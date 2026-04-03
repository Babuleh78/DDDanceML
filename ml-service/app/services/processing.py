"""
Пайплайн обработки танцевального видео:
1. Скачиваем видео из S3
2. video_to_mixamo: MediaPipe → Mixamo JSON (кватернионы костей)
3. skeleton_to_segments: энергетическая сегментация движений
4. labeling: LLM-описания для каждого сегмента
5. Загружаем segments.json в S3 (первый ответ клиенту)
6. Blender headless: по сегментам параллельно → .glb файлы
7. Загружаем .glb в S3 по мере готовности
8. Возвращаем ключи
"""

import sys
print(f"[DEBUG processing.py] Module loading, Python {sys.version}", file=sys.stderr)

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from app.core.config import settings
from app.core import s3 as s3_client
from app.core.redis_client import get_redis
from app.services.skeleton_to_segments import process_skeleton_to_segments
from app.services.skeleton_to_segments import (
    compute_energy,
    detect_boundaries,
    build_segments,
)
from app.services.video_to_mixamo import convert_video_to_mixamo_json
from app.services.labeling.factory import get_labeling_strategy
from app.services.labeling.features import extract_geometric_features
from app.services.labeling.cache import label_cache

CACHE_VERSION = "v2"
def _video_cache_key(video_hash: str) -> str:
            return f"video_result:{CACHE_VERSION}:{video_hash}"
print("[DEBUG processing.py] ALL IMPORTS SUCCESSFUL", file=sys.stderr)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _make_s3_keys(dance_id: str, num_segments: int) -> dict:
    """
    Возвращает все S3-ключи для данного dance_id.

    Структура в S3:
        results/{dance_id}/
            segments.json
            segment_0.glb
            segment_1.glb
            ...
    """
    return {
        "segments": f"results/{dance_id}/segments.json",
        "glb": [
            f"results/{dance_id}/segment_{i}.glb"
            for i in range(num_segments)
        ],
    }


def _video_hash(path: str) -> str:
    """SHA256 по первому и последнему MB файла — быстрый стабильный ключ кэша."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_048_576))
        try:
            f.seek(-1_048_576, 2)
            h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()


def _segment_mixamo(mixamo_frames: list, fps: float) -> tuple[list, dict]:
    """Энергетическая сегментация: frames → segments + debug."""
    logger.info(f"_segment_mixamo: {len(mixamo_frames)} frames, fps={fps}")

    energy, energy_debug = compute_energy(
        mixamo_frames,
        smooth_window=settings.segmenter_smooth_window,
    )
    boundaries = detect_boundaries(
        energy,
        fps=fps,
        min_segment_sec=settings.segmenter_min_seg_sec,
        sensitivity=settings.segmenter_sensitivity,
    )
    segments = build_segments(
        mixamo_frames, boundaries, fps,
        energy=energy,
        min_segment_sec=settings.segmenter_min_seg_sec,
    )
    logger.info(f"_segment_mixamo done: {len(segments)} segments")
    return segments, energy_debug


async def _label_segments(
    segments: list,
    mixamo_frames: list,
    fps: float,
    energy_values,
    enable_labeling: bool = True,
) -> tuple[list, dict]:
    """LLM-разметка сегментов с батчингом и кэшем."""
    labeling_meta = {
        "enabled": enable_labeling,
        "strategy": None,
        "cached_hits": 0,
        "errors": 0,
        "processed_count": 0,
    }

    if not enable_labeling or not settings.labeling_enabled:
        return segments, labeling_meta

    strategy = get_labeling_strategy()
    labeling_meta["strategy"] = strategy.name
    logger.info(f"Labeling strategy: {strategy.name}, segments: {len(segments)}")

    items = [
        (
            extract_geometric_features(
                frames=mixamo_frames,
                start_frame=seg["start_frame"],
                end_frame=seg["end_frame"],
                energy_values=energy_values,
            ),
            seg["duration_sec"],
            i,
        )
        for i, seg in enumerate(segments)
    ]

    cached_indices = set()
    for features, duration_sec, i in items:
        cache_key = strategy.compute_features_hash(features, duration_sec)
        if label_cache.get(cache_key):
            cached_indices.add(i)
            labeling_meta["cached_hits"] += 1

    from .labeling.ollama import OllamaStrategy
    if isinstance(strategy, OllamaStrategy):
        labels = await strategy.generate_labels_batch(items)
    else:
        labels = []
        for features, duration_sec, i in items:
            try:
                label = await strategy.generate_label(features, duration_sec)
                labels.append(label)
            except Exception as e:
                logger.warning(f"Segment #{i} labeling failed: {e}")
                labels.append(f"segment_{i + 1}")
                labeling_meta["errors"] += 1

    for i, (segment, label) in enumerate(zip(segments, labels)):
        segment["label"] = label
        segment["label_source"] = strategy.name
        if i not in cached_indices:
            labeling_meta["processed_count"] += 1

    logger.info(
        f"Labeling done: {labeling_meta['processed_count']} new, "
        f"{labeling_meta['cached_hits']} cached, {labeling_meta['errors']} errors"
    )
    return segments, labeling_meta


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------

def _build_blender_cmd(mixamo_json_path: str, glb_output_path: str, anim_only: bool = True, num_frames: int = None) -> list[str]:
    blender_script = (
        Path(__file__).parent / "blender_logic" / "import_and_export.py"
    ).resolve()
    character_blend = Path(settings.blender_character_blend)
    if not character_blend.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        character_blend = (project_root / character_blend).resolve()

    if not blender_script.exists():
        raise RuntimeError(f"Blender script not found: {blender_script}")
    if not character_blend.exists():
        raise RuntimeError(f"Character blend not found: {character_blend}")

    cmd = [
        settings.blender_executable,
        str(character_blend),
        "--background",
        "--python", str(blender_script),
        "--",
        "--json", mixamo_json_path,
        "--output", glb_output_path,
        "--format", "GLB",
         "--num-frames", str(num_frames)
    ]
    if anim_only:
        cmd.append("--anim-only")
    return cmd


def _run_blender(mixamo_json_path: str, glb_output_path: str, num_frames: int = None) -> None:
    """Запускает Blender синхронно. Вызывается в ProcessPoolExecutor."""
    cmd = _build_blender_cmd(mixamo_json_path, glb_output_path, num_frames=num_frames)
    logger.info(f"Blender start: {glb_output_path}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender exited {result.returncode}.\n"
                f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
            )
        logger.info(f"Blender done: {glb_output_path}")

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Blender timeout: {e}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Blender not in PATH: {e}")

def _run_blender_segment(args: tuple) -> tuple[int, str, str]:
    segment_index, mixamo_json_path, glb_path, s3_key, num_frames = args
    _run_blender(mixamo_json_path, glb_path, num_frames=num_frames)
    return segment_index, glb_path, s3_key


def _slice_mixamo_for_segment(mixamo_data: dict, segment: dict) -> dict:
    """
    start_frame/end_frame — это индексы в массиве frames, не значения time.
    """
    all_frames = mixamo_data["frames"]
    start = segment["start_frame"]
    end = segment["end_frame"]

    # Защита от выхода за границы
    start = max(0, start)
    end = min(len(all_frames), end)

    sliced_frames = all_frames[start:end]

    if not sliced_frames:
        raise ValueError(f"Empty slice: start={start}, end={end}, "
                         f"total={len(all_frames)}")

    # Сбрасываем время к 0 — кадр 0, 1, 2...
    # Это гарантирует что Blender начнёт анимацию с frame_start=0
    sliced_frames = [
        {**f, "time": i}
        for i, f in enumerate(sliced_frames)
    ]

    return {
        **mixamo_data,
        "frames": sliced_frames,
        "duration": len(sliced_frames) - 1,
        "ticksPerSecond": mixamo_data.get("ticksPerSecond", 30),
    }

def _render_segments_parallel(
    segments: list,
    mixamo_data: dict,
    dance_id: str,
    tmpdir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> list[dict]:
    """
    Запускает Blender для каждого сегмента параллельно через потоки.
    Потоки разрешены внутри Celery daemon-воркера, в отличие от процессов.
    Blender всё равно запускается как внешний subprocess — параллельность есть.
    """
    blender_args = []
    for i, segment in enumerate(segments):
        seg_data = _slice_mixamo_for_segment(mixamo_data, segment)
        num_frames = segment["end_frame"] - segment["start_frame"]

        seg_json_path = str(tmpdir / f"seg_{i}.json")
        with open(seg_json_path, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False)

        glb_path = str(tmpdir / f"segment_{i}.glb")
        s3_key = f"results/{dance_id}/segment_{i}.glb"
        blender_args.append((i, seg_json_path, glb_path, s3_key, num_frames))

    results = []

    # ThreadPoolExecutor вместо ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {
            executor.submit(_run_blender_segment, args): args[0]
            for args in blender_args
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                segment_index, glb_path, s3_key = future.result()
                s3_client.upload_file(glb_path, s3_key)
                logger.info(f"Segment {segment_index} uploaded → {s3_key}")

                results.append({
                    "index": segment_index,
                    "glb_key": s3_key,
                    "success": True,
                })

                if progress_callback:
                    progress_callback(segment_index, s3_key)

            except Exception as e:
                logger.error(f"Segment {idx} failed: {e}")
                results.append({
                    "index": idx,
                    "glb_key": None,
                    "success": False,
                    "error": str(e),
                })

    results.sort(key=lambda x: x["index"])
    return results

# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def process_video(
    video_key: str,
    dance_id: str,
    enable_labeling: Optional[bool] = None,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """
    Полный пайплайн обработки видео.

    Args:
        video_key:         S3-ключ входного видео
        dance_id:          UUID танца, определяет папку в S3
        enable_labeling:   включить LLM-разметку (None = из конфига)
        progress_callback: fn(event_name, data) — вызывается при каждом событии.
                           Используется воркером для update_state в Celery.

    Events (progress_callback):
        "segments_ready"  → {"segments_key": str, "num_segments": int}
        "segment_ready"   → {"index": int, "glb_key": str}
        "pipeline_error"  → {"step": str, "error": str}
    """
    logger.info(f"process_video START: video_key={video_key}, dance_id={dance_id}")

    if enable_labeling is None:
        enable_labeling = settings.labeling_enabled

    start_time = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        video_path = str(tmpdir / Path(video_key).name)
    

         # === Шаг 1: Скачать видео из S3 ===
        logger.info(f"Step 1: Downloading {video_key}")
        s3_client.download_file(video_key, video_path)

        if not Path(video_path).exists():
            raise RuntimeError(f"Failed to download video: {video_path}")
        logger.info(f"Video downloaded: {Path(video_path).stat().st_size} bytes")
         # === Кэш по хэшу видео ===
        video_hash = _video_hash(video_path)
        redis = get_redis()
        
        cached = redis.get(_video_cache_key(video_hash))
        if cached:
            logger.info(f"Cache hit: {video_key}")
            return json.loads(cached)



        # === Шаг 2: FPS ===
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        logger.info(f"FPS: {fps}")

        # === Шаг 3: MediaPipe → Mixamo JSON ===
        logger.info("Step 3: MediaPipe...")
        model_path = Path(settings.mixamo_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Mixamo model not found: {settings.mixamo_model_path}")

        with open(settings.mixamo_model_path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        mixamo_data = convert_video_to_mixamo_json(
            video_path=video_path,
            model_json=model_json,
            fps=int(fps),
            min_visibility=settings.mixamo_min_visibility,
            is_hips_move=settings.mixamo_hips_move,
            max_frames=settings.mixamo_max_frames,
            is_show_result=False,
        )
        mixamo_frames = mixamo_data["frames"]
        duration_sec = len(mixamo_frames) / fps if fps > 0 else 0.0
        logger.info(f"MediaPipe done: {len(mixamo_frames)} frames ({duration_sec:.2f}s)")

        # === Шаг 4: Сегментация ===
        logger.info("Step 4: Segmentation...")
        segments, energy_debug = _segment_mixamo(mixamo_frames, fps)
        logger.info(f"Segments: {len(segments)}")

        # === Шаг 5: LLM-разметка ===
        if enable_labeling:
            logger.info("Step 5: LLM labeling...")
            energy_values, _ = compute_energy(
                mixamo_frames,
                smooth_window=settings.segmenter_smooth_window,
            )
            segments, labeling_meta = asyncio.run(
                _label_segments(segments, mixamo_frames, fps, energy_values)
            )
        else:
            labeling_meta = {"enabled": False, "strategy": None}
            logger.info("Step 5: Labeling skipped")

        # === Шаг 6: Сохранить и отдать segments.json ===
        logger.info("Step 6: Uploading segments.json...")
        segments_key = f"results/{dance_id}/segments.json"

        segments_data = {
            "version": "2.0",
            "dance_id": dance_id,
            "meta": {
                "fps": fps,
                "num_frames": len(mixamo_frames),
                "duration_sec": round(duration_sec, 3),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "labeling": labeling_meta,
                "cache_stats": label_cache.stats() if enable_labeling else None,
            },
            "num_segments": len(segments),
            "segments": segments,
            "debug": {
                "energy_stats": {
                    "mean": float(
                        energy_debug.get("energy_smooth", [0])[-1]
                        if energy_debug.get("energy_smooth") else 0
                    ),
                    "max": float(
                        max(energy_debug.get("energy_smooth", [0]))
                        if energy_debug.get("energy_smooth") else 0
                    ),
                }
            } if settings.debug_mode else None,
        }

        segments_path = tmpdir / "segments.json"
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)

        s3_client.upload_file(str(segments_path), segments_key)
        logger.info(f"segments.json uploaded → {segments_key}")

        # Первое событие — Go backend уже может показать список движений
        if progress_callback:
            progress_callback("segments_ready", {
                "segments_key": segments_key,
                "num_segments": len(segments),
            })

        # === Шаг 7: Blender по сегментам параллельно ===
        logger.info("Step 7: Rendering segments in parallel...")

        def on_segment_done(segment_index: int, glb_key: str):
            logger.info(f"Segment {segment_index} ready → {glb_key}")
            if progress_callback:
                progress_callback("segment_ready", {
                    "index": segment_index,
                    "glb_key": glb_key,
                })

        segment_results = _render_segments_parallel(
            segments=segments,
            mixamo_data=mixamo_data,
            dance_id=dance_id,
            tmpdir=tmpdir,
            progress_callback=on_segment_done,
        )

        successful = [r for r in segment_results if r["success"]]
        failed = [r for r in segment_results if not r["success"]]

        if failed:
            logger.warning(f"{len(failed)} segments failed: {failed}")

        # === Итог ===
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = {
            "dance_id": dance_id,
            "segments_key": segments_key,
            "glb_keys": [r["glb_key"] for r in successful],
            "num_frames": len(mixamo_frames),
            "num_segments": len(segments),
            "num_segments_rendered": len(successful),
            "duration_sec": round(duration_sec, 3),
            "processing_time_sec": round(processing_time, 2),
            "labeling_summary": {
                "strategy": labeling_meta.get("strategy"),
                "processed": labeling_meta.get("processed_count", 0),
                "cached": labeling_meta.get("cached_hits", 0),
                "errors": labeling_meta.get("errors", 0),
            } if enable_labeling else None,
        }

        redis.setex(_video_cache_key(video_hash), 86400, json.dumps(result))

        logger.info(
            f"✅ Done: {len(segments)} segments, "
            f"{len(successful)} rendered, {processing_time:.1f}s"
        )
        return result