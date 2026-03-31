"""
Пайплайн обработки танцевального видео:
1. Скачиваем видео из S3
2. video_to_mixamo: MediaPipe → Mixamo JSON (кватернионы костей)
3. skeleton_to_segments: энергетическая сегментация движений
4. [НОВОЕ] labeling: LLM-описания для каждого сегмента
5. Blender headless: Mixamo JSON → .glb анимация
6. Загружаем в S3: animation.glb + segments.json
7. Возвращаем ключи
"""

# === [ОТЛАДКА] Самый первый вывод — до любых импортов ===
import sys
print(f"[DEBUG processing.py] Module loading, Python {sys.version}", file=sys.stderr)

import json
import subprocess
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from app.services.skeleton_to_segments import process_skeleton_to_segments

print("[DEBUG processing.py] Standard imports OK", file=sys.stderr)

import cv2
print("[DEBUG processing.py] cv2 imported", file=sys.stderr)

from app.core.config import settings
print(f"[DEBUG processing.py] settings imported, bucket={settings.s3_bucket}", file=sys.stderr)

from app.core import s3 as s3_client
print("[DEBUG processing.py] s3_client imported", file=sys.stderr)

print("[DEBUG processing.py] About to import skeleton_to_segments...", file=sys.stderr)
from app.services.skeleton_to_segments import (
    compute_energy, 
    detect_boundaries, 
    build_segments,
)
print("[DEBUG processing.py] skeleton_to_segments functions imported", file=sys.stderr)

from app.services.video_to_mixamo import convert_video_to_mixamo_json
print("[DEBUG processing.py] video_to_mixamo imported", file=sys.stderr)

# === Импорты для LLM-разметки ===
print("[DEBUG processing.py] About to import labeling modules...", file=sys.stderr)
from app.services.labeling.factory import get_labeling_strategy
print("[DEBUG processing.py] get_labeling_strategy imported", file=sys.stderr)

from app.services.labeling.features import extract_geometric_features
print("[DEBUG processing.py] extract_geometric_features imported", file=sys.stderr)

from app.services.labeling.cache import label_cache
print("[DEBUG processing.py] label_cache imported", file=sys.stderr)

print("[DEBUG processing.py] ALL IMPORTS SUCCESSFUL", file=sys.stderr)

# === Настройка логгера ===
logger = logging.getLogger(__name__)
logger.info("processing.py module fully loaded")


def _make_keys(video_key: str) -> tuple[str, str]:
    """Генерирует S3-ключи для результатов."""
    logger.debug(f"_make_keys called with video_key={video_key}")
    stem = Path(video_key).stem
    result = (
        f"results/{stem}_animation.glb",
        f"results/{stem}_segments.json",
    )
    logger.debug(f"_make_keys returning: {result}")
    return result

def _segment_mixamo(mixamo_frames: list, fps: float) -> tuple[list, dict]:
    """
    Сегментирует движения и возвращает сегменты + отладочную информацию.
    
    mixamo_frames: список кадров в формате video_to_mixamo:
        [{"time": float, "bones": [{"name": str, "rotation": {...}}, ...]}, ...]
    """
    logger.info(f"_segment_mixamo: {len(mixamo_frames)} frames, fps={fps}")
    
    # === Используем функцию, совместимую с форматом Mixamo ===
    logger.debug("Calling compute_energy_quat (quaternion-based)...")
    energy, energy_debug = compute_energy(
        mixamo_frames,  # ← формат с "bones", а не "joints"
        smooth_window=settings.segmenter_smooth_window,
    )
    logger.debug(f"compute_energy done, energy length: {len(energy)}")
    
    logger.debug("Calling detect_boundaries...")
    boundaries = detect_boundaries(
        energy,
        fps=fps,
        min_segment_sec=settings.segmenter_min_seg_sec,
        sensitivity=settings.segmenter_sensitivity,
    )
    logger.debug(f"detect_boundaries found {len(boundaries)} boundaries")
    
    logger.debug("Calling build_segments...")
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
    """Добавляет LLM-названия к сегментам."""
    logger.info(f"_label_segments: enable={enable_labeling}, {len(segments)} segments")
    
    labeling_meta = {
        "enabled": enable_labeling,
        "strategy": None,
        "cached_hits": 0,
        "errors": 0,
        "processed_count": 0,
    }
    
    if not enable_labeling or not settings.labeling_enabled:
        logger.info("Labeling disabled by config")
        return segments, labeling_meta
    
    logger.info("Getting labeling strategy...")
    strategy = get_labeling_strategy()
    labeling_meta["strategy"] = strategy.name
    logger.info(f"Using strategy: {strategy.name}")
    
    logger.info(f"Starting labeling loop: {len(segments)} segments")
    for i, segment in enumerate(segments):
        try:
            logger.debug(f"Labeling segment #{i}: frames {segment['start_frame']}-{segment['end_frame']}")
            
            # Извлекаем признаки для сегмента
            features = extract_geometric_features(
                frames=mixamo_frames,
                start_frame=segment['start_frame'],
                end_frame=segment['end_frame'],
                energy_values=energy_values,
            )
            logger.debug(f"Features extracted: { {k:v for k,v in features.items() if isinstance(v, (bool, float, int))} }")
            
            # Проверяем кэш
            cache_key = strategy.compute_features_hash(features, segment['duration_sec'])
            if label_cache.get(cache_key):
                labeling_meta["cached_hits"] += 1
                logger.debug(f"Segment #{i}: cache hit")
            else:
                logger.debug(f"Segment #{i}: calling LLM...")
                label = await strategy.generate_label(features, segment['duration_sec'])
                segment['label'] = label
                segment['label_source'] = strategy.name
                segment['features_hash'] = cache_key
                labeling_meta["processed_count"] += 1
                logger.info(f"Segment #{i}: label='{label}'")
                
        except Exception as e:
            logger.warning(f"Failed to label segment #{i}: {type(e).__name__}: {e}", exc_info=False)
            labeling_meta["errors"] += 1
            segment['label_source'] = "fallback"
    
    logger.info(
        f"Labeling done: {labeling_meta['processed_count']} new, "
        f"{labeling_meta['cached_hits']} cached, {labeling_meta['errors']} errors"
    )
    
    return segments, labeling_meta


def _run_blender(mixamo_json_path: str, glb_output_path: str) -> None:
    """Запускает Blender в головном режиме для экспорта .glb."""
    logger.info(f"_run_blender: json={mixamo_json_path}, output={glb_output_path}")
    
    blender_script = (Path(__file__).parent / "blender_logic" / "import_and_export.py").resolve()
    character_blend = Path(settings.blender_character_blend)

    if not character_blend.is_absolute():
        # Корень проекта: ml-service/ (на 3 уровня выше config.py)
        project_root = Path(__file__).parent.parent.parent
        character_blend = (project_root / character_blend).resolve()
    
    logger.debug(f"Blender script path: {blender_script} (exists={blender_script.exists()})")
    logger.debug(f"Character blend path: {character_blend} (exists={character_blend.exists()})")
    
    if not blender_script.exists():
        raise RuntimeError(f"Blender script not found: {blender_script}")
    if not character_blend.exists():
        raise RuntimeError(f"Character file not found: {character_blend}")

    cmd = [
        settings.blender_executable,
        str(character_blend),
        "--background",
        "--python", str(blender_script),
        "--",
        "--json", mixamo_json_path,
        "--output", glb_output_path,
        "--format", "GLB",
    ]
    logger.debug(f"Blender command: {' '.join(cmd)}")

    try:
        logger.info("Starting Blender subprocess...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        logger.debug(f"Blender exited with code {result.returncode}")
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[Blender stdout] {line}")
        if result.stderr:
            for line in result.stderr.splitlines():
                if "WARNING:" in line or "deprecated" in line.lower():
                    logger.warning(f"[Blender stderr] {line}")
                else:
                    logger.error(f"[Blender stderr] {line}")
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender exited with code {result.returncode}.\n"
                f"STDERR:\n{result.stderr}\n"
                f"STDOUT:\n{result.stdout}"
            )
        logger.info("Blender completed successfully")
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"Blender timeout: {e}")
        raise RuntimeError(f"Blender timeout: {e}")
    except FileNotFoundError as e:
        logger.error(f"Blender not found: {e}")
        raise RuntimeError(f"Blender not in PATH: {e}")
    except Exception as e:
        logger.error(f"Blender unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise


def process_video(
    video_key: str,
    enable_labeling: Optional[bool] = None,
) -> dict:
    """Полный пайплайн обработки видео."""
    print(f"[DEBUG] process_video ENTER: video_key={video_key}, enable_labeling={enable_labeling}", file=sys.stderr)
    logger.info(f"process_video START: video_key={video_key}")
    
    if enable_labeling is None:
        enable_labeling = settings.labeling_enabled
        logger.debug(f"labeling_enabled from config: {enable_labeling}")
    
    animation_key, segments_key = _make_keys(video_key)
    start_time = datetime.now(timezone.utc)
    logger.info(f"Result keys: animation={animation_key}, segments={segments_key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = str(tmpdir / Path(video_key).name)
        print(f"[DEBUG] Temp dir: {tmpdir}, video_path: {video_path}", file=sys.stderr)

        # === Шаг 1: Скачать видео из S3 ===
        logger.info(f"Step 1: Downloading video from S3: {settings.s3_bucket}/{video_key}")
        print(f"[DEBUG] About to call s3_client.download_file", file=sys.stderr)
        s3_client.download_file(video_key, video_path)
        print(f"[DEBUG] s3_client.download_file returned", file=sys.stderr)
        
        if not Path(video_path).exists():
            logger.error(f"Video not downloaded: {video_path}")
            raise RuntimeError(f"Failed to download video to {video_path}")
        logger.info(f"Video downloaded: {Path(video_path).stat().st_size} bytes")

        # === Шаг 2: Определить FPS ===
        logger.info("Step 2: Detecting video FPS...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        logger.info(f"Video FPS: {fps}")

        # === Шаг 3: MediaPipe → Mixamo JSON ===
        logger.info("Step 3: MediaPipe pose estimation...")
        
        model_path = Path(settings.mixamo_model_path)
        logger.debug(f"Mixamo model path: {model_path} (exists={model_path.exists()})")
        if not model_path.exists():
            raise RuntimeError(
                f"Mixamo model not found: {settings.mixamo_model_path}. "
                "Set MIXAMO_MODEL_PATH in .env"
            )

        with open(settings.mixamo_model_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        logger.debug("Mixamo model loaded")

        logger.info("Calling convert_video_to_mixamo_json...")
        print("[DEBUG] About to call convert_video_to_mixamo_json", file=sys.stderr)
        mixamo_data = convert_video_to_mixamo_json(
            video_path=video_path,
            model_json=model_json,
            fps=int(fps),
            min_visibility=settings.mixamo_min_visibility,
            is_hips_move=settings.mixamo_hips_move,
            max_frames=settings.mixamo_max_frames,
            is_show_result=False,
        )
        print("[DEBUG] convert_video_to_mixamo_json returned", file=sys.stderr)

        mixamo_frames = mixamo_data['frames']
        duration_sec = len(mixamo_frames) / fps if fps > 0 else 0.0
        logger.info(f"MediaPipe done: {len(mixamo_frames)} frames ({duration_sec:.2f}s)")

        # === Шаг 4: Сегментация ===
        logger.info("Step 4: Computing energy and detecting boundaries...")
        print("[DEBUG] About to call _segment_mixamo", file=sys.stderr)
        segments, energy_debug = _segment_mixamo(mixamo_frames, fps)
        print("[DEBUG] _segment_mixamo returned", file=sys.stderr)
        logger.info(f"Segmentation done: {len(segments)} segments")

        # === Шаг 5: [НОВОЕ] LLM-разметка сегментов ===
        if enable_labeling:
            logger.info("Step 5: Starting LLM labeling phase...")
            print("[DEBUG] About to compute energy for labeling", file=sys.stderr)
            
            energy_values, _ = compute_energy(
                mixamo_frames,
                smooth_window=settings.segmenter_smooth_window,
            )
            print("[DEBUG] Energy computed for labeling", file=sys.stderr)
            
            import asyncio
            print("[DEBUG] About to asyncio.run(_label_segments)", file=sys.stderr)
            segments, labeling_meta = asyncio.run(
                _label_segments(segments, mixamo_frames, fps, energy_values, enable_labeling=True)
            )
            print("[DEBUG] _label_segments completed", file=sys.stderr)
            logger.info(f"Labeling done: strategy={labeling_meta.get('strategy')}")
        else:
            labeling_meta = {"enabled": False, "strategy": None}
            logger.info("Step 5: Labeling skipped")

        # === Шаг 6: Экспорт .glb через Blender ===
        logger.info("Step 6: Exporting animation via Blender...")
        print("[DEBUG] About to save mixamo.json and call Blender", file=sys.stderr)
        
        mixamo_json_path = str(tmpdir / "mixamo.json")
        with open(mixamo_json_path, "w", encoding="utf-8") as f:
            json.dump(mixamo_data, f, ensure_ascii=False)
        logger.debug(f"Mixamo JSON saved: {mixamo_json_path}")

        glb_path = str(tmpdir / "animation.glb")
        print("[DEBUG] About to call _run_blender", file=sys.stderr)
        _run_blender(mixamo_json_path, glb_path)
        print("[DEBUG] _run_blender returned", file=sys.stderr)
        logger.info(f"GLB exported: {glb_path} ({Path(glb_path).stat().st_size} bytes)")

        # === Шаг 7: Подготовка segments.json ===
        logger.info("Step 7: Preparing segments.json...")
        segments_data = {
            "version": "1.1",
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
                    "mean": float(energy_debug.get("energy_smooth", [0])[-1] if energy_debug.get("energy_smooth") else 0),
                    "max": float(max(energy_debug.get("energy_smooth", [0])) if energy_debug.get("energy_smooth") else 0),
                }
            } if settings.debug_mode else None,
        }

        segments_path = tmpdir / "segments.json"
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"segments.json saved: {segments_path}")

        # === Шаг 8: Загрузка результатов в S3 ===
        logger.info("Step 8: Uploading results to S3...")
        print(f"[DEBUG] Uploading GLB: {glb_path} → {animation_key}", file=sys.stderr)
        s3_client.upload_file(glb_path, animation_key)
        print(f"[DEBUG] Uploading segments: {segments_path} → {segments_key}", file=sys.stderr)
        s3_client.upload_file(str(segments_path), segments_key)
        print("[DEBUG] S3 uploads completed", file=sys.stderr)
        logger.info("S3 uploads done")

        # === Итоговая статистика ===
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        result = {
            "animation_key": animation_key,
            "segments_key": segments_key,
            "num_frames": len(mixamo_frames),
            "num_segments": len(segments),
            "duration_sec": round(duration_sec, 3),
            "processing_time_sec": round(processing_time, 2),
        }
        
        if enable_labeling:
            result["labeling_summary"] = {
                "strategy": labeling_meta.get("strategy"),
                "processed": labeling_meta.get("processed_count", 0),
                "cached": labeling_meta.get("cached_hits", 0),
                "errors": labeling_meta.get("errors", 0),
            }
        
        logger.info(
            f"✅ Pipeline complete: {len(segments)} segments, "
            f"{processing_time:.1f}s, labeling={labeling_meta.get('strategy', 'off')}"
        )
        print(f"[DEBUG] process_video EXIT: returning result", file=sys.stderr)
        
        return result


# === [ОТЛАДКА] Прямой запуск для тестов ===
if __name__ == "__main__":
    print("[DEBUG] __main__ block entered", file=sys.stderr)
    
    # Минимальная настройка логирования для прямого запуска
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
        force=True
    )
    
    # Тестовый запуск
    try:
        print("[DEBUG] Calling process_video with test params", file=sys.stderr)
        result = process_video(
            video_key="videos/test.mp4",  # Замените на реальное видео в вашем бакете
            enable_labeling=False,  # Начните без LLM для изоляции проблем
        )
        print(f"[DEBUG] Result: {json.dumps(result, indent=2)}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("[DEBUG] Script completed successfully", file=sys.stderr)