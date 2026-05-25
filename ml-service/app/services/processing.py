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
import subprocess


import numpy as np

from app.core.config import settings
from app.core import s3 as s3_client
from app.core.redis_client import get_redis
from app.services.skeleton_to_segments import (
    compute_energy,
    detect_boundaries,
    build_segments,
)
from app.services.video_to_json import convert_video_to_json
from app.services.body_parts_extractor import extract_body_parts_for_segments
from app.services.dance_features import compute_dance_features
from app.services.skeleton import save_skeleton_json

CACHE_VERSION = "v2"
def _video_cache_key(video_hash: str, dance_id: str = None) -> str:
    if dance_id is None:
        return f"video_result:{CACHE_VERSION}:{video_hash}"
    return f"video_result:{CACHE_VERSION}:{video_hash}:{dance_id}"


def _clone_cached_assets_for_new_dance(
    old_dance_id: str, new_dance_id: str, result: dict
) -> None:
    pairs: list[tuple[str, str]] = []

    def add(key_old: Optional[str]) -> None:
        if not key_old:
            return
        key_new = key_old.replace(old_dance_id, new_dance_id)
        if key_new != key_old:
            pairs.append((key_old, key_new))

    add(result.get("segments_key"))
    add(result.get("full_glb_key"))
    for glb in result.get("glb_keys") or []:
        add(glb)
    add(result.get("video_path"))
    add(f"dance-landmarks-cache/{old_dance_id}.json")

    for src, dst in pairs:
        try:
            s3_client.copy_object(src, dst)
        except FileNotFoundError:
            logger.warning(
                "cached asset missing under old dance_id, skipping: %s", src
            )
        except Exception as e:
            logger.error(
                "failed to clone cached asset %s -> %s: %s", src, dst, e
            )
            raise


def _rewrite_result_paths(result: dict, old_dance_id: str, new_dance_id: str) -> None:
    for field in ("segments_key", "full_glb_key", "video_path"):
        value = result.get(field)
        if isinstance(value, str):
            result[field] = value.replace(old_dance_id, new_dance_id)
    glbs = result.get("glb_keys")
    if isinstance(glbs, list):
        result["glb_keys"] = [
            g.replace(old_dance_id, new_dance_id) if isinstance(g, str) else g
            for g in glbs
        ]

def _glb_cache_key(json_hash: str, cache_type: str = "full") -> str:
    return f"glb_cache/glb_cache:{CACHE_VERSION}:{cache_type}:{json_hash}"

logger = logging.getLogger(__name__)

def _ensure_h264(video_path: str) -> str:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    codec = probe.stdout.strip()
    logger.info(f"Detected codec: '{codec}' for {video_path}")

    if not codec:
        raise ValueError(
            f"No video stream detected in {video_path}. "
            f"ffprobe stderr: {probe.stderr.strip()}"
        )

    TRANSCODE_CODECS = {"av1", "vp9", "vp8", "hevc", "h265"}
    if codec.lower() not in TRANSCODE_CODECS:
        logger.info(f"Codec '{codec}' is fine, no transcode needed")
        return video_path

    out_path = video_path.replace(".mp4", "_h264.mp4")
    logger.info(f"Transcoding {codec} → h264: {video_path} → {out_path}")

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        out_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed for codec '{codec}':\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    logger.info(f"Transcode complete: {out_path}")
    return out_path



def _add_placeholder_descriptions(segments: list) -> list:
    result = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        seg_copy["llm_description"] = None
        result.append(seg_copy)
    return result



def extract_numeric_metrics(segment: dict) -> dict:
    bpd = segment.get("body_parts_description", {})
    raw = bpd.get("raw_analysis", {}) if isinstance(bpd, dict) else {}

    vel_means, smoothness_vals, rom_vals = [], [], []
    for part_data in raw.get("body_parts", {}).values():
        m = part_data.get("metrics", {})
        vs = m.get("velocity_stats", {})
        if vs.get("mean") is not None:
            vel_means.append(vs["mean"])
        if m.get("smoothness") is not None:
            smoothness_vals.append(m["smoothness"])
        if m.get("rom", {}).get("max_distance") is not None:
            rom_vals.append(m["rom"]["max_distance"])

    tempo = raw.get("tempo", {})
    symmetry = raw.get("symmetry", {})
    sym_ratios = [v.get("velocity_ratio", 1.0) for v in symmetry.values() if isinstance(v, dict)]

    return {
        "velocity": {
            "mean": round(float(np.mean(vel_means)) if vel_means else 0.0, 4),
            "max":  round(float(np.max(vel_means))  if vel_means else 0.0, 4),
        },
        "smoothness":     round(float(np.mean(smoothness_vals)) if smoothness_vals else 1.0, 4),
        "rom": {
            "max_distance":  round(float(np.max(rom_vals))  if rom_vals else 0.0, 4),
            "mean_distance": round(float(np.mean(rom_vals)) if rom_vals else 0.0, 4),
        },
        "tempo_bpm":      round(float(tempo.get("beats_per_min", 0.0)), 2),
        "symmetry_ratio": round(float(np.mean(sym_ratios)) if sym_ratios else 1.0, 4),
        "joint_angles": {
            jname: {
                "mean_deg":  jdata.get("mean_deg",  0.0),
                "range_deg": jdata.get("range_deg", 0.0),
            }
            for jname, jdata in raw.get("joint_angles", {}).items()
            if isinstance(jdata, dict)
        },
    }


def _simplify_and_enrich_segments(segments: list) -> list:
    result = []
    for seg in segments:
        body_parts_desc = seg.get("body_parts_description", {})
        text_features = ""
        if isinstance(body_parts_desc, dict):
            text_features = body_parts_desc.get("detailed_descriptions", "")

        result.append({
            "index":           seg.get("index"),
            "label":           seg.get("label"),
            "start_frame":     seg.get("start_frame"),
            "end_frame":       seg.get("end_frame"),
            "llm_description": seg.get("llm_description", ""),
            "text_dance_features": text_features,
            "numeric_metrics": extract_numeric_metrics(seg),
        })
    return result

def _video_hash(path: str) -> str:
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
    return segments, energy_debug


def _enrich_segments_with_body_parts(
    segments: list,
    mixamo_frames: list,
    fps: float
) -> tuple[list, list]:
    try:
        enriched_segments, body_parts_analysis = extract_body_parts_for_segments(
            segments,
            mixamo_frames,
            fps
        )

        return enriched_segments, body_parts_analysis
    except Exception as e:
        logger.error(f"Error in body parts extraction: {e}")
        return segments, []


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
    ]
    if num_frames is not None:
        cmd.extend(["--num-frames", str(num_frames)])
    if anim_only:
        cmd.append("--anim-only")
    return cmd


def _compute_json_hash(json_data: dict) -> str:
    json_str = json.dumps(json_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def _try_load_cached_glb(json_data: dict, cache_type: str = "full") -> Optional[bytes]:
    json_hash = _compute_json_hash(json_data)
    cache_key = _glb_cache_key(json_hash, cache_type)
    
    try:
        if s3_client.file_exists(cache_key):
            logger.debug(f"GLB cache hit: {cache_key}")
            tmp_path = Path(tempfile.gettempdir()) / f"cached_{json_hash}.glb"
            s3_client.download_file(cache_key, str(tmp_path))
            with open(tmp_path, "rb") as f:
                data = f.read()
            tmp_path.unlink()
            return data
    except Exception as e:
        logger.debug(f"Failed to load GLB cache: {e}")
    
    return None


def _cache_glb_to_s3(glb_path: str, json_data: dict, cache_type: str = "full") -> None:
    json_hash = _compute_json_hash(json_data)
    cache_key = _glb_cache_key(json_hash, cache_type)
    
    try:
        s3_client.upload_file(glb_path, cache_key)
        logger.debug(f"GLB cached: {cache_key}")
    except Exception as e:
        logger.debug(f"Failed to cache GLB: {e}")


def _run_blender(mixamo_json_path: str, glb_output_path: str, num_frames: int = None, use_cache: bool = True) -> None:
    if use_cache:
        with open(mixamo_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        cached_glb = _try_load_cached_glb(json_data, cache_type="segment")
        if cached_glb:
            with open(glb_output_path, "wb") as f:
                f.write(cached_glb)
            return
    
    cmd = _build_blender_cmd(mixamo_json_path, glb_output_path, num_frames=num_frames)

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
        
        if use_cache:
            try:
                with open(mixamo_json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                _cache_glb_to_s3(glb_output_path, json_data, cache_type="segment")
            except Exception as e:
                logger.debug(f"Failed to cache GLB result: {e}")

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Blender timeout: {e}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Blender not in PATH: {e}")

def _run_blender_segment(args: tuple) -> tuple[int, str, str]:
    segment_index, mixamo_json_path, glb_path, s3_key, num_frames = args
    _run_blender(mixamo_json_path, glb_path, num_frames=num_frames, use_cache=True)
    return segment_index, glb_path, s3_key


def _slice_mixamo_for_segment(mixamo_data: dict, segment: dict) -> dict:
    all_frames = mixamo_data["frames"]
    start = segment["start_frame"]
    end = segment["end_frame"]

    start = max(0, start)
    end = min(len(all_frames), end)

    sliced_frames = all_frames[start:end]

    if not sliced_frames:
        raise ValueError(f"Empty slice: start={start}, end={end}, "
                         f"total={len(all_frames)}")

    sliced_frames = [
        {**f, "time": i}
        for i, f in enumerate(sliced_frames)
    ]

    return {
        **mixamo_data,
        "frames": sliced_frames,
        "duration": len(sliced_frames) - 1,
        "ticksPerSecond": mixamo_data.get("ticksPerSecond", 24),
    }


def _render_full_animation(
    mixamo_data: dict,
    dance_id: str,
    tmpdir: Path,
) -> dict:
    try:
        num_frames = len(mixamo_data["frames"])

        full_json_path = str(tmpdir / "full_animation.json")
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(mixamo_data, f, ensure_ascii=False)

        glb_path = str(tmpdir / "full_animation.glb")
        s3_key = f"results/{dance_id}/full_animation.glb"

        _run_blender(full_json_path, glb_path, num_frames=num_frames, use_cache=True)

        s3_client.upload_file(glb_path, s3_key)

        return {
            "success": True,
            "glb_key": s3_key,
            "num_frames": num_frames,
        }
    except Exception as e:
        logger.error(f"Full animation rendering failed: {e}")
        return {
            "success": False,
            "glb_key": None,
            "error": str(e),
            "num_frames": len(mixamo_data.get("frames", [])),
        }

def _render_segments_parallel(
    segments: list,
    mixamo_data: dict,
    dance_id: str,
    tmpdir: Path,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> list[dict]:
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(_run_blender_segment, args): args[0]
            for args in blender_args
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                segment_index, glb_path, s3_key = future.result()
                s3_client.upload_file(glb_path, s3_key)

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



def process_video(
    video_key: str,
    dance_id: str,
    enable_labeling: Optional[bool] = None,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
    uploader_user_id: str = "",
) -> dict:
    logger.info(f"process_video START: video_key={video_key}, dance_id={dance_id}")

    if enable_labeling is None:
        enable_labeling = settings.labeling_enabled

    start_time = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        video_path = str(tmpdir / Path(video_key).name)

        # Скачать видео из S3
        s3_client.download_file(video_key, video_path)
        video_path = _ensure_h264(video_path)

        if not Path(video_path).exists():
            raise RuntimeError(f"Failed to download video: {video_path}")

        # Модерация — до любых дорогостоящих операций
        from app.services.moderation import moderate_video_file
        mod_reason = moderate_video_file(
            video_path=video_path,
            dance_id=dance_id,
            uploader_user_id=uploader_user_id,
            video_s3_url=video_key,
        )
        if mod_reason is not None:
            logger.info("process_video STOPPED by moderation: dance_id=%s reason=%s", dance_id, mod_reason)
            return {"status": "moderation_pending", "reason": mod_reason}

        video_hash = _video_hash(video_path)
        redis = get_redis()

        cached = redis.get(_video_cache_key(video_hash))
        if cached:
            result = json.loads(cached)
            old_dance_id = result.get("dance_id")
            if not old_dance_id or old_dance_id == dance_id:
                result["dance_id"] = dance_id
                return result

            critical = [result.get("segments_key"), result.get("full_glb_key")]
            if all(k and s3_client.file_exists(k) for k in critical):
                _clone_cached_assets_for_new_dance(old_dance_id, dance_id, result)
                _rewrite_result_paths(result, old_dance_id, dance_id)
                result["dance_id"] = dance_id
                return result

            logger.warning(
                "cache HIT for hash=%s but artifacts under old dance_id=%s missing — full reprocess",
                video_hash, old_dance_id,
            )

        video_s3_key = f"results/{dance_id}/video.mp4"
        s3_client.upload_file(video_path, video_s3_key)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()

        # MediaPipe в Mixamo JSON
        model_path = Path(settings.mixamo_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Mixamo model not found: {settings.mixamo_model_path}")

        with open(settings.mixamo_model_path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        mixamo_data = convert_video_to_json(
            video_path=video_path,
            model_json=model_json,
            fps=int(fps),
            min_visibility=settings.mixamo_min_visibility,
            is_hips_move=True,
            max_frames=settings.mixamo_max_frames,
            is_show_result=False,
        )
        mixamo_frames = mixamo_data["frames"]


        mixamo_data["frames"] = mixamo_frames
        duration_sec = len(mixamo_frames) / fps if fps > 0 else 0.0

        for frame in mixamo_frames:
            for b in frame.get("bones", []):
                if b.get("name", "").endswith(":Hips"):
                    b["position"]["y"] = 0.0

        segments, energy_debug = _segment_mixamo(mixamo_frames, fps)

        segments, body_parts_analysis = _enrich_segments_with_body_parts(
            segments, mixamo_frames, fps
        )
        segments = _add_placeholder_descriptions(segments)
        segments = _simplify_and_enrich_segments(segments)

        dance_features = compute_dance_features(mixamo_frames, fps)

        # Сохраняем ландмарки в кэш, чтобы compare и keyframe-задача не перезапускали MediaPipe
        landmarks_cache_key = f"dance-landmarks-cache/{dance_id}.json"
        try:
            if not s3_client.file_exists(landmarks_cache_key):
                landmarks_cache_path = tmpdir / "landmarks_cache.json"
                with open(landmarks_cache_path, "w", encoding="utf-8") as f:
                    json.dump({**mixamo_data, "_fps": fps}, f, ensure_ascii=False)
                s3_client.upload_file(str(landmarks_cache_path), landmarks_cache_key)
                logger.info(f"Landmarks cache saved: {landmarks_cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save landmarks cache: {e}")

        segments_key = f"results/{dance_id}/segments.json"

        segments_data = {
            "version": "2.0",
            "dance_id": dance_id,
            "meta": {
                "fps": fps,
                "num_frames": len(mixamo_frames),
                "duration_sec": round(duration_sec, 3),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
            "num_segments": len(segments),
            "segments": segments,
            "dance_features": dance_features,
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
        if progress_callback:
            progress_callback("segments_ready", {
                "segments_key": segments_key,
                "num_segments": len(segments),
            })

        # 2D-скелет эталона для фронт-оверлея
        try:
            save_skeleton_json(
                mixamo_data=mixamo_data,
                fps=float(fps),
                s3_key=f"results/{dance_id}/skeleton.json",
                tmpdir=tmpdir,
                local_name="reference_skeleton.json",
            )
        except Exception as e:
            logger.warning(f"reference skeleton.json upload failed: {e}")

        full_animation_result = _render_full_animation(
            mixamo_data=mixamo_data,
            dance_id=dance_id,
            tmpdir=tmpdir,
        )
        full_glb_key = full_animation_result.get("glb_key") if full_animation_result.get("success") else None

        if progress_callback and full_glb_key:
            progress_callback("full_animation_ready", {
                "glb_key": full_glb_key,
            })

        def on_segment_done(segment_index: int, glb_key: str):
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

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()


        result = {
            "dance_id": dance_id,
            "segments_key": segments_key,
            "full_glb_key": full_glb_key,
            "glb_keys": [r["glb_key"] for r in successful],
            "num_frames": len(mixamo_frames),
            "num_segments": len(segments),
            "num_segments_rendered": len(successful),
            "duration_sec": round(duration_sec, 3),
            "processing_time_sec": round(processing_time, 2),
            "video_path": video_s3_key,

        }

        redis.setex(_video_cache_key(video_hash), 86400, json.dumps(result))

        logger.info(

            f"{len(successful)} rendered, {processing_time:.1f}s"
        )
        return result