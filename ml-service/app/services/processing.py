
import shutil
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
import subprocess

from scipy.ndimage import uniform_filter1d
import numpy as np

from app.core.config import settings
from app.core import s3 as s3_client
from app.core.redis_client import get_redis
from app.services.skeleton_to_segments import process_skeleton_to_segments
from app.services.skeleton_to_segments import (
    compute_energy,
    detect_boundaries,
    build_segments,
)
from app.services.video_to_json import convert_video_to_json
from app.services.body_parts_extractor import extract_body_parts_for_segments

CACHE_VERSION = "v2"
def _video_cache_key(video_hash: str, dance_id: str = None) -> str:
    if dance_id is None:
        return f"video_result:{CACHE_VERSION}:{video_hash}"
    return f"video_result:{CACHE_VERSION}:{video_hash}:{dance_id}"
print("[DEBUG processing.py] ALL IMPORTS SUCCESSFUL", file=sys.stderr)

logger = logging.getLogger(__name__)

def _ensure_h264(video_path: str) -> str:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    codec = probe.stdout.strip()
    logger.info(f"Video codec: {codec}")

    if codec in ("av1", "vp9", "vp8", "hevc"):
        out_path = video_path.replace(".mp4", "_h264.mp4")
        logger.info(f"Transcoding {codec} → h264: {out_path}")
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
            raise RuntimeError(f"ffmpeg transcode failed:\n{result.stderr}")
        return out_path

    return video_path



def _add_placeholder_descriptions(segments: list) -> list:
    result = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        seg_copy["llm_description"] = f"описание сегмента {i}"
        result.append(seg_copy)
    return result



def _simplify_and_enrich_segments(segments: list) -> list:
    from app.services.dance_compare.dance_compare import extract_numeric_metrics
 
    result = []
    for seg in segments:
        body_parts_desc = seg.get("body_parts_description", {})
        features = ""
        if isinstance(body_parts_desc, dict):
            features = body_parts_desc.get("detailed_descriptions", "")
 
        result.append({
            "index":           seg.get("index"),
            "label":           seg.get("label"),
            "start_frame":     seg.get("start_frame"),
            "end_frame":       seg.get("end_frame"),
            "llm_description": seg.get("llm_description", ""),
            "features":        features,
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


def _enrich_segments_with_body_parts(
    segments: list,
    mixamo_frames: list,
    fps: float
) -> tuple[list, list]:
    try:
        logger.info("Step 5.5: Body parts extraction...")
        enriched_segments, body_parts_analysis = extract_body_parts_for_segments(
            segments,
            mixamo_frames,
            fps
        )
        
        logger.info(f"Body parts extraction done: {len(enriched_segments)} segments analyzed")
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


def _run_blender(mixamo_json_path: str, glb_output_path: str, num_frames: int = None) -> None:
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
        "ticksPerSecond": mixamo_data.get("ticksPerSecond", 30),
    }


def _render_full_animation(
    mixamo_data: dict,
    dance_id: str,
    tmpdir: Path,
) -> dict:
    try:
        logger.info("Step 7a: Rendering full animation...")
        num_frames = len(mixamo_data["frames"])
        
        full_json_path = str(tmpdir / "full_animation.json")
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(mixamo_data, f, ensure_ascii=False)
        
        glb_path = str(tmpdir / "full_animation.glb")
        s3_key = f"results/{dance_id}/full_animation.glb"
        
        _run_blender(full_json_path, glb_path, num_frames=num_frames)
        
        s3_client.upload_file(glb_path, s3_key)
        
        logger.info(f"Full animation rendered: {s3_key}")
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
) -> dict:
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
        video_path = _ensure_h264(video_path)

        if not Path(video_path).exists():
            raise RuntimeError(f"Failed to download video: {video_path}")
       
        video_hash = _video_hash(video_path)
        redis = get_redis()
        
        cached = redis.get(_video_cache_key(video_hash))
        if cached:
            logger.info(f"Cache hit for video_hash: {video_hash}, updating dance_id={dance_id}")
            result = json.loads(cached)
            result["dance_id"] = dance_id
            return result

        video_s3_key = f"results/{dance_id}/video.mp4"
        logger.info(f"Uploading video to S3: {video_path} -> {video_s3_key}")
        s3_client.upload_file(video_path, video_s3_key)
        logger.info(f"Video successfully uploaded to S3: {video_s3_key}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()


        # === Шаг 3: MediaPipe → Mixamo JSON ===
        logger.info("Step 3: MediaPipe...")
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
        for i in [0, 1, 5, 10, 50, 100]:
            if i < len(mixamo_frames):
                bones = mixamo_frames[i].get("bones", [])
                for b in bones:
                    if b.get("name", "").endswith(":Hips"):
                        pos = b.get("position", {})
                        print(f"Frame {i} Hips: x={pos.get('x',0):.2f} y={pos.get('y',0):.2f} z={pos.get('z',0):.2f}")
       
        # mixamo_frames = _add_root_motion_to_frames(mixamo_frames)

      
        mixamo_data["frames"] = mixamo_frames
        duration_sec = len(mixamo_frames) / fps if fps > 0 else 0.0
      
        
        z_values = []
        for frame in mixamo_frames:
            lms = frame.get("landmarks")
            if lms and lms[0] is not None:
                z_values.append(lms[0]['z'])
    
        z_bone_values = []
        for frame in mixamo_frames:
            for b in frame.get("bones", []):
                if b.get("name", "").endswith(":Hips"):
                    z_bone_values.append(b["position"].get("z", 0))

        z_arr = np.array(z_bone_values)
        hips_x, hips_y, hips_z, hips_indices = [], [], [], []

        for i, frame in enumerate(mixamo_frames):
            for b in frame.get("bones", []):
                if b.get("name", "").endswith(":Hips"):
                    pos = b.get("position", {})
                    hips_x.append(pos.get("x", 0.0))
                    hips_y.append(pos.get("y", 0.0))
                    hips_z.append(pos.get("z", 0.0))
                    hips_indices.append(i)

        if hips_x:
            window = 25 
            sx = uniform_filter1d(hips_x, size=window)
            sy = uniform_filter1d(hips_y, size=window)
            sz = uniform_filter1d(hips_z, size=window)
            
            sx = sx - sx[0]
            sy = sy - np.median(sy) 
            sz = sz - np.median(sz)
    
            Z_HARD_LIMIT = 1.0 
            sz = Z_HARD_LIMIT * np.tanh(sz / Z_HARD_LIMIT)

            
            for j, i in enumerate(hips_indices):
                for b in mixamo_frames[i].get("bones", []):
                    if b.get("name", "").endswith(":Hips"):
                        b["position"]["x"] = float(sx[j])
                        b["position"]["y"] = float(sy[j])
                        b["position"]["z"] = float(sz[j])

        # === Шаг 4: Сегментация ===
        logger.info("Step 4: Segmentation...")
        segments, energy_debug = _segment_mixamo(mixamo_frames, fps)
    
        segments, body_parts_analysis = _enrich_segments_with_body_parts(
            segments, mixamo_frames, fps
        )
        segments = _add_placeholder_descriptions(segments)
        segments = _simplify_and_enrich_segments(segments)

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
        if progress_callback:
            progress_callback("segments_ready", {
                "segments_key": segments_key,
                "num_segments": len(segments),
            })

        # === Шаг 7: Рендеринг анимаций ===
        logger.info("Step 7: Rendering animations...")

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

        logger.info("Step 7b: Rendering segments in parallel...")

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