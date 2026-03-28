"""
    1. Скачиваем видео из S3
    2. video_to_mixamo: MediaPipe → Mixamo JSON (кватернионы костей)
    3. skeleton_to_segments: энергетическая сегментация движений
    4. Blender headless: Mixamo JSON → .glb анимация
    5. Загружаем в S3: animation.glb + segments.json
    6. Возвращаем ключи
"""

import json
import subprocess
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path
import cv2
from app.core.config import settings
from app.core import s3 as s3_client
from app.services.skeleton_to_segments import compute_energy, detect_boundaries, build_segments
from app.services.video_to_mixamo import convert_video_to_mixamo_json

logger = logging.getLogger(__name__)


def _make_keys(video_key: str) -> tuple[str, str]:
    stem = Path(video_key).stem
    return (
        f"results/{stem}_animation.glb",
        f"results/{stem}_segments.json",
    )


def _frames_to_raw(mixamo_frames: list, fps: float = 30.0) -> list:
    BONE_TO_MP_IDX = {
        'mixamorig:Neck':         0,
        'mixamorig:LeftShoulder': 11,
        'mixamorig:RightShoulder':12,
        'mixamorig:LeftArm':      13,
        'mixamorig:RightArm':     14,
        'mixamorig:LeftForeArm':  15,
        'mixamorig:RightForeArm': 16,
        'mixamorig:LeftUpLeg':    23,
        'mixamorig:RightUpLeg':   24,
        'mixamorig:LeftLeg':      25,
        'mixamorig:RightLeg':     26,
        'mixamorig:LeftFoot':     27,
        'mixamorig:RightFoot':    28,
    }

    raw_frames = []
    for frame in mixamo_frames:
        bone_lookup = {b['name']: b for b in frame.get('bones', [])}
        joints = [{"x": 0.0, "y": 0.0, "z": 0.0, "vis": 0.0} for _ in range(33)]

        for bone_name, mp_idx in BONE_TO_MP_IDX.items():
            bone = bone_lookup.get(bone_name)
            if bone and 'rotation' in bone:
                r = bone['rotation']
                joints[mp_idx] = {"x": r['x'], "y": r['y'], "z": r['z'], "vis": r['w']}

        raw_frames.append({
            "timestamp_ms": frame['time'] * (1000.0 / fps),
            "joints": joints,
        })

    return raw_frames


def _segment_mixamo(mixamo_frames: list, fps: float) -> list:
    raw_frames = _frames_to_raw(mixamo_frames, fps=fps)

    energy, _ = compute_energy(
        raw_frames,
        smooth_window=settings.segmenter_smooth_window,
    )
    boundaries = detect_boundaries(
        energy,
        fps=fps,
        min_segment_sec=settings.segmenter_min_seg_sec,
        sensitivity=settings.segmenter_sensitivity,
    )
    return build_segments(
        raw_frames, boundaries, fps,
        energy=energy,
        min_segment_sec=settings.segmenter_min_seg_sec,
    )


def _run_blender(mixamo_json_path: str, glb_output_path: str) -> None:
    blender_script = (Path(__file__).parent / "blender_logic" / "import_and_export.py").resolve()
    character_blend = Path("/app/blender_data/character.blend")
    
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[Blender] {line}")
        if result.stderr:
            for line in result.stderr.splitlines():
                if "WARNING:" in line or "deprecated" in line.lower():
                    logger.warning(f"[Blender] {line}")
                else:
                    logger.error(f"[Blender stderr] {line}")
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender exited with code {result.returncode}.\n"
                f"STDERR:\n{result.stderr}\n"
                f"STDOUT:\n{result.stdout}"
            )
        
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Blender timeout: {e}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Blender not in PATH: {e}")
    except Exception as e:
        raise

def process_video(video_key: str) -> dict:
    animation_key, segments_key = _make_keys(video_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = str(tmpdir / Path(video_key).name)

        s3_client.download_file(video_key, video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if not Path(settings.mixamo_model_path).exists():
            raise RuntimeError(
                f"Mixamo model not found: {settings.mixamo_model_path}. "
                "Set MIXAMO_MODEL_PATH in .env"
            )

        with open(settings.mixamo_model_path, 'r', encoding='utf-8') as f:
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

        mixamo_frames = mixamo_data['frames']
        duration_sec = len(mixamo_frames) / fps if fps > 0 else 0.0

        mixamo_json_path = str(tmpdir / "mixamo.json")
        with open(mixamo_json_path, "w", encoding="utf-8") as f:
            json.dump(mixamo_data, f, ensure_ascii=False)

        glb_path = str(tmpdir / "animation.glb")
        _run_blender(mixamo_json_path, glb_path)
        segments = _segment_mixamo(mixamo_frames, fps)
      
        segments_data = {
            "version": "1.0",
            "meta": {
                "fps": fps,
                "num_frames": len(mixamo_frames),
                "duration_sec": round(duration_sec, 3),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
            "num_segments": len(segments),
            "segments": segments,
        }

        segments_path = tmpdir / "segments.json"
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False)

        s3_client.upload_file(glb_path, animation_key)
        s3_client.upload_file(str(segments_path), segments_key)

    return {
        "animation_key":  animation_key,
        "segments_key":   segments_key,
        "num_frames":     len(mixamo_frames),
        "num_segments":   len(segments),
        "duration_sec":   round(duration_sec, 3),
    }