import json
import logging
import tempfile
from pathlib import Path

from app.core import s3 as s3_client

logger = logging.getLogger(__name__)

LANDMARKS_CACHE_PREFIX = "dance-landmarks-cache"
KEYFRAMES_S3_KEY = "results/{dance_id}/keyframes.json"


def extract_and_save_keyframes(dance_id: str) -> dict:
    keyframes_key = KEYFRAMES_S3_KEY.format(dance_id=dance_id)

    if s3_client.file_exists(keyframes_key):
        logger.info(f"Keyframes already exist for {dance_id}, skipping")
        return {"dance_id": dance_id, "skipped": True}

    cache_key = f"{LANDMARKS_CACHE_PREFIX}/{dance_id}.json"
    if not s3_client.file_exists(cache_key):
        logger.warning(f"Landmarks cache not found for {dance_id}, cannot extract keyframes")
        return {"dance_id": dance_id, "skipped": True, "reason": "no_landmarks_cache"}

    tmp_landmarks = Path(tempfile.gettempdir()) / f"lm_{dance_id}.json"
    try:
        s3_client.download_file(cache_key, str(tmp_landmarks))
        with open(tmp_landmarks, encoding="utf-8") as f:
            data = json.load(f)
    finally:
        tmp_landmarks.unlink(missing_ok=True)

    frames = data.get("frames", [])
    fps = float(data.get("_fps") or data.get("fps") or 24.0)

    if not frames:
        logger.warning(f"Empty frames in landmarks cache for {dance_id}")
        return {"dance_id": dance_id, "skipped": True, "reason": "empty_frames"}

    from app.services.compare import _extract_pose_array, _find_motion_keyframes

    poses = _extract_pose_array(frames)
    if poses is None:
        logger.warning(f"Could not extract poses for {dance_id}")
        return {"dance_id": dance_id, "skipped": True, "reason": "no_poses"}

    keyframe_indices = _find_motion_keyframes(poses, n_keyframes=10)

    keyframes = [
        {
            "frame_idx": int(idx),
            "timestamp_ms": round(idx / fps * 1000, 1),
        }
        for idx in keyframe_indices
    ]

    result = {
        "dance_id": dance_id,
        "fps": fps,
        "num_frames": len(frames),
        "keyframes": keyframes,
    }

    tmp_out = Path(tempfile.gettempdir()) / f"kf_{dance_id}.json"
    try:
        with open(tmp_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        s3_client.upload_file(str(tmp_out), keyframes_key)
    finally:
        tmp_out.unlink(missing_ok=True)

    logger.info(f"Keyframes saved: {keyframes_key} ({len(keyframes)} keyframes)")
    return {
        "dance_id": dance_id,
        "num_keyframes": len(keyframes),
        "keyframes_key": keyframes_key,
    }
