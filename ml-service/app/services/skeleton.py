import json
import logging
from pathlib import Path
from typing import Optional

from app.core import s3 as s3_client

logger = logging.getLogger(__name__)


def build_skeleton_payload(mixamo_data: dict, fps: float) -> Optional[dict]:
    frames_2d = []
    for fi, frame in enumerate(mixamo_data.get("frames", [])):
        lm2d = frame.get("lm2d")
        if not lm2d:
            continue
        compact = [
            [round(p["x"], 5), round(p["y"], 5), round(p.get("v", 1.0), 3)]
            for p in lm2d
        ]
        frames_2d.append({
            "f": frame.get("time", fi),
            "t": round((frame.get("time", fi)) / max(fps, 1.0), 3),
            "lm": compact,
        })

    if not frames_2d:
        return None

    return {
        "fps": float(fps),
        "width": int(mixamo_data.get("width") or 0),
        "height": int(mixamo_data.get("height") or 0),
        "num_frames": len(mixamo_data.get("frames", [])),
        "frames": frames_2d,
    }


def save_skeleton_json(
    mixamo_data: dict,
    fps: float,
    s3_key: str,
    tmpdir: Path,
    local_name: str = "skeleton.json",
) -> bool:
    payload = build_skeleton_payload(mixamo_data, fps)
    if payload is None:
        logger.warning(f"No lm2d data in frames, skipping skeleton upload: {s3_key}")
        return False

    local_path = Path(tmpdir) / local_name
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    try:
        s3_client.upload_file(str(local_path), s3_key)
        logger.info(f"Skeleton uploaded: {s3_key} ({len(payload['frames'])} frames)")
        return True
    except Exception as e:
        logger.error(f"Failed to upload skeleton.json to {s3_key}: {e}")
        return False
