import json
import sys
import os
from pathlib import Path



from app.services.py_module.mp_manager import MediapipeManager
from app.services.py_module.mp_helper import mediapipe_to_mixamo


def convert_video_to_json(
    video_path: str,
    model_json: dict,
    fps: int = 30,
    min_visibility: float = 0.6,
    is_hips_move: bool = True,
    max_frames: int = 5000,
    is_show_result: bool = False,
) -> dict:
    mp_manager = MediapipeManager()
    mp_manager.fps             = fps
    mp_manager.min_visibility  = min_visibility
    mp_manager.is_hips_move    = is_hips_move
    mp_manager.is_show_result  = is_show_result
    mp_manager.factor          = 0.0
    mp_manager.max_frame_num   = max_frames

    model_json_str = json.dumps(
        model_json if 'node' in model_json else {"node": model_json}
    )

    success, result = mediapipe_to_mixamo(
        mp_manager,
        model_json_str,
        video_path,
    )

    if not success or result is None:
        raise RuntimeError(
            f"video_to_mixamo failed for {video_path}. "
            "Проверь что видео содержит человека в кадре."
        )

    return result