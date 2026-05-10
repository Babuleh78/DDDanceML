import logging
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from dtaidistance import dtw
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False

from app.core import s3 as s3_client
from app.services.video_to_json import convert_video_to_json
from app.core.config import settings

logger = logging.getLogger(__name__)

COMPARE_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARKS_CACHE_PREFIX = "dance-landmarks-cache"


def _get_cached_landmarks(
    video_s3_path: str,
    cache_key: str,
    video_download_path: Path,
    model_json: dict,
    fps: int,
) -> dict:
    if s3_client.file_exists(cache_key):
        try:
            logger.info(f"Cache HIT: {cache_key}")
            cache_local = Path(tempfile.gettempdir()) / f"cache_{Path(cache_key).name}"
            s3_client.download_file(cache_key, str(cache_local))
            
            with open(cache_local, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            cache_local.unlink()
            return cached_data
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}. Will recompute.")
    
    logger.info(f"Cache MISS: {cache_key}. Processing video {video_s3_path}...")
    
    try:
        s3_client.download_file(video_s3_path, str(video_download_path))
        video_data = convert_video_to_json(
            video_path=str(video_download_path),
            model_json=model_json,
            fps=fps,
            min_visibility=settings.mixamo_min_visibility,
            is_hips_move=True,
            max_frames=settings.mixamo_max_frames,
            is_show_result=False,
        )
        try:
            cache_local = Path(tempfile.gettempdir()) / f"cache_{Path(cache_key).name}"
            with open(cache_local, "w", encoding="utf-8") as f:
                json.dump(video_data, f, ensure_ascii=False)
            
            s3_client.upload_file(str(cache_local), cache_key)
            cache_local.unlink()
            logger.info(f"Cached landmarks: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache landmarks: {e}. Continuing without cache.")
        
        return video_data
        
    finally:
        video_download_path.unlink(missing_ok=True)


def _extract_pose_vector(frame: dict, indices: list) -> np.ndarray:
    landmarks = frame.get("landmarks") or []
    coords = []
    for idx in indices:
        if idx < len(landmarks) and landmarks[idx] is not None:
            lm = landmarks[idx]
            coords.extend([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
        else:
            coords.extend([0, 0, 0])
    return np.array(coords, dtype=np.float32)


def _normalize_pose_sequence(frames: list) -> np.ndarray:
    poses = []
    for frame in frames:
        pose = _extract_pose_vector(frame, COMPARE_LANDMARKS)
        poses.append(pose)

    if not poses:
        return np.zeros((1, len(COMPARE_LANDMARKS) * 3), dtype=np.float32)

    poses_array = np.array(poses, dtype=np.float32)

    for i in range(poses_array.shape[1]):
        col = poses_array[:, i]
        col_min = np.min(col)
        col_max = np.max(col)
        col_range = col_max - col_min
        if col_range > 1e-6:
            poses_array[:, i] = (col - col_min) / col_range
        else:
            poses_array[:, i] = 0

    return poses_array


def _dtw_distance_euclidean(s: np.ndarray, t: np.ndarray) -> float:
    if DTAIDISTANCE_AVAILABLE:
        distance = dtw.distance(s, t)
        return float(distance)
    else:
        n, m = len(s), len(t)
        cost_matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = np.linalg.norm(s[i] - t[j])

        # DTW таблица накопления
        dtw_acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dtw_acc[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = cost_matrix[i - 1, j - 1]
                dtw_acc[i, j] = cost + min(
                    dtw_acc[i - 1, j],
                    dtw_acc[i, j - 1],
                    dtw_acc[i - 1, j - 1],
                )

        return float(dtw_acc[n, m])


def _compute_comparison_score(
    original_poses: np.ndarray,
    user_poses: np.ndarray,
) -> tuple[float, float]:
    if len(original_poses) != len(user_poses):
        if len(original_poses) < len(user_poses):
            indices = np.linspace(0, len(original_poses) - 1, len(user_poses))
            original_poses = original_poses[np.round(indices).astype(int)]
        else:
            indices = np.linspace(0, len(user_poses) - 1, len(original_poses))
            user_poses = user_poses[np.round(indices).astype(int)]

    dtw_dist = _dtw_distance_euclidean(original_poses, user_poses)
    max_seq_len = max(len(original_poses), len(user_poses), 1)
    normalized_dtw = dtw_dist / max_seq_len
    
    score = 100.0 * np.exp(-0.5 * normalized_dtw)
    
    return float(score), float(normalized_dtw)


def compare_dance(
    original_video_s3_path: str,
    user_video_s3_path: str,
    user_id: str,
    dance_id: str,
) -> dict:
    logger.info(f"compare_dance START: {original_video_s3_path} vs {user_video_s3_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        try:
            logger.info("Step 1: Downloading videos...")
            original_video_path = tmpdir / "original.mp4"
            user_video_path = tmpdir / "user.mp4"

            s3_client.download_file(original_video_s3_path, str(original_video_path))
            s3_client.download_file(user_video_s3_path, str(user_video_path))

            cap_orig = cv2.VideoCapture(str(original_video_path))
            fps_orig = cap_orig.get(cv2.CAP_PROP_FPS) or 24.0
            cap_orig.release()

            cap_user = cv2.VideoCapture(str(user_video_path))
            fps_user = cap_user.get(cv2.CAP_PROP_FPS) or 24.0
            cap_user.release()

            fps = fps_orig 
            logger.info("Step 2: Processing through MediaPipe...")
            model_path = Path(settings.mixamo_model_path)
            if not model_path.exists():
                raise RuntimeError(f"Mixamo model not found: {settings.mixamo_model_path}")

            with open(settings.mixamo_model_path, "r", encoding="utf-8") as f:
                model_json = json.load(f)

            cache_key = f"{LANDMARKS_CACHE_PREFIX}/{dance_id}.json"
            original_data = _get_cached_landmarks(
                video_s3_path=original_video_s3_path,
                cache_key=cache_key,
                video_download_path=original_video_path,
                model_json=model_json,
                fps=int(fps_orig),
            )
            original_frames = original_data.get("frames", [])

            user_data = convert_video_to_json(
                video_path=str(user_video_path),
                model_json=model_json,
                fps=int(fps_user),
                min_visibility=settings.mixamo_min_visibility,
                is_hips_move=True,
                max_frames=settings.mixamo_max_frames,
                is_show_result=False,
            )
            user_frames = user_data.get("frames", [])

            logger.info(
                f"Extracted frames: original={len(original_frames)}, user={len(user_frames)}"
            )

            logger.info("Step 3: Normalizing pose sequences...")
            original_poses = _normalize_pose_sequence(original_frames)
            user_poses = _normalize_pose_sequence(user_frames)

            logger.info(
                f"Pose matrices: original={original_poses.shape}, user={user_poses.shape}"
            )

            logger.info("Step 4: Computing DTW comparison score...")
            comparison_score, dtw_distance = _compute_comparison_score(
                original_poses, user_poses
            )

            logger.info(f"Comparison score: {comparison_score}/100, DTW distance: {dtw_distance}")

            logger.info("Step 5: Rendering 3D model for user video...")
            from app.services.processing import _run_blender

            user_glb_path = tmpdir / "user_animation.glb"
            user_json_path = tmpdir / "user_animation.json"

            with open(user_json_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False)

            num_frames = len(user_frames)
            _run_blender(str(user_json_path), str(user_glb_path), num_frames=num_frames)

            logger.info("Step 6: Uploading results to S3...")
            user_s3_dir = f"users/{user_id}/{dance_id}"

            user_glb_s3 = f"{user_s3_dir}/user_animation.glb"
            s3_client.upload_file(str(user_glb_path), user_glb_s3)

            result_data = {
                "dance_id": dance_id,
                "user_id": user_id,
                "comparison_score": comparison_score,
                "dtw_distance": dtw_distance,
                "original_video_s3": original_video_s3_path,
                "user_video_s3": user_video_s3_path,
                "user_glb_s3": user_glb_s3,
                "original_frames_count": len(original_frames),
                "user_frames_count": len(user_frames),
                "fps_original": fps_orig,
                "fps_user": fps_user,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            result_json_s3 = f"{user_s3_dir}/comparison_result.json"
            result_json_path = tmpdir / "comparison_result.json"
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            s3_client.upload_file(str(result_json_path), result_json_s3)

            logger.info(f"Comparison complete. Results saved to {user_s3_dir}/")

            return {
                "success": True,
                "dance_id": dance_id,
                "user_id": user_id,
                "comparison_score": comparison_score,
                "dtw_distance": dtw_distance,
                "user_glb_s3": user_glb_s3,
                "result_json_s3": result_json_s3,
            }

        except Exception as e:
            logger.error(f"compare_dance failed: {e}", exc_info=True)
            return {
                "success": False,
                "dance_id": dance_id,
                "user_id": user_id,
                "error": str(e),
            }
