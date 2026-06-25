import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import cdist

from app.domain.comparison import ComparisonResult  # noqa: F401 — domain type, gradual migration

try:
    from dtaidistance import dtw_ndim
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False

from app.core import s3 as s3_client
from app.core.config import settings
from app.services.skeleton import save_skeleton_json
from app.services.video_to_json import convert_video_to_json

logger = logging.getLogger(__name__)

COMPARE_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARKS_CACHE_PREFIX = "dance-landmarks-cache"

JOINT_TRIPLETS = [
    (6, 0, 2),
    (7, 1, 3),
    (0, 2, 4),
    (1, 3, 5),
    (0, 6, 8),
    (1, 7, 9),
    (6, 8, 10),
    (7, 9, 11),
]

HIT_THRESHOLD = 85.0
_BODY_LANDMARKS = (0, 1, 6, 7)
_LIMB_LANDMARKS = (2, 3, 4, 5, 8, 9, 10, 11)
_BODY_WEIGHT_TOTAL = 0.15
_LIMB_WEIGHT_TOTAL = 0.85


_COORDS_PER_LANDMARK = 3


def _build_landmark_weights() -> np.ndarray:
    n_coords = len(COMPARE_LANDMARKS) * _COORDS_PER_LANDMARK
    w = np.zeros(n_coords, dtype=np.float32)
    body_per_coord = _BODY_WEIGHT_TOTAL / (len(_BODY_LANDMARKS) * _COORDS_PER_LANDMARK)
    limb_per_coord = _LIMB_WEIGHT_TOTAL / (len(_LIMB_LANDMARKS) * _COORDS_PER_LANDMARK)
    for lm in _BODY_LANDMARKS:
        w[lm * _COORDS_PER_LANDMARK : (lm + 1) * _COORDS_PER_LANDMARK] = body_per_coord
    for lm in _LIMB_LANDMARKS:
        w[lm * _COORDS_PER_LANDMARK : (lm + 1) * _COORDS_PER_LANDMARK] = limb_per_coord
    return w


_LANDMARK_WEIGHTS = _build_landmark_weights()


def _weighted_cosine(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-8
    w = _LANDMARK_WEIGHTS
    num = float(np.sum(w * a * b))
    na = float(np.sqrt(np.sum(w * a * a)))
    nb = float(np.sqrt(np.sum(w * b * b)))
    if na < eps or nb < eps:
        return 0.0
    return num / (na * nb)


def _get_cached_landmarks(
    video_s3_path: str,
    cache_key: str,
    video_download_path: Path,
    model_json: dict,
    fps: int,
) -> dict:
    if s3_client.file_exists(cache_key):
        try:
            dance_id_hint = Path(cache_key).stem
            logger.info(f"Cache HIT: {cache_key}")
            logger.info(f"Using cached landmarks for dance {dance_id_hint}, saved ~30s")
            cache_local = Path(tempfile.gettempdir()) / f"cache_{Path(cache_key).name}"
            s3_client.download_file(cache_key, str(cache_local))
            
            with open(cache_local, encoding="utf-8") as f:
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


def _frame_detected(frame: dict) -> bool:
    return bool(frame.get("detected", True))


def _detected_mask(frames: list) -> np.ndarray:
    return np.array([_frame_detected(f) for f in frames], dtype=bool)


def _make_not_detected_label(abs_frame: int, fps_user: float) -> dict:
    return {
        "frame_idx": int(abs_frame),
        "timestamp_ms": round(abs_frame / fps_user * 1000, 1),
        "hit": False,
        "reason": "not_detected",
        "timing_score": 0.0,
        "amplitude_score": 0.0,
        "pose_score": 0.0,
        "joint_errors": [1.0] * len(JOINT_TRIPLETS),
    }


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


def _extract_pose_array(frames: list) -> Optional[np.ndarray]:
    poses = [_extract_pose_vector(frame, COMPARE_LANDMARKS) for frame in frames]
    if not poses:
        return None
    return np.array(poses, dtype=np.float32)


def _joint_normalize(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([a, b], axis=0)
    col_min = combined.min(axis=0)
    col_max = combined.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-6, col_max - col_min, 1.0)
    return (
        ((a - col_min) / col_range).astype(np.float32),
        ((b - col_min) / col_range).astype(np.float32),
    )


def _dtw_distance_euclidean(s: np.ndarray, t: np.ndarray) -> float:
    if DTAIDISTANCE_AVAILABLE:
        distance = dtw_ndim.distance(s, t)
        return float(distance)
    else:
        n, m = len(s), len(t)
        cost_matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = np.linalg.norm(s[i] - t[j])

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


def _resample_poses(poses: np.ndarray, target_len: int) -> np.ndarray:
    if len(poses) == target_len:
        return poses
    t_src = np.linspace(0, 1, len(poses))
    t_dst = np.linspace(0, 1, target_len)
    return np.column_stack([
        np.interp(t_dst, t_src, poses[:, col])
        for col in range(poses.shape[1])
    ]).astype(np.float32)


def _compute_comparison_score(
    original_poses: np.ndarray,
    user_poses: np.ndarray,
) -> tuple[float, float]:
    if len(original_poses) != len(user_poses):
        target_len = max(len(original_poses), len(user_poses))
        original_poses = _resample_poses(original_poses, target_len)
        user_poses = _resample_poses(user_poses, target_len)

    dtw_dist = _dtw_distance_euclidean(original_poses, user_poses)
    seq_len = max(len(original_poses), 1)
    normalized_dtw = dtw_dist / seq_len

    score = 100.0 * np.exp(-(normalized_dtw ** 2))

    return float(score), float(normalized_dtw)


def _compute_dtw_with_path(s: np.ndarray, t: np.ndarray) -> tuple[float, list]:
    n, m = len(s), len(t)
    cost = cdist(s.astype(np.float64), t.astype(np.float64), metric="euclidean").astype(np.float32)

    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = float(cost[i - 1, j - 1]) + min(
                acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        min_val = min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        if acc[i - 1, j - 1] == min_val:
            i -= 1; j -= 1
        elif acc[i - 1, j] == min_val:
            i -= 1
        else:
            j -= 1
    while i > 0:
        path.append((i - 1, 0)); i -= 1
    while j > 0:
        path.append((0, j - 1)); j -= 1
    path.reverse()

    return float(acc[n, m]), path


def _compute_joint_angles(poses: np.ndarray) -> np.ndarray:
    angles = []
    for (a, b, c) in JOINT_TRIPLETS:
        pa = poses[:, a * 3: a * 3 + 3]
        pb = poses[:, b * 3: b * 3 + 3]
        pc = poses[:, c * 3: c * 3 + 3]
        ba = pa - pb
        bc = pc - pb
        ba = ba / (np.linalg.norm(ba, axis=1, keepdims=True) + 1e-8)
        bc = bc / (np.linalg.norm(bc, axis=1, keepdims=True) + 1e-8)
        cos_a = np.clip(np.sum(ba * bc, axis=1), -1.0, 1.0)
        angles.append(np.arccos(cos_a))
    return np.column_stack(angles)


def _find_motion_keyframes(poses: np.ndarray, n_keyframes: int = 5) -> list[int]:
    if len(poses) < 3:
        return [len(poses) // 2]
    velocity = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    if len(velocity) >= 5:
        win = min(len(velocity), 11)
        if win % 2 == 0:
            win -= 1
        win = max(win, 3)
        velocity = savgol_filter(velocity, window_length=win, polyorder=2)
    min_dist = max(1, len(velocity) // (n_keyframes + 1))
    peaks, _ = find_peaks(velocity, distance=min_dist)
    if len(peaks) == 0:
        return [len(poses) // 2]
    if len(peaks) > n_keyframes:
        peaks = sorted(peaks[np.argsort(velocity[peaks])[-n_keyframes:]])
    return [int(p) for p in peaks]


def _timing_score(
    path: list,
    n: int,
    m: int,
    orig_norm: Optional[np.ndarray] = None,
    user_norm: Optional[np.ndarray] = None,
) -> float:
    if orig_norm is not None and user_norm is not None and len(orig_norm) > 2 and len(user_norm) > 2:
        orig_vel = np.linalg.norm(np.diff(orig_norm, axis=0), axis=1)
        user_vel = np.linalg.norm(np.diff(user_norm, axis=0), axis=1)
        T = max(len(orig_vel), len(user_vel))
        t_src = np.linspace(0, 1, len(orig_vel))
        t_dst = np.linspace(0, 1, T)
        if len(orig_vel) != T:
            orig_vel = np.interp(t_dst, t_src, orig_vel)
        if len(user_vel) != T:
            t_src_u = np.linspace(0, 1, len(user_vel) if len(user_vel) > 1 else 2)
            user_vel = np.interp(t_dst, t_src_u, user_vel)
        std_o = float(orig_vel.std())
        std_u = float(user_vel.std())
        if std_o < 1e-6 or std_u < 1e-6:
            return 50.0 
        corr = float(np.corrcoef(orig_vel, user_vel)[0, 1])
        return float(100.0 * max(0.0, corr))

    if not path or n <= 1 or m <= 1:
        return 100.0
    deviations = [abs(i / (n - 1) - j / (m - 1)) for (i, j) in path]
    return float(100.0 * np.exp(-3.0 * float(np.mean(deviations))))


def _center_unit_poses(poses: np.ndarray) -> np.ndarray:
    eps = 1e-8
    n_lm = len(COMPARE_LANDMARKS)
    T = len(poses)
    p3d = poses.reshape(T, n_lm, 3)
    centroid = p3d.mean(axis=1, keepdims=True)
    centered = (p3d - centroid).reshape(T, -1)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / (norms + eps)).astype(np.float32)


def _amplitude_score(orig_norm: np.ndarray, user_norm: np.ndarray, path: list) -> float:
    orig_u = _center_unit_poses(orig_norm)
    user_u = _center_unit_poses(user_norm)
    cos_sims = [
        float(np.clip(_weighted_cosine(orig_u[i], user_u[j]), -1.0, 1.0))
        for (i, j) in path
        if i < len(orig_u) and j < len(user_u)
    ]
    if not cos_sims:
        return 50.0
    mean_cos = float(np.mean(cos_sims))
    return float(100.0 * float(np.clip(mean_cos, 0.0, 1.0)) ** 2)


def _pose_accuracy_score(orig_norm: np.ndarray, user_norm: np.ndarray, path: list) -> float:
    keyframes = _find_motion_keyframes(orig_norm, n_keyframes=5)
    path_map: dict[int, int] = {}
    for (i, j) in path:
        if i not in path_map:
            path_map[i] = j

    orig_u = _center_unit_poses(orig_norm)
    user_u = _center_unit_poses(user_norm)

    scores = []
    for kf in keyframes:
        kf = min(kf, len(orig_norm) - 1)
        user_kf = path_map.get(kf)
        if user_kf is None:
            closest = min(path, key=lambda p: abs(p[0] - kf), default=None)
            user_kf = closest[1] if closest else 0
        user_kf = min(user_kf, len(user_norm) - 1)
        cos = float(np.clip(_weighted_cosine(orig_u[kf], user_u[user_kf]), -1.0, 1.0))
        scores.append(100.0 * float(np.clip(cos, 0.0, 1.0)) ** 2)

    return float(np.mean(scores)) if scores else 50.0


def _compute_feedback(timing_score: float, amplitude_score: float, path: list, n: int, m: int) -> str:
    if timing_score > 75:
        return "low_amplitude" if amplitude_score <= 50 else "on_time"
    if path and n > 1 and m > 1:
        signed_devs = [(i / (n - 1)) - (j / (m - 1)) for (i, j) in path]
        return "late" if float(np.mean(signed_devs)) > 0 else "early"
    return "late"


def _load_original_segments(dance_id: str) -> list:
    segments_key = f"results/{dance_id}/segments.json"
    try:
        if not s3_client.file_exists(segments_key):
            return []
        tmp = Path(tempfile.gettempdir()) / f"segs_{dance_id}.json"
        s3_client.download_file(segments_key, str(tmp))
        with open(tmp, encoding="utf-8") as f:
            data = json.load(f)
        tmp.unlink(missing_ok=True)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("segments", [])
    except Exception as e:
        logger.warning(f"Could not load segments for {dance_id}: {e}")
    return []


def _compute_frame_labels(
    orig_seg_raw: np.ndarray,
    user_seg_raw: np.ndarray,
    orig_seg_norm: np.ndarray,
    user_seg_norm: np.ndarray,
    path: list,
    fps_user: float,
    user_abs_frame_idx: list,
) -> list[dict]:
    n, m = len(orig_seg_raw), len(user_seg_raw)
    if n < 2 or m < 2 or not path:
        return []

    orig_angles = _compute_joint_angles(orig_seg_raw)
    user_angles = _compute_joint_angles(user_seg_raw)

    eps = 1e-8
    orig_unit = orig_seg_norm / (np.linalg.norm(orig_seg_norm, axis=1, keepdims=True) + eps)
    user_unit = user_seg_norm / (np.linalg.norm(user_seg_norm, axis=1, keepdims=True) + eps)
    orig_cu = _center_unit_poses(orig_seg_norm)
    user_cu = _center_unit_poses(user_seg_norm)

    user_to_orig: dict[int, tuple[int, float]] = {}
    for (i, j) in path:
        i_norm = i / (n - 1)
        j_norm = j / (m - 1)
        dev = abs(i_norm - j_norm)
        if j not in user_to_orig or dev < user_to_orig[j][1]:
            user_to_orig[j] = (i, dev)

    labels = []
    for j in range(m):
        abs_frame = int(user_abs_frame_idx[j])
        ts_ms = round(abs_frame / fps_user * 1000, 1)

        mapped = user_to_orig.get(j)
        if mapped is None:
            labels.append({
                "frame_idx": abs_frame, "timestamp_ms": ts_ms,
                "hit": False, "reason": "timing",
                "timing_score": 0.0, "amplitude_score": 0.0, "pose_score": 0.0,
                "joint_errors": [1.0] * len(JOINT_TRIPLETS),
            })
            continue

        i, timing_dev = mapped
        timing = round(float(100.0 * np.exp(-3.0 * timing_dev)), 2)

        per_joint_delta = np.abs(orig_angles[i] - user_angles[j])
        joint_errors = [
            round(float(np.clip(d / np.pi, 0.0, 1.0)), 3)
            for d in per_joint_delta
        ]

        cos_amp = float(np.clip(_weighted_cosine(orig_cu[i], user_cu[j]), -1.0, 1.0))
        amplitude = round(float(100.0 * float(np.clip(cos_amp, 0.0, 1.0)) ** 2), 2)

        cos_sim = float(np.clip(_weighted_cosine(orig_unit[i], user_unit[j]), -1.0, 1.0))
        pose = round(float(50.0 * (cos_sim + 1.0)), 2)

        overall = (timing + amplitude + pose) / 3.0
        hit = overall >= HIT_THRESHOLD
        reason = None
        if not hit:
            reason = min(
                [("timing", timing), ("amplitude", amplitude), ("pose", pose)],
                key=lambda x: x[1],
            )[0]

        labels.append({
            "frame_idx": abs_frame,
            "timestamp_ms": ts_ms,
            "hit": hit,
            "reason": reason,
            "timing_score": timing,
            "amplitude_score": amplitude,
            "pose_score": pose,
            "joint_errors": joint_errors,
        })

    return labels


def _analyze_segment(
    orig_seg_raw: np.ndarray,
    user_seg_raw: np.ndarray,
    fps_user: float,
    user_start_frame: int,
    user_detected_mask: Optional[np.ndarray] = None,
) -> tuple[dict, list]:
    default = {
        "timing": 0.0, "amplitude": 0.0,
        "pose_accuracy": 0.0, "score": 0.0,
        "feedback": "not_detected",
    }
    n_user_total = len(user_seg_raw)
    if user_detected_mask is None:
        user_detected_mask = np.ones(n_user_total, dtype=bool)

    detected_local_idx = np.where(user_detected_mask)[0]
    if len(orig_seg_raw) < 2 or len(detected_local_idx) < 2:
      
        nd_labels = [
            _make_not_detected_label(user_start_frame + j, fps_user)
            for j in range(n_user_total) if not user_detected_mask[j]
        ]
        return default, nd_labels

    user_filtered = user_seg_raw[detected_local_idx]
    user_abs_idx = [user_start_frame + int(j) for j in detected_local_idx]

    orig_norm, user_norm = _joint_normalize(orig_seg_raw, user_filtered)
    _, path = _compute_dtw_with_path(orig_norm, user_norm)

    timing = _timing_score(path, len(orig_norm), len(user_norm), orig_norm, user_norm)
    amplitude = _amplitude_score(orig_norm, user_norm, path)
    pose_acc = _pose_accuracy_score(orig_norm, user_norm, path)
    segment_score = 0.2 * timing + 0.4 * amplitude + 0.4 * pose_acc
    feedback = _compute_feedback(timing, amplitude, path, len(orig_norm), len(user_norm))

    scores = {
        "timing": round(timing, 2),
        "amplitude": round(amplitude, 2),
        "pose_accuracy": round(pose_acc, 2),
        "score": round(segment_score, 2),
        "feedback": feedback,
    }
    detected_labels = _compute_frame_labels(
        orig_seg_raw, user_filtered, orig_norm, user_norm,
        path, fps_user, user_abs_idx,
    )

    detected_set = set(int(j) for j in detected_local_idx)
    nd_labels = [
        _make_not_detected_label(user_start_frame + j, fps_user)
        for j in range(n_user_total) if j not in detected_set
    ]
    all_labels = sorted(detected_labels + nd_labels, key=lambda x: x["frame_idx"])
    return scores, all_labels


def _generate_tips(comparison_score: float, segments: list) -> list:
    tips = []

    for seg in segments:
        n = seg.get("segment_id", "?")
        feedback = seg.get("feedback", "on_time")
        score = float(seg.get("score", 100.0))

        if feedback == "low_amplitude":
            tips.append({"type": "warn", "text": f"В сегменте {n} амплитуда движений значительно ниже эталона."})
        elif feedback == "early":
            tips.append({"type": "warn", "text": f"В сегменте {n} движения опережают ритм эталона."})
        elif feedback == "late":
            tips.append({"type": "warn", "text": f"В сегменте {n} движения отстают от ритма эталона."})

        if score >= 85 and feedback == "on_time":
            tips.append({"type": "info", "text": f"Сегмент {n}: отличное исполнение."})

    if segments:
        avg_timing = sum(float(s.get("timing", 0)) for s in segments) / len(segments)
        has_timing_issues = any(s.get("feedback") in ("early", "late") for s in segments)
        if avg_timing >= 75 and not has_timing_issues:
            tips.append({"type": "info", "text": "Хорошая синхронизация ритма."})

    if comparison_score >= 70:
        tips.append({"type": "info", "text": "Отличное исполнение — высокое совпадение с эталоном."})
    elif comparison_score >= 50:
        tips.append({"type": "info", "text": "Хорошая общая синхронизация."})
    elif comparison_score < 30:
        tips.append({"type": "warn", "text": "Значительное расхождение с эталоном. Рекомендуется повторить ключевые движения."})

    return tips


def compare_dance(
    original_video_s3_path: str,
    user_video_s3_path: str,
    user_id: str,
    dance_id: str,
    attempt_id: str = None,
    on_progress=None,
) -> dict:
    attempt_key = attempt_id or dance_id

    def _progress(stage: str, progress: int, label: str):
        if on_progress:
            try:
                on_progress(stage, progress, label)
            except Exception:
                pass

    logger.info(f"compare_dance START: {original_video_s3_path} vs {user_video_s3_path} (attempt={attempt_key})")

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        try:
            start_time = datetime.now(timezone.utc)
            logger.info("Step 1: Downloading videos...")
            original_video_path = tmpdir / "original.mp4"
            user_video_path = tmpdir / "user.mp4"

            s3_client.download_file(original_video_s3_path, str(original_video_path))
            s3_client.download_file(user_video_s3_path, str(user_video_path))
            _progress("pose_extraction", 20, "Извлечение движений")

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

            with open(settings.mixamo_model_path, encoding="utf-8") as f:
                model_json = json.load(f)

            cache_key = f"{LANDMARKS_CACHE_PREFIX}/{dance_id}.json"
            logger.info("Step 2a: loading reference landmarks (cache_key=%s)", cache_key)
            original_data = _get_cached_landmarks(
                video_s3_path=original_video_s3_path,
                cache_key=cache_key,
                video_download_path=original_video_path,
                model_json=model_json,
                fps=int(fps_orig),
            )
            original_frames = original_data.get("frames", [])
            logger.info("Step 2a done: reference frames=%d", len(original_frames))

            logger.info(
                "Step 2b: running MediaPipe on user video (path=%s, fps=%d, max_frames=%s)",
                user_video_path, int(fps_user), settings.mixamo_max_frames,
            )
            _mp_t0 = time.time()
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
                "Step 2b done in %.1fs: user frames=%d",
                time.time() - _mp_t0, len(user_frames),
            )

            original_detected = _detected_mask(original_frames)
            user_detected = _detected_mask(user_frames)
            frames_missing = int((~user_detected).sum()) if len(user_detected) else 0

            logger.info(
                f"Extracted frames: original={len(original_frames)} "
                f"(detected={int(original_detected.sum())}), "
                f"user={len(user_frames)} (detected={int(user_detected.sum())}, missing={frames_missing})"
            )

            _progress("comparing", 45, "Сравнение движений")
            logger.info("Step 3: Extracting and jointly normalizing pose sequences...")
            original_poses = _extract_pose_array(original_frames)
            user_poses = _extract_pose_array(user_frames)

            if original_poses is None:
                raise ValueError("No detectable pose in original video — skeleton extraction returned 0 frames.")
            if user_poses is None:
                raise ValueError("No detectable pose in user video — check lighting, occlusion, or camera distance.")

            duration_orig = len(original_frames) / fps_orig
            duration_user = len(user_frames) / fps_user
            coverage_ratio = min(duration_user / max(duration_orig, 1e-3), 1.0)
            coverage_orig_frames = int(coverage_ratio * len(original_frames))
            logger.info(
                f"Coverage: user={duration_user:.1f}s / orig={duration_orig:.1f}s "
                f"= {coverage_ratio:.1%} ({coverage_orig_frames}/{len(original_frames)} frames)"
            )

            coverage_mask = np.arange(len(original_frames)) < coverage_orig_frames
            covered_orig_detected = original_detected & coverage_mask
            orig_for_dtw = (original_poses[covered_orig_detected]
                            if covered_orig_detected.any()
                            else original_poses[:coverage_orig_frames])
            user_for_dtw = user_poses[user_detected] if user_detected.any() else user_poses
            if len(orig_for_dtw) < 2 or len(user_for_dtw) < 2:
                raise ValueError("Too few detected frames for DTW comparison.")

            orig_for_dtw, user_for_dtw = _joint_normalize(orig_for_dtw, user_for_dtw)

            logger.info(
                f"Pose matrices (detected only): original={orig_for_dtw.shape}, user={user_for_dtw.shape}"
            )

            logger.info("Step 4: Computing DTW comparison score...")
            comparison_score, dtw_distance = _compute_comparison_score(
                orig_for_dtw, user_for_dtw
            )

            logger.info(f"Comparison score: {comparison_score}/100, DTW distance: {dtw_distance}")

            logger.info("Step 5: Computing per-segment diagnostics...")
            original_segments = _load_original_segments(dance_id)
            if not original_segments:
                n_segs = 4
                seg_size = max(1, len(original_frames) // n_segs)
                original_segments = [
                    {
                        "index": i,
                        "start_frame": i * seg_size,
                        "end_frame": min((i + 1) * seg_size, len(original_frames) - 1),
                    }
                    for i in range(n_segs)
                ]

            segment_results = []
            all_frame_labels = []
            for seg in original_segments:
                seg_idx = seg.get("index", 0)
                orig_start = seg.get("start_frame", 0)
                orig_end = min(seg.get("end_frame", len(original_frames) - 1), len(original_frames) - 1)

                if orig_start >= coverage_orig_frames:
                    scores = {
                        "timing": 0.0, "amplitude": 0.0,
                        "pose_accuracy": 0.0, "score": 0.0,
                        "feedback": "not_performed",
                        "partial_coverage": 0.0,
                    }
                    user_start = len(user_frames) - 1
                    user_end = len(user_frames) - 1
                else:
                    user_start = min(int(orig_start * fps_user / fps_orig), len(user_frames) - 1)

                    compare_orig_end = min(orig_end, coverage_orig_frames - 1)
                    user_end = min(int(compare_orig_end * fps_user / fps_orig), len(user_frames) - 1)

                    seg_len = orig_end - orig_start + 1
                    covered_len = compare_orig_end - orig_start + 1
                    partial_coverage = covered_len / max(seg_len, 1)

                    orig_seg_poses = _extract_pose_array(original_frames[orig_start: compare_orig_end + 1])
                    user_seg_poses = _extract_pose_array(user_frames[user_start: user_end + 1])
                    orig_seg_mask = original_detected[orig_start: compare_orig_end + 1]
                    user_seg_mask = user_detected[user_start: user_end + 1]

                    if orig_seg_poses is None or user_seg_poses is None:
                        raw_scores = {
                            "timing": 0.0, "amplitude": 0.0,
                            "pose_accuracy": 0.0, "score": 0.0,
                            "feedback": "not_detected",
                        }
                    else:
                        orig_detected_idx = np.where(orig_seg_mask)[0]
                        orig_seg_for_dtw = (orig_seg_poses[orig_detected_idx]
                                            if len(orig_detected_idx) >= 2
                                            else orig_seg_poses)
                        raw_scores, seg_labels = _analyze_segment(
                            orig_seg_for_dtw, user_seg_poses, fps_user, user_start, user_seg_mask
                        )
                        all_frame_labels.extend(seg_labels)

                    scores = {
                        **raw_scores,
                        "score": round(raw_scores["score"] * partial_coverage, 2),
                        "partial_coverage": round(partial_coverage, 3),
                    }

                segment_results.append({
                    "segment_id": seg_idx + 1,
                    "label": seg.get("label", f"Сегмент {seg_idx + 1}"),
                    "orig_start_frame": orig_start,
                    "orig_end_frame": orig_end,
                    "user_start_frame": user_start,
                    "user_end_frame": user_end,
                    "orig_start_ms": round(orig_start / fps_orig * 1000, 1),
                    "orig_end_ms": round(orig_end / fps_orig * 1000, 1),
                    "user_start_ms": round(user_start / fps_user * 1000, 1),
                    "user_end_ms": round(user_end / fps_user * 1000, 1),
                    **scores,
                })
            logger.info(f"Segment diagnostics done: {len(segment_results)} segments")

            if segment_results:
                dtw_based_score = comparison_score
                segment_mean = float(np.mean([s["score"] for s in segment_results]))
                dtw_factor = float(np.exp(-5.0 * dtw_distance ** 2))
                comparison_score = round(segment_mean * dtw_factor, 2)
                logger.info(
                    f"Score: DTW-global={dtw_based_score:.2f}, "
                    f"segment-mean={segment_mean:.2f}, "
                    f"dtw_factor={dtw_factor:.3f} → final={comparison_score:.2f}"
                )

            _progress("animation_render", 70, "Рендер 3D-анимации")
            logger.info("Step 6: Rendering 3D model for user video (Blender)...")
            from app.services.processing import _run_blender

            user_glb_path = tmpdir / "user_animation.glb"
            user_json_path = tmpdir / "user_animation.json"

            with open(user_json_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False)

            num_frames = len(user_frames)
            _run_blender(str(user_json_path), str(user_glb_path), num_frames=num_frames)

            _progress("saving", 90, "Сохранение результатов")
            logger.info("Step 7: Uploading results to S3...")
            user_s3_dir = f"users/{user_id}/{attempt_key}"

            user_glb_s3 = f"{user_s3_dir}/user_animation.glb"
            s3_client.upload_file(str(user_glb_path), user_glb_s3)

            try:
                save_skeleton_json(
                    mixamo_data=user_data,
                    fps=float(fps_user),
                    s3_key=f"{user_s3_dir}/skeleton.json",
                    tmpdir=tmpdir,
                    local_name="user_skeleton.json",
                )
            except Exception as e:
                logger.warning(f"user skeleton.json upload failed: {e}")

            processing_time_sec = round(
                (datetime.now(timezone.utc) - start_time).total_seconds(), 2
            )
            meta = {
                "duration_sec": round(duration_user, 3),
                "orig_duration_sec": round(duration_orig, 3),
                "coverage_ratio": round(coverage_ratio, 3),
                "fps": fps_user,
                "frames_total": len(user_frames),
                "frames_missing": frames_missing,
                "processing_time_sec": processing_time_sec,
            }

            processed_at = datetime.now(timezone.utc).isoformat()
            tips = _generate_tips(comparison_score, segment_results)

            all_frame_labels.sort(key=lambda x: x["frame_idx"])
            result_data = {
                "dance_id": dance_id,
                "user_id": user_id,
                "comparison_score": comparison_score,
                "dtw_distance": dtw_distance,
                "segments": segment_results,
                "frame_labels": all_frame_labels,
                "tips": tips,
                "meta": meta,
                "original_video_s3": original_video_s3_path,
                "user_video_s3": user_video_s3_path,
                "user_glb_s3": user_glb_s3,
                "original_frames_count": len(original_frames),
                "user_frames_count": len(user_frames),
                "fps_original": fps_orig,
                "fps_user": fps_user,
                "processed_at": processed_at,
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
                "original_video_s3": original_video_s3_path,
                "user_video_s3": user_video_s3_path,
                "user_glb_s3": user_glb_s3,
                "processed_at": processed_at,
                "segments": segment_results,
                "tips": tips,
                "meta": meta,
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
