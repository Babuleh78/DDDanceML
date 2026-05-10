import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)



def extract_positions_for_bones(
    frames: List[Dict],
    bone_indices: List[int],
) -> np.ndarray:
    num_frames = len(frames)
    num_bones = len(bone_indices)
    positions = np.zeros((num_frames, num_bones, 3), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        landmarks = frame.get("landmarks")
        if not landmarks:
            continue
        for bone_pos, bone_idx in enumerate(bone_indices):
            if bone_idx < len(landmarks) and landmarks[bone_idx] is not None:
                lm = landmarks[bone_idx]
                positions[frame_idx, bone_pos] = [lm["x"], lm["y"], lm["z"]]

    return positions


def extract_all_positions(frames: List[Dict], n_bones: int = 26) -> np.ndarray:
    num_frames = len(frames)
    positions = np.zeros((num_frames, n_bones, 3), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        landmarks = frame.get("landmarks")
        if not landmarks:
            continue
        for bone_idx in range(min(n_bones, len(landmarks))):
            if landmarks[bone_idx] is not None:
                lm = landmarks[bone_idx]
                positions[frame_idx, bone_idx] = [lm["x"], lm["y"], lm["z"]]

    return positions



def smooth_signal(signal: np.ndarray, fps: float) -> np.ndarray:
    n = len(signal)
    if n < 5:
        return signal
    window = max(5, int(fps * 0.1) | 1)
    window = min(window, n if n % 2 == 1 else n - 1)
    try:
        return savgol_filter(signal, window_length=window, polyorder=2, axis=0)
    except Exception:
        return signal


def compute_velocity(positions: np.ndarray, fps: float) -> np.ndarray:
    if len(positions) < 2:
        return np.zeros((1, positions.shape[1]), dtype=np.float32)
    diffs = np.diff(positions, axis=0)
    return np.linalg.norm(diffs, axis=2) * fps


def compute_acceleration(velocity: np.ndarray, fps: float) -> np.ndarray:
    if len(velocity) < 2:
        return np.zeros((1, velocity.shape[1]), dtype=np.float32)
    return np.diff(velocity, axis=0) * fps


def compute_jerk(acceleration: np.ndarray, fps: float) -> np.ndarray:
    if len(acceleration) < 2:
        return np.zeros((1, acceleration.shape[1]), dtype=np.float32)
    return np.diff(acceleration, axis=0) * fps


def compute_range_of_motion(positions: np.ndarray) -> Dict:
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    ranges = max_pos - min_pos           # (num_bones, 3)
    distances = np.linalg.norm(ranges, axis=1)  # (num_bones,)

    return {
        "max_distance": float(np.max(distances)),
        "mean_distance": float(np.mean(distances)),
        "min_distance": float(np.min(distances)),
        "axis_range": {
            "x": float(np.max(ranges[:, 0])),
            "y": float(np.max(ranges[:, 1])),
            "z": float(np.max(ranges[:, 2])),
        },
    }


def compute_direction(positions: np.ndarray) -> Dict:
    if len(positions) < 4:
        return {"dominant_axis": "none", "direction_label": "нет движения", "displacement_m": 0.0}

    mid = len(positions) // 2
    start_mean = positions[:mid].mean(axis=(0, 1))   # (3,)
    end_mean   = positions[mid:].mean(axis=(0, 1))   # (3,)
    delta = end_mean - start_mean                    # (3,)

    displacement = float(np.linalg.norm(delta))

    axis_labels = {0: "x", 1: "y", 2: "z"}
    dominant_idx = int(np.argmax(np.abs(delta)))
    dominant_axis = axis_labels[dominant_idx]

    direction_map = {
        "x": ("вправо", "влево"),
        "y": ("вверх", "вниз"),
        "z": ("вперёд", "назад"),
    }
    pos_label, neg_label = direction_map[dominant_axis]
    direction_label = pos_label if delta[dominant_idx] > 0 else neg_label

    return {
        "dominant_axis": dominant_axis,
        "direction_label": direction_label,
        "displacement_m": round(displacement, 3),
        "delta": {
            "x": round(float(delta[0]), 3),
            "y": round(float(delta[1]), 3),
            "z": round(float(delta[2]), 3),
        },
    }



def compute_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _smooth_angles(angles: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames >= 5:
        try:
            window = max(5, (num_frames // 4) | 1)
            window = min(window, num_frames if num_frames % 2 == 1 else num_frames - 1)
            return savgol_filter(angles, window_length=window, polyorder=2)
        except Exception:
            pass
    return angles


def _angle_trend(angles: np.ndarray) -> str:
    half = len(angles) // 2
    delta = float(angles[half:].mean() - angles[:half].mean())
    if abs(delta) < 3.0:
        return "стабильный"
    return "разгибается" if delta > 0 else "сгибается"


def _angle_result(name: str, angles: np.ndarray) -> Dict:
    return {
        "name":      name,
        "mean_deg":  round(float(angles.mean()), 1),
        "min_deg":   round(float(angles.min()),  1),
        "max_deg":   round(float(angles.max()),  1),
        "range_deg": round(float(angles.max() - angles.min()), 1),
        "start_deg": round(float(angles[0]),     1),
        "end_deg":   round(float(angles[-1]),    1),
        "trend":     _angle_trend(angles),
    }


def compute_torso_tilt(all_positions: np.ndarray) -> Dict:
    num_frames = len(all_positions)
    VERTICAL = np.array([0.0, 1.0, 0.0]) 

    angles = np.zeros(num_frames, dtype=np.float32)
    for f in range(num_frames):
        hips = all_positions[f, 0]  
        neck = all_positions[f, 4] 
        spine_vec = neck - hips
        angles[f] = compute_angle_between(spine_vec, VERTICAL)

    angles = _smooth_angles(angles, num_frames)
    result = _angle_result("Наклон корпуса", angles)

    mid = num_frames // 2
    hips_mean = all_positions[mid:, 0].mean(axis=0)
    neck_mean = all_positions[mid:, 4].mean(axis=0)
    spine_vec = neck_mean - hips_mean
    norm = np.linalg.norm(spine_vec)
    if norm > 1e-6:
        spine_vec /= norm
        if abs(spine_vec[2]) > abs(spine_vec[0]):
            result["lean_direction"] = "вперёд" if spine_vec[2] < 0 else "назад"
        elif abs(spine_vec[0]) > 0.1:
            result["lean_direction"] = "вправо" if spine_vec[0] > 0 else "влево"
        else:
            result["lean_direction"] = "вертикально"
    else:
        result["lean_direction"] = "вертикально"

    return result


def compute_joint_angles(
    all_positions: np.ndarray,
    joint_angles_def: Dict,
) -> Dict:
    num_frames = len(all_positions)
    results = {}

    for joint_name, joint_def in joint_angles_def.items():
        # torso_tilt считается своей функцией
        if joint_name == "torso_tilt":
            results["torso_tilt"] = compute_torso_tilt(all_positions)
            continue

        idx_a, idx_v, idx_b = joint_def["points"]
        if (idx_a >= all_positions.shape[1] or
                idx_v >= all_positions.shape[1] or
                idx_b >= all_positions.shape[1]):
            continue

        angles = np.zeros(num_frames, dtype=np.float32)
        for f in range(num_frames):
            pa = all_positions[f, idx_a]
            pv = all_positions[f, idx_v]
            pb = all_positions[f, idx_b]
            angles[f] = compute_angle_between(pa - pv, pb - pv)

        angles = _smooth_angles(angles, num_frames)
        results[joint_name] = _angle_result(joint_def["name"], angles)

    return results


def compute_tempo(frames: List[Dict], fps: float, n_bones: int = 26) -> Dict:
    ACCENT_BONES = [8, 14, 20, 24]

    all_pos = extract_all_positions(frames, n_bones)
    if len(all_pos) < 8:
        return {
            "beats_per_min": 0.0,
            "accent_count": 0,
            "accent_times_sec": [],
            "rhythm_regularity": 0.0,
            "note": "слишком короткий сегмент",
        }

    accent_pos = all_pos[:, ACCENT_BONES, :]
    vel = np.linalg.norm(np.diff(accent_pos, axis=0), axis=2) * fps

    n_frames = vel.shape[0]
    window = max(5, int(fps * 0.2) | 1)
    window = min(window, n_frames if n_frames % 2 == 1 else n_frames - 1)
    for i in range(vel.shape[1]):
        try:
            vel[:, i] = savgol_filter(vel[:, i], window_length=window, polyorder=2)
        except Exception:
            pass

    accent_signal = np.zeros(n_frames, dtype=np.float32)
    for i in range(vel.shape[1]):
        ch = vel[:, i]
        ch_range = ch.max() - ch.min()
        if ch_range > 1e-6:
            accent_signal += (ch - ch.min()) / ch_range

    min_distance = max(3, int(fps * 0.25))
    threshold    = accent_signal.mean() + 1.0 * accent_signal.std()
    if accent_signal.max() > threshold:
        peaks, props = find_peaks(
            accent_signal,
            height=threshold,
            distance=min_distance,
            prominence=accent_signal.std() * 0.5,
        )
    else:
        peaks = np.array([], dtype=int)

    accent_times = [round(float(p / fps), 3) for p in peaks]

    if len(peaks) >= 2:
        intervals_sec = np.diff(peaks) / fps
        mean_interval = float(intervals_sec.mean())
        bpm = round(60.0 / mean_interval, 1) if mean_interval > 0 else 0.0
        bpm = float(np.clip(bpm, 20.0, 240.0))
    else:
        bpm = 0.0

    if len(peaks) >= 3:
        intervals_sec = np.diff(peaks) / fps
        rhythm_regularity = round(float(1.0 / (1.0 + intervals_sec.std())), 3)
    else:
        rhythm_regularity = 0.0

    return {
        "beats_per_min":      bpm,
        "accent_count":       int(len(peaks)),
        "accent_times_sec":   accent_times,
        "rhythm_regularity":  rhythm_regularity,
    }


def compute_symmetry(
    frames: List[Dict],
    symmetry_pairs: List[Tuple[str, str]],
    body_parts_groups: Dict,
    fps: float,
) -> Dict:
    results = {}

    for left_key, right_key in symmetry_pairs:
        left_bones  = body_parts_groups[left_key]["bones"]
        right_bones = body_parts_groups[right_key]["bones"]

        left_pos  = extract_positions_for_bones(frames, left_bones)
        right_pos = extract_positions_for_bones(frames, right_bones)

        left_vel  = compute_velocity(left_pos,  fps).mean()
        right_vel = compute_velocity(right_pos, fps).mean()

        left_rom  = float(np.linalg.norm(
            left_pos.max(axis=0) - left_pos.min(axis=0), axis=1).max())
        right_rom = float(np.linalg.norm(
            right_pos.max(axis=0) - right_pos.min(axis=0), axis=1).max())

        max_vel = max(left_vel, right_vel, 1e-6)
        vel_ratio = round(float(min(left_vel, right_vel) / max_vel), 3)

        lv = compute_velocity(left_pos,  fps).mean(axis=1)
        rv = compute_velocity(right_pos, fps).mean(axis=1)
        min_len = min(len(lv), len(rv))
        if min_len > 4:
            lv, rv = lv[:min_len], rv[:min_len]
            corr = np.correlate(lv - lv.mean(), rv - rv.mean(), mode="full")
            lag = int(np.argmax(corr)) - (min_len - 1)
            phase_offset_sec = round(float(lag / fps), 3)
        else:
            phase_offset_sec = 0.0

        if vel_ratio > 0.8:
            label = "симметричное"
        elif vel_ratio > 0.5:
            label = "умеренно асимметричное"
        else:
            label = "асимметричное"

        pair_name = f"{left_key}_vs_{right_key}"
        results[pair_name] = {
            "velocity_ratio": vel_ratio,
            "rom_ratio": round(
                min(left_rom, right_rom) / max(left_rom, right_rom, 1e-6), 3),
            "phase_offset_sec": phase_offset_sec,
            "label": label,
            "dominant_side": "левая" if left_vel > right_vel else "правая",
        }

    return results

def compute_motion_metrics(
    frames: List[Dict],
    bone_indices: List[int],
    fps: float,
) -> Dict:
    
    positions = extract_positions_for_bones(frames, bone_indices)

    if positions.shape[0] == 0:
        return _empty_metrics()

    for b in range(positions.shape[1]):
        positions[:, b, :] = smooth_signal(positions[:, b, :], fps)

    velocity     = compute_velocity(positions, fps)
    acceleration = compute_acceleration(velocity, fps)
    jerk         = compute_jerk(acceleration, fps)

    def _stats(arr: np.ndarray) -> Dict:
        if arr.size == 0:
            return {"max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "max":  round(float(np.max(arr)),  4),
            "mean": round(float(np.mean(arr)), 4),
            "std":  round(float(np.std(arr)),  4),
        }

    vel_mean_scalar = float(velocity.mean()) if velocity.size > 0 else 0.0
    jerk_mean = float(np.mean(np.abs(jerk))) if jerk.size > 0 else 0.0
    if vel_mean_scalar > 1e-4:
        normalized_jerk = jerk_mean / vel_mean_scalar
    else:
        normalized_jerk = 0.0
    smoothness = round(1.0 / (1.0 + normalized_jerk / 10.0), 4)

    return {
        "velocity_stats":     _stats(velocity),
        "acceleration_stats": _stats(acceleration),
        "jerk_stats":         _stats(np.abs(jerk)),
        "jerk_mean":          round(normalized_jerk, 4),
        "smoothness":         smoothness,
        "rom":                compute_range_of_motion(positions),
        "direction":          compute_direction(positions),
        "num_frames":         len(frames),
    }


def _empty_metrics() -> Dict:
    empty_stats = {"max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "velocity_stats": empty_stats,
        "acceleration_stats": empty_stats,
        "jerk_stats": empty_stats,
        "jerk_mean": 0.0,
        "smoothness": 1.0,
        "rom": {"max_distance": 0.0, "mean_distance": 0.0,
                "min_distance": 0.0, "axis_range": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "direction": {"dominant_axis": "none", "direction_label": "нет движения",
                      "displacement_m": 0.0},
        "num_frames": 0,
    }


def analyze_segment_body_parts(
    segment: Dict,
    mixamo_frames: List[Dict],
    fps: float,
    body_parts_groups: Dict,
    joint_angles_def: Optional[Dict] = None,
    symmetry_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    from .body_parts_groups import JOINT_ANGLES, SYMMETRY_PAIRS

    if joint_angles_def is None:
        joint_angles_def = JOINT_ANGLES
    if symmetry_pairs is None:
        symmetry_pairs = SYMMETRY_PAIRS

    start = max(0, segment["start_frame"])
    end   = min(len(mixamo_frames), segment["end_frame"])
    frames = mixamo_frames[start:end]


    # 1. Метрики по частям тела
    body_parts_analysis = {}
    for part_name, part_config in body_parts_groups.items():
        try:
            metrics = compute_motion_metrics(frames, part_config["bones"], fps)
            body_parts_analysis[part_name] = {
                "display_name": part_config["name"],
                "description":  part_config["description"],
                "metrics":      metrics,
            }
        except Exception as e:
            logger.error(f"Error analyzing {part_name}: {e}", exc_info=True)
            body_parts_analysis[part_name] = {
                "display_name": part_config["name"],
                "error": str(e),
            }

    # 2. Углы суставов
    all_positions = extract_all_positions(frames)
    joint_angles_result = {}
    try:
        joint_angles_result = compute_joint_angles(all_positions, joint_angles_def)
    except Exception as e:
        logger.error(f"Joint angles error: {e}", exc_info=True)

    # 3. Темп
    tempo_result = {}
    try:
        tempo_result = compute_tempo(frames, fps)
    except Exception as e:
        logger.error(f"Tempo error: {e}", exc_info=True)

    # 4. Симметрия
    symmetry_result = {}
    try:
        symmetry_result = compute_symmetry(frames, symmetry_pairs, body_parts_groups, fps)
    except Exception as e:
        logger.error(f"Symmetry error: {e}", exc_info=True)

    return {
        "body_parts":   body_parts_analysis,
        "joint_angles": joint_angles_result,
        "tempo":        tempo_result,
        "symmetry":     symmetry_result,
    }