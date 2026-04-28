"""
Энергетическая сегментация на основе кватернионов (Mixamo format).
Совместимо с форматом: frame = {"time": float, "bones": [{"name", "rotation", "position"}]}
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import Optional, List, Dict

# Кости, участвующие в расчёте энергии (крупные суставы)
KEY_BONES = [
    'mixamorig:LeftArm',
    'mixamorig:RightArm',
    'mixamorig:LeftForeArm',
    'mixamorig:RightForeArm',
    'mixamorig:LeftUpLeg',
    'mixamorig:RightUpLeg',
    'mixamorig:LeftLeg',
    'mixamorig:RightLeg',
    'mixamorig:Spine',
    'mixamorig:Spine1',
    'mixamorig:Spine2',
]


def _quat_to_array(rot: dict) -> np.ndarray:
    """Кватернион из dict → numpy [w, x, y, z]."""
    return np.array([
        float(rot.get('w', 1.0)),
        float(rot.get('x', 0.0)),
        float(rot.get('y', 0.0)),
        float(rot.get('z', 0.0)),
    ], dtype=np.float32)


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Угловое расстояние между двумя кватернионами в радианах [0, π]."""
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return float(2.0 * np.arccos(dot))


def _extract_quats(frames: list) -> Dict[str, np.ndarray]:
    """Извлекает кватернионы KEY_BONES из всех кадров."""
    N = len(frames)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = {bone: np.tile(identity, (N, 1)) for bone in KEY_BONES}

    for t, frame in enumerate(frames):
        bone_lookup = {b['name']: b for b in frame.get('bones', [])}
        for bone_name in KEY_BONES:
            bone = bone_lookup.get(bone_name)
            if bone and 'rotation' in bone:
                result[bone_name][t] = _quat_to_array(bone['rotation'])
    return result


def compute_energy(
    frames: list,
    w_velocity: float = 0.6,
    w_acceleration: float = 0.4,
    smooth_window: int = 28,
) -> tuple[np.ndarray, dict]:
    """Энергия на основе угловой скорости и ускорения кватернионов."""
    N = len(frames)
    quats = _extract_quats(frames)

    # Угловая скорость
    velocity = np.zeros(N, dtype=np.float32)
    for t in range(1, N - 1):
        bone_vels = [
            _quat_angular_distance(quats[bone][t - 1], quats[bone][t + 1]) / 2.0
            for bone in KEY_BONES
        ]
        velocity[t] = float(np.mean(bone_vels))
    velocity[0] = velocity[1]
    velocity[-1] = velocity[-2]

    # Угловое ускорение
    acceleration = np.zeros(N, dtype=np.float32)
    for t in range(1, N - 1):
        acceleration[t] = abs(velocity[t + 1] - velocity[t - 1]) / 2.0
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]

    # Нормализация
    def _norm(sig):
        mn, mx = sig.min(), sig.max()
        if mx - mn < 1e-8:
            return np.zeros_like(sig)
        return (sig - mn) / (mx - mn)

    vel_n = _norm(velocity)
    acc_n = _norm(acceleration)
    energy = w_velocity * vel_n + w_acceleration * acc_n

    if smooth_window % 2 == 0:
        smooth_window += 1
    energy_smooth = savgol_filter(energy, window_length=smooth_window, polyorder=2)
    energy_smooth = np.clip(energy_smooth, 0, None)

    return energy_smooth, {
        "velocity": vel_n.tolist(),
        "acceleration": acc_n.tolist(),
        "energy_raw": energy.tolist(),
        "energy_smooth": energy_smooth.tolist(),
    }


def detect_boundaries(
    energy: np.ndarray,
    fps: float,
    min_segment_sec: float = 2.0,
    sensitivity: float = 0.15,
) -> list[int]:
    """Детектирует границы сегментов по пикам энергии."""
    min_distance = max(1, int(fps * min_segment_sec))
    peaks, _ = find_peaks(energy, distance=min_distance, prominence=sensitivity)
    return peaks.tolist()


def build_segments(
    frames: list,
    boundary_frames: list[int],
    fps: float,
    energy: Optional[np.ndarray] = None,
    min_segment_sec: float = 2.0,
) -> list[dict]:
    """Строит список сегментов с ключевыми кадрами."""
    N = len(frames)
    min_frames = max(2, int(fps * min_segment_sec))
    boundaries = sorted(set([0] + boundary_frames + [N - 1]))

    # Фильтрация слишком коротких сегментов
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if b == boundaries[-1]:
            if b - filtered[-1] < min_frames and len(filtered) > 1:
                filtered.pop()
            filtered.append(b)
        elif b - filtered[-1] >= min_frames:
            filtered.append(b)
    boundaries = filtered

    segments = []
    for i in range(len(boundaries) - 1):
        start_f, end_f = boundaries[i], boundaries[i + 1]
        start_ms = frames[start_f].get("time", start_f / fps) * (1000.0 / fps)
        end_ms = frames[end_f].get("time", end_f / fps) * (1000.0 / fps)
        duration_ms = end_ms - start_ms

        peak_f = (int(np.argmax(energy[start_f:end_f + 1])) + start_f 
                  if energy is not None and len(energy) > end_f 
                  else (start_f + end_f) // 2)
        peak_ms = frames[peak_f].get("time", peak_f / fps) * (1000.0 / fps)

        segments.append({
            "index": len(segments),
            "label": f"segment_{len(segments) + 1}",
            "start_frame": start_f,
            "end_frame": end_f,
            "start_ms": round(start_ms, 2),
            "end_ms": round(end_ms, 2),
            "duration_ms": round(duration_ms, 2),
            "duration_sec": round(duration_ms / 1000, 3),
            "num_frames": end_f - start_f,
            "keyframes": {
                "start": {"frame_idx": start_f, "timestamp_ms": round(start_ms, 2)},
                "peak": {"frame_idx": peak_f, "timestamp_ms": round(peak_ms, 2)},
                "end": {"frame_idx": end_f, "timestamp_ms": round(end_ms, 2)},
            },
        })
    return segments