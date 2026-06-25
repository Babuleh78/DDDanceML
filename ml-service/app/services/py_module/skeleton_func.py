from typing import Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter

KEY_JOINTS = [
    0,
    11, 12,
    13, 14,
    15, 16,
    23, 24,
    25, 26,
    27, 28,
]

BONE_TRIPLES = [
    (11, 13, 15),
    (12, 14, 16),
    (23, 25, 27),
    (24, 26, 28),
    (11, 23, 25),
    (12, 24, 26),
]


def joint_pos(frame: dict, idx: int) -> np.ndarray:
    """Возвращает (x, y, z) сустава как numpy array."""
    j = frame["joints"][idx]
    return np.array([j["x"], j["y"], j["z"]], dtype=np.float32)


def angle_at_joint(frame: dict, a_idx: int, b_idx: int, c_idx: int) -> float:
    A = joint_pos(frame, a_idx)
    B = joint_pos(frame, b_idx)
    C = joint_pos(frame, c_idx)

    BA = A - B
    BC = C - B

    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    if norm_BA < 1e-6 or norm_BC < 1e-6:
        return 0.0

    cos_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))

def compute_joint_velocity(frames: list) -> np.ndarray:
    N = len(frames)
    velocity = np.zeros(N, dtype=np.float32)

    for t in range(1, N - 1):
        prev = np.array([joint_pos(frames[t - 1], i) for i in KEY_JOINTS])
        next_ = np.array([joint_pos(frames[t + 1], i) for i in KEY_JOINTS])
        velocity[t] = np.mean(np.linalg.norm(next_ - prev, axis=1)) / 2.0

    velocity[0] = velocity[1]
    velocity[-1] = velocity[-2]

    return velocity


def compute_bone_angle_delta(frames: list) -> np.ndarray:
    N = len(frames)
    angle_delta = np.zeros(N, dtype=np.float32)

    angles = np.zeros((N, len(BONE_TRIPLES)), dtype=np.float32)
    for t in range(N):
        for k, (a, b, c) in enumerate(BONE_TRIPLES):
            angles[t, k] = angle_at_joint(frames[t], a, b, c)

    for t in range(1, N):
        angle_delta[t] = np.mean(np.abs(angles[t] - angles[t - 1]))

    angle_delta[0] = angle_delta[1]

    return angle_delta


def compute_pose_diff(frames: list, window: int = 3) -> np.ndarray:
    N = len(frames)
    pose_diff = np.zeros(N, dtype=np.float32)

    poses = np.array([
        np.concatenate([joint_pos(f, i) for i in KEY_JOINTS])
        for f in frames
    ], dtype=np.float32)

    for t in range(window, N):
        a = poses[t]
        b = poses[t - window]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            continue
        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        pose_diff[t] = 1.0 - cos_sim

    return pose_diff


def normalize(signal: np.ndarray) -> np.ndarray:
    s_min = signal.min()
    s_max = signal.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)



def compute_energy(
    frames: list,
    w_velocity: float = 0.5,
    w_angle: float = 0.3,
    w_pose: float = 0.2,
    pose_window: int = 3,
    smooth_window: int = 15,
) -> tuple[np.ndarray, dict]:
    vel = normalize(compute_joint_velocity(frames))

    ang = normalize(compute_bone_angle_delta(frames))

    pdiff = normalize(compute_pose_diff(frames, window=pose_window))

    energy = w_velocity * vel + w_angle * ang + w_pose * pdiff

    if smooth_window % 2 == 0:
        smooth_window += 1
    energy_smooth = savgol_filter(energy, window_length=smooth_window, polyorder=2)

    energy_smooth = np.clip(energy_smooth, 0, None)

    return energy_smooth, {
        "velocity": vel.tolist(),
        "angle_delta": ang.tolist(),
        "pose_diff": pdiff.tolist(),
        "energy_raw": energy.tolist(),
        "energy_smooth": energy_smooth.tolist(),
    }

def detect_boundaries(
    energy: np.ndarray,
    fps: float,
    min_segment_sec: float = 1.0,
    sensitivity: float = 0.08,
) -> list[int]:
    min_distance = max(1, int(fps * min_segment_sec))

    peaks, _ = find_peaks(
        energy,
        distance=min_distance,
        prominence=sensitivity,
    )

    return peaks.tolist()


def build_segments(
    frames: list,
    boundary_frames: list[int],
    fps: float,
    energy: Optional[np.ndarray] = None,
    min_segment_sec: float = 1.0,
) -> list[dict]:
    N = len(frames)
    min_frames = max(2, int(fps * min_segment_sec))

    boundaries = sorted(set([0] + boundary_frames + [N - 1]))

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
        start_f = boundaries[i]
        end_f   = boundaries[i + 1]

        start_ms    = frames[start_f].get("timestamp_ms", start_f * (1000 / fps))
        end_ms      = frames[end_f].get("timestamp_ms",   end_f   * (1000 / fps))
        duration_ms = end_ms - start_ms

        if energy is not None and len(energy) > end_f:
            peak_f = int(np.argmax(energy[start_f:end_f + 1])) + start_f
        else:
            peak_f = (start_f + end_f) // 2

        peak_ms = frames[peak_f].get("timestamp_ms", peak_f * (1000 / fps))

        segments.append({
            "index":        len(segments),
            "label":        f"segment_{len(segments) + 1}",
            "start_frame":  start_f,
            "end_frame":    end_f,
            "start_ms":     round(start_ms, 2),
            "end_ms":       round(end_ms, 2),
            "duration_ms":  round(duration_ms, 2),
            "duration_sec": round(duration_ms / 1000, 3),
            "num_frames":   end_f - start_f,
            "keyframes": {
                "start": {"frame_idx": start_f, "timestamp_ms": round(start_ms, 2)},
                "peak":  {"frame_idx": peak_f,  "timestamp_ms": round(peak_ms, 2)},
                "end":   {"frame_idx": end_f,   "timestamp_ms": round(end_ms, 2)},
            },
        })

    return segments