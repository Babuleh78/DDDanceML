import logging

import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)


JOINT_INDICES = {
    "shoulders": [11, 12],
    "elbows": [13, 14],
    "wrists": [15, 16],
    "hips": [23, 24],
    "knees": [25, 26],
    "ankles": [27, 28],
}

KEY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _extract_positions(frame: dict, indices: list) -> np.ndarray:
    landmarks = frame.get("landmarks") or []
    positions = []
    for idx in indices:
        if idx < len(landmarks) and landmarks[idx] is not None:
            lm = landmarks[idx]
            positions.append([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
        else:
            positions.append([0, 0, 0])
    return np.array(positions, dtype=np.float32)


def _angular_velocity(
    frames: list,
    joint_pair_indices: list,
    fps: float,
) -> tuple[float, float]:
    if len(frames) < 2:
        return 0.0, 0.0

    positions = []
    for frame in frames:
        landmarks = frame.get("landmarks") or []
        valid = True
        pair_pos = []
        for idx in joint_pair_indices:
            if idx < len(landmarks) and landmarks[idx] is not None:
                lm = landmarks[idx]
                pair_pos.append(np.array([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)]))
            else:
                valid = False
                break
        if valid:
            positions.append(pair_pos)
        else:
            positions.append(None)

    angles = []
    for i in range(1, len(positions)):
        if positions[i - 1] is not None and positions[i] is not None:
            v1 = positions[i - 1][1] - positions[i - 1][0]
            v2 = positions[i][1] - positions[i][0]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)

    if not angles:
        return 0.0, 0.0

    angles = np.array(angles)
    angular_velocities = angles * fps
    return float(np.mean(angular_velocities)), float(np.max(angular_velocities))


def _com_dispersion(frames: list) -> float:
    if len(frames) < 2:
        return 0.0

    com_positions = []
    for frame in frames:
        pos = _extract_positions(frame, KEY_LANDMARKS)
        valid_pos = pos[pos[:, 0] != 0]
        if len(valid_pos) > 0:
            com = np.mean(valid_pos, axis=0)
            com_positions.append(com)

    if len(com_positions) < 2:
        return 0.0

    com_array = np.array(com_positions)
    return float(np.var(com_array))


def _moving_limbs_count(frames: list, fps: float, velocity_threshold: float = 0.05) -> float:
    if len(frames) < 2:
        return 0.0

    moving_counts = []

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        moving = 0
        for joint_name, indices in JOINT_INDICES.items():
            prev_pos = _extract_positions(prev_frame, indices)
            curr_pos = _extract_positions(curr_frame, indices)

            distances = np.linalg.norm(curr_pos - prev_pos, axis=1)
            avg_distance = np.mean(distances)
            velocity = avg_distance * fps

            if velocity > velocity_threshold:
                moving += 1

        moving_counts.append(moving)

    return float(np.mean(moving_counts)) if moving_counts else 0.0


def _pose_sequence_entropy(frames: list, bins: int = 10) -> float:
    if len(frames) < 2:
        return 0.0

    pose_signatures = []
    for frame in frames:
        pos = _extract_positions(frame, KEY_LANDMARKS)
        valid_pos = pos[np.any(pos != 0, axis=1)]

        if len(valid_pos) > 0:
            quantized = np.digitize(valid_pos, np.linspace(-1, 1, bins))
            signature = tuple(quantized.flatten())
            pose_signatures.append(signature)

    if not pose_signatures:
        return 0.0

    unique_poses, counts = np.unique(
        [str(sig) for sig in pose_signatures],
        return_counts=True
    )
    probabilities = counts / len(pose_signatures)
    return float(entropy(probabilities))


def _detect_jumps_and_rotations(frames: list, fps: float) -> dict:
    jumps = 0
    rotations = 0

    if len(frames) < 2:
        return {"jumps": 0, "rotations": 0}

    y_positions = []
    for frame in frames:
        pos = _extract_positions(frame, KEY_LANDMARKS)
        valid_pos = pos[pos[:, 0] != 0]
        if len(valid_pos) > 0:
            com_y = np.mean(valid_pos[:, 1])
            y_positions.append(com_y)

    if len(y_positions) > 2:
        y_array = np.array(y_positions)
        diffs = np.diff(y_array)
        std_diff = np.std(diffs)
        if std_diff > 0:
            threshold = 2.5 * std_diff
            jumps = int(np.sum(np.abs(diffs) > threshold))

    rotation_angles = []
    for i in range(1, len(frames)):
        prev_pos = _extract_positions(frames[i - 1], KEY_LANDMARKS)
        curr_pos = _extract_positions(frames[i], KEY_LANDMARKS)

        vectors_prev = np.mean(prev_pos[prev_pos[:, 0] != 0], axis=0)
        vectors_curr = np.mean(curr_pos[curr_pos[:, 0] != 0], axis=0)

        if np.linalg.norm(vectors_prev) > 1e-6 and np.linalg.norm(vectors_curr) > 1e-6:
            cos_angle = np.dot(vectors_prev, vectors_curr) / (
                np.linalg.norm(vectors_prev) * np.linalg.norm(vectors_curr)
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rotation_angles.append(angle)

    if rotation_angles:
        rotation_angles = np.array(rotation_angles)
        std_angle = np.std(rotation_angles)
        if std_angle > 0:
            threshold = 2.5 * std_angle
            rotations = int(np.sum(rotation_angles > threshold))

    return {"jumps": int(jumps), "rotations": int(rotations)}


def compute_dance_features(frames: list, fps: float) -> dict:
    try:
        logger.info(f"Computing dance features from {len(frames)} frames at {fps} fps")

        arm_vel_mean, arm_vel_max = _angular_velocity(frames, [11, 13], fps)
        leg_vel_mean, leg_vel_max = _angular_velocity(frames, [23, 25], fps)

        angular_velocity_mean = float(np.mean([arm_vel_mean, leg_vel_mean]))
        angular_velocity_max = float(np.max([arm_vel_max, leg_vel_max]))

        com_disp = _com_dispersion(frames)

        moving_count = _moving_limbs_count(frames, fps)

        pose_ent = _pose_sequence_entropy(frames)

        jump_rot = _detect_jumps_and_rotations(frames, fps)

        features = {
            "angular_velocity_mean": float(np.round(angular_velocity_mean, 4)),
            "angular_velocity_max": float(np.round(angular_velocity_max, 4)),
            "com_dispersion": float(np.round(com_disp, 4)),
            "moving_limbs_avg_count": float(np.round(moving_count, 2)),
            "pose_entropy": float(np.round(pose_ent, 4)),
            "jumps": jump_rot["jumps"],
            "rotations": jump_rot["rotations"],
        }
        return features

    except Exception as e:
        logger.error(f"Error computing dance features: {e}", exc_info=True)
        return {
            "angular_velocity_mean": 0.0,
            "angular_velocity_max": 0.0,
            "com_dispersion": 0.0,
            "moving_limbs_avg_count": 0.0,
            "pose_entropy": 0.0,
            "jumps": 0,
            "rotations": 0,
        }


_DIFFICULTY_RANGES = {
    "angular_velocity_mean": (0.5, 7.0),
    "angular_velocity_max": (3.0, 25.0),
    "moving_limbs_avg_count": (1.0, 5.0),
    "pose_entropy": (1.5, 5.5),
    "acro_density": (0.0, 12.0),      
    "com_dispersion": (0.001, 0.05),
}

_DIFFICULTY_WEIGHTS = {
    "speed": 0.22,
    "coordination": 0.24,
    "variety": 0.18,
    "acrobatics": 0.22,
    "travel": 0.14,
}


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def compute_difficulty(features: dict, duration_sec: float) -> dict:
    try:
        avg_v = features.get("angular_velocity_mean", 0.0)
        limbs = features.get("moving_limbs_avg_count", 0.0)
        entropy_v = features.get("pose_entropy", 0.0)

        if avg_v == 0.0 and limbs == 0.0 and entropy_v == 0.0:
            return {"difficulty_score": 50, "difficulty_label": "medium"}

        r = _DIFFICULTY_RANGES
        speed = (
            0.7 * _norm(avg_v, *r["angular_velocity_mean"])
            + 0.3 * _norm(features.get("angular_velocity_max", 0.0), *r["angular_velocity_max"])
        )
        coordination = _norm(limbs, *r["moving_limbs_avg_count"])
        variety = _norm(entropy_v, *r["pose_entropy"])

        minutes = max(duration_sec / 60.0, 1e-6)
        acro_density = (features.get("jumps", 0) + features.get("rotations", 0)) / minutes
        acrobatics = _norm(acro_density, *r["acro_density"])

        travel = _norm(features.get("com_dispersion", 0.0), *r["com_dispersion"])

        w = _DIFFICULTY_WEIGHTS
        d = (
            w["speed"] * speed
            + w["coordination"] * coordination
            + w["variety"] * variety
            + w["acrobatics"] * acrobatics
            + w["travel"] * travel
        )
        score = max(0, min(100, int(round(100 * d))))

        if score < 34:
            label = "easy"
        elif score < 67:
            label = "medium"
        else:
            label = "hard"

        logger.info(f"Difficulty: score={score}, label={label}")
        return {"difficulty_score": score, "difficulty_label": label}

    except Exception as e:
        logger.error(f"Error computing difficulty: {e}", exc_info=True)
        return {"difficulty_score": 50, "difficulty_label": "medium"}
