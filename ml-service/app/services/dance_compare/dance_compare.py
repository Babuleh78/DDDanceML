import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core import s3 as s3_client
from app.core.config import settings
from app.services.body_parts_extractor.analyzer import compute_velocity
from app.services.body_parts_extractor.extractor import extract_body_parts_for_segments
from app.services.skeleton_to_segments import (
    compute_energy,
    detect_boundaries,
    build_segments,
)
from app.services.video_to_json import convert_video_to_json

logger = logging.getLogger(__name__)

WEIGHTS = {
    "dtw":        0.35,
    "velocity":   0.15,
    "smoothness": 0.10,
    "rom":        0.15,
    "tempo":      0.10,
    "symmetry":   0.10,
    "joints":     0.05,
}

DTW_BONE_GROUPS = {
    "arms": [13, 14, 15, 16],  # локти + запястья
    "legs": [25, 26, 27, 28],  # колени + лодыжки
    "hips": [23, 24],          # бёдра
}


def _mixamo_frames_key(dance_id: str) -> str:
    return f"results/{dance_id}/mixamo_frames.json"


def _try_import_fastdtw():
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        return fastdtw, euclidean
    except ImportError:
        return None, None


def _dtw_numpy(s: np.ndarray, t: np.ndarray) -> float:
    n, m = len(s), len(t)

    diff = s[:, np.newaxis, :] - t[np.newaxis, :, :]
    cost_matrix = np.sqrt((diff ** 2).sum(axis=2))

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

    raw = dtw_acc[n, m]
    return float(raw / (n + m)) if (n + m) > 0 else 0.0


def _dtw_distance(s: np.ndarray, t: np.ndarray) -> float:
    fastdtw_fn, euclidean_fn = _try_import_fastdtw()
    if fastdtw_fn is not None:
        distance, _ = fastdtw_fn(s, t, radius=10, dist=euclidean_fn)
        n, m = len(s), len(t)
        return float(distance / (n + m)) if (n + m) > 0 else 0.0
    return _dtw_numpy(s, t)


def _velocity_signal_multivariate(
    frames: list,
    bone_indices: list,
    fps: float,
) -> np.ndarray:
    num_frames = len(frames)
    positions = np.zeros((num_frames, len(bone_indices), 3), dtype=np.float32)

    for fi, frame in enumerate(frames):
        lms = frame.get("landmarks") or []
        for bi, bone_idx in enumerate(bone_indices):
            if bone_idx < len(lms) and lms[bone_idx] is not None:
                lm = lms[bone_idx]
                positions[fi, bi] = [lm["x"], lm["y"], lm["z"]]

    if num_frames < 2:
        return np.zeros((1, len(bone_indices)), dtype=np.float32)

    vel = compute_velocity(positions, fps)  

    col_min = vel.min(axis=0, keepdims=True)
    col_max = vel.max(axis=0, keepdims=True)
    col_range = np.where(col_max - col_min > 1e-8, col_max - col_min, 1.0)
    return ((vel - col_min) / col_range).astype(np.float32)


def _compute_dtw_scores(
    orig_frames: list,
    user_frames: list,
    fps: float,
) -> dict[str, float]:
    scores = {}
    for group_name, bone_indices in DTW_BONE_GROUPS.items():
        orig_sig = _velocity_signal_multivariate(orig_frames, bone_indices, fps)
        user_sig = _velocity_signal_multivariate(user_frames, bone_indices, fps)

        dist = _dtw_distance(orig_sig, user_sig)

        score = float(np.exp(-dist * 3.0))
        scores[group_name] = round(float(np.clip(score, 0.0, 1.0)), 4)

    return scores


def _proxy_dtw_scores(orig_metrics: dict, user_metrics: dict) -> dict[str, float]:
    o_vel = orig_metrics.get("velocity", {}).get("mean", 0.0)
    u_vel = user_metrics.get("velocity", {}).get("mean", 0.0)
    scale = max(o_vel, 0.1)
    proxy = round(float(np.clip(np.exp(-abs(o_vel - u_vel) / scale * 5.0), 0.0, 1.0)), 4)
    return {group: proxy for group in DTW_BONE_GROUPS}


def _safe_diff(a: float, b: float, scale: float = 1.0) -> float:
    return float(np.clip(abs(a - b) / (scale + 1e-9), 0.0, 1.0))


def _aggregate_diff(orig_metrics: dict, user_metrics: dict) -> dict:
    diffs = {}

    ov = orig_metrics.get("velocity", {}).get("mean", 0.0)
    uv = user_metrics.get("velocity", {}).get("mean", 0.0)
    diffs["velocity"] = _safe_diff(ov, uv, max(ov, 0.5))

    diffs["smoothness"] = _safe_diff(
        orig_metrics.get("smoothness", 1.0),
        user_metrics.get("smoothness", 1.0),
        scale=1.0,
    )

    o_rom = orig_metrics.get("rom", {}).get("max_distance", 0.0)
    u_rom = user_metrics.get("rom", {}).get("max_distance", 0.0)
    diffs["rom"] = _safe_diff(o_rom, u_rom, max(o_rom, 0.1))

    o_bpm = orig_metrics.get("tempo_bpm", 0.0)
    u_bpm = user_metrics.get("tempo_bpm", 0.0)
    diffs["tempo"] = _safe_diff(o_bpm, u_bpm, max(o_bpm, 60.0))

    diffs["symmetry"] = _safe_diff(
        orig_metrics.get("symmetry_ratio", 1.0),
        user_metrics.get("symmetry_ratio", 1.0),
        scale=1.0,
    )

    joint_diffs = {}
    for jname, o_data in orig_metrics.get("joint_angles", {}).items():
        u_data = user_metrics.get("joint_angles", {}).get(jname)
        if u_data is not None:
            joint_diffs[jname] = round(
                _safe_diff(o_data.get("mean_deg", 0.0), u_data.get("mean_deg", 0.0),
                           max(o_data.get("mean_deg", 0.0), 10.0)), 4
            )
    diffs["joint_angles"] = joint_diffs

    return diffs


def _segment_score(dtw_scores: dict, agg_diffs: dict) -> float:
    dtw_mean = float(np.mean(list(dtw_scores.values()))) if dtw_scores else 0.5
    joint_diffs = agg_diffs.get("joint_angles", {})
    joints_mean_diff = float(np.mean(list(joint_diffs.values()))) if joint_diffs else 0.0

    score = (
        WEIGHTS["dtw"]        * dtw_mean
        + WEIGHTS["velocity"]   * (1 - agg_diffs.get("velocity",   0.0))
        + WEIGHTS["smoothness"] * (1 - agg_diffs.get("smoothness", 0.0))
        + WEIGHTS["rom"]        * (1 - agg_diffs.get("rom",        0.0))
        + WEIGHTS["tempo"]      * (1 - agg_diffs.get("tempo",      0.0))
        + WEIGHTS["symmetry"]   * (1 - agg_diffs.get("symmetry",   0.0))
        + WEIGHTS["joints"]     * (1 - joints_mean_diff)
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def _find_weakest(agg_diffs: dict, dtw_scores: dict, threshold: float = 0.4) -> list[str]:
    weak = [m for m, v in agg_diffs.items() if isinstance(v, float) and v > threshold]
    if dtw_scores and float(np.mean(list(dtw_scores.values()))) < (1 - threshold):
        weak.append("movement_dynamics")
    return weak

def extract_numeric_metrics(segment: dict) -> dict:
    bpd = segment.get("body_parts_description", {})
    raw = bpd.get("raw_analysis", {}) if isinstance(bpd, dict) else {}

    vel_means, smoothness_vals, rom_vals = [], [], []
    for part_data in raw.get("body_parts", {}).values():
        m = part_data.get("metrics", {})
        vs = m.get("velocity_stats", {})
        if vs.get("mean") is not None:
            vel_means.append(vs["mean"])
        if m.get("smoothness") is not None:
            smoothness_vals.append(m["smoothness"])
        if m.get("rom", {}).get("max_distance") is not None:
            rom_vals.append(m["rom"]["max_distance"])

    tempo = raw.get("tempo", {})
    symmetry = raw.get("symmetry", {})
    sym_ratios = [v.get("velocity_ratio", 1.0) for v in symmetry.values() if isinstance(v, dict)]

    return {
        "velocity": {
            "mean": round(float(np.mean(vel_means))  if vel_means       else 0.0, 4),
            "max":  round(float(np.max(vel_means))   if vel_means       else 0.0, 4),
        },
        "smoothness":     round(float(np.mean(smoothness_vals)) if smoothness_vals else 1.0, 4),
        "rom": {
            "max_distance":  round(float(np.max(rom_vals))  if rom_vals else 0.0, 4),
            "mean_distance": round(float(np.mean(rom_vals)) if rom_vals else 0.0, 4),
        },
        "tempo_bpm":      round(float(tempo.get("beats_per_min", 0.0)), 2),
        "symmetry_ratio": round(float(np.mean(sym_ratios)) if sym_ratios else 1.0, 4),
        "joint_angles": {
            jname: {
                "mean_deg":  jdata.get("mean_deg",  0.0),
                "range_deg": jdata.get("range_deg", 0.0),
            }
            for jname, jdata in raw.get("joint_angles", {}).items()
            if isinstance(jdata, dict)
        },
    }


def _run_light_pipeline(video_path: str) -> tuple[list, list, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    with open(settings.mixamo_model_path, "r", encoding="utf-8") as f:
        model_json = json.load(f)

    mixamo_data = convert_video_to_json(
        video_path=video_path,
        model_json=model_json,
        fps=int(fps),
        min_visibility=settings.mixamo_min_visibility,
        is_hips_move=True,
        max_frames=settings.mixamo_max_frames,
        is_show_result=False,
    )
    mixamo_frames = mixamo_data["frames"]

    energy, _ = compute_energy(mixamo_frames, smooth_window=settings.segmenter_smooth_window)
    boundaries = detect_boundaries(
        energy, fps=fps,
        min_segment_sec=settings.segmenter_min_seg_sec,
        sensitivity=settings.segmenter_sensitivity,
    )
    segments = build_segments(
        mixamo_frames, boundaries, fps,
        energy=energy,
        min_segment_sec=settings.segmenter_min_seg_sec,
    )

    enriched, _ = extract_body_parts_for_segments(segments, mixamo_frames, fps)
    return (
        mixamo_frames,
        [{**seg, "numeric_metrics": extract_numeric_metrics(seg)} for seg in enriched],
        fps,
    )


def _load_orig_frames(dance_id: str, tmp: Path) -> Optional[list]:
    try:
        local_path = str(tmp / "mixamo_frames.json")
        s3_client.download_file(_mixamo_frames_key(dance_id), local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"mixamo_frames.json недоступен для {dance_id}: {e}")
        return None


def _align_segments(orig: list, user: list) -> list[tuple[dict, dict]]:
    return list(zip(orig[: min(len(orig), len(user))], user[: min(len(orig), len(user))]))

def compare_dance(video_key: str, dance_id: str, segment_idx: int = -1) -> dict:
    logger.info(f"compare_dance: video_key={video_key}, dance_id={dance_id}, seg={segment_idx}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        video_path = str(tmp / Path(video_key).name)
        s3_client.download_file(video_key, video_path)

        user_frames, user_segments, fps = _run_light_pipeline(video_path)

        orig_path = str(tmp / "orig_segments.json")
        s3_client.download_file(f"results/{dance_id}/segments.json", orig_path)
        with open(orig_path, "r", encoding="utf-8") as f:
            orig_data = json.load(f)

        orig_segments = orig_data.get("segments", [])
        orig_fps = orig_data.get("meta", {}).get("fps", fps)

        if orig_segments and "numeric_metrics" not in orig_segments[0]:
            raise ValueError(
                "Оригинальный танец обработан без numeric_metrics. "
                "Переобработайте через /ml/process."
            )

        orig_frames = _load_orig_frames(dance_id, tmp)
        has_real_dtw = orig_frames is not None
        if not has_real_dtw:
            logger.warning(
                f"mixamo_frames.json не найден для {dance_id} — используется прокси DTW."
            )

        if segment_idx == -1:
            pairs = _align_segments(orig_segments, user_segments)
        else:
            if segment_idx >= len(orig_segments):
                raise ValueError(
                    f"segment_idx={segment_idx} за пределами оригинала "
                    f"({len(orig_segments)} сегментов)"
                )
            pairs = [(orig_segments[segment_idx],
                      user_segments[min(segment_idx, len(user_segments) - 1)])]

        segment_details = []
        all_weakest: list[str] = []

        for pair_idx, (orig_seg, user_seg) in enumerate(pairs):
            actual_idx = segment_idx if segment_idx != -1 else pair_idx
            orig_metrics = orig_seg.get("numeric_metrics", {})
            user_metrics = user_seg.get("numeric_metrics", {})

            user_start = user_seg.get("start_frame", 0)
            user_end   = user_seg.get("end_frame",   len(user_frames))
            user_seg_frames = user_frames[user_start:user_end]

            if has_real_dtw:
                orig_start = orig_seg.get("start_frame", 0)
                orig_end   = orig_seg.get("end_frame",   len(orig_frames))
                dtw_scores = _compute_dtw_scores(
                    orig_frames[orig_start:orig_end], user_seg_frames, orig_fps
                )
            else:
                dtw_scores = _proxy_dtw_scores(orig_metrics, user_metrics)

            agg_diffs = _aggregate_diff(orig_metrics, user_metrics)
            score     = _segment_score(dtw_scores, agg_diffs)
            weakest   = _find_weakest(agg_diffs, dtw_scores)
            all_weakest.extend(weakest)

            segment_details.append({
                "segment_idx":       actual_idx,
                "dtw_scores":        dtw_scores,
                "dtw_is_real":       has_real_dtw,
                "velocity_diff":     round(agg_diffs.get("velocity",   0.0), 4),
                "smoothness_diff":   round(agg_diffs.get("smoothness", 0.0), 4),
                "rom_diff":          round(agg_diffs.get("rom",        0.0), 4),
                "tempo_diff":        round(agg_diffs.get("tempo",      0.0), 4),
                "symmetry_diff":     round(agg_diffs.get("symmetry",   0.0), 4),
                "joint_angles_diff": agg_diffs.get("joint_angles", {}),
                "segment_score":     score,
            })

        return {
            "dance_id":        dance_id,
            "segment_idx":     segment_idx,
            "overall_score":   round(float(np.mean([d["segment_score"] for d in segment_details])), 4),
            "segments":        segment_details,
            "weakest_metrics": [m for m, _ in Counter(all_weakest).most_common(3)],
        }