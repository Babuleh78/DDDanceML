"""Обработка скелета: сегментация + семантическая разметка."""
import logging
import numpy as np
from typing import Optional, List, Dict
from .py_module.skeleton_enerqy_quat import compute_energy, detect_boundaries, build_segments

logger = logging.getLogger(__name__)


async def process_skeleton_to_segments(
    frames: List[Dict],
    fps: float,
    energy_params: Optional[Dict] = None,
    boundary_params: Optional[Dict] = None,
    enable_labeling: bool = True,
) -> Dict:
    energy_params = energy_params or {}
    energy_smooth, energy_debug = compute_energy(frames, **energy_params)

    boundary_params = boundary_params or {}

    min_segment_sec = boundary_params.get("min_segment_sec", 2.0)

    detect_kwargs = {
        "min_segment_sec": min_segment_sec,
        "sensitivity": boundary_params.get("sensitivity", 0.15),
    }
    build_kwargs = {
        "min_segment_sec": min_segment_sec,
    }

    boundary_frames = detect_boundaries(energy_smooth, fps=fps, **detect_kwargs)
    segments = build_segments(
        frames, boundary_frames, fps=fps, energy=energy_smooth, **build_kwargs
    )

    labeling_metadata = {
        "enabled": enable_labeling,
        "strategy": None,
        "cached_hits": 0,
        "errors": 0,
    }

    return {
        "segments": segments,
        "metadata": {
            "total_frames": len(frames),
            "fps": fps,
            "n_segments": len(segments),
            "energy_stats": {
                "mean": float(np.mean(energy_smooth)),
                "max": float(np.max(energy_smooth)),
            },
            "labeling": labeling_metadata,
        },
        "debug": {"energy": energy_debug},
    }