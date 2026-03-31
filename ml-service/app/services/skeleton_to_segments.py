"""Обработка скелета: сегментация + семантическая разметка."""
import logging
import numpy as np
from typing import Optional, List, Dict

from .labeling.factory import get_labeling_strategy
from .labeling.features import extract_geometric_features, GeometricFeatures
from .labeling.cache import label_cache

# === Импортируем функции, совместимые с форматом Mixamo (bones + quaternions) ===
from .py_module.skeleton_enerqy_quat import compute_energy, detect_boundaries, build_segments

logger = logging.getLogger(__name__)


async def process_skeleton_to_segments(
    frames: List[Dict],
    fps: float,
    energy_params: Optional[Dict] = None,
    boundary_params: Optional[Dict] = None,
    enable_labeling: bool = True,
) -> Dict:
    """Полный пайплайн: скелет → сегменты → (опционально) LLM-названия."""
    
    # === Шаг 1: Вычисляем энергию движения ===
    energy_params = energy_params or {}
    energy_smooth, energy_debug = compute_energy(frames, **energy_params)
    
    # === Шаг 2: Детектируем границы сегментов ===
    boundary_params = boundary_params or {}
    boundary_frames = detect_boundaries(energy_smooth, fps=fps, **boundary_params)
    
    # === Шаг 3: Строим сегменты ===
    segments = build_segments(
        frames, boundary_frames, fps=fps, energy=energy_smooth, **boundary_params
    )
    
    # === Шаг 4: Семантическая разметка через LLM ===
    labeling_metadata = {
        "enabled": enable_labeling,
        "strategy": None,
        "cached_hits": 0,
        "errors": 0,
    }
    
    if enable_labeling:
        strategy = get_labeling_strategy()
        labeling_metadata["strategy"] = strategy.name
        logger.info(f"Starting labeling for {len(segments)} segments with {strategy.name}")
        
        for i, segment in enumerate(segments):
            try:
                features: GeometricFeatures = extract_geometric_features(
                    frames,
                    segment['start_frame'],
                    segment['end_frame'],
                    energy_values=energy_smooth,
                )
                
                cache_key = strategy.compute_features_hash(features, segment['duration_sec'])
                if label_cache.get(cache_key):
                    labeling_metadata["cached_hits"] += 1
                    continue
                
                label = await strategy.generate_label(features, segment['duration_sec'])
                segment['label'] = label
                segment['label_source'] = strategy.name
                segment['features_hash'] = cache_key
                
            except Exception as e:
                logger.warning(f"Failed to label segment {i}: {e}")
                labeling_metadata["errors"] += 1
                segment['label_source'] = "fallback"
    
    # === Шаг 5: Формируем ответ ===
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
            "cache_stats": label_cache.stats(),
        },
        "debug": {"energy": energy_debug},
    }