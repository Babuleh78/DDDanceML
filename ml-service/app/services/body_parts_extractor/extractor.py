"""
Главный модуль экстрактора движений.
Координирует анализ частей тела, углов суставов, темпа и симметрии.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .body_parts_groups import BODY_PARTS_GROUPS, JOINT_ANGLES, SYMMETRY_PAIRS
from .analyzer import analyze_segment_body_parts
from .description_builder import enrich_segments_with_descriptions, build_segment_description

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_body_parts_for_segments(
    segments,
    mixamo_frames,
    fps,
    body_parts_groups=None,
    joint_angles_def=None,
    symmetry_pairs=None,
):
    from .body_parts_groups import BODY_PARTS_GROUPS, JOINT_ANGLES, SYMMETRY_PAIRS
    from .analyzer import analyze_segment_body_parts
    from .description_builder import enrich_segments_with_descriptions
 
    if body_parts_groups is None:
        body_parts_groups = BODY_PARTS_GROUPS
    if joint_angles_def is None:
        joint_angles_def = JOINT_ANGLES
    if symmetry_pairs is None:
        symmetry_pairs = SYMMETRY_PAIRS
 
    logger.info(f"Analyzing {len(segments)} segments in parallel")
 
    analysis_list = [None] * len(segments)
 
    def _analyze_one(idx_segment):
        idx, segment = idx_segment
        try:
            return idx, analyze_segment_body_parts(
                segment, mixamo_frames, fps,
                body_parts_groups, joint_angles_def, symmetry_pairs,
            )
        except Exception as e:
            logger.error(f"Error in segment {idx}: {e}", exc_info=True)
            return idx, {}
 
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_analyze_one, (idx, seg)): idx
            for idx, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            analysis_list[idx] = result
 
    enriched = enrich_segments_with_descriptions(segments, analysis_list, fps)
    logger.info(f"Done: {len(segments)} segments analyzed")
    return enriched, analysis_list

def get_body_parts_report(analysis: Dict) -> str:
    lines = ["Отчёт анализа движения\n"]

    for part_name, part_data in analysis.get("body_parts", {}).items():
        if "error" in part_data:
            lines.append(f"{part_data['display_name']}: ОШИБКА — {part_data['error']}")
            continue
        m = part_data.get("metrics", {})
        vel = m.get("velocity_stats", {})
        rom = m.get("rom", {})
        direction = m.get("direction", {})
        lines.append(f"\n{part_data['display_name']}:")
        lines.append(f"  Скорость: mean={vel.get('mean',0):.3f}, max={vel.get('max',0):.3f} м/с")
        lines.append(f"  Jerk: mean={m.get('jerk_mean',0):.3f}  Плавность: {m.get('smoothness',0):.2f}")
        lines.append(f"  ROM: max={rom.get('max_distance',0):.3f} м")
        lines.append(f"  Направление: {direction.get('direction_label','?')} ({direction.get('displacement_m',0):.3f} м)")

    lines.append("\n Углы суставов ")
    for joint_name, data in analysis.get("joint_angles", {}).items():
        lines.append(
            f"  {data['name']}: {data['mean_deg']:.0f}° "
            f"(диапазон {data['range_deg']:.0f}°, {data['trend']})"
        )

    # Темп
    tempo = analysis.get("tempo", {})
    if tempo:
        lines.append(f"\n Темп ")
        lines.append(f"  {tempo.get('beats_per_min', 0):.0f} BPM, "
                     f"{tempo.get('accent_count', 0)} акцентов, "
                     f"регулярность {tempo.get('rhythm_regularity', 0):.2f}")

    # Симметрия
    lines.append("\n Симметрия ")
    for pair, data in analysis.get("symmetry", {}).items():
        lines.append(f"  {pair}: {data['label']}, "
                     f"vel_ratio={data['velocity_ratio']:.2f}, "
                     f"phase={data['phase_offset_sec']:+.2f}s")

    return "\n".join(lines)