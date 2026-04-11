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


def extract_body_parts_for_segments(
    segments: List[Dict],
    mixamo_frames: List[Dict],
    fps: float,
    body_parts_groups: Optional[Dict] = None,
    joint_angles_def: Optional[Dict] = None,
    symmetry_pairs: Optional[List[Tuple[str, str]]] = None,
) -> tuple[List[Dict], List[Dict]]:
    if body_parts_groups is None:
        body_parts_groups = BODY_PARTS_GROUPS
    if joint_angles_def is None:
        joint_angles_def = JOINT_ANGLES
    if symmetry_pairs is None:
        symmetry_pairs = SYMMETRY_PAIRS

    logger.info(f"Analyzing {len(segments)} segments")
    analysis_list = []

    for idx, segment in enumerate(segments):
        try:
            result = analyze_segment_body_parts(
                segment,
                mixamo_frames,
                fps,
                body_parts_groups,
                joint_angles_def,
                symmetry_pairs,
            )
            analysis_list.append(result)
        except Exception as e:
            logger.error(f"Error in segment {idx}: {e}", exc_info=True)
            analysis_list.append({})

    enriched = enrich_segments_with_descriptions(segments, analysis_list, fps)
    logger.info(f"Done: {len(segments)} segments analyzed")
    return enriched, analysis_list


def get_body_parts_report(analysis: Dict) -> str:
    """Текстовый отчёт для отладки."""
    lines = ["=== Отчёт анализа движения ===\n"]

    # Части тела
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

    # Углы суставов
    lines.append("\n=== Углы суставов ===")
    for joint_name, data in analysis.get("joint_angles", {}).items():
        lines.append(
            f"  {data['name']}: {data['mean_deg']:.0f}° "
            f"(диапазон {data['range_deg']:.0f}°, {data['trend']})"
        )

    # Темп
    tempo = analysis.get("tempo", {})
    if tempo:
        lines.append(f"\n=== Темп ===")
        lines.append(f"  {tempo.get('beats_per_min', 0):.0f} BPM, "
                     f"{tempo.get('accent_count', 0)} акцентов, "
                     f"регулярность {tempo.get('rhythm_regularity', 0):.2f}")

    # Симметрия
    lines.append("\n=== Симметрия ===")
    for pair, data in analysis.get("symmetry", {}).items():
        lines.append(f"  {pair}: {data['label']}, "
                     f"vel_ratio={data['velocity_ratio']:.2f}, "
                     f"phase={data['phase_offset_sec']:+.2f}s")

    return "\n".join(lines)