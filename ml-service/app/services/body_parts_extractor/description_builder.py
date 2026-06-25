import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def _classify_velocity(mean: float) -> str:
    if mean < 0.05:  return "неподвижна"
    if mean < 0.15:  return "едва движется"
    if mean < 0.35:  return "движется медленно"
    if mean < 0.70:  return "движется умеренно"
    if mean < 1.50:  return "движется быстро"
    return "движется очень быстро"


def _classify_smoothness(smoothness: float, jerk_mean: float = None) -> str:
    if jerk_mean is not None:
        if jerk_mean < 100: return "плавно"
        if jerk_mean < 260: return "относительно плавно"
        if jerk_mean < 430: return "с небольшими рывками"
        return "резко и отрывисто"
    if smoothness > 0.85: return "плавно"
    if smoothness > 0.65: return "относительно плавно"
    if smoothness > 0.45: return "с небольшими рывками"
    return "резко и отрывисто"


def _classify_rom(rom_max: float) -> str:
    if rom_max < 0.05:  return "практически на месте"
    if rom_max < 0.15:  return "с малой амплитудой"
    if rom_max < 0.35:  return "со средней амплитудой"
    if rom_max < 0.65:  return "с большой амплитудой"
    return "с максимальной амплитудой"


def _classify_angle(deg: float) -> str:
    if deg < 30:   return "полностью согнут"
    if deg < 70:   return "сильно согнут"
    if deg < 110:  return "согнут под прямым углом"
    if deg < 150:  return "слегка согнут"
    return "почти выпрямлен"


def _classify_tempo(bpm: float) -> str:
    if bpm < 1:    return "без выраженного ритма"
    if bpm < 60:   return "медленный темп"
    if bpm < 100:  return "умеренный темп"
    if bpm < 140:  return "быстрый темп"
    return "очень быстрый темп"



def generate_body_part_description(
    part_name: str,
    part_display_name: str,
    metrics: Dict,
) -> str:
    if not metrics or "error" in metrics:
        return ""

    vel   = metrics.get("velocity_stats", {})
    rom   = metrics.get("rom", {})
    vel_mean  = vel.get("mean", 0.0)
    vel_max   = vel.get("max",  0.0)
    rom_max   = rom.get("max_distance", 0.0)
    smoothness = metrics.get("smoothness", 0.5)
    jerk_mean  = metrics.get("jerk_mean", None)
    direction  = metrics.get("direction", {})

    velocity_label   = _classify_velocity(vel_mean)
    smoothness_label = _classify_smoothness(smoothness, jerk_mean)
    rom_label        = _classify_rom(rom_max)
    dir_label        = direction.get("direction_label", "")
    displacement     = direction.get("displacement_m", 0.0)

    parts = [f"{part_display_name} {velocity_label}"]

    if vel_mean >= 0.05:
        parts.append(f"({vel_mean:.2f} м/с, пик {vel_max:.2f} м/с)")

    if rom_max >= 0.05:
        parts.append(f"— {rom_label}")

    if dir_label and displacement >= 0.03:
        parts.append(f"— смещение {dir_label} ({displacement:.2f} м)")

    if vel_mean >= 0.05:
        parts.append(f"— движение {smoothness_label}")

    return " ".join(parts)



def generate_joint_angles_description(joint_angles: Dict) -> str:
    if not joint_angles:
        return ""

    lines = []
    active_joints = {
        k: v for k, v in joint_angles.items()
        if v.get("range_deg", 0) > 10.0
    }

    if not active_joints:
        return ""

    lines.append("Суставы:")
    for joint_name, data in active_joints.items():
        name        = data["name"]
        mean_deg    = data["mean_deg"]
        range_deg   = data["range_deg"]
        trend       = data["trend"]
        start_deg   = data["start_deg"]
        end_deg     = data["end_deg"]
        angle_label = _classify_angle(mean_deg)

        if joint_name == "torso_tilt":
            lean = data.get("lean_direction", "")
            lean_str = f", наклон {lean}" if lean and lean != "вертикально" else ""
            lines.append(
                f"  • {name}: {mean_deg:.0f}° от вертикали{lean_str}, "
                f"диапазон {range_deg:.0f}° ({start_deg:.0f}°→{end_deg:.0f}°)"
            )
        else:
            lines.append(
                f"  • {name}: в среднем {mean_deg:.0f}° ({angle_label}), "
                f"диапазон {range_deg:.0f}° ({start_deg:.0f}°→{end_deg:.0f}°), "
                f"{trend}"
            )

    return "\n".join(lines)



def generate_tempo_description(tempo: Dict) -> str:
    if not tempo:
        return ""

    bpm        = tempo.get("beats_per_min", 0.0)
    accents    = tempo.get("accent_count", 0)
    regularity = tempo.get("rhythm_regularity", 0.0)
    tempo_label = _classify_tempo(bpm)

    hz = round(bpm / 60.0, 1) if bpm > 0 else 0.0

    parts = [f"Темп: {tempo_label}"]
    if bpm >= 20:
        hz_str = f", {hz} Гц" if hz > 0 else ""
        parts.append(f"({bpm:.0f} BPM{hz_str})")
    if accents > 0:
        parts.append(f"— {accents} акцент(а) в движении")
    if regularity > 0.7:
        parts.append("— ритм регулярный")
    elif regularity > 0.4:
        parts.append("— ритм умеренно регулярный")
    elif accents > 1:
        parts.append("— ритм нерегулярный")

    return " ".join(parts)


def generate_symmetry_description(symmetry: Dict) -> str:
    if not symmetry:
        return ""

    lines = []
    pair_display = {
        "left_arm_vs_right_arm":   "Руки",
        "left_leg_vs_right_leg":   "Ноги",
        "left_hand_vs_right_hand": "Кисти",
    }

    for pair_name, data in symmetry.items():
        label         = data.get("label", "")
        dominant_side = data.get("dominant_side", "")
        vel_ratio     = data.get("velocity_ratio", 1.0)
        phase_offset  = data.get("phase_offset_sec", 0.0)
        display       = pair_display.get(pair_name, pair_name)

        line = f"  • {display}: {label}"
        if label != "симметричное":
            line += f" (доминирует {dominant_side}, соотношение скоростей {vel_ratio:.2f})"
        if abs(phase_offset) > 0.05:
            line += f", сдвиг фазы {phase_offset:+.2f} сек"
        lines.append(line)

    if not lines:
        return ""

    return "Симметрия:\n" + "\n".join(lines)


def build_segment_description(
    analysis: Dict,
    segment_index: int,
    segment_duration_sec: float,
) -> Dict:
    body_parts_analysis = analysis.get("body_parts", {})
    joint_angles        = analysis.get("joint_angles", {})
    tempo               = analysis.get("tempo", {})
    symmetry            = analysis.get("symmetry", {})

    active_parts = []
    part_descriptions = []

    for part_name, part_data in body_parts_analysis.items():
        if "error" in part_data:
            continue
        metrics  = part_data.get("metrics", {})
        vel_mean = metrics.get("velocity_stats", {}).get("mean", 0.0)

        if vel_mean > 0.05:
            active_parts.append(part_name)

        desc = generate_body_part_description(
            part_name,
            part_data["display_name"],
            metrics,
        )
        if desc:
            part_descriptions.append(desc)

    if not active_parts:
        overall = "Статичная поза — минимальное движение"
    elif len(active_parts) == 1:
        display = body_parts_analysis[active_parts[0]]["display_name"]
        overall = f"Изолированное движение: {display}"
    elif len(active_parts) <= 3:
        displays = [body_parts_analysis[p]["display_name"] for p in active_parts]
        overall = f"Частичное движение: {', '.join(displays)}"
    else:
        overall = f"Активное движение всего тела ({len(active_parts)} сегментов)"

    blocks = []
    if part_descriptions:
        blocks.append("Части тела:\n" + "\n".join(f"  • {d}" for d in part_descriptions))

    joints_desc = generate_joint_angles_description(joint_angles)
    if joints_desc:
        blocks.append(joints_desc)

    tempo_desc = generate_tempo_description(tempo)
    if tempo_desc:
        blocks.append(tempo_desc)

    symmetry_desc = generate_symmetry_description(symmetry)
    if symmetry_desc:
        blocks.append(symmetry_desc)

    detailed = "\n\n".join(blocks)

    return {
        "segment_index":      segment_index,
        "duration_seconds":   round(segment_duration_sec, 2),
        "overall_description": overall,
        "active_body_parts":  active_parts,
        "detailed_descriptions": detailed,
        "body_parts_count":   len(active_parts),
        "timestamp": (
            f"0:00 - "
            f"{int(segment_duration_sec // 60)}:"
            f"{int(segment_duration_sec % 60):02d}"
        ),
        "raw": {
            "joint_angles": joint_angles,
            "tempo":        tempo,
            "symmetry":     symmetry,
        },
    }


def enrich_segments_with_descriptions(
    segments: List[Dict],
    body_parts_analysis_per_segment: List[Dict],
    fps: float,
) -> List[Dict]:
    enriched = []
    for i, segment in enumerate(segments):
        seg_copy = dict(segment)
        if i < len(body_parts_analysis_per_segment):
            duration_frames = segment.get("end_frame", 0) - segment.get("start_frame", 0)
            duration_sec    = duration_frames / fps if fps > 0 else 0.0
            seg_copy["body_parts_description"] = build_segment_description(
                body_parts_analysis_per_segment[i],
                i,
                duration_sec,
            )
        enriched.append(seg_copy)
    return enriched