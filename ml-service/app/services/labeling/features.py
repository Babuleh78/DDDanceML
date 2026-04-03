"""Извлечение геометрических признаков из кватернионов скелета."""
import numpy as np
from typing import List, Dict, Optional
from .base import GeometricFeatures

# Вектор "вверх" в локальной системе координат
UP_VECTOR = np.array([0.0, 1.0, 0.0], dtype=np.float32)
FORWARD_VECTOR = np.array([0.0, 0.0, 1.0], dtype=np.float32)

from app.services.py_module.skeleton_enerqy_quat import KEY_BONES, _quat_to_array, _quat_angular_distance

def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Кватернион [w, x, y, z] → матрица поворота 3×3.
    Формула из https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def _get_bone_direction(bone_quat: np.ndarray, local_axis: np.ndarray) -> np.ndarray:
    """Преобразует локальный вектор кости в глобальную систему координат."""
    rot_mat = _quat_to_rotation_matrix(bone_quat)
    return rot_mat @ local_axis


def _compute_symmetry(left_quat: np.ndarray, right_quat: np.ndarray) -> float:
    """
    Оценивает симметрию поворота парных костей.
    1.0 = идеальная симметрия, 0.0 = полная асимметрия.
    """
    # Сравниваем направления предплечий/голеней
    left_dir = _get_bone_direction(left_quat, FORWARD_VECTOR)
    right_dir = _get_bone_direction(right_quat, FORWARD_VECTOR)
    
    # Зеркалим правую сторону по оси X для сравнения
    right_dir_mirrored = np.array([-right_dir[0], right_dir[1], right_dir[2]])
    
    # Косинусное сходство
    dot = np.dot(left_dir, right_dir_mirrored)
    norm = np.linalg.norm(left_dir) * np.linalg.norm(right_dir_mirrored) + 1e-8
    return float(np.clip(dot / norm, 0.0, 1.0))


# app/services/labeling/features.py

def extract_geometric_features(
    frames: List[Dict],
    start_frame: int,
    end_frame: int,
    energy_values: Optional[np.ndarray] = None,
    segment_index: int = 0,
    total_segments: int = 1,
    prev_segment_label: Optional[str] = None,
) -> GeometricFeatures:
    """
    Извлекает расширенные интерпретируемые признаки для сегмента движения.
    
    Работает с кватернионами и позициями из формата Mixamo.
    """
    # Инициализация с дефолтными значениями
    features = GeometricFeatures(
        # Руки
        arms_raised=False,
        arms_crossed=False,
        arms_extended_sideways=False,
        arms_forward=False,
        arms_asymmetric=False,
        # Ноги
        legs_bent=False,
        legs_wide_stance=False,
        one_leg_lifted=False,
        weight_on_one_leg=False,
        # Корпус
        torso_lean_forward=0.0,
        torso_lean_side=0.0,
        torso_twist=0.0,
        facing_direction="front",
        # Динамика
        level="medium",
        movement_intensity=0.0,
        movement_type="smooth",
        # Сложность
        symmetry_score=1.0,
        active_joints_count=0,
        complexity="medium",
        # Контекст
        segment_position="middle",
        transition_from_prev=prev_segment_label,
    )
    
    if start_frame >= end_frame or not frames:
        return features
    
    # === Счётчики для агрегации ===
    n_frames = max(1, end_frame - start_frame)
    
    # Руки
    arm_up_count = 0
    arm_cross_count = 0
    arm_side_count = 0
    arm_fwd_count = 0
    arm_asym_count = 0
    
    # Ноги
    leg_bend_count = 0
    leg_wide_count = 0
    one_leg_up_count = 0
    weight_shift_count = 0
    
    # Корпус
    torso_lean_fwd = []
    torso_lean_side = []
    torso_twist_angles = []
    facing_votes = {"front": 0, "back": 0, "left": 0, "right": 0}
    
    # Уровень и симметрия
    y_positions = []
    symmetry_scores = []
    active_joint_flags = {bone: False for bone in KEY_BONES}
    
    # === Проход по кадрам сегмента ===
    for t in range(start_frame, end_frame):
        frame = frames[t]
        bones = {b['name']: b for b in frame.get('bones', [])}
        
        if not bones:
            continue
        
        # === РУКИ: левое и правое предплечье ===
        left_forearm = bones.get('mixamorig:LeftForeArm')
        right_forearm = bones.get('mixamorig:RightForeArm')
        
        if left_forearm and 'rotation' in left_forearm:
            left_quat = _quat_to_array(left_forearm['rotation'])
            left_dir = _get_bone_direction(left_quat, FORWARD_VECTOR)
            
            # Рука поднята (Y > 0.5)
            if left_dir[1] > 0.5:
                arm_up_count += 1
                active_joint_flags['mixamorig:LeftForeArm'] = True
            
            # Рука в сторону (|X| > 0.7)
            if abs(left_dir[0]) > 0.7:
                arm_side_count += 1
            
            # Рука вперёд (Z > 0.6)
            if left_dir[2] > 0.6:
                arm_fwd_count += 1
        
        if right_forearm and 'rotation' in right_forearm:
            right_quat = _quat_to_array(right_forearm['rotation'])
            right_dir = _get_bone_direction(right_quat, FORWARD_VECTOR)
            
            if right_dir[1] > 0.5:
                arm_up_count += 1
                active_joint_flags['mixamorig:RightForeArm'] = True
            
            if abs(right_dir[0]) > 0.7:
                arm_side_count += 1
            
            if right_dir[2] > 0.6:
                arm_fwd_count += 1
        
        # Асимметрия рук (разница направлений > 45°)
        if left_forearm and right_forearm and 'rotation' in left_forearm and 'rotation' in right_forearm:
            lq = _quat_to_array(left_forearm['rotation'])
            rq = _quat_to_array(right_forearm['rotation'])
            angle_diff = _quat_angular_distance(lq, rq)
            if angle_diff > 0.78:  # 45° в радианах
                arm_asym_count += 1
        
        # Скрещенные руки (расстояние между запястьями маленькое)
        left_wrist = bones.get('mixamorig:LeftHand')
        right_wrist = bones.get('mixamorig:RightHand')
        if left_wrist and right_wrist and 'position' in left_wrist and 'position' in right_wrist:
            lw_pos = np.array([left_wrist['position'].get(k, 0) for k in ['x','y','z']])
            rw_pos = np.array([right_wrist['position'].get(k, 0) for k in ['x','y','z']])
            if np.linalg.norm(lw_pos - rw_pos) < 0.3:
                arm_cross_count += 1
        
        # === НОГИ ===
        left_leg = bones.get('mixamorig:LeftLeg')
        right_leg = bones.get('mixamorig:RightLeg')
        
        if left_leg and 'rotation' in left_leg:
            leg_quat = _quat_to_array(left_leg['rotation'])
            shin_dir = _get_bone_direction(leg_quat, -FORWARD_VECTOR)  # голень направлена вниз
            
            # Нога согнута (компонента вверх > 0.3)
            if shin_dir[1] > 0.3:
                leg_bend_count += 1
                active_joint_flags['mixamorig:LeftLeg'] = True
        
        if right_leg and 'rotation' in right_leg:
            leg_quat = _quat_to_array(right_leg['rotation'])
            shin_dir = _get_bone_direction(leg_quat, -FORWARD_VECTOR)
            if shin_dir[1] > 0.3:
                leg_bend_count += 1
                active_joint_flags['mixamorig:RightLeg'] = True
        
        # Широкая стойка (расстояние между стопами > 0.4)
        left_foot = bones.get('mixamorig:LeftFoot')
        right_foot = bones.get('mixamorig:RightFoot')
        if left_foot and right_foot and 'position' in left_foot and 'position' in right_foot:
            lf_pos = np.array([left_foot['position'].get(k, 0) for k in ['x','y','z']])
            rf_pos = np.array([right_foot['position'].get(k, 0) for k in ['x','y','z']])
            stance_width = np.linalg.norm(lf_pos - rf_pos)
            if stance_width > 0.4:
                leg_wide_count += 1
            # Одна нога поднята (разница по Y > 0.3)
            if abs(lf_pos[1] - rf_pos[1]) > 0.3:
                one_leg_up_count += 1
        
        # Вес на одной ноге (таз смещён в сторону)
        hip = bones.get('mixamorig:Hips')
        if hip and 'position' in hip:
            hip_x = hip['position'].get('x', 0)
            if abs(hip_x) > 0.2:
                weight_shift_count += 1
            y_positions.append(float(hip['position'].get('y', 0.0)))
        
        # === КОРПУС ===
        spine = bones.get('mixamorig:Spine2')
        if spine and 'rotation' in spine:
            spine_quat = _quat_to_array(spine['rotation'])
            torso_fwd = _get_bone_direction(spine_quat, FORWARD_VECTOR)
            
            # Наклон вперёд/назад (Y-компонента)
            torso_lean_fwd.append(float(torso_fwd[1]))
            # Наклон вбок (X-компонента)
            torso_lean_side.append(float(torso_fwd[0]))
            
            # Скручивание: разница между направлением плеч и бёдер
            left_shoulder = bones.get('mixamorig:LeftShoulder')
            right_shoulder = bones.get('mixamorig:RightShoulder')
            if left_shoulder and right_shoulder and 'position' in left_shoulder and 'position' in right_shoulder:
                shoulder_vec = np.array([
                    right_shoulder['position'].get('x', 0) - left_shoulder['position'].get('x', 0),
                    0,
                    right_shoulder['position'].get('z', 0) - left_shoulder['position'].get('z', 0),
                ])
                # Угол между вектором плеч и направлением корпуса
                if np.linalg.norm(shoulder_vec) > 0.01:
                    shoulder_vec /= np.linalg.norm(shoulder_vec)
                    twist_cos = np.dot(shoulder_vec[:2], torso_fwd[:2])
                    twist_angle = np.degrees(np.arccos(np.clip(twist_cos, -1, 1)))
                    torso_twist_angles.append(float(twist_angle))
            
            # Направление взгляда (по корпусу)
            if abs(torso_fwd[0]) > 0.6:
                facing_votes["left" if torso_fwd[0] < 0 else "right"] += 1
            elif torso_fwd[2] < -0.5:
                facing_votes["back"] += 1
            else:
                facing_votes["front"] += 1
            
            active_joint_flags['mixamorig:Spine2'] = True
        
        # === СИММЕТРИЯ ===
        if left_forearm and right_forearm and 'rotation' in left_forearm and 'rotation' in right_forearm:
            lq = _quat_to_array(left_forearm['rotation'])
            rq = _quat_to_array(right_forearm['rotation'])
            symmetry_scores.append(_compute_symmetry(lq, rq))
    
    # === АГРЕГАЦИЯ по сегменту ===
    
    # Руки
    features['arms_raised'] = (arm_up_count / n_frames) > 0.6
    features['arms_extended_sideways'] = (arm_side_count / n_frames) > 0.5
    features['arms_forward'] = (arm_fwd_count / n_frames) > 0.5
    features['arms_crossed'] = (arm_cross_count / n_frames) > 0.3
    features['arms_asymmetric'] = (arm_asym_count / n_frames) > 0.4
    
    # Ноги
    features['legs_bent'] = (leg_bend_count / n_frames) > 0.4
    features['legs_wide_stance'] = (leg_wide_count / n_frames) > 0.5
    features['one_leg_lifted'] = (one_leg_up_count / n_frames) > 0.3
    features['weight_on_one_leg'] = (weight_shift_count / n_frames) > 0.4
    
    # Корпус: усреднённые наклоны
    if torso_lean_fwd:
        avg_fwd = np.mean(torso_lean_fwd)
        features['torso_lean_forward'] = float(np.clip(avg_fwd, -1.0, 1.0))
    if torso_lean_side:
        avg_side = np.mean(torso_lean_side)
        features['torso_lean_side'] = float(np.clip(avg_side, -1.0, 1.0))
    if torso_twist_angles:
        features['torso_twist'] = float(np.mean(torso_twist_angles))
    
    # Направление: большинство голосов
    if facing_votes:
        features['facing_direction'] = max(facing_votes, key=facing_votes.get)
    
    # Уровень (по высоте таза)
    if y_positions and max(y_positions) - min(y_positions) > 0.1:
        y_norm = (np.mean(y_positions) - min(y_positions)) / (max(y_positions) - min(y_positions))
        if y_norm < 0.33:
            features['level'] = "low"
        elif y_norm > 0.66:
            features['level'] = "high"
    
    # Симметрия
    if symmetry_scores:
        features['symmetry_score'] = float(np.mean(symmetry_scores))
    
    # Интенсивность из energy
    if energy_values is not None and len(energy_values) > end_frame:
        seg_energy = energy_values[start_frame:end_frame]
        features['movement_intensity'] = float(np.mean(seg_energy))
        
        # Тип движения по статистике энергии
        energy_std = np.std(seg_energy)
        if features['movement_intensity'] < 0.2:
            features['movement_type'] = "static"
        elif energy_std < 0.1:
            features['movement_type'] = "smooth"
        elif features['movement_intensity'] > 0.7:
            features['movement_type'] = "explosive"
        else:
            features['movement_type'] = "sharp"
    
    # Активные суставы и сложность
    active_count = sum(1 for v in active_joint_flags.values() if v)
    features['active_joints_count'] = active_count
    if active_count < 4:
        features['complexity'] = "simple"
    elif active_count > 7:
        features['complexity'] = "complex"
    
    # Позиция сегмента в хореографии
    if total_segments <= 3:
        features['segment_position'] = "start" if segment_index == 0 else "end"
    else:
        if segment_index == 0:
            features['segment_position'] = "start"
        elif segment_index == total_segments - 1:
            features['segment_position'] = "end"
        else:
            features['segment_position'] = "middle"
    
    return features