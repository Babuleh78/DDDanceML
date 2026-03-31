"""Извлечение геометрических признаков из кватернионов скелета."""
import numpy as np
from typing import List, Dict, Optional
from .base import GeometricFeatures

# Вектор "вверх" в локальной системе координат
UP_VECTOR = np.array([0.0, 1.0, 0.0], dtype=np.float32)
FORWARD_VECTOR = np.array([0.0, 0.0, 1.0], dtype=np.float32)


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


def extract_geometric_features(
    frames: List[Dict],
    start_frame: int,
    end_frame: int,
    energy_values: Optional[np.ndarray] = None,
) -> GeometricFeatures:
    """
    Извлекает интерпретируемые признаки для сегмента движения.
    
    Работает напрямую с кватернионами из вашего пайплайна.
    """
    features = GeometricFeatures(
        arms_raised=False,
        arms_crossed=False,
        arms_extended_sideways=False,
        legs_bent=False,
        legs_wide_stance=False,
        torso_lean_forward=0.0,
        torso_lean_side=0.0,
        facing_direction="front",
        movement_intensity=0.0,
        symmetry_score=1.0,
        level="medium",
    )
    
    if start_frame >= end_frame or not frames:
        return features
    
    # Усредняем признаки по всем кадрам сегмента
    arm_up_count = 0
    arm_cross_count = 0
    arm_side_count = 0
    leg_bend_count = 0
    torso_lean_fwd = []
    torso_lean_side = []
    symmetry_scores = []
    y_positions = []
    
    # Кэш для костей в каждом кадре
    for t in range(start_frame, end_frame):
        frame = frames[t]
        bones = {b['name']: b for b in frame.get('bones', [])}
        
        if not bones:
            continue
            
        # === РУКИ ===
        left_arm = bones.get('mixamorig:LeftForeArm')
        right_arm = bones.get('mixamorig:RightForeArm')
        
        if left_arm and 'rotation' in left_arm:
            left_quat = np.array([
                left_arm['rotation'].get('w', 1.0),
                left_arm['rotation'].get('x', 0.0),
                left_arm['rotation'].get('y', 0.0),
                left_arm['rotation'].get('z', 0.0),
            ], dtype=np.float32)
            
            # Направление предплечья в глобальной системе
            forearm_dir = _get_bone_direction(left_quat, FORWARD_VECTOR)
            
            # Рука поднята, если вектор направлен вверх (Y > 0.5)
            if forearm_dir[1] > 0.5:
                arm_up_count += 1
                
            # Рука в сторону, если большая компонента по X
            if abs(forearm_dir[0]) > 0.7:
                arm_side_count += 1
                
        # === НОГИ ===
        left_leg = bones.get('mixamorig:LeftLeg')
        if left_leg and 'rotation' in left_leg:
            leg_quat = np.array([
                left_leg['rotation'].get('w', 1.0),
                left_leg['rotation'].get('x', 0.0),
                left_leg['rotation'].get('y', 0.0),
                left_leg['rotation'].get('z', 0.0),
            ], dtype=np.float32)
            
            # Голень направлена вниз в покое; если отклонение > 45° — нога согнута
            shin_dir = _get_bone_direction(leg_quat, -FORWARD_VECTOR)
            if shin_dir[1] > 0.3:  # Компонента вверх = сгибание
                leg_bend_count += 1
        
        # === КОРПУС ===
        spine = bones.get('mixamorig:Spine2')
        if spine and 'rotation' in spine:
            spine_quat = np.array([
                spine['rotation'].get('w', 1.0),
                spine['rotation'].get('x', 0.0),
                spine['rotation'].get('y', 0.0),
                spine['rotation'].get('z', 0.0),
            ], dtype=np.float32)
            
            # Наклон корпуса: проекция "вперёд" на вертикаль
            torso_fwd = _get_bone_direction(spine_quat, FORWARD_VECTOR)
            torso_lean_fwd.append(float(torso_fwd[1]))  # Y-компонента
            torso_lean_side.append(float(torso_fwd[0]))  # X-компонента
            
            # Направление взгляда (приближённо)
            if abs(torso_fwd[0]) > 0.6:
                features['facing_direction'] = "left" if torso_fwd[0] < 0 else "right"
            elif torso_fwd[2] < -0.5:
                features['facing_direction'] = "back"
        
        # === СИММЕТРИЯ ===
        if left_arm and right_arm and 'rotation' in left_arm and 'rotation' in right_arm:
            lq = np.array([left_arm['rotation'].get(k, 1.0 if k=='w' else 0.0) 
                          for k in ['w','x','y','z']], dtype=np.float32)
            rq = np.array([right_arm['rotation'].get(k, 1.0 if k=='w' else 0.0) 
                          for k in ['w','x','y','z']], dtype=np.float32)
            symmetry_scores.append(_compute_symmetry(lq, rq))
        
        # === УРОВЕНЬ (высота) ===
        # Приближённо: по Y-координате таза (если есть позиция)
        hip = bones.get('mixamorig:Hips')
        if hip and 'position' in hip:
            y_positions.append(float(hip['position'].get('y', 0.0)))
    
    # === Агрегация по сегменту ===
    n_frames = max(1, end_frame - start_frame)
    
    # Пороговые значения (настраиваются эмпирически)
    features['arms_raised'] = (arm_up_count / n_frames) > 0.6
    features['arms_extended_sideways'] = (arm_side_count / n_frames) > 0.5
    features['legs_bent'] = (leg_bend_count / n_frames) > 0.4
    
    # Усреднённый наклон корпуса
    if torso_lean_fwd:
        avg_fwd = np.mean(torso_lean_fwd)
        features['torso_lean_forward'] = float(np.clip(avg_fwd, -1.0, 1.0))
    if torso_lean_side:
        avg_side = np.mean(torso_lean_side)
        features['torso_lean_side'] = float(np.clip(avg_side, -1.0, 1.0))
    
    # Симметрия
    if symmetry_scores:
        features['symmetry_score'] = float(np.mean(symmetry_scores))
    
    # Уровень (по высоте таза)
    if y_positions:
        y_min, y_max = min(y_positions), max(y_positions)
        y_avg = np.mean(y_positions)
        if y_max - y_min > 0.1:  # Есть движение по вертикали
            normalized = (y_avg - y_min) / (y_max - y_min)
            if normalized < 0.33:
                features['level'] = "low"
            elif normalized > 0.66:
                features['level'] = "high"
    
    # Интенсивность из energy (если передана)
    if energy_values is not None and len(energy_values) > end_frame:
        seg_energy = energy_values[start_frame:end_frame]
        features['movement_intensity'] = float(np.mean(seg_energy))
    
    return features