"""Базовые интерфейсы для стратегий разметки."""
from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Literal
import hashlib
import json


class GeometricFeatures(TypedDict, total=False):
    # === Позиционные признаки (руки) ===
    arms_raised: bool                    # руки выше плеч
    arms_crossed: bool                   # руки скрещены перед грудью
    arms_extended_sideways: bool         # руки в стороны (горизонтально)
    arms_forward: bool                   # руки вытянуты вперёд
    arms_asymmetric: bool                # одна рука выше/в стороне от другой
    
    # === Позиционные признаки (ноги) ===
    legs_bent: bool                      # колени согнуты
    legs_wide_stance: bool               # ноги широко расставлены
    one_leg_lifted: bool                 # одна нога оторвана от земли
    weight_on_one_leg: bool              # вес тела на одной ноге
    
    # === Ориентация корпуса ===
    torso_lean_forward: float            # -1.0 (назад) ... 0.0 ... 1.0 (вперёд)
    torso_lean_side: float               # -1.0 (влево) ... 0.0 ... 1.0 (вправо)
    torso_twist: float                   # угол скручивания корпуса (-180...180)
    facing_direction: Literal["front", "back", "left", "right"]
    
    # === Уровень и динамика ===
    level: Literal["low", "medium", "high"]  # высота центра масс
    movement_intensity: float            # 0.0 ... 1.0
    movement_type: Literal["static", "smooth", "sharp", "explosive"]
    
    # === Симметрия и сложность ===
    symmetry_score: float                # 0.0 (асимметрия) ... 1.0 (симметрия)
    active_joints_count: int             # сколько суставов активно двигаются
    complexity: Literal["simple", "medium", "complex"]
    
    # === Контекст сегмента ===
    segment_position: Literal["start", "middle", "end"]  # место в хореографии
    transition_from_prev: Optional[str]  # краткое описание предыдущего сегмента


class LabelingStrategy(ABC):
    """Абстракция стратегии генерации названий движений."""
    
    name: str  # Идентификатор стратегии для метаданных
    
    @abstractmethod
    async def generate_label(
        self, 
        features: GeometricFeatures, 
        duration_sec: float
    ) -> str:
        """
        Генерирует человекочитаемое название движения.
        
        Args:
            features: Словарь геометрических признаков
            duration_sec: Длительность сегмента в секундах
            
        Returns:
            Короткое название на русском (3-5 слов)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверяет доступность бэкенда стратегии."""
        pass
    
    @staticmethod
    def compute_features_hash(features: GeometricFeatures, duration_sec: float) -> str:
        """Вычисляет SHA256-хэш для кэширования."""
        payload = json.dumps({**features, "duration": duration_sec}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]