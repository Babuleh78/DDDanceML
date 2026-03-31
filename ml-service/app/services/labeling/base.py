"""Базовые интерфейсы для стратегий разметки."""
from abc import ABC, abstractmethod
from typing import TypedDict, Optional
import hashlib
import json


class GeometricFeatures(TypedDict, total=False):
    """Интерпретируемые признаки движения для LLM."""
    # Позиционные признаки
    arms_raised: bool
    arms_crossed: bool
    arms_extended_sideways: bool
    legs_bent: bool
    legs_wide_stance: bool
    
    # Ориентация корпуса
    torso_lean_forward: float  # -1.0 (назад) ... 0.0 (вертикально) ... 1.0 (вперёд)
    torso_lean_side: float     # -1.0 (влево) ... 0.0 ... 1.0 (вправо)
    facing_direction: str      # "front", "left", "right", "back"
    
    # Динамика
    movement_intensity: float  # 0.0 ... 1.0 (из energy)
    symmetry_score: float      # 0.0 (асимметрия) ... 1.0 (полная симметрия)
    
    # Уровень (высота центра масс)
    level: str                 # "low", "medium", "high"


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