"""Fallback-стратегия для тестов и при недоступности LLM."""
from .base import LabelingStrategy, GeometricFeatures


class MockStrategy(LabelingStrategy):
    """Возвращает детерминированное название на основе хэша признаков."""
    name = "mock"
    
    _labels_pool = [
        "движение руками",
        "поворот корпуса", 
        "шаг с переносом веса",
        "подъём ноги",
        "наклон в сторону",
        "прыжок на месте",
        "скрещивание рук",
        "волна корпусом",
        "фиксация позы",
        "переход между уровнями",
    ]
    
    def is_available(self) -> bool:
        return True  # Всегда доступен
    
    async def generate_label(
        self,
        features: GeometricFeatures,
        duration_sec: float
    ) -> str:
        # Детерминированный выбор по хэшу
        cache_key = self.compute_features_hash(features, duration_sec)
        idx = int(cache_key, 16) % len(self._labels_pool)
        return self._labels_pool[idx]