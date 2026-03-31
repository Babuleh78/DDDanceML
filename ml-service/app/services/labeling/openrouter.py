"""Стратегия разметки через OpenRouter API."""
import httpx
import logging
from typing import Optional

from app.core.config import settings
from .base import LabelingStrategy, GeometricFeatures
from .cache import label_cache

logger = logging.getLogger(__name__)


class OpenRouterStrategy(LabelingStrategy):
    name = "openrouter"
    
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = settings.openrouter_api_key
        self.model = settings.labeling_model
        self.timeout = httpx.Timeout(30.0, connect=10.0)
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _build_prompt(self, features: GeometricFeatures, duration_sec: float) -> str:
        """Формирует структурированный промпт для LLM."""
        return f"""Ты эксперт по хореографии. Опиши движение для танцевального туториала.

Длительность: {duration_sec:.1f} сек
Признаки:
• Руки подняты: {features.get('arms_raised', False)}
• Руки в стороны: {features.get('arms_extended_sideways', False)}
• Ноги согнуты: {features.get('legs_bent', False)}
• Наклон корпуса вперёд: {features.get('torso_lean_forward', 0):.2f}
• Наклон корпуса вбок: {features.get('torso_lean_side', 0):.2f}
• Направление: {features.get('facing_direction', 'front')}
• Уровень: {features.get('level', 'medium')}
• Симметрия: {features.get('symmetry_score', 1):.2f}
• Интенсивность: {features.get('movement_intensity', 0):.2f}

Задача: дай короткое название движению на русском языке.
Требования:
- Только 3-5 слов
- Без пояснений, только название
- Используй танцевальную лексику если уместно

Примеры:
- "подъём рук с поворотом"
- "прыжок с разворотом"
- "плавный наклон вправо"

Название:"""
    
    async def generate_label(
        self,
        features: GeometricFeatures,
        duration_sec: float
    ) -> str:
        # Проверяем кэш
        cache_key = self.compute_features_hash(features, duration_sec)
        cached = label_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        if not self.is_available():
            raise RuntimeError("OpenRouter API key not configured")
        
        prompt = self._build_prompt(features, duration_sec)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://dddance.ml",
                        "X-Title": "DDDanceML",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0.3,
                    }
                )
                response.raise_for_status()
                result = response.json()
                label = result["choices"][0]["message"]["content"].strip()
                
                # Сохраняем в кэш
                label_cache.set(cache_key, label)
                logger.debug(f"Generated label: '{label}'")
                
                return label
                
        except httpx.TimeoutException:
            logger.warning("OpenRouter request timed out")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouterStrategy: {e}")
            raise