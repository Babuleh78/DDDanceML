"""Стратегия разметки через локальный Ollama."""
import httpx
import logging
from typing import Optional

from app.core.config import settings
from .base import LabelingStrategy, GeometricFeatures
from .cache import label_cache

logger = logging.getLogger(__name__)


class OllamaStrategy(LabelingStrategy):
    """Локальная стратегия через Ollama API."""
    name = "ollama"
    
    def __init__(self):
        self.host = settings.ollama_host  # http://localhost:11434
        self.model = settings.labeling_model_ollama or "llama3.2:3b"
        self.timeout = httpx.Timeout(60.0, connect=5.0)
    
    def is_available(self) -> bool:
        """Проверяет доступность Ollama сервера."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
    
    def _build_prompt(self, features: GeometricFeatures, duration_sec: float) -> str:
        """Формирует компактный промпт для локальной модели."""
        # Локальные модели лучше работают с короткими чёткими промптами
        return (
            f"Ты эксперт по хореографии. Дай короткое название движению на русском.\n"
            f"Правила: не больше 7 слов, только название, без пояснений.\n"
            f"Длительность: {duration_sec:.1f} сек\n"
            f"Признаки: руки_{'подняты' if features.get('arms_raised') else 'опущены'}, "
            f"ноги_{'согнуты' if features.get('legs_bent') else 'прямые'}, "
            f"наклон_корпуса={features.get('torso_lean_forward', 0):.1f}, "
            f"интенсивность={features.get('movement_intensity', 0):.2f}\n"
            f"Название:"
        )
    
    async def generate_label(
        self,
        features: GeometricFeatures,
        duration_sec: float
    ) -> str:
        # Кэширование
        cache_key = self.compute_features_hash(features, duration_sec)
        cached = label_cache.get(cache_key)
        if cached:
            logger.debug(f"Ollama cache hit: {cache_key}")
            return cached
        
        if not self.is_available():
            raise RuntimeError(f"Ollama not available at {self.host}")
        
        prompt = self._build_prompt(features, duration_sec)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,  # меньше креатива, больше стабильности
                            "num_predict": 30,   # ограничиваем длину ответа
                            "top_p": 0.9,
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()  # ← единственный вызов
                label = result["response"].strip()
                label = label.strip('"\'').strip()
                label_cache.set(cache_key, label)
                logger.info(f"✅ Got label: '{label}'")
                return label
                
        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout for model {self.model}")
            raise
        except Exception as e:
            logger.error(f"Ollama request failed: {type(e).__name__}: {e}")
            raise