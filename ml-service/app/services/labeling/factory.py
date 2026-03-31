"""Фабрика стратегий разметки."""
from typing import Optional, Literal  # ← Optional для Python 3.9
import logging

from app.core.config import settings
from .base import LabelingStrategy
from .mock import MockStrategy

logger = logging.getLogger(__name__)


def get_labeling_strategy(
    backend: Optional[str] = None  # ← Простой str вместо Literal | None
) -> LabelingStrategy:
    """
    Возвращает стратегию разметки согласно конфигурации.
    
    Порядок выбора:
    1. Явно переданный backend (для тестов)
    2. labeling_backend из config
    3. Fallback на MockStrategy
    """
    # Если передан backend — используем его, иначе берём из настроек
    target = backend or settings.labeling_backend
    
    if target == "mock":
        logger.info("Using MockStrategy for labeling")
        return MockStrategy()
    
    if target == "openrouter":
        try:
            from .openrouter import OpenRouterStrategy
            strategy = OpenRouterStrategy()
            if strategy.is_available():
                logger.info(f"Using OpenRouterStrategy with model: {strategy.model}")
                return strategy
            else:
                logger.warning("OpenRouterStrategy not available (no API key), falling back to Mock")
        except ImportError as e:
            logger.warning(f"Could not import OpenRouterStrategy: {e}")
        except Exception as e:
            logger.warning(f"OpenRouterStrategy error: {e}")
    
    if target == "ollama":
        try:
            from .ollama import OllamaStrategy
            strategy = OllamaStrategy()
            if strategy.is_available():
                logger.info(f"Using OllamaStrategy with model: {strategy.model}")
                return strategy
            else:
                logger.warning("OllamaStrategy not available (server unreachable), falling back to Mock")
        except ImportError as e:
            logger.warning(f"Could not import OllamaStrategy: {e}")
        except Exception as e:
            logger.warning(f"OllamaStrategy error: {e}")
    
    # Fallback — всегда работает
    logger.warning(f"Using fallback MockStrategy (backend={target} unavailable)")
    return MockStrategy()