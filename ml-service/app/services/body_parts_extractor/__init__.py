"""
Body Parts Extractor - модуль для анализа движений по частям тела на основе MediaPipe/Mixamo.

Основные компоненты:
- extractor.py: главный интерфейс
- analyzer.py: анализ метрик движения (скорость, ускорение, ROM)
- description_builder.py: генерация текстовых описаний
- body_parts_groups.py: определение групп костей
"""

from .extractor import extract_body_parts_for_segments, get_body_parts_report
from .body_parts_groups import BODY_PARTS_GROUPS

__all__ = [
    "extract_body_parts_for_segments",
    "get_body_parts_report",
    "BODY_PARTS_GROUPS",
]
