"""Стратегия разметки через локальный Ollama."""
import asyncio
import httpx
import logging
from typing import Optional, List

from app.core.config import settings
from .base import LabelingStrategy, GeometricFeatures
from .cache import label_cache

logger = logging.getLogger(__name__)


class OllamaStrategy(LabelingStrategy):
    name = "ollama"

    def __init__(self):
        self.host = settings.ollama_host
        self.model = settings.labeling_model_ollama or "qwen2.5:7b"
        self.timeout = httpx.Timeout(120.0, connect=5.0)

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

        # app/services/labeling/ollama.py

    def _build_prompt(self, features: GeometricFeatures, duration_sec: float) -> str:
        """
        Формирует структурированный промпт с естественным описанием признаков
        и few-shot примерами для стабильного вывода.
        """
        
        # === Форматирование признаков в человекочитаемый вид ===
        
        # Руки
        arms_desc_parts = []
        if features.get('arms_raised'):
            arms_desc_parts.append("подняты вверх")
        if features.get('arms_crossed'):
            arms_desc_parts.append("скрещены перед грудью")
        if features.get('arms_extended_sideways'):
            arms_desc_parts.append("разведены в стороны")
        if features.get('arms_forward'):
            arms_desc_parts.append("вытянуты вперёд")
        if features.get('arms_asymmetric'):
            arms_desc_parts.append("в асимметричном положении")
        arms_desc = " и ".join(arms_desc_parts) if arms_desc_parts else "опущены вдоль тела"
        
        # Ноги
        legs_desc_parts = []
        if features.get('legs_bent'):
            legs_desc_parts.append("согнуты в коленях")
        if features.get('legs_wide_stance'):
            legs_desc_parts.append("широко расставлены")
        if features.get('one_leg_lifted'):
            legs_desc_parts.append("одна нога поднята")
        if features.get('weight_on_one_leg'):
            legs_desc_parts.append("вес на одной ноге")
        legs_desc = " и ".join(legs_desc_parts) if legs_desc_parts else "прямые, стопы вместе"
        
        # Корпус
        torso_parts = []
        lean_fwd = features.get('torso_lean_forward', 0)
        if abs(lean_fwd) > 0.3:
            direction = "вперёд" if lean_fwd > 0 else "назад"
            torso_parts.append(f"наклон корпуса {direction}")
        lean_side = features.get('torso_lean_side', 0)
        if abs(lean_side) > 0.3:
            direction = "влево" if lean_side < 0 else "вправо"
            torso_parts.append(f"наклон {direction}")
        if abs(features.get('torso_twist', 0)) > 20:
            torso_parts.append("корпус скручен")
        torso_desc = " и ".join(torso_parts) if torso_parts else "корпус вертикально"
        
        # Динамика и уровень
        intensity_desc = {
            "static": "почти без движения",
            "smooth": "плавное движение",
            "sharp": "резкое движение",
            "explosive": "взрывное, быстрое движение"
        }.get(features.get('movement_type', 'smooth'), "движение")
        
        level_desc = {
            "low": "низкий уровень (присед, наклон)",
            "medium": "средний уровень",
            "high": "высокий уровень (подъём, прыжок)"
        }.get(features.get('level', 'medium'), "средний уровень")
        
        # Симметрия и сложность
        symmetry_desc = "симметрично" if features.get('symmetry_score', 1) > 0.7 else "асимметрично"
        complexity_desc = features.get('complexity', 'medium')
        
        # Контекст сегмента
        position_desc = {
            "start": "начало движения",
            "middle": "середина хореографии",
            "end": "завершение движения"
        }.get(features.get('segment_position', 'middle'), "")
        
        prev_context = f" после «{features.get('transition_from_prev')}»" if features.get('transition_from_prev') else ""
        
       
        
        # === Основной промпт ===
        return f"""Ты эксперт по хореографии и создаёшь ПОНЯТЫЕ названия для танцевального туториала.

    ЗАДАЧА: Дай короткое название движению

    ПРАВИЛА:
    1. Не больше 7 слов
    2. Только название НА РУССКОМ ЯЗЫКЕ, без пояснений, знаков препинания и кавычек

    ДАННЫЕ О ДВИЖЕНИИ:
    • Длительность: {duration_sec:.1f} сек
    • Контекст: {position_desc}{prev_context}

    Положение тела:
    • Руки: {arms_desc}
    • Ноги: {legs_desc}
    • Корпус: {torso_desc}, смотрит {features.get('facing_direction', 'вперёд')}

    Характер движения:
    • Уровень: {level_desc}
    • Тип: {intensity_desc}
    • Симметрия: {symmetry_desc}

    Название движения:"""

    async def _generate_one(
        self,
        client: httpx.AsyncClient,
        features: GeometricFeatures,
        duration_sec: float,
        segment_idx: int,
    ) -> str:
        """Генерирует label для одного сегмента, используя переданный клиент."""
        cache_key = self.compute_features_hash(features, duration_sec)
        cached = label_cache.get(cache_key)
        
        prompt = self._build_prompt(features, duration_sec)

        response = await client.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": 30,
                    "top_p": 0.95,
                },
            },
        )
        response.raise_for_status()

        result = response.json()
        label = result["response"].strip().strip('"\'').strip()

        label_cache.set(cache_key, label)
        logger.info(f"Segment #{segment_idx}: label='{label}'")
        return label

    async def generate_label(
        self,
        features: GeometricFeatures,
        duration_sec: float,
    ) -> str:
        """Одиночный вызов — для совместимости с интерфейсом."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._generate_one(client, features, duration_sec, -1)

    async def generate_labels_batch(
        self,
        items: List[tuple],  # [(features, duration_sec, segment_idx), ...]
    ) -> List[str]:
        """
        Параллельная генерация для всех сегментов через один HTTP-клиент.
        Ollama на CPU обрабатывает запросы последовательно внутри,
        но мы не тратим время на создание нового клиента для каждого сегмента.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._generate_one(client, features, duration_sec, idx)
                for features, duration_sec, idx in items
            ]
            # gather с return_exceptions=True — один упавший сегмент не убьёт остальные
            results = await asyncio.gather(*tasks, return_exceptions=True)

        labels = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Segment #{items[i][2]}: labeling failed: {result}")
                labels.append(f"segment_{items[i][2] + 1}")
            else:
                labels.append(result)
        return labels