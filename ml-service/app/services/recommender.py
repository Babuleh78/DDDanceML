from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
_MAX_CACHE_SIZE = 10_000

UNDERGROUND_KEYWORDS = {
    'андеграунд',
    'underground',
    'необычный',
    'необычное',
    'необычную',
    'редкий',
    'редкое',
    'редкую',
    'нишевый',
    'нишевое',
    'нишевую',
}


class DanceRecommender:
    def __init__(self) -> None:
        self._model = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info('Loading sentence-transformer model: %s', MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info('Sentence-transformer model loaded')

    def _dance_text(self, dance: dict) -> str:
        title = dance.get('title') or ''
        description = dance.get('description') or ''
        parts = [p for p in (title, description) if p]
        return '. '.join(parts)

    def _is_underground(self, query: str) -> bool:
        lower = query.lower()
        return any(kw in lower for kw in UNDERGROUND_KEYWORDS)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        mn, mx = float(values.min()), float(values.max())
        if mx == mn:
            return np.zeros_like(values, dtype=float)
        return (values - mn) / (mx - mn)

    def _ensure_cached(self, dances: list[dict]) -> None:
        if self._model is None:
            return
        to_encode_texts: list[str] = []
        to_encode_ids: list[str] = []
        with self._lock:
            for dance in dances:
                did = dance['id']
                if did not in self._cache:
                    to_encode_texts.append(self._dance_text(dance))
                    to_encode_ids.append(did)

        if not to_encode_texts:
            return

        embeddings = self._model.encode(to_encode_texts, convert_to_numpy=True, show_progress_bar=False)

        with self._lock:
            for did, emb in zip(to_encode_ids, embeddings):
                self._cache[did] = emb
                self._cache.move_to_end(did)
            while len(self._cache) > _MAX_CACHE_SIZE:
                self._cache.popitem(last=False)

    def recommend(self, query: str, dances: list[dict], limit: int = 5) -> list[dict]:
        if self._model is None or not dances:
            return []

        self._ensure_cached(dances)

        dance_ids = [d['id'] for d in dances]
        with self._lock:
            dance_embeddings = np.stack([self._cache[did] for did in dance_ids])

        query_embedding = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

        norms_d = np.linalg.norm(dance_embeddings, axis=1, keepdims=True)
        norms_d = np.where(norms_d == 0, 1e-9, norms_d)
        norm_q = max(float(np.linalg.norm(query_embedding)), 1e-9)
        sim = (dance_embeddings / norms_d) @ (query_embedding / norm_q)

        avg_scores = np.array([float(d.get('avg_score') or 0) for d in dances])
        view_counts = np.array([float(d.get('view_count') or 0) for d in dances])

        norm_avg = self._normalize(avg_scores)
        norm_views = self._normalize(view_counts)

        if self._is_underground(query):
            final = sim * 0.6 + norm_avg * 0.3 - norm_views * 0.1
        else:
            final = sim * 0.6 + norm_avg * 0.2 + norm_views * 0.2

        top_indices = list(np.argsort(final)[::-1][:limit])
        return [dances[i] for i in top_indices]

    def similar(self, dance_id: str, dances: list[dict], limit: int = 4) -> list[dict]:
        """Item-to-item похожие танцы.

        Использует уже закэшированный эмбеддинг целевого танца — БЕЗ повторного
        прогона трансформера. На прогретом кэше это чистый numpy (косинус
        вектора против матрицы кандидатов), что быстро работает на CPU.
        """
        if self._model is None or not dances:
            return []

        self._ensure_cached(dances)

        with self._lock:
            target = self._cache.get(dance_id)
            if target is None:
                return []
            dance_ids = [d['id'] for d in dances]
            dance_embeddings = np.stack([self._cache[did] for did in dance_ids])

        norms_d = np.linalg.norm(dance_embeddings, axis=1, keepdims=True)
        norms_d = np.where(norms_d == 0, 1e-9, norms_d)
        norm_t = max(float(np.linalg.norm(target)), 1e-9)
        sim = (dance_embeddings / norms_d) @ (target / norm_t)

        avg_scores = np.array([float(d.get('avg_score') or 0) for d in dances])
        final = sim * 0.85 + self._normalize(avg_scores) * 0.15

        order = np.argsort(final)[::-1]
        result: list[dict] = []
        for i in order:
            if dances[i]['id'] == dance_id:
                continue
            result.append(dances[i])
            if len(result) >= limit:
                break
        return result

    def invalidate(self, dance_id: str | None = None) -> None:
        with self._lock:
            if dance_id is None:
                self._cache.clear()
            else:
                self._cache.pop(dance_id, None)
