from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ReelsRecommender:
    """Personalized reels feed recommender.

    Reuses the SentenceTransformer model already loaded in DanceRecommender.
    Inject the DanceRecommender instance on construction so the model is not
    loaded twice.
    """

    def __init__(self, dance_recommender: Any) -> None:
        self._dr = dance_recommender


    def _norm(self, values: np.ndarray) -> np.ndarray:
        mn, mx = float(values.min()), float(values.max())
        if mx == mn:
            return np.full(len(values), 0.5, dtype=float)
        return (values - mn) / (mx - mn)

    _BEHAVIOR_SCORE_MAP = {'attempted': 0.3, 'watched_full': 0.1, 'skipped_fast': -0.2}

    def _compute_behavior_bonus(self, candidates: list[dict], behavior_log: list[dict] | None) -> np.ndarray:
        bonus = np.zeros(len(candidates), dtype=float)
        if not behavior_log:
            return bonus
        id_to_idx = {d['id']: i for i, d in enumerate(candidates)}
        for entry in behavior_log:
            did = entry.get('dance_id', '')
            action = entry.get('action', '')
            score = self._BEHAVIOR_SCORE_MAP.get(action, 0.0)
            if score != 0.0 and did in id_to_idx:
                bonus[id_to_idx[did]] += score
        return bonus


    def recommend(
        self,
        user_history: list[dict],
        candidate_dances: list[dict],
        limit: int = 10,
        exclude_ids: list[Any] | None = None,
        behavior_log: list[dict] | None = None,
        friend_uploader_ids: list[str] | None = None,
    ) -> list:
        exclude_set = set(exclude_ids or [])
        candidates = [d for d in candidate_dances if d['id'] not in exclude_set]

        if not candidates:
            return []

        history_ids = {h['dance_id'] for h in user_history}

        if not user_history:
            avg_scores = np.array([float(d.get('avg_score') or 0) for d in candidates])
            view_counts = np.array([float(d.get('view_count') or 0) for d in candidates])
            pop = self._norm(avg_scores) * 0.6 + self._norm(np.log1p(view_counts)) * 0.4
            behavior_bonus = self._compute_behavior_bonus(candidates, behavior_log)
            friend_set = set(friend_uploader_ids or [])
            friend_boost = np.array(
                [0.1 if d.get('uploader_id', '') in friend_set and friend_set else 0.0
                 for d in candidates],
                dtype=float,
            )
            final_pop = pop + behavior_bonus + friend_boost
            top20_idx = list(np.argsort(final_pop)[::-1][:20])
            random.shuffle(top20_idx)
            return [candidates[i]['id'] for i in top20_idx[:limit]]

        model = self._dr._model
        if model is not None:
            self._dr._ensure_cached(candidates)

            with self._dr._lock:
                history_cand_embs = [
                    self._dr._cache[d['id']]
                    for d in candidates
                    if d['id'] in history_ids and d['id'] in self._dr._cache
                ]
                cand_embs = np.stack([self._dr._cache[d['id']] for d in candidates])

            if history_cand_embs:
                mean_hist = np.mean(np.stack(history_cand_embs), axis=0)
                norms_c = np.linalg.norm(cand_embs, axis=1, keepdims=True)
                norms_c = np.where(norms_c == 0, 1e-9, norms_c)
                norm_h = max(float(np.linalg.norm(mean_hist)), 1e-9)
                sim = (cand_embs / norms_c) @ (mean_hist / norm_h)
            else:
                sim = np.zeros(len(candidates), dtype=float)
        else:
            sim = np.zeros(len(candidates), dtype=float)

        avg_scores = np.array([float(d.get('avg_score') or 0) for d in candidates])
        view_counts = np.array([float(d.get('view_count') or 0) for d in candidates])
        popularity = self._norm(avg_scores) * 0.6 + self._norm(np.log1p(view_counts)) * 0.4

        novelty = np.zeros(len(candidates), dtype=float)
        now = datetime.now(tz=timezone.utc)
        week_ago = now - timedelta(days=7)
        for i, d in enumerate(candidates):
            created_at = d.get('created_at')
            if created_at:
                try:
                    if isinstance(created_at, str):
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        dt = created_at
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt > week_ago:
                        novelty[i] += 0.2
                except Exception:
                    pass
            if d['id'] in history_ids:
                novelty[i] -= 1.0

        behavior_bonus = self._compute_behavior_bonus(candidates, behavior_log)

        friend_set = set(friend_uploader_ids or [])
        friend_boost = np.array(
            [0.1 if d.get('uploader_id', '') in friend_set and friend_set else 0.0
             for d in candidates],
            dtype=float,
        )

        final = sim * 0.4 + popularity * 0.3 + novelty * 0.3 + behavior_bonus + friend_boost

        top_indices = list(np.argsort(final)[::-1][:limit])
        return [candidates[i]['id'] for i in top_indices]
