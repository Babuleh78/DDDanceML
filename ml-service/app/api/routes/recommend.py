import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import verify_internal_token
from app.api.routes.dependencies import get_recommender
from app.schemas.recommend import (
    RecommendRequest,
    RecommendResponse,
    SimilarRequest,
    SimilarResponse,
)
from app.services.recommender import UNDERGROUND_KEYWORDS

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["recommendation"],
    dependencies=[Depends(verify_internal_token)],
)


def _build_reasoning(query: str, count: int, is_underground: bool) -> str:
    if count == 0:
        return f"По запросу «{query}» подходящих танцев не найдено"
    if is_underground:
        return f"Найдено {count} нишевых танцев по запросу «{query}»"
    return f"Подобрано {count} танцев по запросу «{query}»"


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, recommender=Depends(get_recommender)):
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender is not initialized",
        )

    dances_input = [d.model_dump() for d in req.dances]
    results = recommender.recommend(req.query, dances_input, req.limit)

    is_underground = any(kw in req.query.lower() for kw in UNDERGROUND_KEYWORDS)
    reasoning = _build_reasoning(req.query, len(results), is_underground)

    return RecommendResponse(
        recommended_ids=[d["id"] for d in results],
        reasoning=reasoning,
    )


@router.post("/similar", response_model=SimilarResponse)
async def similar(req: SimilarRequest, recommender=Depends(get_recommender)):
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommender is not initialized",
        )

    dances_input = [d.model_dump() for d in req.dances]
    results = recommender.similar(req.dance_id, dances_input, req.limit)

    return SimilarResponse(recommended_ids=[d["id"] for d in results])
