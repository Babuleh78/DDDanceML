import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import verify_internal_token
from app.api.routes.dependencies import get_reels_recommender
from app.schemas.reels import ReelsFeedRequest, ReelsFeedResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["reels"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/reels_feed", response_model=ReelsFeedResponse)
async def reels_feed(req: ReelsFeedRequest, reels_recommender=Depends(get_reels_recommender)):
    if not req.candidate_dances:
        return ReelsFeedResponse(recommended_ids=[])

    if reels_recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ReelsRecommender is not initialized",
        )

    t0 = time.monotonic()

    history = [h.model_dump() for h in req.user_history]
    candidates = [c.model_dump() for c in req.candidate_dances]
    behavior_log = [b.model_dump() for b in req.behavior_log]

    recommended_ids = reels_recommender.recommend(
        user_history=history,
        candidate_dances=candidates,
        limit=req.limit,
        exclude_ids=req.exclude_ids,
        behavior_log=behavior_log,
        friend_uploader_ids=req.friend_uploader_ids,
    )

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "reels_feed: %d candidates, %d history items → %d results in %.1f ms",
        len(candidates),
        len(history),
        len(recommended_ids),
        elapsed_ms,
    )

    return ReelsFeedResponse(recommended_ids=[str(rid) for rid in recommended_ids])
