import logging

from fastapi import APIRouter, Depends

from app.api.deps import verify_internal_token
from app.schemas.moderate import ModerateRequest, ModerateResponse
from app.services.moderation import moderate_video

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["moderation"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post("/moderate", response_model=ModerateResponse)
async def moderate_endpoint(request: ModerateRequest) -> ModerateResponse:
    result = await moderate_video(
        video_s3_url=request.video_s3_url,
        dance_id=request.dance_id,
        uploader_user_id=request.uploader_user_id,
        uploader_login=request.uploader_login or "",
    )
    return ModerateResponse(**result)
