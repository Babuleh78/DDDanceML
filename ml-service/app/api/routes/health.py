import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.redis_client import get_redis
from app.core.s3 import get_s3_client

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    errors: list[str] = []

    try:
        get_redis().ping()
    except Exception as e:
        logger.warning("Redis not ready: %s", e)
        errors.append("redis")

    try:
        get_s3_client().head_bucket(Bucket=settings.s3_bucket)
    except Exception as e:
        logger.warning("S3 not ready: %s", e)
        errors.append("s3")

    if errors:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "failing": errors},
        )
    return {"status": "ready"}
