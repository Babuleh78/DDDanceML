import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes.health import router as health_router
from app.api.routes.moderate import router as moderate_router
from app.api.routes.process import router as process_router
from app.api.routes.recommend import router as recommend_router
from app.api.routes.reels import router as reels_router
from app.core.exceptions import DanceNotFoundError, S3UploadError
from app.services.recommender import DanceRecommender
from app.services.reels_recommender import ReelsRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Dance ML Service",
    description="Video to skeleton + movement segments",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(process_router)
app.include_router(moderate_router)
app.include_router(recommend_router)
app.include_router(reels_router)

Instrumentator().instrument(app).expose(app)


@app.exception_handler(DanceNotFoundError)
async def _dance_not_found(request: Request, exc: DanceNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(S3UploadError)
async def _s3_upload_error(request: Request, exc: S3UploadError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.on_event("startup")
async def _startup() -> None:
    from app.telegram_bot.bot import start_bot
    asyncio.create_task(start_bot())

    app.state.recommender = None
    app.state.reels_recommender = None

    async def _load_recommender() -> None:
        try:
            recommender = DanceRecommender()
            await asyncio.to_thread(recommender.load)
            app.state.recommender = recommender
            app.state.reels_recommender = ReelsRecommender(recommender)
            logging.getLogger(__name__).info("DanceRecommender loaded")
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to load DanceRecommender — /ml/recommend will be unavailable"
            )

    asyncio.create_task(_load_recommender())


@app.get("/")
async def root():
    return {
        "service": "DDDance ML Service",
        "version": "0.1.0",
        "status": "running",
    }
