import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.process import ProcessRequest
from app.worker.tasks import process_video_task

from app.schemas.process import ProcessUrlRequest
from app.worker.tasks import process_video_url_task
from sse_starlette.sse import EventSourceResponse
from app.worker.celery_app import celery_app
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["processing"])


@router.post("/process")
async def process(req: ProcessRequest):
    task = process_video_task.delay(
        video_key=req.video_key,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}


@router.get("/stream/{task_id}")
async def stream_status(task_id: str):
    """
    SSE endpoint — Go backend подписывается и получает события в реальном времени.
    
    События:
        segments_ready  → сегменты готовы, можно показать список движений
        segment_ready   → один GLB готов
        done            → всё готово, финальный результат
        error           → ошибка
    """
    async def event_generator():
        while True:
            task = celery_app.AsyncResult(task_id)

            if task.state == "PROGRESS":
                yield {
                    "event": task.info.get("event", "progress"),
                    "data": json.dumps(task.info),
                }

            elif task.state == "SUCCESS":
                yield {
                    "event": "done",
                    "data": json.dumps(task.result),
                }
                break

            elif task.state == "FAILURE":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(task.info)}),
                }
                break

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

@router.post("/process-url/")
async def process_url(req: ProcessUrlRequest):
    logger.info(f"Enqueue URL: {req.url}")
    task = process_video_url_task.delay(
        url=req.url,
        enable_labeling=req.enable_labeling,
    )
    return {"task_id": task.id, "status": "queued"}