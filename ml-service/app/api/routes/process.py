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


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    from app.worker.celery_app import celery_app
    task = celery_app.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "done", "result": task.result}
    if task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}
    return {"status": task.state.lower()}

@router.post("/process-url/")
async def process_url(req: ProcessUrlRequest):
    logger.info(f"Enqueue URL: {req.url}, dance_id={req.dance_id}")
    task = process_video_url_task.delay(
        url=req.url,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}