import logging
import json
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from app.schemas.process import ProcessRequest
from app.worker.tasks import process_video_task

from app.schemas.process import ProcessUrlRequest
from app.worker.tasks import process_video_url_task
from sse_starlette.sse import EventSourceResponse
from app.worker.celery_app import celery_app
from app.core import s3 as s3_client
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


@router.get("/segment_description/{dance_id}/{segment_idx}")
async def get_segment_description(dance_id: str, segment_idx: int):
    try:
        segments_key = f"results/{dance_id}/segments.json"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = str(Path(tmpdir) / "segments.json")
            s3_client.download_file(segments_key, local_path)
            
            with open(local_path, "r", encoding="utf-8") as f:
                segments_data = json.load(f)
            
            segments = segments_data.get("segments", [])
            if segment_idx < 0 or segment_idx >= len(segments):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Segment {segment_idx} not found. Available: 0-{len(segments)-1}"
                )
            
            description = _generate_segment_description(segment_idx)
            
            return {
                "dance_id": dance_id,
                "segment_idx": segment_idx,
                "description": description,
            }
    
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segments data not found for dance_id: {dance_id}"
        )
    except Exception as e:
        logger.error(f"Error getting segment description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving segment: {str(e)}"
        )


def _generate_segment_description(segment_idx: int) -> str:
    return f"скоро будет описание сегмента {segment_idx}"