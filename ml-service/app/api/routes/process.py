import asyncio
import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.schemas.process import ProcessRequest, ProcessResponse
from app.services.processing import process_video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])
_processing_semaphore = asyncio.Semaphore(1)


@router.post("/", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    logger.info(f"Processing request: bucket={req.bucket}, key={req.video_key}")
    
    async with _processing_semaphore:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                process_video, 
                req.video_key
            )
            
            return ProcessResponse(
                result_key=result["animation_key"],  
                segments_key=result["segments_key"], 
                num_frames=result["num_frames"],
                num_segments=result["num_segments"],
                duration_sec=result["duration_sec"]
            )
            
        except FileNotFoundError as e:
            logger.error(f"Video not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"NoSuchKey: {req.video_key}"
            )
        except RuntimeError as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "download failed" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=error_msg
                )
            elif "upload failed" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal processing error"
                )
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal processing error"
            )