# app/schemas/process.py
from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    bucket: str = Field(..., description="Название S3 bucket")
    video_key: str = Field(..., description="Путь к видео в S3")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bucket": "dddance",
                "video_key": "videos/vidos.mp4"
            }
        }


class ProcessResponse(BaseModel):
    result_key: str   
    segments_key: Optional[str] 
    num_frames: int
    num_segments: int
    duration_sec: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "result_key": "results/vidos_animation.glb",
                "segments_key": "results/vidos_segments.json",
                "num_frames": 450,
                "num_segments": 5,
                "duration_sec": 15.0
            }
        }