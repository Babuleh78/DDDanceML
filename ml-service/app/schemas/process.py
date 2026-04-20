# app/schemas/process.py
from pydantic import BaseModel, Field, field_validator, validator
from typing import Optional
import uuid
from pydantic import BaseModel, Field

class ProcessRequest(BaseModel):
    video_key: str
    dance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    enable_labeling: bool = True
    
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

class ProcessUrlRequest(BaseModel):
    url: str
    dance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    enable_labeling: bool = True

    @field_validator("url")
    def validate_url(cls, v):
        allowed =[
            "tiktok.com",
            "vm.tiktok.com",
            "instagram.com",
            "instagr.am",
            "youtube.com",
            "youtu.be",
            "vk.com",
            "vk.video",
            "vkvideo.ru",
        ]
        if not any(domain in v for domain in allowed):
            raise ValueError(f"Unsupported platform. Allowed: {allowed}")
        return v