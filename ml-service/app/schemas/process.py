import uuid
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class ProcessRequest(BaseModel):
    video_key: str
    dance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    enable_labeling: bool = True
    uploader_user_id: str = ""
    
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
    uploader_user_id: str = ""

    @field_validator("url")
    def validate_url(cls, v):
        allowed = [
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
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme")
        netloc = parsed.netloc.lower()
        if not any(netloc == d or netloc.endswith("." + d) for d in allowed):
            raise ValueError(f"URL domain not in allowlist: {allowed}")
        return v