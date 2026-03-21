# app/schemas/process.py
from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    """Запрос на обработку видео"""
    bucket: str = Field(..., description="Название S3 bucket")
    video_key: str = Field(..., description="Путь к видео в S3")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bucket": "dance-videos",
                "video_key": "videos/dance.mp4"
            }
        }


class ProcessResponse(BaseModel):
    """Ответ на успешную обработку видео"""
    result_key: str = Field(..., description="Путь к результату в S3")
    num_frames: int = Field(..., description="Количество кадров в видео")
    num_segments: int = Field(..., description="Количество найденных сегментов")
    duration_sec: float = Field(..., description="Длительность видео в секундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result_key": "results/dance_result.json",
                "num_frames": 450,
                "num_segments": 5,
                "duration_sec": 15.0
            }
        }