# app/schemas/compare.py
from pydantic import BaseModel, Field
from typing import Optional


class DanceCompareRequest(BaseModel):
    original_video_s3_path: str = Field(..., description="Путь в S3 к оригинальному видео (результат/{dance_id}/video.mp4)")
    user_video_s3_path: str = Field(..., description="Путь в S3 к видео пользователя")
    user_id: str = Field(..., description="ID пользователя")
    dance_id: str = Field(..., description="ID оригинального танца")

    class Config:
        json_schema_extra = {
            "example": {
                "original_video_s3_path": "results/abc-123/video.mp4",
                "user_video_s3_path": "uploads/user_attempt.mp4",
                "user_id": "user-456",
                "dance_id": "abc-123",
            }
        }


class DanceCompareResponse(BaseModel):
    task_id: str = Field(..., description="ID задачи Celery")
    dance_id: str
    user_id: str
    status: str = Field(default="queued", description="queued, processing, done, failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc-def-123",
                "dance_id": "abc-123",
                "user_id": "user-456",
                "status": "queued",
            }
        }


class ComparisonScoreResult(BaseModel):
    dance_id: str
    user_id: str
    comparison_score: float = Field(..., ge=0, le=100, description="Score 0-100")
    dtw_distance: float = Field(..., description="Нормализованное расстояние DTW")
    original_video_s3: str = Field(..., description="Путь к оригинальному видео в S3")
    user_video_s3: str = Field(..., description="Путь к видео пользователя в S3")
    user_glb_s3: str = Field(..., description="Путь к 3D моделе пользователя в S3")
    processed_at: str = Field(..., description="ISO timestamp обработки")
