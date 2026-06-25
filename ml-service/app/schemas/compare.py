from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DanceCompareRequest(BaseModel):
    original_video_s3_path: str = Field(..., description="Путь в S3 к оригинальному видео (результат/{dance_id}/video.mp4)")
    user_video_s3_path: str = Field(..., description="Путь в S3 к видео пользователя")
    user_id: str = Field(..., description="ID пользователя")
    dance_id: str = Field(..., description="ID оригинального танца")
    attempt_id: Optional[str] = Field(None, description="UUID попытки; задаёт префикс S3 для артефактов")

    class Config:
        json_schema_extra = {
            "example": {
                "original_video_s3_path": "results/abc-123/video.mp4",
                "user_video_s3_path": "uploads/user_attempt.mp4",
                "user_id": "user-456",
                "dance_id": "abc-123",
                "attempt_id": "f5e6d7c8-...-...",
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
    attempt_score: float = Field(..., ge=0, le=100, description="Score 0-100")
    dtw_distance: float = Field(..., description="Нормализованное расстояние DTW")
    original_video_s3: str = Field(..., description="Путь к оригинальному видео в S3")
    user_video_s3: str = Field(..., description="Путь к видео пользователя в S3")
    user_glb_s3: str = Field(..., description="Путь к 3D модели пользователя в S3")
    processed_at: str = Field(..., description="ISO timestamp обработки")


class CompareTipSegmentInput(BaseModel):
    segment_id: int
    label: Optional[str] = None
    score: float = Field(..., ge=0, le=100)
    timing: float = Field(..., ge=0, le=100)
    amplitude: float = Field(..., ge=0, le=100)
    pose_accuracy: float = Field(..., ge=0, le=100)
    feedback: Optional[str] = None


class CompareTipsRequest(BaseModel):
    attempt_score: float = Field(..., ge=0, le=100, description="Глобальный score попытки 0-100")
    segments: List[CompareTipSegmentInput] = Field(default_factory=list)


class CompareTip(BaseModel):
    type: Literal["warn", "info"]
    text: str


class CompareTipsResponse(BaseModel):
    tips: List[CompareTip] = Field(default_factory=list)
