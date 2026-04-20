# app/schemas/compare.py
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class DanceCompareRequest(BaseModel):
    """
    video_key   — S3-ключ видео пользователя (уже загружен)
    dance_id    — ID оригинального танца (откуда берём segments.json)
    segment_idx — индекс сегмента для сравнения; -1 = всё видео целиком
    """
    video_key: str
    dance_id: str
    segment_idx: int = Field(default=-1, ge=-1)

    class Config:
        json_schema_extra = {
            "example": {
                "video_key": "uploads/user_attempt.mp4",
                "dance_id": "abc-123",
                "segment_idx": -1,
            }
        }


class VelocityMetrics(BaseModel):
    mean: float
    max: float
    std: float


class ROMMetrics(BaseModel):
    max_distance: float
    mean_distance: float


class JointAngleMetrics(BaseModel):
    mean_deg: float
    range_deg: float


class SegmentNumericMetrics(BaseModel):
    velocity: VelocityMetrics
    smoothness: float
    rom: ROMMetrics
    tempo_bpm: float
    symmetry_ratio: float     
    joint_angles: dict[str, JointAngleMetrics]  

class SegmentCompareDetail(BaseModel):
    segment_idx: int
    dtw_scores: dict[str, float]
    velocity_diff: float
    smoothness_diff: float
    rom_diff: float
    tempo_diff: float
    symmetry_diff: float
    joint_angles_diff: dict[str, float]
    segment_score: float

class DanceCompareResponse(BaseModel):
    dance_id: str
    segment_idx: int       
    overall_score: float   
    segments: list[SegmentCompareDetail]
    weakest_metrics: list[str]