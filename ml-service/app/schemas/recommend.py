from pydantic import BaseModel, Field


class RecommendDanceItem(BaseModel):
    id: str
    title: str
    description: str = ''
    avg_score: float = 0.0
    view_count: int = 0


class RecommendRequest(BaseModel):
    query: str
    dances: list[RecommendDanceItem]
    limit: int = Field(default=5, ge=1, le=50)


class RecommendResponse(BaseModel):
    recommended_ids: list[str]
    reasoning: str


class SimilarRequest(BaseModel):
    dance_id: str
    dances: list[RecommendDanceItem]
    limit: int = Field(default=4, ge=1, le=50)


class SimilarResponse(BaseModel):
    recommended_ids: list[str]
