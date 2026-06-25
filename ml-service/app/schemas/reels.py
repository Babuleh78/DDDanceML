from pydantic import BaseModel, Field


class UserHistoryItem(BaseModel):
    dance_id: str
    score: float = 0.0
    viewed_at: str = ''
    liked: bool = False


class CandidateDance(BaseModel):
    id: str
    title: str = ''
    description: str = ''
    avg_score: float = 0.0
    view_count: int = 0
    uploader_id: str = ''


class BehaviorLogEntry(BaseModel):
    dance_id: str
    action: str
    timestamp: int = 0


class ReelsFeedRequest(BaseModel):
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    candidate_dances: list[CandidateDance] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    exclude_ids: list[str] = Field(default_factory=list)
    behavior_log: list[BehaviorLogEntry] = Field(default_factory=list)
    friend_uploader_ids: list[str] = Field(default_factory=list)


class ReelsFeedResponse(BaseModel):
    recommended_ids: list[str]
