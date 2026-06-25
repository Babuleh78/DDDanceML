from typing import Literal, Optional

from pydantic import BaseModel


class ModerateRequest(BaseModel):
    video_s3_url: str
    dance_id: str
    uploader_user_id: str
    uploader_login: Optional[str] = None


class ModerateResponse(BaseModel):
    status: Literal["approved", "pending"]
    error_code: Optional[str] = None
    reason: Optional[Literal["animal", "nsfw", "no_person", "multiple_persons", "other"]] = None
