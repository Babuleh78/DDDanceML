from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _with_redis_password(url: str, password: str) -> str:
    """Inject a password into a redis:// URL that has no credentials yet.

    Redis runs with --requirepass, but the connection URLs (env defaults or
    .env) carry no password, which yields "Authentication required". We splice
    the password in here so celery/redis clients connect authenticated, without
    having to duplicate it into every REDIS_* URL in the environment.
    """
    if not password or "@" in url:
        return url
    for scheme in ("rediss://", "redis://"):
        if url.startswith(scheme):
            return f"{scheme}:{password}@{url[len(scheme):]}"
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"), 
        env_file_encoding="utf-8",
        case_sensitive=False, 
        extra="ignore" 
    )
    
    s3_endpoint_url: str = Field(..., env="S3_ENDPOINT_URL")
    s3_access_key: str = Field(..., env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(..., env="S3_SECRET_KEY")
    s3_bucket: str = Field(..., env="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    
    mixamo_model_path: str = Field(default="/app/models/mixamo_model.json", env="MIXAMO_MODEL_PATH")
    mixamo_min_visibility: float = Field(default=0.6, env="MIXAMO_MIN_VISIBILITY")
    mixamo_hips_move: bool = Field(default=False, env="MIXAMO_HIPS_MOVE")
    mixamo_max_frames: Optional[int] = Field(default=5000, env="MIXAMO_MAX_FRAMES")
    
    blender_executable: str = Field(default="blender", env="BLENDER_EXECUTABLE")
    blender_character_blend: str = Field(default="character.blend", env="BLENDER_CHARACTER_BLEND")
    
    
    segmenter_min_seg_sec: float = Field(default=0.5, env="SEGMENTER_MIN_SEG_SEC")
    segmenter_sensitivity: float = Field(default=0.4, env="SEGMENTER_SENSITIVITY")
    segmenter_smooth_window: int = Field(default=11, env="SEGMENTER_SMOOTH_WINDOW")
    
    labeling_cache_ttl: int = Field(default=3600, env="LABELING_CACHE_TTL")
    labeling_cache_size: int = Field(default=1000, env="LABELING_CACHE_SIZE")
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    celery_broker_url: str = Field(default="redis://redis:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://redis:6379/1", env="CELERY_RESULT_BACKEND")
    redis_cache_url: str = Field(default="redis://redis:6379/2", env="REDIS_CACHE_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    labeling_enabled: bool = Field(default=False, env="LABELING_ENABLED")
    
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")

    moderate_power: int = Field(default=5, env="MODERATE_POWER")
    moderate_yolo_model: str = Field(default="yolov8n.pt", env="MODERATE_YOLO_MODEL")
    moderate_multi_person_check: bool = Field(default=True, env="MODERATE_MULTI_PERSON_CHECK")
    moderate_person_min_rel_area: float = Field(default=0.5, env="MODERATE_PERSON_MIN_REL_AREA")
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_admin_chat_id: Optional[int] = Field(default=None, env="TELEGRAM_ADMIN_CHAT_ID")
    go_backend_url: str = Field(default="http://main:5458", env="GO_BACKEND_URL")

    ytdlp_proxy: Optional[str] = None
    proxy_instagram: Optional[str] = None
    proxy_tiktok: Optional[str] = None
    proxy_vk: Optional[str] = None
    telegram_proxy: Optional[str] = None

    admin_token: Optional[str] = Field(default=None, env="ADMIN_TOKEN")

    ml_internal_token: Optional[str] = Field(default=None, env="ML_INTERNAL_TOKEN")

    @model_validator(mode="after")
    def _inject_redis_password(self) -> "Settings":
        if self.redis_password:
            self.redis_url = _with_redis_password(self.redis_url, self.redis_password)
            self.celery_broker_url = _with_redis_password(self.celery_broker_url, self.redis_password)
            self.celery_result_backend = _with_redis_password(self.celery_result_backend, self.redis_password)
            self.redis_cache_url = _with_redis_password(self.redis_cache_url, self.redis_password)
        return self


settings = Settings()
