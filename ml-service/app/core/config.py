# app/core/config.py
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # === Настройка загрузки из .env ===
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),  # Абсолютный путь к .env
        env_file_encoding="utf-8",
        case_sensitive=False,  # Игнорировать регистр: S3_BUCKET == s3_bucket
        extra="ignore"  # Игнорировать неизвестные переменные в .env
    )
    
    # === S3 (обязательные поля с явным маппингом) ===
    s3_endpoint_url: str = Field(..., env="S3_ENDPOINT_URL")
    s3_access_key: str = Field(..., env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(..., env="S3_SECRET_KEY")
    s3_bucket: str = Field(..., env="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    
    # === Mixamo pipeline ===
    mixamo_model_path: str = Field(default="/app/models/mixamo_model.json", env="MIXAMO_MODEL_PATH")
    mixamo_min_visibility: float = Field(default=0.6, env="MIXAMO_MIN_VISIBILITY")
    mixamo_hips_move: bool = Field(default=False, env="MIXAMO_HIPS_MOVE")
    mixamo_max_frames: Optional[int] = Field(default=5000, env="MIXAMO_MAX_FRAMES")
    
    # === Blender ===
    blender_executable: str = Field(default="blender", env="BLENDER_EXECUTABLE")
    blender_character_blend: str = Field(default="character.blend", env="BLENDER_CHARACTER_BLEND")
    
    # === Segmentation ===
    segmenter_min_seg_sec: float = Field(default=0.8, env="SEGMENTER_MIN_SEG_SEC")
    segmenter_sensitivity: float = Field(default=0.05, env="SEGMENTER_SENSITIVITY")
    segmenter_smooth_window: int = Field(default=15, env="SEGMENTER_SMOOTH_WINDOW")
    
    # === LLM Labeling ===
    labeling_enabled: bool = Field(default=True, env="LABELING_ENABLED")
    labeling_backend: Literal["openrouter", "ollama", "mock"] = Field(
        default="ollama",  # ← по умолчанию локально!
        env="LABELING_BACKEND"
    )

    # OpenRouter (если понадобится)
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    labeling_model: str = Field(
        default="meta-llama/llama-3.2-3b-instruct:free",
        env="LABELING_MODEL"
    )

    # Ollama (локальный)
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    labeling_model_ollama: Optional[str] = Field(
        default="llama3.2:3b",  # ← ваша локальная модель
        env="LABELING_MODEL_OLLAMA"
    )

    # Кэширование
    labeling_cache_ttl: int = Field(default=3600, env="LABELING_CACHE_TTL")
    labeling_cache_size: int = Field(default=1000, env="LABELING_CACHE_SIZE")
    redis_url: str = "redis://redis:6379/0"
    
    # === Debug ===
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")


# Глобальный инстанс
settings = Settings()