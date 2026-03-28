from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    # S3
    s3_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str = "us-east-1"

    # Mixamo pipeline
    mixamo_model_path: str = "/app/models/mixamo_model.json"
    mixamo_min_visibility: float = 0.6
    mixamo_hips_move: bool = False
    mixamo_max_frames: int = 5000

    # Blender
    blender_executable: str = "blender" 
    blender_character_blend: str = "character.blend" 

    # Segmentation
    segmenter_min_seg_sec: float = 0.8
    segmenter_sensitivity: float = 0.05
    segmenter_smooth_window: int = 15


settings = Settings()