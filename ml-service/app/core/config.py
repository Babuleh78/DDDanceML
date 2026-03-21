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

    # Skeleton extraction
    skeleton_model_complexity: int = 2
    skeleton_frame_skip: int = 2        # обрабатывать каждый N-й кадр (2 = 15fps из 30)
    skeleton_smoothing_alpha: float = 0.3  # 

    # Segmentation
    segmenter_min_seg_sec: float = 0.8
    segmenter_sensitivity: float = 0.05
    segmenter_smooth_window: int = 15


settings = Settings()