"""Сохранение и загрузка результатов обработки."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.s3 import s3_client  # ваш существующий клиент
from app.core.config import settings

logger = logging.getLogger(__name__)


def save_segments_local(
    segments: list,
    metadata: Dict[str, Any],
    output_path: Path,
    video_id: str,
    fps: float,
    total_frames: int,
) -> Path:
    """Сохраняет результаты в локальный JSON-файл."""
    result = {
        "version": "1.1",
        "video_id": video_id,
        "fps": fps,
        "total_frames": total_frames,
        "generated_at": datetime.utcnow().isoformat(),
        "segments": segments,
        "metadata": metadata,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved segments to {output_path} ({len(segments)} segments)")
    return output_path


async def save_segments_to_s3(
    segments: list,
    metadata: Dict[str, Any],
    video_id: str,
    fps: float,
    total_frames: int,
    bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
) -> str:
    """
    Загружает результаты в S3.
    
    Returns:
        S3 key сохранённого файла
    """
    bucket = bucket or settings.S3_RESULTS_BUCKET
    s3_key = s3_key or f"segments/{video_id}.json"
    
    result = {
        "version": "1.1",
        "video_id": video_id,
        "fps": fps,
        "total_frames": total_frames,
        "generated_at": datetime.utcnow().isoformat(),
        "segments": segments,
        "metadata": metadata,
    }
    
    # Сериализуем в JSON
    json_content = json.dumps(result, ensure_ascii=False).encode("utf-8")
    
    # Загружаем в S3
    await s3_client.upload_fileobj(
        fileobj=json_content,
        bucket=bucket,
        key=s3_key,
        extra_args={
            "ContentType": "application/json",
            "Metadata": {
                "video_id": video_id,
                "n_segments": str(len(segments)),
                "labeling_strategy": metadata.get("labeling", {}).get("strategy", "none"),
            }
        }
    )
    
    logger.info(f"Uploaded segments to s3://{bucket}/{s3_key}")
    return s3_key


def load_segments_from_file(file_path: Path) -> Dict[str, Any]:
    """Загружает результаты из локального JSON для тестов."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)