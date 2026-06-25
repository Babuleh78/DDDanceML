import logging
import os
import time

import requests
from celery.exceptions import SoftTimeLimitExceeded

from app.core.exceptions import S3UploadError
from app.core.metrics import track_task_metrics
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

def _patch_dance_duration(dance_id: str, duration_sec: int) -> None:
    go_url = os.environ.get("GO_SERVICE_URL", "").rstrip("/")
    if not go_url:
        return
    token = os.environ.get("ML_INTERNAL_TOKEN", "")
    try:
        resp = requests.patch(
            f"{go_url}/api/dances/{dance_id}/duration",
            json={"duration_sec": duration_sec},
            headers={"X-Internal-Token": token},
            timeout=5,
        )
        if resp.status_code not in (200, 204):
            logger.warning("patch duration returned %d for dance %s", resp.status_code, dance_id)
    except Exception as exc:
        logger.warning("failed to patch duration for dance %s: %s", dance_id, exc)


_PROCESS_EVENT_TO_STAGE = {
    "segments_ready":       ("segmentation",     55, "Анализ сегментов"),
    "full_animation_ready": ("animation_render",  75, "Рендер 3D-анимации"),
    "segment_ready":        ("animation_render",  82, "Рендер сегментов"),
}


def _log(task_id: str, stage: str, duration_ms: float = 0.0, **kwargs) -> None:
    entry = {"task_id": task_id, "stage": stage, "duration_ms": round(duration_ms, 1)}
    entry.update(kwargs)
    logger.info(" ".join(f"{k}={v}" for k, v in entry.items()))


@celery_app.task(
    bind=True,
    name="process_video",
    queue="video_processing",
    soft_time_limit=600,
    time_limit=660,
    autoretry_for=(S3UploadError,),
    retry_backoff=True,
    max_retries=3,
)
@track_task_metrics
def process_video_task(
    self,
    video_key: str,
    dance_id: str,
    enable_labeling: bool = True,
    uploader_user_id: str = "",
):
    from app.services.processing import process_video

    t0 = time.time()
    task_id = self.request.id or ""
    _log(task_id, "start", dance_id=dance_id)

    self.update_state(
        state="PROGRESS",
        meta={"stage": "pose_extraction", "progress": 15, "stage_label": "Извлечение движений"},
    )

    def on_progress(event: str, data: dict):
        stage, progress, label = _PROCESS_EVENT_TO_STAGE.get(
            event, ("processing", 50, "Обработка")
        )
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "progress": progress, "stage_label": label, "event": event, **data},
        )

    try:
        result = process_video(video_key, dance_id, enable_labeling, on_progress, uploader_user_id)
        if result.get("status") == "moderation_pending":
            _log(task_id, "moderation_pending", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id)
            return result
        _patch_dance_duration(dance_id, int(round(result.get("duration_sec", 0))))
        celery_app.send_task("extract_keyframes", args=[dance_id], queue="background")
        _log(task_id, "done", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id)
        return result
    except S3UploadError:
        raise
    except SoftTimeLimitExceeded:
        logger.error("Task %s soft time limit exceeded", self.request.id)
        raise
    except Exception as exc:
        _log(task_id, "error", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id, error=str(exc))
        raise


@celery_app.task(
    bind=True,
    name="process_video_url",
    queue="video_processing",
    soft_time_limit=600,
    time_limit=660,
    autoretry_for=(S3UploadError,),
    retry_backoff=True,
    max_retries=3,
)
@track_task_metrics
def process_video_url_task(self, url: str, dance_id: str, enable_labeling: bool = True, uploader_user_id: str = ""):
    from app.services.downloader import download_video_from_url
    from app.services.processing import process_video

    t0 = time.time()
    task_id = self.request.id or ""
    _log(task_id, "start", dance_id=dance_id)

    self.update_state(
        state="PROGRESS",
        meta={"stage": "codec_check", "progress": 5, "stage_label": "Загрузка видео"},
    )

    def on_progress(event: str, data: dict):
        stage, progress, label = _PROCESS_EVENT_TO_STAGE.get(
            event, ("processing", 50, "Обработка")
        )
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "progress": progress, "stage_label": label, "event": event, **data},
        )

    try:
        video_key = download_video_from_url(url)
        self.update_state(
            state="PROGRESS",
            meta={"stage": "pose_extraction", "progress": 15, "stage_label": "Извлечение движений"},
        )
        result = process_video(video_key, dance_id, enable_labeling, on_progress, uploader_user_id)
        _patch_dance_duration(dance_id, int(round(result.get("duration_sec", 0))))
        celery_app.send_task("extract_keyframes", args=[dance_id], queue="background")
        _log(task_id, "done", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id)
        return result
    except S3UploadError:
        raise
    except SoftTimeLimitExceeded:
        logger.error("Task %s soft time limit exceeded", self.request.id)
        raise
    except Exception as exc:
        _log(task_id, "error", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id, error=str(exc))
        raise


@celery_app.task(
    bind=True,
    name="extract_keyframes",
    queue="background",
    soft_time_limit=120,
    time_limit=150,
    autoretry_for=(S3UploadError,),
    retry_backoff=True,
    max_retries=3,
)
@track_task_metrics
def extract_keyframes_task(self, dance_id: str):
    from app.services.keyframes import extract_and_save_keyframes

    t0 = time.time()
    task_id = self.request.id or ""
    _log(task_id, "start", dance_id=dance_id)

    try:
        result = extract_and_save_keyframes(dance_id)
        _log(task_id, "done", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id)
        return result
    except S3UploadError:
        raise
    except SoftTimeLimitExceeded:
        logger.error("Task %s soft time limit exceeded", self.request.id)
        raise
    except Exception as exc:
        _log(task_id, "error", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id, error=str(exc))
        raise


@celery_app.task(
    bind=True,
    name="compare_dance",
    queue="video_processing",
    soft_time_limit=300,
    time_limit=330,
    autoretry_for=(S3UploadError,),
    retry_backoff=True,
    max_retries=3,
)
@track_task_metrics
def compare_dance_task(
    self,
    original_video_s3_path: str,
    user_video_s3_path: str,
    user_id: str,
    dance_id: str,
    attempt_id: str = None,
):
    from app.services.compare import compare_dance

    t0 = time.time()
    task_id = self.request.id or ""
    _log(task_id, "start", dance_id=dance_id, attempt_id=attempt_id or "", user_id=user_id)

    self.update_state(
        state="PROGRESS",
        meta={"stage": "pose_extraction", "progress": 10, "stage_label": "Извлечение движений"},
    )

    def on_progress(stage: str, progress: int, label: str):
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "progress": progress, "stage_label": label},
        )

    try:
        result = compare_dance(
            original_video_s3_path=original_video_s3_path,
            user_video_s3_path=user_video_s3_path,
            user_id=user_id,
            dance_id=dance_id,
            attempt_id=attempt_id,
            on_progress=on_progress,
        )
        _log(task_id, "done", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id, attempt_id=attempt_id or "")
        return result
    except S3UploadError:
        raise
    except SoftTimeLimitExceeded:
        logger.error("Task %s soft time limit exceeded", self.request.id)
        raise
    except Exception as exc:
        _log(task_id, "error", duration_ms=(time.time() - t0) * 1000, dance_id=dance_id, attempt_id=attempt_id or "", error=str(exc))
        raise
