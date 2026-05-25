from app.worker.celery_app import celery_app

_PROCESS_EVENT_TO_STAGE = {
    "segments_ready":       ("segmentation",     55, "Анализ сегментов"),
    "full_animation_ready": ("animation_render",  75, "Рендер 3D-анимации"),
    "segment_ready":        ("animation_render",  82, "Рендер сегментов"),
}


@celery_app.task(bind=True, name="process_video")
def process_video_task(
    self,
    video_key: str,
    dance_id: str,
    enable_labeling: bool = True,
    uploader_user_id: str = "",
):
    from app.services.processing import process_video

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
            return result
        celery_app.send_task("extract_keyframes", args=[dance_id], queue="background")
        return result
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="process_video_url",
    max_retries=2,
    default_retry_delay=10,
    queue="video_processing",
)
def process_video_url_task(self, url: str, dance_id: str, enable_labeling: bool = True, uploader_user_id: str = ""):
    from app.services.downloader import download_video_from_url
    from app.services.processing import process_video

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
        celery_app.send_task("extract_keyframes", args=[dance_id], queue="background")
        return result
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="extract_keyframes",
    max_retries=1,
    default_retry_delay=30,
    queue="background",
)
def extract_keyframes_task(self, dance_id: str):
    from app.services.keyframes import extract_and_save_keyframes

    try:
        return extract_and_save_keyframes(dance_id)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="compare_dance",
    max_retries=2,
    default_retry_delay=10,
    queue="video_processing",
    soft_time_limit=300,
    time_limit=330,
)
def compare_dance_task(
    self,
    original_video_s3_path: str,
    user_video_s3_path: str,
    user_id: str,
    dance_id: str,
    attempt_id: str = None,
):
    from app.services.compare import compare_dance

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
        return compare_dance(
            original_video_s3_path=original_video_s3_path,
            user_video_s3_path=user_video_s3_path,
            user_id=user_id,
            dance_id=dance_id,
            attempt_id=attempt_id,
            on_progress=on_progress,
        )
    except Exception as exc:
        raise self.retry(exc=exc)
