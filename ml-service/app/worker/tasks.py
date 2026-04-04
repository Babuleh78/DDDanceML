from app.worker.celery_app import celery_app

@celery_app.task(bind=True, name="process_video")
def process_video_task(self, video_key: str, dance_id: str, enable_labeling: bool = True):
    from app.services.processing import process_video

    def on_progress(event: str, data: dict):
        self.update_state(
            state="PROGRESS",
            meta={"event": event, **data}
        )

    try:
        return process_video(video_key, dance_id, enable_labeling, on_progress)
    except Exception as exc:
        raise self.retry(exc=exc)
    
@celery_app.task(
    bind=True,
    name="process_video_url",
    max_retries=2,
    default_retry_delay=10,
    queue="video_processing",
)
def process_video_url_task(self, url: str, dance_id: str, enable_labeling: bool = True):
    from app.services.downloader import download_video_from_url
    from app.services.processing import process_video

    try:
        video_key = download_video_from_url(url)
        return process_video(video_key, dance_id, enable_labeling)
    except Exception as exc:
        raise self.retry(exc=exc)