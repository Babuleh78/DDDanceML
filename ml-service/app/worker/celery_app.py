from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "dddance",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_expires=3600,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="video_processing",
    task_default_exchange="video_processing",
    task_default_routing_key="video_processing",
    task_queues={
        "video_processing": {},
        "background": {},
    },
)