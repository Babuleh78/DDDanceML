# app/worker/celery_app.py
import os
from celery import Celery

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "dddance",
    broker=redis_url,
    backend=redis_url.replace("/0", "/1"),
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
)