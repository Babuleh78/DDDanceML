import functools
import time
from typing import Callable

from prometheus_client import Counter, Histogram

celery_tasks_total = Counter(
    "ml_celery_tasks_total",
    "Total Celery tasks executed",
    ["task_name", "status"],
)

celery_task_duration = Histogram(
    "ml_celery_task_duration_seconds",
    "Celery task execution duration",
    ["task_name"],
)

ml_inference_duration = Histogram(
    "ml_inference_duration_seconds",
    "ML model inference duration",
    ["model_name"],
)

s3_operations_total = Counter(
    "ml_s3_operations_total",
    "Total S3 operations",
    ["operation", "status"],
)


def track_task_metrics(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        task_name = getattr(self, "name", fn.__name__)
        start = time.time()
        try:
            result = fn(self, *args, **kwargs)
            celery_tasks_total.labels(task_name=task_name, status="success").inc()
            return result
        except Exception:
            celery_tasks_total.labels(task_name=task_name, status="failure").inc()
            raise
        finally:
            celery_task_duration.labels(task_name=task_name).observe(time.time() - start)

    return wrapper
