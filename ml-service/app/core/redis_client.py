import redis
from app.core.config import settings

# БД/2 — отдельно от Celery broker(/0) и backend(/1)
_client = redis.from_url(settings.redis_url.replace("/0", "/2"))

def get_redis():
    return _client