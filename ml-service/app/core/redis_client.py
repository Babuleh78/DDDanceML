import redis
from app.core.config import settings

_client = redis.from_url(settings.redis_cache_url)

def get_redis():
    return _client