import redis
from app.core.config import settings

_client = redis.from_url(settings.redis_url.replace("/0", "/2"))

def get_redis():
    return _client