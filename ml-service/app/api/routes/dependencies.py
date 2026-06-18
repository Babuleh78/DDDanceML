from fastapi import Request

from app.core.redis_client import get_redis as _get_redis
from app.core.s3 import get_s3_client as _get_s3_client


def get_recommender(request: Request):
    return request.app.state.recommender


def get_reels_recommender(request: Request):
    return request.app.state.reels_recommender


def get_s3():
    return _get_s3_client()


def get_redis():
    return _get_redis()
