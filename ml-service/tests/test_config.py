"""Tests for the Redis-password URL splicing helper in app.core.config."""

from app.core.config import _with_redis_password


def test_password_injected_into_bare_url():
    assert (
        _with_redis_password("redis://redis:6379/0", "s3cr3t")
        == "redis://:s3cr3t@redis:6379/0"
    )


def test_rediss_scheme_preserved():
    assert (
        _with_redis_password("rediss://redis:6379/1", "pw")
        == "rediss://:pw@redis:6379/1"
    )


def test_url_with_existing_credentials_is_left_untouched():
    url = "redis://:already@redis:6379/0"
    assert _with_redis_password(url, "new") == url


def test_empty_password_is_a_noop():
    url = "redis://redis:6379/0"
    assert _with_redis_password(url, "") == url


def test_non_redis_scheme_is_left_untouched():
    url = "http://redis:6379/0"
    assert _with_redis_password(url, "pw") == url
