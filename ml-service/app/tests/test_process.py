"""
test_process.py
===============
Тесты для POST /process — основного эндпоинта ML сервиса.

Покрываемые сценарии:
  1. Видео не найдено в S3                → 422
  2. Успешная обработка (happy path)      → 200 + корректный response body
  3. result.json загружен в S3            → ровно 1 upload
  4. S3 upload упал после обработки       → 500
  5. Невалидный request body              → 422 (Pydantic, 4 случая)
  6. GET /health                          → 200
  7. Сервер занят (семафор)               → 503
  8. Ключ результата детерминирован       → results/abc123_result.json
  9. video_key без папки в пути           → results/myvideo_result.json
"""

import pytest
import threading
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Видео не найдено в S3
# ─────────────────────────────────────────────────────────────────────────────

def test_video_not_found_in_s3(client, mock_s3_download_missing):
    """
    Если S3 вернул ошибку при скачивании — эндпоинт должен вернуть 422.
    Golang backend получит чёткий сигнал что видео недоступно.
    """
    response = client.post("/process", json={
        "bucket": "dance-videos",
        "video_key": "videos/nonexistent.mp4",
    })

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert "NoSuchKey" in body["detail"] or len(body["detail"]) > 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Успешная обработка — полный флоу video → result.json
# ─────────────────────────────────────────────────────────────────────────────

def test_successful_processing(client, mock_pipeline):
    """
    Happy path: видео обработано, result.json загружен в S3.
    Response содержит result_key и корректные метрики.
    """
    response = client.post("/process", json={
        "bucket": "dance-videos",
        "video_key": "videos/dance_abc123.mp4",
    })

    assert response.status_code == 200, response.text
    body = response.json()

    # Структура ответа — один ключ вместо двух
    assert "result_key" in body
    assert "skeleton_key" not in body, "skeleton_key больше не должен быть в ответе"
    assert "segments_key" not in body, "segments_key больше не должен быть в ответе"

    assert "num_frames" in body
    assert "num_segments" in body
    assert "duration_sec" in body

    assert body["num_frames"] >= 0
    assert body["num_segments"] >= 0
    assert body["duration_sec"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. result.json загружен в S3 — ровно один upload
# ─────────────────────────────────────────────────────────────────────────────

def test_exactly_one_file_uploaded_to_s3(client, mock_pipeline):
    """
    После обработки в S3 должен оказаться ровно 1 файл (result.json).
    Раньше было 2 (skeleton + segments) — теперь один объединённый.
    """
    response = client.post("/process", json={
        "bucket": "dance-videos",
        "video_key": "videos/dance_abc123.mp4",
    })

    assert response.status_code == 200

    upload_mock = mock_pipeline["upload_mock"]
    assert upload_mock.call_count == 1, (
        f"Ожидали ровно 1 upload, было {upload_mock.call_count}"
    )

    uploaded_key = upload_mock.call_args.args[1]
    assert "result" in uploaded_key, \
        f"Загруженный ключ должен содержать 'result', получили: {uploaded_key}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ключ результата детерминирован
# ─────────────────────────────────────────────────────────────────────────────

def test_result_key_is_deterministic(client, mock_pipeline):
    """
    videos/dance_abc123.mp4 → results/dance_abc123_result.json

    Повторный запрос с тем же видео перезапишет результат без дублей.
    """
    response = client.post("/process", json={
        "bucket": "dance-videos",
        "video_key": "videos/dance_abc123.mp4",
    })

    assert response.status_code == 200
    assert response.json()["result_key"] == "results/dance_abc123_result.json"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ошибка при загрузке в S3 → 500
# ─────────────────────────────────────────────────────────────────────────────

def test_s3_upload_failure_returns_500(client, mock_pipeline):
    """
    Обработка прошла успешно, но S3 upload упал →
    сервер должен вернуть 500, не 200.
    Golang backend должен знать что результат не сохранён.
    """
    with patch(
        "app.core.s3.upload_file",
        side_effect=RuntimeError("S3 upload failed: connection timeout"),
    ):
        response = client.post("/process", json={
            "bucket": "dance-videos",
            "video_key": "videos/dance_abc123.mp4",
        })

    assert response.status_code == 500


# ─────────────────────────────────────────────────────────────────────────────
# 6. Невалидный request body → 422 от Pydantic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload,description", [
    ({}, "пустой body"),
    ({"bucket": "dance-videos"}, "отсутствует video_key"),
    ({"video_key": "videos/a.mp4"}, "отсутствует bucket"),
    ({"bucket": 123, "video_key": None}, "неверные типы полей"),
])
def test_invalid_request_body(client, payload, description):
    """
    Pydantic отклоняет невалидные запросы до бизнес-логики.
    """
    response = client.post("/process", json=payload)
    assert response.status_code == 422, \
        f"Ожидали 422 для случая '{description}', получили {response.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. GET /health → 200
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check(client):
    """
    Health check работает всегда — в том числе во время обработки видео.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Сервер занят → 503
# ─────────────────────────────────────────────────────────────────────────────

def test_server_busy_returns_503(client, mock_pipeline):
    """
    Если семафор заблокирован — второй запрос получает 503 немедленно.
    """
    started = threading.Event()
    release = threading.Event()

    def slow_process(_video_key):
        started.set()
        release.wait(timeout=5)
        return {
            "result_key": "results/x_result.json",
            "num_frames": 10,
            "num_segments": 1,
            "duration_sec": 1.0,
        }

    with patch("app.api.routes.process.process_video", side_effect=slow_process):
        first_result = {}

        def do_first():
            r = client.post("/process", json={
                "bucket": "dance-videos",
                "video_key": "videos/first.mp4",
            })
            first_result["status"] = r.status_code

        t = threading.Thread(target=do_first)
        t.start()
        started.wait(timeout=3)

        second_response = client.post("/process", json={
            "bucket": "dance-videos",
            "video_key": "videos/second.mp4",
        })

        assert second_response.status_code == 503, \
            f"Ожидали 503 при занятом сервере, получили {second_response.status_code}"
        assert "busy" in second_response.json()["detail"].lower()

        release.set()
        t.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# 9. video_key без папки в пути
# ─────────────────────────────────────────────────────────────────────────────

def test_video_key_without_folder(client, mock_pipeline):
    """
    video_key может быть "videos/a.mp4" или просто "a.mp4".
    Ключ результата формируется корректно в обоих случаях.
    """
    response = client.post("/process", json={
        "bucket": "dance-videos",
        "video_key": "myvideo.mp4",
    })

    assert response.status_code == 200
    assert response.json()["result_key"] == "results/myvideo_result.json"