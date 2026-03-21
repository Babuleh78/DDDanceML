"""
conftest.py
===========
Общие фикстуры для всех тестов.

Стратегия мокирования:
  - S3 download/upload  → unittest.mock.patch
  - SkeletonExtractor   → unittest.mock.patch (не запускаем MediaPipe)
  - cv2.VideoCapture    → unittest.mock.patch (не нужен реальный видеофайл)
  - _build_result       → пишет фиктивный result.json на диск

Реальный I/O (tempfile, json.dump, Path.stat) работает — это важно,
чтобы тесты проверяли что файл реально создаётся перед upload.
"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# ── Фиктивные данные ──────────────────────────────────────────────────────────

def _make_fake_joint() -> dict:
    return {"x": 0.0, "y": 0.5, "z": 0.1, "vis": 1.0}


def _make_fake_raw_frame(frame_idx: int, fps: float = 30.0) -> dict:
    """dict-кадр в формате skeleton.json (для skeleton_to_segments)."""
    return {
        "frame_idx": frame_idx,
        "timestamp_ms": round(frame_idx / fps * 1000, 2),
        "joints": [_make_fake_joint() for _ in range(33)],
    }


def make_fake_result(num_frames: int = 90, fps: float = 30.0) -> dict:
    """
    Фиктивный result.json — объединённый формат skeleton + segments.
    Используется в mock_pipeline для записи на диск вместо реальной обработки.
    """
    raw_frames = [_make_fake_raw_frame(i, fps) for i in range(num_frames)]
    duration_sec = round((num_frames - 1) / fps, 3)

    return {
        "version": "1.0",
        "meta": {
            "fps": fps,
            "num_frames": num_frames,
            "duration_sec": duration_sec,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        "joint_names": [f"joint_{i}" for i in range(33)],
        "connections": [],
        "segments": [
            {
                "index": 0,
                "label": "segment_1",
                "start_frame": 0,
                "end_frame": 44,
                "start_ms": 0.0,
                "end_ms": 1466.67,
                "duration_ms": 1466.67,
                "duration_sec": 1.467,
                "num_frames": 44,
            },
            {
                "index": 1,
                "label": "segment_2",
                "start_frame": 44,
                "end_frame": 89,
                "start_ms": 1466.67,
                "end_ms": 2966.67,
                "duration_ms": 1500.0,
                "duration_sec": 1.5,
                "num_frames": 45,
            },
        ],
        "frames": raw_frames,
    }


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """
    TestClient с замоканными env переменными.
    pydantic-settings читает env при импорте — патчим до импорта app.
    """
    env_vars = {
        "S3_ENDPOINT_URL": "http://localhost:9000",
        "S3_ACCESS_KEY": "minioadmin",
        "S3_SECRET_KEY": "minioadmin",
        "S3_BUCKET": "test-bucket",
        "S3_REGION": "us-east-1",
    }
    with patch.dict("os.environ", env_vars):
        from fastapi.testclient import TestClient
        from app.main import app
        yield TestClient(app)


# ── S3 mocks ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_s3_download_ok():
    """S3 download → создаёт пустой файл-заглушку (видео "существует")."""
    def fake_download(s3_key: str, local_path: str) -> None:
        open(local_path, "wb").close()

    with patch("app.core.s3.download_file", side_effect=fake_download):
        yield


@pytest.fixture()
def mock_s3_download_missing():
    """S3 download → RuntimeError (файл не найден в S3)."""
    with patch(
        "app.core.s3.download_file",
        side_effect=RuntimeError("S3 download failed: NoSuchKey"),
    ):
        yield


@pytest.fixture()
def mock_s3_upload():
    """S3 upload → no-op, возвращает мок для проверки вызовов."""
    with patch("app.core.s3.upload_file") as mock_upload:
        yield mock_upload


# ── Полный мок пайплайна ──────────────────────────────────────────────────────

@pytest.fixture()
def mock_pipeline(mock_s3_download_ok, mock_s3_upload):
    """
    Мокает всю цепочку обработки:
      cv2.VideoCapture    → fps=30
      SkeletonExtractor   → возвращает пустой список frames
      _build_result       → записывает фиктивный result.json на диск

    Мок _build_result критичен: processing.py записывает его результат
    в tmpdir и передаёт в upload_file — тест проверяет реальный I/O.
    """
    fake_result = make_fake_result(num_frames=90, fps=30.0)

    mock_cap = MagicMock()
    mock_cap.get.return_value = 30.0

    def fake_build_result(frames, segments, fps):
        return fake_result

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("app.services.processing.SkeletonExtractor") as mock_extractor_cls,
        patch("app.services.processing._build_result", side_effect=fake_build_result),
    ):
        mock_instance = MagicMock()
        mock_instance.process_video.return_value = []
        mock_extractor_cls.return_value.__enter__.return_value = mock_instance
        mock_extractor_cls.return_value.__exit__.return_value = False

        yield {
            "upload_mock": mock_s3_upload,
            "extractor_instance": mock_instance,
            "fake_result": fake_result,
        }