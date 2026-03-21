# Dance ML Service

FastAPI сервис: танцевальное видео → скелет + сегменты движений в одном файле.

## Структура проекта

```
ml-service/
├── app/
│   ├── main.py                        # FastAPI entrypoint
│   ├── api/routes/
│   │   └── process.py                 # POST /process, GET /health
│   ├── core/
│   │   ├── config.py                  # env vars (pydantic-settings)
│   │   └── s3.py                      # boto3 download/upload
│   ├── services/
│   │   ├── processing.py              # пайплайн: S3 → skeleton → segments → S3
│   │   ├── video_to_skeleton.py       # ← твой файл
│   │   └── skeleton_to_segments.py    # ← твой файл
│   └── schemas/
│       └── process.py                 # Pydantic request/response
├── tests/
│   ├── conftest.py                    # фикстуры и моки
│   └── test_process.py                # 9 тестов
├── Dockerfile
├── docker-compose.yml                 # + MinIO для локальной разработки
├── requirements.txt
└── .env.example
```

## Быстрый старт

```bash
cp video_to_skeleton.py app/services/
cp skeleton_to_segments.py app/services/
cp .env.example .env
docker-compose up --build
```

Сервис: http://localhost:8000  
MinIO console: http://localhost:9001 (minioadmin / minioadmin)  
Swagger UI: http://localhost:8000/docs

## API

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /process`
**Request:**
```json
{
  "bucket": "dance-videos",
  "video_key": "videos/abc123.mp4"
}
```
**Response:**
```json
{
  "result_key": "results/abc123_result.json",
  "num_frames": 900,
  "num_segments": 7,
  "duration_sec": 30.0
}
```

**Errors:**
| Code | Причина |
|------|---------|
| `422` | Видео не найдено в S3 или невалидный request |
| `500` | Ошибка обработки или S3 upload |
| `503` | Сервер занят — повторить позже |

## Формат result.json

Один файл содержит и скелет и сегменты:

```json
{
  "version": "1.0",
  "meta": {
    "fps": 30.0,
    "num_frames": 900,
    "duration_sec": 30.0,
    "processed_at": "2026-03-21T12:00:00Z"
  },
  "joint_names": ["nose", "left_eye_inner", ...],
  "connections": [[11, 12], [11, 23], ...],
  "segments": [
    {
      "index": 0,
      "label": "segment_1",
      "start_frame": 0,
      "end_frame": 145,
      "start_ms": 0.0,
      "end_ms": 4833.3,
      "duration_sec": 4.833,
      "num_frames": 145
    }
  ],
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_ms": 0.0,
      "joints": [
        {"x": 0.01, "y": -0.45, "z": -0.12, "vis": 0.99}
      ]
    }
  ]
}
```

## Переменные окружения

| Переменная | Описание | Дефолт |
|---|---|---|
| `S3_ENDPOINT_URL` | URL S3 / MinIO | — |
| `S3_ACCESS_KEY` | Access key | — |
| `S3_SECRET_KEY` | Secret key | — |
| `S3_BUCKET` | Имя бакета | — |
| `SKELETON_MODEL_COMPLEXITY` | MediaPipe complexity (0–2) | `2` |
| `SEGMENTER_MIN_SEG_SEC` | Мин. длина сегмента, сек | `0.8` |
| `SEGMENTER_SENSITIVITY` | Sensitivity пиков (0–1) | `0.05` |
| `SEGMENTER_SMOOTH_WINDOW` | Окно сглаживания, кадры | `15` |

## Запуск тестов

```bash
docker-compose run --rm ml-service pytest -v
```