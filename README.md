# DDDance ML Service

[![CI](https://github.com/Babuleh78/DDDanceML/actions/workflows/ci.yml/badge.svg)](https://github.com/Babuleh78/DDDanceML/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9-blue)

Computer-vision microservice that powers the dance-comparison features of
[DDDance](https://dddance.ru). It turns a video of a person dancing into a 3D
skeleton, splits the choreography into movement segments, and scores how
closely a user's attempt matches a reference dance — per segment, per joint.

> This is the ML/CV component of a larger platform (Go API + React frontend +
> this service). It runs as an independent FastAPI + Celery service and talks to
> the rest of the system over HTTP and shared S3 storage.

---

## What it does

```
        ┌─────────────┐     video      ┌──────────────────────────────────────┐
 user → │  Go backend │ ─────────────► │            ML service (this)         │
        └─────────────┘   (HTTP +      │                                      │
              ▲            Celery)      │  1. MediaPipe  → 3D landmarks/frame  │
              │                         │  2. Mixamo retarget → skeleton/quat  │
              │  status / result        │  3. Quaternion-energy segmentation  │
              └──────────────  S3  ◄────│  4. DTW comparison (ref vs attempt) │
                              JSON/GLB   │  5. Blender → GLB animation render  │
                                         └──────────────────────────────────────┘
```

1. **Pose estimation** — MediaPipe extracts 3D body landmarks per frame
   (`video_to_json.py`). VFR webm uploads are normalized to CFR 30 fps upstream
   so `t = frame / fps` stays accurate.
2. **Retargeting** — landmarks are mapped onto a Mixamo rig as quaternions
   (`py_module/`), giving a rotation-based skeleton that is scale/position
   invariant.
3. **Segmentation** — choreography is cut into discrete moves using
   quaternion-energy peaks (`skeleton_to_segments.py`).
4. **Comparison** — a DTW comparator (`compare.py`) aligns the user's attempt
   to the reference over 12 keypoints and scores each segment with a weighted
   cosine similarity (85% limbs / 15% body), producing per-frame `hit`,
   timing, amplitude and pose sub-scores.
5. **Rendering** — a headless Blender pass turns the skeleton into a `.glb`
   animation for the 3D viewer (`blender_logic/`).
6. **Extras** — pre-upload moderation (YOLO + nudenet), a sentence-transformer
   based dance recommender, a Reels-style feed ranker, and a yt-dlp downloader
   for importing reference dances from VK/YouTube/Instagram.

---

## Architecture

```
app/
├── main.py            # FastAPI app, routers, exception handlers, Prometheus
├── core/              # config (pydantic-settings), redis, s3, metrics, exceptions
├── domain/            # pure dataclasses — no imports from services/core
├── api/routes/        # HTTP endpoints (process, moderate, recommend, reels, health)
├── schemas/           # pydantic request/response models
├── services/          # the actual CV/ML pipeline (compare, segmentation, ...)
└── worker/            # Celery app + tasks (process_video, compare_dance, keyframes)
```

Design notes worth calling out:

- **Config is centralized** in `app/core/config.py` via `pydantic-settings` —
  no scattered `os.getenv`. Redis passwords are spliced into connection URLs in
  one place (see `_with_redis_password`).
- **`app/domain/` is dependency-free** by design (only stdlib + dataclasses), so
  the core comparison/attempt types stay testable in isolation.
- **Celery tasks always set `soft_time_limit` + `time_limit`** — MediaPipe can
  hang on malformed input; `compare_dance` is capped at 300/330 s.
- **Two queues**: `video_processing` (latency-sensitive) and `background`
  (keyframes, warmups), so a slow render can't block a user-facing comparison.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/process` | Enqueue reference-dance processing (returns task id) |
| `POST` | `/process-url/` | Import + process a dance from a video URL |
| `GET`  | `/status/{task_id}` | Poll Celery task status |
| `POST` | `/dance_compare` | Compare a user attempt against a reference |
| `GET`  | `/segment_description/{dance_id}/{segment_idx}` | Per-segment description |
| `POST` | `/moderate` | Pre-upload moderation (YOLO + nudenet) |
| `POST` | `/recommend`, `/similar` | Dance recommendations |
| `POST` | `/reels_feed` | Reels-style feed ranking |
| `GET`  | `/health`, `/ready` | Liveness / readiness (checks Redis + S3) |

Interactive docs at `http://localhost:8000/docs` once running.

---

## Running locally

Requires Docker. The service expects an external `ml-internal` network and a
reachable Redis + S3 (MinIO in dev).

```bash
cp .env.example .env          # then fill in real values
docker network create ml-internal   # once, if it doesn't exist
docker compose up --build
```

This starts three containers: the FastAPI web process, a Celery `worker`, and
`redis`. The API is served on `:8000`.

### Without Docker

```bash
cd ml-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# in another shell:
celery -A app.worker.celery_app worker -Q video_processing,background --loglevel=info
```

### Tests / lint

Run from the repository root (where `pyproject.toml` lives):

```bash
pip install -r requirements-dev.txt
ruff check ml-service       # lint
mypy                        # type-check (config targets ml-service/app)
pytest                      # unit tests
```

---

## S3 layout

```
results/{dance_id}/        # reference: video.mp4, full_animation.glb, segments.json, skeleton.json, keyframes.json
users/{user_id}/{attempt_id}/   # attempt: video.mp4, user_animation.glb, skeleton.json, comparison_result.json
dance-landmarks-cache/{dance_id}.json   # landmark cache (written by processing, read by compare)
```

Anonymous attempts use `attempt_id` as the user id (`users/{attempt_id}/{attempt_id}/...`).

---

## Tech stack

FastAPI · Celery · Redis · MediaPipe · OpenCV · NumPy/SciPy · dtaidistance ·
PyTorch · sentence-transformers · Ultralytics YOLO · nudenet · headless Blender ·
boto3 · Prometheus.

## Security

Credentials live only in `.env` (gitignored — see `.env.example`) and in
`secrets/` (gitignored — see `secrets/README.md`). Nothing secret is baked into
the image or committed to the repo.
