import asyncio
import logging
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from app.core.config import settings
from app.core.s3 import download_file

logger = logging.getLogger(__name__)

_PERSON_CLASS = 0
_ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
_YOLO_CONF = 0.4
_PERSON_CONF = 0.6
_YOLO_NMS_IOU = 0.5
_PERSON_IOU_DEDUP = 0.5 
_PERSON_CONTAINMENT_DEDUP = 0.7 
_MULTI_PERSON_RATIO = 0.65  

_NSFW_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
_NSFW_CONF = 0.5

_yolo_model = None
_nude_detector = None


def _yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLO moderation model: %s", settings.moderate_yolo_model)
        _yolo_model = YOLO(settings.moderate_yolo_model)
    return _yolo_model


def _nudenet():
    global _nude_detector
    if _nude_detector is None:
        from nudenet import NudeDetector
        _nude_detector = NudeDetector()
    return _nude_detector


@dataclass
class _FrameResult:
    frame_idx: int
    person_count: int
    person_max_conf: float
    has_animal: bool
    animal_max_conf: float
    is_nsfw: bool
    nsfw_max_conf: float
    nsfw_label: Optional[str] = None


def _area(b: list) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(b1: list, b2: list) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou(b1: list, b2: list) -> float:
    inter = _intersection(b1, b2)
    denom = _area(b1) + _area(b2) - inter
    return inter / denom if denom > 0 else 0.0


def _containment(b1: list, b2: list) -> float:
    smaller = min(_area(b1), _area(b2))
    return _intersection(b1, b2) / smaller if smaller > 0 else 0.0


def _same_person(b1: list, b2: list) -> bool:
    return _iou(b1, b2) > _PERSON_IOU_DEDUP or _containment(b1, b2) > _PERSON_CONTAINMENT_DEDUP


def _dedup_person_boxes(person_boxes: list[list]) -> list[list]:
    merged: list[list] = []
    for item in sorted(person_boxes, key=lambda x: -x[0]):
        coords = item[1:]
        if not any(_same_person(coords, m[1:]) for m in merged):
            merged.append(item)

    if len(merged) > 1:
        main_area = max(_area(m[1:]) for m in merged)
        min_area = main_area * settings.moderate_person_min_rel_area
        merged = [m for m in merged if _area(m[1:]) >= min_area]

    return merged


def _analyze_frame(frame_idx: int, frame: np.ndarray) -> _FrameResult:
    results = _yolo()(frame, verbose=False, iou=_YOLO_NMS_IOU)
    person_boxes: list[list] = [] 
    has_animal = False
    animal_max_conf = 0.0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == _PERSON_CLASS and conf >= _PERSON_CONF:
            person_boxes.append([conf] + box.xyxy[0].tolist())
        elif cls in _ANIMAL_CLASSES and conf >= _YOLO_CONF:
            has_animal = True
            animal_max_conf = max(animal_max_conf, conf)

    merged = _dedup_person_boxes(person_boxes)
    person_count = len(merged)
    person_max_conf = max((m[0] for m in merged), default=0.0)

    h, w = frame.shape[:2]
    frame_area = float(h * w) or 1.0
    boxes_repr = "; ".join(
        f"conf={m[0]:.2f},size={_area(m[1:]) / frame_area * 100:.1f}%" for m in merged
    )

    is_nsfw = False
    nsfw_max_conf = 0.0
    nsfw_label = None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    try:
        cv2.imwrite(tmp_path, frame)
        detections = _nudenet().detect(tmp_path)
        for det in detections:
            lbl = det.get("class", "")
            score = float(det.get("score", 0.0))
            if lbl in _NSFW_LABELS and score > _NSFW_CONF:
                is_nsfw = True
                if score > nsfw_max_conf:
                    nsfw_max_conf = score
                    nsfw_label = lbl
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result = _FrameResult(
        frame_idx=frame_idx,
        person_count=person_count,
        person_max_conf=person_max_conf,
        has_animal=has_animal,
        animal_max_conf=animal_max_conf,
        is_nsfw=is_nsfw,
        nsfw_max_conf=nsfw_max_conf,
        nsfw_label=nsfw_label,
    )

    logger.info(
        "frame=%d persons=%d (raw=%d) [%s] animal=%s(conf=%.2f) nsfw=%s(conf=%.2f label=%s)",
        frame_idx,
        person_count,
        len(person_boxes),
        boxes_repr,
        has_animal,
        animal_max_conf,
        is_nsfw,
        nsfw_max_conf,
        nsfw_label,
    )
    return result


def _extract_random_frames(video_path: str, n: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = sorted(random.sample(range(total), min(n, total)))
    frames: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def _run_analysis(video_path: str, n_frames: int) -> list[_FrameResult]:
    frame_data = _extract_random_frames(video_path, n_frames)
    if not frame_data:
        raise ValueError("No frames extracted from video")
    return [_analyze_frame(idx, frame) for idx, frame in frame_data]


def _majority(count: int, total: int, ratio: float = 0.5) -> bool:
    return count > total * ratio


def _evaluate(results: list[_FrameResult]) -> Optional[str]:
    total = len(results)

    no_person = sum(1 for r in results if r.person_count == 0)
    multi = sum(1 for r in results if r.person_count > 1)
    if _majority(no_person, total):
        logger.info("FAIL no_person: %d/%d frames", no_person, total)
        return "no_person"
    if not settings.moderate_multi_person_check:
        if multi:
            logger.info("multiple_persons check disabled — %d/%d frames had 2+", multi, total)
    elif _majority(multi, total, ratio=_MULTI_PERSON_RATIO):
        logger.info("FAIL multiple_persons: %d/%d frames", multi, total)
        return "multiple_persons"

    animal = sum(1 for r in results if r.has_animal)
    if _majority(animal, total):
        logger.info("FAIL animal: %d/%d frames", animal, total)
        return "animal"

    if any(r.is_nsfw for r in results):
        worst = max((r.nsfw_max_conf for r in results if r.is_nsfw), default=0.0)
        logger.info("FAIL nsfw: at least 1 frame, max_conf=%.2f", worst)
        return "nsfw"

    return None


def _parse_s3_key(url_or_key: str) -> str:
    if url_or_key.startswith("s3://"):
        parsed = urlparse(url_or_key)
        return parsed.path.lstrip("/")
    if url_or_key.startswith("http"):
        parsed = urlparse(url_or_key)
        path = parsed.path.lstrip("/")
        bucket = settings.s3_bucket
        if path.startswith(bucket + "/"):
            path = path[len(bucket) + 1:]
        return path
    return url_or_key


def _download_sync(s3_key: str, local_path: str) -> None:
    download_file(s3_key, local_path)


def moderate_video_file(
    video_path: str,
    dance_id: str,
    uploader_user_id: str,
    uploader_login: str = "",
    video_s3_url: str = "",
) -> Optional[str]:
    import shutil
    import threading

    n = settings.moderate_power
    logger.info("Moderation check: dance_id=%s n_frames=%d path=%s", dance_id, n, video_path)

    try:
        frame_results = _run_analysis(video_path, n)
    except Exception:
        logger.exception("Moderation analysis error dance_id=%s", dance_id)
        return "other"

    reason = _evaluate(frame_results)

    if reason is not None:
        logger.info("Moderation PENDING dance_id=%s reason=%s", dance_id, reason)

        notify_path: Optional[str] = None
        try:
            tmp_fd, notify_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            shutil.copy2(video_path, notify_path)
        except Exception:
            logger.warning("Could not copy video for admin notification dance_id=%s", dance_id)
            notify_path = None

        def _notify_thread() -> None:
            from app.telegram_bot.bot import notify_admin_sync
            try:
                notify_admin_sync(
                    dance_id=dance_id,
                    reason=reason,
                    video_path=notify_path or "",
                    uploader_user_id=uploader_user_id,
                    uploader_login=uploader_login,
                    video_s3_url=video_s3_url,
                )
            except Exception:
                logger.exception("Admin notification failed dance_id=%s", dance_id)
            finally:
                if notify_path and os.path.exists(notify_path):
                    try:
                        os.unlink(notify_path)
                    except OSError:
                        pass

        threading.Thread(target=_notify_thread, daemon=True).start()

    return reason


async def moderate_video(video_s3_url: str, dance_id: str, uploader_user_id: str, uploader_login: str = "") -> dict:
    s3_key = _parse_s3_key(video_s3_url)
    video_path: Optional[str] = None

    try:
        tmp_fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        logger.info("Downloading s3_key=%s for dance_id=%s", s3_key, dance_id)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_download_sync, s3_key, video_path),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error("S3 download timeout dance_id=%s", dance_id)
            return {"status": "pending", "error_code": "MODERATION_PENDING", "reason": "other"}

        n = settings.moderate_power
        logger.info("Analyzing %d frames for dance_id=%s", n, dance_id)
        try:
            frame_results = await asyncio.wait_for(
                asyncio.to_thread(_run_analysis, video_path, n),
                timeout=180.0,
            )
        except asyncio.TimeoutError:
            logger.error("Analysis timeout dance_id=%s", dance_id)
            return {"status": "pending", "error_code": "MODERATION_PENDING", "reason": "other"}
        except Exception:
            logger.exception("Analysis error dance_id=%s", dance_id)
            return {"status": "pending", "error_code": "MODERATION_PENDING", "reason": "other"}

        reason = _evaluate(frame_results)

        if reason is not None:
            logger.info("Moderation PENDING dance_id=%s reason=%s", dance_id, reason)
            notify_path = video_path
            video_path = None 
            asyncio.create_task(
                _notify_admin_task(dance_id, reason, notify_path, uploader_user_id, uploader_login, video_s3_url)
            )
            return {"status": "pending", "error_code": "MODERATION_PENDING", "reason": reason}

        logger.info("Moderation APPROVED dance_id=%s", dance_id)
        return {"status": "approved"}

    except Exception:
        logger.exception("Unexpected moderation error dance_id=%s", dance_id)
        return {"status": "pending", "error_code": "MODERATION_PENDING", "reason": "other"}

    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass


async def _notify_admin_task(
    dance_id: str, reason: str, video_path: str, uploader_user_id: str, uploader_login: str, video_s3_url: str
) -> None:
    try:
        from app.telegram_bot.bot import notify_admin
        await notify_admin(
            dance_id=dance_id,
            reason=reason,
            video_path=video_path,
            uploader_user_id=uploader_user_id,
            uploader_login=uploader_login,
            video_s3_url=video_s3_url,
        )
    except Exception:
        logger.exception("Admin notification failed dance_id=%s", dance_id)
    finally:
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass
