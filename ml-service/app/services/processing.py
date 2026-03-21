"""
processing.py
Склеивает S3, video_to_skeleton и skeleton_to_segments в один пайплайн.
"""

import json
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from app.core.config import settings
from app.core import s3 as s3_client

from app.services.video_to_skeleton import (
    SkeletonExtractor,
    MP_LANDMARK_NAMES,
    SKELETON_CONNECTIONS,
    Frame,
    Joint,
)
from app.services.skeleton_to_segments import compute_energy, detect_boundaries, build_segments

logger = logging.getLogger(__name__)


def _make_result_key(video_key: str) -> str:
    stem = Path(video_key).stem
    return f"results/{stem}_result.json"


def _generate_test_frames(num_frames: int = 90, fps: float = 30.0) -> List[Frame]:
    frames = []
    for i in range(num_frames):
        joints = []
        for j in range(33):
            x = 0.5 + np.sin(i * 0.1) * 0.1 + np.random.normal(0, 0.01)
            y = 0.5 + np.cos(i * 0.1) * 0.1 + np.random.normal(0, 0.01)
            z = 0.0 + np.random.normal(0, 0.01)
            
            joints.append(Joint(
                x=float(x),
                y=float(y),
                z=float(z),
                visibility=0.95
            ))
        
        frames.append(Frame(
            frame_idx=i,
            timestamp_ms=i * (1000 / fps),
            joints=joints,
        ))
    
    return frames


def _build_result(
    frames: List[Frame],
    segments: list,
    fps: float,
) -> dict:
    """Собирает единый result.json из данных скелета и сегментов."""
    num_frames = len(frames)
    duration_sec = round(frames[-1].timestamp_ms / 1000, 3) if frames else 0.0

    return {
        "version": "1.0",
        "meta": {
            "fps": fps,
            "num_frames": num_frames,
            "duration_sec": duration_sec,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        "joint_names": MP_LANDMARK_NAMES,
        "connections": SKELETON_CONNECTIONS,
        "segments": segments,
        "frames": [
            {
                "frame_idx": f.frame_idx,
                "timestamp_ms": f.timestamp_ms,
                "joints": [
                    {"x": j.x, "y": j.y, "z": j.z, "vis": j.visibility}
                    for j in f.joints
                ],
            }
            for f in frames
        ],
    }


def process_video(video_key: str) -> dict:
    result_key = _make_result_key(video_key)
 
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = str(tmpdir / Path(video_key).name)
 
        s3_client.download_file(video_key, video_path)
 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        with SkeletonExtractor(
            model_complexity=settings.skeleton_model_complexity
        ) as extractor:
            frames = extractor.process_video(
                video_path,
                frame_skip=settings.skeleton_frame_skip,
            )
 
        raw_frames = [
            {
                "frame_idx": f.frame_idx,
                "timestamp_ms": f.timestamp_ms,
                "joints": [
                    {"x": j.x, "y": j.y, "z": j.z, "vis": j.visibility}
                    for j in f.joints
                ],
            }
            for f in frames
        ]
 
        energy, _ = compute_energy(
            raw_frames,
            smooth_window=settings.segmenter_smooth_window,
        )
        boundaries = detect_boundaries(
            energy,
            fps=fps,
            min_segment_sec=settings.segmenter_min_seg_sec,
            sensitivity=settings.segmenter_sensitivity,
        )
        segments = build_segments(raw_frames, boundaries, fps)
        result_data = _build_result(frames, segments, fps)
 
        result_path = tmpdir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
 
        size_mb = result_path.stat().st_size / 1e6
        s3_client.upload_file(str(result_path), result_key)
 
    return {
        "result_key": result_key,
        "num_frames": len(frames),
        "num_segments": len(segments),
        "duration_sec": result_data["meta"]["duration_sec"],
    }

def process_video(video_key: str) -> dict:
    result_key = _make_result_key(video_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = str(tmpdir / Path(video_key).name)

        s3_client.download_file(video_key, video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        with SkeletonExtractor(
            model_complexity=settings.skeleton_model_complexity
        ) as extractor:
            frames = extractor.process_video(
                video_path,
                frame_skip=settings.skeleton_frame_skip,
                smoothing_alpha=settings.skeleton_smoothing_alpha,
            )

        raw_frames = [
            {
                "frame_idx": f.frame_idx,
                "timestamp_ms": f.timestamp_ms,
                "joints": [
                    {"x": j.x, "y": j.y, "z": j.z, "vis": j.visibility}
                    for j in f.joints
                ],
            }
            for f in frames
        ]

        energy, _ = compute_energy(
            raw_frames,
            smooth_window=settings.segmenter_smooth_window,
        )
        boundaries = detect_boundaries(
            energy,
            fps=fps,
            min_segment_sec=settings.segmenter_min_seg_sec,
            sensitivity=settings.segmenter_sensitivity,
        )
        segments = build_segments(raw_frames, boundaries, fps, energy=energy, min_segment_sec=settings.segmenter_min_seg_sec)

        result_data = _build_result(frames, segments, fps)

        result_path = tmpdir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False)

        s3_client.upload_file(str(result_path), result_key)

    return {
        "result_key": result_key,
        "num_frames": len(frames),
        "num_segments": len(segments),
        "duration_sec": result_data["meta"]["duration_sec"],
    }