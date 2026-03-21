import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("pip install mediapipe")


MP_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 11), (0, 12),
]

_PROCESS_WIDTH = 640


@dataclass
class Joint:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class Frame:
    frame_idx: int
    timestamp_ms: float
    joints: List[Joint]


class ExponentialSmoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def smooth(self, joints_arr: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = joints_arr.copy()
            return joints_arr
        result = self.alpha * joints_arr + (1 - self.alpha) * self._prev
        self._prev = result
        return result


class SkeletonExtractor:

    def __init__(self,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):

        self._use_new_api = False

        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except AttributeError:
            self._use_new_api = True
            self._init_tasks_api(min_detection_confidence, min_tracking_confidence)

    def _init_tasks_api(self, det_conf, track_conf):
        import os
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        model_path = "pose_landmarker_heavy.task"
        if not os.path.exists(model_path):
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                   "pose_landmarker_heavy.task")
            urllib.request.urlretrieve(url, model_path)

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def _close(self) -> None:
        if self._use_new_api:
            if hasattr(self, "landmarker"):
                self.landmarker.close()
        else:
            if hasattr(self, "pose"):
                self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False

    def process_video(self,
                      video_path: str,
                      max_frames: Optional[int] = None,
                      frame_skip: int = 1,
                      smoothing_alpha: float = 0.3) -> List[Frame]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        smoother = ExponentialSmoother(alpha=smoothing_alpha)
        frames: List[Frame] = []
        prev_arr: Optional[np.ndarray] = None
        raw_idx = 0

        while True:
            if raw_idx % frame_skip != 0:
                if not cap.grab():
                    break
                raw_idx += 1
                continue

            ret, bgr = cap.read()
            if not ret:
                break

            if max_frames and len(frames) >= max_frames:
                break

            timestamp_ms = raw_idx / fps * 1000.0

            h, w = bgr.shape[:2]
            if w > _PROCESS_WIDTH:
                scale = _PROCESS_WIDTH / w
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            landmarks_3d = None

            if self._use_new_api:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(
                    mp_image, int(timestamp_ms * 1000))
                if result.pose_world_landmarks:
                    landmarks_3d = result.pose_world_landmarks[0]
            else:
                result = self.pose.process(rgb)
                if result.pose_world_landmarks:
                    landmarks_3d = result.pose_world_landmarks.landmark

            if landmarks_3d:
                raw_arr = np.array(
                    [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]
                     for lm in landmarks_3d],
                    dtype=np.float32,
                )
                prev_arr = raw_arr
            else:
                raw_arr = prev_arr if prev_arr is not None else np.zeros((33, 4), dtype=np.float32)

            smoothed_arr = smoother.smooth(raw_arr)

            frames.append(Frame(
                frame_idx=raw_idx,
                timestamp_ms=timestamp_ms,
                joints=[
                    Joint(x=float(smoothed_arr[i, 0]),
                          y=float(smoothed_arr[i, 1]),
                          z=float(smoothed_arr[i, 2]),
                          visibility=float(smoothed_arr[i, 3]))
                    for i in range(33)
                ],
            ))
            raw_idx += 1

        cap.release()
        self._close()
        return frames


def save_skeleton_json(frames: List[Frame], output_path: str,
                       fps: float = 30.0) -> None:
    data = {
        "fps": fps,
        "num_frames": len(frames),
        "joint_names": MP_LANDMARK_NAMES,
        "connections": SKELETON_CONNECTIONS,
        "frames": [
            {
                "frame_idx": f.frame_idx,
                "timestamp_ms": f.timestamp_ms,
                "joints": [
                    {"x": round(j.x, 5),
                     "y": round(j.y, 5),
                     "z": round(j.z, 5),
                     "vis": round(j.visibility, 4)}
                    for j in f.joints
                ],
            }
            for f in frames
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False)