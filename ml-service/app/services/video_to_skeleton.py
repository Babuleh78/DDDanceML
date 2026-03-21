import cv2
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("pip install mediapipe")


MP_LANDMARK_NAMES = [
    "nose",                                                          # 0
    "left_eye_inner", "left_eye", "left_eye_outer",                  # 1-3
    "right_eye_inner", "right_eye", "right_eye_outer",               # 4-6
    "left_ear", "right_ear",                                         # 7-8
    "mouth_left", "mouth_right",                                     # 9-10
    "left_shoulder", "right_shoulder",                               # 11-12
    "left_elbow", "right_elbow",                                     # 13-14
    "left_wrist", "right_wrist",                                     # 15-16
    "left_pinky", "right_pinky",                                     # 17-18
    "left_index", "right_index",                                     # 19-20
    "left_thumb", "right_thumb",                                     # 21-22
    "left_hip", "right_hip",                                         # 23-24
    "left_knee", "right_knee",                                       # 25-26
    "left_ankle", "right_ankle",                                     # 27-28
    "left_heel", "right_heel",                                       # 29-30
    "left_foot_index", "right_foot_index",                           # 31-32
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
        self._prev: Optional[List[Joint]] = None

    def smooth(self, joints: List[Joint]) -> List[Joint]:
        if self._prev is None:
            self._prev = joints
            return joints

        smoothed = [
            Joint(
                x          = self.alpha * c.x          + (1 - self.alpha) * p.x,
                y          = self.alpha * c.y          + (1 - self.alpha) * p.y,
                z          = self.alpha * c.z          + (1 - self.alpha) * p.z,
                visibility = self.alpha * c.visibility + (1 - self.alpha) * p.visibility,
            )
            for c, p in zip(joints, self._prev)
        ]
        self._prev = smoothed
        return smoothed

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
            print("[extractor] MediaPipe solutions API")
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
            print("[extractor] Скачиваем модель...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                   "pose_landmarker_heavy.task")
            urllib.request.urlretrieve(url, model_path)
        else:
            print("[extractor] Используем готовую модель")

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        print("[extractor] MediaPipe tasks API")

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
        print(f"[extractor] Видео: {total} кадров, {fps:.1f} fps, "
              f"frame_skip={frame_skip}, alpha={smoothing_alpha}")

        smoother   = ExponentialSmoother(alpha=smoothing_alpha)
        frames: List[Frame] = []
        prev_joints: Optional[List[Joint]] = None
        raw_idx = 0 

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if raw_idx % frame_skip != 0:
                raw_idx += 1
                continue

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
                raw_joints = [
                    Joint(
                        x=landmarks_3d[i].x,
                        y=landmarks_3d[i].y,
                        z=landmarks_3d[i].z,
                        visibility=getattr(landmarks_3d[i], "visibility", 1.0),
                    )
                    for i in range(33)
                ]
                prev_joints = raw_joints
            else:
                raw_joints = prev_joints if prev_joints else [
                    Joint(0.0, 0.0, 0.0, 0.0) for _ in range(33)
                ]

            # Сглаживание применяется после fallback сглаживаем всегда
            smoothed_joints = smoother.smooth(raw_joints)

            frames.append(Frame(
                frame_idx=raw_idx,
                timestamp_ms=timestamp_ms,
                joints=smoothed_joints,
            ))
            raw_idx += 1

            if len(frames) % 100 == 0:
                pct = raw_idx / total * 100
                print(f"  [{pct:.0f}%] обработано {len(frames)} кадров")

        cap.release()
        self._close()
        print(f"[extractor] Готово: {len(frames)} кадров")
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
                    {"x": j.x, "y": j.y, "z": j.z, "vis": j.visibility}
                    for j in f.joints
                ],
            }
            for f in frames
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)