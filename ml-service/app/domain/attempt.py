from dataclasses import dataclass


@dataclass
class Attempt:
    id: str
    user_id: str
    dance_id: str
    video_path: str
