from __future__ import annotations


class DanceNotFoundError(Exception):
    def __init__(self, dance_id: int | str) -> None:
        self.dance_id = dance_id
        super().__init__(f"Dance not found: {dance_id}")


class AttemptNotFoundError(Exception):
    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        super().__init__(f"Attempt not found: {attempt_id}")


class S3UploadError(Exception):
    def __init__(self, path: str, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"S3 upload failed for {path}: {cause}")


class MLInferenceError(Exception):
    def __init__(self, model: str, cause: Exception) -> None:
        self.model = model
        self.cause = cause
        super().__init__(f"ML inference error in model '{model}': {cause}")


class VideoProcessingError(Exception):
    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"Video processing failed at stage '{stage}': {cause}")
