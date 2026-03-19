"""Video capture wrapper for webcam input."""

from __future__ import annotations

from typing import Any

import cv2 as cv


class VideoSource:
    """Small wrapper around OpenCV video capture."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: Any | None = None

    def open(self) -> None:
        self.cap = cv.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.camera_index}")

    def read(self) -> tuple[bool, Any | None]:
        if self.cap is None:
            raise RuntimeError("VideoSource.open() must be called before read().")
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
