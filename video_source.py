"""Video capture wrapper for webcam or file input."""

from __future__ import annotations

from typing import Any

import cv2 as cv


class VideoSource:
    """Small wrapper around OpenCV video capture."""

    def __init__(self, camera_index: int = 0, video_path: str | None = None, mirror_camera: bool = True) -> None:
        self.camera_index = camera_index
        self.video_path = video_path
        self.mirror_camera = mirror_camera
        self.cap: Any | None = None

    def open(self) -> None:
        source = self.video_path if self.video_path else self.camera_index

        if self.video_path:
            self.cap = cv.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")
            return

        backend_attempts = [
            ("CAP_ANY", cv.CAP_ANY),
            ("CAP_DSHOW", cv.CAP_DSHOW),
            ("CAP_MSMF", cv.CAP_MSMF),
        ]

        errors: list[str] = []
        for backend_name, backend in backend_attempts:
            cap = cv.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                self.cap = cap
                print(f"Opened camera index {self.camera_index} with backend {backend_name}.")
                return
            cap.release()
            errors.append(backend_name)

        attempted = ", ".join(errors)
        raise RuntimeError(
            f"Failed to open camera index {self.camera_index}. Tried backends: {attempted}. "
            "Check Windows camera privacy settings and whether another app is using the webcam."
        )

    def read(self) -> tuple[bool, Any | None]:
        if self.cap is None:
            raise RuntimeError("VideoSource.open() must be called before read().")

        ok, frame = self.cap.read()
        if ok and self.video_path is None and self.mirror_camera:
            frame = cv.flip(frame, 1)
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
