"""Phone detection scaffolding.

This module is reserved for a later YOLO-style object detector that can
identify a cell phone in the frame and report a smoothed detection
signal to the focus state engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PhoneDetectionResult:
    """Phone detection output for the current frame."""

    phone_detected: bool = False
    confidence: float = 0.0


class PhoneDetector:
    """Placeholder phone detector."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def detect(self, frame) -> PhoneDetectionResult:
        return PhoneDetectionResult()
