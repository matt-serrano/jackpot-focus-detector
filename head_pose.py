"""Head pose estimation scaffolding for attention detection."""

from __future__ import annotations

from dataclasses import dataclass

from face_features import FaceFeaturesResult


@dataclass(slots=True)
class HeadPoseResult:
    """Estimated head orientation values for the current frame."""

    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    head_turned_away: bool = False


class HeadPoseEstimator:
    """Estimate head orientation from facial landmarks.

    This starts as a placeholder so the project has a clean module
    boundary before the real pose math is implemented.
    """

    def __init__(self, config) -> None:
        self.config = config

    def estimate(self, face_result: FaceFeaturesResult) -> HeadPoseResult:
        if not face_result.face_present:
            return HeadPoseResult()

        return HeadPoseResult(
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            head_turned_away=False,
        )
