"""Future home for gaze and off-screen attention heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from face_features import FaceFeaturesResult
from head_pose import HeadPoseResult


@dataclass(slots=True)
class GazeAttentionResult:
    """Estimated eye or gaze-based attention direction."""

    looking_off_screen: bool = False
    confidence: float = 0.0


class GazeAttentionEstimator:
    """Placeholder for future gaze estimation logic."""

    def estimate(self, face_result: FaceFeaturesResult, pose_result: HeadPoseResult) -> GazeAttentionResult:
        return GazeAttentionResult()
