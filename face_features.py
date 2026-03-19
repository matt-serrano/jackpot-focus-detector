"""Face-based feature extraction built on top of MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass, field

from FaceMeshModule import FaceMeshGenerator


@dataclass(slots=True)
class FaceFeaturesResult:
    """Per-frame face features derived from landmarks."""

    face_present: bool
    landmarks: dict[int, tuple[int, int]] = field(default_factory=dict)
    average_ear: float | None = None


class FaceFeaturesExtractor:
    """Extract reusable face features from the current frame."""

    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

    def __init__(self) -> None:
        self.generator = FaceMeshGenerator()

    def process(self, frame) -> FaceFeaturesResult:
        _, landmarks = self.generator.create_face_mesh(frame, draw=False)
        if not landmarks:
            return FaceFeaturesResult(face_present=False)

        average_ear = self._compute_average_ear(landmarks)
        return FaceFeaturesResult(
            face_present=True,
            landmarks=landmarks,
            average_ear=average_ear,
        )

    def _compute_average_ear(self, landmarks: dict[int, tuple[int, int]]) -> float:
        right_ear = self._eye_aspect_ratio(self.RIGHT_EYE_EAR, landmarks)
        left_ear = self._eye_aspect_ratio(self.LEFT_EYE_EAR, landmarks)
        return (right_ear + left_ear) / 2.0

    @staticmethod
    def _eye_aspect_ratio(eye_landmarks: list[int], landmarks: dict[int, tuple[int, int]]) -> float:
        import numpy as np

        a = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
        b = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
        c = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
        return (a + b) / (2.0 * c)
