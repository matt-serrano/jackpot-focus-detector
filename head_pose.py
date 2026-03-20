"""Head pose estimation for attention detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from face_features import FaceFeaturesResult


@dataclass(slots=True)
class HeadPoseResult:
    """Estimated head orientation values for the current frame."""

    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    head_turned_away: bool = False
    available: bool = False


class HeadPoseEstimator:
    """Estimate head orientation from facial landmarks using solvePnP."""

    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    LANDMARK_IDS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "mouth_left": 61,
        "mouth_right": 291,
    }

    def __init__(self, config) -> None:
        self.config = config
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0
        self.has_pose = False
        self.head_turned_away = False

    def estimate(self, face_result: FaceFeaturesResult) -> HeadPoseResult:
        if not face_result.face_present:
            self.has_pose = False
            self.head_turned_away = False
            return HeadPoseResult()

        try:
            image_points = np.array(
                [
                    face_result.landmarks[self.LANDMARK_IDS["nose_tip"]],
                    face_result.landmarks[self.LANDMARK_IDS["chin"]],
                    face_result.landmarks[self.LANDMARK_IDS["left_eye_outer"]],
                    face_result.landmarks[self.LANDMARK_IDS["right_eye_outer"]],
                    face_result.landmarks[self.LANDMARK_IDS["mouth_left"]],
                    face_result.landmarks[self.LANDMARK_IDS["mouth_right"]],
                ],
                dtype=np.float64,
            )
        except KeyError:
            self.has_pose = False
            self.head_turned_away = False
            return HeadPoseResult()

        frame_width, frame_height = face_result.frame_size
        focal_length = float(frame_width)
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _translation_vector = cv.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self.has_pose = False
            self.head_turned_away = False
            return HeadPoseResult()

        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        angles = cv.RQDecomp3x3(rotation_matrix)[0]

        yaw_degrees = float(np.clip(float(angles[1]), -90.0, 90.0))
        pitch_degrees = float(np.clip(float(angles[0]), -90.0, 90.0))
        roll_degrees = float(np.clip(float(angles[2]), -90.0, 90.0))

        self._smooth_pose(yaw_degrees, pitch_degrees, roll_degrees)
        self._update_head_state()

        return HeadPoseResult(
            yaw_degrees=self.smoothed_yaw,
            pitch_degrees=self.smoothed_pitch,
            roll_degrees=self.smoothed_roll,
            head_turned_away=self.head_turned_away,
            available=True,
        )

    def _smooth_pose(self, yaw_degrees: float, pitch_degrees: float, roll_degrees: float) -> None:
        alpha = self.config.pose_smoothing
        if not self.has_pose:
            self.smoothed_yaw = yaw_degrees
            self.smoothed_pitch = pitch_degrees
            self.smoothed_roll = roll_degrees
            self.has_pose = True
            return

        self.smoothed_yaw = (1.0 - alpha) * self.smoothed_yaw + alpha * yaw_degrees
        self.smoothed_pitch = (1.0 - alpha) * self.smoothed_pitch + alpha * pitch_degrees
        self.smoothed_roll = (1.0 - alpha) * self.smoothed_roll + alpha * roll_degrees

    def _update_head_state(self) -> None:
        yaw = abs(self.smoothed_yaw)
        pitch = abs(self.smoothed_pitch)

        if self.head_turned_away:
            yaw_back = yaw <= self.config.head_yaw_exit_degrees
            pitch_back = pitch <= self.config.head_pitch_exit_degrees or not self.config.use_pitch_for_head_away
            self.head_turned_away = not (yaw_back and pitch_back)
            return

        yaw_away = yaw >= self.config.head_yaw_enter_degrees
        pitch_away = self.config.use_pitch_for_head_away and pitch >= self.config.head_pitch_enter_degrees
        self.head_turned_away = yaw_away or pitch_away
