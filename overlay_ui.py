"""Overlay helpers for on-screen application status."""

from __future__ import annotations

import cv2 as cv

from face_features import FaceFeaturesResult
from focus_state import FocusStatus
from head_pose import HeadPoseResult
from utils import DrawingUtils


def draw_status_overlay(frame, focus_status: FocusStatus, face_result: FaceFeaturesResult, pose_result: HeadPoseResult) -> None:
    """Draw the current focus state and debug information on the frame."""
    status_text = f"State: {focus_status.state.upper()}"
    reasons_text = "Reasons: " + (", ".join(focus_status.reasons) if focus_status.reasons else "none")
    ear_text = f"EAR: {face_result.average_ear:.3f}" if face_result.average_ear is not None else "EAR: n/a"
    yaw_text = f"Yaw: {pose_result.yaw_degrees:.1f}"
    pitch_text = f"Pitch: {pose_result.pitch_degrees:.1f}"

    DrawingUtils.draw_text_with_bg(frame, status_text, (20, 30), font_scale=0.7, thickness=2)
    DrawingUtils.draw_text_with_bg(frame, reasons_text, (20, 60), font_scale=0.6, thickness=2)
    DrawingUtils.draw_text_with_bg(frame, ear_text, (20, 90), font_scale=0.6, thickness=2)
    DrawingUtils.draw_text_with_bg(frame, yaw_text, (20, 120), font_scale=0.6, thickness=2)
    DrawingUtils.draw_text_with_bg(frame, pitch_text, (20, 150), font_scale=0.6, thickness=2)

    if face_result.face_present:
        cv.circle(frame, (30, 190), 10, (0, 255, 0), cv.FILLED)
    else:
        cv.circle(frame, (30, 190), 10, (0, 0, 255), cv.FILLED)
