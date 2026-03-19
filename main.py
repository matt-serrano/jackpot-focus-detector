"""Main entry point for the focus detector application.

This module wires together the webcam source, face-based signals,
stateful focus logic, and alert playback. The current implementation
is intentionally lightweight so the project can grow into this shape
without disturbing the existing blink-detection scripts.
"""

from __future__ import annotations

import cv2 as cv

from config import AppConfig
from video_source import VideoSource
from face_features import FaceFeaturesExtractor
from head_pose import HeadPoseEstimator
from focus_state import FocusStateEngine
from overlay_ui import draw_status_overlay
from alert_player import AlertPlayer


def run() -> None:
    """Run the webcam processing loop for the focus detector."""
    config = AppConfig()
    source = VideoSource(camera_index=config.camera_index)
    face_features = FaceFeaturesExtractor()
    head_pose = HeadPoseEstimator(config=config)
    focus_engine = FocusStateEngine(config=config)
    alert_player = AlertPlayer(config=config)

    try:
        source.open()

        while True:
            ok, frame = source.read()
            if not ok:
                break

            face_result = face_features.process(frame)
            pose_result = head_pose.estimate(face_result)
            focus_status = focus_engine.update(face_result=face_result, pose_result=pose_result)

            draw_status_overlay(frame, focus_status, face_result, pose_result)

            if focus_status.should_trigger_alert:
                alert_player.play()

            cv.imshow(config.window_name, frame)
            if cv.waitKey(1) & 0xFF == ord(config.quit_key):
                break
    finally:
        source.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    run()
