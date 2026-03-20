"""Main entry point for the focus detector application."""

from __future__ import annotations

import time

import cv2 as cv

from alert_player import AlertPlayer
from config import AppConfig
from face_features import FaceFeaturesExtractor
from focus_state import FocusStateEngine
from head_pose import HeadPoseEstimator
from overlay_ui import draw_status_overlay
from video_source import VideoSource


def run() -> None:
    """Run the webcam processing loop for the focus detector."""
    config = AppConfig()
    source = VideoSource(
        camera_index=config.camera_index,
        video_path=config.source_video_path,
        mirror_camera=config.mirror_camera,
    )
    face_features = FaceFeaturesExtractor()
    head_pose = HeadPoseEstimator(config=config)
    focus_engine = FocusStateEngine(config=config)
    alert_player = AlertPlayer(config=config)
    unfocused_display_until = 0.0

    try:
        source.open()
        cv.namedWindow(config.window_name, cv.WINDOW_NORMAL)

        while True:
            if cv.getWindowProperty(config.window_name, cv.WND_PROP_VISIBLE) < 1:
                break

            ok, frame = source.read()
            if not ok:
                break

            face_result = face_features.process(frame)
            pose_result = head_pose.estimate(face_result)
            focus_status = focus_engine.update(face_result=face_result, pose_result=pose_result)

            if focus_status.should_trigger_alert:
                alert_player.play()
                unfocused_display_until = time.monotonic() + config.ui_unfocused_display_seconds

            display_state = "unfocused" if time.monotonic() < unfocused_display_until else "focused"
            draw_status_overlay(frame, display_state, face_result, pose_result)
            cv.imshow(config.window_name, frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord(config.quit_key):
                break
    finally:
        source.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    run()
