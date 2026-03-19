"""Shared configuration for the focus detector application."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    """Central configuration for thresholds, timers, and paths."""

    camera_index: int = 0
    window_name: str = "Jackpot Focus Detector"
    quit_key: str = "q"

    face_missing_seconds: float = 2.0
    head_away_seconds: float = 2.0
    phone_detected_seconds: float = 2.0
    alert_cooldown_seconds: float = 8.0

    head_yaw_threshold_degrees: float = 20.0
    head_pitch_threshold_degrees: float = 20.0

    blink_ear_threshold: float = 0.30
    blink_consecutive_frames: int = 4

    popup_video_path: str = "DATA/VIDEOS/INPUTS/funny_popup.mp4"
    yolo_model_path: str = "models/phone_detector.pt"
