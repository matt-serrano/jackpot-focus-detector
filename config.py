"""Shared configuration for the focus detector application."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    """Central configuration for thresholds, timers, and paths."""

    camera_index: int = 0
    source_video_path: str | None = None
    mirror_camera: bool = True

    window_name: str = "Jackpot Focus Detector"
    quit_key: str = "q"

    face_missing_seconds: float = 1.5
    head_away_seconds: float = 1.4
    alert_cooldown_seconds: float = 8.0
    ui_unfocused_display_seconds: float = 7.0

    head_yaw_enter_degrees: float = 35.0
    head_yaw_exit_degrees: float = 24.0
    head_pitch_enter_degrees: float = 28.0
    head_pitch_exit_degrees: float = 20.0
    pose_smoothing: float = 0.2
    use_pitch_for_head_away: bool = False

    blink_ear_threshold: float = 0.30
    blink_consecutive_frames: int = 4

    popup_video_path: str = "jackpot.mp4"
