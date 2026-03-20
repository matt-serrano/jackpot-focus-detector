"""Overlay helpers for on-screen application status."""

from __future__ import annotations

from utils import DrawingUtils


def draw_status_overlay(frame, display_state: str, *_args) -> None:
    """Draw only the displayed user state on the frame."""
    state_label = display_state.upper()
    color = _state_color(display_state)
    DrawingUtils.draw_text_with_bg(
        frame,
        state_label,
        (20, 40),
        font_scale=1.0,
        thickness=3,
        bg_color=color,
        text_color=(0, 0, 0),
    )


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "unfocused":
        return (0, 0, 255)
    if state == "warning":
        return (0, 215, 255)
    return (0, 255, 0)
