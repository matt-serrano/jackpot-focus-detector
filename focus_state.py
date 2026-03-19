"""Stateful attention logic for determining focus vs. distraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from face_features import FaceFeaturesResult
from head_pose import HeadPoseResult


@dataclass(slots=True)
class FocusStatus:
    """Current focus state and the reasons behind it."""

    state: str
    reasons: list[str] = field(default_factory=list)
    should_trigger_alert: bool = False


class FocusStateEngine:
    """Combine per-frame signals into a stable focus state."""

    def __init__(self, config) -> None:
        self.config = config
        self.last_alert_time = 0.0

    def update(self, face_result: FaceFeaturesResult, pose_result: HeadPoseResult) -> FocusStatus:
        reasons: list[str] = []

        if not face_result.face_present:
            reasons.append("face missing")

        if pose_result.head_turned_away:
            reasons.append("head turned away")

        state = "focused" if not reasons else "unfocused"
        should_trigger_alert = self._should_trigger_alert(state)

        return FocusStatus(
            state=state,
            reasons=reasons,
            should_trigger_alert=should_trigger_alert,
        )

    def _should_trigger_alert(self, state: str) -> bool:
        if state != "unfocused":
            return False

        now = time.monotonic()
        if now - self.last_alert_time < self.config.alert_cooldown_seconds:
            return False

        self.last_alert_time = now
        return True
