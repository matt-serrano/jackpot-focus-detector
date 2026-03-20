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
    face_missing_duration: float = 0.0
    head_away_duration: float = 0.0


class FocusStateEngine:
    """Combine per-frame signals into a stable focus state."""

    def __init__(self, config) -> None:
        self.config = config
        self.last_alert_time = 0.0
        self.face_missing_started_at: float | None = None
        self.head_away_started_at: float | None = None
        self.alert_latched = False
        self.current_state = "focused"

    def update(self, face_result: FaceFeaturesResult, pose_result: HeadPoseResult) -> FocusStatus:
        now = time.monotonic()

        face_missing_duration = self._update_duration(
            is_active=not face_result.face_present,
            now=now,
            attr_name="face_missing_started_at",
        )
        head_away_duration = self._update_duration(
            is_active=face_result.face_present and pose_result.head_turned_away,
            now=now,
            attr_name="head_away_started_at",
        )

        reasons: list[str] = []
        state = "focused"

        if face_missing_duration >= self.config.face_missing_seconds:
            state = "unfocused"
            reasons.append("face missing")
        elif head_away_duration >= self.config.head_away_seconds:
            state = "unfocused"
            reasons.append("head turned away")
        elif face_missing_duration > 0 or head_away_duration > 0:
            state = "warning"

        if state == "focused":
            self.alert_latched = False

        self.current_state = state
        should_trigger_alert = self._should_trigger_alert(state=state, now=now)

        return FocusStatus(
            state=state,
            reasons=reasons,
            should_trigger_alert=should_trigger_alert,
            face_missing_duration=face_missing_duration,
            head_away_duration=head_away_duration,
        )

    def _update_duration(self, is_active: bool, now: float, attr_name: str) -> float:
        started_at = getattr(self, attr_name)

        if is_active:
            if started_at is None:
                setattr(self, attr_name, now)
                return 0.0
            return now - started_at

        setattr(self, attr_name, None)
        return 0.0

    def _should_trigger_alert(self, state: str, now: float) -> bool:
        if state != "unfocused":
            return False
        if self.alert_latched:
            return False
        if now - self.last_alert_time < self.config.alert_cooldown_seconds:
            return False

        self.last_alert_time = now
        self.alert_latched = True
        return True
