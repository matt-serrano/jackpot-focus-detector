"""Local alert playback utilities for unfocused events."""

from __future__ import annotations


class AlertPlayer:
    """Stub alert player for the future funny video popup."""

    def __init__(self, config) -> None:
        self.config = config

    def play(self) -> None:
        """Play the configured alert.

        This is intentionally a no-op scaffold for now so the main app
        can be wired before we decide how the local popup video should
        be displayed.
        """
        return None
