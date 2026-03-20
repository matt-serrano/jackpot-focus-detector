"""Local alert playback utilities for unfocused events."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading


class AlertPlayer:
    """Launch the configured local popup video in the system media player."""

    def __init__(self, config) -> None:
        self.config = config
        self.popup_path = self._resolve_popup_path(config.popup_video_path)
        self._process_handle = None
        self._fallback_process: subprocess.Popen | None = None
        self._launch_lock = threading.Lock()
        self._launching = False

    def play(self) -> None:
        """Play the configured alert video with the system's default player."""
        if self.popup_path is None or self.is_playing() or self._launching:
            return

        self._launching = True
        threading.Thread(target=self._play_internal, daemon=True).start()

    def _play_internal(self) -> None:
        try:
            if sys.platform.startswith("win"):
                self._launch_windows_default_player()
                return

            self._fallback_process = subprocess.Popen(["xdg-open", str(self.popup_path)])
        finally:
            self._launching = False

    def is_playing(self) -> bool:
        """Return True while the external video player process is still open."""
        if sys.platform.startswith("win"):
            return self._is_windows_process_running() or self._launching

        if self._fallback_process is None:
            return self._launching
        if self._fallback_process.poll() is None:
            return True

        self._fallback_process = None
        return self._launching

    def _launch_windows_default_player(self) -> None:
        import ctypes
        from ctypes import wintypes

        SEE_MASK_NOCLOSEPROCESS = 0x00000040
        SW_SHOWNORMAL = 1

        class SHELLEXECUTEINFOW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("fMask", ctypes.c_ulong),
                ("hwnd", wintypes.HWND),
                ("lpVerb", wintypes.LPCWSTR),
                ("lpFile", wintypes.LPCWSTR),
                ("lpParameters", wintypes.LPCWSTR),
                ("lpDirectory", wintypes.LPCWSTR),
                ("nShow", ctypes.c_int),
                ("hInstApp", wintypes.HINSTANCE),
                ("lpIDList", wintypes.LPVOID),
                ("lpClass", wintypes.LPCWSTR),
                ("hkeyClass", wintypes.HKEY),
                ("dwHotKey", wintypes.DWORD),
                ("hIcon", wintypes.HANDLE),
                ("hProcess", wintypes.HANDLE),
            ]

        execute_info = SHELLEXECUTEINFOW()
        execute_info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
        execute_info.fMask = SEE_MASK_NOCLOSEPROCESS
        execute_info.hwnd = None
        execute_info.lpVerb = "open"
        execute_info.lpFile = str(self.popup_path)
        execute_info.lpParameters = None
        execute_info.lpDirectory = str(self.popup_path.parent)
        execute_info.nShow = SW_SHOWNORMAL
        execute_info.hInstApp = None

        shell32 = ctypes.windll.shell32
        if not shell32.ShellExecuteExW(ctypes.byref(execute_info)):
            raise OSError("Failed to launch the popup video in the default media player.")

        self._process_handle = execute_info.hProcess

    def _is_windows_process_running(self) -> bool:
        if self._process_handle is None:
            return False

        import ctypes

        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32
        status = kernel32.WaitForSingleObject(self._process_handle, 0)
        if status == WAIT_TIMEOUT:
            return True

        kernel32.CloseHandle(self._process_handle)
        self._process_handle = None
        return False

    def _resolve_popup_path(self, configured_path: str) -> Path | None:
        candidate = Path(configured_path)
        if candidate.exists():
            return candidate.resolve()

        fallback = Path(__file__).resolve().parent / configured_path
        if fallback.exists():
            return fallback.resolve()

        return None
