"""Platform-specific helpers to detect the foreground application."""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from shutil import which
from typing import Callable, Dict, Optional


def _safe_run(cmd) -> Optional[str]:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def _read_process_name(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="ignore") as fp:
            return fp.readline().strip()
    except OSError:
        return None


def _read_process_exe(pid: int) -> Optional[str]:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


class ActiveApplicationResolver:
    """Resolve the foreground application using best-effort heuristics."""

    def __init__(self) -> None:
        self._system = platform.system()
        self._xdotool = which("xdotool") if self._system == "Linux" else None

    @property
    def available(self) -> bool:
        if self._system == "Linux":
            return bool(self._xdotool)
        return False

    def resolve(self) -> Optional[Dict[str, Optional[str]]]:
        if self._system == "Linux":
            return self._resolve_linux()
        return None

    # Linux -----------------------------------------------------------------
    def _resolve_linux(self) -> Optional[Dict[str, Optional[str]]]:
        if not self._xdotool:
            return None

        window_id = _safe_run([self._xdotool, "getwindowfocus"])
        if not window_id or window_id == "0":
            return None

        pid_raw = _safe_run([self._xdotool, "getwindowpid", window_id])
        window_title = _safe_run([self._xdotool, "getwindowname", window_id])

        pid_value: Optional[int] = None
        if pid_raw and pid_raw.isdigit():
            pid_value = int(pid_raw)

        process_name = _read_process_name(pid_value) if pid_value is not None else None
        executable = _read_process_exe(pid_value) if pid_value is not None else None

        app_id = process_name or executable or window_title
        if not any([app_id, process_name, window_title]):
            return None

        return {
            "app_id": app_id,
            "process_name": process_name,
            "executable": executable,
            "window_title": window_title,
            "pid": str(pid_value) if pid_value is not None else None,
        }


class ActiveApplicationPoller(threading.Thread):
    """Background thread polling the resolver and notifying on changes."""

    def __init__(
        self,
        *,
        resolver: ActiveApplicationResolver,
        interval: float,
        on_update: Callable[[Optional[Dict[str, Optional[str]]]], None],
    ) -> None:
        super().__init__(name="ActiveApplicationPoller", daemon=True)
        self._resolver = resolver
        self._interval = max(interval, 0.2)
        self._on_update = on_update
        self._stop_event = threading.Event()
        self._last_payload: Optional[Dict[str, Optional[str]]] = None

    def run(self) -> None:  # pragma: no cover - thread with blocking sleep
        while not self._stop_event.is_set():
            payload = None
            if self._resolver.available:
                try:
                    payload = self._resolver.resolve()
                except Exception:
                    payload = None

            if payload != self._last_payload:
                self._last_payload = payload
                try:
                    self._on_update(payload)
                except Exception:
                    # Collector will log/handle errors; keep polling.
                    pass

            self._stop_event.wait(self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=self._interval + 0.5)


__all__ = ["ActiveApplicationResolver", "ActiveApplicationPoller"]
