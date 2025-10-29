"""Minimal Tkinter UI wrapper for the local mouse collector."""
from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Optional

try:  # pragma: no cover - Tkinter optional
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:  # pragma: no cover - Tkinter optional
    tk = None
    ttk = None
    _TK_IMPORT_ERROR = exc
else:  # pragma: no cover - Tkinter optional
    _TK_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class CollectorUI:
    """Simple UI to toggle the collector and manage session labels."""

    def __init__(
        self,
        *,
        collector,
        session_types: Optional[Iterable[str]] = None,
        status_poll_interval: float = 0.5,
    ) -> None:
        if tk is None:
            raise RuntimeError("Tkinter is required for --ui mode") from _TK_IMPORT_ERROR

        self.collector = collector
        self._status_poll_ms = max(int(status_poll_interval * 1000), 200)
        self._status_after_id: Optional[str] = None
        self._last_session_summary: Optional[dict] = None

        self._session_type_values: List[str] = []
        for value in session_types or []:
            value = (value or "").strip()
            if not value:
                continue
            if value not in self._session_type_values:
                self._session_type_values.append(value)

        context = self.collector.get_session_context()
        default_session_type = context.get("session_type") or ""
        if default_session_type and default_session_type not in self._session_type_values:
            self._session_type_values.insert(0, default_session_type)

        self.root = tk.Tk()
        self.root.title("Local Mouse Collector")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.session_type_var = tk.StringVar(value="")
        self.notes_var = tk.StringVar(value=context.get("notes") or "")
        self.status_line_var = tk.StringVar(value="Session idle")
        self.context_line_var = tk.StringVar(value="Session label: type=unspecified")
        self.active_app_var = tk.StringVar(value="Active app: --")
        self.message_var = tk.StringVar(value="Ready")
        self.last_session_var = tk.StringVar(value="Last session: none yet")

        self._previous_on_session_closed = getattr(self.collector, "on_session_closed", None)
        self.collector.on_session_closed = self._handle_session_closed

        self._build_layout()
        self._refresh_status()
        self._schedule_status_poll()

    # UI wiring -----------------------------------------------------------
    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def _build_layout(self) -> None:
        pad = 10

        control_frame = ttk.Frame(self.root, padding=(pad, pad, pad, 0))
        control_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(control_frame, text="Session Type").grid(row=0, column=0, sticky="w")
        self.session_type_box = ttk.Combobox(
            control_frame,
            textvariable=self.session_type_var,
            values=self._session_type_values,
            width=24,
            state="normal",
        )
        self.session_type_box.grid(row=1, column=0, sticky="ew", pady=(0, pad))
        self.session_type_box.bind("<<ComboboxSelected>>", self._on_session_type_changed)
        self.session_type_box.bind("<KeyRelease>", self._on_session_type_changed)

        ttk.Label(control_frame, text="Notes").grid(row=2, column=0, sticky="w")
        self.notes_entry = ttk.Entry(control_frame, textvariable=self.notes_var, width=28)
        self.notes_entry.grid(row=3, column=0, sticky="ew", pady=(0, pad))

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, sticky="ew", pady=(0, pad))
        button_frame.columnconfigure((0, 1), weight=1)

        self.start_button = ttk.Button(button_frame, text="Start", command=self._start_collector)
        self.start_button.grid(row=0, column=0, padx=2, sticky="ew")

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_collector, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=2, sticky="ew")

        status_frame = ttk.Frame(self.root, padding=(pad, pad, pad, pad))
        status_frame.grid(row=1, column=0, sticky="nsew")

        ttk.Label(status_frame, textvariable=self.status_line_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.context_line_var).grid(row=1, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.active_app_var).grid(row=2, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.message_var, foreground="#444").grid(row=3, column=0, sticky="w", pady=(pad // 2, 0))
        ttk.Label(status_frame, textvariable=self.last_session_var, wraplength=360, foreground="#222").grid(row=4, column=0, sticky="w", pady=(pad // 2, 0))

    def _schedule_status_poll(self) -> None:
        self._status_after_id = self.root.after(self._status_poll_ms, self._refresh_status_loop)

    def _refresh_status_loop(self) -> None:
        self._refresh_status()
        self._schedule_status_poll()

    def _refresh_status(self) -> None:
        try:
            status = self.collector.get_status()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read collector status")
            self.message_var.set(f"Status error: {exc}")
            return

        session_id = status.get("session_id") or "--"
        gesture_count = status.get("gesture_count", 0)
        duration = status.get("session_duration_seconds", 0.0) or 0.0
        running = bool(status.get("running"))
        context = status.get("context", {}) or {}

        self.status_line_var.set(f"Session: {session_id} · Gestures: {gesture_count} · Duration: {duration:.1f}s")

        ctx_parts: List[str] = []
        session_type = context.get("session_type") or "unspecified"
        ctx_parts.append(f"type={session_type}")
        notes = context.get("notes")
        if notes:
            ctx_parts.append(f"notes={notes}")
        self.context_line_var.set("Session label: " + ", ".join(ctx_parts))

        active_app = status.get("active_application") or {}
        app_name = active_app.get("app_id") or active_app.get("process_name") or "--"
        window_title = active_app.get("window_title")
        if window_title and window_title != app_name:
            self.active_app_var.set(f"Active app: {app_name} · {window_title}")
        else:
            self.active_app_var.set(f"Active app: {app_name}")

        try:
            focus_widget = self.root.focus_get()
        except (KeyError, AttributeError):  # transient popdown widgets can disappear mid-focus query
            focus_widget = None

        if running:
            if self.session_type_var.get() != session_type:
                self.session_type_var.set(session_type)
            if notes is not None and self.notes_var.get() != (notes or ""):
                self.notes_var.set(notes or "")

        if running:
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            self.session_type_box.configure(state="disabled")
            self.notes_entry.configure(state="disabled")
        else:
            has_session_type = bool(self.session_type_var.get().strip())
            self.start_button.state(["!disabled"] if has_session_type else ["disabled"])
            self.stop_button.state(["disabled"])
            self.session_type_box.configure(state="normal")
            self.notes_entry.configure(state="normal")

    def _on_session_type_changed(self, event=None) -> None:  # noqa: ANN001 - Tk signature
        # Re-evaluate start button state when the user changes the session type.
        self.root.after_idle(self._refresh_status)

    def _start_collector(self) -> None:
        if self.collector.get_status().get("running"):
            return

        session_type = self.session_type_var.get().strip()
        if not session_type:
            self.message_var.set("Select a session type before starting.")
            return

        notes_value = self.notes_var.get().strip()
        notes = notes_value if notes_value else None

        if session_type not in self._session_type_values:
            self._session_type_values.append(session_type)
            self.session_type_box.configure(values=self._session_type_values)

        self.collector.set_session_context(
            session_type=session_type,
            notes=notes,
        )

        def _start() -> None:
            try:
                self.collector.start()
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Failed to start collector")
                self.root.after(0, self._handle_start_failure, exc)

        threading.Thread(target=_start, daemon=True).start()
        self.session_type_box.configure(state="disabled")
        self.notes_entry.configure(state="disabled")
        self.message_var.set(f"Collector starting… (type={session_type})")

    def _handle_start_failure(self, error: Exception) -> None:
        self.message_var.set(f"Start failed: {error}")
        self.session_type_box.configure(state="normal")
        self.notes_entry.configure(state="normal")

    def _stop_collector(self) -> None:
        if not self.collector.get_status().get("running"):
            return

        def _stop() -> None:
            try:
                self.collector.stop()
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Failed to stop collector")
                self.root.after(0, lambda: self.message_var.set(f"Stop failed: {exc}"))

        threading.Thread(target=_stop, daemon=True).start()
        self.message_var.set("Collector stopping…")

    def _handle_session_closed(self, summary: dict) -> None:
        self._last_session_summary = summary

        def _update() -> None:
            self._update_last_session_summary(summary)
            session_id = summary.get("session_id", "--")
            gestures = summary.get("gesture_count", 0)
            self.message_var.set(f"Session {session_id} saved ({gestures} gestures)")

        self.root.after(0, _update)

        previous = self._previous_on_session_closed
        if previous and previous is not self._handle_session_closed:
            try:
                previous(summary)
            except Exception:  # pragma: no cover - defensive chaining
                logger.exception("Chained session_closed callback failed")

    def _update_last_session_summary(self, summary: dict) -> None:
        session_id = summary.get("session_id", "--")
        gestures = summary.get("gesture_count", 0)
        duration = summary.get("duration_seconds", 0.0) or 0.0
        context = summary.get("session_context", {}) or {}
        ctx_bits: List[str] = []
        session_type = context.get("session_type")
        if session_type:
            ctx_bits.append(f"type={session_type}")
        application_usage = summary.get("application_usage") or []
        if application_usage:
            top_app = application_usage[0]
            app_name = top_app.get("application")
            if app_name:
                ctx_bits.append(f"app={app_name}")
        notes = context.get("notes")
        if notes:
            ctx_bits.append(f"notes={notes}")
        ctx_display = ", ".join(ctx_bits) if ctx_bits else "no labels"
        self.last_session_var.set(
            f"Last session: {session_id} · {gestures} gestures · {duration:.1f}s · {ctx_display}"
        )

    def _on_close(self) -> None:
        if self._status_after_id is not None:
            self.root.after_cancel(self._status_after_id)
            self._status_after_id = None

        if self.collector.get_status().get("running"):
            def _shutdown() -> None:
                try:
                    self.collector.stop()
                finally:
                    self.root.after(0, self.root.destroy)

            threading.Thread(target=_shutdown, daemon=True).start()
        else:
            self.root.destroy()


__all__ = ["CollectorUI"]
