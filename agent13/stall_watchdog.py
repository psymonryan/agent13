"""Stall watchdog - capture a Python stack when the main thread stops yielding.

A synchronous loop inside a coroutine monopolises the event loop: input,
signals and ``/save`` all queue behind it, so the process looks alive but
cannot be reached. Observed 2026-11-19: 80 minutes at 100% CPU, and neither
py-spy (needs root) nor lldb (needs the libpython offsets) was available at
the moment it mattered. This samples the main thread from a daemon thread
and, when one call path has been burning CPU for ``stall_seconds``, writes
the stack to the debug log - the evidence that was missing.

CPU time, not wall time, is the signal: an idle ``await`` also holds one
stack shape for minutes, and reporting those would bury the real stall.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback


def _stack_key(frame) -> tuple:
    """The call path, ignoring line numbers - a loop moves between lines."""
    key = []
    while frame is not None:
        key.append((frame.f_code.co_filename, frame.f_code.co_name))
        frame = frame.f_back
    return tuple(key)


def _log_to_debug(text: str, held: float) -> None:
    try:
        from agent13.debug_log import log_event

        log_event(
            "stall_watchdog",
            {"held_seconds": round(held, 1), "stack": text},
            level="ERROR",
        )
    except Exception:
        pass  # A diagnostic must never take the app down.


class StallWatchdog:
    """Report a main-thread stall to *report* (a callable taking text, held).

    ``start()``/``stop()`` are idempotent; the thread is a daemon, so it
    never holds the process open.
    """

    def __init__(
        self,
        stall_seconds: float = 10.0,
        poll_seconds: float = 1.0,
        report=None,
    ) -> None:
        self.stall_seconds = stall_seconds
        self.poll_seconds = poll_seconds
        self._report = report or _log_to_debug
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="stall-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        main = threading.main_thread().ident
        key: tuple | None = None
        since = 0.0
        cpu_at = 0.0
        reported = False
        while not self._stop.wait(self.poll_seconds):
            frame = sys._current_frames().get(main)
            if frame is None:
                continue
            now = time.monotonic()
            cpu = time.process_time()
            current = _stack_key(frame)
            if current != key:
                key, since, cpu_at, reported = current, now, cpu, False
                continue
            if reported or now - since < self.stall_seconds:
                continue
            # Same call path for long enough - but only a stall if it also
            # burned the CPU. An idle await holds one shape indefinitely.
            if cpu - cpu_at < 0.5 * (now - since):
                continue
            reported = True
            try:
                # format_stack returns a LIST of lines - join it, or the log
                # entry is a list rather than text.
                self._report("".join(traceback.format_stack(frame)), now - since)
            except Exception:
                pass  # A diagnostic must never take the app down.
