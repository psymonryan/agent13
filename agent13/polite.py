"""Polite mode — multi-agent coordination via a shared file lock.

When multiple agents target the same provider, polite mode makes them wait for
a shared lock before commencing a turn. The lock is held for the entire turn
and released in the ``finally`` of ``_process_item``.

The ``interval`` (``N``) is a pseudo-priority via poll interval: lower ``N``
polls more often and is statistically more likely to be the next to try
after the lock frees, so it wins the next free slot more often. ``N=0``
yields to the event loop on every iteration (most aggressive, ~always wins).

Algorithm (polling frequency):
    loop:
        if try_acquire_nonblocking(lock):   # we got it
            proceed                          # we win, hold until turn done
        else:
            sleep(N)                         # busy, wait and retry

All waits are explicit ``asyncio.sleep`` — priority is never handed to the OS
scheduler (where all waiters are equal). ``filelock`` is a real mutex, so
once acquired nobody can steal it; the polling frequency alone creates the
statistical priority. (An earlier "sleep-then-recheck courtesy" design was
rejected: a strict mutex cannot be contested mid-hold, so the courtesy sleep
was a no-op.)

Same-filesystem only — agents on different machines will not coordinate
(each sees its own absent lock and proceeds; no deadlock, just no
coordination).
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from filelock import BaseFileLock, Timeout

from agent13.config_paths import get_config_dir


def _sanitize_provider(provider: str) -> str:
    """Sanitize provider string for use in a lock filename.

    Non-alphanumeric characters become ``_`` so URL providers (e.g.
    ``http://localhost:8080/v1``) produce a stable, filesystem-safe name.
    Two agents on the same URL therefore coordinate on the same lock.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", provider).strip("_") or "default"


class PoliteLock:
    """Shared file lock for polite multi-agent coordination.

    The lock file lives at ``~/.agent13/polite_{provider}.lck``. Uses
    ``filelock`` for cross-platform, stale-lock handling via PID.

    Events are emitted via the supplied ``emit`` coroutine (matching
    ``Agent.emit``'s signature):
      - ``POLITE_WAITING`` (rate-capped at ~1/sec) with ``elapsed`` seconds float
      - ``POLITE_ACQUIRED`` once on win
    """

    # Rate cap for POLITE_WAITING events — smooth 1/sec UI ticking.
    _WAIT_EVENT_INTERVAL = 1.0

    def __init__(
        self,
        provider: str,
        interval: float,
        emit: Optional[Callable[..., Coroutine[Any, Any, None]]] = None,
    ) -> None:
        self._interval = max(0.0, float(interval))
        self._emit = emit
        self._lock: Optional[BaseFileLock] = None
        self._provider = provider
        self._safe_name = _sanitize_provider(provider)

        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        self._path = config_dir / f"polite_{self._safe_name}.lck"

    @property
    def path(self) -> Path:
        """Lock file path (for diagnostics/tests)."""
        return self._path

    @property
    def provider(self) -> str:
        """Original provider string (name or URL)."""
        return self._provider

    @property
    def interval(self) -> float:
        """Current poll interval (pseudo-priority N)."""
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        self._interval = max(0.0, float(value))

    def is_held(self) -> bool:
        """True if this PoliteLock currently owns the file lock."""
        return self._lock is not None and self._lock.is_locked

    async def acquire(self) -> None:
        """Acquire the lock via non-blocking polling.

        Emits ``POLITE_WAITING`` (rate-capped at ~1/sec) while waiting with
        ``elapsed`` seconds since waiting began, and ``POLITE_ACQUIRED`` once
        when the lock is won.

        Cancellation safe: if the awaiting task is cancelled mid-wait, any
        lock we hold is released before propagating.
        """
        from filelock import FileLock

        # Construct the lock lazily so the file is only touched on first use.
        lock = FileLock(str(self._path))
        wait_start: Optional[float] = None
        last_emit: float = 0.0

        try:
            while True:
                # Non-blocking acquire attempt — instant, never hands
                # priority to the OS scheduler.
                try:
                    lock.acquire(timeout=0)
                except Timeout:
                    # Busy — emit waiting event (rate-capped) and sleep.
                    now = time.monotonic()
                    if wait_start is None:
                        wait_start = now
                        last_emit = now
                        await self._emit_waiting(0.0)
                    elif now - last_emit >= self._WAIT_EVENT_INTERVAL:
                        last_emit = now
                        await self._emit_waiting(now - wait_start)
                    # asyncio.sleep(0) cooperatively yields without blocking —
                    # this is the N=0 "most aggressive" behaviour.
                    await asyncio.sleep(self._interval)
                    continue

                # We won the lock.
                self._lock = lock
                break

            await self._emit_acquired()
        except BaseException:
            # Cancellation or error — release if we grabbed it, then re-raise.
            self._release_quietly()
            raise

    def release(self) -> None:
        """Release the lock if held. Safe to call when not held."""
        self._release_quietly()

    def _release_quietly(self) -> None:
        if self._lock is not None:
            try:
                if self._lock.is_locked:
                    self._lock.release()
            except Exception:
                pass
            finally:
                self._lock = None

    async def _emit_waiting(self, elapsed: float) -> None:
        if self._emit is None:
            return
        from agent13.events import AgentEvent

        try:
            await self._emit(AgentEvent.POLITE_WAITING, {"elapsed": float(elapsed)})
        except Exception:
            pass

    async def _emit_acquired(self) -> None:
        if self._emit is None:
            return
        from agent13.events import AgentEvent

        try:
            await self._emit(AgentEvent.POLITE_ACQUIRED, {})
        except Exception:
            pass
