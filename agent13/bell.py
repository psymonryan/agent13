"""Bell manager shared between TUI and REPL.

Encapsulates the threshold-timer + ring logic so both UI modes use the
same behaviour.  The only UI-specific detail is the *fallback ring*:
the TUI calls ``self.bell()`` (Textual), the REPL writes ``\\a`` to stdout.
"""

import asyncio
import shutil
import subprocess
import sys
from typing import Callable, Optional


class BellManager:
    """Manages bell state, threshold timer, and ringing.

    Args:
        threshold: Seconds before bell on long turns. 0 = always ring on
            idle.  ``"off"`` disables.  ``None`` uses config default (0).
        enabled: Whether bell is active (from config).
        command: External command to run instead of terminal bell.
        fallback_ring: Called when no command is set (terminal bell).
            TUI passes ``self.bell``; REPL passes a ``\\a``-writer.
    """

    def __init__(
        self,
        threshold: float | str | None = None,
        enabled: bool = True,
        command: str = "",
        fallback_ring: Optional[Callable[[], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.threshold: float = 0.0
        self._armed = False
        self._task: asyncio.Task | None = None
        self._command = command
        self._fallback_ring = fallback_ring

        if threshold is not None:
            if isinstance(threshold, str) and threshold.lower() == "off":
                self.enabled = False
            else:
                try:
                    self.threshold = float(threshold)
                except (ValueError, TypeError):
                    pass  # Fall back to default

        if self._command and not self.validate_command(self._command):
            print(
                f"Warning: bell-command '{self._command}' is not executable, "
                "falling back to terminal bell",
                file=sys.stderr,
            )
            self._command = ""

    # ── Properties (read-only views for status display) ──────────────

    @property
    def command(self) -> str:
        return self._command

    # ── Validation ───────────────────────────────────────────────────

    @staticmethod
    def validate_command(command: str) -> bool:
        """Check that the first token of *command* is executable via PATH."""
        first_token = command.strip().split()[0] if command.strip() else ""
        if not first_token:
            return False
        return shutil.which(first_token) is not None

    # ── Turn lifecycle hooks ─────────────────────────────────────────

    def on_turn_start(self) -> None:
        """Arm the bell and start the threshold timer (if threshold > 0)."""
        self._armed = True
        self._start_timer()

    def on_turn_end(self) -> None:
        """Cancel pending timer; ring immediately if threshold == 0."""
        self._cancel_timer()
        if self._armed and self.enabled and self.threshold == 0:
            self.ring()
        self._armed = False

    def on_pause(self) -> None:
        """Cancel timer and disarm so resume-idle doesn't ring."""
        self._cancel_timer()
        self._armed = False

    # ── Timer ────────────────────────────────────────────────────────

    def _start_timer(self) -> None:
        """Schedule a single bell after *threshold* seconds.

        When threshold is 0 (always ring), no timer is needed — the bell
        rings immediately on idle in ``on_turn_end``.
        """
        if not self.enabled or self.threshold <= 0:
            return
        if self._task is not None:
            return
        secs = self.threshold

        async def _bell_after_delay():
            try:
                await asyncio.sleep(secs)
                self.ring()
            except asyncio.CancelledError:
                pass

        self._task = asyncio.create_task(_bell_after_delay())

    def _cancel_timer(self) -> None:
        """Cancel the pending bell timer."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ── Ring ─────────────────────────────────────────────────────────

    def ring(self) -> None:
        """Ring via external command (if set) or fallback terminal bell."""
        self._task = None
        if self._command:
            try:
                subprocess.Popen(
                    self._command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass  # Fire-and-forget; ignore failures
        elif self._fallback_ring:
            try:
                self._fallback_ring()
            except Exception:
                pass  # Ignore errors during shutdown

    # ── Teardown ─────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Cancel any pending timer (for shutdown / teardown)."""
        self._cancel_timer()

    # ── Runtime setters (used by /bell and /bell-command slash cmds) ─

    def set_threshold(self, val: float) -> None:
        """Set threshold and enable bell (val >= 0)."""
        self.enabled = True
        self.threshold = val
        if val == 0:
            self._cancel_timer()

    def disable(self) -> None:
        """Disable bell and cancel any pending timer."""
        self.enabled = False
        self._cancel_timer()

    def set_command(self, command: str) -> bool:
        """Set external bell command. Returns True if valid."""
        if not self.validate_command(command):
            return False
        self._command = command
        return True

    def clear_command(self) -> None:
        """Clear external command, revert to terminal bell."""
        self._command = ""

    # ── Status helpers ───────────────────────────────────────────────

    def status_text(self) -> str:
        """One-line status for /status display."""
        if not self.enabled:
            return "off"
        if self.threshold == 0:
            return "always"
        return f"{self.threshold:.0f}s"