"""Tests for agent13.stall_watchdog.

The watchdog exists to turn an unreachable hang into evidence: on 2026-11-19 a
synchronous loop inside a coroutine held the event loop at 100% CPU for 80
minutes, and nothing could report it - SIGINT, input and /save all queue
behind the loop.
"""

import time

from agent13.stall_watchdog import StallWatchdog, _stack_key


def _watchdog(stall_seconds, reports):
    return StallWatchdog(
        stall_seconds=stall_seconds,
        poll_seconds=0.05,
        report=lambda text, held: reports.append((text, held)),
    )


class TestStallWatchdog:
    def test_reports_a_spinning_loop(self):
        """A tight CPU loop in one call path is the case that matters."""
        reports = []
        wd = _watchdog(0.3, reports)
        wd.start()
        try:
            end = time.monotonic() + 2.0
            spin = 0
            while not reports and time.monotonic() < end:
                spin += 1
        finally:
            wd.stop()

        assert reports, "a 100%-CPU loop was never reported"
        text, held = reports[0]
        assert "test_reports_a_spinning_loop" in text
        assert held >= 0.3

    def test_ignores_an_idle_wait(self):
        """An idle await holds one stack shape for minutes - not a stall.

        Reporting those would bury the real one, which is why the trigger is
        CPU time rather than wall time.
        """
        reports = []
        wd = _watchdog(0.2, reports)
        wd.start()
        try:
            time.sleep(1.0)
        finally:
            wd.stop()

        assert reports == []

    def test_reports_once_per_stall(self):
        """One episode, one report - not one per poll."""
        reports = []
        wd = _watchdog(0.2, reports)
        wd.start()
        try:
            end = time.monotonic() + 1.2
            spin = 0
            while time.monotonic() < end:
                spin += 1
        finally:
            wd.stop()

        assert len(reports) == 1

    def test_start_and_stop_are_idempotent(self):
        wd = _watchdog(0.2, [])
        wd.start()
        wd.start()
        wd.stop()
        wd.stop()

    def test_a_failing_reporter_does_not_raise(self):
        """A diagnostic must never take the app down."""

        def boom(text, held):
            raise RuntimeError("reporter exploded")

        wd = StallWatchdog(stall_seconds=0.2, poll_seconds=0.05, report=boom)
        wd.start()
        try:
            end = time.monotonic() + 1.0
            spin = 0
            while time.monotonic() < end:
                spin += 1
        finally:
            wd.stop()


class TestStackKey:
    def test_key_ignores_line_numbers(self):
        """A loop moves between lines; the call path is what identifies it."""

        def sample():
            return _stack_key(__import__("sys")._getframe())

        a = sample()
        b = sample()
        assert a == b
        assert any(name == "sample" for _, name in a)
