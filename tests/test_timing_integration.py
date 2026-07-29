"""Integration tests for TokenTimingTracker wiring in REPL and TUI.

These tests verify the full event pipeline through the real tracker,
matching the exact patterns used in agent13/repl.py and ui/tui.py.

Unlike test_timing.py (unit tests on the tracker class), these tests
exercise the tracker as it's actually wired: event sequences, display
string formatting, and state transitions.
"""

import time

from agent13.timing import TokenTimingTracker


# ── REPL wiring pattern ──────────────────────────────────────────────


def _simulate_repl_events(tracker, token_count=200, elapsed=5.0):
    """Simulate the REPL's STREAM_START → TOKEN → TOKEN_USAGE event sequence.

    Returns the TPSResult (or None) from compute_tps().
    """
    # STREAM_START → reset_stream (repl.py:389)
    tracker.reset_stream()

    # Simulate token stream with realistic timing
    start = time.monotonic()
    interval = elapsed / token_count

    for i in range(token_count):
        now = start + (i * interval)
        # First token check (repl.py:396)
        if tracker.is_first_token:
            pass  # would call display.start_response()
        tracker.record_token(now)

    # TOKEN_USAGE event (repl.py:440)
    data = {
        "prompt_tokens": 500,
        "completion_tokens": token_count,
        "total_tokens": 500 + token_count,
    }

    # compute_tps with first/last token times (matching TUI pattern)
    first = start
    last = start + elapsed
    return tracker.compute_tps(data, first_token_time=first, last_token_time=last)


class TestREPLWiringPattern:
    """Verify the REPL event pipeline produces correct TPS display."""

    def test_repl_pattern_produces_tps_result(self):
        """Full REPL event sequence yields a TPSResult."""
        tracker = TokenTimingTracker()
        result = _simulate_repl_events(tracker, token_count=200, elapsed=5.0)

        assert result is not None
        assert abs(result.tps - 40.0) < 1.0  # 200/5 = 40 tok/s

    def test_repl_pattern_short_response_no_tps(self):
        """REPL pattern with few tokens yields no TPS."""
        tracker = TokenTimingTracker()
        result = _simulate_repl_events(tracker, token_count=10, elapsed=0.5)

        assert result is None

    def test_repl_display_string_format(self):
        """TPSResult produces the display string format used in repl.py:442-446."""
        tracker = TokenTimingTracker()
        result = _simulate_repl_events(tracker, token_count=200, elapsed=5.0)

        assert result is not None
        # Format from repl.py:443-445
        display_str = (
            f"  ({result.tps:.0f} tok/s,"
            f" {result.completion_tokens} tokens,"
            f" {result.elapsed:.1f}s)"
        )
        assert "tok/s" in display_str
        assert "tokens" in display_str
        assert "s)" in display_str

    def test_repl_stale_tps_across_short_streams(self):
        """REPL preserves TPS from prior long stream when next is short."""
        tracker = TokenTimingTracker()

        # First stream: long enough
        _simulate_repl_events(tracker, token_count=200, elapsed=5.0)
        assert tracker.last_tps > 0

        # Second stream: too short
        result = _simulate_repl_events(tracker, token_count=5, elapsed=0.2)
        assert result is None

        # Stale TPS preserved (matches TUI behavior post-refactor)
        assert tracker.last_tps > 0

    def test_repl_turn_tracking(self):
        """REPL turn_start/turn_end cycle tracks processing time."""
        tracker = TokenTimingTracker()

        tracker.turn_start()
        time.sleep(0.05)
        tracker.turn_end()

        assert tracker._turn_count == 1
        assert tracker._total_processing_time > 0.0

    def test_repl_multiple_turns_accumulate(self):
        """Multiple REPL turns accumulate correctly."""
        tracker = TokenTimingTracker()

        for _ in range(3):
            tracker.turn_start()
            time.sleep(0.02)
            tracker.turn_end()

        assert tracker._turn_count == 3
        assert tracker._total_processing_time > 0.05


# ── TUI wiring pattern ───────────────────────────────────────────────


class _MinimalTUIShim:
    """Minimal TUI shim that mirrors the real TUI's tracker wiring.

    Matches ui/tui.py after the refactor:
    - self.tracker = TokenTimingTracker()
    - _reset_stream_timing delegates to tracker.reset_stream()
    - _update_token_usage delegates to tracker.compute_tps()
    - _update_status delegates to tracker.turn_start()/turn_end()
    """

    def __init__(self):
        self.tracker = TokenTimingTracker()
        self._last_tps = 0.0  # Textual reactive, kept in sync
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._elapsed_start_time = None

    def _reset_stream_timing(self):
        """Mirrors ui/tui.py post-refactor."""
        self.tracker.reset_stream()

    def _update_token_usage(self, data, first_token_time=None, last_token_time=None):
        """Mirrors ui/tui.py post-refactor."""
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)

        result = self.tracker.compute_tps(data, first_token_time, last_token_time)
        if result is not None:
            self._last_tps = result.tps

    def _update_status(self, status):
        """Mirrors ui/tui.py post-refactor."""
        if status == "processing":
            if self._elapsed_start_time is None:
                self._elapsed_start_time = time.time()
            self.tracker.turn_start()
        elif status == "idle":
            self.tracker.turn_end()
            self._elapsed_start_time = None


class TestTUIWiringPattern:
    """Verify the TUI event pipeline through a minimal shim."""

    def test_update_token_usage_syncs_last_tps(self):
        """TUI _update_token_usage sets _last_tps from tracker."""
        tui = _MinimalTUIShim()
        now = time.monotonic()

        tui.tracker.reset_stream()
        tui.tracker.record_token(now - 5.0)
        for _ in range(199):
            tui.tracker.record_token(now)

        tui._update_token_usage(
            {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            first_token_time=now - 5.0,
            last_token_time=now,
        )

        assert tui._last_tps > 0
        assert abs(tui._last_tps - 40.0) < 1.0

    def test_short_stream_does_not_clobber_last_tps(self):
        """TUI preserves _last_tps when stream is too short."""
        tui = _MinimalTUIShim()
        now = time.monotonic()

        # First: long stream
        tui.tracker.reset_stream()
        tui.tracker.record_token(now - 5.0)
        tui._update_token_usage(
            {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            first_token_time=now - 5.0,
            last_token_time=now,
        )
        saved_tps = tui._last_tps
        assert saved_tps > 0

        # Second: short stream
        tui.tracker.reset_stream()
        tui.tracker.record_token(now)
        tui._update_token_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            first_token_time=now,
            last_token_time=now + 0.1,
        )

        assert tui._last_tps == saved_tps  # preserved

    def test_reset_stream_clears_tracker(self):
        """TUI _reset_stream_timing resets the tracker."""
        tui = _MinimalTUIShim()
        now = time.monotonic()

        tui.tracker.record_token(now)
        tui.tracker.record_token(now + 1.0)
        assert not tui.tracker.is_first_token

        tui._reset_stream_timing()
        assert tui.tracker.is_first_token

    def test_update_status_tracks_turns(self):
        """TUI _update_status properly starts/ends turns."""
        tui = _MinimalTUIShim()

        tui._update_status("processing")
        time.sleep(0.01)
        assert tui.tracker.session_elapsed > 0

        time.sleep(0.02)
        tui._update_status("idle")

        assert tui.tracker._turn_count == 1
        assert tui.tracker._total_processing_time > 0

    def test_full_tui_response_cycle(self):
        """Simulate a complete TUI response cycle: reset → tokens → TPS → idle."""
        tui = _MinimalTUIShim()

        # Agent starts processing
        tui._update_status("processing")

        # Stream starts
        tui._reset_stream_timing()

        # Tokens arrive
        start = time.monotonic()
        for i in range(200):
            tui.tracker.record_token(start + i * 0.025)

        # Token usage arrives
        tui._update_token_usage(
            {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
            first_token_time=start,
            last_token_time=start + 5.0,
        )

        assert tui._last_tps > 0
        assert tui.completion_tokens == 200

        # Agent goes idle
        tui._update_status("idle")
        assert tui.tracker._turn_count == 1

    def test_token_fields_updated_from_data(self):
        """TUI syncs prompt/completion/total tokens from data dict."""
        tui = _MinimalTUIShim()

        data = {
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "total_tokens": 579,
        }
        tui._update_token_usage(data)

        assert tui.prompt_tokens == 123
        assert tui.completion_tokens == 456
        assert tui.total_tokens == 579


# ── Reflection event flow ────────────────────────────────────────────


class TestReflectionTrackerIntegration:
    """Verify the tracker handles reflection TOKEN_USAGE events correctly.

    These tests replace the MockTUI in test_reflection_token_usage.py
    with the real TokenTimingTracker.
    """

    def test_reflection_with_enough_tokens_calculates_tps(self):
        """Reflection stream with >50 tokens and >1.5s elapsed yields TPS."""
        tracker = TokenTimingTracker()
        now = time.monotonic()

        # Simulate reflection stream: 100 tokens over 3 seconds
        tracker.reset_stream()
        tracker.record_token(now - 3.0)
        for _ in range(99):
            tracker.record_token(now)

        data = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        result = tracker.compute_tps(
            data, first_token_time=now - 3.0, last_token_time=now
        )

        assert result is not None
        assert abs(result.tps - 33.3) < 1.0  # ~100/3
        assert result.completion_tokens == 100

    def test_reflection_short_stream_no_tps(self):
        """Short reflection (few tokens) yields no TPS."""
        tracker = TokenTimingTracker()
        now = time.monotonic()

        tracker.reset_stream()
        tracker.record_token(now)

        data = {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }
        result = tracker.compute_tps(
            data, first_token_time=now, last_token_time=now + 0.5
        )

        assert result is None
        # But token fields should still be set
        assert tracker.completion_tokens == 20
