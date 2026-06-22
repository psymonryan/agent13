"""
Characterization tests for TokenTimingTracker.

These were originally written against TUI's _update_token_usage via a minimal
shim. After extracting agent13/timing.py, they now target TokenTimingTracker
directly, preserving all threshold and behavioral semantics.

Key behaviors characterized:
  - MIN_TOKENS = 50
  - MIN_ELAPSED = 1.5s
  - MIN_ELAPSED_CALC = 0.1s (sanity floor)
  - Stale TPS persistence: short streams don't clobber last_tps
  - Race-condition aware: first/last token times passed as args to compute_tps
  - turn_start/turn_end lifecycle with turn_count and total_processing_time
  - session_elapsed via turn_start
"""

import time

import pytest

from agent13.timing import TokenTimingTracker, TPSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(**overrides):
    """Create a fresh tracker with optional attribute overrides."""
    tracker = TokenTimingTracker()
    for k, v in overrides.items():
        setattr(tracker, k, v)
    return tracker


def _data(completion_tokens=100, prompt_tokens=50, total_tokens=150):
    """Build the data dict that compute_tps expects."""
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Threshold gating — MIN_TOKENS=50, MIN_ELAPSED=1.5
# ---------------------------------------------------------------------------


class TestThresholdGating:
    """TPS should only be returned when both MIN_TOKENS and MIN_ELAPSED are met."""

    def test_no_tokens_no_tps(self):
        """Zero completion_tokens → None, last_tps unchanged."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=0), now - 10, now)
        assert result is None
        assert tracker.last_tps is None

    def test_below_min_tokens_no_tps(self):
        """Under 50 completion tokens → suppressed even with enough time."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=49), now - 10, now)
        assert result is None
        assert tracker.last_tps is None

    def test_below_min_elapsed_no_tps(self):
        """Under 1.5s elapsed → suppressed even with enough tokens."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=100), now - 1.0, now)
        assert result is None
        assert tracker.last_tps is None

    def test_both_thresholds_met_returns_result(self):
        """≥50 tokens AND ≥1.5s → TPSResult returned."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=200), now - 10.0, now)
        assert isinstance(result, TPSResult)
        assert 15 < result.tps < 25

    def test_exact_boundary_min_tokens(self):
        """Exactly 50 tokens with enough time → TPS calculated."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=50), now - 5.0, now)
        assert isinstance(result, TPSResult)
        assert 9 < result.tps < 11

    def test_exact_boundary_min_elapsed(self):
        """Exactly 1.5s elapsed with enough tokens → TPS calculated."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(_data(completion_tokens=60), now - 1.5, now)
        assert isinstance(result, TPSResult)
        assert 38 < result.tps < 42


# ---------------------------------------------------------------------------
# Sanity floor — MIN_ELAPSED_CALC=0.1
# ---------------------------------------------------------------------------


class TestSanityFloor:
    """MIN_ELAPSED_CALC = 0.1 prevents division by near-zero elapsed."""

    def test_extremely_short_elapsed_no_tps(self):
        """Even with many tokens, near-zero elapsed → suppressed."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=500), now - 0.001, now
        )
        assert result is None

    def test_elapsed_below_floor_no_tps(self):
        """Elapsed 0.05s < 0.1 floor → suppressed."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=200), now - 0.05, now
        )
        assert result is None

    def test_elapsed_at_floor_still_suppressed_by_min_elapsed(self):
        """Elapsed 0.1s passes sanity floor but fails MIN_ELAPSED (1.5s)."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=200), now - 0.1, now
        )
        assert result is None

    def test_elapsed_above_both_thresholds_calculates(self):
        """Elapsed ≥ 1.5s with enough tokens → TPS calculated."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=100), now - 5.0, now
        )
        assert isinstance(result, TPSResult)
        assert 18 < result.tps < 22


# ---------------------------------------------------------------------------
# Stale TPS persistence
# ---------------------------------------------------------------------------


class TestStaleTPS:
    """Short streams should NOT clobber a previously-set last_tps."""

    def test_short_stream_preserves_last_tps(self):
        """If thresholds aren't met, last_tps stays at its prior value."""
        tracker = _make_tracker()
        tracker._last_tps = 42.0
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=10), now - 2.0, now
        )
        assert result is None
        assert tracker.last_tps == 42.0

    def test_long_stream_updates_last_tps(self):
        """When thresholds are met, last_tps gets the new value."""
        tracker = _make_tracker()
        tracker._last_tps = 10.0
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=200), now - 10.0, now
        )
        assert isinstance(result, TPSResult)
        assert abs(tracker.last_tps - 20.0) < 1.0

    def test_none_last_tps_stays_none_on_short_stream(self):
        """Initial last_tps=None remains None when stream is too short."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=5), now - 0.5, now
        )
        assert result is None
        assert tracker.last_tps is None

    def test_below_min_elapsed_preserves_stale(self):
        """Response under 1.5s keeps prior last_tps."""
        tracker = _make_tracker()
        tracker._last_tps = 35.0
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=80), now - 1.0, now
        )
        assert result is None
        assert tracker.last_tps == 35.0

    def test_below_min_tokens_preserves_stale(self):
        """Response with <50 tokens keeps prior last_tps."""
        tracker = _make_tracker()
        tracker._last_tps = 25.0
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=30), now - 10.0, now
        )
        assert result is None
        assert tracker.last_tps == 25.0


# ---------------------------------------------------------------------------
# Race-condition aware timestamps
# ---------------------------------------------------------------------------


class TestRaceConditionArgs:
    """first/last token times passed as args override internal state."""

    def test_uses_arg_times_over_internal_times(self):
        """When args are provided, they override internal _first_token_time."""
        tracker = _make_tracker(
            _first_token_time=1000.0,
            _last_token_time=1001.0,
        )
        now = time.time()
        # Args say 10s elapsed → valid; internal says 1s → would fail MIN_ELAPSED
        result = tracker.compute_tps(
            _data(completion_tokens=200), now - 10.0, now
        )
        assert isinstance(result, TPSResult)
        assert 15 < result.tps < 25

    def test_falls_back_to_internal_when_args_none(self):
        """When args are None, falls back to internal _first_token_time."""
        tracker = _make_tracker()
        now = time.time()
        tracker._first_token_time = now - 10.0
        tracker._last_token_time = now
        result = tracker.compute_tps(
            _data(completion_tokens=200), None, None
        )
        assert isinstance(result, TPSResult)
        assert 15 < result.tps < 25

    def test_none_both_sides_no_tps(self):
        """If both args and internal times are None, no TPS calculation."""
        tracker = _make_tracker()
        result = tracker.compute_tps(
            _data(completion_tokens=200), None, None
        )
        assert result is None
        assert tracker.last_tps is None


# ---------------------------------------------------------------------------
# TPS computation
# ---------------------------------------------------------------------------


class TestTPSComputation:
    """TPS = completion_tokens / elapsed_seconds."""

    def test_tps_calculation_accuracy(self):
        """Known input → expected TPS value."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=100), now - 5.0, now
        )
        assert isinstance(result, TPSResult)
        assert abs(result.tps - 20.0) < 0.5

    def test_tps_with_fractional_result(self):
        """Non-round division should still work."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=75), now - 3.0, now
        )
        assert isinstance(result, TPSResult)
        assert abs(result.tps - 25.0) < 0.5

    def test_high_tps(self):
        """Fast generation: 500 tokens in 2s = 250 tok/s."""
        tracker = _make_tracker()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=500), now - 2.0, now
        )
        assert isinstance(result, TPSResult)
        assert abs(result.tps - 250.0) < 5.0

    def test_sets_token_fields_from_data(self):
        """completion_tokens, prompt_tokens, total_tokens are set from data."""
        tracker = _make_tracker()
        now = time.time()
        tracker.compute_tps(
            _data(completion_tokens=100, prompt_tokens=200, total_tokens=300),
            now - 5.0,
            now,
        )
        assert tracker.completion_tokens == 100
        assert tracker.prompt_tokens == 200
        assert tracker.total_tokens == 300

    def test_result_carries_turn_context(self):
        """TPSResult includes turn_count and total_processing_time."""
        tracker = _make_tracker()
        tracker.turn_start()
        time.sleep(0.02)
        tracker.turn_end()
        now = time.time()
        result = tracker.compute_tps(
            _data(completion_tokens=200), now - 5.0, now
        )
        assert isinstance(result, TPSResult)
        assert result.turn_count == 1
        assert result.total_processing_time > 0.01


# ---------------------------------------------------------------------------
# Stream lifecycle
# ---------------------------------------------------------------------------


class TestStreamLifecycle:
    """reset_stream and record_token manage per-stream state."""

    def test_reset_stream_clears_times(self):
        """After reset, is_first_token is True."""
        tracker = _make_tracker()
        tracker._first_token_time = 100.0
        tracker._last_token_time = 200.0
        tracker.reset_stream()
        assert tracker.is_first_token is True

    def test_reset_stream_clears_token_count(self):
        """After reset, _token_count is 0."""
        tracker = _make_tracker()
        tracker.record_token(100.0)
        tracker.record_token(101.0)
        assert tracker._token_count == 2
        tracker.reset_stream()
        assert tracker._token_count == 0

    def test_record_token_increments_count(self):
        """Each record_token call increments _token_count."""
        tracker = _make_tracker()
        assert tracker._token_count == 0
        tracker.record_token(100.0)
        assert tracker._token_count == 1
        tracker.record_token(100.5)
        assert tracker._token_count == 2
        tracker.record_token(101.0)
        assert tracker._token_count == 3

    def test_record_token_captures_first_time(self):
        """First record_token sets _first_token_time."""
        tracker = _make_tracker()
        now = time.time()
        tracker.record_token(now)
        assert tracker._first_token_time == now
        assert tracker._last_token_time == now
        assert tracker.is_first_token is False

    def test_record_token_updates_last_time_only(self):
        """Subsequent record_token calls update _last_token_time only."""
        tracker = _make_tracker()
        t1 = 1000.0
        t2 = 1001.0
        tracker.record_token(t1)
        tracker.record_token(t2)
        assert tracker._first_token_time == t1
        assert tracker._last_token_time == t2

    def test_is_first_token_after_reset(self):
        """is_first_token is True immediately after reset."""
        tracker = _make_tracker()
        tracker.record_token(100.0)
        assert tracker.is_first_token is False
        tracker.reset_stream()
        assert tracker.is_first_token is True


# ---------------------------------------------------------------------------
# Turn timing
# ---------------------------------------------------------------------------


class TestTurnTiming:
    """turn_start/turn_end lifecycle: turn_count and total_processing_time."""

    def test_turn_start_and_end(self):
        """After turn_start → turn_end, turn_count increments and time accumulates."""
        tracker = _make_tracker()
        tracker.turn_start()
        time.sleep(0.05)
        tracker.turn_end()

        assert tracker._turn_count == 1
        assert tracker._total_processing_time >= 0.04

    def test_turn_end_noop_when_not_started(self):
        """If turn_start not called, turn_end should not increment."""
        tracker = _make_tracker()
        tracker.turn_end()

        assert tracker._turn_count == 0
        assert tracker._total_processing_time == 0.0

    def test_multiple_turns_accumulate(self):
        """Multiple turn_start/turn_end cycles accumulate correctly."""
        tracker = _make_tracker()

        for _ in range(3):
            tracker.turn_start()
            time.sleep(0.05)
            tracker.turn_end()

        assert tracker._turn_count == 3
        assert tracker._total_processing_time > 0.1


# ---------------------------------------------------------------------------
# Session elapsed
# ---------------------------------------------------------------------------


class TestSessionElapsed:
    """Session elapsed time (from first turn_start)."""

    def test_elapsed_zero_before_first_turn(self):
        """session_elapsed returns 0.0 when no turn has started."""
        tracker = _make_tracker()
        assert tracker.session_elapsed == 0.0

    def test_elapsed_increases_after_turn_start(self):
        """Elapsed should grow as time passes after turn_start."""
        tracker = _make_tracker()
        tracker.turn_start()

        elapsed1 = tracker.session_elapsed
        time.sleep(0.05)
        elapsed2 = tracker.session_elapsed

        assert elapsed2 > elapsed1

    def test_elapsed_resets_on_new_turn_start(self):
        """A new turn_start resets elapsed measurement."""
        tracker = _make_tracker()
        tracker.turn_start()
        time.sleep(0.1)

        tracker.turn_start()  # new turn
        elapsed = tracker.session_elapsed
        assert elapsed < 0.05