"""Unit tests for polite mode (agent13.polite).

Tests the PoliteLock primitive in isolation: path building, sanitization,
acquire/release, interval handling, and the polling algorithm. The
integration test (two agents coordinating) lives in test_polite_integration.py.
"""

import asyncio

import pytest

from agent13.events import AgentEvent
from agent13.polite import PoliteLock, _sanitize_provider


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("studiomlx", "studiomlx"),
        ("studio-mlx", "studio_mlx"),
        ("http://localhost:8080/v1", "http_localhost_8080_v1"),
        ("https://api.example.com:443/v1/", "https_api_example_com_443_v1"),
        ("", "default"),
        ("---", "default"),  # all non-alphanumeric -> stripped to empty -> default
        ("a/b/c", "a_b_c"),
    ],
)
def test_sanitize_provider(raw, expected):
    assert _sanitize_provider(raw) == expected


def test_sanitize_provider_idempotent():
    """Sanitizing an already-sanitized name is a no-op (stable filename)."""
    once = _sanitize_provider("http://localhost:8080")
    twice = _sanitize_provider(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Path building
# ---------------------------------------------------------------------------


def test_lock_path_uses_config_dir(tmp_path, monkeypatch):
    """Lock file lives in ~/.agent13/ (config dir) with sanitized provider."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="studiomlx", interval=0.1)
    assert lock.path.parent == tmp_path
    assert lock.path.name == "polite_studiomlx.lck"


def test_lock_path_url_provider(tmp_path, monkeypatch):
    """URL providers produce a sanitized filename, not raw URL."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="http://localhost:8080/v1", interval=0.1)
    assert lock.path.name == "polite_http_localhost_8080_v1.lck"


def test_lock_path_creates_config_dir(tmp_path, monkeypatch):
    """Config dir is created if it doesn't exist."""
    nested = tmp_path / "nested" / "config"
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(nested))
    lock = PoliteLock(provider="test", interval=0.1)
    assert nested.exists()
    assert lock.path.parent == nested


def test_two_locks_same_provider_same_path(tmp_path, monkeypatch):
    """Two locks with the same provider share the same path."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    l1 = PoliteLock(provider="shared", interval=0.1)
    l2 = PoliteLock(provider="shared", interval=0.2)
    assert l1.path == l2.path


def test_two_locks_different_provider_different_path(tmp_path, monkeypatch):
    """Different providers -> different paths."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    l1 = PoliteLock(provider="alpha", interval=0.1)
    l2 = PoliteLock(provider="beta", interval=0.1)
    assert l1.path != l2.path


# ---------------------------------------------------------------------------
# Interval / properties
# ---------------------------------------------------------------------------


def test_interval_setter_clamps_negative(tmp_path, monkeypatch):
    """Negative intervals are clamped to 0."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="test", interval=-1.0)
    assert lock.interval == 0.0


def test_interval_setter_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="test", interval=0.5)
    assert lock.interval == 0.5
    lock.interval = 0.1
    assert lock.interval == 0.1


def test_is_held_false_before_acquire(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="test", interval=0.1)
    assert not lock.is_held()


def test_provider_property_preserves_raw(tmp_path, monkeypatch):
    """provider property returns the original (unsanitized) string."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="http://localhost:8080", interval=0.1)
    assert lock.provider == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Acquire / release
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Isolated config dir for each test."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    return tmp_path


async def test_acquire_release_basic(config_dir):
    """Acquire then release; is_held tracks correctly."""
    lock = PoliteLock(provider="basic", interval=0.01)
    await lock.acquire()
    assert lock.is_held()
    lock.release()
    assert not lock.is_held()


async def test_acquire_emits_acquired_event(config_dir):
    """POLITE_ACQUIRED fires once on successful acquire."""
    events = []

    async def emit(evt, data):
        events.append((evt, data))

    lock = PoliteLock(provider="basic", interval=0.01, emit=emit)
    await lock.acquire()
    acquired = [e for e in events if e[0] is AgentEvent.POLITE_ACQUIRED]
    assert len(acquired) == 1
    assert acquired[0][1] == {}
    lock.release()


async def test_acquire_no_emit_when_callback_none(config_dir):
    """No emit callback -> no error, still acquires."""
    lock = PoliteLock(provider="basic", interval=0.01, emit=None)
    await lock.acquire()
    assert lock.is_held()
    lock.release()


async def test_release_when_not_held_is_noop(config_dir):
    """release() on an un-acquired lock doesn't raise."""
    lock = PoliteLock(provider="basic", interval=0.01)
    lock.release()  # should not raise
    lock.release()  # idempotent


async def test_release_after_release_is_noop(config_dir):
    lock = PoliteLock(provider="basic", interval=0.01)
    await lock.acquire()
    lock.release()
    lock.release()  # second release
    assert not lock.is_held()


# ---------------------------------------------------------------------------
# Polling algorithm — waiting when busy
# ---------------------------------------------------------------------------


async def test_busy_lock_emits_waiting_events(config_dir):
    """When the lock is held by another, waiting events fire (rate-capped)."""
    events = []

    async def emit(evt, data):
        events.append((evt, data))

    holder = PoliteLock(provider="busy", interval=0.01)
    waiter = PoliteLock(provider="busy", interval=0.05, emit=emit)

    await holder.acquire()
    try:
        task = asyncio.create_task(waiter.acquire())
        # Let the waiter poll a few times
        await asyncio.sleep(0.2)
        waiting = [e for e in events if e[0] is AgentEvent.POLITE_WAITING]
        assert len(waiting) >= 1, "should have waiting events while holder has lock"
        # elapsed should be non-negative
        for _, data in waiting:
            assert "elapsed" in data
            assert data["elapsed"] >= 0.0
        assert not waiter.is_held()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        holder.release()


async def test_wait_event_rate_capped(config_dir):
    """Waiting events are rate-capped at ~1/sec (not one per poll)."""
    events = []

    async def emit(evt, data):
        events.append((evt, data))

    holder = PoliteLock(provider="capped", interval=0.01)
    # Very short interval on waiter so it polls rapidly
    waiter = PoliteLock(provider="capped", interval=0.02, emit=emit)

    await holder.acquire()
    try:
        task = asyncio.create_task(waiter.acquire())
        # Wait 1.5 seconds — with 1/sec rate cap, expect ~1-2 events
        await asyncio.sleep(1.5)
        waiting = [e for e in events if e[0] is AgentEvent.POLITE_WAITING]
        # Should be at most 2 (first immediate + ~1 after 1 sec)
        assert len(waiting) <= 3, f"rate cap failed: {len(waiting)} events in 1.5s"
        assert len(waiting) >= 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        holder.release()


async def test_waiter_acquires_after_holder_releases(config_dir):
    """Waiter proceeds once the holder releases."""
    holder = PoliteLock(provider="handoff", interval=0.01)
    waiter = PoliteLock(provider="handoff", interval=0.05)

    await holder.acquire()
    task = asyncio.create_task(waiter.acquire())
    await asyncio.sleep(0.15)  # waiter is waiting
    assert not waiter.is_held()

    holder.release()
    await asyncio.wait_for(task, timeout=2.0)
    assert waiter.is_held()
    waiter.release()


async def test_elapsed_increases_over_time(config_dir):
    """The elapsed value in waiting events grows as time passes."""
    events = []

    async def emit(evt, data):
        events.append((evt, data))

    holder = PoliteLock(provider="elapsed", interval=0.01)
    waiter = PoliteLock(provider="elapsed", interval=0.3, emit=emit)

    await holder.acquire()
    try:
        task = asyncio.create_task(waiter.acquire())
        # Wait long enough to get at least 2 rate-capped events (1.5s)
        await asyncio.sleep(1.6)
        waiting = [e for e in events if e[0] is AgentEvent.POLITE_WAITING]
        if len(waiting) >= 2:
            assert waiting[-1][1]["elapsed"] > waiting[0][1]["elapsed"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# Cancellation safety
# ---------------------------------------------------------------------------


async def test_cancellation_releases_lock(config_dir):
    """If the waiting task is cancelled, no lock is left held."""
    holder = PoliteLock(provider="cancel", interval=0.01)
    waiter = PoliteLock(provider="cancel", interval=0.05)

    await holder.acquire()
    task = asyncio.create_task(waiter.acquire())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert not waiter.is_held()
    holder.release()


async def test_cancellation_while_holding_releases(config_dir):
    """If cancelled after acquiring (edge case), lock is released."""
    # This is hard to trigger naturally since acquire returns immediately
    # on success. We simulate by acquiring then cancelling a noop task.
    lock = PoliteLock(provider="cancel-held", interval=0.01)
    await lock.acquire()
    assert lock.is_held()

    # Simulate the cancellation path in acquire's except block
    lock._release_quietly()
    assert not lock.is_held()


# ---------------------------------------------------------------------------
# N=0 edge case
# ---------------------------------------------------------------------------


async def test_n_zero_acquires_quickly(config_dir):
    """N=0 (most aggressive) acquires immediately when free."""
    lock = PoliteLock(provider="zero", interval=0.0)
    # Should not hang or busy-loop excessively
    await asyncio.wait_for(lock.acquire(), timeout=1.0)
    assert lock.is_held()
    lock.release()
