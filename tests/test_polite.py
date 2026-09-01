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


def test_lock_path_uses_locks_dir(tmp_path, monkeypatch):
    """Lock file lives in ~/.agent13/locks/ with sanitized provider."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="studiomlx", interval=0.1)
    assert lock.path.parent == tmp_path / "locks"
    assert lock.path.name == "polite_studiomlx.lck"


def test_lock_path_url_provider(tmp_path, monkeypatch):
    """URL providers produce a sanitized filename, not raw URL."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="http://localhost:8080/v1", interval=0.1)
    assert lock.path.name == "polite_http_localhost_8080_v1.lck"


def test_lock_path_creates_locks_dir(tmp_path, monkeypatch):
    """Config dir and locks/ subdir are created if they don't exist."""
    nested = tmp_path / "nested" / "config"
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(nested))
    lock = PoliteLock(provider="test", interval=0.1)
    assert (nested / "locks").exists()
    assert lock.path.parent == nested / "locks"


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


def test_lock_path_with_model(tmp_path, monkeypatch):
    """A model name is appended to the lock filename."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="studiomlx", interval=0.1, model="GLM-5.1")
    assert lock.path.name == "polite_studiomlx_GLM_5_1.lck"


def test_lock_path_model_none_is_legacy(tmp_path, monkeypatch):
    """model=None produces the legacy provider-only filename."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="studiomlx", interval=0.1)
    assert lock.path.name == "polite_studiomlx.lck"


def test_lock_path_model_name_sanitized(tmp_path, monkeypatch):
    """A model-name fallback is sanitized into the filename."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    lock = PoliteLock(provider="x", interval=0.1, model="Qwen/3.8-27B")
    assert lock.path.name == "polite_x_Qwen_3_8_27B.lck"


def test_two_locks_same_provider_different_model(tmp_path, monkeypatch):
    """Same provider, different models -> different paths (run in parallel)."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    l1 = PoliteLock(provider="shared", interval=0.1, model="GLM-5.1")
    l2 = PoliteLock(provider="shared", interval=0.1, model="GLM-5.2")
    assert l1.path != l2.path


def test_two_locks_same_provider_same_model(tmp_path, monkeypatch):
    """Same provider and model -> same path (coordinate)."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    l1 = PoliteLock(provider="shared", interval=0.1, model="GLM-5.1")
    l2 = PoliteLock(provider="shared", interval=0.2, model="GLM-5.1")
    assert l1.path == l2.path


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


def test_model_property_returns_value(tmp_path, monkeypatch):
    """model property returns the value passed in (or None)."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    assert PoliteLock(provider="x", interval=0.1, model="GLM-5.1").model == "GLM-5.1"
    assert PoliteLock(provider="x", interval=0.1).model is None


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
    # ---------------------------------------------------------------------------


# set_client re-keys the polite lock on provider switch
# ---------------------------------------------------------------------------


class _MockClient:
    """Minimal mock client with a settable base_url."""

    def __init__(self, base_url: str):
        self.base_url = base_url


def test_set_client_rekeys_polite_lock(config_dir):
    """set_client with a new base_url rebuilds the polite lock."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://old.local/v1"), model="m")
    agent.set_polite(interval=2.5)
    old_path = agent.polite_lock.path
    assert agent.polite_lock.interval == 2.5

    agent.set_client(_MockClient("http://new.local/v1"))

    assert agent.polite_lock is not None
    assert agent.polite_lock.provider == "http://new.local/v1"
    assert agent.polite_lock.path != old_path
    assert agent.polite_lock.interval == 2.5  # preserved


def test_set_client_same_base_url_no_rekey(config_dir):
    """set_client with the same base_url leaves the lock unchanged."""
    from agent13.core import Agent

    client = _MockClient("http://same.local/v1")
    agent = Agent(client, model="m")
    agent.set_polite(interval=1.0)
    lock_obj = agent.polite_lock

    agent.set_client(_MockClient("http://same.local/v1"))

    assert agent.polite_lock is lock_obj  # same object, not rebuilt


def test_set_client_no_polite_no_crash(config_dir):
    """set_client is safe when polite mode is not enabled."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://old.local/v1"), model="m")
    assert agent.polite_lock is None

    agent.set_client(_MockClient("http://new.local/v1"))
    assert agent.polite_lock is None


async def test_set_client_skips_rekey_when_held(config_dir):
    """Lock held mid-turn is not swapped (avoid orphaning the hold)."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://old.local/v1"), model="m")
    agent.set_polite(interval=0.01)
    await agent.polite_lock.acquire()
    assert agent.polite_lock.is_held()
    held_obj = agent.polite_lock

    agent.set_client(_MockClient("http://new.local/v1"))

    # Lock unchanged — still the old one, still held
    assert agent.polite_lock is held_obj
    assert agent.polite_lock.is_held()
    agent.polite_lock.release()


# ---------------------------------------------------------------------------
# Per-model keying (base name / thinking-suffix strip / re-key)
# ---------------------------------------------------------------------------


def test_model_base_name_no_colon(config_dir):
    """A model name without a colon is returned unchanged."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="GLM-5.1")
    assert agent._model_base_name() == "GLM-5.1"


def test_model_base_name_strips_thinking_suffix(config_dir):
    """A thinking-level suffix (:none/:medium) is stripped from the key."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="GLM-5.1:medium")
    assert agent._model_base_name() == "GLM-5.1"


def test_model_base_name_strips_suffix_even_if_in_model_list(config_dir):
    """Regression: backends expose thinking variants as /v1/models entries.

    The suffix must be stripped by the verb whitelist, not by list membership
    — otherwise "Model:medium" (which IS in the list) would keep the suffix
    and the same model would get multiple locks.
    """
    from agent13.core import Agent

    agent = Agent(
        _MockClient("http://x/v1"),
        model="Qwen3.8-27B-MTPLX-Optimized-Quality:medium",
    )
    agent.available_models = [
        "Qwen3.8-27B-MTPLX-Optimized-Quality",
        "Qwen3.8-27B-MTPLX-Optimized-Quality:none",
        "Qwen3.8-27B-MTPLX-Optimized-Quality:medium",
    ]
    assert agent._model_base_name() == "Qwen3.8-27B-MTPLX-Optimized-Quality"


def test_model_base_name_keeps_non_thinking_suffix(config_dir):
    """A non-thinking colon suffix (OpenRouter) is kept."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="meta-llama/llama-3.1:free")
    assert agent._model_base_name() == "meta-llama/llama-3.1:free"


def test_model_base_name_strips_all_thinking_verbs(config_dir):
    """Every whitelisted thinking verb is stripped."""
    from agent13.core import Agent

    for verb in ("nothink", "none", "low", "medium", "high", "xhigh", "max"):
        agent = Agent(_MockClient("http://x/v1"), model=f"GLM-5.1:{verb}")
        assert agent._model_base_name() == "GLM-5.1", verb


def test_polite_model_key_sanitized_name(config_dir):
    """_polite_model_key returns the sanitized base model name."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="Qwen/3.8-27B")
    assert agent._polite_model_key() == "Qwen_3_8_27B"


def test_polite_model_key_strips_thinking_suffix(config_dir):
    """The thinking suffix is stripped before sanitizing."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="GLM-5.1:medium")
    assert agent._polite_model_key() == "GLM_5_1"


def test_polite_model_key_keeps_non_thinking_suffix(config_dir):
    """A non-thinking colon suffix survives into the key."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="meta-llama/llama-3.1:free")
    assert agent._polite_model_key() == "meta_llama_llama_3_1_free"


def test_polite_model_key_none_without_model(config_dir):
    """No model set -> None (provider-only key)."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="")
    agent.available_models = ["alpha"]
    assert agent._polite_model_key() is None


def test_set_polite_uses_model_name_in_path(config_dir):
    """set_polite keys the lock by provider + base model name."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x:8012/v1"), model="GLM-5.1:medium")
    agent.available_models = ["GLM-5.1"]
    agent.set_polite(interval=3)
    assert agent.polite_lock.path.name == "polite_http_x_8012_v1_GLM_5_1.lck"
    assert agent.polite_lock.model == "GLM_5_1"


def test_set_polite_same_provider_model_no_rekey(config_dir):
    """Re-calling set_polite with the same provider+model keeps the object."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="delta")
    agent.available_models = ["delta"]
    agent.set_polite(interval=1.0)
    lock_obj = agent.polite_lock
    agent.set_polite(interval=2.0)
    assert agent.polite_lock is lock_obj  # same object, interval updated
    assert agent.polite_lock.interval == 2.0


def test_set_model_rekeys_polite_lock(config_dir):
    """set_model re-keys the polite lock to the new model's name."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="alpha")
    agent.available_models = ["alpha", "beta", "gamma"]
    agent.set_polite(interval=3)
    old_path = agent.polite_lock.path
    assert agent.polite_lock.model == "alpha"

    agent.set_model("gamma")

    assert agent.polite_lock.model == "gamma"
    assert agent.polite_lock.path != old_path
    assert agent.polite_lock.interval == 3  # preserved


def test_set_model_no_polite_no_crash(config_dir):
    """set_model is safe when polite mode is not enabled."""
    from agent13.core import Agent

    agent = Agent(_MockClient("http://x/v1"), model="alpha")
    agent.available_models = ["alpha", "beta"]
    assert agent.polite_lock is None
    agent.set_model("beta")  # should not raise
    assert agent.polite_lock is None
