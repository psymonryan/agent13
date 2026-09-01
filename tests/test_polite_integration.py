"""Integration tests for polite mode.

Tests what the user actually experiences: two agents coordinating via the
shared lock, events flowing through the agent event system, and the lock
being released after a turn so the next agent can proceed.

The LLM is mocked via pytest-httpserver; the polite lock is real (file-based).
"""

import asyncio
import json
import time

import pytest
import pytest_httpserver
from openai import AsyncOpenAI
from werkzeug.wrappers import Response

from agent13.core import Agent
from agent13.events import AgentEvent
from agent13.commands import execute_polite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_handler(delay: float = 0.0):
    """Build a chat handler that returns a simple SSE response.

    If ``delay`` > 0, the handler sleeps before responding — used to keep
    the first agent's turn in flight long enough for the second to wait.
    """

    def handler(request):
        if delay:
            import time

            time.sleep(delay)
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        last_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_msg = msg.get("content", "")
                break
        content = f"Reply to: {last_msg[:50]}"
        chunk = {
            "id": "mock",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        final = {
            "id": "mock",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        sse = f"data: {json.dumps(chunk)}\n\n"
        sse += f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"
        return Response(sse, content_type="text/event-stream")

    return handler


def _make_models_handler():
    def handler(request):
        return Response(
            json.dumps(
                {"data": [{"id": "mock-model", "object": "model", "owned_by": "test"}]}
            ),
            content_type="application/json",
        )

    return handler


@pytest.fixture
def mock_server():
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(_make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        _make_chat_handler()
    )
    server.start()
    yield server
    server.stop()


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Isolate ~/.agent13 to a temp dir so lock files don't collide."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _make_agent(server, monkeypatch):
    """Build an Agent wired to the mock server with a no-op tool executor."""
    client = AsyncOpenAI(
        base_url=server.url_for("/v1"),
        api_key="test-key",
    )

    async def execute_tool(name, args):
        return "ok"

    agent = Agent(
        client=client,
        model="mock-model",
        execute_tool=execute_tool,
    )
    return agent


# ---------------------------------------------------------------------------
# Event capture
# ---------------------------------------------------------------------------


async def _wait_for_lock_released(lock, timeout: float = 5.0) -> bool:
    """Wait for the lock to be acquired then released, or timeout.

    Fixed sleeps are inherently flaky across platforms (the mock LLM
    round-trip is markedly slower on Windows CI than macOS). Polling on
    the actual lock state makes the assertion deterministic.

    We must first observe the lock being held (the turn has started) and
    then observe it released (the turn's ``finally`` has run) — polling
    only for "not held" would return True instantly before the turn even
    begins acquiring.
    """
    deadline = time.monotonic() + timeout

    # Phase 1: wait for the turn to acquire the lock.
    while time.monotonic() < deadline:
        if lock.is_held():
            break
        await asyncio.sleep(0.02)
    else:
        return not lock.is_held()

    # Phase 2: wait for the turn to release the lock.
    while time.monotonic() < deadline:
        if not lock.is_held():
            return True
        await asyncio.sleep(0.05)
    return not lock.is_held()


class EventCollector:
    """Collects agent events for assertions."""

    def __init__(self):
        self.events: list[tuple[AgentEvent, dict]] = []

    def make_handler(self):
        async def handler(event):
            self.events.append((event.event, dict(event.data)))

        return handler

    def count(self, event_type: AgentEvent) -> int:
        return sum(1 for e in self.events if e[0] is event_type)

    def has(self, event_type: AgentEvent) -> bool:
        return self.count(event_type) > 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polite_agent_acquires_and_releases(mock_server, isolated_config):
    """A single polite agent acquires the lock for a turn, then releases it.

    The lock should be held during the turn and free after it completes.
    """
    agent = _make_agent(mock_server, isolated_config)
    collector = EventCollector()
    agent.on_event(collector.make_handler())

    agent.set_polite(interval=0.05)
    assert agent.polite_mode

    # The lock file should exist after set_polite (created on construction)
    assert agent.polite_lock is not None
    lock_path = agent.polite_lock.path
    assert lock_path.parent == isolated_config / "locks"

    # Run a turn
    agent.queue.add("hello")
    agent_task = asyncio.create_task(agent.run())
    # Wait for the turn to actually complete and release the lock.
    # Polling (rather than a fixed sleep) is deterministic across platforms:
    # the mock LLM round-trip timing varies between macOS and Windows CI.
    released = await _wait_for_lock_released(agent.polite_lock, timeout=5.0)
    agent.stop()
    await asyncio.sleep(0.1)

    # The lock should NOT be held after the turn completes
    assert released, "lock should be released after the turn completes"
    assert not agent.polite_lock.is_held()

    # POLITE_ACQUIRED should have fired (at least once)
    assert collector.has(AgentEvent.POLITE_ACQUIRED)

    # Clean up
    if not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_second_agent_waits_for_first(mock_server, isolated_config):
    """Two polite agents: the second waits while the first holds the lock.

    This is the core coordination guarantee. We hold agent1's lock manually
    (simulating an in-flight turn), start agent2's turn, and verify agent2
    waits. Then we release agent1's lock and verify agent2 proceeds.
    """
    agent1 = _make_agent(mock_server, isolated_config)
    agent2 = _make_agent(mock_server, isolated_config)

    coll2 = EventCollector()
    agent2.on_event(coll2.make_handler())

    # Both use the same provider (mock server URL) -> same lock
    agent1.set_polite(interval=0.05)
    agent2.set_polite(interval=0.05)
    assert agent1.polite_lock.path == agent2.polite_lock.path

    # Hold the lock as if agent1 is mid-turn
    await agent1.polite_lock.acquire()
    assert agent1.polite_lock.is_held()

    # Start agent2 — it should wait for the lock
    agent2.queue.add("second message")
    task2 = asyncio.create_task(agent2.run())

    # Give agent2 time to try and fail to acquire
    await asyncio.sleep(0.5)

    # Agent2 should be waiting (POLITE_WAITING events)
    assert coll2.has(AgentEvent.POLITE_WAITING), "agent2 should emit POLITE_WAITING"

    # Agent2 should NOT have acquired yet
    assert not coll2.has(AgentEvent.POLITE_ACQUIRED), "agent2 acquired too early"

    # Release agent1's lock — agent2 should proceed
    agent1.polite_lock.release()
    await asyncio.sleep(0.5)

    # Agent2 should now acquire
    assert coll2.has(AgentEvent.POLITE_ACQUIRED), (
        "agent2 should acquire after agent1 releases"
    )

    # Clean up
    agent2.stop()
    await asyncio.sleep(0.2)
    if not task2.done():
        task2.cancel()
        try:
            await task2
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_non_polite_agent_ignores_lock(mock_server, isolated_config):
    """An agent without polite mode ignores the lock entirely.

    A polite agent holding the lock should not block a non-polite agent
    from proceeding.
    """
    polite_agent = _make_agent(mock_server, isolated_config)
    plain_agent = _make_agent(mock_server, isolated_config)

    coll_plain = EventCollector()
    plain_agent.on_event(coll_plain.make_handler())

    # Only the polite agent gets polite mode
    polite_agent.set_polite(interval=0.05)
    assert polite_agent.polite_lock is not None

    # Manually acquire and hold the lock
    await polite_agent.polite_lock.acquire()
    assert polite_agent.polite_lock.is_held()

    # The plain agent should NOT be affected — it has no polite_lock
    assert plain_agent.polite_lock is None

    # Run a turn on the plain agent; it should proceed immediately
    plain_agent.queue.add("hello")
    task = asyncio.create_task(plain_agent.run())
    await asyncio.sleep(0.5)

    # Plain agent should have completed its turn (got ASSISTANT_COMPLETE
    # or at least ITEM_STARTED — not blocked)
    assert coll_plain.has(AgentEvent.ITEM_STARTED), (
        "plain agent should start without waiting"
    )
    assert not coll_plain.has(AgentEvent.POLITE_WAITING), (
        "plain agent should not emit POLITE_WAITING"
    )

    plain_agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    polite_agent.polite_lock.release()


@pytest.mark.asyncio
async def test_lock_released_on_error(mock_server, isolated_config):
    """The lock is released even if the turn errors out.

    Verifies the finally block in _process_item fires on error paths.
    """
    agent = _make_agent(mock_server, isolated_config)
    agent.set_polite(interval=0.05)

    # Force an error by making the client fail
    original_client = agent.client

    # Replace the client's create method to raise
    async def failing_create(**kwargs):
        raise RuntimeError("simulated failure")

    agent.client.chat = type(agent.client.chat)(client=original_client)
    agent.client.chat.completions = type(agent.client.chat.completions)(
        client=original_client
    )
    agent.client.chat.completions.create = failing_create

    # Run a turn — it should error
    agent.queue.add("this will fail")
    task = asyncio.create_task(agent.run())
    # Wait for the turn to actually error out and release the lock.
    # Polling is deterministic across platforms (fixed sleeps are flaky
    # on Windows CI where the mock round-trip is slower).
    released = await _wait_for_lock_released(agent.polite_lock, timeout=5.0)

    # The lock should be released despite the error
    assert released, "lock should be released after error"

    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_polite_command_enables(mock_server, isolated_config):
    """The /polite command (via execute_polite) enables polite mode at runtime."""
    agent = _make_agent(mock_server, isolated_config)
    assert not agent.polite_mode

    # /polite 0.3
    result = execute_polite(agent, "0.3")
    assert result.success
    assert agent.polite_mode
    assert agent.polite_lock.interval == 0.3

    # /polite off
    result = execute_polite(agent, "off")
    assert result.success
    assert not agent.polite_mode
    assert agent.polite_lock is None


@pytest.mark.asyncio
async def test_polite_command_off_when_not_enabled(mock_server, isolated_config):
    """/polite off when never enabled is a silent no-op (success)."""
    agent = _make_agent(mock_server, isolated_config)
    assert not agent.polite_mode

    result = execute_polite(agent, "off")
    assert result.success
    assert not agent.polite_mode


@pytest.mark.asyncio
async def test_polite_command_no_args_shows_status_and_usage(mock_server, isolated_config):
    """/polite with no args shows current status plus usage hint."""
    agent = _make_agent(mock_server, isolated_config)

    # When not enabled — shows "off" plus usage
    result = execute_polite(agent, "")
    assert result.success
    assert "Polite mode: off" in result.message
    assert "Usage" in result.message
    assert not agent.polite_mode

    # When enabled — shows interval plus usage
    agent.set_polite(interval=0.3)
    result = execute_polite(agent, "")
    assert result.success
    assert "enabled" in result.message
    assert "0.3" in result.message
    assert "Usage" in result.message


@pytest.mark.asyncio
async def test_polite_command_invalid_shows_usage(mock_server, isolated_config):
    """/polite abc shows usage (invalid float)."""
    agent = _make_agent(mock_server, isolated_config)

    result = execute_polite(agent, "abc")
    assert not result.success
    assert "Usage" in result.message
    assert not agent.polite_mode


# ---------------------------------------------------------------------------
# Non-GPU items skip the lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_skips_polite_lock(mock_server, isolated_config):
    """/clear (kind='clear') does not acquire the polite lock.

    Local-only operations execute instantly even when another agent holds
    the lock, since they never touch the GPU/LLM.
    """
    holder = _make_agent(mock_server, isolated_config)
    clearer = _make_agent(mock_server, isolated_config)

    collector = EventCollector()
    clearer.on_event(collector.make_handler())

    # Both agents polite, same backend
    holder.set_polite(interval=0.05)
    clearer.set_polite(interval=0.05)
    assert holder.polite_lock.path == clearer.polite_lock.path

    # Hold the lock as if another agent is mid-turn
    await holder.polite_lock.acquire()
    assert holder.polite_lock.is_held()

    # Request clear — should proceed instantly, no POLITE_WAITING
    await clearer.request_clear()
    task = asyncio.create_task(clearer.run())
    await asyncio.sleep(0.5)

    # Clear should have completed without waiting
    assert not collector.has(AgentEvent.POLITE_WAITING), (
        "clear should not wait for the lock"
    )
    assert collector.has(AgentEvent.MESSAGES_CLEARED), (
        "clear should have completed"
    )

    clearer.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    holder.polite_lock.release()


@pytest.mark.asyncio
async def test_load_skips_polite_lock(mock_server, isolated_config, tmp_path):
    """/load (kind='load') does not acquire the polite lock.

    Like clear, load is a local-only operation that should execute
    instantly regardless of lock state.
    """
    # Create a minimal save file to load
    save_path = tmp_path / "test_save.ctx"
    save_path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "saved message"}],
                "system_prompt": "test",
                "model": "mock-model",
            }
        )
    )

    holder = _make_agent(mock_server, isolated_config)
    loader = _make_agent(mock_server, isolated_config)

    collector = EventCollector()
    loader.on_event(collector.make_handler())

    holder.set_polite(interval=0.05)
    loader.set_polite(interval=0.05)

    # Hold the lock
    await holder.polite_lock.acquire()

    # Request load — should proceed instantly
    await loader.request_load(str(save_path))
    task = asyncio.create_task(loader.run())
    await asyncio.sleep(0.5)

    assert not collector.has(AgentEvent.POLITE_WAITING), (
        "load should not wait for the lock"
    )
    assert collector.has(AgentEvent.CONTEXT_LOADED), (
        "load should have completed"
    )

    loader.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    holder.polite_lock.release()


@pytest.mark.asyncio
async def test_normal_prompt_waits_for_polite_lock(mock_server, isolated_config):
    """A normal prompt (no special kind) DOES wait for the polite lock.

    This is the counterpart to the clear/load skip tests — confirms that
    GPU-hitting items still acquire the lock.
    """
    holder = _make_agent(mock_server, isolated_config)
    agent = _make_agent(mock_server, isolated_config)

    collector = EventCollector()
    agent.on_event(collector.make_handler())

    holder.set_polite(interval=0.05)
    agent.set_polite(interval=0.05)

    # Hold the lock
    await holder.polite_lock.acquire()

    # Queue a normal prompt — should wait
    agent.queue.add("hello")
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.5)

    assert collector.has(AgentEvent.POLITE_WAITING), (
        "normal prompt should wait for the lock"
    )

    # Release — should proceed
    holder.polite_lock.release()
    await asyncio.sleep(0.5)
    assert collector.has(AgentEvent.POLITE_ACQUIRED), (
        "normal prompt should acquire after release"
    )

    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Option B: Lock acquired before get_next() — cancel & delete during wait
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_polite_wait_keeps_item_in_queue(mock_server, isolated_config):
    """ESC (task.cancel) during polite wait leaves the item in the pending queue.

    The item was never pulled as 'current' because the lock is now acquired
    BEFORE get_next(). After cancellation the item should still be in
    queue.items, ready to be deleted or re-processed.
    """
    holder = _make_agent(mock_server, isolated_config)
    agent = _make_agent(mock_server, isolated_config)

    coll = EventCollector()
    agent.on_event(coll.make_handler())

    holder.set_polite(interval=0.05)
    agent.set_polite(interval=0.05)

    # Hold the lock so agent can't proceed
    await holder.polite_lock.acquire()

    # Queue a prompt — agent will wait for the lock
    agent.queue.add("waiting message")
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.5)

    # Agent should be waiting
    assert coll.has(AgentEvent.POLITE_WAITING), "agent should be waiting for lock"

    # The item should still be in the pending queue (not pulled as current)
    assert agent.queue.pending_count == 1, "item should still be pending"
    assert agent.queue.current is None, "no item should be current"

    # Cancel the task (simulating ESC)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    # Item should STILL be in the queue after cancellation
    assert agent.queue.pending_count == 1, "item should remain in queue after cancel"
    assert agent.queue.current is None, "no item should be current after cancel"
    assert agent.queue.list_items()[0].text == "waiting message"

    # Clean up
    holder.polite_lock.release()


@pytest.mark.asyncio
async def test_delete_queue_item_during_polite_wait(mock_server, isolated_config):
    """Deleting a queue item during polite wait releases the lock and skips it.

    While the agent is waiting for the polite lock, /delete q 1 should
    remove the pending item. The agent should then release the lock and
    loop back without processing it.
    """
    holder = _make_agent(mock_server, isolated_config)
    agent = _make_agent(mock_server, isolated_config)

    coll = EventCollector()
    agent.on_event(coll.make_handler())

    holder.set_polite(interval=0.05)
    agent.set_polite(interval=0.05)

    # Hold the lock
    await holder.polite_lock.acquire()

    # Queue a prompt — agent will wait for the lock
    agent.queue.add("doomed message")
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.5)

    # Agent should be waiting
    assert coll.has(AgentEvent.POLITE_WAITING), "agent should be waiting"

    # The item is still pending — delete it
    assert agent.queue.pending_count == 1
    removed = agent.queue.remove_at(1)
    assert removed is not None
    assert removed.text == "doomed message"
    assert agent.queue.pending_count == 0

    # Now release the holder's lock — agent will acquire, see item is gone,
    # release the lock, and loop back to idle.
    holder.polite_lock.release()
    await asyncio.sleep(0.5)

    # Agent should NOT have started processing the item
    assert not coll.has(AgentEvent.ITEM_STARTED), (
        "deleted item should not have started processing"
    )

    # Lock should not be held by agent
    assert not agent.polite_lock.is_held(), "lock should be released after skipping"

    # Clean up
    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_delete_then_next_item_proceeds(mock_server, isolated_config):
    """After deleting the waiting item, a subsequent item proceeds normally.

    Queue two items, hold the lock. Delete the first item while waiting.
    Release the lock — the second item should then be processed.
    """
    holder = _make_agent(mock_server, isolated_config)
    agent = _make_agent(mock_server, isolated_config)

    coll = EventCollector()
    agent.on_event(coll.make_handler())

    holder.set_polite(interval=0.05)
    agent.set_polite(interval=0.05)

    await holder.polite_lock.acquire()

    # Queue two items
    agent.queue.add("first (doomed)")
    agent.queue.add("second (should proceed)")
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.5)

    # Waiting for lock on first item
    assert coll.has(AgentEvent.POLITE_WAITING)

    # Delete the first item
    agent.queue.remove_at(1)
    assert agent.queue.pending_count == 1

    # Release lock — agent acquires, sees first item is gone, releases,
    # then loops and acquires for the second item.
    holder.polite_lock.release()
    released = await _wait_for_lock_released(agent.polite_lock, timeout=5.0)
    assert released, "lock should be acquired and released for second item"

    # The second item should have been processed
    assert coll.has(AgentEvent.ITEM_STARTED), "second item should have started"

    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_clear_load_skip_lock_with_option_b(mock_server, isolated_config):
    """Clear/load items still skip the lock after the Option B refactor.

    The peek-based check in run() should correctly identify clear/load
    kinds and skip lock acquisition, same as before.
    """
    holder = _make_agent(mock_server, isolated_config)
    clearer = _make_agent(mock_server, isolated_config)

    coll = EventCollector()
    clearer.on_event(coll.make_handler())

    holder.set_polite(interval=0.05)
    clearer.set_polite(interval=0.05)

    # Hold the lock
    await holder.polite_lock.acquire()

    # Request clear — should proceed instantly
    await clearer.request_clear()
    task = asyncio.create_task(clearer.run())
    await asyncio.sleep(0.5)

    assert not coll.has(AgentEvent.POLITE_WAITING), "clear should not wait"
    assert coll.has(AgentEvent.MESSAGES_CLEARED), "clear should complete"

    clearer.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    holder.polite_lock.release()

# ---------------------------------------------------------------------------
# Lock released during tool execution (Option B level 2)
# ---------------------------------------------------------------------------


def _make_tool_call_chat_handler():
    """Chat handler that returns a tool call on the first request, then a plain reply.

    The tool call references 'test_tool' which the test agent executes as a
    slow operation. On the second request (after tool result), returns a
    normal text response.
    """

    call_count = [0]

    def handler(request):
        call_count[0] += 1

        if call_count[0] == 1:
            # First call: return a tool call
            chunk = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "test_tool",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            final = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
            }
        else:
            # Second call: normal response
            chunk = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": "Done!"}, "finish_reason": None}
                ],
            }
            final = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        sse = f"data: {json.dumps(chunk)}\n\n"
        sse += f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"
        return Response(sse, content_type="text/event-stream")

    return handler


def _make_agent_with_tool(server, monkeypatch, tool_delay: float = 0.0):
    """Build an Agent with a slow test_tool that takes tool_delay seconds."""
    client = AsyncOpenAI(
        base_url=server.url_for("/v1"),
        api_key="test-key",
    )

    async def execute_tool(name, args):
        if name == "test_tool" and tool_delay:
            await asyncio.sleep(tool_delay)
        return "tool result"

    agent = Agent(
        client=client,
        model="mock-model",
        execute_tool=execute_tool,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    return agent


@pytest.fixture
def tool_call_server():
    """Mock server that returns a tool call on first request, plain reply after."""
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(_make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        _make_tool_call_chat_handler()
    )
    server.start()
    yield server
    server.stop()


@pytest.mark.asyncio
async def test_lock_released_during_tool_execution(tool_call_server, isolated_config):
    """The polite lock is released during tool execution so other agents can use the GPU.

    Agent1 runs a turn with a slow tool. While the tool executes (no GPU),
    agent2 should be able to acquire the lock and run its own LLM turn.
    """
    agent1 = _make_agent_with_tool(tool_call_server, isolated_config, tool_delay=1.0)
    agent2 = _make_agent_with_tool(tool_call_server, isolated_config, tool_delay=0.0)

    coll1 = EventCollector()
    agent1.on_event(coll1.make_handler())
    coll2 = EventCollector()
    agent2.on_event(coll2.make_handler())

    agent1.set_polite(interval=0.05)
    agent2.set_polite(interval=0.05)

    # Start agent1 — it will acquire lock, stream LLM (tool call), release lock,
    # then execute the slow tool (1s sleep).
    agent1.queue.add("run tool")
    task1 = asyncio.create_task(agent1.run())

    # Wait for agent1 to start tooling (lock should be released by then)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if coll1.has(AgentEvent.TOOL_CALL):
            break
        await asyncio.sleep(0.02)

    assert coll1.has(AgentEvent.TOOL_CALL), "agent1 should have started tool execution"
    assert not agent1.polite_lock.is_held(), (
        "lock should be released during tool execution"
    )

    # Now agent2 should be able to acquire the lock and run its turn
    agent2.queue.add("my turn")
    task2 = asyncio.create_task(agent2.run())

    # Wait for agent2 to acquire
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if coll2.has(AgentEvent.POLITE_ACQUIRED):
            break
        await asyncio.sleep(0.02)

    assert coll2.has(AgentEvent.POLITE_ACQUIRED), (
        "agent2 should acquire the lock while agent1 executes tools"
    )

    # Clean up
    agent1.stop()
    agent2.stop()
    await asyncio.sleep(0.2)
    for task in (task1, task2):
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_lock_reacquired_for_second_llm_round(tool_call_server, isolated_config):
    """After tool execution, the lock is re-acquired for the next LLM stream.

    Agent runs a turn with one tool round. The lock should be:
    1. Held during first LLM stream
    2. Released during tool execution
    3. Re-acquired for second LLM stream
    4. Released after second stream (no more tools)
    """
    agent = _make_agent_with_tool(tool_call_server, isolated_config, tool_delay=0.3)
    coll = EventCollector()
    agent.on_event(coll.make_handler())

    agent.set_polite(interval=0.05)

    agent.queue.add("run tool")
    task = asyncio.create_task(agent.run())

    # Wait for tool call to start (lock should be released)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if coll.has(AgentEvent.TOOL_CALL):
            break
        await asyncio.sleep(0.02)
    assert coll.has(AgentEvent.TOOL_CALL), "should have tool call"

    # Lock should be released during tool execution
    assert not agent.polite_lock.is_held(), (
        "lock should be released during tool execution"
    )

    # Wait for the turn to complete (ASSISTANT_COMPLETE on second round)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if coll.has(AgentEvent.ASSISTANT_COMPLETE):
            break
        await asyncio.sleep(0.02)
    assert coll.has(AgentEvent.ASSISTANT_COMPLETE), "should complete second LLM round"

    # Lock should be released after the turn
    assert not agent.polite_lock.is_held(), (
        "lock should be released after final LLM stream"
    )

    # POLITE_ACQUIRED should have fired at least twice:
    # once for the initial acquire (by run()), once for re-acquire after tools
    assert coll.count(AgentEvent.POLITE_ACQUIRED) >= 2, (
        "lock should be acquired at least twice (initial + re-acquire after tools)"
    )

    # Clean up
    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_lock_released_on_simple_turn(mock_server, isolated_config):
    """A turn with no tool calls releases the lock after the single LLM stream.

    The lock is acquired by run(), _llm_turn streams once, releases after
    streaming, then breaks (no tools). The lock should be free after the turn.
    """
    agent = _make_agent(mock_server, isolated_config)
    coll = EventCollector()
    agent.on_event(coll.make_handler())

    agent.set_polite(interval=0.05)

    agent.queue.add("hello")
    task = asyncio.create_task(agent.run())

    # Wait for turn to complete
    released = await _wait_for_lock_released(agent.polite_lock, timeout=5.0)
    assert released, "lock should be acquired and released"

    assert coll.has(AgentEvent.POLITE_ACQUIRED), "should have acquired the lock"
    assert coll.has(AgentEvent.ASSISTANT_COMPLETE), "should complete the turn"
    assert not agent.polite_lock.is_held(), "lock should be free after turn"

    # POLITE_ACQUIRED should fire exactly once (no re-acquire needed)
    assert coll.count(AgentEvent.POLITE_ACQUIRED) == 1, (
        "should acquire exactly once for a no-tool turn"
    )

    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_lock_safety_net_on_error(mock_server, isolated_config):
    """The _process_item finally block releases the lock if _llm_turn errors.

    If an error occurs during streaming (before the release point),
    _process_item's finally acts as a safety net.
    """
    agent = _make_agent(mock_server, isolated_config)
    agent.set_polite(interval=0.05)

    # Force an error by making the client fail
    original_client = agent.client

    async def failing_create(**kwargs):
        raise RuntimeError("simulated failure")

    agent.client.chat = type(agent.client.chat)(client=original_client)
    agent.client.chat.completions = type(agent.client.chat.completions)(
        client=original_client
    )
    agent.client.chat.completions.create = failing_create

    agent.queue.add("this will fail")
    task = asyncio.create_task(agent.run())

    # Wait for the turn to error out and release the lock
    released = await _wait_for_lock_released(agent.polite_lock, timeout=5.0)
    assert released, "lock should be released after error (safety net)"

    agent.stop()
    await asyncio.sleep(0.1)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
