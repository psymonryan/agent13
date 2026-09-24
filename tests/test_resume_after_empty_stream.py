"""Failing tests demonstrating the stale-connection empty-stream theory.

Theory (from the "resume fails after a long pause" bug):

  After a long pause the pooled HTTP connection has gone stale (the server's
  keep-alive timeout elapsed while we sat at the pause safe-point, which sits
  *directly after* a tool result). On /resume the next LLM stream is issued on
  that stale connection, which ends with zero bytes.

  For a *streaming* request a zero-byte stream is indistinguishable from a
  legitimate empty response: ``async for chunk in stream`` simply yields
  nothing and raises no exception (the non-streaming "incomplete response"
  error never fires). So ``_llm_turn`` sees no content, no reasoning and no
  tool calls and ends the turn *silently* — no ASSISTANT_COMPLETE, no
  notification, the agent drops to IDLE, and the history dangles on a tool
  result that /resume cannot recover (not paused, and has_incomplete_turn is
  only set on context load).

These tests encode that theory. The first proves the transport mechanism
(a zero-byte stream yields nothing). The next two assert the behaviour the
fix must provide — retry the empty stream, and leave the stranded turn
recoverable via /resume — and therefore FAIL on the current code.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent13.core import Agent, AgentEvent
from agent13.llm import stream_response_with_tools


# ── Mock OpenAI chunk / stream objects ────────────────────────────────────────


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCallChunk:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.function = _Function(name, arguments)


class _Delta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, delta, finish_reason=None, usage=None):
        self.choices = [_Choice(delta, finish_reason)]
        self.usage = usage


class MockStream:
    """An async iterable of chunks, like the OpenAI SDK's streaming response.

    An empty list simulates a stale connection that closes with zero bytes.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        pass


def empty_stream() -> MockStream:
    """A stream that ends with zero bytes — what a stale pooled connection yields."""
    return MockStream([])


def content_stream(text: str) -> MockStream:
    return MockStream(
        [
            _Chunk(_Delta(content=text), None),
            _Chunk(_Delta(), "stop"),
        ]
    )


def make_client(streams) -> MagicMock:
    """A mock AsyncOpenAI client whose create() returns the given streams in order."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(streams))
    return client


# History that ends on a tool result — the exact state at the pause safe-point
# (right after a tool result, about to make the next LLM call).
HISTORY_ENDING_IN_TOOL_RESULT = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do the thing"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fake_tool", "arguments": json.dumps({"x": 5})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "tool-result"},
]


def _make_agent(client) -> Agent:
    agent = Agent(
        client=client,
        model="test-model",
        execute_tool=lambda name, args: "tool-result",
    )
    agent.messages = [dict(m) for m in HISTORY_ENDING_IN_TOOL_RESULT]
    agent._running = True
    return agent


def _events_of(agent, event_type) -> list:
    return [e for e in agent._collected if e.event == event_type]


# ── 1. The transport mechanism ────────────────────────────────────────────────


class TestStaleConnectionMechanism:
    """A zero-byte streaming response is indistinguishable from an empty one."""

    @pytest.mark.asyncio
    async def test_zero_byte_stream_yields_nothing(self):
        """stream_response_with_tools over an empty (stale) stream yields no
        content, no reasoning, no tool_calls_complete and no token_usage — and
        raises no exception. This is the 'what': the empty stream is silent.
        """
        client = make_client([empty_stream()])

        events = []
        async for event_type, _data in stream_response_with_tools(
            client, "test-model", HISTORY_ENDING_IN_TOOL_RESULT
        ):
            events.append(event_type)

        assert events == [], (
            "A zero-byte (stale-connection) stream should yield no events, "
            f"got {events!r}"
        )


# ── 2. The silent turn end (the bug) ──────────────────────────────────────────


class TestEmptyStreamTurnEnd:
    """An empty LLM stream must not silently kill the turn."""

    @pytest.mark.asyncio
    async def test_empty_stream_is_retried_not_silent(self):
        """When the stream comes back empty, _llm_turn should re-issue the call
        (a fresh connection) rather than ending the turn. If the retry returns
        content, the turn completes normally.

        FAILS today: the first (empty) stream ends the turn with a single call
        and no assistant message appended.
        """
        # First call empty (stale conn), retry succeeds with content.
        client = make_client([empty_stream(), content_stream("done")])
        agent = _make_agent(client)
        agent._collected = []

        @agent.on_event
        async def handler(event):
            agent._collected.append(event)

        await agent._llm_turn()

        assert client.chat.completions.create.call_count >= 2, (
            "An empty stream should trigger a retry of the LLM call, but the "
            f"client was only called {client.chat.completions.create.call_count} time(s)"
        )
        # The turn should have completed with the retried content.
        last = agent.messages[-1]
        assert last.get("role") == "assistant" and last.get("content") == "done", (
            "After a successful retry the turn should append the assistant "
            f"response, but history ends with {last!r}"
        )
        assert _events_of(agent, AgentEvent.ASSISTANT_COMPLETE), (
            "A completed (retried) turn should emit ASSISTANT_COMPLETE"
        )

    @pytest.mark.asyncio
    async def test_persistent_empty_stream_is_not_silent(self):
        """If the stream stays empty (retries exhausted), the turn must not end
        with total silence — the user needs to know the model returned nothing.

        FAILS today: no ASSISTANT_COMPLETE, no NOTIFICATION and no ERROR are
        emitted; the turn just vanishes.
        """
        client = make_client([empty_stream() for _ in range(10)])
        agent = _make_agent(client)
        agent._collected = []

        @agent.on_event
        async def handler(event):
            agent._collected.append(event)

        await agent._llm_turn()

        surfaced = (
            _events_of(agent, AgentEvent.NOTIFICATION)
            or _events_of(agent, AgentEvent.ERROR)
            or _events_of(agent, AgentEvent.ASSISTANT_COMPLETE)
        )
        assert surfaced, (
            "A turn that ends on a persistently empty stream should surface "
            "something (notification/error) instead of ending silently, but no "
            f"NOTIFICATION/ERROR/ASSISTANT_COMPLETE was emitted (events: "
            f"{[e.event for e in agent._collected]!r})"
        )


# ── 3. The stranded turn is unrecoverable ─────────────────────────────────────


class TestStrandedTurnRecovery:
    """After an empty-stream turn end, /resume must be able to continue it."""

    @pytest.mark.asyncio
    async def test_stranded_turn_is_recoverable_by_resume(self):
        """When a turn ends with the history dangling on a tool result, the
        agent should report it as an incomplete turn so /resume can continue
        it (case 2 of continue_incomplete_turn: last message is a tool result
        -> call the LLM).

        FAILS today: has_incomplete_turn is only set on context load, so after
        a mid-session empty-stream turn end it is False and /resume reports
        'Not paused' — the dangling tool result is stranded.
        """
        client = make_client([empty_stream() for _ in range(10)])
        agent = _make_agent(client)
        agent._collected = []

        @agent.on_event
        async def handler(event):
            agent._collected.append(event)

        await agent._llm_turn()

        # History genuinely dangles on a tool result...
        assert agent.messages[-1].get("role") == "tool", (
            "Precondition: history should end on a tool result, got "
            f"{agent.messages[-1]!r}"
        )
        # ...so the agent must flag it as an incomplete turn for /resume.
        assert agent.has_incomplete_turn, (
            "After an empty-stream turn end that leaves a dangling tool result, "
            "has_incomplete_turn should be True so /resume can continue the "
            "turn, but it is False (the turn is stranded)"
        )
