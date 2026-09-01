"""Unit tests for the TOOL_CALL_PENDING feature.

Providers like mlx-lm buffer whole tool calls: while generating one they stream
nothing visible - only payload-less keep-alive chunks (empty/role-only deltas).
We surface that "silent but alive" state so the TUI can show a dimmed "⚙ …"
placeholder instead of nothing at all during (potentially minutes of) buffered
tool-call generation.

Covers:
- llm.stream_response_with_tools yields ("keepalive", {}) for payload-less chunks
- AgentTUI._tool_call_markup renders the name-unknown pending variant
- Core emits TOOL_CALL_PENDING once when the stream goes silent past the
  threshold, and not when visible tokens keep flowing
"""

import pytest

from agent13 import Agent, AgentEvent
from agent13.llm import stream_response_with_tools
from ui.tui import AgentTUI


# =============================================================================
# llm layer: payload-less chunks become keepalive events
# =============================================================================


def _chunk(payload: dict) -> str:
    import json

    base = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
    }
    base.update(payload)
    return f"data: {json.dumps(base)}\n\n"


def make_buffered_sse() -> str:
    """SSE stream shaped like mlx-lm buffering a tool call.

    role chunk, two payload-less keep-alive chunks, then the whole tool call
    (name + args together) at the end.
    """
    parts = [
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path": "x"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        _chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        ),
    ]
    parts.append("data: [DONE]\n\n")
    return "".join(parts)


def make_normal_sse() -> str:
    """SSE stream shaped like a normal response: content flows immediately."""
    parts = [
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
                ]
            }
        ),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]
    parts.append("data: [DONE]\n\n")
    return "".join(parts)


class _FakeChoices:
    pass


class TestKeepaliveEvents:
    """stream_response_with_tools must surface payload-less chunks."""

    @pytest.fixture
    def openai_client(self, httpserver):
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key="test-key",
            base_url=httpserver.url_for("/v1"),
            max_retries=0,
        )

    @pytest.mark.asyncio
    async def test_buffered_tool_call_yields_keepalives(
        self, httpserver, openai_client
    ):
        httpserver.expect_request(
            "/v1/chat/completions", method="POST"
        ).respond_with_data(
            make_buffered_sse(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

        events = []
        async for event_type, data in stream_response_with_tools(
            openai_client,
            "test-model",
            [{"role": "user", "content": "write a file"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "write_file", "parameters": {}},
                }
            ],
        ):
            events.append((event_type, data))

        types = [t for t, _ in events]
        # The role chunk plus the two payload-less chunks before the tool call
        assert types.count("keepalive") == 3, f"keepalive events missing: {types}"
        # Keepalives precede the tool call (they ARE the silence signal)
        assert types.index("keepalive") < types.index("tool_call")
        # The buffered tool call still arrives intact
        assert "tool_calls_complete" in types

    @pytest.mark.asyncio
    async def test_normal_response_yields_one_initial_keepalive(
        self, httpserver, openai_client
    ):
        httpserver.expect_request(
            "/v1/chat/completions", method="POST"
        ).respond_with_data(
            make_normal_sse(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

        events = []
        async for event_type, data in stream_response_with_tools(
            openai_client,
            "test-model",
            [{"role": "user", "content": "hello"}],
        ):
            events.append((event_type, data))

        types = [t for t, _ in events]
        # Only the initial role chunk is payload-less
        assert types.count("keepalive") == 1, f"unexpected keepalives: {types}"
        assert types.index("keepalive") < types.index("content")


# =============================================================================
# _tool_call_markup: name-unknown pending variant
# =============================================================================


class TestPendingMarkup:
    def test_pending_without_name_is_gear_ellipsis(self):
        markup = AgentTUI._tool_call_markup("", {}, pending=True)
        assert markup == "[dim]⚙ …[/]"

    def test_pending_with_name_keeps_name(self):
        markup = AgentTUI._tool_call_markup("write_file", {}, pending=True)
        assert "write_file" in markup
        assert "…" in markup


# =============================================================================
# Core emission: TOOL_CALL_PENDING on silence, not on flowing tokens
# =============================================================================


def _make_agent():
    agent = Agent(client=object(), model="test-model")
    agent._running = True

    async def exec_tool(name, arguments):
        return '{"ok": true}'

    agent.execute_tool = exec_tool
    return agent


class TestToolCallPendingEmission:
    @staticmethod
    def _stateful_stream(first_call_events, gap: float = 0.01):
        """Fake _stream_and_emit: first call yields the given events (with a
        small real delay between them so silence measurement is meaningful),
        later calls yield a terminal content-only stream."""
        import asyncio

        state = {"n": 0}

        async def fake_stream_and_emit(
            messages, *, source="assistant", tool_choice="auto"
        ):
            state["n"] += 1
            if state["n"] == 1:
                for ev in first_call_events:
                    if gap:
                        await asyncio.sleep(gap)
                    yield ev
            else:
                yield ("content", "Done.")

        return fake_stream_and_emit

    @pytest.mark.asyncio
    async def test_pending_emitted_on_silence(self, monkeypatch):
        import agent13.core as core_mod

        # Shrink the threshold so the test doesn't sleep for real seconds
        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # mlx-like stream: first keepalive starts the clock (generation began),
        # second keepalive = silence -> pending fires, then the buffered call
        agent._stream_and_emit = self._stateful_stream(
            [
                ("keepalive", {}),
                ("keepalive", {}),
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "write_file",
                                "arguments": '{"path": "test.py"}',
                            }
                        ]
                    },
                ),
            ]
        )
        await agent._llm_turn()

        pending = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_PENDING]
        complete = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL]
        assert pending, "TOOL_CALL_PENDING was not emitted during silence"
        assert complete, "TOOL_CALL was not emitted"
        assert pending[0] < complete[0], "pending must precede the tool call"

    @pytest.mark.asyncio
    async def test_pending_emitted_only_once(self, monkeypatch):
        import agent13.core as core_mod

        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        pending_count = []

        @agent.on_event
        async def record(event):
            if event.event == AgentEvent.TOOL_CALL_PENDING:
                pending_count.append(1)

        # Multiple keepalives during a long silence - only one pending event
        agent._stream_and_emit = self._stateful_stream(
            [
                ("keepalive", {}),
                ("keepalive", {}),
                ("keepalive", {}),
                ("keepalive", {}),
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "write_file",
                                "arguments": "{}",
                            }
                        ]
                    },
                ),
            ]
        )
        await agent._llm_turn()

        assert len(pending_count) == 1, (
            f"expected exactly one pending event, got {len(pending_count)}"
        )

    @pytest.mark.asyncio
    async def test_no_pending_when_tokens_flow(self, monkeypatch):
        import agent13.core as core_mod

        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # Normal response: content flows, no keepalive ever fires
        agent._stream_and_emit = self._stateful_stream(
            [
                ("content", "Hello"),
                ("content", " world"),
            ]
        )
        await agent._llm_turn()

        assert AgentEvent.TOOL_CALL_PENDING not in order

    @pytest.mark.asyncio
    async def test_no_pending_after_tool_call_started(self, monkeypatch):
        import agent13.core as core_mod

        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # Streaming provider: name known early (TOOL_CALL_STARTED), then args
        # stream silently (keepalives). The named widget supersedes the
        # placeholder - no TOOL_CALL_PENDING may fire after the name is known.
        agent._stream_and_emit = self._stateful_stream(
            [
                ("tool_call", {"name": "write_file", "id": "tc_1"}),
                ("keepalive", {}),
                ("keepalive", {}),
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "write_file",
                                "arguments": '{"path": "test.py"}',
                            }
                        ]
                    },
                ),
            ]
        )
        await agent._llm_turn()

        started = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_STARTED]
        pending = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_PENDING]
        assert started, "TOOL_CALL_STARTED was not emitted"
        assert not pending, (
            "TOOL_CALL_PENDING must not fire once the tool name is known"
        )

    @pytest.mark.asyncio
    async def test_pending_rearms_after_reasoning(self, monkeypatch):
        """The user-reported pattern: silent gap (false alarm) -> thinking
        -> tool-call silence. The second silence must get its own signal."""
        import agent13.core as core_mod

        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # Gap before thinking (fires), thinking streams (re-arm + false alarm
        # discard in UI), then the real silence before the buffered tool call
        # (must fire AGAIN).
        agent._stream_and_emit = self._stateful_stream(
            [
                ("keepalive", {}),
                ("keepalive", {}),
                ("reasoning", "Let me plan the essay."),
                ("keepalive", {}),
                ("keepalive", {}),
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "write_file",
                                "arguments": '{"path": "essay.txt"}',
                            }
                        ]
                    },
                ),
            ]
        )
        await agent._llm_turn()

        pending = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_PENDING]
        reasoning = [
            i for i, e in enumerate(order) if e == AgentEvent.ASSISTANT_REASONING
        ]
        complete = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL]
        assert len(pending) == 2, (
            f"expected pending to fire in BOTH silences, got {len(pending)}: "
            f"{[e.name for e in order]}"
        )
        assert pending[0] < reasoning[0] < pending[1], (
            "first pending before thinking, second after (re-armed)"
        )
        assert pending[1] < complete[0], "second pending must precede the tool call"

    @pytest.mark.asyncio
    async def test_pending_rearms_after_content(self, monkeypatch):
        """Same re-arm contract for content tokens (preamble before a tool
        call: 'I'll write that file...' -> silent buffering)."""
        import agent13.core as core_mod

        monkeypatch.setattr(core_mod, "TOOL_CALL_PENDING_SILENCE", 0.0)
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        agent._stream_and_emit = self._stateful_stream(
            [
                ("content", "I'll write that file for you."),
                ("keepalive", {}),
                ("keepalive", {}),
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "write_file",
                                "arguments": '{"path": "test.py"}',
                            }
                        ]
                    },
                ),
            ]
        )
        await agent._llm_turn()

        pending = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_PENDING]
        assert len(pending) == 1, (
            f"content re-arms, so the post-content silence must fire: {pending}"
        )
        assert pending[0] > next(
            i for i, e in enumerate(order) if e == AgentEvent.ASSISTANT_TOKEN
        )
