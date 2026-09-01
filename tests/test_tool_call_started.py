"""Unit tests for the TOOL_CALL_STARTED feature.

The model commits to a tool (name known) before its arguments finish
streaming. We surface that early so the TUI can show a dimmed name-only
widget immediately instead of waiting for the full argument stream.

Covers:
- AgentTUI._tool_call_markup (pending vs complete vs no-args)
- Core emits TOOL_CALL_STARTED before TOOL_CALL, once per tool-call id
"""

import pytest

from agent13 import Agent, AgentEvent
from ui.tui import AgentTUI


# =============================================================================
# _tool_call_markup
# =============================================================================


class TestToolCallMarkup:
    """The markup builder shared by the pending and complete widgets."""

    def test_pending_is_dimmed_name_only(self):
        markup = AgentTUI._tool_call_markup("write_file", {}, pending=True)
        assert "write_file" in markup
        assert "dim" in markup
        assert "…" in markup
        # No args line when pending
        assert "filepath" not in markup

    def test_complete_includes_truncated_args(self):
        args = {"filepath": "test.py", "content": "x" * 500}
        markup = AgentTUI._tool_call_markup("write_file", args, pending=False)
        assert "write_file" in markup
        assert "test.py" in markup
        # Truncated to 200 chars + ellipsis
        assert "..." in markup
        # Not dimmed (bright name)
        assert "bold yellow" in markup

    def test_complete_no_args_is_name_only(self):
        markup = AgentTUI._tool_call_markup("command", {}, pending=False)
        assert "command" in markup
        assert "…" not in markup


# =============================================================================
# Core emission: TOOL_CALL_STARTED before TOOL_CALL
# =============================================================================


def _make_agent():
    agent = Agent(client=object(), model="test-model")
    agent._running = True

    async def exec_tool(name, arguments):
        return '{"ok": true}'

    agent.execute_tool = exec_tool
    return agent


class TestToolCallStartedEmission:
    """Core must emit TOOL_CALL_STARTED (early name) before TOOL_CALL.

    _llm_turn loops until a stream yields no tool calls, so the fake stream is
    stateful: the first call yields the tool call, subsequent calls yield a
    terminal content-only stream so the loop exits.
    """

    @staticmethod
    def _stateful_stream(first_call_events):
        """Build a stateful fake _stream_and_emit.

        First call yields first_call_events; every later call yields a terminal
        content-only stream (no tool calls) so _llm_turn's loop terminates.
        """
        state = {"n": 0}

        async def fake_stream_and_emit(messages, *, source="assistant", tool_choice="auto"):
            state["n"] += 1
            if state["n"] == 1:
                for ev in first_call_events:
                    yield ev
            else:
                yield ("content", "Done.")

        return fake_stream_and_emit

    @pytest.mark.asyncio
    async def test_started_precedes_complete_and_result(self):
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # content, then the early name event, then (silent) args, then complete
        agent._stream_and_emit = self._stateful_stream([
            ("content", "Writing..."),
            ("tool_call", {"name": "write_file", "id": "tc_1"}),
            ("tool_calls_complete", {
                "tool_calls": [{
                    "id": "tc_1",
                    "name": "write_file",
                    "arguments": '{"filepath": "test.py"}',
                }]
            }),
        ])
        await agent._llm_turn()

        started = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL_STARTED]
        complete = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_CALL]
        result = [i for i, e in enumerate(order) if e == AgentEvent.TOOL_RESULT]

        assert started, "TOOL_CALL_STARTED was not emitted"
        assert complete, "TOOL_CALL was not emitted"
        assert result, "TOOL_RESULT was not emitted"
        assert started[0] < complete[0], "started must precede complete"
        assert complete[0] < result[0], "complete must precede result"

    @pytest.mark.asyncio
    async def test_started_emitted_once_per_tool_call_id(self):
        agent = _make_agent()
        started_names = []

        @agent.on_event
        async def record(event):
            if event.event == AgentEvent.TOOL_CALL_STARTED:
                started_names.append(event.data.get("name"))

        # The name event can repeat for the same id (provider quirk) — we must
        # dedupe so the UI only shows one pending widget per tool call.
        agent._stream_and_emit = self._stateful_stream([
            ("tool_call", {"name": "write_file", "id": "tc_1"}),
            ("tool_call", {"name": "write_file", "id": "tc_1"}),  # repeat
            ("tool_calls_complete", {
                "tool_calls": [{
                    "id": "tc_1",
                    "name": "write_file",
                    "arguments": "{}",
                }]
            }),
        ])
        await agent._llm_turn()

        assert started_names == ["write_file"], (
            f"expected exactly one started event, got {started_names}"
        )

    @pytest.mark.asyncio
    async def test_no_started_when_no_tool_call(self):
        agent = _make_agent()
        order = []

        @agent.on_event
        async def record(event):
            order.append(event.event)

        # No tool call at all — the loop terminates on the first stream.
        agent._stream_and_emit = self._stateful_stream([
            ("content", "Just a reply."),
        ])
        await agent._llm_turn()

        assert AgentEvent.TOOL_CALL_STARTED not in order
        assert AgentEvent.TOOL_CALL not in order
