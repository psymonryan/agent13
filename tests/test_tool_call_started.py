"""Unit tests for the TOOL_CALL_STARTED feature.

The model commits to a tool (name known) before its arguments finish
streaming. We surface that early so the TUI can show a dimmed name-only
widget immediately instead of waiting for the full argument stream.

Covers:
- ToolWidget headers + format_tool_args (pending vs complete vs no-args)
- Core emits TOOL_CALL_STARTED before TOOL_CALL, once per tool-call id
"""

import pytest

from agent13 import Agent, AgentEvent
from ui.tui import ToolWidget, format_tool_args


# =============================================================================
# ToolWidget header + format_tool_args
# =============================================================================


class TestToolCallMarkup:
    """The header markup and args body behind a tool widget."""

    def test_pending_is_dimmed_name_only(self):
        widget = ToolWidget("write_file", pending=True)
        header = widget._title_markup()
        assert "write_file" in header
        assert "dim" in header
        assert "…" in header
        # No args body when pending
        assert widget._content == ""

    def test_complete_includes_truncated_args(self):
        args = {"filepath": "test.py", "content": "x" * 500}
        widget = ToolWidget("write_file")
        widget.set_call("write_file", args)
        header = widget._title_markup()
        # Not dimmed (bright name), no pending ellipsis
        assert "bold yellow" in header
        assert "write_file" in header
        assert "…" not in header
        body = widget._content
        assert "test.py" in body
        # Truncated to 200 chars + ellipsis
        assert "..." in body

    def test_complete_no_args_is_name_only(self):
        widget = ToolWidget("command")
        header = widget._title_markup()
        assert "command" in header
        assert "…" not in header
        assert widget._content == ""

    def test_complete_with_seq_numbers_the_header(self):
        widget = ToolWidget("command")
        widget.set_call("command", {"command": "ls -la"}, seq=1)
        assert "⚙ command 1[/]: ls -la" in widget._title_markup()

    def test_complete_without_seq_is_unnumbered(self):
        widget = ToolWidget("command")
        widget.set_call("command", {"command": "ls -la"})
        assert "⚙ command[/]: ls -la" in widget._title_markup()

    def test_result_with_seq_carries_the_call_number(self):
        widget = ToolWidget("command", kind="result")
        widget.set_result("total 8", seq=2)
        assert widget._title_markup() == "[dim]✓ Result 2 (command):[/]"

    def test_result_without_seq_is_unnumbered(self):
        widget = ToolWidget("command", kind="result")
        widget.set_result("total 8")
        assert widget._title_markup() == "[dim]✓ Result (command):[/]"

    def test_format_tool_args_empty_is_blank(self):
        assert format_tool_args({}) == ""

    def test_format_tool_args_truncates_long_values(self):
        body = format_tool_args({"content": "y" * 400})
        assert "..." in body
        assert len(body) < 400


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
    async def test_call_and_result_carry_per_turn_seq(self):
        """A batch of N calls gets seq 1..N on both TOOL_CALL and TOOL_RESULT.

        seq is the call's 1-based number within the user turn. A fresh agent
        starts its counter at 0, so a turn's first batch numbers 1..N. The
        UIs pair a result with its call by this number: a batch's pending
        call widgets mount during the stream, so its results render after all
        of its calls - the number is what keeps them associated.
        """
        agent = _make_agent()
        call_seqs, result_seqs = [], []

        @agent.on_event
        async def record(event):
            if event.event == AgentEvent.TOOL_CALL:
                call_seqs.append(event.data.get("seq"))
            elif event.event == AgentEvent.TOOL_RESULT:
                result_seqs.append(event.data.get("seq"))

        agent._stream_and_emit = self._stateful_stream([
            ("tool_calls_complete", {
                "tool_calls": [
                    {"id": "tc_1", "name": "write_file", "arguments": "{}"},
                    {"id": "tc_2", "name": "write_file", "arguments": "{}"},
                ]
            }),
        ])
        await agent._llm_turn()

        assert call_seqs == [1, 2], f"TOOL_CALL seqs: {call_seqs}"
        assert result_seqs == [1, 2], f"TOOL_RESULT seqs: {result_seqs}"

    @pytest.mark.asyncio
    async def test_seq_continues_across_iterations_in_one_turn(self):
        """Within one user turn, later LLM iterations keep counting.

        _process_item calls _llm_turn repeatedly for a single turn and
        resets the counter once, at the top. So the second iteration's call
        must continue where the first left off - [1] then [2], not [1] then
        [1] (the per-batch behaviour this replaces).
        """
        agent = _make_agent()
        call_seqs = []

        @agent.on_event
        async def record(event):
            if event.event == AgentEvent.TOOL_CALL:
                call_seqs.append(event.data.get("seq"))

        # Iteration 1: one tool call, then the terminal content stream.
        agent._stream_and_emit = self._stateful_stream([
            ("tool_calls_complete", {
                "tool_calls": [
                    {"id": "tc_1", "name": "write_file", "arguments": "{}"},
                ]
            }),
        ])
        await agent._llm_turn()

        # Iteration 2: a fresh _llm_turn in the same turn (no reset), one
        # tool call, then the terminal content stream.
        agent._stream_and_emit = self._stateful_stream([
            ("tool_calls_complete", {
                "tool_calls": [
                    {"id": "tc_2", "name": "write_file", "arguments": "{}"},
                ]
            }),
        ])
        await agent._llm_turn()

        assert call_seqs == [1, 2], (
            f"TOOL_CALL seqs across iterations in one turn: {call_seqs}"
        )

    @pytest.mark.asyncio
    async def test_process_item_resets_the_turn_counter(self):
        """Each user turn (queue item) restarts numbering from 1.

        Simulate a counter left mid-turn by a previous item, then process a
        fresh item: _process_item must zero _turn_tool_seq at its top so the
        new turn's first call numbers 1, not 2.
        """
        from unittest.mock import AsyncMock, patch

        agent = _make_agent()
        agent.messages = []  # no history - keeps the journal a no-op
        agent._turn_tool_seq = 5  # stale counter from a prior turn

        item = type(
            "Item",
            (),
            {
                "text": "Q",
                "id": 1,
                "priority": False,
                "interrupt": False,
                "kind": "prompt",
            },
        )()
        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(item)

        assert agent._turn_tool_seq == 0, (
            f"_process_item must reset the turn counter, got "
            f"{agent._turn_tool_seq}"
        )

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
