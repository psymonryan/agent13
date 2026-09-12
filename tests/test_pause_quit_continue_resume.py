"""Unit tests for the pause→quit→continue→resume fix (StopReason + JIT repair).

Covers:
- stop() records its StopReason; ESC-interrupt vs quit gating of history repair
  (Fix 1 — quit must NOT corrupt a mid-turn history before auto-save).
- JIT repair safety net in _stream_and_emit (Fix 3): a loaded incomplete
  context is closed only when the API requires it (pending tool_calls), and a
  trailing tool result is left untouched (tool-then-user is already valid).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent13 import Agent, AgentEvent
from agent13.core import StopReason
from agent13.persistence import save_context, load_context, get_auto_save_path


class MockClient:
    """Mock OpenAI client for testing."""


def make_agent():
    return Agent(MagicMock(), model="test-model")


# =============================================================================
# Fix 1: stop() reason plumbing
# =============================================================================


class TestStopReason:
    def test_stop_defaults_to_interrupt(self):
        agent = make_agent()
        agent.stop()
        assert agent._stop_reason == StopReason.INTERRUPT

    def test_stop_quit_records_quit(self):
        agent = make_agent()
        agent.stop(StopReason.QUIT)
        assert agent._stop_reason == StopReason.QUIT
        assert agent._running is False

    def test_stop_reason_defaults_interrupt_at_construction(self):
        agent = make_agent()
        # A bare CancelledError path with no stop() call ever made
        # (e.g. external task.cancel() with _running True) must repair.
        assert agent._stop_reason == StopReason.INTERRUPT


# =============================================================================
# Fix 1: quit must not corrupt mid-turn history
# =============================================================================


class TestQuitPreservesMidTurnHistory:
    @pytest.mark.asyncio
    async def test_quit_cancel_does_not_repair(self):
        """Quit-path cancel leaves trailing tool result un-repaired.

        The auto-save then records incomplete_turn=true so --continue +
        /resume can resume the turn with the identical message prefix.
        """
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "square 3"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "9"},
            ]
        )

        task = asyncio_create_run_task(agent)
        await wait_until_running(agent, task)
        agent.stop(StopReason.QUIT)
        await asyncio_wait_cancelled(task)

        # History untouched: NO [Interrupted] marker appended
        assert agent.messages[-1]["role"] == "tool"
        assert not any(m.get("content") == "[Interrupted]" for m in agent.messages)

    @pytest.mark.asyncio
    async def test_interrupt_cancel_still_repairs(self):
        """ESC-path cancel appends the [Interrupted] marker (unchanged)."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "hi"},
            ]
        )

        task = asyncio_create_run_task(agent)
        await wait_until_running(agent, task)
        agent.stop(StopReason.INTERRUPT)
        await asyncio_wait_cancelled(task)

        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == "[Interrupted]"

    @pytest.mark.asyncio
    async def test_quit_cancel_with_pending_tools_no_repair(self):
        """Cancel while tool_calls are pending (repair case 2) — quit keeps them."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
            ]
        )

        task = asyncio_create_run_task(agent)
        await wait_until_running(agent, task)
        agent.stop(StopReason.QUIT)
        await asyncio_wait_cancelled(task)

        # No synthesized tool results, no marker
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1].get("tool_calls") is not None

    @pytest.mark.asyncio
    async def test_quit_before_run_starts_is_honoured(self):
        """stop(QUIT) in the create_task()→run() window still suppresses repair.

        The quit sequence (stop → cancel) can fire before run() executes its
        first line. run() detects the pending stop via _stop_event and must
        honour the recorded reason instead of resetting it to INTERRUPT.
        """
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
            ]
        )

        # stop() BEFORE the task runs — the scheduling-window race
        agent.stop(StopReason.QUIT)
        task = asyncio_create_run_task(agent)
        import asyncio

        await asyncio.sleep(0.05)  # let run() reach the loop (inside try)
        await asyncio_wait_cancelled(task)

        # History untouched: tools still pending, NO [Interrupted] marker
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1].get("tool_calls") is not None
        assert not any(m.get("content") == "[Interrupted]" for m in agent.messages)

    @pytest.mark.asyncio
    async def test_quit_save_preserves_incomplete_flag(self):
        """Full quit→save cycle: trailing tool result saves incomplete=true."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "square 3"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "9"},
            ]
        )
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"
        task = asyncio_create_run_task(agent)
        await wait_until_running(agent, task)
        agent.stop(StopReason.QUIT)
        await asyncio_wait_cancelled(task)

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AGENT13_SAVES_DIR"] = tmp
            try:
                path = get_auto_save_path(session_date="2026-09-08")
                save_context(agent, path)
                data = json.loads(open(path).read())
                assert data["incomplete_turn"] is True
            finally:
                os.environ.pop("AGENT13_SAVES_DIR", None)

    @pytest.mark.asyncio
    async def test_quit_save_user_tail_preserves_incomplete_flag(self):
        """Quit while the assistant is streaming (history ends on a user
        message) saves incomplete=true — so --continue + /resume answers
        the pending user message instead of producing consecutive users."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "square 3"},
                {"role": "assistant", "content": "9"},
                {"role": "user", "content": "and 4?"},  # quit mid-stream here
            ]
        )
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AGENT13_SAVES_DIR"] = tmp
            try:
                path = get_auto_save_path(session_date="2026-09-08")
                save_context(agent, path)
                data = json.loads(open(path).read())
                assert data["incomplete_turn"] is True
            finally:
                os.environ.pop("AGENT13_SAVES_DIR", None)


# =============================================================================
# Fix 3: JIT repair before API calls
# =============================================================================


class TestJITRepair:
    @pytest.mark.asyncio
    async def test_new_message_after_load_repairs_pending_tools(self):
        """New user message after --continue with pending tool_calls:
        repair happens at stream time (marker synthesized), stream proceeds."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "square 3"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
            ]
        )
        agent.mark_incomplete_turn(True)
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"

        seen = {}

        async def fake_stream(
            client,
            model,
            messages,
            system_prompt,
            tools,
            tool_choice=None,
            session_date=None,
        ):
            seen["messages"] = [dict(m) for m in messages]
            seen["tool_choice"] = tool_choice
            yield "content", "ok"
            yield "finish", {"finish_reason": "stop"}
            yield (
                "token_usage",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch("agent13.llm.stream_response_with_tools", fake_stream):
            agent._running = True
            await agent._llm_turn()
            agent._running = False

        # Repair happened: tool result + [Interrupted] marker present
        msgs = seen["messages"]
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] == "[Interrupted]"
        tool_msgs = [
            m
            for m in msgs
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
        ]
        assert len(tool_msgs) == 1
        # Flag cleared
        assert agent.has_incomplete_turn is False

    @pytest.mark.asyncio
    async def test_new_message_after_load_trailing_tool_no_repair(self):
        """Trailing tool result + new user message is already API-valid:
        NO marker inserted, kv-cache prefix preserved."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "square 3"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "square_number",
                                "arguments": '{"x": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "9"},
            ]
        )
        agent.mark_incomplete_turn(True)
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"

        seen = {}

        async def fake_stream(
            client,
            model,
            messages,
            system_prompt,
            tools,
            tool_choice=None,
            session_date=None,
        ):
            seen["messages"] = [dict(m) for m in messages]
            yield "content", "ok"
            yield "finish", {"finish_reason": "stop"}
            yield (
                "token_usage",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch("agent13.llm.stream_response_with_tools", fake_stream):
            agent._running = True
            await agent._llm_turn()
            agent._running = False

        # No [Interrupted] marker — history untouched
        assert not any(m.get("content") == "[Interrupted]" for m in seen["messages"])
        assert agent.has_incomplete_turn is False

    @pytest.mark.asyncio
    async def test_new_message_after_load_user_tail_repairs(self):
        """User tail (quit while streaming) + new user message: [Interrupted]
        marker inserted so roles alternate — no consecutive users sent."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "and now?"},
            ]
        )
        agent.mark_incomplete_turn(True)
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"

        seen = {}

        async def fake_stream(
            client,
            model,
            messages,
            system_prompt,
            tools,
            tool_choice=None,
            session_date=None,
        ):
            seen["messages"] = [dict(m) for m in messages]
            yield "content", "ok"
            yield "finish", {"finish_reason": "stop"}
            yield (
                "token_usage",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch("agent13.llm.stream_response_with_tools", fake_stream):
            agent._running = True
            await agent._llm_turn()
            agent._running = False

        msgs = seen["messages"]
        # Marker closed the dangling turn before the send
        assert any(
            m["role"] == "assistant" and m.get("content") == "[Interrupted]"
            for m in msgs
        )
        # No consecutive user messages in what was sent
        for a, b in zip(msgs, msgs[1 :]):
            assert not (a["role"] == "user" and b["role"] == "user")
        # Flag cleared
        assert agent.has_incomplete_turn is False

    @pytest.mark.asyncio
    async def test_no_repair_when_flag_not_set(self):
        """A normal new message (no loaded incomplete context) never repairs."""
        agent = make_agent()
        agent.messages.append({"role": "user", "content": "hi"})
        agent.system_prompt = "sys"

        seen = {}

        async def fake_stream(
            client,
            model,
            messages,
            system_prompt,
            tools,
            tool_choice=None,
            session_date=None,
        ):
            seen["messages"] = [dict(m) for m in messages]
            yield "content", "ok"
            yield "finish", {"finish_reason": "stop"}
            yield (
                "token_usage",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch("agent13.llm.stream_response_with_tools", fake_stream):
            agent._running = True
            await agent._llm_turn()
            agent._running = False

        assert not any(m.get("content") == "[Interrupted]" for m in seen["messages"])
        assert seen["messages"][-1]["content"] == "hi"


# =============================================================================
# /resume (continue_incomplete_turn) on a user-tail history
# =============================================================================


class TestContinueUserTail:
    @pytest.mark.asyncio
    async def test_continue_user_tail_answers_pending_message(self):
        """/resume on a user-tail history (quit while streaming) calls the LLM
        to answer the pending user message — history sent unmodified
        (kv-cache prefix preserved)."""
        agent = make_agent()
        agent.messages.extend(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "and now?"},
            ]
        )
        agent.mark_incomplete_turn(True)
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"

        seen = {}

        async def fake_stream(
            client,
            model,
            messages,
            system_prompt,
            tools,
            tool_choice=None,
            session_date=None,
        ):
            seen["messages"] = [dict(m) for m in messages]
            yield "content", "ok"
            yield "finish", {"finish_reason": "stop"}
            yield (
                "token_usage",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

        with patch("agent13.llm.stream_response_with_tools", fake_stream):
            agent._running = True
            started = await agent.continue_incomplete_turn()
            agent._running = False

        assert started is True
        # Flag cleared
        assert agent.has_incomplete_turn is False
        # LLM was sent the pending user message, unmodified
        assert seen["messages"][-1] == {"role": "user", "content": "and now?"}
        # NO [Interrupted] marker — /resume continues, it doesn't close
        assert not any(m.get("content") == "[Interrupted]" for m in seen["messages"])
        # Reply appended to history
        assert agent.messages[-1]["role"] == "assistant"


# =============================================================================
# load_context flag semantics
# =============================================================================


class TestLoadContextFlag:
    def test_load_sets_flag_true(self, tmp_path):
        agent = make_agent()
        agent.system_prompt = "sys"
        agent.session_date = "2026-09-08"
        path = tmp_path / "s.ctx"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "messages": [
                        {"role": "user", "content": "u"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "square_number",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    ],
                    "incomplete_turn": True,
                }
            )
        )
        ok, msg, incomplete = load_context(agent, path)
        assert ok and incomplete is True
        assert agent.has_incomplete_turn is True

    def test_load_clears_stale_flag(self, tmp_path):
        """Loading a COMPLETE context after an incomplete one clears the flag."""
        agent = make_agent()
        agent.system_prompt = "sys"
        agent.mark_incomplete_turn(True)

        path = tmp_path / "s.ctx"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                    "incomplete_turn": False,
                }
            )
        )
        ok, msg, incomplete = load_context(agent, path)
        assert ok and incomplete is False
        assert agent.has_incomplete_turn is False


# =============================================================================
# helpers
# =============================================================================


def asyncio_create_run_task(agent):
    import asyncio

    return asyncio.get_event_loop().create_task(agent.run())


async def wait_until_running(agent, task):
    """Wait until run() has entered its while loop (inside the try block).

    STARTED is emitted just before try:; one loop-sleep afterwards guarantees
    the cancel lands inside the loop, not during pre-try setup.
    """
    import asyncio

    started = asyncio.Event()

    @agent.on_event
    async def on_started(event):
        if event.event == AgentEvent.STARTED:
            started.set()

    await asyncio.wait_for(started.wait(), timeout=2.0)
    await asyncio.sleep(0.08)  # > run-loop's 0.05s between-items sleep


async def asyncio_wait_cancelled(task):
    task.cancel()
    try:
        await task
    except BaseException:
        pass
