"""Tests for polite-mode + pause behaviour in continue_incomplete_turn().

Covers the fix for two bugs in continue_incomplete_turn():

1. Polite bypass: the function previously called _llm_turn() with no
   _polite_acquired argument (default False), so a --continue + /resume of
   an incomplete turn silently skipped polite lock coordination — the GPU
   could be stolen mid-turn and the LLM stream ran without the lock.

2. Missing pause safe-points: the pending-tools loop had no
   _wait_if_paused() call, so /pause during the loop kept running to
   completion before the pause took effect (handover's open finding).

Fix: continue_incomplete_turn() now acquires the polite lock before work
(mirroring run()), forwards _polite_acquired to _llm_turn(), adds a pause
safe-point after each tool result, and releases the lock in a finally.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent, PauseState


class MockClient:
    """Minimal mock AsyncOpenAI client."""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()
        # base_url is read by set_polite() to key the lock file.
        self.base_url = "http://test.local/v1"


def _make_agent_with_pending_tools() -> Agent:
    """Build an Agent whose last message is an assistant turn with tool_calls.

    This is the 'incomplete turn' state (case 1: pending tools) that
    --continue / /load can restore.
    """
    client = MockClient()
    agent = Agent(client=client, model="test-model")
    agent.messages = [
        {"role": "user", "content": "please run the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "name": "noop",
                    "arguments": json.dumps({}),
                }
            ],
        },
    ]
    agent.mark_incomplete_turn(True)
    return agent


def _make_agent_with_tool_result() -> Agent:
    """Build an Agent whose last message is a tool result (case 2).

    The fallthrough branch: get_pending_tool_calls() returns None and
    has_incomplete_turn() drives the call to _llm_turn().
    """
    client = MockClient()
    agent = Agent(client=client, model="test-model")
    agent.messages = [
        {"role": "user", "content": "please run the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "name": "noop",
                    "arguments": json.dumps({}),
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    agent.mark_incomplete_turn(True)
    return agent


class TestPoliteContinueIncompleteTurn:
    """continue_incomplete_turn() honours polite mode and pause safe-points."""

    @pytest.mark.asyncio
    async def test_tool_branch_acquires_polite_lock(self, tmp_path, monkeypatch):
        """Pending-tools branch: lock acquired before _llm_turn, released after.

        Without the fix, _polite_acquired defaults to False and the lock is
        never acquired — the polite guarantee is silently violated.
        """
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_pending_tools()
        agent._running = True  # mirror in-flight agent.run() state
        agent.set_polite(interval=0.01)

        lock_states = []

        async def fake_llm_turn(_polite_acquired=False):
            lock_states.append(
                ("at_llm_turn", _polite_acquired, agent.polite_lock.is_held())
            )

        async def fake_execute_tool(name, args):
            return "ok"

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
        ):
            await agent.continue_incomplete_turn()

        # The lock was acquired and _polite_acquired was True when _llm_turn ran.
        assert lock_states, "_llm_turn should have been called"
        _, acquired_flag, held = lock_states[0]
        assert acquired_flag is True, (
            f"_polite_acquired should be True when polite mode is on, got {acquired_flag!r}"
        )
        assert held is True, (
            "Lock should be held when _llm_turn runs (it releases inside _llm_turn)"
        )
        # Safety net released the lock after _llm_turn returned.
        assert not agent.polite_lock.is_held(), (
            "Lock should be released after continue_incomplete_turn completes"
        )

    @pytest.mark.asyncio
    async def test_tool_result_branch_acquires_polite_lock(self, tmp_path, monkeypatch):
        """Tool-result branch (case 2): lock acquired before _llm_turn, released after."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_tool_result()
        agent._running = True
        agent.set_polite(interval=0.01)

        lock_states = []

        async def fake_llm_turn(_polite_acquired=False):
            lock_states.append(
                ("at_llm_turn", _polite_acquired, agent.polite_lock.is_held())
            )

        with patch.object(agent, "_llm_turn", side_effect=fake_llm_turn):
            await agent.continue_incomplete_turn()

        assert lock_states, "_llm_turn should have been called"
        _, acquired_flag, held = lock_states[0]
        assert acquired_flag is True, (
            f"_polite_acquired should be True, got {acquired_flag!r}"
        )
        assert held is True, "Lock should be held when _llm_turn runs"
        assert not agent.polite_lock.is_held(), (
            "Lock should be released after completion"
        )

    @pytest.mark.asyncio
    async def test_no_lock_when_polite_off(self, tmp_path, monkeypatch):
        """Polite mode off: no lock acquired, _polite_acquired stays False."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_pending_tools()
        agent._running = True
        # polite mode NOT enabled

        acquired_flags = []

        async def fake_llm_turn(_polite_acquired=False):
            acquired_flags.append(_polite_acquired)

        async def fake_execute_tool(name, args):
            return "ok"

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
        ):
            await agent.continue_incomplete_turn()

        assert acquired_flags == [False], (
            f"_polite_acquired should be False when polite is off, got {acquired_flags}"
        )
        assert agent.polite_lock is None

    @pytest.mark.asyncio
    async def test_pause_blocks_pending_tools_loop(self, tmp_path, monkeypatch):
        """Pause during the pending-tools loop blocks at the safe-point.

        This is the handover's open finding: without _wait_if_paused() after
        each tool result, /pause shows "Paused" in the TUI but the loop keeps
        running until _llm_turn's own safe point.
        """
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_pending_tools()
        agent._running = True
        agent.set_polite(interval=0.01)

        # Two pending tools so the loop iterates and we can pause between them.
        agent.messages[1]["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "name": "noop",
                "arguments": json.dumps({}),
            },
            {
                "id": "call_2",
                "type": "function",
                "name": "noop",
                "arguments": json.dumps({}),
            },
        ]

        tool_call_count = [0]
        resumed = asyncio.Event()

        async def fake_execute_tool(name, args):
            tool_call_count[0] += 1
            # After the first tool completes, request a pause. The safe-point
            # after the tool result should block before the second tool runs.
            if tool_call_count[0] == 1:
                agent.pause()  # PAUSING state
                # Schedule a resume shortly after, so the test doesn't hang.

                async def _resume():
                    await asyncio.sleep(0.05)
                    agent.resume()
                    resumed.set()

                asyncio.create_task(_resume())
            return "ok"

        async def fake_llm_turn(_polite_acquired=False):
            return None

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
        ):
            await agent.continue_incomplete_turn()

        # Both tools ran, and the pause was observed (resume fired).
        assert tool_call_count[0] == 2, "Both pending tools should have executed"
        assert resumed.is_set(), "Resume should have fired after the pause"
        # The pause state was actually entered (not just requested).
        # After resume, _wait_if_paused transitions back to RUNNING.
        assert agent.pause_state == PauseState.RUNNING, (
            f"Pause state should be RUNNING after resume, got {agent.pause_state}"
        )

    @pytest.mark.asyncio
    async def test_lock_released_on_llm_turn_error(self, tmp_path, monkeypatch):
        """Safety net releases the lock if _llm_turn raises.

        Mirrors test_lock_safety_net_on_error in test_polite_integration.py,
        but for the continue_incomplete_turn path.
        """
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_tool_result()
        agent._running = True
        agent.set_polite(interval=0.01)

        async def failing_llm_turn(_polite_acquired=False):
            raise RuntimeError("LLM exploded")

        with patch.object(agent, "_llm_turn", side_effect=failing_llm_turn):
            with pytest.raises(RuntimeError, match="LLM exploded"):
                await agent.continue_incomplete_turn()

        assert not agent.polite_lock.is_held(), (
            "Lock should be released by the finally safety net after an error"
        )

    @pytest.mark.asyncio
    async def test_lock_released_on_tool_execution_error(self, tmp_path, monkeypatch):
        """Safety net releases the lock if a tool raises in the pending-tools loop."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_pending_tools()
        agent._running = True
        agent.set_polite(interval=0.01)

        async def failing_execute_tool(name, args):
            raise RuntimeError("tool exploded")

        async def fake_llm_turn(_polite_acquired=False):
            return None

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(
                agent, "_execute_tool_async", side_effect=failing_execute_tool
            ),
        ):
            with pytest.raises(RuntimeError, match="tool exploded"):
                await agent.continue_incomplete_turn()

        assert not agent.polite_lock.is_held(), (
            "Lock should be released by the finally safety net after a tool error"
        )


class TestContinueIncompleteRunningState:
    """continue_incomplete_turn() must set _running = True.

    Bug: continue_incomplete_turn() never set _running = True, relying on
    run() having set it. When called from run()'s pre-loop check this is
    fine, but direct callers (e.g. tests) may not have it set. More
    importantly, _llm_turn's `while self._running` loop silently does
    nothing if _running is False.
    """

    @pytest.mark.asyncio
    async def test_running_true_during_turn_even_if_was_false(
        self, tmp_path, monkeypatch
    ):
        """_running is True inside _llm_turn even if it was False on entry."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_tool_result()
        agent._running = False
        agent.mark_incomplete_turn(True)

        running_states = []

        async def fake_llm_turn(_polite_acquired=False):
            running_states.append(agent._running)

        with patch.object(agent, "_llm_turn", side_effect=fake_llm_turn):
            await agent.continue_incomplete_turn()

        assert running_states == [True], (
            f"_running should be True inside _llm_turn, got {running_states}"
        )


class TestMarkerPop:
    """continue_incomplete_turn() drops a trailing [Interrupted] marker.

    The marker exists only to keep an in-process history API-valid after an
    ESC. On resume the turn must continue from the REAL last message (tool
    result or user message) — an assistant message at the tail is an invalid
    last role for the next LLM call.
    """

    def _make_agent_with_marker_tail(self, tail_before_marker: list[dict]) -> Agent:
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.messages = tail_before_marker + [
            {"role": "assistant", "content": "[Interrupted]", "interrupt": True}
        ]
        agent.mark_incomplete_turn(True)
        return agent

    @pytest.mark.asyncio
    async def test_marker_popped_tool_tail(self, tmp_path, monkeypatch):
        """[user, assistant+tc, tool, marker] -> LLM sees [user, assistant+tc, tool]."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = self._make_agent_with_marker_tail(
            [
                {"role": "user", "content": "please run the thing"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "name": "noop",
                            "arguments": json.dumps({}),
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        )
        agent._running = True

        seen = []

        async def fake_llm_turn(_polite_acquired=False):
            seen.append([dict(m) for m in agent.messages])

        with patch.object(agent, "_llm_turn", side_effect=fake_llm_turn):
            started = await agent.continue_incomplete_turn()

        assert started is True
        assert seen, "_llm_turn should have been called"
        msgs = seen[0]
        assert not any(m.get("interrupt") for m in msgs), (
            f"marker must be popped before the LLM call: {msgs[-1]!r}"
        )
        assert msgs[-1]["role"] == "tool", (
            f"turn should continue from the real last message, got {msgs[-1]!r}"
        )

    @pytest.mark.asyncio
    async def test_marker_popped_user_tail(self, tmp_path, monkeypatch):
        """[user, marker] (quit/ESC while streaming) -> LLM sees [user]."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = self._make_agent_with_marker_tail(
            [
                {"role": "user", "content": "hi"},
            ]
        )
        agent._running = True

        seen = []

        async def fake_llm_turn(_polite_acquired=False):
            seen.append([dict(m) for m in agent.messages])

        with patch.object(agent, "_llm_turn", side_effect=fake_llm_turn):
            started = await agent.continue_incomplete_turn()

        assert started is True
        assert seen, "_llm_turn should have been called"
        assert seen[0] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_history_without_marker_untouched(self, tmp_path, monkeypatch):
        """No marker at the tail: nothing is popped (normal /resume path)."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "please run the thing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "name": "noop",
                        "arguments": json.dumps({}),
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        agent.mark_incomplete_turn(True)
        agent._running = True

        seen = []

        async def fake_llm_turn(_polite_acquired=False):
            seen.append([dict(m) for m in agent.messages])

        with patch.object(agent, "_llm_turn", side_effect=fake_llm_turn):
            started = await agent.continue_incomplete_turn()

        assert started is True
        assert seen, "_llm_turn should have been called"
        assert len(seen[0]) == 3, "no message may be dropped without a marker"
        assert seen[0][-1]["role"] == "tool"


class TestContinueIncompleteJournalReflection:
    """continue_incomplete_turn() runs the post-turn journal reflection.

    The resume path has no queue item, so _process_item's Stage 12
    (journal.maybe_reflect_after_turn) never runs for a resumed turn.
    Without the equivalent call here, a resumed turn's tool work stays
    un-compacted and the NEXT turn's retrospective_compact fires the
    reflection before the user's new message is processed (10-06 edmac
    incident: 163k-token prefill, 10-min delay on a simple question).
    The call is unconditional — maybe_reflect_after_turn's internal
    guards (journal mode, tool calls, interrupt) make it a no-op when
    not applicable.
    """

    @pytest.mark.asyncio
    async def test_tool_branch_reflects_after_llm_turn(self, tmp_path, monkeypatch):
        """Case 1 (pending tools): reflection runs after _llm_turn, not before."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_pending_tools()
        agent._running = True

        order = []
        reflect = AsyncMock()

        async def fake_llm_turn(_polite_acquired=False):
            order.append("llm_turn")

        async def fake_execute_tool(name, args):
            return "ok"

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
            patch.object(
                agent.journal, "maybe_reflect_after_turn", side_effect=reflect
            ),
        ):
            started = await agent.continue_incomplete_turn()

        assert started is True
        reflect.assert_awaited_once()
        order.append("reflect")
        assert order == ["llm_turn", "reflect"], (
            f"reflection must run after the resumed turn completes, got {order}"
        )

    @pytest.mark.asyncio
    async def test_tool_result_branch_reflects_after_llm_turn(self, tmp_path, monkeypatch):
        """Case 2 (tool result tail): reflection runs after _llm_turn."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        agent = _make_agent_with_tool_result()
        agent._running = True

        order = []
        reflect = AsyncMock()

        async def fake_llm_turn(_polite_acquired=False):
            order.append("llm_turn")

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(
                agent.journal, "maybe_reflect_after_turn", side_effect=reflect
            ),
        ):
            started = await agent.continue_incomplete_turn()

        assert started is True
        reflect.assert_awaited_once()
        order.append("reflect")
        assert order == ["llm_turn", "reflect"], (
            f"reflection must run after the resumed turn completes, got {order}"
        )

    @pytest.mark.asyncio
    async def test_no_reflection_when_nothing_to_continue(self, tmp_path, monkeypatch):
        """No incomplete turn: returns False and never reflects."""
        monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent.mark_incomplete_turn(False)
        agent._running = True

        reflect = AsyncMock()
        with patch.object(
            agent.journal, "maybe_reflect_after_turn", side_effect=reflect
        ):
            started = await agent.continue_incomplete_turn()

        assert started is False
        reflect.assert_not_awaited()
