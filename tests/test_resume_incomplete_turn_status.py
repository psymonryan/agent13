"""Regression test: /resume on a loaded incomplete turn must not linger on IDLE.

Bug: When a session is loaded via --continue (or /load) with an incomplete
turn, the agent's _incomplete_turn_loaded flag is set but its _status
remains IDLE. Pressing /resume calls Agent.continue_incomplete_turn(),
which previously did NOT transition the status before kicking off work.
The status only changed later, lazily, inside _llm_turn() on the first
reasoning/content/tool_calls_complete event — leaving the UI showing
"idle" for the entire LLM latency window (which can be many seconds,
or 10+ minutes for reasoning models).

Fix: continue_incomplete_turn() now calls _set_status(WAITING) right
after clearing the flag and before any work, mirroring _process_item.
This test pins that behaviour: STATUS_CHANGE to WAITING is observed
synchronously, before _llm_turn() is awaited.

Second case, same family: a mid-turn /resume. _wait_if_paused() used to
report IDLE on resume whenever the queue was empty — correct for the run
loop's between-items pause (the agent then blocks on the queue), but wrong
for the callers that resume *mid-turn* and continue the turn immediately.
Those now pass resume_status=WAITING, so no IDLE is emitted at all.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent, AgentEvent, AgentStatus


class MockClient:
    """Minimal mock AsyncOpenAI client."""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()


def _make_agent_with_incomplete_turn() -> Agent:
    """Build an Agent whose last message is an assistant turn with tool_calls.

    This is the 'incomplete turn' state that --continue / /load can restore.
    """
    client = MockClient()
    agent = Agent(client=client, model="test-model")
    # Simulate a loaded incomplete turn: assistant emitted tool_calls but
    # no tool results follow yet.
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
    # Loaded sessions start with the agent idle and not running its loop.
    # _set_status requires running to emit but is fine to call directly;
    # we mirror real post-load state here.
    return agent


class TestContinueIncompleteTurnStatus:
    """continue_incomplete_turn() must transition out of IDLE synchronously."""

    @pytest.mark.asyncio
    async def test_status_is_waiting_before_llm_turn(self):
        """STATUS_CHANGE to WAITING fires before _llm_turn() is awaited.

        We patch _llm_turn and _execute_tool_async to record the agent's
        status at the moment they're called. If the fix is in place,
        status will be WAITING (not IDLE) by the time either runs.

        This case uses the fallthrough branch (last message is a tool
        result, so get_pending_tool_calls() returns None and we go
        straight to _llm_turn). _running is set True to mirror an
        in-flight agent.run(), which is the realistic post-/resume state.
        """
        agent = _make_agent_with_incomplete_turn()
        # Force the fallthrough branch: last message is a tool result,
        # so get_pending_tool_calls() returns None and has_incomplete_turn()
        # drives the call to _llm_turn().
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
        agent._running = True  # mirror in-flight agent.run() state

        statuses_seen = []
        statuses_at_llm_turn = []

        async def capture_status_change(event):
            if event.event == AgentEvent.STATUS_CHANGE:
                statuses_seen.append(event.data.get("status"))

        agent._handlers.append(capture_status_change)

        async def fake_llm_turn(_polite_acquired=False):
            statuses_at_llm_turn.append(agent.status.value)
            return None

        async def fake_execute_tool(name, args):
            # Should also see WAITING, not IDLE, when tools begin executing.
            return "ok"

        with (
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
        ):
            await agent.continue_incomplete_turn()

        # The WAITING transition was emitted.
        assert "waiting" in statuses_seen, (
            f"Expected STATUS_CHANGE to 'waiting' before work began, "
            f"got: {statuses_seen}"
        )
        # And it was visible by the time _llm_turn actually ran — this is
        # the regression: previously status was still 'idle' here.
        assert statuses_at_llm_turn, "_llm_turn should have been called"
        assert statuses_at_llm_turn[0] == "waiting", (
            f"Expected status to be 'waiting' when _llm_turn ran (before "
            f"any tokens), got: {statuses_at_llm_turn[0]!r}. This is the "
            f"bug: /resume leaves the UI on 'idle' for the LLM latency window."
        )

    @pytest.mark.asyncio
    async def test_status_is_waiting_before_tool_execution(self):
        """For the pending-tools branch, WAITING is set before tools execute.

        The tool-execution branch of continue_incomplete_turn runs each
        pending tool_call via _execute_tool_async before calling _llm_turn.
        The status must already be WAITING when the first tool runs.

        Note: the pending-tools loop guards each iteration on self._running,
        so we set _running=True to actually exercise this branch (mirroring
        what agent.run() does before any turn processing happens).
        """
        agent = _make_agent_with_incomplete_turn()
        agent._running = True  # mirror in-flight agent.run() state

        statuses_at_tool_exec = []

        async def capture(event):
            pass

        agent._handlers.append(capture)

        async def fake_execute_tool(name, args):
            statuses_at_tool_exec.append(agent.status.value)
            return "ok"

        async def fake_llm_turn(_polite_acquired=False):
            return None

        with (
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
        ):
            await agent.continue_incomplete_turn()

        assert statuses_at_tool_exec, "_execute_tool_async should have been called"
        assert statuses_at_tool_exec[0] == "waiting", (
            f"Expected status 'waiting' when first pending tool executed, "
            f"got: {statuses_at_tool_exec[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_no_transition_when_not_incomplete(self):
        """If no incomplete turn is loaded, continue_incomplete_turn is a no-op.

        Returns False and does not emit any STATUS_CHANGE.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.mark_incomplete_turn(False)

        emitted = []

        async def capture(event):
            if event.event == AgentEvent.STATUS_CHANGE:
                emitted.append(event.data.get("status"))

        agent._handlers.append(capture)

        result = await agent.continue_incomplete_turn()

        assert result is False
        assert emitted == [], (
            f"No STATUS_CHANGE should fire when there's no incomplete turn, "
            f"got: {emitted}"
        )


class TestMidTurnResumeStatus:
    """A mid-turn /resume must not report IDLE.

    _wait_if_paused() used to set IDLE on resume whenever the queue was empty.
    Right for the run loop's between-items pause, wrong for the callers that
    resume mid-turn (the /pause safe point in _llm_turn, the pending-tools
    loop in continue_incomplete_turn, the auto-context pause_wait): they
    continue the turn immediately, so the status bar read "idle" while the
    agent was working — and the TUI's idle handling ended the turn (elapsed
    timer reset, bell).

    Fix: those callers pass resume_status=WAITING. The parameter is optional,
    so the run loop keeps its queue-based behaviour.
    """

    @staticmethod
    async def _pause_then_resume(agent, resume_status=None):
        """Drive one pause/resume cycle through _wait_if_paused()."""
        agent._running = True
        assert agent.pause() is True
        task = asyncio.create_task(agent._wait_if_paused(resume_status=resume_status))
        # Let the coroutine promote PAUSING -> PAUSED and block on the event.
        for _ in range(5):
            await asyncio.sleep(0)
        assert agent.is_paused, "should have reached the PAUSED transition"
        agent.resume()
        await task

    @pytest.mark.asyncio
    async def test_mid_turn_resume_reports_waiting_not_idle(self):
        """resume_status=WAITING: IDLE is never emitted."""
        agent = Agent(client=MockClient(), model="test-model")
        seen = []

        async def capture(event):
            if event.event == AgentEvent.STATUS_CHANGE:
                seen.append(event.data.get("status"))

        agent._handlers.append(capture)

        await self._pause_then_resume(agent, resume_status=AgentStatus.WAITING)

        assert agent.status is AgentStatus.WAITING
        assert seen == ["paused", "waiting"], (
            f"Expected paused -> waiting with no idle, got: {seen}"
        )

    @pytest.mark.asyncio
    async def test_explicit_resume_status_wins_over_queue(self):
        """A queued item must not override the caller's explicit status."""
        agent = Agent(client=MockClient(), model="test-model")
        agent.queue = MagicMock()
        agent.queue.pending_count = 3

        await self._pause_then_resume(agent, resume_status=AgentStatus.WAITING)

        assert agent.status is AgentStatus.WAITING

    @pytest.mark.asyncio
    async def test_default_resume_keeps_queue_based_behaviour(self):
        """Without resume_status the run loop's rule is unchanged."""
        idle_agent = Agent(client=MockClient(), model="test-model")
        idle_agent.queue = MagicMock()
        idle_agent.queue.pending_count = 0
        await self._pause_then_resume(idle_agent)
        assert idle_agent.status is AgentStatus.IDLE

        busy_agent = Agent(client=MockClient(), model="test-model")
        busy_agent.queue = MagicMock()
        busy_agent.queue.pending_count = 2
        await self._pause_then_resume(busy_agent)
        assert busy_agent.status is AgentStatus.WAITING

    @pytest.mark.asyncio
    async def test_pending_tools_safe_point_passes_resume_status(self):
        """continue_incomplete_turn's pending-tools safe point passes WAITING.

        That loop resumes mid-turn into _llm_turn, so a bare
        _wait_if_paused() would report IDLE on resume.
        """
        agent = _make_agent_with_incomplete_turn()
        agent._running = True
        seen = []

        async def fake_wait_if_paused(resume_status=None):
            seen.append(resume_status)

        async def fake_execute_tool(name, args):
            return "ok"

        async def fake_llm_turn(_polite_acquired=False):
            return None

        agent._wait_if_paused = fake_wait_if_paused

        with (
            patch.object(agent, "_execute_tool_async", side_effect=fake_execute_tool),
            patch.object(agent, "_llm_turn", side_effect=fake_llm_turn),
        ):
            await agent.continue_incomplete_turn()

        assert seen == [AgentStatus.WAITING], (
            f"the pending-tools safe point must resume as WAITING, got: {seen}"
        )
