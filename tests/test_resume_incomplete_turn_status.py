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
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent, AgentEvent


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
