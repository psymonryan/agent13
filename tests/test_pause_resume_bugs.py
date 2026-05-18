"""Tests for pause/resume state machine bugs.

Bug #1 (wishlist line 29): /pause says "paused" but a bash tool keeps running.
Bug #2 (wishlist line 28): After an error, /resume says "not paused".
    Resolution: We no longer pause after errors. /resume is for
    user-initiated pauses; after an error the agent returns to IDLE
    and the user should /retry directly.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent13.core import Agent, AgentEvent, AgentStatus, PauseState


class MockClient:
    """Minimal mock AsyncOpenAI client."""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()


def _make_tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Create a tool_call dict matching the OpenAI format."""
    return {
        "id": call_id,
        "type": "function",
        "name": name,
        "arguments": json.dumps(args),
    }


class TestPauseResumeAfterError:
    """After an error, the agent returns to IDLE (not PAUSED).

    Previously we paused after error so /resume would work, but /resume
    alone doesn't re-attempt the failed turn — the user needs /retry
    anyway. Pausing just adds a useless intermediate step.
    Now: error → IDLE, user can /retry directly.
    """

    @pytest.mark.asyncio
    async def test_agent_is_idle_after_api_error(self):
        """After an API error, agent should be idle (not paused).

        Pausing after error doesn't help — /resume alone doesn't
        re-attempt the turn. The user needs /retry which works from
        IDLE state.
        """
        client = MockClient()
        # Make the API fail
        client.chat.completions.create = AsyncMock(
            side_effect=Exception("context_length_exceeded")
        )

        agent = Agent(client=client, model="test-model")

        # Track events
        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        # Add a message and run
        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())

        # Wait for the error to occur
        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(e.event == AgentEvent.ERROR for e in events):
                break

        # Stop the agent
        agent.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify we got an ERROR event
        error_events = [e for e in events if e.event == AgentEvent.ERROR]
        assert len(error_events) > 0, "Should have received an ERROR event"

        # After error, agent should be IDLE (not paused)
        assert not agent.is_paused, (
            "After an API error, agent should be IDLE so /retry works directly. "
            f"Got is_paused={agent.is_paused}, status={agent.status}"
        )

    @pytest.mark.asyncio
    async def test_agent_status_is_idle_after_api_error(self):
        """After an API error, agent status should be IDLE, not PAUSED.

        PAUSED is for user-initiated pauses. After an error the agent
        should return to IDLE so the user can /retry immediately.
        """
        client = MockClient()
        client.chat.completions.create = AsyncMock(
            side_effect=Exception("context_length_exceeded")
        )

        agent = Agent(client=client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())

        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(e.event == AgentEvent.ERROR for e in events):
                break

        # Check status BEFORE stopping — stop() would reset state
        error_events = [e for e in events if e.event == AgentEvent.ERROR]
        assert len(error_events) > 0

        assert agent.status == AgentStatus.IDLE, (
            f"After API error, status should be IDLE, got {agent.status}"
        )

        agent.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_resume_returns_false_after_api_error(self):
        """After an API error, resume() should return False (agent is not paused).

        The user should use /retry instead, which works from IDLE state.
        """
        client = MockClient()
        client.chat.completions.create = AsyncMock(
            side_effect=Exception("context_length_exceeded")
        )

        agent = Agent(client=client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())

        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(e.event == AgentEvent.ERROR for e in events):
                break

        agent.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # After error, agent is IDLE so resume() returns False
        result = agent.resume()
        assert result is False, (
            "After API error, resume() should return False (agent is IDLE, not paused). "
            "Use /retry instead."
        )


class TestPauseStateConsistency:
    """Tests that pause state transitions are consistent.

    When pause() is called, _pausing is set True and _paused remains False.
    Only when _wait_if_paused() runs does _pausing→_paused happen.
    This is the core mechanism for Bug #1 — the agent correctly distinguishes
    "pausing" (waiting for safe point) from "paused" (at safe point).
    The TUI bug is setting _paused=True immediately on /pause.
    """

    def test_pause_sets_pausing_not_paused(self):
        """Calling pause() should set is_pausing=True, is_paused=False."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        result = agent.pause()

        assert result is True, "pause() should return True on first pause"
        assert agent.is_pausing, "is_pausing should be True after pause()"
        assert not agent.is_paused, "is_paused should be False after pause()"

    def test_double_pause_returns_false(self):
        """Calling pause() when already pausing should return False.

        Currently pause() only checks _paused, not _pausing, so a second
        pause() returns True. This is a minor inconsistency — pause() should
        probably return False if already pausing OR paused.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        agent.pause()
        result = agent.pause()

        # Currently returns True because pause() only checks _paused.
        # After fix: should return False since we're already pausing.
        assert result is False, (
            "Second pause() should return False when already pausing. "
            "Currently returns True because pause() only checks _paused, "
            "not _pausing."
        )

    def test_pause_then_paused_is_consistent(self):
        """After pause() and _wait_if_paused(), is_paused should be True."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        agent.pause()
        # Simulate _wait_if_paused() transition
        agent._pause_state = PauseState.PAUSED

        assert not agent.is_pausing, "is_pausing should be False after transition"
        assert agent.is_paused, "is_paused should be True after transition"

    @pytest.mark.asyncio
    async def test_pausing_cleared_after_cancel_and_restart(self):
        """After task cancellation during pausing, run() should clear stale pause state.

        When run() is called after an interrupt, it should clear both _paused
        and _pausing so the restart is clean. Previously:
        - run() only checked/cleared _paused (not _pausing)
        - _pausing survived restart, so _wait_if_paused() promoted it to _paused
        - The agent ended up unexpectedly paused after an interrupt+restart
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Simulate pausing state directly (without hitting the error path)
        agent._pause_state = PauseState.PAUSING
        agent._pause_event.clear()
        assert agent._pause_state == PauseState.PAUSING

        # Cancel the task (simulating ESC/interrupt during pausing)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # pause_state is still PAUSING after cancellation
        assert agent._pause_state == PauseState.PAUSING

        # Now restart the agent
        task2 = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # After restart, should NOT be paused or pausing
        # Fix C: run() now clears pause_state to RUNNING unconditionally
        assert not agent.is_paused, (
            f"After interrupt+restart during pausing, is_paused should be False. "
            f"Got pause_state={agent.pause_state}"
        )
        assert not agent.is_pausing, (
            f"After interrupt+restart during pausing, is_pausing should be False. "
            f"Got pause_state={agent.pause_state}"
        )

        # Clean up
        agent.stop()
        try:
            await task2
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_paused_cleared_after_cancel_and_restart(self):
        """After task cancellation while fully paused, run() clears _paused on restart.

        This is the scenario: pause completed (agent is_paused=True), then
        user hits ESC/Ctrl+C. The cancelled task ends. When run() is called
        for restart, it should clear _paused so the agent starts fresh.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        # Manually set paused state (simulating completed pause)
        agent._pause_state = PauseState.PAUSED
        agent._pause_event.clear()

        # Start run() — it should detect PAUSED and clear it
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # After restart, paused state should be cleared
        # (run() clears pause_state to RUNNING on restart)
        assert not agent.is_paused, (
            f"After restart while paused, is_paused should be False. "
            f"Got pause_state={agent.pause_state}"
        )

        agent.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass
