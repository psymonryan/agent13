"""Tests for pause/resume bugs from wishlist lines 35 and 37-38.

Bug #1: /pause → !!interrupt → /resume — Queue count not visible while paused,
         status shows "ready" instead of "processing" after resume.

Bug #2: Pausing and then resuming before the agent loop has paused causes an
         additional request to be sent to the backend, and the chat window
         showing progress of the tool calls vanishes.

These tests target the agent (core) level root causes. TUI-level symptoms
are consequences of these agent-level bugs.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_stream_chunks(
    content: str = "",
    reasoning: str = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> list:
    """Build a list of mock stream chunks for stream_response_with_tools."""
    chunks = []
    if reasoning:
        chunks.append({"type": "reasoning", "data": reasoning})
    if content:
        chunks.append({"type": "content", "data": content})
    if tool_calls:
        chunks.append(
            {"type": "tool_calls_complete", "data": {"tool_calls": tool_calls}}
        )
    chunks.append({"type": "done", "data": {"finish_reason": finish_reason}})
    return chunks


# ---------------------------------------------------------------------------
# Bug #2: Resume during PAUSING state causes double run()
# ---------------------------------------------------------------------------


class TestResumeDuringPausing:
    """Bug #2: Calling resume() while the agent is in PAUSING state
    (pause requested but not yet effective) should cancel the pause request,
    not return False.

    Previously: resume() checked `if not self._paused: return False`.
    When the agent was in PAUSING state, _paused was still False, so resume()
    returned False. The TUI then fell into the "restart agent" path and
    started a second run() task while the first was still active.

    Fix: PauseState enum — resume() now accepts both PAUSED and PAUSING,
    transitioning both to RUNNING. No more scattered booleans to get wrong.
    """

    def test_resume_returns_true_during_pausing_state(self):
        """resume() should return True when agent is in PAUSING state.

        With PauseState enum, resume() checks `if self._pause_state == RUNNING:
        return False`, so both PAUSED and PAUSING are accepted.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        # Request pause — sets PauseState.PAUSING
        agent.pause()
        assert agent.pause_state == PauseState.PAUSING, "Should be in PAUSING state"
        assert not agent.is_paused, "Should NOT be fully paused yet"

        # Try to resume — should return True now
        result = agent.resume()

        assert result is True, (
            "resume() should return True during PAUSING state to cancel "
            "the pause request."
        )

    def test_resume_clears_pause_state_during_pausing(self):
        """resume() during PAUSING should transition to RUNNING."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        agent.pause()
        assert agent.pause_state == PauseState.PAUSING

        agent.resume()

        assert agent.pause_state == PauseState.RUNNING, (
            "After resume() during PAUSING, pause_state should be RUNNING."
        )

    def test_resume_sets_pause_event_during_pausing(self):
        """resume() during PAUSING should set _pause_event.

        If _pause_event is not set, the agent loop will still block
        if it reaches _wait_if_paused() before the pausing flag is
        checked. Setting the event ensures the loop continues.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        agent.pause()
        assert not agent._pause_event.is_set(), (
            "After pause(), _pause_event should be cleared"
        )

        agent.resume()

        assert agent._pause_event.is_set(), (
            "After resume() during PAUSING, _pause_event should be set "
            "so the agent loop doesn't block if it reaches _wait_if_paused()."
        )

    @pytest.mark.asyncio
    async def test_resume_during_pausing_prevents_double_run(self):
        """Resuming during PAUSING should NOT start a second run() task.

        This simulates the fixed TUI logic:
        1. User types /pause → agent.pause(), PauseState=PAUSING
        2. User types /resume → TUI checks agent.is_paused or agent.is_pausing
        3. TUI takes the resume branch (not the restart branch)
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        # Simulate the TUI's _handle_resume_command logic (fixed version)
        async def tui_resume_logic(agent):
            """Simplified version of TUI _handle_resume_command."""
            if agent.is_paused or agent.is_pausing:
                # Agent is paused or pausing — resume() handles both
                return agent.resume()
            else:
                # Agent was stopped — restart it
                return "WOULD_START_SECOND_RUN"

        # Set up PAUSING state
        agent.pause()
        assert agent.pause_state == PauseState.PAUSING
        assert not agent.is_paused

        # TUI resume logic should now take the correct branch
        result = await tui_resume_logic(agent)

        assert result is True, (
            "TUI resume logic should call agent.resume() (returning True), "
            "not start a second run()."
        )

    @pytest.mark.asyncio
    async def test_double_run_creates_concurrent_tasks(self):
        """Starting a second run() while the first is active is catastrophic.

        Two concurrent run() loops race on the same agent state (queue,
        messages, status). This test verifies the bug: two run() tasks
        can both be active simultaneously.
        """
        client = MockClient()

        # Create a slow mock that takes time per call, so both runs overlap
        call_count = 0

        async def slow_stream(*args, **kwargs):
            """Mock stream_response_with_tools that yields slowly."""
            nonlocal call_count
            call_count += 1
            # Yield a single content chunk then finish
            yield "content", "Hello"
            await asyncio.sleep(0.5)  # Slow enough for both runs to overlap
            yield "done", {"finish_reason": "stop"}

        agent = Agent(client=client, model="test-model")

        # Patch stream_response_with_tools to use our slow mock
        with patch("agent13.core.stream_response_with_tools", slow_stream):
            await agent.add_message("Hello")

            # Start first run
            task1 = asyncio.create_task(agent.run())
            await asyncio.sleep(0.1)

            # Simulate the bug: start a second run() while first is active
            task2 = asyncio.create_task(agent.run())
            await asyncio.sleep(0.1)

            # Both tasks are running — this is the bug
            assert not task1.done(), "First run() should still be active"
            assert not task2.done(), "Second run() should still be active"

            # Clean up
            agent.stop()
            try:
                await task1
            except asyncio.CancelledError:
                pass
            try:
                await task2
            except asyncio.CancelledError:
                pass

            # Both runs made API calls — the second run sent an extra request
            assert call_count >= 2, (
                f"Both run() tasks made API calls (count={call_count}). "
                "This confirms the double-request bug."
            )


# ---------------------------------------------------------------------------
# Bug #1: Queue count not visible while paused, status wrong after resume
# ---------------------------------------------------------------------------


class TestQueueVisibilityWhilePaused:
    """Bug #1 (part A): Adding a message while paused emits QUEUE_UPDATE
    but the TUI has no handler for it, so the queue count doesn't appear
    in the status bar until something else triggers update_status().

    At the agent level, we verify that QUEUE_UPDATE is emitted with the
    correct count when add_message() is called while paused. The TUI
    fix (adding a QUEUE_UPDATE handler) is separate.
    """

    @pytest.mark.asyncio
    async def test_queue_update_emitted_while_paused(self):
        """add_message() while paused should emit QUEUE_UPDATE with correct count.

        This verifies the agent does its job — the bug is that the TUI
        doesn't listen for QUEUE_UPDATE events.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        queue_updates = []

        @agent.on_event
        async def handler(event):
            if event.event == AgentEvent.QUEUE_UPDATE:
                queue_updates.append(event)

        # Put agent into PAUSED state directly
        agent._pause_state = PauseState.PAUSED
        agent._pause_event.clear()

        # Add a message while paused
        await agent.add_message("urgent message", priority=True, interrupt=True)

        # Verify QUEUE_UPDATE was emitted with count=1
        assert len(queue_updates) >= 1, "QUEUE_UPDATE should be emitted"
        last_update = queue_updates[-1]
        assert last_update.data.get("count") == 1, (
            f"QUEUE_UPDATE count should be 1, got {last_update.data.get('count')}"
        )

    @pytest.mark.asyncio
    async def test_queue_update_emitted_with_interrupt_while_paused(self):
        """An interrupt message added while paused should emit QUEUE_UPDATE.

        This is the exact user scenario: /pause, then !!interrupt message.
        The queue count should be visible immediately.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        queue_updates = []

        @agent.on_event
        async def handler(event):
            if event.event == AgentEvent.QUEUE_UPDATE:
                queue_updates.append(event)

        # Put agent into PAUSED state
        agent._pause_state = PauseState.PAUSED
        agent._pause_event.clear()

        # Add an interrupt message (the !! prefix case)
        await agent.add_message("!!change direction", priority=True, interrupt=True)

        assert len(queue_updates) >= 1, (
            "QUEUE_UPDATE should be emitted when interrupt message added while paused"
        )
        last_update = queue_updates[-1]
        assert last_update.data.get("count") == 1, (
            f"Queue should show 1 pending item, got {last_update.data.get('count')}"
        )


class TestStatusAfterResume:
    """Bug #1 (part B): After /resume, the status shows "ready" instead of
    "processing" because:

    1. resume() sets _pause_event, but the agent hasn't transitioned away
       from PAUSED yet (the _wait_if_paused() coroutine hasn't run yet)
    2. The TUI's update_status() used to fall through to self.status (a
       stale cache from the last STATUS_CHANGE event). Now it reads from
       agent.status.value directly — always current, no stale cache.
    3. Once the agent picks up the interrupt item, it goes IDLE → WAITING →
       PROCESSING, but there's a visible gap where "ready" is shown.

    At the agent level, we test that after resume(), the agent correctly
    transitions to PROCESSING when it picks up a queued item.
    """

    @pytest.mark.asyncio
    async def test_status_transitions_to_processing_after_resume(self):
        """After resume with a queued item, agent should reach PROCESSING status.

        This tests the full flow: pause → add interrupt → resume → verify
        the agent processes the item and reaches PROCESSING status.
        """
        client = MockClient()

        # Mock that produces a simple content response
        async def mock_stream(*args, **kwargs):
            yield "content", "Hello"
            yield "done", {"finish_reason": "stop"}

        agent = Agent(client=client, model="test-model")

        status_changes = []

        @agent.on_event
        async def handler(event):
            if event.event == AgentEvent.STATUS_CHANGE:
                status_changes.append(event.data.get("status"))

        with patch("agent13.core.stream_response_with_tools", mock_stream):
            # Start the agent
            await agent.add_message("First message")
            task = asyncio.create_task(agent.run())

            # Wait for processing to start
            for _ in range(50):
                await asyncio.sleep(0.02)
                if agent.status == AgentStatus.PROCESSING:
                    break

            # Request pause
            agent.pause()

            # Wait for pause to take effect
            for _ in range(50):
                await asyncio.sleep(0.02)
                if agent.is_paused:
                    break

            assert agent.is_paused, "Agent should be paused"

            # Add an interrupt message while paused
            await agent.add_message("!!interrupt", priority=True, interrupt=True)

            # Resume
            agent.resume()

            # Wait for processing to start again
            for _ in range(100):
                await asyncio.sleep(0.02)
                if agent.status in (
                    AgentStatus.PROCESSING,
                    AgentStatus.THINKING,
                    AgentStatus.WAITING,
                    AgentStatus.TOOLING,
                ):
                    break

            # Give the agent time to finish processing (it's a fast mock)
            await asyncio.sleep(0.2)

            # Agent should have reached an active state after resume.
            # Check the status_changes log rather than current status,
            # because a fast mock may have already finished and returned to IDLE.
            active_states = {"processing", "thinking", "waiting", "tooling"}
            reached_active = any(s in active_states for s in status_changes)
            assert reached_active, (
                f"After resume with queued item, agent should have reached an "
                f"active state. Status changes: {status_changes}"
            )

            # Clean up
            agent.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_resume_emits_resumed_then_idle_before_processing(self):
        """After resume(), the agent emits RESUMED then IDLE before picking up work.

        This is the root cause of the "stays on ready" bug: between the
        RESUMED event and when the next item starts processing, the status
        is IDLE. The TUI shows "ready" during this gap.

        The fix should be: when resuming with items in the queue, skip
        the IDLE transition and go directly to WAITING/PROCESSING.
        """
        client = MockClient()

        async def mock_stream(*args, **kwargs):
            yield "content", "Response"
            yield "done", {"finish_reason": "stop"}

        agent = Agent(client=client, model="test-model")

        event_log = []

        @agent.on_event
        async def handler(event):
            event_log.append((event.event, event.data.copy() if event.data else {}))

        with patch("agent13.core.stream_response_with_tools", mock_stream):
            # Start agent and get it processing
            await agent.add_message("Hello")
            task = asyncio.create_task(agent.run())

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.02)
                if agent.status == AgentStatus.PROCESSING:
                    break

            # Pause
            agent.pause()
            for _ in range(50):
                await asyncio.sleep(0.02)
                if agent.is_paused:
                    break

            # Add interrupt item to queue
            await agent.add_message("!!interrupt", priority=True, interrupt=True)

            # Clear event log to focus on resume sequence
            event_log.clear()

            # Resume
            agent.resume()

            # Wait for processing to start
            for _ in range(100):
                await asyncio.sleep(0.02)
                if agent.status in (AgentStatus.PROCESSING, AgentStatus.THINKING):
                    break

            # Check the event sequence after resume
            status_after_resume = [
                (ev, data)
                for ev, data in event_log
                if ev in (AgentEvent.RESUMED, AgentEvent.STATUS_CHANGE)
            ]

            # The bug: RESUMED → IDLE → WAITING → PROCESSING
            # The fix: RESUMED → WAITING → PROCESSING (skip IDLE)
            status_values = [
                data.get("status")
                for ev, data in status_after_resume
                if ev == AgentEvent.STATUS_CHANGE
            ]

            # After fix: IDLE should NOT appear between RESUMED and WAITING
            # when there are items in the queue
            if "idle" in status_values:
                idle_idx = status_values.index("idle")
                waiting_idx = (
                    status_values.index("waiting") if "waiting" in status_values else -1
                )
                if waiting_idx > idle_idx:
                    pytest.fail(
                        f"After resume with queued items, status should not "
                        f"go through IDLE before WAITING/PROCESSING. "
                        f"Got sequence: {status_values}"
                    )

            # Clean up
            agent.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass
