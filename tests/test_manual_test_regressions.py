"""Tests for bugs found during manual testing (April 2026).

Bug #1 (wishlist #28): After an error in _process_item (outside _llm_turn),
  the agent goes IDLE not PAUSED. The _llm_turn error handler correctly sets
  PAUSED (line 2022-2026), but _process_item's except block doesn't.

Bug #2 (wishlist #70 regression): After ESC interrupt, the agent state should
  be clean for /resume. The _interrupt_agent_loop clears pause state and
  restarts the agent loop, but the TUI's input buffer concatenation means
  /resume text gets merged with next input.

Bug #3 (wishlist #54): After journal operations, the TUI's Ctx counter stays
  at 0 because no TOKEN_USAGE event is emitted after compaction.
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


# ============================================================================
# Bug #1: _process_item except block doesn't set PauseState.PAUSED
# ============================================================================

class TestProcessItemErrorPath:
    """Bug #1: _process_item except block should set PauseState.PAUSED.

    The _llm_turn error handler (line 2022-2026) correctly sets:
        self._pause_state = PauseState.PAUSED
        self._pause_event.clear()
        await self._set_status(AgentStatus.PAUSED)
        await self.emit(AgentEvent.PAUSED, {"reason": "error"})

    But _process_item's except block (line 1579-1592) only emits ERROR
    and completes the queue item. It does NOT set pause_state or status.
    Line 1593-1595 preserves PAUSED if already set by _llm_turn, but
    doesn't set it if the error came from elsewhere in _process_item.

    This means: errors in the retrospective journal path, or any error
    that doesn't originate from _llm_turn, leave the agent in RUNNING
    state with status IDLE. /resume then says "Not paused".
    """

    @pytest.mark.asyncio
    async def test_process_item_exception_sets_paused(self):
        """When _process_item catches an exception outside _llm_turn,
        it should set PauseState.PAUSED so /resume works.

        We simulate this by making _reflect_on_tool_use raise an error
        during the retrospective journal path in _process_item.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        # Enable journal mode to trigger retrospective journal path
        agent.journal_mode = True

        # Add messages with tool calls (triggers retrospective journal)
        agent.messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {
                     "name": "read_file",
                     "arguments": json.dumps({"filepath": "/tmp/test.txt"}),
                 }
             }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "File contents"},
            {"role": "assistant", "content": "Done."},
        ]

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        # Mock _reflect_on_tool_use to raise an error
        with patch.object(
            agent, '_reflect_on_tool_use',
            new_callable=AsyncMock,
            side_effect=Exception("Reflection failed"),
        ):
            await agent.add_message("Next prompt")
            task = asyncio.create_task(agent.run())

            # Wait for error
            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ERROR for e in events):
                    break

            # After error in _process_item, should be PAUSED
            assert agent.pause_state == PauseState.PAUSED, (
                f"After error in _process_item, pause_state should be PAUSED, "
                f"got {agent.pause_state}"
            )

            agent.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_process_item_exception_status_is_paused(self):
        """After _process_item exception, agent status should be PAUSED not IDLE."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.journal_mode = True

        agent.messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {
                     "name": "read_file",
                     "arguments": json.dumps({"filepath": "/tmp/test.txt"}),
                 }
             }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "File contents"},
            {"role": "assistant", "content": "Done."},
        ]

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        with patch.object(
            agent, '_reflect_on_tool_use',
            new_callable=AsyncMock,
            side_effect=Exception("Reflection failed"),
        ):
            await agent.add_message("Next prompt")
            task = asyncio.create_task(agent.run())

            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ERROR for e in events):
                    break

            # Status should be PAUSED, not IDLE
            assert agent.status == AgentStatus.PAUSED, (
                f"After error in _process_item, status should be PAUSED, "
                f"got {agent.status}"
            )

            agent.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_resume_works_after_process_item_error(self):
        """After _process_item exception, agent.resume() should return True."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")
        agent.journal_mode = True

        agent.messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {
                     "name": "read_file",
                     "arguments": json.dumps({"filepath": "/tmp/test.txt"}),
                 }
             }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "File contents"},
            {"role": "assistant", "content": "Done."},
        ]

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        with patch.object(
            agent, '_reflect_on_tool_use',
            new_callable=AsyncMock,
            side_effect=Exception("Reflection failed"),
        ):
            await agent.add_message("Next prompt")
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

            # resume() should return True (currently returns False
            # because pause_state is RUNNING, not PAUSED)
            result = agent.resume()
            assert result is True, (
                f"After error in _process_item, resume() should return True. "
                f"Got False because pause_state={agent.pause_state}"
            )


# ============================================================================
# Bug #2: Agent state after interrupt should be clean for /resume
# ============================================================================

class TestInterruptResumeState:
    """Bug #2: After ESC interrupt, agent should be in a clean state for /resume.

    The _interrupt_agent_loop in tui.py:
    1. Cancels the agent task
    2. Sets _interrupt_available = True
    3. Clears pause state (calls agent.resume() if paused/pausing)
    4. Clears input field
    5. Restarts agent loop

    At the agent level, after interrupt+restart:
    - pause_state should be RUNNING (not PAUSED/PAUSING)
    - The agent should accept new messages normally
    """

    @pytest.mark.asyncio
    async def test_pause_state_clean_after_cancel_and_restart(self):
        """After task cancellation and restart, pause_state should be RUNNING."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Simulate interrupt: cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Restart
        task2 = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        assert agent.pause_state == PauseState.RUNNING, (
            f"After interrupt+restart, pause_state should be RUNNING, "
            f"got {agent.pause_state}"
        )

        agent.stop()
        try:
            await task2
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_interrupt_available_flag_for_resume(self):
        """After interrupt, agent should be able to process a continuation message."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent.add_message("Hello")
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Interrupt
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Restart and send continuation
        task2 = asyncio.create_task(agent.run())
        await asyncio.sleep(0.05)

        await agent.add_message("Actually, please continue")
        await asyncio.sleep(0.2)

        # Agent should process the message
        started_events = [
            e for e in events if e.event == AgentEvent.ITEM_STARTED
        ]
        assert len(started_events) >= 1, (
            "Agent should process continuation message after interrupt"
        )

        agent.stop()
        try:
            await task2
        except asyncio.CancelledError:
            pass


# ============================================================================
# Bug #3: Journal should emit TOKEN_USAGE after compaction
# ============================================================================

class TestJournalTokenUsage:
    """Bug #3: After journal compaction, TOKEN_USAGE event should be emitted.

    The TUI's Ctx counter updates on TOKEN_USAGE events. After journal
    compaction, the message history changes size but no TOKEN_USAGE is
    emitted, so the counter stays at 0 or its previous value.

    This is especially visible with --continue sessions where the Ctx
    starts at 0 and never updates after journal.
    """

    @pytest.mark.asyncio
    async def test_journal_last_turn_emits_token_usage(self):
        """After journal_last_turn compaction, TOKEN_USAGE should be emitted.

        We mock _reflect_on_tool_use to return a summary, then verify
        that a TOKEN_USAGE event is emitted after compaction.
        """
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        # Set up messages with tool calls for journal to compact
        agent.messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {
                     "name": "read_file",
                     "arguments": json.dumps({"filepath": "/tmp/test.txt"}),
                 }
             }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "File contents here with some length"},
            {"role": "assistant", "content": "Here is the file content."},
        ]

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        # Mock _reflect_on_tool_use to return a summary
        with patch.object(
            agent, '_reflect_on_tool_use',
            new_callable=AsyncMock,
            return_value="Summary: read_file returned file contents",
        ):
            success, message = await agent.journal_last_turn()

        # Journal should succeed
        assert success, f"journal_last_turn should succeed, got: {message}"

        # After compaction, TOKEN_USAGE should be emitted
        token_events = [
            e for e in events if e.event == AgentEvent.TOKEN_USAGE
        ]
        assert len(token_events) > 0, (
            "After journal compaction, TOKEN_USAGE should be emitted "
            "so the TUI can update its Ctx counter. "
            f"Got {len(token_events)} TOKEN_USAGE events. "
            f"Events: {[e.event for e in events]}"
        )

    @pytest.mark.asyncio
    async def test_journal_compact_event_contains_token_counts(self):
        """JOURNAL_COMPACT event should contain token counts for TUI to use."""
        client = MockClient()
        agent = Agent(client=client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Read /tmp/test.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {
                     "name": "read_file",
                     "arguments": json.dumps({"filepath": "/tmp/test.txt"}),
                 }
             }]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "File contents here with some length"},
            {"role": "assistant", "content": "Here is the file content."},
        ]

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        with patch.object(
            agent, '_reflect_on_tool_use',
            new_callable=AsyncMock,
            return_value="Summary: read_file returned file contents",
        ):
            success, message = await agent.journal_last_turn()

        assert success, f"journal_last_turn should succeed, got: {message}"

        # Find JOURNAL_COMPACT event
        compact_events = [
            e for e in events if e.event == AgentEvent.JOURNAL_COMPACT
        ]
        assert len(compact_events) > 0, "Should emit JOURNAL_COMPACT event"

        # Check that token counts are present
        data = compact_events[0].data
        assert "tokens_before" in data, "JOURNAL_COMPACT should have tokens_before"
        assert "tokens_after" in data, "JOURNAL_COMPACT should have tokens_after"
        assert data["tokens_before"] > data["tokens_after"], (
            f"tokens_before ({data['tokens_before']}) should be > "
            f"tokens_after ({data['tokens_after']}) after compaction"
        )
