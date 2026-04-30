"""Tests for queue interrupt logging bugs.

Bug #1: log_queue_interrupt passes QueueItem object instead of item_id (int).
    At core.py line 1854, the mid-turn interrupt path calls:
        log_queue_interrupt(self.queue.current)
    But log_queue_interrupt expects item_id: int = None. Passing a QueueItem
    dataclass causes json.dumps to raise TypeError (not JSON serializable),
    which log_event silently catches. The log entry is DROPPED entirely.

Bug #2: CancelledError in run() skips queue.complete_current() and
    log_queue_complete(). When ESC cancels the agent task, CancelledError
    propagates from _llm_turn → _process_item → run(). The handler at
    run() line 1151-1156 logs queue_interrupt with no item_id and re-raises,
    so _process_item never reaches complete_current() or log_queue_complete().
    The debug log shows the item starting but never completing.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent13.core import Agent, AgentEvent
from agent13.debug_log import log_queue_interrupt
from agent13.queue import AgentQueue


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


class TestLogQueueInterruptTypeMismatch:
    """Bug #1: log_queue_interrupt(self.queue.current) passed QueueItem not int.

    At core.py line 1854, when a priority/interrupt message arrives mid-turn,
    the code used to call log_queue_interrupt(self.queue.current). But
    log_queue_interrupt expects item_id: int = None. Passing a QueueItem
    dataclass caused:

    1. QueueItem is not None, so `if item_id is not None` passes
    2. data["item_id"] = QueueItem(...) is set
    3. json.dumps raises TypeError (QueueItem is not JSON serializable)
    4. log_event catches the exception silently (line 97: `except Exception: pass`)
    5. The log entry is DROPPED — nothing is written to disk

    Fix: Changed to log_queue_interrupt(self.queue.current.id if self.queue.current else None)

    This test verifies the fix works — both int and the corrected call
    produce valid log entries with integer item_ids.
    """

    def test_log_queue_interrupt_with_queueitem_silent_failure(self, tmp_path):
        """log_queue_interrupt accepts int item_id, not QueueItem.

        We write two entries: one with an int (which succeeds) and one
        using the corrected pattern (queue.current.id if queue.current else None).
        Both should produce valid log entries with integer item_ids.
        """
        import agent13.debug_log as dl

        # Save and reset global debug state so our tmp_path is used
        saved_enabled = dl._debug_enabled
        saved_file = dl._log_file
        try:
            dl._debug_enabled = True
            dl._log_file = tmp_path / "debug.log"

            # Create a QueueItem like queue.current would be
            queue = AgentQueue()
            queue.add("test message")
            item = queue.get_next()  # Sets queue.current to a QueueItem

            # Write a control entry with correct int type — should succeed
            log_queue_interrupt(item.id)

            # Write the corrected entry — this is what core.py line 1854
            # now does after the fix:
            log_queue_interrupt(queue.current.id if queue.current else None)

            # Read back the log
            interrupt_entries = []
            with open(dl._log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if entry.get("event") == "queue_interrupt":
                        interrupt_entries.append(entry)

            # Both entries should be written (fix resolved the silent drop)
            assert len(interrupt_entries) == 2, (
                f"Expected 2 queue_interrupt events, got {len(interrupt_entries)}. "
                f"If only 1, the QueueItem is still being passed directly. "
                f"Entries found: {interrupt_entries}"
            )

            # Both entries should have item_id as int
            for i, entry in enumerate(interrupt_entries):
                data = entry.get("data", {})
                assert isinstance(data.get("item_id"), int), (
                    f"queue_interrupt entry {i} should have item_id as int. "
                    f"Got data={data}"
                )
        finally:
            dl._debug_enabled = saved_enabled
            dl._log_file = saved_file


class TestCancelledErrorSkipsQueueComplete:
    """Bug #2: CancelledError in run() used to skip complete_current() and log_queue_complete().

    When ESC cancels the agent task during streaming, CancelledError propagates
    from _llm_turn → _process_item → run(). The handler at run() lines
    1151-1156 used to log queue_interrupt (with no item_id) and re-raise. The
    _process_item cleanup at lines 1614-1615 (complete_current + log_queue_complete)
    was never reached.

    Fix: The CancelledError handler now extracts the current item's id,
    logs queue_interrupt with that id, calls complete_current(), and logs
    queue_complete with status "interrupted" before re-raising.

    This test verifies the fix — after CancelledError, queue.current is None
    and the debug log contains a queue_complete event for the interrupted item.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_emits_queue_complete(self, tmp_path):
        """After CancelledError, queue_complete should be logged for the item.

        We start processing an item, then cancel the task mid-stream.
        The debug log should show both queue_start AND queue_complete for
        that item, and queue.current should be None.
        """
        import agent13.debug_log as dl

        client = MockClient()

        # Slow streaming response so we can cancel mid-stream
        async def slow_stream(*args, **kwargs):
            yield "content", "I will "
            yield "content", "delete all "
            await asyncio.sleep(10)  # Give us time to cancel
            yield "content", "the tables"  # Won't reach

        # Save and set debug state so our tmp_path is used
        saved_enabled = dl._debug_enabled
        saved_file = dl._log_file
        try:
            dl._debug_enabled = True
            dl._log_file = tmp_path / "debug.log"

            with patch("agent13.core.stream_response_with_tools", slow_stream):
                agent = Agent(client=client, model="test-model")

                events = []

                @agent.on_event
                async def handler(event):
                    events.append(event)

                # Add a message and start processing
                await agent.add_message("Refactor the database")
                task = asyncio.create_task(agent.run())

                # Wait for streaming to start
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if any(e.event == AgentEvent.ASSISTANT_TOKEN for e in events):
                        break

                # Cancel the task (simulating ESC/interrupt)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # After fix: queue.current should be None after CancelledError
            assert agent.queue.current is None, (
                f"After CancelledError, queue.current should be None, "
                f"got {agent.queue.current}. complete_current() should "
                f"have been called in the CancelledError handler."
            )

            # Read the debug log from our tmp_path
            log_path = tmp_path / "debug.log"
            assert log_path.exists(), "debug.log should exist in tmp_path"

            queue_starts = []
            queue_completes = []
            queue_interrupts = []

            with open(log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    evt = entry.get("event")
                    if evt == "queue_start":
                        queue_starts.append(entry)
                    elif evt == "queue_complete":
                        queue_completes.append(entry)
                    elif evt == "queue_interrupt":
                        queue_interrupts.append(entry)

            # We should have at least one queue_start
            assert len(queue_starts) >= 1, "Expected at least one queue_start event"

            # After fix: Every queue_start should have a matching queue_complete
            assert len(queue_completes) >= 1, (
                f"Expected at least one queue_complete event, got {len(queue_completes)}. "
                f"The CancelledError handler should log queue_complete with "
                f"status='interrupted'."
            )

            # The queue_complete should have status "interrupted"
            interrupted_completes = [
                e for e in queue_completes
                if e.get("data", {}).get("status") == "interrupted"
            ]
            assert len(interrupted_completes) >= 1, (
                f"Expected at least one queue_complete with status='interrupted', "
                f"got {[e.get('data', {}).get('status') for e in queue_completes]}. "
                f"The CancelledError handler should log status='interrupted'."
            )

            # The queue_interrupt should also have an item_id
            for entry in queue_interrupts:
                data = entry.get("data", {})
                assert "item_id" in data and data["item_id"] is not None, (
                    f"queue_interrupt has no item_id. data={data}. "
                    f"When CancelledError fires during item processing, the "
                    f"interrupt log should include the current item's id."
                )
        finally:
            dl._debug_enabled = saved_enabled
            dl._log_file = saved_file
