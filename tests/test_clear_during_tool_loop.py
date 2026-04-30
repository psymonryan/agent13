"""Tests for deferred /clear, /load, /retry commands (L34 bug fix).

Bug: clear_messages(), load_context(), and retry message deletion all
mutate self.messages synchronously from the TUI while _llm_turn() may
be mid-loop. After tool results are appended, the loop continues to the
next LLM call with a corrupted message list (no user message, just
orphaned assistant+tool messages), causing a 500 "No user query found
in messages" error.

Fix: All three commands now use queue items (kind="clear"/"load"/"retry")
that are processed at safe boundaries between items in _process_item,
never mid-loop. The TUI calls request_clear()/request_load()/request_retry()
which add queue items, and the agent processes them when idle.
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
from agent13 import Agent, AgentEvent, AgentStatus


def _make_tool_call(call_id, name, args):
    """Helper to build a tool call dict."""
    return {
        "id": call_id,
        "name": name,
        "arguments": json.dumps(args),
    }


class TestDeferredClearDuringToolLoop:
    """Test that request_clear() defers the clear to a safe boundary."""

    @pytest.mark.asyncio
    async def test_request_clear_does_not_corrupt_mid_loop(self):
        """request_clear() adds a queue item instead of clearing immediately.

        When clear_messages() was called directly from the TUI, it wiped
        self.messages while _llm_turn was mid-loop. The next LLM call
        received [assistant(tool_calls), tool(result)] with no user
        message, causing a 500 error.

        With request_clear(), the clear is deferred to a queue item
        processed between items, so the mid-loop messages are never
        corrupted.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")
        agent._running = True

        # Track what messages are sent to the API
        api_messages_sent = []

        # Mock tool executor
        async def mock_execute_tool(name, arguments):
            return json.dumps({"result": arguments.get("x", 0) ** 2})

        agent._execute_tool_async = mock_execute_tool

        # Build mock stream that produces tool calls then a final response.
        call_count = 0

        async def mock_stream(client_arg, model, messages, system_prompt, tools):
            nonlocal call_count
            call_count += 1
            # Capture what messages the API would receive
            api_messages_sent.append(list(messages))

            if call_count == 1:
                # Round 1: yield tool calls
                yield (
                    "tool_calls_complete",
                    {"tool_calls": [_make_tool_call("tc1", "square_number", {"x": 3})]},
                )
            else:
                # Round 2: yield final content
                yield ("content", "The square of 3 is 9.")

        # Add a user message (as _process_item would do)
        agent.messages.append({"role": "user", "content": "Square 3"})

        # Hook into _set_status to inject request_clear() at the same
        # point where the old /clear would strike (between tool result
        # and next LLM call).
        original_set_status = agent._set_status
        clear_requested = asyncio.Event()
        status_call_count = 0

        async def hooked_set_status(status):
            nonlocal status_call_count
            status_call_count += 1
            result = await original_set_status(status)
            # After tool results, the loop transitions to WAITING before
            # the next LLM call. This is the exact point where /clear
            # from the TUI would strike.
            if status == AgentStatus.WAITING and status_call_count == 2:
                # Simulate deferred /clear from TUI — adds queue item
                # instead of wiping messages directly
                await agent.request_clear()
                clear_requested.set()
            return result

        agent._set_status = hooked_set_status

        with patch("agent13.core.stream_response_with_tools", side_effect=mock_stream):
            # Run the LLM turn
            await agent._llm_turn()

        # Verify request_clear() was called
        assert clear_requested.is_set(), "request_clear() was never injected"

        # THE FIX: With request_clear(), messages are NOT cleared mid-loop.
        # The clear is deferred to a queue item. So the second API call
        # still has the user message.
        assert len(api_messages_sent) >= 2, (
            f"Expected at least 2 API calls, got {len(api_messages_sent)}"
        )

        second_call_messages = api_messages_sent[1]
        user_messages = [m for m in second_call_messages if m.get("role") == "user"]

        assert len(user_messages) > 0, (
            "Second API call has no user message — request_clear() should "
            "defer the clear, not wipe messages mid-loop."
        )

        # The clear should be pending in the queue, not yet executed
        clear_items = [i for i in agent.queue.list_items() if i.kind == "clear"]
        assert len(clear_items) == 1, (
            "Clear should be pending in queue as kind='clear' item"
        )

    @pytest.mark.asyncio
    async def test_clear_messages_between_items_is_safe(self):
        """Calling clear_messages() between queue items (when agent is idle)
        should work correctly — this is the expected/safe usage.

        This test documents the correct behaviour that should continue to work
        after the fix.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # Set up some messages
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # Clear when idle — this should always work
        count = agent.clear_messages()

        assert count == 2
        assert agent.messages == []

    @pytest.mark.asyncio
    async def test_clear_via_queue_kind_is_processed_safely(self):
        """A kind='clear' queue item is processed by _process_item at a
        safe boundary, emitting MESSAGES_CLEARED event.

        This verifies the full deferred flow: request_clear() → queue item
        → _process_item → clear_messages() → MESSAGES_CLEARED event.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # Set up some messages
        agent.messages = [
            {"role": "user", "content": "Old message"},
            {"role": "assistant", "content": "Old response"},
        ]

        # Capture events
        events_received = []

        async def on_event(event_data):
            events_received.append(event_data)

        agent.on_event(on_event)

        # Add a clear item to the queue
        agent.queue.add("", kind="clear")

        # Process it via _process_item (the safe boundary handler)
        item = agent.queue.get_next()
        assert item is not None
        assert item.kind == "clear"

        await agent._process_item(item)

        # Messages should be cleared
        assert agent.messages == []

        # MESSAGES_CLEARED event should have been emitted
        cleared_events = [
            e for e in events_received if e.event == AgentEvent.MESSAGES_CLEARED
        ]
        assert len(cleared_events) == 1
        assert cleared_events[0].data.get("count") == 2


class TestDeferredLoadDuringToolLoop:
    """Test that request_load() defers the load to a safe boundary."""

    @pytest.mark.asyncio
    async def test_request_load_defers_to_queue(self):
        """request_load() adds a queue item instead of loading immediately.

        When load_context() was called directly from the TUI, it replaced
        self.messages while _llm_turn could be mid-loop. With request_load(),
        the load is deferred to a queue item.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # Set up some messages
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # request_load() should add a queue item, not replace messages
        item_id = await agent.request_load("/some/path.ctx")

        # Messages should still be intact (not replaced)
        assert len(agent.messages) == 2

        # The load should be pending in the queue
        load_items = [i for i in agent.queue.list_items() if i.kind == "load"]
        assert len(load_items) == 1, (
            "Load should be pending in queue as kind='load' item"
        )
        assert load_items[0].text == "/some/path.ctx"
        assert load_items[0].id == item_id


class TestDeferredRetryDuringToolLoop:
    """Test that request_retry() defers the retry to a safe boundary."""

    @pytest.mark.asyncio
    async def test_request_retry_defers_to_queue(self):
        """request_retry() adds a queue item instead of deleting messages
        immediately.

        When /retry deleted messages by index directly from the TUI, it
        could corrupt self.messages while _llm_turn was mid-loop. With
        request_retry(), the deletion is deferred to a queue item.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # Set up some messages
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # request_retry() should add a queue item, not delete messages
        item_id = await agent.request_retry()

        # Messages should still be intact (not deleted)
        assert len(agent.messages) == 2

        # The retry should be pending in the queue
        retry_items = [i for i in agent.queue.list_items() if i.kind == "retry"]
        assert len(retry_items) == 1, (
            "Retry should be pending in queue as kind='retry' item"
        )
        assert retry_items[0].id == item_id

    @pytest.mark.asyncio
    async def test_retry_via_process_item_deletes_group(self):
        """A kind='retry' queue item processed by _process_item deletes
        the last message group and emits RETRY_STARTED with the user text.
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # Set up messages with two groups
        agent.messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]

        # Capture events
        events_received = []

        async def on_event(event_data):
            events_received.append(event_data)

        agent.on_event(on_event)

        # Add a retry item to the queue
        agent.queue.add("", kind="retry")

        # Process it
        item = agent.queue.get_next()
        await agent._process_item(item)

        # Last group should be deleted (second question + answer)
        assert len(agent.messages) == 2
        assert agent.messages[0]["content"] == "First question"
        assert agent.messages[1]["content"] == "First answer"

        # RETRY_STARTED event should have been emitted with user_text
        retry_events = [
            e for e in events_received if e.event == AgentEvent.RETRY_STARTED
        ]
        assert len(retry_events) == 1
        assert retry_events[0].data.get("user_text") == "Second question"

    @pytest.mark.asyncio
    async def test_retry_with_no_messages(self):
        """A kind='retry' queue item with no messages emits RETRY_STARTED
        with empty user_text (no crash).
        """
        client = MagicMock()
        agent = Agent(client, model="test-model")

        # No messages
        assert agent.messages == []

        # Capture events
        events_received = []

        async def on_event(event_data):
            events_received.append(event_data)

        agent.on_event(on_event)

        # Add a retry item
        agent.queue.add("", kind="retry")

        item = agent.queue.get_next()
        await agent._process_item(item)

        # Should not crash, just emit with empty user_text
        retry_events = [
            e for e in events_received if e.event == AgentEvent.RETRY_STARTED
        ]
        assert len(retry_events) == 1
        assert retry_events[0].data.get("user_text") == ""
