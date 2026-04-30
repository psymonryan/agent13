"""Tests for journal mode (context compaction via reflection)."""

import pytest
from unittest.mock import AsyncMock, patch
from agent13 import Agent, AgentEvent


class MockClient:
    """Mock OpenAI client for testing."""

    pass


class TestCompactPreviousTurn:
    """Tests for _compact_previous_turn method."""

    def test_compact_previous_turn_basic(self):
        """Should replace messages after last user message with summary."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Setup: user -> assistant -> user -> assistant
        agent.messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer with lots of tokens"},
        ]

        agent._compact_previous_turn("Summary of previous turn")

        # Should keep messages up to last user, then append summary
        # Result: user, assistant, user, summary
        assert len(agent.messages) == 4
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "First question"
        assert agent.messages[1]["role"] == "assistant"
        assert agent.messages[1]["content"] == "First answer"
        assert agent.messages[2]["role"] == "user"
        assert agent.messages[2]["content"] == "Second question"
        # After last user message, the summary replaces the old assistant response
        assert agent.messages[3]["role"] == "assistant"
        assert agent.messages[3]["content"] == "Summary of previous turn"

    def test_compact_preserves_first_user_message(self):
        """Should preserve the first user message even with long history."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Setup: multiple turns
        agent.messages = [
            {"role": "user", "content": "Important first message"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        agent._compact_previous_turn("Summary")

        # First user message should still be there
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "Important first message"

    def test_compact_empty_messages(self):
        """Should handle empty message list gracefully."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = []
        agent._compact_previous_turn("Summary")

        assert agent.messages == []

    def test_compact_no_user_message(self):
        """Should handle case where there's no user message."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Only assistant messages (edge case)
        agent.messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        agent._compact_previous_turn("Summary")

        # Should remain unchanged (no user message to find)
        assert len(agent.messages) == 2

    def test_compact_single_turn(self):
        """Should compact a single turn correctly."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        agent._compact_previous_turn("Summary")

        assert len(agent.messages) == 2
        assert agent.messages[0]["content"] == "Question"
        assert agent.messages[1]["content"] == "Summary"


class TestReflectOnToolUse:
    """Tests for _reflect_on_tool_use method."""

    @pytest.mark.asyncio
    async def test_reflection_returns_summary(self):
        """Should return summary string from reflection API call."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It's 4."},
        ]

        # Mock the stream_response_with_tools function (imported from agent13.llm)
        async def mock_stream(*args, **kwargs):
            yield ("content", "Attempted: math. Found: 4.")

        with patch("agent13.llm.stream_response_with_tools") as mock_reflect:
            mock_reflect.return_value = mock_stream()

            result = await agent._reflect_on_tool_use()

            assert result == "Attempted: math. Found: 4."
            # Verify tool_choice="auto" was passed (possible LCP cache fix)
            call_kwargs = mock_reflect.call_args
            assert call_kwargs[1].get("tool_choice") == "auto"
            # Verify the reflection prompt asks about tools
            messages = call_kwargs[0][2]  # Third argument is messages
            assert messages[-1]["role"] == "user"
            assert "reflect" in messages[-1]["content"].lower()
            assert "tools" in messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_reflection_failure_returns_none(self):
        """Should return None if reflection API call fails."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        # Mock the stream_response_with_tools to raise an exception
        async def mock_stream(*args, **kwargs):
            raise Exception("API error")
            yield  # never reached, but makes this a generator

        with patch("agent13.llm.stream_response_with_tools") as mock_reflect:
            mock_reflect.return_value = mock_stream()

            result = await agent._reflect_on_tool_use()

            assert result is None

    @pytest.mark.asyncio
    async def test_reflection_empty_response_returns_none(self):
        """Should return None if reflection returns empty string."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        # Mock stream_response_with_tools to return empty content
        async def mock_stream(*args, **kwargs):
            yield ("content", "")

        with patch("agent13.llm.stream_response_with_tools") as mock_reflect:
            mock_reflect.return_value = mock_stream()

            result = await agent._reflect_on_tool_use()

            assert result is None

    @pytest.mark.asyncio
    async def test_reflection_whitespace_only_returns_none(self):
        """Should return None if reflection returns only whitespace."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        # Mock stream_response_with_tools to return whitespace
        async def mock_stream(*args, **kwargs):
            yield ("content", "   \n\t  ")

        with patch("agent13.llm.stream_response_with_tools") as mock_reflect:
            mock_reflect.return_value = mock_stream()
            result = await agent._reflect_on_tool_use()

            assert result is None


class TestJournalModeIntegration:
    """Tests for journal mode behavior in _process_item."""

    @pytest.mark.asyncio
    async def test_journal_mode_disabled_no_op(self):
        """Should not compact when journal_mode is False."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=False)

        # Even with tool calls, journal_mode=False means no compaction
        agent.messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "tool_calls": [{"id": "tc1"}]},
        ]

        # Track if _reflect_on_tool_use was called
        reflect_called = False

        async def track_reflect():
            nonlocal reflect_called
            reflect_called = True
            return None

        agent._reflect_on_tool_use = track_reflect

        # Process an item (we need to mock the LLM turn to avoid actual API calls)
        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "Q2",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # Reflection should NOT have been called (journal_mode is False)
        assert reflect_called is False
        # Messages should just have the new user message appended
        assert len(agent.messages) == 3
        assert agent.messages[-1]["content"] == "Q2"

    @pytest.mark.asyncio
    async def test_first_message_no_compaction(self):
        """Should skip compaction on first message (no history)."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)

        agent.messages = []

        # Track if _reflect_on_tool_use was called
        reflect_called = False

        async def track_reflect():
            nonlocal reflect_called
            reflect_called = True
            return None

        agent._reflect_on_tool_use = track_reflect

        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "First",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # No history, so no reflection needed
        assert reflect_called is False
        assert len(agent.messages) == 1

    @pytest.mark.asyncio
    async def test_no_tool_calls_no_compaction(self):
        """Should not compact when last turn had no tool calls."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)

        # No tool_calls in the assistant message
        agent.messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},  # No tool_calls
        ]

        # Track if _reflect_on_tool_use was called
        reflect_called = False

        async def track_reflect():
            nonlocal reflect_called
            reflect_called = True
            return None

        agent._reflect_on_tool_use = track_reflect

        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "Q2",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # No tool calls, so no reflection needed
        assert reflect_called is False
        assert len(agent.messages) == 3

    @pytest.mark.asyncio
    async def test_atomic_mutation_on_failure(self):
        """History should remain intact if reflection fails."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)

        # Need tool_calls for compaction to be attempted
        original_messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "tool_calls": [{"id": "tc1"}]},
        ]
        agent.messages = original_messages.copy()

        # Mock reflection to fail
        async def failing_reflect():
            return None  # Simulates failure

        agent._reflect_on_tool_use = failing_reflect

        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "Q2",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # Original messages should still be there (plus new user message)
        assert len(agent.messages) == 3
        assert agent.messages[0]["content"] == "Q1"
        assert agent.messages[1]["content"] == "A1"
        assert agent.messages[2]["content"] == "Q2"

    @pytest.mark.asyncio
    async def test_retrospective_compaction_on_new_message(self):
        """Should compact previous turn's tool calls when journal is on and new message arrives."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)

        # Set up messages from a previous turn with tool calls
        agent.messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "tool_calls": [{"id": "tc1"}]},
        ]

        # Mock _reflect_on_tool_use to return a summary
        async def mock_reflect():
            return "Summary of tool use"

        agent._reflect_on_tool_use = mock_reflect

        # Capture events
        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "Q2",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # Should have emitted JOURNAL_COMPACT with retrospective=True
        compact_events = [e for e in events if e.event == AgentEvent.JOURNAL_COMPACT]
        assert len(compact_events) == 1
        assert compact_events[0].data["retrospective"] is True

        # Messages should be compacted
        assert len(agent.messages) == 3  # Q1, compacted assistant, Q2
        assert "Summary of tool use" in agent.messages[1]["content"]

    @pytest.mark.asyncio
    async def test_retrospective_compaction_applies_compaction(self):
        """Should apply compaction to message history when journal is on."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)

        # Set up messages from a previous turn with tool calls
        agent.messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": "A1 with lots of content",
                "tool_calls": [{"id": "tc1"}],
            },
        ]

        # Mock _reflect_on_tool_use to return a brief summary
        async def mock_reflect():
            return "Brief summary"

        agent._reflect_on_tool_use = mock_reflect

        with patch.object(agent, "_llm_turn", new_callable=AsyncMock):
            await agent._process_item(
                type(
                    "Item",
                    (),
                    {
                        "text": "Q2",
                        "id": 1,
                        "priority": False,
                        "interrupt": False,
                        "kind": "prompt",
                    },
                )()
            )

        # After compaction, should have: Q1, summary + final_message, Q2
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "Q1"
        assert agent.messages[1]["role"] == "assistant"
        # The compaction combines tool summary with the final assistant message
        assert (
            agent.messages[1]["content"] == "Brief summary\n\nA1 with lots of content"
        )
        assert agent.messages[2]["role"] == "user"
        assert agent.messages[2]["content"] == "Q2"


class TestImmediateCompaction:
    """Tests for immediate compaction (previously deferred as pending_compaction).

    Compaction now happens immediately in _maybe_reflect_after_turn rather than
    being deferred to the next _process_item call.
    """

    def test_journal_compact_event_exists(self):
        """JOURNAL_COMPACT event exists in AgentEvent enum."""
        assert hasattr(AgentEvent, "JOURNAL_COMPACT")
        assert AgentEvent.JOURNAL_COMPACT.value == "journal_compact"

    @pytest.mark.asyncio
    async def test_maybe_reflect_skips_when_journal_off(self):
        """_maybe_reflect_after_turn skips when journal_mode is off."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=False)
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]

        # Should not raise and messages should be unchanged
        await agent._maybe_reflect_after_turn()
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_maybe_reflect_skips_when_no_messages(self):
        """_maybe_reflect_after_turn skips when no messages."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)
        agent.messages = []

        await agent._maybe_reflect_after_turn()
        assert len(agent.messages) == 0

    @pytest.mark.asyncio
    async def test_maybe_reflect_skips_when_no_tool_calls(self):
        """_maybe_reflect_after_turn skips when last turn had no tool calls."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},  # No tool_calls
        ]

        await agent._maybe_reflect_after_turn()
        # Messages unchanged
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_maybe_reflect_skips_with_interrupt(self):
        """_maybe_reflect_after_turn skips when interrupt is pending."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]
        # Add an interrupt to the queue
        agent.queue.add("interrupt message", interrupt=True)

        # Should skip reflection due to interrupt
        await agent._maybe_reflect_after_turn()
        # Messages unchanged
        assert len(agent.messages) == 2

    @pytest.mark.asyncio
    async def test_maybe_reflect_compacts_immediately(self):
        """_maybe_reflect_after_turn applies compaction immediately."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]

        # Mock _reflect_on_tool_use to return a summary
        async def mock_reflect():
            return "Reflected summary"

        agent._reflect_on_tool_use = mock_reflect

        # Capture events
        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent._maybe_reflect_after_turn()

        # Messages should be compacted immediately (not deferred)
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[1]["role"] == "assistant"
        assert "Reflected summary" in agent.messages[1]["content"]

        # JOURNAL_COMPACT should have been emitted
        compact_events = [e for e in events if e.event == AgentEvent.JOURNAL_COMPACT]
        assert len(compact_events) == 1

    @pytest.mark.asyncio
    async def test_maybe_reflect_no_compaction_when_summary_empty(self):
        """_maybe_reflect_after_turn does not compact when reflection returns None."""
        client = MockClient()
        agent = Agent(client, model="test-model", journal_mode=True)
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]

        # Mock _reflect_on_tool_use to return None (failure)
        async def mock_reflect():
            return None

        agent._reflect_on_tool_use = mock_reflect

        await agent._maybe_reflect_after_turn()

        # Messages should be unchanged
        assert len(agent.messages) == 2
        assert agent.messages[1]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_journal_last_turn_compacts_immediately(self):
        """journal_last_turn applies compaction immediately."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]

        # Mock _reflect_on_tool_use
        async def mock_reflect():
            return "Summary of tools"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_last_turn()

        assert success is True
        # Messages should be compacted
        assert len(agent.messages) == 2
        assert "Summary of tools" in agent.messages[1]["content"]

    @pytest.mark.asyncio
    async def test_journal_status_during_reflection(self):
        """Agent sets JOURNALING status during _reflect_on_tool_use."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]

        statuses_seen = []

        @agent.on_event
        async def handler(event):
            if event.event == AgentEvent.STATUS_CHANGE:
                statuses_seen.append(event.data.get("status"))

        # Mock stream_response_with_tools (which _reflect_on_tool_use calls internally)
        async def mock_stream(*args, **kwargs):
            yield ("content", "Reflected summary")

        with patch("agent13.llm.stream_response_with_tools") as mock_sr:
            mock_sr.return_value = mock_stream()
            # Also mock get_all_tools to avoid needing MCP
            with patch.object(
                agent, "get_all_tools", new_callable=AsyncMock, return_value=[]
            ):
                agent.journal_mode = True
                await agent._maybe_reflect_after_turn()

        # JOURNALING status should have been emitted
        assert "journaling" in statuses_seen


class TestHasToolCalls:
    """Tests for _has_tool_calls helper."""

    def test_no_tool_calls(self):
        """Returns False when no messages have tool_calls or tool role."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert agent._has_tool_calls() is False

    def test_with_tool_calls(self):
        """Returns True when an assistant message has tool_calls."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
        ]
        assert agent._has_tool_calls() is True

    def test_with_tool_role(self):
        """Returns True when a message has role 'tool'."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [{"id": "1", "name": "test"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        assert agent._has_tool_calls() is True

    def test_empty_messages(self):
        """Returns False when messages list is empty."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = []
        assert agent._has_tool_calls() is False

    def test_after_compaction_no_tool_calls(self):
        """Returns False after compaction has removed tool messages."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Summary of what I did"},
        ]
        assert agent._has_tool_calls() is False


class TestFindEarliestToolTurn:
    """Tests for _find_earliest_tool_turn helper."""

    def test_no_tool_calls(self):
        """Returns None when there are no tool-using turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert agent._find_earliest_tool_turn() is None

    def test_single_tool_turn(self):
        """Finds the boundary of a single tool-using turn."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "file contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 0  # The user message
        assert end_idx == 3  # The final assistant message

    def test_multiple_tool_turns_finds_earliest(self):
        """Finds the earliest tool turn when multiple exist."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {"role": "assistant", "content": "Here is A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B contents"},
            {"role": "assistant", "content": "Here is B"},
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 0  # First user message
        assert end_idx == 3  # First final assistant message

    def test_multi_round_tool_use(self):
        """Handles a turn with multiple rounds of tool calls."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read then edit"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "edit_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "edited"},
            {"role": "assistant", "content": "Done editing"},
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 0
        assert end_idx == 5  # Final assistant message

    def test_interrupt_user_messages_skipped(self):
        """Interrupt user messages are not treated as turn boundaries."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "user", "content": "wait", "interrupt": True},
            {"role": "assistant", "content": "ok, continuing"},
            {"role": "assistant", "content": "Here is the file"},
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 0  # Non-interrupt user message
        # end_idx should be the final assistant message

    def test_incomplete_turn_uses_end_of_messages(self):
        """Uses end-of-messages as boundary when tool turn has no final assistant response."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            # No final assistant message — turn is incomplete
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 0  # "Read the file" user message
        assert end_idx == 2  # Last message (tool result) as boundary

    def test_mixed_tool_and_non_tool_turns(self):
        """Finds the tool turn among non-tool turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]
        result = agent._find_earliest_tool_turn()
        assert result is not None
        user_idx, end_idx = result
        assert user_idx == 2  # "Read the file" user message
        assert end_idx == 5  # "Here is the file" assistant message

    def test_empty_messages(self):
        """Returns None for empty message list."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = []
        assert agent._find_earliest_tool_turn() is None


class TestCountToolTurns:
    """Tests for _count_tool_turns helper."""

    def test_no_tool_turns(self):
        """Returns 0 when there are no tool-using turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert agent._count_tool_turns() == 0

    def test_single_tool_turn(self):
        """Counts a single tool-using turn."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]
        assert agent._count_tool_turns() == 1

    def test_multiple_tool_turns(self):
        """Counts multiple distinct tool-using turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {"role": "assistant", "content": "Here is A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B contents"},
            {"role": "assistant", "content": "Here is B"},
        ]
        assert agent._count_tool_turns() == 2

    def test_multi_round_counts_as_one(self):
        """Multi-round tool use within one turn counts as one turn."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read then edit"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "edit_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "edited"},
            {"role": "assistant", "content": "Done editing"},
        ]
        assert agent._count_tool_turns() == 1

    def test_empty_messages(self):
        """Returns 0 for empty message list."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = []
        assert agent._count_tool_turns() == 0


class TestJournalAll:
    """Tests for journal_all method."""

    @pytest.mark.asyncio
    async def test_no_messages(self):
        """Returns failure when no messages exist."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = []

        success, message = await agent.journal_all()
        assert success is False
        assert "No messages" in message

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        """Returns failure when no tool-using turns exist."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        success, message = await agent.journal_all()
        assert success is False
        assert "No tool-using turns" in message

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """Journals a single tool-using turn."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "file contents here"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            return "Used read_file to read the file"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert "1 turn" in message
        # After compaction: user message + compacted assistant
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[1]["role"] == "assistant"
        assert "read_file" in agent.messages[1]["content"]
        # No tool_calls or tool role messages remain
        assert not agent._has_tool_calls()

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        """Iteratively journals multiple tool-using turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {"role": "assistant", "content": "Here is A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B contents"},
            {"role": "assistant", "content": "Here is B"},
        ]

        reflect_count = 0

        async def mock_reflect():
            nonlocal reflect_count
            reflect_count += 1
            return f"Summary of tool use round {reflect_count}"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert "2 turn" in message
        assert reflect_count == 2
        # After compacting both turns: user1 + assistant1 + user2 + assistant2
        assert len(agent.messages) == 4
        assert not agent._has_tool_calls()

    @pytest.mark.asyncio
    async def test_mixed_tool_and_non_tool_turns(self):
        """Only compacts tool-using turns, preserves non-tool turns."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            return "Used read_file"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert "1 turn" in message
        # Non-tool turn preserved: user + assistant (2 msgs)
        # Tool turn compacted: user + assistant (2 msgs)
        assert len(agent.messages) == 4
        assert agent.messages[0]["content"] == "Hello"
        assert agent.messages[1]["content"] == "Hi there"
        assert not agent._has_tool_calls()

    @pytest.mark.asyncio
    async def test_reflection_failure_stops_iteration(self):
        """Stops iterating if reflection fails, reports partial success."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {"role": "assistant", "content": "Here is A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B contents"},
            {"role": "assistant", "content": "Here is B"},
        ]

        reflect_count = 0

        async def mock_reflect():
            nonlocal reflect_count
            reflect_count += 1
            if reflect_count == 1:
                return "Summary of first turn"
            return None  # Second reflection fails

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert "1 turn" in message
        # First turn compacted, second turn still has tool calls
        assert agent._has_tool_calls() is True

    @pytest.mark.asyncio
    async def test_first_reflection_failure(self):
        """Returns failure if the very first reflection fails."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            return None

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is False
        assert "no summary" in message.lower()

    @pytest.mark.asyncio
    async def test_emits_progress_events(self):
        """Emits JOURNAL_COMPACT events with mode='all' and iteration info."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            return "Used read_file"

        agent._reflect_on_tool_use = mock_reflect

        events = []

        @agent.on_event
        async def handler(event):
            if event.event == AgentEvent.JOURNAL_COMPACT:
                events.append(event)

        await agent.journal_all()

        assert len(events) == 1
        assert events[0].data.get("mode") == "all"
        assert events[0].data.get("iteration") == 1
        assert events[0].data.get("total_turns") == 1

    @pytest.mark.asyncio
    async def test_preserves_tail_messages(self):
        """Messages after the compacted turn are preserved."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You're welcome"},
        ]

        async def mock_reflect():
            return "Used read_file"

        agent._reflect_on_tool_use = mock_reflect

        await agent.journal_all()

        # The non-tool turn at the end should be preserved
        # Find the "Thanks" user message
        user_contents = [m["content"] for m in agent.messages if m["role"] == "user"]
        assert "Thanks" in user_contents

    @pytest.mark.asyncio
    async def test_already_compacted_history(self):
        """Returns failure when history has already been fully compacted."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "Used read_file to read the file\n\nHere is the file",
            },
        ]

        success, message = await agent.journal_all()
        assert success is False
        assert "No tool-using turns" in message


class TestJournalingViaQueue:
    """Tests that journal commands route through the agent queue correctly."""

    @pytest.mark.asyncio
    async def test_journal_last_via_queue(self):
        """journal_last queued and processed like a normal item."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        reflected = False

        async def mock_reflect():
            nonlocal reflected
            reflected = True
            return "Used read_file"

        agent._reflect_on_tool_use = mock_reflect

        # Add journal item to queue
        await agent.add_message("/journal last", kind="journal_last")
        item = agent.queue.get_next()
        assert item is not None
        assert item.kind == "journal_last"
        assert not reflected  # Not yet processed

    @pytest.mark.asyncio
    async def test_journal_all_via_queue(self):
        """journal_all queued with correct kind."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        # Add journal item to queue
        await agent.add_message("/journal all", kind="journal_all")
        item = agent.queue.get_next()
        assert item is not None
        assert item.kind == "journal_all"

    @pytest.mark.asyncio
    async def test_journal_failure_returns_gracefully(self):
        """journal_last_turn returns (False, message) when reflection fails."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            return None  # Simulate failure

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_last_turn()
        assert success is False
        assert "Reflection produced no summary" in message

    @pytest.mark.asyncio
    async def test_journal_all_exception_propagates(self):
        """journal_all propagates exceptions (handled by _process_item)."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "contents"},
            {"role": "assistant", "content": "Here is the file"},
        ]

        async def mock_reflect():
            raise RuntimeError("API error")

        agent._reflect_on_tool_use = mock_reflect

        with pytest.raises(RuntimeError):
            await agent.journal_all()


class TestJournalAllIterative:
    """More thorough tests for journal_all iterative behavior."""

    @pytest.mark.asyncio
    async def test_two_turns_message_state_between_iterations(self):
        """Verify message state is correct between iterations."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {"role": "assistant", "content": "Here is A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B contents"},
            {"role": "assistant", "content": "Here is B"},
        ]

        call_count = 0
        snapshots = []

        async def mock_reflect():
            nonlocal call_count
            call_count += 1
            # Snapshot the message state at each reflection call
            snapshots.append(
                {
                    "call": call_count,
                    "msg_count": len(agent.messages),
                    "roles": [m["role"] for m in agent.messages],
                    "has_tool_calls": agent._has_tool_calls(),
                }
            )
            return f"Summary of tool use round {call_count}"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert call_count == 2

        # First reflection: messages truncated to [user_A, asst_tool, tool, asst_final]
        assert snapshots[0]["call"] == 1
        assert snapshots[0]["msg_count"] == 4
        assert snapshots[0]["roles"] == ["user", "assistant", "tool", "assistant"]
        assert snapshots[0]["has_tool_calls"] is True

        # Second reflection: first turn compacted, messages are
        # [user_A, compacted_A, user_B, asst_tool, tool, asst_final]
        assert snapshots[1]["call"] == 2
        assert snapshots[1]["msg_count"] == 6
        assert snapshots[1]["roles"] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert snapshots[1]["has_tool_calls"] is True

        # Final state: both turns compacted
        assert len(agent.messages) == 4
        assert not agent._has_tool_calls()

    @pytest.mark.asyncio
    async def test_three_turns_all_compacted(self):
        """Verify three tool-using turns are all compacted."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A"},
            {"role": "assistant", "content": "Found A"},
            {"role": "user", "content": "Read B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "B"},
            {"role": "assistant", "content": "Found B"},
            {"role": "user", "content": "Read C"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "3", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "3", "content": "C"},
            {"role": "assistant", "content": "Found C"},
        ]

        reflect_count = 0

        async def mock_reflect():
            nonlocal reflect_count
            reflect_count += 1
            return f"Round {reflect_count} summary"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert reflect_count == 3
        assert "3 turn" in message
        # user1 + asst1 + user2 + asst2 + user3 + asst3 = 6 messages
        assert len(agent.messages) == 6
        assert not agent._has_tool_calls()

    @pytest.mark.asyncio
    async def test_multi_round_tool_use_in_first_turn(self):
        """First turn has multi-round tool use (2 tool calls), second has single."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Read and edit file A"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "A contents"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "2", "name": "edit_file"}],
            },
            {"role": "tool", "tool_call_id": "2", "content": "edited"},
            {"role": "assistant", "content": "Fixed A"},
            {"role": "user", "content": "Read file B"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "3", "name": "read_file"}],
            },
            {"role": "tool", "tool_call_id": "3", "content": "B contents"},
            {"role": "assistant", "content": "Found B"},
        ]

        reflect_count = 0

        async def mock_reflect():
            nonlocal reflect_count
            reflect_count += 1
            return f"Round {reflect_count} summary"

        agent._reflect_on_tool_use = mock_reflect

        success, message = await agent.journal_all()
        assert success is True
        assert reflect_count == 2  # Multi-round counts as one turn
        assert "2 turn" in message
        # user1 + asst1 + user2 + asst2 = 4 messages
        assert len(agent.messages) == 4
        assert not agent._has_tool_calls()
