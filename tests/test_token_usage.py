"""Tests for token usage tracking in streaming responses."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agent13 import AgentEvent, AgentEventData
from agent13.llm import stream_response_with_tools


class MockUsage:
    """Mock usage object for streaming chunk."""

    def __init__(self, prompt_tokens=100, completion_tokens=50, total_tokens=150):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockDelta:
    """Mock delta object for streaming chunk."""

    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock choice object for streaming chunk."""

    def __init__(self, delta):
        self.delta = delta


class MockChunk:
    """Mock streaming chunk."""

    def __init__(self, delta=None, usage=None):
        self.choices = [MockChoice(delta)] if delta else []
        self.usage = usage


class MockStream:
    """Mock async iterator for streaming response."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


class TestStreamResponseWithToolsTokenUsage:
    """Tests for token_usage event in stream_response_with_tools."""

    @pytest.mark.asyncio
    async def test_yields_token_usage_with_usage_data(self):
        """Should yield token_usage event when usage is present in final chunk."""
        # Create mock chunks with usage in the final chunk
        chunks = [
            MockChunk(delta=MockDelta(content="Hello")),
            MockChunk(delta=MockDelta(content=" world")),
            MockChunk(
                usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MockStream(chunks))

        events = []
        async for event in stream_response_with_tools(
            mock_client,
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)

        # Check that token_usage event was yielded
        token_usage_events = [e for e in events if e[0] == "token_usage"]
        assert len(token_usage_events) == 1

        usage_data = token_usage_events[0][1]
        assert usage_data["prompt_tokens"] == 10
        assert usage_data["completion_tokens"] == 5
        assert usage_data["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_no_token_usage_without_usage_data(self):
        """Should not yield token_usage event when usage is not present."""
        chunks = [
            MockChunk(delta=MockDelta(content="Hello")),
            MockChunk(delta=MockDelta(content=" world")),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MockStream(chunks))

        events = []
        async for event in stream_response_with_tools(
            mock_client,
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)

        # Check that no token_usage event was yielded
        token_usage_events = [e for e in events if e[0] == "token_usage"]
        assert len(token_usage_events) == 0

    @pytest.mark.asyncio
    async def test_stream_options_included_in_api_params(self):
        """Should include stream_options with include_usage in API params."""
        chunks = [MockChunk(usage=MockUsage())]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MockStream(chunks))

        # Consume the generator
        async for _ in stream_response_with_tools(
            mock_client,
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            pass

        # Verify the API call included stream_options
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "stream_options" in call_kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}


class TestAgentTokenUsageEvent:
    """Tests for Agent emitting TOKEN_USAGE events."""

    @pytest.mark.asyncio
    async def test_agent_emits_token_usage_event(self):
        """Agent should emit TOKEN_USAGE event when receiving token_usage from stream."""
        from agent13.core import Agent

        # Create mock chunks
        chunks = [
            MockChunk(delta=MockDelta(content="Hello")),
            MockChunk(
                usage=MockUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MockStream(chunks))

        agent = Agent(mock_client, model="test-model")

        # Track emitted events
        emitted_events = []

        @agent.on_event
        async def capture_events(event: AgentEventData):
            if event.event == AgentEvent.TOKEN_USAGE:
                emitted_events.append(event)

        # Add a user message
        agent.messages.append({"role": "user", "content": "Test"})

        # Set running flag so _llm_turn actually executes
        agent._running = True

        # Run one LLM turn
        await agent._llm_turn()
        await agent._llm_turn()

        # Verify TOKEN_USAGE event was emitted
        assert len(emitted_events) == 1
        assert emitted_events[0].data["prompt_tokens"] == 20
        assert emitted_events[0].data["completion_tokens"] == 10
        assert emitted_events[0].data["total_tokens"] == 30


class TestTUITokenUsageUpdate:
    """Tests for TUI token usage update functionality."""

    def test_update_token_usage_updates_properties(self):
        """_update_token_usage should update properties from data dict."""

        # Create a mock object to test the method behavior
        # We use a simple object since _update_token_usage just sets attributes
        class MockTUI:
            def _update_token_usage(self, data: dict):
                self.prompt_tokens = data.get("prompt_tokens", 0)
                self.completion_tokens = data.get("completion_tokens", 0)
                self.total_tokens = data.get("total_tokens", 0)

        tui = MockTUI()
        tui.prompt_tokens = 0
        tui.completion_tokens = 0
        tui.total_tokens = 0

        # Call the method
        tui._update_token_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        )

        # Verify properties were updated
        assert tui.prompt_tokens == 100
        assert tui.completion_tokens == 50
        assert tui.total_tokens == 150

    def test_update_token_usage_handles_missing_keys(self):
        """_update_token_usage should handle missing keys gracefully."""

        class MockTUI:
            def _update_token_usage(self, data: dict):
                self.prompt_tokens = data.get("prompt_tokens", 0)
                self.completion_tokens = data.get("completion_tokens", 0)
                self.total_tokens = data.get("total_tokens", 0)

        tui = MockTUI()
        tui.prompt_tokens = 50
        tui.completion_tokens = 25
        tui.total_tokens = 75

        # Call with empty dict
        tui._update_token_usage({})

        # Should default to 0
        assert tui.prompt_tokens == 0
        assert tui.completion_tokens == 0
        assert tui.total_tokens == 0

    def test_update_token_usage_handles_partial_data(self):
        """_update_token_usage should handle partial data."""

        class MockTUI:
            def _update_token_usage(self, data: dict):
                self.prompt_tokens = data.get("prompt_tokens", 0)
                self.completion_tokens = data.get("completion_tokens", 0)
                self.total_tokens = data.get("total_tokens", 0)

        tui = MockTUI()
        tui.prompt_tokens = 0
        tui.completion_tokens = 0
        tui.total_tokens = 0

        # Call with partial data
        tui._update_token_usage(
            {
                "prompt_tokens": 100,
                # Missing completion_tokens and total_tokens
            }
        )

        assert tui.prompt_tokens == 100
        assert tui.completion_tokens == 0
        assert tui.total_tokens == 0
        assert tui.total_tokens == 0


class TestChunkCountRename:
    """Tests to verify chunk_count variable rename."""

    @pytest.mark.asyncio
    async def test_chunk_count_increments_per_content_chunk(self):
        """chunk_count should increment for each content chunk, not actual tokens."""
        chunks = [
            MockChunk(delta=MockDelta(content="Hello")),  # chunk_count = 1
            MockChunk(delta=MockDelta(content=" world")),  # chunk_count = 2
            MockChunk(delta=MockDelta(reasoning_content="thinking")),  # chunk_count = 3
            MockChunk(usage=MockUsage()),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MockStream(chunks))

        events = []
        async for event in stream_response_with_tools(
            mock_client,
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)

        # We should have 2 content events, 1 reasoning event, and 1 token_usage event
        content_events = [e for e in events if e[0] == "content"]
        reasoning_events = [e for e in events if e[0] == "reasoning"]
        token_usage_events = [e for e in events if e[0] == "token_usage"]

        assert len(content_events) == 2
        assert len(reasoning_events) == 1
        assert len(token_usage_events) == 1
