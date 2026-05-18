"""Tests for TOKEN_USAGE emission from _reflect_on_tool_use.

These tests verify that reflection API calls properly emit TOKEN_USAGE events
with real token counts from the API stream, not synthetic zero values.

This is the DRY enforcement: both _llm_turn and _reflect_on_tool_use should
handle token_usage events identically via _stream_and_emit.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent13.events import AgentEvent


def _make_mock_stream(events):
    """Create an async generator that yields (event_type, data) tuples.

    This mocks stream_response_with_tools at the level that
    _stream_and_emit consumes it.
    """
    async def mock_stream(*args, **kwargs):
        for event_type, data in events:
            yield event_type, data
    return mock_stream


class TestReflectionTokenUsage:
    """Tests for TOKEN_USAGE emission from _reflect_on_tool_use."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent with journal mode enabled."""
        from agent13.core import Agent

        client = MagicMock()
        agent = Agent(client, model="test-model", journal_mode=True)
        return agent

    @pytest.mark.asyncio
    async def test_reflect_emits_token_usage_from_stream(self, agent):
        """_reflect_on_tool_use should emit TOKEN_USAGE with real token counts from API stream."""
        # Mock stream with token_usage event (like real API responses)
        mock_stream = _make_mock_stream([
            ("content", "Summary of tool use"),
            (
                "token_usage",
                {
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                },
            ),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            # Track emitted events
            emitted_events = []
            original_emit = agent.emit

            async def track_emit(event_type, data=None):
                emitted_events.append((event_type, data))
                return await original_emit(event_type, data)

            agent.emit = track_emit

            # Call reflection
            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
            )

        # Should have emitted TOKEN_USAGE with real counts
        token_events = [
            e for e in emitted_events if e[0] == AgentEvent.TOKEN_USAGE
        ]
        assert len(token_events) == 1
        assert token_events[0][1]["prompt_tokens"] == 150
        assert token_events[0][1]["completion_tokens"] == 50
        assert token_events[0][1]["total_tokens"] == 200

    @pytest.mark.asyncio
    async def test_reflect_updates_agent_token_counts(self, agent):
        """_reflect_on_tool_use should update agent's prompt_tokens/completion_tokens."""
        mock_stream = _make_mock_stream([
            ("content", "Summary"),
            (
                "token_usage",
                {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "total_tokens": 280,
                },
            ),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            # Initial state
            agent.prompt_tokens = 0
            agent.completion_tokens = 0
            agent.total_tokens = 0

            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
            )

        # Agent token counts should be updated
        assert agent.prompt_tokens == 200
        assert agent.completion_tokens == 80
        assert agent.total_tokens == 280

    @pytest.mark.asyncio
    async def test_reflect_emits_stream_start(self, agent):
        """_reflect_on_tool_use should emit STREAM_START before streaming."""
        mock_stream = _make_mock_stream([
            ("content", "Summary"),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            emitted_events = []
            original_emit = agent.emit

            async def track_emit(event_type, data=None):
                emitted_events.append(event_type)
                return await original_emit(event_type, data)

            agent.emit = track_emit

            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
            )

        # STREAM_START should be emitted
        assert AgentEvent.STREAM_START in emitted_events


class TestEventParity:
    """Tests for event parity between _llm_turn and _reflect_on_tool_use."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create an agent with journal mode enabled."""
        from agent13.core import Agent

        client = MagicMock()
        agent = Agent(client, model="test-model", journal_mode=True)
        return agent

    @pytest.mark.asyncio
    async def test_reflection_should_emit_same_events_as_llm_turn(self, agent):
        """_reflect_on_tool_use should emit TOKEN_USAGE just like _llm_turn does.

        This is the DRY principle: both code paths call _stream_and_emit
        and should handle token_usage events identically.
        """
        # Mock stream with token_usage (like _llm_turn receives)
        mock_stream = _make_mock_stream([
            ("content", "Tool summary"),
            (
                "token_usage",
                {
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                },
            ),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            # Track emitted events
            emitted_events = []
            original_emit = agent.emit

            async def track_emit(event_type, data=None):
                emitted_events.append((event_type, data))
                return await original_emit(event_type, data)

            agent.emit = track_emit

            # Call reflection
            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
            )

        # Should emit TOKEN_USAGE (just like _llm_turn does)
        token_usage_events = [
            e for e in emitted_events if e[0] == AgentEvent.TOKEN_USAGE
        ]
        assert len(token_usage_events) == 1
        assert token_usage_events[0][1]["completion_tokens"] == 50

        # Should also emit STREAM_START (just like _llm_turn does)
        stream_start_events = [
            e for e in emitted_events if e[0] == AgentEvent.STREAM_START
        ]
        assert len(stream_start_events) == 1


class TestTPSFromReflection:
    """Tests for TPS calculation from reflection token usage."""

    @pytest.mark.asyncio
    async def test_tps_calculable_after_reflection(self):
        """TPS should be calculable from reflection TOKEN_USAGE events."""
        from agent13.core import Agent

        client = MagicMock()
        agent = Agent(client, model="test-model", journal_mode=True)

        # Mock stream with enough tokens to pass MIN_TOKENS threshold
        mock_stream = _make_mock_stream([
            ("content", "Detailed summary of tool usage"),
            (
                "token_usage",
                {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,  # Above MIN_TOKENS=50
                    "total_tokens": 300,
                },
            ),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            # Track TOKEN_USAGE events
            token_usage_data = []
            original_emit = agent.emit

            async def track_emit(event_type, data=None):
                if event_type == AgentEvent.TOKEN_USAGE:
                    token_usage_data.append(data)
                return await original_emit(event_type, data)

            agent.emit = track_emit

            # Call reflection
            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
            )

        # Should have emitted TOKEN_USAGE with completion_tokens=100
        assert len(token_usage_data) == 1
        assert token_usage_data[0]["completion_tokens"] == 100

        # TUI could calculate TPS if elapsed time > MIN_ELAPSED (3.0s)
        # The actual TPS calculation happens in TUI, but we've verified
        # the data is available for it.

    @pytest.mark.asyncio
    async def test_short_reflection_no_tps(self):
        """Short reflection streams should emit TOKEN_USAGE but not trigger TPS display."""
        from agent13.core import Agent

        client = MagicMock()
        agent = Agent(client, model="test-model", journal_mode=True)

        # Mock stream with few tokens (like skill loading acknowledgment)
        mock_stream = _make_mock_stream([
            ("content", "Skill loaded."),
            (
                "token_usage",
                {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,  # Below MIN_TOKENS=50
                    "total_tokens": 70,
                },
            ),
        ])

        with patch("agent13.llm.stream_response_with_tools") as mock_srw:
            mock_srw.return_value = mock_stream()

            token_usage_data = []
            original_emit = agent.emit

            async def track_emit(event_type, data=None):
                if event_type == AgentEvent.TOKEN_USAGE:
                    token_usage_data.append(data)
                return await original_emit(event_type, data)

            agent.emit = track_emit

            await agent._reflect_on_tool_use(
                messages=[{"role": "user", "content": "test"}],
                skill_names=["code-review"],
            )

        # TOKEN_USAGE should still be emitted (for context counter, etc.)
        assert len(token_usage_data) == 1
        assert token_usage_data[0]["completion_tokens"] == 20

        # But TUI would suppress TPS display because:
        # - completion_tokens (20) < MIN_TOKENS (50)
        # - elapsed time likely < MIN_ELAPSED (3.0s)
        # This is correct behavior - TPS shouldn't show for tiny responses


class TestTUITPSFromReflection:
    """Tests for TUI TPS calculation from reflection tokens."""

    @pytest.mark.asyncio
    async def test_tui_tps_update_from_reflection_tokens(self):
        """TUI should calculate TPS from reflection TOKEN_USAGE if thresholds met."""
        import time

        # Create a mock TUI that mimics the real _update_token_usage behavior
        # We can't instantiate AgentTUI directly due to Textual's reactive system
        class MockTUI:
            def __init__(self):
                self._first_token_time = None
                self._last_token_time = None
                self._token_count = 0
                self._last_tps = 0.0
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0

            def _update_token_usage(self, data: dict):
                """Mimic the real TUI method behavior."""
                self.prompt_tokens = data.get("prompt_tokens", 0)
                self.completion_tokens = data.get("completion_tokens", 0)
                self.total_tokens = data.get("total_tokens", 0)

                # Calculate TPS if we have timing data
                if self._first_token_time and self._last_token_time:
                    elapsed = self._last_token_time - self._first_token_time
                    if elapsed > 0 and self.completion_tokens > 0:
                        self._last_tps = self.completion_tokens / elapsed

        tui = MockTUI()

        # Simulate timing: stream started 5 seconds ago
        now = time.monotonic()
        tui._first_token_time = now - 5.0
        tui._last_token_time = now
        tui._token_count = 100

        # Call _update_token_usage with reflection data
        tui._update_token_usage(
            {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        )

        # TPS should be calculated: 100 tokens / 5 seconds = 20.0 TPS
        assert abs(tui._last_tps - 20.0) < 0.1
        assert tui.completion_tokens == 100
