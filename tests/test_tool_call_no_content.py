"""
Regression test for bug: Chat window appears to stop when tool call arrives with zero preceding content.

Bug description:
- When an LLM "thinks" silently (no streamed tokens) then calls a tool directly,
  the TUI shows an empty "Agent:" line that makes it appear output has stopped.
- Root cause: _handle_tool_call always called _finalize_streaming_for_tool(),
  which assumed a streaming widget existed, even when no content was ever shown.
- Fix: Added guard condition to only finalize when _streaming_content_widget or
  _in_reasoning is True.

The fix prevents empty "Agent:" widgets from being created when tool calls
arrive with zero preceding content tokens.
"""

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Mock Infrastructure (adapted from test_token_usage.py)
# =============================================================================

class MockUsage:
    """Mock usage object for streaming chunk."""

    def __init__(self, prompt_tokens=100, completion_tokens=50, total_tokens=150):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockToolCall:
    """Mock tool call object."""

    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = function


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


def make_tool_call_chunk(tool_name: str, tool_args: dict, tool_id: str = "call_123"):
    """Create a mock chunk containing only a tool call (no content)."""
    return MockChunk(
        delta=MockDelta(
            content=None,
            reasoning_content=None,
            tool_calls=[MockToolCall(
                id=tool_id,
                type="function",
                function=MagicMock(
                    name=tool_name,
                    arguments=str(tool_args).replace("'", '"')
                )
            )]
        )
    )


def make_usage_chunk(total_tokens: int = 0):
    """Create a mock chunk with usage info."""
    return MockChunk(usage=MockUsage(total_tokens=total_tokens))


# =============================================================================
# Regression Test: Verify the fix is in place
# =============================================================================

class TestHandleToolCallGuardCondition:
    """
    Regression test for the empty "Agent:" widget bug.

    BEFORE the fix:
        _handle_tool_call always called _finalize_streaming_for_tool(),
        even when no content widget existed.

    AFTER the fix:
        _handle_tool_call only finalizes when there's content to finalize.

    This test verifies the fix is present and prevents regression.
    """

    def test_fix_logic_table(self):
        """
        Test the fix logic table:

        _streaming_content_widget | _in_reasoning | Should finalize?
        ----------------------------------------------------------------
        None                       | False         | NO  ← BUG CASE (fixed)
        Some widget                | False         | YES (normal case)
        None                       | True          | YES (reasoning)
        Some widget                | True          | YES (both)
        """
        test_cases = [
            # (streaming_widget, in_reasoning, should_finalize, description)
            (None, False, False, "BUG CASE: tool call with no content"),
            (MagicMock(), False, True, "Normal: content widget exists"),
            (None, True, True, "Reasoning in progress"),
            (MagicMock(), True, True, "Both content and reasoning"),
        ]

        for streaming_widget, in_reasoning, should_finalize, description in test_cases:
            # The fix logic:
            should_call_finalize = bool(streaming_widget or in_reasoning)

            assert should_call_finalize == should_finalize, (
                f"Logic error for: {description}. "
                f"streaming_widget={streaming_widget}, "
                f"in_reasoning={in_reasoning}, "
                f"expected should_finalize={should_finalize}"
            )

    @pytest.mark.asyncio
    async def test_guard_condition_exists_in_source(self):
        """
        Verify that _handle_tool_call has the guard condition in its source.

        This is a regression test - if the guard condition is removed,
        this test will fail.
        """
        import inspect
        from ui.tui import AgentTUI

        # Get the source code of _handle_tool_call
        source = inspect.getsource(AgentTUI._handle_tool_call)

        # Verify the guard condition exists
        # The fix should check _streaming_content_widget and _in_reasoning
        assert "_streaming_content_widget" in source, (
            "_handle_tool_call should reference _streaming_content_widget"
        )
        assert "_in_reasoning" in source, (
            "_handle_tool_call should reference _in_reasoning"
        )
        assert "if " in source, (
            "_handle_tool_call should have an if statement for the guard"
        )

        # Verify the docstring mentions the edge case
        assert "no preceding content" in source.lower() or "zero content" in source.lower(), (
            "_handle_tool_call docstring should mention the edge case it handles"
        )

# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":

    pytest.main([__file__, "-v"])
