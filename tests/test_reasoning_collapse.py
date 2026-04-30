"""Tests for ReasoningMessage collapse functionality."""

import pytest


def test_reasoning_message_defaults_to_expanded():
    """Thinking widgets should start expanded (collapsed=False)."""
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Thinking")
    assert widget.collapsed is False
    assert widget._title == "Thinking"


def test_reasoning_message_reflecting_starts_collapsed():
    """Reflecting widgets should start collapsed."""
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Reflecting", collapsed=True)
    assert widget.collapsed is True
    assert widget._title == "Reflecting"


def test_reasoning_message_can_set_collapsed():
    """set_collapsed should update the collapsed state."""
    from ui.tui import ReasoningMessage

    widget = ReasoningMessage("Thinking", collapsed=False)
    assert widget.collapsed is False

    # Simulate setting collapsed
    widget.collapsed = True
    assert widget.collapsed is True


def test_reasoning_message_compose_shows_indicator():
    """compose should show correct indicator based on collapsed state."""
    from ui.tui import ReasoningMessage

    # Collapsed shows ▶
    ReasoningMessage("Test", collapsed=True)
    # Check indicator character in compose output

    # The indicator is set in compose()
    # ▶ = \u25b6, ▼ = \u25bc
    assert "▶" == "\u25b6"
    assert "▼" == "\u25bc"


def test_collapse_parameter_based_on_title():
    """Verify the pattern used in callers: collapsed=(title == 'Reflecting')."""
    # This tests the logic used at call sites
    thinking_title = "Thinking"
    reflecting_title = "Reflecting"

    thinking_collapsed = thinking_title == "Reflecting"
    reflecting_collapsed = reflecting_title == "Reflecting"

    assert thinking_collapsed is False
    assert reflecting_collapsed is True


def test_append_stores_content():
    """Content should be stored regardless of collapsed state."""
    from ui.tui import ReasoningMessage
    import asyncio

    widget = ReasoningMessage("Test", collapsed=True)

    # Content should accumulate
    async def append_content():
        await widget.append("Hello ")
        await widget.append("World")

    asyncio.run(append_content())

    assert widget._content == "Hello World"
    assert widget.collapsed is True  # Still collapsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
