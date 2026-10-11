#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for the TUI UI-event guard.

Textual 8.2.8's Screen._forward_event selection branch dereferences the
widget under the cursor. If that widget is unmounted in the same instant
as a MouseDown (a race with the streaming/collapse re-render in the
chat), the lookup returns a detached widget and the branch crashes on
``container.region`` (parent is None) - screen.py:1914, AttributeError.
The exception propagates into the App's message loop and Textual's
default _handle_exception exits the whole app.

The guard (ui/tui.py) contains the failure in two layers:
1. _install_screen_event_guard wraps Screen._forward_event so the known
   crash is swallowed where it happens - the TUI stays fully interactive.
2. _handle_exception backstop: any other leaked runtime exception keeps
   the process and the agent alive (forced quit saves as usual);
   pre-mount it still defers to Textual's loud exit.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.events import MouseDown
from textual.geometry import Offset
from textual.widgets import Markdown
from textual.widgets._markdown import MarkdownParagraph

from agent13.queue import AgentQueue
from agent13.durable_records import DurableRecords


class FakeMCP:
    """Stands in for MCPManager - just the attributes the TUI reads."""

    def __init__(self, connecting: bool = False):
        self.connecting = connecting
        self.connect_all = AsyncMock(return_value={})
        self.servers = {}
        self._shutting_down = False

    def is_connected(self):
        return not self.connecting


def _make_app(mcp):
    """Build an AgentTUI with a stub agent (same pattern as the other
    test_tui_* files)."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
        )

    agent = MagicMock()
    agent.durable_records = DurableRecords()
    agent.mcp = mcp
    agent._mcp = mcp
    agent._mcp_server_configs = []
    agent._ensure_mcp = AsyncMock(return_value=mcp)
    agent.add_message = AsyncMock()
    agent.queue = AgentQueue()
    agent.is_pausing = False
    agent.is_paused = False
    agent.status = MagicMock()
    agent.status.value = "idle"
    agent.tool_stats = MagicMock()
    agent.tool_stats.total_calls = 0
    agent.auto_context_chain_used = 0

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    app.agent = agent
    return app


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Keep test history/saves out of the real ~/.agent13."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))


def _make_stale_mouse_down(app, paragraph) -> MouseDown:
    """Simulate the race: the cursor lookup returns a widget that is
    already detached (parent is None)."""
    app.screen.get_widget_and_offset_at = MagicMock(
        return_value=(paragraph, Offset(0, 0))
    )
    return MouseDown(
        None, 0, 0, 0, 0, button=1, shift=False, meta=False, ctrl=False,
        screen_x=0, screen_y=0,
    )


@pytest.mark.asyncio
async def test_mouse_down_on_detached_widget_does_not_kill_app(isolated_config):
    """The exact screen.py:1914 crash is contained: no exit, the dim
    notice lands in the chat, and the app's message loop keeps running."""
    app = _make_app(FakeMCP())

    async with app.run_test() as pilot:
        chat = app.query_one("#chat")
        md = Markdown("**hello** *world*")
        await chat.mount(md)
        await pilot.pause()

        paragraph = md.query_one(MarkdownParagraph)
        # The unguarded original, bound from the class (the instance attr
        # is already the guard installed in on_mount).
        from textual.screen import Screen

        original_forward = Screen._forward_event.__get__(app.screen, Screen)
        await paragraph.remove()  # detach: parent becomes None

        event = _make_stale_mouse_down(app, paragraph)

        # Negative control: the unguarded path really does crash here -
        # proves the repro drives the buggy selection branch.
        with pytest.raises(AttributeError):
            original_forward(event)

        # The guarded path contains it. (Fresh event: the one above was
        # marked forwarded by the unguarded call and would no-op.)
        app.screen._forward_event(_make_stale_mouse_down(app, paragraph))
        await pilot.pause(0.2)

        assert app.return_code is None, "app must not exit"
        assert app.screen.is_mounted, "screen must stay mounted"
        notices = app.query_one("#chat").query(".system-message")
        assert any("UI event error ignored" in str(n.render()) for n in notices)


@pytest.mark.asyncio
async def test_screen_still_interactive_after_contained_error(isolated_config):
    """After the contained crash the App's message loop is alive: keys
    still reach the focused input field."""
    app = _make_app(FakeMCP())

    async with app.run_test() as pilot:
        chat = app.query_one("#chat")
        md = Markdown("**hello** *world*")
        await chat.mount(md)
        await pilot.pause()

        paragraph = md.query_one(MarkdownParagraph)
        await paragraph.remove()
        event = _make_stale_mouse_down(app, paragraph)
        app.screen._forward_event(event)
        await pilot.pause(0.2)

        input_field = app.query_one("#input-field")
        input_field.focus()
        await pilot.press("a")
        await pilot.pause(0.2)
        assert input_field.text == "a", (
            "input must still work after a contained UI event error"
        )


@pytest.mark.asyncio
async def test_handle_exception_backstop_keeps_app_alive(isolated_config):
    """The backstop: a runtime exception that still leaks out (any other
    Textual internal) keeps the process alive with the dim notice."""
    app = _make_app(FakeMCP())

    async with app.run_test() as pilot:
        app._handle_exception(RuntimeError("boom"))
        await pilot.pause(0.2)

        assert app.return_code is None
        assert app.screen.is_mounted
        chat = app.query_one("#chat")
        notices = chat.query(".system-message")
        assert any("UI event error ignored" in str(n.render()) for n in notices)
