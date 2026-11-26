#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""The input cursor doubles as a terminal-focus indicator.

The signal comes from the terminal's focus reporting (CSI ?1004), which
Textual turns into AppFocus/AppBlur. tmux only forwards it when
``focus-events`` is on - see ~/.tmux.conf.

Two things are deliberately NOT Textual's own:

1. ``AgentTUI._on_app_blur`` does not call super(). Textual's handler sets
   ``app_focus``, which triggers ``_watch_app_focus`` ->
   ``screen.update_node_styles()`` - a CSS re-application over EVERY widget.
   Measured on a real chat: 1.65 s blur / 2.12 s focus at 6,109 widgets,
   5.18 s / 5.00 s at 18,309. We track our own flag and repaint one widget.

2. The input's border is always present, only its colour changes on focus.
   It used to exist only on ``:focus``, so losing focus also removed 2 rows
   and reflowed the whole screen.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual import events

from agent13.durable_records import DurableRecords
from agent13.queue import AgentQueue
from ui.chat_input import ChatTextArea


class FakeMCP:
    """Stands in for MCPManager - just the attributes the TUI reads."""

    def __init__(self):
        self.connecting = False
        self.connect_all = AsyncMock(return_value={})
        self.servers = {}
        self._shutting_down = False

    def is_connected(self):
        return True


def _make_app():
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
    agent.model = "test-model"
    agent.provider_name = "test"
    mcp = FakeMCP()
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
    """Keep test history/saves out of the real ~/.agent13, and make the
    tmux probe deterministic.

    TMUX/TMUX_PANE are dropped because this suite may itself run inside a
    tmux pane: without this, every blur test would fork a real `tmux
    display-message` and the answer would depend on which pane the test
    happens to be running in.
    """
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("TMUX_PANE", raising=False)


@pytest.mark.asyncio
async def test_cursor_drawn_when_terminal_focused(isolated_config):
    app = _make_app()

    async with app.run_test():
        field = app.query_one("#input-field", ChatTextArea)
        assert app._terminal_focused is True
        assert field._draw_cursor is True


@pytest.mark.asyncio
async def test_app_blur_event_hides_the_cursor(isolated_config):
    app = _make_app()

    async with app.run_test() as pilot:
        field = app.query_one("#input-field", ChatTextArea)

        app.post_message(events.AppBlur())
        await pilot.pause(0.2)
        assert app._terminal_focused is False
        assert field._draw_cursor is False

        app.post_message(events.AppFocus())
        await pilot.pause(0.2)
        assert app._terminal_focused is True
        assert field._draw_cursor is True


@pytest.mark.asyncio
async def test_blur_does_not_touch_app_focus(isolated_config):
    """The performance contract: Textual's app_focus machinery is bypassed.

    If this ever fails, Textual's handler is running again and every window
    switch is back to a multi-second full-tree style walk.
    """
    app = _make_app()

    async with app.run_test() as pilot:
        app.post_message(events.AppBlur())
        await pilot.pause(0.2)
        assert app.app_focus is True, "app_focus must stay untouched"

        app.post_message(events.AppFocus())
        await pilot.pause(0.2)
        assert app.app_focus is True


@pytest.mark.asyncio
async def test_blur_event_does_not_reflow_the_input(isolated_config):
    """The regression: the border used to exist only on :focus, so losing
    focus removed 2 rows and reflowed the whole screen."""
    app = _make_app()

    async with app.run_test() as pilot:
        field = app.query_one("#input-field", ChatTextArea)
        await pilot.pause(0.2)
        focused_region = field.region

        app.post_message(events.AppBlur())
        await pilot.pause(0.2)
        assert field.region == focused_region, "blur must not move or resize it"

        app.post_message(events.AppFocus())
        await pilot.pause(0.2)
        assert field.region == focused_region


@pytest.mark.asyncio
async def test_refresh_focus_appearance_drops_the_line_cache(isolated_config):
    """The cache key does not include the focus flag, so refresh() alone
    would repaint the same cached Strip - cursor and all."""
    app = _make_app()

    async with app.run_test():
        field = app.query_one("#input-field", ChatTextArea)
        field._line_cache[("sentinel",)] = "cached line"

        field.refresh_focus_appearance()
        # The stale entry is gone. (The repaint may already have cached a
        # fresh line, so assert on the key, not the size.)
        assert ("sentinel",) not in field._line_cache


# -- the tmux pane-visibility probe -----------------------------------------
#
# tmux sends the same focus-out whether focus moved to another pane in this
# window or the whole window was switched away. Repainting in the second case
# is pure noise: it sets the window's activity flag (the yellow tab) for a
# pane nobody is looking at. The distinction lives only in the tmux server,
# so it is asked for - and ONLY when tmux is actually in use.


def _fake_proc(value: bytes):
    """A stand-in for an asyncio subprocess whose output is ``value``."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(value, b""))
    return proc


@pytest.mark.asyncio
async def test_pane_visible_without_tmux(isolated_config):
    """No tmux: nothing to ask, so the pre-existing behaviour stands."""
    app = _make_app()
    async with app.run_test():
        assert await app._pane_still_visible() is True


@pytest.mark.asyncio
async def test_pane_visible_when_window_is_active(isolated_config, monkeypatch):
    """Focus moved to another pane in the SAME window: still on screen."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()
    async with app.run_test():
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(b"1\n")),
        ):
            assert await app._pane_still_visible() is True


@pytest.mark.asyncio
async def test_pane_not_visible_when_window_switched_away(
    isolated_config, monkeypatch
):
    """The whole window went away: the pane is not on screen."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()
    async with app.run_test():
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(b"0\n")),
        ):
            assert await app._pane_still_visible() is False


@pytest.mark.asyncio
async def test_pane_probe_fails_safe(isolated_config, monkeypatch):
    """If tmux cannot be asked, behave as we did before (hide the cursor)."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()
    async with app.run_test():
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=OSError("no tmux")),
        ):
            assert await app._pane_still_visible() is True


@pytest.mark.asyncio
async def test_window_switch_does_not_repaint(isolated_config, monkeypatch):
    """The whole point: a window switch must write NOTHING.

    A repaint here is what sets tmux's activity flag on a window the user is
    looking at, which is the yellow tab on every ^a^a.
    """
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()

    async with app.run_test() as pilot:
        app._refresh_input_cursor = MagicMock()

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(b"0\n")),
        ):
            app.post_message(events.AppBlur())
            await pilot.pause(0.3)

        assert app._terminal_focused is True, "cursor must stay as it was"
        app._refresh_input_cursor.assert_not_called()

        # And coming back must not repaint either.
        app.post_message(events.AppFocus())
        await pilot.pause(0.3)
        app._refresh_input_cursor.assert_not_called()


@pytest.mark.asyncio
async def test_pane_switch_still_hides_the_cursor(isolated_config, monkeypatch):
    """The feature must still work where it is wanted."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()

    async with app.run_test() as pilot:
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(b"1\n")),
        ):
            app.post_message(events.AppBlur())
            await pilot.pause(0.3)

        assert app._terminal_focused is False

        app.post_message(events.AppFocus())
        await pilot.pause(0.3)
        assert app._terminal_focused is True


@pytest.mark.asyncio
async def test_focus_returning_during_the_probe_keeps_the_cursor(
    isolated_config, monkeypatch
):
    """A slow tmux query must not hide the cursor after focus came back."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    app = _make_app()

    release = asyncio.Event()

    async def slow_communicate():
        await release.wait()
        return (b"1\n", b"")

    proc = MagicMock()
    proc.communicate = slow_communicate

    async with app.run_test() as pilot:
        with patch(
            "asyncio.create_subprocess_exec", AsyncMock(return_value=proc)
        ):
            app.post_message(events.AppBlur())
            await pilot.pause(0.1)  # blur is now parked in the probe

            app.post_message(events.AppFocus())
            await pilot.pause(0.1)

            release.set()
            await pilot.pause(0.3)

        assert app._terminal_focused is True, (
            "the stale blur must not hide the cursor"
        )
