#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for TUI notification (toast) text colors.

The toast text is markup-colored per level in ``_show_notification``.
"blue" (ANSI 34, #0000ff) is a DARK blue - unreadable on the dark-mode
panel background the toasts render on. Info-level toasts must use a
bright blue (SGR 94) so the writing stays legible.
"""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
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


async def _render_notification_ansi(app, level: str, message: str = "toast text") -> str:
    """Show a toast at `level` and return its rendered ANSI output (16-color
    console, so colors come out as stable SGR codes)."""
    await app._show_notification(message, duration=60.0, level=level)
    await asyncio.sleep(0)
    # Latest panel: earlier toasts in the same app are still mounted
    # (60 s dismiss) and query_one would return the first one.
    panels = app.query_one("#chat").query(".notification-panel")
    panel = panels[-1]
    buf = io.StringIO()
    console = Console(
        file=buf, color_system="standard", force_terminal=True, width=100
    )
    console.print(panel.render())
    return buf.getvalue()


@pytest.mark.asyncio
async def test_info_notification_uses_bright_blue(isolated_config):
    """Info toasts (e.g. 'Chain restarted - records injected') must be
    legible on the dark panel: bright blue (SGR 94), not dark blue (SGR 34)."""
    app = _make_app(FakeMCP())

    async with app.run_test() as pilot:
        await pilot.pause()
        out = await _render_notification_ansi(app, "info")

    assert "\x1b[94m" in out, f"expected bright blue (94), got: {out!r}"
    assert "\x1b[34m" not in out, "dark blue (34) is unreadable on dark bg"


@pytest.mark.asyncio
async def test_warning_and_error_levels_unchanged(isolated_config):
    """Only the info level was dark; warning (bright yellow 93) and error
    (red 31) must keep their colors."""
    app = _make_app(FakeMCP())

    async with app.run_test() as pilot:
        await pilot.pause()
        warning_out = await _render_notification_ansi(app, "warning")
        error_out = await _render_notification_ansi(app, "error")

    assert "\x1b[93m" in warning_out, f"expected bright yellow (93), got: {warning_out!r}"
    assert "\x1b[31m" in error_out, f"expected red (31), got: {error_out!r}"


def test_textual_markup_drops_unknown_style_names():
    """Trap pin: Textual's Content.from_markup SILENTLY DROPS style names it
    does not recognize (e.g. rich's "bright_blue" renders as PLAIN TEXT)
    instead of erroring. Any color added to a toast markup must be a
    Textual-palette name, verified by this rendering test."""
    from textual.content import Content

    buf = io.StringIO()
    console = Console(
        file=buf, color_system="standard", force_terminal=True, width=80
    )
    console.print(Content.from_markup("[bright_blue]hi[/]"))
    out = buf.getvalue()
    assert "\x1b[" not in out, (
        "precondition: 'bright_blue' is not a Textual markup color and "
        f"renders unstyled - {out!r}"
    )
