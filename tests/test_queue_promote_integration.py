#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""Integration tests: a bare "!"/"!!" in the TUI promotes the queued item.

Drives the real AgentTUI through Textual's run_test(), types into the real
input field and presses Enter, so the whole submit path runs - the prefix
parse, the promotion, and the notice in the chat.

The bug this pins: a bare "!!" used to strip to "" and call
add_message("", interrupt=True), queueing an EMPTY interrupt item that the
turn loop then injected as a blank user turn.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.durable_records import DurableRecords
from agent13.queue import AgentQueue


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
    # The status bar reads model/provider from the agent (single source of
    # truth), so the stub must carry real strings, not MagicMocks.
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
    """Keep test history/saves out of the real ~/.agent13."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))


async def _submit(pilot, app, text):
    """Type *text* into the real input field and press Enter."""
    field = app.query_one("#input-field")
    field.focus()
    field.load_text(text)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.2)


def _notice_text(app) -> str:
    """The chat's system-message lines, joined."""
    return " ".join(
        str(n.render()) for n in app.query_one("#chat").query(".system-message")
    )


@pytest.mark.asyncio
async def test_bare_steer_promotes_the_queued_item(isolated_config):
    app = _make_app()

    async with app.run_test() as pilot:
        app.agent.queue.add("queued work")
        await _submit(pilot, app, "!!")

        # Still exactly one item - no empty message was queued
        assert app.agent.queue.pending_count == 1
        item = app.agent.queue.items[0]
        assert item.text == "queued work"
        assert item.interrupt is True
        app.agent.add_message.assert_not_called()

        # The notice names the item, with the bracket rendered literally:
        # a "\]" here would mean escape_markup left a visible backslash.
        rendered = _notice_text(app)
        assert '[steer] "queued work"' in rendered
        assert "\\]" not in rendered


@pytest.mark.asyncio
async def test_bare_priority_promotes_the_queued_item(isolated_config):
    app = _make_app()

    async with app.run_test() as pilot:
        app.agent.queue.add("queued work")
        await _submit(pilot, app, "!")

        assert app.agent.queue.pending_count == 1
        item = app.agent.queue.items[0]
        assert item.text == "queued work"
        assert item.priority is True
        assert item.interrupt is False
        app.agent.add_message.assert_not_called()

        assert '[priority] "queued work"' in _notice_text(app)


@pytest.mark.asyncio
async def test_bare_steer_with_nothing_queued(isolated_config):
    app = _make_app()

    async with app.run_test() as pilot:
        await _submit(pilot, app, "!!")

        assert app.agent.queue.pending_count == 0
        app.agent.add_message.assert_not_called()
        assert "[steer] nothing queued" in _notice_text(app)


@pytest.mark.asyncio
async def test_bare_prefix_stays_out_of_history(isolated_config):
    """A bare "!"/"!!" is a control, not a message."""
    app = _make_app()

    async with app.run_test() as pilot:
        app.agent.queue.add("queued work")
        await _submit(pilot, app, "!!")

        assert "!!" not in app._history.get_all()


@pytest.mark.asyncio
async def test_prefixed_message_still_sends_normally(isolated_config):
    """Regression guard: "!!text" is a message, not a promotion."""
    app = _make_app()

    async with app.run_test() as pilot:
        app.agent.queue.add("queued work")
        await _submit(pilot, app, "!!do the thing")

        app.agent.add_message.assert_awaited_once_with(
            "do the thing", priority=True, interrupt=True
        )
        # The queued item was NOT promoted
        assert app.agent.queue.items[0].interrupt is False
