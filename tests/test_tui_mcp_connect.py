#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for the MCP connect gate (TUI side).

Contract:
1. While an MCP connect/reconnect is in flight (MCPManager.connecting),
   a user message goes straight into the queue — the agent loop defers
   starting a turn until the tool list is complete. The queue pane opens
   so the user sees the message queued; no extra "why" notice.
2. When a connect finishes (startup --mcp connect or /mcp connect), the
   status bar is refreshed so it shows "mcp: on".

The core gate (agent loop deferral) is covered in tests/test_core.py
(TestMCPConnectGate); the MCPManager.connecting flag lifecycle in
tests/test_mcp.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.queue import AgentQueue
from agent13.durable_records import DurableRecords


class FakeMCP:
    """Stands in for MCPManager — just the attributes the TUI reads."""

    def __init__(self, connecting: bool = True):
        self.connecting = connecting
        self.connect_all = AsyncMock(return_value={"slowtest": ["hello"]})
        self.servers = {}  # on_unmount teardown iterates this
        self._shutting_down = False

    def is_connected(self):
        return not self.connecting


def _make_app(mcp):
    """Build an AgentTUI with a stub agent wired to *mcp*."""
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
    agent._mcp_server_configs = [object()]
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
        import asyncio

        await asyncio.Event().wait()

    agent.run = _run_forever
    app.agent = agent
    return app


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Keep test history/saves out of the real ~/.agent13."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))


@pytest.mark.asyncio
async def test_mcp_connect_updates_status_bar(isolated_config):
    """/mcp connect refreshes the status bar when the connect finishes."""
    import asyncio

    app = _make_app(FakeMCP(connecting=True))
    app.update_status = MagicMock()

    async with app.run_test():
        app._handle_mcp_command("connect")
        # do_connect runs as a background task — poll for the refresh
        for _ in range(100):
            if app.update_status.called:
                break
            await asyncio.sleep(0.05)
        assert app.update_status.called, (
            "status bar must be refreshed when /mcp connect finishes"
        )


@pytest.mark.asyncio
async def test_startup_connect_updates_status_bar(isolated_config):
    """The startup (--mcp) background connect refreshes the status bar."""
    app = _make_app(FakeMCP(connecting=True))
    app.update_status = MagicMock()

    async with app.run_test():
        await app._connect_mcp_on_startup()
        assert app.update_status.called


@pytest.mark.asyncio
async def test_startup_connect_updates_status_bar_on_error(isolated_config):
    """A failed startup connect still refreshes the status bar."""
    mcp = FakeMCP(connecting=True)
    mcp.connect_all = AsyncMock(side_effect=RuntimeError("all servers down"))
    app = _make_app(mcp)
    app.update_status = MagicMock()

    async with app.run_test():
        await app._connect_mcp_on_startup()
        assert app.update_status.called


@pytest.mark.asyncio
async def test_submit_during_mcp_connect_opens_queue_pane(isolated_config):
    """A message sent while MCP is connecting goes to the queue and the
    queue pane opens (the user sees it queued) — no turn starts."""
    mcp = FakeMCP(connecting=True)
    app = _make_app(mcp)

    async with app.run_test() as pilot:
        input_field = app.query_one("#input-field")
        input_field.focus()
        await pilot.pause()

        await pilot.press(*"hello there")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Message went to the agent queue...
        agent_add = app.agent.add_message
        assert agent_add.await_count == 1
        (text_arg,), _ = agent_add.await_args
        assert "hello there" in text_arg
        # ...and the queue pane opened so the user sees it queued
        assert app._info_pane_mode == "queue"
        # ...and no turn is in flight (agent idle during the connect)
        assert not app.processing


@pytest.mark.asyncio
async def test_submit_when_idle_and_not_connecting_no_queue_pane(isolated_config):
    """Negative control: idle agent + no connect in flight -> no queue pane."""
    mcp = FakeMCP(connecting=False)
    app = _make_app(mcp)
    app._info_pane_mode = None

    async with app.run_test() as pilot:
        input_field = app.query_one("#input-field")
        input_field.focus()
        await pilot.pause()

        await pilot.press(*"hello there")
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert app.agent.add_message.await_count == 1
        assert app._info_pane_mode != "queue"
