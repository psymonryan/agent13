#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for AgentTUI._shutdown MCP session draining (windows_release_fixes).

Quitting the TUI while an MCP server is still connecting leaves its
_session_runner parked in an await. on_unmount cancels the task, but the
SDK's shielded stdio shutdown (closing the child pipes + transport) needs
loop iterations to finish. Without a drain, asyncio.run's teardown
mass-cancels the task mid-unwind and the transports warn
"unclosed transport" / ValueError("I/O operation on closed pipe") at exit.

Fix: _shutdown awaits the cancelled session tasks (bounded) after
super()._shutdown(), while the loop is still open. These tests pin that.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.config import create_client


def _make_app(client):
    """Build an AgentTUI with a stub agent (same pattern as
    test_tui_client_loop.py)."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=client,
            model="test-model",
            model_names=["test-model"],
            provider="test",
        )

    agent = MagicMock()
    agent.set_client = MagicMock()
    agent.add_message = AsyncMock()
    agent.set_system_prompt = MagicMock()
    agent.stop = MagicMock()

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    agent.queue = MagicMock()
    agent.queue.pending_count = 0
    agent.queue.current = None
    agent.devel_mode = False
    agent.auto_context_chain_used = 0
    app.agent = agent
    app._update_info_content = MagicMock()
    return app, agent


@pytest.mark.asyncio
async def test_shutdown_drains_cancelled_mcp_session_tasks():
    """_shutdown must await the session tasks that on_unmount cancelled, so
    their async unwind (SDK stdio shutdown) completes before the loop
    closes."""
    original = create_client("http://localhost:9/v1", "key")
    app, agent = _make_app(original)
    try:
        # Stub MCP with one server whose session task is parked mid-connect.
        mcp = MagicMock()
        server = MagicMock()
        server.session_task = None
        mcp.servers = {"test": server}
        agent.mcp = mcp

        unwound = asyncio.Event()

        async def _session():
            try:
                await asyncio.Event().wait()  # parked: still connecting
            except asyncio.CancelledError:
                # Simulate the SDK's shielded shutdown: needs loop
                # iterations to close the child pipes + transport.
                await asyncio.sleep(0.05)
                unwound.set()
                raise

        async with app.run_test():
            server.session_task = asyncio.create_task(_session())
            await asyncio.sleep(0.05)  # let it park
            assert not server.session_task.done()

        # Exiting run_test() ran app._shutdown(): on_unmount cancelled the
        # task, then the drain awaited it.
        assert server.session_task.done(), (
            "session task not drained before loop close"
        )
        assert unwound.is_set(), (
            "session task was cancelled mid-unwind (drain did not run)"
        )
    finally:
        await original.close()
