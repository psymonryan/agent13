#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for TUI client loop separation.

The CLI runs the model fetch in a setup event loop (asyncio.run(async_main())),
then Textual runs the TUI in a fresh loop (app.run()). A client that has
already connected in the setup loop holds pooled HTTP connections bound to
that now-closed loop; its first TUI request fails with
"RuntimeError: Event loop is closed" (visible with openai >= 3.14, which no
longer retries raw RuntimeErrors).

Fix: the CLI closes the setup-loop client and passes provider_args to the
TUI, which creates a fresh client in on_mount (the live loop). These tests
pin that wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.config import create_client


def _make_app(client, provider_args=None):
    """Build an AgentTUI with a stub agent that records set_client calls."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=client,
            model="test-model",
            model_names=["test-model"],
            provider="test",
            provider_args=provider_args,
        )

    agent = MagicMock()
    agent.set_client = MagicMock()
    agent.add_message = AsyncMock()
    agent.set_system_prompt = MagicMock()

    async def _run_forever():
        import asyncio

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
async def test_on_mount_creates_fresh_client_when_provider_args_given():
    """With provider_args, on_mount builds a new client in the live loop and
    points the agent at it (the setup-loop client is not reused)."""
    original = create_client("http://localhost:9/v1", "key")
    app, agent = _make_app(
        original, provider_args=("http://localhost:9/v1", "key", 1.0, 1.0)
    )
    try:
        async with app.run_test():
            assert app.client is not original
            assert isinstance(app.client, type(original))
            # Agent must be pointed at the new client (single source of truth).
            agent.set_client.assert_called_once_with(app.client)
    finally:
        await original.close()
        await app.client.close()


@pytest.mark.asyncio
async def test_on_mount_keeps_client_when_no_provider_args():
    """Without provider_args (offline providers like fake), the original
    client is kept - no new client is created."""
    original = create_client("http://localhost:9/v1", "key")
    app, agent = _make_app(original)
    try:
        async with app.run_test():
            assert app.client is original
            agent.set_client.assert_not_called()
    finally:
        await original.close()
