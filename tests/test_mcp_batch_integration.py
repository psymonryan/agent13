"""Integration tests for `--mcp` in batch (`-p`) mode.

Regression for the silent-no-op bug where `run_batch_with_display` accepted
no `connect_mcp` kwarg and `args.mcp` was never wired into the batch call
site (cli.py). The flag only worked in TUI mode.

Only the LLM is mocked. Everything else is real: real MCPManager, real
FastMCP in-memory server (transport patched), real Agent, real
run_batch_with_display. This mirrors the pattern in
test_mcp_integration.py.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from agent13.config import MCPServerConfig
from agent13.mcp import MCPManager

# Reuse the in-memory stateful server helpers from the sibling test module.
from tests.test_mcp_integration import _make_stateful_server


# ---------------------------------------------------------------------------
# Wiring test: signature + call site
# ---------------------------------------------------------------------------


def test_run_batch_with_display_accepts_connect_mcp_kwarg():
    """The batch entry point must accept connect_mcp (regression: was missing)."""
    from agent13.cli import run_batch_with_display

    sig = inspect.signature(run_batch_with_display)
    assert "connect_mcp" in sig.parameters, (
        "run_batch_with_display must accept connect_mcp kwarg so --mcp works "
        "in batch mode"
    )
    assert sig.parameters["connect_mcp"].default is False


# ---------------------------------------------------------------------------
# Integration test: MCP tools actually load + disconnect runs on exit
# ---------------------------------------------------------------------------


@pytest.fixture
def stdio_config_for_batch():
    return MCPServerConfig(
        name="batchsrv",
        transport="stdio",
        command="echo",
        args=["ignored"],
        connect_timeout=10.0,
        tool_timeout=10.0,
    )


class TestBatchModeMCP:
    """End-to-end: run_batch_with_display(connect_mcp=True) wires MCP tools."""

    async def test_mcp_tools_registered_and_disconnect_called(
        self, stdio_config_for_batch, monkeypatch
    ):
        """Verify the full batch+MCP lifecycle:

        1. Agent.set_mcp_servers is called with config
        2. After connect, the MCP tool appears in agent.get_all_tools()
        3. disconnect_mcp is invoked on exit (finally block)
        """
        from agent13.cli import run_batch_with_display
        from agent13.config import Config
        from agent13.core import Agent

        # Build a fresh stateful FastMCP server.
        fastmcp, _state = _make_stateful_server()

        # Patch MCPManager._open_transport_cm on the class so any manager
        # created during the batch run uses in-memory transport. We patch the
        # class method (not an instance) because the manager is constructed
        # inside run_batch_with_display.
        from contextlib import asynccontextmanager

        lowlevel = fastmcp._lowlevel_server

        @asynccontextmanager
        async def fake_open_transport_cm(self, config, stderr_capture):
            from mcp.shared.memory import create_client_server_memory_streams

            async with create_client_server_memory_streams() as (
                client_streams,
                server_streams,
            ):
                client_read, client_write = client_streams
                server_read, server_write = server_streams

                async def run_server():
                    await lowlevel.run(
                        server_read,
                        server_write,
                        lowlevel.create_initialization_options(),
                        raise_exceptions=True,
                    )

                task = asyncio.create_task(run_server())
                try:
                    yield client_read, client_write, None
                finally:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

        monkeypatch.setattr(MCPManager, "_open_transport_cm", fake_open_transport_cm)

        # Minimal stub config: only mcp_servers matters for this path.
        cfg = Config(mcp_servers=[stdio_config_for_batch])
        from agent13 import cli as cli_mod

        monkeypatch.setattr(
            cli_mod, "get_config", lambda: cfg, raising=True
        )
        # run_batch_with_display reads config via a local _get_config alias.
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: cfg, raising=True
        )

        # Stub run_batch so we can inspect the agent BEFORE it actually runs,
        # and so we don't need a mocked LLM for this test. We assert inside
        # the stub where agent + mcp are both alive.
        captured: dict[str, Any] = {}

        async def fake_run_batch(agent, prompt, **kwargs):
            captured["agent"] = agent
            # MCP should be connected now — tool list should include the
            # stateful server's tools.
            tools = await agent.get_all_tools()
            tool_names = {
                t.get("function", {}).get("name", "") for t in tools
            }
            captured["tool_names"] = tool_names
            captured["mcp_is_not_none"] = agent.mcp is not None

        monkeypatch.setattr(cli_mod, "run_batch", fake_run_batch)

        # Also stub out the Agent so we don't need a real client. We use a
        # thin subclass that records disconnect_mcp calls.
        disconnect_calls = {"count": 0}

        class RecordingAgent(Agent):
            async def disconnect_mcp(self) -> bool:
                disconnect_calls["count"] += 1
                return await super().disconnect_mcp()

        monkeypatch.setattr(cli_mod, "Agent", RecordingAgent)

        # Execute. pretty=False keeps it simple (no Rich display).
        await run_batch_with_display(
            client=None,
            model="test-model",
            prompt="ignored by stub",
            pretty=False,
            connect_mcp=True,
        )

        # Assertions
        assert "agent" in captured, "fake_run_batch was never called"
        assert captured["mcp_is_not_none"] is True, (
            "MCP manager was not initialized during batch run"
        )
        # The stateful server registers: put, get, count
        assert "mcp://batchsrv/put" in captured["tool_names"], (
            f"MCP tools not registered; got: {captured['tool_names']}"
        )
        assert "mcp://batchsrv/get" in captured["tool_names"]
        assert disconnect_calls["count"] == 1, (
            f"disconnect_mcp should be called exactly once on exit; "
            f"got {disconnect_calls['count']}"
        )

    async def test_connect_mcp_false_skips_mcp(self, monkeypatch):
        """When connect_mcp=False (default), no MCP manager is created."""
        from agent13.cli import run_batch_with_display
        from agent13.config import Config
        from agent13 import cli as cli_mod

        # Config WITH mcp_servers — but connect_mcp=False should ignore them.
        cfg = Config(mcp_servers=[
            MCPServerConfig(
                name="should_not_connect",
                transport="stdio",
                command="echo",
                args=["ignored"],
                connect_timeout=10.0,
                tool_timeout=10.0,
            )
        ])
        monkeypatch.setattr("agent13.config.get_config", lambda: cfg)

        captured: dict[str, Any] = {}

        async def fake_run_batch(agent, prompt, **kwargs):
            captured["mcp_is_none"] = agent.mcp is None
            captured["mcp_server_configs_empty"] = (
                len(agent._mcp_server_configs) == 0
            )

        monkeypatch.setattr(cli_mod, "run_batch", fake_run_batch)

        await run_batch_with_display(
            client=None,
            model="test-model",
            prompt="ignored",
            pretty=False,
            connect_mcp=False,
        )

        assert captured["mcp_is_none"] is True, (
            "connect_mcp=False should not initialize MCP"
        )
        assert captured["mcp_server_configs_empty"] is True, (
            "connect_mcp=False should not call set_mcp_servers"
        )
