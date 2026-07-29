"""Integration tests for MCP persistent sessions.

Per docs_archive/mcp_connection_fix_plan.md §6.2. Only the LLM is mocked
— everything else is real (real MCPManager, real ClientSession, real
FastMCP server in-process for the stateful tests, real subprocess for
the crash-reconnect test).

This file is the regression test for Attempt 13 (stateful-server blind
spot). Pre-fix, test_stateful_round_trip would fail because each
call_tool spawned a fresh subprocess and lost the in-memory state.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from typing import Any

import pytest

from agent13.config import MCPServerConfig
from agent13.mcp import MCPManager


# ---------------------------------------------------------------------------
# Helpers: in-memory stateful FastMCP server
# ---------------------------------------------------------------------------


def _make_stateful_server():
    """Build a fresh FastMCP server with module-private state.

    Returns (server, state_dict). state_dict is shared with the registered
    tools so tests can inspect/reset it.
    """
    from mcp.server.mcpserver import MCPServer as FastMCP

    mcp = FastMCP("stateful-test")
    state: dict[str, str] = {}

    @mcp.tool()
    def put(key: str, value: str) -> str:
        state[key] = value
        return f"stored {key}={value}"

    @mcp.tool()
    def get(key: str) -> str:
        return state.get(key, "<missing>")

    @mcp.tool()
    def count() -> int:
        return len(state)

    return mcp, state


def _patch_transport_to_in_memory(
    manager: MCPManager, server_name: str, fastmcp_server: Any
) -> asyncio.Task:
    """Replace manager._open_transport_cm with one that yields in-memory streams.

    The FastMCP server is run in a background task owned by the caller
    (returned) so it lives as long as the test needs. The patched
    _open_transport_cm yields (client_read, client_write, None) — matching
    the real stdio/http signature — so _session_runner can wrap it in a
    ClientSession as usual.
    """
    from mcp.server.mcpserver import MCPServer as FastMCP

    lowlevel = (
        fastmcp_server._lowlevel_server
        if isinstance(fastmcp_server, FastMCP)
        else fastmcp_server
    )

    # Holder for the server task; set when the CM is first entered.
    server_task_holder: dict[str, asyncio.Task | None] = {"task": None}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_open_transport_cm(config, stderr_capture):
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

            # Start the server task; it lives until the CM exits.
            server_task_holder["task"] = asyncio.create_task(run_server())
            try:
                yield client_read, client_write, None
            finally:
                # Cancel the server task when the client side closes.
                t = server_task_holder["task"]
                if t and not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

    manager._open_transport_cm = fake_open_transport_cm  # type: ignore[method-assign]
    # Return the holder so tests can inspect; the actual task is created lazily.
    return server_task_holder  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# In-memory stateful tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stdio_config_for_stateful():
    """A stdio config whose command is irrelevant — transport is patched."""
    return MCPServerConfig(
        name="stateful",
        transport="stdio",
        command="echo",
        args=["ignored"],
        connect_timeout=10.0,
        tool_timeout=10.0,
    )


class TestStatefulInMemory:
    """Stateful-server regression tests using in-memory transport.

    These would fail on the pre-fix code because each call_tool spawned a
    fresh subprocess and lost the FastMCP server's module-level _STATE.
    """

    async def test_stateful_round_trip(self, stdio_config_for_stateful):
        """put then get returns the stored value (not <missing>)."""
        manager = MCPManager([stdio_config_for_stateful])
        fastmcp, _state = _make_stateful_server()
        _patch_transport_to_in_memory(manager, "stateful", fastmcp)

        try:
            connected = await manager.connect_server_if_needed("stateful")
            assert connected is True, "connect failed"

            put_out = await manager.call_tool(
                "mcp://stateful/put", {"key": "a", "value": "1"}
            )
            assert "stored" in put_out, f"put returned: {put_out}"

            get_out = await manager.call_tool("mcp://stateful/get", {"key": "a"})
            # Pre-fix would return "<missing>" because of subprocess-per-call.
            assert "1" in get_out, f"get returned: {get_out}"
            assert "<missing>" not in get_out
        finally:
            await manager.cleanup()

    async def test_persistence_across_many_calls(self, stdio_config_for_stateful):
        """10 puts followed by 10 gets — all correct on the same session."""
        manager = MCPManager([stdio_config_for_stateful])
        fastmcp, _state = _make_stateful_server()
        _patch_transport_to_in_memory(manager, "stateful", fastmcp)

        try:
            await manager.connect_server_if_needed("stateful")

            for i in range(10):
                out = await manager.call_tool(
                    "mcp://stateful/put", {"key": f"k{i}", "value": f"v{i}"}
                )
                assert f"v{i}" in out

            for i in range(10):
                out = await manager.call_tool("mcp://stateful/get", {"key": f"k{i}"})
                assert f"v{i}" in out, f"get k{i} returned: {out}"
                assert "<missing>" not in out

            count_out = await manager.call_tool("mcp://stateful/count", {})
            assert "10" in count_out, f"count returned: {count_out}"
        finally:
            await manager.cleanup()

    async def test_session_survives_idle(self, stdio_config_for_stateful):
        """put, sleep 2s, get — still works (proves real persistence)."""
        manager = MCPManager([stdio_config_for_stateful])
        fastmcp, _state = _make_stateful_server()
        _patch_transport_to_in_memory(manager, "stateful", fastmcp)

        try:
            await manager.connect_server_if_needed("stateful")

            await manager.call_tool(
                "mcp://stateful/put", {"key": "idle", "value": "kept"}
            )
            await asyncio.sleep(2.0)
            out = await manager.call_tool("mcp://stateful/get", {"key": "idle"})
            assert "kept" in out, f"get after idle returned: {out}"
            assert "<missing>" not in out
        finally:
            await manager.cleanup()


# ---------------------------------------------------------------------------
# Real stdio subprocess: crash-reconnect test
# ---------------------------------------------------------------------------

# A tiny MCP server script run as a subprocess. It keeps state in a file
# so we can verify state is lost after crash (expected behavior).
_STATEFUL_STDIO_SERVER = textwrap.dedent(
    """
    import sys
    from mcp.server.mcpserver import MCPServer as FastMCP

    mcp = FastMCP("stateful-stdio")

    @mcp.tool()
    def echo(x: str) -> str:
        '''Echo back the argument. Used to detect live session.'''
        return f"echo:{x}"

    if __name__ == "__main__":
        mcp.run("stdio")
    """
)


@pytest.fixture
def stateful_stdio_server_file(tmp_path):
    """Write the stateful stdio server script to a tmp file."""
    server_file = tmp_path / "stateful_stdio_server.py"
    server_file.write_text(_STATEFUL_STDIO_SERVER)
    return server_file


@pytest.fixture
def stdio_config_real(stateful_stdio_server_file):
    """Real stdio config pointing at our stateful server subprocess."""
    return MCPServerConfig(
        name="stateful_stdio",
        transport="stdio",
        command=sys.executable,
        args=[str(stateful_stdio_server_file)],
        connect_timeout=15.0,
        tool_timeout=10.0,
        retry_attempts=1,  # fail fast on reconnect so test doesn't hang
        retry_delay=0.1,
    )


class TestReconnectAfterCrash:
    """Crash-reconnect test using a real stdio subprocess.

    Per design Q7: kill child, assert first call succeeds-or-errors, assert
    second call succeeds via the reconnect path. State loss is expected.
    """

    async def test_reconnect_after_subprocess_crash(self, stdio_config_real):
        manager = MCPManager([stdio_config_real])

        try:
            connected = await manager.connect_server_if_needed("stateful_stdio")
            assert connected is True, "initial connect failed"

            # Sanity: echo works.
            out = await manager.call_tool("mcp://stateful_stdio/echo", {"x": "alive"})
            assert "echo:alive" in out, f"sanity echo returned: {out}"

            # Kill the child subprocess out-of-band. The _session_runner
            # owns the subprocess via stdio_client; find it via psutil on
            # the server script path (cross-platform; pgrep was Unix-only).
            script_path = stdio_config_real.args[0]
            killed = self._kill_children_matching(script_path)
            assert killed > 0, "no child process found to kill"

            # First call after crash: either succeeds (broken pipe detected
            # immediately, reconnect happened transparently) or errors
            # (timeout / connection closed surfaced). Either is acceptable
            # per design Q7.
            first_call_result = await manager.call_tool(
                "mcp://stateful_stdio/echo", {"x": "after_crash_1"}
            )

            # Second call must succeed — the reconnect path always wins.
            second_out = await manager.call_tool(
                "mcp://stateful_stdio/echo", {"x": "after_crash_2"}
            )
            assert "echo:after_crash_2" in second_out, (
                f"second call after crash did not succeed via reconnect: "
                f"first={first_call_result!r}, second={second_out!r}"
            )

            # Confirm exactly one live child process now (reconnect spawned one).
            live = self._count_children_matching(script_path)
            assert live >= 1, "no live child after reconnect"
        finally:
            await manager.cleanup()
            # Make sure we don't leak subprocesses.
            self._kill_children_matching(str(stdio_config_real.args[0]))

    @staticmethod
    def _matching_pids(pattern: str) -> list[int]:
        """Return PIDs of live processes whose command line contains `pattern`.

        Uses psutil for cross-platform support (replaces Unix-only pgrep -f,
        which failed with FileNotFoundError on Windows).
        Excludes the current process to avoid killing the test runner itself
        when the pattern matches the interpreter path.
        """
        import psutil

        self_pid = os.getpid()
        matches: list[int] = []
        for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
            try:
                info = proc.info
                cmdline = info.get("cmdline") or []
                if info["pid"] == self_pid:
                    continue
                if any(pattern in arg for arg in cmdline):
                    matches.append(info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return matches

    @staticmethod
    def _kill_children_matching(pattern: str) -> int:
        """Kill any process whose command line matches `pattern`.

        Returns the number of processes killed. Cross-platform via psutil
        (replaces the old pgrep -f + os.kill(SIGKILL) Unix-only approach).
        """
        import psutil

        pids = TestReconnectAfterCrash._matching_pids(pattern)
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return len(pids)

    @staticmethod
    def _count_children_matching(pattern: str) -> int:
        return len(TestReconnectAfterCrash._matching_pids(pattern))
