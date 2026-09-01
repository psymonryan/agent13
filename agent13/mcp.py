"""MCP (Model Context Protocol) client manager.

Manages long-lived persistent connections to MCP servers. Each connected
server owns a background asyncio.Task (_session_runner) that holds the
transport context manager and ClientSession open for the lifetime of the
connection; tool calls reuse the live session. If a session dies
mid-call, call_tool attempts one transparent reconnect via
_reconnect_server before surfacing the error.

See docs_archive/mcp_connection_fix_plan.md for the design and
docs_archive/mcp_journey.md § Attempt 13 for the history.
"""

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional, Literal, Callable, Awaitable
from dataclasses import dataclass, field

from agent13.config import MCPServerConfig
from agent13.debug_log import log_event, log_error
from agent13.events import AgentEvent, AgentEventData


class StderrCapture:
    """Capture stderr from MCP subprocess using a pipe and emit as events.

    The MCP SDK's stdio_client passes errlog directly to subprocess.Popen(stderr=...),
    which requires a real file descriptor. We create a pipe, pass the write end
    to the subprocess, and read from the read end to emit lines as events.
    """

    def __init__(self, server_name: str, emit_callback: Callable[[str], None]):
        self.server_name = server_name
        self.emit_callback = emit_callback
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = ""

    def fileno(self) -> int:
        """Return write end of pipe for subprocess.stderr."""
        return self._write_fd

    def start(self) -> None:
        """Start the reader thread."""
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Read from pipe and emit lines."""
        os.set_blocking(self._read_fd, False)
        while self._running:
            try:
                data = os.read(self._read_fd, 4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                self._buffer += text

                # Process complete lines
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if line.strip():
                        self.emit_callback(line)
            except BlockingIOError:
                # No data available, wait a bit
                import time

                time.sleep(0.01)
            except OSError:
                break

    def stop(self) -> None:
        """Stop the reader and clean up."""
        self._running = False
        # Flush any remaining buffer
        if self._buffer.strip():
            self.emit_callback(self._buffer.rstrip("\r"))
        # Close write end first to unblock reader
        try:
            os.close(self._write_fd)
        except OSError:
            pass
        # Wait for reader to finish
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        # Close read end
        try:
            os.close(self._read_fd)
        except OSError:
            pass


# Import MCP SDK - will fail gracefully if not installed
try:
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import (
        streamable_http_client as streamablehttp_client,
    )
    from mcp.shared.exceptions import MCPError as McpError
    from mcp.types import CONNECTION_CLOSED

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    anyio = None
    ClientSession = None
    McpError = None
    CONNECTION_CLOSED = None
    stdio_client = None
    StdioServerParameters = None
    streamablehttp_client = None


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""

    server_name: str
    name: str  # Full URI: mcp://server_name/tool_name
    original_name: str  # Original tool name from server
    description: str
    input_schema: dict


@dataclass
class ServerInfo:
    """Tracks a connected MCP server and its live session.

    Persistent-session fields (session, stderr_capture, session_task,
    _stop_event, _ready_event) are None/disconnected until a background
    session task owns the transport. They are private implementation
    details and default to None so existing fixtures that construct
    ServerInfo with only config=/tools=/status= keep working.
    """

    config: MCPServerConfig
    tools: list = field(default_factory=list)
    status: Literal["connected", "disconnected", "error"] = "disconnected"
    last_error: Optional[str] = None

    # --- Persistent-session state (private; None when disconnected) ---
    session: Optional["ClientSession"] = None
    stderr_capture: Optional[StderrCapture] = None
    session_task: Optional[asyncio.Task] = None
    _stop_event: Optional[asyncio.Event] = None
    _ready_event: Optional[asyncio.Event] = None


# Maximum retry backoff (10 minutes)
MAX_RETRY_DELAY = 600.0


class MCPManager:
    """Manages MCP server connections with persistent sessions.

    Each connected server is backed by a long-lived _session_runner
    asyncio.Task that owns the transport (stdio_client /
    streamablehttp_client) and ClientSession context managers. The live
    session is stored on ServerInfo.session and reused by every call_tool
    invocation. If the session dies mid-call (subprocess crash, network
    drop), call_tool attempts ONE transparent reconnect before erroring.

    Usage:
        manager = MCPManager(server_configs)
        manager.set_event_callback(my_callback)

        # Connect and discover tools (launches _session_runner tasks)
        await manager.connect_all()

        # Get tools in OpenAI format
        tools = manager.get_openai_tools()

        # Execute a tool (reuses the live session)
        result = await manager.call_tool("mcp://hvac_server/set_temperature", {"temp": 72})

        # Cleanup on shutdown (cancels runner tasks, stops subprocesses)
        await manager.cleanup()
    """

    def __init__(self, server_configs: list[MCPServerConfig]):
        self.server_configs = {c.name: c for c in server_configs}
        self.servers: dict[str, ServerInfo] = {}
        self.tools: list[MCPTool] = []
        self._semaphore = asyncio.Semaphore(5)
        self._event_callback: Optional[
            Callable[[AgentEvent, AgentEventData], Awaitable[None]]
        ] = None
        self._shutting_down = False

    def set_event_callback(
        self, callback: Callable[[AgentEvent, AgentEventData], Awaitable[None]]
    ) -> None:
        """Set callback for user notifications."""
        self._event_callback = callback

    async def _emit_event(self, event: AgentEvent, data: dict) -> None:
        """Emit event to callback if set."""
        if self._event_callback:
            await self._event_callback(event, AgentEventData(event=event, data=data))

    def _emit_stderr_sync(
        self, loop: asyncio.AbstractEventLoop, server_name: str, line: str
    ) -> None:
        """Emit stderr line as event (thread-safe, called from StderrCapture).

        This is called from the subprocess reader thread, so we use
        asyncio.run_coroutine_threadsafe to schedule the event emission
        on the main event loop.
        """
        log_event("mcp_stderr", {"server": server_name, "line": line})
        # Schedule the coroutine on the provided loop
        if loop and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._emit_event(
                    AgentEvent.MCP_SERVER_STDERR,
                    {"server_name": server_name, "line": line},
                ),
                loop,
            )

    async def connect_server_if_needed(self, server_name: str) -> bool:
        """Connect to a server and discover its tools.

        Returns:
            True if connected (or already was), False on failure
        """
        if not MCP_AVAILABLE:
            await self._emit_event(
                AgentEvent.MCP_SERVER_ERROR,
                {
                    "server_name": server_name,
                    "error": "MCP SDK not installed. Install with: pip install mcp",
                },
            )
            return False

        if (
            server_name in self.servers
            and self.servers[server_name].status == "connected"
        ):
            return True

        if server_name not in self.server_configs:
            log_error(
                Exception(f"Unknown MCP server: {server_name}"), {"context": "mcp"}
            )
            return False

        config = self.server_configs[server_name]

        # Notify user that server is starting
        await self._emit_event(
            AgentEvent.MCP_SERVER_STARTED,
            {"server_name": server_name, "transport": config.transport},
        )

        success = await self._connect_with_retry(config)

        # --- MCP SDK 2.0 fallback ---
        # uvx resolves the latest MCP SDK (2.0+) which removed
        # mcp.server.fastmcp. Many servers still import it and crash on
        # startup. If the connection failed and this is a uvx server
        # without an explicit mcp pin, retry once with --with "mcp<2".
        if not success and self._is_uvx_needing_mcp_fallback(config):
            patched = self._patched_config_mcp1x(config)
            log_event(
                "mcp_sdk2_fallback",
                {"server": server_name, "original_args": config.args},
            )
            # _connect_once looks up config by name from self.server_configs,
            # so we must swap in the patched config for the retry.
            original_config = self.server_configs[server_name]
            self.server_configs[server_name] = patched
            success = await self._connect_with_retry(patched)

            if success:
                # Persist the patched args on the original config so
                # future connects don't need the fallback.
                original_config.args = patched.args
                self.server_configs[server_name] = original_config
                await self._emit_event(
                    AgentEvent.MCP_SERVER_WARNING,
                    self._mcp_fallback_warning(server_name, original_config),
                )
                await self._emit_event(
                    AgentEvent.MCP_SERVER_READY,
                    {
                        "server_name": server_name,
                        "tool_count": len(self.servers[server_name].tools),
                    },
                )
                return True
            else:
                # Restore original config on failure.
                self.server_configs[server_name] = original_config

        if success:
            await self._emit_event(
                AgentEvent.MCP_SERVER_READY,
                {
                    "server_name": server_name,
                    "tool_count": len(self.servers[server_name].tools),
                },
            )
        else:
            await self._emit_event(
                AgentEvent.MCP_SERVER_ERROR,
                {
                    "server_name": server_name,
                    "error": f"Connection failed after {config.retry_attempts} attempts",
                },
            )

        return success

    # ------------------------------------------------------------------
    # MCP SDK 2.0 fallback helpers
    # ------------------------------------------------------------------

    _MCP_PIN_TOKENS = ("--with", "-w")
    _MCP_PIN_PREFIX = "mcp"

    @staticmethod
    def _is_uvx_needing_mcp_fallback(config: MCPServerConfig) -> bool:
        """True if this is a uvx stdio server with no explicit mcp pin.

        We only attempt the fallback for uvx-launched servers (the common
        case for third-party MCP tools). If the user already has
        ``--with mcp...`` (or ``-w mcp...``) in their args, they've
        taken control of the MCP version and we respect that.
        """
        if config.transport != "stdio" or not config.command:
            return False
        if config.command != "uvx" and not config.command.endswith("/uvx"):
            return False
        args = config.args
        for i, tok in enumerate(args):
            if tok in MCPManager._MCP_PIN_TOKENS:
                # Check if the next arg pins mcp
                if i + 1 < len(args) and args[i + 1].startswith(
                    MCPManager._MCP_PIN_PREFIX
                ):
                    return False
        return True

    @staticmethod
    def _patched_config_mcp1x(config: MCPServerConfig) -> MCPServerConfig:
        """Return a shallow copy of config with --with mcp<2 injected."""
        import copy

        patched = copy.copy(config)
        patched.args = ["--with", "mcp<2"] + list(config.args)
        return patched

    @staticmethod
    def _mcp_fallback_warning(
        server_name: str, config: MCPServerConfig
    ) -> dict:
        """Build the spoon-feed warning payload for the SDK 2.0 fallback.

        ``config.args`` must already contain the injected ``--with mcp<2``
        (the caller persists the patched args before calling this).
        """
        args_repr = ", ".join(f'"{a}"' for a in config.args)
        return {
            "server_name": server_name,
            "warning": (
                f"MCP server '{server_name}' failed to start because it uses "
                f"an older MCP SDK API (mcp.server.fastmcp) that was removed "
                f"in MCP SDK 2.0. It was auto-fixed by pinning mcp<2.\n\n"
                f"To make this permanent, edit your config (~/.agent13/config.toml) "
                f"and change the args line to:\n"
                f'  args = [{args_repr}]\n\n'
                f"Once the server releases an update compatible with MCP SDK 2.0, "
                f"remove the --with mcp<2 to use the latest version."
            ),
        }

    async def _connect_once(self, server_name: str) -> bool:
        """Single connect attempt. No backoff. Returns True on success.

        Launches the _session_runner background task and waits (up to
        connect_timeout) for it to signal _ready_event. On success the
        server is connected with tools registered. On failure or timeout,
        status is set to "error" / "disconnected" and the task is torn
        down. Idempotent for already-connected servers.

        Used by both _connect_with_retry (loops with backoff) and
        _reconnect_server (single attempt after a mid-call failure).
        """
        if (
            server_name in self.servers
            and self.servers[server_name].status == "connected"
        ):
            return True

        if server_name not in self.server_configs:
            log_error(
                Exception(f"Unknown MCP server: {server_name}"), {"context": "mcp"}
            )
            return False

        config = self.server_configs[server_name]

        # If reconnecting after a failure, tear down any prior runner first
        # so we don't leak a parked task / subprocess.
        if server_name in self.servers:
            await self._teardown_server(server_name)

        server = ServerInfo(config=config)
        server._stop_event = asyncio.Event()
        server._ready_event = asyncio.Event()
        if config.transport == "stdio":
            loop = asyncio.get_running_loop()
            server.stderr_capture = StderrCapture(
                server_name=config.name,
                emit_callback=lambda line: self._emit_stderr_sync(
                    loop, config.name, line
                ),
            )
            server.stderr_capture.start()
        self.servers[server_name] = server

        server.session_task = asyncio.create_task(
            self._session_runner(server_name),
            name=f"mcp_session:{server_name}",
        )

        try:
            await asyncio.wait_for(
                server._ready_event.wait(), timeout=config.connect_timeout
            )
        except asyncio.TimeoutError:
            server.last_error = f"connect timeout after {config.connect_timeout}s"
            log_error(
                TimeoutError(server.last_error),
                {"context": "mcp_connect", "server": server_name},
            )
            await self._teardown_server(server_name)
            return False

        return server.status == "connected"

    async def _connect_with_retry(self, config: MCPServerConfig) -> bool:
        """Connect with exponential backoff retry.

        Each attempt delegates to _connect_once (single, no-backoff
        attempt). _connect_once tears down any prior runner before
        starting a fresh one, so retries don't accumulate parked tasks.
        """
        delay = config.retry_delay

        for attempt in range(config.retry_attempts):
            try:
                if await self._connect_once(config.name):
                    return True
                # _connect_once already set status/last_error on the server.
                err = self.servers[config.name].last_error or "connect failed"
                raise RuntimeError(err)
            except Exception as e:
                if attempt < config.retry_attempts - 1:
                    log_event(
                        "mcp_connect_retry",
                        {
                            "server": config.name,
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, MAX_RETRY_DELAY)
                else:
                    log_error(e, {"context": "mcp_connect", "server": config.name})
                    if config.name in self.servers:
                        self.servers[config.name].status = "error"
                        self.servers[config.name].last_error = str(e)
                    return False

        return False

    @asynccontextmanager
    async def _open_transport_cm(
        self,
        config: MCPServerConfig,
        stderr_capture: Optional[StderrCapture],
    ):
        """Open the transport context manager for the configured transport.

        Yields (read_stream, write_stream, stderr_capture). The caller owns
        the async-with block — closing happens when the caller exits. This
        helper exists so both the connect-time test (which closes before
        returning) and the persistent session runner (which keeps it open)
        can share the same transport-opening code.

        For stdio, stderr_capture must be provided (already started); the
        caller is responsible for stopping it. For http, stderr_capture is
        ignored (pass None).
        """
        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
            )
            async with stdio_client(server_params, errlog=stderr_capture) as (
                read_stream,
                write_stream,
            ):
                yield read_stream, write_stream, stderr_capture
        else:
            async with streamablehttp_client(config.url) as result:
                # mcp SDK 1.x yields (read, write, get_session_id_callback);
                # mcp SDK 2.x yields (read, write). Take the first two either way.
                read_stream, write_stream = result[0], result[1]
                yield read_stream, write_stream, None

    async def _session_runner(self, server_name: str) -> None:
        """Own the async-with stack for a persistent session.

        Entered once at connect time; exits only on disconnect (via
        _stop_event) or if the transport itself dies. The live session is
        handed to ServerInfo.session and tools are registered before
        signalling _ready_event. On any failure, status is set to
        "error" and last_error populated; _ready_event is always set so
        the connect path doesn't hang.
        """
        server = self.servers.get(server_name)
        if server is None:
            return
        config = server.config
        try:
            async with self._open_transport_cm(config, server.stderr_capture) as (
                read_stream,
                write_stream,
                _,
            ):
                timeout = (
                    float(config.connect_timeout)
                    if config.connect_timeout
                    else None
                )
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=timeout
                ) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    server.session = session
                    server.tools = tools_result.tools
                    self._register_tools(server_name, server.tools)
                    server.status = "connected"
                    server.last_error = None
                    log_event(
                        "mcp_connected",
                        {
                            "server": server_name,
                            "transport": config.transport,
                            "tool_count": len(server.tools),
                        },
                    )
                    # Signal readiness before parking, so the connect path
                    # observes the connected status.
                    if server._ready_event:
                        server._ready_event.set()
                    # Park until disconnect flips _stop_event.
                    if server._stop_event:
                        await server._stop_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # On Windows (Proactor event loop), closing subprocess stdio
            # transports can raise ValueError("I/O operation on closed
            # pipe") during shutdown — the transport closes pipes while
            # pending async I/O is still in flight.  When _stop_event is
            # set we know this is an intentional disconnect, so suppress
            # the noise rather than alarming the user.
            if (
                isinstance(e, ValueError)
                and "closed pipe" in str(e).lower()
                and server._stop_event
                and server._stop_event.is_set()
            ):
                pass  # expected on Windows during shutdown
            else:
                server.status = "error"
                server.last_error = str(e)
                log_error(e, {"context": "mcp_session_runner", "server": server_name})
        finally:
            server.session = None
            # Always release the connect path even on error.
            if server._ready_event:
                server._ready_event.set()

    async def _teardown_server(self, server_name: str) -> None:
        """Tear down a live session: stop the runner task, stop stderr.

        Safe to call on a server that's already disconnected or absent.
        Leaves the ServerInfo record in place with status="disconnected";
        callers that want to remove it entirely do so after.
        """
        server = self.servers.get(server_name)
        if server is None:
            return
        if server._stop_event:
            server._stop_event.set()
        if server.session_task and not server.session_task.done():
            # Graceful first: let the _session_runner unwind its async-with
            # stack (closing ClientSession and transport) naturally.  This
            # avoids the Windows Proactor "I/O operation on closed pipe"
            # error that occurs when cancel() interrupts mid-close.
            try:
                await asyncio.wait_for(server.session_task, timeout=3.0)
            except asyncio.TimeoutError:
                # Graceful shutdown didn't finish in time — force cancel.
                server.session_task.cancel()
                try:
                    await asyncio.wait_for(server.session_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            except asyncio.CancelledError:
                pass
        if server.stderr_capture:
            server.stderr_capture.stop()
        server.session = None
        server.status = "disconnected"

    def _is_reconnectable(self, exc: BaseException) -> bool:
        """Return True if exc indicates the session died and a reconnect is worth trying.

        Trigger set (per plan §4.4, narrowed per design Q5):
          - anyio.BrokenResourceError  (write side broken before read loop notices)
          - anyio.ClosedResourceError  (transport closed under us)
          - McpError with error.code == CONNECTION_CLOSED  (primary signal:
            the SDK sends a JSONRPCError(code=-32000) to pending requests
            when the read loop detects the closed stream)

        asyncio.CancelledError is intentionally NOT here — it must
        propagate (project interrupt semantics, see mcp_journey.md §8).
        """
        if anyio is None:
            return False
        if isinstance(exc, anyio.BrokenResourceError):
            return True
        if isinstance(exc, anyio.ClosedResourceError):
            return True
        if McpError is not None and isinstance(exc, McpError):
            code = getattr(getattr(exc, "error", None), "code", None)
            if code == CONNECTION_CLOSED:
                return True
        return False

    def _register_tools(self, server_name: str, tools: list) -> None:
        """Register tools with collision prevention."""
        from agent13.tools import name_matches

        config = self.server_configs[server_name]
        enabled_patterns = config.enabled_tools
        disabled_patterns = config.disabled_tools

        for tool in tools:
            # Skip if matching disabled patterns
            if name_matches(tool.name, disabled_patterns):
                continue
            # Skip if whitelist exists and tool not in it
            if enabled_patterns and not name_matches(tool.name, enabled_patterns):
                continue

            # Use URI-style naming to prevent collisions
            full_name = f"mcp://{server_name}/{tool.name}"

            # Check for collisions with existing tools
            if any(t.name == full_name for t in self.tools):
                log_event("mcp_duplicate_tool", {"tool": full_name})
                continue

            # Get input schema - handle both inputSchema and input_schema
            schema = getattr(tool, "inputSchema", None) or getattr(
                tool, "input_schema", {}
            )

            self.tools.append(
                MCPTool(
                    server_name=server_name,
                    name=full_name,
                    original_name=tool.name,
                    description=tool.description or f"Tool from {server_name}",
                    input_schema=schema,
                )
            )

    def get_openai_tools(self) -> list[dict]:
        """Convert MCP tools to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool on the persistent session.

        Reuses the live session owned by _session_runner. If the session
        died mid-call (subprocess crash, network drop), attempts ONE
        transparent reconnect via _reconnect_server before surfacing the
        error. The reconnect trigger set is defined by _is_reconnectable
        and intentionally excludes asyncio.CancelledError (interrupt
        semantics must propagate).

        Args:
            tool_name: Full tool URI (e.g., "mcp://hvac_server/set_temperature")
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if self._shutting_down:
            return json.dumps({"error": "MCPManager is shutting down"})

        # Parse URI format: mcp://server_name/tool_name
        if not tool_name.startswith("mcp://"):
            return json.dumps({"error": f"Invalid MCP tool name: {tool_name}"})
        parts = tool_name[6:].split("/", 1)
        if len(parts) != 2:
            return json.dumps({"error": f"Invalid MCP tool name: {tool_name}"})
        server_name, actual_tool_name = parts

        # Check server is connected
        if server_name not in self.servers:
            return json.dumps({"error": f"MCP server '{server_name}' not configured"})

        server = self.servers[server_name]
        if server.status != "connected" or server.session is None:
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})

        config = server.config
        timeout = config.tool_timeout

        log_event(
            "mcp_tool_call_start",
            {"server": server_name, "tool": actual_tool_name, "arguments": arguments},
        )

        start_time = asyncio.get_event_loop().time()

        try:
            try:
                result = await self._invoke_with_session(
                    server, actual_tool_name, arguments, timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._is_reconnectable(e):
                    raise
                # Session died mid-call — try ONE transparent reconnect.
                log_event(
                    "mcp_session_reconnecting",
                    {"server": server_name, "reason": str(e)},
                )
                if await self._reconnect_server(server_name):
                    # _reconnect_server replaced server.session; re-fetch.
                    server = self.servers[server_name]
                    result = await self._invoke_with_session(
                        server, actual_tool_name, arguments, timeout
                    )
                else:
                    raise

            duration = asyncio.get_event_loop().time() - start_time
            log_event(
                "mcp_tool_call_end",
                {
                    "server": server_name,
                    "tool": actual_tool_name,
                    "duration_ms": int(duration * 1000),
                    "success": True,
                },
            )

            return self._format_result(result)

        except asyncio.TimeoutError:
            duration = asyncio.get_event_loop().time() - start_time
            error_msg = f"MCP tool '{tool_name}' timed out after {timeout}s"
            log_error(
                TimeoutError(error_msg),
                {
                    "context": "mcp_tool_call",
                    "server": server_name,
                    "tool": actual_tool_name,
                    "duration_ms": int(duration * 1000),
                },
            )
            return json.dumps({"error": error_msg})

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            error_msg = f"MCP tool '{tool_name}' failed: {e}"
            log_error(
                e,
                {
                    "context": "mcp_tool_call",
                    "server": server_name,
                    "tool": actual_tool_name,
                    "duration_ms": int(duration * 1000),
                },
            )
            return json.dumps({"error": error_msg})

    async def _invoke_with_session(
        self,
        server: ServerInfo,
        tool_name: str,
        arguments: dict,
        timeout: float,
    ):
        """Call a tool on server.session under semaphore + tool timeout.

        No transport management here — the session is already alive (owned
        by _session_runner). Raises whatever the SDK raises; call_tool
        decides whether to reconnect.
        """
        async with asyncio.timeout(timeout):
            async with self._semaphore:
                return await server.session.call_tool(
                    tool_name, arguments, read_timeout_seconds=timeout
                )

    async def _reconnect_server(self, server_name: str) -> bool:
        """Tear down the dead session and start a fresh one. Returns True on success.

        Single attempt via _connect_once — no backoff. The caller has
        already failed once; if this also fails, the caller's original
        exception propagates.
        """
        await self._teardown_server(server_name)
        return await self._connect_once(server_name)

    def _format_result(self, result) -> str:
        """Format MCP tool result as string."""
        # Check for content blocks
        if hasattr(result, "content"):
            content = result.content
            # Handle empty content list
            if content is None or (hasattr(content, "__len__") and len(content) == 0):
                return "[Empty result]"

            # Extract text from content blocks
            parts = []
            for block in content:
                block_type = getattr(block, "type", None)
                text = getattr(block, "text", None)

                # Handle text blocks - check if text is actually a string
                if isinstance(text, str):
                    parts.append(text)
                elif block_type == "image":
                    parts.append("[Image content received from MCP tool]")
                elif block_type == "resource":
                    uri = getattr(block, "uri", "unknown")
                    parts.append(f"[Resource: {uri}]")
                elif hasattr(block, "data"):
                    parts.append(str(block.data))

            if parts:
                return "\n".join(parts)
            else:
                return "[Empty result]"

        if hasattr(result, "structuredContent") and result.structuredContent:
            return json.dumps(result.structuredContent, indent=2)

        return str(result)

    def get_server_info(self) -> dict[str, list[str]]:
        """Get info about connected servers.

        Returns:
            Dict of server_name -> list of tool names
        """
        return {
            name: [t.name for t in self.tools if t.server_name == name]
            for name in self.servers
        }

    def is_connected(self) -> bool:
        """Check if any MCP server is currently connected.

        Returns:
            True if at least one server has status 'connected', False otherwise
        """
        return any(server.status == "connected" for server in self.servers.values())

    async def connect_all(self) -> dict[str, list[str]]:
        """Connect to all configured servers.

        Returns:
            Dict of server_name -> list of tool names for successfully connected servers
        """
        for config in self.server_configs.values():
            log_event(
                "mcp_connect_start",
                {"server": config.name, "transport": config.transport},
            )
            try:
                await self.connect_server_if_needed(config.name)
            except Exception as e:
                log_error(e, {"context": "mcp_connect_all", "server": config.name})

        return self.get_server_info()

    async def reload(self) -> dict[str, list[str]]:
        """Reload configuration and reconnect to all servers.

        Tears down live sessions first (cancels runner tasks, stops stderr
        captures, ends subprocesses) before reconnecting — prevents leaking
        parked tasks / child processes from the previous configuration.

        Delegates to ``connect_all`` so a single failing server doesn't
        abort the entire reload (per-server errors are logged, not raised).
        """
        await self.disconnect()
        return await self.connect_all()

    async def disconnect(self) -> dict[str, list[str]]:
        """Disconnect from all servers and clear tools.

        Actually tears down live sessions (cancels _session_runner tasks,
        stops StderrCapture threads, ends child subprocesses via the
        transport context manager exit). Safe to call on already-
        disconnected servers.
        """
        for name in list(self.servers.keys()):
            await self._teardown_server(name)
        self.servers = {}
        self.tools = []

        return self.get_server_info()

    async def cleanup(self, timeout: float = 5.0) -> None:
        """Disconnect from all servers gracefully and inhibit further calls.

        Sets _shutting_down so concurrent call_tool invocations return
        early, then tears down all live sessions.
        """
        self._shutting_down = True
        await self.disconnect()
