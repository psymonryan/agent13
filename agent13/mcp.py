"""MCP (Model Context Protocol) client manager.

Manages connections to MCP servers with lazy loading and user notifications.
Uses a reconnect-per-operation pattern for reliability.
"""

import asyncio
import json
import os
import threading
from datetime import timedelta
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
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.streamable_http import streamablehttp_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
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
    """Tracks server info (we reconnect for each operation)."""

    config: MCPServerConfig
    tools: list = field(default_factory=list)
    status: Literal["connected", "disconnected", "error"] = "disconnected"
    last_error: Optional[str] = None


# Maximum retry backoff (10 minutes)
MAX_RETRY_DELAY = 600.0


class MCPManager:
    """Manages MCP server connections.

    Uses a reconnect-per-operation pattern for reliability.
    Each tool call establishes a fresh connection to the server.

    Usage:
        manager = MCPManager(server_configs)
        manager.set_event_callback(my_callback)

        # Connect and discover tools
        await manager.connect_all()

        # Get tools in OpenAI format
        tools = manager.get_openai_tools()

        # Execute a tool (reconnects automatically)
        result = await manager.call_tool("mcp://hvac_server/set_temperature", {"temp": 72})

        # Cleanup on shutdown
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

    async def _connect_with_retry(self, config: MCPServerConfig) -> bool:
        """Connect with exponential backoff retry."""
        delay = config.retry_delay

        for attempt in range(config.retry_attempts):
            try:
                if config.transport == "stdio":
                    tools = await self._test_and_list_tools_stdio(config)
                else:
                    tools = await self._test_and_list_tools_http(config)

                self.servers[config.name] = ServerInfo(
                    config=config, tools=tools, status="connected"
                )
                self._register_tools(config.name, tools)
                return True

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
                    self.servers[config.name] = ServerInfo(
                        config=config, status="error", last_error=str(e)
                    )
                    return False

        return False

    async def _test_and_list_tools_stdio(self, config: MCPServerConfig) -> list:
        """Test stdio server connection and list tools."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )

        log_event(
            "mcp_stdio_connecting", {"server": config.name, "command": config.command}
        )
        timeout = (
            timedelta(seconds=config.connect_timeout)
            if config.connect_timeout
            else None
        )

        # Create stderr capture to intercept subprocess output
        # Capture the event loop for thread-safe event emission
        loop = asyncio.get_running_loop()
        stderr_capture = StderrCapture(
            server_name=config.name,
            emit_callback=lambda line: self._emit_stderr_sync(loop, config.name, line),
        )
        stderr_capture.start()

        try:
            async with asyncio.timeout(config.connect_timeout):
                async with stdio_client(server_params, errlog=stderr_capture) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(
                        read_stream, write_stream, read_timeout_seconds=timeout
                    ) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        log_event(
                            "mcp_connected",
                            {
                                "server": config.name,
                                "transport": "stdio",
                                "tool_count": len(tools_result.tools),
                            },
                        )
                        return tools_result.tools
        finally:
            stderr_capture.stop()

    async def _test_and_list_tools_http(self, config: MCPServerConfig) -> list:
        """Test HTTP server connection and list tools."""
        log_event("mcp_http_connecting", {"server": config.name, "url": config.url})
        timeout = (
            timedelta(seconds=config.connect_timeout)
            if config.connect_timeout
            else None
        )

        async with asyncio.timeout(config.connect_timeout):
            async with streamablehttp_client(config.url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=timeout
                ) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    log_event(
                        "mcp_connected",
                        {
                            "server": config.name,
                            "transport": "http",
                            "tool_count": len(tools_result.tools),
                        },
                    )
                    return tools_result.tools

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
        """Call an MCP tool (reconnects for each call).

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
        if server.status != "connected":
            return json.dumps({"error": f"MCP server '{server_name}' not connected"})

        config = server.config
        timeout = config.tool_timeout

        log_event(
            "mcp_tool_call_start",
            {"server": server_name, "tool": actual_tool_name, "arguments": arguments},
        )

        start_time = asyncio.get_event_loop().time()

        try:
            async with asyncio.timeout(timeout):
                async with self._semaphore:
                    # Reconnect for each call
                    if config.transport == "stdio":
                        result = await self._call_stdio_tool(
                            config, actual_tool_name, arguments
                        )
                    else:
                        result = await self._call_http_tool(
                            config, actual_tool_name, arguments
                        )

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

    async def _call_stdio_tool(
        self, config: MCPServerConfig, tool_name: str, arguments: dict
    ):
        """Call a tool on a stdio server."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )
        timeout = (
            timedelta(seconds=config.tool_timeout) if config.tool_timeout else None
        )

        # Create stderr capture to intercept subprocess output
        # Capture the event loop for thread-safe event emission
        loop = asyncio.get_running_loop()
        stderr_capture = StderrCapture(
            server_name=config.name,
            emit_callback=lambda line: self._emit_stderr_sync(loop, config.name, line),
        )
        stderr_capture.start()

        try:
            async with stdio_client(server_params, errlog=stderr_capture) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=timeout
                ) as session:
                    await session.initialize()
                    return await session.call_tool(
                        tool_name, arguments, read_timeout_seconds=timeout
                    )
        finally:
            stderr_capture.stop()

    async def _call_http_tool(
        self, config: MCPServerConfig, tool_name: str, arguments: dict
    ):
        """Call a tool on an HTTP server."""
        timeout = (
            timedelta(seconds=config.tool_timeout) if config.tool_timeout else None
        )

        async with streamablehttp_client(config.url) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream, write_stream, read_timeout_seconds=timeout
            ) as session:
                await session.initialize()
                return await session.call_tool(
                    tool_name, arguments, read_timeout_seconds=timeout
                )

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
            import json

            return json.dumps(result.structuredContent, indent=2)

        return str(result)
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
        """Reload configuration and reconnect to all servers."""
        self.servers = {}
        self.tools = []

        for config in self.server_configs.values():
            await self.connect_server_if_needed(config.name)

        return self.get_server_info()

    async def disconnect(self) -> dict[str, list[str]]:
        """Disconnect from all servers and clear tools.

        Returns:
            Empty server info dict
        """
        # Mark all servers as disconnected
        for name, server in self.servers.items():
            server.status = "disconnected"

        # Clear state
        self.servers = {}
        self.tools = []

        return self.get_server_info()

    async def cleanup(self, timeout: float = 5.0) -> None:
        """Disconnect from all servers gracefully.

        With reconnect-per-operation, this just marks us as shutting down.
        """
        self._shutting_down = True
        for name, server in self.servers.items():
            server.status = "disconnected"
