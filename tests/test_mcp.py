"""Tests for MCP (Model Context Protocol) integration."""

import asyncio

import pytest
from unittest.mock import MagicMock
from agent13.mcp import MCPManager, MCPTool, MCP_AVAILABLE
from agent13.config import MCPServerConfig
from agent13.events import AgentEvent


@pytest.fixture
def http_config():
    """Create a basic HTTP MCP server config."""
    return MCPServerConfig(
        name="test_server", transport="http", url="http://localhost:8080/mcp"
    )


@pytest.fixture
def stdio_config():
    """Create a basic stdio MCP server config."""
    return MCPServerConfig(
        name="stdio_server", transport="stdio", command="uvx", args=["mcp-server-fetch"]
    )


class TestMCPServerConfig:
    """Tests for MCPServerConfig validation."""

    def test_validate_valid_http(self, http_config):
        """Valid HTTP config should have no errors."""
        assert http_config.validate() == []

    def test_validate_valid_stdio(self, stdio_config):
        """Valid stdio config should have no errors."""
        assert stdio_config.validate() == []

    def test_validate_invalid_name_slash(self):
        """Name with slash should fail validation."""
        config = MCPServerConfig(
            name="bad/name", transport="http", url="http://localhost"
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid server name" in errors[0]

    def test_validate_invalid_name_space(self):
        """Name with space should fail validation."""
        config = MCPServerConfig(
            name="bad name", transport="http", url="http://localhost"
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid server name" in errors[0]

    def test_validate_invalid_transport(self):
        """Invalid transport should fail validation."""
        config = MCPServerConfig(
            name="test", transport="websocket", url="http://localhost"
        )
        errors = config.validate()
        assert any("Invalid transport" in e for e in errors)

    def test_validate_missing_url(self):
        """HTTP transport without URL should fail."""
        config = MCPServerConfig(name="test", transport="http")
        errors = config.validate()
        assert any("requires 'url'" in e for e in errors)

    def test_validate_invalid_url_scheme(self):
        """URL without http/https should fail."""
        config = MCPServerConfig(name="test", transport="http", url="ftp://localhost")
        errors = config.validate()
        assert any("Invalid URL scheme" in e for e in errors)

    def test_validate_missing_command(self):
        """stdio transport without command should fail."""
        config = MCPServerConfig(name="test", transport="stdio")
        errors = config.validate()
        assert any("requires 'command'" in e for e in errors)

    def test_validate_overlapping_tools(self):
        """Tools in both enabled and disabled should fail."""
        config = MCPServerConfig(
            name="test",
            transport="http",
            url="http://localhost",
            enabled_tools=["tool1", "tool2"],
            disabled_tools=["tool2", "tool3"],
        )
        errors = config.validate()
        assert any("both enabled and disabled" in e for e in errors)

    def test_default_timeouts(self, http_config):
        """Default timeouts should be set."""
        assert http_config.connect_timeout == 240.0
        assert http_config.tool_timeout == 60.0
        assert http_config.retry_attempts == 3
        assert http_config.retry_delay == 1.0


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_mcp_tool_creation(self):
        """Basic MCPTool creation."""
        tool = MCPTool(
            server_name="test_server",
            name="mcp://test_server/fetch",
            original_name="fetch",
            description="Fetch a URL",
            input_schema={"type": "object"},
        )
        assert tool.server_name == "test_server"
        assert tool.name == "mcp://test_server/fetch"
        assert tool.original_name == "fetch"


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestMCPManager:
    """Tests for MCPManager class."""

    def test_init_empty(self):
        """MCPManager with no servers."""
        manager = MCPManager([])
        assert manager.servers == {}
        assert manager.tools == []

    def test_init_with_configs(self, http_config, stdio_config):
        """MCPManager with server configs."""
        manager = MCPManager([http_config, stdio_config])
        assert "test_server" in manager.server_configs
        assert "stdio_server" in manager.server_configs

    def test_uri_tool_naming(self, http_config):
        """Tool names use URI format to prevent collisions."""
        manager = MCPManager([http_config])
        manager.servers["test_server"] = MagicMock(tools=[])

        tool = MagicMock()
        tool.name = "fetch"
        tool.description = "Fetch URL"
        tool.inputSchema = {}
        manager._register_tools("test_server", [tool])

        assert len(manager.tools) == 1
        assert manager.tools[0].name == "mcp://test_server/fetch"

    def test_collision_prevention(self, http_config):
        """Duplicate tools are skipped."""
        manager = MCPManager([http_config])
        manager.servers["test_server"] = MagicMock(tools=[])

        tool = MagicMock()
        tool.name = "fetch"
        tool.description = "Fetch"
        tool.inputSchema = {}
        manager._register_tools("test_server", [tool])
        manager._register_tools("test_server", [tool])  # Same tool again

        assert len(manager.tools) == 1

    def test_disabled_tools_skipped(self, http_config):
        """Disabled tools are not registered."""
        http_config.disabled_tools = ["bad_tool"]
        manager = MCPManager([http_config])
        manager.servers["test_server"] = MagicMock(tools=[])

        good_tool = MagicMock()
        good_tool.name = "good_tool"
        good_tool.description = "Good"
        good_tool.inputSchema = {}

        bad_tool = MagicMock()
        bad_tool.name = "bad_tool"
        bad_tool.description = "Bad"
        bad_tool.inputSchema = {}

        manager._register_tools("test_server", [good_tool, bad_tool])

        assert len(manager.tools) == 1
        assert manager.tools[0].original_name == "good_tool"

    def test_enabled_tools_whitelist(self, http_config):
        """Only enabled tools are registered when whitelist exists."""
        http_config.enabled_tools = ["good_tool"]
        manager = MCPManager([http_config])
        manager.servers["test_server"] = MagicMock(tools=[])

        good_tool = MagicMock()
        good_tool.name = "good_tool"
        good_tool.description = "Good"
        good_tool.inputSchema = {}

        other_tool = MagicMock()
        other_tool.name = "other_tool"
        other_tool.description = "Other"
        other_tool.inputSchema = {}

        manager._register_tools("test_server", [good_tool, other_tool])

        assert len(manager.tools) == 1
        assert manager.tools[0].original_name == "good_tool"

    def test_get_openai_tools(self, http_config):
        """Tools are converted to OpenAI format."""
        manager = MCPManager([http_config])
        manager.tools = [
            MCPTool(
                server_name="test",
                name="mcp://test/fetch",
                original_name="fetch",
                description="Fetch URL",
                input_schema={"type": "object", "properties": {}},
            )
        ]

        openai_tools = manager.get_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "mcp://test/fetch"
        assert openai_tools[0]["function"]["description"] == "Fetch URL"

    def test_format_result_text(self):
        """Text content is extracted correctly."""
        manager = MCPManager([])

        result = MagicMock()
        result.content = [MagicMock(type="text", text="Hello")]

        assert manager._format_result(result) == "Hello"

    def test_format_result_multiple(self):
        """Multiple content items are joined."""
        manager = MCPManager([])

        result = MagicMock()
        result.content = [
            MagicMock(type="text", text="Hello"),
            MagicMock(type="text", text="World"),
        ]

        assert manager._format_result(result) == "Hello\nWorld"

    def test_format_result_image(self):
        """Image content gets placeholder."""
        manager = MCPManager([])

        result = MagicMock()
        result.content = [
            MagicMock(type="text", text="Here's an image:"),
            MagicMock(type="image"),
        ]

        formatted = manager._format_result(result)
        assert "[Image content received from MCP tool]" in formatted

    def test_format_result_resource(self):
        """Resource content gets placeholder with URI."""
        manager = MCPManager([])

        result = MagicMock()
        result.content = [MagicMock(type="resource", uri="file:///test.txt")]

        formatted = manager._format_result(result)
        assert "[Resource: file:///test.txt]" in formatted

    def test_format_result_empty(self):
        """Empty content returns placeholder."""
        manager = MCPManager([])

        result = MagicMock()
        result.content = []
        result.structuredContent = None

        assert manager._format_result(result) == "[Empty result]"

    def test_get_server_info_empty(self):
        """get_server_info with no servers."""
        manager = MCPManager([])
        assert manager.get_server_info() == {}

    def test_get_server_info_connected(self, http_config):
        """get_server_info returns connected servers."""
        from agent13.mcp import ServerInfo, MCPTool

        manager = MCPManager([http_config])
        manager.servers["test_server"] = ServerInfo(
            config=http_config,
            tools=[MagicMock(name="tool1"), MagicMock(name="tool2")],
            status="connected",
        )
        # Add tools to manager.tools for get_server_info to find
        manager.tools = [
            MCPTool(
                server_name="test_server",
                name="mcp://test_server/tool1",
                original_name="tool1",
                description="",
                input_schema={},
            ),
            MCPTool(
                server_name="test_server",
                name="mcp://test_server/tool2",
                original_name="tool2",
                description="",
                input_schema={},
            ),
        ]

        info = manager.get_server_info()
        assert "test_server" in info
        assert len(info["test_server"]) == 2


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
@pytest.mark.asyncio
class TestMCPManagerAsync:
    """Async tests for MCPManager."""

    async def test_call_tool_invalid_format(self, http_config):
        """call_tool returns error JSON for invalid tool name format."""
        manager = MCPManager([http_config])

        import json

        result = json.loads(await manager.call_tool("not_mcp_format", {}))
        assert "error" in result
        assert "Invalid MCP tool name" in result["error"]

        result = json.loads(await manager.call_tool("mcp://missing_tool_name", {}))
        assert "error" in result
        assert "Invalid MCP tool name" in result["error"]

    async def test_call_tool_unknown_server(self, http_config):
        """call_tool returns error JSON for unknown server."""
        manager = MCPManager([http_config])

        import json

        result = json.loads(await manager.call_tool("mcp://unknown/tool", {}))
        assert "error" in result
        assert "not configured" in result["error"]

    async def test_cleanup_sets_flag(self, http_config):
        """cleanup sets shutting_down flag."""
        manager = MCPManager([http_config])
        assert manager._shutting_down is False

        await manager.cleanup()

        assert manager._shutting_down is True  # Set by cleanup


class TestMCPConfigIntegration:
    """Tests for MCP config parsing integration."""

    def test_config_mcp_servers_empty(self, tmp_path):
        """Config with no MCP servers."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "http://localhost:8080/v1"
""")

        from agent13.config import Config

        config = Config.from_file(config_file)
        assert config.mcp_servers == []

    def test_config_mcp_servers_http(self, tmp_path):
        """Config with HTTP MCP server."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "http://localhost:8080/v1"

[[mcp_servers]]
name = "hvac"
transport = "http"
url = "http://localhost:4040/mcp"
""")

        from agent13.config import Config

        config = Config.from_file(config_file)
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].name == "hvac"
        assert config.mcp_servers[0].transport == "http"
        assert config.mcp_servers[0].url == "http://localhost:4040/mcp"

    def test_config_mcp_servers_stdio(self, tmp_path):
        """Config with stdio MCP server."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "http://localhost:8080/v1"

[[mcp_servers]]
name = "fetch"
transport = "stdio"
command = "uvx"
args = ["mcp-server-fetch"]
""")

        from agent13.config import Config

        config = Config.from_file(config_file)
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].name == "fetch"
        assert config.mcp_servers[0].transport == "stdio"
        assert config.mcp_servers[0].command == "uvx"
        assert config.mcp_servers[0].args == ["mcp-server-fetch"]

    def test_config_mcp_servers_invalid(self, tmp_path):
        """Config with invalid MCP server raises error."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "http://localhost:8080/v1"

[[mcp_servers]]
name = "bad/server"
transport = "http"
url = "http://localhost:4040/mcp"
""")

        from agent13.config import Config

        with pytest.raises(ValueError, match="Invalid server name"):
            Config.from_file(config_file)

    def test_config_mcp_servers_missing_url(self, tmp_path):
        """Config with HTTP server missing URL raises error."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "http://localhost:8080/v1"

[[mcp_servers]]
name = "hvac"
transport = "http"
""")

        from agent13.config import Config

        with pytest.raises(ValueError, match="requires 'url'"):
            Config.from_file(config_file)

    async def test_call_tool_rejects_when_shutting_down(self, http_config):
        """call_tool returns error JSON when manager is shutting down."""
        manager = MCPManager([http_config])

        # Set the shutdown flag manually
        manager._shutting_down = True

        import json

        result = json.loads(await manager.call_tool("mcp://test_server/test_tool", {}))
        assert "error" in result
        assert "shutting down" in result["error"]


class TestPersistentSessions:
    """Regression tests for the persistent-session refactor.

    These cover the contracts in docs_archive/mcp_connection_fix_plan.md §6.1:
    ServerInfo defaults, connect launches a task, call_tool reuses the
    stored session, broken sessions trigger exactly one reconnect,
    disconnect cancels the task, reload doesn't leak.
    """

    def test_server_info_defaults_unchanged(self):
        """New ServerInfo fields must default to None for back-compat."""
        from agent13.mcp import ServerInfo

        s = ServerInfo(config=None)
        assert s.session is None
        assert s.stderr_capture is None
        assert s.session_task is None
        assert s._stop_event is None
        assert s._ready_event is None
        assert s.tools == []
        assert s.status == "disconnected"
        assert s.last_error is None

    async def test_connect_creates_session_task(self, monkeypatch, http_config):
        """connect_server_if_needed launches _session_runner and stores the task."""
        manager = MCPManager([http_config])

        # Stub _session_runner so it does the minimum: set session, signal
        # _ready_event, then park on _stop_event. Avoids real transport.
        async def fake_runner(name):
            server = manager.servers[name]
            server.session = MagicMock()  # stand-in for live ClientSession
            server.tools = []
            server.status = "connected"
            server._ready_event.set()
            await server._stop_event.wait()

        monkeypatch.setattr(manager, "_session_runner", fake_runner)

        connected = await manager.connect_server_if_needed("test_server")
        assert connected is True

        server = manager.servers["test_server"]
        assert server.session_task is not None
        assert isinstance(server.session_task, asyncio.Task)
        assert server.session is not None
        assert server.status == "connected"

        await manager.cleanup()
        assert server.session_task.done()

    async def test_call_tool_reuses_session(self, monkeypatch, http_config):
        """call_tool must invoke server.session.call_tool and not open a transport."""
        from agent13.mcp import ServerInfo

        manager = MCPManager([http_config])

        # Wire a fake connected server with a mock session.
        server = manager.servers["test_server"] = ServerInfo(
            config=http_config, status="connected"
        )
        server._stop_event = asyncio.Event()
        server._ready_event = asyncio.Event()
        mock_session = MagicMock()
        # call_tool awaits session.call_tool(...) -> return a result with .content
        call_tool_calls = []

        async def fake_call_tool(name, args, read_timeout_seconds=None):
            call_tool_calls.append((name, args))
            result = MagicMock()
            result.content = [MagicMock(text="ok")]
            return result

        mock_session.call_tool = fake_call_tool
        server.session = mock_session

        # _open_transport_cm must NOT be called — patch it to explode if it is.
        async def explode(*a, **kw):
            raise AssertionError("call_tool opened a transport; should reuse session")

        monkeypatch.setattr(manager, "_open_transport_cm", explode)

        out = await manager.call_tool("mcp://test_server/my_tool", {"x": 1})
        assert "ok" in out
        assert len(call_tool_calls) == 1
        assert call_tool_calls[0] == ("my_tool", {"x": 1})

        await manager.cleanup()

    async def test_call_tool_reconnects_on_broken_session(
        self, monkeypatch, http_config
    ):
        """A BrokenResourceError on first call triggers ONE reconnect + retry."""
        import anyio

        from agent13.mcp import ServerInfo

        manager = MCPManager([http_config])

        # Seed a connected server whose session raises on first call_tool.
        server = manager.servers["test_server"] = ServerInfo(
            config=http_config, status="connected"
        )
        server._stop_event = asyncio.Event()
        server._ready_event = asyncio.Event()

        call_count = {"n": 0}

        async def first_session_call(name, args, read_timeout_seconds=None):
            call_count["n"] += 1
            raise anyio.BrokenResourceError("pipe broke")

        first_session = MagicMock()
        first_session.call_tool = first_session_call
        server.session = first_session

        # After reconnect, _connect_once must produce a fresh server/session.
        second_session = MagicMock()

        async def second_session_call(name, args, read_timeout_seconds=None):
            call_count["n"] += 1
            r = MagicMock()
            r.content = [MagicMock(text="recovered")]
            return r

        second_session.call_tool = second_session_call

        async def fake_connect_once(name):
            # Replace the server with a fresh connected one holding second_session.
            new_server = ServerInfo(config=http_config, status="connected")
            new_server._stop_event = asyncio.Event()
            new_server._ready_event = asyncio.Event()
            new_server.session = second_session
            manager.servers[name] = new_server
            return True

        monkeypatch.setattr(manager, "_connect_once", fake_connect_once)

        out = await manager.call_tool("mcp://test_server/my_tool", {"x": 1})
        # First call raised BrokenResourceError, reconnect happened, second
        # call succeeded.
        assert call_count["n"] == 2, f"Expected 2 calls, got {call_count['n']}"
        assert "recovered" in out

        await manager.cleanup()

    async def test_disconnect_cancels_session_task(self, monkeypatch, http_config):
        """disconnect() cancels the session_task and stops stderr_capture."""
        manager = MCPManager([http_config])

        async def fake_runner(name):
            server = manager.servers[name]
            try:
                server._ready_event.set()
                await server._stop_event.wait()
            except asyncio.CancelledError:
                raise

        monkeypatch.setattr(manager, "_session_runner", fake_runner)

        await manager.connect_server_if_needed("test_server")
        server = manager.servers["test_server"]

        # Inject a fake stderr_capture to verify stop() is called.
        stopped = {"called": False}

        class FakeCapture:
            def stop(self):
                stopped["called"] = True

        server.stderr_capture = FakeCapture()

        await manager.disconnect()
        assert server.session_task.done()
        assert stopped["called"] is True

    async def test_reload_does_not_leak(self, monkeypatch, http_config):
        """reload() tears down the prior session before starting a new one."""
        manager = MCPManager([http_config])
        teardown_calls = {"names": []}

        async def spy_teardown(name):
            teardown_calls["names"].append(name)
            # mimic real teardown: cancel task, mark disconnected
            server = manager.servers.get(name)
            if server and server.session_task and not server.session_task.done():
                server.session_task.cancel()
                try:
                    await asyncio.wait_for(server.session_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            if server:
                server.status = "disconnected"

        monkeypatch.setattr(manager, "_teardown_server", spy_teardown)

        async def fake_runner(name):
            server = manager.servers[name]
            server.session = MagicMock()
            server.tools = []
            server.status = "connected"
            server._ready_event.set()
            await server._stop_event.wait()

        monkeypatch.setattr(manager, "_session_runner", fake_runner)

        await manager.connect_server_if_needed("test_server")
        first_task = manager.servers["test_server"].session_task

        await manager.reload()
        # Reload should have torn down the old server before reconnecting.
        assert "test_server" in teardown_calls["names"]
        assert first_task.done() or first_task.cancelled()


class TestMCPSDK2Fallback:
    """Tests for the MCP SDK 2.0 fallback (auto-inject --with mcp<2)."""

    def test_is_uvx_needing_fallback_plain_uvx(self):
        """Plain uvx command without mcp pin → needs fallback."""
        config = MCPServerConfig(
            name="srv", transport="stdio", command="uvx", args=["some-mcp"]
        )
        assert MCPManager._is_uvx_needing_mcp_fallback(config) is True

    def test_is_uvx_needing_fallback_already_pinned(self):
        """uvx with --with mcp<2 already present → no fallback needed."""
        config = MCPServerConfig(
            name="srv",
            transport="stdio",
            command="uvx",
            args=["--with", "mcp<2", "some-mcp"],
        )
        assert MCPManager._is_uvx_needing_mcp_fallback(config) is False

    def test_is_uvx_needing_fallback_other_with(self):
        """uvx with --with for a different package → still needs fallback."""
        config = MCPServerConfig(
            name="srv",
            transport="stdio",
            command="uvx",
            args=["--with", "httpx", "some-mcp"],
        )
        assert MCPManager._is_uvx_needing_mcp_fallback(config) is True

    def test_is_uvx_needing_fallback_not_uvx(self):
        """Non-uvx command → no fallback."""
        config = MCPServerConfig(
            name="srv", transport="stdio", command="python", args=["server.py"]
        )
        assert MCPManager._is_uvx_needing_mcp_fallback(config) is False

    def test_is_uvx_needing_fallback_http_transport(self):
        """HTTP transport → no fallback (only stdio/uvx)."""
        config = MCPServerConfig(
            name="srv", transport="http", url="http://localhost:8080"
        )
        assert MCPManager._is_uvx_needing_mcp_fallback(config) is False

    def test_patched_config_injects_mcp_pin(self):
        """Patched config should have --with mcp<2 prepended."""
        config = MCPServerConfig(
            name="srv", transport="stdio", command="uvx", args=["drawio-mcp"]
        )
        patched = MCPManager._patched_config_mcp1x(config)
        assert patched.args == ["--with", "mcp<2", "drawio-mcp"]
        # Original should be untouched.
        assert config.args == ["drawio-mcp"]

    def test_fallback_warning_message_format(self):
        """Warning message should contain config fix and explanation."""
        config = MCPServerConfig(
            name="drawio",
            transport="stdio",
            command="uvx",
            args=["--with", "mcp<2", "drawio-mcp"],
        )
        result = MCPManager._mcp_fallback_warning("drawio", config)
        assert result["server_name"] == "drawio"
        warning = result["warning"]
        assert "mcp.server.fastmcp" in warning
        assert "MCP SDK 2.0" in warning
        assert 'args = ["--with", "mcp<2", "drawio-mcp"]' in warning
        assert "config.toml" in warning

    async def test_fallback_triggers_on_failure(self, monkeypatch):
        """When first connect fails, fallback should retry with mcp<2."""
        config = MCPServerConfig(
            name="test_srv",
            transport="stdio",
            command="uvx",
            args=["fake-mcp"],
            retry_attempts=1,
            retry_delay=0.01,
            connect_timeout=1.0,
        )
        manager = MCPManager([config])

        call_count = {"n": 0}
        events = []

        async def fake_connect_with_retry(cfg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call (original config) fails.
                return False
            # Second call (patched config) succeeds.
            # Simulate a connected server.
            from agent13.mcp import ServerInfo

            server = ServerInfo(config=cfg, status="connected", tools=[])
            manager.servers[config.name] = server
            return True

        monkeypatch.setattr(manager, "_connect_with_retry", fake_connect_with_retry)

        async def capture_event(event, data):
            events.append((event, data.data))

        manager.set_event_callback(capture_event)

        result = await manager.connect_server_if_needed("test_srv")

        assert result is True
        assert call_count["n"] == 2  # original + fallback
        # Config args should be persisted with the pin.
        assert config.args == ["--with", "mcp<2", "fake-mcp"]
        # Warning event should have been emitted.
        warning_events = [e for e in events if e[0] == AgentEvent.MCP_SERVER_WARNING]
        assert len(warning_events) == 1
        assert "fake-mcp" in warning_events[0][1]["warning"]

    async def test_fallback_skipped_when_already_pinned(self, monkeypatch):
        """Server with --with mcp<2 should not trigger fallback on failure."""
        config = MCPServerConfig(
            name="test_srv",
            transport="stdio",
            command="uvx",
            args=["--with", "mcp<2", "fake-mcp"],
            retry_attempts=1,
            retry_delay=0.01,
            connect_timeout=1.0,
        )
        manager = MCPManager([config])

        call_count = {"n": 0}

        async def fake_connect_with_retry(cfg):
            call_count["n"] += 1
            return False  # Always fails.

        monkeypatch.setattr(manager, "_connect_with_retry", fake_connect_with_retry)

        result = await manager.connect_server_if_needed("test_srv")

        assert result is False
        assert call_count["n"] == 1  # No fallback retry.
        # Config args should be unchanged.
        assert config.args == ["--with", "mcp<2", "fake-mcp"]
