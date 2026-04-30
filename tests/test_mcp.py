"""Tests for MCP (Model Context Protocol) integration."""

import pytest
from unittest.mock import MagicMock
from agent13.mcp import MCPManager, MCPTool, MCP_AVAILABLE
from agent13.config import MCPServerConfig


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
