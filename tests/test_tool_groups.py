"""Tests for tool groups, name_matches, get_filtered_tools, and devel mode."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (  # noqa: E402
    get_tools,
    get_filtered_tools,
    get_tool_groups,
    name_matches,
    _ensure_discovered,
)


# --- name_matches tests ---


class TestNameMatches:
    """Test the name_matches pattern matching function."""

    def test_exact_match(self):
        assert name_matches("read_file", ["read_file"]) is True

    def test_exact_no_match(self):
        assert name_matches("read_file", ["write_file"]) is False

    def test_glob_wildcard_match(self):
        assert name_matches("tui_launch", ["tui_*"]) is True

    def test_glob_wildcard_no_match(self):
        assert name_matches("read_file", ["tui_*"]) is False

    def test_glob_question_mark(self):
        assert name_matches("tui_a", ["tui_?"]) is True

    def test_glob_question_mark_no_match(self):
        assert name_matches("tui_ab", ["tui_?"]) is False

    def test_regex_match(self):
        assert name_matches("tui_launch", ["re:^tui_.*"]) is True

    def test_regex_no_match(self):
        assert name_matches("read_file", ["re:^tui_.*"]) is False

    def test_case_insensitive_glob(self):
        assert name_matches("TUI_Launch", ["tui_*"]) is True

    def test_case_insensitive_exact(self):
        assert name_matches("Read_File", ["read_file"]) is True

    def test_multiple_patterns_any_match(self):
        assert name_matches("read_file", ["tui_*", "read_*"]) is True

    def test_multiple_patterns_none_match(self):
        assert name_matches("edit_file", ["tui_*", "read_*"]) is False

    def test_empty_patterns(self):
        assert name_matches("read_file", []) is False

    def test_empty_string_pattern_skipped(self):
        assert name_matches("read_file", [""]) is False

    def test_whitespace_pattern_skipped(self):
        assert name_matches("read_file", ["  "]) is False

    def test_regex_invalid_pattern_skipped(self):
        # Invalid regex should not crash, just not match
        assert name_matches("read_file", ["re:[invalid"]) is False

    def test_mixed_glob_and_regex(self):
        assert name_matches("tui_launch", ["read_*", "re:^tui.*"]) is True


# --- get_filtered_tools tests ---


class TestGetFilteredTools:
    """Test the get_filtered_tools function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure tools are discovered before each test."""
        _ensure_discovered()

    def test_no_filters_returns_non_devel(self):
        """Without filters and devel=False, non-devel tools are returned."""
        all_tools = get_tools()
        filtered = get_filtered_tools()
        # devel=False by default, so should have fewer tools if any devel tools exist
        assert len(filtered) <= len(all_tools)

    def test_devel_false_hides_devel_tools(self):
        """With devel=False, devel-group tools are hidden."""
        all_tools = get_tools()
        filtered = get_filtered_tools(devel=False)
        # Should have fewer tools if any devel tools exist
        assert len(filtered) <= len(all_tools)

        # Check that no devel tools are in the result
        for schema in filtered:
            name = schema["function"]["name"]
            groups = get_tool_groups(name)
            assert "devel" not in groups

    def test_devel_true_shows_devel_tools(self):
        """With devel=True, devel-group tools are included."""
        filtered_off = get_filtered_tools(devel=False)
        filtered_on = get_filtered_tools(devel=True)
        # Should have more (or equal) tools with devel on
        assert len(filtered_on) >= len(filtered_off)

    def test_enabled_tools_whitelist(self):
        """enabled_tools acts as a whitelist."""
        filtered = get_filtered_tools(enabled_tools=["read_file"])
        names = [s["function"]["name"] for s in filtered]
        # Only read_file should be present (or any matching the pattern)
        assert "read_file" in names
        # tui tools should not be present (they don't match "read_file")
        for name in names:
            assert name == "read_file"

    def test_disabled_tools_blacklist(self):
        """disabled_tools acts as a blacklist when enabled_tools is empty."""
        filtered = get_filtered_tools(disabled_tools=["square_number"])
        names = [s["function"]["name"] for s in filtered]
        assert "square_number" not in names
        assert "read_file" in names

    def test_enabled_overrides_disabled(self):
        """When enabled_tools is non-empty, disabled_tools is ignored."""
        # enabled_tools=["read_file"] should whitelist to just read_file
        # even if disabled_tools would also exclude it
        filtered = get_filtered_tools(
            enabled_tools=["read_file"],
            disabled_tools=["read_file"],
        )
        names = [s["function"]["name"] for s in filtered]
        assert "read_file" in names

    def test_glob_pattern_in_enabled(self):
        """Glob patterns work in enabled_tools."""
        filtered = get_filtered_tools(enabled_tools=["tui_*"])
        names = [s["function"]["name"] for s in filtered]
        # All should start with tui_
        for name in names:
            assert name.startswith("tui_")

    def test_glob_pattern_in_disabled(self):
        """Glob patterns work in disabled_tools."""
        filtered = get_filtered_tools(
            devel=True,
            disabled_tools=["tui_*"],
        )
        names = [s["function"]["name"] for s in filtered]
        for name in names:
            assert not name.startswith("tui_")

    def test_devel_and_disabled_combined(self):
        """Devel filter and disabled_tools compose correctly."""
        # devel=False hides devel tools, disabled hides more
        filtered = get_filtered_tools(
            devel=False,
            disabled_tools=["square_number"],
        )
        names = [s["function"]["name"] for s in filtered]
        assert "square_number" not in names
        # No devel tools either
        for name in names:
            groups = get_tool_groups(name)
            assert "devel" not in groups


# --- get_tool_groups tests ---


class TestGetToolGroups:
    """Test the get_tool_groups function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _ensure_discovered()

    def test_devel_tool_has_devel_group(self):
        """TUI viewer tools should have the 'devel' group."""
        groups = get_tool_groups("tui_launch")
        assert "devel" in groups

    def test_regular_tool_has_no_groups(self):
        """Non-devel tools should have empty groups."""
        groups = get_tool_groups("read_file")
        assert groups == []

    def test_unknown_tool_has_no_groups(self):
        """Unknown tools should have empty groups."""
        groups = get_tool_groups("nonexistent_tool")
        assert groups == []


# --- Agent devel_mode tests ---


class TestAgentDevelMode:
    """Test Agent.devel_mode and Agent.set_devel_mode()."""

    def test_default_devel_mode_off(self):
        from agent13 import Agent
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1234")
        agent = Agent(client=client, model="test")
        assert agent.devel_mode is False

    def test_devel_mode_on_init(self):
        from agent13 import Agent
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1234")
        agent = Agent(client=client, model="test", devel_mode=True)
        assert agent.devel_mode is True

    def test_set_devel_mode_toggles(self):
        from agent13 import Agent
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1234")
        agent = Agent(client=client, model="test")
        assert agent.devel_mode is False
        agent.set_devel_mode(True)
        assert agent.devel_mode is True
        agent.set_devel_mode(False)
        assert agent.devel_mode is False

    def test_devel_mode_tools_count(self):
        """When devel mode is on, more tools should be visible."""
        from agent13 import Agent
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1234")

        agent_off = Agent(client=client, model="test", devel_mode=False)
        agent_on = Agent(client=client, model="test", devel_mode=True)
        assert len(agent_on.tools) >= len(agent_off.tools)


# --- Config enabled/disabled_tools tests ---


class TestConfigEnabledDisabledTools:
    """Test Config.enabled_tools and Config.disabled_tools parsing."""

    def test_default_empty(self):
        from agent13.config import Config

        config = Config()
        assert config.enabled_tools == []
        assert config.disabled_tools == []

    def test_from_toml(self, tmp_path):
        from agent13.config import Config

        config_file = tmp_path / "config.toml"
        config_file.write_text("""enabled_tools = ["read_*"]
disabled_tools = ["square_number"]

[[providers]]
name = "test"
api_base = "http://localhost:1234"
api_key_env_var = "TEST_KEY"
""")
        config = Config.from_file(config_file)
        assert config.enabled_tools == ["read_*"]
        assert config.disabled_tools == ["square_number"]
