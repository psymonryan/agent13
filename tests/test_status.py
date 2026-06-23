"""Tests for agent13/status.py — shared status data gathering."""

import os
import time
from unittest import mock

import pytest

from agent13.status import (
    StatusData,
    format_duration,
    fmt_tokens,
    gather_status,
    toggle_enum,
    get_tool_stats_summary,
)


# ── Mock helpers ──


class MockToolStats:
    """Minimal tool stats mock."""

    def __init__(self, successes=0, calls=0):
        self.total_successes = successes
        self.total_calls = calls


class MockMCP:
    """Minimal MCP mock."""

    def __init__(self, connected=False):
        self._connected = connected

    def is_connected(self):
        return self._connected


class MockQueue:
    """Minimal queue mock."""

    def __init__(self, pending=0):
        self.pending_count = pending


class MockAgent:
    """Minimal agent mock for status gathering."""

    def __init__(self, **kwargs):
        self.messages = kwargs.get("messages", [])
        self.queue = MockQueue(kwargs.get("pending", 0))
        self.is_pausing = kwargs.get("is_pausing", False)
        self.is_paused = kwargs.get("is_paused", False)
        self.mcp = kwargs.get("mcp", None)
        self.tool_stats = kwargs.get("tool_stats", MockToolStats())
        self.journal_mode = kwargs.get("journal_mode", False)
        self.devel_mode = kwargs.get("devel_mode", False)
        self.skills_mode = kwargs.get("skills_mode", False)
        self.remove_reasoning = kwargs.get("remove_reasoning", False)

    @property
    def status(self):
        from agent13.core import AgentStatus
        if self.is_paused:
            return AgentStatus.IDLE
        return AgentStatus.IDLE

    @property
    def is_idle(self):
        return not self.is_pausing and not self.is_paused


class MockTracker:
    """Minimal tracker mock."""

    def __init__(self, **kwargs):
        self.prompt_tokens = kwargs.get("prompt_tokens", 0)
        self.completion_tokens = kwargs.get("completion_tokens", 0)
        self._turn_count = kwargs.get("turn_count", 0)
        self._total_processing_time = kwargs.get("total_processing_time", 0.0)


class MockPromptManager:
    """Minimal prompt manager mock."""

    def __init__(self, active="default"):
        self.active_prompt = active


# ── format_duration tests ──


class TestFormatDuration:
    """Tests for format_duration()."""

    def test_seconds_only(self):
        assert format_duration(42) == "42s"

    def test_minutes_and_seconds(self):
        assert format_duration(65) == "1m 5s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_exact_minute(self):
        assert format_duration(120) == "2m 0s"

    def test_exact_hour(self):
        assert format_duration(3600) == "1h 0m 0s"

    def test_float_truncated(self):
        assert format_duration(61.7) == "1m 1s"


# ── fmt_tokens tests ──


class TestFmtTokens:
    """Tests for fmt_tokens()."""

    def test_small_number(self):
        assert fmt_tokens(42) == "42"

    def test_zero(self):
        assert fmt_tokens(0) == "0"

    def test_thousands(self):
        assert fmt_tokens(12345) == "12.3k"

    def test_exact_thousand(self):
        assert fmt_tokens(1000) == "1.0k"

    def test_large(self):
        assert fmt_tokens(1234567) == "1234.6k"


# ── gather_status tests ──


class TestGatherStatus:
    """Tests for gather_status()."""

    def _make_agent(self, **kwargs):
        return MockAgent(**kwargs)

    def test_basic_idle(self):
        agent = self._make_agent()
        sd = gather_status(agent, "test-provider", "test-model", time.time())
        assert sd.agent_status == "idle"
        assert sd.provider == "test-provider"
        assert sd.model == "test-model"
        assert sd.message_count == 0
        assert sd.queue_count == 0

    def test_pausing_status(self):
        agent = self._make_agent(is_pausing=True)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.agent_status == "pausing"

    def test_paused_status(self):
        agent = self._make_agent(is_paused=True)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.agent_status == "paused"

    def test_queued_status(self):
        agent = self._make_agent(pending=3)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.agent_status == "queued"
        assert sd.queue_count == 3

    def test_message_count(self):
        agent = self._make_agent(messages=["a", "b", "c"])
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.message_count == 3

    def test_run_time(self):
        start = time.time() - 125  # 2m 5s ago
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", start)
        assert "2m" in sd.run_time
        assert "s" in sd.run_time

    def test_cwd(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.cwd == os.getcwd()

    def test_tracker_tokens(self):
        tracker = MockTracker(prompt_tokens=5000, completion_tokens=3000)
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time(), tracker=tracker)
        assert sd.prompt_tokens == 5000
        assert sd.completion_tokens == 3000
        assert sd.prompt_tokens_fmt == "5.0k"
        assert sd.completion_tokens_fmt == "3.0k"
        assert sd.total_tokens_fmt == "8.0k"

    def test_tracker_turns(self):
        tracker = MockTracker(turn_count=5, total_processing_time=120.0)
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time(), tracker=tracker)
        assert sd.turn_count == 5
        assert sd.total_processing == "2m 0s"

    def test_no_tracker(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.prompt_tokens == 0
        assert sd.completion_tokens == 0
        assert sd.turn_count == 0

    def test_mcp_connected(self):
        mcp = MockMCP(connected=True)
        agent = self._make_agent(mcp=mcp)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.mcp_status == "connected"

    def test_mcp_off(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.mcp_status == "off"

    def test_mcp_disconnected(self):
        mcp = MockMCP(connected=False)
        agent = self._make_agent(mcp=mcp)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.mcp_status == "off"

    def test_tool_stats(self):
        stats = MockToolStats(successes=45, calls=52)
        agent = self._make_agent(tool_stats=stats)
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.tool_successes == 45
        assert sd.tool_calls == 52

    def test_tool_stats_zero(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.tool_successes == 0
        assert sd.tool_calls == 0

    def test_settings_defaults(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.journal_mode is False
        assert sd.devel_mode is False
        assert sd.skills_mode is False
        assert sd.remove_reasoning is False

    def test_settings_enabled(self):
        agent = self._make_agent(
            journal_mode=True, devel_mode=True,
            skills_mode=True, remove_reasoning=True,
        )
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.journal_mode is True
        assert sd.devel_mode is True
        assert sd.skills_mode is True
        assert sd.remove_reasoning is True

    def test_active_prompt(self):
        pm = MockPromptManager(active="coding")
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time(), prompt_manager=pm)
        assert sd.active_prompt == "coding"

    def test_no_prompt_manager(self):
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.active_prompt == ""

    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_sandbox_mode(self, mock_sandbox):
        from agent13.sandbox import SandboxMode
        mock_sandbox.return_value = SandboxMode.OFF
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.sandbox_mode == "off"

    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_sandbox_mode_error(self, mock_sandbox):
        mock_sandbox.side_effect = RuntimeError("no sandbox")
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.sandbox_mode == "unknown"

    def test_tui_fields_default_none(self):
        """TUI-only optional fields default to None."""
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        assert sd.pretty is None
        assert sd.tool_response_format is None
        assert sd.spinner_speed is None
        assert sd.clipboard_method is None
        assert sd.saves_auto_count is None
        assert sd.resumed_turn_count == 0

    def test_tui_fields_can_be_set(self):
        """TUI can populate its optional fields after gather."""
        agent = self._make_agent()
        sd = gather_status(agent, "p", "m", time.time())
        sd.pretty = True
        sd.tool_response_format = "auto"
        sd.resumed_turn_count = 5
        assert sd.pretty is True
        assert sd.resumed_turn_count == 5


class TestStatusData:
    """Tests for StatusData dataclass."""

    def test_default_construction(self):
        sd = StatusData(agent_status="idle", run_time="0s", cwd="/tmp")
        assert sd.agent_status == "idle"
        assert sd.run_time == "0s"
        assert sd.cwd == "/tmp"
        assert sd.provider == ""
        assert sd.model == ""
        assert sd.prompt_tokens == 0
        assert sd.completion_tokens == 0

    def test_full_construction(self):
        sd = StatusData(
            agent_status="processing",
            run_time="5m 32s",
            cwd="/home/user",
            provider="openrouter",
            model="mistral/devstral2",
            active_prompt="coding",
            prompt_tokens=12300,
            completion_tokens=8100,
            prompt_tokens_fmt="12.3k",
            completion_tokens_fmt="8.1k",
            total_tokens_fmt="20.4k",
            queue_count=2,
            message_count=15,
            mcp_status="connected",
            tool_successes=45,
            tool_calls=52,
            sandbox_mode="off",
            journal_mode=True,
        )
        assert sd.provider == "openrouter"
        assert sd.mcp_status == "connected"
        assert sd.journal_mode is True
        assert sd.devel_mode is False  # default

# -- Tests for toggle_enum --


from enum import Enum


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestToggleEnum:
    """Tests for toggle_enum utility."""

    def test_cycles_to_next(self):
        """toggle_enum returns the next enum member."""
        assert toggle_enum(Color, Color.RED) == Color.GREEN

    def test_cycles_from_last_to_first(self):
        """toggle_enum wraps from last to first."""
        assert toggle_enum(Color, Color.BLUE) == Color.RED

    def test_cycles_middle(self):
        """toggle_enum works from middle member."""
        assert toggle_enum(Color, Color.GREEN) == Color.BLUE

    def test_single_member_enum(self):
        """toggle_enum works with single-member enum."""

        class One(Enum):
            ONLY = "only"

        assert toggle_enum(One, One.ONLY) == One.ONLY


# -- Tests for get_tool_stats_summary --


class MockToolStatsWithDict:
    """Mock tool stats with calls/successes dicts."""

    def __init__(self, calls=None, successes=None):
        self.calls = calls or {}
        self.successes = successes or {}

    @property
    def total_calls(self):
        return sum(self.calls.values())

    @property
    def total_successes(self):
        return sum(self.successes.values())


class MockAgentForTools:
    """Minimal agent mock for tool stats testing."""

    def __init__(self, tool_stats):
        self.tool_stats = tool_stats


class TestGetToolStatsSummary:
    """Tests for get_tool_stats_summary."""

    def test_empty_stats(self):
        """Empty stats returns zero counts."""
        agent = MockAgentForTools(MockToolStatsWithDict())
        result = get_tool_stats_summary(agent)
        assert result["total_calls"] == 0
        assert result["total_successes"] == 0
        assert result["total_failures"] == 0
        assert result["success_rate"] == 0.0
        assert result["per_tool"] == []

    def test_with_calls(self):
        """Stats with calls returns correct counts."""
        stats = MockToolStatsWithDict(
            calls={"read_file": 5, "edit_file": 3},
            successes={"read_file": 4, "edit_file": 2},
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)
        assert result["total_calls"] == 8
        assert result["total_successes"] == 6
        assert result["total_failures"] == 2
        assert result["success_rate"] == 75.0

    def test_per_tool_sorted_by_calls(self):
        """Per-tool list is sorted by call count descending."""
        stats = MockToolStatsWithDict(
            calls={"edit_file": 3, "read_file": 10, "command": 7},
            successes={"edit_file": 2, "read_file": 8, "command": 5},
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)
        names = [t["name"] for t in result["per_tool"]]
        assert names == ["read_file", "command", "edit_file"]

    def test_per_tool_includes_successes(self):
        """Each per-tool entry includes successes count."""
        stats = MockToolStatsWithDict(
            calls={"read_file": 5},
            successes={"read_file": 3},
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)
        assert result["per_tool"][0]["successes"] == 3

    def test_success_rate_100_percent(self):
        """All calls successful gives 100% rate."""
        stats = MockToolStatsWithDict(
            calls={"read_file": 10},
            successes={"read_file": 10},
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)
        assert result["success_rate"] == 100.0
