"""Tests for agent13/status.py — shared status data gathering."""

import os
import time
from enum import Enum
from types import SimpleNamespace
from unittest import mock

import pytest

from agent13 import Agent
from agent13.core import AgentStatus, PauseState, ToolStats
from agent13.queue import AgentQueue
from agent13.status import (
    CTX_CRIT_RATIO,
    CTX_WARN_RATIO,
    StatusData,
    StatusSource,
    build_status,
    ctx_alert_style,
    format_duration,
    fmt_tokens,
    gather_status,
    parse_token_count,
    toggle_enum,
    get_tool_stats_summary,
)
from agent13.timing import TokenTimingTracker


# ── Test doubles ──
#
# There is deliberately no fake Agent here. ``StatusSource.from_agent`` is the
# ONE place that reads the Agent's shape, so its tests use a REAL one: a fake
# would keep passing after the Agent renamed an attribute, which is exactly the
# drift this file exists to catch. Construction is ~26 ms and touches no network.
#
# The doubles below stand in for COLLABORATORS (LLM client, MCP manager, prompt
# manager), never for the Agent.


class _Client:
    """Stands in for the LLM client - Agent construction never touches it."""


class _MCPStub:
    """Only ``is_connected`` is read from the MCP manager."""

    def __init__(self, connected: bool):
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected


class _PromptManagerStub:
    """Only ``active_prompt`` is read from the prompt manager."""

    def __init__(self, active: str):
        self.active_prompt = active


def make_agent(**attrs):
    """A real Agent with ``attrs`` set on it - the subject of the from_agent tests.

    A name the Agent does not have raises instead of becoming a new attribute,
    so a typo in a test fails here rather than silently asserting a default.
    """
    agent = Agent(_Client(), model="test-model")
    for name, value in attrs.items():
        if not hasattr(agent, name):
            raise AttributeError(f"Agent has no attribute {name!r}")
        setattr(agent, name, value)
    return agent


def minimal_agent():
    """The smallest object ``from_agent`` accepts.

    Everything it reads WITHOUT a getattr guard is here; the guarded ones
    (chain_count, session_started_at, focus name, tracker, mcp, prompt manager)
    are absent, which is the only way to reach their defaults - a real Agent
    always has them.
    """
    return SimpleNamespace(
        is_pausing=False,
        is_paused=False,
        status=AgentStatus.IDLE,
        queue=AgentQueue(),
        mcp=None,
        messages=[],
        tool_stats=ToolStats(),
        journal_mode=False,
        devel_mode=False,
        skills_mode=False,
        remove_reasoning=False,
        auto_context_threshold=220000,
        auto_context_action="report_and_compact",
        auto_context_chain=3,
        auto_context_chain_used=0,
    )


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


# ── ctx_alert_style tests ────────────────────────────────────────────────────


class TestCtxAlertStyle:
    """Tests for ctx_alert_style() - the status bar's context-pressure bands.

    The bands are fractions of the auto-context threshold: warn at 80%, crit at
    90%. The boundary itself is IN the band (>=), so a value sitting exactly on
    the threshold is crit, not warn.

    The four colours are config values ([ui] ctx_*_fg / ctx_*_bg). "" leaves a
    channel alone, which is what makes the shipped default "coloured text on
    the bar's own background" and the alternative "black on red" a solid block.
    """

    THRESHOLD = 200000

    def test_below_warn_is_uncoloured(self):
        assert ctx_alert_style(100000, self.THRESHOLD) is None

    def test_just_below_warn(self):
        warn = int(self.THRESHOLD * CTX_WARN_RATIO)
        assert ctx_alert_style(warn - 1, self.THRESHOLD) is None

    def test_at_warn_is_yellow(self):
        warn = int(self.THRESHOLD * CTX_WARN_RATIO)
        assert ctx_alert_style(warn, self.THRESHOLD) == "yellow"

    def test_between_bands_is_yellow(self):
        assert ctx_alert_style(170000, self.THRESHOLD) == "yellow"

    def test_just_below_crit_is_yellow(self):
        crit = int(self.THRESHOLD * CTX_CRIT_RATIO)
        assert ctx_alert_style(crit - 1, self.THRESHOLD) == "yellow"

    def test_at_crit_is_red(self):
        crit = int(self.THRESHOLD * CTX_CRIT_RATIO)
        assert ctx_alert_style(crit, self.THRESHOLD) == "red"

    def test_over_threshold_is_red(self):
        assert ctx_alert_style(250000, self.THRESHOLD) == "red"

    def test_disabled_threshold_is_uncoloured(self):
        """threshold 0 means auto-context is off: nothing to be close to."""
        assert ctx_alert_style(250000, 0) is None

    def test_negative_threshold_is_uncoloured(self):
        assert ctx_alert_style(250000, -1) is None

    def test_zero_value_is_uncoloured(self):
        assert ctx_alert_style(0, self.THRESHOLD) is None

    def test_bool_threshold_is_rejected(self):
        """isinstance(True, int) is True - a bool must not be read as 1."""
        assert ctx_alert_style(250000, True) is None

    def test_bool_value_is_rejected(self):
        assert ctx_alert_style(True, self.THRESHOLD) is None

    def test_non_int_arguments_are_rejected(self):
        """A stub agent's MagicMock threshold must not raise."""
        from unittest.mock import MagicMock

        assert ctx_alert_style(250000, MagicMock()) is None
        assert ctx_alert_style(MagicMock(), self.THRESHOLD) is None

    # ── the four configurable colours ────────────────────────────────────

    def test_default_is_foreground_only(self):
        """The shipped default: coloured text, no background."""
        assert ctx_alert_style(170000, self.THRESHOLD) == "yellow"
        assert ctx_alert_style(190000, self.THRESHOLD) == "red"

    def test_solid_block_style(self):
        """black on yellow / black on red - the documented alternative."""
        kwargs = dict(warn_fg="black", warn_bg="yellow", crit_fg="black", crit_bg="red")
        assert ctx_alert_style(170000, self.THRESHOLD, **kwargs) == "black on yellow"
        assert ctx_alert_style(190000, self.THRESHOLD, **kwargs) == "black on red"

    def test_background_only(self):
        """An empty fg keeps the bar's text colour and paints only the bg."""
        assert (
            ctx_alert_style(190000, self.THRESHOLD, crit_fg="", crit_bg="red")
            == "on red"
        )

    def test_all_channels_empty_means_no_style(self):
        """A user who blanks all four gets an unstyled segment, not "[None]"."""
        assert (
            ctx_alert_style(
                190000,
                self.THRESHOLD,
                warn_fg="",
                warn_bg="",
                crit_fg="",
                crit_bg="",
            )
            is None
        )

    def test_bands_use_their_own_pair(self):
        """The warn pair must not leak into the crit band (or vice versa)."""
        assert (
            ctx_alert_style(
                170000, self.THRESHOLD, warn_fg="cyan", crit_fg="magenta"
            )
            == "cyan"
        )
        assert (
            ctx_alert_style(
                190000, self.THRESHOLD, warn_fg="cyan", crit_fg="magenta"
            )
            == "magenta"
        )

    def test_whitespace_is_trimmed(self):
        """TOML values are hand-edited; ' red ' must not produce ' red  on  x'."""
        assert (
            ctx_alert_style(190000, self.THRESHOLD, crit_fg=" red ", crit_bg=" black ")
            == "red on black"
        )


# ── parse_token_count tests ──────────────────────────────────────────────────


class TestParseTokenCount:
    """Tests for parse_token_count() - the k-suffix inverse of fmt_tokens.

    Shared by the --auto-context-threshold CLI flag and the
    /auto_context_threshold slash command (TUI + REPL).
    """

    def test_plain_number(self):
        assert parse_token_count("150000") == 150000

    def test_k_suffix(self):
        assert parse_token_count("150k") == 150000

    def test_k_suffix_uppercase(self):
        assert parse_token_count("150K") == 150000

    def test_zero(self):
        assert parse_token_count("0") == 0

    def test_whitespace_tolerated(self):
        assert parse_token_count(" 200k ") == 200000

    @pytest.mark.parametrize("bad", ["", "k", "1.5k", "abc", "15kk", "-5"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            parse_token_count(bad)


# ── status tests ──


class TestBuildStatus:
    """build_status takes a plain dataclass: no Agent, no mock, no globals.

    This is the seam the split bought. These tests construct the two or three
    fields they care about instead of faking an object with 24 attributes and
    two private tracker fields.
    """

    def test_formats_a_source_with_no_agent(self):
        sd = build_status(
            StatusSource(provider="p", model="m", prompt_tokens=1500, completion_tokens=500)
        )
        assert sd.provider == "p"
        assert sd.model == "m"
        assert sd.agent_status == "idle"
        assert sd.prompt_tokens_fmt == "1.5k"
        assert sd.total_tokens_fmt == "2.0k"
        assert sd.mcp_status == "off"

    def test_mcp_connected(self):
        assert build_status(StatusSource(mcp_connected=True)).mcp_status == "connected"

    def test_total_processing_is_blank_at_zero(self):
        assert build_status(StatusSource()).total_processing == "0s"
        assert build_status(StatusSource(total_processing_time=65)).total_processing == "1m 5s"

    def test_goal_line_is_carried_through(self):
        sd = build_status(StatusSource(goal_line="Grind: 2h left"))
        assert sd.goal_line == "Grind: 2h left"

    def test_resumed_fields_are_left_at_their_defaults(self):
        """Neither half fills the resumed-session fields - the /status handler does."""
        sd = build_status(StatusSource(turn_count=9))
        assert sd.resumed_turn_count == 0
        assert sd.resumed_saved_at is None

    def test_the_adapter_is_exactly_the_two_halves(self):
        agent = make_agent()
        start = time.time()
        assert build_status(
            StatusSource.from_agent(agent, "p", "m", start)
        ) == gather_status(agent, "p", "m", start)

    # -- carried through unchanged -------------------------------------------
    # Every other field is a straight copy. run_time, cwd and total_processing
    # are computed, so they keep their own tests above.

    def test_status_words_are_carried_through(self):
        for word in ("idle", "pausing", "paused", "queued", "processing"):
            assert build_status(StatusSource(agent_status=word)).agent_status == word

    def test_counts_are_carried_through(self):
        sd = build_status(StatusSource(message_count=3, queue_count=2))
        assert (sd.message_count, sd.queue_count) == (3, 2)

    def test_run_time_is_measured_from_the_session_start(self):
        sd = build_status(StatusSource(session_start_time=time.time() - 125))
        assert "2m" in sd.run_time

    def test_cwd_is_the_process_cwd(self):
        assert build_status(StatusSource()).cwd == os.getcwd()

    def test_focus_chain_and_session_start_are_carried_through(self):
        sd = build_status(
            StatusSource(
                focus_name="auto_context",
                chain_count=20,
                session_started_at="2026-09-21T14:32:00",
            )
        )
        assert sd.focus_name == "auto_context"
        assert sd.chain_count == 20
        assert sd.session_started_at == "2026-09-21T14:32:00"

    def test_tool_counts_are_carried_through(self):
        sd = build_status(StatusSource(tool_successes=45, tool_calls=52, bounded_count=3))
        assert (sd.tool_successes, sd.tool_calls, sd.bounded_count) == (45, 52, 3)

    def test_settings_are_carried_through(self):
        sd = build_status(
            StatusSource(
                journal_mode=True,
                devel_mode=True,
                skills_mode=True,
                remove_reasoning=True,
            )
        )
        assert sd.journal_mode is True
        assert sd.devel_mode is True
        assert sd.skills_mode is True
        assert sd.remove_reasoning is True

    def test_auto_context_settings_are_carried_through(self):
        sd = build_status(
            StatusSource(
                auto_context_threshold=150000,
                auto_context_action="compact",
                auto_context_chain=2,
                auto_context_chain_used=1,
            )
        )
        assert sd.auto_context_threshold == 150000
        assert sd.auto_context_action == "compact"
        assert sd.auto_context_chain == 2
        assert sd.auto_context_chain_used == 1

    def test_active_prompt_is_carried_through(self):
        assert build_status(StatusSource(active_prompt="coding")).active_prompt == "coding"

    def test_sandbox_and_pins_are_carried_through(self):
        sd = build_status(
            StatusSource(
                sandbox_mode="off",
                sandbox_pinned="permissive-closed",
                devel_pinned=True,
            )
        )
        assert sd.sandbox_mode == "off"
        assert sd.sandbox_pinned == "permissive-closed"
        assert sd.devel_pinned is True


class TestFromAgent:
    """The adapter half: what ``from_agent`` reads off a REAL Agent.

    A fake Agent would pass even if the Agent renamed an attribute, so these
    tests take the real thing. ``_read`` is how a front-end calls it: provider
    and model only, everything else from the agent.
    """

    @staticmethod
    def _read(agent, **kwargs):
        return StatusSource.from_agent(agent, "p", "m", **kwargs)

    def test_provider_and_model_come_from_the_caller(self):
        source = self._read(make_agent())
        assert (source.provider, source.model) == ("p", "m")

    def test_provider_and_model_fall_back_to_the_agent(self):
        agent = make_agent()
        agent.provider_name = "from-agent"
        source = StatusSource.from_agent(agent)
        assert (source.provider, source.model) == ("from-agent", "test-model")

    def test_status_word_is_the_enum_lowercased(self):
        agent = make_agent()
        agent._status = AgentStatus.IDLE
        assert self._read(agent).agent_status == "idle"

    def test_pausing_wins_over_paused_and_queued(self):
        agent = make_agent()
        agent._pause_state = PauseState.PAUSING
        agent.queue.add("queued too")
        assert self._read(agent).agent_status == "pausing"

    def test_paused(self):
        agent = make_agent()
        agent._pause_state = PauseState.PAUSED
        assert self._read(agent).agent_status == "paused"

    def test_queued_when_the_queue_has_items(self):
        agent = make_agent()
        for text in ("a", "b", "c"):
            agent.queue.add(text)
        source = self._read(agent)
        assert source.agent_status == "queued"
        assert source.queue_count == 3

    def test_message_count_is_the_message_list_length(self):
        assert self._read(make_agent(messages=["a", "b", "c"])).message_count == 3

    def test_session_start_time_is_read(self):
        start = time.time() - 125
        assert self._read(make_agent(session_start_time=start)).session_start_time == start

    def test_session_start_time_falls_back_to_now(self):
        """0.0 is the Agent's own default - a front-end that never set it still gets a run time."""
        source = self._read(make_agent(session_start_time=0.0))
        assert source.session_start_time == pytest.approx(time.time(), abs=5)

    def test_focus_name_is_read(self):
        agent = make_agent()
        agent.apply_focus_name("auto_context")
        assert self._read(agent).focus_name == "auto_context"

    def test_focus_name_defaults_to_empty(self):
        assert self._read(make_agent()).focus_name == ""

    def test_chain_count_and_session_start_are_read(self):
        agent = make_agent()
        agent._chain_count = 20
        agent._session_started_at = "2026-09-21T14:32:00"
        source = self._read(agent)
        assert source.chain_count == 20
        assert source.session_started_at == "2026-09-21T14:32:00"

    def test_guarded_attributes_default_when_absent(self):
        source = self._read(minimal_agent())
        assert source.chain_count == 0
        assert source.session_started_at == ""
        assert source.focus_name == ""

    def test_goal_line_is_empty_without_a_window(self):
        assert self._read(make_agent()).goal_line == ""

    def test_tracker_tokens(self):
        tracker = TokenTimingTracker()
        tracker.prompt_tokens = 5000
        tracker.completion_tokens = 3000
        source = self._read(make_agent(), tracker=tracker)
        assert (source.prompt_tokens, source.completion_tokens) == (5000, 3000)

    def test_tracker_turns(self):
        tracker = TokenTimingTracker()
        tracker._turn_count = 5
        tracker._total_processing_time = 120.0
        source = self._read(make_agent(), tracker=tracker)
        assert source.turn_count == 5
        assert source.total_processing_time == 120.0

    def test_no_tracker(self):
        source = self._read(make_agent(tracker=None))
        assert source.prompt_tokens == 0
        assert source.completion_tokens == 0
        assert source.turn_count == 0

    def test_mcp_connected(self):
        agent = make_agent()
        agent._mcp = _MCPStub(connected=True)
        assert self._read(agent).mcp_connected is True

    def test_mcp_off_when_the_agent_has_no_manager(self):
        assert self._read(make_agent()).mcp_connected is False

    def test_mcp_disconnected(self):
        agent = make_agent()
        agent._mcp = _MCPStub(connected=False)
        assert self._read(agent).mcp_connected is False

    def test_tool_stats(self):
        agent = make_agent()
        agent.tool_stats.calls = {"t": 52}
        agent.tool_stats.successes = {"t": 45}
        agent.tool_stats.bounded = 3
        source = self._read(agent)
        assert source.tool_successes == 45
        assert source.tool_calls == 52
        assert source.bounded_count == 3

    def test_tool_stats_zero(self):
        source = self._read(make_agent())
        assert source.tool_successes == 0
        assert source.tool_calls == 0
        assert source.bounded_count == 0

    def test_settings_defaults(self):
        source = self._read(make_agent())
        assert source.journal_mode is False
        assert source.devel_mode is False
        assert source.skills_mode is False
        assert source.remove_reasoning is False

    def test_settings_enabled(self):
        source = self._read(
            make_agent(
                journal_mode=True,
                _devel_mode=True,
                _skills_mode=True,
                remove_reasoning=True,
            )
        )
        assert source.journal_mode is True
        assert source.devel_mode is True
        assert source.skills_mode is True
        assert source.remove_reasoning is True

    def test_auto_context_settings(self):
        source = self._read(
            make_agent(
                auto_context_threshold=150000,
                auto_context_action="compact",
                auto_context_chain=2,
                _auto_context_chain_used=1,
            )
        )
        assert source.auto_context_threshold == 150000
        assert source.auto_context_action == "compact"
        assert source.auto_context_chain == 2
        assert source.auto_context_chain_used == 1

    def test_active_prompt(self):
        source = self._read(make_agent(), prompt_manager=_PromptManagerStub("coding"))
        assert source.active_prompt == "coding"

    def test_no_prompt_manager(self):
        assert self._read(make_agent(prompt_manager=None)).active_prompt == ""

    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_sandbox_mode(self, mock_sandbox):
        from agent13.sandbox import SandboxMode

        mock_sandbox.return_value = SandboxMode.OFF
        assert self._read(make_agent()).sandbox_mode == "off"

    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_sandbox_mode_error(self, mock_sandbox):
        mock_sandbox.side_effect = RuntimeError("no sandbox")
        assert self._read(make_agent()).sandbox_mode == "unknown"

    @mock.patch("agent13.pins.get_pinned_devel")
    @mock.patch("agent13.sandbox.get_pinned_sandbox_mode")
    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_pins_populated(self, mock_sandbox, mock_sandbox_pin, mock_devel_pin):
        from agent13.sandbox import SandboxMode

        mock_sandbox.return_value = SandboxMode.OFF
        mock_sandbox_pin.return_value = SandboxMode.PERMISSIVE_CLOSED
        mock_devel_pin.return_value = True
        source = self._read(make_agent())
        assert source.sandbox_pinned == "permissive-closed"
        assert source.devel_pinned is True

    @mock.patch("agent13.pins.get_pinned_devel")
    @mock.patch("agent13.sandbox.get_pinned_sandbox_mode")
    @mock.patch("tools.security.get_current_sandbox_mode")
    def test_pins_none_when_unpinned(self, mock_sandbox, mock_sandbox_pin, mock_devel_pin):
        from agent13.sandbox import SandboxMode

        mock_sandbox.return_value = SandboxMode.OFF
        mock_sandbox_pin.return_value = None
        mock_devel_pin.return_value = None
        source = self._read(make_agent())
        assert source.sandbox_pinned is None
        assert source.devel_pinned is None

    def test_resumed_fields_default_to_zero(self):
        """gather_status leaves the resumed-session fields alone."""
        sd = gather_status(make_agent(), "p", "m", time.time())
        assert sd.resumed_turn_count == 0
        assert sd.resumed_saved_at is None
        assert sd.resumed_prompt_tokens == 0
        assert sd.resumed_completion_tokens == 0


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

    def test_resumed_fields_can_be_set(self):
        """The /status handler fills them from the agent after gather."""
        sd = StatusData(agent_status="idle", run_time="0s", cwd="/tmp")
        sd.resumed_turn_count = 5
        sd.resumed_saved_at = "2026-10-26T12:00:00"
        assert sd.resumed_turn_count == 5
        assert sd.resumed_saved_at == "2026-10-26T12:00:00"

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
            sandbox_pinned="off",
            journal_mode=True,
            devel_pinned=False,
        )
        assert sd.provider == "openrouter"
        assert sd.mcp_status == "connected"
        assert sd.journal_mode is True
        assert sd.devel_mode is False  # default
        assert sd.sandbox_pinned == "off"
        assert sd.devel_pinned is False  # pinned off, not "no pin"


# -- Tests for toggle_enum --


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
    """Mock tool stats with calls/successes dicts.

    Mirrors the real ToolStats fields the summary reads - the mode and
    param-combo breakdowns included, so a missing attribute fails here.
    """

    def __init__(
        self,
        calls=None,
        successes=None,
        modes=None,
        mode_successes=None,
        param_combos=None,
        param_combo_successes=None,
    ):
        self.calls = calls or {}
        self.successes = successes or {}
        self.modes = modes or {}
        self.mode_successes = mode_successes or {}
        self.param_combos = param_combos or {}
        self.param_combo_successes = param_combo_successes or {}

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

    def test_per_tool_mode_breakdown(self):
        """Each per-tool entry carries its mode breakdown, sorted by mode."""
        stats = MockToolStatsWithDict(
            calls={"read_file": 5},
            successes={"read_file": 4},
            modes={"read_file": {"raw": 3, "lines": 2}},
            mode_successes={"read_file": {"raw": 2, "lines": 2}},
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)

        assert result["per_tool"][0]["modes"] == [
            {"mode": "lines", "calls": 2, "successes": 2},
            {"mode": "raw", "calls": 3, "successes": 2},
        ]

    def test_per_tool_param_combos_sorted_by_calls(self):
        """Param combos come back most-used first."""
        stats = MockToolStatsWithDict(
            calls={"edit_file": 6},
            successes={"edit_file": 5},
            param_combos={"edit_file": {"filepath,find": 2, "filepath,find,content": 4}},
            param_combo_successes={
                "edit_file": {"filepath,find": 1, "filepath,find,content": 4}
            },
        )
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)

        assert result["per_tool"][0]["param_combos"] == [
            {"combo": "filepath,find,content", "calls": 4, "successes": 4},
            {"combo": "filepath,find", "calls": 2, "successes": 1},
        ]

    def test_per_tool_without_breakdowns_gets_empty_lists(self):
        """A tool with neither modes nor combos gets empty lists, not keys."""
        stats = MockToolStatsWithDict(calls={"command": 1}, successes={"command": 1})
        agent = MockAgentForTools(stats)
        result = get_tool_stats_summary(agent)

        assert result["per_tool"][0]["modes"] == []
        assert result["per_tool"][0]["param_combos"] == []
