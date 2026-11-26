"""Shared status data gathering for /status command.

Single source of truth for what /status knows about. Both TUI and REPL
consume the same StatusData, format it differently:
- TUI: Rich markup in info pane
- REPL: plain text to stdout

Design follows the same pattern as models.py's resolve_from_list():
shared data logic, thin UI wrappers.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string.

    >>> format_duration(65)
    '1m 5s'
    >>> format_duration(3661)
    '1h 1m 1s'
    >>> format_duration(42)
    '42s'
    """
    secs = int(seconds)
    hours, remainder = divmod(secs, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"


def fmt_tokens(n: int) -> str:
    """Format token count with k suffix for large numbers.

    >>> fmt_tokens(12345)
    '12.3k'
    >>> fmt_tokens(42)
    '42'
    """
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def parse_token_count(s: str) -> int:
    """Parse a token count that may carry a k suffix ("150k", "150K").

    The inverse of fmt_tokens' display form: plain integers ("150000") and
    k-suffixed values ("150k") both work. Raises ValueError for anything
    else (empty string, bare "k", "1.5k", negatives) - callers show their
    own usage message.

    >>> parse_token_count("150k")
    150000
    >>> parse_token_count("150000")
    150000
    >>> parse_token_count("0")
    0
    """
    val = s.strip().lower()
    if val.endswith("k"):
        count = int(val[:-1]) * 1000
    else:
        count = int(val)
    if count < 0:
        raise ValueError(f"negative token count: {s!r}")
    return count


# Context-pressure bands, as a fraction of the auto-context threshold. The
# status bar's Ctx segment is styled at the first and the second, so the
# wrap-up is visible before it fires rather than as a surprise.
CTX_WARN_RATIO = 0.80
CTX_CRIT_RATIO = 0.90

# Defaults match the shipped config: coloured text on the bar's own background.
# A user wanting a solid block sets fg="black" and a bg (see [ui] in
# default_config.toml).
CTX_WARN_FG = "yellow"
CTX_WARN_BG = ""
CTX_CRIT_FG = "red"
CTX_CRIT_BG = ""


def _colour_style(fg: str, bg: str) -> Optional[str]:
    """Compose a Textual markup style from a foreground/background pair.

    "" means "leave that channel alone", so the pair (fg, "") is plain text
    colour and ("", bg) is a background block in the bar's own text colour.
    Both empty means no style at all - the segment renders as the rest of the
    bar does.
    """
    fg = (fg or "").strip()
    bg = (bg or "").strip()
    if fg and bg:
        return f"{fg} on {bg}"
    if fg:
        return fg
    if bg:
        return f"on {bg}"
    return None


def ctx_alert_style(
    value: int,
    threshold: int,
    warn_fg: str = CTX_WARN_FG,
    warn_bg: str = CTX_WARN_BG,
    crit_fg: str = CTX_CRIT_FG,
    crit_bg: str = CTX_CRIT_BG,
) -> Optional[str]:
    """Markup style for the status bar's context segment, or None for default.

    ``value`` and ``threshold`` are raw token counts: ``value`` is whatever the
    bar is displaying (the live estimate while the count is estimated, the
    grounded prompt-token count otherwise) and ``threshold`` is
    ``agent.auto_context_threshold``. Styling the displayed number is
    deliberate - a style derived from a different number than the digits on
    screen would contradict them.

    ``threshold <= 0`` means auto-context is disabled: there is nothing to be
    close to, so no style. Non-int arguments (a bool, or a stub agent's
    MagicMock) also yield None rather than a TypeError - ``isinstance(True,
    int)`` is True, so bools are rejected explicitly.

    >>> ctx_alert_style(150000, 220000)
    >>> ctx_alert_style(176000, 220000)
    'yellow'
    >>> ctx_alert_style(198000, 220000)
    'red'
    >>> ctx_alert_style(198000, 220000, "black", "red")
    'black on red'
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if threshold <= 0 or value <= 0:
        return None
    ratio = value / threshold
    if ratio >= CTX_CRIT_RATIO:
        return _colour_style(crit_fg, crit_bg)
    if ratio >= CTX_WARN_RATIO:
        return _colour_style(warn_fg, warn_bg)
    return None


@dataclass
class StatusData:
    """Structured data for /status display.

    Fields are grouped by section. The resumed-session fields are filled by
    the /status handler from the agent; gather_status leaves them at their
    defaults.
    """

    # ── Session ──
    agent_status: str  # idle, processing, queued, pausing, paused
    run_time: str  # "5m 32s"
    cwd: str
    focus_name: str = ""  # effective focus (explicit or path-derived); "" = none
    chain_count: int = 0  # session-lifetime compaction total (persists in saves)
    session_started_at: str = ""  # ISO original session start (persists in saves)
    last_turn_duration: Optional[str] = None  # "3s"
    last_turn_ago: Optional[str] = None  # "30s ago"
    turn_count: int = 0
    total_processing: str = "0s"

    # ── Provider ──
    provider: str = ""
    model: str = ""
    active_prompt: str = ""

    # ── Context ──
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_fmt: str = "0"
    completion_tokens_fmt: str = "0"
    total_tokens_fmt: str = "0"
    queue_count: int = 0
    message_count: int = 0

    # ── Connectivity ──
    mcp_status: str = "off"

    # ── Tools ──
    tool_successes: int = 0
    tool_calls: int = 0
    bounded_count: int = 0  # results truncated by tool output bounding

    # ── Settings ──
    sandbox_mode: str = "unknown"
    sandbox_pinned: Optional[str] = None  # pinned mode value, None = no pin
    journal_mode: bool = False
    devel_mode: bool = False
    devel_pinned: Optional[bool] = None  # pinned devel state, None = no pin
    skills_mode: bool = False
    remove_reasoning: bool = False
    auto_context_threshold: int = 0
    auto_context_action: str = ""
    auto_context_chain: int = 0
    auto_context_chain_used: int = 0
    nudges: int = 0  # session-lifetime degenerate-nudge total (persists in saves)
    # Goal window (/grind), user-facing only: the remaining time must never
    # reach the model (a visible budget invites "let me hand back").
    goal_line: str = ""  # "" = no window armed
    auto_save_idle_seconds: int = 0  # idle seconds before one auto-save (0 = off)

    # ── Resumed session ──
    # Filled by the shared /status handler from the agent's resumed-* fields.
    resumed_turn_count: int = 0
    resumed_saved_at: Optional[str] = None
    resumed_prompt_tokens: int = 0
    resumed_completion_tokens: int = 0


@dataclass(frozen=True)
class StatusSource:
    """Everything ``build_status`` needs, in one typed place.

    The point of the split: ``gather_status(agent)`` used to reach into ~24
    attributes of the Agent (two of them private on the tracker) and three
    module-level singletons, so its real interface was "an object shaped like
    an Agent" - which a test could only satisfy by faking all 24. The reaching
    now lives in ``from_agent``; everything downstream is a plain dataclass,
    and a test constructs the three fields it cares about.
    """

    # Session
    agent_status: str = "idle"
    session_start_time: float = 0.0
    focus_name: str = ""
    chain_count: int = 0
    session_started_at: str = ""
    turn_count: int = 0
    total_processing_time: float = 0.0
    # Provider
    provider: str = ""
    model: str = ""
    active_prompt: str = ""
    # Context
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queue_count: int = 0
    message_count: int = 0
    # Connectivity
    mcp_connected: bool = False
    # Tools
    tool_successes: int = 0
    tool_calls: int = 0
    bounded_count: int = 0
    # Settings
    sandbox_mode: str = "unknown"
    sandbox_pinned: Optional[str] = None
    journal_mode: bool = False
    devel_mode: bool = False
    devel_pinned: Optional[bool] = None
    skills_mode: bool = False
    remove_reasoning: bool = False
    auto_context_threshold: int = 0
    auto_context_action: str = ""
    auto_context_chain: int = 0
    auto_context_chain_used: int = 0
    nudges: int = 0
    auto_save_idle_seconds: int = 0
    goal_line: str = ""

    @classmethod
    def from_agent(
        cls,
        agent,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        session_start_time: Optional[float] = None,
        prompt_manager=None,
        tracker=None,
    ) -> "StatusSource":
        """Read an Agent. The ONLY place that knows the Agent's shape.

        Every argument after ``agent`` is optional and defaults to the value
        the agent carries, so a shared caller needs nothing but the agent. The
        explicit forms remain for callers that hold a value the agent does not
        (tests, and any front-end with a session-local override).
        """
        from tools.security import get_current_sandbox_mode

        if provider is None:
            provider = getattr(agent, "provider_name", "") or ""
        if model is None:
            model = getattr(agent, "model", "") or ""
        if session_start_time is None:
            session_start_time = getattr(agent, "session_start_time", 0.0) or time.time()
        if prompt_manager is None:
            prompt_manager = getattr(agent, "prompt_manager", None)
        if tracker is None:
            tracker = getattr(agent, "tracker", None)

        if agent.is_pausing:
            agent_status = "pausing"
        elif agent.is_paused:
            agent_status = "paused"
        elif agent.queue.pending_count > 0:
            agent_status = "queued"
        else:
            agent_status = agent.status.value.lower()

        turn_count = 0
        total_processing = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        if tracker is not None:
            turn_count = tracker._turn_count
            total_processing = tracker._total_processing_time
            prompt_tokens = tracker.prompt_tokens
            completion_tokens = tracker.completion_tokens

        try:
            sandbox_mode = get_current_sandbox_mode().value
        except Exception:
            sandbox_mode = "unknown"

        from agent13.pins import get_pinned_devel
        from agent13.sandbox import get_pinned_sandbox_mode

        sandbox_pinned: Optional[str] = None
        try:
            pinned = get_pinned_sandbox_mode()
            if pinned is not None:
                sandbox_pinned = pinned.value
        except Exception:
            pass

        devel_pinned: Optional[bool] = None
        try:
            devel_pinned = get_pinned_devel()
        except Exception:
            pass

        stats = agent.tool_stats
        return cls(
            agent_status=agent_status,
            session_start_time=session_start_time,
            focus_name=getattr(agent, "_session_focus_name", ""),
            chain_count=getattr(agent, "chain_count", 0),
            session_started_at=getattr(agent, "session_started_at", ""),
            turn_count=turn_count,
            total_processing_time=total_processing,
            provider=provider,
            model=model,
            active_prompt=prompt_manager.active_prompt if prompt_manager is not None else "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            queue_count=agent.queue.pending_count,
            message_count=len(agent.messages),
            mcp_connected=bool(agent.mcp and agent.mcp.is_connected()),
            tool_successes=stats.total_successes,
            tool_calls=stats.total_calls,
            bounded_count=getattr(stats, "bounded", 0),
            sandbox_mode=sandbox_mode,
            sandbox_pinned=sandbox_pinned,
            journal_mode=agent.journal_mode,
            devel_mode=agent.devel_mode,
            devel_pinned=devel_pinned,
            skills_mode=agent.skills_mode,
            remove_reasoning=agent.remove_reasoning,
            auto_context_threshold=agent.auto_context_threshold,
            auto_context_action=agent.auto_context_action,
            auto_context_chain=agent.auto_context_chain,
            auto_context_chain_used=agent.auto_context_chain_used,
            nudges=getattr(agent, "degenerate_nudges", 0),
            auto_save_idle_seconds=getattr(agent, "auto_save_idle_seconds", 0),
            goal_line=_goal_line(agent),
        )


def build_status(source: StatusSource) -> StatusData:
    """Format a StatusSource. Pure: no Agent, no globals, no clock beyond now."""
    total_processing = (
        format_duration(source.total_processing_time)
        if source.total_processing_time > 0
        else "0s"
    )
    return StatusData(
        # Session
        agent_status=source.agent_status,
        run_time=format_duration(time.time() - source.session_start_time),
        cwd=str(Path.cwd()),
        focus_name=source.focus_name,
        chain_count=source.chain_count,
        session_started_at=source.session_started_at,
        turn_count=source.turn_count,
        total_processing=total_processing,
        # Provider
        provider=source.provider,
        model=source.model,
        active_prompt=source.active_prompt,
        # Context
        prompt_tokens=source.prompt_tokens,
        completion_tokens=source.completion_tokens,
        prompt_tokens_fmt=fmt_tokens(source.prompt_tokens),
        completion_tokens_fmt=fmt_tokens(source.completion_tokens),
        total_tokens_fmt=fmt_tokens(source.prompt_tokens + source.completion_tokens),
        queue_count=source.queue_count,
        message_count=source.message_count,
        # Connectivity
        mcp_status="connected" if source.mcp_connected else "off",
        # Tools
        tool_successes=source.tool_successes,
        tool_calls=source.tool_calls,
        bounded_count=source.bounded_count,
        # Settings
        sandbox_mode=source.sandbox_mode,
        sandbox_pinned=source.sandbox_pinned,
        journal_mode=source.journal_mode,
        devel_mode=source.devel_mode,
        devel_pinned=source.devel_pinned,
        skills_mode=source.skills_mode,
        remove_reasoning=source.remove_reasoning,
        auto_context_threshold=source.auto_context_threshold,
        auto_context_action=source.auto_context_action,
        auto_context_chain=source.auto_context_chain,
        auto_context_chain_used=source.auto_context_chain_used,
        nudges=source.nudges,
        auto_save_idle_seconds=source.auto_save_idle_seconds,
        goal_line=source.goal_line,
    )


def gather_status(
    agent,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    session_start_time: Optional[float] = None,
    prompt_manager=None,
    tracker=None,
) -> StatusData:
    """Gather all core status data from the agent.

    A thin adapter: read the agent once (``StatusSource.from_agent``), then
    format (``build_status``). Kept so every existing caller and test keeps
    working; the two halves are separately testable.
    """
    return build_status(
        StatusSource.from_agent(
            agent, provider, model, session_start_time, prompt_manager, tracker
        )
    )


def _goal_line(agent) -> str:
    """The /status goal line, or "" when no window is armed.

    One home for the wording: /grind with no args composes its own summary, but
    this is what /status shows, and the two must agree about what "armed" means
    (goal_active, i.e. the deadline has not passed).
    """
    from agent13 import goal as goal_rules

    if not getattr(agent, "goal_active", False):
        return ""
    return "Grind: {text} (window {window}, {left} left)".format(
        text=getattr(agent, "_goal_text", "") or "no goal text",
        window=goal_rules.format_window(getattr(agent, "goal_window", 0.0)),
        left=goal_rules.format_window(getattr(agent, "goal_remaining", 0.0)),
    )


def toggle_enum(enum_class, current_value):
    """Cycle to the next value in an enum, wrapping to first.

    Args:
        enum_class: The enum type (e.g. SandboxMode).
        current_value: Current enum member.

    Returns:
        Next enum member, wrapping from last to first.

    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = "red"
    ...     GREEN = "green"
    ...     BLUE = "blue"
    >>> toggle_enum(Color, Color.RED)
    <Color.GREEN: 'green'>
    >>> toggle_enum(Color, Color.BLUE)
    <Color.RED: 'red'>
    """
    members = list(enum_class)
    idx = members.index(current_value)
    return members[(idx + 1) % len(members)]


def get_tool_stats_summary(agent) -> dict:
    """Get tool usage statistics from agent.

    Returns dict with keys:
        total_calls, total_successes, total_failures, success_rate,
        per_tool: list sorted by calls desc, each
            {name, calls, successes,
             modes: [{mode, calls, successes}] sorted by mode,
             param_combos: [{combo, calls, successes}] sorted by calls desc}.

    The mode and param-combo breakdowns are what /tools shows per tool; they
    are empty lists for tools that have neither.
    """
    stats = agent.tool_stats
    total = stats.total_calls
    successes = stats.total_successes
    failures = total - successes
    rate = (successes / total * 100) if total > 0 else 0.0

    per_tool = []
    for name, count in sorted(stats.calls.items(), key=lambda x: x[1], reverse=True):
        mode_successes = stats.mode_successes.get(name, {})
        modes = [
            {"mode": mode, "calls": calls, "successes": mode_successes.get(mode, 0)}
            for mode, calls in sorted(stats.modes.get(name, {}).items())
        ]
        combo_successes = stats.param_combo_successes.get(name, {})
        param_combos = [
            {"combo": combo, "calls": calls, "successes": combo_successes.get(combo, 0)}
            for combo, calls in sorted(
                stats.param_combos.get(name, {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        per_tool.append(
            {
                "name": name,
                "calls": count,
                "successes": stats.successes.get(name, 0),
                "modes": modes,
                "param_combos": param_combos,
            }
        )

    return {
        "total_calls": total,
        "total_successes": successes,
        "total_failures": failures,
        "success_rate": rate,
        "per_tool": per_tool,
    }
