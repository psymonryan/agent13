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


@dataclass
class StatusData:
    """Structured data for /status display.

    Fields are grouped by section. TUI-only fields (resumed, saves,
    display settings) are Optional and populated only by TUI.
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

    # ── TUI-only (Optional) ──
    # These fields are populated by the TUI for its richer status display.
    # REPL leaves them as None.
    pretty: Optional[bool] = None
    tool_response_format: Optional[str] = None
    spinner_speed: Optional[str] = None
    clipboard_method: Optional[str] = None

    # ── Saves (TUI-only) ──
    saves_auto_count: Optional[int] = None
    saves_manual_count: Optional[int] = None
    saves_location: Optional[str] = None
    saves_restorable: Optional[bool] = None
    saves_restorable_date: Optional[str] = None

    # ── Resumed session (TUI-only) ──
    resumed_turn_count: int = 0
    resumed_saved_at: Optional[str] = None
    resumed_prompt_tokens: int = 0
    resumed_completion_tokens: int = 0


def gather_status(
    agent,
    provider: str,
    model: str,
    session_start_time: float,
    prompt_manager=None,
    tracker=None,
) -> StatusData:
    """Gather all core status data from agent and session state.

    Args:
        agent: The Agent instance.
        provider: Provider name string.
        model: Model name string.
        session_start_time: time.time() when session started.
        prompt_manager: PromptManager instance (for active prompt name).
        tracker: TokenTimingTracker instance (for token counts and turn stats).

    Returns:
        StatusData with all core fields populated.
    """
    from tools.security import get_current_sandbox_mode

    # ── Agent status ──
    if agent.is_pausing:
        agent_status = "pausing"
    elif agent.is_paused:
        agent_status = "paused"
    elif agent.queue.pending_count > 0:
        agent_status = "queued"
    else:
        agent_status = agent.status.value.lower()

    # ── Run time ──
    run_time = format_duration(time.time() - session_start_time)

    # ── Last turn ──
    turn_count = 0
    total_processing = "0s"

    if tracker is not None:
        turn_count = tracker._turn_count
        if tracker._total_processing_time > 0:
            total_processing = format_duration(tracker._total_processing_time)

    # ── Tokens ──
    prompt_tokens = 0
    completion_tokens = 0
    if tracker is not None:
        prompt_tokens = tracker.prompt_tokens
        completion_tokens = tracker.completion_tokens

    # ── MCP ──
    mcp_status = "off"
    if agent.mcp and agent.mcp.is_connected():
        mcp_status = "connected"

    # ── Tools ──
    stats = agent.tool_stats

    # ── Active prompt ──
    active_prompt = ""
    if prompt_manager is not None:
        active_prompt = prompt_manager.active_prompt

    # ── Sandbox ──
    try:
        sandbox_mode = get_current_sandbox_mode().value
    except Exception:
        sandbox_mode = "unknown"

    # ── Pins (per-project; rendered as "(pinned: <value>)" in /status) ──
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

    return StatusData(
        # Session
        agent_status=agent_status,
        run_time=run_time,
        cwd=str(Path.cwd()),
        focus_name=getattr(agent, "_session_focus_name", ""),
        chain_count=getattr(agent, "chain_count", 0),
        session_started_at=getattr(agent, "session_started_at", ""),
        turn_count=turn_count,
        total_processing=total_processing,
        # Provider
        provider=provider,
        model=model,
        active_prompt=active_prompt,
        # Context
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_fmt=fmt_tokens(prompt_tokens),
        completion_tokens_fmt=fmt_tokens(completion_tokens),
        total_tokens_fmt=fmt_tokens(prompt_tokens + completion_tokens),
        queue_count=agent.queue.pending_count,
        message_count=len(agent.messages),
        # Connectivity
        mcp_status=mcp_status,
        # Tools
        tool_successes=stats.total_successes,
        tool_calls=stats.total_calls,
        bounded_count=getattr(stats, "bounded", 0),
        # Settings
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
        per_tool: list of {name, calls, successes} sorted by calls desc.
    """
    stats = agent.tool_stats
    total = stats.total_calls
    successes = stats.total_successes
    failures = total - successes
    rate = (successes / total * 100) if total > 0 else 0.0

    per_tool = []
    for name, count in sorted(stats.calls.items(), key=lambda x: x[1], reverse=True):
        per_tool.append(
            {
                "name": name,
                "calls": count,
                "successes": stats.successes.get(name, 0),
            }
        )

    return {
        "total_calls": total,
        "total_successes": successes,
        "total_failures": failures,
        "success_rate": rate,
        "per_tool": per_tool,
    }
