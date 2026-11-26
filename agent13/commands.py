"""Shared command validation and execution for REPL and TUI.

Each function handles the logic of a slash command and returns a CommandResult.
UIs call these functions and render the result in their own format.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import io
import json
import os

_DELETE_USAGE = (
    "Usage: /delete h N | /delete q N | /delete s NAME\n"
    "  N can be: number, 'last', negative index (-1), or range (1:3, -2:)"
)

_POLITE_USAGE = (
    "Usage: /polite N | /polite off\n"
    "  N    - poll interval in seconds (pseudo-priority; lower = more aggressive)\n"
    "  off  - disable polite mode"
)


@dataclass
class CommandResult:
    """Result of a shared command operation.

    The message is plain text - each front-end renders it in its own format.
    Two optional hints cover the only UI side-effects a shared command needs:

    refresh_status: the agent/config state changed, re-read the status bar.
    pane: keep the info pane open in this mode ("history", "queue", ...).
    """

    success: bool
    message: str  # Plain text, no Rich markup
    data: dict[str, Any] = field(default_factory=dict)
    refresh_status: bool = False
    pane: str | None = None


def execute_save(agent, args: str) -> CommandResult:
    """Execute /save command.

    Parses args, validates name, checks overwrite, saves context.
    Returns CommandResult with message count in data['msg_count'] on success.
    """
    from agent13.persistence import save_context, resolve_save_path

    args = args.strip()
    if not args:
        return CommandResult(
            False,
            "Usage: /save <name> [-y]\n"
            "  /save mycontext  - Save to ./.agent13/saves/mycontext.ctx\n"
            "  /save mycontext -y  - Overwrite without prompting",
        )

    # -y flag may appear anywhere in the args; strip it and keep the rest as the name
    parts = args.split()
    force = "-y" in parts
    parts = [p for p in parts if p != "-y"]
    name = " ".join(parts)
    if not name or name.startswith("-"):
        return CommandResult(False, "Please provide a valid save name")

    path = resolve_save_path(name)

    if path.exists() and not force:
        return CommandResult(
            False,
            f"File already exists: {path}\nUse /save {name} -y to overwrite",
        )

    try:
        save_context(agent, str(path))
        msg_count = len(agent.messages)
        return CommandResult(
            True, f"Saved: {path} ({msg_count} messages)", {"msg_count": msg_count}
        )
    except Exception as e:
        return CommandResult(False, f"Error saving: {e}")


def execute_delete(agent, args: str) -> CommandResult:
    """Execute /delete command.

    Parses target (h/q/s) and spec, validates, executes deletion.
    Returns CommandResult with kind in data['kind'] on success
    ('history', 'queue', or 'save') for UI side effects.
    """
    target = args.strip()
    if not target:
        return CommandResult(False, _DELETE_USAGE)

    parts = target.split()
    if len(parts) < 2:
        return CommandResult(False, _DELETE_USAGE)

    kind = parts[0].lower()
    spec = parts[1]

    if kind == "h":
        return _delete_history(agent, spec)
    elif kind == "q":
        return _delete_queue(agent, spec)
    elif kind == "s":
        return _delete_save(spec)
    else:
        return CommandResult(False, _DELETE_USAGE)


def _normalize_idx(idx: int, total: int) -> int:
    """Convert a (possibly negative) 1-based index to a positive 1-based index.

    -1 maps to ``total``, -2 to ``total - 1``, etc. Positive indices are
    returned unchanged. Used by ``_parse_index_spec`` for the negative-index
    keyword convention (``-1`` = last item, mirroring Python's negative-index
    ergonomics but on a 1-based scale).
    """
    return total + idx + 1 if idx < 0 else idx


def _parse_index_spec(spec: str, total: int) -> tuple[list[int], str | None]:
    """Parse an index specification with support for 'last', negative indices, and ranges.

    Args:
        spec: Index specification (e.g., "1", "last", "-1", "1:3", "-2:", "-3:-1")
        total: Total number of items (for bounds checking and negative index conversion)

    Returns:
        Tuple of (list of 1-based indices, error message or None)

    Supported formats:
        - "N" - single positive index (1-based)
        - "last" - alias for -1 (last item)
        - "-N" - negative index (-1 = last, -2 = second-to-last)
        - "N:M" - range from N to M (inclusive, 1-based)
        - "N:" - range from N to end
        - ":M" - range from start to M
        - "-N:M" - range with negative start
        - "N:-M" - range with negative end
    """
    # Handle 'last' keyword
    if spec.lower() == "last":
        if total == 0:
            return [], "No items to delete"
        return [total], None

    # Handle range syntax with ':'
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            return [], f"Invalid range format: {spec}"

        start_str, end_str = parts

        # Parse start
        if start_str == "":
            start = 1
        elif start_str.lower() == "last":
            start = total
        else:
            try:
                start = int(start_str)
            except ValueError:
                return [], f"Invalid start index: {start_str}"
            start = _normalize_idx(start, total)

        # Parse end
        if end_str == "":
            end = total
        elif end_str.lower() == "last":
            end = total
        else:
            try:
                end = int(end_str)
            except ValueError:
                return [], f"Invalid end index: {end_str}"
            end = _normalize_idx(end, total)

        # Validate range
        if start < 1 or start > total:
            rng = f" (1-{total})" if total else ""
            return [], f"Start index {start} out of range{rng}"
        if end < 1 or end > total:
            rng = f" (1-{total})" if total else ""
            return [], f"End index {end} out of range{rng}"
        if start > end:
            return [], f"Invalid range: start ({start}) > end ({end})"

        return list(range(start, end + 1)), None

    # Handle single index
    try:
        idx = int(spec)
    except ValueError:
        return [], f"Invalid index: {spec}"
    idx = _normalize_idx(idx, total)

    # Validate
    if idx < 1 or idx > total:
        rng = f" (1-{total})" if total else ""
        return [], f"Index {idx} out of range{rng}"

    return [idx], None


def _delete_history(agent, spec: str) -> CommandResult:
    """Delete history groups by number or range.

    Supports: N, last, -N (negative), N:M (range), N: (to end), :M (from start)
    """
    groups = agent.history.get_message_groups()
    total = len(groups)

    indices, error = _parse_index_spec(spec, total)
    if error:
        return CommandResult(False, error)

    # Collect all message indices to delete
    all_indices = []
    for g in indices:
        all_indices.extend(groups[g - 1])

    # Delete in reverse order to preserve indices
    for idx in sorted(all_indices, reverse=True):
        del agent.messages[idx]

    # Build response message
    if len(indices) == 1:
        return CommandResult(
            True,
            f"Deleted group {indices[0]}",
            {"kind": "history"},
            refresh_status=True,
        )
    else:
        return CommandResult(
            True,
            f"Deleted groups {indices[0]}-{indices[-1]} ({len(all_indices)} messages)",
            {"kind": "history"},
            refresh_status=True,
        )


def _delete_queue(agent, spec: str) -> CommandResult:
    """Delete queue item(s) by index.

    Supports: N, last, -N (negative), N:M (range), N: (to end), :M (from start)
    """
    total = agent.queue.pending_count

    indices, error = _parse_index_spec(spec, total)
    if error:
        return CommandResult(False, error)

    # Delete items (in reverse order to preserve indices)
    removed_items = []
    for idx in sorted(indices, reverse=True):
        removed = agent.queue.remove_at(idx)
        if removed:
            removed_items.append(removed)
        else:
            # This shouldn't happen if parsing is correct, but handle it
            return CommandResult(False, f"Failed to remove queue item at index {idx}")

    # Build response
    if len(removed_items) == 1:
        text_preview = removed_items[0].text[:50]
        return CommandResult(
            True,
            f"Removed queue item: {text_preview}",
            {"kind": "queue"},
        )
    else:
        return CommandResult(
            True,
            f"Removed {len(removed_items)} queue items ({indices[0]}-{indices[-1]})",
            {"kind": "queue"},
        )


def _delete_save(spec: str) -> CommandResult:
    """Delete a save file by name."""
    from agent13.persistence import resolve_save_path

    save_path = resolve_save_path(spec)
    if save_path.exists():
        save_path.unlink()
        return CommandResult(True, f"Deleted save: {spec}", {"kind": "save"})
    else:
        return CommandResult(False, f"Save not found: {spec}")


# ---------------------------------------------------------------------------
# /retry — shared command logic
# ---------------------------------------------------------------------------


def execute_retry(agent, args: str = "") -> CommandResult:
    """Execute /retry — delete last message group and return user text for re-queueing.

    Sync: validates agent is idle, deletes the last group from agent.messages.
    The caller owns the tail, because the two front-ends differ on purpose:
    the TUI loads data["user_text"] into the input for editing, the REPL
    re-queues it with await agent.add_message(...) straight away.

    `args` is accepted for the handler signature and ignored - /retry takes
    no arguments.

    Skips trailing priming pairs (left by journal compaction) so the priming
    prompt is never offered as retry text. Priming pairs are left in place;
    journal's sweep-on-next-run handles multiples.

    Returns CommandResult with data["user_text"] on success.
    """
    from agent13.prompts import PRIMING_PROMPT, PRIMING_RESPONSE, JOURNAL_USER_MESSAGE_PREFIX
    from agent13.message_history import content_to_text

    def _strip_journal_prefix(content) -> str:
        """Flatten a message's content and remove the journal prefix if present.

        After journal compaction, the last user message is rewritten to
        '[previous user message] "<original>"'.  /retry should offer the
        user their original text for editing, not the prefixed wrapper.

        Content is flattened first: multimodal messages (native vision)
        carry a list of blocks instead of a string, and old sessions loaded
        from disk can still have one at a group boundary.
        """
        text = content_to_text(content)
        prefix = JOURNAL_USER_MESSAGE_PREFIX + ' "'
        if text.startswith(prefix) and text.endswith('"') and len(text) > len(prefix):
            return text[len(prefix):-1]
        return text

    if not agent.is_idle:
        return CommandResult(False, "Agent is busy")
    if not agent.messages:
        return CommandResult(False, "No messages to retry")

    groups = agent.history.get_message_groups()
    if not groups:
        return CommandResult(False, "No messages to retry")

    def _is_priming_group(group):
        """True if a group is a priming pair (priming prompt + 'ok')."""
        if len(group) != 2:
            return False
        user_msg = agent.messages[group[0]]
        asst_msg = agent.messages[group[1]]
        return (
            user_msg.get("role") == "user"
            and user_msg.get("content") == PRIMING_PROMPT
            and asst_msg.get("role") == "assistant"
            and asst_msg.get("content") == PRIMING_RESPONSE
        )

    # Walk backwards past trailing priming-pair groups to find the real
    # last group the user would want to retry.
    target_idx = len(groups) - 1
    while target_idx >= 0 and _is_priming_group(groups[target_idx]):
        target_idx -= 1

    if target_idx < 0:
        return CommandResult(False, "No messages to retry")

    last_group = groups[target_idx]
    first_msg_idx = last_group[0]
    user_text = _strip_journal_prefix(
        agent.messages[first_msg_idx].get("content", "")
    )

    for idx in sorted(last_group, reverse=True):
        del agent.messages[idx]

    return CommandResult(True, "Retrying", {"user_text": user_text})


# ---------------------------------------------------------------------------
# /prioritise, /deprioritise — shared command logic
# ---------------------------------------------------------------------------


def execute_prioritise(agent, args: str) -> CommandResult:
    """Execute /prioritise — mark queue item as priority."""
    if not args.strip():
        return CommandResult(False, "Usage: /prioritise N")
    try:
        idx = int(args.strip())
    except ValueError:
        return CommandResult(False, "Invalid index format")
    if agent.queue.set_priority_at(idx, True):
        return CommandResult(True, f"Item {idx} marked as priority", {"kind": "queue"})
    return CommandResult(False, f"Invalid queue index: {idx}")


def execute_deprioritise(agent, args: str) -> CommandResult:
    """Execute /deprioritise — remove priority from queue item."""
    if not args.strip():
        return CommandResult(False, "Usage: /deprioritise N")
    try:
        idx = int(args.strip())
    except ValueError:
        return CommandResult(False, "Invalid index format")
    if agent.queue.set_priority_at(idx, False):
        return CommandResult(True, f"Item {idx} priority removed", {"kind": "queue"})
    return CommandResult(False, f"Invalid queue index: {idx}")


_GRIND_USAGE = (
    "/grind <dur> [text]   work until the window closes (8h, 90m, 2h30m)\n"
    "/grind                show the window and the goal text\n"
    "/grind off            stop the window, keep the goal text\n"
    "/grind clear          stop the window and forget the goal text"
)


def execute_grind(agent, args: str) -> CommandResult:
    """Execute /grind — arm a goal window that survives turn ends.

    Forms:
      /grind <dur> [text] - arm (or re-arm) the window; the text is sticky
      /grind              - show the window, the goal text and the time left
      /grind off          - stop the window, KEEPING the goal text
      /grind clear        - stop the window and FORGET the goal text

    ``off`` and ``clear`` differ only in the text, and that difference is the
    whole point: the text is what a later bare ``/grind 4h`` re-arms, so
    forgetting it is a separate, deliberate act. Neither interrupts the current
    turn - the window bounds the decision to keep going, not the work in
    flight. Both stop the window, because a window with no goal would keep
    grinding toward the record's ticket list.

    A HANDLER command, not a queue item: both front-ends dispatch it
    synchronously at input time, so it takes effect immediately, mid-turn or
    not, and never touches the queue or the interrupt path (an interrupt-class
    command would be delivered to the model as literal text - core.py's
    interrupt path appends item.text with no kind check). Handler commands are
    not turns, so the status bar's task clock is untouched.
    """
    from agent13 import goal as goal_rules

    args = args.strip()
    if not args:
        if not agent.goal_active:
            kept = f" - goal text kept: {agent._goal_text}" if agent._goal_text else ""
            _log_goal_command(agent, "show", args)
            return CommandResult(True, f"No grind active{kept}")
        _log_goal_command(agent, "show", args)
        return CommandResult(
            True,
            "Grind: {text}\n  window {window}, {left} remaining".format(
                text=agent._goal_text or "no goal text",
                window=goal_rules.format_window(agent.goal_window),
                left=goal_rules.format_window(agent.goal_remaining),
            ),
            {"kind": "goal"},
            refresh_status=True,
        )

    if args.lower() in ("off", "clear"):
        # Two commands, one difference: whether the goal text survives. The
        # log action names follow the command, not the shared effect - after
        # the fact the log is the only place the two can be told apart.
        if args.lower() == "off":
            kept = agent._goal_text
            agent.stop_goal()
            _log_goal_command(agent, "stop", args, kept_text=kept)
            message = (
                f"Grind stopped - goal text kept: {kept}"
                if kept
                else "Grind stopped - no goal text to keep"
            )
        else:
            dropped = agent._goal_text
            agent.clear_goal()
            _log_goal_command(agent, "clear", args, cleared_text=dropped)
            message = "Grind cleared"
        return CommandResult(True, message, {"kind": "goal"}, refresh_status=True)

    seconds, text = goal_rules.parse_duration(args)
    if seconds is None:
        _log_goal_command(agent, "reject", args, reason="no duration")
        return CommandResult(
            False,
            "Duration must come first, in hours and minutes (8h, 90m, 2h30m; "
            "no seconds).\n" + _GRIND_USAGE,
        )
    if seconds <= 0:
        _log_goal_command(agent, "reject", args, reason="zero duration")
        return CommandResult(
            False,
            "Duration must be at least one minute - a window that closes on "
            "arrival is a typo.\n" + _GRIND_USAGE,
        )
    agent.set_goal(seconds, text or None)
    _log_goal_command(agent, "arm", args)
    return CommandResult(
        True,
        "Grind armed: {text}\n  window {window}".format(
            text=agent._goal_text or "no goal text",
            window=goal_rules.format_window(agent.goal_window),
        ),
        {"kind": "goal"},
        refresh_status=True,
    )


def _log_goal_command(agent, action: str, raw: str, **extra) -> None:
    """Record a /grind invocation: what was asked, and the state it left.

    Logged from the command rather than from set_goal/clear_goal so a REJECTED
    duration is recorded too - "I typed /grind 90s and nothing happened" is the
    question this answers. The goal text is logged verbatim: it is the user's
    own words, and "why did it work on the wrong thing" is answered by it.
    """
    from agent13.debug_log import is_debug_enabled, log_tps_event

    if not is_debug_enabled():
        return
    data = {
        "action": action,
        "raw": raw,
        "goal_text": getattr(agent, "_goal_text", ""),
        "goal_active": bool(getattr(agent, "goal_active", False)),
        "goal_window_s": round(getattr(agent, "goal_window", 0.0), 1),
        "goal_remaining_s": round(getattr(agent, "goal_remaining", 0.0), 1),
    }
    data.update(extra)
    log_tps_event("goal_command", data)


def execute_polite(agent, args: str) -> CommandResult:
    """Execute /polite command — enable/disable polite multi-agent mode.

    Forms:
      /polite N   - enable with poll interval N seconds
      /polite off - disable (silent no-op if not enabled)
      /polite     - show usage

    Returns CommandResult with data['action'] = 'enabled'|'disabled' and
    data['interval'] (when enabled) for UI side-effects.
    """
    args = args.strip()
    if not args:
        # Show current status plus usage hint.
        if agent.polite_lock is not None:
            status = f"Polite mode: enabled (interval {agent.polite_lock.interval}s)"
        else:
            status = "Polite mode: off"
        return CommandResult(True, f"{status}\n{_POLITE_USAGE}")

    if args.lower() == "off":
        # Silent no-op if never enabled (per design).
        if agent.polite_lock is None:
            return CommandResult(
                True,
                "Polite mode not enabled",
                {"action": "disabled"},
                refresh_status=True,
            )
        agent.disable_polite()
        return CommandResult(
            True,
            "Polite mode disabled",
            {"action": "disabled"},
            refresh_status=True,
        )

    # Parse interval N (float, must be non-negative).
    try:
        interval = float(args)
    except ValueError:
        return CommandResult(False, _POLITE_USAGE)
    if interval < 0:
        return CommandResult(False, f"Interval must be non-negative\n{_POLITE_USAGE}")

    agent.set_polite(interval=interval)
    if interval == 0:
        return CommandResult(
            True,
            "Rude mode enabled",
            {"action": "enabled", "interval": interval},
            refresh_status=True,
        )
    return CommandResult(
        True,
        f"Polite mode enabled (interval {interval}s)",
        {"action": "enabled", "interval": interval},
        refresh_status=True,
    )


# ---------------------------------------------------------------------------
# /cwd - shared command logic
# ---------------------------------------------------------------------------


def execute_cwd(agent, args: str) -> CommandResult:
    """Execute /cwd - show or change the working directory.

    A leading @ (from path autocomplete) is stripped from the path.
    """
    args = args.strip()
    if not args:
        return CommandResult(True, f"Working directory: {Path.cwd()}")

    target = Path(args.lstrip("@")).expanduser().resolve()
    if not target.exists():
        return CommandResult(False, f"Path does not exist: {target}")
    if not target.is_dir():
        return CommandResult(False, f"Not a directory: {target}")
    try:
        os.chdir(target)
    except OSError as e:
        return CommandResult(False, f"Cannot change directory: {e}")
    return CommandResult(True, f"Working directory changed to: {target}")


def execute_tools(agent, args: str = "") -> CommandResult:
    """Execute /tools - this session's tool usage statistics.

    Args are ignored (there are none). Per tool: calls, successes, the mode
    breakdown, and the argument-combination breakdown (only when a tool was
    called with more than one combination - a single row is noise).
    """
    from agent13.status import get_tool_stats_summary

    summary = get_tool_stats_summary(agent)
    if summary["total_calls"] == 0:
        return CommandResult(True, "No tool calls yet this session", pane="tools")

    rate = summary["success_rate"]
    lines = [
        f"Tool Usage  {summary['total_successes']}/{summary['total_calls']} "
        f"successful ({rate:.0f}%)",
        "",
    ]

    for tool in summary["per_tool"]:
        line = f"  {tool['name']}  {tool['successes']}/{tool['calls']}"
        if tool["modes"]:
            mode_strs = [
                f"{m['mode']}={m['successes']}/{m['calls']}" for m in tool["modes"]
            ]
            line += f"  modes: {', '.join(mode_strs)}"
        lines.append(line)

        if len(tool["param_combos"]) > 1:
            for combo in tool["param_combos"]:
                lines.append(
                    f"    {combo['combo']}={combo['successes']}/{combo['calls']}"
                )

    return CommandResult(True, "\n".join(lines), pane="tools")


_MCP_USAGE = (
    "Usage: /mcp [connect|disconnect|reload]\n"
    "  /mcp             list servers and their tools\n"
    "  /mcp connect     connect to all configured servers\n"
    "  /mcp disconnect  disconnect from all servers\n"
    "  /mcp reload      reconnect (re-read the server config)"
)


def execute_mcp(agent, args: str) -> CommandResult:
    """Execute /mcp - list MCP servers, or start a connection change.

    The listing is synchronous. connect/disconnect/reload are async, so the
    handler does the pre-checks and leaves the action in data["mcp_action"];
    the front-end awaits run_mcp_action() and renders what it returns.
    """
    subcmd = args.strip().split(maxsplit=1)[0].lower() if args.strip() else ""

    if subcmd == "connect":
        if not agent._mcp_server_configs:
            return CommandResult(False, "No MCP servers configured")
        return CommandResult(
            True, "Connecting to MCP servers...", {"mcp_action": "connect"}
        )

    if subcmd == "disconnect":
        if not agent.mcp:
            return CommandResult(False, "MCP not connected")
        return CommandResult(
            True, "Disconnecting MCP servers...", {"mcp_action": "disconnect"}
        )

    if subcmd == "reload":
        if not agent.mcp:
            return CommandResult(False, "MCP not initialized (no servers configured)")
        return CommandResult(
            True, "Reloading MCP servers...", {"mcp_action": "reload"}
        )

    if subcmd:
        return CommandResult(False, _MCP_USAGE)

    # ── Listing ──
    from ui.display import format_mcp_servers

    configured = len(agent._mcp_server_configs)
    if not agent.mcp:
        return CommandResult(
            True,
            "MCP Status:\n"
            "  Status: Not initialized\n"
            f"  Configured servers: {configured}\n"
            "\n"
            "Use /mcp connect to connect to MCP servers.",
            pane="mcp",
        )

    servers = agent.mcp.get_server_info()
    if not servers:
        return CommandResult(
            True,
            "MCP Status:\n"
            f"  Configured servers: {configured}\n"
            "\n"
            "Use /mcp connect to connect to servers.",
            pane="mcp",
        )

    return CommandResult(True, format_mcp_servers(servers, use_rich=False), pane="mcp")


async def run_mcp_action(agent, action: str) -> tuple[bool, str]:
    """Run the async half of /mcp connect|disconnect|reload.

    Returns (success, message) for the front-end to render: the message only
    exists after the await, so it cannot live in the CommandResult.
    """
    try:
        if action == "connect":
            mcp = await agent._ensure_mcp()
            if not mcp:
                return False, "MCP connect failed: no MCP client"
            info = await mcp.connect_all()
            return True, f"MCP connected: {list(info.keys())}"
        if action == "disconnect":
            await agent.disconnect_mcp()
            return True, "MCP servers disconnected"
        if action == "reload":
            servers = await agent.mcp.reload()
            return True, f"MCP reconnected: {list(servers.keys())}"
    except Exception as e:
        return False, f"MCP {action} error: {e}"
    return False, f"Unknown MCP action: {action}"


def execute_focus(agent, args: str) -> CommandResult:
    """Execute /focus - set (or report) the authoritative focus name.

    Runs at call time, not at the next queue slot: the name must land before
    a threshold-triggered wrap-up builds its prompt, so record targeting
    follows a mid-turn /focus. On an empty history the focus's durable
    records are seeded into it; mid-session it is set-only. No arg reports
    the current name.

    The status bar shows the focus name, so the result asks for a refresh.
    """
    message, _level = agent.apply_focus_name(args.strip())
    return CommandResult(True, message, refresh_status=True)


# ---------------------------------------------------------------------------
# /model, /provider - shared command logic
# ---------------------------------------------------------------------------
#
# Both commands have an async tail the handler cannot own: after a provider
# switch the front-end fetches the new model list (the TUI in a task, the REPL
# with an await). The handler does the synchronous part and leaves the new
# base_url in data["base_url"]; each front-end runs the fetch and renders it
# with format_model_list.


def execute_model(agent, args: str) -> CommandResult:
    """Execute /model - list the models, or select one by number/name."""
    from agent13.models import format_model_list, resolve_model_selection

    args = args.strip()
    models = list(getattr(agent, "available_models", None) or [])

    if not args:
        if not models:
            return CommandResult(
                False, "No models loaded. Use /provider to switch providers."
            )
        return CommandResult(
            True,
            format_model_list(models, agent.model),
            data={"models": models, "current": agent.model},
        )

    captured = io.StringIO()
    model = resolve_model_selection(models, args, output=captured)
    if model is None:
        # resolve_model_selection explains what went wrong (bad number,
        # ambiguous prefix, unknown name) - pass that through.
        return CommandResult(
            False, captured.getvalue().strip() or f"Unknown model: {args}"
        )

    agent.set_model(model)
    return CommandResult(
        True, f"Model set to: {model}", data={"model": model}, refresh_status=True
    )


def execute_provider(agent, args: str) -> CommandResult:
    """Execute /provider - list the providers, or switch to one by name/URL."""
    from agent13.config import (
        current_provider_name,
        format_provider_list,
        get_provider_names,
        resolve_provider_arg,
        resolve_provider_selection,
    )

    args = args.strip()

    if not args:
        names = get_provider_names()
        if not names:
            return CommandResult(
                False, "No providers configured in ~/.agent13/config.toml"
            )
        current = current_provider_name(agent)
        return CommandResult(
            True,
            format_provider_list(current),
            data={"providers": names, "current": current},
        )

    captured = io.StringIO()
    resolved = resolve_provider_selection(args, output=captured)
    if resolved is None:
        return CommandResult(
            False, captured.getvalue().strip() or f"No provider matching '{args}'"
        )

    try:
        base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
            resolved
        )
    except ValueError as e:
        return CommandResult(False, f"Error: {e}")

    from agent13.config import create_client

    agent.set_client(
        create_client(
            base_url,
            api_key,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
        )
    )
    # Remember the chosen name (empty for a raw URL) so the status bar shows
    # what the user picked instead of guessing from the base_url.
    agent.provider_name = (
        "" if resolved.startswith(("http://", "https://")) else resolved
    )
    return CommandResult(
        True,
        f"Provider changed to: {base_url}",
        data={"base_url": base_url},
        refresh_status=True,
    )


# ---------------------------------------------------------------------------
# /remove-api-reasoning, /tool-response - shared command logic
# ---------------------------------------------------------------------------

_REMOVE_REASONING_USAGE = (
    "Usage: /remove-api-reasoning [on|off]\n"
    "  /remove-api-reasoning on - Strip reasoning between turns (API context)\n"
    "  /remove-api-reasoning off - Preserve reasoning between turns\n"
    "  To fold the widgets in the chat, use /reasoning hide"
)


def execute_remove_api_reasoning(agent, args: str) -> CommandResult:
    """Execute /remove-api-reasoning - strip reasoning from the API context.

    No args toggles: the setting is a two-state switch, so a bare invocation
    flips it rather than reporting it.
    """
    args = args.strip().lower()
    if args == "on":
        agent.remove_reasoning = True
        return CommandResult(
            True,
            "Remove API reasoning enabled\n"
            "Reasoning tokens will be stripped between turns to save context.",
        )
    if args == "off":
        agent.remove_reasoning = False
        return CommandResult(
            True,
            "Remove API reasoning disabled\n"
            "Reasoning tokens will be preserved between turns.",
        )
    if not args:
        agent.remove_reasoning = not agent.remove_reasoning
        status = "on" if agent.remove_reasoning else "off"
        return CommandResult(True, f"Remove API reasoning: {status}")

    status = "on" if agent.remove_reasoning else "off"
    return CommandResult(False, f"{_REMOVE_REASONING_USAGE}\n  Current: {status}")


def execute_tool_response(agent, args: str) -> CommandResult:
    """Execute /tool-response - set the format the model answers tools in."""
    current = "json" if getattr(agent, "response_format", None) else "raw"

    args = args.strip().lower()
    if not args:
        return CommandResult(
            True,
            f"Tool response format: {current}\nUsage: /tool-response [raw|json]",
        )
    if args not in ("raw", "json"):
        return CommandResult(False, "Invalid format. Use 'raw' or 'json'")

    agent.set_response_format({"type": "json_object"} if args == "json" else None)
    return CommandResult(True, f"Tool response format set to: {args}")


# ---------------------------------------------------------------------------
# /prompt, /skills, /snippet - shared command logic
# ---------------------------------------------------------------------------
# The three managers are attached to the agent by cli.py (one instance for
# every mode). A front-end with none attached - batch, pipe - degrades to a
# one-line explanation instead of crashing.

_PROMPT_USAGE = "Usage: /prompt [name]   (no args lists prompts)"

_SKILLS_USAGE = (
    "Usage: /skills [on|off|list|status|refresh]\n"
    "  /skills on       add skill descriptions to the system prompt, enable the tool\n"
    "  /skills off      remove the descriptions (the tool has no off-switch)\n"
    "  /skills status   show both switches\n"
    "  /skills refresh  re-sync bundled default skills (-y overwrites local changes)\n"
    "  /skills          list the available skills"
)

_SNIPPET_USAGE = (
    "Snippets: saved user messages invoked as slash commands.\n"
    "Usage:\n"
    "  /snippet list                 list all snippets (re-reads the file first)\n"
    "  /snippet add last <name> [-y] save the last user message as a snippet\n"
    "  /snippet delete <name>        delete a snippet\n"
    "  /snippet rename <old> <new>   rename a snippet\n"
    "  /snippet use <name>           load a snippet into the input\n"
    "\n"
    "You can also type /<snippet_name> to load it directly."
)


def execute_prompt(agent, args: str) -> CommandResult:
    """Execute /prompt - switch the active system prompt.

    No args lists the prompts. The TUI turns that into its completion picker
    (data key ``prompt_picker`` = (names, active)); the REPL prints the list.
    """
    pm = getattr(agent, "prompt_manager", None)
    if pm is None:
        return CommandResult(False, "No prompt manager configured")

    name = args.strip()
    if not name:
        names = list(pm.prompts.keys())
        lines = [
            f"  {'*' if n == pm.active_prompt else ' '} {n}" for n in names
        ]
        return CommandResult(
            True,
            "Available prompts:\n" + "\n".join(lines),
            {"prompt_picker": (names, pm.active_prompt)},
        )

    if not pm.set_active(name):
        return CommandResult(
            False,
            f"Prompt not found: {name}\nAvailable: {', '.join(pm.prompts)}",
        )

    agent.reload_agents_md()
    agent.set_system_prompt(pm.get_prompt())
    return CommandResult(True, f"Switched to prompt: {name}", refresh_status=True)


def _apply_skills_list(agent, include: bool) -> None:
    """Rebuild the system prompt with or without the skills section."""
    from agent13.prompts import get_skills_section

    pm = getattr(agent, "prompt_manager", None)
    base = pm.get_prompt() if pm is not None else agent.system_prompt
    sm = getattr(agent, "skill_manager", None)
    if include and sm is not None and sm.skills:
        section = get_skills_section(sm.skills)
        if section:
            agent.set_system_prompt(f"{base}\n\n{section}")
            return
    agent.set_system_prompt(base)


def _execute_skills_refresh(sm, force: bool) -> CommandResult:
    """Re-sync bundled default skills (mechanism: skills/defaults.py).

    Without -y: installs new defaults, updates untouched ones, and WARNS
    about locally-modified ones it did not touch (the /save -y convention).
    With -y: overwrites the modified ones too. The manager is rescanned so
    the running session serves the refreshed content without a restart.
    """
    from agent13.skills.defaults import sync_default_skills

    report = sync_default_skills(force=force)
    if hasattr(sm, "rescan"):
        sm.rescan()
    if not report:
        return CommandResult(True, "All default skills are up to date.")
    return CommandResult(True, report.render())


def execute_skills(agent, args: str) -> CommandResult:
    """Execute /skills - toggle the skills list or list the available skills.

    on/off toggle the skills *list* (descriptions in the system prompt); 'on'
    also enables the *tool*. The tool is enabled by --skills, by /skills on,
    or by the first skill invocation - and never switches off.
    """
    sm = getattr(agent, "skill_manager", None)
    if sm is None:
        return CommandResult(False, "No skill manager configured")

    arg = args.strip().lower()

    if arg == "on":
        agent.skills_list_enabled = True
        agent.set_skills_mode(True)
        _apply_skills_list(agent, True)
        return CommandResult(
            True,
            "Skills enabled\n"
            "Skill descriptions added to system prompt and the "
            "skill tool is available.",
            refresh_status=True,
        )

    if arg == "off":
        agent.skills_list_enabled = False
        _apply_skills_list(agent, False)
        return CommandResult(
            True,
            "Skills list disabled\n"
            "Skill descriptions removed from the system prompt. "
            "Note: the skill tool, once enabled, has no off-switch. "
            "You can still invoke skills via slash commands.",
            refresh_status=True,
        )

    if arg == "status":
        count = len(sm.skills) if sm.skills else 0
        return CommandResult(
            True,
            "Skills status\n"
            f"  List in prompt: {'on' if agent.skills_list_enabled else 'off'}\n"
            f"  Skill tool:     {'on' if agent.skills_mode else 'off'} "
            "(auto-enables after first skill use)\n"
            f"  {count} skill(s) available",
        )

    parts = arg.split()
    if parts and parts[0] == "refresh":
        return _execute_skills_refresh(sm, force="-y" in parts[1:])

    if arg not in ("", "list"):
        return CommandResult(False, f"{_SKILLS_USAGE}\n\nUnknown: {arg}")

    skills = sm.skills
    if not skills:
        return CommandResult(True, "No skills available")

    model_invocable = [
        (name, info)
        for name, info in sorted(skills.items())
        if not info.disable_model_invocation
    ]
    user_invoked = [
        (name, info)
        for name, info in sorted(skills.items())
        if info.disable_model_invocation
    ]

    lines = [f"Available skills (tool: {'on' if agent.skills_mode else 'off'})", ""]

    def _entry(name: str, info) -> str:
        # One line per skill: descriptions are often YAML block scalars and
        # run to a paragraph, which would swamp the listing in either pane.
        return f"  /{name} - {_collapse(info.description or '', 100)}"

    for name, info in model_invocable:
        lines.append(_entry(name, info))

    if user_invoked:
        if model_invocable:
            lines.append("")
        lines.append("User-invoked only (hidden from the model; invoke via /name):")
        for name, info in user_invoked:
            lines.append(_entry(name, info))

    lines.append("")
    lines.append(
        "Invoke a skill to enable the tool - /skills on|off toggles the list"
    )
    return CommandResult(True, "\n".join(lines))


def _last_user_message(agent) -> Optional[str]:
    """The text of the last non-slash-command user message, or None."""
    from agent13.message_history import content_to_text

    idx = agent.history.find_last_user_idx()
    if idx is None:
        return None
    content = content_to_text(agent.messages[idx].get("content", ""))
    if not content or not content.strip():
        return None
    return content


def execute_snippet(agent, args: str) -> CommandResult:
    """Execute /snippet - manage the saved user messages.

    Mutations announce themselves in ``data`` so a front-end can re-register
    its dynamic slash commands: ``snippet_added`` / ``snippet_removed`` (name).
    ``/snippet use`` carries the text in ``snippet_text`` - the TUI loads it
    into the input, the REPL prints it.
    """
    sm = getattr(agent, "snippet_manager", None)
    if sm is None:
        return CommandResult(False, "No snippet manager configured")

    if not args.strip():
        return CommandResult(True, _SNIPPET_USAGE)

    parts = args.split(maxsplit=2)
    sub = parts[0].lower()

    if sub == "list":
        snippets = sm.list_snippets()
        if not snippets:
            return CommandResult(True, "No snippets defined.")
        lines = ["Available snippets:"]
        for s in snippets:
            collision = (
                " (collision with built-in)" if s["collision"] else ""
            )
            lines.append(f"  {s['name']}{collision}")
            lines.append(f"    {s['preview']}")
        return CommandResult(True, "\n".join(lines))

    if sub == "add":
        all_parts = args.split()
        if len(all_parts) < 3 or all_parts[1].lower() != "last":
            return CommandResult(
                False,
                "Usage: /snippet add last <name> [-y]\n"
                "  Saves your previous user message as a snippet.",
            )
        force = "-y" in all_parts
        name_parts = [p for p in all_parts[2:] if p != "-y"]
        if not name_parts:
            return CommandResult(False, "Usage: /snippet add last <name> [-y]")
        name = name_parts[0]

        validation_error = sm.validate_name(name)
        if validation_error:
            return CommandResult(False, validation_error)

        if sm.get_snippet(name) is not None and not force:
            return CommandResult(
                False,
                f"Snippet '{name}' already exists.\n"
                f"Use /snippet add last {name} -y to overwrite",
            )

        content = _last_user_message(agent)
        if content is None:
            return CommandResult(False, "No user message to save")

        warning = sm.add_snippet(name, content)
        message = f"Saved snippet '{name}'"
        if warning:
            message += f"\nWARNING: {warning}"
        return CommandResult(True, message, {"snippet_added": name})

    if sub == "delete":
        if len(parts) < 2:
            return CommandResult(False, "Usage: /snippet delete <name>")
        name = parts[1]
        if not sm.delete_snippet(name):
            return CommandResult(False, f"Snippet not found: {name}")
        return CommandResult(
            True, f"Deleted snippet '{name}'", {"snippet_removed": name}
        )

    if sub == "rename":
        if len(parts) < 3:
            return CommandResult(False, "Usage: /snippet rename <old> <new>")
        old_name, new_name = parts[1], parts[2]
        result = sm.rename_snippet(old_name, new_name)
        if result and result.startswith("Snippet not found"):
            return CommandResult(False, result)
        if result and result.startswith("Invalid"):
            return CommandResult(False, result)
        message = f"Renamed snippet '{old_name}' -> '{new_name}'"
        if result:
            message += f"\nWARNING: {result}"
        return CommandResult(
            True,
            message,
            {"snippet_removed": old_name, "snippet_added": new_name},
        )

    if sub == "use":
        if len(parts) < 2:
            return CommandResult(False, "Usage: /snippet use <name>")
        name = parts[1]
        snippet = sm.get_snippet(name)
        if snippet is None:
            return CommandResult(False, f"Snippet not found: {name}")
        return CommandResult(
            True, f"Snippet '{name}':\n{snippet}", {"snippet_text": snippet}
        )

    return CommandResult(
        False, f"Unknown snippet command: {sub}\n\n{_SNIPPET_USAGE}"
    )


def execute_skill_invocation(
    agent, name: str, args: str = ""
) -> Optional[CommandResult]:
    """Load a skill by name and queue its content as the next user turn.

    This is the dynamic ``/skillname`` dispatch, not a catalogue command: both
    front-ends call it when the typed command matches no built-in. Returns None
    when no skill has that name, so the caller can report an unknown command.

    The skill tool is ratcheted on (one-way) by the first invocation, and the
    content is recorded for chain-restart re-injection.
    """
    from agent13.file_injection import expand_file_mentions

    sm = getattr(agent, "skill_manager", None)
    if sm is None or not sm.get_skill(name):
        return None

    content = sm.format_skill_content(name)
    if not content:
        return CommandResult(False, f"Failed to load skill: {name}")

    if not agent.skills_mode:
        agent.set_skills_mode(True)
    if args:
        content = f"{content}\n\n{args}"
    # Track for chain-restart re-injection (the content as sent, args included).
    agent.note_skill_loaded(name, content)
    return CommandResult(
        True,
        f"Skill loaded: {name}",
        {"agent_message": (expand_file_mentions(content), "prompt", {})},
    )


# ---------------------------------------------------------------------------
# /bell, /bell-command - shared command logic
# ---------------------------------------------------------------------------

_BELL_USAGE = (
    "Usage: /bell [N|off]\n"
    "  /bell 30 - Ring bell after 30s\n"
    "  /bell 0 - Always ring on idle\n"
    "  /bell off - Disable bell\n"
    "  /bell - Show current status"
)

_BELL_COMMAND_HINTS = (
    "  /bell-command off - Revert to terminal bell\n"
    "  /bell-command - Show current status"
)


def _bell_manager(agent):
    """The session's BellManager, or None when this mode has none.

    Each front-end builds the manager (it supplies the fallback ring: Textual's
    bell in the TUI, a literal \\a in the REPL) and attaches it to the agent as
    ``agent.bell``, so the shared handlers can reach it without a UI callback.
    """
    return getattr(agent, "bell", None)


def execute_bell(agent, args: str) -> CommandResult:
    """Execute /bell - set or show the terminal bell threshold."""
    bell = _bell_manager(agent)
    if bell is None:
        return CommandResult(False, "Bell is not available in this mode")

    args = args.strip().lower()
    if not args:
        status = bell.status_text()
        if status == "off":
            return CommandResult(True, "Bell: off")
        if status == "always":
            return CommandResult(True, "Bell: on (always)")
        return CommandResult(True, f"Bell: on ({status})")
    if args == "off":
        bell.disable()
        return CommandResult(True, "Bell: off")

    try:
        val = float(args)
        if val < 0:
            raise ValueError("negative")
    except ValueError:
        return CommandResult(False, _BELL_USAGE)

    bell.set_threshold(val)
    if val == 0:
        return CommandResult(True, "Bell: on (always)")
    return CommandResult(True, f"Bell: on ({val:.0f}s)")


def execute_bell_command(agent, args: str) -> CommandResult:
    """Execute /bell-command - set or show the external bell command."""
    bell = _bell_manager(agent)
    if bell is None:
        return CommandResult(False, "Bell is not available in this mode")

    args = args.strip()
    if not args:
        if bell.command:
            return CommandResult(True, f"Bell command: {bell.command}")
        return CommandResult(True, "Bell command: (terminal bell)")
    if args.lower() == "off":
        bell.clear_command()
        return CommandResult(True, "Bell command: cleared (terminal bell)")

    # Strip surrounding quotes if present (single or double)
    if len(args) >= 2 and args[0] in "\"'" and args[-1] == args[0]:
        args = args[1:-1].strip()
    if not args:
        return CommandResult(False, "Usage: /bell-command <cmd>\n" + _BELL_COMMAND_HINTS)
    if not bell.set_command(args):
        return CommandResult(
            False,
            f"Error: '{args.split()[0]}' is not executable\n"
            "  The first token must be found in PATH.\n" + _BELL_COMMAND_HINTS,
        )
    return CommandResult(True, f"Bell command: {args}")


# ---------------------------------------------------------------------------
# /auto_context_* - shared command logic
# ---------------------------------------------------------------------------

_ACTION_USAGE = (
    "Usage: /auto_context_action [compact|report_and_compact|none]\n"
    "  /auto_context_action compact - Full compact at the threshold\n"
    "  /auto_context_action report_and_compact - Wrap up, report, then compact\n"
    "  /auto_context_action none - Pause at the threshold (no action)\n"
    "  /auto_context_action - Show current value"
)

_THRESHOLD_USAGE = (
    "Usage: /auto_context_threshold [N|0]\n"
    "  /auto_context_threshold 150k - Compact at 150,000 tokens\n"
    "  /auto_context_threshold 0 - Disable auto-context\n"
    "  /auto_context_threshold - Show current value"
)

_CHAIN_USAGE = (
    "Usage: /auto_context_chain [N]\n"
    "  /auto_context_chain 2 - Restart the turn up to 2 times per turn\n"
    "  /auto_context_chain - Show current value"
)

_NUDGE_CAP_USAGE = (
    "Usage: /auto_context_nudge_cap [N]\n"
    "  /auto_context_nudge_cap 2 - Stop after 2 consecutive degenerate ends\n"
    "  /auto_context_nudge_cap 0 - Disable the guard\n"
    "  /auto_context_nudge_cap - Show current value"
)


def execute_auto_context_threshold(agent, args: str) -> CommandResult:
    """Execute /auto_context_threshold - set or show the context threshold."""
    from agent13.status import parse_token_count

    args = args.strip()
    if not args:
        if agent.auto_context_threshold <= 0:
            return CommandResult(True, "Auto-context: off (threshold 0)")
        if agent.auto_context_action == "none":
            action_line = "Action: none (chain ignored)"
        else:
            action_line = (
                f"Action: {agent.auto_context_action} "
                f"(chain: {agent.auto_context_chain})"
            )
        return CommandResult(
            True,
            f"Auto-context: on ({agent.auto_context_threshold:,} tokens)\n"
            f"{action_line}",
        )

    try:
        threshold = parse_token_count(args)
    except ValueError:
        return CommandResult(False, _THRESHOLD_USAGE)
    agent.auto_context_threshold = threshold
    # The status bar's Ctx colour is a function of this threshold (warn at
    # 80%, crit at 90%), so the bar must repaint or it keeps the old band.
    if threshold == 0:
        return CommandResult(True, "Auto-context: off", refresh_status=True)
    return CommandResult(
        True, f"Auto-context: on ({threshold:,} tokens)", refresh_status=True
    )


def execute_auto_context_action(agent, args: str) -> CommandResult:
    """Execute /auto_context_action - set or show what happens at the threshold."""
    args = args.strip()
    valid = ("compact", "report_and_compact", "none")
    if not args:
        return CommandResult(
            True,
            f"Auto-context action: {agent.auto_context_action} "
            "(valid: compact | report_and_compact | none)",
        )
    if args not in valid:
        return CommandResult(False, _ACTION_USAGE)
    agent.auto_context_action = args
    return CommandResult(True, f"Auto-context action: {args}")


def execute_auto_context_chain(agent, args: str) -> CommandResult:
    """Execute /auto_context_chain - set or show the restarts-per-turn cap."""
    args = args.strip()
    if not args:
        return CommandResult(
            True,
            f"Auto-context chain: {agent.auto_context_chain} "
            "(restarts per turn; 0 = stop after the action)",
        )
    try:
        chain = int(args)
        if chain < 0:
            raise ValueError("must be >= 0")
    except ValueError:
        return CommandResult(False, _CHAIN_USAGE)
    agent.auto_context_chain = chain
    return CommandResult(True, f"Auto-context chain: {chain}")


def execute_auto_context_nudge_cap(agent, args: str) -> CommandResult:
    """Execute /auto_context_nudge_cap - set or show the degeneration guard cap."""
    args = args.strip()
    if not args:
        cap = agent.auto_context_nudge_cap
        if cap > 0:
            return CommandResult(
                True,
                f"Degeneration guard: on (cap {cap})\n"
                f"  nudges: {agent.degenerate_nudges}",
            )
        return CommandResult(True, "Degeneration guard: off (nudge cap 0)")

    try:
        nudge_cap = int(args)
        if nudge_cap < 0:
            raise ValueError("must be >= 0")
    except ValueError:
        return CommandResult(False, _NUDGE_CAP_USAGE)
    agent.auto_context_nudge_cap = nudge_cap
    if nudge_cap == 0:
        return CommandResult(True, "Degeneration guard: off (nudge cap 0)")
    return CommandResult(True, f"Degeneration guard cap: {nudge_cap}")


# ---------------------------------------------------------------------------
# /wrapup, /chain, /journal last|all - commands that queue an agent turn
# ---------------------------------------------------------------------------
#
# The turn is queued with agent.add_message, which is async, so the handler
# cannot own it: it leaves the call in data["agent_message"] as
# (text, kind, data) and each front-end runs it - the TUI in a task, the REPL
# with an await. One key, one shape, for every command that queues a turn.


def _queued_turn(text: str, kind: str, data: dict, message: str) -> CommandResult:
    """A CommandResult whose tail queues one agent turn."""
    return CommandResult(True, message, {"agent_message": (text, kind, data)})


def execute_wrapup(agent, args: str) -> CommandResult:
    """Execute /wrapup - queue the wrap-up turn (journal + current-truth).

    Optional argument: the focus name (rest of line, may contain spaces),
    which becomes the authoritative focus name for the {focus} files even if
    it was never named in the conversation.
    """
    arg = args.strip()
    if not arg:
        return _queued_turn("/wrapup", "wrapup", {}, "Wrap-up queued")
    return _queued_turn(
        f"/wrapup {arg}",
        "wrapup",
        {"focus_name": arg},
        f"Wrap-up queued (focus: {arg})",
    )


def execute_status(agent, args: str = "") -> CommandResult:
    """Execute /status - the session's state and settings as plain text.

    Everything comes off the agent: gather_status() fills the core fields and
    the front-end keeps the rest (last turn, resumed stats, display flags) on
    the agent as it runs. The argument is ignored.
    """
    import time as _time
    from datetime import datetime

    from agent13.status import fmt_tokens, format_duration, gather_status

    sd = gather_status(agent)

    # Last turn: the front-end records when a turn ended.
    if agent.last_turn_duration is not None:
        sd.last_turn_duration = format_duration(agent.last_turn_duration)
        if agent.last_turn_end_time is not None:
            ago = format_duration(_time.time() - agent.last_turn_end_time)
            sd.last_turn_ago = f"{ago} ago"

    sd.resumed_turn_count = agent.resumed_turn_count
    sd.resumed_saved_at = agent.resumed_saved_at
    sd.resumed_prompt_tokens = agent.resumed_prompt_tokens
    sd.resumed_completion_tokens = agent.resumed_completion_tokens

    lines = ["Session", f"  status: {sd.agent_status}", f"  run time: {sd.run_time}"]
    if sd.session_started_at:
        try:
            start = datetime.fromisoformat(sd.session_started_at)
            age = format_duration(_time.time() - start.timestamp())
            lines.append(
                f"  session started: {start.strftime('%Y-%m-%d %H:%M')} "
                f"({age} ago)"
            )
        except (ValueError, OSError):
            pass
    lines.append(f"  total chains: {sd.chain_count}")
    lines.append(f"  cwd: {sd.cwd}")
    if sd.goal_line:
        lines.append(f"  {sd.goal_line}")
    if sd.focus_name:
        lines.append(f"  focus: {sd.focus_name}")
    if sd.last_turn_duration:
        lines.append(
            f"  last turn: {sd.last_turn_duration} (completed {sd.last_turn_ago})"
        )
    if sd.turn_count > 0:
        lines.append(
            f"  turns: {sd.turn_count} | total processing: {sd.total_processing}"
        )

    if sd.resumed_turn_count > 0:
        lines += ["", "Resumed session"]
        if sd.resumed_saved_at:
            try:
                saved = datetime.fromisoformat(sd.resumed_saved_at)
                ago = format_duration(_time.time() - saved.timestamp())
                lines.append(
                    f"  saved: {saved.strftime('%Y-%m-%d %H:%M')} ({ago} ago)"
                )
            except (ValueError, OSError):
                pass
        prev_p = fmt_tokens(sd.resumed_prompt_tokens)
        prev_c = fmt_tokens(sd.resumed_completion_tokens)
        prev_t = fmt_tokens(sd.resumed_prompt_tokens + sd.resumed_completion_tokens)
        lines.append(
            f"  previous: {sd.resumed_turn_count} turns | "
            f"{prev_p} prompt / {prev_c} completion / {prev_t} total"
        )
        delta_p = fmt_tokens(sd.prompt_tokens - sd.resumed_prompt_tokens)
        delta_c = fmt_tokens(sd.completion_tokens - sd.resumed_completion_tokens)
        delta_t = fmt_tokens(
            (sd.prompt_tokens - sd.resumed_prompt_tokens)
            + (sd.completion_tokens - sd.resumed_completion_tokens)
        )
        lines.append(
            f"  this session: {sd.turn_count} turns | "
            f"{delta_p} prompt / {delta_c} completion / {delta_t} total"
        )

    try:
        from agent13.config import get_config
        from agent13.persistence import count_saves, find_latest_auto_save

        auto_count, manual_count = count_saves()
        loc_label = "local" if get_config().saves_location == "local" else "central"
        lines += ["", "Saves"]
        lines.append(f"  auto: {auto_count} ({loc_label}) | manual: {manual_count}")
        latest = find_latest_auto_save()
        if latest:
            try:
                with open(latest) as f:
                    saved_at = json.load(f).get("saved_at")
                if saved_at:
                    saved_dt = datetime.fromisoformat(saved_at)
                    lines.append(
                        f"  restorable: yes ({saved_dt.strftime('%Y-%m-%d %H:%M')})"
                    )
                else:
                    lines.append("  restorable: yes")
            except (OSError, json.JSONDecodeError, ValueError):
                lines.append("  restorable: yes")
        else:
            lines.append("  restorable: no auto-save found")
    except OSError:
        pass

    bell = _bell_manager(agent)
    bell_text = bell.status_text() if bell is not None else "off"
    bell_cmd = (bell.command if bell is not None and bell.command else "") or (
        "(terminal bell)"
    )

    tools_str = f"{sd.tool_successes}/{sd.tool_calls}" if sd.tool_calls > 0 else "0"
    sandbox_suffix = (
        f" (pinned: {sd.sandbox_pinned})" if sd.sandbox_pinned is not None else ""
    )
    devel_suffix = (
        f" (pinned: {'on' if sd.devel_pinned else 'off'})"
        if sd.devel_pinned is not None
        else ""
    )
    auto_ctx = (
        "off"
        if sd.auto_context_threshold == 0
        else f"{sd.auto_context_threshold:,} tokens"
    )
    idle_save = (
        "off" if sd.auto_save_idle_seconds <= 0 else f"{sd.auto_save_idle_seconds}s idle"
    )
    response_format = "json" if getattr(agent, "response_format", None) else "raw"

    lines += [
        "",
        "Provider",
        f"  provider: {sd.provider}",
        f"  model: {sd.model}",
        f"  prompt: {sd.active_prompt}",
        "",
        "Context",
        f"  prompt tokens: {sd.prompt_tokens_fmt}",
        f"  completion tokens: {sd.completion_tokens_fmt}",
        f"  total tokens: {sd.total_tokens_fmt}",
        f"  queue: {sd.queue_count}  messages: {sd.message_count}",
        "",
        "Connectivity",
        f"  mcp: {sd.mcp_status}",
        "",
        "Tools",
        f"  success/calls: {tools_str}",
        f"  bounded: {sd.bounded_count}",
        "",
        "Settings",
        f"  sandbox: {sd.sandbox_mode}{sandbox_suffix}",
        f"  tool-response: {response_format}",
        f"  spinner: {agent.spinner_speed}",
        f"  clipboard: {agent.clipboard_method}",
        f"  bell: {bell_text}",
        f"  bell-cmd: {bell_cmd}",
        f"  remove-api-reasoning: {'on' if sd.remove_reasoning else 'off'}",
        f"  reasoning widgets: "
        f"{'collapsed' if agent.reasoning_collapsed else 'expanded'}",
        f"  tool widgets: {'collapsed' if agent.toolview_collapsed else 'expanded'}",
        f"  devel: {'on' if sd.devel_mode else 'off'}{devel_suffix}",
        f"  skill tool: {'on' if sd.skills_mode else 'off'}",
        f"  journal: {'on' if sd.journal_mode else 'off'}",
        f"  auto-context: {auto_ctx}",
        f"  auto-action: {sd.auto_context_action}",
        f"  auto-chain: {sd.auto_context_chain}",
        f"  chains: {sd.auto_context_chain_used}/{sd.auto_context_chain}",
        f"  nudges: {sd.nudges}",
        f"  idle-save: {idle_save}",
    ]
    return CommandResult(True, "\n".join(lines), pane="status")


def execute_compact(agent, args: str) -> CommandResult:
    """Execute /compact - queue a compaction turn for the whole history.

    The argument is either a focus for the next task (free text) or
    ``--prompt <name>``; resolve_compact_prompt owns that split and the
    error wording.
    """
    from agent13.prompts import resolve_compact_prompt

    arg = args.strip()
    compact_prompt_text, error = resolve_compact_prompt(
        getattr(agent, "prompt_manager", None), arg
    )
    if error:
        return CommandResult(False, error)

    display = f"/compact {arg}" if arg else "/compact"
    return _queued_turn(
        display,
        "compact",
        {
            "compact_prompt": compact_prompt_text,
            "user_specified_prompt": bool(arg),
        },
        "Compaction queued",
    )


def execute_chain(agent, args: str) -> CommandResult:
    """Execute /chain - queue a manual chain restart.

    The restart runs wrap-up (if stale) -> clear history -> inject records;
    no AI call when the wrap-up is already current.

    Args:
        args: "no-skills" (also accepted as --no-skills / no_skills) drops
            manually-loaded skills from this restart AND forgets them; they
            are one /skillname away if still needed.
    """
    no_skills = (
        args.strip().lower().lstrip("-").replace("-", "").replace("_", "")
        == "noskills"
    )
    if not no_skills:
        return _queued_turn("/chain", "chain", {}, "Chain restart queued")
    return _queued_turn(
        "/chain",
        "chain",
        {"no_skills": True},
        "Chain restart queued (loaded skills dropped)",
    )


# ---------------------------------------------------------------------------
# /journal - shared command logic
# ---------------------------------------------------------------------------

_JOURNAL_USAGE = (
    "Usage: /journal [on|off|last|all|status]\n"
    "  /journal on      - Enable context compaction\n"
    "  /journal off     - Disable context compaction\n"
    "  /journal last    - Journal the most recent tool-using turn\n"
    "  /journal all     - Journal all tool-using turns iteratively\n"
    "  /journal status  - Show current state"
)


def execute_journal(agent, args: str) -> CommandResult:
    """Execute /journal - toggle journal mode, or journal past turns.

    on/off flip `agent.journal_mode` (the status bar shows it, hence
    refresh_status). last/all cannot run here: they queue an agent message,
    which is async - the front-end schedules it from
    data["agent_message"] = (text, kind, data).
    """
    args = args.strip().lower()

    if args == "on":
        agent.journal_mode = True
        return CommandResult(
            True,
            "Journal mode enabled\n"
            "Context will be compacted via reflection before each new message.",
            refresh_status=True,
        )

    if args == "off":
        agent.journal_mode = False
        return CommandResult(
            True, "Journal mode disabled", refresh_status=True
        )

    if args == "last":
        return _queued_turn(
            "/journal last",
            "journal_last",
            {},
            "Journaling the most recent tool-using turn",
        )

    if args == "all":
        return _queued_turn(
            "/journal all", "journal_all", {}, "Journaling all tool-using turns"
        )

    status = "on" if agent.journal_mode else "off"
    if args in ("", "status"):
        return CommandResult(True, f"Journal mode: {status}")

    return CommandResult(False, f"{_JOURNAL_USAGE}\n  Current: {status}")


# ---------------------------------------------------------------------------
# /sandbox, /devel - shared command logic
# ---------------------------------------------------------------------------

_SANDBOX_USAGE = (
    "  /sandbox <mode>  set session mode\n"
    "  /sandbox pin     pin current mode for this project\n"
    "  /sandbox unpin   remove pin for this project"
)

_DEVEL_USAGE = (
    "Usage: /devel [on|off|status|pin|unpin]\n"
    "  /devel on      - Show devel-group tools to the AI\n"
    "  /devel off     - Hide devel-group tools from the AI\n"
    "  /devel status  - Show current state\n"
    "  /devel pin     - Pin current state for this project\n"
    "  /devel unpin   - Remove pin for this project"
)


def execute_sandbox(agent, args: str) -> CommandResult:
    """Execute /sandbox - show or set the sandbox mode, pin it per project.

    Forms:
      /sandbox <mode> - set the session override
      /sandbox pin    - pin the current mode for this project
      /sandbox unpin  - remove the project pin
      /sandbox        - show effective mode, override, default and pin
    """
    from agent13.sandbox import (
        format_all_sandbox_modes,
        get_default_sandbox_mode,
        get_pinned_sandbox_mode,
        parse_sandbox_mode,
        pin_sandbox_mode,
        unpin_sandbox_mode,
    )
    from tools.security import (
        get_current_sandbox_mode,
        get_session_sandbox_mode,
        set_session_sandbox_mode,
    )

    mode_str = args.strip().lower()

    if not mode_str:
        current = get_current_sandbox_mode()
        session = get_session_sandbox_mode()
        config_default = get_default_sandbox_mode()
        pinned = get_pinned_sandbox_mode()
        lines = [
            "Sandbox Configuration:",
            f"  Current mode: {current.value}",
            (
                f"  Session override: {session.value}"
                if session
                else "  Session override: none (using config default)"
            ),
            f"  Config default: {config_default.value}",
            f"  Pinned: {pinned.value if pinned else 'no'}",
            "",
            format_all_sandbox_modes(),
            "",
            _SANDBOX_USAGE,
        ]
        return CommandResult(True, "\n".join(lines))

    if mode_str == "pin":
        current = get_current_sandbox_mode()
        pin_sandbox_mode(current)
        return CommandResult(
            True,
            f"Pinned sandbox mode '{current.value}' for this project\n"
            "This mode will auto-apply on startup in this directory.",
        )

    if mode_str == "unpin":
        if unpin_sandbox_mode():
            return CommandResult(True, "Removed sandbox pin for this project")
        return CommandResult(False, "No sandbox pin exists for this project")

    try:
        mode = parse_sandbox_mode(mode_str)
    except ValueError as e:
        return CommandResult(False, f"Error: {e}\n{_SANDBOX_USAGE}")
    set_session_sandbox_mode(mode)
    return CommandResult(True, f"Sandbox mode set to: {mode.value}")


def execute_devel(agent, args: str) -> CommandResult:
    """Execute /devel - show or toggle devel-group tool visibility.

    Forms:
      /devel on|off   - show/hide devel-group tools for this session
      /devel pin      - pin the CURRENT state for this project
      /devel unpin    - remove the project pin
      /devel [status] - show the session state and the pin
    """
    from agent13.pins import get_pinned_devel, pin_devel, unpin_devel

    args = args.strip().lower()

    if args in ("on", "off"):
        agent.set_devel_mode(args == "on")
        if args == "on":
            return CommandResult(
                True,
                "Devel mode enabled\n"
                "Devel-group tools are now visible to the AI.",
                refresh_status=True,
            )
        return CommandResult(
            True,
            "Devel mode disabled\n"
            "Devel-group tools are now hidden from the AI.",
            refresh_status=True,
        )

    if args == "pin":
        enabled = agent.devel_mode
        pin_devel(enabled)
        return CommandResult(
            True,
            f"Pinned devel mode '{'on' if enabled else 'off'}' for this project\n"
            "Devel mode will auto-apply on startup in this directory.",
        )

    if args == "unpin":
        if unpin_devel():
            return CommandResult(True, "Removed devel pin for this project")
        return CommandResult(False, "No devel pin exists for this project")

    status = "on" if agent.devel_mode else "off"
    pinned = get_pinned_devel()
    pinned_text = ("on" if pinned else "off") if pinned is not None else "no"

    if args in ("", "status"):
        return CommandResult(True, f"Devel mode: {status}\n  Pinned: {pinned_text}")

    return CommandResult(False, f"{_DEVEL_USAGE}\n  Current: {status}")


# ---------------------------------------------------------------------------
# /queue — shared command logic
# ---------------------------------------------------------------------------


@dataclass
class QueueItemDisplay:
    """Display representation of a queue item."""

    index: int
    text: str
    interrupt: bool
    priority: bool
    running: bool


def format_queue_items(queue) -> list[QueueItemDisplay]:
    """Format queue items for display.

    The currently running item (if any) is included first, WITHOUT a
    deletable index number (index=0).  Pending items are then numbered
    1..N so that the numbers shown to the user match the indices accepted
    by ``/delete q N`` (which operates on pending items only).
    Returns structured data — each UI renders in its own format.
    """
    result = []

    # Running item: shown as a header, not numbered (not deletable)
    if queue.current:
        result.append(
            QueueItemDisplay(
                index=0,
                text=queue.current.text,
                interrupt=queue.current.interrupt,
                priority=queue.current.priority,
                running=True,
            )
        )

    # Pending items numbered 1..N — matches /delete q indexing
    for i, item in enumerate(queue.list_items(), start=1):
        result.append(
            QueueItemDisplay(
                index=i,
                text=item.text,
                interrupt=item.interrupt,
                priority=item.priority,
                running=False,
            )
        )

    return result


def execute_queue(agent, args: str = "", polite_waiting: bool = False) -> CommandResult:
    """Execute /queue — list the running item and the pending queue.

    The running item is an unnumbered header line (it cannot be deleted);
    pending items are numbered 1..N so the numbers match ``/delete q N``.
    Item text is whitespace-collapsed but never truncated: both front-ends
    wrap, and the text is the user's own.

    ``args`` is accepted for the handler signature and ignored - /queue takes
    no arguments. ``polite_waiting`` is not a command form: the TUI passes it
    from its polite-lock tick to prepend the "Waiting for lock" header.
    """
    items = format_queue_items(agent.queue)
    if not items:
        return CommandResult(True, "Queue is empty", pane="queue")

    pending_count = sum(1 for it in items if not it.running)
    lines = []
    if polite_waiting:
        lines.append("Waiting for lock")
    lines.append(f"Queue ({pending_count} pending):")
    for item in items:
        if item.interrupt:
            marker = "!! "
        elif item.priority:
            marker = "!  "
        else:
            marker = "   "
        text = " ".join(item.text.split())
        if item.running:
            lines.append(f"  {marker}→ {text} (running)")
        else:
            lines.append(f"  {marker}{item.index}. {text}")
    return CommandResult(True, "\n".join(lines), pane="queue")


def promote_latest(agent, interrupt: bool) -> str:
    """Promote the most recently queued item to steer (``!!``) or priority (``!``).

    Typing the prefix ALONE is a promotion, not an empty message: the user
    has already queued something and is now saying "run that next" (``!``)
    or "steer it into the running turn" (``!!``). Before this, a bare ``!!``
    queued an EMPTY interrupt item, which the turn loop then injected as a
    blank user message.

    Returns the one-line status both front-ends display. The rule and the
    wording live here so the TUI and the REPL cannot drift.
    """
    tag = "[steer]" if interrupt else "[priority]"
    item = agent.queue.latest_pending()
    if item is None:
        return f"{tag} nothing queued"
    if interrupt and item.interrupt:
        return f"{tag} already steering"
    if not interrupt and item.priority:
        return f"{tag} already priority"

    if interrupt:
        agent.queue.set_interrupt(item.id, True)
    else:
        agent.queue.set_priority(item.id, True)
    return f'{tag} "{" ".join(item.text.split())}"'


# ---------------------------------------------------------------------------
# /history — shared data extraction
# ---------------------------------------------------------------------------


@dataclass
class HistoryEntry:
    """A single entry within a history group."""

    role: str
    content: str
    tool_calls: list[dict]  # [{name, arguments}]
    is_interrupt: bool
    is_injected: bool = False  # native-vision image injection (mid-turn)


@dataclass
class HistoryGroup:
    """A group of related messages (user prompt + responses)."""

    number: int
    first_role: str
    first_content: str
    entries: list[HistoryEntry]


def format_history_groups(agent) -> list[HistoryGroup]:
    """Format message history as grouped entries for display.

    Returns structured data — each UI renders in its own format.
    Content is flattened to text, so multimodal (image) messages never leak
    content-block lists into the UIs.
    """
    from agent13.message_history import content_to_text, is_injected

    groups = agent.history.get_message_groups()
    result = []
    for group_num, group in enumerate(groups, 1):
        first_idx = group[0]
        first_msg = agent.messages[first_idx]
        entries = []
        for idx in group[1:]:
            msg = agent.messages[idx]
            is_user = msg.get("role") == "user"
            entries.append(
                HistoryEntry(
                    role=msg.get("role", "unknown"),
                    content=content_to_text(msg.get("content", "")),
                    tool_calls=[
                        {
                            "name": tc.get("function", {}).get("name", "?"),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        }
                        for tc in msg.get("tool_calls", [])
                    ],
                    is_interrupt=is_user and msg.get("interrupt", False),
                    is_injected=is_user and is_injected(msg),
                )
            )
        result.append(
            HistoryGroup(
                number=group_num,
                first_role=first_msg.get("role", "unknown"),
                first_content=content_to_text(first_msg.get("content", "")),
                entries=entries,
            )
        )
    return result


def _collapse(text, max_len: int = 120) -> str:
    """Collapse whitespace and truncate - one listing entry per line."""
    text = " ".join(str(text).split())
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# The most informative argument per tool, for the /history tool-call line.
_TOOL_KEY_ARGS = {
    "command": "command",
    "read_file": "filepath",
    "write_file": "filepath",
    "edit_file": "filepath",
    "skill": "name",
    "square_number": "x",
}


def format_tool_call_preview(name: str, args_json: str) -> str:
    """Extract the key argument from a tool call as one plain line.

    Tool arguments (e.g. edit_file's ``content``) routinely contain
    newlines that would corrupt the one-entry-per-line history layout.
    """
    try:
        args = json.loads(args_json) if args_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    key = _TOOL_KEY_ARGS.get(name)
    if key and key in args:
        preview = str(args[key])
    elif args:
        preview = str(next(iter(args.values()), ""))
    else:
        return name
    return f"{name} \u2192 {_collapse(preview, 60)}"


def format_tool_result_preview(content) -> str:
    """Summarise a tool result message as one plain line.

    Tool result messages (especially edit_file's ``Replaced ... of '<find>'
    with '<content>'``) embed raw user content that frequently contains
    newlines - collapsing here keeps the history listing readable.
    """
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not content:
        return "tool result (empty)"

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _collapse(content, 80)

    if not isinstance(data, dict):
        return _collapse(str(data), 80)

    ok = data.get("success", not data.get("error"))
    tag = "\u2713" if ok else "\u2717"

    if "exit_code" in data:
        exit_code = data.get("exit_code", "?")
        stdout = _collapse(data.get("stdout") or "", 60)
        stderr = _collapse(data.get("stderr") or "", 60)
        if exit_code == 0:
            return f"{tag} exit {exit_code}: {stdout}"
        return f"{tag} exit {exit_code}: {stderr or stdout}"

    if "filepath" in data and "total_lines" in data:
        # read_file result
        return (
            f"{tag} {data.get('filepath', '')} "
            f"({data.get('total_lines', 0)} lines, {data.get('view', '')})"
        )

    if data.get("error"):
        return f"{tag} {_collapse(data['error'], 80)}"

    if "message" in data:
        # write_file / edit_file success
        return f"{tag} {_collapse(data['message'], 80)}"

    if "skill_name" in data:
        return f"{tag} Skill loaded: {data.get('skill_name', '?')}"

    if data:
        k, v = next(iter(data.items()))
        return f"{tag} {_collapse(f'{k}: {v}', 80)}"

    return f"{tag} (empty result)"


def execute_new_session(agent, args: str = "") -> CommandResult:
    """Execute /new_session - wipe history and start a new session identity.

    Args are ignored (there are none).

    Destructive, so the wipe is always deferred: it goes through the queue and
    lands at a safe boundary, behind anything already waiting. That is why the
    handler does not clear anything itself - it only asks for the clear
    (data["clear_mode"]) and says so. Each front-end owns the scheduling,
    because the TUI resets its chat display from the MESSAGES_CLEARED event
    rather than from this command, and the REPL has to print the confirmation
    above a redrawn prompt.

    "Idle" is not a safe shortcut: agent.is_idle looks at the running turn, not
    at the queue, so an inline wipe could jump ahead of queued work.
    """
    return CommandResult(
        True,
        "New session queued (history + counters reset at the next safe point)",
        {"clear_mode": "all"},
    )


def execute_load(agent, args: str) -> CommandResult:
    """Execute /load - validate a save name and ask for a deferred load.

    No arg lists the available saves and the usage.

    The load always goes through the queue (data["load_path"]) so it lands at a
    safe boundary, behind anything already waiting - the same reasoning as
    /new_session. Each front-end runs it: the TUI records the path so it can
    capture the resumed token stats, the REPL just awaits.
    """
    from agent13.persistence import resolve_save_path

    name = args.strip()
    if not name:
        names = list_save_names()
        lines = ["Available saves:"] if names else ["No saves found"]
        lines.extend(f"  {n}" for n in names)
        lines.append("")
        lines.append("Usage: /load <name>")
        return CommandResult(True, "\n".join(lines))

    path = resolve_save_path(name)
    if not path.exists():
        return CommandResult(False, f"Not found: {path}\nUse /save {name} to create it")

    return CommandResult(
        True,
        "Load queued (will take effect after current response)",
        {"load_path": str(path)},
    )


def execute_history(agent, args: str = "") -> CommandResult:
    """Execute /history - the grouped message history as a plain listing.

    Args are ignored (there are none yet).
    """
    if not agent.messages:
        return CommandResult(True, "No messages in history", pane="history")

    groups = format_history_groups(agent)
    count = f"{len(groups)} group" + ("" if len(groups) == 1 else "s")
    lines = [f"Message history ({count}):", ""]

    for g in groups:
        first = _collapse(g.first_content) if g.first_content else "(empty)"
        lines.append(f"{g.number:3d}. {g.first_role}: {first}")
        for entry in g.entries:
            if entry.role == "tool":
                lines.append(
                    f"     tool result: {format_tool_result_preview(entry.content)}"
                )
            elif entry.role == "assistant":
                # An assistant message with tool calls often has no text.
                text = _collapse(entry.content) if entry.content else ""
                if text:
                    lines.append(f"     assistant: {text}")
                for tc in entry.tool_calls:
                    preview = format_tool_call_preview(tc["name"], tc["arguments"])
                    lines.append(f"     tool call: {preview}")
            elif entry.is_interrupt:
                lines.append(f"     interrupt: {_collapse(entry.content) or '(empty)'}")
            elif entry.is_injected:
                lines.append(f"     injected: {_collapse(entry.content) or '(empty)'}")
            else:
                text = _collapse(entry.content) if entry.content else ""
                if text:
                    lines.append(f"     {entry.role}: {text}")

    lines.append("")
    lines.append("Use /delete h N to delete group N")
    return CommandResult(True, "\n".join(lines), pane="history")


def list_save_names() -> list[str]:
    """List available save names (stems without .ctx extension).

    Searches both the project-local and central saves dirs — manual saves
    written under a central saves_location live in the central dir.
    """
    from agent13.persistence import list_all_saves

    return [s.stem for s in list_all_saves()]


# ---------------------------------------------------------------------------
# /reasoning and /toolview — widget folding (display only)
# ---------------------------------------------------------------------------

_COLLAPSE_LABELS = {
    "reasoning": "Reasoning widgets (Thoughts / Reflection)",
    "toolview": "Tool call / result widgets",
}


def _collapse_helpers(kind: str):
    """(get_pinned, pin, unpin) for a collapse kind, from the pin store."""
    from agent13 import pins

    return {
        "reasoning": (
            pins.get_pinned_reasoning,
            pins.pin_reasoning,
            pins.unpin_reasoning,
        ),
        "toolview": (
            pins.get_pinned_toolview,
            pins.pin_toolview,
            pins.unpin_toolview,
        ),
    }[kind]


def execute_collapse(
    kind: str, args: str, collapsed: bool, set_collapsed
) -> CommandResult:
    """Execute /reasoning and /toolview — fold or expand a chat widget family.

    Forms:
      /kind hide  - fold new widgets to a one-line header (session)
      /kind show  - expand new widgets (session)
      /kind pin   - pin the CURRENT setting for this project (startup default)
      /kind unpin - remove the project pin
      /kind       - show status (effective value, pin, config default)

    Args:
        kind: "reasoning" or "toolview".
        args: Raw argument string.
        collapsed: The current effective setting (True = folded).
        set_collapsed: Callable applying a session change to the UI.

    Widgets already in the chat are never re-rendered - the setting applies
    to widgets created from now on (including a /rebuild, which creates them
    afresh).

    Returns CommandResult with data['collapsed'] when the value changed.
    """
    label = _COLLAPSE_LABELS[kind]
    get_pinned, pin, unpin = _collapse_helpers(kind)
    usage = (
        f"Usage: /{kind} hide | show | pin | unpin\n"
        f"  hide  - fold to a one-line header\n"
        f"  show  - expand\n"
        f"  pin   - keep the current setting for this project\n"
        f"  unpin - remove the project pin\n"
        f"  Current: {'collapsed' if collapsed else 'expanded'}"
    )
    args = args.strip().lower()

    if args in ("hide", "show"):
        want = args == "hide"
        set_collapsed(want)
        state = "collapsed" if want else "expanded"
        return CommandResult(
            True, f"{label}: {state} for this session", {"collapsed": want}
        )

    if args == "pin":
        pin(collapsed)
        state = "collapsed" if collapsed else "expanded"
        return CommandResult(
            True,
            f"Pinned '{state}' for this project - applies on startup here",
        )

    if args == "unpin":
        if unpin():
            return CommandResult(True, f"Removed the {kind} pin for this project")
        return CommandResult(False, f"No {kind} pin exists for this project")

    if not args:
        from agent13.config import get_config

        state = "collapsed" if collapsed else "expanded"
        pinned = get_pinned()
        pinned_text = (
            ("collapsed" if pinned else "expanded")
            if pinned is not None
            else "no"
        )
        cfg_default = getattr(get_config(), f"{kind}_collapsed", True)
        lines = [
            f"{label}: {state}",
            f"  Pinned for this project: {pinned_text}",
            f"  Config default: {'collapsed' if cfg_default else 'expanded'}",
        ]
        if kind == "reasoning":
            lines.append(
                "  [Display only - /remove-api-reasoning controls what the API gets]"
            )
        return CommandResult(True, "\n".join(lines))

    return CommandResult(False, f"Unknown {kind} option: {args}\n{usage}")


def execute_pretty(args: str, pretty: bool, set_pretty) -> CommandResult:
    """Execute /pretty - toggle markdown rendering for the transcript.

    Forms:
      /pretty      - toggle
      /pretty on   - markdown rendering
      /pretty off  - plain text

    Args:
        args: Raw argument string.
        pretty: The current setting (True = markdown).
        set_pretty: Callable applying the change to this front-end's renderer.

    The setting lives in the front-end's renderer, not on the agent, so this is
    a shared helper rather than a registry handler - the same shape as
    execute_collapse. It applies to output from now on; messages already on
    screen are not re-rendered.
    """
    arg = args.strip().lower()
    if not arg:
        want = not pretty
    elif arg == "on":
        want = True
    elif arg == "off":
        want = False
    else:
        return CommandResult(
            False,
            "Usage: /pretty [on|off]\n"
            "  /pretty     - toggle\n"
            "  /pretty on  - markdown rendering\n"
            "  /pretty off - plain text\n"
            f"  Current: {'on' if pretty else 'off'}",
        )

    set_pretty(want)
    return CommandResult(
        True, f"Pretty mode: {'on' if want else 'off'}", {"pretty": want}
    )


# ---------------------------------------------------------------------------
# Command registry - the one index of every slash command
# ---------------------------------------------------------------------------
#
# A command is *shared* when its work is the same in the TUI and the REPL: it
# acts on the agent/config and returns a CommandResult. Shared commands carry a
# handler here and both front-ends run them through run_command().
#
# A command whose work is about this front-end's own screen or input (clearing
# the chat widget, multi-line entry, spinner animation, ...) carries no handler.
# It is still listed, so /help and Tab are complete from one source, and each
# front-end supplies the body. unimplemented() reports the gaps.

HELP_GROUPS = (
    "Session",
    "Queue",
    "Context",
    "Config",
    "Tools",
    "Display",
    "Updates",
)


@dataclass(frozen=True)
class Command:
    """One slash command: its catalogue entry and, when shared, its handler.

    name:        command without the leading slash
    description: one line for /help and completion
    group:       /help section, one of HELP_GROUPS
    handler:     (agent, args) -> CommandResult, or None when front-end-local
    completions: static options, or (agent, partial, full_text) -> options
    detail:      extra form lines shown under the entry in /help
    aliases:     alternative spellings accepted on input
    """

    name: str
    description: str
    group: str = "Session"
    handler: Optional[Callable[[Any, str], CommandResult]] = None
    completions: Optional[
        Union[list[str], Callable[[Any, str, str], list[str]]]
    ] = None
    detail: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


# --- completion helpers ----------------------------------------------------


def _prefix_filter(options: list[str], partial: str) -> list[str]:
    """Filter options by case-insensitive prefix."""
    if not partial:
        return list(options)
    p = partial.lower()
    return [o for o in options if o.lower().startswith(p)]


def _complete_model(agent, partial: str, full_text: str = "") -> list[str]:
    """Model names, or the name a number selects."""
    models = list(getattr(agent, "available_models", None) or [])
    if partial.isdigit():
        idx = int(partial) - 1
        return [models[idx]] if 0 <= idx < len(models) else []
    return _prefix_filter(models, partial)


def _complete_provider(agent, partial: str, full_text: str = "") -> list[str]:
    """Provider names from config, or the name a number selects."""
    from agent13.config import get_config

    providers = [p.name for p in get_config().providers]
    if partial.isdigit():
        idx = int(partial) - 1
        return [providers[idx]] if 0 <= idx < len(providers) else []
    return _prefix_filter(providers, partial)


def _complete_save(agent, partial: str, full_text: str = "") -> list[str]:
    """Manual save names, newest first."""
    from agent13.persistence import list_saves

    try:
        saves = sorted(list_saves(), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []
    return _prefix_filter([s.stem for s in saves], partial)


def _complete_load(agent, partial: str, full_text: str = "") -> list[str]:
    """Manual save names then auto-saves, newest first."""
    from agent13.debug_log import log_event
    from agent13.persistence import list_all_saves

    try:
        saves = list_all_saves()
    except Exception as e:
        # Surface the failure where the user's eye already is - a silent empty
        # list reads as "no saves" and hides real errors (e.g. permissions).
        log_event("load_completions_failed", {"error": str(e)})
        return [f"saves list unavailable: {e}"[:100]]
    return _prefix_filter([s.stem for s in saves], partial)


def _complete_prompt(agent, partial: str, full_text: str = "") -> list[str]:
    """Prompt names (flat command, no subcommands)."""
    pm = getattr(agent, "prompt_manager", None)
    if pm is None:
        return []
    return _prefix_filter(list(pm.prompts.keys()), partial)


def _complete_snippet(agent, partial: str, full_text: str = "") -> list[str]:
    """Subcommands, then snippet names after delete/rename/use and 'add last'."""
    sm = getattr(agent, "snippet_manager", None)
    if sm is None:
        return []

    if not partial or " " not in partial:
        return _prefix_filter(["list", "add", "delete", "rename", "use"], partial)

    subcmd_part, arg_partial = partial.split(" ", 1)
    subcmd = subcmd_part.lower()
    names = list(sm.snippets.keys())

    if subcmd == "add":
        if not arg_partial:
            return ["last"]
        if "last".startswith(arg_partial.lower()):
            return ["last"]
        if arg_partial.startswith("last"):
            rest = arg_partial[4:].lstrip()
            if rest or arg_partial.rstrip() == "last":
                return _prefix_filter(names, rest)
        return []

    if subcmd in ("delete", "rename", "use"):
        return _prefix_filter(names, arg_partial)

    return []


def _complete_delete(agent, partial: str, full_text: str = "") -> list[str]:
    """Subcommands h/q/s, then save names after 's '."""
    if not partial or " " not in partial:
        return _prefix_filter(["h", "q", "s"], partial)
    subcmd, arg_partial = partial.split(" ", 1)
    if subcmd.lower() != "s":
        return []
    from agent13.persistence import list_saves

    try:
        saves = sorted(list_saves(), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []
    names = _prefix_filter([s.stem for s in saves], arg_partial)
    return [f"s {n}" for n in names]


def _complete_sandbox(agent, partial: str, full_text: str = "") -> list[str]:
    """Sandbox modes plus the pin/unpin subcommands."""
    from agent13.sandbox import SandboxMode

    return _prefix_filter([m.value for m in SandboxMode] + ["pin", "unpin"], partial)


# --- the catalogue ---------------------------------------------------------

_COMMAND_LIST = (
    # -- Session ------------------------------------------------------------
    Command(
        "help",
        "Show this help",
        "Session",
        detail=("Type /help <command> for that command's forms.",),
    ),
    Command(
        "status",
        "Show session status and settings",
        "Session",
        handler=execute_status,
    ),
    Command(
        "new_session",
        "Wipe message history and start a new session",
        "Session",
        handler=execute_new_session,
        detail=("Resets the compaction total and session start time.",),
    ),
    Command(
        "history",
        "Show message history (grouped)",
        "Session",
        handler=execute_history,
    ),
    Command(
        "save",
        "Save session to ./.agent13/saves/<name>.ctx",
        "Session",
        handler=execute_save,
        completions=_complete_save,
        detail=("Usage: /save <name> [-y]   (-y overwrites without asking)",),
    ),
    Command(
        "load",
        "Load a saved session",
        "Session",
        handler=execute_load,
        completions=_complete_load,
        detail=("Usage: /load <name>",),
    ),
    Command(
        "exit",
        "Exit",
        "Session",
        aliases=("quit",),
    ),
    Command(
        "multi",
        "Enter multi-line mode (end with . on its own line)",
        "Session",
        detail=("A trailing \\ on a line also continues it.",),
    ),
    Command("stop", "Interrupt the current response", "Session"),
    # -- Queue --------------------------------------------------------------
    Command("queue", "Show queued messages", "Queue", handler=execute_queue),
    Command("pause", "Pause agent processing", "Queue"),
    Command("resume", "Resume agent processing", "Queue"),
    Command("retry", "Retry the last message", "Queue", handler=execute_retry),
    Command(
        "prioritise",
        "Mark a queue item as priority",
        "Queue",
        handler=execute_prioritise,
        aliases=("prioritize",),
        detail=("Usage: /prioritise N",),
    ),
    Command(
        "deprioritise",
        "Remove priority from a queue item",
        "Queue",
        handler=execute_deprioritise,
        aliases=("deprioritize",),
        detail=("Usage: /deprioritise N",),
    ),
    Command(
        "delete",
        "Delete a history group, queue item, or save",
        "Queue",
        handler=execute_delete,
        completions=_complete_delete,
        detail=(
            "/delete h N     delete history group N",
            "/delete q N     delete queue item N",
            "/delete s NAME  delete a saved session",
            "N can be a number, 'last', a negative index (-1), or a range (1:3, -2:)",
        ),
    ),
    # -- Context ------------------------------------------------------------
    Command(
        "journal",
        "Context compaction via reflection",
        "Context",
        completions=["on", "off", "last", "all", "status"],
        handler=execute_journal,
    ),
    Command(
        "compact",
        "Compact history, steered toward your next task",
        "Context",
        completions=_complete_prompt,
        detail=(
            "/compact [next-task focus]",
            "/compact --prompt <name>",
        ),
        handler=execute_compact,
    ),
    Command(
        "wrapup",
        "Update journal + current-truth files now, then idle",
        "Context",
        handler=execute_wrapup,
        detail=("Usage: /wrapup [name]   (optional name = focus name)",),
    ),
    Command(
        "focus",
        "Set the focus name",
        "Context",
        handler=execute_focus,
        detail=(
            "/focus          show the current focus",
            "/focus <name>   set it (on an empty history, loads its records)",
        ),
    ),
    Command(
        "grind",
        "Work until a time window closes",
        "Context",
        handler=execute_grind,
        completions=["off", "clear"],
        detail=(
            "/grind <dur> [text]   work until the window closes (8h, 90m, 2h30m)",
            "/grind                show the window and the goal text",
            "/grind off            stop the window, keep the goal text",
            "/grind clear          stop the window and forget the goal text",
        ),
    ),
    Command(
        "chain",
        "Restart the loop: wrap-up (if stale) -> clear -> inject records",
        "Context",
        handler=execute_chain,
    ),
    Command(
        "auto_context_threshold",
        "Context threshold: act when context exceeds N tokens",
        "Context",
        handler=execute_auto_context_threshold,
        detail=(
            "/auto_context_threshold N    N supports a k suffix",
            "/auto_context_threshold 0    disable auto-context",
            "/auto_context_threshold      show current value",
        ),
    ),
    Command(
        "auto_context_action",
        "What to do at the threshold",
        "Context",
        handler=execute_auto_context_action,
        completions=["compact", "report_and_compact", "none"],
    ),
    Command(
        "auto_context_chain",
        "Restart the turn up to N times per turn after a compact",
        "Context",
        handler=execute_auto_context_chain,
    ),
    Command(
        "auto_context_nudge_cap",
        "Stop after N consecutive cut-off/empty outputs (0 = off)",
        "Context",
        handler=execute_auto_context_nudge_cap,
    ),
    # -- Config -------------------------------------------------------------
    Command(
        "model",
        "Select model",
        "Config",
        completions=_complete_model,
        detail=("Usage: /model [name|number]   (no args lists models)",),
        handler=execute_model,
    ),
    Command(
        "provider",
        "Switch provider",
        "Config",
        completions=_complete_provider,
        detail=("Usage: /provider <name|url>   (no args lists providers)",),
        handler=execute_provider,
    ),
    Command(
        "prompt",
        "Switch system prompt",
        "Config",
        completions=_complete_prompt,
        detail=("Usage: /prompt [name]   (no args lists prompts)",),
        handler=execute_prompt,
    ),
    Command(
        "sandbox",
        "Show/set sandbox mode, pin per-project",
        "Config",
        completions=_complete_sandbox,
        handler=execute_sandbox,
    ),
    Command(
        "devel",
        "Show devel-group tools, pin per-project",
        "Config",
        completions=["on", "off", "status", "pin", "unpin"],
        handler=execute_devel,
    ),
    Command(
        "skills",
        "Toggle/list skills mode",
        "Config",
        completions=["on", "off", "list", "status"],
        handler=execute_skills,
    ),
    Command(
        "polite",
        "Multi-agent lock coordination",
        "Config",
        completions=["off"],
        detail=(
            "/polite N    enable with poll interval N seconds",
            "/polite off  disable",
        ),
        handler=execute_polite,
    ),
    Command(
        "bell",
        "Ring the bell when a turn takes longer than N seconds",
        "Config",
        detail=(
            "/bell N     ring if a turn takes longer than N seconds",
            "/bell 0     always ring on return to idle",
            "/bell off   disable",
            "/bell       show current status",
        ),
        handler=execute_bell,
    ),
    Command(
        "bell-command",
        "Run an external command instead of the terminal bell",
        "Config",
        detail=(
            "/bell-command <cmd>  run <cmd> instead of the bell",
            "/bell-command off    revert to the terminal bell",
            "/bell-command        show current command",
        ),
        handler=execute_bell_command,
    ),
    Command(
        "remove-api-reasoning",
        "Strip reasoning tokens between turns (API context)",
        "Config",
        completions=["on", "off"],
        handler=execute_remove_api_reasoning,
    ),
    Command(
        "tool-response",
        "Set the tool response format",
        "Config",
        completions=["raw", "json"],
        handler=execute_tool_response,
    ),
    Command(
        "cwd",
        "Show or change working directory",
        "Config",
        handler=execute_cwd,
        detail=("Usage: /cwd [path]   (prefix @ for path completion)",),
    ),
    # -- Tools --------------------------------------------------------------
    Command("tools", "Show tool usage statistics", "Tools", handler=execute_tools),
    Command(
        "mcp",
        "List/manage MCP servers",
        "Tools",
        handler=execute_mcp,
        completions=["connect", "disconnect", "reload"],
    ),
    Command(
        "snippet",
        "Manage snippets",
        "Tools",
        completions=_complete_snippet,
        detail=(
            "/snippet list",
            "/snippet add last <name> [-y]   (saves the last user message)",
            "/snippet delete <name> | /snippet rename <old> <new> | /snippet use <name>",
        ),
        handler=execute_snippet,
    ),
    # -- Display ------------------------------------------------------------
    Command(
        "clear",
        "Clear the chat window (history preserved)",
        "Display",
        detail=(
            "/clear      clear the chat window",
            "/clear all  moved: use /new_session",
            "/clear N    moved: use /rebuild N",
        ),
    ),
    Command(
        "rebuild",
        "Rebuild the chat window from message history",
        "Display",
        detail=(
            "/rebuild    rebuild from the full history",
            "/rebuild N  rebuild showing the last N turns",
        ),
    ),
    Command("pretty", "Toggle markdown rendering", "Display", completions=["on", "off"]),
    Command(
        "reasoning",
        "Fold the Thoughts/Reflection widgets",
        "Display",
        completions=["hide", "show", "pin", "unpin"],
    ),
    Command(
        "toolview",
        "Fold the tool call/result widgets",
        "Display",
        completions=["hide", "show", "pin", "unpin"],
    ),
    Command(
        "spinner",
        "Spinner style and speed",
        "Display",
        completions=["fast", "slow", "off", "status"],
    ),
    Command(
        "clipboard",
        "Show or set clipboard method",
        "Display",
        completions=["osc52", "system"],
    ),
    # -- Updates ------------------------------------------------------------
    Command(
        "upgrade",
        "Check for updates and apply",
        "Updates",
        completions=["--copy"],
        detail=("Usage: /upgrade [--copy]   (--copy puts the command on the clipboard)",),
    ),
)


def _build_index(commands) -> dict[str, Command]:
    """Map every name and alias to its Command."""
    index: dict[str, Command] = {}
    for cmd in commands:
        index[cmd.name] = cmd
        for alias in cmd.aliases:
            index[alias] = cmd
    return index


COMMANDS: dict[str, Command] = _build_index(_COMMAND_LIST)


# --- registry interface ----------------------------------------------------


def get_command(name: str) -> Optional[Command]:
    """Look up a command by name or alias (no leading slash)."""
    return COMMANDS.get(name.lower())


def command_names() -> list[str]:
    """Every command name, aliases excluded, in catalogue order."""
    return [c.name for c in _COMMAND_LIST]


def run_command(name: str, agent, args: str) -> Optional[CommandResult]:
    """Run a shared command.

    Returns None when the command is front-end-local (no handler) or unknown -
    the caller then runs its own dispatch.
    """
    cmd = get_command(name)
    if cmd is None or cmd.handler is None:
        return None
    return cmd.handler(agent, args)


def unimplemented(local_names) -> list[str]:
    """Commands a front-end neither shares nor implements locally.

    Pass the names that front-end handles itself. An empty list means the
    front-end covers the whole catalogue.
    """
    local = set(local_names)
    return [
        c.name
        for c in _COMMAND_LIST
        if c.handler is None and c.name not in local
    ]


def complete_param(name: str, agent, partial: str, full_text: str = "") -> list[str]:
    """Tab completions for a command's arguments.

    Static lists are prefix-filtered here; callables filter themselves (they
    have the context to do it, e.g. a number selecting a model).
    """
    cmd = get_command(name)
    if cmd is None or cmd.completions is None:
        return []
    if callable(cmd.completions):
        return cmd.completions(agent, partial, full_text)
    return _prefix_filter(cmd.completions, partial)


@dataclass(frozen=True)
class HelpEntry:
    """One command as /help shows it."""

    name: str
    description: str
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class HelpSection:
    """A /help group."""

    title: str
    entries: list[HelpEntry]


def format_help_sections(local_names=None) -> list[HelpSection]:
    """The catalogue as /help sections, in HELP_GROUPS order.

    Pass the names this front-end implements itself to leave out commands it
    does not offer. Omit the argument for the full catalogue.
    """
    local = None if local_names is None else set(local_names)
    sections = []
    for group in HELP_GROUPS:
        entries = [
            HelpEntry(c.name, c.description, c.detail)
            for c in _COMMAND_LIST
            if c.group == group
            and (c.handler is not None or local is None or c.name in local)
        ]
        if entries:
            sections.append(HelpSection(group, entries))
    return sections
