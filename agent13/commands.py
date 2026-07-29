"""Shared command validation and execution for REPL and TUI.

Each function handles the logic of a slash command and returns a CommandResult.
UIs call these functions and render the result in their own format.
"""

from dataclasses import dataclass, field
from typing import Any

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
    """Result of a shared command operation."""

    success: bool
    message: str  # Plain text, no Rich markup
    data: dict[str, Any] = field(default_factory=dict)


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
        )
    else:
        return CommandResult(
            True,
            f"Deleted groups {indices[0]}-{indices[-1]} ({len(all_indices)} messages)",
            {"kind": "history"},
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


def execute_retry(agent) -> CommandResult:
    """Execute /retry — delete last message group and return user text for re-queueing.

    Sync: validates agent is idle, deletes the last group from agent.messages.
    Caller must await agent.add_message(result.data["user_text"]) to re-queue.

    Skips trailing priming pairs (left by journal compaction) so the priming
    prompt is never offered as retry text. Priming pairs are left in place;
    journal's sweep-on-next-run handles multiples.

    Returns CommandResult with data["user_text"] on success.
    """
    from agent13.prompts import PRIMING_PROMPT, PRIMING_RESPONSE

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
    user_text = agent.messages[first_msg_idx].get("content", "")

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
                True, "Polite mode not enabled", {"action": "disabled"}
            )
        agent.disable_polite()
        return CommandResult(True, "Polite mode disabled", {"action": "disabled"})

    # Parse interval N (float, must be non-negative).
    try:
        interval = float(args)
    except ValueError:
        return CommandResult(False, _POLITE_USAGE)
    if interval < 0:
        return CommandResult(False, f"Interval must be non-negative\n{_POLITE_USAGE}")

    agent.set_polite(interval=interval)
    return CommandResult(
        True,
        f"Polite mode enabled (interval {interval}s)",
        {"action": "enabled", "interval": interval},
    )


# ---------------------------------------------------------------------------
# /queue — shared data extraction
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
    """
    groups = agent.history.get_message_groups()
    result = []
    for group_num, group in enumerate(groups, 1):
        first_idx = group[0]
        first_msg = agent.messages[first_idx]
        entries = []
        for idx in group[1:]:
            msg = agent.messages[idx]
            entries.append(
                HistoryEntry(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    tool_calls=[
                        {
                            "name": tc.get("function", {}).get("name", "?"),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        }
                        for tc in msg.get("tool_calls", [])
                    ],
                    is_interrupt=msg.get("role") == "user"
                    and msg.get("interrupt", False),
                )
            )
        result.append(
            HistoryGroup(
                number=group_num,
                first_role=first_msg.get("role", "unknown"),
                first_content=first_msg.get("content", ""),
                entries=entries,
            )
        )
    return result


def list_save_names() -> list[str]:
    """List available save names (stems without .ctx extension)."""
    from agent13.persistence import list_saves

    return [s.stem for s in list_saves()]
