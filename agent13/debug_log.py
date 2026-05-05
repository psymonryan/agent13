"""Debug logging for agent13.

Provides JSONL (JSON Lines) logging for debugging agent/LLM interactions.
Each log entry is a single JSON line for easy parsing.

Usage:
    from agent13.debug_log import init_debug, log_event, log_error, is_debug_enabled

    # Initialize at startup with --debug flag
    init_debug()

    # Log events
    log_event("user_message", {"text": "Hello"})

    # Log errors with traceback
    try:
        ...
    except Exception as e:
        log_error(e, {"context": "api_call"})
"""

import hashlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Global debug state
_debug_enabled = False
_log_file: Optional[Path] = None

# Log rotation threshold (10 MB)
MAX_LOG_SIZE = 10_000_000


def init_debug(log_dir: Optional[Path] = None) -> None:
    """Initialize debug logging.

    Creates the log directory if needed, rotates old logs if too large,
    and writes a session_start event.

    Args:
        log_dir: Directory for debug.log (defaults to ~/.agent13/)
    """
    global _debug_enabled, _log_file
    _debug_enabled = True

    if log_dir is None:
        log_dir = Path.home() / ".agent13"

    log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = log_dir / "debug.log"

    # Rotate if too large
    if _log_file.exists() and _log_file.stat().st_size > MAX_LOG_SIZE:
        old_file = _log_file.with_suffix(".log.old")
        if old_file.exists():
            old_file.unlink()
        _log_file.rename(old_file)

    # Write session header
    log_event("session_start", {"pid": os.getpid()})


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled.

    Returns:
        True if debug logging is enabled, False otherwise
    """
    return _debug_enabled


def log_event(event: str, data: dict[str, Any], level: str = "DEBUG") -> None:
    """Log an event to the debug log.

    Args:
        event: Event name (e.g., "user_message", "tool_call")
        data: Event data dictionary
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    if not _debug_enabled or _log_file is None:
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "event": event,
        "data": data,
    }

    try:
        with open(_log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Never fail the app due to logging


def log_error(error: Exception, context: Optional[dict] = None) -> None:
    """Log an error with traceback.

    Args:
        error: The exception that occurred
        context: Optional context dictionary
    """
    log_event(
        "error",
        {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        },
        level="ERROR",
    )


def truncate_for_log(text: str, max_len: int = 1000) -> str:
    """Truncate text for logging.

    Args:
        text: Text to truncate
        max_len: Maximum length (default 1000)

    Returns:
        Truncated text with "... [N chars total]" suffix if truncated
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text)} chars total]"


def log_session_end() -> None:
    """Log session end event."""
    log_event("session_end", {"pid": os.getpid()})


def log_user_message(
    text: str, priority: bool = False, interrupt: bool = False, item_id: int = None
) -> None:
    """Log a user message event.

    Args:
        text: The message text
        priority: Whether it was a priority message
        interrupt: Whether it was an interrupt-level message
        item_id: Queue item ID if available
    """
    data = {
        "text": truncate_for_log(text),
        "priority": priority,
        "interrupt": interrupt,
    }
    if item_id is not None:
        data["item_id"] = item_id
    log_event("user_message", data)


def log_api_request(
    model: str, message_count: int, tool_count: int, params: dict = None
) -> None:
    """Log an API request event.

    Args:
        model: Model name
        message_count: Number of messages in request
        tool_count: Number of tools available
        params: Additional request parameters
    """
    data = {
        "model": model,
        "message_count": message_count,
        "tool_count": tool_count,
    }
    if params:
        data["params"] = params
    log_event("api_request", data)


def _short_hash(content: str) -> str:
    """Return first 8 hex chars of md5 hash for change detection."""
    return hashlib.md5(content.encode()).hexdigest()[:8]


def log_api_hash(
    system_prompt: str,
    tools: list[dict] | None,
    messages: list[dict],
) -> None:
    """Log per-component hashes of the API request payload for cache analysis.

    By comparing consecutive api_hash entries, you can find the exact point
    where the prompt prefix diverges — and thus where the server's LCP cache
    will break. Everything before the first differing hash should be in the
    KV cache; everything from that point onward must be reprocessed.

    Args:
        system_prompt: The system prompt string (after date injection)
        tools: The tools list (may be None/empty)
        messages: The api_messages list (as built by build_messages_with_system)
    """
    try:
        parts = [f"sys={_short_hash(system_prompt)}"]
        if tools:
            tools_json = json.dumps(tools, sort_keys=True)
            parts.append(f"tools={_short_hash(tools_json)}")
        else:
            parts.append("tools=none")
        for i, msg in enumerate(messages):
            msg_json = json.dumps(msg, sort_keys=True, default=str)
            parts.append(f"msg{i}={_short_hash(msg_json)}")
        log_event("api_hash", {"hashes": " ".join(parts)})
    except Exception as e:
        log_event("api_hash_error", {"error": str(e)})


def log_api_response(tokens_used: dict = None, finish_reason: str = None) -> None:
    """Log an API response event.

    Args:
        tokens_used: Dict with prompt_tokens, completion_tokens, total_tokens
        finish_reason: Why the response finished (stop, tool_calls, etc.)
    """
    data = {}
    if tokens_used:
        data["tokens"] = tokens_used
    if finish_reason:
        data["finish_reason"] = finish_reason
    log_event("api_response", data)


def log_tool_call(name: str, arguments: dict) -> None:
    """Log a tool call event.

    Args:
        name: Tool name
        arguments: Tool arguments
    """
    log_event(
        "tool_call",
        {
            "name": name,
            "arguments": arguments,
        },
    )


def log_tool_result(name: str, result: str) -> None:
    """Log a tool result event.

    Args:
        name: Tool name
        result: Tool result
    """
    log_event(
        "tool_result",
        {
            "name": name,
            "result": result,
        },
    )


def log_stream_start(model: str) -> None:
    """Log stream start event.

    Args:
        model: Model name
    """
    log_event("stream_start", {"model": model})


def log_stream_chunk(tokens_received: int) -> None:
    """Log a stream chunk summary.

    Args:
        tokens_received: Number of tokens received so far
    """
    log_event("stream_chunk", {"tokens_received": tokens_received})


def log_stream_end(total_tokens: int) -> None:
    """Log stream end event.

    Args:
        total_tokens: Total tokens received
    """
    log_event("stream_end", {"total_tokens": total_tokens})


def log_queue_start(text: str, item_id: int) -> None:
    """Log queue start processing event.

    Args:
        text: Message text (truncated)
        item_id: Queue item ID
    """
    log_event(
        "queue_start",
        {
            "text": truncate_for_log(text),
            "item_id": item_id,
        },
    )


def log_queue_complete(item_id: int, status: str) -> None:
    """Log queue complete event.

    Args:
        item_id: Queue item ID
        status: Final status (complete, error, etc.)
    """
    log_event(
        "queue_complete",
        {
            "item_id": item_id,
            "status": status,
        },
    )


def log_queue_interrupt(item_id: int = None) -> None:
    """Log queue interrupt event.

    Args:
        item_id: Queue item ID if available
    """
    data = {}
    if item_id is not None:
        data["item_id"] = item_id
    log_event("queue_interrupt", data)


def log_assistant_response(text: str, reasoning: str = None) -> None:
    """Log assistant response event.

    Args:
        text: Response text (truncated)
        reasoning: Reasoning content if any (truncated)
    """
    data = {"text": truncate_for_log(text)}
    if reasoning:
        data["reasoning"] = truncate_for_log(reasoning)
    log_event("assistant_response", data)


def log_journal_debug(step: str, data: dict[str, Any]) -> None:
    """Log journal debug event for diagnosing journal_all issues.

    Args:
        step: Decision point name (e.g. "journal_all_start", "has_tool_calls")
        data: Diagnostic data dictionary
    """
    log_event("journal_debug", {"step": step, **data})


def log_journal_reflection(
    incoming_message: str, summary: str, message_count: int
) -> None:
    """Log journal reflection event for debugging.

    Args:
        incoming_message: The incoming user message that triggered reflection
        summary: The summary returned by the LLM
        message_count: Number of messages before compaction
    """
    log_event(
        "journal_reflection",
        {
            "incoming_message": truncate_for_log(incoming_message, 200),
            "summary": truncate_for_log(summary, 500),
            "message_count": message_count,
        },
    )


# =============================================================================
# TPS Debug Logging - Track tokens-per-second calculation issues
# =============================================================================


def log_tps_event(event_type: str, data: dict[str, Any] = None) -> None:
    """Log a TPS-related event for debugging.

    Args:
        event_type: Type of TPS event (e.g., "token_usage", "is_final", "timing_reset")
        data: Event data dictionary
    """
    if data is None:
        data = {}
    data["source"] = "tps_debug"
    log_event(f"tps_{event_type}", data)


def log_tps_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    first_token_time: float = None,
    last_token_time: float = None,
    token_count: int = None,
) -> None:
    """Log TOKEN_USAGE event arrival with timing state.

    Args:
        prompt_tokens: Prompt tokens from API
        completion_tokens: Completion tokens from API
        total_tokens: Total tokens from API
        first_token_time: Current _first_token_time value (or None)
        last_token_time: Current _last_token_time value (or None)
        token_count: Current _token_count value (or None)
    """
    log_tps_event(
        "token_usage",
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "first_token_time": first_token_time,
            "last_token_time": last_token_time,
            "token_count": token_count,
        },
    )


def log_tps_first_token(token_count: int = None) -> None:
    """Log first token arrival.

    Args:
        token_count: Current _token_count value
    """
    log_tps_event("first_token", {"token_count": token_count})


def log_tps_stream_start(
    source: str,
    first_token_time: float = None,
    last_token_time: float = None,
    token_count: int = None,
) -> None:
    """Log stream start event.

    Args:
        source: What triggered the stream start (e.g., "status_processing", "tool_complete")
        first_token_time: Current _first_token_time value
        last_token_time: Current _last_token_time value
        token_count: Current _token_count value
    """
    log_tps_event(
        "stream_start",
        {
            "source": source,
            "first_token_time": first_token_time,
            "last_token_time": last_token_time,
            "token_count": token_count,
        },
    )


def log_tps_stream_end(
    reason: str,
    first_token_time: float = None,
    last_token_time: float = None,
    token_count: int = None,
    pending_token_usage: dict = None,
) -> None:
    """Log stream end event.

    Args:
        reason: Why stream ended (e.g., "is_final", "tool_call", "interrupt")
        first_token_time: Current _first_token_time value
        last_token_time: Current _last_token_time value
        token_count: Current _token_count value
        pending_token_usage: Current _pending_token_usage dict
    """
    data = {
        "reason": reason,
        "first_token_time": first_token_time,
        "last_token_time": last_token_time,
        "token_count": token_count,
        "pending_token_usage_set": pending_token_usage is not None,
    }
    if pending_token_usage:
        data["completion_tokens"] = pending_token_usage.get("completion_tokens")
    log_tps_event("stream_end", data)


def log_tps_timing_reset(
    source: str, old_first: float = None, old_last: float = None, old_count: int = None
) -> None:
    """Log when timing variables are reset.

    Args:
        source: What triggered the reset (e.g., "status_processing", "is_final", "tool_finalize")
        old_first: Previous _first_token_time value
        old_last: Previous _last_token_time value
        old_count: Previous _token_count value
    """
    log_tps_event(
        "timing_reset",
        {
            "source": source,
            "old_first_token_time": old_first,
            "old_last_token_time": old_last,
            "old_token_count": old_count,
        },
    )


def log_tps_tool_call(
    tool_name: str,
    first_token_time: float = None,
    last_token_time: float = None,
    token_count: int = None,
    pending_token_usage: dict = None,
) -> None:
    """Log tool call event with timing state.

    Args:
        tool_name: Name of the tool being called
        first_token_time: Current _first_token_time value
        last_token_time: Current _last_token_time value
        token_count: Current _token_count value
        pending_token_usage: Current _pending_token_usage dict
    """
    data = {
        "tool_name": tool_name,
        "first_token_time": first_token_time,
        "last_token_time": last_token_time,
        "token_count": token_count,
        "pending_token_usage_set": pending_token_usage is not None,
    }
    if pending_token_usage:
        data["completion_tokens"] = pending_token_usage.get("completion_tokens")
    log_tps_event("tool_call", data)


def log_tps_calculation(
    elapsed: float,
    completion_tokens: int,
    min_elapsed: float,
    min_tokens: int,
    threshold_passed: bool,
    tps_value: float = None,
) -> None:
    """Log TPS calculation details including threshold check.

    Args:
        elapsed: Calculated elapsed time in seconds
        completion_tokens: Number of completion tokens
        min_elapsed: Minimum elapsed time threshold
        min_tokens: Minimum token count threshold
        threshold_passed: Whether both thresholds were met
        tps_value: Calculated TPS value (if threshold passed)
    """
    log_tps_event(
        "calculation",
        {
            "elapsed": elapsed,
            "completion_tokens": completion_tokens,
            "min_elapsed": min_elapsed,
            "min_tokens": min_tokens,
            "threshold_passed": threshold_passed,
            "tps_value": tps_value,
        },
    )
