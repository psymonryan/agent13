"""Pipe mode — JSON stream protocol for orchestrator integration.

Reads NDJSON turn prompts from stdin, writes NDJSON stream events to stdout.
Long-running: multiple turns per process lifetime, exits on stdin EOF.

Usage:
    agent13 <provider> --model N --io-format json

Protocol: see agent13-pipe-mode-spec.md
"""

import asyncio
import json
import sys
import time
import uuid
from typing import Optional

from agent13 import (
    Agent,
    AgentEvent,
    PromptManager,
    StopReason,
    get_filtered_tools,
    execute_tool,
    skill_manager_ctx,
    init_debug,
    get_config,
)


# ── I/O helpers ────────────────────────────────────────────────────────────


def _emit(obj: dict) -> None:
    """Write one JSON event to stdout (the protocol channel)."""
    print(json.dumps(obj, default=str), flush=True)


def _log(msg: str) -> None:
    """Write a diagnostic message to stderr."""
    print(msg, file=sys.stderr, flush=True)


def _usage_zeros() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


# ── Input parsing ──────────────────────────────────────────────────────────


def _parse_turn(line: str) -> str:
    """Validate and extract prompt text from a NDJSON turn line.

    Raises ValueError with an actionable message on bad input.
    """
    try:
        turn = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e

    if not isinstance(turn, dict):
        raise ValueError(f"turn must be a JSON object, got {type(turn).__name__}")

    if turn.get("type") != "user":
        raise ValueError(f"type must be 'user', got {turn.get('type')!r}")

    msg = turn.get("message")
    if not isinstance(msg, dict):
        raise ValueError("missing 'message' object")

    if msg.get("role") != "user":
        raise ValueError(f"message.role must be 'user', got {msg.get('role')!r}")

    content = msg.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("message.content must be a non-empty array")

    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content block must be an object")
        if block.get("type") != "text":
            raise ValueError(f"unsupported content block type: {block.get('type')!r}")
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError("text content block requires a string 'text' field")
        parts.append(text)

    return "".join(parts)


# ── Main entry point ───────────────────────────────────────────────────────


async def run_pipe_mode(
    client,
    model: str,
    provider: str = "",
    debug: bool = False,
    prompt_manager: Optional[PromptManager] = None,
    system_prompt: Optional[str] = None,
    journal_mode: bool = False,
    remove_reasoning: bool = False,
    devel_mode: bool = False,
    skills_mode: bool = False,
    skill_manager=None,
    continue_session: bool = False,
    read_files: list[str] | None = None,
    polite_interval: float | None = None,
    priming_enabled: bool = False,
    auto_context_threshold: int = 220000,
    auto_context_action: str = "report_and_compact",
    auto_context_chain: int = 3,
    report_and_compact_prompt: str = "",
) -> None:
    """Long-running pipe session: NDJSON in on stdin, NDJSON out on stdout.

    Mirrors run_repl()'s agent lifecycle but swaps:
      - input()            → asyncio stdin readline
      - RichDisplay        → _emit() JSON to stdout
      - slash commands     → not present

    Returns normally on stdin EOF; caller handles process exit.
    """
    if debug:
        init_debug()

    prompt_manager = prompt_manager or PromptManager()

    if skill_manager:
        skill_manager_ctx.set(skill_manager)

    # ── Create agent (same construction as REPL) ──────────────────────
    config = get_config()
    agent = Agent(
        client=client,
        model=model,
        system_prompt=system_prompt or prompt_manager.get_prompt(),
        tools=get_filtered_tools(
            devel=devel_mode,
            skills=skills_mode,
            enabled_tools=config.enabled_tools or None,
            disabled_tools=config.disabled_tools or None,
        ),
        execute_tool=execute_tool,
        remove_reasoning=remove_reasoning,
        devel_mode=devel_mode,
        skills_mode=skills_mode,
        journal_mode=journal_mode,
        priming_enabled=priming_enabled,
        auto_context_threshold=auto_context_threshold,
        auto_context_action=auto_context_action,
        auto_context_chain=auto_context_chain,
        report_and_compact_prompt=report_and_compact_prompt,
    )

    agent.available_models = []

    if config and config.mcp_servers:
        agent.set_mcp_servers(config.mcp_servers)

    if polite_interval is not None:
        agent.set_polite(interval=polite_interval)

    if continue_session:
        from agent13.persistence import find_latest_auto_save, load_context

        latest = find_latest_auto_save()
        if latest:
            success, msg, _incomplete = load_context(agent, str(latest))
            if success:
                _log(f"Resumed session from {latest} ({msg})")
            else:
                _log(f"Could not resume: {msg}")
        else:
            _log("No saved session found, starting fresh")

    # ── Session ID ─────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())

    # ── Turn state (reset per turn) ───────────────────────────────────
    text_buf: list[str] = []
    reasoning_buf: list[str] = []
    tool_count = 0
    pending_tool_ids: list[str] = []
    turn_prompt_tokens = 0
    turn_completion_tokens = 0
    turn_result_text = ""
    turn_start = 0.0
    turn_error = ""

    # ── Turn completion tracking (same pattern as batch.py) ──────────
    processing_done = asyncio.Event()
    work_started = False
    shutting_down = False

    # ── Event handlers ─────────────────────────────────────────────────

    def _flush_text() -> None:
        """Flush accumulated text/reasoning buffers as assistant events."""
        nonlocal turn_result_text

        if reasoning_buf:
            reasoning_text = "".join(reasoning_buf).strip()
            if reasoning_text:
                _emit(
                    {
                        "type": "assistant",
                        "session_id": session_id,
                        "subtype": "reasoning",
                        "message": {
                            "content": [{"type": "text", "text": reasoning_text}]
                        },
                    }
                )
            reasoning_buf.clear()

        if text_buf:
            text = "".join(text_buf).strip()
            if text:
                turn_result_text = text
                _emit(
                    {
                        "type": "assistant",
                        "session_id": session_id,
                        "message": {"content": [{"type": "text", "text": text}]},
                    }
                )
            text_buf.clear()

    def _emit_result(is_error: bool = False, error_message: str = "") -> None:
        """Emit the result event for the current turn and reset turn state."""
        nonlocal tool_count, turn_prompt_tokens, turn_completion_tokens
        nonlocal turn_result_text, turn_error

        duration_ms = int((time.monotonic() - turn_start) * 1000)

        # If an error occurred and no output was produced, mark as failed
        if not is_error and turn_error and not turn_result_text:
            is_error = True
            error_message = turn_error

        result: dict = {
            "type": "result",
            "session_id": session_id,
            "subtype": "success" if not is_error else "error",
            "is_error": is_error,
            "result": turn_result_text if not is_error else "",
            "model": model,
            "usage": {
                "input_tokens": turn_prompt_tokens,
                "output_tokens": turn_completion_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "num_turns": tool_count,
            "duration_ms": duration_ms,
            "duration_api_ms": duration_ms,
        }

        if is_error and error_message:
            result["error"] = {"message": error_message}

        _emit(result)

        # Reset turn state
        tool_count = 0
        pending_tool_ids.clear()
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        turn_result_text = ""
        turn_error = ""

    @agent.on_event
    async def on_item_started(event):
        nonlocal work_started
        if event.event == AgentEvent.ITEM_STARTED:
            work_started = True

    @agent.on_event
    async def on_status(event):
        if event.event != AgentEvent.STATUS_CHANGE or shutting_down:
            return
        status = event.data.get("status", "")
        if status == "idle" and work_started:
            processing_done.set()

    @agent.on_event
    async def on_token(event):
        if event.event != AgentEvent.ASSISTANT_TOKEN or shutting_down:
            return
        text_buf.append(event.text or "")

    @agent.on_event
    async def on_reasoning(event):
        if event.event != AgentEvent.ASSISTANT_REASONING or shutting_down:
            return
        reasoning_buf.append(event.text or "")

    @agent.on_event
    async def on_notification(event):
        if event.event != AgentEvent.NOTIFICATION or shutting_down:
            return
        message = event.data.get("message", "")
        if message:
            _log(f"[notice] {message}")

    @agent.on_event
    async def on_tool_call(event):
        nonlocal tool_count
        if event.event != AgentEvent.TOOL_CALL or shutting_down:
            return
        name = event.data.get("name", "")
        arguments = event.data.get("arguments", {})

        _flush_text()

        tool_count += 1
        tool_id = f"toolu_{tool_count:02d}"
        pending_tool_ids.append(tool_id)

        _emit(
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": name,
                            "input": arguments,
                        }
                    ]
                },
            }
        )

    @agent.on_event
    async def on_tool_result(event):
        if event.event != AgentEvent.TOOL_RESULT or shutting_down:
            return
        result = event.data.get("result", "")

        # Pop the matching tool call ID (FIFO — tools are sequential)
        tool_id = pending_tool_ids.pop(0) if pending_tool_ids else "toolu_00"

        _emit(
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                            "is_error": False,
                        }
                    ]
                },
            }
        )

    @agent.on_event
    async def on_token_usage(event):
        nonlocal turn_prompt_tokens, turn_completion_tokens
        if event.event != AgentEvent.TOKEN_USAGE or shutting_down:
            return
        turn_prompt_tokens += event.data.get("prompt_tokens", 0)
        turn_completion_tokens += event.data.get("completion_tokens", 0)

    @agent.on_event
    async def on_error(event):
        nonlocal turn_error
        if event.event != AgentEvent.ERROR or shutting_down:
            return
        message = event.message or "Unknown error"
        turn_error = message
        _log(f"[pipe] error: {message}")

    @agent.on_event
    async def on_mcp_started(event):
        if event.event != AgentEvent.MCP_SERVER_STARTED or shutting_down:
            return
        _log(f"  MCP: Starting {event.server_name or 'unknown'}")

    @agent.on_event
    async def on_mcp_ready(event):
        if event.event != AgentEvent.MCP_SERVER_READY or shutting_down:
            return
        _log(
            f"  MCP: {event.server_name or 'unknown'} ready "
            f"({event.tool_count or 0} tools)"
        )

    @agent.on_event
    async def on_mcp_error(event):
        if event.event != AgentEvent.MCP_SERVER_ERROR or shutting_down:
            return
        _log(
            f"  MCP error: {event.server_name or 'unknown'}: "
            f"{event.error or 'Unknown error'}"
        )

    # ── Start agent ────────────────────────────────────────────────────
    agent_task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.1)  # let agent initialise

    # Inject files from --read flag (if provided)
    if read_files:
        from agent13.file_injection import build_read_message

        read_msg = build_read_message(read_files)
        work_started = True
        await agent.add_message(read_msg)
        await processing_done.wait()
        work_started = False
        processing_done.clear()

    # ── Emit system event (first line on stdout) ───────────────────────
    _emit(
        {
            "type": "system",
            "session_id": session_id,
            "subtype": "session_start",
            "data": {
                "session_id": session_id,
                "model": model,
                "provider": provider,
            },
        }
    )

    # ── Main loop: read NDJSON from stdin ──────────────────────────────
    # Read stdin in a worker thread (same pattern as the REPL and headless),
    # NOT via the event loop's pipe transport: on Windows the
    # ProactorEventLoop crashes with OSError [WinError 6] "The handle is
    # invalid" when the stdin write end has closed by the time the first
    # read is registered (e.g. `subprocess.run(input=...)` closes stdin
    # immediately after writing) — the selector backend (macOS/Linux)
    # returns a clean EOF in the same situation.
    loop = asyncio.get_running_loop()

    while True:
        try:
            raw = await loop.run_in_executor(None, sys.stdin.buffer.readline)
        except (OSError, ValueError):
            # Stdin handle invalid or closed mid-session — treat as EOF.
            raw = b""
        if not raw:  # EOF
            break

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        # Parse the turn prompt
        try:
            prompt = _parse_turn(line)
        except ValueError as e:
            _emit(
                {
                    "type": "result",
                    "session_id": session_id,
                    "subtype": "invalid_input",
                    "is_error": True,
                    "result": "",
                    "error": {"message": str(e)},
                    "model": model,
                    "usage": _usage_zeros(),
                    "num_turns": 0,
                    "duration_ms": 0,
                    "duration_api_ms": 0,
                }
            )
            continue

        # Process the turn
        turn_start = time.monotonic()
        processing_done.clear()

        await agent.add_message(prompt)
        await processing_done.wait()

        # Flush remaining text and emit result
        _flush_text()
        _emit_result(is_error=False)

    # ── EOF: clean shutdown ────────────────────────────────────────────
    shutting_down = True
    agent.stop(StopReason.QUIT)
    agent_task.cancel()
    try:
        await agent_task
    except asyncio.CancelledError:
        pass

    _log(f"Pipe session ended ({session_id})")
