"""Interactive REPL mode for agent13.

Two-state interaction model:
- STREAMING: no prompt, agent output flows freely. Press Enter to request input.
- INPUT: prompt shown, output frozen. Type message, press Enter to send.

Phase 2: --output flag redirects chat to a file for clean split-pane viewing.

Usage:
    agent13 <provider> --repl                    # Interactive REPL
    agent13 <provider> --output out.log          # REPL + file output (--output implies --repl)
    agent13 <provider> --repl --output out.log   # Same, explicit
"""

import asyncio
import os
import sys
from pathlib import Path
import time
from typing import Optional

from rich.console import Console

from agent13 import (
    Agent,
    AgentEvent,
    History,
    PromptManager,
    get_filtered_tools,
    execute_tool,
    skill_manager_ctx,
    init_debug,
    get_config,
)
from ui.display import RichDisplay
from agent13.timing import TokenTimingTracker
from agent13.file_injection import expand_file_mentions
from agent13.bell import BellManager
from agent13.models import fetch_models, resolve_model_selection
from agent13.config import (
    resolve_provider_arg,
    resolve_provider_selection,
    create_client,
    get_provider_names,
)
from agent13.commands import (
    execute_save,
    execute_delete,
    execute_retry,
    execute_prioritise,
    execute_deprioritise,
    list_save_names,
    format_queue_items,
    format_history_groups,
)
from agent13.message_history import content_to_text
from agent13.status import get_tool_stats_summary
from agent13.sandbox import (
    parse_sandbox_mode,
    get_default_sandbox_mode,
    get_pinned_sandbox_mode,
    pin_sandbox_mode,
    unpin_sandbox_mode,
)
from tools.security import (
    set_session_sandbox_mode,
    get_session_sandbox_mode,
    get_current_sandbox_mode,
)
from ui.display import format_mcp_servers


# Slash commands available in REPL mode
COMMANDS = {
    "/quit": "Exit the REPL",
    "/exit": "Exit the REPL",
    "/help": "Show available commands",
    "/status": "Show agent status and session info",
    "/pause": "Pause agent processing",
    "/resume": "Resume agent processing",
    "/stop": "Interrupt current response",
    "/save": "Save session (e.g. /save mysession)",
    "/load": "Load session (e.g. /load mysession)",
    "/multi": "Enter multi-line mode (end with . on its own line) or just put a \\ on the end of the line",
    "/clear": "Clear display: /clear (display only), /clear N (show last N turns), /clear all (wipe history + display)",
    "/history": "Show message history (grouped)",
    "/queue": "Show queued messages",
    "/retry": "Retry the last message",
    "/prioritise": "Mark queue item as priority: /prioritise N",
    "/deprioritise": "Remove priority from queue item: /deprioritise N",
    "/delete": "Delete items: /delete h N (history group), /delete q N (queue item), /delete s NAME (save)",
    "/journal": "Control journal mode: /journal [on|off|last|all|status]",
    "/compact": "Compact history: /compact [next-task focus] | /compact --prompt <name>",
    "/model": "Switch model: /model [name|number] (no args lists available models)",
    "/provider": "Switch provider: /provider <name|url>",
    "/sandbox": "Sandbox mode: /sandbox [mode|pin|unpin] (no args shows status)",
    "/bell": "Bell on idle: /bell [N|off] (0=always, N=threshold seconds)",
    "/bell-command": "Set external bell command: /bell-command [cmd|off]",
    "/devel": "Toggle devel mode: /devel [on|off|status]",
    "/tools": "Show tool usage statistics",
    "/mcp": "Show MCP server status: /mcp [connect|disconnect|reload]",
    "/cwd": "Show or change working directory: /cwd [path]",
    "/upgrade": "Check for and apply updates",
    "/polite": "Multi-agent lock coordination: /polite N | /polite off",
    "/auto_compact_threshold": "Auto-compact threshold: /auto_compact_threshold [N|0] (supports k suffix)",
    "/auto_compact_max": "Auto-compact max cycles: /auto_compact_max [N] (min 1)",
}


class OutputController:
    """Wraps stdout — buffers output while user is in input mode.

    Two modes:
    - Streaming (default): output writes through to stdout immediately.
    - Input (user at prompt): output buffers, flushed when streaming resumes.
    """

    def __init__(self):
        self._real = sys.stdout
        self._buffer: list[str] = []
        self._frozen = False

    def write(self, text: str) -> None:
        """Write output. Buffers when frozen (input mode)."""
        if self._frozen:
            self._buffer.append(text)
        else:
            self._real.write(text)
            self._real.flush()

    def flush(self) -> None:
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    def freeze(self) -> None:
        """Freeze output — buffer everything (entering input mode)."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze — flush buffer and resume streaming."""
        self._frozen = False
        for text in self._buffer:
            self._real.write(text)
        self._buffer.clear()
        self._real.flush()


def _sanitize(text: str, max_len: int = 120) -> str:
    """Collapse newlines and truncate for display in history/queue listings."""
    if not text:
        return "(empty)"
    text = text.replace("\n", " ").replace("\r", "").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _display_loaded_messages(agent) -> None:
    """Display loaded conversation messages to the console.

    Shows a plain-text summary of the loaded conversation so the user
    can see what was restored. Truncates long messages for readability.
    """
    messages = agent.messages
    if not messages:
        print("  (empty conversation)")
        return

    print(f"\n  Loaded conversation ({len(messages)} messages):")
    print("  " + "-" * 50)

    pending_tool_calls: dict = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id")

        # Normalize multimodal content (list of blocks) to display text
        content = content_to_text(content)

        if role == "user":
            if content:
                display = content[:200] + "..." if len(content) > 200 else content
                # Collapse multi-line for compact display
                display = display.replace("\n", " ")
                print(f"  > {display}")

        elif role == "assistant":
            if content and content.strip():
                display = content[:300] + "..." if len(content) > 300 else content
                display = display.replace("\n", " ")
                print(f"  Agent: {display}")

            for tc in tool_calls:
                tc_id = tc.get("id")
                tc_func = tc.get("function", {})
                tc_name = tc_func.get("name", "unknown")
                tc_args = tc_func.get("arguments", "{}")
                if tc_id:
                    pending_tool_calls[tc_id] = (tc_name, tc_args)
                    # Show tool name and brief args
                    args_brief = (
                        tc_args[:100] + "..." if len(tc_args) > 100 else tc_args
                    )
                    print(f"    Tool: {tc_name}({args_brief})")

        elif role == "tool":
            if tool_call_id and tool_call_id in pending_tool_calls:
                tc_name, _ = pending_tool_calls.pop(tool_call_id)
                display = content[:200] + "..." if len(content) > 200 else content
                display = display.replace("\n", " ")
                print(f"    Result ({tc_name}): {display}")

    print("  " + "-" * 50 + "\n")


async def run_repl(
    client,
    model: str,
    provider: str = "",
    pretty: bool = True,
    debug: bool = False,
    prompt_manager: Optional[PromptManager] = None,
    system_prompt: Optional[str] = None,
    journal_mode: bool = False,
    remove_reasoning: bool = False,
    devel_mode: bool = False,
    skills_mode: bool = False,
    skill_manager=None,
    continue_session: bool = False,
    output_path: Optional[str] = None,
    model_names: Optional[list[str]] = None,
    read_files: list[str] | None = None,
    polite_interval: float | None = None,
    bell_threshold: float | str | None = None,
    bell_enabled: bool = True,
    bell_command: str = "",
    priming_enabled: bool = False,
    auto_compact_threshold: int = 0,
    auto_compact_max_iterations: int = 3,
):
    """Run the agent in interactive REPL mode.

    Two-state interaction model:
    - STREAMING: no prompt, agent output flows to stdout.
      Press Enter to request input.
    - INPUT: prompt shown, output frozen.
      Type message and press Enter to send.

    When output_path is set (--output), chat output goes to the file
    and info/commands stay on stdout — enabling split-pane viewing
    with ``tail -f``.
    """
    if debug:
        init_debug()

    prompt_manager = prompt_manager or PromptManager()

    # Set up skill manager context
    if skill_manager:
        skill_manager_ctx.set(skill_manager)

    # Set up readline for history and line editing
    _histfile = os.path.join(os.path.expanduser("~"), ".agent13", "repl_history")
    try:
        import readline  # noqa: F811

        readline.set_history_length(1000)
        try:
            readline.read_history_file(_histfile)
        except (FileNotFoundError, OSError):
            pass
    except ImportError:
        _histfile = None  # readline not available

    # Shared history — same store as TUI (dated per-project files)
    history = History()

    # Pre-seed readline with history from the History store so TUI entries
    # are visible via readline up/down navigation.
    try:
        import readline as _rl

        for _ts, _cmd in history.session_items:
            _rl.add_history(_cmd)
    except (ImportError, NameError):
        pass

    # Tab completion for /save and /load commands
    def _completer(text, state):
        """Tab-complete save names for /save and /load."""
        import readline as _rl

        # Only complete when cursor is after "/save " or "/load "
        buffer = _rl.get_line_buffer()
        if buffer.startswith("/save ") or buffer.startswith("/load "):
            from agent13.persistence import list_saves

            prefix = buffer.split(maxsplit=1)[1] if " " in buffer else ""
            if state == 0:
                # Build candidates on first call
                saves = list_saves()
                _completer.candidates = [
                    s.stem for s in saves if s.stem.startswith(prefix)
                ]
            candidates = getattr(_completer, "candidates", [])
            return candidates[state] if state < len(candidates) else None
        return None

    try:
        import readline as _rl2

        _rl2.set_completer(_completer)
        _rl2.parse_and_bind("tab: complete")
    except (ImportError, NameError):
        pass

    # Set up output controller and console
    # Without --output: chat goes through OutputController (freezes in input mode)
    # With --output:    chat goes to file directly, status goes through OutputController
    output_ctrl = OutputController()

    output_file = None
    if output_path:
        output_file = open(output_path, "a", buffering=1)
        chat_console = Console(file=output_file)
        pretty = False  # Live display doesn't work on file targets
    else:
        chat_console = Console(file=output_ctrl)

    display = RichDisplay(console=chat_console, pretty=pretty, debug=debug)

    # Create agent
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
        auto_compact_threshold=auto_compact_threshold,
        auto_compact_max_iterations=auto_compact_max_iterations,
    )

    # Store available models on agent
    agent.available_models = model_names or []

    # Load MCP server configs
    if config and config.mcp_servers:
        agent.set_mcp_servers(config.mcp_servers)

    # Enable polite mode if requested (--polite N)
    if polite_interval is not None:
        agent.set_polite(interval=polite_interval)

    # Bell manager — shared logic with TUI. REPL writes \a to stdout
    # as the fallback (terminal bell) instead of Textual's self.bell().
    def _ring_terminal_bell():
        sys.stdout.write("\a")
        sys.stdout.flush()

    bell = BellManager(
        threshold=bell_threshold,
        enabled=bell_enabled,
        command=bell_command,
        fallback_ring=_ring_terminal_bell,
    )

    # Load previous session if --continue
    if continue_session:
        from agent13.persistence import find_latest_auto_save, load_context

        latest = find_latest_auto_save()
        if latest:
            success, msg, _incomplete = load_context(agent, str(latest))
            if success:
                print(f"Resumed session from {latest} ({msg})")
            else:
                print(f"Could not resume: {msg}")
        else:
            print("No saved session found, starting fresh")

    # ── State tracking ───────────────────────────────────────────────
    # Note: agent.is_idle, agent.status, agent.pause_state are the
    # single source of truth — same as TUI. No duplicate flags.
    shutting_down = False  # Guards event handlers during cleanup
    enter_streaming = False  # Set by /resume to switch to streaming mode
    in_multi_mode = False
    multi_buffer: list[str] = []

    # TPS / timing tracking (delegated to TokenTimingTracker)
    tracker = TokenTimingTracker()
    _processing_start_time: Optional[float] = None
    _session_start_time: float = time.time()
    _last_turn_duration: Optional[float] = None  # seconds
    _last_turn_end_time: Optional[float] = None  # epoch

    # ── Event handlers ───────────────────────────────────────────────
    # Wire agent events to RichDisplay (chat content) and OutputController
    # (status notifications). Status goes through output_ctrl so it buffers
    # when the user is in input mode.

    @agent.on_event
    async def on_status_change(event):
        if event.event != AgentEvent.STATUS_CHANGE or shutting_down:
            return
        nonlocal _processing_start_time, _last_turn_duration, _last_turn_end_time
        status = event.data.get("status", "")
        if status == "paused":
            output_ctrl.write("[paused]\n")
            bell.on_pause()
            return
        if status == "idle" and agent.messages:
            tracker.turn_end()
            bell.on_turn_end()
            # Duration notification
            if _processing_start_time:
                elapsed = time.time() - _processing_start_time
                _last_turn_duration = elapsed
                _last_turn_end_time = time.time()
                # Unfreeze before writing — output may be frozen if we're
                # resuming from pause (read_no_prompt/with_prompt froze it).
                # Without this, completion message buffers forever.
                output_ctrl.unfreeze()
                output_ctrl.write(f"\n[complete — {elapsed:.1f}s]")
                _processing_start_time = None
                sys.__stdout__.write("\n> ")
                sys.__stdout__.flush()
                output_ctrl.freeze()
        elif status in ("waiting", "thinking", "processing", "tooling"):
            tracker.turn_start()
            bell.on_turn_start()

    @agent.on_event
    async def on_item_started(event):
        if event.event != AgentEvent.ITEM_STARTED:
            return

    @agent.on_event
    async def on_stream_start(event):
        if event.event != AgentEvent.STREAM_START or shutting_down:
            return
        tracker.reset_stream()

    @agent.on_event
    async def on_token(event):
        if event.event != AgentEvent.ASSISTANT_TOKEN or shutting_down:
            return
        now = time.time()
        if tracker.is_first_token:
            display.start_response()
        tracker.record_token(now)
        display.add_token(event.text or "")

    @agent.on_event
    async def on_reasoning(event):
        if event.event != AgentEvent.ASSISTANT_REASONING or shutting_down:
            return
        display.add_reasoning(event.text or "")

    @agent.on_event
    async def on_complete(event):
        if event.event != AgentEvent.ASSISTANT_COMPLETE or shutting_down:
            return
        display.complete_response()

    @agent.on_event
    async def on_tool_call(event):
        if event.event != AgentEvent.TOOL_CALL or shutting_down:
            return
        name = event.data.get("name", "")
        arguments = event.data.get("arguments", {})
        output_ctrl.write(f"[tool: {name}]\n")
        display.show_tool_call(name, arguments)

    @agent.on_event
    async def on_tool_result(event):
        if event.event != AgentEvent.TOOL_RESULT or shutting_down:
            return
        result = event.data.get("result", "")
        display.show_tool_result(result)

    @agent.on_event
    async def on_error(event):
        if event.event != AgentEvent.ERROR or shutting_down:
            return
        message = event.message or "Unknown error"
        display.show_error(message)

    @agent.on_event
    async def on_token_usage(event):
        if event.event != AgentEvent.TOKEN_USAGE or shutting_down:
            return
        result = tracker.compute_tps(event.data)
        if result is not None:
            output_ctrl.write(
                f"  ({result.tps:.0f} tok/s,"
                f" {result.completion_tokens} tokens,"
                f" {result.elapsed:.1f}s)\n"
            )

    @agent.on_event
    async def on_context_loaded(event):
        if event.event != AgentEvent.CONTEXT_LOADED or shutting_down:
            return
        success = event.data.get("success", False)
        message = event.data.get("message", "")
        incomplete = event.data.get("incomplete", False)
        if success:
            print(f"\n  {message}")
            if incomplete:
                print("  Warning: Incomplete turn detected. Use /resume to continue.")
            _display_loaded_messages(agent)
        else:
            print(f"  Error: {message}")

    @agent.on_event
    async def on_journal_result(event):
        if event.event != AgentEvent.JOURNAL_RESULT or shutting_down:
            return
        success = event.data.get("success", False)
        message = event.data.get("message", "")
        if success:
            print(f"\n  {message}")
        else:
            print(f"  Journal error: {message}")

    @agent.on_event
    async def on_journal_compact(event):
        if event.event != AgentEvent.JOURNAL_COMPACT or shutting_down:
            return
        summary = event.data.get("summary", "")
        words = event.data.get("word_count", 0)
        print(f"\n  [Journal] {summary}")
        print(f"  ({words} words)")

    @agent.on_event
    async def on_mcp_started(event):
        if event.event != AgentEvent.MCP_SERVER_STARTED or shutting_down:
            return
        server_name = event.server_name or "unknown"
        transport = event.transport or "unknown"
        print(f"  MCP: Starting {server_name} ({transport})")

    @agent.on_event
    async def on_mcp_ready(event):
        if event.event != AgentEvent.MCP_SERVER_READY or shutting_down:
            return
        server_name = event.server_name or "unknown"
        tool_count = event.tool_count or 0
        print(f"  MCP: {server_name} ready ({tool_count} tools)")

    @agent.on_event
    async def on_mcp_error(event):
        if event.event != AgentEvent.MCP_SERVER_ERROR or shutting_down:
            return
        server_name = event.server_name or "unknown"
        error = event.error or "Unknown error"
        print(f"  MCP error: {server_name}: {error}")

    @agent.on_event
    async def on_mcp_stderr(event):
        if event.event != AgentEvent.MCP_SERVER_STDERR or shutting_down:
            return
        server_name = event.server_name or "unknown"
        line = event.line or ""
        if line.strip():
            print(f"  MCP {server_name}: {line}")

    # -- Start agent --------------------------------------------------------
    agent_task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.1)  # Let agent initialise

    # Inject files from --read flag if provided
    if read_files:
        from agent13.file_injection import build_read_message

        read_msg = build_read_message(read_files)
        await agent.add_message(read_msg)

    # Print banner (directly to stdout — visible even when frozen)
    provider_str = f"{provider}/" if provider else ""
    print(f"agent13 REPL \u2014 {provider_str}{model}")
    if output_path:
        print(f"Chat output: tail -f {output_path}")
    print("Type /help for commands, /quit to exit")
    print("Press Enter (empty) to switch to input mode\n")

    # ── Input helpers ────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    def read_with_prompt():
        """Blocking input with prompt — INPUT mode."""
        try:
            return input("> ")
        finally:
            # Freeze output before returning — close race window
            output_ctrl.freeze()

    def read_no_prompt():
        """Blocking input without prompt — STREAMING mode.

        Uses sys.stdin.readline() to bypass readline — no history/completion
        interference during streaming. User only needs to press Enter.
        """
        try:
            line = sys.stdin.readline()
            if not line:  # EOF
                return "/quit"
            return line.rstrip("\n")
        except KeyboardInterrupt:
            return "/quit"
        finally:
            # Freeze output before returning — close race window
            output_ctrl.freeze()

    def read_multi_prompt():
        """Blocking input with multi-line prompt — MULTI-LINE mode."""
        try:
            return input("... ")
        finally:
            output_ctrl.freeze()

    # ── REPL loop ────────────────────────────────────────────────────
    # Two states, toggled by Enter:
    #
    #   STREAMING: no prompt, agent output flows, user watches.
    #              Enter → INPUT mode.
    #
    #   INPUT: prompt shown, output frozen/buffered, user types.
    #              Enter → send message → STREAMING mode.

    try:
        # Start in INPUT mode — show initial prompt
        output_ctrl.freeze()
        user_input = await loop.run_in_executor(None, read_with_prompt)

        while True:
            user_input = user_input.strip()

            # ── Multi-line accumulation (before empty-line check) ──
            from_multi = False
            if in_multi_mode:
                if user_input == ".":
                    if multi_buffer:
                        user_input = "\n".join(multi_buffer)
                        in_multi_mode = False
                        multi_buffer = []
                        from_multi = True
                        # Fall through to send dispatch
                    else:
                        in_multi_mode = False
                        print("  Multi-line mode cancelled (empty)")
                        user_input = await loop.run_in_executor(None, read_with_prompt)
                        continue
                elif user_input in ("/quit", "/exit"):
                    in_multi_mode = False
                    multi_buffer = []
                    break
                elif user_input == "/cancel":
                    in_multi_mode = False
                    multi_buffer = []
                    print("  Multi-line input cancelled")
                    user_input = await loop.run_in_executor(None, read_with_prompt)
                    continue
                else:
                    multi_buffer.append(user_input)
                    user_input = await loop.run_in_executor(None, read_multi_prompt)
                    continue
            # ── Backslash continuation (enter multi-line mode) ──
            if not in_multi_mode and not from_multi and user_input.endswith("\\"):
                in_multi_mode = True
                multi_buffer = [user_input[:-1]]  # strip trailing backslash
                print("  Multi-line mode (\\). End with . on its own line.")
                user_input = await loop.run_in_executor(None, read_multi_prompt)
                continue
            if not user_input:
                if agent.is_idle:
                    # Agent is idle — just re-show the prompt
                    user_input = await loop.run_in_executor(None, read_with_prompt)
                else:
                    # Agent is streaming — switch to streaming, wait for Enter
                    output_ctrl.unfreeze()
                    user_input = await loop.run_in_executor(None, read_no_prompt)
                    # Back to input mode (freeze set in read_no_prompt finally)
                    if not user_input.strip():
                        # Another empty — show prompt
                        user_input = await loop.run_in_executor(None, read_with_prompt)
                continue

            # ── Slash commands ────────────────────────────────────
            if user_input.startswith("/") and not from_multi:
                cmd = user_input.split()[0].lower()
                args_parts = user_input.split(maxsplit=1)
                cmd_arg = args_parts[1] if len(args_parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    break

                elif cmd == "/help":
                    print("\n  Commands:")
                    for c, desc in COMMANDS.items():
                        print(f"    {c:12s}  {desc}")
                    print("\n  Press Enter (empty line) to switch between modes.")
                    print("  - Streaming mode: output flows, no prompt.")
                    print(
                        "  - Input mode: prompt shown, output pauses, type your message."
                    )

                elif cmd == "/status":
                    from agent13.status import gather_status, format_duration
                    import time as _time

                    # Build last-turn info if available
                    sd = gather_status(
                        agent,
                        provider,
                        model,
                        _session_start_time,
                        prompt_manager,
                        tracker,
                    )
                    # Add last-turn data from local tracking
                    if _last_turn_duration is not None:
                        sd.last_turn_duration = format_duration(_last_turn_duration)
                        if _last_turn_end_time is not None:
                            ago = format_duration(_time.time() - _last_turn_end_time)
                            sd.last_turn_ago = f"{ago} ago"

                    print()
                    print("  Session")
                    print(f"    status:    {sd.agent_status}")
                    print(f"    run time:  {sd.run_time}")
                    print(f"    cwd:       {sd.cwd}")
                    if sd.last_turn_duration:
                        print(
                            f"    last turn: {sd.last_turn_duration} (completed {sd.last_turn_ago})"
                        )
                    if sd.turn_count > 0:
                        print(
                            f"    turns:     {sd.turn_count} | total processing: {sd.total_processing}"
                        )
                    print()
                    print("  Provider")
                    print(f"    provider:  {sd.provider}")
                    print(f"    model:     {sd.model}")
                    print(f"    prompt:    {sd.active_prompt}")
                    print()
                    print("  Context")
                    print(f"    prompt tokens:       {sd.prompt_tokens_fmt}")
                    print(f"    completion tokens:   {sd.completion_tokens_fmt}")
                    print(f"    total tokens:        {sd.total_tokens_fmt}")
                    print(f"    queue: {sd.queue_count}  messages: {sd.message_count}")
                    print()
                    print("  Connectivity")
                    print(f"    mcp: {sd.mcp_status}")
                    tools_str = (
                        f"{sd.tool_successes}/{sd.tool_calls}"
                        if sd.tool_calls > 0
                        else "0"
                    )
                    print()
                    print("  Tools")
                    print(f"    success/calls: {tools_str}")
                    print()
                    print("  Settings")
                    print(f"    sandbox:           {sd.sandbox_mode}")
                    print(
                        f"    remove-reasoning:  {'on' if sd.remove_reasoning else 'off'}"
                    )
                    print(f"    devel:             {'on' if sd.devel_mode else 'off'}")
                    print(f"    skills:            {'on' if sd.skills_mode else 'off'}")
                    print(
                        f"    journal:           {'on' if sd.journal_mode else 'off'}"
                    )
                    print(f"    bell:              {bell.status_text()}")
                    print(
                        f"    bell-cmd:          {bell.command if bell.command else '(terminal bell)'}"
                    )
                    print()

                elif cmd == "/pause":
                    if agent.is_paused:
                        print("  Already paused")
                    elif agent.is_pausing:
                        print("  Already pausing")
                    elif agent.is_idle:
                        print("  Nothing to pause (agent is idle)")
                    else:
                        agent.pause()
                        print(
                            "  Pausing at next safe point \u2014 use /resume to cancel"
                        )

                elif cmd == "/resume":
                    if agent.is_paused or agent.is_pausing:
                        agent.resume()
                        # Flush buffered output (streaming tokens, [paused], etc.)
                        # that accumulated while output was frozen during pause.
                        output_ctrl.unfreeze()
                        print("  Resumed")
                        # If agent has work to do, switch to streaming mode
                        # so the user can see output resume flowing.
                        if not agent.is_idle:
                            enter_streaming = True
                        else:
                            output_ctrl.freeze()
                    else:
                        print("  Not paused")

                elif cmd == "/stop":
                    if not agent.is_idle:
                        await agent.add_message(
                            "[interrupted by user]",
                            interrupt=True,
                        )
                        print("  Interrupted")
                    else:
                        print("  Nothing to stop")

                elif cmd == "/save":
                    result = execute_save(agent, cmd_arg)
                    if result.success:
                        print(f"  {result.message}")
                    else:
                        print(f"  {result.message}")

                elif cmd == "/load":
                    if not cmd_arg:
                        # Show available saves
                        names = list_save_names()
                        if names:
                            print("  Available saves:")
                            for name in names:
                                print(f"    {name}")
                        else:
                            print("  No saves found")
                        print("  Usage: /load <name>")
                    else:
                        from agent13.persistence import (
                            load_context,
                            resolve_save_path,
                        )

                        path = resolve_save_path(cmd_arg)
                        if not path.exists():
                            print(f"  Not found: {path}")
                            print(f"  Use /save {cmd_arg} to create it")
                        else:
                            # Check if agent is busy
                            if not agent.is_idle:
                                # Defer the load to a safe boundary
                                await agent.request_load(str(path))
                                print(
                                    "  Load queued (will take effect after current response)"
                                )
                            else:
                                # Load immediately
                                try:
                                    success, msg, incomplete = load_context(
                                        agent, str(path)
                                    )
                                    if success:
                                        print(f"  {msg}")
                                        if incomplete:
                                            print(
                                                "  Warning: Incomplete turn. Use /resume to continue."
                                            )
                                        _display_loaded_messages(agent)
                                    else:
                                        print(f"  Error: {msg}")
                                except Exception as e:
                                    print(f"  Error loading: {e}")

                elif cmd == "/multi":
                    in_multi_mode = True
                    multi_buffer = []
                    print("  Multi-line mode. End with . on its own line.")
                    user_input = await loop.run_in_executor(None, read_multi_prompt)
                    continue

                elif cmd == "/clear":
                    arg = cmd_arg.strip().lower()
                    if arg == "all":
                        if agent.is_idle:
                            count = agent.clear_messages()
                            print(f"  Cleared {count} messages")
                        else:
                            await agent.request_clear(mode="all")
                            print(
                                "  Clear queued (will take effect after current response)"
                            )
                    elif arg.isdigit():
                        # /clear N in REPL — no display to rebuild,
                        # history untouched
                        print(
                            f"  History preserved ({len(agent.messages)} messages). "
                            "Use /clear all to wipe."
                        )
                    else:
                        # /clear with no args in REPL — no display to wipe,
                        # history untouched
                        print(
                            f"  History preserved ({len(agent.messages)} messages). "
                            "Use /clear all to wipe."
                        )

                elif cmd == "/history":
                    if not agent.messages:
                        print("  No messages in history")
                    else:
                        groups = format_history_groups(agent)
                        print(f"\n  Message history ({len(groups)} groups):\n")
                        for g in groups:
                            first_content = _sanitize(g.first_content)
                            print(f"  {g.number:3d}. {g.first_role}: {first_content}")
                            for entry in g.entries:
                                if entry.role == "tool":
                                    print(
                                        f"         tool result: {_sanitize(entry.content, 80)}"
                                    )
                                elif entry.role == "assistant":
                                    if entry.content:
                                        print(
                                            f"         assistant: {_sanitize(entry.content, 80)}"
                                        )
                                    for tc in entry.tool_calls:
                                        print(
                                            f"         tool call: {tc['name']}({_sanitize(tc['arguments'], 60)})"
                                        )
                                elif entry.is_interrupt:
                                    print(
                                        f"         interrupt: {_sanitize(entry.content)}"
                                    )
                                elif entry.is_injected:
                                    print(
                                        f"         injected: {_sanitize(entry.content)}"
                                    )
                                elif entry.content:
                                    print(
                                        f"         {entry.role}: {_sanitize(entry.content)}"
                                    )
                        print("\n  Use /delete h N to delete group N")

                elif cmd == "/queue":
                    items = format_queue_items(agent.queue)
                    if not items:
                        print("  Queue is empty")
                    else:
                        pending_count = sum(1 for it in items if not it.running)
                        print(f"\n  Queue ({pending_count} pending):")
                        for item in items:
                            marker = ""
                            if item.interrupt:
                                marker = "!! "
                            elif item.priority:
                                marker = "!  "
                            if item.running:
                                # Running item: header line, no index number
                                print(
                                    f"  {marker}→ {_sanitize(item.text, 80)} (running)"
                                )
                            else:
                                print(
                                    f"  {marker}{item.index}. {_sanitize(item.text, 80)}"
                                )

                elif cmd == "/retry":
                    result = execute_retry(agent)
                    if result.success:
                        await agent.add_message(result.data["user_text"])
                    else:
                        print(f"  {result.message}")

                elif cmd == "/prioritise":
                    result = execute_prioritise(agent, cmd_arg)
                    print(f"  {result.message}")

                elif cmd == "/deprioritise":
                    result = execute_deprioritise(agent, cmd_arg)
                    print(f"  {result.message}")

                elif cmd == "/delete":
                    result = execute_delete(agent, cmd_arg)
                    if result.success:
                        print(f"  {result.message}")
                    else:
                        print(f"  {result.message}")

                elif cmd == "/journal":
                    args = cmd_arg.strip().lower()
                    if args == "on":
                        agent.journal_mode = True
                        print("  Journal mode enabled")
                        print(
                            "  Context will be compacted via reflection before each new message."
                        )
                    elif args == "off":
                        agent.journal_mode = False
                        print("  Journal mode disabled")
                    elif args == "last":
                        await agent.add_message("/journal last", kind="journal_last")
                    elif args == "all":
                        await agent.add_message("/journal all", kind="journal_all")
                    elif args == "status" or not args:
                        status = "on" if agent.journal_mode else "off"
                        print(f"  Journal mode: {status}")
                    else:
                        print("  Usage: /journal [on|off|last|all|status]")
                        print("    /journal on      - Enable context compaction")
                        print("    /journal off     - Disable context compaction")
                        print(
                            "    /journal last    - Journal the most recent tool-using turn"
                        )
                        print(
                            "    /journal all     - Journal all tool-using turns iteratively"
                        )
                        print("    /journal status  - Show current state")

                elif cmd == "/compact":
                    from agent13.prompts import resolve_compact_prompt

                    arg = cmd_arg.strip()
                    compact_prompt_text, error = resolve_compact_prompt(prompt_manager, arg)
                    if error:
                        for error_line in error.splitlines():
                            print(f"  {error_line}")
                    else:
                        # Track turn start so the idle handler prints
                        # "[complete]" and re-shows the prompt (same as the
                        # normal message dispatch below).
                        _processing_start_time = time.time()
                        queued_text = f"/compact {arg}" if arg else "/compact"
                        await agent.add_message(
                            queued_text,
                            kind="compact",
                            data={"compact_prompt": compact_prompt_text},
                        )

                elif cmd == "/model":
                    if not cmd_arg:
                        # List available models
                        if agent.available_models:
                            print("\nAvailable models:")
                            for i, name in enumerate(agent.available_models, 1):
                                marker = " *" if name == agent.model else ""
                                print(f"  {i}. {name}{marker}")
                            print()
                        else:
                            print(
                                "  No models loaded. Use /provider to switch providers."
                            )
                    else:
                        model = resolve_model_selection(agent.available_models, cmd_arg)
                        if model:
                            agent.set_model(model)
                            print(f"  Model set to: {model}")
                        # resolve_model_selection prints its own error

                elif cmd == "/provider":
                    if not cmd_arg:
                        names = get_provider_names()
                        if not names:
                            print("  No providers configured in ~/.agent13/config.toml")
                            user_input = await loop.run_in_executor(
                                None, read_with_prompt
                            )
                            continue
                        # Determine current provider from client base_url
                        current = ""
                        try:
                            current_url = str(agent.client.base_url)
                            for name in names:
                                prov = get_config().get_provider(name)
                                if prov and prov.api_base.rstrip("/") in current_url:
                                    current = name
                                    break
                        except Exception:
                            pass
                        print("\nAvailable providers:")
                        for i, name in enumerate(names, 1):
                            marker = " *" if name == current else ""
                            print(f"  {i}. {name}{marker}")
                        print()
                        print("  Use /provider <name> to switch")
                        user_input = await loop.run_in_executor(None, read_with_prompt)
                        continue
                    # Resolve numeric selection to provider name
                    resolved = resolve_provider_selection(cmd_arg)
                    if resolved is None:
                        # resolve_provider_selection prints its own error
                        user_input = await loop.run_in_executor(None, read_with_prompt)
                        continue
                    try:
                        base_url, api_key, read_timeout, connect_timeout = (
                            resolve_provider_arg(resolved)
                        )
                    except ValueError as e:
                        print(f"  Error: {e}")
                        user_input = await loop.run_in_executor(None, read_with_prompt)
                        continue
                    new_client = create_client(
                        base_url,
                        api_key,
                        read_timeout=read_timeout,
                        connect_timeout=connect_timeout,
                    )
                    agent.set_client(new_client)
                    provider = (
                        ""
                        if resolved.startswith("http://")
                        or resolved.startswith("https://")
                        else resolved
                    )
                    print(f"  Provider changed to: {base_url}")
                    # Fetch and display models
                    try:
                        models = await fetch_models(new_client)
                        agent.available_models = models
                        print("  Available models:")
                        for i, name in enumerate(models, 1):
                            print(f"    {i}. {name}")
                        print()
                        print("  Use /model <name|number> to select")
                    except Exception as e:
                        print(f"  Warning: Could not fetch models: {e}")

                elif cmd == "/sandbox":
                    args = cmd_arg.strip().lower()
                    if not args:
                        current = get_current_sandbox_mode()
                        session = get_session_sandbox_mode()
                        config_default = get_default_sandbox_mode()
                        pinned = get_pinned_sandbox_mode()
                        print("\n  Sandbox Configuration:")
                        print(f"    Current mode: {current.value}")
                        if session:
                            print(f"    Session override: {session.value}")
                        else:
                            print("    Session override: none (using config default)")
                        print(f"    Config default: {config_default.value}")
                        if pinned:
                            print(f"    Pinned for this project: {pinned.value}")
                        else:
                            print("    Pinned for this project: none")
                        print()
                        from agent13.sandbox import SandboxMode

                        for mode in SandboxMode:
                            print(f"    - {mode.value}")
                        print()
                        print("  /sandbox <mode>  set session mode")
                        print("  /sandbox pin     pin current mode for this project")
                        print("  /sandbox unpin   remove pin for this project")
                    elif args == "pin":
                        current = get_current_sandbox_mode()
                        pin_sandbox_mode(current)
                        print(f"  Pinned sandbox mode '{current.value}' for this project")
                        print("  This mode will auto-apply on startup in this directory.")
                    elif args == "unpin":
                        if unpin_sandbox_mode():
                            print("  Removed sandbox pin for this project")
                        else:
                            print("  No sandbox pin exists for this project")
                    else:
                        try:
                            mode = parse_sandbox_mode(args)
                            set_session_sandbox_mode(mode)
                            print(f"  Sandbox mode set to: {mode.value}")
                        except ValueError as e:
                            print(f"  Error: {e}")

                elif cmd == "/devel":
                    args = cmd_arg.strip().lower()
                    if args == "on":
                        agent.set_devel_mode(True)
                        print("  Devel mode enabled")
                        print("  Devel-group tools are now visible to the AI.")
                    elif args == "off":
                        agent.set_devel_mode(False)
                        print("  Devel mode disabled")
                        print("  Devel-group tools are now hidden from the AI.")
                    elif args == "status" or not args:
                        status = "on" if agent.devel_mode else "off"
                        print(f"  Devel mode: {status}")
                    else:
                        status = "on" if agent.devel_mode else "off"
                        print("  Usage: /devel [on|off|status]")
                        print("    /devel on      - Show devel-group tools to the AI")
                        print("    /devel off     - Hide devel-group tools from the AI")
                        print("    /devel status  - Show current state")
                        print(f"  Current: {status}")

                elif cmd == "/tools":
                    summary = get_tool_stats_summary(agent)
                    if summary["total_calls"] == 0:
                        print("  No tool calls yet this session")
                    else:
                        rate = summary["success_rate"]
                        print(
                            f"\n  Tool Usage  {summary['total_successes']}/{summary['total_calls']} successful ({rate:.0f}%)"
                        )
                        print()
                        for tool in summary["per_tool"]:
                            print(
                                f"    {tool['name']}  {tool['successes']}/{tool['calls']}"
                            )

                elif cmd == "/mcp":
                    args = cmd_arg.strip().lower()
                    if args == "connect":
                        if not agent._mcp_server_configs:
                            print("  No MCP servers configured")
                        else:
                            try:
                                mcp = await agent._ensure_mcp()
                                if mcp:
                                    info = await mcp.connect_all()
                                    print(f"  MCP connected: {list(info.keys())}")
                            except Exception as e:
                                print(f"  MCP connect error: {e}")
                    elif args == "disconnect":
                        if not agent.mcp:
                            print("  MCP not connected")
                        else:
                            await agent.disconnect_mcp()
                            print("  MCP servers disconnected")
                    elif args == "reload":
                        if not agent.mcp:
                            print("  MCP not initialized (no servers configured)")
                        else:
                            servers = await agent.mcp.reload()
                            print(f"  MCP reconnected: {list(servers.keys())}")
                    else:
                        # List MCP servers
                        if not agent.mcp:
                            configured = len(agent._mcp_server_configs)
                            print("\n  MCP Status:")
                            print("    Status: Not initialized")
                            print(f"    Configured servers: {configured}")
                            print()
                            print("  Use /mcp connect to connect to MCP servers.")
                        else:
                            servers = agent.mcp.get_server_info()
                            if not servers:
                                configured = len(agent._mcp_server_configs)
                                print("\n  MCP Status:")
                                print(f"    Configured servers: {configured}")
                                print()
                                print("  Use /mcp connect to connect to servers.")
                            else:
                                print(format_mcp_servers(servers, use_rich=False))

                elif cmd == "/cwd":
                    if cmd_arg:
                        path_str = cmd_arg.strip().lstrip("@")
                        path = os.path.abspath(os.path.expanduser(path_str))
                        if not os.path.exists(path):
                            print(f"  Path does not exist: {path}")
                        elif not os.path.isdir(path):
                            print(f"  Not a directory: {path}")
                        else:
                            try:
                                os.chdir(path)
                                print(f"  Changed directory: {path}")
                            except OSError as e:
                                print(f"  Cannot change directory: {e}")
                    else:
                        print(f"  {Path.cwd()}")

                elif cmd == "/upgrade":
                    try:
                        from agent13.updater import (
                            check_and_apply_update,
                            UpdateStatus,
                        )

                        def _confirm(tag: str) -> bool:
                            try:
                                c = input(f"  Update to {tag}? [y/N] ")
                            except (EOFError, KeyboardInterrupt):
                                return False
                            return c.strip().lower() in ("y", "yes")

                        def _on_status(msg: str) -> None:
                            print(f"  {msg}")

                        result = check_and_apply_update(
                            confirm=_confirm, on_status=_on_status
                        )

                        if result.status is UpdateStatus.UPDATED:
                            print(f"  {result.message}")
                            print("  Restart agent13 to use the new version.")
                        elif result.status is UpdateStatus.UP_TO_DATE:
                            print(f"  {result.message}")
                        elif result.status is UpdateStatus.CANCELLED:
                            print(f"  {result.message}")
                        elif result.status is UpdateStatus.UNREACHABLE:
                            print(f"  {result.message}")
                            print("  You can manually upgrade with:")
                            print("    uv tool install --force agent13")
                        elif result.status is UpdateStatus.FAILED:
                            print(f"  Update failed: {result.message}")
                            if result.manual_cmd:
                                print("  Manual install:")
                                print(f"    {result.manual_cmd}")
                            else:
                                print("  No wheel asset available for manual install.")
                    except Exception as e:
                        print(f"  Update check failed: {e}")

                elif cmd == "/polite":
                    from agent13.commands import execute_polite

                    result = execute_polite(agent, cmd_arg)
                    if result.success:
                        print(f"  {result.message}")
                    else:
                        # Usage/error — print multi-line message indented
                        for line in result.message.splitlines():
                            print(f"  {line}")

                elif cmd == "/bell":
                    args = cmd_arg.strip().lower()
                    if not args:
                        st = bell.status_text()
                        if st == "off":
                            print("  Bell: off")
                        elif st == "always":
                            print("  Bell: on (always)")
                        else:
                            print(f"  Bell: on ({st})")
                    elif args == "off":
                        bell.disable()
                        print("  Bell: off")
                    else:
                        try:
                            val = float(args)
                            if val < 0:
                                raise ValueError("negative")
                            bell.set_threshold(val)
                            if val == 0:
                                print("  Bell: on (always)")
                            else:
                                print(f"  Bell: on ({val:.0f}s)")
                        except ValueError:
                            print("  Usage: /bell [N|off]")
                            print("    /bell 30  - Ring bell after 30s")
                            print("    /bell 0   - Always ring on idle")
                            print("    /bell off - Disable bell")
                            print("    /bell     - Show current status")

                elif cmd == "/bell-command":
                    args = cmd_arg.strip()
                    if not args:
                        if bell.command:
                            print(f"  Bell command: {bell.command}")
                        else:
                            print("  Bell command: (terminal bell)")
                    elif args.lower() == "off":
                        bell.clear_command()
                        print("  Bell command: cleared (terminal bell)")
                    else:
                        # Strip surrounding quotes if present
                        if len(args) >= 2 and args[0] in "\"'" and args[-1] == args[0]:
                            args = args[1:-1]
                        if not bell.set_command(args):
                            print(f"  Error: '{args.split()[0]}' is not executable")
                            print("  The first token must be found in PATH.")
                            print("  /bell-command off - Revert to terminal bell")
                            print("  /bell-command     - Show current status")
                        else:
                            print(f"  Bell command: {args}")

                elif cmd == "/auto_compact_threshold":
                    args = cmd_arg.strip()
                    if not args:
                        if agent.auto_compact_threshold > 0:
                            print(f"  Auto-compact: on ({agent.auto_compact_threshold:,} tokens)")
                        else:
                            print("  Auto-compact: off")
                    else:
                        try:
                            val = args.lower()
                            if val.endswith("k"):
                                threshold = int(val[:-1]) * 1000
                            else:
                                threshold = int(val)
                            if threshold < 0:
                                raise ValueError("negative")
                            agent.auto_compact_threshold = threshold
                            if threshold == 0:
                                print("  Auto-compact: off")
                            else:
                                print(f"  Auto-compact: on ({threshold:,} tokens)")
                        except ValueError:
                            print("  Usage: /auto_compact_threshold [N|0]")
                            print("    /auto_compact_threshold 150k - Compact at 150,000 tokens")
                            print("    /auto_compact_threshold 0    - Disable")
                            print("    /auto_compact_threshold      - Show current status")

                elif cmd == "/auto_compact_max":
                    args = cmd_arg.strip()
                    if not args:
                        print(
                            f"  Auto-compact max cycles: {agent.auto_compact_max_iterations}"
                        )
                    else:
                        try:
                            max_iter = int(args)
                            if max_iter < 1:
                                raise ValueError("must be >= 1")
                            agent.auto_compact_max_iterations = max_iter
                            print(f"  Auto-compact max cycles: {max_iter}")
                        except ValueError:
                            print("  Usage: /auto_compact_max [N]")
                            print("    /auto_compact_max 3 - Pause after 3 compact-and-continue cycles")
                            print("    /auto_compact_max   - Show current value")

                else:
                    print(f"  Unknown command: {user_input}")
                    print("  Type /help for available commands")

                # Stay in input mode unless entering streaming mode
                if not enter_streaming:
                    user_input = await loop.run_in_executor(None, read_with_prompt)
                    continue
                # else: fall through to STREAMING MODE below

            # ── Parse priority/interrupt prefixes ─────────────────
            if user_input.startswith("!!"):
                interrupt = True
                priority = True
                message_text = user_input[2:].strip()
            elif user_input.startswith("!"):
                interrupt = False
                priority = True
                message_text = user_input[1:].strip()
            else:
                interrupt = False
                priority = False
                message_text = user_input

            if not message_text:
                user_input = await loop.run_in_executor(None, read_with_prompt)
                continue

            # Record in shared history (same store as TUI)
            history.add(message_text)
            # ── Send message to agent ─────────────────────────────
            if interrupt or priority:
                # Always send immediately — let the agent queue handle ordering
                _processing_start_time = time.time()
                await agent.add_message(
                    message_text, priority=priority, interrupt=interrupt
                )
                if interrupt:
                    output_ctrl.write(
                        "[interrupt] Sent \u2014 will take over"
                        " at next natural boundary\n"
                    )
                else:
                    output_ctrl.write("[priority] Sent \u2014 will process next\n")
            elif agent.is_idle:
                # Agent is idle — send immediately
                _processing_start_time = time.time()
                output_ctrl.write("[processing]\n")
                await agent.add_message(expand_file_mentions(message_text))
            else:
                # Agent is busy — queue via agent (same as TUI)
                await agent.add_message(expand_file_mentions(message_text))
                output_ctrl.write(
                    "  (queued \u2014 will process after current response)\n"
                )

            # ── STREAMING MODE ────────────────────────────────────
            # Unfreeze output — agent events flow to stdout
            enter_streaming = False
            output_ctrl.unfreeze()

            # Wait for Enter (no prompt — user watches streaming output)
            user_input = await loop.run_in_executor(None, read_no_prompt)
            # Enter pressed — freeze set in read_no_prompt finally

            if user_input.strip():
                # User typed during streaming — it's their next message
                continue
            else:
                # Empty Enter — show prompt for input
                user_input = await loop.run_in_executor(None, read_with_prompt)

    except (KeyboardInterrupt, EOFError) as exc:
        is_eof = isinstance(exc, EOFError)
        if in_multi_mode:
            # Both Ctrl+C and Ctrl+D in multi-mode: cancel buffer + exit
            in_multi_mode = False
            multi_buffer = []
            print("\n  Multi-line input cancelled")
        elif is_eof:
            # Ctrl+D on empty line: exit cleanly (same as /quit)
            print("\n  EOF — exiting")
        else:
            print("\n  Interrupted")

    finally:
        # Flush any buffered output before cleanup
        output_ctrl.unfreeze()

        # ── Cleanup ──────────────────────────────────────────────
        shutting_down = True  # Stop event handlers from writing

        # Cancel any pending bell timer so it can't fire after teardown.
        bell.cancel()

        # Auto-save session
        if agent.messages:
            from agent13.persistence import save_context, get_auto_save_path

            auto_path = get_auto_save_path(session_date=agent.session_date)
            try:
                save_context(agent, auto_path)
                print(f"\nSession saved to {auto_path}")
            except Exception as e:
                print(f"\nWarning: Could not save session: {e}")

        # Save readline history
        if _histfile:
            try:
                import readline as _rl

                os.makedirs(os.path.dirname(_histfile), exist_ok=True)
                _rl.write_history_file(_histfile)
            except Exception:
                pass

        agent.stop()
        agent_task.cancel()
        try:
            await asyncio.wait_for(agent_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        if output_file:
            try:
                output_file.flush()
            except ValueError:
                pass  # Already closed
            try:
                output_file.close()
            except ValueError:
                pass

        print("Goodbye!")
