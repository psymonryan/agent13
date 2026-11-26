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
import time
from typing import Optional

from rich.console import Console

from agent13 import (
    Agent,
    AgentEvent,
    History,
    PromptManager,
    StopReason,
    get_filtered_tools,
    execute_tool,
    skill_manager_ctx,
    init_debug,
    get_config,
)
from ui.display import RichDisplay
from agent13.timing import TokenTimingTracker
from agent13.durable_records import startup_focus
from agent13.file_injection import expand_file_mentions
from agent13.bell import BellManager
from agent13.models import fetch_models, format_model_list
from agent13.commands import (
    command_names,
    execute_collapse,
    execute_pretty,
    format_help_sections,
    get_command,
    promote_latest,
    run_command,
    run_mcp_action,
)
from agent13.message_history import content_to_text


# Commands the REPL implements itself (the dispatch chain in run_repl).
# Shared commands come from the registry in agent13.commands.
_REPL_LOCAL_COMMANDS = frozenset(
    {
        "exit",
        "help",
        "multi",
        "pause",
        "pretty",
        "reasoning",
        "resume",
        "stop",
        "toolview",
        "upgrade",
    }
)

# Catalogue entries the REPL deliberately does not offer - their work is about
# the TUI's own widgets (spinner, clipboard, chat rebuild, display wipe).
#
# /clear is listed here but still has a branch in the dispatch chain: it used
# to do something in the REPL, so typing it earns a pointer at /new_session
# rather than "Unknown command".
_NOT_IN_REPL = frozenset(
    {
        "clear",
        "clipboard",
        "rebuild",
        "spinner",
    }
)

# Slash commands available in REPL mode, derived from the shared registry.
COMMANDS = {
    f"/{name}": get_command(name).description
    for name in command_names()
    if get_command(name).handler is not None or name in _REPL_LOCAL_COMMANDS
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

    @property
    def frozen(self) -> bool:
        return self._frozen

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


def _drain_if_busy(agent) -> None:
    """Let an in-flight turn finish before exiting on stdin EOF.

    SYNC variant — for the input worker thread (read_no_prompt): the main
    thread is parked in the event loop awaiting this worker, so a sleep
    poll does not block the turn from completing. A stdin EOF (pipe
    closed, or a spurious Windows-pipe EOF) must not cancel the user's
    running turn — finish it, then exit. Deliberately NOT applied to
    Ctrl+C: an interrupt should cancel immediately.
    """
    if agent.is_idle:
        return
    print("\n  stdin closed — waiting for the current turn to finish…", flush=True)
    while not agent.is_idle:
        time.sleep(0.2)


async def _drain_if_busy_async(agent) -> None:
    """ASYNC variant of _drain_if_busy — for the main thread (EOFError path).

    Must yield to the event loop while waiting: the in-flight turn runs
    ON that loop, so a blocking sleep here would deadlock (turn can't
    finish -> is_idle never flips -> drain polls forever).
    """
    if agent.is_idle:
        return
    print("\n  stdin closed — waiting for the current turn to finish…", flush=True)
    while not agent.is_idle:
        await asyncio.sleep(0.2)


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
    skills_notice: Optional[str] = None,
    snippet_manager=None,
    continue_session: bool = False,
    output_path: Optional[str] = None,
    model_names: Optional[list[str]] = None,
    read_files: list[str] | None = None,
    focus_name: str | None = None,
    polite_interval: float | None = None,
    bell_threshold: float | str | None = None,
    bell_enabled: bool = True,
    bell_command: str = "",
    priming_enabled: bool = False,
    reasoning_collapsed: bool = True,
    toolview_collapsed: bool = True,
    auto_context_threshold: int = 220000,
    auto_context_action: str = "report_and_compact",
    auto_context_chain: int = 3,
    auto_context_nudge_cap: int = 2,
    auto_save_idle_seconds: int = 300,
    datetime_nudge_seconds: int = 3600,
    report_and_compact_prompt: str = "",
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

    display = RichDisplay(
        console=chat_console,
        pretty=pretty,
        debug=debug,
        reasoning_collapsed=reasoning_collapsed,
        toolview_collapsed=toolview_collapsed,
    )

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
        auto_context_threshold=auto_context_threshold,
        auto_context_action=auto_context_action,
        auto_context_chain=auto_context_chain,
        auto_context_nudge_cap=auto_context_nudge_cap,
        auto_save_idle_seconds=auto_save_idle_seconds,
        datetime_nudge_seconds=datetime_nudge_seconds,
        report_and_compact_prompt=report_and_compact_prompt,
    )

    # Store available models on agent
    agent.available_models = model_names or []
    # Provider name for /status - the agent owns it so /provider can change it
    # without leaving a stale copy in this scope.
    agent.provider_name = provider
    # Managers the shared command handlers read (/prompt, /skills, /snippet,
    # /compact, /status). cli.py constructs them once for every mode.
    agent.prompt_manager = prompt_manager
    agent.skill_manager = skill_manager
    agent.snippet_manager = snippet_manager
    # Display flags /status reports. The REPL has no /spinner or /clipboard,
    # so those keep the agent's defaults.
    agent.reasoning_collapsed = reasoning_collapsed
    agent.toolview_collapsed = toolview_collapsed

    # Focus name from --focus: authoritative state; on a fresh session the
    # records become the first turn - the model reads them and replies a
    # minimal ack while the user types (the --read pattern). The agent loop
    # starts below, so the queued item is processed immediately.
    # One wiring for all four UIs - see agent13.durable_records.startup_focus.
    message, _level, enqueue_msg = startup_focus(
        agent.durable_records, focus_name, agent.messages
    )
    if enqueue_msg:
        await agent.add_message(enqueue_msg, kind="prompt")
    if message:
        print(message, file=sys.stderr)

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
    # The shared /bell and /bell-command handlers reach the manager through
    # the agent, so the same code serves both front-ends.
    agent.bell = bell

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

    # Startup notice from the default-skills sync (cli.py runs it before any
    # front-end exists, so it hands the report over instead of printing).
    if skills_notice:
        print(skills_notice)

    # ── State tracking ───────────────────────────────────────────────
    # Note: agent.is_idle, agent.status, agent.pause_state are the
    # single source of truth — same as TUI. No duplicate flags.
    shutting_down = False  # Guards event handlers during cleanup
    enter_streaming = False  # Set by /resume to switch to streaming mode
    in_multi_mode = False
    multi_buffer: list[str] = []

    # TPS / timing tracking (delegated to TokenTimingTracker). The tracker,
    # the session clock and the last-turn stamps live on the agent so the
    # shared /status can read them.
    agent.tracker = TokenTimingTracker()
    agent.session_start_time = time.time()
    agent.last_turn_duration = None  # seconds
    agent.last_turn_end_time = None  # epoch
    _processing_start_time: Optional[float] = None

    # ── Event handlers ───────────────────────────────────────────────
    # Wire agent events to RichDisplay (chat content) and OutputController
    # (status notifications). Status goes through output_ctrl so it buffers
    # when the user is in input mode.

    def _print_above_prompt(write) -> None:
        """Run *write* and, if the user is at the prompt, redraw it after.

        A deferred action (/load, /new_session) completes in the agent loop
        while the REPL is parked at the ``> `` prompt. Printing there would
        leave the text glued to the prompt with no new one after it, so
        release the buffer, print, then redraw. Mid-turn the output is
        already streaming, so just print.
        """
        if not output_ctrl.frozen:
            write()
            return
        output_ctrl.unfreeze()
        write()
        sys.__stdout__.write("\n> ")
        sys.__stdout__.flush()
        output_ctrl.freeze()

    @agent.on_event
    async def on_status_change(event):
        if event.event != AgentEvent.STATUS_CHANGE or shutting_down:
            return
        nonlocal _processing_start_time
        status = event.data.get("status", "")
        if status == "paused":
            output_ctrl.write("[paused]\n")
            bell.on_paused()
            return
        if status == "idle" and agent.messages:
            agent.tracker.turn_end()
            bell.on_turn_end()
            # Duration notification
            if _processing_start_time:
                elapsed = time.time() - _processing_start_time
                agent.last_turn_duration = elapsed
                agent.last_turn_end_time = time.time()
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
            agent.tracker.turn_start()
            bell.on_turn_start()

    @agent.on_event
    async def on_item_started(event):
        if event.event != AgentEvent.ITEM_STARTED:
            return

    @agent.on_event
    async def on_stream_start(event):
        if event.event != AgentEvent.STREAM_START or shutting_down:
            return
        agent.tracker.reset_stream()

    @agent.on_event
    async def on_token(event):
        if event.event != AgentEvent.ASSISTANT_TOKEN or shutting_down:
            return
        now = time.time()
        if agent.tracker.is_first_token:
            display.start_response()
        agent.tracker.record_token(now)
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
    async def on_notification(event):
        if event.event != AgentEvent.NOTIFICATION or shutting_down:
            return
        message = event.data.get("message", "")
        level = event.data.get("level", "info")
        color = {"warning": "yellow", "error": "red"}.get(level, "bright_blue")
        output_ctrl.write(f"[{color}]{message}[/]\n")

    @agent.on_event
    async def on_degenerate_nudge(event):
        if event.event != AgentEvent.DEGENERATE_NUDGE or shutting_down:
            return
        message = event.data.get("message", "")
        level = event.data.get("level", "info")
        color = "yellow" if level == "warning" else "bright_blue"
        output_ctrl.write(f"[{color}]{message}[/]\n")

    @agent.on_event
    async def on_tool_call(event):
        if event.event != AgentEvent.TOOL_CALL or shutting_down:
            return
        name = event.data.get("name", "")
        arguments = event.data.get("arguments", {})
        seq = event.data.get("seq")
        # Folded: the gear line below is the only line for the call, so drop
        # the redundant [tool: name] progress marker. Expanded keeps it, as a
        # sibling of [processing].
        if not display.toolview_collapsed:
            output_ctrl.write(f"[tool: {name}]\n")
        display.show_tool_call(name, arguments, seq)

    @agent.on_event
    async def on_tool_result(event):
        if event.event != AgentEvent.TOOL_RESULT or shutting_down:
            return
        result = event.data.get("result", "")
        seq = event.data.get("seq")
        display.show_tool_result(result, seq)

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
        result = agent.tracker.compute_tps(event.data)
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

        def _show() -> None:
            if success:
                print(f"\n  {message}")
                if incomplete:
                    print(
                        "  Warning: Incomplete turn detected. Use /resume to continue."
                    )
                _display_loaded_messages(agent)
            else:
                print(f"  Error: {message}")

        # /load is deferred, so this lands while the user may be at the prompt.
        _print_above_prompt(_show)

    @agent.on_event
    async def on_messages_cleared(event):
        if event.event != AgentEvent.MESSAGES_CLEARED or shutting_down:
            return
        # A deferred clear (/new_session) landed at a safe boundary. The user
        # is usually sitting at the prompt, so print above it and redraw -
        # the same dance the turn-complete line does. Mid-turn the output is
        # already streaming, so just write.
        count = event.data.get("count", 0)
        _print_above_prompt(lambda: print(f"  Cleared {count} messages"))

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
                _drain_if_busy(agent)
                return "/quit"
            return line.rstrip("\n")
        except KeyboardInterrupt:
            return "/quit"
        finally:
            # Freeze output before returning — close race window
            output_ctrl.freeze()

    async def wait_stream_exit() -> str:
        """Streaming-mode exit — unfreeze, wait for Enter, return next input.

        Typed-ahead text is the user's next message (dispatched at loop top);
        empty Enter returns to input mode (read with prompt).
        """
        output_ctrl.unfreeze()
        user_input = await loop.run_in_executor(None, read_no_prompt)
        # Enter pressed — freeze set in read_no_prompt finally
        if user_input.strip():
            # User typed during streaming — it's their next message
            return user_input
        # Empty Enter — show prompt for input
        return await loop.run_in_executor(None, read_with_prompt)

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

                # Shared commands run from the registry. The chain below
                # covers the ones this front-end implements itself.
                shared = run_command(cmd.lstrip("/"), agent, cmd_arg)
                if shared is not None:
                    for line in shared.message.splitlines():
                        # Indent the text, but leave blank lines blank rather
                        # than emitting two spaces of trailing whitespace.
                        print(f"  {line}" if line else "")
                    # /provider leaves the async tail to the front-end: fetch
                    # the new provider's models and show them.
                    base_url = shared.data.get("base_url")
                    if base_url:
                        try:
                            models = await fetch_models(agent.client)
                            agent.available_models = models
                            for line in format_model_list(models).splitlines():
                                print(f"  {line}")
                            print("  Use /model <name|number> to select")
                        except Exception as e:
                            print(f"  Warning: Could not fetch models: {e}")
                    # /retry leaves its tail here too: the REPL re-queues the
                    # text at once, where the TUI loads it into the input.
                    user_text = shared.data.get("user_text")
                    if user_text is not None:
                        await agent.add_message(user_text)
                    # /wrapup, /chain, /journal last|all queue an agent turn
                    # the same way. Track the turn start so the idle handler
                    # prints "[complete]" and re-shows the prompt.
                    agent_message = shared.data.get("agent_message")
                    if agent_message:
                        text, kind, data = agent_message
                        _processing_start_time = time.time()
                        await agent.add_message(text, kind=kind, data=data)
                    # /mcp connect|disconnect|reload: the action is async.
                    mcp_action = shared.data.get("mcp_action")
                    if mcp_action:
                        _ok, mcp_message = await run_mcp_action(agent, mcp_action)
                        print(f"  {mcp_message}")
                    # /new_session: always deferred through the queue. The
                    # confirmation comes from the MESSAGES_CLEARED handler
                    # below, so the wording matches the TUI's.
                    clear_mode = shared.data.get("clear_mode")
                    if clear_mode:
                        await agent.request_clear(mode=clear_mode)
                    # /load: deferred through the queue; on_context_loaded
                    # prints the result and re-displays the loaded messages.
                    load_path = shared.data.get("load_path")
                    if load_path:
                        await agent.request_load(load_path)
                    user_input = await loop.run_in_executor(None, read_with_prompt)
                    continue

                if cmd in ("/quit", "/exit"):
                    break

                elif cmd == "/help":
                    for section in format_help_sections(_REPL_LOCAL_COMMANDS):
                        print(f"\n  {section.title}:")
                        for entry in section.entries:
                            print(f"    /{entry.name:22s} {entry.description}")
                            for detail in entry.detail:
                                print(f"      {detail}")
                    print("\n  Press Enter (empty line) to switch between modes.")
                    print("  - Streaming mode: output flows, no prompt.")
                    print(
                        "  - Input mode: prompt shown, output pauses, type your message."
                    )

                elif cmd == "/pause":
                    if agent.is_paused:
                        print("  Already paused")
                    elif agent.is_pausing:
                        print("  Already pausing")
                    elif not agent.has_pending_work:
                        # has_pending_work, not is_idle: a message that was
                        # just sent is queued but not yet started, and the
                        # agent still reads idle for that window.
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
                    elif agent.has_incomplete_turn:
                        # Resume an incomplete turn loaded via --continue or
                        # /load. Mirrors the TUI: signal run() to call
                        # continue_incomplete_turn() inside its own loop —
                        # pending tools execute, then the LLM continues with
                        # the identical message prefix (kv-cache friendly).
                        agent.request_continue_incomplete()
                        output_ctrl.unfreeze()
                        print("  Continuing incomplete turn...")
                        enter_streaming = True
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

                elif cmd == "/multi":
                    in_multi_mode = True
                    multi_buffer = []
                    print("  Multi-line mode. End with . on its own line.")
                    user_input = await loop.run_in_executor(None, read_multi_prompt)
                    continue

                elif cmd == "/clear":
                    arg = cmd_arg.strip().lower()
                    if arg == "all":
                        print(
                            "  /clear all has moved: use /new_session "
                            "(wipe history, start a new session)"
                        )
                    elif arg.isdigit():
                        # /clear N in REPL — no display to rebuild,
                        # history untouched
                        print(
                            f"  History preserved ({len(agent.messages)} messages). "
                            "Use /new_session to wipe."
                        )
                    else:
                        # /clear with no args in REPL — no display to wipe,
                        # history untouched
                        print(
                            f"  History preserved ({len(agent.messages)} messages). "
                            "Use /new_session to wipe."
                        )

                elif cmd == "/pretty":
                    def _set_pretty(value: bool) -> None:
                        display.pretty = value

                    result = execute_pretty(cmd_arg, display.pretty, _set_pretty)
                    for line in result.message.splitlines():
                        print(f"  {line}")

                elif cmd in ("/reasoning", "/toolview"):
                    kind = cmd[1:]
                    current = (
                        agent.reasoning_collapsed
                        if kind == "reasoning"
                        else agent.toolview_collapsed
                    )

                    def _set_collapse(value: bool, _kind=kind) -> None:
                        # The agent owns the value (/status reads it); the
                        # display keeps the copy its renderers consult.
                        if _kind == "reasoning":
                            agent.reasoning_collapsed = value
                            display.reasoning_collapsed = value
                        else:
                            agent.toolview_collapsed = value
                            display.toolview_collapsed = value

                    result = execute_collapse(kind, cmd_arg, current, _set_collapse)
                    for line in result.message.splitlines():
                        print(f"  {line}")

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

                else:
                    # Dynamic /skillname: the shared helper loads the skill,
                    # ratchets the tool on and queues the turn.
                    from agent13.commands import execute_skill_invocation

                    cmd_name = cmd[1:]
                    cmd_args = user_input.split(maxsplit=1)
                    cmd_args = cmd_args[1] if len(cmd_args) > 1 else ""
                    skill_result = execute_skill_invocation(
                        agent, cmd_name, cmd_args
                    )
                    if skill_result is not None:
                        print(f"  {skill_result.message}")
                        agent_message = skill_result.data.get("agent_message")
                        if agent_message:
                            text, kind, data = agent_message
                            _processing_start_time = time.time()
                            await agent.add_message(text, kind=kind, data=data)
                    else:
                        print(f"  Unknown command: {user_input}")
                        print("  Type /help for available commands")

                # Stay in input mode unless entering streaming mode
                if not enter_streaming:
                    user_input = await loop.run_in_executor(None, read_with_prompt)
                    continue
                # Switch to streaming mode directly — the command text itself
                # must NOT leak into the send path below as a user message
                # (previously "/resume" was queued as a phantom message).
                enter_streaming = False
                user_input = await wait_stream_exit()
                continue

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
                # A bare "!"/"!!" promotes the most recently queued item.
                # It is a control, not a message: nothing is queued and
                # nothing enters history.
                if interrupt or priority:
                    # print, not output_ctrl.write: the controller BUFFERS
                    # while the user is at the prompt, and nothing unfreezes
                    # it before the next turn - the hint would never appear.
                    print(promote_latest(agent, interrupt))
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
                        "[steer] Sent \u2014 will take over"
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

            # Wait for Enter (no prompt — user watches streaming output);
            # typed-ahead text is their next message, empty Enter re-prompts.
            user_input = await wait_stream_exit()

    except (KeyboardInterrupt, EOFError) as exc:
        is_eof = isinstance(exc, EOFError)
        if in_multi_mode:
            # Both Ctrl+C and Ctrl+D in multi-mode: cancel buffer + exit
            in_multi_mode = False
            multi_buffer = []
            print("\n  Multi-line input cancelled")
        elif is_eof:
            # Ctrl+D on empty line: exit cleanly (same as /quit) — but let
            # a running turn finish first (EOF must not cancel it). Async
            # drain: we're on the main thread; the turn runs on this loop.
            await _drain_if_busy_async(agent)
            print("\n  EOF — exiting")
        else:
            print("\n  Interrupted")

    finally:
        # Unblock any worker thread still parked in a blocking stdin read —
        # a blocking C call that cancellation cannot reach. Without this,
        # asyncio.run()'s executor-shutdown join hangs the process at exit
        # (observed on Windows: REPL unable to exit while a streaming-mode
        # stdin read is in flight).
        try:
            sys.stdin.close()
        except Exception:
            pass

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

        agent.stop(StopReason.QUIT)
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
