#!/usr/bin/env python3

"""Headless agent runner for programmatic testing.

Minimal test harness that passes stdin to the agent and prints events
in a parseable format. No complex state management - just observe what
the agent does.

Output format:
    READY                          - Agent started, ready for input
    EVENT: STARTED|STOPPED|PAUSED|RESUMED
    STATUS: <status> | MODEL: <model> | QUEUE: <count>
    USER: <message>                - User message being processed
    ASSISTANT: <text>              - Assistant response
    TOOL: <name>(<args>)           - Tool call
    RESULT: <name>: <result>       - Tool result
    ERROR: [<type>] <message>      - Error occurred

Commands (stdin):
    /pause   - Pause during processing
    /resume  - Resume from pause
    /status  - Print current status
    /quit    - Exit

Example pexpect test:
    child = pexpect.spawn('uv run headless.py test --model 19')
    child.expect('READY')
    child.sendline('hello')
    child.expect('USER: hello')
    child.sendline('/quit')
    child.expect('DONE')
"""

import argparse
import asyncio
import sys
import time
from typing import Optional

from openai import AsyncOpenAI

from agent13.config import resolve_provider_arg, create_client, get_config
from agent13.core import Agent, AgentEvent
from agent13.prompts import PromptManager
from agent13.debug_log import init_debug
from agent13.models import fetch_models, select_model
from agent13.skills import SkillManager
from agent13.context import skill_manager_ctx
from tools import get_tools, execute_tool


async def run_headless(
    client: AsyncOpenAI,
    model: str,
    provider: str = "",
    prompt_manager: Optional[PromptManager] = None,
    debug: bool = False,
    journal_mode: bool = False,
    continue_session: bool = False,
):
    """Run the agent in headless mode, reading commands from stdin.

    This is a minimal test harness - it just passes stdin to the agent
    and prints events. No complex state management.

    Output format:
        STARTED | STOPPED | PAUSED | RESUMED
        STATUS: <status> | MODEL: <model> | QUEUE: <count>
        USER: <message>
        ASSISTANT: <text>
        TOOL: <name>(<args>)
        RESULT: <result>
        ERROR: <message>

    Commands: /pause, /resume, /quit
    """
    if debug:
        init_debug()

    prompt_manager = prompt_manager or PromptManager()

    # Set up skill manager context for skill tool
    skill_manager = SkillManager(lambda: get_config())
    skill_manager_ctx.set(skill_manager)

    def print_status(status: str, queue_count: int = 0):
        """Print current status in parseable format."""
        if provider:
            print(f"STATUS: {status} | MODEL: {provider}/{model} | QUEUE: {queue_count}", flush=True)
        else:
            print(f"STATUS: {status} | MODEL: {model} | QUEUE: {queue_count}", flush=True)

    # Create agent
    agent = Agent(
        client=client,
        model=model,
        system_prompt=prompt_manager.get_prompt(),
        tools=get_tools(),
        execute_tool=execute_tool,
        journal_mode=journal_mode,
    )

    # Load MCP server configs
    config = get_config()
    if config and config.mcp_servers:
        agent.set_mcp_servers(config.mcp_servers)
        print(f"MCP: {len(config.mcp_servers)} servers configured", flush=True)

    # Load previous session if --continue
    if continue_session:
        from agent13.persistence import find_latest_auto_save, load_context
        latest = find_latest_auto_save()
        if latest:
            success, msg = load_context(agent, str(latest))
            if success:
                print(f"CONTINUED: {latest} ({msg})", flush=True)
            else:
                print(f"CONTINUE_ERROR: {msg}", flush=True)
        else:
            print("CONTINUE: No auto-saved session found, starting fresh", flush=True)

    # Track status locally for display
    status = "initialising"

    # Status mapping for display (internal status -> display string)
    STATUS_DISPLAY = {
        "initialising": "initialising",
        "idle": "ready",
        "waiting": "waiting",
        "thinking": "thinking",
        "processing": "processing",
        "tooling": "tooling",
        "paused": "paused",
    }

    # Register event handlers - just print what the agent tells us
    # NOTE: Each handler MUST check event.event type, otherwise all handlers
    # run for every event (the @agent.on_event decorator registers for ALL events)
    @agent.on_event
    async def on_status_change(event):
        if event.event != AgentEvent.STATUS_CHANGE:
            return
        nonlocal status
        status = event.data.get("status", "unknown")
        display_status = STATUS_DISPLAY.get(status, status)
        print(f"STATUS: {display_status}", flush=True)

    @agent.on_event
    async def on_started(event):
        if event.event != AgentEvent.STARTED:
            return
        print("EVENT: STARTED", flush=True)
    @agent.on_event
    async def on_stopped(event):
        if event.event != AgentEvent.STOPPED:
            return
        nonlocal status
        status = "stopped"
        print("EVENT: STOPPED", flush=True)
    @agent.on_event
    async def on_paused(event):
        if event.event != AgentEvent.PAUSED:
            return
        print("EVENT: PAUSED", flush=True)
    @agent.on_event
    async def on_resumed(event):
        if event.event != AgentEvent.RESUMED:
            return
        print("EVENT: RESUMED", flush=True)
    @agent.on_event
    async def on_item_started(event):
        if event.event != AgentEvent.ITEM_STARTED:
            return
        text = event.data.get("text", "")
        print(f"USER: {text}", flush=True)
    # Track TPS timing
    _first_token_time: float | None = None
    _last_token_time: float | None = None
    _token_count: int = 0

    @agent.on_event
    async def on_stream_start(event):
        if event.event != AgentEvent.STREAM_START:
            return
        nonlocal _first_token_time, _last_token_time, _token_count
        _first_token_time = None
        _last_token_time = None
        _token_count = 0

    @agent.on_event
    async def on_assistant_token(event):
        if event.event != AgentEvent.ASSISTANT_TOKEN:
            return
        nonlocal _first_token_time, _last_token_time, _token_count
        now = time.time()
        if _first_token_time is None:
            _first_token_time = now
        _last_token_time = now
        _token_count += 1

    @agent.on_event
    async def on_assistant_complete(event):
        if event.event != AgentEvent.ASSISTANT_COMPLETE:
            return
        text = event.data.get("text", "")
        reasoning = event.data.get("reasoning", "")
        if reasoning:
            print(f"REASONING: {reasoning[:200]}..." if len(reasoning) > 200 else f"REASONING: {reasoning}", flush=True)
        if text:
            print(f"ASSISTANT: {text[:500]}..." if len(text) > 500 else f"ASSISTANT: {text}", flush=True)

    @agent.on_event
    async def on_tool_call(event):
        if event.event != AgentEvent.TOOL_CALL:
            return
        name = event.data.get("name", "")
        args = event.data.get("arguments", {})
        args_str = str(args)
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."
        print(f"TOOL: {name}({args_str})", flush=True)

    @agent.on_event
    async def on_tool_result(event):
        if event.event != AgentEvent.TOOL_RESULT:
            return
        name = event.data.get("name", "")
        result = event.data.get("result", "")
        if len(result) > 200:
            result = result[:197] + "..."
        result = result.replace("\n", "\\n")
        print(f"RESULT: {name}: {result}", flush=True)

    @agent.on_event
    async def on_error(event):
        if event.event != AgentEvent.ERROR:
            return
        message = event.message or "Unknown error"
        error_type = event.data.get("error_type", "unknown")
        print(f"ERROR: [{error_type}] {message}", flush=True)
    @agent.on_event
    async def on_journal_compact(event):
        if event.event != AgentEvent.JOURNAL_COMPACT:
            return
        summary = event.data.get("summary", "")
        tokens_before = event.data.get("tokens_before", 0)
        tokens_after = event.data.get("tokens_after", 0)
        # Truncate summary for display
        display_summary = summary[:100] + "..." if len(summary) > 100 else summary
        display_summary = display_summary.replace("\n", "\\n")
        print(f"JOURNAL_COMPACT: {tokens_before} -> {tokens_after} tokens", flush=True)
        print(f"  summary: {display_summary}", flush=True)

    @agent.on_event
    async def on_queue_update(event):
        if event.event != AgentEvent.QUEUE_UPDATE:
            return
        count = event.data.get("count", 0)
        print_status(status, count)

    @agent.on_event
    async def on_token_usage(event):
        if event.event != AgentEvent.TOKEN_USAGE:
            return
        data = event.data
        completion_tokens = data.get("completion_tokens", 0)

        # Calculate TPS with same thresholds as TUI
        MIN_TOKENS = 50
        MIN_ELAPSED = 1.0

        if _first_token_time and _last_token_time:
            elapsed = _last_token_time - _first_token_time
            if elapsed >= MIN_ELAPSED and completion_tokens >= MIN_TOKENS:
                tps = completion_tokens / elapsed
                print(f"TPS: {tps:.1f} (tokens={completion_tokens}, elapsed={elapsed:.2f}s)", flush=True)
            else:
                print(f"TPS: skipped (tokens={completion_tokens}, elapsed={elapsed:.2f}s < thresholds)", flush=True)

    @agent.on_event
    async def on_mcp_started(event):
        if event.event != AgentEvent.MCP_SERVER_STARTED:
            return
        server_name = event.data.get("server_name", "unknown")
        transport = event.data.get("transport", "unknown")
        print(f"MCP_STARTED: {server_name} ({transport})", flush=True)

    @agent.on_event
    async def on_mcp_ready(event):
        if event.event != AgentEvent.MCP_SERVER_READY:
            return
        server_name = event.data.get("server_name", "unknown")
        tool_count = event.data.get("tool_count", 0)
        print(f"MCP_READY: {server_name} ({tool_count} tools)", flush=True)

    @agent.on_event
    async def on_mcp_error(event):
        if event.event != AgentEvent.MCP_SERVER_ERROR:
            return
        server_name = event.data.get("server_name", "unknown")
        error = event.data.get("error", "unknown error")
        print(f"MCP_ERROR: {server_name}: {error}", flush=True)

    @agent.on_event
    async def on_mcp_stderr(event):
        if event.event != AgentEvent.MCP_SERVER_STDERR:
            return
        server_name = event.data.get("server_name", "unknown")
        line = event.data.get("line", "")
        if line.strip():
            print(f"MCP_STDERR: {server_name}: {line}", flush=True)

    # Start agent in background
    agent_task = asyncio.create_task(agent.run())

    print("READY", flush=True)

    # Create a queue for stdin lines
    input_queue: asyncio.Queue[str] = asyncio.Queue()
    stdin_done = False

    async def stdin_reader():
        """Background task to read stdin and put lines in queue."""
        nonlocal stdin_done
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:  # EOF
                    stdin_done = True
                    break
                await input_queue.put(line.rstrip('\n\r'))
            except Exception:
                break

    reader_task = asyncio.create_task(stdin_reader())

    try:
        while not stdin_done:
            try:
                line = await asyncio.wait_for(input_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if not line:
                continue

            # Handle commands - just pass through to agent
            if line.startswith("/"):
                cmd = line.lower()
                if cmd in ("/quit", "/exit"):
                    print("QUITTING", flush=True)
                    break
                elif cmd == "/pause":
                    if agent.is_paused:
                        print("ALREADY_PAUSED", flush=True)
                    elif status == "processing":
                        agent.pause()
                        print("PAUSING", flush=True)
                    else:
                        print("NOTHING_TO_PAUSE", flush=True)
                elif cmd == "/resume":
                    if agent.is_paused:
                        agent.resume()
                        print("RESUMING", flush=True)
                    else:
                        print("NOT_PAUSED", flush=True)
                elif cmd == "/status":
                    print_status(status, agent.queue.pending_count)
                elif cmd.startswith("/mcp"):
                    # Handle /mcp command
                    parts = line.split(maxsplit=1)
                    subcmd = parts[1].lower() if len(parts) > 1 else ""
                    if subcmd == "connect":
                        print("MCP: Connecting...", flush=True)
                        mcp = await agent._ensure_mcp()
                        if mcp:
                            info = await mcp.connect_all()
                            if info:
                                for name, tools in info.items():
                                    print(f"MCP_CONNECTED: {name} ({len(tools)} tools)", flush=True)
                            else:
                                print("MCP: No servers connected", flush=True)
                        else:
                            print("MCP: No servers configured", flush=True)
                    else:
                        if agent.mcp:
                            info = agent.mcp.get_server_info()
                            if info:
                                for name, tools in info.items():
                                    print(f"MCP_STATUS: {name} ({len(tools)} tools)", flush=True)
                            else:
                                print("MCP: No servers connected. Use /mcp connect", flush=True)
                        else:
                            count = len(agent._mcp_server_configs) if agent._mcp_server_configs else 0
                            print(f"MCP: Not initialized ({count} servers configured). Use /mcp connect", flush=True)
                elif cmd.startswith("/save"):
                    # Handle /save command
                    parts = line.split(maxsplit=2)
                    if len(parts) < 2:
                        print("USAGE: /save <name> [-y]", flush=True)
                    else:
                        name = parts[1]
                        force = len(parts) > 2 and parts[2] == "-y"
                        from agent13.persistence import save_context, get_saves_dir
                        saves_dir = get_saves_dir()
                        saves_dir.mkdir(parents=True, exist_ok=True)
                        path = saves_dir / f"{name}.ctx"
                        if path.exists() and not force:
                            print(f"EXISTS: {path}. Use /save {name} -y to overwrite", flush=True)
                        else:
                            save_context(agent, str(path))
                            print(f"SAVED: {path}", flush=True)
                elif cmd.startswith("/load"):
                    # Handle /load command
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        print("USAGE: /load <name>", flush=True)
                    else:
                        name = parts[1]
                        from agent13.persistence import load_context, get_saves_dir
                        path = get_saves_dir() / f"{name}.ctx"
                        if not path.exists():
                            print(f"NOT_FOUND: {path}", flush=True)
                        else:
                            success, msg = load_context(agent, str(path))
                            if success:
                                print(f"LOADED: {path} ({msg})", flush=True)
                            else:
                                print(f"LOAD_ERROR: {msg}", flush=True)
                else:
                    print(f"UNKNOWN_COMMAND: {line}", flush=True)
            else:
                # Send message to agent
                await agent.add_message(line)

    finally:
        # Auto-save on exit if there are messages
        if agent.messages:
            from agent13.persistence import save_context, get_auto_save_path
            auto_path = get_auto_save_path()
            save_context(agent, str(auto_path))
            print(f"AUTO_SAVED: {auto_path}", flush=True)
        agent.stop()
        reader_task.cancel()
        try:
            await asyncio.wait_for(agent_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        print("DONE", flush=True)


async def async_main():
    """Parse args and run headless mode."""
    parser = argparse.ArgumentParser(
        description="Headless agent runner for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Output format:
    READY | EVENT: STARTED|STOPPED|PAUSED|RESUMED
    STATUS: <status> | MODEL: <model> | QUEUE: <count>
    USER: <message> | ASSISTANT: <text> | TOOL: <name>(<args>)
    RESULT: <name>: <result> | ERROR: [<type>] <message>

Commands: /pause, /resume, /status, /quit
"""
    )
    parser.add_argument(
        "provider",
        nargs="?",
        help="Provider name from config or OpenAI-compatible URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (number or name)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Enable journal mode (context compaction)"
    )
    parser.add_argument(
        "-c", "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue from last auto-saved session"
    )

    args = parser.parse_args()

    if not args.provider:
        parser.error("provider argument is required")

    # Resolve provider
    try:
        base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(args.provider)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    provider_name = "" if args.provider.startswith("http") else args.provider

    # Create client
    client = create_client(base_url, api_key, read_timeout=read_timeout, connect_timeout=connect_timeout)

    # Fetch models
    try:
        model_names = await fetch_models(client)
    except RuntimeError as e:
        err_msg = str(e)
        is_connection_error = any(
            s in err_msg.lower()
            for s in ["connection error", "connection refused", "could not resolve", "timed out", "name or service not known"]
        )

        if is_connection_error:
            # Provider is unreachable - no point continuing
            print(f"Error: Provider is unreachable: {e}", file=sys.stderr)
            sys.exit(1)
        elif not args.model:
            # No model specified - can't proceed without model list
            print(f"Error: Could not fetch models: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            # Non-connection error (e.g. /models endpoint broken) - try proceeding with specified model
            print(f"Warning: Could not fetch models: {e}", file=sys.stderr)
            model_names = []

    # Select model - if model list unavailable and user specified a model, use it directly
    if not model_names and args.model:
        model = args.model
        print(f"Using model '{model}' (model list unavailable, using specified name directly)", file=sys.stderr)
    else:
        model = await select_model(model_names, args.model)

    # Run headless
    await run_headless(
        client=client,
        model=model,
        provider=provider_name,
        debug=args.debug,
        journal_mode=args.journal,
        continue_session=args.continue_session,
    )


if __name__ == "__main__":
    asyncio.run(async_main())
