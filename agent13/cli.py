"""Unified entry point for agent13.

Usage:
  agent13 studio                    # Interactive TUI mode
  agent13 openrouter --model devstral   # TUI with model selection
  agent13 test -p "Hello"          # Batch mode
  agent13 --list-providers         # List providers
  agent13 test --model             # List models for provider
"""

import json
import sys

import argparse
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ui.display import RichDisplay

from agent13 import (
    Agent,
    PromptManager,
    get_filtered_tools,
    execute_tool,
    resolve_provider_arg,
    create_client,
    init_debug,
    log_session_end,
    get_config,
    run_batch,
    skill_manager_ctx,
    __version__,
)
from agent13.persistence import save_context, get_auto_save_path
from agent13.models import fetch_models, select_model, print_model_list
from agent13.sandbox import parse_sandbox_mode
from agent13.skills import SkillManager, ensure_default_skills
from agent13.prompts import get_skills_section
from tools.security import set_session_sandbox_mode
from agent13.config_paths import get_global_env_file

# Load environment variables from ~/.env
load_dotenv(get_global_env_file())


def print_provider_list():
    """Print available providers from config."""
    config = get_config()
    if not config.providers:
        print("No providers configured in ~/.agent13/config.toml")
        return
    print("\nAvailable providers:")
    for provider in config.providers:
        key_status = (
            f" (key: {provider.api_key_env_var})"
            if provider.api_key_env_var
            else " (no key required)"
        )
        print(f"  {provider.name}{key_status}")
        print(f"    {provider.api_base}")
    print()


async def run_batch_with_display(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    pretty: bool = True,
    debug: bool = False,
    tool_response_format: str = "raw",
    prompt_manager: PromptManager = None,
    system_prompt: str = None,
    send_reasoning: bool = False,
    remove_reasoning: bool = False,
    devel_mode: bool = False,
):
    """Run batch mode with Rich display."""
    # Initialize prompt manager
    prompt_manager = prompt_manager or PromptManager()

    # Use provided system_prompt or get from prompt manager
    if system_prompt is None:
        system_prompt = prompt_manager.get_prompt()

    # Convert tool_response_format string to response_format dict
    response_format = (
        {"type": "json_object"} if tool_response_format == "json" else None
    )

    # Create agent
    from agent13.config import get_config as _get_config

    _config = _get_config()
    agent = Agent(
        client=client,
        model=model,
        system_prompt=system_prompt,
        tools=get_filtered_tools(
            devel=devel_mode,
            enabled_tools=_config.enabled_tools or None,
            disabled_tools=_config.disabled_tools or None,
        ),
        execute_tool=execute_tool,
        response_format=response_format,
        send_reasoning=send_reasoning,
        remove_reasoning=remove_reasoning,
        devel_mode=devel_mode,
    )

    # Use RichDisplay for pretty mode, simple print for non-pretty
    if pretty:
        display = RichDisplay(pretty=True, debug=debug)

        async def on_token(text: str):
            if not display._in_content:
                display.start_response()
                await asyncio.sleep(0.05)  # Allow spinner cleanup
            display.add_token(text)

        async def on_reasoning(text: str):
            display.add_reasoning(text)

        async def on_tool_call(name: str, arguments: dict):
            display.show_tool_call(name, arguments)

        async def on_tool_result(result: str):
            display.show_tool_result(result)

        async def on_error(message: str):
            display.show_error(message)

        async def on_complete():
            display.complete_response()

    else:
        # Simple mode - just print raw text
        response_started = False

        async def on_token(text: str):
            nonlocal response_started
            if not response_started:
                print()  # Start response on new line
                response_started = True
            print(text, end="", flush=True)

        async def on_reasoning(text: str):
            print(text, end="", flush=True)

        async def on_tool_call(name: str, arguments: dict):
            args_str = json.dumps(arguments)
            print(f"\n[Tool: {name}({args_str})]")

        async def on_tool_result(result: str):
            display_result = result[:200] + "..." if len(result) > 200 else result
            print(f"[Result: {display_result}]")

        async def on_error(message: str):
            print(f"\nError: {message}")

        async def on_complete():
            print()  # End response with newline

    # Run batch
    await run_batch(
        agent,
        prompt,
        on_token=on_token,
        on_reasoning=on_reasoning,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_error=on_error,
        on_complete=on_complete,
    )


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults in help, preserve epilog formatting."""

    pass


async def async_main():
    """Async main entry point."""
    parser = argparse.ArgumentParser(
        description="agent13 - AI agent with TUI and batch modes",
        formatter_class=_HelpFormatter,
        epilog="""Examples:
  agent13 studio                      # Interactive TUI mode
  agent13 openrouter --model devstral # TUI with model selection
  agent13 test -p "What is 5^2?"      # Batch mode
  agent13 --list-providers            # List providers
  agent13 test --model                # List models for provider

Provider names are read from ~/.agent13/config.toml
""",
    )
    parser.add_argument(
        "provider", nargs="?", help="Provider name from config or OpenAI-compatible URL"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers from config and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Run in batch mode with this prompt (exits after processing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Model to select: number (1, 2, ...) or name. With no value, lists models",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to use (name from prompt manager)",
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        help="Sandbox mode for bash tool (permissive-open, permissive-closed, etc.)",
    )
    parser.add_argument(
        "--pretty",
        choices=["on", "off"],
        default="on",
        help="Enable/disable markdown rendering",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--tool-response",
        choices=["raw", "json"],
        default="raw",
        help="Tool response format: 'raw' or 'json'",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Connect to MCP servers on startup (TUI mode only)",
    )
    parser.add_argument(
        "--skills",
        action="store_true",
        help="Include discovered skills in the system prompt",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Enable journal mode (context compaction via reflection)",
    )
    parser.add_argument(
        "--send-reasoning",
        action="store_true",
        help="Include reasoning_content in message history",
    )
    parser.add_argument(
        "--remove-reasoning",
        action="store_true",
        help="Strip reasoning tokens between turns",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue from last auto-saved session",
    )
    parser.add_argument(
        "--devel",
        action="store_true",
        help="Enable devel mode (show devel-group tools like TUI viewer to the AI)",
    )
    parser.add_argument(
        "--spinner",
        choices=["fast", "slow", "off"],
        default="fast",
        help="Spinner style/speed: fast (braille, 100ms), slow (classic, 250ms), or off",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Check for updates and install if available, then exit",
    )
    parser.add_argument(
        "--clipboard",
        choices=["osc52", "system"],
        default="osc52",
        help="Clipboard method: osc52 (terminal escape sequence) or system (OS clipboard command)",
    )

    args = parser.parse_args()
    # Track whether --clipboard was explicitly passed (vs default)
    args._clipboard_explicit = "--clipboard" in sys.argv

    # Ensure default skills are available for new users
    ensure_default_skills()

    # Handle --upgrade flag (check + apply, then exit)
    if args.upgrade:
        from agent13.updater import perform_update

        success, message = perform_update()
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}", file=sys.stderr)
        sys.exit(0 if success else 1)

    # Check for updates (throttled, respects config)
    from agent13.updater import check_for_update, format_update_notice, perform_update

    cfg = get_config()
    if cfg.update_check_enabled:
        update_info = check_for_update(cfg.update_check_interval_hours)
        if update_info:
            notice = format_update_notice(update_info)
            print(f"\n{notice}\n", file=sys.stderr)

            # Interactive prompt: only in TUI mode on a real terminal
            is_batch = args.prompt is not None
            if not is_batch and sys.stdin.isatty():
                try:
                    print(file=sys.stderr)
                    choice = input("  Apply update now? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = ""
                if choice in ("y", "yes"):
                    success, message = perform_update()
                    if success:
                        print(f"✓ {message}")
                    else:
                        print(f"✗ {message}", file=sys.stderr)
                    sys.exit(0 if success else 1)
                # Any other input → continue to TUI as normal

    # Initialize debug logging if --debug flag is set
    if args.debug:
        init_debug()

    # Handle --list-providers flag (doesn't require provider argument)
    if args.list_providers:
        print_provider_list()
        sys.exit(0)

    # Provider is required for all other operations
    if not args.provider:
        parser.error(
            "provider argument is required (use --list-providers to see available providers)"
        )

    # Resolve provider
    try:
        base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
            args.provider
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine provider name for display
    provider_name = (
        ""
        if args.provider.startswith("http://") or args.provider.startswith("https://")
        else args.provider
    )

    # Initialize client
    client = create_client(
        base_url, api_key, read_timeout=read_timeout, connect_timeout=connect_timeout
    )

    # Fetch models
    try:
        model_names = await fetch_models(client)
    except RuntimeError as e:
        err_msg = str(e)
        is_connection_error = any(
            s in err_msg.lower()
            for s in [
                "connection error",
                "connection refused",
                "could not resolve",
                "timed out",
                "name or service not known",
            ]
        )

        if is_connection_error:
            # Provider is unreachable - no point continuing
            print(f"Error: Provider is unreachable: {e}", file=sys.stderr)
            sys.exit(1)
        elif args.model == "":
            # --model with no value means "list models" - can't do that without fetching
            print(f"Error: Could not fetch models: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            # Non-connection error (e.g. /models endpoint broken) - try proceeding with specified model
            print(f"Warning: Could not fetch models: {e}", file=sys.stderr)
            model_names = []

    # Handle --model with no value: list models and exit
    if args.model == "":
        print_model_list(model_names)
        sys.exit(0)

    # Select model - if model list unavailable and user specified a model, use it directly
    if not model_names and args.model:
        model = args.model
        print(
            f"Using model '{model}' (model list unavailable, using specified name directly)",
            file=sys.stderr,
        )
    else:
        model = await select_model(model_names, args.model)

    # Initialize prompt manager
    prompt_manager = PromptManager()
    if args.system_prompt:
        if not prompt_manager.set_active(args.system_prompt):
            print(f"Error: Prompt '{args.system_prompt}' not found", file=sys.stderr)
            print(
                f"Available prompts: {', '.join(prompt_manager.prompts)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Set sandbox mode if specified
    if args.sandbox:
        try:
            sandbox_mode = parse_sandbox_mode(args.sandbox)
            set_session_sandbox_mode(sandbox_mode)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Create skill manager (needed for both batch and TUI if skills enabled)
    skill_manager = SkillManager(lambda: get_config())

    # Build system prompt (optionally with skills)
    system_prompt = prompt_manager.get_prompt()
    include_skills = args.skills or get_config().include_skills
    if include_skills and skill_manager.skills:
        skills_section = get_skills_section(skill_manager.skills)
        if skills_section:
            system_prompt = f"{system_prompt}\n\n{skills_section}"

    # Batch mode
    if args.prompt:
        # Set skill manager context for skill tool
        if include_skills and skill_manager.skills:
            skill_manager_ctx.set(skill_manager)
        await run_batch_with_display(
            client=client,
            model=model,
            prompt=args.prompt,
            pretty=args.pretty == "on",
            debug=args.debug,
            tool_response_format=args.tool_response,
            prompt_manager=prompt_manager,
            system_prompt=system_prompt,
            send_reasoning=args.send_reasoning,
            remove_reasoning=args.remove_reasoning,
            devel_mode=args.devel,
        )
        log_session_end()
        sys.exit(0)

    # TUI mode - import here to avoid loading Textual for batch mode
    from ui.tui import AgentTUI

    # Return app for TUI
    return AgentTUI(
        client=client,
        model=model,
        model_names=model_names,
        provider=provider_name,
        pretty=args.pretty == "on",
        debug=args.debug,
        tool_response_format=args.tool_response,
        prompt_manager=prompt_manager,
        connect_mcp=args.mcp,
        skill_manager=skill_manager,
        system_prompt=system_prompt,
        journal_mode=args.journal,
        send_reasoning=args.send_reasoning,
        remove_reasoning=args.remove_reasoning,
        continue_session=args.continue_session,
        devel_mode=args.devel,
        spinner_speed=args.spinner,
        clipboard_method=args.clipboard if args._clipboard_explicit else cfg.clipboard_method,
    )


def main():
    """Main entry point."""
    # Check if running in batch mode (has -p or --prompt)
    is_batch = "-p" in sys.argv or "--prompt" in sys.argv

    if is_batch:
        # Batch mode - run async directly
        try:
            asyncio.run(async_main())
        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(0)
    else:
        # TUI mode - run async setup, then run app
        app = None
        try:
            app = asyncio.run(async_main())
            app.run()
        except KeyboardInterrupt:
            log_session_end()
            print("\nExiting on keyboard interrupt", flush=True)
        except EOFError:
            print("\nGoodbye!")
        finally:
            # Auto-save on exit if there are messages
            if app is not None and hasattr(app, "agent") and app.agent.messages:
                auto_save_path = get_auto_save_path()
                try:
                    save_context(app.agent, auto_save_path)
                    print(f"\nSession saved to {auto_save_path}")
                except Exception as e:
                    print(f"\nWarning: Could not auto-save session: {e}")
