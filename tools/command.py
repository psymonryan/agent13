"""Command tool with sandbox support for secure command execution."""

from pathlib import Path
from typing import Optional

from tools import tool
from tools.security import (
    get_current_sandbox_mode,
    set_session_sandbox_mode,
    get_session_sandbox_mode,
)

__all__ = ["command", "set_session_sandbox_mode", "get_session_sandbox_mode"]
from agent13.sandbox import run_sandboxed_async


@tool(is_async=True)
async def command(command: str, timeout: Optional[float] = None) -> dict:
    """Run a command. Sandboxed by default (macOS Seatbelt; unrestricted on other platforms). User controls mode via /sandbox command.

    Args:
        command: The command to run
        timeout: Timeout in seconds (default 30, max 600)

    Returns: Dict with success, exit_code, stdout, stderr, truncated, timed_out, sandbox_mode
    """
    # Validate and clamp timeout
    if timeout is None:
        timeout = 30.0
    else:
        # Convert to float in case LLM passes string
        try:
            timeout = float(timeout)
        except (ValueError, TypeError):
            timeout = 30.0
    timeout = max(0.1, min(timeout, 600.0))  # Clamp to 0.1-600 seconds

    # Get the current sandbox mode (user-controlled only)
    mode = get_current_sandbox_mode()

    # Run the command asynchronously
    result = await run_sandboxed_async(
        command=command,
        mode=mode,
        timeout=timeout,
        max_output=100000,  # 100KB
        project_dir=Path.cwd(),
    )

    return result
