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
from agent13.remote_exec import run_remote_command


@tool(is_async=True)
async def command(
    command: str,
    timeout: Optional[float] = None,
    remote: Optional[str] = None,
    remote_shell: Optional[str] = None,
) -> dict:
    """Run a command. Sandboxed by default (macOS Seatbelt; unrestricted on other platforms). User controls mode via /sandbox command.

    To run a command on a REMOTE machine, pass remote="user@host" — do NOT
    write `ssh host "..."` yourself. The harness ships your script over ssh
    with zero quoting layers: $_, $?, $(...), backticks, quotes all arrive
    intact. Just write the script as if running locally on the target OS.
    Remote shell auto-detected (posix/powershell), cached per host.
    Override with remote_shell="posix" or remote_shell="powershell".

    On Windows, commands run in PowerShell (no intermediate shell — your text
    arrives verbatim; no ^-escaping, no cmd.exe quirks). Use PowerShell syntax
    on Windows: Get-ChildItem, Where-Object, Select-String; $env:PATH;
    `;` to chain. PowerShell 5.1 rules: no `&&`/`||`/ternary; prefer
    `;` chaining; add `-Encoding utf8` to Out-File/Set-Content (5.1 defaults
    to UTF-16). On macOS/Linux commands run in /bin/sh.

    Args:
        command: The command to run (plain script, no ssh wrapping)
        timeout: Timeout in seconds (default 120, max 600)
        remote: ssh target (user@host or host). Set this instead of writing
                ssh yourself. The script runs remotely with zero quoting layers.
        remote_shell: Override auto-detect: 'posix' or 'powershell'.

    Returns: Dict with success, exit_code, stdout, stderr, truncated, timed_out, sandbox_mode
    """
    # Validate and clamp timeout
    if timeout is None:
        timeout = 120.0
    else:
        # Convert to float in case LLM passes string
        try:
            timeout = float(timeout)
        except (ValueError, TypeError):
            timeout = 120.0
    timeout = max(0.1, min(timeout, 600.0))  # Clamp to 0.1-600 seconds

    # Remote execution path
    if remote:
        return await run_remote_command(
            host=remote,
            command=command,
            remote_shell=remote_shell,
            timeout=timeout,
            max_output=100000,
        )

    # Local execution path
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
