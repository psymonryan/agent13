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
from agent13.remote_jobs import detach_remote_job, poll_remote_job


def _arg_error(message: str) -> dict:
    """Result dict for a missing/invalid argument (teaches the fix)."""
    return {
        "success": False,
        "exit_code": -1,
        "stdout": "",
        "stderr": message,
        "truncated": False,
        "timed_out": False,
        "status": "error",
        "output_encoding": "utf-8",
    }


@tool(is_async=True)
async def command(
    command: Optional[str] = None,
    timeout: Optional[float] = None,
    remote: Optional[str] = None,
    remote_shell: Optional[str] = None,
    mode: Optional[str] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Run a command. Sandboxed by default (macOS Seatbelt; unrestricted on other platforms). User controls mode via /sandbox command.

    To run a command on a REMOTE machine, pass remote="user@host" — do NOT
    write `ssh host "..."` yourself. The harness ships your script over ssh
    with zero quoting layers: $_, $?, $(...), backticks, quotes all arrive
    intact. Just write the script as if running locally on the target OS.
    Remote shell auto-detected (posix/powershell), cached per host.
    Override with remote_shell="posix" or remote_shell="powershell".

    Long-running remote jobs (remote only) — for anything past the 120 s
    cap (game runs, debuggers, builds):
      mode="detach"  -> runs the script as a detached remote job that
                        survives this session; returns job_id + log path.
      mode="poll"    -> with job_id (+ remote), returns the log tail
                        (last 64 KB), live status, and exit code.
    Jobs auto-clean after 7 days.

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
        mode: 'run' (default), 'detach' (launch a long-running remote job),
              or 'poll' (check a job's status/log tail).
        job_id: Job id from a mode='detach' result; required with mode='poll'.

    Returns: Dict with success, exit_code, stdout, stderr, truncated, timed_out,
             status, output_encoding, sandbox_mode. Detach adds job_id, pid,
             log_path; poll adds status, job_exit_code, log_tail.
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

    # Mode dispatch (remote jobs)
    mode_norm = (mode or "run").strip().lower()
    if mode_norm not in ("run", "detach", "poll"):
        return _arg_error(
            f"Invalid mode {mode!r}: use 'run' (default), 'detach', or "
            "'poll' (detach/poll require remote)."
        )
    if mode_norm == "poll":
        if not remote:
            return _arg_error("mode='poll' requires remote='user@host'")
        if not job_id:
            return _arg_error(
                "mode='poll' requires job_id (from a mode='detach' result)"
            )
        try:
            return await poll_remote_job(
                host=remote, job_id=job_id, remote_shell=remote_shell
            )
        except ValueError as e:
            return _arg_error(str(e))
    if mode_norm == "detach":
        if not remote:
            return _arg_error("mode='detach' requires remote='user@host'")
        if not command:
            return _arg_error(
                "mode='detach' requires command (the job script to run remotely)"
            )
        return await detach_remote_job(
            host=remote, command=command, remote_shell=remote_shell
        )
    if not command:
        return _arg_error("Missing required argument: command")

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
    sandbox_mode = get_current_sandbox_mode()

    # Run the command asynchronously
    result = await run_sandboxed_async(
        command=command,
        mode=sandbox_mode,
        timeout=timeout,
        max_output=100000,  # 100KB
        project_dir=Path.cwd(),
    )

    return result
