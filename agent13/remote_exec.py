"""Remote command execution over ssh (command v2, Phase 2).

The core law (from two independent agent journals + the c209 session):
NEVER let a variable payload pass through a quoting layer. Local bash,
the ssh transport, and the remote login shell each re-parse anything
that sits on a command line. So:

- The remote-parsed part of the argv is a FIXED SAFE CONSTANT
  (``sh -s`` for POSIX, ``powershell -Command -`` for Windows).
- The model's script travels as base64 (the universal wire format —
  [A-Za-z0-9+/=], no character any shell interprets) on ssh's stdin,
  and is decoded on the remote side straight into the tool's stdin.

The harness does all encoding; the model writes plain shell/PowerShell/
Python and never sees base64.

Transport cases:
    ssh host 'b64=$(cat); printf %s "$b64" | base64 -d | sh -s'
        POSIX. Runs in plain sh, not login bash (no .bashrc) —
        deterministic, matches the local tool's stdin-closed behaviour.
        No temp file, no command-line cap.
    ssh host '<powershell decode template>'
        Windows. The outer powershell reads base64 from stdin, decodes
        it (UTF-8), and pipes the script into an inner
        ``powershell -NoProfile -NonInteractive -Command -``. No
        -EncodedCommand, so the ~3KB command-line cap on 5.1 does not
        apply. $LASTEXITCODE from native exes is propagated.

Remote shell auto-detect (one probe per host, session-cached):
    ssh host uname
    non-empty stdout => POSIX; empty/error => Windows (PowerShell).
"""

import asyncio
import base64
from typing import Optional

from agent13.sandbox import _SUBPROCESS_ENCODING, _kill_process_tree

# Session cache: host -> "posix" | "powershell"
_remote_shell_cache: dict[str, str] = {}

# Base64 alphabet (no newlines, no padding beyond '='): no character any
# local or remote shell can misinterpret.
_B64_SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def clear_remote_shell_cache() -> None:
    """Clear the cached remote-shell detection results (tests)."""
    _remote_shell_cache.clear()


def _validate_host(host: str) -> str:
    """Validate an ssh host spec: no shell metacharacters allowed.

    The host is interpolated into the ssh argv (after "ssh", before the
    fixed remote command), so it must be inert. Allowed: user@host, host,
    digits, letters, dots, dashes, underscores. Port flags are NOT
    supported (use an ssh config alias instead).
    """
    host = host.strip()
    if not host:
        raise ValueError("Empty host")
    # user@ is optional
    body = host.split("@", 1)[1] if "@" in host else host
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    bad = sorted(set(body) - allowed)
    if bad:
        raise ValueError(
            f"Invalid host {host!r}: characters {bad} are not allowed. "
            "Use a plain user@host or an ssh config alias (no port flags, "
            "no shell metacharacters)."
        )
    return host


def _validate_remote_shell(shell: Optional[str]) -> Optional[str]:
    """Normalize the remote_shell override.

    Returns None for auto/empty, 'posix' or 'powershell' for explicit.
    Raises ValueError for anything else.
    """
    if shell is None:
        return None
    shell = shell.strip().lower()
    if shell in ("auto", ""):
        return None
    if shell in ("sh", "bash", "posix"):
        return "posix"
    if shell in ("powershell", "ps"):
        return "powershell"
    raise ValueError(
        f"Invalid remote_shell {shell!r}: use 'auto', 'sh' (or 'bash'/'posix'), "
        "or 'powershell'."
    )


async def detect_remote_shell(host: str) -> str:
    """Detect the remote shell: 'posix' or 'powershell'.

    Probe: ``ssh host uname`` — non-empty stdout means a POSIX box;
    empty/error means Windows (uname is not a command there).
    Cached per host for the session.
    """
    if host in _remote_shell_cache:
        return _remote_shell_cache[host]

    result = await _run_ssh_probe(host, "uname", timeout=15)
    if result["exit_code"] == 0 and result["stdout"].strip():
        shell = "posix"
    else:
        shell = "powershell"
    _remote_shell_cache[host] = shell
    return shell


def build_remote_argv(
    host: str, remote_shell: str
) -> list[str]:
    """Build the local ssh argv for a remote command.

    The model's command is base64-encoded and shipped on ssh's stdin; the
    remote side decodes it and runs it through a fixed safe template
    (``sh -s`` for POSIX, ``powershell -Command -`` for Windows). No layer
    ever parses the payload — the base64 alphabet is inert to every shell.

    Args:
        host: ssh target (user@host or host)
        remote_shell: 'posix' or 'powershell' (resolved by caller)

    Returns:
        argv suitable for asyncio.create_subprocess_exec. The caller
        writes the base64 payload to the process's stdin.
    """
    host = _validate_host(host)
    remote_shell = _validate_remote_shell(remote_shell)
    if remote_shell is None:
        raise ValueError("remote_shell must be 'posix' or 'powershell'")

    if remote_shell == "posix":
        # Default hop: base64 on the wire, decoded on the remote side
        # straight into sh's stdin. The fixed template (sh -s) is
        # trivially parseable by any remote login shell — the payload
        # (base64) never appears on the command line, and no temp file
        # is created.
        #
        #   b64=$(cat)  -> read all of stdin (the base64 payload)
        #   base64 -d   -> decode to raw script bytes
        #   sh -s       -> reads the script from stdin (POSIX)
        #
        # Runs in sh, not login bash (no .bashrc) — deterministic.
        # sh -s exits with the script's exit code. The script's own
        # stdin is the pipe (not a tty) — interactive commands will
        # block and hit the timeout, same as the local tool.
        remote_cmd = 'b64=$(cat); printf %s "$b64" | base64 -d | sh -s'
    else:
        # PowerShell hop: decode base64 (UTF-8) on the remote side and
        # pipe it into powershell -Command - (reads script from stdin).
        # No temp file, no -EncodedCommand (so the ~3KB command-line
        # cap on 5.1 does not apply). $LASTEXITCODE from native exes
        # is propagated via the epilogue.
        #
        #   [Console]::In.ReadToEnd() -> read all of stdin (base64)
        #   [Convert]::FromBase64String -> bytes
        #   [Text.Encoding]::UTF8.GetString -> script text
        #   powershell -Command -     -> reads script from stdin
        #
        # The outer powershell is the fixed safe template; the inner
        # powershell runs the model's script. $LASTEXITCODE from the
        # inner process (native exes) is captured and propagated.
        remote_cmd = (
            "$b64 = [Console]::In.ReadToEnd().Trim(); "
            "$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b64)); "
            "$rc = 0; $text | powershell -NoProfile -NonInteractive -Command -; "
            "$rc = $LASTEXITCODE; if ($null -eq $rc) { $rc = 0 }; "
            "exit $rc"
        )

    return ["ssh", host, remote_cmd]


def _validate_b64_payload(b64: str) -> str:
    """Ensure the base64 payload is inert (only [A-Za-z0-9+/=])."""
    bad = sorted(set(b64) - _B64_SAFE)
    if bad:
        # Should be impossible (we just encoded it) — fail loudly.
        raise ValueError(f"Base64 payload contains unsafe characters: {bad}")
    return b64


async def _run_ssh_probe(host: str, remote_cmd: str, timeout: float) -> dict:
    """Run a short ssh command capturing output (used for auto-detect)."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ssh", host, remote_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "ssh not found on this machine",
        }

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _kill_process_tree(process)
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"ssh probe timed out after {timeout}s",
        }

    return {
        "success": process.returncode == 0,
        "exit_code": process.returncode,
        "stdout": stdout_b.decode(_SUBPROCESS_ENCODING, errors="replace"),
        "stderr": stderr_b.decode(_SUBPROCESS_ENCODING, errors="replace"),
    }


async def run_remote_command(
    host: str,
    command: str,
    remote_shell: Optional[str] = None,
    timeout: float = 120.0,
    max_output: int = 100000,
) -> dict:
    """Run a script on a remote host over ssh with zero quoting layers.

    The script is base64-encoded locally and shipped on ssh's stdin. The
    remote side (fixed safe template) decodes it and runs it with ``sh -s``
    (POSIX) or ``powershell -Command -`` (Windows). Exit code, stdout and
    stderr come back verbatim.

    Args:
        host: ssh target (user@host or host; no flags)
        command: the script to run remotely
        remote_shell: 'posix', 'powershell', or None (auto-detect, cached)
        timeout: timeout in seconds
        max_output: max output bytes before truncation

    Returns:
        Dict with success, exit_code, stdout, stderr, truncated, timed_out,
        remote (host), remote_shell (resolved).
    """
    host = _validate_host(host)
    forced = _validate_remote_shell(remote_shell)

    # Resolve the remote shell (probe once per host, session-cached)
    if forced is not None:
        shell = forced
    else:
        shell = await detect_remote_shell(host)

    # Encode the payload: base64 of UTF-8, no newlines.
    b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
    _validate_b64_payload(b64)

    argv = build_remote_argv(host, shell)

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                "ssh not found on this machine. Install OpenSSH client "
                "or run locally (omit the remote parameter)."
            ),
            "truncated": False,
            "timed_out": False,
            "remote": host,
            "remote_shell": shell,
        }

    # Write the base64 payload to stdin, then close it. The remote
    # template reads all of stdin (b64=$(cat) / ReadToEnd), decodes,
    # and runs the script.
    try:
        process.stdin.write(b64.encode("ascii"))
        process.stdin.close()
    except Exception:
        pass  # stdin write failure will surface as a remote error

    # Pump both streams into chunk lists so partial output survives a
    # timeout kill (same pattern as run_sandboxed_async).
    chunks_out: list[bytes] = []
    chunks_err: list[bytes] = []

    async def _pump(stream, chunks: list[bytes]) -> None:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)

    pump_out = asyncio.create_task(_pump(process.stdout, chunks_out))
    pump_err = asyncio.create_task(_pump(process.stderr, chunks_err))

    try:
        timed_out = False
        try:
            await asyncio.wait_for(
                process.wait(), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            await _kill_process_tree(process)

        try:
            await asyncio.wait_for(asyncio.gather(pump_out, pump_err), timeout=5)
        except asyncio.TimeoutError:
            await _kill_process_tree(process)
            pump_out.cancel()
            pump_err.cancel()

        stdout = b"".join(chunks_out).decode(_SUBPROCESS_ENCODING, errors="replace")
        stderr = b"".join(chunks_err).decode(_SUBPROCESS_ENCODING, errors="replace")

        if timed_out:
            return _remote_timeout_result(
                timeout, stdout, stderr, host, shell, max_output
            )

        truncated = False
        if len(stdout) > max_output:
            stdout = (
                stdout[:max_output] + f"\n... [Output truncated at {max_output} bytes]"
            )
            truncated = True
        if len(stderr) > max_output:
            stderr = (
                stderr[:max_output] + f"\n... [Output truncated at {max_output} bytes]"
            )
            truncated = True

        return {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "timed_out": False,
            "remote": host,
            "remote_shell": shell,
        }

    except Exception as e:
        pump_out.cancel()
        pump_err.cancel()
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Error running remote command: {e}",
            "truncated": False,
            "timed_out": False,
            "remote": host,
            "remote_shell": shell,
        }


def _remote_timeout_result(
    timeout: float,
    stdout: str,
    stderr: str,
    host: str,
    shell: str,
    max_output: int,
) -> dict:
    """Build the result dict for a timed-out remote command."""
    truncated = False
    if len(stdout) > max_output:
        stdout = stdout[:max_output] + f"\n... [Output truncated at {max_output} bytes]"
        truncated = True
    if len(stderr) > max_output:
        stderr = stderr[:max_output] + f"\n... [Output truncated at {max_output} bytes]"
        truncated = True

    message = (
        f"Remote command timed out after {timeout} seconds on {host}. "
        "If the script is interactive (reads stdin, prompts), it was "
        "blocked - the script's stdin is the base64 payload, not a tty."
    )
    stderr = (stderr + "\n" if stderr else "") + message

    return {
        "success": False,
        "exit_code": -1,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
        "timed_out": True,
        "remote": host,
        "remote_shell": shell,
    }
