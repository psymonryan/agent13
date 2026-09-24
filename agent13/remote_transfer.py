"""Binary-exact file transfer over sftp (command v2, Phase 3).

Kills the recurring base64-hand-transfer corruption class: bytes travel
over the sftp protocol (binary-safe, no encoding layer), and both ends
compute SHA-256 so any mismatch or truncation fails loudly instead of
silently "succeeding".

Flow (put):
    1. SHA-256 of the local file (before send)
    2. Ensure the remote parent directory exists
    3. ``sftp -b -`` (batch mode, single command on stdin)
    4. SHA-256 of the remote file (after write)
    5. Compare; return both hashes; mismatch => success False

Flow (get) is the mirror: remote hash first (also proves the file
exists), transfer, then local hash.
"""

import asyncio
import hashlib
import os
import re
import shlex
from typing import Optional

from agent13.remote_exec import (
    _validate_host,
    _validate_remote_shell,
    detect_remote_shell,
    run_remote_command,
)
from agent13.sandbox import _SUBPROCESS_ENCODING, _kill_process_tree

_SFTP_TIMEOUT = 300.0  # 300 s: ample for <100 MB over a slow link
_PROBE_TIMEOUT = 30.0
_HASH_CHUNK = 1024 * 1024


def _sha256_file(path: str) -> str:
    """Stream SHA-256 of a local file (hex digest)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sftp_quote(path: str) -> str:
    """Quote a path for an sftp batch command (double-quote, escape \\ and ")."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ps_quote(s: str) -> str:
    """Quote a string as a PowerShell single-quoted literal."""
    return "'" + s.replace("'", "''") + "'"


def _parent_dir(path: str, shell: str) -> str:
    """Parent directory of a remote path ('' if the path has no dir part)."""
    if shell == "powershell":
        for sep in ("\\", "/"):
            idx = path.rfind(sep)
            if idx > 0:
                return path[:idx]
        return ""
    idx = max(path.rfind("/"), path.rfind("\\"))
    if idx > 0:
        return path[:idx]
    return ""


def _normalize_remote_path(path: str, shell: str) -> str:
    """Canonicalize a remote path to the target OS's native form.

    Accepts every reasonable spelling so callers never have to know the
    remote's dialect:
      posix:      /abs/path, rel/path          (unchanged)
      powershell: C:\\x, C:/x, /C:/x, /C:\\x   (-> C:\\x)
                  rel/path                     (unchanged; both consumers
                                                resolve it against home)
    """
    if shell != "powershell":
        return path
    m = re.match(r"^/?([A-Za-z]):[/\\](.*)$", path)
    if m:
        return m.group(1) + ":\\" + m.group(2).replace("/", "\\")
    return path


def _sftp_wire_path(path: str, shell: str) -> str:
    """Path in the form the sftp protocol expects on the remote.

    Windows sftp-server has no drive-letter concept: ``C:\\x`` is treated
    as relative (home-joined), the absolute form is ``/C:/x``. POSIX and
    relative paths pass through unchanged.
    """
    if shell != "powershell":
        return path
    m = re.match(r"^([A-Za-z]):[/\\](.*)$", path)
    if m:
        return f"/{m.group(1)}:/" + m.group(2).replace("\\", "/")
    return path


def _transfer_result(
    op: str, host: str, local_path: str, remote_path: str, shell: str
) -> dict:
    """Base result dict for a transfer (filled in by the caller)."""
    return {
        "success": False,
        "exit_code": -1,
        "operation": op,
        "local_path": local_path,
        "remote_path": remote_path,
        "remote": host,
        "remote_shell": shell,
        "bytes": 0,
        "local_sha256": None,
        "remote_sha256": None,
        "verified": False,
        "output_encoding": "utf-8",
        "stderr": "",
    }


async def _run_sftp(
    host: str, batch_command: str, timeout: float = _SFTP_TIMEOUT
) -> dict:
    """Run a single sftp batch command (sent on stdin).

    Batch mode: no prompts, non-zero exit on any failure. Returns
    success / exit_code / stdout / stderr.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "sftp",
            "-b",
            "-",
            "-o",
            "BatchMode=yes",
            host,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                "sftp not found on this machine. Install the OpenSSH client "
                "(sftp ships with it) and retry."
            ),
        }

    try:
        process.stdin.write((batch_command + "\n").encode("utf-8"))
        process.stdin.close()
    except Exception:
        pass  # surfaces as a non-zero sftp exit

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
            "stderr": f"sftp timed out after {timeout}s",
        }

    stderr = stderr_b.decode(_SUBPROCESS_ENCODING, errors="replace")
    if process.returncode != 0 and "subsystem request failed" in stderr:
        stderr += (
            "\nThe remote sshd does not offer the sftp subsystem. Enable it "
            "(sshd_config: Subsystem sftp ...) or copy the file in base64 "
            "chunks via command(remote=...) instead."
        )
    return {
        "success": process.returncode == 0,
        "exit_code": process.returncode,
        "stdout": stdout_b.decode(_SUBPROCESS_ENCODING, errors="replace"),
        "stderr": stderr,
    }


async def _ensure_remote_dir(host: str, shell: str, dir_path: str) -> dict:
    """Create the remote parent directory if missing (idempotent)."""
    if not dir_path:
        return {"success": True}
    if shell == "posix":
        script = f"mkdir -p -- {shlex.quote(dir_path)}"
    else:
        script = (
            f"New-Item -ItemType Directory -Force -Path {_ps_quote(dir_path)} "
            "| Out-Null"
        )
    return await run_remote_command(
        host=host, command=script, remote_shell=shell, timeout=_PROBE_TIMEOUT
    )


async def _remote_sha256(host: str, shell: str, path: str) -> dict:
    """Compute SHA-256 of a remote file. Returns {success, sha256, stderr}."""
    if shell == "posix":
        script = f"sha256sum -- {shlex.quote(path)}"
    else:
        script = (
            f"(Get-FileHash -LiteralPath {_ps_quote(path)} -Algorithm SHA256)"
            ".Hash.ToLower()"
        )
    result = await run_remote_command(
        host=host, command=script, remote_shell=shell, timeout=_PROBE_TIMEOUT
    )
    if not result["success"]:
        return {"success": False, "sha256": None, "stderr": result["stderr"]}
    parts = result["stdout"].strip().split()
    if not parts or len(parts[0]) != 64:
        return {
            "success": False,
            "sha256": None,
            "stderr": f"Unexpected sha256 output from remote: {result['stdout']!r}",
        }
    return {"success": True, "sha256": parts[0], "stderr": ""}


async def remote_put(
    host: str,
    local_path: str,
    remote_path: str,
    remote_shell: Optional[str] = None,
) -> dict:
    """Copy a local file to a remote host. Binary-exact, SHA-256 verified.

    Args:
        host: ssh target (user@host or host)
        local_path: file on THIS machine
        remote_path: destination path on the remote
        remote_shell: 'posix', 'powershell', or None (auto-detect)

    Returns:
        Dict with success, exit_code, operation, local_path, remote_path,
        remote, bytes, local_sha256, remote_sha256, verified, stderr.
    """
    host = _validate_host(host)
    shell = _validate_remote_shell(remote_shell) or await detect_remote_shell(host)
    remote_path = _normalize_remote_path(remote_path, shell)
    result = _transfer_result("put", host, local_path, remote_path, shell)

    if not os.path.isfile(local_path):
        result["stderr"] = (
            f"Local file not found: {local_path} (must be an existing file)"
        )
        return result

    result["bytes"] = os.path.getsize(local_path)
    local_sha = _sha256_file(local_path)
    result["local_sha256"] = local_sha

    dir_result = await _ensure_remote_dir(host, shell, _parent_dir(remote_path, shell))
    if not dir_result["success"]:
        result["stderr"] = (
            f"Failed to create remote directory for {remote_path}: "
            f"{dir_result['stderr']}"
        )
        return result

    sftp = await _run_sftp(
        host,
        f"put {_sftp_quote(local_path)} "
        f"{_sftp_quote(_sftp_wire_path(remote_path, shell))}",
    )
    if not sftp["success"]:
        result["exit_code"] = sftp["exit_code"]
        result["stderr"] = (
            sftp["stderr"] or f"sftp put failed (exit {sftp['exit_code']})"
        )
        return result

    remote_hash = await _remote_sha256(host, shell, remote_path)
    if not remote_hash["success"]:
        result["exit_code"] = 1
        result["stderr"] = (
            "Transfer completed but remote verification failed: "
            f"{remote_hash['stderr']}"
        )
        return result
    result["remote_sha256"] = remote_hash["sha256"]
    if remote_hash["sha256"] != local_sha:
        result["exit_code"] = 1
        result["stderr"] = (
            f"SHA-256 mismatch after transfer: local {local_sha} != remote "
            f"{remote_hash['sha256']}. The file may be truncated or "
            "corrupted - do not trust the remote copy."
        )
        return result

    result["success"] = True
    result["exit_code"] = 0
    result["verified"] = True
    return result


async def remote_get(
    host: str,
    remote_path: str,
    local_path: str,
    remote_shell: Optional[str] = None,
) -> dict:
    """Copy a file from a remote host to local. Binary-exact, SHA-256 verified.

    Args:
        host: ssh target (user@host or host)
        remote_path: file on the remote
        local_path: destination path on THIS machine
        remote_shell: 'posix', 'powershell', or None (auto-detect)

    Returns:
        Dict with success, exit_code, operation, local_path, remote_path,
        remote, bytes, local_sha256, remote_sha256, verified, stderr.
    """
    host = _validate_host(host)
    shell = _validate_remote_shell(remote_shell) or await detect_remote_shell(host)
    remote_path = _normalize_remote_path(remote_path, shell)
    result = _transfer_result("get", host, local_path, remote_path, shell)

    remote_hash = await _remote_sha256(host, shell, remote_path)
    if not remote_hash["success"]:
        result["stderr"] = (
            f"Remote file not readable: {remote_path} "
            f"(missing, or hash probe failed) - {remote_hash['stderr']}"
        )
        return result
    result["remote_sha256"] = remote_hash["sha256"]

    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    sftp = await _run_sftp(
        host,
        f"get {_sftp_quote(_sftp_wire_path(remote_path, shell))} "
        f"{_sftp_quote(local_path)}",
    )
    if not sftp["success"]:
        result["exit_code"] = sftp["exit_code"]
        result["stderr"] = (
            sftp["stderr"] or f"sftp get failed (exit {sftp['exit_code']})"
        )
        return result

    if not os.path.isfile(local_path):
        result["exit_code"] = 1
        result["stderr"] = (
            "sftp reported success but the local file is missing - "
            "treat the transfer as failed."
        )
        return result

    result["bytes"] = os.path.getsize(local_path)
    local_sha = _sha256_file(local_path)
    result["local_sha256"] = local_sha
    if local_sha != remote_hash["sha256"]:
        result["exit_code"] = 1
        result["stderr"] = (
            f"SHA-256 mismatch after transfer: remote {remote_hash['sha256']} "
            f"!= local {local_sha}. The file may be truncated or corrupted - "
            "do not trust the local copy."
        )
        return result

    result["success"] = True
    result["exit_code"] = 0
    result["verified"] = True
    return result
