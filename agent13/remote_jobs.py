"""Detached long-running remote jobs (command v2, Phase 3).

Lifts the 120 s command cap for game runs, debuggers, and builds:
``mode="detach"`` launches the script as a detached remote job (survives
the ssh session and the agent), ``mode="poll"`` reads its status, exit
code, and log tail.

Job state lives entirely on the remote, named by job_id (self-contained -
no local registry, survives agent restarts):

    POSIX:    /tmp/agent13-jobs/<job_id>/{script.sh,log,status,pid}
    Windows:  %TEMP%\\agent13-jobs\\<job_id>\\{script.ps1,wrapper.ps1,log,status,pid}

Jobs older than TTL_DAYS are removed on every detach and poll.
"""

import base64
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

from agent13.remote_exec import (
    _validate_host,
    _validate_remote_shell,
    detect_remote_shell,
    run_remote_command,
)

JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
TTL_DAYS = 7
TAIL_BYTES = 65536
_DETACH_TIMEOUT = 30.0
_POLL_TIMEOUT = 30.0

_POSIX_ROOT = "/tmp/agent13-jobs"
_PS_ROOT_PART = "agent13-jobs"
_TASK_PREFIX = "agent13-job-"


def new_job_id() -> str:
    """Stable, single-use job id: UTC timestamp + 3 random bytes."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def validate_job_id(job_id: str) -> str:
    """Validate a job_id format; raise ValueError with guidance otherwise."""
    job_id = (job_id or "").strip()
    if not JOB_ID_RE.match(job_id):
        raise ValueError(
            f"Invalid job_id {job_id!r}: expected format "
            "YYYYMMDDTHHMMSSZ-xxxxxx (from a mode='detach' result)."
        )
    return job_id


def job_dir(shell: str, job_id: str) -> str:
    """Remote job directory for a job (shell-specific path convention)."""
    if shell == "powershell":
        return f"$env:TEMP\\{_PS_ROOT_PART}\\{job_id}"
    return f"{_POSIX_ROOT}/{job_id}"


def _crlf(command: str) -> str:
    """Normalize line endings for Windows (spec item 3)."""
    return command.replace("\r\n", "\n").replace("\n", "\r\n")


def _build_detach_script(shell: str, job_id: str, command: str) -> str:
    """Build the remote script that delivers + launches a detached job."""
    if shell == "powershell":
        b64 = base64.b64encode(_crlf(command).encode("utf-8")).decode("ascii")
        return f"""$ErrorActionPreference = 'Stop'
$root = Join-Path $env:TEMP '{_PS_ROOT_PART}'
$d = Join-Path $root '{job_id}'
Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Where-Object {{ $_.LastWriteTime -lt (Get-Date).AddDays(-{TTL_DAYS}) }} | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ScheduledTask -TaskName '{_TASK_PREFIX}*' -ErrorAction SilentlyContinue | Where-Object {{ $_.LastRunTime -ne [datetime]::MinValue -and $_.LastRunTime -lt (Get-Date).AddDays(-{TTL_DAYS}) }} | Unregister-ScheduledTask -Confirm:$false -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $d | Out-Null
[IO.File]::WriteAllBytes((Join-Path $d 'script.ps1'), [Convert]::FromBase64String('{b64}'))
$wrapper = @"
`$d = Join-Path (Join-Path $env:TEMP '{_PS_ROOT_PART}') '{job_id}'
Set-Content -Path (Join-Path `$d 'pid') -Value "`$PID" -Encoding ascii
`$ErrorActionPreference = 'Continue'
& (Join-Path `$d 'script.ps1') 2>&1 | Out-File -FilePath (Join-Path `$d 'log') -Encoding utf8
`$code = `$LASTEXITCODE
if (`$null -eq `$code) {{ if (`$Error.Count -gt 0) {{ `$code = 1 }} else {{ `$code = 0 }} }}
Set-Content -Path (Join-Path `$d 'status') -Value "`$code" -Encoding ascii
Unregister-ScheduledTask -TaskName '{_TASK_PREFIX}{job_id}' -Confirm:$false -ErrorAction SilentlyContinue
"@
Set-Content -Path (Join-Path $d 'wrapper.ps1') -Value $wrapper -Encoding ascii
# Start-Process children die when the ssh session ends (OpenSSH-for-Windows
# tears down the session's process tree); a scheduled task runs outside it.
$taskName = '{_TASK_PREFIX}{job_id}'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $d 'wrapper.ps1')`""
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Output "TASK $taskName"
"""

    b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"""root={_POSIX_ROOT}
d="$root/{job_id}"
find "$root" -mindepth 1 -maxdepth 1 -type d -mtime +{TTL_DAYS} -exec rm -rf {{}} + 2>/dev/null || true
mkdir -p "$d"
printf %s '{b64}' | base64 -d > "$d/script.sh"
cd "$d"
nohup sh -c 'sh script.sh > log 2>&1; echo $? > status' </dev/null >/dev/null 2>&1 &
echo $! > pid
echo "PID $(cat pid)"
"""


def _build_poll_script(shell: str, job_id: str) -> str:
    """Build the remote script that reports a job's status + log tail.

    Emits machine-readable markers: STATUS=<running|exited|not_found|
    unknown>, optional EXIT_CODE=<n>, then the log tail between
    TAIL_BEGIN / TAIL_END.
    """
    if shell == "powershell":
        return f"""$root = Join-Path $env:TEMP '{_PS_ROOT_PART}'
$d = Join-Path $root '{job_id}'
Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Where-Object {{ $_.LastWriteTime -lt (Get-Date).AddDays(-{TTL_DAYS}) }} | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ScheduledTask -TaskName '{_TASK_PREFIX}*' -ErrorAction SilentlyContinue | Where-Object {{ $_.LastRunTime -ne [datetime]::MinValue -and $_.LastRunTime -lt (Get-Date).AddDays(-{TTL_DAYS}) }} | Unregister-ScheduledTask -Confirm:$false -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $d)) {{ Write-Output 'STATUS=not_found'; exit 0 }}
if (Test-Path -LiteralPath (Join-Path $d 'status')) {{
  Write-Output 'STATUS=exited'
  Write-Output "EXIT_CODE=$((Get-Content -LiteralPath (Join-Path $d 'status') -Raw).Trim())"
}} else {{ Write-Output 'STATUS=running' }}
Write-Output 'TAIL_BEGIN'
$logPath = Join-Path $d 'log'
if (Test-Path -LiteralPath $logPath) {{
  $fs = [IO.File]::Open($logPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {{
    $take = [Math]::Min({TAIL_BYTES}, $fs.Length)
    if ($take -gt 0) {{
      $fs.Seek(-$take, [IO.SeekOrigin]::End) | Out-Null
      $buf = New-Object byte[] $take
      $n = $fs.Read($buf, 0, $take)
      [Text.Encoding]::UTF8.GetString($buf, 0, $n) | Write-Output
    }}
  }} finally {{ $fs.Close() }}
}}
Write-Output 'TAIL_END'
"""

    return f"""root={_POSIX_ROOT}
d="$root/{job_id}"
find "$root" -mindepth 1 -maxdepth 1 -type d -mtime +{TTL_DAYS} -exec rm -rf {{}} + 2>/dev/null || true
if [ ! -d "$d" ]; then echo "STATUS=not_found"; exit 0; fi
if [ -f "$d/status" ]; then
  echo "STATUS=exited"
  echo "EXIT_CODE=$(cat "$d/status" 2>/dev/null)"
elif [ -f "$d/pid" ] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then
  echo "STATUS=running"
else
  echo "STATUS=unknown"
fi
echo "TAIL_BEGIN"
tail -c {TAIL_BYTES} "$d/log" 2>/dev/null
echo "TAIL_END"
"""


def parse_poll_output(stdout: str) -> dict:
    """Parse the marker-based poll script output.

    Returns {status, exit_code, log_tail}.
    """
    status = "unknown"
    exit_code: Optional[int] = None
    m = re.search(r"^STATUS=(\w+)", stdout, re.M)
    if m:
        status = m.group(1)
    m = re.search(r"^EXIT_CODE=(-?\d+)", stdout, re.M)
    if m:
        exit_code = int(m.group(1))
    tail = ""
    m = re.search(r"TAIL_BEGIN\r?\n(.*?)\r?\nTAIL_END", stdout, re.S)
    if m:
        tail = m.group(1)
    return {"status": status, "exit_code": exit_code, "log_tail": tail}


def _paths(shell: str, job_id: str) -> tuple[str, str]:
    d = job_dir(shell, job_id)
    if shell == "powershell":
        return f"{d}\\log", f"{d}\\status"
    return f"{d}/log", f"{d}/status"


async def detach_remote_job(
    host: str, command: str, remote_shell: Optional[str] = None
) -> dict:
    """Launch a script as a detached remote job.

    The job survives the ssh session and the agent. Returns job_id,
    pid (POSIX) / task_name (Windows), log_path, and a poll hint.

    Args:
        host: ssh target (user@host or host)
        command: the job script (plain shell/PowerShell)
        remote_shell: 'posix', 'powershell', or None (auto-detect)

    Returns:
        Dict with success, exit_code, job_id, pid, log_path, status_path,
        remote, remote_shell, hint (or stderr on failure).
    """
    host = _validate_host(host)
    shell = _validate_remote_shell(remote_shell) or await detect_remote_shell(host)
    job_id = new_job_id()
    script = _build_detach_script(shell, job_id, command)
    result = await run_remote_command(
        host=host, command=script, remote_shell=shell, timeout=_DETACH_TIMEOUT
    )
    log_path, status_path = _paths(shell, job_id)
    base = {
        "job_id": job_id,
        "remote": host,
        "remote_shell": shell,
        "log_path": log_path,
        "status_path": status_path,
        "output_encoding": "utf-8",
    }
    if not result["success"]:
        return {
            **base,
            "success": False,
            "exit_code": result["exit_code"],
            "pid": None,
            "stderr": result["stderr"] or f"detach failed (exit {result['exit_code']})",
        }
    m = re.search(r"PID (\d+)", result["stdout"])
    m_task = re.search(r"TASK (\S+)", result["stdout"])
    return {
        **base,
        "success": True,
        "exit_code": 0,
        "pid": int(m.group(1)) if m else None,
        "task_name": m_task.group(1) if m_task else None,
        "hint": (
            f'Poll progress: command(mode="poll", job_id="{job_id}", remote="{host}")'
        ),
    }


async def poll_remote_job(
    host: str,
    job_id: str,
    remote_shell: Optional[str] = None,
    tail_bytes: int = TAIL_BYTES,
) -> dict:
    """Poll a detached remote job: status, exit code, log tail.

    Args:
        host: ssh target (user@host or host)
        job_id: from a detach_remote_job result
        remote_shell: 'posix', 'powershell', or None (auto-detect)
        tail_bytes: max log tail bytes (default 64 KB)

    Returns:
        Dict with success, exit_code (of the poll), status (running /
        exited / not_found / unknown / poll_failed), job_exit_code,
        log_tail, log_path, remote, remote_shell (or stderr on failure).
    """
    host = _validate_host(host)
    job_id = validate_job_id(job_id)
    shell = _validate_remote_shell(remote_shell) or await detect_remote_shell(host)
    script = _build_poll_script(shell, job_id)
    result = await run_remote_command(
        host=host, command=script, remote_shell=shell, timeout=_POLL_TIMEOUT
    )
    log_path, _ = _paths(shell, job_id)
    base = {
        "job_id": job_id,
        "remote": host,
        "remote_shell": shell,
        "log_path": log_path,
        "output_encoding": "utf-8",
    }
    if not result["success"]:
        return {
            **base,
            "success": False,
            "exit_code": result["exit_code"],
            "status": "poll_failed",
            "job_exit_code": None,
            "log_tail": "",
            "stderr": result["stderr"] or f"poll failed (exit {result['exit_code']})",
        }
    parsed = parse_poll_output(result["stdout"])
    out = {
        **base,
        "success": True,
        "exit_code": 0,
        "status": parsed["status"],
        "job_exit_code": parsed["exit_code"],
        "log_tail": parsed["log_tail"],
    }
    if parsed["status"] == "not_found":
        out["note"] = (
            "Job directory not found on the remote - the host rebooted, "
            f"or the job was cleaned up (TTL {TTL_DAYS} days)."
        )
    return out
