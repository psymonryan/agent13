"""Self-update checker for agent13.

Checks GitHub releases for newer versions, throttled to once per day.
Can perform in-place upgrade via uv tool and prompt user to restart.

On Windows, the Scripts directory (containing both agent13.exe and
python.exe) is locked by the OS and cannot be deleted or replaced.
However, Windows *does* allow renaming locked files and directories.
We exploit this by renaming Scripts/ to a temp location before running
``uv tool install --force``, which can then create a fresh Scripts/
directory unimpeded.  Any leftover temp directory is cleaned up on
next launch.

The rename is not guaranteed to succeed: the directory is locked for
renaming if any process holds it as its current working directory
(WinError 32) or has a file inside it open without FILE_SHARE_DELETE
(WinError 5, e.g. antivirus or a file explorer window).  When the
rename fails we fall back to a *detached helper* process that runs
``uv tool install --force`` after the current process has exited, at
which point the locks are released.  The helper records its outcome in
a result file that the next launch reports on.

Config keys (in ~/.agent13/config.toml):
    [updates]
    check_enabled = true          # Set to false to disable update checks
    check_interval_hours = 24    # Minimum hours between checks
"""

import json
import logging
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

import httpx

from agent13 import __version__
from agent13.config_paths import ensure_config_dir, get_config_dir

logger = logging.getLogger(__name__)

# GitHub repo for releases
GITHUB_OWNER = "psymonryan"
GITHUB_REPO = "agent13"
GITHUB_RELEASES_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# Throttle state file
_LAST_CHECK_FILE = get_config_dir() / "last_update_check.json"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '0.1.8' into a comparable tuple."""
    # Strip leading 'v' if present
    version_str = version_str.lstrip("v")
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _is_newer(remote_version: str, local_version: str) -> bool:
    """Return True if remote_version is newer than local_version."""
    remote = _parse_version(remote_version)
    local = _parse_version(local_version)
    if not remote or not local:
        # Can't compare, assume not newer
        return False
    return remote > local


def _read_last_check() -> Optional[datetime]:
    """Read the timestamp of the last update check from the state file."""
    if not _LAST_CHECK_FILE.exists():
        return None
    try:
        data = json.loads(_LAST_CHECK_FILE.read_text())
        ts = data.get("last_check")
        if ts:
            return datetime.fromisoformat(ts)
    except (json.JSONDecodeError, ValueError, KeyError, OSError):
        pass
    return None


def _write_last_check(now: datetime) -> None:
    """Write the current check timestamp to the state file."""
    try:
        ensure_config_dir()
        _LAST_CHECK_FILE.write_text(
            json.dumps({"last_check": now.isoformat()})
        )
    except OSError as e:
        logger.warning("Failed to write update check timestamp: %s", e)


def _should_check(interval_hours: float) -> bool:
    """Return True if enough time has passed since the last check."""
    last = _read_last_check()
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    # Make last check timezone-aware if it isn't
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    elapsed_hours = (now - last).total_seconds() / 3600
    return elapsed_hours >= interval_hours


def _find_wheel_asset(assets: list[dict]) -> Optional[str]:
    """Find the .whl asset URL from a GitHub release assets list."""
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith("-py3-none-any.whl"):
            return asset.get("browser_download_url")
    return None


def fetch_latest_release() -> Optional[dict]:
    """Fetch the latest release info from GitHub.

    Returns dict with 'tag_name', 'html_url', and 'wheel_url' keys,
    or None on failure.
    """
    try:
        resp = httpx.get(
            GITHUB_RELEASES_URL,
            headers={"Accept": "application/vnd.github+json"},
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            data = resp.json()
            tag = data.get("tag_name", "")
            wheel_url = _find_wheel_asset(data.get("assets", []))
            return {
                "tag_name": tag,
                "html_url": data.get("html_url", ""),
                "wheel_url": wheel_url or "",
            }
        # 404 = no releases yet, rate-limited, etc -- not an error worth reporting
        logger.debug("GitHub releases returned status %d", resp.status_code)
    except (httpx.HTTPError, OSError) as e:
        logger.debug("Failed to check for updates: %s", e)
    return None


def _build_manual_command(wheel_url: str) -> str:
    """Build the manual uv tool install command from a wheel URL."""
    return f"uv tool install --force {wheel_url}"


def _find_scripts_dir() -> Optional[str]:
    """Find the Scripts directory containing the agent13 executables.

    Under ``uv tool``, sys.executable points to the Python interpreter
    inside the Scripts/ directory.  On Windows, this directory also
    contains agent13.exe (the shim) and python.exe -- both are locked
    while the process is running.

    Returns:
        Path to the Scripts directory, or None if not found.
    """
    exe_dir = os.path.dirname(sys.executable)
    if os.path.isdir(exe_dir):
        return exe_dir
    return None


def _rename_locked_scripts_dir() -> Optional[str]:
    """Rename the Scripts directory to a temp location on Windows.

    Windows allows renaming a directory that contains locked (running)
    executables, but not deleting it.  By renaming Scripts/ to a
    temporary location outside the uv tools tree, ``uv tool install``
    can create a fresh Scripts directory without hitting "Access is
    denied".  Using a temp location (rather than Scripts.old in the
    same parent) prevents uv from trying to remove the old directory.

    Returns:
        The temp path on success, None if not applicable or failed.
    """
    if os.name != "nt":
        return None

    scripts_dir = _find_scripts_dir()
    if scripts_dir is None:
        return None

    # Move to temp dir outside the uv tools tree so uv doesn't
    # try to remove it during reinstall
    tmp_old = os.path.join(
        tempfile.gettempdir(),
        f"agent13-scripts-{os.getpid()}.old",
    )

    # Remove any stale temp dir from a previous interrupted update
    if os.path.exists(tmp_old):
        try:
            shutil.rmtree(tmp_old)
        except OSError:
            pass

    # Retry a few times: transient locks (antivirus scanning a file in
    # the dir, a file explorer window) clear within a second or two.
    # A persistent lock (our own cwd, a held file handle) will not clear,
    # so we give up after a bounded number of attempts and let the caller
    # fall back to the detached helper.
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            os.rename(scripts_dir, tmp_old)
            logger.info("Renamed locked Scripts dir to %s", tmp_old)
            return tmp_old
        except OSError as e:
            logger.warning(
                "Could not rename Scripts dir (attempt %d/%d): %s",
                attempt, attempts, e,
            )
            if attempt < attempts:
                time.sleep(1.0)
    return None


def _restore_renamed_scripts_dir(old_dir: str, scripts_dir: str) -> None:
    """Rollback: rename temp dir back to the original Scripts path."""
    try:
        # If uv already created a new Scripts dir, remove it first
        if os.path.exists(scripts_dir):
            shutil.rmtree(scripts_dir)
        os.rename(old_dir, scripts_dir)
        logger.info("Rolled back Scripts dir rename: %s -> %s", old_dir, scripts_dir)
    except OSError as e:
        logger.warning("Could not restore Scripts dir: %s", e)


def cleanup_old_scripts_dir() -> None:
    """Remove any leftover temp directory from a previous update.

    Call this at startup to clean up stale temp dirs.  The old Scripts
    directory is moved to %TEMP%\agent13-scripts-<pid>.old before
    install (in-process path), and the whole tool dir to
    %TEMP%\agent13-tool-helper-<pid>.old (detached helper path); both
    are cleaned up after success, and this is a safety net for
    interrupted updates.
    """
    if os.name != "nt":
        return

    # Look for agent13-scripts-*.old / agent13-tool-*.old in the temp dir
    tmp_dir = tempfile.gettempdir()
    try:
        for entry in os.listdir(tmp_dir):
            if (
                entry.startswith(("agent13-scripts-", "agent13-tool-"))
                and entry.endswith(".old")
            ):
                old_path = os.path.join(tmp_dir, entry)
                try:
                    shutil.rmtree(old_path)
                    logger.info("Cleaned up stale %s", old_path)
                except OSError as e:
                    logger.debug("Could not remove stale dir: %s", e)
    except OSError:
        pass


# Result file written by the detached updater helper, read on next launch.
_UPDATE_RESULT_FILE = get_config_dir() / "last_update.json"

# Script run by the detached helper process.  It waits for the parent
# (the running agent13) to exit so the tool dir locks are released,
# moves the whole tool dir out of the way (polling until the transient
# lock clears), runs `uv tool install --force` as a fresh install, and
# records the outcome.  On install failure it rolls the tool dir back
# so the old version keeps working.
# Args: <parent_pid> <wheel_path> <uv_path> <tool_dir>
_HELPER_SCRIPT = r"""
import json, os, shutil, subprocess, sys, time

_LOG = os.path.join(
    os.environ.get("TEMP", os.environ.get("TMP", "/tmp")),
    "agent13-updater-%s.log" % os.path.basename(sys.argv[0]),
)

def _log(msg):
    try:
        with open(_LOG, "a") as f:
            f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    except Exception:
        pass

def _alive(pid):
    # Use tasklist (not ctypes OpenProcess, which returns stale handles in
    # detached processes) to check if the PID is alive.
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        except Exception:
            return True  # assume alive on error (keep waiting)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def main():
    parent_pid = int(sys.argv[1])
    wheel = sys.argv[2]
    uv = sys.argv[3]
    tool_dir = sys.argv[4] if len(sys.argv) > 4 else ""
    # We inherit the parent's cwd, which may be the Scripts dir itself.
    # A process whose cwd is inside a dir locks that dir for removal on
    # Windows, so move away before uv tries to remove it.
    try:
        os.chdir(os.path.expanduser("~"))
    except OSError:
        pass
    _log("started parent=%d uv=%s" % (parent_pid, uv))
    # Wait (up to 60s) for the parent to exit and release the file locks.
    # Check every 1s (tasklist is a bit slow; 0.1s would spawn too many).
    deadline = time.time() + 60
    while time.time() < deadline:
        if not _alive(parent_pid):
            _log("parent exited")
            break
        time.sleep(1.0)
    else:
        _log("timed out waiting for parent")
    # Move the whole tool dir out of the way so uv installs fresh (no
    # removal needed) and a failed install can be rolled back
    # completely.  After the parent exits, the OS/AV keeps the dir tree
    # locked for a few seconds (antivirus scanning); poll (1s apart)
    # until the rename succeeds.  If a persistent lock never clears, run
    # uv anyway (it may still succeed).
    old_dir = ""
    if tool_dir and os.path.isdir(tool_dir):
        old_dir = os.path.join(
            os.environ.get("TEMP", os.environ.get("TMP", "/tmp")),
            "agent13-tool-helper-%d.old" % os.getpid(),
        )
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)
        for attempt in range(60):
            try:
                os.rename(tool_dir, old_dir)
                _log("moved tool dir to %s" % old_dir)
                break
            except OSError as e:
                _log("tool rename attempt %d failed: %s" % (attempt + 1, e))
                if attempt < 59:
                    time.sleep(1.0)
        else:
            old_dir = ""
    for attempt in (1, 2):
        _log("running uv (attempt %d)" % attempt)
        try:
            result = subprocess.run(
                [uv, "tool", "install", "--force", wheel],
                capture_output=True, text=True, timeout=300,
            )
            ok = result.returncode == 0
            detail = (result.stderr or result.stdout or "").strip()
        except Exception as e:  # noqa: BLE001 - report any failure
            ok = False
            detail = str(e)
        if ok:
            break
        # One retry: the lock may have been transient (e.g. antivirus).
        if attempt == 1:
            _log("uv failed, retrying in 3s")
            time.sleep(3)
    _log("uv done ok=%s" % ok)
    if old_dir:
        if ok:
            # Old tool dir is no longer needed; if this fails the
            # startup cleanup (agent13-tool-*.old) removes it later.
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            # Roll back so the old version keeps working.
            try:
                if os.path.isdir(tool_dir):
                    shutil.rmtree(tool_dir, ignore_errors=True)
                os.rename(old_dir, tool_dir)
                _log("restored tool dir")
            except OSError as e:
                _log("could not restore tool dir: %s" % e)
    try:
        os.unlink(wheel)
    except OSError:
        pass
    try:
        result_file = os.path.join(
            os.path.expanduser("~"), ".agent13", "last_update.json"
        )
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            json.dump({"ok": ok, "detail": detail[:2000], "ts": time.time()}, f)
        _log("wrote result")
    except Exception:  # noqa: BLE001 - result file is best-effort
        pass
    # Clean up this script file (we were run as `python <thisfile> ...`).
    try:
        os.unlink(sys.argv[0])
    except OSError:
        pass

try:
    main()
except Exception as _e:  # noqa: BLE001 - log any crash for diagnosis
    import traceback
    _log("CRASH: %s %s" % (_e, traceback.format_exc()))
"""


def _is_process_alive(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _find_helper_python() -> str:
    """Find a Python interpreter OUTSIDE the Scripts dir to run the helper.

    The helper must not use the Scripts dir's own python.exe: that
    interpreter lives inside the directory that ``uv tool install`` needs
    to remove, and a running python locks the directory for removal
    (WinError 5).  Prefer a uv-managed python, then a system python, and
    fall back to ``sys.executable`` (which may be in the Scripts dir, but
    is better than nothing).
    """
    scripts_dir = _find_scripts_dir()
    # Use ntpath (Windows semantics) so the comparison is consistent on
    # every platform -- this code path is Windows-only, but tests run on
    # macOS where os.path.normcase/os.sep are no-ops.
    scripts_prefix = (
        ntpath.normcase(ntpath.normpath(scripts_dir)) + ntpath.sep
        if scripts_dir
        else None
    )

    candidates: list[str] = []
    # 1. uv-managed python (most reliable, lives outside the Scripts dir)
    try:
        result = subprocess.run(
            ["uv", "python", "find"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    # 2. System python on PATH
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path:
            candidates.append(path)

    for path in candidates:
        if not os.path.isfile(path):
            continue
        norm = ntpath.normcase(ntpath.normpath(path))
        if scripts_prefix is None or not norm.startswith(scripts_prefix):
            return path
    # Fallback: the current interpreter (may be in the Scripts dir).
    return sys.executable


def _spawn_detached_updater(wheel_path: str, uv_path: str) -> bool:
    """Spawn a detached process that installs the wheel after we exit.

    The helper waits for this process to exit (releasing the Scripts dir
    locks), then runs ``uv tool install --force <wheel>`` and records the
    outcome in ``_UPDATE_RESULT_FILE``.

    Returns:
        True if the helper was spawned, False otherwise.
    """
    # Write the helper to a temp file and run it as `python <file> ...`.
    # Passing it via `-c` is unreliable on Windows: the script contains
    # double quotes that subprocess.list2cmdline does not escape, so the
    # command line is mis-parsed.  A file avoids all quoting issues.
    script_path = os.path.join(
        tempfile.gettempdir(), f"agent13-updater-{os.getpid()}.py"
    )
    try:
        with open(script_path, "w") as f:
            f.write(_HELPER_SCRIPT)
    except OSError as e:
        logger.warning("Could not write helper script: %s", e)
        return False

    scripts_dir = _find_scripts_dir()
    cmd = [
        _find_helper_python(),
        script_path,
        str(os.getpid()),
        wheel_path,
        uv_path,
        os.path.dirname(scripts_dir) if scripts_dir else "",
    ]
    common = dict(
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                cmd,
                creationflags=(
                    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                ),
                **common,
            )
        else:
            subprocess.Popen(cmd, start_new_session=True, **common)
        logger.info("Spawned detached updater helper for %s", wheel_path)
        return True
    except OSError as e:
        logger.warning("Could not spawn detached updater: %s", e)
        return False


def check_last_update_result() -> Optional[str]:
    """Read and clear the detached updater's result file.

    Called at startup to report on a scheduled update from a previous
    launch.  Returns a human-readable message, or None if there is no
    recorded result (or it could not be read).
    """
    try:
        if not _UPDATE_RESULT_FILE.exists():
            return None
        with open(_UPDATE_RESULT_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug("Could not read last update result: %s", e)
        return None
    finally:
        try:
            _UPDATE_RESULT_FILE.unlink()
        except OSError:
            pass

    if data.get("ok"):
        return "Your scheduled update completed successfully."
    detail = (data.get("detail") or "").strip()
    msg = "Your scheduled update did not complete."
    if detail:
        msg += f" ({detail[:300]})"
    msg += (
        " Close any file explorer windows showing the agent13 install "
        "directory and temporarily disable antivirus, then run "
        "`agent13 --upgrade` again."
    )
    return msg


def check_for_update(
    interval_hours: float = 24,
) -> Optional[dict]:
    """Check if a newer version is available on GitHub.

    Args:
        interval_hours: Minimum hours between checks (throttle).

    Returns:
        A dict with update info if an update is available, None otherwise.
        Dict keys: remote_tag, local_version, wheel_url, manual_cmd
    """
    if not _should_check(interval_hours):
        return None

    release = fetch_latest_release()
    if release is None:
        return None

    now = datetime.now(timezone.utc)
    _write_last_check(now)

    remote_tag = release["tag_name"]
    if _is_newer(remote_tag, __version__):
        wheel_url = release.get("wheel_url", "")
        manual_cmd = _build_manual_command(wheel_url) if wheel_url else ""
        return {
            "remote_tag": remote_tag,
            "local_version": __version__,
            "wheel_url": wheel_url,
            "manual_cmd": manual_cmd,
        }
    return None


def format_update_notice(info: dict) -> str:
    """Format update info dict into a human-readable multi-line notice.

    Args:
        info: Dict from check_for_update() with keys:
              remote_tag, local_version, wheel_url, manual_cmd

    Returns:
        Formatted multi-line string suitable for terminal display.
    """
    remote_tag = info["remote_tag"]
    local_version = info["local_version"]
    manual_cmd = info.get("manual_cmd", "")

    lines = [
        f">> Update available: {remote_tag} (you have {local_version})",
        "",
        "  From TUI use:  /upgrade",
        "  Or run:        agent13 --upgrade",
    ]
    if manual_cmd:
        lines.append(f"  Or run:        {manual_cmd}")
    lines.append("")
    lines.append(
        "  To disable this check set:\n"
        "      check_enabled = false in [updates] section\n"
        "  of ~/.agent13/config.toml"
    )
    return "\n".join(lines)


class UpdateStatus(Enum):
    """Outcome of check_and_apply_update."""

    UPDATED = "updated"  # Upgrade applied successfully
    UP_TO_DATE = "up_to_date"  # No newer version available
    CANCELLED = "cancelled"  # User declined the confirm prompt
    COPIED = "copied"  # Manual command copied to clipboard (copy_mode)
    FAILED = "failed"  # perform_update failed; manual_cmd is the fallback
    UNREACHABLE = "unreachable"  # Could not reach GitHub releases API


@dataclass
class UpdateResult:
    """Structured outcome of check_and_apply_update.

    Attributes:
        status: Outcome category. Callers render based on this.
        message: Human-readable detail (success message, failure reason,
            or "Already on latest..." etc.). Does NOT include a restart hint;
            callers add context-appropriate hints.
        manual_cmd: The ``uv tool install --force <wheel_url>`` command, or
            empty string if no wheel asset was found. Present for FAILED
            (fallback) and COPIED, and also populated for UPDATED (in case
            the caller wants to show "or run manually").
        remote_tag: The remote version tag, or empty string if unreachable.
    """

    status: UpdateStatus
    message: str
    manual_cmd: str = ""
    remote_tag: str = ""


def check_and_apply_update(
    copy_mode: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
    confirm: Optional[Callable[[str], bool]] = None,
) -> UpdateResult:
    """Check for an update and optionally apply it.

    Centralizes the ``/upgrade`` flow used by the REPL and TUI: fetch the
    latest release, write the last-check timestamp, compare versions, then
    either copy the manual install command (``copy_mode``) or perform the
    upgrade (after optional user confirmation via ``confirm``).

    Args:
        copy_mode: If True, do not apply the upgrade. Instead, the manual
            install command is built and returned in the result with status
            COPIED. The caller is responsible for the actual clipboard write
            (keeps this function pure of UI concerns). ``confirm`` is not
            called in copy mode (copying is non-destructive).
        on_status: Optional callback for progress messages (e.g.
            "Checking for updates...", "Downloading and installing...").
            Called zero or more times. If None, messages are ignored.
        confirm: Optional callback invoked with the remote tag (e.g.
            "v0.2.0") before applying the upgrade. Returns True to proceed,
            False to cancel. If None, the upgrade proceeds without asking.
            Not called when ``copy_mode`` is True or when no update is
            available.

    Returns:
        UpdateResult describing the outcome. Callers decide how to render
        each status (plain text for REPL, Rich markup for TUI, etc.).
    """
    if on_status:
        on_status("Checking for updates...")

    release = fetch_latest_release()
    if release is None:
        return UpdateResult(
            status=UpdateStatus.UNREACHABLE,
            message="Could not reach GitHub releases API.",
        )

    # Record the check timestamp so the throttled startup check (cli.py)
    # doesn't keep nagging after a /upgrade invocation. Previously the REPL
    # path skipped this, causing repeated notices.
    _write_last_check(datetime.now(timezone.utc))

    remote_tag = release["tag_name"]
    if not _is_newer(remote_tag, __version__):
        return UpdateResult(
            status=UpdateStatus.UP_TO_DATE,
            message=f"Already on latest version ({__version__}).",
            remote_tag=remote_tag,
        )

    wheel_url = release.get("wheel_url", "")
    manual_cmd = _build_manual_command(wheel_url) if wheel_url else ""

    if copy_mode:
        if not manual_cmd:
            return UpdateResult(
                status=UpdateStatus.FAILED,
                message=(
                    f"No wheel asset found for {remote_tag}. "
                    f"Cannot build install command."
                ),
                remote_tag=remote_tag,
            )
        return UpdateResult(
            status=UpdateStatus.COPIED,
            message=manual_cmd,
            manual_cmd=manual_cmd,
            remote_tag=remote_tag,
        )

    if confirm is not None:
        if not confirm(remote_tag):
            return UpdateResult(
                status=UpdateStatus.CANCELLED,
                message="Update cancelled.",
                manual_cmd=manual_cmd,
                remote_tag=remote_tag,
            )

    if on_status:
        on_status(
            f"Update available: {remote_tag} (you have {__version__})."
        )

    success, message = perform_update(on_status=on_status)
    if success:
        return UpdateResult(
            status=UpdateStatus.UPDATED,
            message=message,
            manual_cmd=manual_cmd,
            remote_tag=remote_tag,
        )
    return UpdateResult(
        status=UpdateStatus.FAILED,
        message=message,
        manual_cmd=manual_cmd,
        remote_tag=remote_tag,
    )


def _download_with_progress(
    url: str,
    dest_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Optional[int]:
    """Download a file with streaming progress reporting.

    Args:
        url: URL to download from.
        dest_path: Local path to write the file.
        on_progress: Optional callback called with progress messages.

    Returns:
        None on success. On HTTP failure (non-200), the HTTP status code.
        On network/OS errors, raises (httpx.HTTPError or OSError).
    """
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        if resp.status_code != 200:
            return resp.status_code

        # Try to get content length for progress
        content_length = resp.headers.get("content-length")
        try:
            total = int(content_length) if content_length else None
        except ValueError:
            total = None

        downloaded = 0
        last_pct = -1

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and total > 0:
                        pct = int((downloaded / total) * 100)
                        # Report every 10% to avoid spam
                        if pct >= last_pct + 10:
                            last_pct = pct
                            if on_progress:
                                on_progress(f"Downloading: {pct}%")

        return None


def perform_update(
    on_status: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """Attempt an in-place upgrade by downloading the wheel from GitHub.

    Downloads the .whl from the latest GitHub release and installs it
    via `uv tool install --force <wheel_path>`.

    On Windows, the Scripts directory contains both the agent13.exe shim
    and python.exe (the running interpreter), both of which are locked.
    We rename the entire Scripts directory to Scripts.old (Windows allows
    this), then run uv tool install, which creates a fresh Scripts dir.
    If install fails, we roll back the rename.

    Args:
        on_status: Optional callback for progress messages. Called with
            plain text strings (no formatting). If None, messages are
            silently dropped.

    Returns:
        Tuple of (success: bool, message: str).
        On success the message does NOT include a restart hint -- callers
        add context-appropriate hints (TUI vs CLI vs --upgrade).
    """
    def _say(msg: str):
        if on_status:
            on_status(msg)

    # Step 1: Fetch latest release info
    release = fetch_latest_release()
    if release is None:
        return False, "Could not reach GitHub releases API."

    remote_tag = release["tag_name"]
    wheel_url = release.get("wheel_url", "")

    if not _is_newer(remote_tag, __version__):
        return True, f"Already on latest version ({__version__})."

    if not wheel_url:
        return False, (
            f"Update available ({remote_tag}) but no wheel asset found "
            f"on GitHub release. Install manually."
        )

    # Step 2: Download the wheel to a temp file (streaming with progress).
    #   Use the real wheel filename from the URL: uv validates wheel filenames
    #   against PEP 427, which requires
    #   {distribution}-{version}-{python}-{abi}-{platform}.whl -- a bare
    #   tmpXXXX.whl would be rejected.  Extract the real filename so the temp
    #   path passes validation.
    wheel_name = wheel_url.rsplit("/", 1)[-1]
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, wheel_name)

    if os.name == "nt":
        _say("Downloading update (may take a moment)...")
    else:
        _say("Downloading update...")

    try:
        status = _download_with_progress(
            wheel_url, tmp_path, on_progress=on_status,
        )
        if status is not None:
            # Clean up any partial file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False, (
                f"Failed to download wheel (HTTP {status}). "
                f"Try manually: {_build_manual_command(wheel_url)}"
            )
    except (httpx.HTTPError, OSError) as e:
        # Clean up partial file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False, (
            f"Failed to download wheel: {e}. "
            f"Try manually: {_build_manual_command(wheel_url)}"
        )

    # Step 3: On Windows, rename the locked Scripts dir so uv can replace it
    scripts_dir = _find_scripts_dir()
    renamed_old = _rename_locked_scripts_dir()

    # Step 3b: If the rename failed, the Scripts dir is still locked (a
    # process holds it as its cwd, or has a file inside it open).  An
    # in-process `uv tool install` would fail for the same reason.  Fall
    # back to a detached helper that runs the install after we exit, when
    # our own locks are released.  The wheel is kept (not unlinked in the
    # finally block below) because the helper consumes it.
    if os.name == "nt" and renamed_old is None and scripts_dir is not None:
        uv_path = shutil.which("uv") or "uv"
        if _spawn_detached_updater(tmp_path, uv_path):
            _say(
                "Scripts dir is locked; scheduling the update to finish "
                "after agent13 exits."
            )
            return True, (
                "Update scheduled. It will complete after agent13 exits. "
                "Restart agent13 to use the new version."
            )
        # Could not spawn the helper; fall through to the in-process
        # install so the user sees the real error.

    # Step 4: Install via uv
    if os.name == "nt":
        _say("Installing update (may take a few minutes on Windows)...")
    else:
        _say("Installing update...")

    try:
        result = subprocess.run(
            ["uv", "tool", "install", "--force", tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Clean up the old Scripts directory
            if renamed_old and os.path.exists(renamed_old):
                try:
                    shutil.rmtree(renamed_old)
                except OSError:
                    pass
            return True, f"Updated to {remote_tag} successfully."

        # On Windows, uv may fail to copy the entrypoint shim to
        # ~/.local/bin/ because that file is locked by the running
        # process.  The packages are already installed at this point --
        # only the tiny launcher exe wasn't refreshed.  Since the
        # launcher is version-agnostic, the old one works fine.
        stderr = result.stderr.strip()
        if os.name == "nt" and "Failed to install entrypoint" in stderr:
            # Packages installed successfully, entrypoint just wasn't
            # refreshed.  Clean up old Scripts dir and report success.
            if renamed_old and os.path.exists(renamed_old):
                try:
                    shutil.rmtree(renamed_old)
                except OSError:
                    pass
            return True, f"Updated to {remote_tag} successfully."

        # Install failed -- roll back the Scripts dir rename on Windows
        if renamed_old:
            _restore_renamed_scripts_dir(renamed_old, scripts_dir)

        return False, (
            f"Install failed: {stderr}. "
            f"Try manually: {_build_manual_command(wheel_url)}"
        )
    except subprocess.TimeoutExpired:
        if renamed_old:
            _restore_renamed_scripts_dir(renamed_old, scripts_dir)
        return False, (
            f"Install timed out. "
            f"Try manually: {_build_manual_command(wheel_url)}"
        )
    except OSError as e:
        if renamed_old:
            _restore_renamed_scripts_dir(renamed_old, scripts_dir)
        return False, (
            f"Install failed: {e}. "
            f"Try manually: {_build_manual_command(wheel_url)}"
        )
    finally:
        # Clean up temp wheel file (always defined by this point)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
