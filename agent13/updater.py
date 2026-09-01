"""Self-update checker for agent13.

Checks GitHub releases for newer versions, throttled to once per day.
Can perform in-place upgrade via uv tool and prompt user to restart.

On Windows, the Scripts directory (containing both agent13.exe and
python.exe) is locked by the OS and cannot be deleted or replaced.
However, Windows *does* allow renaming locked files and directories.
We exploit this by renaming Scripts/ -> Scripts.old/ before running
``uv tool install --force``, which can then create a fresh Scripts/
directory unimpeded.  Any leftover .old directory is cleaned up on
next launch.

Config keys (in ~/.agent13/config.toml):
    [updates]
    check_enabled = true          # Set to false to disable update checks
    check_interval_hours = 24    # Minimum hours between checks
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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

    try:
        os.rename(scripts_dir, tmp_old)
        logger.info("Renamed locked Scripts dir to %s", tmp_old)
        return tmp_old
    except OSError as e:
        logger.warning("Could not rename Scripts dir: %s", e)
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
    """Remove any leftover temp Scripts directory from a previous update.

    Call this at startup to clean up stale temp dirs.  The old Scripts
    directory is moved to %TEMP%\agent13-scripts-<pid>.old before
    install and cleaned up after success, but this is a safety net for
    interrupted updates.
    """
    if os.name != "nt":
        return

    # Look for agent13-scripts-*.old in the temp directory
    tmp_dir = tempfile.gettempdir()
    try:
        for entry in os.listdir(tmp_dir):
            if entry.startswith("agent13-scripts-") and entry.endswith(".old"):
                old_path = os.path.join(tmp_dir, entry)
                try:
                    shutil.rmtree(old_path)
                    logger.info("Cleaned up stale %s", old_path)
                except OSError as e:
                    logger.debug("Could not remove stale dir: %s", e)
    except OSError:
        pass


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
