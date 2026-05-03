"""Self-update tool for agent13.

Checks GitHub releases for newer versions and can perform in-place upgrades
by downloading the wheel from the latest GitHub release.
"""

from agent13.tools import tool
from agent13.clipboard import copy_via_system
from agent13.updater import (
    perform_update,
    fetch_latest_release,
    _is_newer,
    _build_manual_command,
    _write_last_check,
)
from agent13 import __version__
from datetime import datetime, timezone


@tool()
def self_update(action: str = "check") -> str:
    """Check for agent13 updates, perform an in-place upgrade, or copy the install command.

    Args:
        action: "check" to check for updates (default),
                "apply" to download and install the latest version,
                "copy" to copy the manual install command to clipboard

    Returns:
        Status message about available updates or update result
    """
    if action not in ("check", "apply", "copy"):
        return (
            "Usage: self_update(action='check' | 'apply' | 'copy'). "
            "'check' shows if an update is available, "
            "'apply' downloads and installs the latest version, "
            "'copy' copies the manual install command to clipboard."
        )

    # Fetch latest release info
    release = fetch_latest_release()
    if release is None:
        return f"Could not reach GitHub releases API. Current version: {__version__}"

    now = datetime.now(timezone.utc)
    _write_last_check(now)

    remote_tag = release["tag_name"]
    if not _is_newer(remote_tag, __version__):
        return f"You're up to date (version {__version__})."

    wheel_url = release.get("wheel_url", "")
    manual_cmd = _build_manual_command(wheel_url) if wheel_url else ""

    if action == "check":
        msg = (
            f"Update available: {remote_tag} (you have {__version__}). "
            f"Use self_update(action='apply') to upgrade."
        )
        if manual_cmd:
            msg += f" Or run manually: {manual_cmd}"
        return msg

    if action == "copy":
        if not manual_cmd:
            return f"No wheel asset found for {remote_tag}. Cannot build install command."
        if copy_via_system(manual_cmd):
            return f"Copied to clipboard: {manual_cmd}"
        return f"Could not copy to clipboard. Manual command: {manual_cmd}"

    if action == "apply":
        success, message = perform_update()
        if not success and manual_cmd:
            message += f" Manual command: {manual_cmd}"
        return message
