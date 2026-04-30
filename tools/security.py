"""Shared security utilities for file operation tools.

Provides soft enforcement of sandbox policies for read_file and edit_file.
Uses the same sandbox modes as bash (controlled via /sandbox command).
"""

from pathlib import Path
from typing import Optional, Tuple

from agent13.sandbox import (
    SandboxMode,
    SandboxCapabilities,
    get_effective_sandbox_mode,
    parse_sandbox_paths,
    clear_sandbox_paths_cache,
    SANDBOX_CAPABILITIES,
)

# Module-level session override (shared across all tools)
_session_sandbox_mode: Optional[SandboxMode] = None


def set_session_sandbox_mode(mode: Optional[SandboxMode]) -> None:
    """Set session-level sandbox mode override.

    Args:
        mode: The sandbox mode to use, or None to use config default
    """
    global _session_sandbox_mode
    _session_sandbox_mode = mode
    # Clear the sandbox paths cache so it re-parses for the new mode
    clear_sandbox_paths_cache()


def get_session_sandbox_mode() -> Optional[SandboxMode]:
    """Get the current session-level sandbox mode override.

    Returns:
        The session override, or None if using config default
    """
    return _session_sandbox_mode


def get_current_sandbox_mode() -> SandboxMode:
    """Get the current effective sandbox mode.

    Returns:
        The effective sandbox mode (session override or config default)
    """
    return get_effective_sandbox_mode(_session_sandbox_mode)


def get_current_capabilities() -> SandboxCapabilities:
    """Get capabilities for current sandbox mode."""
    return SANDBOX_CAPABILITIES[get_current_sandbox_mode()]


def validate_path_for_read(filepath: str, cwd: Path = None) -> Tuple[bool, str]:
    """Validate if a path can be read based on current sandbox mode.

    Uses the actual sandbox profile files to determine allowed paths,
    ensuring consistency with the bash tool's sandbox enforcement.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    mode = get_current_sandbox_mode()

    path = Path(filepath)

    # Always block path traversal
    if ".." in filepath:
        return False, "Path traversal not allowed: '..' in path"

    base_dir = (cwd or Path.cwd()).resolve()

    # Resolve path relative to cwd if provided
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (base_dir / path).resolve()

    # Parse sandbox profile to get allowed paths
    paths = parse_sandbox_paths(mode)

    # If reads are allowed anywhere, return immediately
    if paths.allow_any_read:
        return True, ""

    # Check if under project directory (if allowed)
    if paths.project_read:
        try:
            resolved.relative_to(base_dir)
            return True, ""
        except ValueError:
            pass  # Not under project, check other paths

    # Check against explicit read paths from sandbox profile
    for allowed_dir in paths.read_paths:
        if _is_path_under_directory(resolved, allowed_dir):
            return True, ""

    # Deny access
    return False, (
        f"Read access denied: path not in allowed directories.\n"
        f"  Path: {filepath}\n"
        f"  Sandbox mode: {mode.value}\n"
        f"  Use '/sandbox none' to allow reads anywhere."
    )


def _is_path_under_directory(path: Path, directory: str) -> bool:
    """Check if a path is under a given directory.

    Args:
        path: The resolved path to check
        directory: The directory path (string, possibly with ~)

    Returns:
        True if path is under the directory
    """
    expanded = Path(directory).expanduser().resolve()
    try:
        path.relative_to(expanded)
        return True
    except ValueError:
        return False


def validate_path_for_write(filepath: str, cwd: Path = None) -> Tuple[bool, str]:
    """Validate if a path can be written based on current sandbox mode.

    Uses the actual sandbox profile files to determine allowed paths,
    ensuring consistency with the bash tool's sandbox enforcement.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    mode = get_current_sandbox_mode()

    path = Path(filepath)

    # Always block path traversal
    if ".." in filepath:
        return False, "Path traversal not allowed: '..' in path"

    base_dir = (cwd or Path.cwd()).resolve()

    # Resolve path relative to cwd if provided
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (base_dir / path).resolve()

    # Parse sandbox profile to get allowed paths
    paths = parse_sandbox_paths(mode)

    # If writes are allowed anywhere, return immediately
    if paths.allow_any_write:
        return True, ""

    # Check if under project directory (if allowed)
    if paths.project_write:
        try:
            resolved.relative_to(base_dir)
            return True, ""
        except ValueError:
            pass  # Not under project, check other paths

    # Check against explicit write paths from sandbox profile
    for allowed_dir in paths.write_paths:
        if _is_path_under_directory(resolved, allowed_dir):
            return True, ""

    # Deny access
    return False, (
        f"Write access denied: path not in allowed directories.\n"
        f"  Path: {filepath}\n"
        f"  Sandbox mode: {mode.value}\n"
        f"  Use '/sandbox none' to allow writes anywhere."
    )
