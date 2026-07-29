"""Shared security utilities for file operation tools.

Provides soft enforcement of sandbox policies for read_file and edit_file.
Uses the same sandbox modes as bash (controlled via /sandbox command).
"""

import re
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

# Matches a `..` path component (traversal), not `..` inside a filename like
# `foo..bar.txt` or `..hidden`. A `..` is a traversal iff it's bounded by a
# path separator or the start/end of the string.
_TRAVERSAL_RE = re.compile(r"(^\.\.$|^\.\./|/\.\./|/\.\.$)")

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


def _validate_path(
    filepath: str,
    cwd: Path,
    allow_any: bool,
    project_allowed: bool,
    explicit_dirs: list[str],
    verb: str,
) -> Tuple[bool, str]:
    """Shared core for read/write path validation.

    Args:
        filepath: Original path string (used in error messages).
        cwd: Base directory for resolving relative paths.
        allow_any: If True, allow reads/writes anywhere (skip further checks).
        project_allowed: If True, allow paths under ``cwd``.
        explicit_dirs: Extra allowed directories from the sandbox profile.
        verb: "Read" or "Write" — used in the denial message.

    Returns:
        (is_allowed, error_message) — error_message is empty when allowed.
    """
    mode = get_current_sandbox_mode()

    # Cheap early-exit for path traversal: reject any `..` that appears as a
    # path component (e.g. `../secret`, `foo/../bar`) while allowing legitimate
    # `..` inside filenames like `foo..bar.txt`. The resolved-path checks below
    # also catch traversal, but this gives a clearer error message and avoids
    # touching the filesystem for obvious cases.
    if _TRAVERSAL_RE.search(filepath):
        return False, "Path traversal not allowed: '..' in path"

    base_dir = (cwd or Path.cwd()).resolve()
    path = Path(filepath)

    # Resolve path relative to cwd if provided
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (base_dir / path).resolve()

    # If this operation is allowed anywhere, return immediately
    if allow_any:
        return True, ""

    # Check if under project directory (if allowed)
    if project_allowed:
        try:
            resolved.relative_to(base_dir)
            return True, ""
        except ValueError:
            pass  # Not under project, check other paths

    # Check against explicit paths from sandbox profile
    for allowed_dir in explicit_dirs:
        if _is_path_under_directory(resolved, allowed_dir):
            return True, ""

    # Deny access
    return False, (
        f"{verb} access denied: path not in allowed directories.\n"
        f"  Path: {filepath}\n"
        f"  Sandbox mode: {mode.value}\n"
        f"  Use '/sandbox off' to allow {verb.lower()}s anywhere."
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


def validate_path_for_read(filepath: str, cwd: Path = None) -> Tuple[bool, str]:
    """Validate if a path can be read based on current sandbox mode.

    Uses the actual sandbox profile files to determine allowed paths,
    ensuring consistency with the bash tool's sandbox enforcement.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    paths = parse_sandbox_paths(get_current_sandbox_mode())
    return _validate_path(
        filepath,
        cwd,
        allow_any=paths.allow_any_read,
        project_allowed=paths.project_read,
        explicit_dirs=paths.read_paths,
        verb="Read",
    )


def validate_path_for_write(filepath: str, cwd: Path = None) -> Tuple[bool, str]:
    """Validate if a path can be written based on current sandbox mode.

    Uses the actual sandbox profile files to determine allowed paths,
    ensuring consistency with the bash tool's sandbox enforcement.

    Returns:
        Tuple of (is_allowed, error_message)
    """
    paths = parse_sandbox_paths(get_current_sandbox_mode())
    return _validate_path(
        filepath,
        cwd,
        allow_any=paths.allow_any_write,
        project_allowed=paths.project_write,
        explicit_dirs=paths.write_paths,
        verb="Write",
    )
