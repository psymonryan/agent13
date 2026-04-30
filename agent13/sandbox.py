"""Sandbox management for secure command execution.

Uses macOS Seatbelt sandboxing via sandbox-exec to restrict file and network access.
"""

import os
import platform
import subprocess
import asyncio
import signal
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from agent13.config_paths import get_config_file

# Encoding for subprocess output
# Windows: Use OEM code page (what cmd.exe/native programs output)
# Unix: Use UTF-8 (standard terminal encoding)
if sys.platform == "win32":
    import ctypes

    _SUBPROCESS_ENCODING = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
else:
    _SUBPROCESS_ENCODING = "utf-8"


def get_temp_dir() -> str:
    """Return the appropriate temp directory for the current platform.

    On POSIX systems (Linux, macOS, BSD) returns '/tmp' which is the
    standard temp directory and is included in sandbox profiles.
    On Windows returns the OS-provided temp directory (e.g.
    %LOCALAPPDATA%\\Temp) since there is no sandbox enforcement.

    This ensures tests use a path that is both writable AND allowed
    by the sandbox profile on each platform.
    """
    if os.name == "nt":
        import tempfile

        return tempfile.gettempdir()
    return "/tmp"


class SandboxMode(Enum):
    """Sandbox restriction modes."""

    PERMISSIVE_OPEN = "permissive-open"
    PERMISSIVE_CLOSED = "permissive-closed"
    RESTRICTIVE_OPEN = "restrictive-open"
    RESTRICTIVE_CLOSED = "restrictive-closed"
    NONE = "none"


@dataclass
class SandboxCapabilities:
    """Describes what a sandbox mode allows."""

    file_write: str  # "project" or "anywhere"
    file_read: str  # "project" or "anywhere"
    network: bool  # True if network access allowed


@dataclass
class SandboxPaths:
    """Parsed paths from sandbox profile file."""

    write_paths: list[str]  # Absolute paths allowed for writes (excludes project dir)
    read_paths: list[str]  # Absolute paths allowed for reads (excludes project dir)
    project_write: bool  # Whether PROJECT_DIR param is allowed for writes
    project_read: bool  # Whether PROJECT_DIR param is allowed for reads
    allow_any_write: (
        bool  # Whether writes are allowed anywhere (no subpath restrictions)
    )
    allow_any_read: bool  # Whether reads are allowed anywhere (no subpath restrictions)


SANDBOX_CAPABILITIES = {
    SandboxMode.PERMISSIVE_OPEN: SandboxCapabilities(
        file_write="project",
        file_read="anywhere",
        network=True,
    ),
    SandboxMode.PERMISSIVE_CLOSED: SandboxCapabilities(
        file_write="project",
        file_read="anywhere",
        network=False,
    ),
    SandboxMode.RESTRICTIVE_OPEN: SandboxCapabilities(
        file_write="project",
        file_read="project",
        network=True,
    ),
    SandboxMode.RESTRICTIVE_CLOSED: SandboxCapabilities(
        file_write="project",
        file_read="project",
        network=False,
    ),
    SandboxMode.NONE: SandboxCapabilities(
        file_write="anywhere",
        file_read="anywhere",
        network=True,
    ),
}

# Cache for parsed sandbox paths, keyed by SandboxMode
_sandbox_paths_cache: dict[SandboxMode, SandboxPaths] = {}


def get_sandbox_profiles_dir() -> Path:
    """Get the directory containing sandbox profile files."""
    return Path(__file__).parent / "sandbox_profiles"


def get_sandbox_profile_path(mode: SandboxMode) -> Path:
    """Get the path to a sandbox profile file.

    Args:
        mode: The sandbox mode

    Returns:
        Path to the .sb profile file
    """
    return get_sandbox_profiles_dir() / f"{mode.value}.sb"


def load_sandbox_profile(mode: SandboxMode) -> str:
    """Load a sandbox profile file.

    Args:
        mode: The sandbox mode

    Returns:
        The profile content as a string

    Raises:
        FileNotFoundError: If the profile file doesn't exist
    """
    profile_path = get_sandbox_profile_path(mode)
    if not profile_path.exists():
        raise FileNotFoundError(f"Sandbox profile not found: {profile_path}")
    return profile_path.read_text()


def parse_sandbox_paths(mode: SandboxMode) -> SandboxPaths:
    """Parse allowed paths from sandbox profile file.

    Extracts file-read* and file-write* paths from the .sb profile.
    Results are cached per mode to avoid re-parsing.

    Args:
        mode: The sandbox mode

    Returns:
        SandboxPaths with parsed read/write paths and project flags
    """
    # Check cache first
    if mode in _sandbox_paths_cache:
        return _sandbox_paths_cache[mode]

    # For 'none' mode, allow everything
    if mode == SandboxMode.NONE:
        return SandboxPaths(
            write_paths=[],
            read_paths=[],
            project_write=True,
            project_read=True,
            allow_any_write=True,
            allow_any_read=True,
        )

    # Load and parse the profile
    content = load_sandbox_profile(mode)
    write_paths = []
    read_paths = []
    allow_any_write = False
    allow_any_read = False

    def finalize_block(allow_type: str, paths: list) -> None:
        """Add accumulated paths to the appropriate list.

        If paths is empty for this block, it means the allow directive
        had no subpath restrictions, so we set allow_any_* instead.
        """
        nonlocal allow_any_write, allow_any_read
        if allow_type == "write":
            if not paths:
                # (allow file-write*) with no subpaths = allow any write
                allow_any_write = True
            else:
                write_paths.extend(paths)
        elif allow_type == "read":
            if not paths:
                # (allow file-read*) with no subpaths = allow any read
                allow_any_read = True
            else:
                read_paths.extend(paths)

    current_allow_type = None
    current_block_paths = []

    for line in content.split("\n"):
        stripped = line.strip()

        # Skip comments and empty lines (but finalize any pending block first)
        if not stripped or stripped.startswith(";"):
            if current_allow_type:
                finalize_block(current_allow_type, current_block_paths)
                current_allow_type = None
                current_block_paths = []
            continue

        # Start of a new allow file-write* block
        if stripped.startswith("(allow file-write"):
            # Finalize any previous block
            if current_allow_type:
                finalize_block(current_allow_type, current_block_paths)
            current_allow_type = "write"
            current_block_paths = []
            _extract_subpaths_from_line(stripped, current_block_paths)
            continue

        # Start of a new allow file-read* block
        if stripped.startswith("(allow file-read"):
            # Finalize any previous block
            if current_allow_type:
                finalize_block(current_allow_type, current_block_paths)
            current_allow_type = "read"
            current_block_paths = []
            _extract_subpaths_from_line(stripped, current_block_paths)
            continue

        # If we're in an allow block, look for subpaths or literals
        if current_allow_type:
            if stripped.startswith("(subpath") or stripped.startswith("(literal"):
                _extract_subpaths_from_line(stripped, current_block_paths)
            else:
                # Any other line ends the current block
                finalize_block(current_allow_type, current_block_paths)
                current_allow_type = None
                current_block_paths = []

    # Finalize any remaining block
    if current_allow_type:
        finalize_block(current_allow_type, current_block_paths)

    # Check for PROJECT_DIR in the paths
    project_write = any('(param "PROJECT_DIR")' in p for p in write_paths)
    project_read = any('(param "PROJECT_DIR")' in p for p in read_paths)

    # Resolve param references to actual paths and keep them
    # (previously we dropped param refs, but now we have AGENT_DIR,
    # USER_CACHE, USER_LOCAL that need to be resolved for path validation)
    from agent13.config_paths import get_config_dir as _get_config_dir

    _param_resolvers = {
        "PROJECT_DIR": str(Path.cwd().resolve()),
        "AGENT_DIR": str(_get_config_dir().resolve()),
        "USER_CACHE": str(Path.home() / ".cache"),
        "USER_LOCAL": str(Path.home() / ".local" / "share"),
        "USER_LOCAL_BIN": str(Path.home() / ".local" / "bin"),
    }

    def _resolve_param_path(p: str) -> str:
        """Resolve a (param "NAME") reference to an actual path."""
        import re

        m = re.match(r'\(param "([^"]+)"\)', p)
        if m and m.group(1) in _param_resolvers:
            return _param_resolvers[m.group(1)]
        return _expand_path(p)

    write_paths = [_resolve_param_path(p) for p in write_paths]
    read_paths = [_resolve_param_path(p) for p in read_paths]

    # On non-macOS platforms the .sb profile paths (e.g. /tmp) don't exist.
    # Add the platform's actual temp directory so validate_path_for_write
    # allows writes there. On macOS this is redundant (/tmp is already in
    # the profile) but harmless.
    if not is_macos():
        temp_dir = get_temp_dir()
        if temp_dir not in write_paths:
            write_paths.append(temp_dir)
        if temp_dir not in read_paths:
            read_paths.append(temp_dir)

    result = SandboxPaths(
        write_paths=write_paths,
        read_paths=read_paths,
        project_write=project_write,
        project_read=project_read,
        allow_any_write=allow_any_write,
        allow_any_read=allow_any_read,
    )

    # Cache the result
    _sandbox_paths_cache[mode] = result
    return result


def clear_sandbox_paths_cache() -> None:
    """Clear the cached sandbox paths.

    Call this when the sandbox mode changes to force re-parsing.
    """
    global _sandbox_paths_cache
    _sandbox_paths_cache = {}


def _extract_subpaths_from_line(line: str, paths: list) -> None:
    """Extract subpath and literal values from a line.

    Handles formats like:
        (subpath "/path/to/dir")
        (subpath (param "PROJECT_DIR"))
        (literal "/dev/null")
    """
    import re

    # Match (subpath "...") or (subpath (param "..."))
    subpath_pattern = r'\(subpath\s+((?:\([^)]+\)|"[^"]+"))\)'
    for match in re.finditer(subpath_pattern, line):
        paths.append(match.group(1))

    # Also match (literal "...") - these are specific file paths
    literal_pattern = r'\(literal\s+("[^"]+")\)'
    for match in re.finditer(literal_pattern, line):
        paths.append(match.group(1))


def _finalize_allow_block(
    allow_type: str, paths: list, write_paths: list, read_paths: list
) -> None:
    """Finalize an allow block, adding paths to the appropriate list."""
    if allow_type == "write":
        write_paths.extend(paths)
    elif allow_type == "read":
        read_paths.extend(paths)


def _expand_path(path: str) -> str:
    """Expand a path string, handling ~ and quotes."""
    # Remove surrounding quotes
    path = path.strip('"')

    # Expand ~ to home directory
    if path.startswith("~"):
        path = os.path.expanduser(path)

    return path


def is_macos() -> bool:
    """Check if we're running on macOS."""
    return platform.system() == "Darwin"


def get_default_sandbox_mode() -> SandboxMode:
    """Get the default sandbox mode from config.

    Returns:
        The default sandbox mode (defaults to PERMISSIVE_OPEN if not configured)
    """
    config_path = get_config_file()
    if not config_path.exists():
        return SandboxMode.PERMISSIVE_OPEN

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        sandbox_config = data.get("sandbox", {})
        default_mode = sandbox_config.get("default", "permissive-open")

        # Parse the mode string
        return parse_sandbox_mode(default_mode)
    except Exception:
        # On any error, use default
        return SandboxMode.PERMISSIVE_OPEN


def parse_sandbox_mode(mode_str: str) -> SandboxMode:
    """Parse a sandbox mode string.

    Args:
        mode_str: The mode string (e.g., "permissive-open", "none")

    Returns:
        The corresponding SandboxMode

    Raises:
        ValueError: If the mode string is invalid
    """
    mode_str = mode_str.lower().strip()
    mode_map = {
        "permissive-open": SandboxMode.PERMISSIVE_OPEN,
        "permissive-closed": SandboxMode.PERMISSIVE_CLOSED,
        "restrictive-open": SandboxMode.RESTRICTIVE_OPEN,
        "restrictive-closed": SandboxMode.RESTRICTIVE_CLOSED,
        "none": SandboxMode.NONE,
        "disabled": SandboxMode.NONE,  # alias
        "off": SandboxMode.NONE,  # alias
    }
    if mode_str not in mode_map:
        valid_modes = ", ".join(m.value for m in SandboxMode)
        raise ValueError(
            f"Invalid sandbox mode: {mode_str}. Valid modes: {valid_modes}"
        )
    return mode_map[mode_str]


def get_effective_sandbox_mode(
    session_override: Optional[SandboxMode] = None,
) -> SandboxMode:
    """Get the effective sandbox mode.

    Priority: session_override > config default > hardcoded default

    Args:
        session_override: Optional session-level override

    Returns:
        The effective sandbox mode
    """
    if session_override is not None:
        return session_override
    return get_default_sandbox_mode()


def build_sandbox_command(
    command: str, mode: SandboxMode, project_dir: Optional[Path] = None
) -> list[str]:
    """Build a sandboxed command.

    Args:
        command: The command to run
        mode: The sandbox mode
        project_dir: The project directory (defaults to cwd)

    Returns:
        The command as a list of arguments (suitable for subprocess)

    Note:
        On non-macOS systems or with mode=NONE, returns the command unchanged
        (wrapped in shell for subprocess compatibility).
    """
    if not is_macos() or mode == SandboxMode.NONE:
        # No sandboxing on non-macOS or when disabled
        if sys.platform == "win32":
            return ["cmd.exe", "/c", command]
        return ["/bin/sh", "-c", command]

    if project_dir is None:
        project_dir = Path.cwd()

    profile_path = get_sandbox_profile_path(mode)

    # Build -D parameters for profile variable substitution
    # These replace hardcoded home-directory paths so profiles work
    # for any user, not just the developer.
    from agent13.config_paths import get_config_dir

    d_params = [
        f"PROJECT_DIR={project_dir.resolve()}",
        f"AGENT_DIR={get_config_dir().resolve()}",
        f"USER_CACHE={Path.home() / '.cache'}",
        f"USER_LOCAL={Path.home() / '.local' / 'share'}",
        f"USER_LOCAL_BIN={Path.home() / '.local' / 'bin'}",
        f"USER_HOME_SSH={Path.home() / '.ssh'}",
    ]

    # Build sandbox-exec command
    # sandbox-exec -f profile.sb -D K=V ... -- command
    cmd = ["sandbox-exec", "-f", str(profile_path)]
    for p in d_params:
        cmd.extend(["-D", p])
    cmd.extend(["--", "/bin/sh", "-c", command])
    return cmd


def run_sandboxed(
    command: str,
    mode: SandboxMode,
    timeout: float = 30.0,
    max_output: int = 100000,
    project_dir: Optional[Path] = None,
) -> dict:
    """Run a command in a sandbox.

    Args:
        command: The command to run
        mode: The sandbox mode
        timeout: Timeout in seconds (default 30)
        max_output: Maximum output size in bytes (default 100KB)
        project_dir: The project directory (defaults to cwd)

    Returns:
        Dict with:
        - success: bool
        - exit_code: int
        - stdout: str
        - stderr: str
        - truncated: bool (if output was truncated)
        - timed_out: bool (if command timed out)
        - sandbox_mode: str (the mode used)
    """
    sandbox_cmd = build_sandbox_command(command, mode, project_dir)

    try:
        result = subprocess.run(
            sandbox_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir) if project_dir else None,
        )

        stdout = result.stdout
        stderr = result.stderr
        truncated = False

        # Truncate output if needed
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
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "timed_out": False,
            "sandbox_mode": mode.value,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds. You can specify a larger timeout (up to 300 seconds) using the timeout parameter.",
            "truncated": False,
            "timed_out": True,
            "sandbox_mode": mode.value,
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "truncated": False,
            "timed_out": False,
            "sandbox_mode": mode.value,
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Error running command: {e}",
            "truncated": False,
            "timed_out": False,
            "sandbox_mode": mode.value,
        }


def format_sandbox_mode_info(mode: SandboxMode) -> str:
    """Format a sandbox mode for display.

    Args:
        mode: The sandbox mode

    Returns:
        A human-readable description
    """
    caps = SANDBOX_CAPABILITIES[mode]

    lines = [
        f"Mode: {mode.value}",
        f"  File write: {caps.file_write}",
        f"  File read: {caps.file_read}",
        f"  Network: {'allowed' if caps.network else 'blocked'}",
    ]
    return "\n".join(lines)


def format_all_sandbox_modes() -> str:
    """Format all sandbox modes for display.

    Returns:
        A human-readable list of all modes
    """
    lines = ["Available sandbox modes:"]
    for mode in SandboxMode:
        caps = SANDBOX_CAPABILITIES[mode]
        network_str = "network" if caps.network else "no network"
        read_str = "read anywhere" if caps.file_read == "anywhere" else "read project"
        write_str = (
            "write anywhere" if caps.file_write == "anywhere" else "write project"
        )
        lines.append(f"  {mode.value}: {write_str}, {read_str}, {network_str}")
    return "\n".join(lines)


def validate_sandbox_profiles() -> list[str]:
    """Validate all sandbox profile files exist and are readable.

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    for mode in SandboxMode:
        if mode == SandboxMode.NONE:
            continue  # No profile file for 'none' mode
        try:
            profile_path = get_sandbox_profile_path(mode)
            if not profile_path.exists():
                errors.append(f"Missing sandbox profile: {profile_path}")
            else:
                # Try to read the file
                profile_path.read_text()
        except Exception as e:
            errors.append(f"Error reading sandbox profile for {mode.value}: {e}")
    return errors


async def run_sandboxed_async(
    command: str,
    mode: SandboxMode,
    timeout: float = 30.0,
    max_output: int = 100000,
    project_dir: Optional[Path] = None,
) -> dict:
    """Run a command in a sandbox asynchronously.

    This is the async version of run_sandboxed that uses asyncio subprocess
    to avoid blocking the event loop.

    Args:
        command: The command to run
        mode: The sandbox mode
        timeout: Timeout in seconds (default 30)
        max_output: Maximum output size in bytes (default 100KB)
        project_dir: The project directory (defaults to cwd)

    Returns:
        Dict with:
        - success: bool
        - exit_code: int
        - stdout: str
        - stderr: str
        - truncated: bool (if output was truncated)
        - timed_out: bool (if command timed out)
        - sandbox_mode: str (the mode used)
    """
    # Windows requires different subprocess handling:
    # - create_subprocess_shell() for proper command string interpretation
    # - CREATE_NEW_PROCESS_GROUP instead of start_new_session
    # Unix uses create_subprocess_exec() with start_new_session for process groups
    if sys.platform == "win32":
        # On Windows, use shell=True equivalent via create_subprocess_shell
        # This properly handles command strings with quotes and special chars
        # CREATE_NO_WINDOW (0x08000000): Prevents console window from appearing
        # CREATE_NEW_PROCESS_GROUP (0x200): Allows process tree termination
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x08000200,  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
            cwd=str(project_dir) if project_dir else None,
        )
    else:
        # Unix: use build_sandbox_command for sandbox-exec wrapper
        sandbox_cmd = build_sandbox_command(command, mode, project_dir)
        # Use start_new_session to create a new process group for clean termination
        process = await asyncio.create_subprocess_exec(
            *sandbox_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=str(project_dir) if project_dir else None,
        )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode(_SUBPROCESS_ENCODING, errors="replace")
        stderr = stderr.decode(_SUBPROCESS_ENCODING, errors="replace")
        truncated = False

        # Truncate output if needed
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
            "sandbox_mode": mode.value,
        }

    except asyncio.TimeoutError:
        # Kill the process tree
        await _kill_process_tree(process)
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds. You can specify a larger timeout (up to 300 seconds) using the timeout parameter.",
            "truncated": False,
            "timed_out": True,
            "sandbox_mode": mode.value,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "truncated": False,
            "timed_out": False,
            "sandbox_mode": mode.value,
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Error running command: {e}",
            "truncated": False,
            "timed_out": False,
            "sandbox_mode": mode.value,
        }


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a process and all its children."""
    if proc.returncode is not None:
        return

    try:
        if sys.platform == "win32":
            # Windows: use taskkill for process tree
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            # Unix: kill the process group
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        await proc.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass  # Process already dead
