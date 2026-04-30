"""History management for agent commands with project-scoped storage.

History is stored in dated files scoped to the current project (basename of cwd).
Navigation (up/down) operates on session items only - commands typed in this
agent instance. Persistence is handled via atomic append to the dated file.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional
from agent13.config_paths import get_history_path as _get_history_path


def _is_testing() -> bool:
    """Check if we're running under pytest by looking for marker file.

    This uses a marker file approach instead of environment variables because
    pexpect-spawned processes don't inherit env vars reliably.

    Returns:
        True if tests/.testing marker file exists.
    """
    # Look for marker file in tests directory (relative to this file)
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(agent_dir)
    marker_path = os.path.join(project_root, "tests", ".testing")
    return os.path.exists(marker_path)


def get_default_history_path() -> str:
    """Get the default history file path (backward compatibility).

    Returns:
        Path in format ~/.agent13/history-{basename-cwd}-{YYYY-MM-DD}
    """
    return get_history_path()


def get_history_path(project_name: str = None) -> str:
    """Get the history file path for a project.

    Args:
        project_name: Project identifier. If None, uses basename of cwd.
                      Falls back to "global" if no cwd available.

    Returns:
        Path in format ~/.agent13/history-{project}[-test]-{YYYY-MM-DD}
        The -test suffix is added when running under pytest.
    """
    # Add _test suffix when running under pytest
    suffix = "_test" if _is_testing() else ""
    return str(_get_history_path(project_name, suffix))


class History:
    """Command history with project-scoped storage and session-based navigation.

    On startup, session_items is seeded from file history so up/down navigation
    works immediately. New commands are added to both session_items and the file.
    The file path is computed fresh on each write, so date rollover works correctly.

    Attributes:
        project_name: Project identifier (basename of cwd at instantiation).
        session_items: List of (timestamp, command) for navigation (oldest-first).
        file_items: List of (timestamp, command) from file (newest-first for display).
        _index: Current navigation position in session_items.
        _prefix: The original prefix for prefix-based navigation (or None).
        _prefix_matches: Cached list of (index, command) tuples matching prefix.
        _prefix_match_idx: Current position in _prefix_matches.
    """

    def __init__(self, project_name: str = None, file_path: str = None):
        """Initialize history for a project.

        Args:
            project_name: Project identifier. If None, uses basename of cwd.
            file_path: Optional explicit file path to use instead of auto-generated.
                       Used for testing with specific files.
        """
        self._file_path = file_path

        if file_path:
            # If explicit file path provided, derive project name from it
            self.project_name = os.path.splitext(os.path.basename(file_path))[0]
        elif project_name is None:
            cwd = os.getcwd()
            if cwd:
                self.project_name = os.path.basename(cwd)
            else:
                self.project_name = "global"
        else:
            self.project_name = project_name

        self.session_items: list[tuple[datetime, str]] = []
        self.file_items: list[tuple[datetime, str]] = []
        self._index: int = 0
        # Prefix-based navigation state
        self._prefix: Optional[str] = None
        self._prefix_matches: list[tuple[int, str]] = []
        self._prefix_match_idx: int = 0
        self._load_file()

    def _get_path(self) -> str:
        """Get the current history file path (computed fresh for date rollover).

        When running under pytest (detected via marker file), adds _test suffix
        to prevent polluting user's real history file.
        """
        if self._file_path:
            return self._file_path

        # Add _test suffix when running under pytest
        suffix = "_test" if _is_testing() else ""

        return str(_get_history_path(self.project_name, suffix))

    def _get_path_for_date(self, date: datetime) -> str:
        """Get the history file path for a specific date.

        Uses _get_path() to get the base path format, then replaces the date.

        Args:
            date: The date to get the path for.

        Returns:
            Path in format ~/.agent13/history-{project}[-test]-{YYYY-MM-DD}
        """
        base_path = self._get_path()
        # Replace the date portion (last 10 chars before extension if present,
        # or just the last 10 chars if they match YYYY-MM-DD format)
        date_str = date.strftime("%Y-%m-%d")
        # The base path ends with -YYYY-MM-DD, find and replace that pattern
        return re.sub(r"-\d{4}-\d{2}-\d{2}$", f"-{date_str}", base_path)

    def _load_file(self) -> None:
        """Load history from the last 7 days into file_items."""
        # If an explicit file path is set, only load from that single file
        if self._file_path:
            self._load_single_file(self._file_path)
            # Reverse to get newest-first
            self.file_items.reverse()
            # Seed session_items for navigation
            self.session_items = list(reversed(self.file_items))
            self._index = len(self.session_items)
            return

        today = datetime.now()

        # Load from oldest to newest day (6 days ago to today)
        # This way items are added in chronological order
        for days_ago in range(6, -1, -1):
            date = today - timedelta(days=days_ago)
            path = self._get_path_for_date(date)
            if not os.path.exists(path):
                continue
            self._load_single_file(path)

        # Reverse to get newest-first (file_items is now oldest-first)
        self.file_items.reverse()

        # Seed session_items with file history so up/down navigation works immediately
        # Store in oldest-first order for navigation (reverse of file_items)
        self.session_items = list(reversed(self.file_items))
        self._index = len(self.session_items)

    def _load_single_file(self, path: str) -> None:
        """Load history from a single file into file_items.

        Args:
            path: Path to the history file to load.

        File format:
            # YYYY-MM-DD HH:MM:SS
            command text (may span multiple lines)
            # YYYY-MM-DD HH:MM:SS
            another command

        Multi-line commands are stored as-is between timestamp lines.
        """
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                content = f.read()
            # Split on timestamp lines: # YYYY-MM-DD HH:MM:SS[.microseconds]
            # Pattern captures timestamp in group 1, handles both our format
            # and prompt_toolkit format (with microseconds)
            pattern = r"^# (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)$"
            parts = re.split(pattern, content, flags=re.MULTILINE)

            # parts[0] is content before first timestamp (usually empty)
            # Then alternating: timestamp, command, timestamp, command, ...
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    timestamp_str = parts[i]
                    command = parts[i + 1].strip("\n")
                    if command:  # skip empty commands
                        try:
                            # Try with microseconds first (prompt_toolkit format)
                            timestamp = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                            )
                        except ValueError:
                            try:
                                # Fall back to our format
                                timestamp = datetime.strptime(
                                    timestamp_str, "%Y-%m-%d %H:%M:%S"
                                )
                            except ValueError:
                                timestamp = datetime.now()
                        self.file_items.append((timestamp, command))

        except (IOError, OSError):
            pass

    def _append_to_file(self, command: str, timestamp: datetime) -> None:
        """Atomically append a command to the history file.

        Args:
            command: The command string to append.
            timestamp: Timestamp for the command.
        """
        path = self._get_path()
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Pre-create the file to avoid creation race on Windows.
        # On all platforms, appends under 1024 bytes are atomic,
        # so no file locking is needed for our ~40 byte writes.
        if not os.path.exists(path):
            open(path, "w").close()

        try:
            with open(path, "a") as f:
                f.write(f"# {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{command}\n")
        except (IOError, OSError):
            pass

    def add(self, command: str) -> None:
        """Add a command to history.

        Adds to session items (for navigation) and appends to file (for persistence).
        Slash commands (starting with '/') are blacklisted and not added to history.

        Args:
            command: The command string to add.
        """
        if not command or not command.strip():
            return

        # Blacklist slash commands - they shouldn't be in history
        if command.strip().startswith("/"):
            return

        # Don't add consecutive duplicates to session
        if self.session_items and self.session_items[-1][1] == command:
            return

        timestamp = datetime.now()
        self.session_items.append((timestamp, command))
        self._index = len(self.session_items)  # Position at end (after newest)
        self._append_to_file(command, timestamp)

        # Also add to file_items for /history display
        self.file_items.insert(0, (timestamp, command))

    def up(self) -> Optional[str]:
        """Navigate up in session history (towards older commands).

        Returns:
            The command at the new position, or oldest command if at end.
        """
        if not self.session_items:
            return None

        # Move towards older (lower index)
        if self._index > 0:
            self._index -= 1
            return self.session_items[self._index][1]
        else:
            # At oldest, return it again
            return self.session_items[0][1]

    def down(self) -> Optional[str]:
        """Navigate down in session history (towards newer commands).

        Returns:
            The command at the new position, or None if at the newest.
        """
        if not self.session_items:
            return None

        # Move towards newer (higher index)
        if self._index < len(self.session_items) - 1:
            self._index += 1
            return self.session_items[self._index][1]
        else:
            # At newest, return None to clear input
            self._index = len(self.session_items)
            return None

    def reset(self) -> None:
        """Reset navigation position to after the newest item."""
        self._index = len(self.session_items)
        # Also reset prefix navigation state
        self._prefix = None
        self._prefix_matches = []
        self._prefix_match_idx = 0

    def start_prefix_navigation(self, prefix: str) -> None:
        """Start prefix-based navigation mode.

        This is called when the user presses Ctrl+B to search history
        with the current input as a prefix filter.

        Args:
            prefix: The prefix to match (the current input buffer content).
                   Empty string matches all history items.
        """
        self._prefix = prefix
        self._prefix_matches = []
        self._prefix_match_idx = -1  # Will be incremented on first up()

        # Find all history items that start with this prefix (newest to oldest)
        # session_items is oldest-first, so iterate in reverse
        # Empty prefix matches everything
        # Deduplicate - skip commands we've already seen
        seen = set()
        for i in range(len(self.session_items) - 1, -1, -1):
            cmd = self.session_items[i][1]
            if cmd.startswith(prefix) and cmd not in seen:
                seen.add(cmd)
                self._prefix_matches.append((i, cmd))

    def up_with_prefix(self) -> Optional[str]:
        """Navigate up in history, filtering by the saved prefix.

        Returns:
            The next matching command (older), or None if no more matches.
        """
        if not self._prefix_matches:
            return None

        # Move to next older match
        if self._prefix_match_idx < len(self._prefix_matches) - 1:
            self._prefix_match_idx += 1
            idx, cmd = self._prefix_matches[self._prefix_match_idx]
            self._index = idx
            return cmd
        else:
            # At oldest match, return it again
            idx, cmd = self._prefix_matches[-1]
            return cmd

    def down_with_prefix(self) -> Optional[str]:
        """Navigate down in history, filtering by the saved prefix.

        Returns:
            The next matching command (newer), or None to clear input.
        """
        if not self._prefix_matches:
            return None

        # Move to newer match
        if self._prefix_match_idx > 0:
            self._prefix_match_idx -= 1
            idx, cmd = self._prefix_matches[self._prefix_match_idx]
            self._index = idx
            return cmd
        else:
            # At newest match or past it, return None to show prefix
            self._prefix_match_idx = -1
            return None

    def in_prefix_mode(self) -> bool:
        """Check if we're in prefix navigation mode."""
        return self._prefix is not None

    def get_prefix(self) -> Optional[str]:
        """Get the current prefix for prefix navigation mode."""
        return self._prefix

    def get_all(self) -> list[str]:
        """Get all commands from file history (newest first).

        Returns:
            List of command strings from file.
        """
        return [item[1] for item in self.file_items]

    def get_session(self) -> list[str]:
        """Get all commands from this session (oldest first for display).

        Returns:
            List of command strings from this session.
        """
        return [item[1] for item in self.session_items]

    def get_with_timestamps(self) -> list[tuple[datetime, str]]:
        """Get all commands from file with timestamps (newest first).

        Returns:
            List of (timestamp, command) tuples.
        """
        return self.file_items.copy()

    def clear_session(self) -> None:
        """Clear session history only (not file)."""
        self.session_items = []
        self._index = 0

    def clear(self) -> None:
        """Clear all history (session and file)."""
        self.session_items = []
        self.file_items = []
        self._index = 0
        # Clear the file
        path = self._get_path()
        try:
            with open(path, "w"):
                pass
        except (IOError, OSError):
            pass

    def __len__(self) -> int:
        """Return number of items in session history."""
        return len(self.session_items)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"History(project={self.project_name!r}, "
            f"session={len(self.session_items)}, file={len(self.file_items)})"
        )
