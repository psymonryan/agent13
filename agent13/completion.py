"""@path completion: what to offer, and what a Tab press does with it.

Pure decision + a filesystem scan, with no Textual, no widget and no Agent.
They lived in ``ui/tui.py``, where the only way to test the decision was to
import the TUI and reach for a private method (``AgentTUI._filename_completion_scan``,
``AgentTUI._descend_target(_Stub(), ...)``) - a test that has to construct the
whole app to check a string decision.

What stays in the TUI: the keybindings, the info-pane rendering, and
``_descend_target``, which needs the TUI's own completion-context parser to
decide whether the cursor sits in an ``@filename`` token. This module owns the
part that has no such dependency.
"""

from __future__ import annotations

import os

# The info pane windows 8 at a time; a longer list is unreachable by cycling.
MAX_MATCHES = 50


def scan_level(dir_path: str, base: str, prefix: str = "") -> list[str]:
    """List a directory as ``@<base><entry>`` completions.

    Directories get a trailing ``/``. Hidden entries are skipped and the list is
    capped at ``MAX_MATCHES``.
    """
    try:
        entries = os.listdir(dir_path)
    except (PermissionError, OSError):
        return []

    matches = []
    for entry in sorted(entries):
        if entry.startswith("."):
            continue  # Skip hidden files
        if prefix and not entry.lower().startswith(prefix.lower()):
            continue

        # Build the completion preserving the user's input style
        completion = f"@{base}{entry}"

        # Add trailing / for directories
        if os.path.isdir(os.path.join(dir_path, entry)):
            completion += "/"

        matches.append(completion)

        if len(matches) >= MAX_MATCHES:
            break
    return matches


def filename_completion_scan(partial: str) -> tuple[list[str], list[str]]:
    """Scan for an ``@path`` partial.

    Returns:
        (matches, level) - ``matches`` are the entries matching the typed
        prefix (what Tab completes to); ``level`` is EVERY entry in the
        directory the completion was made in (what Tab cycles through
        afterwards, so a unique match cannot trap the user).
    """
    if not partial.startswith("@"):
        return [], []

    # Strip the leading @ to get the path
    path_part = partial[1:]

    # Split the typed path into (directory to scan, prefix to filter by,
    # text that precedes each entry in the completion).
    if not path_part:
        # Just "@" - list files in current directory
        dir_path, prefix, base_for_completion = ".", "", ""
    else:
        expanded = os.path.expanduser(path_part)
        if os.path.isdir(expanded):
            if not path_part.endswith("/"):
                # The typed path IS a directory, named in full. Complete
                # to the name with a trailing "/" - do NOT descend here.
                # Descending dropped the directory from the completion, so
                # "@docs_archive" completed to "@<first entry in it>" and
                # the name vanished from the path. The level to cycle is
                # the PARENT, where the completion was made.
                parent = os.path.dirname(path_part)
                level = scan_level(
                    os.path.expanduser(parent) if parent else ".",
                    f"{parent}/" if parent else "",
                )
                return [f"@{path_part}/"], level
            dir_path, prefix, base_for_completion = expanded, "", path_part
        else:
            dir_path = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)
            # Preserve the user's input style (~, relative, absolute)
            base_for_completion = os.path.dirname(path_part)
            if base_for_completion and not base_for_completion.endswith("/"):
                base_for_completion += "/"

    matches = scan_level(dir_path, base_for_completion, prefix)
    # A prefix-less scan IS the level - don't list the directory twice
    level = matches if not prefix else scan_level(dir_path, base_for_completion)
    return matches, level


def plan_completion(
    matches: list[str], level: list[str], partial: str
) -> tuple[list[str], int] | None:
    """Decide what a Tab press does with the matches it found.

    Returns (cycle, index) - the list Tab/Shift+Tab will walk and where to
    start - or None when Tab must do nothing because the typed text is
    already a complete entry: completing it could only REPLACE what the user
    typed in full, silently and destructively.

    ``level`` is every entry in the directory the completion was made in, so a
    unique match does not trap the user: Tab keeps moving through the
    siblings instead of descending into the directory.
    """
    if not matches:
        return None
    if len(matches) == 1 and matches[0] == partial:
        return None
    cycle = level or matches
    index = cycle.index(matches[0]) if matches[0] in cycle else 0
    return cycle, index
