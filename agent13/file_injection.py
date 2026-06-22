"""File injection: expand @filename tokens and build --read messages."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MAX_EMBED_BYTES = 256 * 1024  # 256 KB — same as mistral-vibe


def expand_file_mentions(text: str, base_dir: str | Path | None = None) -> str:
    """Expand @path tokens in text, inlining file contents.

    Scans for @path tokens where @ is preceded by start-of-string or a
    non-alphanumeric, non-underscore character (so email@addresses are safe).
    Each @path that resolves to an existing file is replaced with:

        <file path="path">
        {contents}
        </file>

    Files larger than DEFAULT_MAX_EMBED_BYTES or that are binary are left
    as-is (the @path token stays literal so the agent can use the read_file
    tool on it). Directories are left as-is too.

    Args:
        text: The user's message text potentially containing @path tokens.
        base_dir: Base directory for resolving relative @paths. Defaults
            to cwd.

    Returns:
        The text with @path tokens expanded where the file exists and is
        embeddable.
    """
    if not text or "@" not in text:
        return text

    resolved_base = Path(base_dir) if base_dir else Path.cwd()

    result = []
    pos = 0
    while pos < len(text):
        at_pos = text.find("@", pos)
        if at_pos == -1:
            result.append(text[pos:])
            break

        # Append text before the @
        result.append(text[pos:at_pos])

        # Check if this @ is a valid anchor (start of string or preceded
        # by non-alphanumeric, non-underscore)
        if at_pos > 0 and (text[at_pos - 1].isalnum() or text[at_pos - 1] == "_"):
            # Not a file anchor — literal @
            result.append("@")
            pos = at_pos + 1
            continue

        # Extract the candidate path after @
        candidate, new_pos = _extract_path_candidate(text, at_pos + 1)
        if not candidate:
            result.append("@")
            pos = at_pos + 1
            continue

        # Try to resolve and read the file
        expanded = _try_expand(candidate, resolved_base)
        if expanded is not None:
            result.append(expanded)
        else:
            # Could not expand — leave the @token literal
            result.append("@" + candidate)

        pos = new_pos

    return "".join(result)


def _extract_path_candidate(text: str, start: int) -> tuple[str | None, int]:
    """Extract a path candidate starting at position `start` (after the @).

    Supports quoted paths (@'path with spaces' or @"path with spaces") and
    bare paths (alphanumeric, ._/-()[]{} and backslash).
    """
    if start >= len(text):
        return None, start

    # Quoted path
    if text[start] in ("'", '"'):
        quote = text[start]
        end_quote = text.find(quote, start + 1)
        if end_quote == -1:
            return None, start
        return text[start + 1 : end_quote], end_quote + 1

    # Bare path
    end = start
    while end < len(text) and _is_path_char(text[end]):
        end += 1

    if end == start:
        return None, start

    return text[start:end], end


def _is_path_char(ch: str) -> bool:
    """Characters allowed in a bare @path token."""
    return ch.isalnum() or ch in "._/\\-()[]{}~:"


def _try_expand(candidate: str, base_dir: Path) -> str | None:
    """Try to read candidate as a file relative to base_dir.

    Returns the expanded <file path="..."> block, or None if the candidate
    is not an embeddable file (doesn't exist, is a directory, is binary,
    or exceeds the size cap).
    """
    # Expand ~ for home directory paths
    expanded = os.path.expanduser(candidate)

    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path

    try:
        path = path.resolve()
    except (OSError, RuntimeError):
        return None

    if not path.exists() or not path.is_file():
        return None

    try:
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) > DEFAULT_MAX_EMBED_BYTES:
        return None

    if not _is_probably_text(data):
        return None

    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    return f'<file path="{candidate}">\n{content}\n</file>'


def _is_probably_text(data: bytes) -> bool:
    """Heuristic: reject data with null bytes or too many non-printables."""
    if not data:
        return True
    if b"\x00" in data:
        return False
    non_text = sum(1 for b in data if b <= 31 and b not in (9, 10, 11, 12, 13) or b == 127)
    return (non_text / len(data)) < 0.1


def build_read_message(
    file_paths: list[str], base_dir: str | Path | None = None
) -> str:
    """Build the --read injection message with acknowledgement instruction.

    Reads each file, wraps it in <file path="..."> tags, and appends the
    acknowledgement instruction. Files that don't exist or can't be read
    are included as an error note so the user sees which files failed.

    Args:
        file_paths: List of file paths to read (from --read CLI args).
        base_dir: Base directory for relative paths. Defaults to cwd.

    Returns:
        A string suitable for sending as the first user message.
    """
    resolved_base = Path(base_dir) if base_dir else Path.cwd()
    parts = ["The following files have been read into context for this session."]

    for path_str in file_paths:
        expanded = os.path.expanduser(path_str)
        path = Path(expanded)
        if not path.is_absolute():
            path = resolved_base / path

        try:
            path = path.resolve()
        except (OSError, RuntimeError):
            parts.append(
                f'\n<file_error path="{path_str}">Could not resolve path</file_error>'
            )
            continue

        if not path.exists():
            parts.append(f'\n<file_error path="{path_str}">File not found</file_error>')
            continue

        if not path.is_file():
            parts.append(
                f'\n<file_error path="{path_str}">Not a file (directory?)</file_error>'
            )
            continue

        try:
            data = path.read_bytes()
        except OSError as e:
            parts.append(
                f'\n<file_error path="{path_str}">Read error: {e}</file_error>'
            )
            continue

        if len(data) > DEFAULT_MAX_EMBED_BYTES:
            parts.append(
                f'\n<file_error path="{path_str}">File too large '
                f"({len(data)} bytes > {DEFAULT_MAX_EMBED_BYTES} bytes limit)</file_error>"
            )
            continue

        if not _is_probably_text(data):
            parts.append(
                f'\n<file_error path="{path_str}">Binary file, not embedded</file_error>'
            )
            continue

        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            parts.append(
                f'\n<file_error path="{path_str}">Not valid UTF-8</file_error>'
            )
            continue

        parts.append(f'\n<file path="{path_str}">\n{content}\n</file>')

    parts.append('\nReply with "files read" and nothing else.')
    return "\n".join(parts)
