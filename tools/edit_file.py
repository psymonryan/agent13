"""Edit file tool - line-based editing for AI agents.

Every edit result includes a snapshot_id for potential rollback.
Line numbers are 1-indexed (first line is line 1).
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from tools import tool
from tools.security import validate_path_for_write, get_current_sandbox_mode

# =============================================================================
# Snapshot storage — per-session undo for edit_file
# =============================================================================
# Snapshots are keyed by integer ID, stable across FIFO eviction.
# The AI discovers snapshot_id from edit results and uses it with mode="rollback".
_snapshots: dict[str, dict[int, str]] = {}  # filepath -> {id: content}
_snapshot_counter: dict[str, int] = {}  # filepath -> next available ID
MAX_SNAPSHOTS_PER_FILE = 10


# =============================================================================
# Helpers
# =============================================================================


def _is_binary_file(content: bytes) -> bool:
    """Check if content appears to be binary (non-text).

    Simple check: look for null bytes.
    """
    return b"\x00" in content


def _do_replace(
    lines: list[str],
    find: str,
    content: str,
    start_line: int = None,
    end_line: int = None,
    match_target_indentation: bool = True,
    replace_all: bool = False,
) -> tuple[bool, list[str], str, int]:
    """Find and replace text within a line range.

    Supports multi-line patterns - the find string can span multiple lines.

    When match_target_indentation=True (default), the function:
    - Matches code structure by normalizing indentation in the find pattern
    - Applies correct indentation from the target location to the replacement

    Args:
        lines: File content as list of lines
        find: Text to find (can contain newlines for multi-line patterns)
        content: Text to replace with
        start_line: Start line (0-indexed, inclusive), None for global
        end_line: End line (0-indexed, inclusive), None for global
        match_target_indentation: If True, normalize indentation when matching and replacing
        replace_all: If True, replace all occurrences; if False, error on multiple matches

    Returns:
        Tuple of (success, modified_lines, error_message, num_replacements)
    """
    # Clean whitespace from content text (strip trailing, convert whitespace-only lines to empty)
    content_lines = content.splitlines()
    content_lines = _clean_whitespace_lines(content_lines)
    content = "\n".join(content_lines)

    search_start = start_line if start_line is not None else 0
    search_end = end_line if end_line is not None else len(lines) - 1

    # Extract the search region as a single string (supports multi-line patterns)
    search_lines = lines[search_start : search_end + 1]
    search_content = "\n".join(search_lines)

    if match_target_indentation and "\n" in find:
        # Multi-line pattern with match_target_indentation: normalize indentation for matching
        find_lines = find.splitlines()
        find_normalized, replace_normalized = _normalize_pattern_indentation(
            find_lines, content, search_lines
        )
        search_normalized = "\n".join(_strip_line_indents(search_lines))

        # Find match in normalized content
        match_pos = _find_normalized_match(
            search_normalized, find_normalized, search_lines
        )

        if match_pos is None:
            # No exact match - try fuzzy matching for better error message
            fuzzy_match = _find_fuzzy_match(
                lines, find, search_start, search_end, FUZZY_THRESHOLD
            )
            error_msg = _format_fuzzy_error(
                find,
                lines,
                fuzzy_match,
                start_line=search_start + 1 if start_line is not None else None,
                end_line=search_end + 1 if end_line is not None else None,
            )
            return False, lines, error_msg, 0

        # Apply target indentation to replacement
        target_indent = _detect_indentation(search_lines, match_pos["line_idx"])
        replace_indented = _apply_indentation(replace_normalized, target_indent)

        # Reconstruct with proper line breaks
        match_start_line = search_start + match_pos["line_idx"]
        match_end_line = search_start + match_pos["line_idx"] + len(find_lines) - 1

        result = (
            lines[:match_start_line]
            + replace_indented.splitlines()
            + lines[match_end_line + 1 :]
        )
        return True, result, "", 1  # 1 replacement made

    else:
        # Exact matching (original behavior or single-line patterns)
        # Find all matches
        matches = []
        pos = 0
        while True:
            idx = search_content.find(find, pos)
            if idx == -1:
                break
            matches.append(idx)
            pos = idx + 1

        if len(matches) == 0:
            # No exact match - try fuzzy matching for better error message
            fuzzy_match = _find_fuzzy_match(
                lines, find, search_start, search_end, FUZZY_THRESHOLD
            )
            error_msg = _format_fuzzy_error(
                find,
                lines,
                fuzzy_match,
                start_line=search_start + 1 if start_line is not None else None,
                end_line=search_end + 1 if end_line is not None else None,
            )
            return False, lines, error_msg, 0

        # Handle multiple matches based on replace_all flag
        if len(matches) > 1 and not replace_all:
            # Calculate line numbers for each match position
            match_info = []
            for match_pos in matches[:5]:
                # Count newlines before this position to get line number
                lines_before = search_content[:match_pos].count("\n")
                line_num = search_start + lines_before + 1
                # Get a snippet of the line where match starts
                line_start = search_content.rfind("\n", 0, match_pos) + 1
                line_end = search_content.find("\n", match_pos)
                if line_end == -1:
                    line_end = len(search_content)
                snippet = search_content[line_start : min(line_end, line_start + 50)]
                match_info.append(f"  Line {line_num}: {snippet}...")
            if len(matches) > 5:
                match_info.append(f"  ... and {len(matches) - 5} more matches")
            return (
                False,
                lines,
                f"Multiple matches found ({len(matches)} total). Use replace_all=True to replace all, or be more specific:\n"
                + "\n".join(match_info),
                0,
            )

        # Replace all matches (either replace_all=True or single match)
        # Work backwards to preserve positions
        result_content = search_content
        for idx in reversed(matches):
            result_content = (
                result_content[:idx] + content + result_content[idx + len(find) :]
            )

        new_lines = result_content.splitlines()

        # Reconstruct full file
        result = lines[:search_start] + new_lines + lines[search_end + 1 :]
        return True, result, "", len(matches)  # Return count of replacements made


def _detect_indentation(lines: list[str], start_line: int) -> str:
    """Detect the indentation of a line.

    Args:
        lines: File content as list of lines
        start_line: Line to check (0-indexed)

    Returns:
        The indentation string (spaces/tabs)
    """
    if start_line < 0 or start_line >= len(lines):
        return ""

    line = lines[start_line]
    # Get leading whitespace
    stripped = line.lstrip()
    if not stripped:
        return ""

    indent_len = len(line) - len(stripped)
    return line[:indent_len]


def _normalize_indentation(content_lines: list[str], target_indent: str) -> list[str]:
    """Normalize indentation of content to match target indentation.

    Args:
        content_lines: Lines of content to normalize
        target_indent: Target indentation string

    Returns:
        Content lines with normalized indentation
    """
    if not content_lines or not target_indent:
        return content_lines

    # Detect the indentation of the first non-empty line in content
    first_indent = ""
    for line in content_lines:
        if line.strip():
            stripped = line.lstrip()
            if stripped:
                indent_len = len(line) - len(stripped)
                first_indent = line[:indent_len]
                break

    # If content has no indentation, add target_indent to all non-empty lines
    if not first_indent:
        result = []
        for line in content_lines:
            if line.strip():
                result.append(target_indent + line.lstrip())
            else:
                result.append(line)
        return result

    # If content already has correct indentation, return as-is
    if first_indent == target_indent:
        return content_lines

    # Normalize: remove first_indent from each line and add target_indent
    result = []
    for line in content_lines:
        if line.startswith(first_indent):
            result.append(target_indent + line[len(first_indent) :])
        elif line.strip():
            # Line doesn't have the expected indent, just add target indent
            result.append(target_indent + line.lstrip())
        else:
            # Empty line
            result.append(line)

    return result


def _strip_line_indents(lines: list[str]) -> list[str]:
    """Strip leading whitespace from each line, preserving relative structure.

    Args:
        lines: List of lines

    Returns:
        Lines with leading whitespace stripped from each
    """
    return [line.lstrip() for line in lines]


def _normalize_pattern_indentation(
    find_lines: list[str], content: str, search_lines: list[str]
) -> tuple[str, str]:
    """Normalize indentation in find pattern and prepare replacement.

    This function:
    1. Strips leading whitespace from find pattern lines (normalizing indentation)
    2. Returns the normalized find pattern and replacement ready for matching

    Args:
        find_lines: Lines of the find pattern
        content: Replacement text
        search_lines: Search region lines (for context)

    Returns:
        Tuple of (normalized_find_string, normalized_replace_string)
    """
    # Normalize find pattern by stripping each line
    find_normalized = "\n".join(line.lstrip() for line in find_lines)

    # For replacement, we'll apply indentation later based on match location
    # Just return it as-is for now (splitlines preserves the content)
    replace_normalized = content

    return find_normalized, replace_normalized


def _find_normalized_match(
    search_normalized: str, find_normalized: str, search_lines: list[str]
) -> dict | None:
    """Find a match in normalized content, returning match info.

    Args:
        search_normalized: Search content with stripped indentation
        find_normalized: Find pattern with stripped indentation
        search_lines: Original search lines (for line number calculation)

    Returns:
        Dict with 'line_idx' (0-indexed line in search_lines) or None if not found
    """
    idx = search_normalized.find(find_normalized)
    if idx == -1:
        return None

    # Calculate which line the match starts on
    lines_before = search_normalized[:idx].count("\n")

    return {"line_idx": lines_before}


def _clean_whitespace_lines(lines: list[str]) -> list[str]:
    """
    Clean whitespace issues from lines:
    - Strip trailing whitespace from all lines
    - Convert whitespace-only lines to empty lines

    Args:
        lines: List of lines to clean

    Returns:
        List of cleaned lines
    """
    cleaned = []
    for line in lines:
        # Strip trailing whitespace (preserve leading)
        line = line.rstrip()
        # If line is now empty (was whitespace-only), keep it as empty
        cleaned.append(line)
    return cleaned


def _find_matches(
    lines: list[str], find: str, start_line: int = None, end_line: int = None
) -> tuple[list[tuple[int, int]], str | None]:
    """
    Find all matches of text in file content, supporting multi-line patterns.

    This function searches for the find text within the specified line range,
    supporting both single-line and multi-line patterns. It returns the line
    numbers (0-indexed) where matches start and end.

    When no exact match is found, fuzzy matching is attempted to provide
    helpful suggestions.

    Args:
        lines: File content as list of lines (0-indexed)
        find: Text to find (can contain newlines for multi-line patterns)
        start_line: Start line (0-indexed, inclusive), None for start of file
        end_line: End line (0-indexed, inclusive), None for end of file

    Returns:
        Tuple of (matches, error_message):
        - matches: List of (start_line_idx, end_line_idx) tuples (0-indexed, inclusive)
        - error_message: None if at least one match found, otherwise an error message
    """
    search_start = start_line if start_line is not None else 0
    search_end = end_line if end_line is not None else len(lines) - 1

    # Extract the search region as a single string (supports multi-line patterns)
    search_lines = lines[search_start : search_end + 1]
    search_content = "\n".join(search_lines)

    # Find all matches
    matches = []
    pos = 0
    while True:
        idx = search_content.find(find, pos)
        if idx == -1:
            break
        matches.append(idx)
        pos = idx + 1

    if len(matches) == 0:
        # No exact match found - try fuzzy matching for better error message
        fuzzy_match = _find_fuzzy_match(
            lines, find, search_start, search_end, FUZZY_THRESHOLD
        )

        # Format error with fuzzy suggestion
        error_msg = _format_fuzzy_error(
            find,
            lines,
            fuzzy_match,
            start_line=search_start + 1 if start_line is not None else None,
            end_line=search_end + 1 if end_line is not None else None,
        )
        return [], error_msg

    # Convert character positions to line numbers
    result = []
    find_line_count = find.count("\n")  # Number of newlines in find pattern

    for match_pos in matches:
        # Count newlines before this position to get start line
        lines_before = search_content[:match_pos].count("\n")
        start_line_idx = search_start + lines_before
        # End line is start line + number of lines in the pattern
        end_line_idx = start_line_idx + find_line_count
        result.append((start_line_idx, end_line_idx))

    return result, None


def _apply_indentation(content: str, indent: str) -> str:
    """Apply indentation to each line of content, preserving relative indentation.

    This function preserves the relative indentation structure of multi-line content.
    For example, if the content has lines at indent levels 0, 4, and 8 spaces,
    and the target indent is 4 spaces, the result will have lines at 4, 8, and 12 spaces.

    Args:
        content: Content to indent (can be multi-line)
        indent: Base indentation string to apply to the first non-empty line

    Returns:
        Content with indentation applied, preserving relative offsets between lines
    """
    lines = content.splitlines()
    if not lines:
        return content

    # Detect base indentation from first non-empty line
    base_indent_len = 0
    for line in lines:
        if line.strip():
            base_indent_len = len(line) - len(line.lstrip())
            break

    # Apply indentation preserving relative offsets
    result = []
    for line in lines:
        if line.strip():
            # Calculate this line's indent length
            line_indent_len = len(line) - len(line.lstrip())
            # Calculate relative offset (how much more/less than base)
            relative_offset = line_indent_len - base_indent_len
            # Apply target base + relative offset
            if relative_offset > 0:
                result.append(indent + (" " * relative_offset) + line.lstrip())
            else:
                result.append(indent + line.lstrip())
        else:
            result.append(line)

    return "\n".join(result)


# =============================================================================
# Fuzzy Matching
# =============================================================================

FUZZY_THRESHOLD = 0.9  # Minimum similarity ratio to consider a fuzzy match


@dataclass
class FuzzyMatch:
    """Represents a fuzzy match result."""

    line_start: int  # 0-indexed line where match starts
    line_end: int  # 0-indexed line where match ends
    similarity: float  # Similarity ratio (0.0 to 1.0)
    matched_text: str  # The actual text that was matched


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison.

    - Strips leading/trailing whitespace from each line
    - Normalizes multiple spaces to single space
    - Preserves line structure

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    lines = text.splitlines()
    normalized_lines = []
    for line in lines:
        # Strip leading/trailing whitespace, normalize internal whitespace
        stripped = line.strip()
        # Normalize multiple spaces to single space (but preserve structure)
        import re

        stripped = re.sub(r" +", " ", stripped)
        normalized_lines.append(stripped)
    return "\n".join(normalized_lines)


def _find_fuzzy_match(
    content_lines: list[str],
    search_text: str,
    start_line: int = 0,
    end_line: int = None,
    threshold: float = FUZZY_THRESHOLD,
) -> Optional[FuzzyMatch]:
    """Find the best fuzzy match for search_text in content.

    Uses anchor-based search for efficiency:
    1. Find lines matching first/last non-empty lines of search_text
    2. Check windows of same size as search text around those anchors
    3. Return best match above threshold

    Args:
        content_lines: File content as list of lines (0-indexed)
        search_text: Text to search for
        start_line: Start line for search (0-indexed, inclusive)
        end_line: End line for search (0-indexed, inclusive), None for end
        threshold: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        FuzzyMatch if found above threshold, None otherwise
    """
    if end_line is None:
        end_line = len(content_lines) - 1

    search_lines = search_text.splitlines()
    if not search_lines:
        return None

    # Normalize search text for comparison
    search_normalized = _normalize_text(search_text)
    search_line_count = len(search_lines)

    # Find anchor lines (first and last non-empty lines of search text)
    first_anchor = None
    last_anchor = None
    for line in search_lines:
        if line.strip():
            if first_anchor is None:
                first_anchor = line.strip()
            last_anchor = line.strip()

    if first_anchor is None:
        # Search text is all whitespace
        return None

    # Find candidate anchor positions in content
    candidates = []
    for i in range(start_line, min(end_line + 1, len(content_lines))):
        line_stripped = content_lines[i].strip()
        # Check if this line matches either anchor (fuzzy match on single line)
        if (
            first_anchor
            and SequenceMatcher(None, line_stripped, first_anchor).ratio() > 0.7
        ):
            candidates.append(i)
        elif (
            last_anchor
            and SequenceMatcher(None, line_stripped, last_anchor).ratio() > 0.7
        ):
            candidates.append(i)

    # If no anchor candidates, slide window through first 100 lines
    if not candidates:
        check_end = min(
            start_line + 100, end_line + 1, len(content_lines) - search_line_count + 1
        )
        for i in range(start_line, max(start_line, check_end)):
            candidates.append(i)

    # Check each candidate window
    best_match = None
    best_similarity = 0.0

    for anchor_line in candidates:
        # Try windows starting at different positions around the anchor
        for offset in range(-2, 3):  # Try -2, -1, 0, +1, +2
            window_start = anchor_line + offset
            window_end = window_start + search_line_count - 1

            # Validate window bounds
            if window_start < start_line or window_end > end_line:
                continue
            if window_start < 0 or window_end >= len(content_lines):
                continue

            # Extract window content (exact same size as search)
            window_lines = content_lines[window_start : window_end + 1]
            window_text = "\n".join(window_lines)
            window_normalized = _normalize_text(window_text)

            # Calculate similarity
            similarity = SequenceMatcher(
                None, window_normalized, search_normalized
            ).ratio()

            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = FuzzyMatch(
                    line_start=window_start,
                    line_end=window_end,
                    similarity=similarity,
                    matched_text=window_text,
                )

    return best_match


def _format_fuzzy_error(
    search_text: str,
    content_lines: list[str],
    fuzzy_match: Optional[FuzzyMatch],
    start_line: int = None,
    end_line: int = None,
) -> str:
    """Format an error message with fuzzy match suggestion.

    Args:
        search_text: The text that was searched for
        content_lines: Full file content as list of lines
        fuzzy_match: The fuzzy match found (or None)
        start_line: Start line of search scope (1-indexed, for display)
        end_line: End line of search scope (1-indexed, for display)

    Returns:
        Formatted error message
    """
    scope_desc = f" in lines {start_line}-{end_line}" if start_line is not None else ""

    lines = []
    lines.append(f"Text not found{scope_desc}.")
    lines.append("")

    if fuzzy_match:
        lines.append(f"Closest fuzzy match ({fuzzy_match.similarity:.1%} similarity):")
        lines.append(
            f"  Lines {fuzzy_match.line_start + 1}-{fuzzy_match.line_end + 1}:"
        )
        lines.append("")

        # Show a diff-like comparison
        search_lines = search_text.splitlines()
        matched_lines = fuzzy_match.matched_text.splitlines()

        lines.append("  --- SEARCH (what you provided)")
        lines.append("  +++ FOUND (closest match in file)")
        lines.append("")

        # Show first 10 lines of each
        max_show = 10
        for i, (search_line, matched_line) in enumerate(
            zip(search_lines[:max_show], matched_lines[:max_show])
        ):
            search_display = (
                search_line[:60] + "..." if len(search_line) > 60 else search_line
            )
            matched_display = (
                matched_line[:60] + "..." if len(matched_line) > 60 else matched_line
            )
            if search_line.rstrip() != matched_line.rstrip():
                lines.append(f"  - {search_display}")
                lines.append(f"  + {matched_display}")
            else:
                lines.append(f"    {search_display}")

        if len(search_lines) > max_show or len(matched_lines) > max_show:
            lines.append(
                f"  ... ({max(len(search_lines), len(matched_lines)) - max_show} more lines)"
            )

        lines.append("")
        lines.append("Debugging tips:")
        lines.append("  1. Check for exact whitespace/indentation match")
        lines.append("  2. Verify line endings match (\\r\\n vs \\n)")
        lines.append("  3. The file may have been modified since you last read it")
    else:
        lines.append("No similar text found in the search region.")
        lines.append("")
        lines.append("Debugging tips:")
        lines.append("  1. Verify the text exists in the file")
        lines.append("  2. Check if the file has been modified")
        lines.append("  3. Try reading the file again to see current content")

    return "\n".join(lines)


# =============================================================================
# Indentation Auto-Correction
# =============================================================================


def _count_leading_spaces(line: str) -> int:
    """Count the number of leading spaces in a line."""
    import re

    match = re.match(r"^( *)", line)
    return len(match.group(1)) if match else 0


def _infer_base_indent(lines: list[str], start_line_0: int) -> int:
    """Infer the correct base indentation for a replace_range edit in a Python file.

    Uses two rules:
    1. If the previous line ends with ':', the block body should be indented
       one level deeper (prev_indent + 4).
    2. Otherwise, use the indentation of the line being replaced (maintain level).

    Args:
        lines: All lines in the file (0-indexed)
        start_line_0: The 0-indexed start line of the replacement region

    Returns:
        Number of spaces for the base indentation
    """
    # Rule 1: previous line opens a block → go deeper
    if start_line_0 > 0:
        prev_line = lines[start_line_0 - 1]
        prev_stripped = prev_line.strip()
        prev_indent = _count_leading_spaces(prev_line)
        if prev_stripped.endswith(":"):
            return prev_indent + 4

    # Rule 2: maintain the indent of the line(s) being replaced
    if start_line_0 < len(lines):
        orig_line = lines[start_line_0]
        return _count_leading_spaces(orig_line)

    # Rule 3: past end of file (append at end) — match last line's indent
    if start_line_0 == len(lines) and len(lines) > 0:
        last_line = lines[-1]
        last_stripped = last_line.strip()
        last_indent = _count_leading_spaces(last_line)
        if last_stripped.endswith(":"):
            return last_indent + 4
        return last_indent

    # Fallback: top of file, no previous context
    return 0


def _auto_correct_first_line_indent(
    new_lines: list[str], lines: list[str], insert_point_0: int, file_extension: str
) -> tuple[list[str], Optional[int]]:
    """
    Auto-correct the first line's indentation for Python files.

    LLMs frequently lose indent tokens on the first line, while remaining lines are
    usually correct. This function corrects the first line upward to match the natural
    AST indent level inferred from surrounding code. It never removes spaces — if the
    model sends more indent than expected, we trust it and let the AST gate decide.

    Args:
        new_lines: The lines of content to be inserted
        lines: The original file lines (0-indexed)
        insert_point_0: The 0-indexed line where content will be inserted
        file_extension: File extension (e.g., ".py")

    Returns:
        tuple: (corrected_new_lines, auto_indent_value_or_None)
    """
    # Only apply to Python files
    if file_extension != ".py":
        return new_lines, None

    if not new_lines or not new_lines[0].strip():
        return new_lines, None

    # Infer the natural indent from surrounding code context
    natural_indent = _infer_base_indent(lines, insert_point_0)
    first_line = new_lines[0]
    first_indent = _count_leading_spaces(first_line)

    if first_indent < natural_indent:
        # Model dropped tokens — correct upward to natural AST level
        first_line_fixed = " " * natural_indent + first_line.lstrip()
        corrected = [first_line_fixed] + new_lines[1:]
        return corrected, natural_indent

    # first_indent >= natural_indent: model got it right or overshot — leave alone
    # AST gate will catch genuinely wrong indentation
    return new_lines, None


def _validate_python_syntax(content: str, filepath: str) -> Optional[str]:
    """Validate Python syntax. Returns error message if invalid, None if valid.

    Uses compile() instead of ast.parse() to catch errors like 'return outside
    function' that ast.parse() misses.
    """
    try:
        compile(content, filepath, "exec")
        return None
    except SyntaxError as e:
        return (
            f"Python syntax error in proposed edit: {e.msg} at line {e.lineno}"
            + (f", column {e.offset}" if e.offset else "")
            + ". File NOT modified. Fix the syntax/indentation and retry."
        )


def _build_preview(
    lines: list[str],
    edit_start: int,
    edit_end: int,
    context: int = 3,
    max_edit_show: int = 6,
) -> str:
    """Build a preview showing context around an edit region.

    Shows 3 lines before the edit, the first few lines of the edit,
    a skip marker if the edit is long, the last few lines of the edit,
    and 3 lines after the edit. Line numbers included (1-indexed).
    """
    total = len(lines)
    result_parts = []

    # Context before edit
    before_start = max(0, edit_start - context)
    for i in range(before_start, edit_start):
        result_parts.append(f"  {i + 1:>4} | {lines[i]}")

    # Edit region
    edit_len = edit_end - edit_start
    if edit_len <= max_edit_show:
        # Show entire edit region
        for i in range(edit_start, edit_end):
            result_parts.append(f"  {i + 1:>4} | {lines[i]}")
    else:
        # Show start of edit
        for i in range(edit_start, edit_start + context):
            if i < edit_end:
                result_parts.append(f"  {i + 1:>4} | {lines[i]}")
        # Skip marker
        skipped = edit_len - (2 * context)
        result_parts.append("       | ... ({} lines)".format(skipped))
        # Show end of edit
        for i in range(edit_end - context, edit_end):
            result_parts.append(f"  {i + 1:>4} | {lines[i]}")

    # Context after edit
    after_end = min(total, edit_end + context)
    for i in range(edit_end, after_end):
        result_parts.append(f"  {i + 1:>4} | {lines[i]}")

    return "\n".join(result_parts)


# =============================================================================
# edit_file Tool
# =============================================================================


@tool
def edit_file(
    filepath: str,
    find: Optional[str] = None,
    content: Optional[str] = None,
    mode: str = "replace",
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    replace_all: bool = False,
    snapshot_id: Optional[int] = None,
) -> dict:
    """Edit a file using line-based navigation. Modes: replace (default), append, prepend, replace_range, delete, rollback.

    Args:
        filepath: Path to file
        find: Text to find (replace/delete/append/prepend modes)
        content: Content to insert or replace with (all modes except delete/rollback). If python, the first line indent may be autocorrected
        mode: Edit mode — "rollback" undoes last edit; "rollback" + snapshot_id restores specific snapshot
        start_line: Start line number (1-indexed, inclusive)
        end_line: End line number (1-indexed, inclusive)
        replace_all: Act on all matches (default: False)
        snapshot_id: Snapshot to restore (rollback mode only; omit for latest)

    Returns:
        Dict with success status and details. Every edit result includes
        snapshot_id for potential rollback.
    """
    # Validate mode
    valid_modes = (
        "replace",
        "append",
        "prepend",
        "replace_range",
        "delete",
        "rollback",
    )
    if mode not in valid_modes:
        return {"error": f"Invalid mode: {mode}. Valid modes: {valid_modes}"}

    # --- Rollback: early exit before file I/O and parameter validation ---
    if mode == "rollback":
        if filepath not in _snapshots or not _snapshots[filepath]:
            return {"success": False, "error": f"No snapshots exist for {filepath}"}
        snapshots = _snapshots[filepath]
        # Resolve snapshot_id: None or -1 means "latest"
        target_id = snapshot_id
        if target_id is None or target_id == -1:
            target_id = max(snapshots.keys())
        if target_id not in snapshots:
            available = sorted(snapshots.keys())
            return {
                "success": False,
                "error": f"Snapshot {target_id} not found. Available: {available}",
            }
        # Grab the target content before any eviction
        restored = snapshots[target_id]
        # Snapshot current state first — undo is undoable
        current = Path(filepath).read_text("utf-8")
        counter = _snapshot_counter[filepath]
        _snapshots[filepath][counter] = current
        new_snapshot_id = counter
        _snapshot_counter[filepath] = counter + 1
        # Evict oldest if over cap (target is safe — already extracted)
        while len(_snapshots[filepath]) > MAX_SNAPSHOTS_PER_FILE:
            oldest = min(snapshots.keys())
            del snapshots[oldest]
        # Write restored content
        path = Path(filepath)
        path.write_text(restored, "utf-8")
        return {
            "success": True,
            "message": f"Restored {filepath} to snapshot {target_id}",
            "snapshot_id": new_snapshot_id,
            "filepath": filepath,
            "mode": "rollback",
            "sandbox_mode": get_current_sandbox_mode().value
            if get_current_sandbox_mode()
            else "default",
        }

    # Validate parameters for each non-rollback mode
    if mode == "replace":
        if find is None or content is None:
            return {"error": f"Mode '{mode}' requires 'find' and 'content' parameters"}
    elif mode == "delete":
        # Delete mode requires either find OR start_line/end_line
        if find is None and (start_line is None or end_line is None):
            return {
                "error": "Mode 'delete' requires either 'find' parameter or 'start_line' and 'end_line' parameters"
            }
    else:
        if content is None:
            return {"error": f"Mode '{mode}' requires 'content' parameter"}

    if mode == "replace_range":
        if start_line is None or end_line is None:
            return {
                "error": "Mode 'replace_range' requires 'start_line' and 'end_line' parameters"
            }

    # Validate line numbers
    if start_line is not None:
        if start_line < 1:
            return {"error": f"start_line must be >= 1, got {start_line}"}
    if end_line is not None:
        if end_line < 1:
            return {"error": f"end_line must be >= 1, got {end_line}"}
    if start_line is not None and end_line is not None:
        if start_line > end_line:
            return {
                "error": f"start_line ({start_line}) must be <= end_line ({end_line})"
            }
    # Validate path with sandbox enforcement
    is_valid, error = validate_path_for_write(filepath)
    if not is_valid:
        return {
            "error": error,
            "sandbox_mode": get_current_sandbox_mode().value
            if get_current_sandbox_mode()
            else "default",
        }

    path = Path(filepath)

    # Check file exists
    if not path.exists():
        return {
            "error": f"File not found: {filepath}\n\nUse write_file to create new files."
        }

    if not path.is_file():
        return {"error": f"Not a file: {filepath}"}

    # Read file content
    try:
        raw_content = path.read_bytes()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Check for binary content
    if _is_binary_file(raw_content):
        return {"error": f"Binary file not supported: {filepath}"}

    # Decode content
    try:
        original_content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": f"File is not valid UTF-8: {filepath}"}

    # Detect line ending style
    has_trailing_newline = original_content.endswith("\n")
    lines = original_content.splitlines()

    # Convert 1-indexed line numbers to 0-indexed
    range_start = (start_line - 1) if start_line is not None else None
    range_end = (end_line - 1) if end_line is not None else None

    # Validate line range against file size
    if range_start is not None and range_start >= len(lines):
        return {
            "error": f"start_line ({start_line}) exceeds file length ({len(lines)} lines)"
        }
    if range_end is not None and range_end >= len(lines):
        return {
            "error": f"end_line ({end_line}) exceeds file length ({len(lines)} lines)"
        }

    file_extension = path.suffix.lower()

    # Perform the edit
    modified_lines = lines.copy()
    edit_description = ""
    original_lines = []  # For modes that replace content, store what was replaced
    auto_indent = None  # Set by append/prepend/replace_range if auto-correct fires

    if mode == "replace":
        success, modified_lines, error, num_replacements = _do_replace(
            modified_lines, find, content, range_start, range_end, False, replace_all
        )  # match_target_indentation disabled
        if not success:
            return {"success": False, "error": error}
        # Build informative message with replacement count
        if num_replacements == 1:
            edit_description = f"Replaced 1 occurrence of '{find}' with '{content}'"
        else:
            edit_description = (
                f"Replaced {num_replacements} occurrences of '{find}' with '{content}'"
            )
        if start_line is not None:
            edit_description += f" in lines {start_line}-{end_line}"
    elif mode == "append":
        # Clean whitespace from content
        new_lines = content.splitlines()
        new_lines = _clean_whitespace_lines(new_lines)
        # Auto-correct first line indentation for Python files
        if file_extension == ".py" and new_lines:
            # Compute insert point for indent inference
            # Append inserts AFTER the anchor line, so insert_point is one past it
            if range_end is not None:
                _insert_point = range_end + 1
            elif range_start is not None:
                _insert_point = range_start + 1
            elif find is None:
                _insert_point = len(lines)  # append at end (past last line)
            else:
                _insert_point = None  # find-based: defer to after match
            if _insert_point is not None:
                new_lines, auto_indent = _auto_correct_first_line_indent(
                    new_lines, lines, _insert_point, file_extension
                )

        if find is not None:
            # Find matches using multi-line aware matching
            search_start = range_start if range_start is not None else None
            search_end = range_end if range_end is not None else None

            matches, error = _find_matches(
                modified_lines, find, search_start, search_end
            )

            if error:
                return {"success": False, "error": error}

            if len(matches) > 1 and not replace_all:
                match_info = []
                for start_idx, end_idx in matches[:5]:
                    snippet = (
                        modified_lines[start_idx][:50]
                        if modified_lines[start_idx]
                        else ""
                    )
                    if start_idx == end_idx:
                        match_info.append(f"  Line {start_idx + 1}: {snippet}...")
                    else:
                        match_info.append(
                            f"  Lines {start_idx + 1}-{end_idx + 1}: {snippet}..."
                        )
                if len(matches) > 5:
                    match_info.append(f"  ... and {len(matches) - 5} more matches")
                return {
                    "success": False,
                    "error": f"Multiple matches found for '{find}' ({len(matches)} total). Use replace_all=True to append after all, or be more specific:\n"
                    + "\n".join(match_info),
                }

            # Insert after each match
            # For multi-line matches, insert after the last line of the match
            if replace_all:
                # Insert after each match, in reverse order to preserve indices
                for start_idx, end_idx in reversed(matches):
                    modified_lines = (
                        modified_lines[: end_idx + 1]
                        + new_lines
                        + modified_lines[end_idx + 1 :]
                    )
                edit_description = (
                    f"Appended content after {len(matches)} matches of '{find}'"
                )
            else:
                start_idx, end_idx = matches[0]
                modified_lines = (
                    modified_lines[: end_idx + 1]
                    + new_lines
                    + modified_lines[end_idx + 1 :]
                )
                if start_idx == end_idx:
                    edit_description = f"Appended content after line {end_idx + 1}"
                else:
                    edit_description = (
                        f"Appended content after lines {start_idx + 1}-{end_idx + 1}"
                    )
        elif range_end is not None or range_start is not None:
            # Either line number can anchor the insertion point
            # With both, append after the bottom of the region (end_line)
            if range_end is not None:
                insert_line = range_end
                label = end_line
            else:
                insert_line = range_start
                label = start_line
            modified_lines = (
                modified_lines[: insert_line + 1]
                + new_lines
                + modified_lines[insert_line + 1 :]
            )
            if (
                start_line is not None
                and end_line is not None
                and start_line != end_line
            ):
                edit_description = f"Appended content after line {end_line}"
            else:
                edit_description = f"Appended content after line {label}"
        else:
            # Default: append at end of file
            modified_lines = modified_lines + new_lines
            edit_description = "Appended content to end of file"

    elif mode == "prepend":
        # Clean whitespace from content
        new_lines = content.splitlines()
        new_lines = _clean_whitespace_lines(new_lines)
        # Auto-correct first line indentation for Python files
        if file_extension == ".py" and new_lines:
            # Compute insert point for indent inference
            if range_start is not None:
                _insert_point = range_start
            elif range_end is not None:
                _insert_point = range_end
            elif find is None:
                _insert_point = 0  # prepend at start
            else:
                _insert_point = None  # find-based: defer
            if _insert_point is not None:
                new_lines, auto_indent = _auto_correct_first_line_indent(
                    new_lines, lines, _insert_point, file_extension
                )

        # Determine insertion point
        if find is not None:
            # Find matches using multi-line aware matching
            search_start = range_start if range_start is not None else None
            search_end = range_end if range_end is not None else None

            matches, error = _find_matches(
                modified_lines, find, search_start, search_end
            )

            if error:
                return {"success": False, "error": error}

            if len(matches) > 1 and not replace_all:
                match_info = []
                for start_idx, end_idx in matches[:5]:
                    snippet = (
                        modified_lines[start_idx][:50]
                        if modified_lines[start_idx]
                        else ""
                    )
                    if start_idx == end_idx:
                        match_info.append(f"  Line {start_idx + 1}: {snippet}...")
                    else:
                        match_info.append(
                            f"  Lines {start_idx + 1}-{end_idx + 1}: {snippet}..."
                        )
                if len(matches) > 5:
                    match_info.append(f"  ... and {len(matches) - 5} more matches")
                return {
                    "success": False,
                    "error": f"Multiple matches found for '{find}' ({len(matches)} total). Use replace_all=True to prepend before all, or be more specific:\n"
                    + "\n".join(match_info),
                }

            # Insert before each match
            # For multi-line matches, insert before the first line of the match
            if replace_all:
                # Insert before each match, in reverse order to preserve indices
                for start_idx, end_idx in reversed(matches):
                    modified_lines = (
                        modified_lines[:start_idx]
                        + new_lines
                        + modified_lines[start_idx:]
                    )
                edit_description = (
                    f"Prepended content before {len(matches)} matches of '{find}'"
                )
            else:
                start_idx, end_idx = matches[0]
                modified_lines = (
                    modified_lines[:start_idx] + new_lines + modified_lines[start_idx:]
                )
                if start_idx == end_idx:
                    edit_description = f"Prepended content before line {start_idx + 1}"
                else:
                    edit_description = (
                        f"Prepended content before lines {start_idx + 1}-{end_idx + 1}"
                    )
        elif range_start is not None or range_end is not None:
            # Either line number can anchor the insertion point
            # With both, prepend before the top of the region (start_line)
            if range_start is not None:
                insert_at = range_start
                label = start_line
            else:
                insert_at = range_end
                label = end_line
            modified_lines = (
                modified_lines[:insert_at] + new_lines + modified_lines[insert_at:]
            )
            if (
                start_line is not None
                and end_line is not None
                and start_line != end_line
            ):
                edit_description = f"Prepended content before line {start_line}"
            else:
                edit_description = f"Prepended content before line {label}"
        else:
            # Default: prepend at beginning of file
            modified_lines = new_lines + modified_lines
            edit_description = "Prepended content to start of file"

    elif mode == "replace_range":
        # Clean whitespace from content
        original_lines = (
            lines[range_start : range_end + 1]
            if range_start is not None and range_end is not None
            else []
        )
        new_lines = content.splitlines()
        new_lines = _clean_whitespace_lines(new_lines)
        # Auto-correct first line indentation for Python files
        if file_extension == ".py" and range_start is not None and new_lines:
            new_lines, auto_indent = _auto_correct_first_line_indent(
                new_lines, lines, range_start, file_extension
            )
        if auto_indent is not None:
            edit_description = (
                f"Replaced lines {start_line}-{end_line} (auto_indent={auto_indent})"
            )
        else:
            edit_description = f"Replaced lines {start_line}-{end_line}"
        modified_lines = (
            modified_lines[:range_start] + new_lines + modified_lines[range_end + 1 :]
        )

    elif mode == "delete":
        if find is not None:
            # Delete entire lines containing the pattern
            # First, find which lines contain the pattern
            search_start = range_start if range_start is not None else 0
            search_end = range_end if range_end is not None else len(modified_lines) - 1

            # Find matching lines
            matching_indices = []
            for i in range(search_start, search_end + 1):
                if i < len(modified_lines) and find in modified_lines[i]:
                    matching_indices.append(i)

            if len(matching_indices) == 0:
                # No exact match - try fuzzy matching for better error message
                fuzzy_match = _find_fuzzy_match(
                    modified_lines, find, search_start, search_end, FUZZY_THRESHOLD
                )
                error_msg = _format_fuzzy_error(
                    find,
                    modified_lines,
                    fuzzy_match,
                    start_line=search_start + 1 if range_start is not None else None,
                    end_line=search_end + 1 if range_end is not None else None,
                )
                return {"success": False, "error": error_msg}

            # Handle multiple matches based on replace_all flag
            if len(matching_indices) > 1 and not replace_all:
                match_info = []
                for idx in matching_indices[:5]:
                    snippet = modified_lines[idx][:50]
                    match_info.append(f"  Line {idx + 1}: {snippet}...")
                if len(matching_indices) > 5:
                    match_info.append(
                        f"  ... and {len(matching_indices) - 5} more matches"
                    )
                return {
                    "success": False,
                    "error": f"Multiple lines contain '{find}' ({len(matching_indices)} total). Use replace_all=True to delete all, or be more specific:\n"
                    + "\n".join(match_info),
                }

            # Delete the matching line(s)
            if replace_all:
                lines_to_delete = set(matching_indices)
            else:
                lines_to_delete = {matching_indices[0]}

            original_lines = [modified_lines[i] for i in sorted(lines_to_delete)]
            modified_lines = [
                line
                for i, line in enumerate(modified_lines)
                if i not in lines_to_delete
            ]
            num_deletions = len(lines_to_delete)

            # Build informative message
            if num_deletions == 1:
                edit_description = (
                    f"Deleted line {matching_indices[0] + 1} containing '{find}'"
                )
            else:
                line_nums = ", ".join(str(i + 1) for i in sorted(lines_to_delete)[:5])
                if len(lines_to_delete) > 5:
                    line_nums += f" ... ({len(lines_to_delete) - 5} more)"
                edit_description = (
                    f"Deleted {num_deletions} lines containing '{find}': {line_nums}"
                )
            num_replacements = num_deletions  # For result dict
        else:
            # Delete by line range
            original_lines = lines[range_start : range_end + 1]
            modified_lines = (
                modified_lines[:range_start] + modified_lines[range_end + 1 :]
            )
            edit_description = (
                f"Deleted lines {start_line}-{end_line} ({len(original_lines)} lines)"
            )

    # Find the edit region by diffing original vs modified lines
    # This is universal — works for all modes without per-mode tracking
    edit_start = 0
    while (
        edit_start < len(lines)
        and edit_start < len(modified_lines)
        and lines[edit_start] == modified_lines[edit_start]
    ):
        edit_start += 1
    # For the tail, compare from the end of both arrays
    # When lengths differ, the extra lines are always part of the edit region
    edit_end = len(
        modified_lines
    )  # Use modified_lines length (may be longer for append)
    tail_match = 0
    min_len = min(len(lines), len(modified_lines))
    while (
        tail_match < min_len - edit_start
        and lines[len(lines) - 1 - tail_match]
        == modified_lines[len(modified_lines) - 1 - tail_match]
    ):
        tail_match += 1
    edit_end = len(modified_lines) - tail_match

    # Snapshot before write — enables rollback
    if filepath not in _snapshots:
        _snapshots[filepath] = {}
        _snapshot_counter[filepath] = 0
    _snapshots[filepath][_snapshot_counter[filepath]] = original_content
    current_snapshot_id = _snapshot_counter[filepath]
    _snapshot_counter[filepath] += 1
    # Evict oldest snapshots if over cap
    while len(_snapshots[filepath]) > MAX_SNAPSHOTS_PER_FILE:
        oldest = min(_snapshots[filepath].keys())
        del _snapshots[filepath][oldest]

    # Write back to file
    new_content = "\n".join(modified_lines)
    if has_trailing_newline:
        new_content += "\n"

    # AST validation gate for Python files — reject invalid syntax before writing
    if file_extension == ".py":
        syntax_error = _validate_python_syntax(new_content, filepath)
        if syntax_error:
            return {"success": False, "error": syntax_error}

    try:
        path.write_text(new_content, "utf-8")
    except Exception as e:
        return {"error": f"Failed to write file: {e}"}

    result = {
        "success": True,
        "message": edit_description,
        "filepath": filepath,
        "mode": mode,
        "snapshot_id": current_snapshot_id,
        "sandbox_mode": get_current_sandbox_mode().value
        if get_current_sandbox_mode()
        else "default",
    }
    # Include replacement/deletion count for replace and delete modes
    if mode == "replace":
        result["replacements"] = num_replacements
    elif mode == "delete" and find is not None:
        result["deletions"] = num_replacements  # num_replacements holds deletion count
    # Include original content for verification when lines were replaced
    if original_lines:
        result["original_lines"] = original_lines
    # Include auto_indent info when correction was applied
    if auto_indent is not None:
        result["auto_indent"] = auto_indent
    # Include context preview showing the edit region with surrounding lines
    if edit_start < edit_end:
        result["preview"] = _build_preview(modified_lines, edit_start, edit_end)
    return result
