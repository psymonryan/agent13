"""Read file tool - smart file reading for AI agents.

As a tool, returns dict with filepath, total_lines, view type, and content.
"""

import re
from pathlib import Path
from typing import Optional, Dict, List

from tools import tool
from tools.security import validate_path_for_read, get_current_sandbox_mode


# =============================================================================
# Helpers
# =============================================================================


def _is_binary_file(content: bytes) -> bool:
    """Check if content appears to be binary (non-text).

    Simple check: look for null bytes.
    """
    return b"\x00" in content


# =============================================================================
# Custom Symbol Extractor (Regex-based, Language-Agnostic)
# =============================================================================


def _extract_symbols_python(filepath: Path, lines: List[str]) -> Dict[str, dict]:
    """Extract symbols from Python file using regex.

    Returns dict: {symbol_name: {'line': int, 'type': str, 'indent': str, 'end_line': int}}

    Handles:
    - Functions: def name(...)
    - Classes: class Name(...)
    - Async functions: async def name(...)
    - Decorated functions/classes (includes decorators in symbol range)
    - Nested classes/functions
    - Marimo notebook cells (@app.cell decorator)

    Note: For decorated symbols, 'line' refers to the first decorator line,
    not the def/class line. This makes read_file and replace_symbol more intuitive.
    """
    symbols = {}
    class_stack = []  # Stack of (class_name, indent_level)

    # Patterns
    class_pattern = re.compile(r"^(\s*)(?:async\s+)?class\s+(\w+)")
    func_pattern = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)")
    decorator_pattern = re.compile(r"^(\s*)@\w+(?:\.\w+)*")

    def find_decorator_start(line_idx: int, base_indent: str) -> int:
        """Look backwards from a def/class line to find the first decorator."""
        start_idx = line_idx
        i = line_idx - 1

        while i >= 0:
            line = lines[i]
            stripped = line.strip()

            # Stop at blank lines
            if not stripped:
                break

            # Check if this is a decorator at the same or greater indentation
            dec_match = decorator_pattern.match(line)
            if dec_match:
                dec_indent = dec_match.group(1)
                # Decorator must be at same or greater indentation than the def/class
                if len(dec_indent) >= len(base_indent):
                    start_idx = i
                    i -= 1
                else:
                    # Decorator is less indented, probably belongs to outer scope
                    break
            else:
                # Not a decorator, stop looking
                break

        return start_idx

    i = 0
    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # Check for decorator
        dec_match = decorator_pattern.match(line)
        if dec_match:
            # Look ahead to see what this decorates
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                next_line = lines[j]
                # Check if it decorates a class or function
                if class_pattern.match(next_line) or func_pattern.match(next_line):
                    # Skip to the decorated item
                    i = j
                    line = lines[i]
                    line_num = i + 1

        # Check for class
        class_match = class_pattern.match(line)
        if class_match:
            indent, name = class_match.groups()
            indent_level = len(indent)

            # Look backwards to find decorators
            decorator_start_idx = find_decorator_start(i, indent)
            decorator_start_line = decorator_start_idx + 1

            # Update class stack
            while class_stack and class_stack[-1][1] >= indent_level:
                class_stack.pop()

            if class_stack:
                qualified_name = f"{class_stack[-1][0]}.{name}"
            else:
                qualified_name = name

            symbols[qualified_name] = {
                "line": decorator_start_line,  # Use decorator line if present
                "type": "class",
                "indent": indent,
                "indent_level": indent_level,
            }

            class_stack.append((name, indent_level))
            i += 1
            continue

        # Check for function
        func_match = func_pattern.match(line)
        if func_match:
            indent, name = func_match.groups()
            indent_level = len(indent)

            # Look backwards to find decorators
            decorator_start_idx = find_decorator_start(i, indent)
            decorator_start_line = decorator_start_idx + 1

            # Update class stack based on indent
            while class_stack and class_stack[-1][1] >= indent_level:
                class_stack.pop()

            # Determine qualified name
            if class_stack:
                qualified_name = f"{class_stack[-1][0]}.{name}"
            else:
                qualified_name = name

            # For anonymous functions, make the name unique by including line number
            if name == "_":
                qualified_name = f"_{line_num}"

            symbols[qualified_name] = {
                "line": decorator_start_line,  # Use decorator line if present
                "type": "function",
                "indent": indent,
                "indent_level": indent_level,
            }

            i += 1
            continue

        i += 1

    # Calculate end lines
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["line"])
    for idx, (name, info) in enumerate(sorted_symbols):
        if idx < len(sorted_symbols) - 1:
            # End line is start of next symbol at same or lower indent level
            next_info = sorted_symbols[idx + 1][1]
            if next_info["indent_level"] <= info["indent_level"]:
                info["end_line"] = next_info["line"] - 1
            else:
                info["end_line"] = next_info["line"] - 1
        else:
            info["end_line"] = len(lines)

    return symbols


def _extract_symbols_javascript(filepath: Path, lines: List[str]) -> Dict[str, dict]:
    """Extract symbols from JavaScript/TypeScript file using regex.

    Handles:
    - Functions: function name(...)
    - Classes: class Name
    - Arrow functions: const/let/var name = (...) =>
    - Export statements
    - Methods in classes
    """
    symbols = {}
    class_stack = []

    # Patterns
    class_pattern = re.compile(r"^(\s*)(?:export\s+)?(?:abstract\s+)?class\s+(\w+)")
    func_pattern = re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)")
    arrow_pattern = re.compile(
        r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"
    )
    method_pattern = re.compile(
        r"^(\s+)(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{"
    )

    for i, line in enumerate(lines):
        line_num = i + 1

        # Check for class
        class_match = class_pattern.match(line)
        if class_match:
            indent, name = class_match.groups()
            indent_level = len(indent)

            while class_stack and class_stack[-1][1] >= indent_level:
                class_stack.pop()

            qualified_name = f"{class_stack[-1][0]}.{name}" if class_stack else name

            symbols[qualified_name] = {
                "line": line_num,
                "type": "class",
                "indent": indent,
                "indent_level": indent_level,
            }

            class_stack.append((name, indent_level))
            continue

        # Check for function
        func_match = func_pattern.match(line)
        if func_match:
            indent, name = func_match.groups()
            indent_level = len(indent)

            while class_stack and class_stack[-1][1] >= indent_level:
                class_stack.pop()

            qualified_name = f"{class_stack[-1][0]}.{name}" if class_stack else name

            symbols[qualified_name] = {
                "line": line_num,
                "type": "function",
                "indent": indent,
                "indent_level": indent_level,
            }
            continue

        # Check for arrow function
        arrow_match = arrow_pattern.match(line)
        if arrow_match:
            indent, name = arrow_match.groups()
            indent_level = len(indent)

            while class_stack and class_stack[-1][1] >= indent_level:
                class_stack.pop()

            qualified_name = f"{class_stack[-1][0]}.{name}" if class_stack else name

            symbols[qualified_name] = {
                "line": line_num,
                "type": "function",
                "indent": indent,
                "indent_level": indent_level,
            }
            continue

        # Check for method (inside class)
        if class_stack:
            method_match = method_pattern.match(line)
            if method_match:
                indent, name = method_match.groups()
                indent_level = len(indent)

                qualified_name = f"{class_stack[-1][0]}.{name}"

                symbols[qualified_name] = {
                    "line": line_num,
                    "type": "method",
                    "indent": indent,
                    "indent_level": indent_level,
                }

    # Calculate end lines
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["line"])
    for idx, (name, info) in enumerate(sorted_symbols):
        if idx < len(sorted_symbols) - 1:
            info["end_line"] = sorted_symbols[idx + 1][1]["line"] - 1
        else:
            info["end_line"] = len(lines)

    return symbols


def _extract_symbols_markdown(filepath: Path, lines: List[str]) -> Dict[str, dict]:
    """Extract symbols from Markdown file (headers as symbols)."""
    symbols = {}

    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for i, line in enumerate(lines):
        line_num = i + 1
        match = header_pattern.match(line)
        if match:
            hashes, title = match.groups()
            level = len(hashes)
            indent = "    " * (level - 1)  # Visual indentation

            # Create qualified name based on header hierarchy
            symbols[title.strip()] = {
                "line": line_num,
                "type": "header",
                "indent": indent,
                "indent_level": level,
                "header_level": level,
            }

    # Calculate end lines
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["line"])
    for idx, (name, info) in enumerate(sorted_symbols):
        if idx < len(sorted_symbols) - 1:
            info["end_line"] = sorted_symbols[idx + 1][1]["line"] - 1
        else:
            info["end_line"] = len(lines)

    return symbols


def _extract_symbols_css(filepath: Path, lines: List[str]) -> Dict[str, dict]:
    """Extract symbols from CSS file (selectors as symbols)."""
    symbols = {}

    # Match CSS selectors (simplified)
    selector_pattern = re.compile(r"^([^{]+)\s*\{")

    for i, line in enumerate(lines):
        line_num = i + 1
        match = selector_pattern.match(line.strip())
        if match:
            selector = match.group(1).strip()
            # Clean up selector
            selector = " ".join(selector.split())

            if selector and not selector.startswith("@"):
                symbols[selector] = {
                    "line": line_num,
                    "type": "selector",
                    "indent": "",
                    "indent_level": 0,
                }

    # Calculate end lines
    sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["line"])
    for idx, (name, info) in enumerate(sorted_symbols):
        if idx < len(sorted_symbols) - 1:
            info["end_line"] = sorted_symbols[idx + 1][1]["line"] - 1
        else:
            info["end_line"] = len(lines)

    return symbols


def extract_symbols(filepath: Path) -> Dict[str, dict]:
    """Extract symbols from file based on language.

    Returns dict: {symbol_name: {'line': int, 'type': str, 'indent': str, 'end_line': int}}

    Falls back to empty dict if language not supported.
    """
    if not filepath.exists():
        return {}

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return {}

    ext = filepath.suffix.lower()

    if ext == ".py":
        return _extract_symbols_python(filepath, lines)
    elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
        return _extract_symbols_javascript(filepath, lines)
    elif ext in (".md", ".markdown"):
        return _extract_symbols_markdown(filepath, lines)
    elif ext == ".css":
        return _extract_symbols_css(filepath, lines)
    else:
        return {}


def _generate_compact_python_skim(sorted_symbols: list, lines: list) -> list:
    """Generate compact skim view for Python files by grouping anonymous functions.

    Shows named functions/classes in full, groups anonymous functions into ranges.

    Args:
        sorted_symbols: List of (symbol_name, info) tuples sorted by line number
        lines: File content as list of lines

    Returns:
        List of (line_num, content) tuples for output
    """
    output = []

    # Group consecutive anonymous functions
    i = 0
    while i < len(sorted_symbols):
        symbol_name, info = sorted_symbols[i]
        line_num = info["line"]
        sym_type = info["type"]

        # Check if this is an anonymous function
        is_anonymous = False
        if sym_type == "function":
            if symbol_name.startswith("_") and symbol_name[1:].isdigit():
                is_anonymous = True
            elif symbol_name == "_":
                is_anonymous = True
            elif "." in symbol_name:
                parts = symbol_name.split(".")
                last_part = parts[-1]
                if last_part == "_" or (
                    last_part.startswith("_") and last_part[1:].isdigit()
                ):
                    is_anonymous = True

        if is_anonymous:
            # Count consecutive anonymous functions
            anon_start = line_num
            anon_count = 1
            j = i + 1

            while j < len(sorted_symbols):
                next_name, next_info = sorted_symbols[j]
                next_type = next_info["type"]

                # Check if next is also anonymous
                next_is_anon = False
                if next_type == "function":
                    if next_name.startswith("_") and next_name[1:].isdigit():
                        next_is_anon = True
                    elif next_name == "_":
                        next_is_anon = True
                    elif "." in next_name:
                        parts = next_name.split(".")
                        last_part = parts[-1]
                        if last_part == "_" or (
                            last_part.startswith("_") and last_part[1:].isdigit()
                        ):
                            next_is_anon = True

                if next_is_anon:
                    anon_count += 1
                    j += 1
                else:
                    break

            anon_end = sorted_symbols[j - 1][1]["line"] if j > i + 1 else anon_start

            # Add grouped entry
            if anon_count == 1:
                # Single anonymous function - skip it (too much noise for single cells)
                pass
            else:
                # Multiple anonymous functions - group them
                output.append(
                    (
                        anon_start,
                        f"{anon_start:4d}-{anon_end:4d}: {anon_count} anonymous functions",
                    )
                )

            i = j
        else:
            # Named function/class - show in full
            indent_str = info["indent"]
            line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""

            if sym_type == "class":
                output.append((line_num, f"{line_num:4d}: {indent_str}{line_content}"))
            elif sym_type == "function":
                output.append((line_num, f"{line_num:4d}: {indent_str}{line_content}"))
            elif sym_type == "method":
                output.append((line_num, f"{line_num:4d}: {indent_str}{line_content}"))
            elif sym_type == "header":
                output.append((line_num, f"{line_num:4d}: {line_content}"))
            elif sym_type == "selector":
                output.append((line_num, f"{line_num:4d}: {line_content}"))
            else:
                output.append((line_num, f"{line_num:4d}: {indent_str}{line_content}"))

            i += 1

    return output


def _generate_skim_view(filepath: str, source_bytes: bytes, lines: list) -> str:
    """Generate ultra-compact skim view for supported file types.

    Uses custom symbol extractor first (more reliable), falls back to tree-sitter.
    For Python files, groups anonymous functions to keep output compact.
    """
    path = Path(filepath)

    # Try custom symbol extractor first (works on any text, even with syntax errors)
    symbols = extract_symbols(path)

    if symbols:
        # Use custom extractor results
        output = []
        sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["line"])

        # For Python files, group anonymous functions to keep output compact
        if path.suffix.lower() == ".py":
            output = _generate_compact_python_skim(sorted_symbols, lines)
        else:
            # For other languages, show all symbols
            for symbol_name, info in sorted_symbols:
                line_num = info["line"]
                indent_str = info["indent"]
                sym_type = info["type"]

                # Get the actual line content
                line_content = (
                    lines[line_num - 1].strip() if line_num <= len(lines) else ""
                )

                # Format based on type
                if sym_type == "class":
                    output.append(
                        (line_num, f"{line_num:4d}: {indent_str}{line_content}")
                    )
                elif sym_type == "function":
                    output.append(
                        (line_num, f"{line_num:4d}: {indent_str}{line_content}")
                    )
                elif sym_type == "method":
                    output.append(
                        (line_num, f"{line_num:4d}: {indent_str}{line_content}")
                    )
                elif sym_type == "header":
                    output.append((line_num, f"{line_num:4d}: {line_content}"))
                elif sym_type == "selector":
                    output.append((line_num, f"{line_num:4d}: {line_content}"))
                else:
                    output.append(
                        (line_num, f"{line_num:4d}: {indent_str}{line_content}")
                    )

        # Format output
        result = [f"skim: {filepath} ({len(lines)} lines)", ""]
        for line_num, content in output:
            result.append(content)

        return "\n".join(result)

    # No tree-sitter fallback - custom extractor is primary and sufficient
    return None


def _generate_raw_view(
    filepath: str, lines: list, offset: int = 1, limit: int = 50
) -> str:
    """Generate raw view with line numbers.

    Shows file content with line numbers, prefixed with "raw:" to distinguish
    from the "skim:" structured view. Used when:
    - offset/limit is specified (specific line range)
    - File type doesn't have a skim view extractor
    """
    total_lines = len(lines)
    start_idx = offset - 1  # Convert to 0-indexed
    end_idx = min(start_idx + limit, total_lines)

    result = [
        f"raw: {filepath} (showing lines {offset}-{end_idx} of {total_lines})",
        "",
    ]
    for i in range(start_idx, end_idx):
        result.append(f"{i + 1:4d}: {lines[i]}")

    if end_idx < total_lines:
        result.append("")
        result.append(
            f"[{total_lines - end_idx} more lines not shown. Use --offset {end_idx + 1} to see more.]"
        )

    return "\n".join(result)


@tool
def read_file(
    filepath: str, offset: Optional[int] = None, limit: Optional[int] = None
) -> dict:
    """Read a file. No params = skim (symbols). offset/limit = raw lines.

    Args:
        filepath: Path to file
        offset: Starting line (1-indexed)
        limit: Max lines to return

    Returns:
        Dict with filepath, total_lines, view type, and content
    """
    # Validate path with sandbox enforcement
    is_valid, error = validate_path_for_read(filepath)
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
        return {"error": f"File not found: {filepath}"}

    if not path.is_file():
        return {"error": f"Not a file: {filepath}"}

    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024
    file_size = path.stat().st_size
    if file_size > max_size:
        return {
            "error": f"File too large ({file_size} bytes). Use offset/limit to read in chunks."
        }

    # Read file content
    try:
        raw_content = path.read_bytes()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Check for binary content
    if _is_binary_file(raw_content):
        return {"error": f"Binary file not supported: {filepath}"}

    # Decode and split into lines
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": f"File is not valid UTF-8: {filepath}"}

    lines = content.splitlines()
    total_lines = len(lines)

    # Short files: show full content instead of skim (skip the skim→read dance)
    short_file_threshold = 300
    is_short_file = total_lines <= short_file_threshold

    # No params: try skim view first (but not for short files)
    # Note: CLI sets limit=50 by default, so check for None or default 50
    use_skim = not is_short_file and (
        (offset is None and limit is None) or (offset is None and limit == 50)
    )
    if use_skim:
        # Try skim view (works for any file with custom symbol extractor)
        content_output = _generate_skim_view(filepath, raw_content, lines)
        if content_output:
            return {
                "filepath": filepath,
                "total_lines": total_lines,
                "view": "skim",
                "content": content_output,
                "sandbox_mode": get_current_sandbox_mode().value
                if get_current_sandbox_mode()
                else "default",
            }

    # Raw view: show file content with line numbers
    # For short files with no explicit limit, show all lines
    raw_limit = total_lines if is_short_file and limit in (None, 50) else (limit or 50)
    content_output = _generate_raw_view(filepath, lines, offset or 1, raw_limit)
    return {
        "filepath": filepath,
        "total_lines": total_lines,
        "view": "raw",
        "content": content_output,
        "sandbox_mode": get_current_sandbox_mode().value
        if get_current_sandbox_mode()
        else "default",
    }
