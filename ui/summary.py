"""The one-line "what" for a folded tool call and a folded reasoning block.

Folding is about hiding the bulk of a turn while still showing *what is being
done* - which command is running, which file is being edited. Both the TUI
(widget headers) and the REPL (the linear stream) render that one line, so the
extraction and the cell-aware truncation live here once, and the two surfaces
always agree.
"""

from __future__ import annotations

import math

from rich.cells import cell_len

#: A one-line hint is a glance, not a paragraph. Above this, the value is
#: "payload" (file contents, big args) and is not useful folded.
_MAX_ARG_LEN = 80

#: Budget bounds for a width-relative hint. Floor keeps a narrow split-pane
#: readable; cap stops a wide monitor from yielding a 200-column line.
MIN_BUDGET = 24
MAX_BUDGET = 80
DEFAULT_BUDGET = 48

#: Which argument carries the "what" for the tools that matter, in priority
#: order. Anything not listed falls back to the first short string argument.
_TOOL_WHAT_KEYS = {
    "command": ("command",),
    "read_file": ("filepath",),
    "write_file": ("filepath",),
    "edit_file": ("filepath",),
    "grep": ("pattern", "query"),
    "search": ("query", "pattern"),
}


def _collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace (incl. newlines) to single spaces."""
    return " ".join(text.split())


def _scalar_text(value: bool | int | float) -> str:
    """Plain rendering of a scalar argument ('' when it is not displayable)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def tool_what(name: str, arguments: dict | None) -> str:
    """The untruncated one-line "what" for a tool call ('' if none).

    ``command`` -> the command string; ``read_file``/``write_file``/``edit_file``
    -> the filepath; else the first short string argument; else the first short
    scalar (bool/int/float) argument; '' when there is nothing worth showing
    (a name-only header).
    """
    if not arguments:
        return ""
    for key in _TOOL_WHAT_KEYS.get(name, ()):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _collapse_ws(value)
    for value in arguments.values():
        if isinstance(value, str) and value.strip() and len(value) <= _MAX_ARG_LEN:
            return _collapse_ws(value)
    for value in arguments.values():
        text = _scalar_text(value) if isinstance(value, (bool, int, float)) else ""
        if text and len(text) <= _MAX_ARG_LEN:
            return text
    return ""


def first_line(text: str) -> str:
    """The first non-empty line of a block, whitespace-collapsed ('' if none)."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _collapse_ws(line)
    return ""


def truncate_line(text: str, budget: int) -> str:
    """Trim ``text`` to ``budget`` cells, appending an ellipsis when cut."""
    if budget <= 0:
        return ""
    if cell_len(text) <= budget:
        return text
    cut = budget - 1  # leave one cell for the ellipsis
    out: list[str] = []
    cells = 0
    for ch in text:
        n = cell_len(ch)
        if cells + n > cut:
            break
        out.append(ch)
        cells += n
    return "".join(out) + "…"


def budget_for(label: str, available: int, indicator: bool = True) -> int:
    """Cell budget for a hint, given a fixed label and (optional) tail indicator.

    ``label`` is the whole visible prefix (e.g. ``"⚙ command: "`` /
    ``"Thoughts: "``). ``available`` is the header's current width; a non-positive
    or non-numeric value (pre-layout, mocked console) falls back to the default.
    """
    try:
        available = int(available)
    except (TypeError, ValueError):
        available = 0
    if available <= 0:
        return DEFAULT_BUDGET
    overhead = cell_len(label) + (cell_len(" ▶") if indicator else 0)
    return max(MIN_BUDGET, min(MAX_BUDGET, available - overhead))
