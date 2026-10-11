"""Tool result text bounding - Phase 1 of tool output bounding.

A tool result whose text exceeds the cap is spilled: the FULL text goes to
a file under the platform temp dir (the OS auto-cleans it - no lifecycle
code), and the model receives the first HEAD_CHARS + a marker + the last
TAIL_CHARS. For multi-line output the cuts land on line boundaries (the
marker reports line numbers that map 1:1 onto read_file's offset/limit);
when the elided region is a single line - e.g. a JSON envelope whose payload
is one escaped line - the cuts are char-based IN PLACE so the line (and any
JSON structure around it) survives and only the middle of the payload is
elided (the marker then reports a char range). The marker always names the
spill file so the model can read the middle back (read_file for multi-line,
grep/command for a single-line payload).

Design: docs_archive/tool_output_bound_design.md (Phase 1, approved
2026-09-29). Head+tail (not head-only): for logs and command output the END
is where errors and final state live. Access is via read_file on the spill
file, NOT re-invoking the tool (re-invocation re-executes = side effects;
MCP schemas aren't ours to extend; the file gives grep for free).

Text only - image payloads pass through untouched (Phase 2).
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime
from pathlib import Path

from agent13.debug_log import log_event

SPILL_DIRNAME = "agent13_tool_output"
HEAD_CHARS = 15_000
TAIL_CHARS = 15_000

# Per-process sequence so spills in the same second never collide.
_seq = itertools.count(1)


def _spill_path(tool_name: str) -> Path:
    """<tempdir>/agent13_tool_output/<YYYYmmdd-HHMMSS>_<tool>_<seq>.txt.

    The temp dir comes from agent13.sandbox.get_temp_dir() (%TEMP% on
    Windows, /tmp on Unix) - the same dir the sandbox profiles allow, so a
    spilled file is always readable back (read_file in every mode; command +
    grep in permissive). Never a hardcoded path. Tool names are sanitized
    (MCP names contain '://' and '/').
    """
    from agent13.sandbox import get_temp_dir

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", tool_name).strip("_") or "tool"
    return Path(get_temp_dir()) / SPILL_DIRNAME / f"{ts}_{safe}_{next(_seq)}.txt"


def _cut(text: str, n: int, from_end: bool = False) -> str:
    """First/last n chars, cut back/forward to a line boundary.

    Degrades to a char-boundary cut when the chunk has no newlines (e.g.
    minified JSON).
    """
    if from_end:
        chunk = text[-n:] if n < len(text) else text
        nl = chunk.find("\n")
        return chunk[nl + 1 :] if nl != -1 else chunk
    chunk = text[:n]
    nl = chunk.rfind("\n")
    return chunk[: nl + 1] if nl != -1 else chunk


def bound_tool_result(
    text: str, cap: int, tool_name: str = "tool"
) -> tuple[str, str | None]:
    """Bound a tool result's text to `cap` chars (cap <= 0 = off).

    Args:
        text: The full tool result text.
        cap: Maximum chars for the model-facing content (0 = disabled).
        tool_name: Tool name, used in the spill file name.

    Returns:
        (content_for_model, spill_path_or_None).

        - cap <= 0 or len(text) <= cap: text unchanged, no spill file
          (zero overhead on the common path).
        - len(text) > cap: full text written to a temp file; the model gets
          head + marker + tail. The cuts are line-aligned for multi-line
          output, or char-based in place when the elided region is a single
          line (a JSON envelope's escaped payload) so the structure is
          preserved. If the spill write fails (disk full etc.) the marker
          omits the file pointer and says the middle is unavailable - the
          turn is never failed by a spill error.

        A cap smaller than HEAD_CHARS + TAIL_CHARS leaves no room to elide;
        such text is returned unchanged (the launch cap equals
        HEAD_CHARS + TAIL_CHARS, so this only concerns future per-tool
        overrides set below 30k).
    """
    if cap <= 0 or len(text) <= cap or len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text, None

    head_pos = HEAD_CHARS
    tail_pos = len(text) - TAIL_CHARS
    # The elided middle is a single line (no raw newline) when the payload is
    # one escaped line - a JSON envelope's stdout/content field, or minified
    # JSON. Char-cut it in place so the line (and any JSON structure around
    # it) survives; a line-boundary cut would back up past the whole payload
    # and elide it entirely, leaving the model an empty shell.
    if "\n" not in text[head_pos:tail_pos]:
        head, tail = text[:head_pos], text[tail_pos:]
        elided = tail_pos - head_pos
        span = f"chars {head_pos + 1}-{tail_pos} of {len(text)}"
        joiner = ""  # marker sits inline, keeping the payload line whole
    else:
        head = _cut(text, HEAD_CHARS)
        tail = _cut(text, TAIL_CHARS, from_end=True)
        elided = len(text) - len(head) - len(tail)
        total_lines = text.count("\n") + 1
        # head ends on a line boundary (complete lines); tail starts on one.
        first_elided = head.count("\n") + 1
        last_elided = total_lines - (tail.count("\n") + 1)
        span = f"lines {first_elided}-{last_elided} of {total_lines}"
        joiner = "\n"  # marker on its own line between complete lines

    spill_path: str | None = None
    try:
        path = _spill_path(tool_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        spill_path = str(path)
    except OSError:
        # Never fail the turn on a spill write failure.
        spill_path = None

    if spill_path:
        marker = (
            f"[... {elided:,} chars elided ({span}) - full output saved to "
            f"{spill_path} for reference]"
        )
    else:
        marker = (
            f"[... {elided:,} chars elided ({span}) - full output "
            f"unavailable (spill write failed)]"
        )

    # Observability: one debug-log event per bounded result (log_event is a
    # no-op when debug mode is off). Here spill_path=None means the spill
    # write failed - this branch only runs for bound results.
    log_event(
        "tool_output_bounded",
        {
            "tool": tool_name,
            "original_chars": len(text),
            "cap": cap,
            "elided_chars": elided,
            "span": span,
            "spill_path": spill_path,
        },
    )
    return head + joiner + marker + joiner + tail, spill_path
