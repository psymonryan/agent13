"""Mind-map tools - read, edit, create and check an iThoughts (.itmz) map.

Thin wrappers over ``agent13.mindmap``, which owns the format, and
``agent13.mindmap_check``, which owns diagnosis. Models choose by name, and a
``mode`` argument would turn the first decision into a schema-reading exercise.

The check also rides along: every tool that touches a map appends its findings
as a trailer, so verification does not depend on the model deciding to look.
``check_map`` exists anyway as the explicit on-demand validation - the trailer
only fires on a tool call, and link breakage happens OUTSIDE the tool (a
folder rename), so "check without writing" is a real request, not a free ride.

Visibility: default-visible, because the tool is inert without an explicit
path. A session that does not want it can add ``disabled_tools = ["*map*"]``
to ``[ui]``/the global tool filter.
"""

from pathlib import Path
from typing import Optional

from tools import ToolResult, tool
from tools.security import (
    get_current_sandbox_mode,
    validate_path_for_read,
    validate_path_for_write,
)
from agent13.mindmap import (
    QUIET_SECONDS_DEFAULT,
    STORE,
    MindMap,
    MindMapError,
    _preview,
    default_style,
    read_style,
)
from agent13.mindmap_check import check as _check


def _sandbox_mode() -> str:
    mode = get_current_sandbox_mode()
    return mode.value if mode else "default"


def _quiet_seconds() -> int:
    """``[tools] mindmap_quiet_seconds`` - 0 disables the unsaved-edit guard."""
    try:
        from agent13.config import get_config

        return int(get_config().tools_mindmap_quiet_seconds)
    except Exception:
        return QUIET_SECONDS_DEFAULT


def _resolve_path(filepath: str) -> Path:
    return Path(filepath).expanduser().resolve()


EDIT_CHECK_NOTE = (
    "a finding may pre-date this write; no op in this tool can repair a position."
)


def _with_check(text: str, path: Path, extra: str = "") -> str:
    """Append the check trailer. A clean map leaves ``text`` unchanged."""
    trailer = _check(path).trailer(extra=extra)
    return f"{text}\n\n{trailer}" if trailer else text


def _style_bytes(style_path: Optional[Path]) -> bytes:
    """``style_from`` -> ``[tools] mindmap_style_template`` -> bundled default."""
    if style_path is not None:
        return read_style(style_path)
    configured = ""
    try:
        from agent13.config import get_config

        configured = (get_config().tools_mindmap_style_template or "").strip()
    except Exception:
        configured = ""
    if configured:
        return read_style(_resolve_path(configured))
    return default_style()


@tool
def read_map(
    filepath: str,
    topic: str = "",
    depth: int = 1,
    query: str = "",
    text_offset: int = 0,
) -> str:
    """Read an iThoughts mind map (.itmz). Root + immediate children by default.

    Use a map for STRUCTURE - plans, decisions, outlines, the shape of a
    problem. Not for evidence: logs, measurements, code and tables belong in
    files, because a .itmz is a ZIP that cannot be diffed, merged or grepped
    mechanically.

    Only call this when the user names a .itmz file (or asks about a map).
    Long prose and pasted documents are fine in a topic - read them with
    depth=0 and text_offset if the text is large.

    Args:
        filepath: Path to the .itmz file
        topic: Address from a previous read (the 8 chars in [brackets]), or a
               topic's exact text. Omit for the root.
        depth: Levels to return. 0 = the topic only, 1 = the topic and its
               children (default).
        query: Search the whole map for this text (case-insensitive) and
               return matching addresses with context instead of a tree.
        text_offset: Character offset into the topic's own text, for reading a
               long topic in windows. Use with depth=0.

    Returns:
        The tree or the matches. An over-budget request is REFUSED with the
        count and the drill options, never truncated - a truncated tree reads
        as an empty branch. A long topic's text is paged, with the offset to
        continue from.
    """
    if "\0" in filepath:
        return {"error": "Invalid path: embedded null character."}

    path = _resolve_path(filepath)
    allowed, error = validate_path_for_read(str(path))
    if not allowed:
        return {"error": error, "sandbox_mode": _sandbox_mode()}

    try:
        mind_map = MindMap.open(path)
    except MindMapError as exc:
        return {"error": str(exc)}

    try:
        # ToolResult, not a bare str: execute_tool json.dumps a str, which
        # would hand the model a quoted string with escaped newlines.
        text = mind_map.render(
            topic=topic or None, depth=depth, query=query, text_offset=text_offset
        )
    except MindMapError as exc:
        # The over-budget refusal is where a check matters most: a huge foreign
        # map the model cannot read. The findings ride along with the refusal.
        return {"error": _with_check(str(exc), path)}

    return ToolResult(text=_with_check(text, path))


@tool
def edit_map(filepath: str, ops: list, expect_version: str = "") -> str:
    """Edit an iThoughts mind map (.itmz). A batch of ops, ONE atomic write.

    All ops are applied or none: a failure names the op index and writes
    nothing, so retrying the whole batch is safe. Pass expect_version (the
    "version:" line from read_map) so a map that changed since you read it is
    refused instead of clobbered.

    Ops - each an object with "op":
      {"op": "add_branch", "parent": "4F76A79F", "outline": "..."}
          The workhorse. outline is an indented outline: one topic per line,
          tabs (or spaces) for depth. A leading "- ", "* " or "+ " is treated
          as a bullet and stripped; start the line with a backslash to keep it.
      {"op": "replace_branch", "topic": "4F76A79F", "outline": "..."}
          Replaces the topic's CHILDREN. The topic's own text is untouched -
          use rename for that, and do not repeat the topic's text as the
          first outline line.
      {"op": "rename", "topic": "4F76A79F", "text": "New text"}
      {"op": "move", "topic": "4F76A79F", "parent": "3B5F5DCB", "index": 0}
      {"op": "delete", "topic": "4F76A79F", "keep_children": false}
      {"op": "set_note", "topic": "4F76A79F", "text": "..."}
      {"op": "set_link", "topic": "4F76A79F", "url": "https://..."}
          url is stored verbatim - any scheme, relative path or /Local/...
          is fine. "" or null removes the link.

    Args:
        filepath: Path to the .itmz file
        ops: The batch, applied in order
        expect_version: The "version:" value from read_map (recommended)

    Returns:
        What was written, the new version, and the address of every topic
        created - so you do not have to re-read to learn them.
    """
    if "\0" in filepath:
        return {"error": "Invalid path: embedded null character."}

    path = _resolve_path(filepath)
    allowed, error = validate_path_for_write(str(path))
    if not allowed:
        return {"error": error, "sandbox_mode": _sandbox_mode()}

    # Read the baseline BEFORE opening: MindMap.open records one, and a
    # baseline this call created itself would silently satisfy the check.
    baseline = STORE.baseline(path)
    try:
        mind_map = MindMap.open(path)
    except MindMapError as exc:
        return {"error": str(exc)}

    expect = (expect_version or "").strip().lower()
    if not expect and baseline is not None:
        expect = baseline.short
    if not expect:
        return {
            "error": (
                f"no version to check {path} against. Call "
                f'read_map("{path}") first (it prints a "version:" line), or pass '
                "expect_version with the value you already have. Nothing was written."
            )
        }

    try:
        report = mind_map.apply_ops(ops)
        new_version = mind_map.save(expect_short=expect, quiet_seconds=_quiet_seconds())
    except MindMapError as exc:
        return {"error": str(exc)}

    lines = [f"wrote {path}  (version {expect} -> {new_version.short})"]
    lines.extend(report)
    lines.append(f"new version: {new_version.short}")
    return ToolResult(text=_with_check("\n".join(lines), path, extra=EDIT_CHECK_NOTE))


@tool
def create_map(
    filepath: str,
    root_text: str = "Central Topic",
    outline: str = "",
    style_from: str = "",
) -> str:
    """Create a new iThoughts mind map (.itmz). Refuses to overwrite.

    Only when the user asks for a NEW map - use edit_map to change an existing one.

    Args:
        filepath: Where to write. Parent directories are created if missing.
        root_text: The root topic's text.
        outline: Optional. Children of the root, same outline syntax as add_branch.
        style_from: Optional .itmz whose look (style.xml) to copy. Omit for the default.

    Returns:
        The path, the new version, and the address of the root plus every topic
        created from outline - so you do not have to re-read to learn them.
    """
    if "\0" in filepath:
        return {"error": "Invalid path: embedded null character."}

    path = _resolve_path(filepath)
    allowed, error = validate_path_for_write(str(path))
    if not allowed:
        return {"error": error, "sandbox_mode": _sandbox_mode()}

    style_path: Optional[Path] = None
    if style_from:
        style_path = _resolve_path(style_from)
        allowed, error = validate_path_for_read(str(style_path))
        if not allowed:
            return {"error": error, "sandbox_mode": _sandbox_mode()}

    try:
        style = _style_bytes(style_path)
    except MindMapError as exc:
        return {"error": str(exc)}

    made_dirs = not path.parent.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mind_map = MindMap.create(path, root_text=root_text, style=style)
        created = (
            mind_map.add_outline(mind_map.root_topic, outline)
            if outline.strip()
            else []
        )
        version = mind_map.save(create=True)
    except MindMapError as exc:
        return {"error": str(exc)}
    except OSError as exc:
        return {"error": f"could not write {path}: {exc}"}

    lines = [f"created {path}"]
    if made_dirs:
        lines.append(f"  created the missing directory {path.parent}")
    lines.append(
        f"  root {mind_map.address_of(mind_map.root_topic)}  {_preview(root_text)}"
    )
    for el in created:
        lines.append(f"  {mind_map.address_of(el)}  {_preview(el.get('text') or '')}")
    lines.append(f"new version: {version.short}")
    return ToolResult(text=_with_check("\n".join(lines), path))


@tool
def check_map(filepath: str) -> str:
    """Diagnose an iThoughts map (.itmz): structural findings + link validation.

    Validates every topic's link against the LOCAL filesystem: file:// URLs
    (percent-decoded) and relative paths (resolved against the map's own
    directory). Web links (http/https) are counted but NEVER fetched, and
    /Local/... targets are skipped. Nothing is ever written.

    Use this to check WITHOUT writing - e.g. after renaming files or folders
    outside the tool. Broken links are also reported automatically by every
    edit_map save.

    Args:
        filepath: Path to the .itmz file

    Returns:
        The full report: structural findings, then a links summary ("52 local
        checked, all resolve, 17 web not checked"). Each broken link lists the
        topic title, its address, the parent chain, and the target path - fix
        with edit_map set_link.
    """
    if "\0" in filepath:
        return {"error": "Invalid path: embedded null character."}

    path = _resolve_path(filepath)
    allowed, error = validate_path_for_read(str(path))
    if not allowed:
        return {"error": error, "sandbox_mode": _sandbox_mode()}

    # check() reports a missing/unreadable file as a finding, never a raise.
    return ToolResult(text=_check(path).render())
