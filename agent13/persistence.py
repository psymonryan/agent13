"""Context persistence for saving and loading conversation state.

Provides save/load functionality for agent message history, enabling:
- Manual save/load via /save and /load commands
- Auto-save on exit for session continuation
- --continue flag to resume from last session
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent13.config_paths import get_global_saves_dir

if TYPE_CHECKING:
    from agent13.core import Agent

_AUTO_SAVE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _ensure_ctx_stem(name: str) -> str:
    """Strip .ctx suffix if present, for safe path construction.

    Prevents double-extension when user types e.g. ``/load foo.ctx``.
    """
    if name.endswith(".ctx"):
        return name[:-4]
    return name


def resolve_save_path(name: str) -> Path:
    """Resolve a user-supplied save name to a .ctx file path.

    Handles three cases:
    - Absolute path (starts with /): use as-is (strip .ctx if present, re-add it)
    - Tilde path (starts with ~): expand user, use as-is
    - Bare name: join with the saves directory

    This prevents the bugs where /save and /load split on spaces
    (dropping the rest of the name) and where ~ is not expanded.

    Args:
        name: The raw user argument (may contain spaces, ~, or be absolute).

    Returns:
        Path to the .ctx file (with .ctx extension ensured).
    """
    name = name.strip()
    if not name:
        raise ValueError("Save name cannot be empty")

    # Absolute path or tilde path — use directly
    if name.startswith("/") or name.startswith("~"):
        path = Path(name).expanduser()
        if path.suffix == ".ctx":
            return path
        # Preserve any other existing extension (e.g. foo.backup -> foo.backup.ctx)
        return path.with_name(f"{path.name}.ctx")

    # Bare name — join with saves directory
    return get_saves_dir() / f"{_ensure_ctx_stem(name)}.ctx"


# Context file format version
CONTEXT_VERSION = 1


def get_saves_dir() -> Path:
    """Get the manual saves directory (project-local).

    Respects AGENT13_SAVES_DIR env var (used by tests for isolation).
    Falls back to ./.agent13/saves/.

    Returns:
        Path to the saves directory, created if needed.
    """
    env_dir = os.environ.get("AGENT13_SAVES_DIR")
    if env_dir:
        saves_dir = Path(env_dir)
    else:
        saves_dir = Path.cwd() / ".agent13" / "saves"
    saves_dir.mkdir(parents=True, exist_ok=True)
    return saves_dir


def get_auto_save_dir() -> Path:
    """Get the auto-save directory.

    Respects config: [saves] location = "central" or "local" (default).
    - central: ~/.agent13/saves/
    - local:   ./.agent13/saves/  (default — see default_config.toml)

    Returns:
        Path to the auto-save directory, created if needed.
    """
    from agent13.config import get_config

    cfg = get_config()
    if cfg.saves_location == "local":
        return get_saves_dir()  # project-local (default)
    return get_global_saves_dir()  # central


def get_auto_save_path(
    project_name: str | None = None, session_date: str | None = None
) -> Path:
    """Get the auto-save path for the current session.

    Respects config: [saves] location = "central" or "local" (default).
    - central: <project>-YYYY-MM-DD.ctx (project prefix needed to distinguish projects)
    - local:   YYYY-MM-DD.ctx (no prefix, directory is already project-specific; default)

    Args:
        project_name: Optional project name. If not provided, uses cwd name.
        session_date: Optional ISO date string for the session start date.
            If None, defaults to today. Using the session date ensures the
            auto-save always writes to the original session's file, even if
            the session spans multiple days.

    Returns:
        Path like ~/.agent13/saves/myproject-2026-04-01.ctx (central)
        or ./.agent13/saves/2026-04-01.ctx (local)
    """
    from agent13.config import get_config

    if project_name is None:
        project_name = Path.cwd().name

    cfg = get_config()
    date_str = session_date or datetime.now().strftime("%Y-%m-%d")

    if cfg.saves_location == "local":
        # Local mode (default): no project prefix needed (directory is project-specific)
        return get_auto_save_dir() / f"{date_str}.ctx"
    else:
        # Central mode: include project prefix to distinguish projects
        return get_auto_save_dir() / f"{project_name}-{date_str}.ctx"


def _is_central_dir(path: Path) -> bool:
    """Check if a path is the central saves directory.

    Args:
        path: Path to check.

    Returns:
        True if this is the central (~/.agent13/saves/) directory.
    """
    return path == get_global_saves_dir()


def _get_auto_save_pattern(project_name: str, is_central: bool) -> str:
    """Get the glob pattern for auto-saves in a given directory type.

    Args:
        project_name: Project name for central mode patterns.
        is_central: True if searching in central directory.

    Returns:
        Glob pattern string.
    """
    if is_central:
        # Central: files have project prefix (project-YYYY-MM-DD.ctx)
        return f"{project_name}-*.ctx"
    else:
        # Local: files are date-only (YYYY-MM-DD.ctx)
        return "*.ctx"


def find_latest_auto_save(project_name: str | None = None) -> Path | None:
    """Find the most recent auto-save file.

    Searches the configured save location first, then falls back to the
    other location if nothing is found. This handles the case where the
    user changed their saves_location config or moved to a different
    project directory.

    Args:
        project_name: Optional project name to filter. If not provided,
            uses cwd name.

    Returns:
        Path to the most recent .ctx file, or None if none exist in
        either location.
    """
    from agent13.config import get_config

    if project_name is None:
        project_name = Path.cwd().name

    cfg = get_config()
    auto_dir = get_auto_save_dir()

    # Primary: configured save location
    primary = _find_latest_in_dir(auto_dir, project_name, _is_central_dir(auto_dir))
    if primary is not None:
        return primary

    # Fallback: try the other location
    if cfg.saves_location == "central":
        # Configured for central; try local (project dir)
        fallback_dir = get_saves_dir()
    else:
        # Configured for local; try central (~/.agent13/saves/)
        fallback_dir = get_global_saves_dir()

    return _find_latest_in_dir(
        fallback_dir, project_name, _is_central_dir(fallback_dir)
    )


def _find_latest_in_dir(
    directory: Path, project_name: str, is_central: bool
) -> Path | None:
    """Find the most recent auto-save in a single directory.

    Globs the directory using the appropriate pattern, filters to actual
    auto-saves, and returns the newest by mtime.

    Args:
        directory: Directory to search (central or local saves dir).
        project_name: Project name used to build the glob pattern.
        is_central: Whether ``directory`` is the central saves dir (affects
            pattern and name validation).

    Returns:
        Path to the most recent .ctx file, or None if none exist.
    """
    pattern = _get_auto_save_pattern(project_name, is_central)
    matches = [
        m
        for m in directory.glob(pattern)
        if _is_auto_save_name_for_dir(m.stem, is_central, project_name)
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _is_incomplete_turn(messages: list) -> bool:
    """Check if the conversation has an incomplete turn.

    A turn is incomplete if:
    - Last message is assistant with tool_calls (tools not yet executed)
    - Last message is tool (results not yet processed by LLM)

    Args:
        messages: List of message dicts.

    Returns:
        True if the turn is incomplete.
    """
    if not messages:
        return False

    last_msg = messages[-1]

    # Case 1: Assistant with pending tool calls
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        return True

    # Case 2: Tool result waiting for LLM to process
    if last_msg.get("role") == "tool":
        return True

    return False


def save_context(agent: "Agent", path: Path | str) -> None:
    """Save agent context to a file.

    Saves messages, model, system_prompt, token usage, and incomplete turn flag.
    Strips reasoning tokens from messages before saving.
    Applies pending compaction if available (without modifying agent state).

    Args:
        agent: The Agent instance to save.
        path: Path to save the context file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compaction now happens immediately in _maybe_reflect_after_turn,
    # so agent.messages always reflects the compacted state.
    # Save messages as-is (including reasoning_content) so that on load
    # the KV cache prefix still matches and avoids a cache miss.
    messages_copy = [dict(msg) for msg in agent.messages]

    context = {
        "version": CONTEXT_VERSION,
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "session_date": getattr(agent, "session_date", None),
        "messages": messages_copy,
        "token_usage": {
            "prompt": agent.prompt_tokens,
            "completion": agent.completion_tokens,
        },
        "saved_at": datetime.now().isoformat(),
        "incomplete_turn": _is_incomplete_turn(messages_copy),
    }

    with open(path, "w") as f:
        json.dump(context, f, indent=2)


def load_context(agent: "Agent", path: Path | str) -> tuple[bool, str, bool]:
    """Load agent context from a file.

    Replaces the agent's messages with the loaded context.
    If the saved context had an incomplete turn, sets agent flag for resume handling.

    Args:
        agent: The Agent instance to load into.
        path: Path to the context file.

    Returns:
        Tuple of (success, message, incomplete_turn).
        If success is False, message contains error.
        incomplete_turn is True if the saved context was mid-turn.
    """
    path = Path(path)

    if not path.exists():
        return False, f"Context file not found: {path}", False

    try:
        with open(path) as f:
            context = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in context file: {e}", False

    # Check version compatibility
    version = context.get("version", 0)
    if version > CONTEXT_VERSION:
        return (
            False,
            f"Context file version {version} is newer than supported {CONTEXT_VERSION}",
            False,
        )

    # Validate required fields
    if "messages" not in context:
        return False, "Context file missing 'messages' field", False

    # Load into agent
    agent.messages = context["messages"]

    # Restore other fields if present
    # Note: We intentionally do NOT restore model - user keeps their current
    # provider/model settings. The saved model is informational only.
    if "system_prompt" in context:
        agent.system_prompt = context["system_prompt"]
    if "session_date" in context and context["session_date"]:
        agent.session_date = context["session_date"]
    if "token_usage" in context:
        agent.prompt_tokens = context["token_usage"].get("prompt", 0)
        agent.completion_tokens = context["token_usage"].get("completion", 0)

    # Check for incomplete turn and set agent flag
    incomplete_turn = context.get("incomplete_turn", False)
    if incomplete_turn:
        agent.mark_incomplete_turn(True)

    return True, f"Loaded context from {path}", incomplete_turn


def list_saves() -> list[Path]:
    """List available manual save files.

    Returns:
        List of paths to .ctx files in the manual saves directory.
    """
    saves_dir = get_saves_dir()
    return sorted(saves_dir.glob("*.ctx"))


def _is_auto_save_name_for_dir(stem: str, is_central: bool, project_name: str) -> bool:
    """Check if a filename stem is an auto-save for a specific directory type.

    Args:
        stem: Filename stem (without .ctx extension).
        is_central: True if checking a file from the central directory.
        project_name: Project name for central mode validation.

    Returns:
        True if this is a valid auto-save name for the given directory type.
    """
    if is_central:
        # Central mode: must have project prefix
        if not stem.startswith(f"{project_name}-"):
            return False
        # Check the date part
        date_part = stem[len(project_name) + 1:]  # Skip "project-"
        return bool(_AUTO_SAVE_RE.match(date_part))
    else:
        # Local mode: must be date-only
        return bool(_AUTO_SAVE_RE.match(stem))


def list_all_saves() -> list[Path]:
    """List all save files (manual + auto) sorted by mtime, newest first.

    Manual saves appear first, auto-saves (date-based names) last.
    Used for /load tab completion.

    Returns:
        List of paths sorted: manual (mtime desc), then auto-saves (mtime desc).
    """
    manual = list_saves()
    auto_dir = get_auto_save_dir()
    saves_dir = get_saves_dir()
    project_name = Path.cwd().name

    auto_saves: list[Path] = []
    if auto_dir == saves_dir:
        # Local mode: auto-saves are in same dir, filter by date-only pattern
        for f in saves_dir.glob("*.ctx"):
            if _is_auto_save_name_for_dir(f.stem, is_central=False, project_name=project_name):
                auto_saves.append(f)
    else:
        # Central mode: filter by project-date pattern
        for f in auto_dir.glob("*.ctx"):
            if _is_auto_save_name_for_dir(f.stem, is_central=True, project_name=project_name):
                auto_saves.append(f)

    # Deduplicate (if local mode and same file somehow)
    manual_set = set(manual)
    auto_saves = [f for f in auto_saves if f not in manual_set]

    # Sort each group by mtime descending
    manual.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    auto_saves.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return manual + auto_saves
