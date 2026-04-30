"""Context persistence for saving and loading conversation state.

Provides save/load functionality for agent message history, enabling:
- Manual save/load via /save and /load commands
- Auto-save on exit for session continuation
- --continue flag to resume from last session
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent13.config_paths import get_saves_dir as _get_global_saves_dir

if TYPE_CHECKING:
    from agent13.core import Agent

# Context file format version
CONTEXT_VERSION = 1


def get_saves_dir() -> Path:
    """Get the manual saves directory (project-local).

    Returns:
        Path to ./.agent13/saves/, created if needed.
    """
    saves_dir = Path.cwd() / ".agent13" / "saves"
    saves_dir.mkdir(parents=True, exist_ok=True)
    return saves_dir


def get_auto_save_dir() -> Path:
    """Get the auto-save directory (global).

    Returns:
        Path to ~/.agent13/saves/, created if needed.
    """
    return _get_global_saves_dir()


def get_auto_save_path(project_name: str | None = None) -> Path:
    """Get the auto-save path for the current session.

    Args:
        project_name: Optional project name. If not provided, uses cwd name.

    Returns:
        Path like ~/.agent13/saves/myproject-2026-04-01.ctx
    """
    if project_name is None:
        project_name = Path.cwd().name

    date_str = datetime.now().strftime("%Y-%m-%d")
    return get_auto_save_dir() / f"{project_name}-{date_str}.ctx"


def find_latest_auto_save(project_name: str | None = None) -> Path | None:
    """Find the most recent auto-save file.

    Args:
        project_name: Optional project name to filter. If not provided, uses cwd name.

    Returns:
        Path to the most recent .ctx file, or None if none exist.
    """
    if project_name is None:
        project_name = Path.cwd().name

    auto_dir = get_auto_save_dir()
    pattern = f"{project_name}-*.ctx"

    matches = list(auto_dir.glob(pattern))
    if not matches:
        return None

    # Sort by modification time, most recent first
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
        List of paths to .ctx files in the saves directory.
    """
    saves_dir = get_saves_dir()
    return sorted(saves_dir.glob("*.ctx"))
