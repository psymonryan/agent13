"""Prompt management for system prompts."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from agent13.config_paths import get_prompts_file, ensure_config_dir
from agent13.yaml_store import load_yaml, save_yaml

DEFAULT_PROMPT = "You are a tool using AI assistant."

REFLECTION_PROMPT = (
    "Since you have just used tools, tersely reflect on each one, then stop.\n"
    "- what was your goal when calling the tools\n"
    "- what did you achieve with these calls\n"
    "Skip where the goal was not achieved"
)

PRIMING_PROMPT = (
    "The previous turns have been journalled to reduce context. "
    "From now on, actually call the tools rather than reflecting. Dont reflect again until asked. "
    "Acknowledge this imperative with an `ok`"
)

PRIMING_RESPONSE = "ok"

# Injected as a user message after an auto-compact so the model resumes the
# in-progress turn (finishing the original task) instead of stopping.
AUTO_COMPACT_CONTINUE_HINT = (
    "[Context was compacted to fit. Continue with your previous task and complete it.]"
)

JOURNAL_USER_MESSAGE_PREFIX = "[previous user message]"
JOURNAL_USER_MESSAGE = f'{JOURNAL_USER_MESSAGE_PREFIX} "{{original}}"'

# Default compaction prompt for /compact command.
# Users can override by adding a "compaction" entry to prompts.yaml
# or by passing a named prompt: /compact --prompt <name>
DEFAULT_COMPACT_PROMPT = (
    "Summarize our conversation so far into a concise but complete context summary.\n"
    "Preserve:\n"
    "- All key decisions, their rationale, and current status\n"
    "- Important code, file paths, and technical details\n"
    "- New knowledge gained by doing: how to connect to hosts/services,\n"
    "  exact commands that work, environment quirks and gotchas\n"
    "- Open questions and unresolved issues\n"
    "- The current direction/next steps\n"
    "Skip pleasantries and hedging. Write as a direct reference document\n"
    "that lets you continue the work seamlessly.\n"
    "If you'd have to re-discover it from scratch, it belongs in the summary."
)

# Appended to the base compaction prompt when /compact is given a focus
# ("next task") string. Steers the summary sections and next steps toward
# the upcoming work and allows harder compression of unrelated detail.
COMPACT_STEERING_TEMPLATE = (
    "\nNext task: {focus}\n"
    "- Organize the summary sections around that task\n"
    "- Make \"next steps\" concrete actions toward it\n"
    "- Details clearly unrelated to it can be compressed harder"
)


def resolve_compact_prompt(prompt_manager, arg: str) -> tuple:
    """Resolve the /compact argument into the compaction prompt to send.

    Shared by the REPL, headless, and TUI command handlers so all
    interfaces accept the same syntax:

    - (no arg)        → the 'compaction' prompt from prompts.yaml, or
                        DEFAULT_COMPACT_PROMPT if absent
    - --prompt <name> → swap in a named prompt (existing behavior)
    - <free text>     → base prompt + steering block focusing the summary
                        on the user's next task

    Args:
        prompt_manager: PromptManager for prompt lookups.
        arg: Raw argument text after /compact (may be empty).

    Returns:
        (prompt_text, error) — exactly one is None.
    """
    arg = arg.strip()
    if arg.startswith("--prompt"):
        prompt_name = arg[len("--prompt"):].strip()
        if not prompt_name:
            return None, (
                "Usage: /compact --prompt <name>\n"
                f"Available: {', '.join(prompt_manager.prompts.keys())}"
            )
        candidate = prompt_manager.get_prompt(prompt_name)
        if candidate == prompt_manager.get_prompt("default") and prompt_name != "default":
            return None, (
                f"Prompt '{prompt_name}' not found\n"
                f"Available: {', '.join(prompt_manager.prompts.keys())}"
            )
        return candidate, None
    base = prompt_manager.prompts.get("compaction", DEFAULT_COMPACT_PROMPT)
    if arg:
        return base + COMPACT_STEERING_TEMPLATE.format(focus=arg), None
    return base, None


# The lightweight user message that replaces the full compaction prompt
# in history after compaction. Small so it doesn't re-bloat context.
COMPACT_REPLACEMENT_MESSAGE = "Give me a summary of our previous session"

# Default prompts bundled with the package
DEFAULT_PROMPTS_FILE = Path(__file__).parent / "default_prompts.yaml"

if TYPE_CHECKING:
    from agent13.skills import SkillInfo


def ensure_default_prompts() -> None:
    """Copy default prompts to user's config directory if they don't exist.

    This provides starter prompts for new users.
    """
    prompts_file = get_prompts_file()

    # If prompts already exist, don't overwrite
    if prompts_file.exists():
        return

    # Check if we have a default prompts file to copy
    if not DEFAULT_PROMPTS_FILE.exists():
        return

    # Ensure config directory exists
    ensure_config_dir()

    # Copy default prompts
    try:
        prompts_file.write_text(DEFAULT_PROMPTS_FILE.read_text())
    except OSError as e:
        # Log warning but don't fail
        import logging

        logging.getLogger(__name__).warning("Failed to copy default prompts: %s", e)


class PromptManager:
    """Manages system prompts stored in ~/.agent13/prompts.yaml

    Prompts are stored in YAML format with prompt names as keys.
    The active prompt is used for system messages in conversations.
    """

    def __init__(self, config_path: str = None):
        """Initialize prompt manager.

        Args:
            config_path: Path to prompts YAML file (defaults to ~/.agent13/prompts.yaml).
        """
        self.config_path = Path(config_path) if config_path else get_prompts_file()
        self.prompts: dict[str, str] = {}
        self.active_prompt: str = "default"
        self.custom_additions: list[str] = []
        self.load_prompts()

    def load_prompts(self) -> None:
        """Load prompts from config file.

        Raises:
            yaml.YAMLError: If prompts file exists but is invalid YAML.
            ValueError: If prompts file exists but has wrong structure
                or contains non-string values.
        """
        self.prompts = load_yaml(self.config_path)
        # load_yaml already validates top-level is a dict (or missing → {})
        # Now strictly validate all values are strings
        for key, value in self.prompts.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"Prompt '{key}' in {self.config_path} must be a "
                    f"string, got {type(value).__name__}"
                )

        # Ensure default exists
        if "default" not in self.prompts:
            self.prompts["default"] = DEFAULT_PROMPT

    def save_prompts(self) -> None:
        """Save prompts to config file."""
        save_yaml(self.config_path, self.prompts)

    def get_prompt(self, name: str = None) -> str:
        """Get a prompt by name, or the active prompt.

        Args:
            name: Prompt name, or None for active prompt.

        Returns:
            The prompt content.
        """
        name = name or self.active_prompt
        return self.prompts.get(name, self.prompts.get("default", DEFAULT_PROMPT))

    def set_active(self, name: str) -> bool:
        """Set the active prompt.

        Args:
            name: Name of the prompt to activate.

        Returns:
            True if the prompt exists and was activated.
        """
        if name in self.prompts:
            self.active_prompt = name
            return True
        return False

    def add_prompt(self, name: str, content: str) -> None:
        """Add or update a prompt.

        Args:
            name: Prompt name.
            content: Prompt content.
        """
        self.prompts[name] = content
        self.save_prompts()

    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt.

        Args:
            name: Name of the prompt to delete.

        Returns:
            True if the prompt was deleted.
        """
        if name in self.prompts and name != "default":
            del self.prompts[name]
            if self.active_prompt == name:
                self.active_prompt = "default"
            self.save_prompts()
            return True
        return False

    def append_to_active(self, addition: str) -> None:
        """Add temporary content to the active prompt.

        Args:
            addition: Text to append to the system message.
        """
        self.custom_additions.append(addition)

    def clear_additions(self) -> None:
        """Clear temporary prompt additions."""
        self.custom_additions.clear()

    def build_system_message(self) -> str:
        """Build the complete system message.

        Returns:
            The active prompt with any custom additions.
        """
        base = self.get_prompt()
        if self.custom_additions:
            additions = "\n\n".join(self.custom_additions)
            return f"{base}\n\n{additions}"
        return base

    def list_prompts(self) -> list[dict]:
        """List all available prompts.

        Returns:
            List of dicts with name, active status, and preview.
        """
        return [
            {
                "name": name,
                "active": name == self.active_prompt,
                "preview": content[:100] + "..." if len(content) > 100 else content,
            }
            for name, content in self.prompts.items()
        ]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromptManager(path={self.config_path!r}, prompts={len(self.prompts)}, active={self.active_prompt!r})"


def get_skills_section(skills: dict[str, "SkillInfo"]) -> str:
    """Generate the skills section for the system prompt.

    Args:
        skills: Dictionary of skill name to SkillInfo

    Returns:
        Formatted skills section string, or empty string if no skills
    """
    if not skills:
        return ""

    lines = [
        "# Available Skills",
        "",
        "You have access to skills for specialized workflows. When a task matches",
        "a skill's description, use the `skill` tool to load its full instructions.",
        "",
        "<available_skills>",
    ]

    for name, info in sorted(skills.items()):
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{info.description}</description>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)
