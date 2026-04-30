"""Prompt management for system prompts."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from agent13.config_paths import get_prompts_file
from agent13.yaml_store import load_yaml, save_yaml

DEFAULT_PROMPT = "You are a tool using AI assistant."

if TYPE_CHECKING:
    from agent13.skills import SkillInfo


class PromptManager:
    """Manages system prompts stored in ~/.agent/prompts.yaml

    Prompts are stored in YAML format with prompt names as keys.
    The active prompt is used for system messages in conversations.
    """

    def __init__(self, config_path: str = None):
        """Initialize prompt manager.

        Args:
            config_path: Path to prompts YAML file (defaults to ~/.agent/prompts.yaml).
        """
        self.config_path = Path(config_path) if config_path else get_prompts_file()
        self.prompts: dict[str, str] = {}
        self.active_prompt: str = "default"
        self.custom_additions: list[str] = []
        self.load_prompts()

    def load_prompts(self) -> None:
        """Load prompts from config file."""
        self.prompts = load_yaml(self.config_path)
        # Ensure it's a dict of strings
        if not isinstance(self.prompts, dict):
            self.prompts = {}

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
            additions = "\\n\\n".join(self.custom_additions)
            return f"{base}\\n\\n{additions}"
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
