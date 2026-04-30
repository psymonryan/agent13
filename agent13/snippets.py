"""Snippet management for saved user messages."""

from __future__ import annotations

import re
from pathlib import Path

from agent13.config_paths import get_snippets_file
from agent13.yaml_store import load_yaml, save_yaml


# Valid snippet names: alphanumeric, underscores, hyphens only
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class SnippetManager:
    """Manages user snippets stored in ~/.agent13/snippets.yaml

    Snippets are saved user messages that can be invoked via slash commands.
    Unlike prompts (which are system messages), snippets are injected as
    user messages — they represent the user's voice, not instructions to the AI.
    """

    def __init__(
        self, config_path: str | None = None, reserved_names: set[str] | None = None
    ):
        """Initialize snippet manager.

        Args:
            config_path: Path to snippets YAML file
                (defaults to ~/.agent13/snippets.yaml).
            reserved_names: Set of command names that are reserved
                (built-in slash commands). Used for collision detection.
        """
        self.config_path = Path(config_path) if config_path else get_snippets_file()
        self.reserved_names = reserved_names or set()
        self.snippets: dict[str, str] = {}
        self._collisions: list[str] = []
        self.load_snippets()

    def load_snippets(self) -> list[str]:
        """Load snippets from config file.

        Returns:
            List of collision warning strings for names that conflict
            with reserved (built-in) command names.
        """
        self.snippets = load_yaml(self.config_path)
        # Ensure it's a dict of strings
        if not isinstance(self.snippets, dict):
            self.snippets = {}
        # Filter to string values only
        self.snippets = {
            k: v
            for k, v in self.snippets.items()
            if isinstance(k, str) and isinstance(v, str)
        }

        # Check for collisions with reserved names
        self._collisions = []
        warnings = []
        for name in self.snippets:
            if name in self.reserved_names:
                self._collisions.append(name)
                warnings.append(
                    f"Snippet '{name}' conflicts with built-in '/{name}' "
                    f"command. Use /snippet use {name} to invoke it."
                )
        return warnings

    def save_snippets(self) -> None:
        """Save snippets to config file."""
        save_yaml(self.config_path, self.snippets)

    def get_snippet(self, name: str) -> str | None:
        """Get a snippet by name.

        Args:
            name: Snippet name.

        Returns:
            The snippet content, or None if not found.
        """
        return self.snippets.get(name)

    def validate_name(self, name: str) -> str | None:
        """Validate a snippet name.

        Args:
            name: Proposed snippet name.

        Returns:
            Error message if invalid, None if valid.
        """
        if not name:
            return "Snippet name cannot be empty"
        if not _NAME_PATTERN.match(name):
            return (
                f"Invalid snippet name '{name}': "
                "use only letters, numbers, underscores, and hyphens"
            )
        return None

    def add_snippet(self, name: str, content: str) -> str | None:
        """Add or update a snippet.

        Args:
            name: Snippet name.
            content: Snippet message text.

        Returns:
            Collision warning string if name conflicts with a reserved
            command, None otherwise. The snippet is always saved
            (collisions are warnings, not errors).
        """
        self.snippets[name] = content
        self.save_snippets()

        # Check collision
        if name in self.reserved_names:
            if name not in self._collisions:
                self._collisions.append(name)
            return (
                f"Snippet '{name}' conflicts with built-in '/{name}' "
                f"command. Use /snippet use {name} to invoke it."
            )
        return None

    def rename_snippet(self, old_name: str, new_name: str) -> str | None:
        """Rename a snippet.

        Args:
            old_name: Current snippet name.
            new_name: New snippet name.

        Returns:
            Error message if rename failed, collision warning string
            if new name conflicts, None on success.
        """
        if old_name not in self.snippets:
            return f"Snippet not found: {old_name}"

        validation_error = self.validate_name(new_name)
        if validation_error:
            return validation_error

        content = self.snippets.pop(old_name)
        self.snippets[new_name] = content
        self.save_snippets()

        # Update collision tracking
        if old_name in self._collisions:
            self._collisions.remove(old_name)

        # Check collision for new name
        if new_name in self.reserved_names:
            if new_name not in self._collisions:
                self._collisions.append(new_name)
            return (
                f"Snippet '{new_name}' conflicts with built-in '/{new_name}' "
                f"command. Use /snippet use {new_name} to invoke it."
            )
        return None

    def delete_snippet(self, name: str) -> bool:
        """Delete a snippet.

        Args:
            name: Snippet name to delete.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self.snippets:
            return False

        del self.snippets[name]
        if name in self._collisions:
            self._collisions.remove(name)
        self.save_snippets()
        return True

    def list_snippets(self) -> list[dict]:
        """List all snippets with preview and collision status.

        Returns:
            List of dicts with keys: name, preview, collision.
        """
        result = []
        for name, content in self.snippets.items():
            first_line = content.split("\n", 1)[0]
            preview = first_line[:80] + "..." if len(first_line) > 80 else first_line
            result.append(
                {
                    "name": name,
                    "preview": preview,
                    "collision": name in self._collisions,
                }
            )
        return result

    def __repr__(self) -> str:
        return f"SnippetManager({len(self.snippets)} snippets)"
