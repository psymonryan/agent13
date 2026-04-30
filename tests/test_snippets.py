"""Tests for agent13.snippets module."""

import os
from agent13.snippets import SnippetManager


class TestSnippetManager:
    """Tests for SnippetManager class."""

    def test_create_manager(self, temp_dir):
        """Should create manager with default path."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        assert sm.config_path.name == "snippets.yaml"
        assert sm.snippets == {}

    def test_add_snippet(self, temp_dir):
        """Should add a snippet and persist to file."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        result = sm.add_snippet("journal", "Create a journal document")
        assert result is None  # No collision warning
        assert sm.get_snippet("journal") == "Create a journal document"

        # Should persist to file
        sm2 = SnippetManager(config_path=path)
        assert sm2.get_snippet("journal") == "Create a journal document"

    def test_add_snippet_collision(self, temp_dir):
        """Should return collision warning for reserved name."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path, reserved_names={"quit", "help"})
        result = sm.add_snippet("quit", "Quit the app")
        assert result is not None
        assert "conflicts" in result
        assert "quit" in result
        # Snippet should still be saved
        assert sm.get_snippet("quit") == "Quit the app"

    def test_delete_snippet(self, temp_dir):
        """Should delete a snippet."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        sm.add_snippet("journal", "Create a journal document")
        assert sm.delete_snippet("journal") is True
        assert sm.get_snippet("journal") is None

    def test_delete_nonexistent(self, temp_dir):
        """Should return False for nonexistent snippet."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        assert sm.delete_snippet("nonexistent") is False

    def test_rename_snippet(self, temp_dir):
        """Should rename a snippet."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        sm.add_snippet("journal", "Create a journal document")
        result = sm.rename_snippet("journal", "dev-journal")
        assert result is None
        assert sm.get_snippet("journal") is None
        assert sm.get_snippet("dev-journal") == "Create a journal document"

    def test_rename_nonexistent(self, temp_dir):
        """Should return error for nonexistent source."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        result = sm.rename_snippet("nonexistent", "new-name")
        assert result is not None
        assert "not found" in result

    def test_rename_collision(self, temp_dir):
        """Should return collision warning for reserved new name."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path, reserved_names={"quit"})
        sm.add_snippet("journal", "Create a journal document")
        result = sm.rename_snippet("journal", "quit")
        assert result is not None
        assert "conflicts" in result
        # New name should be saved
        assert sm.get_snippet("quit") == "Create a journal document"
        assert sm.get_snippet("journal") is None

    def test_validate_name_valid(self, temp_dir):
        """Should accept valid names."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        assert sm.validate_name("journal") is None
        assert sm.validate_name("create-journal") is None
        assert sm.validate_name("create_journal") is None
        assert sm.validate_name("snippet123") is None
        assert sm.validate_name("ABC") is None

    def test_validate_name_invalid(self, temp_dir):
        """Should reject invalid names."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        assert sm.validate_name("") is not None
        assert sm.validate_name("my snippet") is not None  # spaces
        assert sm.validate_name("my.snippet") is not None  # dots
        assert sm.validate_name("my:snippet") is not None  # colons

    def test_list_snippets(self, temp_dir):
        """Should list snippets with preview and collision status."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path, reserved_names={"quit"})
        sm.add_snippet("journal", "Create a journal document")
        sm.add_snippet("quit", "Quit the app")
        result = sm.list_snippets()
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"journal", "quit"}
        # Check collision flag
        journal = next(s for s in result if s["name"] == "journal")
        assert journal["collision"] is False
        quit_snippet = next(s for s in result if s["name"] == "quit")
        assert quit_snippet["collision"] is True

    def test_load_collision_warnings(self, temp_dir):
        """Should return collision warnings on load."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path, reserved_names={"quit"})
        sm.add_snippet("quit", "Quit the app")
        # Reload and check warnings
        sm2 = SnippetManager(config_path=path, reserved_names={"quit"})
        warnings = sm2.load_snippets()
        assert len(warnings) == 1
        assert "quit" in warnings[0]

    def test_overwrite_with_add(self, temp_dir):
        """Should overwrite existing snippet with add."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        sm.add_snippet("journal", "Old content")
        sm.add_snippet("journal", "New content")
        assert sm.get_snippet("journal") == "New content"

    def test_repr(self, temp_dir):
        """Should have useful repr."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        sm.add_snippet("journal", "Create journal")
        sm.add_snippet("review", "Review code")
        assert "2 snippets" in repr(sm)

    def test_load_corrupt_file(self, temp_dir):
        """Should handle corrupt YAML gracefully."""
        path = os.path.join(temp_dir, "snippets.yaml")
        with open(path, "w") as f:
            f.write("{{{{invalid yaml")
        sm = SnippetManager(config_path=path)
        assert sm.snippets == {}

    def test_load_non_dict_file(self, temp_dir):
        """Should handle non-dict YAML gracefully."""
        path = os.path.join(temp_dir, "snippets.yaml")
        with open(path, "w") as f:
            f.write("- item1\n- item2\n")
        sm = SnippetManager(config_path=path)
        assert sm.snippets == {}

    def test_preview_truncation(self, temp_dir):
        """Should truncate long previews."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path)
        long_content = "x" * 200
        sm.add_snippet("long", long_content)
        result = sm.list_snippets()
        assert len(result) == 1
        assert result[0]["preview"].endswith("...")
        assert len(result[0]["preview"]) <= 83  # 80 + "..."

    def test_delete_removes_collision(self, temp_dir):
        """Should remove from collisions list on delete."""
        path = os.path.join(temp_dir, "snippets.yaml")
        sm = SnippetManager(config_path=path, reserved_names={"quit"})
        sm.add_snippet("quit", "Quit the app")
        assert "quit" in sm._collisions
        sm.delete_snippet("quit")
        assert "quit" not in sm._collisions
