"""Tests for agent.prompts module."""

import os
from agent13 import PromptManager


class TestPromptManager:
    """Tests for PromptManager class."""

    def test_create_manager(self, temp_dir):
        """Should create manager with default path."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        assert pm.config_path.name == "prompts.yaml"

    def test_default_prompt_exists(self, temp_dir):
        """Should always have a default prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)

        assert "default" in pm.prompts
        assert pm.get_prompt("default") is not None

    def test_get_prompt(self, temp_dir):
        """Should get prompt by name."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")

        assert pm.get_prompt("coder") == "You are an expert coder."

    def test_get_active_prompt(self, temp_dir):
        """Should get active prompt when no name specified."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")
        pm.set_active("coder")

        assert pm.get_prompt() == "You are an expert coder."

    def test_get_nonexistent_prompt(self, temp_dir):
        """Should return default for nonexistent prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)

        assert pm.get_prompt("nonexistent") == pm.get_prompt("default")

    def test_set_active(self, temp_dir):
        """Should set active prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")

        result = pm.set_active("coder")

        assert result is True
        assert pm.active_prompt == "coder"

    def test_set_active_nonexistent(self, temp_dir):
        """Should return False for nonexistent prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)

        result = pm.set_active("nonexistent")

        assert result is False

    def test_add_prompt(self, temp_dir):
        """Should add new prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("writer", "You are a creative writer.")

        assert "writer" in pm.prompts
        assert pm.prompts["writer"] == "You are a creative writer."

    def test_update_prompt(self, temp_dir):
        """Should update existing prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "Original")
        pm.add_prompt("coder", "Updated")

        assert pm.prompts["coder"] == "Updated"

    def test_delete_prompt(self, temp_dir):
        """Should delete prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")

        result = pm.delete_prompt("coder")

        assert result is True
        assert "coder" not in pm.prompts

    def test_delete_nonexistent_prompt(self, temp_dir):
        """Should return False for nonexistent prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)

        result = pm.delete_prompt("nonexistent")

        assert result is False

    def test_delete_default_prompt(self, temp_dir):
        """Should not delete default prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)

        result = pm.delete_prompt("default")

        assert result is False
        assert "default" in pm.prompts

    def test_delete_active_prompt_resets_to_default(self, temp_dir):
        """Should reset to default when deleting active prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")
        pm.set_active("coder")

        pm.delete_prompt("coder")

        assert pm.active_prompt == "default"

    def test_append_to_active(self, temp_dir):
        """Should append to active prompt."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.append_to_active("Be concise.")
        pm.append_to_active("Use examples.")

        message = pm.build_system_message()

        assert pm.get_prompt() in message
        assert "Be concise." in message
        assert "Use examples." in message

    def test_clear_additions(self, temp_dir):
        """Should clear custom additions."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.append_to_active("Be concise.")

        pm.clear_additions()

        message = pm.build_system_message()
        assert message == pm.get_prompt()

    def test_build_system_message(self, temp_dir):
        """Should build system message with additions."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")
        pm.set_active("coder")
        pm.append_to_active("Focus on Python.")

        message = pm.build_system_message()

        assert message.startswith("You are an expert coder.")
        assert "Focus on Python." in message

    def test_list_prompts(self, temp_dir):
        """Should list all prompts with metadata."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")
        pm.add_prompt("writer", "You are a creative writer.")
        pm.set_active("coder")

        prompts = pm.list_prompts()

        assert len(prompts) == 3  # default, coder, writer
        coder = next(p for p in prompts if p["name"] == "coder")
        assert coder["active"] is True
        writer = next(p for p in prompts if p["name"] == "writer")
        assert writer["active"] is False

    def test_list_prompts_preview_truncation(self, temp_dir):
        """Should truncate long prompts in preview."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        long_prompt = "x" * 200
        pm.add_prompt("long", long_prompt)

        prompts = pm.list_prompts()
        long_entry = next(p for p in prompts if p["name"] == "long")

        assert len(long_entry["preview"]) == 103  # 100 + "..."
        assert long_entry["preview"].endswith("...")

    def test_persistence(self, temp_dir):
        """Should persist prompts to file."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm1 = PromptManager(path)
        pm1.add_prompt("coder", "You are an expert coder.")

        # Create new manager with same path
        pm2 = PromptManager(path)

        assert "coder" in pm2.prompts
        assert pm2.prompts["coder"] == "You are an expert coder."

    def test_repr(self, temp_dir):
        """Should return string representation."""
        path = os.path.join(temp_dir, "prompts.yaml")
        pm = PromptManager(path)
        pm.add_prompt("coder", "You are an expert coder.")

        repr_str = repr(pm)

        assert "PromptManager" in repr_str
        assert "prompts=2" in repr_str  # default + coder
