#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for info pane toggle behavior.

When the info pane is visible and the user presses Enter on empty input,
the pane hides. Pressing Enter again on empty input should reopen the pane
showing the last content — a true toggle.
"""

import inspect
from unittest.mock import MagicMock

import pytest

from ui.tui import AgentTUI


class TestInfoPaneToggle:
    """Info pane should toggle: close on Enter, reopen on next Enter."""

    def test_hide_saves_mode_and_content(self):
        """_hide_info_pane must save the current mode for later restore."""
        source = inspect.getsource(AgentTUI._hide_info_pane)
        assert "_last_info_mode" in source, (
            "_hide_info_pane does not save _last_info_mode.\n"
            "When hiding the pane, the current mode must be saved so "
            "Enter can reopen the same view."
        )

    def test_update_info_content_saves_content(self):
        """_update_info_content must save content for later restore."""
        source = inspect.getsource(AgentTUI._update_info_content)
        assert "_last_info_content" in source, (
            "_update_info_content does not save _last_info_content.\n"
            "The content string must be saved so Enter can reopen "
            "showing the same message."
        )

    def test_empty_submit_has_toggle_logic(self):
        """on_chat_text_area_submitted must reopen pane on empty Enter when hidden."""
        source = inspect.getsource(AgentTUI.on_chat_text_area_submitted)
        assert "_last_info_content" in source, (
            "on_chat_text_area_submitted does not reference _last_info_content.\n"
            "When the pane is hidden and Enter is pressed on empty input, "
            "it should reopen showing the last content."
        )
        assert "_show_info_pane" in source, (
            "on_chat_text_area_submitted does not call _show_info_pane.\n"
            "The toggle reopen path must call _show_info_pane to make "
            "the pane visible again."
        )

    def test_init_has_last_info_fields(self):
        """__init__ must initialize _last_info_content and _last_info_mode."""
        source = inspect.getsource(AgentTUI.__init__)
        assert "_last_info_content" in source, (
            "__init__ does not initialize _last_info_content.\n"
            "Both _last_info_content and _last_info_mode must be "
            "initialized to None."
        )
        assert "_last_info_mode" in source, (
            "__init__ does not initialize _last_info_mode."
        )

    def test_toggle_close_then_reopen_with_mock(self):
        """End-to-end toggle: show content → hide → reopen shows same content.

        Uses a mock for the Textual widgets to test the logic without
        mounting the full app.
        """
        # Create a minimal mock that mimics the relevant TUI state
        app = MagicMock(spec=[
            "_info_pane", "_info_content", "_info_pane_mode",
            "_last_info_content", "_last_info_mode",
            "_show_info_pane", "_hide_info_pane", "_update_info_content",
        ])

        # Simulate initial state: pane hidden, no last content
        app._info_pane = MagicMock()
        app._info_pane.styles.display = "none"
        app._info_content = MagicMock()
        app._info_pane_mode = None
        app._last_info_content = None
        app._last_info_mode = None

        # Wire up _show_info_pane
        def show():
            app._info_pane.styles.display = "block"

        def hide():
            if app._info_pane.styles.display != "none":
                app._last_info_mode = app._info_pane_mode
            app._info_pane.styles.display = "none"
            app._info_pane_mode = None

        def update_content(content):
            app._info_content.update(content)
            app._last_info_content = content
            show()

        app._show_info_pane = show
        app._hide_info_pane = hide
        app._update_info_content = update_content

        # Step 1: Show some content
        AgentTUI._update_info_content(app, "[green]Success![/]")
        assert app._info_pane.styles.display == "block"
        assert app._last_info_content == "[green]Success![/]"

        # Step 2: Hide (simulates Enter on empty input while visible)
        AgentTUI._hide_info_pane(app)
        assert app._info_pane.styles.display == "none"
        assert app._info_pane_mode is None
        # Content and mode should be preserved
        assert app._last_info_content == "[green]Success![/]"

        # Step 3: Reopen (simulates Enter on empty input while hidden)
        # This is the toggle logic from on_chat_text_area_submitted
        if app._info_pane.styles.display != "none":
            hide()
        elif app._last_info_content is not None:
            app._info_pane_mode = app._last_info_mode
            app._info_content.update(app._last_info_content)
            show()

        assert app._info_pane.styles.display == "block"
        assert app._last_info_content == "[green]Success![/]"

        # Step 4: Close again (toggle back)
        if app._info_pane.styles.display != "none":
            hide()

        assert app._info_pane.styles.display == "none"

    def test_toggle_with_named_mode(self):
        """Toggle preserves named modes (history, help, status)."""
        app = MagicMock(spec=[
            "_info_pane", "_info_content", "_info_pane_mode",
            "_last_info_content", "_last_info_mode",
            "_show_info_pane", "_hide_info_pane", "_update_info_content",
        ])

        app._info_pane = MagicMock()
        app._info_pane.styles.display = "none"
        app._info_content = MagicMock()
        app._info_pane_mode = None
        app._last_info_content = None
        app._last_info_mode = None

        def show():
            app._info_pane.styles.display = "block"

        def hide():
            if app._info_pane.styles.display != "none":
                app._last_info_mode = app._info_pane_mode
            app._info_pane.styles.display = "none"
            app._info_pane_mode = None

        def update_content(content):
            app._info_content.update(content)
            app._last_info_content = content
            show()

        app._show_info_pane = show
        app._hide_info_pane = hide
        app._update_info_content = update_content

        # Show /history content and set mode
        AgentTUI._update_info_content(app, "History content here")
        app._info_pane_mode = "history"

        # Hide
        AgentTUI._hide_info_pane(app)
        assert app._last_info_mode == "history"

        # Reopen
        if app._info_pane.styles.display != "none":
            hide()
        elif app._last_info_content is not None:
            app._info_pane_mode = app._last_info_mode
            app._info_content.update(app._last_info_content)
            show()

        assert app._info_pane_mode == "history"
        assert app._info_pane.styles.display == "block"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
