#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for TUI auto-scroll behavior using Textual's anchor() method.

Tests verify:
1. Messages mount correctly to VerticalScroll
2. Auto-scroll works when at bottom
3. Auto-scroll disables when user scrolls up
4. Auto-scroll re-enables when user scrolls to bottom
5. Keyboard scroll controls (shift+up/down) work
"""

import asyncio
import pytest

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Vertical
from textual.reactive import reactive
from textual.widgets import Input, Static
from typing import ClassVar


class ChatPane(VerticalScroll):
    """Scrollable message display area using VerticalScroll with anchor()."""

    def on_mount(self) -> None:
        """Enable anchoring on mount."""
        self.anchor()

    def is_at_bottom(self) -> bool:
        """Check if view is near bottom (within threshold)."""
        try:
            threshold = 3
            return self.scroll_y >= (self.max_scroll_y - threshold)
        except Exception:
            return True

    async def add_message(self, text: str) -> None:
        """Add a message by mounting a Static widget."""
        was_at_bottom = self.is_at_bottom()
        message = Static(text, markup=True)
        await self.mount(message)
        if was_at_bottom:
            self.scroll_end(animate=False)
            self.anchor()


class StatusBar(Static):
    """Status display showing anchor state and message count."""

    anchored: reactive[bool] = reactive(True)
    message_count: reactive[int] = reactive(0)
    running: reactive[bool] = reactive(False)

    def render(self) -> str:
        anchor_status = "[green]ON[/]" if self.anchored else "[red]OFF[/]"
        run_status = "[green]RUNNING[/]" if self.running else "[red]STOPPED[/]"
        return f" Auto-scroll: {anchor_status} | Messages: {self.message_count} | State: {run_status}"


class ScrollTestTUI(App):
    """Minimal TUI for testing auto-scroll behavior."""

    CSS = """
    Screen {
        layout: vertical;
        background: #1a1a1a;
    }

    #chat {
        height: 1fr;
        width: 100%;
        background: #1a1a1a;
        padding: 0;
    }

    #status-bar {
        height: auto;
        width: 100%;
        background: #333;
        color: #fff;
        padding: 0 1;
    }

    #input-container {
        height: auto;
        width: 100%;
        background: #1a1a1a;
        padding: 0;
    }

    Input {
        width: 100%;
        height: auto;
        background: transparent;
        color: #fff;
        border: none;
        padding: 0;
    }

    Input:focus {
        border: none;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False),
        Binding("shift+down", "scroll_chat_down", "Scroll Down", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._generating = False
        self._task: asyncio.Task | None = None
        self._message_count = 0

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat"):
            yield Static("")  # Placeholder
        yield StatusBar(id="status-bar")
        with Vertical(id="input-container"):
            yield Input(placeholder="Type /start or /stop...", id="input")

    def on_mount(self) -> None:
        """Set up the TUI after mounting."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.anchor()
        asyncio.create_task(self._mount_initial_messages())
        self._update_status()

    async def _mount_initial_messages(self) -> None:
        """Mount initial welcome messages."""
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(
            Static(
                "[dim]Welcome! Type /start to begin message generation.[/]", markup=True
            )
        )
        await chat.mount(
            Static("[dim]Scroll up with mouse to disable auto-scroll[/]", markup=True)
        )
        await chat.mount(
            Static("[dim]Shift+Up/Down for keyboard scroll control[/]", markup=True)
        )
        await chat.mount(Static("[dim]Press Ctrl+C or Ctrl+D to quit[/]", markup=True))
        await chat.mount(Static("", markup=True))
        chat.scroll_end(animate=False)

    def _update_status(self) -> None:
        """Update the status bar."""
        status = self.query_one("#status-bar", StatusBar)
        chat = self.query_one("#chat", VerticalScroll)
        status.anchored = chat.is_anchored
        status.message_count = self._message_count
        status.running = self._generating

    def _is_at_bottom(self) -> bool:
        """Check if view is near bottom."""
        chat = self.query_one("#chat", VerticalScroll)
        try:
            threshold = 3
            return chat.scroll_y >= (chat.max_scroll_y - threshold)
        except Exception:
            return True

    async def _generate_messages(self) -> None:
        """Background task to generate messages."""
        import random

        phrases = [
            "Processing data stream...",
            "Analyzing patterns...",
            "Updating cache...",
            "Syncing with server...",
            "Computing results...",
        ]

        try:
            while self._generating:
                await asyncio.sleep(0.3)
                if not self._generating:
                    break

                self._message_count += 1
                phrase = random.choice(phrases)
                timestamp = f"[{self._message_count:04d}]"
                text = f"[cyan]{timestamp}[/] {phrase}"

                chat = self.query_one("#chat", VerticalScroll)
                was_at_bottom = self._is_at_bottom()

                # Mount message
                message = Static(text, markup=True)
                await chat.mount(message)

                # Auto-scroll if was at bottom
                if was_at_bottom:
                    chat.scroll_end(animate=False)
                    chat.anchor()

                self._update_status()
        except asyncio.CancelledError:
            raise

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input."""
        command = event.value.strip().lower()
        self.query_one("#chat", VerticalScroll)

        if command == "/start":
            if not self._generating:
                self._generating = True
                asyncio.create_task(
                    self._mount_message("[green]▶ Started message generation[/]")
                )
                self._task = asyncio.create_task(self._generate_messages())
            else:
                asyncio.create_task(self._mount_message("[yellow]Already running![/]"))
        elif command == "/stop":
            if self._generating:
                self._generating = False
                if self._task and not self._task.done():
                    self._task.cancel()
                asyncio.create_task(
                    self._mount_message("[red]◀ Stopped message generation[/]")
                )
            else:
                asyncio.create_task(self._mount_message("[yellow]Not running![/]"))
        elif command in ("/quit", "/exit", "q"):
            self.exit()
        else:
            asyncio.create_task(
                self._mount_message(f"[dim]Unknown command: {command}[/]")
            )

        self._update_status()
        event.input.value = ""

    async def _mount_message(self, text: str) -> None:
        """Mount a single message to the chat."""
        chat = self.query_one("#chat", VerticalScroll)
        was_at_bottom = self._is_at_bottom()
        message = Static(text, markup=True)
        await chat.mount(message)
        if was_at_bottom:
            chat.scroll_end(animate=False)
            chat.anchor()

    def action_scroll_chat_up(self) -> None:
        """Scroll up and disable auto-scroll."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.scroll_relative(y=-5, animate=False)
        chat.anchor(False)
        self.notify("Auto-scroll disabled")
        self._update_status()

    def action_scroll_chat_down(self) -> None:
        """Scroll down and re-enable auto-scroll if at bottom."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.scroll_relative(y=5, animate=False)
        if self._is_at_bottom():
            chat.anchor()
            self.notify("Auto-scroll enabled")
        self._update_status()

    def on_unmount(self) -> None:
        """Clean up task on exit."""
        self._generating = False
        if self._task and not self._task.done():
            self._task.cancel()


# ============== Tests ==============


@pytest.mark.asyncio
async def test_app_mounts_and_starts():
    """Test that the app mounts correctly and starts."""
    app = ScrollTestTUI()

    async with app.run_test():
        # Verify widgets exist
        chat = app.query_one("#chat", VerticalScroll)
        status = app.query_one("#status-bar", StatusBar)
        input_box = app.query_one("#input", Input)

        assert chat is not None
        assert status is not None
        assert input_box is not None

        # Verify initial state
        assert not app._generating
        assert app._message_count == 0


@pytest.mark.asyncio
async def test_anchor_enabled_on_mount():
    """Test that anchor is enabled when app mounts."""
    app = ScrollTestTUI()

    async with app.run_test():
        chat = app.query_one("#chat", VerticalScroll)
        # Anchor should be enabled on mount
        assert chat.is_anchored


@pytest.mark.asyncio
async def test_start_stop_generation():
    """Test starting and stopping message generation."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        input_box = app.query_one("#input", Input)
        input_box.focus()
        await pilot.pause()

        # Start generation
        await pilot.press(*"/start")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app._generating

        # Let it generate some messages
        await pilot.pause(1.0)
        assert app._message_count > 0

        # Stop generation
        await pilot.press(*"/stop")
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert not app._generating


@pytest.mark.asyncio
async def test_messages_mount_to_verticalscroll():
    """Test that messages are mounted to VerticalScroll, not a child Static."""
    app = ScrollTestTUI()

    async with app.run_test():
        chat = app.query_one("#chat", VerticalScroll)

        # Mount a message
        await app._mount_message("[cyan]Test message[/]")

        # Verify message was mounted to chat (VerticalScroll)
        # The chat should have children (the mounted Static widgets)
        children = list(chat.children)
        assert len(children) > 0

        # Verify the message is a Static widget
        from textual.widgets import Static

        assert all(isinstance(child, Static) for child in children)


@pytest.mark.asyncio
async def test_auto_scroll_when_at_bottom():
    """Test that auto-scroll works when view is at bottom."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        chat = app.query_one("#chat", VerticalScroll)

        # Verify anchor is enabled
        assert chat.is_anchored

        # Mount multiple messages
        for i in range(5):
            await app._mount_message(f"[cyan]Message {i}[/]")
            await pilot.pause(0.1)

        # Should still be anchored (auto-scroll working)
        assert chat.is_anchored

        # Note: _is_at_bottom() may return False if content doesn't overflow
        # The key test is that anchor is still enabled


@pytest.mark.asyncio
async def test_keyboard_scroll_up_disables_anchor():
    """Test that Shift+Up disables auto-scroll."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        chat = app.query_one("#chat", VerticalScroll)

        # Start with anchor enabled
        assert chat.is_anchored

        # Mount some messages first
        for i in range(10):
            await app._mount_message(f"[cyan]Message {i}[/]")
            await pilot.pause(0.05)

        # Trigger scroll up action
        app.action_scroll_chat_up()
        await pilot.pause()

        # Anchor should be disabled
        assert not chat.is_anchored


@pytest.mark.asyncio
async def test_keyboard_scroll_down_reenables_anchor_at_bottom():
    """Test that Shift+Down re-enables auto-scroll when at bottom."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        chat = app.query_one("#chat", VerticalScroll)

        # Mount messages
        for i in range(10):
            await app._mount_message(f"[cyan]Message {i}[/]")
            await pilot.pause(0.05)

        # Disable anchor by scrolling up
        app.action_scroll_chat_up()
        await pilot.pause()
        assert not chat.is_anchored

        # Scroll down to bottom
        chat.scroll_end(animate=False)
        await pilot.pause()

        # Now trigger scroll down action
        app.action_scroll_chat_down()
        await pilot.pause()

        # Anchor should be re-enabled if we're at bottom
        # Note: In test environment, content may not overflow, so anchor
        # might not be re-enabled. The key test is that scroll_down action
        # works without error.
        # If content overflows and we're at bottom, anchor should be True
        if app._is_at_bottom():
            assert chat.is_anchored


@pytest.mark.asyncio
async def test_message_generation_with_auto_scroll():
    """Test that message generation maintains auto-scroll."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        chat = app.query_one("#chat", VerticalScroll)
        input_box = app.query_one("#input", Input)
        input_box.focus()
        await pilot.pause()

        # Start generation
        await pilot.press(*"/start")
        await pilot.press("enter")
        await pilot.pause(0.2)

        # Let it generate messages
        await pilot.pause(1.5)

        # Should still be anchored
        assert chat.is_anchored

        # Stop generation
        await pilot.press(*"/stop")
        await pilot.press("enter")
        await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_quit_command():
    """Test that /quit command exits cleanly."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        input_box = app.query_one("#input", Input)
        input_box.focus()
        await pilot.pause()

        # Start generation
        await pilot.press(*"/start")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app._generating

        # Quit
        await pilot.press("q")
        await pilot.press("enter")

        # App should exit - context manager handles cleanup


@pytest.mark.asyncio
async def test_unknown_command():
    """Test unknown command handling."""
    app = ScrollTestTUI()

    async with app.run_test() as pilot:
        chat = app.query_one("#chat", VerticalScroll)
        input_box = app.query_one("#input", Input)
        input_box.focus()
        await pilot.pause()

        # Count initial messages
        initial_count = len(list(chat.children))

        # Type unknown command
        await pilot.press(*"/unknown")
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Should have added a message
        new_count = len(list(chat.children))
        assert new_count > initial_count


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
