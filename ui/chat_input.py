"""Multi-line chat input widget using TextArea."""

from textual.widgets import TextArea
from textual.binding import Binding
from textual.message import Message
from textual import events


class ChatTextArea(TextArea):
    """Multi-line input widget with history and completion support.

    Key bindings:
    - Enter: Submit message
    - Ctrl+J: New line
    - Ctrl+B: History previous
    - Ctrl+F: History next
    - Up/Down: Cursor movement (TextArea default)
    - Tab: Completion (handled by App)
    """

    BINDINGS = [
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("ctrl+j", "insert_newline", "New Line", show=False, priority=True),
    ]

    class Submitted(Message):
        """Posted when user presses Enter to submit."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class HistoryPrevious(Message):
        """Posted when user presses Ctrl+B."""

        def __init__(self, prefix: str) -> None:
            self.prefix = prefix
            super().__init__()

    class HistoryNext(Message):
        """Posted when user presses Ctrl+F."""

        def __init__(self, prefix: str) -> None:
            self.prefix = prefix
            super().__init__()

    def action_submit(self) -> None:
        """Submit the current text."""
        value = self.text.strip()
        # Always post Submitted, even if empty (allows TUI to close todo area on empty Enter)
        self.post_message(self.Submitted(value))

    def action_insert_newline(self) -> None:
        """Insert a newline character and scroll to keep cursor visible."""
        self.insert("\n")
        # Scroll to keep cursor in view after inserting newline
        self.scroll_cursor_visible()

    async def _on_key(self, event: events.Key) -> None:
        """Handle key events for history navigation."""
        # Ctrl+B = history previous (backward)
        if event.key == "ctrl+b":
            event.prevent_default()
            event.stop()
            self.post_message(self.HistoryPrevious(self.text))
            return

        # Ctrl+F = history next (forward)
        if event.key == "ctrl+f":
            event.prevent_default()
            event.stop()
            self.post_message(self.HistoryNext(self.text))
            return

        # Tab = completion (handled by App)
        # Don't prevent default - App's on_key will catch it
        # Note: We let Tab bubble up so App can handle completion

        await super()._on_key(event)
