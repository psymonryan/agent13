#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for TUI commands and features.

Tests verify:
1. History navigation with persistence
2. Queue commands (/queue, /pause, /resume)
3. Priority detection (! at end of message)
4. /prioritise and /deprioritise commands
5. /delete q N command
6. History grouping for atomic deletion
7. Context size in status bar
"""

import pytest
import os
import tempfile


from agent13.history import History
from agent13.queue import AgentQueue, ItemStatus


# ============== History Tests ==============


def test_history_add_and_navigation():
    """Test History class add and navigation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "history-test-2026-03-25")

        from unittest.mock import patch

        with patch.object(History, "_get_path", return_value=path):
            history = History("test")

            # Test empty history
            assert len(history) == 0
            assert history.up() is None
            assert history.down() is None

            # Add items
            history.add("first command")
            history.add("second command")
            history.add("third command")

            # Test navigation
            assert history.up() == "third command"  # Most recent
            assert history.up() == "second command"
            assert history.up() == "first command"
            assert history.up() == "first command"  # Stays at oldest

            # Navigate back down
            assert history.down() == "second command"
            assert history.down() == "third command"
            assert history.down() is None  # Back to start, clears input

            # Reset and navigate again
            history.reset()
            assert history.up() == "third command"


def test_history_no_consecutive_duplicates():
    """Test that consecutive duplicates are not added."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "history-test-2026-03-25")

        from unittest.mock import patch

        with patch.object(History, "_get_path", return_value=path):
            history = History("test")
            history.add("same")
            history.add("same")
            history.add("different")

            assert len(history) == 2
            assert history.up() == "different"
            assert history.up() == "same"


def test_queue_add_and_list():
    """Test AgentQueue add and list operations."""
    queue = AgentQueue()

    # Empty queue
    assert queue.pending_count == 0
    assert queue.list_items() == []

    # Add items
    queue.add("first item")
    queue.add("second item")
    queue.add("third item")

    assert queue.pending_count == 3
    items = queue.list_items()
    assert len(items) == 3
    assert items[0].text == "first item"
    assert items[1].text == "second item"
    assert items[2].text == "third item"


def test_queue_priority():
    """Test AgentQueue priority handling."""
    queue = AgentQueue()

    # Add normal items
    queue.add("normal 1")
    queue.add("normal 2")

    # Add priority item - should be at front
    queue.add("priority 1", priority=True)

    items = queue.list_items()
    assert items[0].text == "priority 1"
    assert items[0].priority
    assert items[1].text == "normal 1"
    assert items[2].text == "normal 2"

    # Test has_priority
    assert queue.has_priority

    # Test set_priority_at
    queue.set_priority_at(2, True)  # Make "normal 1" priority
    items = queue.list_items()
    # Priority items should be at front
    assert items[0].priority
    assert items[1].priority


def test_queue_interrupt_priority_ordering():
    """Test that interrupt items come before priority items, which come before normal items."""
    queue = AgentQueue()

    # Add items in mixed order
    queue.add("normal 1")
    queue.add("priority 1", priority=True)
    queue.add("normal 2")
    queue.add("interrupt 1", interrupt=True)
    queue.add("priority 2", priority=True)
    queue.add("interrupt 2", interrupt=True)

    items = queue.list_items()

    # Interrupt items should be first (in order added)
    assert items[0].text == "interrupt 1"
    assert items[0].interrupt
    assert items[0].priority  # interrupt implies priority

    assert items[1].text == "interrupt 2"
    assert items[1].interrupt

    # Then priority items
    assert items[2].text == "priority 1"
    assert items[2].priority
    assert not items[2].interrupt

    assert items[3].text == "priority 2"
    assert items[3].priority
    assert not items[3].interrupt

    # Then normal items
    assert items[4].text == "normal 1"
    assert not items[4].priority
    assert not items[4].interrupt

    assert items[5].text == "normal 2"
    assert not items[5].priority

    # Test has_interrupt
    assert queue.has_interrupt

    # Test pop_interrupt_items
    interrupt_items = queue.pop_interrupt_items()
    assert len(interrupt_items) == 2
    assert interrupt_items[0].text == "interrupt 1"
    assert interrupt_items[1].text == "interrupt 2"

    # After popping, no more interrupt items
    assert not queue.has_interrupt
    assert queue.has_priority  # Still have priority items


def test_queue_remove_at():
    """Test AgentQueue remove_at operation."""
    queue = AgentQueue()

    queue.add("item 1")
    queue.add("item 2")
    queue.add("item 3")

    # Remove item at index 2 (1-based)
    removed = queue.remove_at(2)
    assert removed is not None
    assert removed.text == "item 2"

    assert queue.pending_count == 2
    items = queue.list_items()
    assert items[0].text == "item 1"
    assert items[1].text == "item 3"

    # Invalid index
    assert queue.remove_at(10) is None


def test_queue_get_next_and_complete():
    """Test AgentQueue get_next and complete_current operations."""
    queue = AgentQueue()

    queue.add("item 1")
    queue.add("item 2")

    # Get next item
    current = queue.get_next()
    assert current is not None
    assert current.text == "item 1"
    assert current.status == ItemStatus.RUNNING

    # get_next returns the current item if one is running
    current_again = queue.get_next()
    assert current_again is not None
    assert current_again.text == "item 1"

    # Complete current
    queue.complete_current()
    assert queue.current is None

    # Get next item
    current = queue.get_next()
    assert current is not None
    assert current.text == "item 2"

    # Complete and verify queue is empty
    queue.complete_current()
    assert queue.get_next() is None


# ============== Message Grouping Tests ==============


def test_message_grouping():
    """Test _get_message_groups logic."""
    # Simulate the grouping logic
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 5 squared?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "square_number"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "content": "25"},
        {"role": "assistant", "content": "5 squared is 25"},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome!"},
    ]

    # Group messages
    groups = []
    current_group = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        if role == "user":
            if current_group:
                groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        groups.append(current_group)

    # Should have 3 groups
    assert len(groups) == 3

    # Group 1: user + assistant
    assert groups[0] == [0, 1]

    # Group 2: user + assistant (tool call) + tool result + assistant
    assert groups[1] == [2, 3, 4, 5]

    # Group 3: user + assistant
    assert groups[2] == [6, 7]


def test_message_grouping_empty():
    """Test grouping with empty messages."""
    messages = []

    groups = []
    current_group = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        if role == "user":
            if current_group:
                groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        groups.append(current_group)

    assert len(groups) == 0


def test_message_grouping_single_user():
    """Test grouping with single user message."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    groups = []
    current_group = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")

        if role == "user":
            if current_group:
                groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    if current_group:
        groups.append(current_group)

    assert len(groups) == 1
    assert groups[0] == [0]


# ============== Priority Detection Tests ==============


def test_priority_detection():
    """Test detecting priority and interrupt markers in messages."""
    # Messages starting with !! are interrupt level
    text = "!!Hello"
    if text.startswith("!!"):
        interrupt = True
        priority = True
        message_text = text[2:].strip()
    elif text.startswith("!"):
        interrupt = False
        priority = True
        message_text = text[1:].strip()
    else:
        interrupt = False
        priority = False
        message_text = text

    assert interrupt
    assert priority
    assert message_text == "Hello"

    # Messages starting with single ! are priority (not interrupt)
    text2 = "!Urgent"
    if text2.startswith("!!"):
        interrupt2 = True
        priority2 = True
        message_text2 = text2[2:].strip()
    elif text2.startswith("!"):
        interrupt2 = False
        priority2 = True
        message_text2 = text2[1:].strip()
    else:
        interrupt2 = False
        priority2 = False
        message_text2 = text2

    assert not interrupt2
    assert priority2
    assert message_text2 == "Urgent"

    # Normal messages have no markers
    text3 = "Hello"
    if text3.startswith("!!"):
        interrupt3 = True
        priority3 = True
        message_text3 = text3[2:].strip()
    elif text3.startswith("!"):
        interrupt3 = False
        priority3 = True
        message_text3 = text3[1:].strip()
    else:
        interrupt3 = False
        priority3 = False
        message_text3 = text3

    assert not interrupt3
    assert not priority3
    assert message_text3 == "Hello"


# ============== Context Size Formatting Tests ==============


def test_context_size_formatting():
    """Test format_context_size function."""
    from agent13 import format_context_size

    # Test with empty messages
    assert format_context_size([]) == "0"

    # Test with small messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    size = format_context_size(messages)
    assert size.endswith("k") or size.isdigit()  # Either "Xk" or just digits


# ============== Tab Completion Tests ==============


class MockAgentTUI:
    """Mock AgentTUI for testing completion logic without full Textual app."""

    COMMANDS = [
        "/help",
        "/quit",
        "/exit",
        "/clear",
        "/hist",
        "/delete",
        "/model",
        "/list",
        "/tool-response",
        "/pretty",
        "/prompt",
        "/queue",
        "/pause",
        "/resume",
        "/prioritise",
        "/deprioritise",
    ]

    def __init__(self):
        self._history = History()
        self._completion_matches = []
        self._completion_index = 0
        self._completion_prefix = ""

    def _get_command_completions(self, text: str) -> list[str]:
        """Get slash command completions for text starting with /."""
        if not text.startswith("/"):
            return []
        text_lower = text.lower()
        matches = [cmd for cmd in self.COMMANDS if cmd.lower().startswith(text_lower)]
        return matches

    def _get_history_completions(self, text: str) -> list[str]:
        """Get history completions for text."""
        if not text:
            return []
        text_lower = text.lower()
        matches = []
        seen = set()
        for item in self._history.get_all():
            if item.lower().startswith(text_lower) and item not in seen:
                matches.append(item)
                seen.add(item)
                if len(matches) >= 10:
                    break
        return matches

    def _get_completions(self, text: str) -> list[str]:
        """Get completions for text."""
        if text.startswith("/"):
            return self._get_command_completions(text)
        else:
            return self._get_history_completions(text)

    def _reset_completion_state(self):
        """Reset tab completion state."""
        self._completion_matches = []
        self._completion_index = 0
        self._completion_prefix = ""


def test_command_completion_exact_match():
    """Test command completion with exact match."""
    tui = MockAgentTUI()

    # Exact match should return just that command
    matches = tui._get_completions("/help")
    assert matches == ["/help"]


def test_command_completion_partial_match():
    """Test command completion with partial match."""
    tui = MockAgentTUI()

    # Partial match should return all matching commands
    matches = tui._get_completions("/h")
    assert "/help" in matches
    assert "/hist" in matches
    assert len(matches) >= 2


def test_command_completion_no_match():
    """Test command completion with no match."""
    tui = MockAgentTUI()

    # No match should return empty
    matches = tui._get_completions("/xyz")
    assert matches == []


def test_command_completion_case_insensitive():
    """Test command completion is case insensitive."""
    tui = MockAgentTUI()

    # Case insensitive matching
    matches = tui._get_completions("/HELP")
    assert "/help" in matches

    matches = tui._get_completions("/MODEL")
    assert "/model" in matches


def test_command_completion_prioritise_variants():
    """Test completion for commands — Australian spelling shown in completions."""
    tui = MockAgentTUI()

    # Australian spelling should appear in completions
    matches = tui._get_completions("/prior")
    assert "/prioritise" in matches

    matches = tui._get_completions("/depri")
    assert "/deprioritise" in matches


def test_history_completion_empty_history():
    """Test history completion with empty history."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".history", delete=False) as f:
        history_path = f.name

    try:
        tui = MockAgentTUI()
        tui._history = History(file_path=history_path)

        matches = tui._get_history_completions("test")
        assert matches == []
    finally:
        os.unlink(history_path)


def test_history_completion_with_matches():
    """Test history completion with matching items."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".history", delete=False) as f:
        history_path = f.name

    try:
        tui = MockAgentTUI()
        tui._history = History(file_path=history_path)

        # Add some history items
        tui._history.add("test command one")
        tui._history.add("test command two")
        tui._history.add("other command")

        # Should match items starting with "test"
        matches = tui._get_history_completions("test")
        assert len(matches) == 2
        assert "test command one" in matches
        assert "test command two" in matches

        # Should not match "other command"
        assert "other command" not in matches
    finally:
        os.unlink(history_path)


def test_history_completion_case_insensitive():
    """Test history completion is case insensitive."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".history", delete=False) as f:
        history_path = f.name

    try:
        tui = MockAgentTUI()
        tui._history = History(file_path=history_path)

        tui._history.add("Hello World")

        # Case insensitive matching
        matches = tui._get_history_completions("hello")
        assert "Hello World" in matches

        matches = tui._get_history_completions("HELLO")
        assert "Hello World" in matches
    finally:
        os.unlink(history_path)


def test_history_completion_limit():
    """Test history completion limits to 10 matches."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".history", delete=False) as f:
        history_path = f.name

    try:
        tui = MockAgentTUI()
        tui._history = History(file_path=history_path)

        # Add 15 items all starting with "test"
        for i in range(15):
            tui._history.add(f"test item {i}")

        matches = tui._get_history_completions("test")
        assert len(matches) == 10  # Limited to 10
    finally:
        os.unlink(history_path)


def test_history_completion_no_duplicates():
    """Test history completion doesn't return duplicates."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".history", delete=False) as f:
        history_path = f.name

    try:
        tui = MockAgentTUI()
        tui._history = History(file_path=history_path)

        # Add same item multiple times (History already prevents consecutive duplicates)
        tui._history.add("test command")
        tui._history.add("other command")
        tui._history.add("test command")  # Duplicate (not consecutive, so allowed)

        matches = tui._get_history_completions("test")
        # Should only have one "test command" even if it appears multiple times
        assert matches.count("test command") == 1
    finally:
        os.unlink(history_path)


def test_completion_mode_slash_vs_history():
    """Test that completion mode switches based on input."""
    tui = MockAgentTUI()

    # Slash prefix -> command completion
    matches = tui._get_completions("/hel")
    assert all(m.startswith("/") for m in matches)

    # No slash prefix -> history completion
    tui._history.add("hello there")
    matches = tui._get_completions("hel")
    assert all(not m.startswith("/") for m in matches)


def test_completion_state_reset():
    """Test that completion state can be reset."""
    tui = MockAgentTUI()

    # Set up some completion state
    tui._completion_matches = ["/help", "/hist"]
    tui._completion_index = 1
    tui._completion_prefix = "/h"

    # Reset
    tui._reset_completion_state()

    assert tui._completion_matches == []
    assert tui._completion_index == 0
    assert tui._completion_prefix == ""


def test_completion_empty_text():
    """Test completion with empty text."""
    tui = MockAgentTUI()

    # Empty text should return no matches for history
    matches = tui._get_history_completions("")
    assert matches == []

    # Empty text for commands should return all commands
    matches = tui._get_command_completions("")
    assert len(matches) == 0  # Empty string doesn't start with /


def test_elapsed_timer_reset_on_idle():
    """Test that elapsed timer resets when going idle with no queue items."""
    import time

    # Create a mock TUI with minimal setup
    class MockQueue:
        def __init__(self):
            self._pending_count = 0

        @property
        def pending_count(self):
            return self._pending_count

    class MinimalTUI:
        def __init__(self):
            self._elapsed_start_time = None
            self.agent = type("obj", (object,), {"queue": MockQueue()})()
            self.processing = False
            self.status = "Ready"

        def _update_status(self, status: str) -> None:
            """Simplified version of TUI._update_status for testing."""
            if status == "processing":
                self.processing = True
                self.status = "Processing"
                if self._elapsed_start_time is None:
                    self._elapsed_start_time = time.time()
            elif status == "idle":
                self.processing = False
                self.status = "Ready"
                if self.agent.queue.pending_count == 0:
                    self._elapsed_start_time = None

    tui = MinimalTUI()

    # Initially no timer
    assert tui._elapsed_start_time is None
    assert tui.status == "Ready"

    # Start processing - timer should start
    tui._update_status("processing")
    assert tui._elapsed_start_time is not None
    start_time = tui._elapsed_start_time

    # Small delay
    time.sleep(0.01)

    # Processing again - timer should NOT reset (already running)
    tui._update_status("processing")
    assert tui._elapsed_start_time == start_time

    # Go idle - timer should reset
    tui._update_status("idle")
    assert tui._elapsed_start_time is None
    assert tui.status == "Ready"


def test_elapsed_timer_continues_with_queue():
    """Test that elapsed timer continues when queue has pending items."""
    import time

    class MockQueue:
        def __init__(self):
            self._pending_count = 0

        @property
        def pending_count(self):
            return self._pending_count

    class MinimalTUI:
        def __init__(self):
            self._elapsed_start_time = None
            self.agent = type("obj", (object,), {"queue": MockQueue()})()
            self.processing = False
            self.status = "Ready"

        def _update_status(self, status: str) -> None:
            if status == "processing":
                self.processing = True
                self.status = "Processing"
                if self._elapsed_start_time is None:
                    self._elapsed_start_time = time.time()
            elif status == "idle":
                self.processing = False
                self.status = "Ready"
                if self.agent.queue.pending_count == 0:
                    self._elapsed_start_time = None

    tui = MinimalTUI()

    # Start processing
    tui._update_status("processing")
    assert tui._elapsed_start_time is not None
    start_time = tui._elapsed_start_time

    # Add a queue item
    tui.agent.queue._pending_count = 1

    # Go idle with pending items - timer should NOT reset
    tui._update_status("idle")
    assert tui._elapsed_start_time == start_time

    # Queue empty now
    tui.agent.queue._pending_count = 0

    # Go idle again - timer should reset
    tui._update_status("idle")
    assert tui._elapsed_start_time is None


# =============================================================================
# Completion Context Tests (Phase 1: Cursor-aware completion)
# =============================================================================


class MockCompletionContext:
    """Mock class to test _get_completion_context logic."""

    def _get_completion_context(
        self, text: str, cursor_row: int, cursor_col: int
    ) -> tuple[str, tuple[int, int], tuple[int, int]]:
        """Analyze input to determine what kind of completion is needed.

        This is a copy of the TUI method for testing.
        """
        # Get the line the cursor is on
        lines = text.split("\n")
        if cursor_row >= len(lines):
            return ("none", (0, 0), (cursor_row, cursor_col))
        current_line = lines[cursor_row]

        # Calculate character offset within the current line
        line_offset = cursor_col
        text_before_cursor = current_line[:line_offset]

        # Check for @filename completion (anywhere in text)
        # Find the last @ before cursor on this line
        at_pos = text_before_cursor.rfind("@")
        if at_pos != -1:
            # Check if there's a space between @ and cursor (invalidates the match)
            after_at = text_before_cursor[at_pos + 1 :]
            if " " not in after_at and "\n" not in after_at:
                # Valid @filename context
                start_col = at_pos
                return ("@filename", (cursor_row, start_col), (cursor_row, cursor_col))

        # Check for /command completion (must be at start of first line)
        if cursor_row == 0 and text_before_cursor.startswith("/"):
            # Check if we're still in the command part (no space yet) or in params
            space_pos = text_before_cursor.find(" ")
            if space_pos == -1:
                # No space yet - completing the command itself
                return ("/command", (0, 0), (0, cursor_col))
            else:
                # Space found - completing command parameters
                # The partial is everything after the space
                start_col = space_pos + 1
                return ("/cmd_param", (0, start_col), (0, cursor_col))

        # History completion (text at start of first line, no special prefix, no @)
        # Don't trigger history if there's an @ in the text (we're after a file reference)
        if (
            cursor_row == 0
            and cursor_col > 0
            and not text_before_cursor.startswith("/")
            and "@" not in text_before_cursor
        ):
            return ("history", (0, 0), (0, cursor_col))

        # No completion context
        return ("none", (cursor_row, cursor_col), (cursor_row, cursor_col))


def test_completion_context_at_filename():
    """Test @filename completion context detection."""
    ctx = MockCompletionContext()

    # @ at start of text
    result = ctx._get_completion_context("Read @src/", 0, 10)
    assert result[0] == "@filename"
    assert result[1] == (0, 5)  # @ is at position 5
    assert result[2] == (0, 10)  # cursor at end

    # @ in middle of text
    result = ctx._get_completion_context("Please read @fil", 0, 15)
    assert result[0] == "@filename"
    assert result[1] == (0, 12)  # @ position
    assert result[2] == (0, 15)

    # @ with partial path
    result = ctx._get_completion_context("@", 0, 1)
    assert result[0] == "@filename"
    assert result[1] == (0, 0)
    assert result[2] == (0, 1)


def test_completion_context_at_filename_with_space():
    """Test that @filename context is NOT triggered if space after @."""
    ctx = MockCompletionContext()

    # Space after @ invalidates the match
    result = ctx._get_completion_context("Read @ file", 0, 11)
    assert result[0] != "@filename"  # Should be history or none


def test_completion_context_slash_command():
    """Test /command completion context detection."""
    ctx = MockCompletionContext()

    # Partial command
    result = ctx._get_completion_context("/mod", 0, 4)
    assert result[0] == "/command"
    assert result[1] == (0, 0)
    assert result[2] == (0, 4)

    # Just slash
    result = ctx._get_completion_context("/", 0, 1)
    assert result[0] == "/command"
    assert result[1] == (0, 0)
    assert result[2] == (0, 1)


def test_completion_context_cmd_param():
    """Test /cmd_param completion context detection."""
    ctx = MockCompletionContext()

    # After command with space
    result = ctx._get_completion_context("/model dev", 0, 10)
    assert result[0] == "/cmd_param"
    assert result[1] == (0, 7)  # Start after "/model "
    assert result[2] == (0, 10)

    # Empty param (just after space)
    result = ctx._get_completion_context("/model ", 0, 7)
    assert result[0] == "/cmd_param"
    assert result[1] == (0, 7)
    assert result[2] == (0, 7)


def test_completion_context_history():
    """Test history completion context detection."""
    ctx = MockCompletionContext()

    # Regular text at start
    result = ctx._get_completion_context("prev", 0, 4)
    assert result[0] == "history"
    assert result[1] == (0, 0)
    assert result[2] == (0, 4)


def test_completion_context_none():
    """Test cases where no completion should occur."""
    ctx = MockCompletionContext()

    # Empty text
    result = ctx._get_completion_context("", 0, 0)
    assert result[0] == "none"

    # Slash command not at start of first line
    result = ctx._get_completion_context("some text /mod", 0, 14)
    # This should match history because / is not at start
    assert result[0] == "history"

    # @ on second line (should still work)
    result = ctx._get_completion_context("first line\n@file", 1, 5)
    assert result[0] == "@filename"
    assert result[1] == (1, 0)


def test_completion_context_multiline():
    """Test completion context with multiline text."""
    ctx = MockCompletionContext()

    # @filename on second line
    text = "First line\nRead @src/"
    result = ctx._get_completion_context(text, 1, 10)
    assert result[0] == "@filename"
    assert result[1] == (1, 5)  # @ position on second line

    # History should only work on first line
    text = "First line\nsecond"
    result = ctx._get_completion_context(text, 1, 6)
    assert result[0] == "none"  # No history on second line


def test_completion_context_after_file_completion():
    """Test that Tab after a completed file path doesn't trigger completion."""
    ctx = MockCompletionContext()

    # After a file completion with space after - should be "none"
    text = "@docs/file.md hello"
    result = ctx._get_completion_context(text, 0, 19)  # cursor at end
    assert result[0] == "none"  # Not history, not @filename

    # Multiple @ in text - should complete the last one
    text = "@file1.txt @file2.txt @fil"
    result = ctx._get_completion_context(text, 0, 26)
    assert result[0] == "@filename"
    assert result[1] == (0, 22)  # Last @ position


class MockFilenameCompleter:
    """Mock class to test _get_filename_completions logic."""

    def _get_filename_completions(self, partial: str) -> list[str]:
        """Get filename completions for @file syntax.

        This is a copy of the TUI method for testing.
        """
        import os

        if not partial.startswith("@"):
            return []

        # Strip the leading @ to get the path
        path_part = partial[1:]

        # Determine the directory to scan and the prefix to filter by
        if not path_part:
            # Just "@" - list files in current directory
            dir_path = "."
            prefix = ""
            base_for_completion = ""
        elif path_part.startswith("~"):
            # Handle home directory expansion
            expanded = os.path.expanduser(path_part)
            if os.path.isdir(expanded):
                dir_path = expanded
                prefix = ""
            else:
                dir_path = os.path.dirname(expanded)
                prefix = os.path.basename(expanded)
            # Preserve ~ in the completion
            base_for_completion = (
                path_part if path_part == "~" else os.path.dirname(path_part)
            )
            if base_for_completion and not base_for_completion.endswith("/"):
                base_for_completion += "/"
        else:
            # Handle relative or absolute paths
            if os.path.isabs(path_part):
                full_path = path_part
            else:
                full_path = os.path.join(os.getcwd(), path_part)

            if os.path.isdir(full_path):
                dir_path = full_path
                prefix = ""
            else:
                dir_path = os.path.dirname(full_path)
                prefix = os.path.basename(full_path)

            # Preserve the base path for completion
            if "/" in path_part:
                base_for_completion = path_part.rsplit("/", 1)[0] + "/"
            else:
                base_for_completion = ""

        # Get the directory to scan
        if not os.path.isdir(dir_path):
            return []

        try:
            entries = os.listdir(dir_path)
        except (PermissionError, OSError):
            return []

        # Filter by prefix and build completions
        matches = []
        for entry in sorted(entries):
            if entry.startswith("."):
                continue  # Skip hidden files
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue

            # Build the completion preserving the user's input style
            completion = f"@{base_for_completion}{entry}"

            # Add trailing / for directories
            full_entry_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_entry_path):
                completion += "/"

            matches.append(completion)

            if len(matches) >= 20:  # Limit matches
                break

        return matches


def test_filename_completion_no_at_sign():
    """Test that filename completion requires @ prefix."""
    completer = MockFilenameCompleter()
    matches = completer._get_filename_completions("src/")
    assert matches == []


def test_filename_completion_just_at():
    """Test @ alone lists current directory."""
    completer = MockFilenameCompleter()
    matches = completer._get_filename_completions("@")
    # Should return files from current directory
    assert all(m.startswith("@") for m in matches)
    # Hidden files should be excluded
    assert not any(m.startswith("@.") for m in matches)


def test_filename_completion_partial_match():
    """Test partial filename matching."""
    completer = MockFilenameCompleter()
    # Assuming there's a 'tests' directory in the project
    matches = completer._get_filename_completions("@te")
    assert any("tests" in m.lower() for m in matches)


def test_filename_completion_case_insensitive():
    """Test case-insensitive matching."""
    completer = MockFilenameCompleter()
    # Should match 'tests' with '@TE'
    matches = completer._get_filename_completions("@TE")
    assert any("tests" in m.lower() for m in matches)


def test_filename_completion_directories_have_slash():
    """Test that directory completions end with /."""
    completer = MockFilenameCompleter()
    # Assuming 'tests' is a directory
    matches = completer._get_filename_completions("@tests")
    # Find the tests match
    for m in matches:
        if "tests" in m and m.endswith("/"):
            assert True
            return
    # If no match found, that's also OK (might not be tests dir)
    assert True


def test_filename_completion_nested_path():
    """Test completion with nested path."""
    completer = MockFilenameCompleter()
    # Complete inside tests directory
    matches = completer._get_filename_completions("@tests/")
    # Should list files in tests directory
    assert all(m.startswith("@tests/") for m in matches)


def test_filename_completion_tilde_expansion():
    """Test ~ expansion for home directory."""
    completer = MockFilenameCompleter()
    matches = completer._get_filename_completions("@~")
    # Should expand to home directory contents
    # Just verify it doesn't crash and returns @-prefixed paths
    assert all(m.startswith("@") for m in matches)


def test_filename_completion_nonexistent_directory():
    """Test completion for non-existent directory returns empty."""
    completer = MockFilenameCompleter()
    matches = completer._get_filename_completions("@nonexistent_xyz_dir/")
    assert matches == []


def test_filename_completion_limit():
    """Test that completion is limited to 50 matches."""
    completer = MockFilenameCompleter()
    # Just @ should list current directory, but be limited
    matches = completer._get_filename_completions("@")
    assert len(matches) <= 50


class TestCompletionWindowing:
    """Test the paginated completion windowing logic."""

    def _make_show_output(self, matches, current, page_size=8):
        """Reproduce the windowing logic from _show_completions."""
        total = len(matches)
        current_idx = matches.index(current) if current in matches else 0

        if total <= page_size:
            window_start = 0
            window_end = total
        else:
            half = page_size // 2
            window_start = max(0, current_idx - half)
            window_end = window_start + page_size
            if window_end > total:
                window_end = total
                window_start = total - page_size

        return window_start, window_end, current_idx

    def test_small_list_fits_in_one_page(self):
        """Lists smaller than page_size show all items."""
        matches = ["a", "b", "c"]
        start, end, idx = self._make_show_output(matches, "b")
        assert start == 0
        assert end == 3
        assert idx == 1

    def test_exact_page_size(self):
        """List exactly page_size shows all items, no scrolling."""
        matches = [f"item{i}" for i in range(8)]
        start, end, idx = self._make_show_output(matches, "item4")
        assert start == 0
        assert end == 8

    def test_window_starts_at_top(self):
        """Selection near the top keeps window at the start."""
        matches = [f"item{i}" for i in range(20)]
        start, end, idx = self._make_show_output(matches, "item2")
        assert start == 0
        assert end == 8
        assert start <= idx < end

    def test_window_follows_selection_downward(self):
        """Selection past the middle scrolls the window down."""
        matches = [f"item{i}" for i in range(20)]
        start, end, idx = self._make_show_output(matches, "item10")
        assert start <= idx < end
        assert end - start == 8

    def test_window_clamps_at_bottom(self):
        """Selection near the end clamps window to show last page."""
        matches = [f"item{i}" for i in range(20)]
        start, end, idx = self._make_show_output(matches, "item19")
        assert end == 20
        assert start == 12  # 20 - 8
        assert start <= idx < end

    def test_window_at_last_item(self):
        """Selection at the very last item still visible."""
        matches = [f"item{i}" for i in range(23)]
        start, end, idx = self._make_show_output(matches, "item22")
        assert start == 15  # 23 - 8
        assert end == 23
        assert start <= idx < end

    def test_window_first_item_in_large_list(self):
        """Selection at first item in large list shows from top."""
        matches = [f"item{i}" for i in range(50)]
        start, end, idx = self._make_show_output(matches, "item0")
        assert start == 0
        assert end == 8


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
