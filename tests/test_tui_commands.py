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
from pathlib import Path


from agent13.history import History
from agent13.queue import AgentQueue, ItemStatus
from agent13.completion import filename_completion_scan, plan_completion
from ui.tui import AgentTUI


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


def test_queue_display_running_not_numbered():
    """Running item should not consume an index number; pending items start at 1."""
    from agent13.commands import format_queue_items

    queue = AgentQueue()
    queue.add("first pending")
    queue.add("second pending")
    queue.get_next()  # "first pending" becomes current/running

    items = format_queue_items(queue)

    assert len(items) == 2
    # Running item: index 0, running=True
    assert items[0].running is True
    assert items[0].index == 0
    assert items[0].text == "first pending"
    # First pending: index 1 (NOT 2)
    assert items[1].running is False
    assert items[1].index == 1
    assert items[1].text == "second pending"


def test_queue_display_no_running():
    """With nothing running, pending items are numbered 1..N as before."""
    from agent13.commands import format_queue_items

    queue = AgentQueue()
    queue.add("item a")
    queue.add("item b")

    items = format_queue_items(queue)

    assert len(items) == 2
    assert items[0].running is False
    assert items[0].index == 1
    assert items[1].running is False
    assert items[1].index == 2


def test_queue_display_running_only():
    """Running item with no pending items: single unnumbered entry."""
    from agent13.commands import format_queue_items

    queue = AgentQueue()
    queue.add("only item")
    queue.get_next()

    items = format_queue_items(queue)

    assert len(items) == 1
    assert items[0].running is True
    assert items[0].index == 0


def test_queue_delete_index_matches_display():
    """/delete q N index must match the number shown by /queue.

    Regression test for the bug where /queue numbered the running item as 1,
    so the first pending item appeared as 2 — but /delete q operated on
    pending-only indices, making /delete q 2 fail with out-of-range.
    """
    from agent13.commands import format_queue_items, _delete_queue
    from unittest.mock import MagicMock

    queue = AgentQueue()
    queue.add("running item")
    queue.add("pending item")
    queue.get_next()  # "running item" becomes current/running

    # What the user sees: the pending item is index 1 (not 2)
    items = format_queue_items(queue)
    pending = [it for it in items if not it.running]
    assert len(pending) == 1
    assert pending[0].index == 1

    # /delete q 1 should remove it — previously this would have failed or
    # removed the wrong item because the display offset was wrong.
    agent = MagicMock()
    agent.queue = queue
    result = _delete_queue(agent, "1")

    assert result.success
    assert queue.pending_count == 0


def test_queue_delete_last_matches_display():
    """'last' keyword in /delete q should refer to the last pending item,
    consistent with the display numbering."""
    from agent13.commands import format_queue_items, _delete_queue
    from unittest.mock import MagicMock

    queue = AgentQueue()
    queue.add("first pending")
    queue.add("second pending")
    queue.get_next()  # "first pending" becomes running

    # Display: running (index 0), then "second pending" at index 1
    items = format_queue_items(queue)
    pending = [it for it in items if not it.running]
    assert len(pending) == 1
    assert pending[0].index == 1  # only pending item, also the "last"

    # /delete q last should remove the last (and only) pending item
    agent = MagicMock()
    agent.queue = queue
    result = _delete_queue(agent, "last")

    assert result.success
    assert queue.pending_count == 0


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
        "/exit",
        "/clear",
        "/hist",
        "/delete",
        "/model",
        "/list",
        "/tool-response",
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


def test_command_completion_quit_not_suggested():
    """Test /quit is hidden from completions in favour of /exit.

    Regression: /quit is too similar to /queue and users accidentally quit
    when tab-completing /qu. /exit is the completion offered instead.
    """
    tui = MockAgentTUI()

    # /qu should not complete to /quit (only /queue matches)
    matches = tui._get_completions("/qu")
    assert "/quit" not in matches
    assert "/queue" in matches

    # /ex completes to /exit
    matches = tui._get_completions("/ex")
    assert "/exit" in matches
    assert "/quit" not in matches

    # /q only matches /queue now
    matches = tui._get_completions("/q")
    assert set(matches) == {"/queue"}


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


class _ScanStub:
    """The real scanner methods, without a Textual app."""

    _get_filename_completions = AgentTUI._get_filename_completions


def _complete(partial: str) -> list[str]:
    """Call the REAL TUI filename completer.

    These tests used to run against a hand-copied duplicate of the method,
    which silently drifted: the copy still had the bug that dropped a typed
    directory name from the completion.
    """
    return _ScanStub()._get_filename_completions(partial)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A small directory tree to complete against, with cwd inside it.

    docs/          <- directory whose name is typed in full
      a.md
      sub/
        b.md
    doc.txt        <- file sharing a prefix with docs/
    """
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("a")
    (tmp_path / "docs" / "sub" / "b.md").write_text("b")
    (tmp_path / "doc.txt").write_text("c")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_filename_completion_no_at_sign():
    """Test that filename completion requires @ prefix."""
    assert _complete("docs/") == []


def test_filename_completion_just_at(tree):
    """Test @ alone lists current directory."""
    matches = _complete("@")
    assert sorted(matches) == ["@doc.txt", "@docs/"]


def test_filename_completion_partial_match(tree):
    """Test partial filename matching."""
    assert _complete("@do") == ["@doc.txt", "@docs/"]


def test_filename_completion_case_insensitive(tree):
    """Test case-insensitive matching."""
    assert _complete("@DO") == ["@doc.txt", "@docs/"]


def test_filename_completion_directories_have_slash(tree):
    """Test that directory completions end with /."""
    assert _complete("@docs") == ["@docs/"]


def test_typed_directory_name_is_not_dropped_from_the_completion(tree):
    """Typing a complete directory name completes to `dir/`, not its contents.

    Regression: `@docs` used to descend into docs/ and return `@a.md`,
    `@sub/` - the directory name vanished from the path, so accepting the
    first match produced a path that did not exist.
    """
    assert _complete("@docs") == ["@docs/"]


def test_typed_nested_directory_name_keeps_the_whole_path(tree):
    """The same bug on a nested path: `@docs/sub` -> `@docs/sub/`."""
    assert _complete("@docs/sub") == ["@docs/sub/"]


def test_filename_completion_nested_path(tree):
    """Test completion with nested path (trailing slash descends)."""
    matches = _complete("@docs/")
    assert matches == ["@docs/a.md", "@docs/sub/"]


def test_nested_partial_file_keeps_the_directory_prefix(tree):
    """A partial inside a directory still carries the directory prefix."""
    assert _complete("@docs/su") == ["@docs/sub/"]


def test_filename_completion_tilde_expansion(tmp_path, monkeypatch):
    """Test ~ expansion for home directory."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows expanduser

    assert _complete("@~") == ["@~/"]
    assert _complete("@~/docs") == ["@~/docs/"]
    assert _complete("@~/docs/") == ["@~/docs/a.md"]


def test_filename_completion_nonexistent_directory(tree):
    """Test completion for non-existent directory returns empty."""
    assert _complete("@nonexistent_xyz_dir/") == []


def test_filename_completion_limit(tree):
    """Test that completion is limited to 50 matches."""
    for i in range(60):
        (tree / f"f{i:02d}.txt").write_text("x")
    assert len(_complete("@")) == 50


# ===== Cycling: Tab walks the LEVEL, not just the prefix matches =====
#
# Tab completes to the first prefix match, then keeps moving through the
# siblings at that level - a unique match must not trap the user inside a
# directory. `→` opens a directory; a path typed in full leaves Tab inert.


def _scan(partial: str):
    """(matches, level) for an @path partial, from the real scanner."""
    return filename_completion_scan(partial)


def _plan(matches, level, partial):
    return plan_completion(matches, level, partial)


def _descend(text: str, row: int, col: int):
    """The `→` gesture: descend only at the end of a directory path."""

    class _Stub:
        _get_completion_context = AgentTUI._get_completion_context

    return AgentTUI._descend_target(_Stub(), text, (row, col))


def test_scan_level_is_the_parent_when_a_directory_is_named_in_full(tree):
    """`@docs` completes to `@docs/`; the level to cycle is the PARENT."""
    matches, level = _scan("@docs")
    assert matches == ["@docs/"]
    assert level == ["@doc.txt", "@docs/"]


def test_scan_level_is_the_directory_when_a_slash_was_typed(tree):
    """`@docs/` descends: matches and level are both the directory's entries."""
    matches, level = _scan("@docs/")
    assert matches == ["@docs/a.md", "@docs/sub/"]
    assert level == matches


def test_scan_level_keeps_the_whole_level_for_a_partial(tree):
    """A partial's level is unfiltered, so Tab can move past its match."""
    matches, level = _scan("@docs/su")
    assert matches == ["@docs/sub/"]
    assert level == ["@docs/a.md", "@docs/sub/"]


def test_plan_completion_cycles_the_level_after_a_unique_match():
    """The reported trap: one match, but Tab must still move on."""
    plan = _plan(["@docs/"], ["@doc.txt", "@docs/", "@done/"], "@doc")
    assert plan == (["@doc.txt", "@docs/", "@done/"], 1)


def test_plan_completion_is_inert_on_a_path_typed_in_full():
    """Tab must never replace a complete path the user typed."""
    assert _plan(["@docs/a.md"], ["@docs/a.md", "@docs/sub/"], "@docs/a.md") is None


def test_plan_completion_cycles_when_another_entry_extends_the_text(tree):
    """`@a.py` is complete but also a prefix of `@a.py.bak` - still cycles."""
    (tree / "a.py").write_text("x")
    (tree / "a.py.bak").write_text("y")
    matches, level = _scan("@a.py")
    assert matches == ["@a.py", "@a.py.bak"]
    assert _plan(matches, level, "@a.py") == (level, 0)


def test_plan_completion_nothing_matched():
    assert _plan([], [], "@zzz") is None


def test_right_arrow_opens_a_directory_at_the_end_of_the_line(tree):
    assert _descend("@docs/", 0, 6) == ("@docs/", (0, 0), (0, 6))


def test_right_arrow_opens_a_directory_named_without_a_slash(tree):
    assert _descend("@docs", 0, 5) == ("@docs", (0, 0), (0, 5))


def test_right_arrow_is_ordinary_cursor_movement_mid_line(tree):
    """Mid-line `→` must move the cursor, not rewrite the path."""
    assert _descend("@docs/ x", 0, 6) is None


def test_right_arrow_does_not_open_a_file(tree):
    assert _descend("@doc.txt", 0, 8) is None


def test_right_arrow_needs_an_at_reference(tree):
    """A line ending in a real directory path is not a file reference."""
    assert _descend("see /tmp/", 0, 9) is None


class _FakeField:
    """Just enough TextArea for _apply_completion (single-line spans)."""

    def __init__(self, text: str):
        self.text = text
        self.cursor_location = (0, len(text))

    def replace(self, completion: str, start, end) -> None:
        self.text = self.text[: start[1]] + completion + self.text[end[1] :]

    def move_cursor(self, location) -> None:
        self.cursor_location = location


class _CompletionStub:
    """The real completion methods on a stub app (no Textual, no widgets)."""

    _get_completions_for_context = AgentTUI._get_completions_for_context
    _get_filename_completions = AgentTUI._get_filename_completions
    _apply_completion = AgentTUI._apply_completion
    _reset_completion_state = AgentTUI._reset_completion_state
    _start_completion = AgentTUI._start_completion

    def __init__(self):
        self._completion_matches = []
        self._completion_index = 0
        self._completion_prefix = ""
        self._completion_start = (0, 0)
        self._completion_end = (0, 0)
        self._completion_text = ""
        self._info_pane_mode = None
        self.shown = None
        self.hidden = False

    def _show_completions(self, matches, current=None):
        self.shown = (matches, current)

    def _hide_info_pane(self):
        self.hidden = True


def test_start_completion_enters_a_cycle_over_the_level(tree):
    """`@docs` + Tab completes, and leaves Tab able to move to `@doc.txt`."""
    stub = _CompletionStub()
    field = _FakeField("@docs")

    stub._start_completion(field, "@filename", "@docs", "@docs", (0, 0), (0, 5))

    assert field.text == "@docs/"
    assert stub._completion_matches == ["@doc.txt", "@docs/"]
    assert stub._completion_index == 1
    assert stub._completion_text == "@docs/"
    assert stub.shown == (["@doc.txt", "@docs/"], "@docs/")


def test_start_completion_is_inert_on_a_complete_path(tree):
    """No rewrite, no cycle, no pane - Tab does nothing."""
    stub = _CompletionStub()
    field = _FakeField("@docs/a.md")

    stub._start_completion(field, "@filename", "@docs/a.md", "@docs/a.md", (0, 0), (0, 9))

    assert field.text == "@docs/a.md"
    assert stub._completion_matches == []
    assert stub.shown is None


class TestPoliteWaitOverlay:
    """The "wait: Ns" overlay must clear when the wait ends WITHOUT the lock.

    Reported live 2026-11-09: /polite off during a lock wait started the turn
    correctly, but the status bar stayed on "wait: 0s" for the whole turn -
    POLITE_ACQUIRED never fires on that path and the status stays
    "processing", so nothing cleared it.
    """

    def _app(self):
        from unittest.mock import MagicMock

        app = MagicMock()
        app._polite_wait_elapsed = 12.0
        app._info_pane_mode = None
        return app

    def test_the_overlay_is_cleared(self):
        app = self._app()

        AgentTUI._handle_polite_wait_ended(app)

        app._clear_polite_waiting.assert_called_once_with()

    def test_the_queue_pane_is_refreshed_when_it_is_open(self):
        """It still says "Waiting for lock" otherwise."""
        app = self._app()
        app._info_pane_mode = "queue"

        AgentTUI._handle_polite_wait_ended(app)

        app.call_later.assert_called_once_with(app._show_queue)

    def test_no_pane_refresh_when_the_queue_pane_is_not_open(self):
        app = self._app()

        AgentTUI._handle_polite_wait_ended(app)

        app.call_later.assert_not_called()


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


class TestCwdCommand:
    """Tests for /cwd command logic."""

    def test_show_current_directory(self):
        """Bare /cwd shows the current working directory."""
        cwd = Path.cwd()
        assert cwd.exists()
        assert cwd.is_dir()

    def test_change_to_existing_directory(self, tmp_path):
        """Changing to an existing directory succeeds."""
        original = Path.cwd()
        try:
            os.chdir(tmp_path)
            assert Path.cwd() == tmp_path
        finally:
            os.chdir(original)

    def test_nonexistent_path_reports_error(self):
        """Changing to a nonexistent path is detected."""
        target = Path("/nonexistent_path_abc123")
        assert not target.exists()

    def test_not_a_directory_reports_error(self, tmp_path):
        """Changing to a file (not a directory) is detected."""
        test_file = tmp_path / "testfile.txt"
        test_file.write_text("hello")
        assert test_file.exists()
        assert not test_file.is_dir()

    def test_expanduser_in_path(self):
        """Tilde ~ gets expanded to home directory."""
        expanded = Path("~").expanduser()
        assert expanded.exists()
        assert expanded.is_dir()
        assert str(expanded) == str(Path.home())

    def test_resolve_symlinks(self, tmp_path):
        """Paths with symlinks get resolved."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        resolved = link_dir.resolve()
        assert resolved == real_dir

    def test_chdir_restores_original(self, tmp_path):
        """Test cleanup: cwd is restored after test."""
        original = Path.cwd()
        os.chdir(tmp_path)
        assert Path.cwd() != original
        os.chdir(original)
        assert Path.cwd() == original

    def test_at_prefix_stripped_from_args(self, tmp_path):
        """Args with @ prefix (from autocomplete) get stripped before chdir."""
        args_with_at = f"@{tmp_path}"
        # Simulate the handler's stripping logic
        target = Path(args_with_at.strip().lstrip("@")).expanduser().resolve()
        assert target == tmp_path
        assert target.is_dir()

    def test_at_prefix_single_char_stripped(self, tmp_path):
        """Single @ as sole arg becomes empty after strip, falls through to 'bare'."""
        args = "@"
        stripped = args.strip().lstrip("@")
        # After stripping, this is empty — same as bare /cwd (show current dir)
        assert stripped == ""


# ============== /provider completion Tests ==============


class TestProviderCompletion:
    """Tests for /provider command showing navigable list."""

    def test_get_provider_completions_returns_config_names(self):
        """_complete_provider returns provider names from config."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.providers = [
            MagicMock(name="alpha"),
            MagicMock(name="beta"),
            MagicMock(name="gamma"),
        ]
        mock_config.providers[0].name = "alpha"
        mock_config.providers[1].name = "beta"
        mock_config.providers[2].name = "gamma"

        from agent13.commands import _complete_provider

        with patch("agent13.config.get_config", return_value=mock_config):
            result = _complete_provider(None, "")

        assert result == ["alpha", "beta", "gamma"]

    def test_get_provider_completions_filters_partial(self):
        """_complete_provider filters by partial input."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.providers = [
            MagicMock(name="openai"),
            MagicMock(name="openrouter"),
            MagicMock(name="local"),
        ]
        mock_config.providers[0].name = "openai"
        mock_config.providers[1].name = "openrouter"
        mock_config.providers[2].name = "local"

        from agent13.commands import _complete_provider

        with patch("agent13.config.get_config", return_value=mock_config):
            result = _complete_provider(None, "open")

        assert result == ["openai", "openrouter"]
        assert "local" not in result

    def test_handle_provider_no_args_enters_completion(self):
        """/provider with no args opens the picker at the current provider.

        The listing itself is shared (execute_provider); the picker is the
        TUI's own tail, driven by data["providers"].
        """
        from unittest.mock import MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        app._enter_command_completion = MagicMock()
        result = CommandResult(
            True,
            "Available providers:\n  1. alpha *\n  2. beta",
            data={"providers": ["alpha", "beta"], "current": "alpha"},
        )

        consumed = AgentTUI._run_command_follow_up(app, result, "")

        assert consumed is True
        app._enter_command_completion.assert_called_once_with(
            "provider", ["alpha", "beta"], "alpha"
        )

    def test_handle_provider_no_args_no_providers_shows_error(self):
        """/provider with no args and no providers shows error."""
        from unittest.mock import MagicMock, patch

        from agent13 import commands as C

        with patch("agent13.config.get_provider_names", return_value=[]):
            result = C.run_command("provider", MagicMock(), "")

        assert not result.success
        assert "No providers" in result.message


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])

    def test_get_provider_completions_numeric(self):
        """_get_provider_completions resolves numeric input."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.providers = [
            MagicMock(name="alpha"),
            MagicMock(name="beta"),
            MagicMock(name="gamma"),
        ]
        mock_config.providers[0].name = "alpha"
        mock_config.providers[1].name = "beta"
        mock_config.providers[2].name = "gamma"

        app = MagicMock()
        with patch("ui.tui.get_config", return_value=mock_config):
            from ui.tui import AgentTUI

            result = AgentTUI._get_provider_completions(app, "2")

        assert result == ["beta"]

    def test_get_provider_completions_numeric_out_of_range(self):
        """_get_provider_completions returns empty for out-of-range number."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.providers = [
            MagicMock(name="alpha"),
            MagicMock(name="beta"),
        ]
        mock_config.providers[0].name = "alpha"
        mock_config.providers[1].name = "beta"

        app = MagicMock()
        with patch("ui.tui.get_config", return_value=mock_config):
            from ui.tui import AgentTUI

            result = AgentTUI._get_provider_completions(app, "99")

        assert result == []
# ============== Retry Command Tests ==============


class TestRetryTail:
    """/retry: the shared handler removes the group, the TUI loads the text.

    The REPL re-queues the same text instead - that tail is its own.
    """

    def test_follow_up_loads_the_text_into_the_input(self):
        from unittest.mock import MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        result = CommandResult(True, "Retrying", {"user_text": "second question"})

        consumed = AgentTUI._run_command_follow_up(app, result, "")

        assert consumed is True
        app._load_retry_text.assert_called_once_with("second question")

    def test_load_retry_text_fills_the_input_field(self):
        from unittest.mock import MagicMock

        from ui.tui import AgentTUI

        app = MagicMock()
        field = MagicMock()
        app.query_one.return_value = field

        AgentTUI._load_retry_text(app, "second question")

        assert field.text == "second question"
        field.focus.assert_called_once()
        app.update_status.assert_called_once()


# ============== Queued-turn tails ==============
# /wrapup, /chain and /journal last|all all leave one agent turn in
# data["agent_message"] as (text, kind, data); the TUI schedules it and still
# renders the handler's confirmation.


class TestQueuedTurnTail:
    async def test_journal_queues_the_agent_message(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        app.agent.add_message = AsyncMock()
        result = CommandResult(
            True,
            "Journaling the most recent tool-using turn",
            {"agent_message": ("/journal last", "journal_last", {})},
        )

        consumed = AgentTUI._run_command_follow_up(app, result, "last")
        await asyncio.sleep(0)  # let the scheduled task run

        assert consumed is False, "the confirmation must still render"
        app.agent.add_message.assert_awaited_once_with(
            "/journal last", kind="journal_last", data={}
        )

    async def test_wrapup_queues_the_turn_with_its_focus_name(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from agent13.commands import execute_wrapup
        from ui.tui import AgentTUI

        app = MagicMock()
        app.agent.add_message = AsyncMock()

        consumed = AgentTUI._run_command_follow_up(
            app, execute_wrapup(app.agent, "  my cool game  "), "my cool game"
        )
        await asyncio.sleep(0)

        assert consumed is False
        app.agent.add_message.assert_awaited_once_with(
            "/wrapup my cool game", kind="wrapup", data={"focus_name": "my cool game"}
        )

    async def test_chain_queues_the_no_skills_flag(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from agent13.commands import execute_chain
        from ui.tui import AgentTUI

        app = MagicMock()
        app.agent.add_message = AsyncMock()

        AgentTUI._run_command_follow_up(app, execute_chain(app.agent, "no-skills"), "no-skills")
        await asyncio.sleep(0)

        app.agent.add_message.assert_awaited_once_with(
            "/chain", kind="chain", data={"no_skills": True}
        )

    async def test_new_session_queues_the_deferred_clear(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from agent13.commands import execute_new_session
        from ui.tui import AgentTUI

        app = MagicMock()
        app.agent.request_clear = AsyncMock()

        consumed = AgentTUI._run_command_follow_up(
            app, execute_new_session(app.agent, ""), ""
        )
        await asyncio.sleep(0)

        # Not consumed - the handler's "New session queued" still renders.
        assert consumed is False
        app.agent.request_clear.assert_awaited_once_with(mode="all")
        # The display is reset by the MESSAGES_CLEARED handler, not here.
        app.agent.clear_messages.assert_not_called()

    async def test_load_queues_and_records_the_path(self):
        import asyncio
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock

        from agent13.commands import execute_load
        from ui.tui import AgentTUI

        app = MagicMock()
        app.agent.request_load = AsyncMock()

        result = execute_load(app.agent, "mine")
        result.data["load_path"] = "/tmp/mine.ctx"  # bypass the filesystem check

        consumed = AgentTUI._run_command_follow_up(app, result, "mine")
        await asyncio.sleep(0)

        assert consumed is False
        # Recorded first so on_context_loaded can capture the resumed stats.
        assert app._pending_load_path == Path("/tmp/mine.ctx")
        app.agent.request_load.assert_awaited_once_with("/tmp/mine.ctx")


# ============== Bell Command Tests ==============


class TestBellCommandValidation:
    """Tests for BellManager.validate_command logic."""

    def test_valid_command(self):
        """A known executable validates successfully (cross-platform)."""
        import sys
        import os
        from agent13.bell import BellManager

        exe = os.path.basename(sys.executable)
        assert BellManager.validate_command(f"{exe} hello") is True

    def test_invalid_command(self):
        """A nonexistent executable fails validation."""
        from agent13.bell import BellManager

        assert BellManager.validate_command("nonexistent_xyz_123") is False

    def test_empty_command(self):
        """Empty string fails validation."""
        from agent13.bell import BellManager

        assert BellManager.validate_command("") is False

    def test_whitespace_only(self):
        """Whitespace-only string fails validation."""
        from agent13.bell import BellManager

        assert BellManager.validate_command("   ") is False

    def test_command_with_args(self):
        """Command with arguments validates on first token only."""
        import sys
        import os
        from agent13.bell import BellManager

        exe = os.path.basename(sys.executable)
        assert BellManager.validate_command(f"{exe} --flag value") is True

    def test_unquoted_path_not_stripped(self):
        """An unquoted path like ./script is not modified."""
        from agent13.bell import BellManager

        # ./echo doesn't exist but the validator should handle it gracefully
        result = BellManager.validate_command("./nonexistent_script")
        assert result is False


# ============== /load completion Tests ==============


def _make_tui_app():
    """Build an AgentTUI with stubs, matching test_tui_client_loop's pattern."""
    from unittest.mock import MagicMock, patch

    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        return ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
        )


class TestLoadCompletions:
    """/load tab completion lists saves; failures surface, not silence.

    Regression: a user on Windows (running from his home dir) saw auto-saves
    written but /load <tab> showed nothing — list_all_saves() raising was
    swallowed by `except Exception: return []`, indistinguishable from "no
    saves".
    """

    def test_lists_save_stems(self):
        from unittest.mock import patch

        from agent13.commands import _complete_load

        with patch(
            "agent13.persistence.list_all_saves",
            return_value=[Path("/saves/manual.ctx"), Path("/saves/2026-09-22.ctx")],
        ):
            result = _complete_load(None, "")
        assert result == ["manual", "2026-09-22"]

    def test_filters_by_partial(self):
        from unittest.mock import patch

        from agent13.commands import _complete_load

        with patch(
            "agent13.persistence.list_all_saves",
            return_value=[Path("/saves/manual.ctx"), Path("/saves/2026-09-22.ctx")],
        ):
            result = _complete_load(None, "2026")
        assert result == ["2026-09-22"]

    def test_failure_shows_hint_with_error(self):
        """A raising list_all_saves yields a visible hint, not an empty list."""
        from unittest.mock import patch

        from agent13.commands import _complete_load

        with patch(
            "agent13.persistence.list_all_saves",
            side_effect=PermissionError("[WinError 5] Access is denied"),
        ):
            result = _complete_load(None, "")
        assert len(result) == 1
        assert result[0].startswith("saves list unavailable:")
        assert "Access is denied" in result[0]

    def test_failure_hint_is_truncated(self):
        """Long exception strings can't blow out the completion dropdown."""
        from unittest.mock import patch

        from agent13.commands import _complete_load

        with patch(
            "agent13.persistence.list_all_saves",
            side_effect=RuntimeError("x" * 500),
        ):
            result = _complete_load(None, "")
        assert len(result) == 1
        assert len(result[0]) <= 100


# ============== Prompt / snippet tails ==============


class TestPromptTail:
    """/prompt with no args opens the TUI picker instead of printing a list."""

    def test_follow_up_opens_the_picker(self):
        from unittest.mock import MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        app._enter_command_completion = MagicMock()
        result = CommandResult(
            True,
            "Available prompts:\n  * default\n    concise",
            data={"prompt_picker": (["default", "concise"], "default")},
        )

        consumed = AgentTUI._run_command_follow_up(app, result, "")

        assert consumed is True
        app._enter_command_completion.assert_called_once_with(
            "prompt", ["default", "concise"], "default"
        )


class TestSnippetTail:
    """/snippet: the TUI fills the input and keeps its slash commands in step."""

    def test_use_fills_the_input(self):
        from unittest.mock import MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        field = MagicMock()
        app.query_one.return_value = field
        result = CommandResult(
            True, "Snippet 'mine':\nthe text", {"snippet_text": "the text"}
        )

        consumed = AgentTUI._run_command_follow_up(app, result, "mine")

        assert consumed is True
        assert field.text == "the text"
        field.focus.assert_called_once()

    def test_add_registers_the_slash_command(self):
        from unittest.mock import MagicMock

        from ui.tui import AgentTUI

        app = MagicMock()
        app.SLASH_COMMANDS = list(AgentTUI._BUILTIN_SLASH_COMMANDS)
        app._BUILTIN_SLASH_COMMANDS = AgentTUI._BUILTIN_SLASH_COMMANDS

        AgentTUI._add_snippet_slash_command(app, "mine")

        assert "/mine" in app.SLASH_COMMANDS

    def test_add_skips_a_name_that_collides_with_a_builtin(self):
        from unittest.mock import MagicMock

        from ui.tui import AgentTUI

        app = MagicMock()
        app.SLASH_COMMANDS = list(AgentTUI._BUILTIN_SLASH_COMMANDS)
        app._BUILTIN_SLASH_COMMANDS = AgentTUI._BUILTIN_SLASH_COMMANDS

        AgentTUI._add_snippet_slash_command(app, "help")

        assert app.SLASH_COMMANDS.count("/help") == 1

    def test_delete_drops_the_slash_command(self):
        from unittest.mock import MagicMock

        from ui.tui import AgentTUI

        app = MagicMock()
        app.SLASH_COMMANDS = list(AgentTUI._BUILTIN_SLASH_COMMANDS) + ["/mine"]
        app._BUILTIN_SLASH_COMMANDS = AgentTUI._BUILTIN_SLASH_COMMANDS

        AgentTUI._remove_snippet_slash_command(app, "mine")

        assert "/mine" not in app.SLASH_COMMANDS

    def test_follow_up_dispatches_add_and_remove(self):
        from unittest.mock import MagicMock

        from agent13.commands import CommandResult
        from ui.tui import AgentTUI

        app = MagicMock()
        result = CommandResult(
            True,
            "Renamed snippet 'old' -> 'new'",
            {"snippet_removed": "old", "snippet_added": "new"},
        )

        consumed = AgentTUI._run_command_follow_up(app, result, "rename old new")

        assert consumed is False, "the confirmation still renders"
        app._remove_snippet_slash_command.assert_called_once_with("old")
        app._add_snippet_slash_command.assert_called_once_with("new")
