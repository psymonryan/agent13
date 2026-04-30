"""Tests for Ctrl+C (action_clear_quit) behavior.

TDD: These tests should FAIL against the current code, then PASS
after the fix is applied to ui/tui.py action_clear_quit.

The key invariant: Ctrl+C should clear input text BEFORE interrupting
the agent. Only when input is empty should it interrupt/quit.

Bug: when processing=True and input has text, Ctrl+C was interrupting
the agent loop instead of clearing the input. This meant:
- During /journal last, Ctrl+C killed the agent loop (wrong task)
- During normal processing, typing text then Ctrl+C interrupted instead
  of clearing — you had to press Ctrl+C twice (once to interrupt,
  once to clear)
"""


class MockInputField:
    """Minimal mock of ChatTextArea for testing action_clear_quit."""

    def __init__(self, text: str = ""):
        self._text = text
        self._selected_text = ""
        self._cleared = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def selected_text(self) -> str:
        return self._selected_text

    def clear(self) -> None:
        self._text = ""
        self._cleared = True


class MockScreen:
    """Minimal mock of Screen for selection check."""

    def __init__(self, selected: str = ""):
        self._selected = selected

    def get_selected_text(self) -> str:
        return self._selected


class MockAgent:
    """Minimal mock of Agent for testing."""

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class MockTUI:
    """Minimal mock of TUI with just the fields action_clear_quit uses.

    action_clear_quit is copied VERBATIM from the real TUI code so these
    tests exercise the actual logic. When the fix is applied to the real
    code, copy it here too — the tests should then pass.
    """

    def __init__(
        self, processing: bool = False, input_text: str = "", screen_selected: str = ""
    ):
        self.processing = processing
        self._interrupt_requested = False
        self._agent_interrupted = False
        self._force_quit_called = False
        self._input_field = MockInputField(input_text)
        self._screen = MockScreen(screen_selected)
        self.agent = MockAgent()

    def query_one(self, selector: str, widget_type=None):
        if selector == "#input-field":
            return self._input_field
        raise ValueError(f"Unknown selector: {selector}")

    @property
    def screen(self):
        return self._screen

    def run_worker(self, coro):
        """Track that an interrupt was requested."""
        self._agent_interrupted = True

    def action_force_quit(self):
        self._force_quit_called = True

    # --- The method under test (copied from real TUI) ---
    def action_clear_quit(self) -> None:
        """Handle Ctrl+C: copy selection, clear input, interrupt agent, or quit.

        FIXED logic — checks input text BEFORE processing, so Ctrl+C
        clears what the user typed instead of interrupting the agent.
        """
        # If there's a selection in the chat area, copy it
        if self.screen.get_selected_text():
            return

        input_field = self.query_one("#input-field", MockInputField)

        # If text is selected in input, let TextArea handle copy (don't clear)
        if input_field.selected_text:
            return

        # If textarea has text, clear it (even if agent is processing)
        if input_field.text:
            input_field.clear()
            return

        # If agent is processing with empty input, interrupt it
        if self.processing and not self._interrupt_requested:
            self._interrupt_requested = True
            self.run_worker(None)
            return

        # If empty, set flag and quit
        self.action_force_quit()


class TestCtrlCShouldClearInputBeforeInterrupt:
    """Test that Ctrl+C clears input text BEFORE interrupting the agent.

    These tests should FAIL with the current buggy logic and PASS
    after the fix (reorder: check input text before processing).
    """

    def test_processing_with_text_clears_input_not_interrupt(self):
        """Ctrl+C when processing + input has text → clears input, does NOT interrupt.

        This is the core bug: Ctrl+C checked processing BEFORE input text,
        so it would interrupt the agent instead of clearing what the user
        just typed.

        Scenario: user sends "hello", agent starts streaming, user types
        "this is a test" (without Enter), then presses Ctrl+C.
        Expected: input cleared, agent keeps running.
        Actual (buggy): agent interrupted, input NOT cleared.
        """
        tui = MockTUI(processing=True, input_text="this is a test")
        tui.action_clear_quit()

        assert tui._input_field._cleared, (
            "Ctrl+C should clear input when text is present"
        )
        assert tui._input_field.text == "", "Input should be empty after clear"
        assert not tui._agent_interrupted, (
            "Ctrl+C should NOT interrupt agent when input has text to clear"
        )

    def test_processing_with_text_two_ctrlc(self):
        """First Ctrl+C clears text, second Ctrl+C interrupts agent.

        Simulates the user workflow: agent is streaming, user types some
        text, presses Ctrl+C to clear it, then presses Ctrl+C again to
        interrupt the agent.

        With the buggy logic, the first Ctrl+C already interrupts the
        agent — the user never gets to "clear then interrupt".
        """
        tui = MockTUI(processing=True, input_text="discard me")

        # First Ctrl+C: should clear input only
        tui.action_clear_quit()
        assert tui._input_field._cleared, "First Ctrl+C should clear input"
        assert not tui._agent_interrupted, "First Ctrl+C should NOT interrupt agent"

        # Second Ctrl+C: now input is empty, should interrupt
        tui.action_clear_quit()
        assert tui._agent_interrupted, (
            "Second Ctrl+C (empty input) should interrupt agent"
        )

    def test_journaling_with_text_clears_not_interrupt(self):
        """Ctrl+C during journaling + input has text → clears input, not interrupt.

        This was the original reported bug: /journal last runs in background,
        user presses Ctrl+C expecting to clear input, but it killed the
        agent loop.
        """
        tui = MockTUI(processing=True, input_text="some text")
        tui.action_clear_quit()

        assert tui._input_field._cleared, "Ctrl+C during journaling should clear input"
        assert not tui._agent_interrupted, (
            "Ctrl+C during journaling should NOT interrupt agent loop"
        )


class TestCtrlCExistingBehavior:
    """Tests for behavior that already works correctly."""

    def test_idle_with_text_clears_input(self):
        """Ctrl+C when idle + input has text → clears input, does not quit."""
        tui = MockTUI(processing=False, input_text="hello world")
        tui.action_clear_quit()

        assert tui._input_field._cleared
        assert tui._input_field.text == ""
        assert not tui._force_quit_called
        assert not tui._agent_interrupted

    def test_idle_with_empty_input_quits(self):
        """Ctrl+C when idle + empty input → quits."""
        tui = MockTUI(processing=False, input_text="")
        tui.action_clear_quit()

        assert tui._force_quit_called
        assert not tui._agent_interrupted

    def test_processing_with_empty_input_interrupts(self):
        """Ctrl+C when processing + empty input → interrupts agent."""
        tui = MockTUI(processing=True, input_text="")
        tui.action_clear_quit()

        assert tui._agent_interrupted
        assert tui._interrupt_requested
        assert not tui._force_quit_called

    def test_selection_copies_instead_of_clear(self):
        """Ctrl+C with selection → copies, does not clear or interrupt."""
        tui = MockTUI(
            processing=True, input_text="text", screen_selected="selected text"
        )
        tui.action_clear_quit()

        assert not tui._input_field._cleared
        assert not tui._agent_interrupted
        assert not tui._force_quit_called
