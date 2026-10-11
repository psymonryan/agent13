"""Tests for ui.display module."""

from unittest.mock import MagicMock, patch
from io import StringIO

from rich.console import Console
from rich.panel import Panel

from ui.display import RichDisplay


class TestRichDisplay:
    """Tests for RichDisplay class."""

    def test_create_display_default(self):
        """Should create display with default settings."""
        display = RichDisplay()

        assert display.pretty is True
        assert display.debug is False
        # Widgets fold by default (interactive UIs); batch mode opts out.
        assert display.reasoning_collapsed is True
        assert display.toolview_collapsed is True
        assert display._in_reasoning is False
        assert display._in_content is False
        assert display._response_buffer == ""

    def test_create_display_with_options(self):
        """Should create display with custom settings."""
        console = Console()
        display = RichDisplay(console=console, pretty=False, debug=True)

        assert display.console is console
        assert display.pretty is False
        assert display.debug is True

    def test_start_waiting_pretty(self):
        """Should start spinner in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.start_waiting()

        # Should have called console.status
        assert console.status.called
        assert display._status is not None

        # Clean up
        display.stop_waiting()

    def test_start_waiting_non_pretty(self):
        """Should not start spinner in non-pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display.start_waiting()

        # Should not have called console.status
        assert not console.status.called
        assert display._status is None

    def test_stop_waiting(self):
        """Should stop spinner."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.start_waiting()
        assert display._status is not None

        display.stop_waiting()

        assert display._status is None

    def test_add_token_accumulates(self):
        """Should accumulate tokens in buffer."""
        # Use a real Console to avoid mock issues with Live
        console = Console(file=StringIO(), force_terminal=True)
        display = RichDisplay(console=console, pretty=True)

        display.start_response()
        display.add_token("Hello")
        display.add_token(" ")
        display.add_token("World")

        assert display._response_buffer == "Hello World"

        # Clean up
        display.complete_response()

    def test_add_token_debug_mode(self):
        """Should print tokens with markers in debug mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True, debug=True)

        display.start_response()
        display.add_token("Hello")
        display.add_token("World")

        # Debug mode prints directly with markers
        assert console.print.call_count >= 2

    def test_add_token_plain_mode(self):
        """Should print tokens directly in plain mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display.start_response()
        display.add_token("Hello")
        display.add_token("World")

        # Plain mode prints directly via console
        assert console.print.call_count >= 2

    def test_add_reasoning_pretty(self):
        """Should print reasoning in italic in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, reasoning_collapsed=False
        )

        display.add_reasoning("thinking...")

        # Should print with italic style
        console.print.assert_called()
        call_args = str(console.print.call_args)
        assert "italic" in call_args

    def test_add_reasoning_plain(self):
        """Should print reasoning without styling in plain mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=False, reasoning_collapsed=False
        )

        display.add_reasoning("thinking...")

        # Should print via console (Thinking: prefix + text)
        assert console.print.call_count >= 1

    def test_reasoning_to_content_transition(self):
        """Should handle transition from reasoning to content."""
        display = RichDisplay(pretty=False, reasoning_collapsed=False)

        display.add_reasoning("thinking...")
        assert display._in_reasoning is True
        assert display._in_content is False

        # start_response resets _in_reasoning and sets _in_content
        display.start_response()
        assert display._in_content is True

    def test_show_tool_call_pretty(self):
        """Should show tool call in panel in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_call("square_number", {"n": 5})

        # Should have printed a panel
        console.print.assert_called()

    def test_show_tool_call_plain(self):
        """Should show tool call as text in plain mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display.show_tool_call("square_number", {"n": 5})

        # Should have printed via console
        assert console.print.call_count >= 1

    def test_show_tool_result(self):
        """Should show tool result."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, toolview_collapsed=False
        )

        display.show_tool_result("25")

        console.print.assert_called()

    def test_show_tool_result_truncates_long(self):
        """Should truncate long tool results in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, toolview_collapsed=False
        )

        long_result = "x" * 600
        display.show_tool_result(long_result)

        # Should have truncated
        call_args = str(console.print.call_args)
        assert "..." in call_args

    def test_show_error_pretty(self):
        """Should show error in red in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_error("Something went wrong")

        console.print.assert_called()
        call_args = str(console.print.call_args)
        assert "red" in call_args.lower()

    def test_show_error_plain(self):
        """Should show error in plain mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display.show_error("Something went wrong")

        # Should have printed via console
        assert console.print.call_count >= 1

    def test_show_separator_pretty(self):
        """Should show separator in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_separator()

        console.print.assert_called()

    def test_show_separator_plain(self):
        """Should not show separator in plain mode."""
        display = RichDisplay(pretty=False)

        with patch("builtins.print") as mock_print:
            display.show_separator()

            # Plain mode doesn't show separator (no print calls)
            mock_print.assert_not_called()

    def test_print_with_style(self):
        """Should print with Rich styling in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display._print("Hello", style="bold red")

        console.print.assert_called_with("Hello", style="bold red", end="\n")

    def test_print_without_style(self):
        """Should print without styling in plain mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display._print("Hello")

        console.print.assert_called_with("Hello", end="\n", highlight=False)

    def test_reset_state_on_start_response(self):
        """Should reset state when starting new response."""
        display = RichDisplay(pretty=False)

        # Simulate previous response
        display._response_buffer = "old content"
        display._in_reasoning = True

        display.start_response()

        assert display._response_buffer == ""
        assert display._in_reasoning is False
        assert display._in_content is True

    def test_live_display_pretty_mode(self):
        """Should use Live display for markdown in pretty mode."""
        # Use a real Console with StringIO for this test
        console = Console(file=StringIO(), force_terminal=True)
        display = RichDisplay(console=console, pretty=True)

        display.start_response()
        display.add_token("Hello")

        # Should have created a Live display
        assert display._live_display is not None

        # Clean up
        display.complete_response()

    def test_no_live_display_debug_mode(self):
        """Should not use Live display in debug mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True, debug=True)

        display.start_response()
        display.add_token("Hello")

        # Should not have created a Live display
        assert display._live_display is None

        display.complete_response()

    def test_no_live_display_plain_mode(self):
        """Should not use Live display in plain mode."""
        display = RichDisplay(pretty=False)

        display.start_response()
        display.add_token("Hello")

        # Should not have created a Live display
        assert display._live_display is None

        display.complete_response()


class TestCollapsedDisplay:
    """The folded (default) view: the REPL has no widgets, so it prints less.

    Reasoning prints a one-line "Thoughts: <first line>" hint when the block
    ends - the linear stream cannot expand a folded block later, and the full
    thought is the bulk. A tool call prints one dim line carrying its "what"
    (the command running, the file being edited): that line is the "what is it
    doing right now" signal.
    """

    @staticmethod
    def _printed(console):
        """The text arguments of every console.print call."""
        return [c.args[0] if c.args else "" for c in console.print.call_args_list]

    def test_defaults_are_folded(self):
        display = RichDisplay()
        assert display.reasoning_collapsed is True
        assert display.toolview_collapsed is True

    def test_flags_are_public_for_the_repl_command(self):
        """/reasoning and /toolview set these on the live display."""
        display = RichDisplay()
        display.reasoning_collapsed = False
        display.toolview_collapsed = False
        assert display.reasoning_collapsed is False
        assert display.toolview_collapsed is False

    def test_folded_reasoning_prints_nothing(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.add_reasoning("thinking...")

        assert console.print.call_args_list == []
        # No "Thinking:" prefix, so the state flag stays down
        assert display._in_reasoning is False

    def test_folded_reasoning_prints_a_one_line_hint(self):
        """start_response() shows the block's one-line 'what', not the block.

        The hint is the first line of the thought (not the whole block), and no
        empty newline is printed - that only happens for expanded reasoning.
        """
        console = MagicMock(spec=Console)
        console.size.width = 120
        display = RichDisplay(console=console, pretty=True, debug=True)

        display.add_reasoning("let me look at the log first\nthen the failing test")
        display.start_response()

        printed = self._printed(console)
        assert any("Assistant:" in str(p) for p in printed)
        assert any("Thoughts:" in str(p) for p in printed)
        assert "let me look at the log first" in str(printed)
        assert "failing test" not in str(printed), "only the first line, not the whole block"
        assert "" not in printed, "folded reasoning must not print a newline"

    def test_expanded_reasoning_does_end_its_line(self):
        """The control: expanded reasoning DOES print the trailing newline."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console,
            pretty=True,
            debug=True,
            reasoning_collapsed=False,
        )

        display.add_reasoning("thinking...")
        display.start_response()

        assert "" in self._printed(console)

    def test_folded_tool_call_is_one_dim_line(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_call("write_file", {"filepath": "test.py"})

        printed = self._printed(console)
        assert len(printed) == 1
        assert "⚙ write_file" in printed[0]
        assert "test.py" in printed[0], "the one line carries the 'what'"
        assert "{" not in printed[0], "the full args JSON stays folded"
        assert "dim" in printed[0]

    def test_folded_tool_result_prints_nothing(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_result("x" * 5000)

        assert console.print.call_args_list == []

    def test_folded_tool_call_plain_mode_has_no_markup(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=False)

        display.show_tool_call("write_file", {"filepath": "test.py"})

        assert self._printed(console) == ["⚙ write_file: test.py"]

    def test_expanded_tool_call_prints_a_panel(self):
        """The control: expanded keeps the old panel with args."""
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, toolview_collapsed=False
        )

        display.show_tool_call("write_file", {"filepath": "test.py"})

        args = console.print.call_args_list[0].args
        assert isinstance(args[0], Panel)
        assert "test.py" in str(args[0].renderable)

    def test_expanded_tool_result_prints_the_body(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, toolview_collapsed=False
        )

        display.show_tool_result("25")

        assert any("25" in str(p) for p in self._printed(console))

    def test_folded_tool_call_line_carries_the_seq(self):
        """The gear line numbers the call so its result can be paired with it."""
        console = MagicMock(spec=Console)
        console.size.width = 100
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_call("command", {"command": "ls -la"}, seq=1)

        assert self._printed(console) == ["[dim]⚙ command 1: ls -la[/]"]

    def test_folded_tool_call_line_without_seq_is_unnumbered(self):
        console = MagicMock(spec=Console)
        console.size.width = 100
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_call("command", {"command": "ls -la"})

        assert self._printed(console) == ["[dim]⚙ command: ls -la[/]"]

    def test_folded_tool_call_seq_shrinks_the_what_budget(self):
        """'⚙ name N: ' is wider than '⚙ name: ', so the hint truncates earlier."""
        console = MagicMock(spec=Console)
        console.size.width = 40
        display = RichDisplay(console=console, pretty=True)
        long_cmd = "command" * 20

        display.show_tool_call("command", {"command": long_cmd}, seq=1)
        numbered = self._printed(console)[0]
        console.print.call_args_list.clear()

        display.show_tool_call("command", {"command": long_cmd})
        unnumbered = self._printed(console)[0]

        assert "…" in numbered and "…" in unnumbered
        numbered_what = numbered.split(": ", 1)[1].removesuffix("[/]")
        unnumbered_what = unnumbered.split(": ", 1)[1].removesuffix("[/]")
        assert len(numbered_what) == len(unnumbered_what) - 2, (
            "the number costs cells in the what budget"
        )

    def test_expanded_tool_result_line_carries_the_seq(self):
        console = MagicMock(spec=Console)
        display = RichDisplay(
            console=console, pretty=True, toolview_collapsed=False
        )

        display.show_tool_result("total 8", seq=2)

        assert any("→ 2: total 8" in str(p) for p in self._printed(console))
