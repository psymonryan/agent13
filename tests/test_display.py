"""Tests for ui.display module."""

import time
from unittest.mock import MagicMock, patch
from io import StringIO

from rich.console import Console

from ui.display import RichDisplay


class TestRichDisplay:
    """Tests for RichDisplay class."""

    def test_create_display_default(self):
        """Should create display with default settings."""
        display = RichDisplay()

        assert display.pretty is True
        assert display.debug is False
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
        display = RichDisplay(pretty=False)

        display.start_response()
        with patch("builtins.print") as mock_print:
            display.add_token("Hello")
            display.add_token("World")

            # Plain mode prints directly
            assert mock_print.call_count >= 2

    def test_add_reasoning_pretty(self):
        """Should print reasoning in italic in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.add_reasoning("thinking...")

        # Should print with italic style
        console.print.assert_called()
        call_args = str(console.print.call_args)
        assert "italic" in call_args

    def test_add_reasoning_plain(self):
        """Should print reasoning without styling in plain mode."""
        display = RichDisplay(pretty=False)

        with patch("builtins.print") as mock_print:
            display.add_reasoning("thinking...")

            # Should print without styling
            mock_print.assert_called()

    def test_reasoning_to_content_transition(self):
        """Should handle transition from reasoning to content."""
        display = RichDisplay(pretty=False)

        display.add_reasoning("thinking...")
        assert display._in_reasoning is True
        assert display._in_content is False

        # start_response resets _in_reasoning and sets _in_content
        display.start_response()
        assert display._in_content is True

    def test_complete_response_shows_timing(self):
        """Should show elapsed time on completion."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display._response_start_time = time.time() - 1.5  # 1.5 seconds ago
        display._response_buffer = "Hello"

        display.complete_response()

        # Should have printed timing
        console.print.assert_called()
        # Check that timing was printed (format: (X.XXs))
        calls_str = str(console.print.call_args_list)
        assert "s)" in calls_str

    def test_complete_response_plain_mode(self):
        """Should show timing in plain mode."""
        display = RichDisplay(pretty=False)

        display._response_start_time = time.time() - 0.5
        display._response_buffer = "Hello"

        with patch("builtins.print") as mock_print:
            display.complete_response()

            # Should have printed timing
            mock_print.assert_called()

    def test_show_tool_call_pretty(self):
        """Should show tool call in panel in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_call("square_number", {"n": 5})

        # Should have printed a panel
        console.print.assert_called()

    def test_show_tool_call_plain(self):
        """Should show tool call as text in plain mode."""
        display = RichDisplay(pretty=False)

        with patch("builtins.print") as mock_print:
            display.show_tool_call("square_number", {"n": 5})

            # Should have printed text
            mock_print.assert_called()

    def test_show_tool_result(self):
        """Should show tool result."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

        display.show_tool_result("25")

        console.print.assert_called()

    def test_show_tool_result_truncates_long(self):
        """Should truncate long tool results in pretty mode."""
        console = MagicMock(spec=Console)
        display = RichDisplay(console=console, pretty=True)

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
        """Should show error to stderr in plain mode."""
        display = RichDisplay(pretty=False)

        with patch("builtins.print") as mock_print:
            display.show_error("Something went wrong")

            mock_print.assert_called()

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
        display = RichDisplay(pretty=False)

        with patch("builtins.print") as mock_print:
            display._print("Hello")

            mock_print.assert_called_with("Hello", end="\n", flush=False)

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
