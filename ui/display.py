# /// script
# dependencies = [
#     "rich",
# ]
# ///

"""Shared display logic for CLI and TUI interfaces.

Provides RichDisplay class that encapsulates all Rich-based output formatting,
including spinners, markdown rendering, panels, and timing display.
"""

import sys
import time
import json
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule


class RichDisplay:
    """Shared display class for CLI and TUI interfaces.

    Handles all Rich-based output formatting including:
    - Waiting spinner
    - Markdown streaming with Live display
    - Reasoning display
    - Tool call/result display
    - Error display
    - Timing display

    Supports modes:
    - pretty=True, debug_streaming=False: Full Rich formatting with markdown rendering
    - pretty=True, debug_streaming=True: Rich formatting with token markers (| for content, · for reasoning)
    - pretty=False: Plain text output
    """

    # Throttle interval for Live display updates (seconds)
    UPDATE_INTERVAL = 0.05

    def __init__(
        self,
        console: Optional[Console] = None,
        pretty: bool = True,
        debug: bool = False,
        debug_streaming: bool = False,
    ):
        """Initialize the display.

        Args:
            console: Rich Console instance (creates new one if None)
            pretty: Enable pretty output (spinners, panels, markdown)
            debug: Enable debug mode (for logging, not display)
            debug_streaming: Show token boundaries during streaming
        """
        self.console = console if console is not None else Console()
        self.pretty = pretty
        self.debug = debug
        self.debug_streaming = debug_streaming

        # State tracking
        self._in_reasoning = False
        self._in_content = False
        self._response_start_time: Optional[float] = None
        self._status = None  # Rich status spinner

        # Markdown streaming state (pretty mode only)
        self._response_buffer = ""
        self._live_display: Optional[Live] = None
        self._last_update_time = 0.0

    def start_waiting(self, status: str = "Waiting"):
        """Start the waiting spinner.

        Called when agent starts processing a message.

        Args:
            status: Status text to display (e.g., "Waiting", "Thinking", "Processing", "Tooling")
        """
        if self.pretty:
            # Stop any existing spinner first to avoid conflicts
            self.stop_waiting()
            self._status = self.console.status(
                f"[bold blue]{status}...", spinner="dots"
            )
            self._status.__enter__()

    def stop_waiting(self):
        """Stop the waiting spinner if running."""
        if self._status:
            try:
                self._status.__exit__(None, None, None)
            except Exception:
                # Ignore any errors during spinner cleanup
                pass
            finally:
                self._status = None

    def start_response(self):
        """Start a new response.

        Resets state and prepares for content tokens.
        Called on first content token after reasoning (or directly if no reasoning).
        """
        # Stop spinner if still running - ensure it's completely stopped
        self.stop_waiting()
        # Note: Small delay should be added by caller in async context
        # to allow spinner cleanup to complete before Live display starts

        # Check if we were in reasoning mode BEFORE resetting
        was_in_reasoning = self._in_reasoning

        # Reset state for new response
        self._in_reasoning = False
        self._in_content = True
        self._response_buffer = ""
        self._live_display = None

        # Start timing
        self._response_start_time = time.time()

        # Add newline after reasoning if we were thinking
        if was_in_reasoning:
            self._print()  # End the thinking line

        # Print assistant header
        self._print("[bold green]Assistant:[/bold green]")

        # Start Live display for markdown rendering (pretty mode, not debug)
        if self.pretty and not self.debug:
            self._live_display = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=4,
                transient=False,  # Keep final output
            )
            self._live_display.__enter__()

    def add_token(self, text: str):
        """Add a content token to the response.

        Args:
            text: Token text to add
        """
        if self.debug_streaming:
            # Debug streaming mode: show token boundaries with | markers
            marker = "│"
            self._print(f"{marker}{text}", end="")
        elif self.pretty and self._live_display:
            # Pretty mode: accumulate and render markdown
            self._response_buffer += text
            current_time = time.time()
            if current_time - self._last_update_time >= self.UPDATE_INTERVAL:
                self._live_display.update(Markdown(self._response_buffer))
                self._last_update_time = current_time
        else:
            # Non-pretty mode: print raw text
            self._print(text, end="")

    def add_reasoning(self, text: str):
        """Add reasoning text.

        Args:
            text: Reasoning text to display
        """
        # Stop spinner and show "Thinking:" prefix before first reasoning token
        if not self._in_reasoning:
            self.stop_waiting()
            if self.pretty:
                self._print("[italic cyan]Thinking:[/italic cyan] ", end="")
            else:
                self._print("Thinking: ", end="")
            self._in_reasoning = True

        if self.pretty:
            self._print(text, style="italic", end="")
        else:
            self._print(text, end="")

        if self.debug_streaming:
            marker = "·" if self.pretty else "."
            self._print(marker, style="dim", end="")

    def complete_response(self):
        """Complete the response and show timing.

        Called when all tokens have been received.
        """
        # Stop any remaining spinner (safety check)
        self.stop_waiting()
        # Finalize Live display if active
        if self._live_display:
            # Final update with complete buffer
            self._live_display.update(Markdown(self._response_buffer))
            try:
                self._live_display.__exit__(None, None, None)
            except Exception:
                # Ignore any errors during Live display cleanup
                pass
            finally:
                self._live_display = None

        # Show elapsed time
        if self._response_start_time:
            elapsed = time.time() - self._response_start_time
            if self.pretty:
                self._print(f"[dim]({elapsed:.2f}s)[/dim]")
            else:
                self._print(f"({elapsed:.2f}s)")

        # Show separator in pretty mode
        if self.pretty:
            self.show_separator()

        # Reset content flag for next response
        self._in_content = False

    def show_tool_call(self, name: str, arguments: dict):
        """Show a tool call.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        args_str = (
            json.dumps(arguments, indent=2)
            if len(json.dumps(arguments)) > 50
            else json.dumps(arguments)
        )
        if self.pretty:
            self._print_panel(f"[bold]{name}[/bold]({args_str})", "Tool Call")
        else:
            self._print_panel(f"{name}({json.dumps(arguments)})", "Tool Call")

    def show_tool_result(self, result: str):
        """Show a tool result.

        Args:
            result: Tool result string
        """
        result_display = (
            result[:500] + "..." if len(result) > 500 and self.pretty else result
        )
        self._print(f"→ {result_display}", style="dim")

    def show_error(self, message: str):
        """Show an error message.

        Args:
            message: Error message to display
        """
        if self.pretty:
            self.console.print(f"[bold red]Error: {message}[/bold red]")
        else:
            print(f"Error: {message}", file=sys.stderr)

    def show_notification(
        self, message: str, duration: float = 5.0, level: str = "info"
    ):
        """Show a notification message.

        Args:
            message: Notification message to display
            duration: Duration in seconds (not used in CLI)
            level: Notification level (info, warning, error)
        """
        if level == "warning":
            style = "bold yellow"
            prefix = "⚠ Warning:"
        elif level == "error":
            style = "bold red"
            prefix = "✗ Error:"
        else:  # info
            style = "bold blue"
            prefix = "ℹ Info:"

        if self.pretty:
            self.console.print(f"[{style}]{prefix} {message}[/{style}]")
        else:
            print(f"{prefix} {message}")

    def show_separator(self):
        """Show a visual separator between responses."""
        if self.pretty:
            self._print()
            self.console.print(Rule(style="dim"))
            self._print()

    def _print(self, text: str = "", style: Optional[str] = None, end: str = "\n"):
        """Print text with optional Rich styling.

        Args:
            text: Text to print
            style: Rich style string (e.g., "bold red", "italic", "dim")
            end: String appended after the text (default: newline)
        """
        if self.pretty:
            if style:
                self.console.print(text, style=style, end=end)
            else:
                self.console.print(text, end=end)
        else:
            print(text, end=end, flush=(end == ""))

    def _print_panel(self, content: str, title: str, border_style: str = "yellow"):
        """Print content in a Rich panel.

        Args:
            content: Content to display in panel
            title: Panel title
            border_style: Border color style
        """
        if self.pretty:
            self.console.print(Panel(content, title=title, border_style=border_style))
        else:
            print(f"{title}: {content}")


def format_mcp_servers(servers: dict, use_rich: bool = True) -> str:
    """Format MCP server information for display.

    Args:
        servers: Dict mapping server name to list of tool URIs
        use_rich: If True, use Rich markup; if False, plain text

    Returns:
        Formatted string listing all servers and their tools
    """
    if not servers:
        return "No MCP servers connected"

    lines = []
    if use_rich:
        lines.append("[bold]MCP Servers:[/]")
    else:
        lines.append("MCP Servers:")

    for name, tools in servers.items():
        if use_rich:
            lines.append(f"  [cyan]{name}[/]: {len(tools)} tools")
        else:
            lines.append(f"  {name}: {len(tools)} tools")

        for tool in tools:
            # Extract tool name from URI
            tool_name = tool.split("/")[-1] if "/" in tool else tool
            if use_rich:
                lines.append(f"    - {tool_name}")
            else:
                lines.append(f"    - {tool_name}")

    return "\n".join(lines)
