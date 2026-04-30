"""TUI Viewer Tool - Launch and interact with the Agent13 TUI headlessly.

This tool allows the AI to see what the user sees when running the TUI,
enabling visual verification of changes instead of guessing.

Note: This tool is Unix-only (requires PTY). On Windows, these tools
return an error message explaining the limitation.
"""

import os
import re
import sys
import time

from tools import tool

# Unix-only imports - will fail gracefully on Windows
if sys.platform != "win32":
    import pty
    import select
    import signal
    import fcntl
    import termios
    import struct
    import pyte

# Global state for the running TUI session
_tui_state: dict = {}


@tool(groups=["devel"])
def tui_launch(
    provider: str = "test",
    model: str = "Qwen-3.5-27B",
    rows: int = 24,
    cols: int = 80,
    wait: float = 2.0,
    format: str = "text",
    continue_session: bool = False,
) -> str:
    """
    Launch the Agent13 TUI in a headless PTY.

    Args:
        provider: Provider name from config
        model: Model name to use
        rows: Terminal height
        cols: Terminal width
        wait: Seconds to wait for TUI to initialize
        format: Output format - "text" or "raw"
        continue_session: If True, pass --continue to restore last session
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    global _tui_state
    if _tui_state.get("pid"):
        try:
            os.kill(_tui_state["pid"], signal.SIGTERM)
            os.waitpid(_tui_state["pid"], 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        _tui_state = {}

    # Create pyte screen
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream()
    stream.attach(screen)

    # Create pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    # Set terminal size
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)

    # Fork and exec
    args = ["./agent13.py", provider, "--model", model]
    if continue_session:
        args.append("--continue")
    pid = os.fork()

    if pid == 0:  # Child process
        os.close(master_fd)
        os.setsid()
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execvp("./agent13.py", args)
    else:  # Parent process
        os.close(slave_fd)

        # Wait for TUI to initialize
        time.sleep(wait)

        # Read initial output
        output = _read_all_available(master_fd)
        stream.feed(output.decode("utf-8", errors="replace"))

        # Store state
        _tui_state = {
            "pid": pid,
            "master_fd": master_fd,
            "screen": screen,
            "stream": stream,
            "rows": rows,
            "cols": cols,
        }

        if format == "raw":
            return _format_screen(screen)
        return _format_screen_text(screen)


@tool(groups=["devel"])
def tui_screenshot(format: str = "text") -> str:
    """
    Screenshot the current TUI state.

    Args:
        format: "text" or "raw"
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    if not _tui_state:
        return "Error: TUI not running. Use tui_launch first."

    if format not in ("raw", "text"):
        return f"Error: Unknown format '{format}'. Use 'raw' or 'text'."

    # Read any new output
    output = _read_all_available(_tui_state["master_fd"])
    if output:
        _tui_state["stream"].feed(output.decode("utf-8", errors="replace"))

    if format == "text":
        return _format_screen_text(_tui_state["screen"])
    return _format_screen(_tui_state["screen"])


@tool(groups=["devel"])
def tui_type(
    text: str, wait: float = 0.5, format: str = "text", enter: bool = False
) -> str:
    """
    Type text into the TUI. enter=False by default — use tui_press("enter")
    separately to submit, because Textual's async event loop may process Enter
    before text insertion completes when sent in the same tool call.

    Args:
        text: Text to type
        wait: Seconds to wait after typing
        format: "text" or "raw"
        enter: Press Enter after typing (default: False; use tui_press instead)
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    if not _tui_state:
        return "Error: TUI not running. Use tui_launch first."

    # Write text to PTY. When enter=True, we must ensure Textual has fully
    # processed all character insertions before sending Enter, otherwise
    # action_submit fires with empty/partial text (Textual async event loop).
    os.write(_tui_state["master_fd"], text.encode("utf-8"))
    if enter:
        # NOTE: enter=True is unreliable due to Textual async event loop.
        # Prefer enter=False + tui_press("enter") for reliable submission.
        time.sleep(0.3)
        try:
            os.write(_tui_state["master_fd"], b"\r")
        except OSError as e:
            return f"Error writing Enter: {e}"
    time.sleep(wait)

    return tui_screenshot(format=format)


@tool(groups=["devel"])
def tui_press(key: str, wait: float = 0.5, format: str = "text") -> str:
    """
    Press a special key in the TUI.

    Args:
        key: Key name (enter, escape, tab, backspace, up, down, left, right, etc.)
        wait: Seconds to wait after pressing
        format: "text" or "raw"
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    if not _tui_state:
        return "Error: TUI not running. Use tui_launch first."

    key_map = {
        "enter": "\r",
        "escape": "\x1b",
        "tab": "\t",
        "backspace": "\x7f",
        "up": "\x1b[A",
        "down": "\x1b[B",
        "right": "\x1b[C",
        "left": "\x1b[D",
        "ctrl_c": "\x03",
        "ctrl_d": "\x04",
        "ctrl_l": "\x0c",
        "pageup": "\x1b[5~",
        "pagedown": "\x1b[6~",
        "home": "\x1b[H",
        "end": "\x1b[F",
    }

    if key.lower() not in key_map:
        return f"Error: Unknown key '{key}'. Available: {', '.join(key_map.keys())}"

    os.write(_tui_state["master_fd"], key_map[key.lower()].encode("utf-8"))
    time.sleep(wait)

    return tui_screenshot(format=format)


@tool(groups=["devel"])
def tui_wait_until(
    contains: str, timeout: float = 30.0, interval: float = 1.0, format: str = "text"
) -> str:
    """
    Wait until the TUI screen contains text. Polls until found or timeout.

    Args:
        contains: Text to search for (case-sensitive)
        timeout: Maximum seconds to wait
        interval: Seconds between checks
        format: "text" or "raw"
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    if not _tui_state:
        return "Error: TUI not running. Use tui_launch first."

    _format = _format_screen_text if format == "text" else _format_screen

    deadline = time.monotonic() + timeout
    last_screen = ""

    while time.monotonic() < deadline:
        # Read any new output and update screen
        output = _read_all_available(_tui_state["master_fd"])
        if output:
            _tui_state["stream"].feed(output.decode("utf-8", errors="replace"))

        # Get plain text for searching
        last_screen = _format_screen_text(_tui_state["screen"])

        if contains in last_screen:
            return _format(_tui_state["screen"])

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        time.sleep(min(interval, remaining))

    # Timeout - return error with current state for debugging
    return (
        f"Timeout: '{contains}' not found within {timeout}s.\n"
        f"Current screen:\n{_format(_tui_state['screen'])}"
    )


@tool(groups=["devel"])
def tui_quit() -> str:
    """
    Close the TUI session.
    """
    if sys.platform == "win32":
        return "Error: TUI viewer tools require Unix (PTY not available on Windows)"

    global _tui_state

    if not _tui_state:
        return "No TUI session running."

    try:
        # Send Ctrl+D for clean exit (triggers auto-save in finally block)
        os.write(_tui_state["master_fd"], b"\x04")
        time.sleep(2.0)  # Wait for async cleanup and auto-save to complete

        # If still running, send SIGTERM
        try:
            os.kill(_tui_state["pid"], 0)  # Check if alive
            os.write(_tui_state["master_fd"], b"\x04")
            time.sleep(0.5)
        except ProcessLookupError:
            pass  # Already exited cleanly

        # Force kill if still running
        try:
            os.kill(_tui_state["pid"], signal.SIGTERM)
            os.waitpid(_tui_state["pid"], 0)
        except (ProcessLookupError, ChildProcessError):
            pass

        os.close(_tui_state["master_fd"])
    except Exception:
        pass

    _tui_state = {}
    return "TUI session closed."


def _read_all_available(fd: int, timeout: float = 0.1) -> bytes:
    """Read all available data from file descriptor."""
    output = b""
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
            if ready:
                chunk = os.read(fd, 4096)
                if chunk:
                    output += chunk
                else:
                    break
            else:
                break
        except OSError:
            break
    return output


def _format_screen(screen: pyte.Screen) -> str:
    """Format the pyte screen as a readable string."""
    lines = []
    for i, line in enumerate(screen.display):
        # Strip trailing whitespace
        display_line = line.rstrip()
        # Show line number and content
        lines.append(f"{i:02d}│{display_line}")

    # Remove trailing empty lines
    while lines and lines[-1] == f"{len(lines) - 1:02d}│":
        lines.pop()

    return "\n".join(lines)


# Box-drawing and decorative Unicode characters to strip for plain text
_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF\u2800-\u28FF]")


def _format_screen_text(screen: pyte.Screen) -> str:
    """Format the pyte screen as plain text without box-drawing or line numbers."""
    lines = []
    for line in screen.display:
        stripped = line.rstrip()
        # Remove box-drawing characters
        cleaned = _BOX_DRAWING_RE.sub("", stripped)
        # Collapse multiple spaces left by removed chars
        cleaned = re.sub(r"  +", " ", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)
