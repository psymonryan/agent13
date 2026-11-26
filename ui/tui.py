# /// script
# dependencies = [
#     "textual",
#     "openai",
#     "python-dotenv",
#     "pyyaml"
# ]
# ///

"""TUI interface using Textual and the event-driven Agent."""

import sys
import os
import asyncio
import json
from contextlib import contextmanager

from dotenv import load_dotenv
from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widgets import Static, Markdown
from textual.widgets._markdown import MarkdownStream
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual.events import Click, MouseUp
from textual.widget import MountError, Widget
from ui.chat_input import ChatTextArea
from ui.summary import budget_for, first_line, tool_what, truncate_line
import time
import argparse
from pathlib import Path

# Re-export from shared module for backwards compatibility
from agent13.models import (
    fetch_models,
    format_model_list,
    print_model_list,
    select_model,
)
from agent13.config import current_provider_name
from agent13.timing import TokenTimingTracker
from agent13.file_injection import expand_file_mentions
from agent13.bell import BellManager
from agent13.completion import filename_completion_scan, plan_completion


from agent13 import (
    Agent,
    AgentEvent,
    AgentEventData,
    AgentStatus,
    PauseState,
    SpinnerSpeed,
    StopReason,
    History,
    PromptManager,
    SnippetManager,
    get_filtered_tools,
    execute_tool,
    resolve_provider_arg,
    create_client,
    init_debug,
    log_session_end,
    get_config,
    # TPS debug logging
    is_debug_enabled,
    log_tps_event,
    log_tps_token_usage,
    log_tps_first_token,
    log_tps_stream_start,
    log_tps_stream_end,
    log_tps_timing_reset,
    log_tps_tool_call,
    log_tps_calculation,
    # Skills
    SkillManager,
    skill_manager_ctx,
)
from agent13 import goal as goal_rules
from agent13.status import ctx_alert_style
from agent13.durable_records import startup_focus
from agent13.persistence import (
    load_context,
    find_latest_auto_save,
)
from agent13.commands import (
    execute_queue,
    get_command,
    promote_latest,
    run_command,
    command_names,
    complete_param,
    format_help_sections,
)
from agent13.message_history import content_to_text, is_turn_start
from agent13.stall_watchdog import StallWatchdog
from agent13.prompts import get_skills_section, resolve_report_and_compact_prompt
from agent13.sandbox import (
    parse_sandbox_mode,
    validate_sandbox_profiles,
    get_pinned_sandbox_mode,
)
from tools.security import (
    set_session_sandbox_mode,
)

# Track if exit was triggered by Ctrl+C to show message after TUI closes
_ctrl_c_pressed = False

# Custom escape function that escapes ALL brackets for Textual markup
# Both rich.markup.escape and textual.markup.escape only escape brackets
# that match specific patterns (e.g., [a-z]), but Textual's parser fails
# on brackets like [Showing...] with uppercase letters


def escape_markup(text: str) -> str:
    """Escape all brackets in text for safe use in Textual markup."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def escape_markup_open(text: str) -> str:
    """Escape only opening brackets.

    Textual unescapes '\\[' but not '\\]', so escaping both leaves a visible
    backslash in front of every closing bracket. Use this where the text
    carries bracketed usage forms like '[-y]'.
    """
    return text.replace("\\", "\\\\").replace("[", "\\[")


# Max chars for the model name in the status bar before it's truncated.
_MODEL_DISPLAY_BUDGET = 16

# Max TOP-LEVEL chat widgets kept live. One per message (a message's own
# children are its ~60 markdown widgets), so a trim never cuts mid-message.
# The chat used to keep every message for the life of the session: 100
# messages = 6,109 widgets, and any full-tree operation cost seconds
# (update_node_styles measured 1.7 s at 6,109 and 5.0 s at 18,309).
# The message HISTORY is untouched - /history, /rebuild and the model's
# context all still see everything; only the live widgets are bounded.
MAX_CHAT_MESSAGES = 40


def compress_model(model: str, budget: int = _MODEL_DISPLAY_BUDGET) -> str:
    """Compress a model name for the status bar.

    Keeps dash-separated pairs up to ``budget`` chars (breaking at a dash),
    marks truncation with ``__``, and shortens a trailing ``:value`` thinking
    suffix to its first three letters (e.g. ``:medium`` -> ``:med``).

    Examples:
        Qwen3.8-27B-MTPLX-Optimized-Quality:medium -> Qwen3.8-27B__:med
        gpt-oss-120b-oQ8                          -> gpt-oss-120b-oQ8 (fits)
        GLM-5.1                                   -> GLM-5.1
    """
    base, sep, value = model.rpartition(":")
    if not sep:
        base, value = model, ""
    parts = base.split("-")
    kept = parts[0]
    chopped = False
    for part in parts[1:]:
        candidate = f"{kept}-{part}"
        if len(candidate) <= budget:
            kept = candidate
        else:
            chopped = True
            break
    # No dash to break at and the first part alone overran the budget.
    if len(kept) > budget:
        kept = kept[:budget]
        chopped = True
    if chopped:
        kept += "__"
    if value:
        kept += ":" + value[:3]
    return kept


# Load environment variables from ~/.env
load_dotenv(os.path.expanduser("~/.env"))


class TokenMessage(Message):
    """Message sent for each streaming token."""

    def __init__(
        self,
        text: str,
        is_final: bool = False,
        markup: bool = False,
        is_reasoning: bool = False,
        generation: int = 0,
        source: str = "assistant",
    ) -> None:
        super().__init__()
        self.text = text
        self.is_final = is_final
        self.markup = markup
        self.is_reasoning = is_reasoning
        self.generation = generation  # Track which streaming session this belongs to
        self.source = source  # "assistant" or "reflection"


class ToolCallMessage(Message):
    """Message sent when a tool is called."""

    def __init__(self, name: str, arguments: dict, seq: int | None = None) -> None:
        super().__init__()
        self.name = name
        self.arguments = arguments
        # 1-based number of this call within the user turn - pairs the result
        # with its call. None when the emitter didn't provide one.
        self.seq = seq


class ToolCallStartedMessage(Message):
    """Message sent when a tool call's name is known but args are still streaming.

    Lets the UI show the tool name immediately (dimmed, args pending) instead of
    waiting for the full argument stream to complete.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class ToolCallPendingMessage(Message):
    """Message sent when the provider is buffering a tool call (name unknown).

    Providers like mlx-lm buffer whole tool calls: while generating one they
    stream nothing visible - only payload-less keep-alive chunks. This message
    lets the UI show a dimmed "⚙ …" placeholder so the user sees the agent is
    preparing a tool call. It is replaced in place by TOOL_CALL_STARTED (name
    known) or TOOL_CALL (complete), and silently removed if content tokens
    start arriving instead (the silence was just slow thinking, not a tool).
    """

    def __init__(self) -> None:
        super().__init__()


class ToolResultMessage(Message):
    """Message sent when a tool result is received."""

    def __init__(self, name: str, result: str, seq: int | None = None) -> None:
        super().__init__()
        self.name = name
        self.result = result
        # Per-turn 1-based index of the call this result belongs to.
        self.seq = seq


class StatusMessage(Message):
    """Message sent for status changes."""

    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class ErrorMessage(Message):
    """Message sent for errors."""

    def __init__(self, message: str, error_type: str = "unknown") -> None:
        super().__init__()
        self.message = message
        self.error_type = error_type


class NotificationMessage(Message):
    """Message sent for user notifications."""

    def __init__(
        self, message: str, duration: float = 5.0, level: str = "info"
    ) -> None:
        super().__init__()
        self.message = message
        self.duration = duration
        self.level = level


class SystemQueueMessage(Message):
    """System message routed through _token_queue for ordered rendering.

    Used when a system message must appear AFTER pending tool results
    (e.g. "Agent paused") to avoid visual reordering.
    """

    def __init__(self, text: str, escape_text: bool = True) -> None:
        super().__init__()
        self.text = text
        self.escape_text = escape_text


class StreamingMessage(Static):
    """A message widget that supports incremental streaming via MarkdownStream."""

    def __init__(self, content: str = "", title: str | None = None):
        super().__init__()
        self._title = title
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:
        # Pre-render title if provided to avoid race conditions with streaming
        prefix = f"**{self._title}:**\n" if self._title else ""
        self._markdown = Markdown(prefix + self._content)
        yield self._markdown

    async def append(self, text: str) -> None:
        """Append text to the message, streaming it to the Markdown widget."""
        if not text:
            return
        self._content += text
        if self._markdown:
            if self._stream is None:
                self._stream = Markdown.get_stream(self._markdown)
            await self._stream.write(text)

    async def finalize(self) -> None:
        """Stop the stream when done."""
        if self._stream:
            await self._stream.stop()
            self._stream = None


class CollapsibleMessage(Static):
    """A chat widget with a clickable header that folds its body.

    The header (title + a ▶/▼ indicator) is always visible; the body is a
    child widget hidden while collapsed. Clicking the header toggles.
    Subclasses set the header text via ``_title_markup()`` and the body
    widget type via ``_make_body()``.

    Content is buffered in ``_content``: text appended while folded is not
    written to the body, so expanding re-renders the whole body. Content set
    before mount is applied in ``compose()``.
    """

    def __init__(self, title: str = "", collapsed: bool = False):
        super().__init__()
        self._title = title
        self._content = ""
        self.collapsed = collapsed
        self._body: Static | None = None
        self._stream: MarkdownStream | None = None
        self._header: Static | None = None

    # -- header -----------------------------------------------------------

    def _header_class(self) -> str:
        """CSS classes for the header row."""
        return "reasoning-header"

    def _title_markup(self) -> str:
        """Rich markup for the header title (the indicator is appended)."""
        return f"[dim]{escape_markup(self._title)}[/]"

    def _header_markup(self) -> str:
        indicator = "▶" if self.collapsed else "▼"
        return f"{self._title_markup()} {indicator}"

    def _refresh_header(self) -> None:
        if self._header:
            self._header.update(self._header_markup())

    def _header_width(self) -> int:
        """Columns available to the header text (0 before first layout)."""
        if self._header is None:
            return 0
        size = getattr(self._header, "size", None)
        width = getattr(size, "width", 0) if size is not None else 0
        try:
            return int(width)
        except (TypeError, ValueError):
            return 0

    def on_resize(self, event) -> None:
        """The header is width-relative; re-emit it so it stays one line."""
        self._refresh_header()

    def set_title(self, title: str) -> None:
        """Update the header title (re-renders the indicator)."""
        self._title = title
        self._refresh_header()

    # -- body -------------------------------------------------------------

    def _make_body(self) -> Static:
        """The foldable body widget. Markdown (streamable) by default."""
        return Markdown("")

    def compose(self) -> ComposeResult:
        self._header = Static(self._header_markup(), classes=self._header_class())
        yield self._header
        self._body = self._make_body()
        if self._content:
            self._body.update(self._content)
        self._body.display = not self.collapsed
        yield self._body

    def set_body(self, text: str) -> None:
        """Set the whole body in one shot (safe before or after mounting)."""
        self._content = text
        if self._body is not None and not self.collapsed:
            self._body.update(text)

    async def append(self, text: str) -> None:
        """Append text to the body, streaming it while expanded."""
        if not text:
            return
        self._content += text
        if self._body is not None and not self.collapsed:
            if self._stream is None:
                self._stream = Markdown.get_stream(self._body)
            await self._stream.write(text)

    async def finalize(self) -> None:
        """Stop the stream when done."""
        if self._stream:
            await self._stream.stop()
            self._stream = None

    # -- folding ----------------------------------------------------------

    async def on_click(self, event: Click) -> None:
        """Toggle collapsed state only when clicking the header."""
        if event.widget is self._header:
            await self.set_collapsed(not self.collapsed)

    async def set_collapsed(self, collapsed: bool) -> None:
        """Set collapsed state and update display."""
        self.collapsed = collapsed
        self._refresh_header()
        if self._body is None:
            return
        self._body.display = not collapsed
        if not collapsed and self._content:
            # Re-render: content buffered while folded was never written.
            self._body.update(self._content)


class ReasoningMessage(CollapsibleMessage):
    """A message widget for reasoning/thinking content.

    Supports incremental streaming and renders as dim/italic markdown.
    Can be collapsed/expanded via click.
    """

    #: In-progress title -> the title a finished block carries.
    _COMPLETED_TITLES = {"Thinking": "Thoughts", "Reflecting": "Reflection"}

    def __init__(self, title: str = "Thinking", collapsed: bool = False):
        super().__init__(title=title, collapsed=collapsed)
        # The "what it's thinking about" hint is only meaningful once the
        # block is finished, so it is revealed at finalize() - not streamed.
        self._completed = False

    def _title_markup(self) -> str:
        """Header: the title, plus the first line of the thought once done.

        While streaming it is just the in-progress title ("Thinking"); at
        completion it gains a width-relative hint of what it was thinking.
        """
        title = escape_markup(self._title)
        hint = ""
        if self._completed:
            hint = truncate_line(
                first_line(self._content),
                budget_for(f"{self._title}: ", self._header_width()),
            )
        if hint:
            return f"[dim]{title}: {escape_markup(hint)}[/]"
        return f"[dim]{title}[/]"

    async def finalize(self) -> None:
        """Stop the stream and put the header in its completed tense.

        A finished block reads "Thoughts" / "Reflection", not "Thinking" /
        "Reflecting". Renaming here - rather than in the streaming caller -
        is what makes the live path and the rebuild path (which mounts and
        finalizes in one step, from history that only ever holds completed
        blocks) label the same block identically. Also reveals the first-line
        hint, which the rebuild path gets for free the same way.
        """
        completed = self._COMPLETED_TITLES.get(self._title)
        if completed:
            self.set_title(completed)
        self._completed = True
        self._refresh_header()
        await super().finalize()


def format_tool_args(arguments: dict, limit: int = 200) -> str:
    """Dim Rich markup for a tool call's arguments, truncated to `limit`."""
    if not arguments:
        return ""
    try:
        args_str = json.dumps(arguments, indent=2)
        if len(args_str) > limit:
            args_str = args_str[:limit] + "..."
    except Exception:
        args_str = str(arguments)[:limit]
    return f"[dim]{escape_markup(args_str)}[/]"


def format_tool_result(result: str, limit: int = 300) -> str:
    """Dim Rich markup for a tool result, truncated to `limit` chars."""
    text = result[:limit] + "..." if len(result) > limit else result
    return f"[dim]{escape_markup(text)}[/]"


class ToolWidget(CollapsibleMessage):
    """A tool call or tool result, folded to a one-line header by default.

    The header (``⚙ name`` / ``✓ Result (name)``) is always visible so the
    transcript keeps its "what is it doing right now" signal; the args and
    result body fold away until the header is clicked.
    """

    def __init__(
        self,
        name: str,
        collapsed: bool = False,
        kind: str = "call",
        pending: bool = False,
        pending_unknown: bool = False,
    ):
        super().__init__(title=name, collapsed=collapsed)
        self.kind = kind  # "call" | "result"
        self.pending = pending
        # Name not known yet (provider buffering the tool call)
        self.pending_unknown = pending_unknown
        # The call's arguments - the source of the folded one-line "what".
        self._arguments: dict = {}
        # 1-based number of this call within the user turn (set by set_call /
        # set_result) - lets a result be paired with its call.
        self._seq: int | None = None

    def _header_class(self) -> str:
        return "tool-header"

    def _make_body(self) -> Static:
        # Tool bodies are escaped Rich markup (JSON / raw output), not markdown.
        return Static("", markup=True)

    def _seq_suffix(self) -> str:
        """The header's number part: ' 1' when this widget carries a seq."""
        return f" {self._seq}" if self._seq else ""

    def _title_markup(self) -> str:
        name = escape_markup(self._title)
        seq = self._seq_suffix()
        if self.pending:
            if self.pending_unknown or not name:
                return "[dim]⚙ …[/]"
            return f"[dim]⚙ {name} …[/]"
        if self.kind == "result":
            return f"[dim]✓ Result{seq} ({name}):[/]"
        # A completed call carries a width-relative "what" - the command
        # running, the file being edited - so one line says what it's doing.
        what = truncate_line(
            tool_what(self._title, self._arguments),
            budget_for(f"⚙ {self._title}{seq}: ", self._header_width()),
        )
        if what:
            return f"[bold yellow]⚙ {name}{seq}[/]: {escape_markup(what)}"
        return f"[bold yellow]⚙ {name}{seq}[/]"

    def set_pending_name(self, name: str) -> None:
        """Fill in the name of a name-unknown placeholder (still pending)."""
        self._title = name
        self.pending_unknown = False
        self._refresh_header()

    def set_call(self, name: str, arguments: dict, seq: int | None = None) -> None:
        """Complete a pending widget: set the name and the args body."""
        self._title = name
        self.pending = False
        self.pending_unknown = False
        self._arguments = arguments
        self._seq = seq
        self.set_body(format_tool_args(arguments))
        self._refresh_header()

    def set_result(self, result: str, seq: int | None = None) -> None:
        """Set the result body."""
        self._seq = seq
        self.set_body(format_tool_result(result))


class InterruptMessage(Static):
    """Widget displayed when user interrupts the agent."""

    def __init__(self):
        super().__init__(
            "[yellow]⚠ Interrupted by user[/]", classes="system-message", markup=True
        )


# Commands the TUI implements itself - their work is about this UI's own screen
# or input, so there is nothing to share with the REPL. Shared commands come
# from the registry in agent13.commands.
_TUI_LOCAL_COMMANDS = frozenset(
    {
        "help",
        "exit",
        "clear",
        "rebuild",
        "reasoning",
        "toolview",
        "spinner",
        "clipboard",
        "pause",
        "resume",
    }
)

# Catalogue entries this front-end deliberately does not offer.
#
# /upgrade: the startup check already offers the update at the moment the user
# learns about it, `agent13 --upgrade` covers every other case, and this
# front-end's version applied the update with no confirmation prompt.
#
# /pretty: Textual always renders markdown, so the flag had no effect here -
# it only made /status report a setting that did nothing.
_TUI_NOT_IMPLEMENTED = frozenset({"multi", "stop", "upgrade", "pretty"})


class AgentTUI(App):
    """TUI with streaming messages, smart scroll, and real Agent integration."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #chat {
        height: 1fr;
        border: solid green;
        overflow-y: auto;
        padding: 0 1;
        background: #0d0d0d;
    }

    #messages {
        layout: stream;
        padding: 1;
    }

    .user-message {
        color: cyan;
        margin-top: 1;
    }

    .reasoning-message {
        color: dimgrey;
        margin-top: 1;
        margin-bottom: 0;
        text-style: italic;
    }

    .reasoning-message Markdown {
        color: dimgrey;
        text-style: italic;
    }

    .reasoning-message Markdown Strong {
        text-style: bold italic;
    }

    .reasoning-header {
        color: dimgrey;
        text-style: italic;
        margin-bottom: 0;
    }

    .tool-header {
        margin-bottom: 0;
    }

    .assistant-message {
        color: magenta;
        margin-top: 0;
        margin-bottom: 1;
    }

    .system-message {
        color: dimgrey;
        margin-top: 1;
    }

    .tool-call {
        color: yellow;
        margin-top: 1;
        margin-bottom: 0;
        padding: 0 2;
        border-left: outer dimgrey;
    }

    .tool-result {
        color: orange;
        margin-top: 0;
        margin-bottom: 1;
        padding: 0 2;
        border-left: outer dimgrey;
    }

    .error-message {
        color: red;
        margin-bottom: 1;
        padding: 0 2;
    }

    .error-panel {
        background: $panel;
        border: solid red;
        color: red;
        margin: 1;
        padding: 1;
    }

    .notification-panel {
        background: $panel;
        border: solid dodgerblue;
        color: white;
        margin: 1;
        padding: 1;
    }

    .error-network {
        color: yellow;
    }

    .error-auth {
        color: red;
    }

    .error-rate-limit {
        color: magenta;
    }

    #info-pane {
        height: auto;
        background: $panel;
        border: solid yellow;
        padding: 0;
        display: none;
    }

    #info-content {
        height: auto;
    }

    .queue-item {
        color: yellow;
    }

    .command-output {
        color: white;
    }

    #input-area {
        height: auto;
        padding: 0 1;
        background: $panel;
    }

    ChatTextArea {
        width: 100%;
        height: auto;
        min-height: 1;
        max-height: 10;
        background: $surface;
        color: $text;
        /* The border is ALWAYS present; only its colour changes on focus.
           Textual drops widget focus when the TERMINAL loses focus - that is
           what hides the cursor, so you can see a click has not landed yet.
           If the border only existed on :focus, that same event would also
           remove 2 rows and reflow the whole screen. */
        border: solid transparent;
    }

    ChatTextArea:focus {
        border: solid $accent;
    }

    ChatTextArea .text-area--cursor {
        color: $text;
    }

    #status-line {
        height: 1;
        width: 100%;
        background: $primary;
        color: white;
        padding: 0 1;
        layout: horizontal;
        align: left middle;
    }

    #status-left {
        height: auto;
        width: auto;
        color: white;
    }

    #status-spacer {
        width: 1fr;
        height: auto;
    }

    #status-right {
        height: auto;
        width: auto;
        color: white;
        text-align: right;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "clear_quit", "Quit", show=False),
        Binding("ctrl+d", "force_quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "force_quit", "Quit", show=False, priority=True),
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        # NOTE: Shift+key combos removed — they don't survive
        # terminal → ssh → tmux → Textual chain.
        # Use mouse scroll instead of shift+up/down.
        # Use Ctrl+Y instead of ctrl+shift+c.
        # Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False),
        # Binding("shift+down", "scroll_chat_down", "Scroll Down", show=False),
        # Ctrl+Y: Copy full markdown of message containing selection
        Binding(
            "ctrl+y", "copy_as_markdown", "Copy markdown", show=False, priority=True
        ),
        # Ctrl+Shift+C removed — doesn't survive terminal chain.
        # Use Ctrl+Y (copy_as_markdown) instead.
        # Binding("ctrl+shift+c", "copy_selection", "Copy selection", show=False),
        # Ctrl+O: Toggle collapse of most recent reasoning widget
        Binding("ctrl+o", "toggle_collapsed", "Toggle Reasoning", show=False),
    ]

    # Auto-copy rendered text to clipboard on mouse up
    # Can be disabled by setting this to False
    autocopy_to_clipboard: bool = True

    # Available slash commands for tab completion
    # Note: Named SLASH_COMMANDS to avoid conflict with Textual's command palette
    # which expects app.COMMANDS to be a list of command provider classes
    #
    # The catalogue lives in agent13.commands. This list is derived from it:
    # shared commands (those with a handler) plus the ones the TUI implements
    # itself. Skills and snippets append their own names in __init__.
    #
    # Commands the TUI implements locally (see _TUI_LOCAL_COMMANDS above).
    _LOCAL_COMMANDS = _TUI_LOCAL_COMMANDS

    # Catalogue entries this front-end deliberately does not offer.
    _NOT_IN_TUI = _TUI_NOT_IMPLEMENTED

    _BUILTIN_SLASH_COMMANDS = [
        f"/{name}"
        for name in command_names()
        if get_command(name).handler is not None or name in _TUI_LOCAL_COMMANDS
    ]

    # Class-level attribute for type checking (instance copy created in __init__)
    SLASH_COMMANDS = _BUILTIN_SLASH_COMMANDS

    # Spinner animation frames by speed mode
    SPINNER_FRAMES_FAST = [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    ]  # braille, 10 frames
    SPINNER_FRAMES_SLOW = ["-", "\\", "|", "/"]  # classic, 4 frames
    SPINNER_INTERVAL = 0.1  # 100ms per frame (fast mode)

    status = reactive("Ready")
    queue_count = reactive(0)
    processing = reactive(False)
    _streaming = reactive(False)  # True when receiving tokens from LLM
    _spinner_index = reactive(0)
    # Polite mode: None when not waiting for lock, else elapsed seconds (float).
    # Drives the "polite: Ns" status overlay; cleared on POLITE_ACQUIRED.
    _polite_wait_elapsed: reactive = reactive(None)

    # Token usage tracking
    prompt_tokens = reactive(0)
    completion_tokens = reactive(0)
    total_tokens = reactive(0)
    # Live context estimate at the safe point (prompt_tokens is stale there -
    # it predates the tool results just added). Shown with a "~" until the next
    # LLM call reports the grounded count via TOKEN_USAGE.
    estimated_context = reactive(0)
    context_is_estimated = reactive(False)

    # TPS (tokens per second) tracking
    _last_tps = reactive(0.0)  # TPS from the most recent API call

    # Re-entrancy guard for update_status (a plain attribute, NOT a reactive:
    # setting a reactive here would fire the very watchers it guards against).
    _updating_status: bool = False

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        model_names: list[str],
        provider: str = "",
        provider_args: tuple[str, str, float, float] | None = None,
        debug: bool = False,
        tool_response_format: str = "raw",
        prompt_manager: PromptManager = None,
        connect_mcp: bool = False,
        skill_manager: SkillManager = None,
        snippet_manager: SnippetManager = None,
        system_prompt: str = None,
        include_skills: bool = False,
        journal_mode: bool = False,
        remove_reasoning: bool = False,
        continue_session: bool = False,
        devel_mode: bool = False,
        spinner_speed: str = "fast",
        clipboard_method: str = "osc52",
        read_files: list[str] | None = None,
        focus_name: str | None = None,
        polite_interval: float | None = None,
        bell_threshold: float | str | None = None,
        bell_enabled: bool = True,
        bell_command: str = "",
        priming_enabled: bool = False,
        cursor_blink: bool = False,
        reasoning_collapsed: bool = True,
        toolview_collapsed: bool = True,
        ctx_warn_fg: str = "yellow",
        ctx_warn_bg: str = "",
        ctx_crit_fg: str = "red",
        ctx_crit_bg: str = "",
        status_fg: str = "",
        status_bg: str = "",
        auto_context_threshold: int = 220000,
        auto_context_action: str = "report_and_compact",
        auto_context_chain: int = 3,
        auto_context_nudge_cap: int = 2,
        auto_save_idle_seconds: int = 300,
        datetime_nudge_seconds: int = 3600,
        report_and_compact_prompt: str = "",
        skills_notice: str | None = None,
    ):
        """Initialize the TUI.

        Args:
            client: AsyncOpenAI client for API calls
            model: Model name to use
            model_names: List of available model names
            provider: Provider name for status bar display
            provider_args: (base_url, api_key, read_timeout, connect_timeout)
                for creating a fresh client in the TUI's own event loop
                (see on_mount). None for offline providers (fake).
            debug: Enable debug mode
            tool_response_format: Tool response format ('raw' or 'json')
            prompt_manager: Optional pre-configured PromptManager
            connect_mcp: Connect to MCP servers on startup
            skill_manager: Optional skill manager for skill commands
            system_prompt: Optional system prompt (overrides prompt_manager if provided)
            include_skills: Whether skills are opted in (via --skills or config.include_skills).
                Gates visibility of the skill *tool* to the AI. Slash commands are
                unaffected. Default False (minimal context).
            journal_mode: Enable context compaction via reflection
            remove_reasoning: Strip reasoning tokens between turns
            continue_session: Load latest auto-save on startup
            devel_mode: Enable devel mode (show devel-group tools to the AI)
            spinner_speed: Spinner speed - 'fast' (100ms, braille), 'slow' (250ms, classic), or 'off'
            clipboard_method: Clipboard method - 'osc52' (terminal escape) or 'system' (OS commands)
            polite_interval: Optional poll interval N in seconds for polite mode
                (multi-agent lock coordination). None = disabled.
            bell_threshold: Bell threshold in seconds. 0 = always ring on idle.
                None = use config value. "off" = disable.
            bell_enabled: Whether bell is active (from config). Overridden if
                bell_threshold is "off".
            bell_command: External command to run instead of terminal bell.
                Validated on startup; invalid commands warn and fall back.
            cursor_blink: Whether the input cursor blinks. Default False
                (no periodic terminal output, tmux-friendly).
            reasoning_collapsed: Reasoning widgets (Thoughts/Reflection) start
                folded to a one-line header. Default True.
            toolview_collapsed: Tool call/result widgets start folded to a
                one-line header. Default True.
            ctx_warn_fg / ctx_warn_bg: Status-bar Ctx text/background colour
                at 80% of auto_context_threshold. "" = leave that channel
                alone (fg = the bar's text colour, bg = its background).
            ctx_crit_fg / ctx_crit_bg: The same pair at 90%.
            status_fg / status_bg: The status bar's own text/background colour.
                "" = the theme's (white on the primary colour, the shipped
                look). The Ctx segment's "" channels inherit these.
            auto_context_threshold: Token threshold for the auto-context check.
            auto_context_action: What to do at the threshold - "compact",
                "report_and_compact", or "none".
            auto_context_chain: Turn restarts after a compact per turn (0 = none).
            auto_context_nudge_cap: Consecutive degenerate outputs before the
                guard stops the turn (0 = guard off).
            auto_save_idle_seconds: Write the daily auto-save once after this
                many idle seconds (0 = off).
            report_and_compact_prompt: Prompt text for the wrap-up turn. Empty
                string = resolve from prompts.yaml / built-in default.
        """
        super().__init__()
        # The Agent owns the live model/client; the TUI reads them from there
        # so a shared /model or /provider cannot leave a stale copy behind.
        self._provider_args = provider_args
        self._debug_mode = debug
        self._connect_mcp = connect_mcp
        self.read_files = read_files
        self.focus_name = focus_name

        # Managers are constructed by cli.py so every mode shares one. They are
        # attached to the agent below - the agent owns them, and the shared
        # command handlers read them from there.
        prompt_manager = prompt_manager or PromptManager()

        # Whether skills are opted in (--skills flag or config.include_skills).
        # Drives the skill *tool*'s initial visibility AND whether the skills
        # list starts in the system prompt. The tool is additionally ratcheted
        # on by the first skill slash-invocation (and never switches off); the
        # list is toggled at runtime by /skills on|off. Slash commands are
        # always available regardless.
        self._include_skills = include_skills
        # Snippet manager: cli.py builds it with the catalogue's names reserved;
        # a direct construction here covers the standalone main() path.
        if snippet_manager is None:
            reserved = {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}
            try:
                snippet_manager = SnippetManager(reserved_names=reserved)
            except Exception as e:
                import sys

                print(f"Error: Failed to load snippets: {e}", file=sys.stderr)
                sys.exit(1)

        # Initialize slash commands (instance attribute for skill extensibility)
        self.SLASH_COMMANDS = list(self._BUILTIN_SLASH_COMMANDS)

        # Convert tool_response_format string to response_format dict
        response_format = (
            {"type": "json_object"} if tool_response_format == "json" else None
        )

        # Determine system prompt
        effective_system_prompt = (
            system_prompt if system_prompt else prompt_manager.get_prompt()
        )

        # Initialize agent with tools
        config = get_config()
        # Determine skills mode: the skill tool is visible when skills are opted
        # in via --skills (include_skills) and skills exist — mirroring cli.py's
        # gate. It is ALSO ratcheted on by the first skill slash-invocation and
        # never switches off (see the /skill handler below). This only affects
        # what the LLM sees, per the minimal-context principle.
        skills_mode = self._include_skills and bool(
            skill_manager and skill_manager.skills
        )
        self.agent = Agent(
            client=client,
            model=model,
            system_prompt=effective_system_prompt,
            tools=get_filtered_tools(
                devel=devel_mode,
                skills=skills_mode,
                enabled_tools=config.enabled_tools or None,
                disabled_tools=config.disabled_tools or None,
            ),
            execute_tool=execute_tool,
            response_format=response_format,
            journal_mode=journal_mode,
            remove_reasoning=remove_reasoning,
            devel_mode=devel_mode,
            skills_mode=skills_mode,
            priming_enabled=priming_enabled,
            auto_context_threshold=auto_context_threshold,
            auto_context_action=auto_context_action,
            auto_context_chain=auto_context_chain,
            auto_context_nudge_cap=auto_context_nudge_cap,
            auto_save_idle_seconds=auto_save_idle_seconds,
            datetime_nudge_seconds=datetime_nudge_seconds,
            report_and_compact_prompt=report_and_compact_prompt
            or resolve_report_and_compact_prompt(prompt_manager),
        )

        # Store available models on agent (single source of truth)
        self.agent.available_models = model_names
        # Provider name for the status bar - the agent owns it so /provider
        # can change it without leaving a stale copy here.
        self.agent.provider_name = provider

        # Managers the shared command handlers read (/prompt, /skills,
        # /snippet, /compact, /status). The agent owns them so both front-ends
        # read one instance.
        self.agent.prompt_manager = prompt_manager
        self.agent.skill_manager = skill_manager
        self.agent.snippet_manager = snippet_manager
        # Startup notice from the default-skills sync ("" or None = silent).
        self._skills_notice = skills_notice
        # The skills *list* starts in the prompt when skills are opted in
        # (--skills / config.include_skills); /skills on|off toggles it.
        self.agent.skills_list_enabled = include_skills

        # Register skill and snippet slash commands (they read the managers
        # from the agent, so this runs after the agent exists).
        if self.agent.skill_manager:
            self._register_skill_commands()
        self._register_snippet_commands()

        # Set MCP server configs for lazy initialization
        if config.mcp_servers:
            self.agent.set_mcp_servers(config.mcp_servers)

        # Enable polite mode if requested (--polite N)
        if polite_interval is not None:
            self.agent.set_polite(interval=polite_interval)

        # Load auto-save if continuing session
        self._continue_session = continue_session
        self._session_loaded = False  # Track if we loaded a session

        # Spinner speed ("fast", "slow", "off") - the agent owns the value so
        # the shared /status can report it; the spinner reads it here.
        self.agent.spinner_speed = spinner_speed

        # Clipboard method (osc52 or system)
        self.agent.clipboard_method = clipboard_method

        # Widget folding (display only): which widget families start folded.
        self.agent.reasoning_collapsed = reasoning_collapsed
        self.agent.toolview_collapsed = toolview_collapsed

        # Status-bar colours (display only), all TUI-local: no shared handler
        # reads them, so they have no business on the Agent - core carries no
        # UI config. The Ctx quad is read by update_status; the bar's own pair
        # is applied to widgets that only exist once the app is mounted.
        self._ctx_warn_fg = ctx_warn_fg
        self._ctx_warn_bg = ctx_warn_bg
        self._ctx_crit_fg = ctx_crit_fg
        self._ctx_crit_bg = ctx_crit_bg
        self._status_fg = status_fg
        self._status_bg = status_bg

        # Bell: BellManager encapsulates threshold timer, command validation,
        # and ringing. TUI passes self.bell as the fallback (terminal bell).
        self._cursor_blink = cursor_blink
        self._bell = BellManager(
            threshold=bell_threshold,
            enabled=bell_enabled,
            command=bell_command,
            fallback_ring=self.bell,
        )
        # The shared /bell and /bell-command handlers reach the manager through
        # the agent, so the same code serves both front-ends.
        self.agent.bell = self._bell

        # State
        self._history = (
            History()
        )  # Persistent command history (uses default dated path)
        # Separate widgets for reasoning and content
        self._streaming_reasoning_widget: ReasoningMessage | None = None
        self._streaming_content_widget: StreamingMessage | None = None
        self._in_reasoning: bool = False  # Track if currently in reasoning phase
        self._finalize_before_tool: bool = (
            False  # True if we finalized before a tool call
        )
        # Tool-call widgets shown at TOOL_CALL_STARTED (name only, args pending).
        # Consumed in FIFO order by the matching TOOL_CALL (complete) event, which
        # updates the widget in place to add the args.
        self._pending_tool_widgets: list[Static] = []
        self._stream_generation: int = (
            0  # Incremented on each new streaming session to discard stale tokens
        )

        # Sequential token processor: guarantees ordering of all widget operations
        # by processing tokens one at a time from an async queue, eliminating
        # all race conditions caused by asyncio.create_task fire-and-forget.
        self._token_queue: asyncio.Queue[
            TokenMessage
            | ToolCallMessage
            | ToolCallStartedMessage
            | ToolResultMessage
            | SystemQueueMessage
            | None
        ] = asyncio.Queue()
        self._token_processor_task: asyncio.Task | None = None
        self._agent_task: asyncio.Task | None = None
        self._mcp_connect_task: asyncio.Task | None = None
        self._shutting_down = False  # Set in on_unmount to prevent mount errors
        # Nesting-safe suspension counter for the live-widget cap: a rebuild
        # mounts everything (including its own status line) with the cap off.
        self._chat_cap_suspends = 0
        # Bumped on every focus-in so a blur whose tmux query is still in
        # flight cannot hide the cursor after focus has come back.
        self._focus_generation = 0
        self._agent_started = asyncio.Event()
        self._interrupt_requested = False  # Prevent double-cancellation
        self._interrupt_available = (
            False  # /resume after ESC sends "Sorry, please continue"
        )
        # Note: pause state is read from agent.pause_state (single source of truth)
        self._error_state = False  # Track if agent stopped due to error
        self._last_error_type = None  # Type of last error

        # TPS tracking — delegated to TokenTimingTracker
        # _last_tps stays as a Textual reactive for status bar binding
        self.agent.tracker = TokenTimingTracker()
        self._stream_start_time: float | None = (
            None  # When current stream started (debug tracking only)
        )

        # Session start time (when TUI was created, for /status run time)
        self.agent.session_start_time = time.time()
        # Last turn tracking (persists after idle for status bar + /status)
        self.agent.last_turn_duration = None  # seconds of last completed turn
        self.agent.last_turn_end_time = None  # when last turn finished (epoch)
        # Resumed session tracking (for --continue and /load /status)
        self.agent.resumed_saved_at = None  # ISO timestamp from .ctx file
        self.agent.resumed_turn_count = 0  # user-role messages in loaded session
        self.agent.resumed_prompt_tokens = 0  # prompt tokens at load time
        self.agent.resumed_completion_tokens = 0  # completion tokens at load time
        self._pending_load_path: Path | None = None  # stored before deferred /load

        # Tab completion state
        self._completion_matches: list[str] = []  # Current completion matches
        self._completion_index: int = 0  # Current index in matches
        self._completion_prefix: str = ""  # Original text being completed
        self._completion_start: tuple[int, int] = (
            0,
            0,
        )  # Start location for partial replacement
        self._completion_end: tuple[int, int] = (0, 0)  # End location (cursor position)
        self._completion_text: str = (
            ""  # Text at time of last completion (to detect changes)
        )

        # Info pane mode tracking - what content is currently displayed
        self._info_pane_mode: str | None = (
            None  # None, "history", "queue", "help", "status"
        )
        # Track whether the user explicitly dismissed the queue pane so
        # background events (POLITE_WAITING) don't reopen it.
        self._queue_pane_dismissed: bool = False
        # Last info pane content/mode — saved on hide so Enter can toggle it back open
        self._last_info_content: str | None = None
        self._last_info_mode: str | None = None

        # Register event handlers
        self._register_handlers()

    def _register_skill_commands(self) -> None:
        """Register slash commands for all skills."""
        if not self.agent.skill_manager:
            return
        self._register_dynamic_commands(self.agent.skill_manager.skills, "Skill")

    def _register_snippet_commands(self) -> None:
        """Register slash commands for snippets and show collision warnings."""
        self._register_dynamic_commands(self.agent.snippet_manager.snippets, "Snippet")
        # Show collision warnings from initial load
        for warning in self.agent.snippet_manager._collisions:
            print(
                f"WARNING: Snippet '{warning}' conflicts with built-in '/{warning}' command. "
                f"Use /snippet use {warning} to invoke it.",
                file=sys.stderr,
            )

    def _register_dynamic_commands(self, items: dict, label: str) -> list[str]:
        """Register items as slash commands, returning collision names.

        Args:
            items: Dict of name -> value to register as /name commands.
            label: Label for warning messages (e.g. "Skill", "Snippet").

        Returns:
            List of names that collided with built-in commands.
        """
        reserved = {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}
        collisions = []
        for name in items:
            if name in reserved:
                print(
                    f"WARNING: {label} '{name}' conflicts with built-in '/{name}' command.",
                    file=sys.stderr,
                )
                collisions.append(name)
            else:
                self.SLASH_COMMANDS.append(f"/{name}")
        return collisions

    def _add_snippet_slash_command(self, name: str) -> None:
        """Offer /name for a new or renamed snippet.

        A name that collides with a built-in command is left out - the snippet
        is still reachable through /snippet use.
        """
        if name in {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}:
            return
        slash = f"/{name}"
        if slash not in self.SLASH_COMMANDS:
            self.SLASH_COMMANDS.append(slash)

    def _remove_snippet_slash_command(self, name: str) -> None:
        """Drop /name after a snippet is deleted or renamed away."""
        slash = f"/{name}"
        if slash in self.SLASH_COMMANDS:
            self.SLASH_COMMANDS.remove(slash)

    def _is_at_bottom(self) -> bool:
        """Check if view is near bottom (within threshold)."""
        try:
            threshold = 3
            return self._chat.scroll_y >= (self._chat.max_scroll_y - threshold)
        except Exception:
            return True

    def _get_command_completions(self, text: str) -> list[str]:
        """Get slash command completions for text starting with /.

        Args:
            text: The text to complete (should start with /)

        Returns:
            List of matching commands
        """
        if not text.startswith("/"):
            return []

        text_lower = text.lower()
        matches = [
            cmd for cmd in self.SLASH_COMMANDS if cmd.lower().startswith(text_lower)
        ]
        return matches

    def _get_history_completions(self, text: str) -> list[str]:
        """Get history completions for text.

        Args:
            text: The text to complete

        Returns:
            List of matching history items (newest first)
        """
        if not text:
            return []

        text_lower = text.lower()
        matches = []
        seen = set()

        for item in self._history.get_all():
            if item.lower().startswith(text_lower) and item not in seen:
                matches.append(item)
                seen.add(item)
                if len(matches) >= 10:  # Limit to 10 matches
                    break

        return matches

    def _get_filename_completions(self, partial: str) -> list[str]:
        """Get filename completions for @file syntax.

        Args:
            partial: The partial text starting with '@' (e.g., "@src/ut" or "@~/Doc")

        Returns:
            List of matching file/directory paths with '@' prefix
        """
        return filename_completion_scan(partial)[0]

    def _get_param_completions(
        self, cmd: str, partial: str, full_text: str = ""
    ) -> list[str]:
        """Get completions for command parameters.

        Args:
            cmd: The command name without slash (e.g., "model", "prompt")
            partial: The partial parameter text to complete
            full_text: Full input text (for context, e.g., detecting subcommands)

        Returns:
            List of matching completions
        """
        return complete_param(cmd, self.agent, partial, full_text)

    def _get_completions_for_context(
        self, completion_type: str, partial: str, full_text: str = ""
    ) -> list[str]:
        """Get completions based on completion context type.

        Args:
            completion_type: Type of completion ("@filename", "/command", "/cmd_param", "history")
            partial: The partial text to complete (e.g., "@src/ut" or "/mod" or "prev")
            full_text: Full input text (needed for some context types)

        Returns:
            List of matching completions
        """
        if completion_type == "@filename":
            return self._get_filename_completions(partial)
        elif completion_type == "/command":
            return self._get_command_completions(partial)
        elif completion_type == "/cmd_param":
            # Parse the command and get parameter completions
            # full_text is like "/model dev" or "/prompt concise"
            parts = full_text.split(maxsplit=1)
            if len(parts) >= 1:
                cmd = parts[0][1:]  # Remove leading /
                return self._get_param_completions(cmd, partial, full_text)
            return []
        elif completion_type == "history":
            return self._get_history_completions(partial)
        else:
            return []

    def _get_completions(self, text: str) -> list[str]:
        """Get completions for text (legacy method for backward compatibility).

        Args:
            text: The text to complete

        Returns:
            List of matching completions
        """
        if text.startswith("/"):
            return self._get_command_completions(text)
        else:
            return self._get_history_completions(text)

    COMPLETION_PAGE_SIZE = 8  # Max items shown in completion dropdown

    def _show_completions(self, matches: list[str], current: str | None = None) -> None:
        """Show available completions in the info pane.

        If info-pane is showing reference content (history, queue, etc.),
        skip showing completions to preserve the reference view.

        Renders a fixed-size window of items around the current selection,
        scrolling the window as the user cycles through matches.

        Args:
            matches: List of completion options
            current: Currently selected completion (for highlighting)
        """
        if not matches:
            return

        # Don't show completions if info-pane has reference content
        if self._info_pane_mode is not None:
            return

        total = len(matches)
        page_size = self.COMPLETION_PAGE_SIZE

        # Find the index of the current selection
        current_idx = 0
        if current and current in matches:
            current_idx = matches.index(current)

        # Calculate the visible window around the current selection
        # Keep the selection visible, scrolling the window when it hits edges
        if total <= page_size:
            # Everything fits in one page
            window_start = 0
            window_end = total
        else:
            # Calculate window start so current_idx is visible
            # Prefer showing context below the selection
            half = page_size // 2
            window_start = max(0, current_idx - half)
            window_end = window_start + page_size

            # Clamp to valid range
            if window_end > total:
                window_end = total
                window_start = total - page_size

        has_above = window_start > 0
        has_below = window_end < total

        # Build header
        lines = [f"[bold]Completions ({total}):[/]"]

        # "more above" indicator
        if has_above:
            lines.append(f"  [dim]↑{window_start} more above[/]")

        # Render visible window
        for i in range(window_start, window_end):
            match = matches[i]
            if match == current:
                lines.append(f"  [cyan bold]▶ {escape_markup(match)}[/]")
            else:
                lines.append(f"    {escape_markup(match)}")

        # "more below" indicator
        if has_below:
            lines.append(f"  [dim]↓{total - window_end} more below[/]")

        # Only teach the → gesture when it applies - it opens a DIRECTORY
        hint = "Tab/Shift+Tab/Up/Down to cycle, "
        if current and current.endswith("/"):
            hint += "\u2192 opens a dir, "
        hint += "Enter to accept, Escape to cancel"
        lines.append("")
        lines.append(f"[dim]{hint}[/]")
        self._update_info_content("\n".join(lines))

    def _reset_completion_state(self) -> None:
        """Reset tab completion state."""
        self._completion_matches = []
        self._completion_index = 0
        self._completion_prefix = ""
        self._completion_start = (0, 0)  # (row, col) where completion target starts
        self._completion_end = (0, 0)  # (row, col) where cursor is (end of target)
        self._completion_text = ""  # Clear saved text

    def _location_to_offset(self, text: str, row: int, col: int) -> int:
        """Convert (row, col) location to character offset in text.

        Args:
            text: The full text content
            row: Line number (0-indexed)
            col: Column number (0-indexed)

        Returns:
            Character offset from start of text
        """
        lines = text.split("\n")
        offset = 0
        for i in range(row):
            offset += len(lines[i]) + 1  # +1 for newline
        offset += col
        return offset

    def _offset_to_location(self, text: str, offset: int) -> tuple[int, int]:
        """Convert character offset to (row, col) location.

        Args:
            text: The full text content
            offset: Character offset from start of text

        Returns:
            Tuple of (row, col)
        """
        lines = text.split("\n")
        current_offset = 0
        for row, line in enumerate(lines):
            if current_offset + len(line) >= offset:
                return (row, offset - current_offset)
            current_offset += len(line) + 1  # +1 for newline
        # If offset is at or past end of text
        return (len(lines) - 1, len(lines[-1]))

    def _get_completion_context(
        self, text: str, cursor_row: int, cursor_col: int
    ) -> tuple[str, tuple[int, int], tuple[int, int]]:
        """Analyze input to determine what kind of completion is needed.

        Args:
            text: Full text content of input field
            cursor_row: Current cursor row (0-indexed)
            cursor_col: Current cursor column (0-indexed)

        Returns:
            Tuple of (completion_type, start_location, end_location)
            - completion_type: "none", "@filename", "/command", "/cmd_param", "history"
            - start_location: (row, col) where the completion target begins
            - end_location: (row, col) where cursor is (what to replace up to)
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

        # Check for /command completion (must be at start of line)
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

    def _register_handlers(self):
        """Register event handlers for agent events."""

        @self.agent.on_event
        async def on_started(event: AgentEventData):
            if event.event != AgentEvent.STARTED:
                return
            self._agent_started.set()
            self._error_state = False  # Clear error state on successful start
            self._last_error_type = None
            asyncio.create_task(
                self._write_system(f"Agent started with {self.agent.model}")
            )
            asyncio.create_task(
                self._write_system(
                    "Type your message, /help for commands, or Ctrl+D to exit"
                )
            )

        @self.agent.on_event
        async def on_stopped(event: AgentEventData):
            if event.event != AgentEvent.STOPPED:
                return
            self.call_later(self._set_running, False)
            # Show retry hint if stopped due to error
            if self._error_state:
                asyncio.create_task(
                    self._write_system(
                        "[dim]Agent stopped due to error. Use /retry to restart.[/]",
                        escape_text=False,
                    )
                )

        @self.agent.on_event
        async def on_status_change(event: AgentEventData):
            if event.event != AgentEvent.STATUS_CHANGE:
                return
            status = event.data.get("status")
            self.call_later(self._update_status, status)

        @self.agent.on_event
        async def on_item_started(event: AgentEventData):
            if event.event != AgentEvent.ITEM_STARTED:
                return
            text = event.data.get("text", "")
            # Clear error state at start of new turn - if we get here, the agent
            # is healthy and processing a new item, so any previous error is stale
            self._error_state = False
            self._last_error_type = None
            # Increment stream generation for new conversation turn
            # This ensures any stale tokens from previous turns are discarded
            self._stream_generation += 1
            # Drop any orphaned pending tool widgets left by a prior turn that
            # errored mid-tool-call (name shown, args never completed).
            await self._discard_pending_tool_widgets()
            # Show user message in chat when processing starts
            # MUST await (not create_task) to ensure user message appears before
            # any subsequent events like JOURNAL_COMPACT that mount widgets
            await self._write_user(text)

        @self.agent.on_event
        async def on_interrupt_injected(event: AgentEventData):
            if event.event != AgentEvent.INTERRUPT_INJECTED:
                return
            text = event.data.get("text", "")
            # Route through post_message → _token_queue so interrupt message
            # renders AFTER any pending tool results. This matches the fix
            # already applied to on_paused (see comment at line 1647).
            # Previously _write_interrupt() was called directly, which
            # bypassed the queue and caused interrupts to appear before
            # tool results that were still waiting to render.
            self.post_message(
                SystemQueueMessage(
                    f"[bold yellow]⚡ Interrupt:[/] {escape_markup(text)}",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_assistant_token(event: AgentEventData):
            if event.event != AgentEvent.ASSISTANT_TOKEN:
                return
            text = event.text or ""
            self.post_message(TokenMessage(text, generation=self._stream_generation))

        @self.agent.on_event
        async def on_assistant_reasoning(event: AgentEventData):
            if event.event != AgentEvent.ASSISTANT_REASONING:
                return
            text = event.text or ""
            source = event.data.get("source", "assistant")
            # Post reasoning tokens with is_reasoning flag and source
            self.post_message(
                TokenMessage(
                    text,
                    is_reasoning=True,
                    generation=self._stream_generation,
                    source=source,
                )
            )

        @self.agent.on_event
        async def on_assistant_complete(event: AgentEventData):
            if event.event != AgentEvent.ASSISTANT_COMPLETE:
                return
            self.post_message(
                TokenMessage("", is_final=True, generation=self._stream_generation)
            )

        @self.agent.on_event
        async def on_tool_call_pending(event: AgentEventData):
            if event.event != AgentEvent.TOOL_CALL_PENDING:
                return
            self.post_message(ToolCallPendingMessage())

        @self.agent.on_event
        async def on_tool_call_started(event: AgentEventData):
            if event.event != AgentEvent.TOOL_CALL_STARTED:
                return
            name = event.data.get("name", "")
            self.post_message(ToolCallStartedMessage(name))

        @self.agent.on_event
        async def on_tool_call(event: AgentEventData):
            if event.event != AgentEvent.TOOL_CALL:
                return
            name = event.data.get("name", "")
            arguments = event.data.get("arguments", {})
            seq = event.data.get("seq")
            self.post_message(ToolCallMessage(name, arguments, seq))

        @self.agent.on_event
        async def on_tool_result(event: AgentEventData):
            if event.event != AgentEvent.TOOL_RESULT:
                return
            name = event.data.get("name", "")
            result = event.data.get("result", "")
            seq = event.data.get("seq")
            # Post through message queue to preserve ordering with tool calls
            self.post_message(ToolResultMessage(name, result, seq))

        @self.agent.on_event
        async def on_error(event: AgentEventData):
            if event.event != AgentEvent.ERROR:
                return
            message = event.message or "Unknown error"
            error_type = event.data.get("error_type", "unknown")
            if error_type == "cancelled":
                # User-initiated interrupt (ESC) — not an error.
                # The ESC feedback is already shown by _interrupt_agent_loop,
                # so silently absorb here (no error panel, no _error_state).
                return
            # Real errors: show error panel and set error state
            self._error_state = True
            self._last_error_type = error_type
            self.post_message(ErrorMessage(message, error_type))

        @self.agent.on_event
        async def on_notification(event: AgentEventData):
            if event.event != AgentEvent.NOTIFICATION:
                return
            message = event.data.get("message", "")
            duration = event.data.get("duration", 5.0)
            level = event.data.get("level", "info")
            self.post_message(NotificationMessage(message, duration, level))

        @self.agent.on_event
        async def on_journal_compact(event: AgentEventData):
            if event.event != AgentEvent.JOURNAL_COMPACT:
                return
            tokens_before = event.data.get("tokens_before", 0)
            tokens_after = event.data.get("tokens_after", 0)
            retrospective = event.data.get("retrospective", False)
            mode = event.data.get("mode", "")
            iteration = event.data.get("iteration")
            total_turns = event.data.get("total_turns")
            # Show simple notification - summary was already streamed during reflection
            savings = tokens_before - tokens_after
            if mode == "all" and iteration and total_turns:
                label = f"Journal all ({iteration}/{total_turns})"
            elif retrospective:
                label = "Retrospective compact"
            else:
                label = "Journal compact"
            asyncio.create_task(
                self._write_system(
                    f"[dim]{label}: {tokens_before}→{tokens_after} words (saved {savings})[/]",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_journal_result(event: AgentEventData):
            if event.event != AgentEvent.JOURNAL_RESULT:
                return
            success = event.data.get("success", False)
            message = event.data.get("message", "")
            if success:
                self._update_info_content(f"[green]{message}[/]")
            else:
                self._update_info_content(f"[yellow]{message}[/]")

        @self.agent.on_event
        async def on_interrupted(event: AgentEventData):
            if event.event != AgentEvent.INTERRUPTED:
                return
            self._interrupt_requested = False
            if self._streaming_reasoning_widget:
                await self._streaming_reasoning_widget.finalize()
                self._streaming_reasoning_widget = None
            if self._streaming_content_widget:
                await self._streaming_content_widget.finalize()
                self._streaming_content_widget = None
            self._in_reasoning = False
            # Drop any pending tool widgets (name shown, args never completed)
            await self._discard_pending_tool_widgets()
            # Increment stream generation to discard any stale tokens
            self._stream_generation += 1
            # Note: Don't show message here - _interrupt_agent_loop handles it
            # to avoid duplicate messages

        @self.agent.on_event
        async def on_paused(event: AgentEventData):
            if event.event != AgentEvent.PAUSED:
                return
            # Agent has paused at a safe point
            self.call_later(self.update_status)
            self._update_info_content("[yellow]Agent paused[/]")
            # The user has usually walked away by the time the pause lands;
            # the bell is what turns the tmux tab yellow to say it is done.
            self._bell.on_paused()

        @self.agent.on_event
        async def on_resumed(event: AgentEventData):
            if event.event != AgentEvent.RESUMED:
                return
            # Agent has resumed from pause
            self.call_later(self.update_status)

        @self.agent.on_event
        async def on_queue_update(event: AgentEventData):
            if event.event != AgentEvent.QUEUE_UPDATE:
                return
            # Queue count changed — refresh status bar to show/hide "Queue: N"
            self.call_later(self.update_status)
            if self._info_pane_mode == "queue":
                # Already showing queue — refresh with current state
                self.call_later(self._show_queue)
            # Background events never open the pane — only user actions do
            # (see on_chat_text_area_submitted is_queued check)

        @self.agent.on_event
        async def on_messages_cleared(event: AgentEventData):
            if event.event != AgentEvent.MESSAGES_CLEARED:
                return
            # Deferred /clear or /clear all completed at safe boundary
            count = event.data.get("count", 0)
            # Both "all" and "trim" modes clear the display
            if self._chat_ready():
                await self._chat.remove_children()
            # Reset TUI token counters to match agent, and drop the
            # safe-point estimate - it describes the history just cleared.
            self._ground_token_display(0, 0)
            self.update_status()
            asyncio.create_task(self._write_system(f"Cleared {count} messages"))

        @self.agent.on_event
        async def on_chat_clear(event: AgentEventData):
            if event.event != AgentEvent.CHAT_CLEAR:
                return
            # Clear the chat display immediately - the user should see the
            # cleared state as soon as the history is cleared, not after a
            # deferred task. Matches the MESSAGES_CLEARED pattern below.
            if self._chat_ready():
                await self._chat.remove_children()

        @self.agent.on_event
        async def on_chat_rebuild(event: AgentEventData):
            if event.event != AgentEvent.CHAT_REBUILD:
                return
            # Rebuild the chat from the current message history. Emitted
            # after chain-restart records are injected, so the user sees
            # the new content. Deferred (heavier widget-tree operation),
            # but self.agent.messages is already populated by this point.
            system_line = event.data.get("system_line")

            async def _do_rebuild():
                await self._rebuild_chat()
                if system_line:
                    await self._write_system(system_line)

            asyncio.create_task(_do_rebuild())

        @self.agent.on_event
        async def on_context_loaded(event: AgentEventData):
            if event.event != AgentEvent.CONTEXT_LOADED:
                return
            # Deferred /load completed at safe boundary
            success = event.data.get("success", False)
            message = event.data.get("message", "")
            incomplete = event.data.get("incomplete", False)
            if success:
                # Sync the token display to the loaded record, and drop the
                # safe-point estimate - it describes the history just replaced.
                self._ground_token_display(
                    self.agent.prompt_tokens, self.agent.completion_tokens
                )
                # Capture resumed session stats for /status
                self._capture_resumed_stats(path=self._pending_load_path)
                self._pending_load_path = None
                self.update_status()
                if incomplete:
                    self._update_info_content(
                        f"[yellow]{message} (incomplete turn - use /resume to continue)[/]"
                    )
                else:
                    self._update_info_content(f"[green]{message}[/]")
                self._session_loaded = True
                asyncio.create_task(self._replay_loaded_messages())
            else:
                self._update_info_content(f"[red]{message}[/]")

        @self.agent.on_event
        async def on_mcp_connecting(event: AgentEventData):
            if event.event != AgentEvent.MCP_CONNECTING:
                return
            asyncio.create_task(
                self._write_system(
                    "[dim]MCP servers connecting…[/]",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_degenerate_nudge(event: AgentEventData):
            if event.event != AgentEvent.DEGENERATE_NUDGE:
                return
            message = event.data.get("message", "")
            level = event.data.get("level", "info")
            style = "yellow" if level == "warning" else "dim"
            asyncio.create_task(
                self._write_system(
                    f"[{style}]{message}[/]",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_mcp_started(event: AgentEventData):
            if event.event != AgentEvent.MCP_SERVER_STARTED:
                return
            server_name = event.server_name or "unknown"
            transport = event.transport or "unknown"
            asyncio.create_task(
                self._write_system(
                    f"[dim]MCP: Starting {server_name} ({transport})[/]",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_mcp_ready(event: AgentEventData):
            if event.event != AgentEvent.MCP_SERVER_READY:
                return
            server_name = event.server_name or "unknown"
            tool_count = event.tool_count or 0
            asyncio.create_task(
                self._write_system(
                    f"[dim]MCP: {server_name} ready ({tool_count} tools)[/]",
                    escape_text=False,
                )
            )

        @self.agent.on_event
        async def on_mcp_error(event: AgentEventData):
            if event.event != AgentEvent.MCP_SERVER_ERROR:
                return
            server_name = event.server_name or "unknown"
            error = event.error or "Unknown error"
            self.post_message(ErrorMessage(f"MCP {server_name}: {error}", "mcp"))

        @self.agent.on_event
        async def on_mcp_stderr(event: AgentEventData):
            if event.event != AgentEvent.MCP_SERVER_STDERR:
                return
            server_name = event.server_name or "unknown"
            line = event.line or ""
            if line.strip():
                asyncio.create_task(
                    self._write_system(
                        f"[dim yellow]MCP {escape_markup(server_name)}:[/] "
                        f"[dim]{escape_markup(line)}[/]",
                        escape_text=False,
                    )
                )

        @self.agent.on_event
        async def on_mcp_warning(event: AgentEventData):
            if event.event != AgentEvent.MCP_SERVER_WARNING:
                return
            warning = event.warning or ""
            if warning:
                asyncio.create_task(
                    self._write_system(
                        f"[yellow]⚠ MCP WARNING:[/]\n{escape_markup(warning)}",
                        escape_text=False,
                    )
                )

        @self.agent.on_event
        async def on_token_usage(event: AgentEventData):
            if event.event != AgentEvent.TOKEN_USAGE:
                return
            # Capture timing data IMMEDIATELY to avoid race condition with _reset_stream_timing
            # Both this and STREAM_START use call_later, so order is non-deterministic
            first_time = self.agent.tracker._first_token_time
            # Use current time as effective "last token time" - TOKEN_USAGE arrives when
            # the API has finished generating, which is the true end of the stream.
            # This handles cases where the API generates tokens that don't arrive as
            # content chunks (e.g., tool call JSON structure).
            effective_last_time = time.time()

            # Update reactive properties (thread-safe via call_later)
            self.call_later(
                self._update_token_usage, event.data, first_time, effective_last_time
            )

        @self.agent.on_event
        async def on_context_estimate(event: AgentEventData):
            if event.event != AgentEvent.CONTEXT_ESTIMATE:
                return
            est = event.data.get("estimated_tokens", 0)
            self.call_later(self._set_context_estimate, est)

        @self.agent.on_event
        async def on_stream_start(event: AgentEventData):
            if event.event != AgentEvent.STREAM_START:
                return

            # NOTE: Do NOT finalize streaming widgets here.
            # This handler runs in the agent task, but _streaming_content_widget
            # is managed by _process_tokens in the token processor task.
            # Finalizing here races with _process_tokens: if ASSISTANT_COMPLETE
            # is still queued, this would prematurely finalize the widget, causing
            # subsequent content tokens to create a NEW "Agent:" widget for the
            # last few words of the message.
            # The _token_queue guarantees ASSISTANT_COMPLETE is always processed
            # before new content tokens, so _finalize_streaming() handles cleanup.
            # Error/interrupt paths finalize via on_interrupted/_interrupt_agent_loop.

            # Update status bar display
            self.call_later(self.update_status)
            # Reset per-stream timing for accurate TPS calculation
            self.call_later(self._reset_stream_timing)

        @self.agent.on_event
        async def on_polite_waiting(event: AgentEventData):
            if event.event != AgentEvent.POLITE_WAITING:
                return
            elapsed = event.data.get("elapsed", 0.0)
            self.call_later(self._set_polite_waiting, elapsed)
            if self._info_pane_mode == "queue":
                # Already showing queue — refresh with "Waiting for lock" header
                self.call_later(self._show_queue, True)
            elif elapsed == 0.0 and not self._queue_pane_dismissed:
                # First polite-wait tick and user hasn't dismissed the pane —
                # open it with the "Waiting for lock" header.  This covers the
                # case where the user's first message to a polite-mode agent
                # was submitted while the agent was still IDLE (processing=False,
                # so is_queued was False and the pane didn't open at submit time).
                self.call_later(self._show_queue, True)
            # Subsequent ticks or dismissed pane: only refresh if already showing

        @self.agent.on_event
        async def on_polite_acquired(event: AgentEventData):
            if event.event != AgentEvent.POLITE_ACQUIRED:
                return
            self.call_later(self._clear_polite_waiting)
            # If queue pane is open, refresh without the "Waiting" header
            if self._info_pane_mode == "queue":
                self.call_later(self._show_queue)

        @self.agent.on_event
        async def on_polite_wait_ended(event: AgentEventData):
            if event.event != AgentEvent.POLITE_WAIT_ENDED:
                return
            self._handle_polite_wait_ended()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat"):
            pass  # Messages mount directly to VerticalScroll
        with Vertical(id="info-pane"):
            yield Static(id="info-content")
        with Vertical(id="input-area"):
            yield ChatTextArea(id="input-field")
        with Horizontal(id="status-line"):
            yield Static("Ready", id="status-left")
            yield Static(id="status-spacer")
            yield Static("", id="status-right")

    def _install_screen_event_guard(self) -> None:
        """Contain Textual's mouse-selection race inside the screen.

        Screen._forward_event's selection branch dereferences the widget
        under the cursor. If that widget is unmounted in the same instant
        as the click (a race with the streaming/collapse re-render), the
        lookup returns a detached widget and the branch crashes on
        ``container.region`` (parent is None). The exception would
        propagate into the App's message loop; Textual's default
        _handle_exception then exits the whole app. Wrapping the method
        at the instance level contains the failure where it happens, so
        the TUI keeps running. If Textual ever renames the internal we
        fall through to the _handle_exception backstop.
        """
        screen = self.screen
        original = getattr(screen, "_forward_event", None)
        if original is None:
            return

        def guarded(event):
            try:
                return original(event)
            except Exception as error:
                self._log_ui_event_error(error)

        screen._forward_event = guarded

    def _log_ui_event_error(self, error: BaseException) -> None:
        """Log a contained UI event error, note it in the chat, keep going."""
        try:
            from agent13.debug_log import log_error

            log_error(error, {"context": "textual_event"})
        except Exception:
            pass
        self.bell()
        asyncio.create_task(
            self._write_system("[dim]⚠ UI event error ignored[/]", escape_text=False)
        )

    def _handle_exception(self, error: BaseException) -> None:
        """Backstop: keep the process alive through runtime Textual exceptions.

        Textual's default (App._handle_exception) always exits on an
        unhandled event exception. The known crash (a MouseDown on a chat
        widget unmounted in the same instant) is contained earlier, in
        _install_screen_event_guard, so the screen stays interactive. This
        backstop covers any other exception that still leaks out: the app
        process and the agent survive; a forced quit (Ctrl+C) saves as
        usual. Before mount we defer to Textual so startup failures still
        fail loudly.
        """
        if not self.is_mounted:
            super()._handle_exception(error)
            return
        self._log_ui_event_error(error)

    async def on_mount(self) -> None:
        """Set up the TUI after mounting."""
        self._install_screen_event_guard()
        # If the event loop ever stops yielding, nothing else can report it -
        # this writes the stack to the debug log (agent13/stall_watchdog.py).
        self._stall_watchdog = StallWatchdog()
        self._stall_watchdog.start()
        # Report from the default-skills sync (new/updated/skipped defaults).
        # cli.py ran the sync before this app existed, so it hands the text
        # over instead of printing into a terminal the TUI is about to own.
        if self._skills_notice:
            self.notify(self._skills_notice, title="Default skills", timeout=12)
        # Create the LLM client in THIS event loop. The client passed at
        # construction was used in the setup loop (model fetch), so its
        # pooled connections are bound to that now-closed loop; reusing
        # them raises "Event loop is closed" on the first request. The
        # setup-loop client was already closed by the CLI.
        if self._provider_args is not None:
            base_url, api_key, read_timeout, connect_timeout = self._provider_args
            self.agent.set_client(
                create_client(
                    base_url,
                    api_key,
                    read_timeout=read_timeout,
                    connect_timeout=connect_timeout,
                )
            )

        # Set skill manager context for skill tool
        if self.agent.skill_manager:
            skill_manager_ctx.set(self.agent.skill_manager)

        # Start the sequential token processor
        self._token_processor_task = asyncio.create_task(self._process_tokens())

        # Validate sandbox profiles on startup
        sandbox_errors = validate_sandbox_profiles()
        if sandbox_errors:
            for error in sandbox_errors:
                print(f"Warning: {error}", file=sys.stderr)

        input_field = self.query_one("#input-field", ChatTextArea)
        input_field.cursor_blink = self._cursor_blink
        input_field.focus()
        self._info_pane = self.query_one("#info-pane", Vertical)
        self._info_content = self.query_one("#info-content", Static)
        self._chat = self.query_one("#chat", VerticalScroll)
        self._status_left = self.query_one("#status-left", Static)
        self._status_right = self.query_one("#status-right", Static)
        self._apply_status_colours()

        # Prevent chat area from taking focus
        self._chat.can_focus = False

        # Enable anchoring for auto-scroll
        self._chat.anchor()

        # Start spinner animation (respects spinner_speed setting)
        self._restart_spinner(self.agent.spinner_speed)

        # Initial status update
        self.update_status()

        # Load auto-save if continuing session
        if self._continue_session:
            self._load_auto_save()

        # Start the agent as a background task
        self._agent_task = asyncio.create_task(self.agent.run())

        # Inject files from --read flag if provided
        if self.read_files:
            from agent13.file_injection import build_read_message

            read_msg = build_read_message(self.read_files)
            await self.agent.add_message(read_msg)

        # Focus name from --focus flag: authoritative state; on a fresh session
        # the records become the session's first turn (the --read pattern).
        # After the auto-save load, so a resumed history is never
        # double-seeded. One wiring for all four UIs - see
        # agent13.durable_records.startup_focus.
        await self._apply_focus_startup()

        # Connect to MCP servers if requested
        if self._connect_mcp and self.agent._mcp_server_configs:
            self._mcp_connect_task = asyncio.create_task(self._connect_mcp_on_startup())

    async def _apply_focus_startup(self) -> None:
        """Apply the ``--focus`` startup flag: set the authoritative name
        and, on a fresh session, enqueue the focus's durable records as the
        session's FIRST TURN - the model reads them and replies a minimal
        ack while the user types their first prompt (the --read pattern).
        A dim system line says what is happening before the ack lands."""
        message, _level, enqueue_msg = startup_focus(
            self.agent.durable_records, self.focus_name, self.agent.messages
        )
        if enqueue_msg:
            # Say what is happening BEFORE the ack lands - the user can
            # start typing their first prompt immediately.
            await self._write_system(
                f"Preprocessing records for focus {escape_markup(self.focus_name)}…",
                escape_text=False,
            )
            await self.agent.add_message(enqueue_msg, kind="prompt")
        if message:
            self._update_info_content(f"[dim]{escape_markup(message)}[/]")

    async def _connect_mcp_on_startup(self) -> None:
        """Connect to MCP servers on startup."""
        try:
            mcp = await self.agent._ensure_mcp()
            if mcp:
                info = await mcp.connect_all()
                if info:
                    await self._write_system(
                        f"[dim]MCP connected: {escape_markup(str(list(info.keys())))}[/]",
                        escape_text=False,
                    )
                else:
                    await self._write_system("[dim]MCP: No servers connected[/]")
        except Exception as e:
            await self._write_system(f"[red]MCP connection error: {e}[/]")
        finally:
            # Refresh the status bar ("mcp: on" right side) — connect runs in
            # the background, so no other event will update it for us.
            self.update_status()

    def on_click(self, event) -> None:
        """Handle click events - ensure input stays focused."""
        # If click is in chat area, refocus input
        input_field = self.query_one("#input-field", ChatTextArea)
        if not input_field.has_focus:
            input_field.focus()

    def on_mouse_up(self, event: MouseUp) -> None:
        """Auto-copy rendered text to clipboard on mouse up.

        Copies the selected rendered text (what you see) and keeps the selection
        visible so you can see what was copied. Use Ctrl+Y to copy the full
        markdown source of the message instead.
        """
        if not self.autocopy_to_clipboard:
            return

        # Check chat area selection
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)
            self.notify(f"Copied {len(text)} chars", title="Copied")
            # Keep selection visible - don't clear
            return

        # Check input field selection
        input_field = self.query_one("#input-field", ChatTextArea)
        if input_field.selected_text:
            self.copy_to_clipboard(input_field.selected_text)
            self.notify(
                f"Copied {len(input_field.selected_text)} chars", title="Copied"
            )
            # Keep selection visible - don't clear

    async def _shutdown(self) -> None:
        await super()._shutdown()
        # on_unmount (dispatched during super()._shutdown) cancels the MCP
        # session tasks, but their unwind is async: the SDK's shielded stdio
        # shutdown needs loop iterations to close the child pipes and
        # transport. asyncio.run's teardown mass-cancels any task still
        # unwinding, leaving transports unclosed whose __del__ raises
        # ValueError("I/O operation on closed pipe") at exit. Drain them
        # here while the loop is still open (bounded: the SDK's internal
        # shutdown timeouts sum to ~4.5s worst case).
        mcp = self.agent.mcp
        if mcp is None:
            return
        tasks = [
            server.session_task
            for server in mcp.servers.values()
            if server.session_task is not None and not server.session_task.done()
        ]
        if tasks:
            await asyncio.wait(tasks, timeout=5.0)

    # -- terminal focus ----------------------------------------------------
    #
    # Tracked WITHOUT Textual's app_focus machinery. Textual's AppBlur
    # handler sets app_focus, which triggers _watch_app_focus ->
    # screen.update_node_styles() (a CSS re-application over EVERY widget)
    # plus a focus restore/repaint. Measured on a real chat: 1.65 s blur /
    # 2.12 s focus at 6,109 widgets, 5.18 s / 5.00 s at 18,309. All we need
    # is for the input cursor to change, so we repaint one widget instead.
    #
    # Not calling super() is deliberate: app_focus stays True. Nothing in
    # this app reads it except CSS App:focus/App:blur rules, of which there
    # are none.

    _terminal_focused: bool = True

    async def _on_app_blur(self, event) -> None:
        """The terminal lost focus - hide the input cursor.

        prevent_default() is load-bearing: Textual dispatches handlers up
        the whole MRO, so without it App._on_app_blur runs too and sets
        app_focus, bringing back the full-tree style walk.

        Under tmux the repaint is SKIPPED when the whole window was switched
        away (see _pane_still_visible): the pane is not on screen, so the
        only effect would be to set the window's activity flag - the yellow
        tab the user sees on every ^a^a.
        """
        event.prevent_default()
        generation = self._focus_generation
        if not await self._pane_still_visible():
            return
        if generation != self._focus_generation:
            # Focus came back while we were asking; the focus handler has
            # already settled the cursor.
            return
        self._terminal_focused = False
        self._refresh_input_cursor()

    async def _on_app_focus(self, event) -> None:
        """The terminal regained focus - draw the input cursor.

        Repaints only if the cursor was actually hidden, so a window switch
        (where the blur was suppressed) writes nothing in either direction.
        """
        event.prevent_default()
        self._focus_generation += 1
        if self._terminal_focused:
            return
        self._terminal_focused = True
        self._refresh_input_cursor()

    async def _pane_still_visible(self) -> bool:
        """True when the terminal we lost focus to still shows our pane.

        tmux sends the SAME focus-out sequence whether focus moved to
        another pane in this window or the whole window was switched away
        (verified 2026-11-24: identical bytes, and the pane-focus-out hook
        fires for both). Only the first case is worth repainting for - in
        the second the pane is not on screen, and the repaint sets the
        window's activity flag, which is the "yellow tab" the user sees on
        every switch away.

        The distinction lives only in the tmux server, so it has to be
        asked for. Costs ~5 ms (measured) on a human-driven transition.

        Outside tmux there is nothing to ask and no activity flag to set,
        so the answer is True: the pre-existing behaviour (hide the cursor
        on any focus loss) is preserved. Any failure also answers True -
        doing what we did before is the safe direction.
        """
        pane = os.environ.get("TMUX_PANE")
        if not os.environ.get("TMUX") or not pane:
            return True
        try:
            proc = await asyncio.create_subprocess_exec(
                "tmux",
                "display-message",
                "-p",
                "-t",
                pane,
                "#{window_active}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
            return out.strip() == b"1"
        except (OSError, asyncio.TimeoutError):
            return True

    def _refresh_input_cursor(self) -> None:
        """Repaint the input after a terminal focus change.

        AppFocus/AppBlur are posted to the App with bubble=False, so the
        input widget never sees them and has to be told.
        """
        for widget in self.query("#input-field"):
            if isinstance(widget, ChatTextArea):
                widget.refresh_focus_appearance()

    def on_unmount(self) -> None:
        """Clean up when unmounting."""
        self._shutting_down = True

        watchdog = getattr(self, "_stall_watchdog", None)
        if watchdog is not None:
            watchdog.stop()

        # Cancel any pending bell timer so it can't fire after teardown.
        self._bell.cancel()

        # Stop the sequential token processor
        if self._token_queue:
            self._token_queue.put_nowait(None)  # Shutdown sentinel
        if self._token_processor_task:
            self._token_processor_task.cancel()

        # Stop the spinner timer
        if hasattr(self, "_spinner_timer") and self._spinner_timer is not None:
            self._spinner_timer.stop()
        self.agent.stop(StopReason.QUIT)
        if self._agent_task:
            self._agent_task.cancel()

        # Synchronously cancel MCP server subprocesses. on_unmount runs during
        # _shutdown, but a fast quit (e.g. /quit right after MCP starts) cancels
        # the run() coroutine, so no *awaited* teardown below it is guaranteed
        # to run (and asyncio.run() would cancel any task we created). Cancelling
        # the _session_runner tasks here is fully synchronous: it injects
        # CancelledError into each runner, whose exit path unwinds the
        # stdio_client transport and closes the child pipes while the loop is
        # still open. That releases the transports before GC, preventing the
        # "RuntimeError: Event loop is closed" from their __del__ at exit.
        mcp = self.agent.mcp
        if mcp is not None:
            mcp._shutting_down = True  # suppress late reconnect attempts
            for server in list(mcp.servers.values()):
                session_task: asyncio.Task | None = server.session_task
                if session_task is not None and not session_task.done():
                    session_task.cancel()
        if self._mcp_connect_task is not None:
            self._mcp_connect_task.cancel()

    async def _safe_mount(self, widget: Widget) -> bool:
        """Mount a widget to the chat, safely handling unmount races.

        Returns True if mounted successfully, False if the TUI is shutting
        down or the chat widget has been unmounted.
        """
        if self._shutting_down:
            return False
        try:
            if not hasattr(self, "_chat") or not self._chat.is_mounted:
                return False
            await self._chat.mount(widget)
            await self._trim_chat()
            return True
        except MountError:
            return False

    @contextmanager
    def _cap_suspended(self):
        """Suspend the live-widget cap for the duration of a rebuild.

        A counter, not a bool, because rebuilds nest: ``_rebuild_chat``
        calls ``_rebuild_from_messages`` and then writes its own status
        line, and that line is mounted through ``_safe_mount`` like any
        other widget. A bool would let the inner call re-enable the cap and
        the status line would trim the rebuild straight back down.
        """
        self._chat_cap_suspends += 1
        try:
            yield
        finally:
            self._chat_cap_suspends -= 1

    async def _trim_chat(self) -> None:
        """Drop the oldest chat widgets beyond MAX_CHAT_MESSAGES.

        Every message is a top-level child of the chat for the life of the
        session, so a long conversation makes any full-tree operation (a
        style walk, a focus toggle) cost seconds. Bounding the live widgets
        is the fix; the message HISTORY is untouched, so /history, /rebuild
        and the model's context still see everything.

        Called from _safe_mount, the single choke point every chat widget
        passes through. The newest widget is never a candidate (the excess
        is taken from the FRONT), so a streaming message cannot be cut.
        """
        if self._chat_cap_suspends:
            return
        try:
            children = list(self._chat.children)
            excess = len(children) - MAX_CHAT_MESSAGES
            if excess <= 0:
                return
            for child in children[:excess]:
                await child.remove()
        except Exception:
            # A trim is housekeeping: never let it break the message that
            # triggered it. Same rule as _scroll_if_at_bottom.
            pass

    async def _scroll_if_at_bottom(self, was_at_bottom: bool) -> None:
        """Scroll to end if the user was at the bottom before a write."""
        if was_at_bottom:
            try:
                self._chat.scroll_end(animate=False)
                self._chat.anchor()
            except Exception:
                pass

    def _chat_ready(self) -> bool:
        """Check if the chat widget is available and mounted."""
        return (
            not self._shutting_down and hasattr(self, "_chat") and self._chat.is_mounted
        )

    async def _clear_chat_display(self) -> None:
        """Clear chat window widgets only (display-only /clear).

        Does not touch message history. Safe to call from sync context
        via asyncio.create_task.
        """
        if not self._chat_ready():
            return
        try:
            await self._chat.remove_children()
        except Exception:
            pass

    async def _clear_and_show_last_n(
        self, keep_turns: int, system_line: str | None = None
    ) -> None:
        """Clear chat window and rebuild showing only last N turns.

        Does not touch message history. A turn = one user message +
        the assistant response + any intervening tool calls/results.

        system_line overrides the footer (chain restart passes
        "Context compacted — showing session summary" instead of the
        default "Showing last N turns..." / "Rebuilt chat from..." line).
        """
        if not self._chat_ready():
            return
        messages = self.agent.messages
        if not messages:
            try:
                await self._chat.remove_children()
            except Exception:
                pass
            return

        # Find the index of the first message in the last N turns
        user_indices = [i for i, m in enumerate(messages) if is_turn_start(m)]
        if len(user_indices) <= keep_turns:
            # Not enough turns to trim — show everything
            if system_line is None:
                await self._rebuild_chat()
                return
            recent = messages
        else:
            cut_index = user_indices[-keep_turns]
            recent = messages[cut_index:]

        with self._cap_suspended():
            try:
                await self._chat.remove_children()
            except Exception:
                pass
            await self._rebuild_from_messages(recent)
            if system_line is None:
                system_line = (
                    f"Showing last {keep_turns} turns ({len(recent)} messages, "
                    "history untouched)"
                )
            await self._write_system(system_line)
            self._chat.scroll_end(animate=False)
            self._chat.anchor()

    async def _write_system(self, text: str, escape_text: bool = True) -> None:
        """Write a system message (non-streaming).

        Args:
            text: The message text to display
            escape_text: If True, escape Rich markup characters in text.
                         Set to False when text contains intentional Rich markup.
        """
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()
        content = escape_markup(text) if escape_text else text
        msg = Static(f"[dim]{content}[/]", classes="system-message", markup=True)
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_user(self, text: str) -> None:
        """Write a user message."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()
        msg = Static(
            f"[cyan]You:[/] {escape_markup(text)}", classes="user-message", markup=True
        )
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_interrupt(self, text: str) -> None:
        """Write an interrupt message injected mid-turn."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()
        msg = Static(
            f"[bold yellow]⚡ Interrupt:[/] {escape_markup(text)}",
            classes="user-message",
            markup=True,
        )
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_tool_call(
        self, name: str, arguments: dict, seq: int | None = None
    ) -> None:
        """Write a tool-call widget (folded per the toolview setting)."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()
        msg = ToolWidget(
            name, collapsed=self.agent.toolview_collapsed, kind="call"
        )
        msg.set_call(name, arguments, seq)
        msg.add_class("tool-call")
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_tool_result(
        self, name: str, result: str, seq: int | None = None
    ) -> None:
        """Write a tool-result widget (folded per the toolview setting)."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()

        msg = ToolWidget(
            name, collapsed=self.agent.toolview_collapsed, kind="result"
        )
        msg.set_result(result, seq)
        msg.add_class("tool-result")
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_error(self, message: str, error_type: str = "unknown") -> None:
        """Write an error message as a styled panel.

        Args:
            message: The error message
            error_type: Type of error (network, auth, rate_limit, permission,
                timeout, model, api, unknown)
        """
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()

        # Choose icon and color based on error type
        if error_type == "network":
            icon = "⚠"
            color = "yellow"
            hint = "\n  [dim]Check your network connection and try again.[/]"
        elif error_type == "auth":
            icon = "🔒"
            color = "red"
            hint = "\n  [dim]Verify your API key is correct.[/]"
        elif error_type == "rate_limit":
            icon = "⏱"
            color = "magenta"
            hint = "\n  [dim]Wait a moment and try again.[/]"
        elif error_type == "permission":
            icon = "🚫"
            color = "red"
            hint = (
                "\n  [dim]Your API key lacks permission or has exceeded its limit.[/]"
            )
        elif error_type == "timeout":
            icon = "⏳"
            color = "yellow"
            hint = "\n  [dim]The request timed out. Try increasing read_timeout in config.[/]"
        elif error_type == "model":
            icon = "❓"
            color = "red"
            hint = ""
        else:
            icon = "✗"
            color = "red"
            hint = ""

        # Create error panel
        error_text = f"[{color}]{icon} Error:[/] {escape_markup(message)}{hint}"
        msg = Static(error_text, classes="error-panel", markup=True)
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _replay_loaded_messages(self) -> None:
        """Replay loaded messages into the chat window after /load command.

        Clears the chat window and repopulates it with messages from the
        loaded context, showing user messages, assistant messages, and
        tool call/result pairs.
        """
        # Clear the chat window
        if not self._chat_ready():
            return
        with self._cap_suspended():
            await self._chat.remove_children()

            # Get messages from agent
            messages = self.agent.messages

            if not messages:
                await self._write_system("[dim]No messages in history[/]")
                return

            await self._rebuild_from_messages(messages)

    async def _rebuild_chat(self) -> None:
        """Rebuild the chat window from message history (/rebuild command).

        Clears the chat window and repopulates it from agent.messages,
        reconstructing user messages, assistant content (with markdown),
        tool calls/results, and reasoning blocks.
        """
        if not self._chat_ready():
            return
        with self._cap_suspended():
            await self._chat.remove_children()

            messages = self.agent.messages

            if not messages:
                await self._write_system("[dim]No messages in history[/]")
                return

            await self._rebuild_from_messages(messages)
            count = len(messages)
            await self._write_system(f"Rebuilt chat from {count} messages")
            self._chat.scroll_end(animate=False)
            self._chat.anchor()

    async def _rebuild_from_messages(self, messages: list) -> None:
        """Rebuild chat widgets from a list of message dicts.

        Shared by _replay_loaded_messages and _rebuild_chat.

        CALLERS MUST WRAP THIS IN ``_cap_suspended()``: a rebuild is the
        deliberate "show me everything" path, so it must not trim itself back
        to MAX_CHAT_MESSAGES as it goes. The suspension lives in the callers
        rather than here because each of them also mounts a status line
        afterwards, and that line goes through _safe_mount like any other
        widget - a suspension that ended at the end of THIS method would let
        the status line trim the rebuild straight back down.

        Renders user messages, assistant content (StreamingMessage with
        markdown), reasoning blocks (ReasoningMessage, folded per the
        /reasoning setting), and tool call/result pairs (ToolWidget, folded
        per the /toolview setting).
        """
        # Track tool calls so we can pair them with results
        pending_tool_calls = {}  # tool_call_id -> (name, arguments)
        # Per-user-turn seq, mirroring the live path: resets on each user
        # message, increments per tool call across the turn's LLM iterations.
        turn_seq = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")

            # Normalize multimodal content (list of blocks) to display text
            content = content_to_text(content)

            if role == "user":
                # Text-only injected user messages (chain-restart nudge)
                # carry no user intent - never render them as user widgets.
                # Image injections (list content) keep their
                # "[Image from tool: X]" trace.
                if msg.get("injected") and isinstance(msg.get("content"), str):
                    continue
                # A real user message starts a new turn: its tool calls
                # number from 1 (injected nudges don't - the turn continues).
                turn_seq = 0
                # Display user message
                if content:
                    await self._write_user(content)

            elif role == "assistant":
                # Injected assistant text (the datetime nudge's role-alternation
                # stub) is not model output - never render it. Same rule as the
                # injected user messages above.
                if msg.get("injected") and isinstance(msg.get("content"), str):
                    continue
                # Display reasoning if present
                reasoning = msg.get("reasoning_content")
                if reasoning and reasoning.strip():
                    was_at_bottom = self._is_at_bottom()
                    # History only holds completed blocks; finalize() puts
                    # the in-progress title in its completed tense, so this
                    # mounts as "Thoughts" exactly like the live path.
                    r_widget = ReasoningMessage(
                        title="Thinking", collapsed=self.agent.reasoning_collapsed
                    )
                    r_widget.add_class("assistant-message")
                    if await self._safe_mount(r_widget):
                        await r_widget.append(reasoning)
                        await r_widget.finalize()
                        await self._scroll_if_at_bottom(was_at_bottom)

                # Display assistant content (if any text before tool calls)
                if content and content.strip():
                    was_at_bottom = self._is_at_bottom()
                    # Use StreamingMessage for replayed assistant content
                    # so markdown is rendered (same as live streaming)
                    msg_widget = StreamingMessage(content=content, title="Agent")
                    msg_widget.add_class("assistant-message")
                    if await self._safe_mount(msg_widget):
                        await self._scroll_if_at_bottom(was_at_bottom)

                # Track tool calls for pairing with results. The per-turn seq
                # mirrors the live path: a running counter that resets on each
                # user message, so rebuilt headers number like the ones that
                # streamed (calls continue 1, 2, ... across the turn's
                # assistant messages).
                for tc in tool_calls:
                    turn_seq += 1
                    seq = turn_seq
                    tc_id = tc.get("id")
                    tc_func = tc.get("function", {})
                    tc_name = tc_func.get("name", "unknown")
                    tc_args = tc_func.get("arguments", "{}")
                    if tc_id:
                        pending_tool_calls[tc_id] = (tc_name, tc_args, seq)

            elif role == "tool":
                # Tool result - pair with the tool call
                if tool_call_id and tool_call_id in pending_tool_calls:
                    tc_name, tc_args_str, seq = pending_tool_calls.pop(tool_call_id)
                    # Parse arguments from JSON string to dict
                    try:
                        tc_args_dict = (
                            json.loads(tc_args_str)
                            if isinstance(tc_args_str, str)
                            else tc_args_str
                        )
                    except json.JSONDecodeError:
                        tc_args_dict = {"raw": tc_args_str}
                    # Display tool call with its result
                    await self._write_tool_call(tc_name, tc_args_dict, seq)
                    # Truncate long tool results for display
                    display_result = content
                    if len(display_result) > 500:
                        display_result = display_result[:500] + "...[truncated]"
                    await self._write_tool_result(tc_name, display_result, seq)
                else:
                    # Orphan tool result (shouldn't happen often)
                    was_at_bottom = self._is_at_bottom()
                    result_widget = Static(
                        f"[dim]Tool result:[/] {escape_markup(content[:200] if len(content) > 200 else content)}",
                        classes="tool-result",
                        markup=True,
                    )
                    if await self._safe_mount(result_widget):
                        await self._scroll_if_at_bottom(was_at_bottom)

        # Scroll to bottom after replay
        self._chat.scroll_end(animate=False)
        self._chat.anchor()

    async def _show_notification(
        self, message: str, duration: float = 5.0, level: str = "info"
    ) -> None:
        """Show a notification message that auto-dismisses after duration.

        Args:
            message: The notification message
            duration: Duration in seconds before auto-dismissal
            level: Notification level (info, warning, error)
        """
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()

        # Choose icon and color based on level
        if level == "warning":
            icon = "⚠"  # Warning sign
            color = "yellow"
        elif level == "error":
            icon = "✗"  # Cross mark
            color = "red"
        else:  # info
            icon = "ℹ"  # Information sign
            # "blue" (#0000ff, SGR 34) is unreadable on the dark panel.
            # NOTE: Textual's Content.from_markup SILENTLY DROPS style names
            # it does not know (e.g. "bright_blue" renders as plain text), so
            # this must be a Textual-palette name: dodgerblue = #1e90ff,
            # SGR 94 (bright blue) on 16-color terminals.
            color = "dodgerblue"

        # Create notification panel
        notification_text = f"[{color}]{icon} {message}[/]"
        notification_widget = Static(
            notification_text, classes="notification-panel", markup=True
        )

        # Mount the notification
        if not await self._safe_mount(notification_widget):
            return
        await self._scroll_if_at_bottom(was_at_bottom)

        # Auto-dismiss after duration
        async def dismiss_after_delay():
            await asyncio.sleep(duration)
            try:
                if notification_widget in self._chat.children:
                    await notification_widget.remove()
            except Exception:
                pass  # Widget may have been unmounted

        asyncio.create_task(dismiss_after_delay())

    def _show_info_pane(self) -> None:
        """Show the info pane."""
        self._info_pane.styles.display = "block"

    def _hide_info_pane(self) -> None:
        """Hide the info pane and save state for toggle-reopen."""
        # Save current mode so Enter can reopen the same view
        if self._info_pane.styles.display != "none":
            self._last_info_mode = self._info_pane_mode
        self._info_pane.styles.display = "none"
        self._info_pane_mode = None

    def _update_info_content(self, content: str) -> None:
        """Update the info pane content."""
        self._info_content.update(content)
        self._last_info_content = content
        self._show_info_pane()

    # ── Info pane formatting helpers ──────────────────────────────────

    @staticmethod
    def _sanitize_text(text: "str | list", max_len: int = 120) -> str:
        """Collapse newlines and truncate for single-line display."""
        # Normalize multimodal content (list of blocks) to text
        if isinstance(text, list):
            text = " ".join(
                part.get("text", "")
                for part in text
                if isinstance(part, dict) and part.get("type") == "text"
            )
        text = " ".join(text.split())
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

    def _reset_stream_timing(self) -> None:
        """Reset timing for a new LLM stream (called on STREAM_START event).

        This ensures each stream (including after tool calls) gets its own
        accurate TPS measurement, rather than accumulating timing across
        multiple streams.
        """
        if is_debug_enabled():
            log_tps_timing_reset(
                source="stream_start",
                old_first=self.agent.tracker._first_token_time,
                old_last=self.agent.tracker._last_token_time,
                old_count=self.agent.tracker._token_count,
                old_stream_start=self._stream_start_time,
            )
        self.agent.tracker.reset_stream()
        self._stream_start_time = time.time()
        if is_debug_enabled():
            log_tps_stream_start(
                source="stream_start",
                first_token_time=self.agent.tracker._first_token_time,
                token_count=self.agent.tracker._token_count,
                stream_start_time=self._stream_start_time,
            )

    def _set_polite_waiting(self, elapsed: float) -> None:
        """Update the polite-waiting overlay (called from event handler)."""
        self._polite_wait_elapsed = float(elapsed)
        self.update_status()

    def _clear_polite_waiting(self) -> None:
        """Clear the polite-waiting overlay (lock acquired, or wait over)."""
        self._polite_wait_elapsed = None
        self.update_status()

    def _handle_polite_wait_ended(self) -> None:
        """The wait ended WITHOUT the lock: /polite off, a re-key, or /pause.

        Cleared from its own event rather than inferred from the next status
        change: on the /polite off path the turn starts immediately and the
        status stays "processing", so nothing else fires and the overlay read
        "wait: Ns" for the whole turn (reported live 2026-11-09).
        """
        self._clear_polite_waiting()
        if self._info_pane_mode == "queue":
            self.call_later(self._show_queue)

    def _update_status(self, status: str) -> None:
        """Update status from agent event."""
        # Map internal status to display status
        status_map = {
            "initialising": "Initialising",
            "idle": "Ready",
            "waiting": "Waiting",
            "thinking": "Thinking",
            "processing": "Processing",
            "tooling": "Tooling",
            "journaling": "Journaling",
            "compacting": "Compacting",
            "paused": "Paused",
        }
        display_status = status_map.get(status, status.capitalize())
        self.status = display_status

        # Update processing state for spinner
        if status in (
            "waiting",
            "thinking",
            "processing",
            "tooling",
            "journaling",
            "compacting",
        ):
            self.processing = True
            # Note: TPS timing reset is handled by STREAM_START event only
            # (see on_stream_start handler). Do NOT reset here - status changes
            # happen during a stream (thinking→processing→tooling) and resetting
            # would wipe out timing before TOKEN_USAGE arrives.
            # _stream_start_time is set per-stream in _reset_stream_timing
            self.agent.tracker.turn_start()
            # Start bell timer if threshold is set and not already running
            self._bell.on_turn_start()
        elif status == "idle":
            self.processing = False
            # Bell: cancel any pending timer, and if threshold is 0 (always
            # ring), ring immediately on return to idle — but only if a
            # turn actually started (not the initial startup idle).
            self._bell.on_turn_end()
            # Clear polite-waiting overlay — covers ESC cancel and /delete q N
            # during polite wait (status goes IDLE without POLITE_ACQUIRED).
            if self._polite_wait_elapsed is not None:
                self._polite_wait_elapsed = None
            # Record the last turn's duration for /status, but only when no
            # queue items remain (a queued message means the turn continues).
            # The status bar's "dur:" comes from the agent's task clock, not
            # from this value.
            if self.agent.queue.pending_count == 0:
                duration = self.agent.tracker.turn_end()
                if duration is not None:
                    self.agent.last_turn_duration = duration
                    self.agent.last_turn_end_time = time.time()
        elif status == "paused":
            self.processing = False
            # Clear the polite-waiting overlay — covers a /pause that took
            # effect DURING a lock wait (the agent went PAUSED without
            # acquiring, so no POLITE_ACQUIRED/IDLE event will clear it).
            if self._polite_wait_elapsed is not None:
                self._polite_wait_elapsed = None
            # Bell: cancel the pending timer and clear _armed so the
            # momentary IDLE emitted on resume (before the agent re-enters
            # processing) doesn't ring. When the turn actually resumes and
            # hits the processing branch, _armed is re-set and a fresh
            # timer starts — so the bell still rings when the resumed turn
            # genuinely completes (including threshold=0 on idle).
            self._bell.on_pause()

    def _set_running(self, running: bool) -> None:
        """Set running state (called when agent stops)."""
        if not running:
            self.processing = False
            self.status = "Stopped"

    @staticmethod
    def _validate_bell_command(command: str) -> bool:
        """Backwards-compatible shim — delegates to BellManager.validate_command."""
        return BellManager.validate_command(command)

    def _set_context_estimate(self, est: int) -> None:
        """Show the live context estimate (with "~") until the next grounded count."""
        self.estimated_context = est
        self.context_is_estimated = True

    def _ground_token_display(self, prompt: int, completion: int) -> None:
        """Set the status bar's token counts and drop the safe-point estimate.

        After the history is REPLACED (/load, /clear) the estimate describes
        a history that no longer exists. Only a TOKEN_USAGE event clears the
        flag, and none arrives until the next LLM call - so without this the
        bar keeps showing the stale "~N" indefinitely (a /load of a 180k save
        into a session sitting at a 13k estimate displayed 13k until the next
        turn).

        The counts are passed in, not read from the agent: /clear trim leaves
        the agent's counters describing the pre-trim history, where 0 is the
        honest display.
        """
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.context_is_estimated = False

    def _update_token_usage(
        self, data: dict, first_token_time: float = None, last_token_time: float = None
    ) -> None:
        """Update token usage from agent event and calculate TPS.

        Delegates TPS computation to TokenTimingTracker. Syncs Textual
        reactive properties from agent data and handles debug logging.

        Args:
            data: Token usage data from TOKEN_USAGE event
            first_token_time: When first token arrived (captured synchronously)
            last_token_time: When last token arrived (captured synchronously)
        """
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)
        # A grounded count just arrived - the estimate is no longer current.
        self.context_is_estimated = False

        result = self.agent.tracker.compute_tps(data, first_token_time, last_token_time)
        if result is not None:
            self._last_tps = result.tps

        if is_debug_enabled():
            first_time = (
                first_token_time
                if first_token_time is not None
                else self.agent.tracker._first_token_time
            )
            last_time = (
                last_token_time
                if last_token_time is not None
                else self.agent.tracker._last_token_time
            )
            elapsed = (last_time - first_time) if (first_time and last_time) else 0
            threshold_passed = result is not None
            tps_value = result.tps if result else None
            log_tps_calculation(
                elapsed=elapsed,
                completion_tokens=self.completion_tokens,
                min_elapsed=self.agent.tracker.MIN_ELAPSED,
                min_tokens=self.agent.tracker.MIN_TOKENS,
                threshold_passed=threshold_passed,
                tps_value=tps_value,
            )
            log_tps_token_usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
                first_token_time=first_time,
                last_token_time=last_time,
                token_count=self.agent.tracker._token_count,
            )

    def _get_spinner_frames(self) -> list[str]:
        """Get the appropriate spinner frames for current speed.

        Fast mode uses braille (10 frames, smooth), slow uses classic (4 frames, chunky).
        Both give ~1 revolution per second at their respective intervals.
        """
        if self.agent.spinner_speed == "slow":
            return self.SPINNER_FRAMES_SLOW
        return self.SPINNER_FRAMES_FAST

    def _advance_spinner(self) -> None:
        """Advance the spinner animation frame."""
        # No animation when spinner is off
        if self.agent.spinner_speed == "off":
            return
        # Only animate spinner when processing (waiting or streaming)
        if self.processing:
            frames = self._get_spinner_frames()
            self._spinner_index = (self._spinner_index + 1) % len(frames)

    def _restart_spinner(self, speed: str) -> None:
        """Restart the spinner timer with a new speed, or stop it.

        Args:
            speed: 'fast' (100ms, braille), 'slow' (250ms, classic), or 'off'
        """
        # Stop existing timer if any
        if hasattr(self, "_spinner_timer") and self._spinner_timer is not None:
            self._spinner_timer.stop()

        # Map speed string to interval
        speed_map = {
            "fast": SpinnerSpeed.FAST.value,  # 0.1
            "slow": SpinnerSpeed.SLOW.value,  # 0.25
            "off": SpinnerSpeed.OFF.value,  # 0
        }
        interval = speed_map.get(speed, SpinnerSpeed.FAST.value)

        if interval > 0:
            self._spinner_timer = self.set_interval(interval, self._advance_spinner)
        else:
            # OFF mode — no timer
            self._spinner_timer = None

        self.agent.spinner_speed = speed
        self._spinner_index = 0  # Reset to avoid OOB after frame set change
        self.update_status()

    def _apply_status_colours(self) -> None:
        """Paint the status bar from [ui] status_fg / status_bg.

        "" leaves that channel to the theme, which is the shipped look (white
        on the theme's primary colour). The two child Statics carry their own
        `color: white` in CSS, so a configured foreground has to be applied to
        them as well - setting it on the container alone loses to them.

        Called from on_mount: the widgets do not exist before that.
        """
        fg = (self._status_fg or "").strip()
        bg = (self._status_bg or "").strip()
        if not fg and not bg:
            return
        bar = self.query_one("#status-line", Horizontal)
        if bg:
            bar.styles.background = bg
        if fg:
            bar.styles.color = fg
            for widget in (self._status_left, self._status_right):
                widget.styles.color = fg

    def update_status(self) -> None:
        """Render the status bar, guarded against re-entry.

        A watcher fired by a reactive this method READS can call straight
        back in, and the cycle then spins the event loop without ever
        yielding - input, signals and /save all queue behind it (observed
        2026-11-19: 80 minutes at 100% CPU, ~19k calls/s, unreachable).
        The nested call has nothing new to draw, so dropping it is correct
        as well as safe.
        """
        if self._updating_status:
            return
        self._updating_status = True
        try:
            self._update_status_body()
        finally:
            self._updating_status = False

    def _update_status_body(self) -> None:
        """Update the status line display.

        Uses event-driven status from agent (self.status) as primary source,
        with special handling for TUI-specific states (pausing, stopped).
        Spinner animates based on processing state.
        """
        # Guard: bail if widgets aren't mounted yet.
        # Reactive watchers (e.g. watch_processing) can fire before on_mount
        # assigns _status_left/_status_right, and reading other reactives
        # inside this method can trigger further watchers (re-entrancy).
        if not hasattr(self, "_status_left"):
            return

        # Spinner logic based on processing state and speed setting
        if self.agent.spinner_speed == "off":
            # Show a static indicator when processing, nothing when idle
            spinner = "..." if self.processing else ""
        elif self.agent.is_pausing:
            spinner = self._get_spinner_frames()[self._spinner_index]
        elif self.processing and self._polite_wait_elapsed is None:
            # Spinner only when actually processing, not during polite wait
            spinner = self._get_spinner_frames()[self._spinner_index]
        else:
            spinner = ""

        # Get status - use agent.pause_state as primary source (single source of truth)
        if self.agent.is_pausing:
            status_text = "pausing"
        elif not self._agent_task or self._agent_task.done():
            status_text = "stopped"
        elif self.agent.is_paused:
            status_text = "paused"
        else:
            # Use the agent's current status (always up-to-date, no stale cache)
            status_text = self.agent.status.value.lower()

        # Polite-mode overlay: when waiting for the shared lock, append the
        # elapsed seconds to the normal status text (e.g. "wait: 5s").
        # The setting value (M) is shown on the right side as "pol: M".
        if self._polite_wait_elapsed is not None:
            status_text = f"wait: {int(self._polite_wait_elapsed)}s"

        # Format token counts with k suffix for thousands
        def format_tokens(n: int) -> str:
            if n >= 1000:
                return f"{n / 1000:.1f}k"
            return str(n)

        # Show the live estimate (with "~") at a safe point; the grounded
        # count (no "~") once the next LLM call reports it.
        if self.context_is_estimated:
            total_str = f"~{format_tokens(self.estimated_context)}"
        else:
            total_str = format_tokens(self.prompt_tokens)

        # Queue count: when polite-waiting, the current item is dequeued but
        # blocked on the lock — it hasn't started processing. Count it so the
        # user sees "Queue: 1" (not 0) for their blocked message.
        queue_count = self.agent.queue.pending_count
        if (
            self._polite_wait_elapsed is not None
            and self.agent.queue.current is not None
        ):
            queue_count += 1
        queue_info = f" | Queue: {queue_count}" if queue_count > 0 else ""

        # Update left side (status info)
        # Compress the model name to fit the bar (provider stays full).
        compressed_model = compress_model(self.agent.model)
        provider_name = current_provider_name(self.agent)
        model_display = (
            f"{provider_name}/{compressed_model}" if provider_name else compressed_model
        )
        # Focus name (session identity; hidden when unset)
        focus_name = self.agent._session_focus_name
        focus_info = f" | focus: {focus_name}" if focus_name else ""
        left_side = f"{spinner} {status_text} | {model_display}{focus_info}{queue_info}".strip()
        self._status_left.update(left_side)

        # Update right side (duration, context, mcp, tools, tps)
        # Task duration: time on the current task, excluding paused time.
        # The agent owns the clock (it survives the auto-context wrap-up and
        # chain restarts, and freezes while paused); the bar only formats it.
        # A frozen clock keeps showing the finished task's total until the
        # next task starts.
        #
        # While a goal window (/grind) is armed the segment gains a
        # denominator: the window's length. Both sides are H:MM - the
        # numerator used to be m:ss with unbounded minutes, so a 3.5-hour task
        # rendered "210:00" next to a "4:00" window. The has_value gate is
        # bypassed while armed: an open window is information even before the
        # first turn.
        elapsed_str = ""
        goal_active = getattr(self.agent, "goal_active", False)
        if self.agent.task_clock.has_value or goal_active:
            elapsed = goal_rules.format_window(self.agent.task_clock.elapsed())
            if goal_active:
                window = goal_rules.format_window(self.agent.goal_window)
                elapsed_str = f"dur: {elapsed}/{window} | "
            else:
                elapsed_str = f"dur: {elapsed} | "

        # Context (always shown, positioned after duration). The segment is
        # styled at 80% and 90% of the auto-context threshold, so the wrap-up
        # is visible before it fires. The style is derived from the SAME
        # number the bar displays (estimate while estimated, grounded count
        # otherwise) - a style computed from the other one would contradict
        # the digits. Disabled auto-context (threshold 0) means nothing to be
        # close to, so no style. Colours come from [ui] in the config file;
        # "" leaves a channel alone, so the default is coloured text on the
        # bar's own background and a "black on red" pair gives a solid block.
        ctx_str = f"Ctx: {total_str}"
        ctx_value = (
            self.estimated_context if self.context_is_estimated else self.prompt_tokens
        )
        ctx_style = ctx_alert_style(
            ctx_value,
            getattr(self.agent, "auto_context_threshold", 0),
            self._ctx_warn_fg,
            self._ctx_warn_bg,
            self._ctx_crit_fg,
            self._ctx_crit_bg,
        )
        if ctx_style:
            ctx_str = f"[{ctx_style}]{ctx_str}[/]"

        # Turn count: "trn: total:turn:turns" - session-lifetime compaction
        # total (persists across saves), chain restarts THIS turn (resets per
        # turn), turns in history. Hidden while both are zero (brand-new
        # session), shown from the first turn/chain onward.
        turn_count = sum(1 for m in self.agent.messages if is_turn_start(m))
        chain_used = getattr(self.agent, "auto_context_chain_used", 0)
        chain_total = getattr(self.agent, "chain_count", 0)
        if chain_used > 0 or turn_count > 0:
            trn_str = f" | trn: {chain_total}:{chain_used}:{turn_count}"
        else:
            trn_str = ""
        # MCP connection status (only show when connected)
        mcp_str = ""
        if self.agent.mcp and self.agent.mcp.is_connected():
            mcp_str = " | mcp: on"

        # Journal mode status (only show when enabled)
        jnl_str = ""
        if self.agent.journal_mode:
            jnl_str = " | jnl: on"

        # Polite mode setting (only show when enabled)
        pol_str = ""
        if self.agent.polite_lock is not None:
            pol_str = f" | pol: {int(self.agent.polite_lock.interval)}"

        # Tool stats (only shown in devel mode)
        tool_stats = self.agent.tool_stats
        if self.agent.devel_mode and tool_stats.total_calls > 0:
            tools_str = (
                f" | Tools: {tool_stats.total_successes}/{tool_stats.total_calls}"
            )
        else:
            tools_str = ""

        # TPS (shown at the end, only when valid)
        tps_str = ""
        if self._last_tps > 0 and self._last_tps <= 500:
            tps_str = f" | tps: {self._last_tps:.1f}"
        elif self._last_tps > 500:
            # Log the anomaly but don't display it
            if is_debug_enabled():
                log_tps_event("sanity_check_failed", {"absurd_tps": self._last_tps})
        # Build right side: dur: | Ctx: | trn: | mcp: | jnl: | pol: | Tools: | tps:
        right_side = f"{elapsed_str}{ctx_str}{trn_str}{mcp_str}{jnl_str}{pol_str}{tools_str}{tps_str}".strip()
        self._status_right.update(right_side)

    def watch_status(self, old: str, new: str) -> None:
        self.update_status()

    def watch_processing(self, old: bool, new: bool) -> None:
        self.update_status()

    def watch__spinner_index(self, old: int, new: int) -> None:
        """Update status when spinner frame changes."""
        self.update_status()

    def watch__streaming(self, old: bool, new: bool) -> None:
        """Update status when streaming state changes."""
        self.update_status()

    def watch_prompt_tokens(self, old: int, new: int) -> None:
        """Update status when prompt tokens change."""
        self.update_status()

    def watch_estimated_context(self, old: int, new: int) -> None:
        """Update status when the context estimate changes."""
        self.update_status()

    def watch_context_is_estimated(self, old: bool, new: bool) -> None:
        """Update status when the estimate/grounded flag changes."""
        self.update_status()

    def watch_completion_tokens(self, old: int, new: int) -> None:
        """Update status when completion tokens change."""
        self.update_status()

    def watch_total_tokens(self, old: int, new: int) -> None:
        """Update status when total tokens change."""
        self.update_status()

    def watch__last_tps(self, old: float, new: float) -> None:
        """Update status when TPS changes."""
        self.update_status()

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        """Handle user input submission from ChatTextArea."""
        text = event.value.strip()
        if not text:
            # Toggle info pane on empty submit: close if visible,
            # reopen last content if hidden
            if self._info_pane.styles.display != "none":
                # Track that user dismissed the queue pane so background
                # events (POLITE_WAITING) don't reopen it.
                if self._info_pane_mode == "queue":
                    self._queue_pane_dismissed = True
                self._hide_info_pane()
            elif self._last_info_content is not None:
                self._info_pane_mode = self._last_info_mode
                self._info_content.update(self._last_info_content)
                self._show_info_pane()
            return

        input_field = self.query_one("#input-field", ChatTextArea)
        input_field.clear()
        input_field.focus()

        # Reset completion state after submission
        self._reset_completion_state()
        self._hide_info_pane()
        # New user input — allow queue pane to open again if polite wait
        # fires later (e.g. first message to a polite-mode agent).
        self._queue_pane_dismissed = False

        # Any new input (except /resume) closes the /resume-after-interrupt window
        if not text.startswith("/resume"):
            self._interrupt_available = False

        # A bare "!"/"!!" is a control, not a message - keep it out of history.
        bare_promotion = text in ("!", "!!")

        # Add to persistent history
        if not bare_promotion:
            self._history.add(text)
        self._history.reset()

        if text.startswith("/"):
            self._handle_command(text)
        else:
            # Check for interrupt (!! at start) or priority (! at start) markers
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

            # "!!" or "!" ALONE promotes the most recently queued item
            # instead of queueing an empty one.
            if bare_promotion:
                self._promote_latest(interrupt)
                return
            # If message will be queued (agent busy, or an MCP connect is in
            # flight — the agent defers starting turns until it finishes),
            # show queue pane so the user sees their message queued
            is_queued = (
                self.processing
                or self.agent.queue.pending_count > 0
                or (self.agent.mcp is not None and self.agent.mcp.connecting)
            )
            if is_queued:
                self.call_later(self._show_queue)

            # Send message to agent (ITEM_STARTED handler will show in chat when processed)
            asyncio.create_task(self._send_message(message_text, priority, interrupt))

    def on_chat_text_area_history_previous(
        self, event: ChatTextArea.HistoryPrevious
    ) -> None:
        """Handle history previous (Ctrl+B) with prefix matching."""
        input_field = self.query_one("#input-field", ChatTextArea)
        prefix = event.prefix

        if self._history.in_prefix_mode():
            # Check if buffer changed from what we expect
            saved_prefix = self._history.get_prefix()
            current_match = (
                self._history._prefix_matches[self._history._prefix_match_idx][1]
                if self._history._prefix_match_idx >= 0
                else None
            )
            # Continue if buffer is still the prefix or a match we navigated to
            if prefix != saved_prefix and prefix != current_match:
                # User changed input, restart prefix mode
                self._history.start_prefix_navigation(prefix)
        else:
            # Start prefix mode (empty prefix matches all)
            self._history.start_prefix_navigation(prefix)

        cmd = self._history.up_with_prefix()

        if cmd is not None:
            input_field.clear()
            input_field.insert(cmd)

    def on_chat_text_area_history_next(self, event: ChatTextArea.HistoryNext) -> None:
        """Handle history next (Ctrl+F) with prefix matching."""
        input_field = self.query_one("#input-field", ChatTextArea)

        if self._history.in_prefix_mode():
            # Prefix mode: continue filtered navigation
            cmd = self._history.down_with_prefix()
            if cmd is None:
                # Return to showing the original prefix
                cmd = self._history.get_prefix()
        else:
            # Normal mode: regular history navigation
            cmd = self._history.down()

        input_field.clear()
        input_field.insert(cmd if cmd is not None else "")

    async def _send_message(
        self, text: str, priority: bool = False, interrupt: bool = False
    ) -> None:
        """Send a message to the agent.

        Args:
            text: The message text
            priority: If True, add to front of queue
            interrupt: If True, interrupt agent loop (implies priority)
        """
        expanded = expand_file_mentions(text)
        await self.agent.add_message(expanded, priority=priority, interrupt=interrupt)

    def _promote_latest(self, interrupt: bool) -> None:
        """Promote the most recently queued item (a bare "!"/"!!" submit).

        The shared helper owns the rule and the wording; this only renders
        the line and refreshes an open queue pane, whose markers just moved.
        """
        line = promote_latest(self.agent, interrupt)
        # escape_markup_open, not _write_system's default: the line carries a
        # bracketed tag ("[steer]"), and escaping both brackets leaves a
        # visible backslash before the "]".
        asyncio.create_task(
            self._write_system(escape_markup_open(line), escape_text=False)
        )
        if self._info_pane_mode == "queue":
            self.call_later(self._show_queue)

    def _run_command_follow_up(self, result, args: str) -> bool:
        """Run the TUI-only tail a shared command asks for via ``result.data``.

        Some commands cannot finish inside a handler: a listing opens the
        completion picker instead of printing, a snippet fills the input, a
        provider switch has to fetch the new model list asynchronously. Each is
        announced in the result's data, so the handler stays free of UI
        callbacks.

        Returns True when this consumed the result (nothing to render).
        """
        # /retry: the shared handler removed the last group; the TUI's tail is
        # to load the text into the input for editing rather than re-send it.
        if "user_text" in result.data:
            self._load_retry_text(result.data["user_text"])
            return True

        # /prompt with no args: the TUI's tail is its completion picker, not a
        # printed list (the REPL prints the list).
        prompt_picker = result.data.get("prompt_picker")
        if prompt_picker:
            names, active = prompt_picker
            self._enter_command_completion("prompt", names, active)
            return True

        # /snippet use: load the text into the input for editing (the REPL has
        # no input field, so it prints the text instead).
        if "snippet_text" in result.data:
            input_field = self.query_one("#input-field", ChatTextArea)
            input_field.text = result.data["snippet_text"]
            input_field.focus()
            return True

        # /snippet add|delete|rename: keep the dynamic slash commands in step.
        # Not consumed - the handler's confirmation still renders.
        added = result.data.get("snippet_added")
        if added:
            self._add_snippet_slash_command(added)
        removed = result.data.get("snippet_removed")
        if removed:
            self._remove_snippet_slash_command(removed)

        # /wrapup, /chain, /journal last|all: queue the agent turn. Not
        # consumed - the handler's confirmation still renders.
        agent_message = result.data.get("agent_message")
        if agent_message:
            text, kind, data = agent_message
            asyncio.create_task(self.agent.add_message(text, kind=kind, data=data))

        # /mcp connect|disconnect|reload: the action is async. Not consumed -
        # the handler's "Connecting..." shows until the result replaces it.
        mcp_action = result.data.get("mcp_action")
        if mcp_action:
            asyncio.create_task(self._run_mcp_action(mcp_action))

        # /new_session: the wipe is always deferred through the queue, so it
        # lands at a safe boundary. The chat display is reset by the
        # MESSAGES_CLEARED handler, not here.
        clear_mode = result.data.get("clear_mode")
        if clear_mode:
            asyncio.create_task(self.agent.request_clear(mode=clear_mode))

        # /load: deferred through the queue. The path is recorded first so
        # on_context_loaded can capture the resumed token stats.
        load_path = result.data.get("load_path")
        if load_path:
            self._pending_load_path = Path(load_path)
            asyncio.create_task(self.agent.request_load(load_path))
        if not args.strip():
            if "models" in result.data:
                self._enter_command_completion(
                    "model", result.data["models"], result.data.get("current", "")
                )
                return True
            if "providers" in result.data:
                self._enter_command_completion(
                    "provider", result.data["providers"], result.data.get("current", "")
                )
                return True

        base_url = result.data.get("base_url")
        if base_url:
            self._render_command_result(result, args)
            asyncio.create_task(self._fetch_and_show_models(base_url))
        return False

    def _render_command_result(self, result, args: str = "") -> None:
        """Render a shared command's result in the info pane.

        The only side-effects a shared command can ask for are a status-bar
        refresh and keeping the info pane open in a given mode.

        Colour follows the info-pane convention: a no-arg invocation is a
        status query (yellow), a change reads green, a failure reads red.
        """
        if not result.success:
            colour = "red"
        elif not args.strip():
            colour = "yellow"
        else:
            colour = "green"
        self._update_info_content(
            f"[{colour}]{escape_markup_open(result.message)}[/]"
        )
        if result.pane:
            self._info_pane_mode = result.pane
        if result.refresh_status:
            self.update_status()

    def _handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
        parts = cmd[1:].split(maxsplit=1)
        if not parts:
            # User entered just "/" or "/ " with no command
            asyncio.create_task(
                self._write_error("Empty command. Type /help for available commands.")
            )
            return
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Shared commands run from the registry. The chain below covers the
        # ones this front-end implements itself (screen, input, widgets).
        result = run_command(command, self.agent, args)
        if result is not None:
            if not self._run_command_follow_up(result, args):
                self._render_command_result(result, args)
            return

        if command == "help":
            self._handle_help_command()
        elif command == "clear":
            arg = args.strip().lower()
            if arg == "all":
                asyncio.create_task(
                    self._write_system(
                        "/clear all has moved: use /new_session "
                        "(wipe history + display, start a new session)"
                    )
                )
            elif arg.isdigit():
                asyncio.create_task(
                    self._write_system(
                        f"/clear N has moved: use /rebuild {arg} "
                        "(re-render the chat showing the last N turns)"
                    )
                )
            else:
                # /clear with no args — pure UI operation, history untouched
                asyncio.create_task(self._clear_chat_display())
        elif command == "rebuild":
            arg = args.strip()
            if arg.isdigit():
                # Scoped re-render: clear display, show last N turns
                asyncio.create_task(
                    self._clear_and_show_last_n(int(arg))
                )
            elif arg:
                asyncio.create_task(
                    self._write_system("Usage: /rebuild [N] — N = turns to show")
                )
            else:
                asyncio.create_task(self._rebuild_chat())
        elif command == "pause":
            self._handle_pause_command()
        elif command == "resume":
            self._handle_resume_command()
        elif command == "reasoning":
            self._handle_collapse_command("reasoning", args)
        elif command == "toolview":
            self._handle_collapse_command("toolview", args)
        elif command == "spinner":
            self._handle_spinner_command(args)
        elif command == "pretty":
            # Deliberately not offered here (see _TUI_NOT_IMPLEMENTED):
            # Textual always renders markdown. Point at where it does apply.
            asyncio.create_task(
                self._write_system(
                    "/pretty is not available in the TUI (it always renders "
                    "markdown); it applies to REPL and batch mode"
                )
            )
        elif command == "upgrade":
            # Deliberately not offered here (see _TUI_NOT_IMPLEMENTED): the
            # startup check offers the update when the user learns of it, and
            # `agent13 --upgrade` covers every other case. Point there rather
            # than applying an update with no confirmation prompt.
            asyncio.create_task(
                self._write_system(
                    "/upgrade is not available in the TUI: the startup check "
                    "offers it, or run `agent13 --upgrade`"
                )
            )
        elif command == "clipboard":
            self._handle_clipboard_command(args)
        elif command in ("quit", "exit"):
            self.agent.stop(StopReason.QUIT)
            self.exit()
        else:
            # Dynamic /skillname: the shared helper loads the skill, ratchets
            # the tool on and queues the turn. None means "not a skill".
            from agent13.commands import execute_skill_invocation

            skill_result = execute_skill_invocation(self.agent, command, args)
            if skill_result is not None:
                if not self._run_command_follow_up(skill_result, args):
                    self._render_command_result(skill_result, args)
                return
            # Check for snippet invocation
            snippet = self.agent.snippet_manager.get_snippet(command)
            if snippet is not None:
                input_field = self.query_one("#input-field", ChatTextArea)
                input_field.text = snippet
                input_field.focus()
                return
            self._update_info_content(f"[red]Unknown command: {escape_markup(cmd)}[/]")

    def _enter_command_completion(
        self, command: str, matches: list[str], current_value: str | None = None
    ) -> None:
        """Enter tab-completion mode for a slash command.

        Sets the input to /command and shows completions, starting at
        the current_value if provided (e.g. current model or active prompt).

        Args:
            command: Slash command name (e.g. "model", "prompt")
            matches: List of completion matches
            current_value: Current value to highlight and start at (optional)
        """
        input_field = self.query_one("#input-field", ChatTextArea)
        input_field.text = f"/{command} "
        cursor_col = len(f"/{command} ")
        input_field.move_cursor((0, cursor_col))

        # Trigger completion for the current input context
        completion_type, start_loc, end_loc = self._get_completion_context(
            f"/{command} ", 0, cursor_col
        )
        if completion_type == "none" or not matches:
            self.call_later(input_field.focus)
            return

        # Find starting index — default to 0, or current_value's position
        start_idx = 0
        if current_value and current_value in matches:
            start_idx = matches.index(current_value)

        if len(matches) == 1:
            # Single match - complete immediately
            self._apply_completion(input_field, matches[0], start_loc, end_loc)
        else:
            # Multiple matches - enter completion mode
            self._completion_matches = matches
            self._completion_index = start_idx
            self._completion_prefix = ""
            self._completion_start = start_loc
            self._completion_end = end_loc
            self._apply_completion(input_field, matches[start_idx], start_loc, end_loc)
            self._completion_text = input_field.text
            self._show_completions(matches, matches[start_idx])

        self.call_later(input_field.focus)

    def _handle_help_command(self) -> None:
        """Handle /help command - generated from the shared command registry."""
        lines = []
        for section in format_help_sections(_TUI_LOCAL_COMMANDS):
            lines.append(f"[bold]{section.title}:[/]")
            for entry in section.entries:
                lines.append(
                    f"  [yellow]/{entry.name}[/] - {escape_markup(entry.description)}"
                )
                for detail in entry.detail:
                    lines.append(f"      [dim]{escape_markup_open(detail)}[/]")
            lines.append("")
        lines.extend(
            [
                "[bold]Keyboard shortcuts:[/]",
                "  [yellow]ESC[/] - Cancel request (use /resume to continue)",
                "  [yellow]Ctrl+C[/] - Clear input or quit",
                "  [yellow]Ctrl+D / Ctrl+Q[/] - Quit app",
                "  [yellow]Tab[/] - Tab completion for commands/history",
                "  [yellow]Up/Down[/] - Navigate input history",
                "  [yellow]Shift+Up/Down[/] - Scroll chat",
                "",
                "[bold]Tips:[/]",
                "  [yellow]![/]message - Priority: queued ahead of normal items "
                "(after any steers)",
                "  [yellow]!![/]message - Steer: add context into the agent's "
                "current turn without cancelling",
                "  [yellow]![/] or [yellow]!![/] alone - promote the most "
                "recently queued message to that level",
            ]
        )
        self._update_info_content("\n".join(lines))
        self._info_pane_mode = "help"

    def _capture_resumed_stats(self, path: Path | None = None) -> None:
        """Snapshot loaded session stats for /status display.

        Called after a successful context load to capture turn count,
        token counts, and save timestamp. Resets session timing since
        this is effectively a new conversation.

        Args:
            path: Optional path to .ctx file (to read saved_at).
        """
        # Read saved_at from .ctx file if path provided
        if path:
            try:
                with open(path) as f:
                    ctx_data = json.load(f)
                self.agent.resumed_saved_at = ctx_data.get("saved_at")
            except (OSError, json.JSONDecodeError):
                pass
        # Count turn-start user messages for turn count (skips interrupts
        # and native-vision image injections)
        self.agent.resumed_turn_count = sum(
            1 for m in self.agent.messages if is_turn_start(m)
        )
        # Snapshot token counters
        self.agent.resumed_prompt_tokens = self.prompt_tokens
        self.agent.resumed_completion_tokens = self.completion_tokens
        # Reset session timing (conversation effectively restarts)
        self.agent.session_start_time = time.time()
        self.agent.tracker._turn_count = 0
        self.agent.tracker._total_processing_time = 0.0
        self.agent.last_turn_duration = None
        self.agent.last_turn_end_time = None

    def _show_queue(self, polite_waiting: bool = False) -> None:
        """Render the queue listing (the shared handler owns the text).

        Called by the command path through the registry, and internally by
        the queue-change and polite-wait ticks. ``polite_waiting`` prepends
        the "Waiting for lock" header; it is not a command form.
        """
        result = execute_queue(self.agent, "", polite_waiting=polite_waiting)
        self._render_command_result(result, "")

    def _handle_pause_command(self) -> None:
        """Handle /pause command - pause agent processing.

        When the agent is actively processing (in the middle of tool calls,
        including a polite multi-agent lock wait), this requests a pause at
        the next safe point and shows 'Pausing' status. A lock wait IS a safe
        point: acquire() notices the pause flag on its next poll tick and
        backs off (PolitePauseRequested), so the agent goes PAUSED without
        interrupting the turn — the queue item and history are untouched, and
        /resume re-enters the lock wait. When the pause takes effect, status
        changes to 'paused'.
        """
        if self.agent.is_paused:
            self._update_info_content("[yellow]Already paused[/]")
            return

        if self.agent.is_pausing:
            self._update_info_content("[yellow]Already pausing[/]")
            return

        if self.processing or self.agent.has_pending_work:
            # Agent is actively processing (turn in flight, or waiting on the
            # polite lock - both report WAITING status) - request pause at
            # the next safe point. has_pending_work also covers the window
            # between submitting a message and the run loop starting it,
            # where self.processing is still False: taking the idle branch
            # there stopped the agent loop instead of pausing it.
            self.agent.pause()
            self.update_status()  # Update status to show "pausing"
            self._update_info_content("[yellow]Pausing at next safe point...[/]")
        else:
            # Agent is idle - pause immediately
            self.agent.stop()
            self._update_info_content("[yellow]Agent paused[/]")

    def _handle_resume_command(self) -> None:
        """Handle /resume command - resume agent processing.

        Resumes from where the agent left off. Handles four cases:
        1. After ESC interrupt - sends "Actually, please continue" to continue
        2. Agent paused mid-turn (PAUSED) - unpauses and continues
        3. Agent pausing mid-turn (PAUSING) - cancels pause request
        4. Agent stopped - restarts the agent loop
        """
        # After ESC interrupt: send a continuation message
        if self._interrupt_available:
            self._interrupt_available = False
            asyncio.create_task(self._send_message("Actually, please continue"))
            self._update_info_content("[green]Continuing from interrupt[/]")
            return

        # Check for incomplete turn from loaded context (even if not paused)
        if self.agent.has_incomplete_turn:
            # Signal run() to continue the incomplete turn at its next
            # loop iteration. This avoids a separate task — run() handles
            # it inside its own loop, so the pause state machine stays
            # single-tasked. If run() is dead, restart it. If paused,
            # resume so it can reach the check.
            if not self._agent_task or self._agent_task.done():
                self._agent_task = asyncio.create_task(self.agent.run())
            elif self.agent.is_paused or self.agent.is_pausing:
                self.agent.resume()
            self.agent.request_continue_incomplete()
            self.update_status()
            self._update_info_content("[green]Continuing incomplete turn...[/]")
            return

        if self.agent.pause_state == PauseState.RUNNING:
            self._update_info_content("[yellow]Not paused[/]")
            return

        # Resume the agent — works for both PAUSED and PAUSING states
        if self.agent.is_paused or self.agent.is_pausing:
            # Agent is paused or pausing — resume() handles both
            self.agent.resume()
        else:
            # Agent was stopped — restart it
            self._agent_task = asyncio.create_task(self.agent.run())

        # Immediately update status display (agent will emit STATUS_CHANGE shortly)
        self.update_status()
        self._update_info_content("[green]Agent resumed[/]")

    def _load_retry_text(self, user_text: str) -> None:
        """Load /retry's text into the input for editing (like /snippet use).

        The TUI's half of /retry: the shared handler already removed the last
        group from the history.
        """
        # Clear error/interrupt state
        self._error_state = False
        self._last_error_type = None
        self._interrupt_requested = False
        self._interrupt_available = False

        input_field = self.query_one("#input-field", ChatTextArea)
        input_field.text = user_text
        input_field.focus()
        self.update_status()
        self._update_info_content(
            "[dim]Last message removed and loaded for editing. Press Enter to retry.[/]"
        )

    async def _fetch_and_show_models(self, base_url: str) -> None:
        """Fetch models from the new provider and show them.

        The async tail of /provider: the shared handler swapped the client,
        this fills the model list the picker and /model read.
        """
        try:
            model_names = await fetch_models(self.agent.client)
        except Exception as e:
            self._update_info_content(
                f"[red]Error fetching models: {escape_markup(str(e))}[/]"
            )
            return

        if not model_names:
            self._update_info_content("[red]No models available from provider[/]")
            return

        self.agent.available_models = model_names

        lines = [f"[bold]Provider changed to: {escape_markup(base_url)}[/]"]
        for line in format_model_list(model_names).splitlines():
            lines.append(escape_markup(line))
        lines.append("")
        lines.append("[dim]Use /model <name> or /model <tab> to select[/]")
        self._update_info_content("\n".join(lines))

    async def _run_mcp_action(self, action: str) -> None:
        """Run /mcp connect|disconnect|reload and show the result."""
        from agent13.commands import run_mcp_action

        ok, message = await run_mcp_action(self.agent, action)
        colour = "green" if ok else "red"
        self._update_info_content(f"[{colour}]{escape_markup_open(message)}[/]")
        self.update_status()

    def _handle_collapse_command(self, kind: str, args: str) -> None:
        """Handle /reasoning and /toolview - fold a widget family (display)."""
        from agent13.commands import execute_collapse

        current = (
            self.agent.reasoning_collapsed
            if kind == "reasoning"
            else self.agent.toolview_collapsed
        )

        def _set(value: bool) -> None:
            if kind == "reasoning":
                self.agent.reasoning_collapsed = value
            else:
                self.agent.toolview_collapsed = value

        result = execute_collapse(kind, args, current, _set)
        # Plain text from the shared command; escape it for the info pane.
        message = escape_markup(result.message)
        color = "green" if result.success else "red"
        self._update_info_content(f"[{color}]{message}[/]")

    def _handle_spinner_command(self, args: str) -> None:
        """Handle /spinner command - set spinner animation speed."""
        args = args.strip().lower()
        if args in ("fast", "slow", "off"):
            self._restart_spinner(args)
            label = {
                "fast": "Fast (100ms, braille)",
                "slow": "Slow (250ms, classic)",
                "off": "Off",
            }[args]
            color = {"fast": "green", "slow": "yellow", "off": "red"}[args]
            self._update_info_content(f"[{color}]Spinner: {label}[/]")
        elif args == "status" or not args:
            speed = self.agent.spinner_speed
            label = {
                "fast": "Fast (100ms, braille)",
                "slow": "Slow (250ms, classic)",
                "off": "Off",
            }.get(speed, speed)
            color = {"fast": "green", "slow": "yellow", "off": "red"}.get(
                speed, "green"
            )
            self._update_info_content(f"[{color}]Spinner: {label}[/]")
        else:
            speed = self.agent.spinner_speed
            self._update_info_content(
                f"[red]Usage: /spinner [fast|slow|off|status][/]\n"
                f"  [yellow]/spinner fast[/] - Braille spinner, 100ms (default)\n"
                f"  [yellow]/spinner slow[/] - Classic spinner, 250ms (slow links)\n"
                f"  [yellow]/spinner off[/] - No spinner\n"
                f"  [yellow]/spinner status[/] - Show current speed\n"
                f"  Current: {speed}"
            )

    def _handle_clipboard_command(self, args: str) -> None:
        """Handle /clipboard command - show or set clipboard method."""
        from agent13.clipboard import VALID_METHODS

        args = args.strip().lower()

        if not args:
            # Show current method
            method = self.agent.clipboard_method
            desc = (
                "terminal escape sequence (works over SSH)"
                if method == "osc52"
                else "OS-level commands (works in tmux, screen, PowerShell)"
            )
            self._update_info_content(
                f"Clipboard method: [bold]{method}[/]\n"
                f"  {desc}\n\n"
                f"Change with: [yellow]/clipboard osc52[/] or "
                f"[yellow]/clipboard system[/]"
            )
            return

        if args not in VALID_METHODS:
            self._update_info_content(
                f"[red]Unknown method: {args}[/]\n"
                f"Valid methods: [yellow]{', '.join(VALID_METHODS)}[/]"
            )
            return

        # Update in-memory
        self.agent.clipboard_method = args

        # Persist to config file
        try:
            from agent13.config_paths import get_config_file

            config_path = get_config_file()
            if config_path.exists():
                content = config_path.read_text()
                # Update or add [clipboard] section
                import re

                pattern = r'\[clipboard]\s*\nmethod\s*=\s*"[^"]*"'
                replacement = f'[clipboard]\nmethod = "{args}"'
                if re.search(pattern, content):
                    new_content = re.sub(pattern, replacement, content)
                else:
                    # Add [clipboard] section at the end
                    new_content = (
                        content.rstrip() + f'\n\n[clipboard]\nmethod = "{args}"\n'
                    )
                config_path.write_text(new_content)
        except OSError as e:
            self._update_info_content(
                f"[yellow]Clipboard set to {args} for this session, "
                f"but could not save to config: {e}[/]"
            )
            return

        desc = (
            "terminal escape sequence (works over SSH)"
            if args == "osc52"
            else "OS-level commands (works in tmux, screen, PowerShell)"
        )
        self._update_info_content(f"[green]Clipboard method: {args}[/]\n  {desc}")

    def _load_auto_save(self) -> None:
        """Load the latest auto-save if available."""
        path = find_latest_auto_save()
        if path is None:
            self._update_info_content(
                "[dim]No auto-save found, starting fresh session[/]"
            )
            return

        success, message, incomplete = load_context(self.agent, path)
        from agent13.debug_log import log_journal_debug

        log_journal_debug(
            "auto_save_load",
            {
                "success": success,
                "message": message,
                "path": str(path),
                "messages_count": len(self.agent.messages),
                "has_tool_calls": self.agent.history.has_tool_calls(),
            },
        )
        if success:
            # Reset TUI token counters
            self.prompt_tokens = self.agent.prompt_tokens
            self.completion_tokens = self.agent.completion_tokens
            self.total_tokens = self.agent.prompt_tokens + self.agent.completion_tokens
            # Capture resumed session stats for /status
            self._capture_resumed_stats(path=path)
            # Note: We intentionally do NOT sync model from loaded context
            # User keeps their current provider/model settings
            self.update_status()
            if incomplete:
                self._update_info_content(
                    f"[yellow]Resumed session from {path.name} (incomplete turn - use /resume to continue)[/]"
                )
            else:
                self._update_info_content(f"[green]Resumed session from {path.name}[/]")
            self._session_loaded = True
            # Replay loaded messages into chat window
            asyncio.create_task(self._replay_loaded_messages())
        else:
            self._update_info_content(f"[yellow]Could not load auto-save: {message}[/]")

    def on_token_message(self, message: TokenMessage) -> None:
        """Enqueue token for sequential processing by _process_tokens.

        This handler is synchronous (as required by Textual message handlers),
        so it only enqueues. All async widget operations happen in _process_tokens
        which processes tokens one at a time, guaranteeing ordering.
        """
        # Discard stale tokens from previous streaming sessions,
        # but always allow is_final through so the old widget gets finalized.
        # Without this, on_item_started increments _stream_generation before
        # the prior turn's is_final (posted via post_message) is processed,
        # causing it to be discarded and leaving the old widget un-finalized.
        if message.generation != self._stream_generation and not message.is_final:
            return
        self._token_queue.put_nowait(message)

    def on_tool_call_started_message(self, message: ToolCallStartedMessage) -> None:
        """Enqueue a tool-call-started message for sequential processing.

        Must go through _token_queue (like ToolCallMessage) so the pending
        name-only widget is mounted in order with the streaming content and the
        subsequent complete tool call that updates it in place.
        """
        self._token_queue.put_nowait(message)

    def on_tool_call_pending_message(self, message: ToolCallPendingMessage) -> None:
        """Enqueue a tool-call-pending message for sequential processing.

        Same ordering requirements as ToolCallStartedMessage: the placeholder
        must mount in order with streaming content, and before the tool call
        that replaces it.
        """
        self._token_queue.put_nowait(message)

    def on_tool_call_message(self, message: ToolCallMessage) -> None:
        """Enqueue tool call for sequential processing.

        Tool calls must finalize any streaming widget BEFORE writing the tool
        call message. Enqueuing ensures this ordering without create_task races.
        """
        # Log tool call with timing state
        if is_debug_enabled():
            log_tps_tool_call(
                tool_name=message.name,
                first_token_time=self.agent.tracker._first_token_time,
                last_token_time=self.agent.tracker._last_token_time,
                token_count=self.agent.tracker._token_count,
                pending_token_usage=None,
            )
        self._token_queue.put_nowait(message)

    async def _process_tokens(self) -> None:
        """Process tokens and tool calls sequentially from the queue.

        This is the single consumer that guarantees all widget operations
        (mount, append, finalize, transition) happen in order. No more
        create_task fire-and-forget for widget operations — every await
        completes before the next operation begins.

        This eliminates the class of race conditions where:
        - Finalize and start-content run concurrently (content bleeds into reasoning)
        - Finalize-for-tool and write-tool-call run concurrently (tool call appears
          before streaming widget is cleaned up)
        - Multiple append tasks interleave (visual corruption)
        """
        while True:
            message = await self._token_queue.get()
            if message is None:
                # Shutdown sentinel
                break

            try:
                if isinstance(message, ToolCallMessage):
                    await self._handle_tool_call(message)
                    continue

                if isinstance(message, ToolCallStartedMessage):
                    await self._handle_tool_call_started(message)
                    continue

                if isinstance(message, ToolCallPendingMessage):
                    await self._handle_tool_call_pending(message)
                    continue

                if isinstance(message, ToolResultMessage):
                    await self._handle_tool_result(message)
                    continue

                if isinstance(message, SystemQueueMessage):
                    await self._write_system(
                        message.text, escape_text=message.escape_text
                    )
                    continue

                # It's a TokenMessage
                await self._handle_token(message)
            except Exception as e:
                # Log but don't die — the processor must keep running
                # or the chat window becomes permanently disconnected
                # from the agent's output.
                from agent13.debug_log import log_error

                log_error(
                    e,
                    {
                        "context": "_process_tokens",
                        "message_type": type(message).__name__,
                    },
                )
                # Try to show error in chat so user knows something went wrong
                try:
                    await self._write_system(
                        f"[red]Display error: {escape_markup(str(e))}[/]",
                        escape_text=False,
                    )
                except Exception:
                    pass  # If we can't even write the error, just continue

    async def _handle_token(self, message: TokenMessage) -> None:
        """Process a single token message sequentially."""
        if message.is_final:
            # Streaming complete
            if is_debug_enabled():
                log_tps_stream_end(
                    reason="is_final",
                    first_token_time=self.agent.tracker._first_token_time,
                    last_token_time=self.agent.tracker._last_token_time,
                    token_count=self.agent.tracker._token_count,
                    pending_token_usage=None,
                )
            self._streaming = False
            self.update_status()
            await self._finalize_streaming()
        else:
            # Track timing for TPS calculation
            now = time.time()
            if self.agent.tracker.is_first_token:
                if is_debug_enabled():
                    log_tps_first_token(token_count=self.agent.tracker._token_count)
            self.agent.tracker.record_token(now)

            # Visible tokens are flowing again: any "⚙ …" placeholder mounted
            # during a silent stretch was a false alarm (the model was just
            # thinking slowly, not preparing a buffered tool call). Remove it
            # before the content/reasoning widget mounts.
            if self._pending_tool_widgets:
                await self._discard_pending_tool_widgets()

            was_at_bottom = self._is_at_bottom()

            if message.is_reasoning:
                # Determine title based on source (reflection vs normal thinking)
                title = "Reflecting" if message.source == "reflection" else "Thinking"
                # Start reasoning widget if not already in reasoning phase
                if not self._in_reasoning:
                    self._in_reasoning = True
                    await self._start_reasoning_stream(
                        was_at_bottom, message.text, title
                    )
                else:
                    # Append to reasoning widget
                    await self._append_to_reasoning(message.text, title)
            else:
                # End reasoning phase if we were in one
                if self._in_reasoning:
                    self._in_reasoning = False
                    await self._finalize_reasoning()

                # Start content widget if not already started
                if self._streaming_content_widget is None:
                    await self._start_content_stream(was_at_bottom, message.text)
                    # Mark that we're now streaming
                    self._streaming = True
                    self.update_status()
                    # Clear the flag since we've created a new streaming message
                    self._finalize_before_tool = False
                elif not self._streaming:
                    # Widget exists but stream ended (error path: ASSISTANT_COMPLETE
                    # was never emitted). Finalize the stale widget and start fresh.
                    await self._streaming_content_widget.finalize()
                    self._streaming_content_widget = None
                    await self._start_content_stream(was_at_bottom, message.text)
                    self._streaming = True
                    self.update_status()
                    self._finalize_before_tool = False
                else:
                    # Append to content widget
                    await self._append_to_content(message.text)

    async def _handle_tool_call(self, message: ToolCallMessage) -> None:
        """Process a tool call message: finalize streaming, then write tool call.

        If a matching pending widget exists (created at TOOL_CALL_STARTED when
        only the name was known), update it in place to add the args. Otherwise
        write a fresh widget (e.g. if the started event was missed).

        Handles the edge case where a tool call arrives with no preceding content
        (the LLM "thought" silently and then called a tool). In this case, we
        skip creating an empty "Agent:" widget.
        """
        # Only finalize if there's actually something to finalize
        # This prevents empty "Agent:" widgets when tool calls arrive
        # after a stream with zero content tokens
        if self._streaming_content_widget or self._in_reasoning:
            await self._finalize_streaming_for_tool()
        else:
            # No content was ever shown, just set the flag
            self._finalize_before_tool = True

        # Reuse the pending name-only widget if one is waiting for this call.
        # FIFO match: started events and complete events arrive in the same order.
        if self._pending_tool_widgets:
            widget = self._pending_tool_widgets.pop(0)
            widget.set_call(message.name, message.arguments, message.seq)
            return

        await self._write_tool_call(message.name, message.arguments, message.seq)

    async def _handle_tool_call_started(self, message: ToolCallStartedMessage) -> None:
        """Process a tool-call-started message: show the tool name immediately.

        The model has committed to a tool (name known) but its arguments are still
        streaming. Mount a dimmed name-only widget now so the user gets feedback
        during (potentially long) argument generation; the matching TOOL_CALL
        later updates it in place with the full args.

        If a TOOL_CALL_PENDING placeholder is already mounted (provider buffers
        tool calls, so the name only became known now), update it in place
        instead of mounting a second widget.
        """
        # A tool call means no more pre-tool content is coming for this turn, so
        # finalize any streaming widget (same as _handle_tool_call).
        if self._streaming_content_widget or self._in_reasoning:
            await self._finalize_streaming_for_tool()
        else:
            self._finalize_before_tool = True

        # Upgrade an existing "⚙ …" placeholder to the named pending widget.
        # Only a name-unknown placeholder (from TOOL_CALL_PENDING) is upgraded
        # in place; a named pending widget from an earlier TOOL_CALL_STARTED
        # (parallel tool calls) must not be overwritten - mount a new one.
        if self._pending_tool_widgets:
            last = self._pending_tool_widgets[-1]
            if last.pending_unknown:
                last.set_pending_name(message.name)
                return

        was_at_bottom = self._is_at_bottom()
        widget = ToolWidget(
            message.name,
            collapsed=self.agent.toolview_collapsed,
            kind="call",
            pending=True,
        )
        widget.add_class("tool-call")
        if await self._safe_mount(widget):
            self._pending_tool_widgets.append(widget)
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _handle_tool_call_pending(self, message: ToolCallPendingMessage) -> None:
        """Process a tool-call-pending message: show a name-unknown placeholder.

        The provider is buffering a tool call: the stream is alive (keep-alive
        chunks) but nothing visible is being emitted, and the tool name won't be
        known until generation finishes. Mount a dimmed "⚙ …" placeholder so the
        user sees the agent is preparing a tool call. TOOL_CALL_STARTED or
        TOOL_CALL later replaces it in place; if content arrives instead, the
        placeholder is discarded (false alarm - it was silent thinking).
        """
        # Already showing a pending/placeholder widget - nothing to add.
        if self._pending_tool_widgets:
            return

        was_at_bottom = self._is_at_bottom()
        widget = ToolWidget(
            "",
            collapsed=self.agent.toolview_collapsed,
            kind="call",
            pending=True,
            pending_unknown=True,
        )
        widget.add_class("tool-call")
        if await self._safe_mount(widget):
            self._pending_tool_widgets.append(widget)
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _discard_pending_tool_widgets(self) -> None:
        """Remove any orphaned pending tool-call widgets from the chat.

        Called at turn boundaries (new item / interrupt) to clean up widgets that
        were shown at TOOL_CALL_STARTED but never completed (e.g. the stream
        errored after the tool name but before the arguments finished).
        """
        for widget in self._pending_tool_widgets:
            try:
                if widget.is_mounted:
                    await widget.remove()
            except Exception:
                pass
        self._pending_tool_widgets = []

    async def _handle_tool_result(self, message: ToolResultMessage) -> None:
        """Process a tool result message: write the result widget."""
        # Log tool result with timing state (stream will resume after this)
        if is_debug_enabled():
            log_tps_event(
                "tool_result",
                {
                    "tool_name": message.name,
                    "first_token_time": self.agent.tracker._first_token_time,
                    "last_token_time": self.agent.tracker._last_token_time,
                    "token_count": self.agent.tracker._token_count,
                },
            )
        await self._write_tool_result(message.name, message.result, message.seq)

    async def _start_reasoning_stream(
        self, was_at_bottom: bool, first_token: str, title: str = "Thinking"
    ) -> None:
        """Create and mount reasoning widget with first token.

        The widget is created with title pre-set to avoid race conditions.
        """
        # Only create widget if there's actual content to show
        if not first_token or not first_token.strip():
            # No meaningful content yet, don't create widget
            return
        # Widget is created with title pre-rendered
        self._streaming_reasoning_widget = ReasoningMessage(
            title, collapsed=self.agent.reasoning_collapsed
        )
        self._streaming_reasoning_widget.add_class("reasoning-message")
        if not await self._safe_mount(self._streaming_reasoning_widget):
            self._streaming_reasoning_widget = None
            return
        await self._scroll_if_at_bottom(was_at_bottom)
        # Just append the first token - title is already in place
        await self._streaming_reasoning_widget.append(first_token)

    async def _append_to_reasoning(self, text: str, title: str = "Thinking") -> None:
        """Append text to reasoning widget."""
        # If widget doesn't exist yet and we have meaningful content, create it
        if not self._streaming_reasoning_widget and (text and text.strip()):
            # First meaningful token - create the widget with title pre-set
            was_at_bottom = self._is_at_bottom()
            self._streaming_reasoning_widget = ReasoningMessage(
                title, collapsed=self.agent.reasoning_collapsed
            )
            self._streaming_reasoning_widget.add_class("reasoning-message")
            if not await self._safe_mount(self._streaming_reasoning_widget):
                self._streaming_reasoning_widget = None
                return
            await self._scroll_if_at_bottom(was_at_bottom)
            # Just append the token - title is already in place
            await self._streaming_reasoning_widget.append(text)
        elif self._streaming_reasoning_widget:
            # Widget exists, just append
            was_at_bottom = self._is_at_bottom()
            await self._streaming_reasoning_widget.append(text)
            if was_at_bottom:
                self._chat.scroll_end(animate=False)
                self._chat.anchor()

    async def _finalize_reasoning(self) -> None:
        """Finalize reasoning widget."""
        if self._streaming_reasoning_widget:
            # finalize() renames the header to its completed tense
            # ("Thinking" -> "Thoughts", "Reflecting" -> "Reflection")
            await self._streaming_reasoning_widget.finalize()
            # Remove widget if it has no meaningful content (title is pre-rendered)
            content = getattr(self._streaming_reasoning_widget, "_content", "")
            if not content or not content.strip():
                # Remove the empty reasoning widget
                await self._streaming_reasoning_widget.remove()
            self._streaming_reasoning_widget = None

    async def _start_content_stream(
        self, was_at_bottom: bool, first_token: str
    ) -> None:
        """Create and mount content widget."""
        # Widget is created with "Agent:" title pre-rendered to avoid race conditions
        self._streaming_content_widget = StreamingMessage(title="Agent")
        self._streaming_content_widget.add_class("assistant-message")
        if not await self._safe_mount(self._streaming_content_widget):
            self._streaming_content_widget = None
            return
        if was_at_bottom:
            self._chat.anchor()
        # No _streaming_ready event needed — sequential processing guarantees
        # mount completes before any _append_to_content call
        await self._streaming_content_widget.append(first_token)

    async def _append_to_content(self, text: str) -> None:
        """Append text to content widget."""
        if not self._streaming_content_widget:
            return

        # No _streaming_ready wait needed — sequential processing guarantees
        # the widget is mounted before we get here
        was_at_bottom = self._is_at_bottom()
        await self._streaming_content_widget.append(text)
        if was_at_bottom:
            self._chat.scroll_end(animate=False)
            self._chat.anchor()

    async def _finalize_streaming_for_tool(self) -> None:
        """Finalize streaming widgets before tool call, preparing for new widgets after."""
        # Log timing state when tool call interrupts streaming
        if is_debug_enabled():
            log_tps_stream_end(
                reason="tool_call",
                first_token_time=self.agent.tracker._first_token_time,
                last_token_time=self.agent.tracker._last_token_time,
                token_count=self.agent.tracker._token_count,
                pending_token_usage=None,
            )
        # Finalize reasoning if still open
        if self._in_reasoning:
            self._in_reasoning = False
            await self._finalize_reasoning()

        # Finalize content widget
        if self._streaming_content_widget:
            await self._streaming_content_widget.finalize()
            self._streaming_content_widget = None

        # Set flag so we know to create new widgets when tokens resume
        self._finalize_before_tool = True

    async def _finalize_streaming(self) -> None:
        """Finalize all streaming widgets."""
        # Finalize reasoning if still open
        if self._in_reasoning:
            self._in_reasoning = False
            await self._finalize_reasoning()

        # Finalize content widget
        if self._streaming_content_widget:
            await self._streaming_content_widget.finalize()
            self._streaming_content_widget = None

        # Clear the tool flag
        self._finalize_before_tool = False
        # Scroll to bottom after streaming completes
        was_at_bottom = self._is_at_bottom()
        if was_at_bottom:
            self._chat.scroll_end(animate=False)
            self._chat.anchor()

    def on_tool_result_message(self, message: ToolResultMessage) -> None:
        """Enqueue tool result for sequential processing.

        Tool results must go through _token_queue to guarantee they render
        AFTER the corresponding tool call. Previously this used create_task
        which could race with the queued tool call, causing results to appear
        before their tool calls in the chat window.
        """
        self._token_queue.put_nowait(message)

    def on_system_queue_message(self, message: SystemQueueMessage) -> None:
        """Enqueue system message for ordered sequential processing.

        System messages (e.g. "Agent paused") must go through _token_queue
        via post_message first, so they preserve ordering with tool results.
        Both use the same two-stage pipeline: post_message → _token_queue.
        """
        self._token_queue.put_nowait(message)

    def on_error_message(self, message: ErrorMessage) -> None:
        """Handle error messages."""
        asyncio.create_task(self._write_error(message.message, message.error_type))

    def on_notification_message(self, message: NotificationMessage) -> None:
        """Handle notification messages."""
        asyncio.create_task(
            self._show_notification(message.message, message.duration, message.level)
        )

    def on_key(self, event) -> None:
        """Handle key events for tab completion."""
        input_field = self.query_one("#input-field", ChatTextArea)
        current_text = input_field.text

        # Tab/Shift+Tab completion - cycle through matches
        if event.key in ("tab", "shift+tab"):
            event.stop()  # Prevent default tab behavior (indent)

            # Determine direction: Shift+Tab goes backward, Tab goes forward
            reverse = event.key == "shift+tab"

            # If already in cycling mode and text unchanged (only our completion
            # This prevents spaces in completed items (e.g. filenames with
            # spaces) from invalidating the context on the next tab press.
            if self._completion_matches and self._completion_text == current_text:
                # Still cycling — use stored start_loc for consistency check
                start_loc = self._completion_start
            else:
                # Not cycling, or text changed — re-evaluate context
                cursor_row, cursor_col = input_field.cursor_location
                completion_type, start_loc, end_loc = self._get_completion_context(
                    current_text, cursor_row, cursor_col
                )

                # If in "none" context, clear any stale completion state and exit
                # This prevents Tab from cycling with stale matches after user has moved on
                if completion_type == "none":
                    if self._completion_matches:
                        self._reset_completion_state()
                        # Only hide info-pane if completions were shown there
                        if self._info_pane_mode is None:
                            self._hide_info_pane()
                    # Re-focus input and exit (Tab does nothing in "none" context)
                    self.call_later(input_field.focus)
                    return

                # Text changed while in completion mode — reset and start fresh
                # This is needed because TextArea stops key events from bubbling,
                # so we don't receive space/character events to reset completion state
                if self._completion_matches and self._completion_text != current_text:
                    self._reset_completion_state()
                    # Only hide info-pane if completions were shown there
                    if self._info_pane_mode is None:
                        self._hide_info_pane()

            # If we're still in completion mode (text unchanged), cycle
            if self._completion_matches and start_loc == self._completion_start:
                if reverse:
                    self._completion_index = (self._completion_index - 1) % len(
                        self._completion_matches
                    )
                else:
                    self._completion_index = (self._completion_index + 1) % len(
                        self._completion_matches
                    )
                match = self._completion_matches[self._completion_index]
                self._apply_completion(input_field, match)
                self._completion_text = input_field.text  # Update saved text
                self._show_completions(self._completion_matches, match)
            else:
                # Start new completion
                # Extract the partial text to complete
                start_row, start_col = start_loc
                end_row, end_col = end_loc

                # Get partial text from the relevant line(s)
                lines = current_text.split("\n")
                if start_row == end_row:
                    partial = lines[start_row][start_col:end_col]
                else:
                    # Multi-line completion (rare but handle it)
                    partial = lines[start_row][start_col:]
                    for r in range(start_row + 1, end_row):
                        partial += "\n" + lines[r]
                    partial += "\n" + lines[end_row][:end_col]

                self._start_completion(
                    input_field, completion_type, partial, current_text, start_loc, end_loc
                )

            # Re-focus input after Tab handling
            self.call_later(input_field.focus)
            return

        # Up/Down in completion mode - navigate through matches
        if event.key == "up":
            if self._completion_matches:
                event.stop()
                self._completion_index = (self._completion_index - 1) % len(
                    self._completion_matches
                )
                match = self._completion_matches[self._completion_index]
                self._apply_completion(input_field, match)
                self._show_completions(self._completion_matches, match)
                self.call_later(input_field.focus)
            # Without completion mode, Up/Down are handled by TextArea (cursor movement)
            return

        if event.key == "down":
            if self._completion_matches:
                event.stop()
                self._completion_index = (self._completion_index + 1) % len(
                    self._completion_matches
                )
                match = self._completion_matches[self._completion_index]
                self._apply_completion(input_field, match)
                self._show_completions(self._completion_matches, match)
                self.call_later(input_field.focus)
            # Without completion mode, Up/Down are handled by TextArea (cursor movement)
            return

        # Right arrow at the end of a directory path opens it. This is the way
        # OUT of a live completion cycle: Tab cycles the level and never
        # descends, so without this the only escape is an edit (any key ends
        # the cycle) - a trick nobody can discover.
        if event.key == "right":
            descend = self._descend_target(current_text, input_field.cursor_location)
            if descend is not None:
                event.stop()
                partial, start_loc, end_loc = descend
                self._reset_completion_state()
                self._start_completion(
                    input_field, "@filename", partial, current_text, start_loc, end_loc
                )
                self.call_later(input_field.focus)
                return

        # Escape is handled by action_interrupt binding
        # No need to handle here - the binding with priority=True catches it first

        # Any other key cancels completion mode
        if (
            event.key not in ("tab", "shift+tab", "escape", "up", "down")
            and self._completion_matches
        ):
            self._reset_completion_state()
            # Only hide info-pane if completions were shown there
            if self._info_pane_mode is None:
                self._hide_info_pane()

        # Enter triggers submission via ChatTextArea.action_submit()
        # Ctrl+J inserts newline via ChatTextArea.action_insert_newline()
        # No special handling needed here

    def _descend_target(
        self, text: str, cursor: tuple[int, int]
    ) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
        """Return the @filename context if the cursor sits at the end of a
        directory path - the `→` gesture that opens it.

        Requires the cursor at the END OF THE LINE (mid-line `→` is ordinary
        cursor movement) and the typed path to name a directory.
        """
        row, col = cursor
        lines = text.split("\n")
        if row >= len(lines) or col != len(lines[row]):
            return None
        completion_type, start_loc, end_loc = self._get_completion_context(
            text, row, col
        )
        if completion_type != "@filename" or start_loc[0] != end_loc[0]:
            return None
        partial = lines[start_loc[0]][start_loc[1] : end_loc[1]]
        if not partial.endswith("/") and not os.path.isdir(
            os.path.expanduser(partial[1:])
        ):
            return None
        return partial, start_loc, end_loc

    def _start_completion(
        self,
        input_field: ChatTextArea,
        completion_type: str,
        partial: str,
        current_text: str,
        start_loc: tuple[int, int],
        end_loc: tuple[int, int],
    ) -> None:
        """Complete `partial` and enter cycling mode.

        The cycle list is the whole directory level for @filename (so a unique
        match cannot trap the user - Tab keeps moving through the siblings),
        and the match list for every other context. A partial that is already
        a complete entry leaves Tab inert: see plan_completion.
        """
        matches = self._get_completions_for_context(completion_type, partial, current_text)
        level = []
        if completion_type == "@filename":
            level = filename_completion_scan(partial)[1]
        plan = plan_completion(matches, level, partial)
        if plan is None:
            # Nothing to complete - never replace what the user typed in full
            self._reset_completion_state()
            if self._info_pane_mode is None:
                self._hide_info_pane()
            return

        cycle, index = plan
        self._completion_matches = cycle
        self._completion_index = index
        self._completion_prefix = partial
        self._completion_start = start_loc
        self._completion_end = end_loc
        self._apply_completion(input_field, matches[0], start_loc, end_loc)
        self._completion_text = input_field.text  # For change detection
        self._show_completions(cycle, matches[0])

    def _apply_completion(
        self,
        input_field: ChatTextArea,
        completion: str,
        start_loc: tuple[int, int] | None = None,
        end_loc: tuple[int, int] | None = None,
        update_state: bool = True,
    ) -> None:
        """Apply a completion by replacing text at the specified location.

        Args:
            input_field: The ChatTextArea widget
            completion: The completion text to insert
            start_loc: (row, col) where to start replacement, or None to use stored state
            end_loc: (row, col) where to end replacement, or None to use stored state
            update_state: If True, update _completion_end to the new end position (for cycling)
        """
        if start_loc is None:
            start_loc = self._completion_start
        if end_loc is None:
            end_loc = self._completion_end

        # Use TextArea's replace method for partial replacement
        input_field.replace(completion, start_loc, end_loc)

        # Calculate new cursor position (end of completion)
        new_cursor_col = start_loc[1] + len(completion)
        # Handle multi-line completions if needed
        if "\n" in completion:
            lines = completion.split("\n")
            new_cursor_row = start_loc[0] + len(lines) - 1
            new_cursor_col = len(lines[-1])
            input_field.move_cursor((new_cursor_row, new_cursor_col))
            if update_state:
                self._completion_end = (new_cursor_row, new_cursor_col)
        else:
            input_field.move_cursor((start_loc[0], new_cursor_col))
            if update_state:
                self._completion_end = (start_loc[0], new_cursor_col)
        # NOTE: action_scroll_chat_up/down removed — shift+up/down doesn't
        # survive terminal → ssh → tmux → Textual chain. Use mouse scroll.
        # def action_scroll_chat_up(self) -> None:
        #     """Scroll up and disable auto-scroll."""
        #     self._chat.scroll_relative(y=-5, animate=False)
        #     self._chat.anchor(False)  # Explicitly clear anchor
        #     self.notify("Auto-scroll disabled")
        #
        # def action_scroll_chat_down(self) -> None:
        #     """Scroll down and re-enable auto-scroll if at bottom."""
        #     self._chat.scroll_relative(y=5, animate=False)
        #     if self._is_at_bottom():
        #         self._chat.anchor()
        #         self.notify("Auto-scroll enabled")

    def action_toggle_collapsed(self) -> None:
        """Toggle collapse state of most recent reasoning widget (Ctrl+O)."""
        # Find the most recent ReasoningMessage widget
        try:
            widgets = list(self.query(ReasoningMessage))
            if widgets:
                last_widget = widgets[-1]
                # Toggle its collapsed state
                asyncio.create_task(
                    last_widget.set_collapsed(not last_widget.collapsed)
                )
        except Exception:
            pass  # Silently ignore if no widgets found

    def action_close_info(self) -> None:
        """Close the info pane."""
        self._hide_info_pane()

    def action_interrupt(self) -> None:
        """Handle ESC key - cancel the running agent task."""
        # Only cancel if actively processing a message (not just waiting idle)
        if self.processing and not self._interrupt_requested:
            self._interrupt_requested = True
            self.run_worker(self._interrupt_agent_loop())

    def copy_to_clipboard(self, text: str) -> None:
        """Override Textual's clipboard to respect the configured method.

        Uses the [clipboard] method from config (default: osc52).
        Falls back to Textual's OSC 52 for "osc52" mode,
        or uses OS-level subprocess for "system" mode.
        """
        from agent13.clipboard import copy_to_clipboard as _copy

        method = self.agent.clipboard_method
        if method == "system":
            if not _copy(text, method="system"):
                self._write_error(
                    "Clipboard (system) failed. "
                    "Install xclip, wl-copy, or try /clipboard osc52"
                )
        else:
            # OSC 52 — delegate to Textual's built-in
            _copy(text, method="osc52", osc52_handler=super().copy_to_clipboard)

    # NOTE: action_copy_selection removed — ctrl+shift+c doesn't survive
    # terminal → ssh → tmux → Textual chain. Use Ctrl+Y instead.
    # def action_copy_selection(self) -> None:
    #     """Copy rendered selection to clipboard (Ctrl+Shift+C).
    #
    #     Copies the selected rendered text and keeps the selection visible.
    #     """
    #     # Check chat area selection
    #     text = self.screen.get_selected_text()
    #     if text:
    #         self.copy_to_clipboard(text)
    #         self.notify(f"Copied {len(text)} chars", title="Copied")
    #         # Keep selection visible
    #         return
    #
    #     # Check input field selection
    #     input_field = self.query_one("#input-field", ChatTextArea)
    #     if input_field.selected_text:
    #         self.copy_to_clipboard(input_field.selected_text)
    #         self.notify(
    #             f"Copied {len(input_field.selected_text)} chars", title="Copied"
    #         )

    def action_copy_as_markdown(self) -> None:
        """Copy full markdown of message containing selection (Ctrl+Y).

        When you select part of a message and press Ctrl+Y, this copies the
        entire raw markdown source of that message, not just the selection.
        Useful for copying assistant responses with formatting intact.
        """
        # Find widget with selection in chat area
        seen_ids: set[int] = set()
        markdown_parts: list[str] = []
        for widget in self.query("*"):
            if not hasattr(widget, "text_selection") or not widget.text_selection:
                continue

            # Collect markdown from all messages containing selected text.
            # Hierarchy: StreamingMessage > Markdown > MarkdownParagraph
            # Selection can span multiple paragraphs/messages.
            # Walk up from each selected widget to find _content on the parent message.
            node = widget
            while node is not None:
                if hasattr(node, "_content") and isinstance(node._content, str):
                    widget_id = id(node)
                    if widget_id not in seen_ids:
                        seen_ids.add(widget_id)
                        markdown_parts.append(node._content)
                    break
                node = node.parent

        if markdown_parts:
            content = "\n\n".join(markdown_parts)
            self.copy_to_clipboard(content)
            self.notify(f"Copied markdown ({len(content)} chars)", title="Copied")
            return

        # No selection in chat - check input field
        input_field = self.query_one("#input-field", ChatTextArea)
        if input_field.selected_text:
            self.copy_to_clipboard(input_field.selected_text)
            self.notify(
                f"Copied {len(input_field.selected_text)} chars", title="Copied"
            )

    async def _interrupt_agent_loop(self) -> None:
        """Cancel the running agent task."""
        if not self.processing or not self._agent_task:
            return

        # ESC during polite wait: pause instead of cancel-and-restart.
        # The queue item stays pending so the user can inspect/edit/delete
        # it via /queue or /delete. Use /resume to continue (re-enters
        # the polite wait) or /delete q N to remove it.
        if self._polite_wait_elapsed is not None:
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
                try:
                    await self._agent_task
                except asyncio.CancelledError:
                    pass  # Expected

            # Clear polite overlay and interrupt flag
            self._polite_wait_elapsed = None
            self._interrupt_requested = False

            # Restart the loop in paused state — item stays in queue.
            # _pause_on_start makes run() enter PAUSING on its first
            # iteration, so _wait_if_paused() blocks immediately.
            self.agent._pause_on_start = True
            self._agent_task = asyncio.create_task(self.agent.run())

            # Clear input field to prevent keystroke concatenation
            try:
                input_field = self.query_one("#input-field", ChatTextArea)
                if input_field.text:
                    input_field.clear()
                input_field.focus()
            except Exception:
                pass

            self._update_info_content(
                "[yellow]ESC — paused[/]\n"
                "Queue item retained.\n"
                "[green]/resume[/] to continue, "
                "[red]/delete q N[/] to remove."
            )
            return

        # Capture whether we're interrupting during tool execution,
        # before the cancel resets the agent state.
        was_tooling = self.agent.status == AgentStatus.TOOLING

        # Cancel the asyncio task
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass  # Expected - task was cancelled

        # Finalize any streaming widgets
        # Finalize reasoning widget if active
        if self._streaming_reasoning_widget:
            try:
                await self._streaming_reasoning_widget.finalize()
            except Exception:
                pass
            self._streaming_reasoning_widget = None

        # Finalize content widget if active
        if self._streaming_content_widget:
            try:
                await self._streaming_content_widget.finalize()
            except Exception:
                pass
            self._streaming_content_widget = None

        # Reset reasoning state
        self._in_reasoning = False

        # Increment stream generation to discard any stale tokens
        self._stream_generation += 1

        # Complete the current queue item to prevent it from being processed again
        self.agent.queue.complete_current()

        # Reset state
        self._interrupt_requested = False
        self._interrupt_available = True  # Allow /resume to continue
        # Clear polite-waiting overlay (ESC during polite wait cancels the
        # acquire; POLITE_ACQUIRED never fires so we must clear manually).
        self._polite_wait_elapsed = None

        # Ensure agent isn't stuck in paused state (e.g. from error path
        # setting pause_state=PAUSED before the interrupt). Escape means
        # "stop and redirect", not "pause to resume later".
        if self.agent.is_paused or self.agent.is_pausing:
            self.agent.resume()
        self.update_status()  # Refresh status bar after clearing pause state

        # Clear input field to prevent keystroke concatenation (Bug #5)
        try:
            input_field = self.query_one("#input-field", ChatTextArea)
            if input_field.text:
                input_field.clear()
            input_field.focus()
        except Exception:
            pass

        # Stop the old agent loop so it exits cleanly (sets _running = False).
        # The CancelledError was already absorbed in _process_item; this
        # ensures the while-loop condition fails and the finally block
        # runs with the correct MCP cleanup path.
        self.agent.stop()

        # Restart the agent loop so it can process future messages
        self._agent_task = asyncio.create_task(self.agent.run())

        # Show interrupt message — immediate, prominent yellow panel
        await self._write_interrupt("ESC pressed")

        # Notify if we were in TOOLING state — the tool may still be running
        # in the background (Python threads can't be killed; see known
        # limitation comment in core.py _llm_turn tool loop).
        if was_tooling:
            self.notify(
                "Tool may still be running in background",
                title="Interrupted during tool execution",
                severity="warning",
            )

    def action_clear_quit(self) -> None:
        """Handle Ctrl+C: clear input, interrupt agent, or quit."""
        input_field = self.query_one("#input-field", ChatTextArea)

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
            self.run_worker(self._interrupt_agent_loop())
            return

        # If empty, set flag and quit
        global _ctrl_c_pressed
        _ctrl_c_pressed = True
        self.action_force_quit()

    def action_ctrl_c(self) -> None:
        """Handle Ctrl+C - set flag for atexit message and exit."""
        global _ctrl_c_pressed
        _ctrl_c_pressed = True
        self.exit()

    def action_force_quit(self) -> None:
        """Quit the application with clean shutdown via /quit command."""
        # Stop the agent first
        self.agent.stop(StopReason.QUIT)
        # Cancel the agent task if running
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        # Exit cleanly without the keyboard interrupt message
        self.exit()


def print_provider_list():
    """Print available providers from config."""
    config = get_config()
    if not config.providers:
        print("No providers configured in ~/.agent13/config.toml")
        return

    print("\nAvailable providers:")
    for provider in config.providers:
        key_status = (
            f" (key: {provider.api_key_env_var})"
            if provider.api_key_env_var
            else " (no key required)"
        )
        print(f"  {provider.name}{key_status}")
        print(f"    {provider.api_base}")
    print()


async def async_main():
    """Async main entry point - handles setup, then returns app for sync run."""

    parser = argparse.ArgumentParser(
        description="TUI agent for OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ./tui.py studio
  ./tui.py openrouter --model devstral
  ./tui.py http://localhost:8012/v1
  ./tui.py --model devstral studio

Provider names are read from ~/.agent13/config.toml
""",
    )
    parser.add_argument(
        "provider", nargs="?", help="Provider name from config or OpenAI-compatible URL"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers from config and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Model to select: number (1, 2, ...) or name. With no value, lists available models and exits",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to use (name from prompt manager)",
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        help="Sandbox mode for bash tool (permissive-open, permissive-closed, restrictive-open, restrictive-closed, none)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--tool-response",
        choices=["raw", "json"],
        default="raw",
        help="Tool response format: 'raw' (default) or 'json'",
    )
    parser.add_argument(
        "--mcp", action="store_true", help="Connect to MCP servers on startup"
    )
    parser.add_argument(
        "--skills",
        action="store_true",
        help="Include discovered skills in the system prompt",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Enable journal mode (context compaction via reflection)",
    )

    args = parser.parse_args()

    # Initialize debug logging if --debug flag is set
    if args.debug:
        init_debug()

    # Handle --list-providers flag (doesn't require provider argument)
    if args.list_providers:
        print_provider_list()
        sys.exit(0)

    # Provider is required for all other operations
    if not args.provider:
        parser.error(
            "provider argument is required (use --list-providers to see available providers)"
        )

    # Resolve provider
    try:
        base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
            args.provider
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine provider name for display (only if not a URL)
    provider_name = (
        ""
        if args.provider.startswith("http://") or args.provider.startswith("https://")
        else args.provider
    )

    # Initialize client
    client = create_client(
        base_url, api_key, read_timeout=read_timeout, connect_timeout=connect_timeout
    )

    # Fetch models
    try:
        model_names = await fetch_models(client)
    except RuntimeError as e:
        err_msg = str(e)
        is_connection_error = any(
            s in err_msg.lower()
            for s in [
                "connection error",
                "connection refused",
                "could not resolve",
                "timed out",
                "name or service not known",
            ]
        )

        if is_connection_error:
            # Provider is unreachable - no point continuing
            print(f"Error: Provider is unreachable: {e}", file=sys.stderr)
            sys.exit(1)
        elif args.model == "":
            # --model with no value means "list models" - can't do that without fetching
            print(f"Error: Could not fetch models: {e}", file=sys.stderr)
            sys.exit(1)
        else:
            # Non-connection error (e.g. /models endpoint broken) - try proceeding with specified model
            print(f"Warning: Could not fetch models: {e}", file=sys.stderr)
            model_names = []

    # Handle --model with no value: list models and exit
    if args.model == "":
        print_model_list(model_names)
        sys.exit(0)

    # Select model - if model list unavailable and user specified a model, use it directly
    if not model_names and args.model:
        model = args.model
        print(
            f"Using model '{model}' (model list unavailable, using specified name directly)",
            file=sys.stderr,
        )
    else:
        model = await select_model(model_names, args.model)

    # Initialize prompt manager and set active prompt if specified
    try:
        prompt_manager = PromptManager()
    except Exception as e:
        print(f"Error: Failed to load prompts: {e}", file=sys.stderr)
        sys.exit(1)
    if args.system_prompt:
        if not prompt_manager.set_active(args.system_prompt):
            print(f"Error: Prompt '{args.system_prompt}' not found", file=sys.stderr)
            print(
                f"Available prompts: {', '.join(prompt_manager.prompts)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Set sandbox mode if specified
    if args.sandbox:
        try:
            sandbox_mode = parse_sandbox_mode(args.sandbox)
            set_session_sandbox_mode(sandbox_mode)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # No explicit flag — check for a pinned mode for this project
        pinned = get_pinned_sandbox_mode()
        if pinned is not None:
            set_session_sandbox_mode(pinned)

    # Create skill manager
    skill_manager = SkillManager(lambda: get_config())

    # Build system prompt (optionally with skills)
    system_prompt = prompt_manager.get_prompt()
    include_skills = args.skills or get_config().include_skills
    if include_skills and skill_manager.skills:
        skills_section = get_skills_section(skill_manager.skills)
        if skills_section:
            system_prompt = f"{system_prompt}\n\n{skills_section}"

    # Return app and config for sync run
    return AgentTUI(
        client=client,
        model=model,
        model_names=model_names,
        provider=provider_name,
        debug=args.debug,
        tool_response_format=args.tool_response,
        prompt_manager=prompt_manager,
        connect_mcp=args.mcp,
        skill_manager=skill_manager,
        system_prompt=system_prompt,
        include_skills=include_skills,
        journal_mode=args.journal,
    )


def main():
    """Main entry point."""
    global _ctrl_c_pressed
    _ctrl_c_pressed = False

    try:
        # Run async setup, then run app synchronously
        app = asyncio.run(async_main())
        app.run()
        # After TUI exits, check if Ctrl+C was pressed
        if _ctrl_c_pressed:
            log_session_end()
            print("\nExiting on keyboard interrupt", flush=True)
    except KeyboardInterrupt:
        # Fallback: Should not normally reach here as Textual handles Ctrl+C,
        # but catch it just in case
        log_session_end()
        print("\nExiting on keyboard interrupt", flush=True)
        sys.exit(0)
    except EOFError:
        # Clean exit on Ctrl+D - no interrupt message
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
