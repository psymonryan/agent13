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

from dotenv import load_dotenv
from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widgets import Static, Markdown
from textual.widgets._markdown import MarkdownStream
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual.events import MouseUp
from textual.widget import MountError, Widget
from ui.chat_input import ChatTextArea
import time
import argparse

# Re-export from shared module for backwards compatibility
from agent13.models import fetch_models, print_model_list, select_model


from agent13 import (
    Agent,
    AgentEvent,
    AgentEventData,
    AgentStatus,
    PauseState,
    SpinnerSpeed,
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
from agent13.persistence import (
    save_context,
    load_context,
    get_saves_dir,
    find_latest_auto_save,
    list_saves,
)
from agent13.prompts import get_skills_section
from agent13.sandbox import (
    format_all_sandbox_modes,
    parse_sandbox_mode,
    get_default_sandbox_mode,
    validate_sandbox_profiles,
)
from tools.security import (
    set_session_sandbox_mode,
    get_session_sandbox_mode,
    get_current_sandbox_mode,
)
from ui.display import format_mcp_servers

# Track if exit was triggered by Ctrl+C to show message after TUI closes
_ctrl_c_pressed = False

# Custom escape function that escapes ALL brackets for Textual markup
# Both rich.markup.escape and textual.markup.escape only escape brackets
# that match specific patterns (e.g., [a-z]), but Textual's parser fails
# on brackets like [Showing...] with uppercase letters


def escape_markup(text: str) -> str:
    """Escape all brackets in text for safe use in Textual markup."""
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


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

    def __init__(self, name: str, arguments: dict) -> None:
        super().__init__()
        self.name = name
        self.arguments = arguments


class ToolResultMessage(Message):
    """Message sent when a tool result is received."""

    def __init__(self, name: str, result: str) -> None:
        super().__init__()
        self.name = name
        self.result = result


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


class ReasoningMessage(Static):
    """A message widget for reasoning/thinking content.

    Supports incremental streaming and renders as dim/italic markdown.
    Can be collapsed/expanded via click or Ctrl+O.
    """

    def __init__(self, title: str = "Thinking", collapsed: bool = False):
        super().__init__()
        self._title = title
        self._content = ""
        self.collapsed = collapsed
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None
        self._header: Static | None = None

    def compose(self) -> ComposeResult:
        # Header row with title and collapse indicator
        indicator = "▶" if self.collapsed else "▼"
        self._header = Static(
            f"[dim]{self._title}[/] {indicator}", classes="reasoning-header"
        )
        yield self._header
        # Markdown content (hidden when collapsed)
        self._markdown = Markdown("")
        self._markdown.display = not self.collapsed
        yield self._markdown

    async def on_click(self) -> None:
        """Toggle collapsed state on click."""
        await self.set_collapsed(not self.collapsed)

    async def set_collapsed(self, collapsed: bool) -> None:
        """Set collapsed state and update display."""
        self.collapsed = collapsed
        # Update indicator
        indicator = "▶" if collapsed else "▼"
        if self._header:
            self._header.update(f"[dim]{self._title}[/] {indicator}")
        # Show/hide content
        if self._markdown:
            self._markdown.display = not collapsed
            # If expanding and we have content, re-render it
            if not collapsed and self._content:
                self._markdown.update(self._content)

    async def append(self, text: str) -> None:
        """Append text to the reasoning message."""
        if not text:
            return
        self._content += text
        # Only write to stream if not collapsed
        if self._markdown and not self.collapsed:
            if self._stream is None:
                self._stream = Markdown.get_stream(self._markdown)
            await self._stream.write(text)

    async def finalize(self) -> None:
        """Stop the stream when done."""
        if self._stream:
            await self._stream.stop()
            self._stream = None

    def set_title(self, title: str) -> None:
        """Update the widget title."""
        self._title = title
        # Update header if it exists
        if self._header:
            indicator = "▶" if self.collapsed else "▼"
            self._header.update(f"[dim]{self._title}[/] {indicator}")


class InterruptMessage(Static):
    """Widget displayed when user interrupts the agent."""

    def __init__(self):
        super().__init__(
            "[yellow]⚠ Interrupted by user[/]", classes="system-message", markup=True
        )


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
        border: solid blue;
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
        border: none;
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
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False),
        Binding("shift+down", "scroll_chat_down", "Scroll Down", show=False),
        # Ctrl+Y: Copy full markdown of message containing selection
        Binding(
            "ctrl+y", "copy_as_markdown", "Copy markdown", show=False, priority=True
        ),
        # Ctrl+Shift+C: Manual copy of rendered selection
        Binding("ctrl+shift+c", "copy_selection", "Copy selection", show=False),
        # Ctrl+O: Toggle collapse of most recent reasoning widget
        Binding("ctrl+o", "toggle_collapsed", "Toggle Reasoning", show=False),
    ]

    # Auto-copy rendered text to clipboard on mouse up
    # Can be disabled by setting this to False
    autocopy_to_clipboard: bool = True

    # Available slash commands for tab completion
    # Note: Named SLASH_COMMANDS to avoid conflict with Textual's command palette
    # which expects app.COMMANDS to be a list of command provider classes
    # Built-in slash commands (base list, skills added in __init__)
    _BUILTIN_SLASH_COMMANDS = [
        "/help",
        "/quit",
        "/exit",
        "/clear",
        "/history",
        "/delete",
        "/model",
        "/list",
        "/tool-response",
        "/pretty",
        "/prompt",
        "/mcp",
        "/queue",
        "/pause",
        "/resume",
        "/retry",
        "/prioritise",
        "/deprioritise",
        "/sandbox",
        "/provider",
        "/tools",
        "/skills",
        "/journal",
        "/remove-reasoning",
        "/save",
        "/load",
        "/devel",
        "/snippet",
        "/spinner",
        "/upgrade",
        "/clipboard",
    ]
    # Class-level attribute for type checking (instance copy created in __init__)
    SLASH_COMMANDS = _BUILTIN_SLASH_COMMANDS

    # Parameter completers for slash commands
    # Values can be:
    #   - list: static list of completions
    #   - str starting with '_': method name to call
    #   - str: attribute name to get
    _PARAM_COMPLETERS = {
        "model": "_get_model_completions",  # Method to get model names
        "mcp": ["connect", "disconnect", "reload"],  # MCP subcommands
        "sandbox": None,  # Will use _get_sandbox_completions method
        "pretty": ["on", "off"],
        "tool-response": ["raw", "json"],
        "provider": "_get_provider_completions",  # Method to get provider names
        "prompt": "_get_prompt_completions",  # Method for subcommand handling
        "delete": "_get_delete_completions",  # Delete from history, queue, or saves
        "journal": ["on", "off", "last", "all", "status"],
        "remove-reasoning": ["on", "off"],
        "save": "_get_save_completions",  # Method to list save files
        "load": "_get_save_completions",  # Same method for load
        "devel": ["on", "off", "status"],
        "snippet": "_get_snippet_completions",
        "spinner": ["fast", "slow", "off", "status"],
        "upgrade": ["--copy"],
        "clipboard": ["osc52", "system"],
    }
    # Spinner animation frames by speed mode
    SPINNER_FRAMES_FAST = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]  # braille, 10 frames
    SPINNER_FRAMES_SLOW = ["-", "\\", "|", "/"]  # classic, 4 frames
    SPINNER_INTERVAL = 0.1  # 100ms per frame (fast mode)

    status = reactive("Ready")
    queue_count = reactive(0)
    processing = reactive(False)
    _streaming = reactive(False)  # True when receiving tokens from LLM
    _spinner_index = reactive(0)
    _spinner_speed: reactive = reactive("fast")  # "fast", "slow", or "off"

    # Token usage tracking
    prompt_tokens = reactive(0)
    completion_tokens = reactive(0)
    total_tokens = reactive(0)

    # TPS (tokens per second) tracking
    _last_tps = reactive(0.0)  # TPS from the most recent API call

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        model_names: list[str],
        provider: str = "",
        pretty: bool = True,
        debug: bool = False,
        tool_response_format: str = "raw",
        prompt_manager: PromptManager = None,
        connect_mcp: bool = False,
        skill_manager: SkillManager = None,
        system_prompt: str = None,
        journal_mode: bool = False,
        send_reasoning: bool = False,
        remove_reasoning: bool = False,
        continue_session: bool = False,
        devel_mode: bool = False,
        spinner_speed: str = "fast",
        clipboard_method: str = "osc52",
    ):
        """Initialize the TUI.

        Args:
            client: AsyncOpenAI client for API calls
            model: Model name to use
            model_names: List of available model names
            provider: Provider name for status bar display
            pretty: Enable pretty output (markdown rendering)
            debug: Enable debug mode
            tool_response_format: Tool response format ('raw' or 'json')
            prompt_manager: Optional pre-configured PromptManager
            connect_mcp: Connect to MCP servers on startup
            skill_manager: Optional skill manager for skill commands
            system_prompt: Optional system prompt (overrides prompt_manager if provided)
            journal_mode: Enable context compaction via reflection
            send_reasoning: Include reasoning_content in message history
            remove_reasoning: Strip reasoning tokens between turns
            continue_session: Load latest auto-save on startup
            devel_mode: Enable devel mode (show devel-group tools to the AI)
            spinner_speed: Spinner speed - 'fast' (100ms, braille), 'slow' (250ms, classic), or 'off'
            clipboard_method: Clipboard method - 'osc52' (terminal escape) or 'system' (OS commands)
        """
        super().__init__()
        self.client = client
        self.model = model
        self.model_names = model_names
        self.provider = provider
        self.pretty = pretty
        self._debug_mode = debug
        self.tool_response_format = tool_response_format
        self._connect_mcp = connect_mcp

        # Initialize prompt manager
        self.prompt_manager = prompt_manager or PromptManager()

        # Skill manager for skill slash commands
        self.skill_manager = skill_manager

        # Snippet manager for snippet slash commands
        reserved = {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}
        self.snippet_manager = SnippetManager(reserved_names=reserved)

        # Initialize slash commands (instance attribute for skill extensibility)
        self.SLASH_COMMANDS = list(self._BUILTIN_SLASH_COMMANDS)

        # Register skill commands
        if self.skill_manager:
            self._register_skill_commands()

        # Register snippet commands and show collision warnings
        self._register_snippet_commands()

        # Convert tool_response_format string to response_format dict
        response_format = (
            {"type": "json_object"} if tool_response_format == "json" else None
        )

        # Determine system prompt
        effective_system_prompt = (
            system_prompt if system_prompt else self.prompt_manager.get_prompt()
        )

        # Initialize agent with tools
        config = get_config()
        self.agent = Agent(
            client=client,
            model=model,
            system_prompt=effective_system_prompt,
            tools=get_filtered_tools(
                devel=devel_mode,
                enabled_tools=config.enabled_tools or None,
                disabled_tools=config.disabled_tools or None,
            ),
            execute_tool=execute_tool,
            response_format=response_format,
            journal_mode=journal_mode,
            send_reasoning=send_reasoning,
            remove_reasoning=remove_reasoning,
            devel_mode=devel_mode,
        )

        # Set MCP server configs for lazy initialization
        if config.mcp_servers:
            self.agent.set_mcp_servers(config.mcp_servers)

        # Load auto-save if continuing session
        self._continue_session = continue_session
        self._session_loaded = False  # Track if we loaded a session

        # Spinner speed (set reactive after super().__init__)
        self._spinner_speed = spinner_speed

        # Clipboard method (osc52 or system)
        self._clipboard_method = clipboard_method

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
        self._stream_generation: int = (
            0  # Incremented on each new streaming session to discard stale tokens
        )

        # Sequential token processor: guarantees ordering of all widget operations
        # by processing tokens one at a time from an async queue, eliminating
        # all race conditions caused by asyncio.create_task fire-and-forget.
        self._token_queue: asyncio.Queue[TokenMessage | ToolCallMessage | ToolResultMessage | SystemQueueMessage | None] = (
            asyncio.Queue()
        )
        self._token_processor_task: asyncio.Task | None = None
        self._agent_task: asyncio.Task | None = None
        self._shutting_down = False  # Set in on_unmount to prevent mount errors
        self._agent_started = asyncio.Event()
        self._agent_running = False  # Track if agent is actively processing
        self._interrupt_requested = False  # Prevent double-cancellation
        self._interrupt_available = (
            False  # /resume after ESC sends "Sorry, please continue"
        )
        # Note: pause state is read from agent.pause_state (single source of truth)
        self._error_state = False  # Track if agent stopped due to error
        self._last_error_type = None  # Type of last error

        # TPS tracking
        self._stream_start_time: float | None = (
            None  # When streaming started (for display timing)
        )
        self._first_token_time: float | None = (
            None  # When first token arrived (for TPS calculation)
        )
        self._last_token_time: float | None = (
            None  # When last token arrived (for TPS calculation)
        )
        self._token_count: int = 0  # Count of chunks received in current stream

        # Elapsed time tracking
        self._elapsed_start_time: float | None = (
            None  # When current processing session started
        )

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
        self._info_pane_mode: str | None = None  # None, "history", "queue", "help"

        # Register event handlers
        self._register_handlers()

    def _register_skill_commands(self) -> None:
        """Register slash commands for all skills."""
        if not self.skill_manager:
            return
        self._register_dynamic_commands(
            self.skill_manager.skills, "Skill"
        )

    def _register_snippet_commands(self) -> None:
        """Register slash commands for snippets and show collision warnings."""
        self._register_dynamic_commands(self.snippet_manager.snippets, "Snippet")
        # Show collision warnings from initial load
        for warning in self.snippet_manager._collisions:
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

            if len(matches) >= 50:  # Limit matches (windowing shows 8 at a time)
                break
        return matches

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
        completer_spec = self._PARAM_COMPLETERS.get(cmd)
        if completer_spec is None:
            # No completer defined - check for special cases
            if cmd == "sandbox":
                return self._get_sandbox_completions(partial)
            return []
        # Get the completer value
        if isinstance(completer_spec, list):
            # Static list - apply filtering below
            options = completer_spec
            skip_filter = False
        elif isinstance(completer_spec, str):
            if completer_spec.startswith("_"):
                # Method call - method does its own context-aware filtering
                method = getattr(self, completer_spec, None)
                if method:
                    options = method(partial, full_text)
                    skip_filter = True  # Method already filtered
                else:
                    return []
            else:
                # Attribute name
                options = getattr(self, completer_spec, [])
                skip_filter = False
        else:
            return []

        # Filter by partial (skip for method-based completers that already filtered)
        if skip_filter:
            return options[:20]  # Just limit results
        if not partial:
            return options[:20]  # Return first 20 if no partial

        partial_lower = partial.lower()
        return [opt for opt in options if opt.lower().startswith(partial_lower)][:20]

    def _get_sandbox_completions(self, partial: str) -> list[str]:
        """Get completions for sandbox modes."""
        from agent13.sandbox import SandboxMode

        modes = [mode.value for mode in SandboxMode]
        if not partial:
            return modes
        partial_lower = partial.lower()
        return [m for m in modes if m.lower().startswith(partial_lower)]

    def _get_provider_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for provider names from config."""
        config = get_config()
        providers = [p.name for p in config.providers]
        if not partial:
            return providers
        partial_lower = partial.lower()
        return [p for p in providers if p.lower().startswith(partial_lower)]

    def _get_model_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for /model command.

        Returns model names for tab completion. If the partial text is a
        number, returns the corresponding model name so number selection works.

        Args:
            partial: The partial model name to complete
            full_text: Full input text (unused, kept for API consistency)

        Returns:
            List of matching model names
        """
        if not partial:
            return self.model_names[:20]

        # If partial is a number, return the corresponding model name
        if partial.isdigit():
            idx = int(partial) - 1
            if 0 <= idx < len(self.model_names):
                return [self.model_names[idx]]
            return []

        # Filter by partial name (case-insensitive prefix match)
        partial_lower = partial.lower()
        matches = [m for m in self.model_names if m.lower().startswith(partial_lower)]
        return matches[:20]

    def _get_prompt_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for /prompt command.

        Handles subcommands: list, use <name>

        Args:
            partial: Everything after "/prompt " (e.g., "u", "use ", "use dev")
            full_text: Full input text (unused, kept for API consistency)
        """
        if not partial or " " not in partial:
            # Completing subcommand: "", "u", "us", "use", "li", "list"
            subcmds = ["list", "use"]
            if not partial:
                return subcmds
            partial_lower = partial.lower()
            return [s for s in subcmds if s.lower().startswith(partial_lower)]

        # Has a space - completing argument after subcommand
        # e.g., "use ", "use d", "use dev"
        subcmd_part, arg_partial = partial.split(" ", 1)
        subcmd = subcmd_part.lower()

        if subcmd == "use":
            # Complete prompt names
            prompt_names = list(self.prompt_manager.prompts.keys())
            if not arg_partial:
                return prompt_names
            arg_lower = arg_partial.lower()
            return [n for n in prompt_names if n.lower().startswith(arg_lower)]

        # Unknown subcommand - no completions
        return []

    def _get_snippet_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for /snippet command.

        Handles subcommands: list, add, delete, rename, use

        Args:
            partial: Everything after "/snippet " (e.g., "a", "add ", "add last ")
            full_text: Full input text (unused, kept for API consistency)
        """
        if not partial or " " not in partial:
            # Completing subcommand: "", "a", "ad", "add", "li", "list", etc.
            subcmds = ["list", "add", "delete", "rename", "use"]
            if not partial:
                return subcmds
            partial_lower = partial.lower()
            return [s for s in subcmds if s.lower().startswith(partial_lower)]

        # Has a space - completing argument after subcommand
        subcmd_part, arg_partial = partial.split(" ", 1)
        subcmd = subcmd_part.lower()

        snippet_names = list(self.snippet_manager.snippets.keys())

        if subcmd == "add":
            # Complete "last" keyword
            if not arg_partial:
                return ["last"]
            if "last".startswith(arg_partial.lower()):
                return ["last"]
            # After "last ", complete snippet names for the <name> arg
            if arg_partial.startswith("last"):
                rest = arg_partial[4:].lstrip()
                if rest or arg_partial.rstrip() == "last":
                    if not rest:
                        return snippet_names
                    rest_lower = rest.lower()
                    return [
                        n for n in snippet_names if n.lower().startswith(rest_lower)
                    ]
            return []

        if subcmd in ("delete", "rename", "use"):
            # Complete snippet names
            if not arg_partial:
                return snippet_names
            arg_lower = arg_partial.lower()
            return [n for n in snippet_names if n.lower().startswith(arg_lower)]

        # Unknown subcommand - no completions
        return []

    def _get_delete_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for /delete command.

        Handles: h <num>, q <num>, s <save_name>
        """
        # If no space, complete the subcommand (h, q, s)
        # User adds space manually to trigger argument completion
        if not partial or " " not in partial:
            subcmds = ["h", "q", "s"]
            if not partial:
                return subcmds
            partial_lower = partial.lower()
            return [s for s in subcmds if s.lower().startswith(partial_lower)]

        # Has a space - completing argument after subcommand
        subcmd_part, arg_partial = partial.split(" ", 1)
        subcmd = subcmd_part.lower()

        if subcmd == "s":
            # Complete save names - return full text including subcommand
            # since replacement range includes "s "
            try:
                from agent13.persistence import list_saves

                saves = list_saves()
            except Exception:
                return []

            # Sort by modification time, newest first
            saves = sorted(saves, key=lambda p: p.stat().st_mtime, reverse=True)

            # Extract names without .ctx extension
            names = [s.stem for s in saves]

            # Return completions with subcommand prefix included
            if not arg_partial:
                return [f"s {n}" for n in names[:20]]

            arg_lower = arg_partial.lower()
            matches = [n for n in names if n.lower().startswith(arg_lower)]
            return [f"s {n}" for n in matches[:20]]

        # For h and q, no completions (just numbers)
        return []

    def _get_save_completions(self, partial: str, full_text: str = "") -> list[str]:
        """Get completions for /save and /load commands.

        Lists available save files from ./.agent13/saves/, sorted by newest first.

        Args:
            partial: The partial save name to complete
            full_text: Full input text (unused, kept for API consistency)

        Returns:
            List of matching save names (without .ctx extension), sorted by newest first
        """
        try:
            saves = list_saves()
        except Exception:
            return []

        # Sort by modification time, newest first
        saves = sorted(saves, key=lambda p: p.stat().st_mtime, reverse=True)

        # Extract names without .ctx extension
        names = [s.stem for s in saves]

        if not partial:
            return names  # Return all saves

        partial_lower = partial.lower()
        matches = [n for n in names if n.lower().startswith(partial_lower)]
        return matches

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
            # full_text is like "/model dev" or "/prompt use dev"
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

        lines.append("")
        lines.append(
            "[dim]Tab/Shift+Tab/Up/Down to cycle, Enter to accept, Escape to cancel[/]"
        )
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
            self._agent_running = True
            self._error_state = False  # Clear error state on successful start
            self._last_error_type = None
            asyncio.create_task(self._write_system(f"Agent started with {self.model}"))
            asyncio.create_task(
                self._write_system(
                    "Type your message, /help for commands, or Ctrl+D to exit"
                )
            )

        @self.agent.on_event
        async def on_stopped(event: AgentEventData):
            if event.event != AgentEvent.STOPPED:
                return
            self._agent_running = False
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
            # Increment stream generation for new conversation turn
            # This ensures any stale tokens from previous turns are discarded
            self._stream_generation += 1
            # Show user message in chat when processing starts
            # MUST await (not create_task) to ensure user message appears before
            # any subsequent events like JOURNAL_COMPACT that mount widgets
            await self._write_user(text)

        @self.agent.on_event
        async def on_interrupt_injected(event: AgentEventData):
            if event.event != AgentEvent.INTERRUPT_INJECTED:
                return
            text = event.data.get("text", "")
            # Show the injected interrupt message in chat with a visual indicator
            await self._write_interrupt(text)

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
        async def on_tool_call(event: AgentEventData):
            if event.event != AgentEvent.TOOL_CALL:
                return
            name = event.data.get("name", "")
            arguments = event.data.get("arguments", {})
            self.post_message(ToolCallMessage(name, arguments))

        @self.agent.on_event
        async def on_tool_result(event: AgentEventData):
            if event.event != AgentEvent.TOOL_RESULT:
                return
            name = event.data.get("name", "")
            result = event.data.get("result", "")
            # Post through message queue to preserve ordering with tool calls
            self.post_message(ToolResultMessage(name, result))

        @self.agent.on_event
        async def on_error(event: AgentEventData):
            if event.event != AgentEvent.ERROR:
                return
            message = event.message or "Unknown error"
            error_type = event.data.get("error_type", "unknown")
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
            self._agent_running = False
            self._interrupt_requested = False
            if self._streaming_reasoning_widget:
                await self._streaming_reasoning_widget.finalize()
                self._streaming_reasoning_widget = None
            if self._streaming_content_widget:
                await self._streaming_content_widget.finalize()
                self._streaming_content_widget = None
            self._in_reasoning = False
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
            # Route through post_message → _token_queue so "Agent paused"
            # renders AFTER any pending tool results. Both tool results and
            # this message go through the same two-stage pipeline:
            #   1. post_message() → Textual message queue (preserves order)
            #   2. on_xxx_message() → _token_queue (sequential rendering)
            # Previously put_nowait() skipped stage 1, racing ahead of
            # tool results that were still in Textual's queue.
            self.post_message(
                SystemQueueMessage("[yellow]Agent paused[/]", escape_text=False)
            )

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

        @self.agent.on_event
        async def on_messages_cleared(event: AgentEventData):
            if event.event != AgentEvent.MESSAGES_CLEARED:
                return
            # Deferred /clear completed at safe boundary
            count = event.data.get("count", 0)
            clear_widgets = event.data.get("clear_widgets", False)
            # Only clear chat window widgets if user explicitly
            # requested /clear all — default /clear preserves
            # scrollback so users can review prior results
            if clear_widgets and self._chat_ready():
                await self._chat.remove_children()
            # Reset TUI token counters to match agent
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.update_status()
            asyncio.create_task(self._write_system(f"Cleared {count} messages"))

        @self.agent.on_event
        async def on_context_loaded(event: AgentEventData):
            if event.event != AgentEvent.CONTEXT_LOADED:
                return
            # Deferred /load completed at safe boundary
            success = event.data.get("success", False)
            message = event.data.get("message", "")
            incomplete = event.data.get("incomplete", False)
            if success:
                # Reset TUI token counters
                self.prompt_tokens = self.agent.prompt_tokens
                self.completion_tokens = self.agent.completion_tokens
                self.total_tokens = (
                    self.agent.prompt_tokens + self.agent.completion_tokens
                )
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
        async def on_retry_started(event: AgentEventData):
            if event.event != AgentEvent.RETRY_STARTED:
                return
            # Deferred /retry completed at safe boundary
            user_text = event.data.get("user_text", "")
            if user_text:
                # Re-add the user message and restart agent
                asyncio.create_task(self._retry_message(user_text))

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
                        f"[dim yellow]MCP {server_name}:[/] [dim]{line}[/]",
                        escape_text=False,
                    )
                )

        @self.agent.on_event
        async def on_token_usage(event: AgentEventData):
            if event.event != AgentEvent.TOKEN_USAGE:
                return
            # Capture timing data IMMEDIATELY to avoid race condition with _reset_stream_timing
            # Both this and STREAM_START use call_later, so order is non-deterministic
            first_time = self._first_token_time
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
        async def on_stream_start(event: AgentEventData):
            if event.event != AgentEvent.STREAM_START:
                return

            # Finalize any pending streaming widgets from previous stream.
            # This handles the race condition where is_final token processing
            # (via asyncio.create_task) hasn't completed before the next stream starts.
            # This can happen during journal reflection followed immediately by a turn.
            if self._streaming_reasoning_widget:
                await self._streaming_reasoning_widget.finalize()
                self._streaming_reasoning_widget = None
            if self._streaming_content_widget:
                await self._streaming_content_widget.finalize()
                self._streaming_content_widget = None
            self._in_reasoning = False

            # Update status bar display
            self.call_later(self.update_status)
            # Reset per-stream timing for accurate TPS calculation
            self.call_later(self._reset_stream_timing)

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

    def on_mount(self) -> None:
        """Set up the TUI after mounting."""
        # Set skill manager context for skill tool
        if self.skill_manager:
            skill_manager_ctx.set(self.skill_manager)

        # Start the sequential token processor
        self._token_processor_task = asyncio.create_task(self._process_tokens())

        # Validate sandbox profiles on startup
        sandbox_errors = validate_sandbox_profiles()
        if sandbox_errors:
            for error in sandbox_errors:
                print(f"Warning: {error}", file=sys.stderr)

        self.query_one("#input-field", ChatTextArea).focus()
        self._info_pane = self.query_one("#info-pane", Vertical)
        self._info_content = self.query_one("#info-content", Static)
        self._chat = self.query_one("#chat", VerticalScroll)
        self._status_left = self.query_one("#status-left", Static)
        self._status_right = self.query_one("#status-right", Static)

        # Prevent chat area from taking focus
        self._chat.can_focus = False

        # Enable anchoring for auto-scroll
        self._chat.anchor()

        # Start spinner animation (respects spinner_speed setting)
        self._restart_spinner(self._spinner_speed)

        # Initial status update
        self.update_status()

        # Load auto-save if continuing session
        if self._continue_session:
            self._load_auto_save()

        # Start the agent as a background task
        self._agent_task = asyncio.create_task(self.agent.run())

        # Connect to MCP servers if requested
        if self._connect_mcp and self.agent._mcp_server_configs:
            asyncio.create_task(self._connect_mcp_on_startup())

    async def _connect_mcp_on_startup(self) -> None:
        """Connect to MCP servers on startup."""
        try:
            mcp = await self.agent._ensure_mcp()
            if mcp:
                info = await mcp.connect_all()
                if info:
                    await self._write_system(
                        f"[dim]MCP connected: {list(info.keys())}[/]", escape_text=False
                    )
                else:
                    await self._write_system("[dim]MCP: No servers connected[/]")
        except Exception as e:
            await self._write_system(f"[red]MCP connection error: {e}[/]")

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

    def on_unmount(self) -> None:
        """Clean up when unmounting."""
        self._shutting_down = True

        # Stop the sequential token processor
        if self._token_queue:
            self._token_queue.put_nowait(None)  # Shutdown sentinel
        if self._token_processor_task:
            self._token_processor_task.cancel()

        # Stop the spinner timer
        if hasattr(self, "_spinner_timer") and self._spinner_timer is not None:
            self._spinner_timer.stop()
        self.agent.stop()
        if self._agent_task:
            self._agent_task.cancel()

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
            return True
        except MountError:
            return False

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

    async def _write_tool_call(self, name: str, arguments: dict) -> None:
        """Write a tool call message with formatted panel."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()

        # Format arguments nicely
        if arguments:
            try:
                args_str = json.dumps(arguments, indent=2)
                # Truncate if too long
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
            except Exception:
                args_str = str(arguments)[:200]
        else:
            args_str = ""

        # Create formatted tool call panel
        lines = [f"[bold yellow]⚙ {escape_markup(name)}[/]"]
        if args_str:
            lines.append(f"[dim]{escape_markup(args_str)}[/]")

        msg = Static("\n".join(lines), classes="tool-call", markup=True)
        if await self._safe_mount(msg):
            await self._scroll_if_at_bottom(was_at_bottom)

    async def _write_tool_result(self, name: str, result: str) -> None:
        """Write a tool result message with formatted panel."""
        if not self._chat_ready():
            return
        was_at_bottom = self._is_at_bottom()

        # Format result nicely
        result_str = result[:300] + "..." if len(result) > 300 else result

        # Create formatted result panel
        lines = [f"[dim]✓ Result ({escape_markup(name)}):[/]"]
        lines.append(f"[dim]{escape_markup(result_str)}[/]")

        msg = Static("\n".join(lines), classes="tool-result", markup=True)
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
            hint = "\n  [dim]Your API key lacks permission or has exceeded its limit.[/]"
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
        await self._chat.remove_children()

        # Get messages from agent
        messages = self.agent.messages

        if not messages:
            await self._write_system("[dim]No messages in loaded context[/]")
            return

        # Track tool calls so we can pair them with results
        pending_tool_calls = {}  # tool_call_id -> (name, arguments)

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")

            if role == "user":
                # Display user message
                if content:
                    await self._write_user(content)

            elif role == "assistant":
                # Display assistant content (if any text before tool calls)
                if content and content.strip():
                    was_at_bottom = self._is_at_bottom()
                    # Use StreamingMessage for replayed assistant content
                    # so markdown is rendered (same as live streaming)
                    msg_widget = StreamingMessage(content=content, title="Agent")
                    msg_widget.add_class("assistant-message")
                    if await self._safe_mount(msg_widget):
                        await self._scroll_if_at_bottom(was_at_bottom)

                # Track tool calls for pairing with results
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    tc_func = tc.get("function", {})
                    tc_name = tc_func.get("name", "unknown")
                    tc_args = tc_func.get("arguments", "{}")
                    if tc_id:
                        pending_tool_calls[tc_id] = (tc_name, tc_args)

            elif role == "tool":
                # Tool result - pair with the tool call
                if tool_call_id and tool_call_id in pending_tool_calls:
                    tc_name, tc_args_str = pending_tool_calls.pop(tool_call_id)
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
                    await self._write_tool_call(tc_name, tc_args_dict)
                    # Truncate long tool results for display
                    display_result = content
                    if len(display_result) > 500:
                        display_result = display_result[:500] + "...[truncated]"
                    await self._write_tool_result(tc_name, display_result)
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
            color = "blue"

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
        """Hide the info pane and reset mode tracking."""
        self._info_pane.styles.display = "none"
        self._info_pane_mode = None

    def _update_info_content(self, content: str) -> None:
        """Update the info pane content."""
        self._info_content.update(content)
        self._show_info_pane()

    def _reset_stream_timing(self) -> None:
        """Reset timing for a new LLM stream (called on STREAM_START event).

        This ensures each stream (including after tool calls) gets its own
        accurate TPS measurement, rather than accumulating timing across
        multiple streams.
        """
        if is_debug_enabled():
            log_tps_timing_reset(
                source="stream_start",
                old_first=self._first_token_time,
                old_last=self._last_token_time,
                old_count=self._token_count,
            )
        self._first_token_time = None
        self._last_token_time = None
        self._token_count = 0
        if is_debug_enabled():
            log_tps_stream_start(
                source="stream_start",
                first_token_time=self._first_token_time,
                token_count=self._token_count,
            )

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
            "paused": "Paused",
        }
        display_status = status_map.get(status, status.capitalize())
        self.status = display_status

        # Update processing state for spinner
        if status in ("waiting", "thinking", "processing", "tooling", "journaling"):
            self.processing = True
            # Note: TPS timing reset is handled by STREAM_START event only
            # (see on_stream_start handler). Do NOT reset here - status changes
            # happen during a stream (thinking→processing→tooling) and resetting
            # would wipe out timing before TOKEN_USAGE arrives.
            if self._stream_start_time is None:
                self._stream_start_time = time.time()
            # Start elapsed timer if not already running (new session)
            if self._elapsed_start_time is None:
                self._elapsed_start_time = time.time()
        elif status == "idle":
            self.processing = False
            # Only reset elapsed timer if no pending queue items
            # (timer continues for queued messages)
            if self.agent.queue.pending_count == 0:
                self._elapsed_start_time = None
        elif status == "paused":
            self.processing = False

    def _set_running(self, running: bool) -> None:
        """Set running state (called when agent stops)."""
        if not running:
            self.processing = False
            self.status = "Stopped"

    def _update_token_usage(
        self, data: dict, first_token_time: float = None, last_token_time: float = None
    ) -> None:
        """Update token usage from agent event and calculate TPS.

        Args:
            data: Token usage data from TOKEN_USAGE event
            first_token_time: Captured first token time (to avoid race condition)
            last_token_time: Captured last token time (to avoid race condition)
        """
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)

        # Use captured timing if provided (to avoid race condition with _reset_stream_timing)
        # Otherwise fall back to instance variables
        first_time = (
            first_token_time if first_token_time is not None else self._first_token_time
        )
        last_time = (
            last_token_time if last_token_time is not None else self._last_token_time
        )
        # Calculate TPS only if we have enough samples for accuracy
        # Short streams have high variance due to TTFT and chunk granularity
        if first_time and last_time:
            elapsed = last_time - first_time
            MIN_TOKENS = 50
            MIN_ELAPSED = 3.0  # seconds - increased to reduce TPS noise
            MIN_ELAPSED_FOR_CALC = (
                0.1  # Minimum elapsed to prevent division by tiny numbers
            )

            # Sanity check: reject absurdly short elapsed times (likely race condition)
            if elapsed < MIN_ELAPSED_FOR_CALC:
                if is_debug_enabled():
                    log_tps_event(
                        "sanity_check_elapsed_too_short",
                        {
                            "elapsed": elapsed,
                            "completion_tokens": self.completion_tokens,
                        },
                    )
                threshold_passed = False
                tps_value = None
            else:
                threshold_passed = (
                    elapsed >= MIN_ELAPSED and self.completion_tokens >= MIN_TOKENS
                )
                tps_value = None

                if threshold_passed:
                    tps_value = self.completion_tokens / elapsed
                    self._last_tps = tps_value
            # else: keep previous TPS value, don't update with noisy data

            if is_debug_enabled():
                log_tps_calculation(
                    elapsed=elapsed,
                    completion_tokens=self.completion_tokens,
                    min_elapsed=MIN_ELAPSED,
                    min_tokens=MIN_TOKENS,
                    threshold_passed=threshold_passed,
                    tps_value=tps_value,
                )
        else:
            if is_debug_enabled():
                log_tps_calculation(
                    elapsed=0,
                    completion_tokens=self.completion_tokens,
                    min_elapsed=3.0,
                    min_tokens=50,
                    threshold_passed=False,
                    tps_value=None,
                )

        if is_debug_enabled():
            log_tps_token_usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
                first_token_time=first_time,
                last_token_time=last_time,
                token_count=self._token_count,
            )

    def _get_spinner_frames(self) -> list[str]:
        """Get the appropriate spinner frames for current speed.

        Fast mode uses braille (10 frames, smooth), slow uses classic (4 frames, chunky).
        Both give ~1 revolution per second at their respective intervals.
        """
        if self._spinner_speed == "slow":
            return self.SPINNER_FRAMES_SLOW
        return self.SPINNER_FRAMES_FAST

    def _advance_spinner(self) -> None:
        """Advance the spinner animation frame."""
        # No animation when spinner is off
        if self._spinner_speed == "off":
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
            "fast": SpinnerSpeed.FAST.value,   # 0.1
            "slow": SpinnerSpeed.SLOW.value,     # 0.25
            "off": SpinnerSpeed.OFF.value,       # 0
        }
        interval = speed_map.get(speed, SpinnerSpeed.FAST.value)

        if interval > 0:
            self._spinner_timer = self.set_interval(interval, self._advance_spinner)
        else:
            # OFF mode — no timer
            self._spinner_timer = None

        self._spinner_speed = speed
        self._spinner_index = 0  # Reset to avoid OOB after frame set change
        self.update_status()

    def update_status(self) -> None:
        """Update the status line display.

        Uses event-driven status from agent (self.status) as primary source,
        with special handling for TUI-specific states (pausing, stopped).
        Spinner animates based on processing state.
        """
        # Spinner logic based on processing state and speed setting
        if self._spinner_speed == "off":
            # Show a static indicator when processing, nothing when idle
            spinner = "..." if self.processing else ""
        elif self.agent.is_pausing:
            spinner = self._get_spinner_frames()[self._spinner_index]
        elif self.processing:
            spinner = self._get_spinner_frames()[self._spinner_index]
        else:
            spinner = ""

        # Get status - use agent.pause_state as primary source (single source of truth)
        if self.agent.is_pausing:
            status_text = "pausing"
        elif not self._agent_running:
            status_text = "stopped"
        elif self.agent.is_paused:
            status_text = "paused"
        else:
            # Use the agent's current status (always up-to-date, no stale cache)
            status_text = self.agent.status.value.lower()

        # Format token counts with k suffix for thousands
        def format_tokens(n: int) -> str:
            if n >= 1000:
                return f"{n / 1000:.1f}k"
            return str(n)

        total_str = format_tokens(self.total_tokens)

        queue_info = (
            f" | Queue: {self.agent.queue.pending_count}"
            if self.agent.queue.pending_count > 0
            else ""
        )

        # Update left side (status info)
        model_display = f"{self.provider}/{self.model}" if self.provider else self.model
        left_side = f"{spinner} {status_text} | {model_display}{queue_info}".strip()
        self._status_left.update(left_side)

        # Update right side (duration, context, mcp, tools, tps)
        # Calculate elapsed time if timer is running (during active session)
        elapsed_str = ""
        if self._elapsed_start_time is not None:
            elapsed_seconds = int(time.time() - self._elapsed_start_time)
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60
            elapsed_str = f"dur: {minutes}:{seconds:02d} | "

        # Context (always shown, positioned after duration)
        ctx_str = f"Ctx: {total_str}"
        # MCP connection status (only show when connected)
        mcp_str = ""
        if self.agent.mcp and self.agent.mcp.is_connected():
            mcp_str = " | mcp: on"

        # Journal mode status (only show when enabled)
        jnl_str = ""
        if self.agent.journal_mode:
            jnl_str = " | jnl: on"

        # Tool stats
        tool_stats = self.agent.tool_stats
        if tool_stats.total_calls > 0:
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
        # Build right side: dur: | Ctx: | mcp: | jnl: | Tools: | tps:
        right_side = (
            f"{elapsed_str}{ctx_str}{mcp_str}{jnl_str}{tools_str}{tps_str}".strip()
        )
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
            # If info pane is visible, close it on empty submit
            if self._info_pane.styles.display != "none":
                self._hide_info_pane()
            return

        input_field = self.query_one("#input-field", ChatTextArea)
        input_field.clear()
        input_field.focus()

        # Reset completion state after submission
        self._reset_completion_state()
        self._hide_info_pane()

        # Any new input (except /resume) closes the /resume-after-interrupt window
        if not text.startswith("/resume"):
            self._interrupt_available = False

        # Add to persistent history
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

            # Check if message will be queued (agent is processing or queue has pending items)
            is_queued = self.processing or self.agent.queue.pending_count > 0

            if is_queued:
                # Show notification for queued item
                queue_pos = self.agent.queue.pending_count + 1
                if interrupt:
                    level_text = " (interrupt)"
                elif priority:
                    level_text = " (priority)"
                else:
                    level_text = ""
                self.notify(
                    f"Queued #{queue_pos}{level_text}: {message_text[:30]}{'...' if len(message_text) > 30 else ''}"
                )

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
        await self.agent.add_message(text, priority=priority, interrupt=interrupt)

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

        if command == "help":
            self._handle_help_command()
        elif command == "model":
            self._handle_model_command(args)
        elif command == "history":
            self._handle_hist_command()
        elif command == "delete":
            self._handle_delete_command(args)
        elif command == "clear":
            clear_all = args.strip().lower() == "all"
            asyncio.create_task(
                self.agent.request_clear(clear_widgets=clear_all)
            )
        elif command == "prompt":
            self._handle_prompt_command(args)
        elif command == "pretty":
            self._handle_pretty_command(args)
        elif command == "tool-response":
            self._handle_tool_response_command(args)
        elif command == "queue":
            self._handle_queue_command()
        elif command == "pause":
            self._handle_pause_command()
        elif command == "resume":
            self._handle_resume_command()
        elif command == "retry":
            self._handle_retry_command()
        elif command == "prioritise" or command == "prioritize":
            self._handle_prioritise_command(args)
        elif command == "deprioritise" or command == "deprioritize":
            self._handle_deprioritise_command(args)
        elif command == "sandbox":
            self._handle_sandbox_command(args)
        elif command == "provider":
            self._handle_provider_command(args)
        elif command == "mcp":
            self._handle_mcp_command(args)
        elif command == "tools":
            self._handle_tools_command()
        elif command == "skills":
            self._handle_skills_command()
        elif command == "journal":
            self._handle_journal_command(args)
        elif command == "remove-reasoning":
            self._handle_remove_reasoning_command(args)
        elif command == "devel":
            self._handle_devel_command(args)
        elif command == "save":
            self._handle_save_command(args)
        elif command == "load":
            self._handle_load_command(args)
        elif command == "snippet":
            self._handle_snippet_command(args)
        elif command == "spinner":
            self._handle_spinner_command(args)
        elif command == "upgrade":
            self._handle_upgrade_command(args)
        elif command == "clipboard":
            self._handle_clipboard_command(args)
        elif command in ("quit", "exit"):
            self.agent.stop()
            self.exit()
        else:
            # Check for skill invocation
            if self.skill_manager:
                skill = self.skill_manager.get_skill(command)
                if skill:
                    content = self.skill_manager.format_skill_content(command)
                    if content:
                        asyncio.create_task(self._send_message(content))
                    else:
                        self._update_info_content(
                            f"[red]Failed to load skill: {command}[/]"
                        )
                    return
            # Check for snippet invocation
            snippet = self.snippet_manager.get_snippet(command)
            if snippet is not None:
                input_field = self.query_one("#input-field", ChatTextArea)
                input_field.text = snippet
                input_field.focus()
                return
            self._update_info_content(f"[red]Unknown command: {escape_markup(cmd)}[/]")

    def _handle_model_command(self, args: str) -> None:
        """Handle /model command."""
        if not args:
            # Show model list via completion system
            input_field = self.query_one("#input-field", ChatTextArea)
            input_field.text = "/model "
            cursor_col = len("/model ")
            input_field.move_cursor((0, cursor_col))
            # Trigger completion for the current input context
            completion_type, start_loc, end_loc = self._get_completion_context(
                "/model ", 0, cursor_col
            )
            if completion_type != "none":
                matches = self._get_completions_for_context(
                    completion_type, "", "/model "
                )
                if matches:
                    if len(matches) == 1:
                        # Single match - complete immediately
                        self._apply_completion(
                            input_field, matches[0], start_loc, end_loc
                        )
                    else:
                        # Multiple matches - enter completion mode
                        self._completion_matches = matches
                        self._completion_index = 0
                        self._completion_prefix = ""
                        self._completion_start = start_loc
                        self._completion_end = end_loc
                        self._apply_completion(
                            input_field, matches[0], start_loc, end_loc
                        )
                        self._completion_text = input_field.text
                        self._show_completions(matches, matches[0])
            self.call_later(input_field.focus)
            return

        model = self._resolve_model(args)
        if model:
            self.model = model
            self.agent.set_model(self.model)
            asyncio.create_task(self._write_system(f"Model set to: {self.model}"))
            self.update_status()
        else:
            self._update_info_content(f"[red]Unknown model: {escape_markup(args)}[/]")

    def _resolve_model(self, choice: str) -> str | None:
        """Resolve model selection by number or name."""
        choice = choice.strip()
        if not choice:
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(self.model_names):
                return self.model_names[idx]
            return None

        # Check for exact match (case-insensitive)
        for m in self.model_names:
            if m.lower() == choice.lower():
                return m

        # Check for substring matches
        matches = [m for m in self.model_names if choice.lower() in m.lower()]
        if len(matches) == 1:
            return matches[0]
        return None

    def _get_message_groups(self) -> list[list[int]]:
        """Group messages for atomic deletion.

        Each group starts with a non-interrupt user message and includes all
        subsequent messages (interrupt user messages, tool calls, tool results,
        assistant responses) until the next non-interrupt user message.

        Interrupt user messages (marked with "interrupt": True) are kept in
        the same group as the turn they interrupted, so /history shows them
        as part of that turn and /journal compacts the entire turn together.

        Returns:
            List of groups, where each group is a list of message indices.
        """
        groups = []
        current_group = []

        for i, msg in enumerate(self.agent.messages):
            role = msg.get("role", "unknown")

            if role == "user" and not msg.get("interrupt"):
                # Start a new group (non-interrupt user message)
                if current_group:
                    groups.append(current_group)
                current_group = [i]
            else:
                # Add to current group (interrupt user msgs, tools, assistants)
                current_group.append(i)

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _handle_hist_command(self) -> None:
        """Handle /history command - show grouped message history."""
        if not self.agent.messages:
            self._update_info_content("[dim]No messages in history[/]")
            self._info_pane_mode = "history"
            return

        groups = self._get_message_groups()
        lines = ["[bold]Message history (grouped):[/]"]

        for group_num, group in enumerate(groups, 1):
            # First message in group is always user (or should be)
            first_idx = group[0]
            first_msg = self.agent.messages[first_idx]
            first_role = first_msg.get("role", "unknown")
            first_content = first_msg.get("content", "")

            # Truncate content for display
            if first_content:
                display = first_content[:80]
                if len(first_content) > 80:
                    display += "..."
            else:
                display = "(empty)"

            # Show group header
            lines.append(
                f"  [cyan]{group_num}.[/] [bold]{first_role}:[/] {escape_markup(display)}"
            )

            # Show subsequent messages in group (indented)
            for idx in group[1:]:
                msg = self.agent.messages[idx]
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "tool":
                    tool_call_id = msg.get("tool_call_id", "?")
                    lines.append(
                        f"      [dim]tool result (id={escape_markup(tool_call_id)})[/]"
                    )
                elif role == "assistant":
                    if content:
                        display = content[:60]
                        if len(content) > 60:
                            display += "..."
                        lines.append(
                            f"      [magenta]assistant:[/] {escape_markup(display)}"
                        )
                    else:
                        # Check for tool calls
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                tc_name = tc.get("function", {}).get("name", "?")
                                lines.append(
                                    f"      [yellow]tool call: {escape_markup(tc_name)}[/]"
                                )
                else:
                    # Check for interrupt user messages
                    if role == "user" and msg.get("interrupt"):
                        if content:
                            display = content[:60]
                            if len(content) > 60:
                                display += "..."
                        else:
                            display = "(empty)"
                        lines.append(
                            f"      [yellow]⚡ interrupt:[/] {escape_markup(display)}"
                        )
                    elif content:
                        display = content[:60]
                        if len(content) > 60:
                            display += "..."
                        lines.append(f"      [dim]{role}:[/] {escape_markup(display)}")

        lines.append("")
        lines.append("[dim]Use /delete h N to delete group N[/]")
        self._update_info_content("\n".join(lines))
        self._info_pane_mode = "history"

    def _handle_delete_command(self, args: str) -> None:
        """Handle /delete command."""
        if not args:
            self._update_info_content(
                "[red]Usage: /delete h N | /delete q N | /delete s NAME[/]"
            )
            return

        parts = args.split()
        if len(parts) < 2:
            self._update_info_content(
                "[red]Usage: /delete h N | /delete q N | /delete s NAME[/]"
            )
            return

        target = parts[0].lower()
        spec = parts[1]

        if target == "h":
            # Delete from history (by group)
            groups = self._get_message_groups()
            try:
                if "-" in spec:
                    # Delete range of groups
                    start, end = map(int, spec.split("-"))
                    if start < 1 or end > len(groups) or start > end:
                        self._update_info_content(
                            f"[red]Invalid group range: {start}-{end}[/]"
                        )
                        return
                    # Collect all indices to delete
                    indices_to_delete = []
                    for group_num in range(start, end + 1):
                        indices_to_delete.extend(groups[group_num - 1])
                    # Delete in reverse order to preserve indices
                    for idx in sorted(indices_to_delete, reverse=True):
                        del self.agent.messages[idx]
                    self.update_status()
                    asyncio.create_task(
                        self._write_system(
                            f"Deleted groups {start}-{end} ({len(indices_to_delete)} messages)"
                        )
                    )
                else:
                    # Delete single group
                    group_num = int(spec)
                    if group_num < 1 or group_num > len(groups):
                        self._update_info_content(
                            f"[red]Invalid group number: {group_num}[/]"
                        )
                        return
                    indices = groups[group_num - 1]
                    # Delete in reverse order to preserve indices
                    for idx in sorted(indices, reverse=True):
                        del self.agent.messages[idx]
                    self.update_status()
                    asyncio.create_task(
                        self._write_system(
                            f"Deleted group {group_num} ({len(indices)} messages)"
                        )
                    )
            except ValueError:
                self._update_info_content("[red]Invalid index format[/]")
        elif target == "q":
            # Delete from queue
            try:
                idx = int(spec)
                removed = self.agent.queue.remove_at(idx)
                if removed:
                    asyncio.create_task(
                        self._write_system(f"Removed queue item: {removed.text[:50]}")
                    )
                else:
                    self._update_info_content(f"[red]Invalid queue index: {idx}[/]")
            except ValueError:
                self._update_info_content("[red]Invalid index format[/]")
        elif target == "s":
            # Delete save file
            from agent13.persistence import get_saves_dir

            saves_dir = get_saves_dir()
            save_path = saves_dir / f"{spec}.ctx"
            if save_path.exists():
                save_path.unlink()
                asyncio.create_task(self._write_system(f"Deleted save: {spec}"))
            else:
                self._update_info_content(f"[red]Save not found: {spec}[/]")
        else:
            self._update_info_content(
                "[red]Usage: /delete h N | /delete q N | /delete s NAME[/]"
            )

    def _handle_help_command(self) -> None:
        """Handle /help command."""
        sandbox_mode = get_current_sandbox_mode()
        self._update_info_content(
            "[bold]Commands:[/]\n"
            "  [yellow]/help[/] - Show this help\n"
            "  [yellow]/model [name][/] - Select model (tab to list)\n"
            "  [yellow]/history[/] - Show message history\n"
            "  [yellow]/delete h N[/] - Delete message group N from history\n"
            "  [yellow]/delete q N[/] - Delete queue item N\n"
            "  [yellow]/delete s NAME[/] - Delete saved context\n"
            "  [yellow]/clear [all][/] - Clear context (all: also clear scrollback)\n"
            "  [yellow]/prompt [list|use NAME][/] - Manage prompts\n"
            "  [yellow]/pretty [on|off][/] - Toggle markdown rendering\n"
            "  [yellow]/tool-response [raw|json][/] - Set tool response format\n"
            "  [yellow]/sandbox [mode][/] - Show/set sandbox mode\n"
            "  [yellow]/mcp [connect|disconnect|reload][/] - List/manage MCP servers\n"
            "  [yellow]/tools[/] - Show tool usage statistics\n"
            "  [yellow]/skills[/] - List available skills\n"
            "\n[bold]Save/Load:[/]\n"
            "  [yellow]/save <name> [-y][/] - Save context to ./.agent13/saves/<name>.ctx\n"
            "  [yellow]/load <name>[/] - Load context from ./.agent13/saves/<name>.ctx\n"
            "\n[bold]Queue commands:[/]\n"
            "  [yellow]/queue[/] - Show queue items\n"
            "  [yellow]/pause[/] - Pause agent processing\n"
            "  [yellow]/resume[/] - Resume agent (after ESC or /pause)\n"
            "  [yellow]/retry[/] - Retry last message\n"
            "  [yellow]/prioritise N[/] - Mark queue item as priority\n"
            "  [yellow]/provider [name|url][/] - Change provider\n"
            "  [yellow]/deprioritise N[/] - Remove priority from item\n"
            "\n[bold]Journal mode:[/]\n"
            "  [yellow]/journal [on|off|last|all|status][/] - Context compaction via reflection\n"
            "\n[bold]Reasoning:[/]\n"
            "  [yellow]/remove-reasoning [on|off][/] - Strip reasoning tokens between turns\n"
            "\n[bold]Spinner:[/]\n"
            "  [yellow]/spinner [fast|slow|off|status][/] - Spinner style and speed\n"
            "\n[bold]Updates:[/]\n"
            "  [yellow]/upgrade [--copy][/] - Check for updates and apply (or copy command to clipboard)\n"
            "\n[bold]Clipboard:[/]\n"
            "  [yellow]/clipboard [osc52|system][/] - Show or set clipboard method\n"
            "\n[bold]Keyboard shortcuts:[/]\n"
            "  [yellow]ESC[/] - Cancel request (use /resume to continue)\n"
            "  [yellow]Ctrl+C[/] - Clear input or quit\n"
            "  [yellow]Ctrl+D / Ctrl+Q[/] - Quit app\n"
            "  [yellow]Tab[/] - Tab completion for commands/history\n"
            "  [yellow]Up/Down[/] - Navigate input history\n"
            "  [yellow]Shift+Up/Down[/] - Scroll chat\n"
            "\n[bold]Tips:[/]\n"
            "  Start message with [yellow]![/] to add to front of queue (processed next)\n"
            "  Start message with [yellow]!![/] to interrupt and process immediately\n"
            "\n[bold]Current settings:[/]\n"
            f"  model: {escape_markup(self.model)}\n"
            f"  tool-response: {escape_markup(self.tool_response_format)}\n"
            f"  pretty: {'on' if self.pretty else 'off'}\n"
            f"  prompt: {escape_markup(self.prompt_manager.active_prompt)}\n"
            f"  sandbox: {escape_markup(sandbox_mode.value)}\n"
            f"  remove-reasoning: {'on' if self.agent.remove_reasoning else 'off'}\n"
            f"  spinner: {self._spinner_speed}\n"
            f"  clipboard: {self._clipboard_method}"
        )
        self._info_pane_mode = "help"

    def _handle_prompt_command(self, args: str) -> None:
        """Handle /prompt command."""
        if not args:
            # Show current prompt
            prompt = self.prompt_manager.get_prompt()
            preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self._update_info_content(
                f"[bold]Current prompt ({self.prompt_manager.active_prompt}):[/]\n"
                f"  {preview}\n\n"
                "[dim]Usage: /prompt [list|use NAME][/]"
            )
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            lines = ["[bold]Available prompts:[/]"]
            for name in self.prompt_manager.prompts:
                marker = (
                    " (active)" if name == self.prompt_manager.active_prompt else ""
                )
                lines.append(f"  [yellow]{escape_markup(name)}[/]{marker}")
            self._update_info_content("\n".join(lines))
        elif subcmd == "use":
            if not subargs:
                self._update_info_content("[red]Usage: /prompt use NAME[/]")
                return
            if self.prompt_manager.set_active(subargs.strip()):
                self.agent.set_system_prompt(self.prompt_manager.get_prompt())
                asyncio.create_task(
                    self._write_system(f"Switched to prompt: {escape_markup(subargs)}")
                )
            else:
                self._update_info_content(
                    f"[red]Prompt not found: {escape_markup(subargs)}[/]"
                )
        else:
            self._update_info_content(
                f"[red]Unknown prompt command: {escape_markup(subcmd)}[/]\n"
                "[dim]Usage: /prompt [list|use NAME][/]"
            )

    def _handle_snippet_command(self, args: str) -> None:
        """Handle /snippet command."""
        if not args:
            self._update_info_content(
                "[bold]Snippets:[/] Saved user messages invoked as slash commands.\n\n"
                "[dim]Usage:[/]\n"
                "  [yellow]/snippet list[/] - List all snippets\n"
                "  [yellow]/snippet add last <name> [-y][/] - Save last user message as snippet\n"
                "  [yellow]/snippet delete <name>[/] - Delete a snippet\n"
                "  [yellow]/snippet rename <old> <new>[/] - Rename a snippet\n"
                "  [yellow]/snippet use <name>[/] - Fill input with snippet\n\n"
                "[dim]You can also type /<snippet_name> to fill input directly.[/]"
            )
            return

        parts = args.split(maxsplit=2)
        subcmd = parts[0].lower()

        if subcmd == "list":
            snippets = self.snippet_manager.list_snippets()
            if not snippets:
                self._update_info_content("[dim]No snippets defined.[/]")
                return
            lines = ["[bold]Available snippets:[/]"]
            for s in snippets:
                collision = (
                    " [red](collision with built-in)[/]" if s["collision"] else ""
                )
                lines.append(f"  [yellow]{escape_markup(s['name'])}[/]{collision}")
                lines.append(f"    [dim]{escape_markup(s['preview'])}[/]")
            self._update_info_content("\n".join(lines))

        elif subcmd == "add":
            # /snippet add last <name> [-y]
            all_parts = args.split()
            if len(all_parts) < 3 or all_parts[1].lower() != "last":
                self._update_info_content(
                    "[red]Usage: /snippet add last <name> [-y][/]\n"
                    "  Saves your previous user message as a snippet."
                )
                return
            # Parse name and -y flag from remaining parts
            force = "-y" in all_parts
            name_parts = [p for p in all_parts[2:] if p != "-y"]
            if not name_parts:
                self._update_info_content(
                    "[red]Usage: /snippet add last <name> [-y][/]"
                )
                return
            name = name_parts[0]

            # Validate name
            validation_error = self.snippet_manager.validate_name(name)
            if validation_error:
                self._update_info_content(f"[red]{validation_error}[/]")
                return

            # Check overwrite
            if self.snippet_manager.get_snippet(name) is not None and not force:
                self._update_info_content(
                    f"[yellow]Snippet '{escape_markup(name)}' already exists.[/]\n"
                    f"Use [yellow]/snippet add last {name} -y[/] to overwrite"
                )
                return

            # Get last user message
            content = self._get_last_user_message()
            if content is None:
                self._update_info_content("[red]No user message to save[/]")
                return

            warning = self.snippet_manager.add_snippet(name, content)

            # Register new slash command if no collision
            if name not in {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}:
                slash = f"/{name}"
                if slash not in self.SLASH_COMMANDS:
                    self.SLASH_COMMANDS.append(slash)

            msg = f"[green]Saved snippet '{escape_markup(name)}'[/]"
            if warning:
                msg += f"\n[yellow]WARNING: {warning}[/]"
            self._update_info_content(msg)

        elif subcmd == "delete":
            if len(parts) < 2:
                self._update_info_content("[red]Usage: /snippet delete <name>[/]")
                return
            name = parts[1]
            if self.snippet_manager.delete_snippet(name):
                # Remove from slash commands
                slash = f"/{name}"
                if slash in self.SLASH_COMMANDS:
                    self.SLASH_COMMANDS.remove(slash)
                self._update_info_content(
                    f"[green]Deleted snippet '{escape_markup(name)}'[/]"
                )
            else:
                self._update_info_content(
                    f"[red]Snippet not found: {escape_markup(name)}[/]"
                )

        elif subcmd == "rename":
            if len(parts) < 3:
                self._update_info_content("[red]Usage: /snippet rename <old> <new>[/]")
                return
            old_name = parts[1]
            new_name = parts[2]
            # Remove old slash command
            old_slash = f"/{old_name}"
            if old_slash in self.SLASH_COMMANDS:
                self.SLASH_COMMANDS.remove(old_slash)
            result = self.snippet_manager.rename_snippet(old_name, new_name)
            if result and result.startswith("Snippet not found"):
                self._update_info_content(f"[red]{result}[/]")
                return
            if result and result.startswith("Invalid"):
                self._update_info_content(f"[red]{result}[/]")
                return
            # Add new slash command if no collision
            if new_name not in {cmd[1:] for cmd in self._BUILTIN_SLASH_COMMANDS}:
                new_slash = f"/{new_name}"
                if new_slash not in self.SLASH_COMMANDS:
                    self.SLASH_COMMANDS.append(new_slash)
            msg = f"[green]Renamed snippet '{escape_markup(old_name)}' → '{escape_markup(new_name)}'[/]"
            if result:
                msg += f"\n[yellow]WARNING: {result}[/]"
            self._update_info_content(msg)

        elif subcmd == "use":
            if len(parts) < 2:
                self._update_info_content("[red]Usage: /snippet use <name>[/]")
                return
            name = parts[1]
            snippet = self.snippet_manager.get_snippet(name)
            if snippet is None:
                self._update_info_content(
                    f"[red]Snippet not found: {escape_markup(name)}[/]"
                )
                return
            input_field = self.query_one("#input-field", ChatTextArea)
            input_field.text = snippet
            input_field.focus()

        else:
            self._update_info_content(
                f"[red]Unknown snippet command: {escape_markup(subcmd)}[/]\n"
                "[dim]Usage: /snippet [list|add|delete|rename|use][/]"
            )

    def _get_last_user_message(self) -> str | None:
        """Get the text of the last non-slash-command user message.

        Returns None if no suitable message is found.
        """
        idx = self.agent._find_last_user_idx()
        if idx is None:
            return None
        content = self.agent.messages[idx].get("content", "")
        if not content or not content.strip():
            return None
        return content

    def _handle_pretty_command(self, args: str) -> None:
        """Handle /pretty command."""
        if not args:
            self.pretty = not self.pretty
        else:
            arg = args.strip().lower()
            if arg == "on":
                self.pretty = True
            elif arg == "off":
                self.pretty = False
            else:
                self._update_info_content("[red]Usage: /pretty [on|off][/]")
                return
        asyncio.create_task(
            self._write_system(f"Pretty mode: {'on' if self.pretty else 'off'}")
        )

    def _handle_tool_response_command(self, args: str) -> None:
        """Handle /tool-response command."""
        if not args:
            self._update_info_content(
                f"[bold]Tool response format:[/] {self.tool_response_format}\n"
                "[dim]Usage: /tool-response [raw|json][/]"
            )
            return

        format_arg = args.strip().lower()
        if format_arg in ("raw", "json"):
            self.tool_response_format = format_arg
            response_format = {"type": "json_object"} if format_arg == "json" else None
            self.agent.set_response_format(response_format)
            asyncio.create_task(
                self._write_system(f"Tool response format set to: {format_arg}")
            )
        else:
            self._update_info_content("[red]Invalid format. Use 'raw' or 'json'[/]")

    def _handle_queue_command(self) -> None:
        """Handle /queue command - show queue items."""
        items = self.agent.queue.list_items()
        if not items:
            self._update_info_content("[dim]Queue is empty[/]")
            self._info_pane_mode = "queue"
            return

        lines = ["[bold]Queue items:[/]"]
        for i, item in enumerate(items, 1):
            if item.interrupt:
                priority_marker = "[red]!![/] "
            elif item.priority:
                priority_marker = "[yellow]![/] "
            else:
                priority_marker = "   "
            status_marker = "running" if item.status.value == "running" else ""
            text_preview = item.text[:40] + "..." if len(item.text) > 40 else item.text
            lines.append(
                f"  {priority_marker}[cyan]{i}.[/] {escape_markup(text_preview)} {status_marker}"
            )
        self._update_info_content("\n".join(lines))
        self._info_pane_mode = "queue"

    def _handle_pause_command(self) -> None:
        """Handle /pause command - pause agent processing.

        When the agent is actively processing (in the middle of tool calls),
        this requests a pause at the next safe point and shows 'Pausing' status.
        When the pause takes effect, status changes to 'paused'.
        """
        if self.agent.is_paused:
            self._update_info_content("[yellow]Already paused[/]")
            return

        if self.agent.is_pausing:
            self._update_info_content("[yellow]Already pausing[/]")
            return

        if self.processing:
            # Agent is actively processing - request pause at next safe point
            self.agent.pause()
            self.update_status()  # Update status to show "pausing"
            asyncio.create_task(
                self._write_system(
                    "[yellow]Pausing at next safe point...[/]", escape_text=False
                )
            )
        else:
            # Agent is idle - pause immediately
            self.agent.stop()
            asyncio.create_task(
                self._write_system("[yellow]Agent paused[/]", escape_text=False)
            )

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
            asyncio.create_task(
                self._write_system(
                    "[green]Continuing from interrupt[/]", escape_text=False
                )
            )
            return

        # Check for incomplete turn from loaded context (even if not paused)
        if self.agent.has_incomplete_turn:
            self._agent_running = True
            self.update_status()
            asyncio.create_task(self._continue_incomplete_turn())
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
            self._agent_running = True
            self._agent_task = asyncio.create_task(self.agent.run())

        # Immediately update status display (agent will emit STATUS_CHANGE shortly)
        self.update_status()
        asyncio.create_task(
            self._write_system("[green]Agent resumed[/]", escape_text=False)
        )

    async def _continue_incomplete_turn(self) -> None:
        """Continue an incomplete turn from a loaded context."""
        await self._write_system(
            "[yellow]Continuing incomplete turn...[/]", escape_text=False
        )
        try:
            await self.agent.continue_incomplete_turn()
            await self._write_system("[green]Turn completed[/]", escape_text=False)
        except Exception as e:
            await self._write_system(
                f"[red]Error continuing turn: {escape_markup(str(e))}[/]",
                escape_text=False,
            )
        finally:
            self._agent_running = False
            self.update_status()

    def _handle_retry_command(self) -> None:
        """Handle /retry command - retry the last message.

        Queues a kind='retry' item for safe deferred processing.
        The agent processes it at a safe boundary between items,
        deleting the last message group and re-adding the user message.
        """
        if not self.agent.messages:
            self._update_info_content("[yellow]No messages to retry[/]")
            return

        # Clear error/interrupt state (pause state read from agent.pause_state)
        self._error_state = False
        self._last_error_type = None
        self._interrupt_requested = False
        self._interrupt_available = False

        # Queue the retry for safe deferred processing
        asyncio.create_task(self.agent.request_retry())

    async def _retry_message(self, text: str) -> None:
        """Re-add a message to the queue for retry.

        If the agent loop is running, just adds the message to the queue
        and the existing loop will pick it up. If the loop has stopped
        (e.g. after an error), restarts it.

        Args:
            text: The user message text to retry
        """
        if self._agent_task and not self._agent_task.done():
            # Agent loop is still running — just add the message to the queue
            await self.agent.add_message(text)
        else:
            # Agent loop has stopped — wait for cleanup, then restart
            if self._agent_task:
                try:
                    await self._agent_task
                except asyncio.CancelledError:
                    pass
            self._agent_running = True
            await self.agent.add_message(text)
            self._agent_task = asyncio.create_task(self.agent.run())
        await self._write_system(
            f"[green]Retrying: {escape_markup(text[:50])}{'...' if len(text) > 50 else ''}[/]",
            escape_text=False,
        )

    def _handle_prioritise_command(self, args: str) -> None:
        """Handle /prioritise command - mark queue item as priority."""
        if not args:
            self._update_info_content("[red]Usage: /prioritise N[/]")
            return

        try:
            idx = int(args.strip())
            if self.agent.queue.set_priority_at(idx, True):
                asyncio.create_task(
                    self._write_system(f"Item {idx} marked as priority")
                )
            else:
                self._update_info_content(f"[red]Invalid queue index: {idx}[/]")
        except ValueError:
            self._update_info_content("[red]Invalid index format[/]")

    def _handle_deprioritise_command(self, args: str) -> None:
        """Handle /deprioritise command - remove priority from queue item."""
        if not args:
            self._update_info_content("[red]Usage: /deprioritise N[/]")
            return

        try:
            idx = int(args.strip())
            if self.agent.queue.set_priority_at(idx, False):
                asyncio.create_task(self._write_system(f"Item {idx} priority removed"))
            else:
                self._update_info_content(f"[red]Invalid queue index: {idx}[/]")
        except ValueError:
            self._update_info_content("[red]Invalid index format[/]")

    def _handle_provider_command(self, args: str) -> None:
        """Handle /provider command - change provider (use /model to select model after)."""
        if not args:
            self._update_info_content("[red]Usage: /provider <provider_name_or_url>[/]")
            return

        try:
            base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                args
            )
        except ValueError as e:
            self._update_info_content(f"[red]Error: {escape_markup(str(e))}[/]")
            return

        # Create new client with the new provider
        self.client = create_client(
            base_url,
            api_key,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
        )
        self.agent.set_client(self.client)

        # Update provider name for status bar (empty if URL)
        self.provider = (
            "" if args.startswith("http://") or args.startswith("https://") else args
        )
        self.update_status()

        # Fetch models from the new provider
        asyncio.create_task(self._fetch_and_show_models(base_url))

    async def _fetch_and_show_models(self, base_url: str) -> None:
        """Fetch models from provider and show them."""
        try:
            model_names = await fetch_models(self.client)
        except Exception as e:
            await self._write_system(
                f"[red]Error fetching models: {escape_markup(str(e))}[/]",
                escape_text=False,
            )
            return

        if not model_names:
            await self._write_system(
                "[red]No models available from provider[/]", escape_text=False
            )
            return

        # Update model list
        self.model_names = model_names

        # Show model list
        lines = [f"[bold]Provider changed to: {escape_markup(base_url)}[/]"]
        lines.append("[bold]Available models:[/]")
        for i, name in enumerate(model_names, 1):
            lines.append(f"  [yellow]{i}.[/] {escape_markup(name)}")
        lines.append("")
        lines.append("[dim]Use /model <name> or /model <tab> to select[/]")
        self._update_info_content("\n".join(lines))

    def _handle_sandbox_command(self, args: str) -> None:
        """Handle /sandbox command - show or set sandbox mode."""
        if not args:
            # Show current mode
            current = get_current_sandbox_mode()
            session = get_session_sandbox_mode()
            config_default = get_default_sandbox_mode()

            lines = ["[bold]Sandbox Configuration:[/]"]
            lines.append(f"  Current mode: [cyan]{current.value}[/]")
            if session:
                lines.append(f"  Session override: [yellow]{session.value}[/]")
            else:
                lines.append("  Session override: [dim]none (using config default)[/]")
            lines.append(f"  Config default: [dim]{config_default.value}[/]")
            lines.append("")
            lines.append(format_all_sandbox_modes())
            lines.append("")
            lines.append("[dim]Usage: /sandbox <mode> to set session override[/]")
            self._update_info_content("\n".join(lines))
            return

        # Set session override
        mode_str = args.strip().lower()
        try:
            mode = parse_sandbox_mode(mode_str)
            set_session_sandbox_mode(mode)
            asyncio.create_task(
                self._write_system(f"Sandbox mode set to: {mode.value}")
            )
        except ValueError as e:
            self._update_info_content(f"[red]{escape_markup(str(e))}[/]")

    def _handle_mcp_command(self, args: str) -> None:
        """Handle /mcp command - list MCP servers or manage connections."""
        parts = args.split(maxsplit=1) if args else []
        subcmd = parts[0].lower() if parts else ""

        if subcmd == "reload":
            # Reload MCP servers
            if not self.agent.mcp:
                asyncio.create_task(
                    self._write_system(
                        "[dim]MCP not initialized (no servers configured)[/]"
                    )
                )
                return

            async def do_reload():
                servers = await self.agent.mcp.reload()
                await self._write_system(
                    f"[dim]MCP reconnected: {list(servers.keys())}[/]"
                )

            asyncio.create_task(do_reload())

        elif subcmd == "connect":
            # Connect to all MCP servers
            if not self.agent._mcp_server_configs:
                asyncio.create_task(
                    self._write_system("[dim]No MCP servers configured[/]")
                )
                return

            async def do_connect():
                try:
                    mcp = await self.agent._ensure_mcp()
                    if mcp:
                        info = await mcp.connect_all()
                        await self._write_system(
                            f"[dim]MCP connected: {list(info.keys())}[/]",
                            escape_text=False,
                        )
                except Exception as e:
                    self.post_message(ErrorMessage(f"MCP connect error: {e}", "mcp"))

            asyncio.create_task(do_connect())

        elif subcmd == "disconnect":
            # Disconnect from all MCP servers
            if not self.agent.mcp:
                asyncio.create_task(self._write_system("[dim]MCP not connected[/]"))
                return

            async def do_disconnect():
                await self.agent.disconnect_mcp()
                await self._write_system(
                    "[dim]MCP servers disconnected[/]", escape_text=False
                )

            asyncio.create_task(do_disconnect())

        else:
            # List MCP servers
            if not self.agent.mcp:
                configured = len(self.agent._mcp_server_configs)
                self._update_info_content(
                    f"[bold]MCP Status:[/]\n"
                    f"  Status: [dim]Not initialized[/]\n"
                    f"  Configured servers: {configured}\n\n"
                    f"[dim]Use /mcp connect to connect to MCP servers.[/]"
                )
                return

            servers = self.agent.mcp.get_server_info()
            if not servers:
                configured = len(self.agent._mcp_server_configs)
                self._update_info_content(
                    f"[bold]MCP Status:[/]\n"
                    f"  Configured servers: {configured}\n\n"
                    f"[dim]Use /mcp connect to connect to servers.[/]"
                )
                return
            formatted = format_mcp_servers(servers, use_rich=True)
            self._update_info_content(formatted)

    def _handle_tools_command(self) -> None:
        """Handle /tools command - show tool usage statistics."""
        stats = self.agent.tool_stats
        total = stats.total_calls
        successes = stats.total_successes

        if total == 0:
            self._update_info_content("[dim]No tool calls yet this session[/]")
            return

        lines = [f"[bold]Tool Usage[/]  {successes}/{total} successful", ""]

        # Sort by call count descending
        sorted_tools = sorted(stats.calls.items(), key=lambda x: x[1], reverse=True)

        for name, calls in sorted_tools:
            tool_successes = stats.successes.get(name, 0)
            success_str = f"{tool_successes}/{calls}"

            # Mode breakdown if available
            modes = stats.modes.get(name, {})
            if modes:
                mode_successes = stats.mode_successes.get(name, {})
                mode_strs = [
                    f"{m}={mode_successes.get(m, 0)}/{c}"
                    for m, c in sorted(modes.items())
                ]
                mode_str = f"  modes: {', '.join(mode_strs)}"
            else:
                mode_str = ""

            lines.append(f"  [yellow]{name}[/]  {success_str}{mode_str}")

        self._update_info_content("\n".join(lines))

    def _handle_skills_command(self) -> None:
        """Handle /skills command - list available skills."""
        if not self.skill_manager:
            self._update_info_content("[dim]No skill manager configured[/]")
            return

        skills = self.skill_manager.skills
        if not skills:
            self._update_info_content("[dim]No skills available[/]")
            return

        lines = ["[bold]Available skills:[/]", ""]
        for name, info in sorted(skills.items()):
            desc = (
                info.description[:60] + "..."
                if len(info.description) > 60
                else info.description
            )
            lines.append(
                f"  [yellow]/{escape_markup(name)}[/] - {escape_markup(desc)}"
            )

        lines.append("")
        self._update_info_content("\n".join(lines))

    def _handle_journal_command(self, args: str) -> None:
        """Handle /journal command - control journal mode."""
        args = args.strip().lower()
        if args == "on":
            self.agent.journal_mode = True
            self._update_status(self.status)
            self._update_info_content(
                "[green]Journal mode enabled[/]\n"
                "Context will be compacted via reflection before each new message."
            )
        elif args == "off":
            self.agent.journal_mode = False
            self._update_status(self.status)
            self._update_info_content("[yellow]Journal mode disabled[/]")
        elif args == "last":
            # Journal the most recent tool-using turn via the agent queue
            asyncio.create_task(
                self.agent.add_message("/journal last", kind="journal_last")
            )
        elif args == "all":
            # Iteratively journal all tool-using turns via the agent queue
            asyncio.create_task(
                self.agent.add_message("/journal all", kind="journal_all")
            )
        elif args == "status" or not args:
            status = "on" if self.agent.journal_mode else "off"
            color = "green" if self.agent.journal_mode else "yellow"
            self._update_info_content(f"[{color}]Journal mode: {status}[/]")
        else:
            self._update_info_content(
                "[red]Usage: /journal [on|off|last|all|status][/]\n"
                "  [yellow]/journal on[/] - Enable context compaction\n"
                "  [yellow]/journal off[/] - Disable context compaction\n"
                "  [yellow]/journal last[/] - Journal the most recent tool-using turn\n"
                "  [yellow]/journal all[/] - Journal all tool-using turns iteratively\n"
                "  [yellow]/journal status[/] - Show current state"
            )

    def _handle_remove_reasoning_command(self, args: str) -> None:
        """Handle /remove-reasoning command - control reasoning token stripping."""
        args = args.strip().lower()
        if args == "on":
            self.agent.remove_reasoning = True
            self._update_info_content(
                "[green]Remove reasoning enabled[/]\n"
                "Reasoning tokens will be stripped between turns to save context."
            )
        elif args == "off":
            self.agent.remove_reasoning = False
            self._update_info_content(
                "[yellow]Remove reasoning disabled[/]\n"
                "Reasoning tokens will be preserved between turns."
            )
        elif not args:
            # Toggle
            self.agent.remove_reasoning = not self.agent.remove_reasoning
            status = "on" if self.agent.remove_reasoning else "off"
            asyncio.create_task(self._write_system(f"Remove reasoning: {status}"))
        else:
            status = "on" if self.agent.remove_reasoning else "off"
            self._update_info_content(
                f"[red]Usage: /remove-reasoning [on|off][/]\n"
                f"  [yellow]/remove-reasoning on[/] - Strip reasoning between turns\n"
                f"  [yellow]/remove-reasoning off[/] - Preserve reasoning between turns\n"
                f"  Current: {status}"
            )

    def _handle_devel_command(self, args: str) -> None:
        """Handle /devel command - toggle devel mode (show/hide devel-group tools)."""
        args = args.strip().lower()
        if args == "on":
            self.agent.set_devel_mode(True)
            self._update_info_content(
                "[green]Devel mode enabled[/]\n"
                "Devel-group tools (e.g. TUI viewer) are now visible to the AI."
            )
        elif args == "off":
            self.agent.set_devel_mode(False)
            self._update_info_content(
                "[yellow]Devel mode disabled[/]\n"
                "Devel-group tools are now hidden from the AI."
            )
        elif args == "status" or not args:
            status = "on" if self.agent.devel_mode else "off"
            color = "green" if self.agent.devel_mode else "yellow"
            self._update_info_content(f"[{color}]Devel mode: {status}[/]")
        else:
            status = "on" if self.agent.devel_mode else "off"
            self._update_info_content(
                f"[red]Usage: /devel [on|off|status][/]\n"
                f"  [yellow]/devel on[/] - Show devel-group tools to the AI\n"
                f"  [yellow]/devel off[/] - Hide devel-group tools from the AI\n"
                f"  [yellow]/devel status[/] - Show current state\n"
                f"  Current: {status}"
            )

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
            self._update_info_content(
                f"[{color}]Spinner: {label}[/]"
            )
        elif args == "status" or not args:
            speed = self._spinner_speed
            label = {
                "fast": "Fast (100ms, braille)",
                "slow": "Slow (250ms, classic)",
                "off": "Off",
            }.get(speed, speed)
            color = {"fast": "green", "slow": "yellow", "off": "red"}.get(speed, "green")
            self._update_info_content(f"[{color}]Spinner: {label}[/]")
        else:
            speed = self._spinner_speed
            self._update_info_content(
                f"[red]Usage: /spinner [fast|slow|off|status][/]\n"
                f"  [yellow]/spinner fast[/] - Braille spinner, 100ms (default)\n"
                f"  [yellow]/spinner slow[/] - Classic spinner, 250ms (slow links)\n"
                f"  [yellow]/spinner off[/] - No spinner\n"
                f"  [yellow]/spinner status[/] - Show current speed\n"
                f"  Current: {speed}"
            )

    def _handle_upgrade_command(self, args: str) -> None:
        """Handle /upgrade command - check for updates and apply or copy."""
        from agent13.updater import (
            perform_update,
            fetch_latest_release,
            _is_newer,
            _build_manual_command,
            _write_last_check,
        )
        from agent13 import __version__
        from datetime import datetime, timezone

        args = args.strip().lower()
        copy_mode = "--copy" in args or "copy" in args

        # Check for update first
        release = fetch_latest_release()
        if release is None:
            self._update_info_content(
                "[red]Could not reach GitHub releases API.[/]"
            )
            return

        now = datetime.now(timezone.utc)
        _write_last_check(now)

        remote_tag = release["tag_name"]
        if not _is_newer(remote_tag, __version__):
            self._update_info_content(
                f"[green]Already on latest version ({__version__}).[/]"
            )
            return

        wheel_url = release.get("wheel_url", "")
        manual_cmd = _build_manual_command(wheel_url) if wheel_url else ""

        if copy_mode:
            # Copy the manual command to clipboard
            if not manual_cmd:
                self._update_info_content(
                    f"[red]No wheel asset found for {remote_tag}. "
                    f"Cannot build install command.[/]"
                )
                return
            self.copy_to_clipboard(manual_cmd)
            self._update_info_content(
                f"[green]Copied to clipboard:[/]\n"
                f"  [dim]{manual_cmd}[/]"
            )
        else:
            # Perform the upgrade
            self._update_info_content(
                f"[yellow]Checking for updates...[/]\n"
                f"  Update available: {remote_tag} (you have {__version__})\n"
                f"  Downloading and installing..."
            )
            success, message = perform_update()
            if success:
                self._update_info_content(
                    f"[green]✓ {message}[/]"
                )
            else:
                parts = [f"[red]✗ {message}[/]"]
                if manual_cmd:
                    parts.append(
                        f"\n  [dim]Manual command: {manual_cmd}[/]"
                    )
                self._update_info_content(" ".join(parts))

    def _handle_clipboard_command(self, args: str) -> None:
        """Handle /clipboard command - show or set clipboard method."""
        from agent13.clipboard import VALID_METHODS

        args = args.strip().lower()

        if not args:
            # Show current method
            method = self._clipboard_method
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
        self._clipboard_method = args

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
                    new_content = content.rstrip() + f"\n\n[clipboard]\nmethod = \"{args}\"\n"
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
        self._update_info_content(
            f"[green]Clipboard method: {args}[/]\n"
            f"  {desc}"
        )

    def _handle_save_command(self, args: str) -> None:
        """Handle /save command - save context to file."""
        args = args.strip()

        # Parse args: /save <name> [-y]
        parts = args.split()
        if not parts:
            self._update_info_content(
                "[red]Usage: /save <name> [-y][/]\n"
                "  [yellow]/save mycontext[/] - Save to ./agent13/saves/mycontext.ctx\n"
                "  [yellow]/save mycontext -y[/] - Overwrite without prompting"
            )
            return

        name = parts[0]
        force = "-y" in parts

        # Validate name
        if not name or name.startswith("-"):
            self._update_info_content("[red]Please provide a valid save name[/]")
            return

        saves_dir = get_saves_dir()
        path = saves_dir / f"{name}.ctx"

        # Check for overwrite
        if path.exists() and not force:
            self._update_info_content(
                f"[yellow]File already exists: {path}[/]\n"
                f"Use [yellow]/save {name} -y[/] to overwrite"
            )
            return

        # Save
        try:
            save_context(self.agent, path)
            self._update_info_content(f"[green]Saved context to {path}[/]")
        except Exception as e:
            self._update_info_content(
                f"[red]Failed to save: {escape_markup(str(e))}[/]"
            )

    def _handle_load_command(self, args: str) -> None:
        """Handle /load command - load context from file.

        Validates the path synchronously, then queues a kind='load'
        item for safe deferred processing. The agent processes it at a
        safe boundary between items, replacing messages without race risk.
        """
        args = args.strip()

        if not args:
            self._update_info_content(
                "[red]Usage: /load <name>[/]\n"
                "  [yellow]/load mycontext[/] - Load from ./agent13/saves/mycontext.ctx"
            )
            return

        name = args.split()[0]
        saves_dir = get_saves_dir()
        path = saves_dir / f"{name}.ctx"

        if not path.exists():
            self._update_info_content(
                f"[red]Save file not found: {path}[/]\n"
                f"Use [yellow]/save {name}[/] to create it"
            )
            return

        # Queue the load for safe deferred processing (pass path as string)
        asyncio.create_task(self.agent.request_load(str(path)))

    def _load_auto_save(self) -> None:
        """Load the latest auto-save if available."""
        path = find_latest_auto_save()
        if path is None:
            self._update_info_content(
                "[dim]No auto-save found, starting fresh session[/]"
            )
            return

        success, message, incomplete = load_context(self.agent, path)
        if success:
            # Reset TUI token counters
            self.prompt_tokens = self.agent.prompt_tokens
            self.completion_tokens = self.agent.completion_tokens
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
            self._update_info_content(f"[yellow]Could not load auto-save: {message}[/]")

    def on_token_message(self, message: TokenMessage) -> None:
        """Enqueue token for sequential processing by _process_tokens.

        This handler is synchronous (as required by Textual message handlers),
        so it only enqueues. All async widget operations happen in _process_tokens
        which processes tokens one at a time, guaranteeing ordering.
        """
        # Discard stale tokens from previous streaming sessions
        if message.generation != self._stream_generation:
            return
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
                first_token_time=self._first_token_time,
                last_token_time=self._last_token_time,
                token_count=self._token_count,
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

            if isinstance(message, ToolCallMessage):
                await self._handle_tool_call(message)
                continue

            if isinstance(message, ToolResultMessage):
                await self._handle_tool_result(message)
                continue

            if isinstance(message, SystemQueueMessage):
                await self._write_system(message.text, escape_text=message.escape_text)
                continue

            # It's a TokenMessage
            await self._handle_token(message)

    async def _handle_token(self, message: TokenMessage) -> None:
        """Process a single token message sequentially."""
        if message.is_final:
            # Streaming complete
            if is_debug_enabled():
                log_tps_stream_end(
                    reason="is_final",
                    first_token_time=self._first_token_time,
                    last_token_time=self._last_token_time,
                    token_count=self._token_count,
                    pending_token_usage=None,
                )
            self._streaming = False
            self.update_status()
            await self._finalize_streaming()
        else:
            # Track timing for TPS calculation
            now = time.time()
            if self._first_token_time is None:
                self._first_token_time = now
                if is_debug_enabled():
                    log_tps_first_token(token_count=self._token_count)
            # Always update last token time for accurate streaming window
            self._last_token_time = now
            self._token_count += 1

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
                else:
                    # Append to content widget
                    await self._append_to_content(message.text)

    async def _handle_tool_call(self, message: ToolCallMessage) -> None:
        """Process a tool call message: finalize streaming, then write tool call.

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
        await self._write_tool_call(message.name, message.arguments)

    async def _handle_tool_result(self, message: ToolResultMessage) -> None:
        """Process a tool result message: write the result widget."""
        # Log tool result with timing state (stream will resume after this)
        if is_debug_enabled():
            log_tps_event(
                "tool_result",
                {
                    "tool_name": message.name,
                    "first_token_time": self._first_token_time,
                    "last_token_time": self._last_token_time,
                    "token_count": self._token_count,
                },
            )
        await self._write_tool_result(message.name, message.result)

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
            title, collapsed=(title == "Reflecting")
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
                title, collapsed=(title == "Reflecting")
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
            # Change "Reflecting" to "Reflection", "Thinking" to "Thoughts" when done
            if self._streaming_reasoning_widget._title == "Reflecting":
                self._streaming_reasoning_widget.set_title("Reflection")
            elif self._streaming_reasoning_widget._title == "Thinking":
                self._streaming_reasoning_widget.set_title("Thoughts")
            await self._streaming_reasoning_widget.finalize()
            # Remove widget if it has no meaningful content (title is pre-rendered)
            content = getattr(self._streaming_reasoning_widget, "_content", "")
            if not content or not content.strip():
                # Remove the empty reasoning widget
                await self._chat.remove(self._streaming_reasoning_widget)
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
                first_token_time=self._first_token_time,
                last_token_time=self._last_token_time,
                token_count=self._token_count,
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

            # Always check current context FIRST - this is the source of truth
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

            # If we're already in completion mode, check if the text has changed
            # (user typed something after the last completion)
            # This is needed because TextArea stops key events from bubbling,
            # so we don't receive space/character events to reset completion state
            if self._completion_matches and self._completion_text != current_text:
                # Text changed - reset and start fresh
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

                matches = self._get_completions_for_context(
                    completion_type, partial, current_text
                )
                if matches:
                    if len(matches) == 1:
                        # Single match - complete immediately
                        self._apply_completion(
                            input_field, matches[0], start_loc, end_loc
                        )
                        self._reset_completion_state()
                        # Only hide info-pane if completions were shown there
                        if self._info_pane_mode is None:
                            self._hide_info_pane()
                    else:
                        # Multiple matches - enter completion mode
                        self._completion_matches = matches
                        self._completion_index = 0
                        self._completion_prefix = partial
                        self._completion_start = start_loc
                        self._completion_end = end_loc
                        self._apply_completion(
                            input_field, matches[0], start_loc, end_loc
                        )
                        self._completion_text = (
                            input_field.text
                        )  # Save text AFTER completion for change detection
                        self._show_completions(matches, matches[0])

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

    def action_scroll_chat_up(self) -> None:
        """Scroll up and disable auto-scroll."""
        self._chat.scroll_relative(y=-5, animate=False)
        self._chat.anchor(False)  # Explicitly clear anchor
        self.notify("Auto-scroll disabled")

    def action_scroll_chat_down(self) -> None:
        """Scroll down and re-enable auto-scroll if at bottom."""
        self._chat.scroll_relative(y=5, animate=False)
        if self._is_at_bottom():
            self._chat.anchor()
            self.notify("Auto-scroll enabled")

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

        method = self._clipboard_method
        if method == "system":
            if not _copy(text, method="system"):
                self._write_error(
                    "Clipboard (system) failed. "
                    "Install xclip, wl-copy, or try /clipboard osc52"
                )
        else:
            # OSC 52 — delegate to Textual's built-in
            _copy(text, method="osc52", osc52_handler=super().copy_to_clipboard)

    def action_copy_selection(self) -> None:
        """Copy rendered selection to clipboard (Ctrl+Shift+C).

        Copies the selected rendered text and keeps the selection visible.
        """
        # Check chat area selection
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)
            self.notify(f"Copied {len(text)} chars", title="Copied")
            # Keep selection visible
            return

        # Check input field selection
        input_field = self.query_one("#input-field", ChatTextArea)
        if input_field.selected_text:
            self.copy_to_clipboard(input_field.selected_text)
            self.notify(
                f"Copied {len(input_field.selected_text)} chars", title="Copied"
            )

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

        # Ensure agent isn't stuck in paused state (e.g. from error path
        # setting pause_state=PAUSED before the interrupt). Escape means
        # "stop and redirect", not "pause to resume later".
        if self.agent.is_paused or self.agent.is_pausing:
            self.agent.resume()

        # Clear input field to prevent keystroke concatenation (Bug #5)
        try:
            input_field = self.query_one("#input-field", ChatTextArea)
            if input_field.text:
                input_field.clear()
            input_field.focus()
        except Exception:
            pass

        # Restart the agent loop so it can process future messages
        self._agent_running = True
        self._agent_task = asyncio.create_task(self.agent.run())

        # Show interrupt message
        await self._write_system("[yellow]⚠ Interrupted by user[/]", escape_text=False)

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
        """Handle Ctrl+C: copy selection, clear input, interrupt agent, or quit."""
        # If there's a selection in the chat area, copy it
        if self.screen.get_selected_text():
            self.action_copy_selection()
            return

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
        self.agent.stop()
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
    parser.add_argument(
        "--pretty",
        choices=["on", "off"],
        default="on",
        help="Enable/disable markdown rendering. Default: on",
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
    prompt_manager = PromptManager()
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
        pretty=args.pretty == "on",
        debug=args.debug,
        tool_response_format=args.tool_response,
        prompt_manager=prompt_manager,
        connect_mcp=args.mcp,
        skill_manager=skill_manager,
        system_prompt=system_prompt,
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
