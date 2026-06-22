"""Agent core class - event-driven agent implementation."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from agent13.events import AgentEvent, AgentEventData, EventHandler
from agent13.journal import JournalManager
from agent13.message_history import MessageHistory
from agent13.prompts import DEFAULT_PROMPT
from agent13.queue import AgentQueue, QueueItem
from agent13.llm import (
    append_assistant_message,
    categorize_error,
    detect_tool_calls_in_reasoning,
    LLMError,
)
from agent13.debug_log import (
    log_error,
    log_user_message,
    log_queue_start,
    log_queue_complete,
    log_queue_interrupt,
    log_assistant_response,
    log_tool_call,
    log_tool_result,
    log_journal_debug,
    # TPS debug logging
    is_debug_enabled,
    log_tps_event,
)

if TYPE_CHECKING:
    from agent13.mcp import MCPManager


# Configuration constants
REASONING_TOOL_CALL_NOTIFICATION_DURATION = 30.0  # seconds


def _sanitize_json_args(tool_name: str, raw: str) -> str:
    """Validate and re-serialise tool call arguments for message history.

    Streaming may accumulate malformed JSON (truncated responses, bad escapes,
    unterminated strings).  Storing raw corrupts every subsequent API call —
    the backend re-parses the full message history and 500s on bad JSON.

    Round-trip through json.loads/dumps to guarantee valid JSON in storage.
    Falls back to '{}' on any parse failure so the message sequence is
    always API-safe, independent of tool execution error handling.
    """
    try:
        return json.dumps(json.loads(raw)) if raw else "{}"
    except (json.JSONDecodeError, TypeError):
        log_error(
            Exception("malformed tool call arguments"),
            {"context": "sanitize_json_args", "name": tool_name,
             "arguments": raw[:200] if raw else ""},
        )
        return "{}"


@dataclass
class ToolStats:
    """Track tool usage statistics."""

    # Per-tool counts
    calls: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    # Mode tracking (for tools with 'mode' parameter)
    modes: dict[str, dict[str, int]] = field(default_factory=dict)
    mode_successes: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, name: str, arguments: dict, result: str) -> None:
        """Record a tool call and its result."""
        # Increment call count
        self.calls[name] = self.calls.get(name, 0) + 1

        # Check for success (no error in result)
        try:
            result_data = json.loads(result)
            is_success = "error" not in result_data
        except (json.JSONDecodeError, TypeError):
            is_success = True  # Non-JSON result is success

        if is_success:
            self.successes[name] = self.successes.get(name, 0) + 1

        # Track mode if present
        if "mode" in arguments:
            if name not in self.modes:
                self.modes[name] = {}
            mode = arguments["mode"]
            self.modes[name][mode] = self.modes[name].get(mode, 0) + 1
            # Track mode successes
            if is_success:
                if name not in self.mode_successes:
                    self.mode_successes[name] = {}
                self.mode_successes[name][mode] = (
                    self.mode_successes[name].get(mode, 0) + 1
                )

    @property
    def total_calls(self) -> int:
        """Total number of tool calls."""
        return sum(self.calls.values())

    @property
    def total_successes(self) -> int:
        """Total number of successful tool calls."""
        return sum(self.successes.values())

    def reset(self) -> None:
        """Reset all statistics."""
        self.calls.clear()
        self.successes.clear()
        self.modes.clear()
        self.mode_successes.clear()

    def summary(self) -> dict:
        """Get a summary for display."""
        return {
            "total": self.total_calls,
            "successes": self.total_successes,
            "by_tool": {
                name: {
                    "calls": self.calls.get(name, 0),
                    "successes": self.successes.get(name, 0),
                    "modes": self.modes.get(name, {}),
                    "mode_successes": self.mode_successes.get(name, {}),
                }
                for name in self.calls
            },
        }


# Lazy import to avoid issues if MCP SDK is not installed
def _get_mcp_manager_class():
    """Lazy import of MCPManager to avoid import errors if MCP SDK is not installed."""
    from agent13.mcp import MCPManager

    return MCPManager


class AgentStatus(Enum):
    """Agent status states."""

    INITIALISING = "initialising"
    IDLE = "idle"
    WAITING = "waiting"
    THINKING = "thinking"
    PROCESSING = "processing"
    TOOLING = "tooling"
    JOURNALING = "journaling"
    PAUSED = "paused"


class PauseState(Enum):
    """Pause state machine — single source of truth.

    Replaces the previous _paused/_pausing booleans which admitted
    an invalid state (_pausing=True, _paused=True) and required
    duplicated tracking in the TUI.
    """

    RUNNING = "running"  # Normal operation
    PAUSING = "pausing"  # Pause requested, not yet at safe point
    PAUSED = "paused"  # Paused at safe point


class SpinnerSpeed(Enum):
    """Spinner animation speed — single source of truth.

    fast: 100ms per frame (4 rev/sec with 4-frame spinner)
    slow: 250ms per frame (1 rev/sec with 4-frame spinner)
    off:  no spinner animation
    """

    FAST = 0.1
    SLOW = 0.25
    OFF = 0


class Agent:
    """Event-driven agent that processes messages from a queue.

    The agent emits events during processing that can be handled by UI layers.
    Events include:
    - STARTED/STOPPED: Lifecycle events
    - QUEUE_UPDATE: Queue state changes
    - USER_MESSAGE: User message added
    - ASSISTANT_TOKEN: Streaming tokens
    - ASSISTANT_COMPLETE: Response finished
    - TOOL_CALL/TOOL_RESULT: Tool execution
    - STATUS_CHANGE: Processing state changes
    - ERROR: Error events

    Usage:
        agent = Agent(client, model="devstral")

        @agent.on_event
        async def handler(event: AgentEventData):
            if event.event == AgentEvent.ASSISTANT_TOKEN:
                print(event.text, end="")

        await agent.add_message("Hello!")
        await agent.run()
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        queue: AgentQueue = None,
        system_prompt: str = None,
        messages: list[dict] = None,
        tools: list[dict] = None,
        execute_tool: Callable[[str, dict], str]
        | Callable[[str, dict], Awaitable[str]] = None,
        response_format: dict = None,
        journal_mode: bool = False,
        send_reasoning: bool = False,
        remove_reasoning: bool = False,
        devel_mode: bool = False,
        skills_mode: bool = False,
    ):
        """Initialize the agent.

        Args:
            client: AsyncOpenAI client for API calls
            model: Model name to use
            queue: AgentQueue for message processing (created if None)
            system_prompt: System prompt text
            messages: Initial message history
            tools: List of tool schemas for function calling
            execute_tool: Function to execute tools (name, args) -> result.
                         Can be sync or async.
            response_format: Optional response format (e.g., {"type": "json_object"})
            journal_mode: Enable context compaction via journal summaries.
            send_reasoning: If True, include reasoning_content in message history.
            remove_reasoning: If True, strip reasoning tokens between turns.
                             Defaults to False (preserve reasoning between turns).
            devel_mode: If True, include tools in the "devel" group (e.g. TUI viewer).
            skills_mode: If True, include tools in the "skills" group (e.g. skill tool).
        """
        self.client = client
        self.model = model
        self.queue = queue or AgentQueue()
        self.system_prompt = system_prompt or DEFAULT_PROMPT
        self.history = MessageHistory(messages)
        self.tools = tools or []
        self.execute_tool = execute_tool
        self.response_format = response_format
        self.journal_mode = journal_mode
        self.send_reasoning = send_reasoning
        self.remove_reasoning = remove_reasoning
        self._devel_mode = devel_mode
        self._skills_mode = skills_mode
        self.execute_tool = execute_tool
        # Available models list (populated by caller or set_client)
        self.available_models: list[str] = []
        self.response_format = response_format
        self.journal_mode = journal_mode
        self.send_reasoning = send_reasoning
        self.remove_reasoning = remove_reasoning

        self._handlers: list[EventHandler] = []
        self._running = False
        self._stop_event = asyncio.Event()
        self._cancel_requested = False  # Set by _process_item on CancelledError
        self._status = AgentStatus.INITIALISING

        # Pause/resume state (single source of truth — PauseState enum)
        self._pause_state = PauseState.RUNNING
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default

        # Token usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        # Tool usage statistics
        self.tool_stats = ToolStats()

        # MCP manager (lazy initialization)
        self._mcp: Optional["MCPManager"] = None
        self._mcp_server_configs: list = []  # Set via set_mcp_servers()

        # Incomplete turn tracking (set when loading a saved incomplete context)
        self._incomplete_turn_loaded: bool = False

        # Journal manager for context compaction
        self.journal = JournalManager(
            history=self.history,
            stream_fn=self._stream_and_emit,
            emit_fn=self.emit,
            set_status_fn=self._set_status,
            get_status_fn=lambda: self._status,
            get_prompt_tokens_fn=lambda: self.prompt_tokens,
            is_interrupted_fn=lambda: self.queue.has_interrupt,
            journal_mode_fn=lambda: self.journal_mode,
            status_journaling=AgentStatus.JOURNALING,
            status_idle=AgentStatus.IDLE,
        )

    @property
    def messages(self) -> list[dict]:
        """Message history — delegates to MessageHistory."""
        return self.history.messages

    @messages.setter
    def messages(self, value: list[dict]):
        self.history.messages = value

    def on_event(self, handler: EventHandler) -> EventHandler:
        """Register an event handler.

        Can be used as a decorator:
            @agent.on_event
            async def handler(event):
                ...

        Or as a method:
            agent.on_event(my_handler)

        Args:
            handler: Function that receives AgentEventData

        Returns:
            The handler (for decorator chaining)
        """
        self._handlers.append(handler)
        return handler

    async def emit(self, event: AgentEvent, data: dict = None) -> None:
        """Emit an event to all registered handlers.

        Args:
            event: The event type
            data: Optional data dictionary
        """
        event_data = AgentEventData(event=event, data=data or {})

        for handler in self._handlers:
            try:
                result = handler(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except BaseException as e:
                # Don't let handler errors (including CancelledError from
                # UI framework) crash the agent loop.  CancelledError is a
                # BaseException in Python 3.9+, so catching Exception alone
                # lets it escape and kill run().
                print(
                    f"Error in event handler: {e}, event_data: {event_data}, result: {result}"
                )

    async def _stream_and_emit(
        self,
        messages: list[dict],
        *,
        source: str = "assistant",
        tool_choice: str = "auto",
    ):
        """Stream an LLM response with centralized event handling.

        Emits STREAM_START and TOKEN_USAGE events centrally so they can't
        be accidentally skipped by callers. Yields all other events
        (content, reasoning, tool_call, tool_calls_complete) for
        caller-specific handling.

        This is the DRY enforcement: both _llm_turn and
        _reflect_on_tool_use use this, so token_usage is always
        stored on self and emitted — it can't be forgotten.

        Args:
            messages: Messages to send to the LLM.
            source: Source label for STREAM_START event
                (e.g. "assistant", "reflection").
            tool_choice: Tool choice mode ("auto", "none").

        Yields:
            (event_type, data) tuples for all non-token_usage events.
        """
        from agent13.llm import stream_response_with_tools

        # Emit STREAM_START for TUI to reset per-stream timing
        await self.emit(AgentEvent.STREAM_START, {"source": source})

        # Get all tools including MCP tools
        tools = await self.get_all_tools()

        async for event_type, data in stream_response_with_tools(
            self.client,
            self.model,
            messages,
            self.system_prompt,
            tools,
            tool_choice=tool_choice,
        ):
            if event_type == "token_usage":
                # Store as source of truth for context size
                self.prompt_tokens = data.get("prompt_tokens", 0)
                self.completion_tokens = data.get("completion_tokens", 0)
                self.total_tokens = data.get("total_tokens", 0)
                await self.emit(AgentEvent.TOKEN_USAGE, data)
                if is_debug_enabled():
                    log_tps_event(
                        "agent_token_usage",
                        {
                            "prompt_tokens": data.get("prompt_tokens"),
                            "completion_tokens": data.get("completion_tokens"),
                            "total_tokens": data.get("total_tokens"),
                        },
                    )
                # Handled centrally — don't yield to caller
                continue

            yield event_type, data

    async def add_message(
        self,
        text: str,
        priority: bool = False,
        interrupt: bool = False,
        kind: str = "prompt",
    ) -> int:
        """Add a user message to the queue.

        Args:
            text: The message text
            priority: Whether to process with high priority (front of queue)
            interrupt: Whether to interrupt the agent loop (implies priority)
            kind: Item kind - "prompt", "journal_last", or "journal_all"

        Returns:
            The queue item ID
        """
        item_id = self.queue.add(
            text, priority=priority, interrupt=interrupt, kind=kind
        )

        # Log user message
        log_user_message(text, priority=priority, interrupt=interrupt, item_id=item_id)

        await self.emit(
            AgentEvent.USER_MESSAGE,
            {
                "text": text,
                "priority": priority,
                "interrupt": interrupt,
                "item_id": item_id,
            },
        )

        await self._emit_queue_update()

        return item_id

    async def run(self) -> None:
        """Run the agent loop, processing messages from the queue.

        This method runs continuously until stop() is called.
        Raises asyncio.CancelledError when interrupted by the user.
        """
        self._running = True
        self._stop_event.clear()

        # Clear any stale pause state unconditionally on (re)start
        was_paused = self._pause_state == PauseState.PAUSED
        self._pause_state = PauseState.RUNNING
        self._pause_event.set()
        if was_paused:
            await self.emit(AgentEvent.RESUMED, {})

        # Transition from INITIALISING to IDLE
        await self._set_status(AgentStatus.IDLE)
        await self.emit(AgentEvent.STARTED, {})

        try:
            while self._running and not self._stop_event.is_set():
                # Check if paused and wait
                await self._wait_if_paused()

                # Check if we should stop after potential pause
                if not self._running or self._stop_event.is_set():
                    break

                # Get next item from queue
                current = self.queue.get_next()

                if current:
                    await self._process_item(current)

                    # If a CancelledError was absorbed in _process_item,
                    # exit the loop cleanly so the ESC handler can restart
                    # with a fresh task.
                    if self._cancel_requested:
                        self._cancel_requested = False
                        break
                else:
                    # Nothing to do, wait briefly
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # CancelledError hit between items (not inside _process_item).
            # Repair message history and emit event, then re-raise so
            # finally block can clean up.
            current_id = self.queue.current.id if self.queue.current else None
            log_queue_interrupt(current_id)
            self.history.repair_interrupted()
            if self.queue.current:
                self.queue.complete_current()
                log_queue_complete(current_id, "interrupted")
            await self.emit(AgentEvent.INTERRUPTED, {})
            raise
        finally:
            # Only cleanup MCP if we're truly stopping (stop() was called).
            # If _running is still True, we were interrupted (ESC) and will
            # be restarted, so we keep MCP connections alive.
            if not self._running and self._mcp:
                await self._mcp.cleanup()
            # Reset status to IDLE so is_idle reflects reality after stop/interrupt.
            # Without this, status stays as e.g. TOOLING after a cancel.
            self._status = AgentStatus.IDLE
            await self.emit(AgentEvent.STOPPED, {})

    def stop(self) -> None:
        """Signal the agent to stop processing."""
        self._running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused so run() can exit

    def pause(self) -> bool:
        """Request the agent to pause.

        Returns True if pause was requested, False if already paused/pausing.
        The agent will pause at the next safe point (between tool calls).
        """
        if self._pause_state != PauseState.RUNNING:
            return False
        self._pause_state = PauseState.PAUSING
        self._pause_event.clear()  # Block the pause event
        return True

    def resume(self) -> bool:
        """Resume the agent from a paused or pausing state.

        Returns True if resumed, False if not paused/pausing.
        Handles both PAUSED (at safe point) and PAUSING (pause requested
        but not yet effective) — cancelling the pause request before it
        takes effect.
        """
        if self._pause_state == PauseState.RUNNING:
            return False
        self._pause_state = PauseState.RUNNING
        self._pause_event.set()  # Unblock the pause event
        return True

    @property
    def pause_state(self) -> PauseState:
        """Get the current pause state — single source of truth for UIs."""
        return self._pause_state

    @property
    def is_paused(self) -> bool:
        """Check if the agent is paused (at a safe point)."""
        return self._pause_state == PauseState.PAUSED

    @property
    def is_pausing(self) -> bool:
        """Check if pausing (pause requested but not yet effective)."""
        return self._pause_state == PauseState.PAUSING

    @property
    def is_idle(self) -> bool:
        """Check if the agent is fully idle (not processing, not paused).

        Single source of truth for both REPL and TUI.
        """
        return (
            self._pause_state == PauseState.RUNNING
            and self._status == AgentStatus.IDLE
        )

    @property
    def has_incomplete_turn(self) -> bool:
        """Check if the agent has an incomplete turn from a loaded context."""
        return self._incomplete_turn_loaded

    def mark_incomplete_turn(self, incomplete: bool) -> None:
        """Mark that the agent has an incomplete turn (called on load)."""
        self._incomplete_turn_loaded = incomplete

    async def continue_incomplete_turn(self) -> bool:
        """Continue an incomplete turn by executing pending tools or calling LLM.

        This is called after loading a context with an incomplete turn.
        It handles two cases:
        1. Last message is assistant with tool_calls -> execute pending tools
        2. Last message is tool -> call LLM to process results

        Returns:
            True if continuation was started, False if no continuation needed.
        """
        if not self._incomplete_turn_loaded:
            return False

        # Clear the flag first
        self._incomplete_turn_loaded = False

        # Check the state of messages
        pending_tools = self.history.get_pending_tool_calls()

        if pending_tools:
            # Case 1: We have pending tool calls to execute
            # Execute each pending tool
            for tc in pending_tools:
                if not self._running:
                    break

                name = tc["name"]
                args_str = tc["arguments"]

                try:
                    import json

                    arguments = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    arguments = {}

                # Emit tool call event
                log_tool_call(name, arguments)
                await self.emit(
                    AgentEvent.TOOL_CALL,
                    {
                        "name": name,
                        "arguments": arguments,
                    },
                )

                # Execute the tool
                result = await self._execute_tool_async(name, arguments)

                # Record tool statistics
                self.tool_stats.record(name, arguments, result)

                # Emit result
                log_tool_result(name, result)
                await self.emit(
                    AgentEvent.TOOL_RESULT,
                    {
                        "name": name,
                        "result": result,
                    },
                )

                # Add tool result to messages
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )

            # Now call LLM to continue
            await self._llm_turn()
            return True

        elif self.history.has_incomplete_turn():
            # Case 2: Last message is tool result, call LLM to process
            await self._llm_turn()
            return True

        return False

    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self._status

    async def _wait_if_paused(self) -> None:
        """Wait while the agent is paused.

        This is called at safe pause points. When pause is requested,
        this will block until resume() is called.
        """
        if self._pause_state == PauseState.PAUSING:
            # Transition to fully paused
            self._pause_state = PauseState.PAUSED
            await self._set_status(AgentStatus.PAUSED)
            await self.emit(AgentEvent.PAUSED, {})

        if self._pause_state == PauseState.PAUSED:
            await self._pause_event.wait()
            # Only transition if we woke from a resume,
            # not from a stop() — check _running to distinguish.
            if self._running:
                # Skip IDLE if there are queued items — go straight to
                # WAITING so the user never sees "ready" between resume
                # and processing the next item.
                if self.queue.pending_count > 0:
                    await self._set_status(AgentStatus.WAITING)
                else:
                    await self._set_status(AgentStatus.IDLE)
                await self.emit(AgentEvent.RESUMED, {})

    async def _execute_tool_async(self, name: str, arguments: dict) -> str:
        """Execute a tool (handles both sync and async callables).

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        # Check if this is an MCP tool
        if name.startswith("mcp://"):
            mcp = await self._ensure_mcp()
            if mcp:
                return await mcp.call_tool(name, arguments)
            return '{"error": "MCP not available"}'

        if self.execute_tool is None:
            return '{"error": "No tool executor configured"}'

        # Check if the callable is a coroutine function
        if asyncio.iscoroutinefunction(self.execute_tool):
            return await self.execute_tool(name, arguments)
        else:
            # Sync callable - run in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.execute_tool(name, arguments)
            )

    def set_mcp_servers(self, server_configs: list) -> None:
        """Set MCP server configurations (does not connect).

        Args:
            server_configs: List of MCPServerConfig objects
        """
        self._mcp_server_configs = server_configs

    async def _ensure_mcp(self):
        """Ensure MCP manager is created (does not connect to servers).

        Returns:
            MCPManager instance or None if no servers configured
        """
        if self._mcp is None and self._mcp_server_configs:
            MCPManager = _get_mcp_manager_class()
            self._mcp = MCPManager(self._mcp_server_configs)
            self._mcp.set_event_callback(self._emit_mcp_event)
        return self._mcp

    async def _emit_mcp_event(self, event: AgentEvent, data: AgentEventData) -> None:
        """Emit MCP events to registered handlers."""
        await self.emit(event, data.data)

    async def get_mcp_tools(self) -> list[dict]:
        """Get MCP tools (returns empty list if not connected).

        Returns:
            List of tool definitions in OpenAI format
        """
        mcp = await self._ensure_mcp()
        if mcp:
            return mcp.get_openai_tools()
        return []

    async def get_all_tools(self) -> list[dict]:
        """Get combined built-in and MCP tools.

        Built-in tools are already filtered via get_filtered_tools() at
        init time and when set_devel_mode() is called.  MCP tools are
        filtered per-server at registration time.  This method also
        applies the global config enabled_tools/disabled_tools filter
        to MCP tools (which weren't filtered at registration).

        Returns:
            List of all tool definitions in OpenAI format
        """
        from agent13.tools import name_matches
        from agent13.config import get_config

        config = get_config()
        enabled = config.enabled_tools
        disabled = config.disabled_tools

        all_tools = list(self.tools)  # Start with built-in tools (already filtered)
        mcp = await self._ensure_mcp()
        if mcp:
            for tool_schema in mcp.get_openai_tools():
                tool_name = tool_schema.get("function", {}).get("name", "")
                # Apply global config filter to MCP tools
                if enabled:
                    if not name_matches(tool_name, enabled):
                        continue
                elif disabled:
                    if name_matches(tool_name, disabled):
                        continue
                all_tools.append(tool_schema)
        return all_tools

    async def disconnect_mcp(self) -> bool:
        """Disconnect from all MCP servers.

        Returns:
            True if disconnected, False if MCP was not initialized
        """
        if self._mcp:
            await self._mcp.disconnect()
            return True
        return False

    @property
    def mcp(self):
        """Get the MCP manager (may be None if not initialized)."""
        return self._mcp

    async def _process_item(self, item: QueueItem) -> None:
        """Process a single queue item.

        Args:
            item: The queue item to process
        """
        await self._set_status(AgentStatus.WAITING)

        # Log queue processing start
        log_queue_start(item.text, item.id)

        # Emit event that processing is starting (for UI to show user message)
        await self.emit(
            AgentEvent.ITEM_STARTED,
            {
                "text": item.text,
                "priority": item.priority,
                "item_id": item.id,
            },
        )
        try:
            # Handle journal queue items — these run reflection/compaction
            # instead of a normal LLM turn
            if item.kind == "journal_last":
                success, message = await self.journal.journal_last_turn()
                if success:
                    await self.emit(
                        AgentEvent.JOURNAL_RESULT,
                        {
                            "success": True,
                            "message": message,
                        },
                    )
                else:
                    await self.emit(
                        AgentEvent.JOURNAL_RESULT,
                        {
                            "success": False,
                            "message": message,
                        },
                    )
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
            elif item.kind == "journal_all":
                success, message = await self.journal.journal_all()
                if success:
                    await self.emit(
                        AgentEvent.JOURNAL_RESULT,
                        {
                            "success": True,
                            "message": message,
                        },
                    )
                else:
                    await self.emit(
                        AgentEvent.JOURNAL_RESULT,
                        {
                            "success": False,
                            "message": message,
                        },
                    )
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
            elif item.kind == "clear":
                # Deferred /clear — safe at this boundary between items
                count = self.clear_messages()
                clear_widgets = (item.data or {}).get("clear_widgets", False)
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
                await self.emit(
                    AgentEvent.MESSAGES_CLEARED,
                    {
                        "count": count,
                        "clear_widgets": clear_widgets,
                    },
                )
            elif item.kind == "load":
                # Deferred /load — safe at this boundary between items
                from agent13.persistence import load_context

                success, message, incomplete = load_context(self, item.text)
                log_journal_debug("load_context", {
                    "success": success,
                    "message": message,
                    "messages_count": len(self.messages),
                    "has_tool_calls": self.history.has_tool_calls(),
                    "first_3_roles": [m.get("role") for m in self.messages[:3]] if self.messages else [],
                })
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
                await self.emit(
                    AgentEvent.CONTEXT_LOADED,
                    {
                        "success": success,
                        "message": message,
                        "incomplete": incomplete,
                    },
                )
            else:
                # Normal prompt processing
                # Retrospective compaction: journal was off during the previous
                # turn, but user has now turned it on. Apply compaction now so
                # the tool calls get summarized before the new turn starts.
                await self.journal.retrospective_compact(is_interrupt=item.interrupt)
                # Strip reasoning from previous turns when remove_reasoning is enabled.
                # In journal mode, the turn gets replaced with a summary anyway.
                # When remove_reasoning is off (default), reasoning is preserved between
                # turns for better multi-step continuity and user visibility.
                if not self.journal_mode and self.remove_reasoning:
                    self.history.strip_reasoning()

                # Add user message to history
                self.messages.append({"role": "user", "content": item.text})

                # Process with LLM (may include multiple tool call rounds)
                await self._llm_turn()

                # Stage 12: Run reflection after turn completes
                await self.journal.maybe_reflect_after_turn()

                # Mark item as complete
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()

        except BaseException as e:
            # Categorize the error for better user feedback.
            # Catch BaseException (not just Exception) so that
            # CancelledError (Python 3.9+ BaseException subclass) is
            # handled here instead of escaping to run() and killing
            # the loop.
            #
            # We do NOT re-raise CancelledError because:
            #   - The ESC interrupt handler (_interrupt_agent_loop)
            #     does its own cleanup (queue.complete_current,
            #     widget finalization) and restarts the agent loop.
            #   - If we re-raise, run()'s finally emits STOPPED which
            #     races with the restarted loop and resets agent status
            #     on the NEW task — causing "stopped" mode.
            #   - Absorbing lets the loop survive spurious cancellations
            #     (framework glitches, executor issues) transparently.
            if isinstance(e, asyncio.CancelledError):
                log_error(e, {"context": "process_item_cancelled",
                              "item_id": item.id})
                # Repair message history (add [Interrupted] assistant
                # message if needed) so role alternation is preserved.
                self.history.repair_interrupted()
                await self.emit(
                    AgentEvent.ERROR,
                    {
                        "message": "Processing cancelled",
                        "error_type": "cancelled",
                        "exception": e,
                    },
                )
                self.queue.complete_current()
                log_queue_complete(item.id, "interrupted")
                self._cancel_requested = True
            else:
                llm_error = (
                    categorize_error(e) if not isinstance(e, LLMError) else e
                )
                log_error(e, {"context": "process_item", "item_id": item.id})
                await self.emit(
                    AgentEvent.ERROR,
                    {
                        "message": str(llm_error),
                        "error_type": llm_error.error_type,
                        "exception": e,
                    },
                )
                self.queue.complete_current()
                log_queue_complete(item.id, "error")

        await self._set_status(AgentStatus.IDLE)

    async def _llm_turn(self) -> None:
        """Execute one LLM turn (may include multiple tool call rounds).

        A turn continues until the LLM responds without tool calls.
        Uses streaming for all phases to capture reasoning tokens.
        """
        if is_debug_enabled():
            log_tps_event("agent_llm_turn_start", {"note": "Starting new LLM turn"})
        while self._running:
            try:
                # Stream response, capturing reasoning and tool calls
                content = ""
                reasoning = ""
                tool_calls = None

                if is_debug_enabled():
                    log_tps_event(
                        "agent_stream_start",
                        {"note": "Starting _stream_and_emit"},
                    )

                # Track first tokens for status transitions
                _first_reasoning = True
                _first_content = True

                # Stream via _stream_and_emit which handles STREAM_START
                # and TOKEN_USAGE centrally (DRY).
                async for event_type, data in self._stream_and_emit(
                    self.messages,
                    source="assistant",
                ):
                    if event_type == "content":
                        content += data
                        # Transition to PROCESSING on first content token
                        if _first_content:
                            _first_content = False
                            await self._set_status(AgentStatus.PROCESSING)
                        await self.emit(
                            AgentEvent.ASSISTANT_TOKEN,
                            {
                                "text": data,
                            },
                        )
                    elif event_type == "reasoning":
                        reasoning += data
                        # Transition to THINKING on first reasoning token
                        if _first_reasoning:
                            _first_reasoning = False
                            await self._set_status(AgentStatus.THINKING)
                        await self.emit(
                            AgentEvent.ASSISTANT_REASONING,
                            {
                                "text": data,
                            },
                        )
                    elif event_type == "tool_calls_complete":
                        tool_calls = data["tool_calls"]
                        # Transition to TOOLING state when tools are about to execute
                        await self._set_status(AgentStatus.TOOLING)
                        if is_debug_enabled():
                            log_tps_event(
                                "agent_tool_calls_complete",
                                {
                                    "tool_count": len(tool_calls),
                                    "tool_names": [tc.get("name") for tc in tool_calls],
                                },
                            )

                # Handle tool calls if present
                if tool_calls:
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_tool_calls_detected",
                            {
                                "tool_count": len(tool_calls),
                                "content_length": len(content),
                                "reasoning_length": len(reasoning),
                                "note": "NO ASSISTANT_COMPLETE will be emitted - loop will continue",
                            },
                        )
                    # Add assistant message with tool calls to history
                    # Include reasoning for within-turn continuity
                    # Sanitize tool call arguments before storage — streaming
                    # may accumulate malformed JSON (truncated, bad escapes)
                    # which poisons every subsequent API call with 500s.
                    msg = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": _sanitize_json_args(
                                        tc["name"], tc["arguments"]
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                    self.messages.append(msg)

                    # Execute each tool call
                    for tc in tool_calls:
                        # Check if stop requested - exit tool loop immediately
                        if not self._running:
                            break

                        # Note: we do NOT check for interrupts inside the tool
                        # loop. All tools in the batch must complete before we
                        # check for interrupts at a natural boundary (after all
                        # tool results are in). This ensures API message validity
                        # (all tool_calls must have matching tool results).
                        #
                        # Known limitation: if the user presses Escape during
                        # tool execution, CancelledError propagates out of
                        # _execute_tool_async() but the underlying thread (for
                        # sync tools) or subprocess (for command tool) continues
                        # running in the background. Python threads cannot be
                        # killed. The tool result is discarded when the queue item
                        # is completed, but side effects (file writes, subprocess
                        # output) are not rolled back. Future fix: track
                        # subprocess PIDs and SIGTERM on interrupt.

                        name = tc["name"]
                        args_str = tc["arguments"]

                        # Parse arguments with better error handling
                        try:
                            arguments = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError as e:
                            log_error(
                                e,
                                {
                                    "context": "tool_args_parse",
                                    "name": name,
                                    "arguments": args_str[:200] if args_str else "",
                                },
                            )
                            # Use empty dict and let the tool handle missing args
                            arguments = {}
                        # Emit tool call BEFORE execution so UI can show it immediately
                        log_tool_call(name, arguments)
                        await self.emit(
                            AgentEvent.TOOL_CALL,
                            {
                                "name": name,
                                "arguments": arguments,
                            },
                        )

                        # Execute the tool
                        result = await self._execute_tool_async(name, arguments)

                        # Record tool statistics
                        self.tool_stats.record(name, arguments, result)

                        # Emit result after execution completes
                        log_tool_result(name, result)
                        await self.emit(
                            AgentEvent.TOOL_RESULT,
                            {
                                "name": name,
                                "result": result,
                            },
                        )

                        # Add tool result to messages
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            }
                        )

                    # Safe pause point - check if pause requested
                    await self._wait_if_paused()

                    # Check for interrupt messages at natural boundary
                    # (after all tool results are in, before next LLM call)
                    # Inject interrupt into current turn to preserve KV cache
                    if self.queue.has_interrupt:
                        interrupt_items = self.queue.pop_interrupt_items()

                        # Log the interruption
                        log_queue_interrupt(self.queue.current.id if self.queue.current else None)

                        # Inject a fake assistant+user pair for each interrupt.
                        # The fake assistant preserves the role sequence so the
                        # LLM backend's LCP/KV-cache still sees a valid prefix
                        # (assistant→user instead of bare user mid-turn).
                        for item in interrupt_items:
                            self.messages.append(
                                {
                                    "role": "assistant",
                                    "content": "[Interrupted]",
                                    "interrupt": True,
                                }
                            )
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": item.text,
                                    "interrupt": True,
                                }
                            )
                            # Notify UI that message was injected mid-turn
                            await self.emit(
                                AgentEvent.INTERRUPT_INJECTED,
                                {
                                    "text": item.text,
                                    "item_id": item.id,
                                },
                            )

                        # Stay in turn - continue to next LLM call with the
                        # injected message in context. This preserves the KV
                        # cache because reasoning tokens are not stripped.
                        continue

                    # Continue loop for next LLM call
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_loop_continue",
                            {
                                "note": "Continuing to next LLM call after tool execution",
                            },
                        )
                    # Transition back to WAITING for next LLM call
                    await self._set_status(AgentStatus.WAITING)
                    continue

                # Check for tool calls in reasoning content (Qwen workaround)
                reasoning_tool_calls = detect_tool_calls_in_reasoning(reasoning)

                if reasoning_tool_calls:
                    # Tool calls detected in reasoning - handle them
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_tool_calls_in_reasoning_detected",
                            {
                                "tool_count": len(reasoning_tool_calls),
                                "tool_names": [
                                    tc.get("name") for tc in reasoning_tool_calls
                                ],
                            },
                        )

                    # Show notification about this issue
                    notification_message = (
                        "⚠️  Tool calls detected in reasoning field. "
                        "This model sometimes places tool calls in reasoning instead of "
                        "using the standard format. Executing detected tools."
                    )
                    await self.emit(
                        AgentEvent.NOTIFICATION,
                        {
                            "message": notification_message,
                            "duration": REASONING_TOOL_CALL_NOTIFICATION_DURATION,
                            "level": "warning",
                        },
                    )

                    # Clean reasoning content by removing tool call information
                    # This prevents confusion in subsequent LLM calls
                    clean_reasoning = reasoning
                    for tc in reasoning_tool_calls:
                        # Remove tool call patterns from reasoning
                        # Remove common tool call patterns
                        patterns = [
                            r"<tool_call>.*?</tool_call>",
                            r"<arg_key>.*?</arg_key>",
                            r"\"tool\":\s*\"[^\"]+\"",
                            r"\"function\":\s*\"[^\"]+\"",
                        ]
                        for pattern in patterns:
                            clean_reasoning = re.sub(
                                pattern,
                                "",
                                clean_reasoning,
                                flags=re.DOTALL | re.IGNORECASE,
                            )
                    clean_reasoning = clean_reasoning.strip()

                    # Add assistant message with cleaned reasoning and tool calls
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [
                                {
                                    "id": f"tc_reasoning_{i}",
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["arguments"]),
                                    },
                                }
                                for i, tc in enumerate(reasoning_tool_calls)
                            ],
                        }
                    )

                    if clean_reasoning:
                        # Notify about extracted reasoning instead of adding synthetic message
                        await self.emit(
                            AgentEvent.NOTIFICATION,
                            {
                                "message": f"Extracted reasoning from tool call: {clean_reasoning[:200]}{'...' if len(clean_reasoning) > 200 else ''}",
                                "duration": 10.0,
                                "level": "info",
                            },
                        )

                    # Transition to TOOLING state
                    await self._set_status(AgentStatus.TOOLING)

                    # Execute each detected tool call
                    for i, tc in enumerate(reasoning_tool_calls):
                        name = tc["name"]
                        arguments = tc["arguments"]

                        # Emit tool call event
                        log_tool_call(name, arguments)
                        await self.emit(
                            AgentEvent.TOOL_CALL,
                            {
                                "name": name,
                                "arguments": arguments,
                            },
                        )

                        # Execute the tool
                        result = await self._execute_tool_async(name, arguments)

                        # Record tool statistics
                        self.tool_stats.record(name, arguments, result)

                        # Emit result
                        log_tool_result(name, result)
                        await self.emit(
                            AgentEvent.TOOL_RESULT,
                            {
                                "name": name,
                                "result": result,
                            },
                        )

                        # Add tool result to messages
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"tc_reasoning_{i}",
                                "content": result,
                            }
                        )

                    # Continue loop for next LLM call
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_loop_continue",
                            {
                                "note": "Continuing after handling reasoning tool calls",
                            },
                        )
                    await self._set_status(AgentStatus.WAITING)
                    continue

                # No tool calls - add final response to history
                # Always include reasoning for within-turn continuity; stripping
                # happens between turns via _strip_reasoning_from_messages()
                # only when remove_reasoning is enabled.
                if content or reasoning:
                    append_assistant_message(self.messages, content, reasoning)
                    log_assistant_response(content, reasoning if reasoning else None)
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_assistant_complete",
                            {
                                "content_length": len(content),
                                "reasoning_length": len(reasoning) if reasoning else 0,
                                "note": "ASSISTANT_COMPLETE will be emitted",
                            },
                        )
                    await self.emit(
                        AgentEvent.ASSISTANT_COMPLETE,
                        {
                            "text": content,
                            "reasoning": reasoning,
                        },
                    )

                break

            except Exception as e:
                # Categorize the error for better user feedback
                llm_error = categorize_error(e) if not isinstance(e, LLMError) else e
                log_error(e, {"context": "llm_turn"})
                await self.emit(
                    AgentEvent.ERROR,
                    {
                        "message": str(llm_error),
                        "error_type": llm_error.error_type,
                        "exception": e,
                    },
                )

                break

    async def _set_status(self, status: AgentStatus) -> None:
        """Set the agent status and emit event if changed."""
        if self._status != status:
            self._status = status
            await self.emit(
                AgentEvent.STATUS_CHANGE,
                {
                    "status": status.value,  # Emit string value for compatibility
                },
            )

    async def _emit_queue_update(self) -> None:
        """Emit a queue update event."""
        await self.emit(
            AgentEvent.QUEUE_UPDATE,
            {
                "count": self.queue.pending_count,
                "current": self.queue.current,
            },
        )

    def set_model(self, model: str) -> None:
        """Set the model name.

        Args:
            model: The model name to use
        """
        self.model = model
        self.tool_stats.reset()  # Reset stats on model change

    def set_client(
        self, client: AsyncOpenAI, models: list[str] | None = None
    ) -> None:
        """Set the OpenAI client.

        Args:
            client: The AsyncOpenAI client to use
            models: Optional list of available model names to store
        """
        self.client = client
        if models is not None:
            self.available_models = models

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt.

        Args:
            prompt: The system prompt text
        """
        self.system_prompt = prompt

    def set_response_format(self, response_format: dict) -> None:
        """Set the response format.

        Args:
            response_format: The response format dict (e.g., {"type": "json_object"})
        """
        self.response_format = response_format

    @property
    def devel_mode(self) -> bool:
        """Whether devel-mode tools are visible to the AI."""
        return self._devel_mode

    def set_devel_mode(self, enabled: bool) -> None:
        """Toggle devel mode and rebuild the built-in tool list.

        Args:
            enabled: True to show devel-group tools, False to hide them
        """
        from agent13.tools import get_filtered_tools
        from agent13.config import get_config

        config = get_config()
        self._devel_mode = enabled
        self.tools = get_filtered_tools(
            devel=enabled,
            skills=self._skills_mode,
            enabled_tools=config.enabled_tools or None,
            disabled_tools=config.disabled_tools or None,
        )

    @property
    def skills_mode(self) -> bool:
        """Whether skills-mode tools are visible to the AI."""
        return self._skills_mode

    def set_skills_mode(self, enabled: bool) -> None:
        """Toggle skills mode and rebuild the built-in tool list.

        Args:
            enabled: True to show skills-group tools, False to hide them
        """
        from agent13.tools import get_filtered_tools
        from agent13.config import get_config

        config = get_config()
        self._skills_mode = enabled
        self.tools = get_filtered_tools(
            devel=self._devel_mode,
            skills=enabled,
            enabled_tools=config.enabled_tools or None,
            disabled_tools=config.disabled_tools or None,
        )

    def clear_messages(self) -> int:
        """Clear the message history and reset token usage.

        Returns:
            Number of messages cleared
        """
        count = len(self.messages)
        self.messages.clear()
        self.reset_token_usage()
        return count

    async def request_clear(self, clear_widgets: bool = False) -> int:
        """Request a clear of message history via the queue.

        Adds a kind="clear" item to the queue so the clear happens at a
        safe boundary between items, not mid-loop. This prevents the race
        condition where /clear wipes messages while _llm_turn is iterating.

        Args:
            clear_widgets: If True, also clear the TUI chat window widgets
                (i.e. /clear all). Default False preserves scrollback.

        Returns:
            The queue item ID
        """
        item_id = self.queue.add(
            "", kind="clear",
            data={"clear_widgets": clear_widgets})
        await self._emit_queue_update()
        return item_id

    async def request_load(self, path: str) -> int:
        """Request a context load via the queue.

        Adds a kind="load" item to the queue so the load happens at a
        safe boundary between items, not mid-loop. This prevents the race
        condition where /load replaces messages while _llm_turn is iterating.

        Args:
            path: Path to the context file

        Returns:
            The queue item ID
        """
        item_id = self.queue.add(path, kind="load")
        await self._emit_queue_update()
        return item_id

    def reset_token_usage(self) -> None:
        """Reset token usage counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
