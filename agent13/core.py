"""Agent core class - event-driven agent implementation."""

import asyncio
import datetime
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from agent13.events import AgentEvent, AgentEventData, EventHandler
from agent13.journal import JournalManager
from agent13.polite import PoliteLock, _sanitize_provider
from agent13.message_history import MessageHistory, is_turn_start
from agent13.prompts import DEFAULT_PROMPT, AUTO_COMPACT_CONTINUE_HINT
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
    log_compact_start,
    # TPS debug logging
    is_debug_enabled,
    log_tps_event,
)
from agent13.vision import describe_image, resize_image_uri
from agent13.vision import SIDECAR_MAX_DIMENSION, NATIVE_MAX_DIMENSION
from tools import ToolResult

# Seconds of stream silence (no visible tokens, only keep-alive chunks) before
# core emits TOOL_CALL_PENDING. Providers that buffer whole tool calls (mlx-lm)
# send a keep-alive every ~10s while generating; 5s means the placeholder
# appears on the first keep-alive past the threshold. Normal responses start
# streaming content within ~1s of the role chunk, so they never trip this.
TOOL_CALL_PENDING_SILENCE = 5.0

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
            {
                "context": "sanitize_json_args",
                "name": tool_name,
                "arguments": raw[:200] if raw else "",
            },
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
    # Param combo tracking — sorted tuple of non-None argument names
    param_combos: dict[str, dict[str, int]] = field(default_factory=dict)
    param_combo_successes: dict[str, dict[str, int]] = field(default_factory=dict)

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

        # Track param combo (sorted tuple of non-None argument names)
        combo = ",".join(sorted(k for k, v in arguments.items() if v is not None))
        if name not in self.param_combos:
            self.param_combos[name] = {}
            self.param_combo_successes[name] = {}
        self.param_combos[name][combo] = self.param_combos[name].get(combo, 0) + 1
        if is_success:
            self.param_combo_successes[name][combo] = (
                self.param_combo_successes[name].get(combo, 0) + 1
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
        self.param_combos.clear()
        self.param_combo_successes.clear()

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
                    "param_combos": self.param_combos.get(name, {}),
                    "param_combo_successes": self.param_combo_successes.get(name, {}),
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
    COMPACTING = "compacting"
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

    # Thinking-level suffixes stripped from the polite lock key so agents on
    # the same model with different thinking budgets share one lock. Some
    # backends expose these as first-class /v1/models entries (e.g.
    # "Model:medium"), so membership in the model list can't detect them — a
    # fixed whitelist is the reliable signal.
    _THINKING_SUFFIXES = {"nothink", "none", "low", "medium", "high", "xhigh", "max"}

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
        remove_reasoning: bool = False,
        devel_mode: bool = False,
        skills_mode: bool = False,
        priming_enabled: bool = False,
        auto_compact_threshold: int = 0,
        auto_compact_max_iterations: int = 3,
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
        self.remove_reasoning = remove_reasoning
        self._devel_mode = devel_mode
        self._skills_mode = skills_mode
        self.priming_enabled = priming_enabled
        self.auto_compact_threshold = auto_compact_threshold
        self.auto_compact_max_iterations = auto_compact_max_iterations
        self._auto_compact_failures = 0  # Circuit breaker counter
        self._auto_compact_triggered = False  # Set when threshold hit mid-turn
        self._auto_compact_iterations = 0  # Compact cycles this turn (resets per turn)
        self._auto_compact_snapshot_count = (
            0  # Per-session monotonic (snapshot filenames)
        )
        # Message count at the start of the most recent LLM stream. Used to
        # estimate the true context size at the safe point (prompt_tokens is
        # stale there - it predates the tool results just added).
        self._msg_count_before_stream = 0
        self.execute_tool = execute_tool
        # Session date: stable for the lifetime of the session.
        # Used in the system prompt to prevent midnight date changes
        # from invalidating the KV cache prefix.
        self.session_date = datetime.date.today().isoformat()
        # Available models list (populated by caller or set_client)
        self.available_models: list[str] = []
        self.response_format = response_format
        self.journal_mode = journal_mode
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
        # When True, run() enters PAUSING state immediately after startup
        # (used by ESC-during-polite-wait to pause instead of cancel).
        self._pause_on_start = False

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
        # Set by /resume to signal run() to call continue_incomplete_turn()
        # at the next loop iteration (inside run(), not a separate task).
        self._incomplete_turn_pending: bool = False

        # Polite mode (multi-agent lock coordination); None = disabled.
        self.polite_lock: Optional[PoliteLock] = None

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
            priming_enabled=priming_enabled,
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
            session_date=self.session_date,
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
        data: dict = None,
    ) -> int:
        """Add a user message to the queue.

        Args:
            text: The message text
            priority: Whether to process with high priority (front of queue)
            interrupt: Whether to interrupt the agent loop (implies priority)
            kind: Item kind - "prompt", "journal_last", "journal_all",
                  "compact", "clear", or "load"
            data: Optional metadata dict (e.g. {"compact_prompt": "..."})

        Returns:
            The queue item ID
        """
        item_id = self.queue.add(
            text, priority=priority, interrupt=interrupt, kind=kind, data=data
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

        # ESC-during-polite-wait: re-enter pausing state so the loop
        # blocks at _wait_if_paused() on the very first iteration.
        if self._pause_on_start:
            self._pause_on_start = False
            self._pause_state = PauseState.PAUSING
            self._pause_event.clear()

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

                # Continue an incomplete turn from a loaded context.
                # Triggered by /resume (sets _incomplete_turn_pending via
                # request_continue_incomplete). Runs inside run() so there's
                # exactly one agent task driving the pause state machine.
                if self._incomplete_turn_pending:
                    self._incomplete_turn_pending = False
                    await self.continue_incomplete_turn()
                    # Return to IDLE so the TUI shows "ready" after the
                    # continued turn finishes (mirrors _process_item's
                    # _set_status(IDLE) at its end).
                    await self._set_status(AgentStatus.IDLE)
                    continue
                # Peek at the next item (without removing it) to decide
                # whether we need the polite lock. Acquiring the lock BEFORE
                # get_next() means the item stays in the pending queue during
                # the wait — so /delete q N can remove it and ESC can cancel
                # the wait cleanly without orphaning a "current" item.
                next_item = self.queue.peek_next()

                if (
                    next_item
                    and self.polite_lock is not None
                    and next_item.kind not in ("clear", "load")
                ):
                    # Set WAITING status so the TUI knows we're busy (processing=True).
                    # This makes ESC work and is_queued check pass during the polite wait.
                    await self._set_status(AgentStatus.WAITING)
                    try:
                        await self.polite_lock.acquire()
                    except asyncio.CancelledError:
                        # ESC or task cancellation during polite wait.
                        # The item is still in the pending queue (not yet
                        # pulled as current), so nothing to clean up —
                        # just break so the loop restarts fresh.
                        await self._set_status(AgentStatus.IDLE)
                        break
                    _polite_acquired = True

                    # The item may have been removed from the queue (via
                    # /delete q N) while we were waiting for the lock.
                    # If so, release the lock and loop back — nothing to
                    # process.
                    if next_item not in self.queue.items:
                        if self.polite_lock is not None:
                            self.polite_lock.release()
                        _polite_acquired = False
                        await self._set_status(AgentStatus.IDLE)
                        continue
                else:
                    _polite_acquired = False

                # Get next item from queue
                current = self.queue.get_next()

                if current:
                    await self._process_item(current, _polite_acquired)

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
            self._pause_state == PauseState.RUNNING and self._status == AgentStatus.IDLE
        )

    @property
    def has_incomplete_turn(self) -> bool:
        """Check if the agent has an incomplete turn from a loaded context."""
        return self._incomplete_turn_loaded

    def mark_incomplete_turn(self, incomplete: bool) -> None:
        """Mark that the agent has an incomplete turn (called on load)."""
        self._incomplete_turn_loaded = incomplete

    def request_continue_incomplete(self) -> None:
        """Signal run() to continue an incomplete turn at the next iteration.

        Called by the TUI's /resume when has_incomplete_turn is True. Sets
        _incomplete_turn_pending so run()'s loop picks it up — no separate
        task needed, avoiding concurrent-task races in the pause state machine.
        """
        self._incomplete_turn_pending = True

    async def continue_incomplete_turn(self) -> bool:
        """Continue an incomplete turn by executing pending tools or calling LLM.

        This is called after loading a context with an incomplete turn.
        It handles two cases:
        1. Last message is assistant with tool_calls -> execute pending tools
        2. Last message is tool -> call LLM to process results

        Polite mode (if enabled) is honoured: the shared lock is acquired
        before any work and released in the ``finally`` safety net, mirroring
        ``_process_item``. ``_polite_acquired`` is forwarded to ``_llm_turn``
        so the lock is released during tool execution and re-acquired before
        each LLM stream. Without this, a ``--continue`` + ``/resume`` of an
        incomplete turn would silently bypass polite coordination.

        Pause safe-points are checked after each tool result (mirroring
        ``_llm_turn``'s tool loop) so ``/pause`` during the pending-tools
        branch takes effect promptly instead of running to completion.

        Returns:
            True if continuation was started, False if no continuation needed.
        """
        if not self._incomplete_turn_loaded:
            return False

        # Clear the flag first
        self._incomplete_turn_loaded = False

        # Ensure _running is True so _llm_turn's `while self._running` loop
        # executes. run() normally sets this before calling us, but direct
        # callers (e.g. tests) may not have it set.
        self._running = True
        self._stop_event.clear()

        # Transition to WAITING immediately so the UI reflects activity
        # before the first LLM token arrives (mirrors _process_item).
        # Without this, /resume on a loaded incomplete turn leaves the
        # status at IDLE for the entire LLM latency window.
        await self._set_status(AgentStatus.WAITING)

        # Polite mode: acquire the shared lock before any work, mirroring
        # run()'s acquire-before-get_next(). The turn is already loaded
        # into self.messages (no queue item to peek), so we acquire
        # unconditionally when polite mode is on. Cancellation-safe: on
        # CancelledError the lock is released (via acquire()'s own
        # BaseException handler) before propagating.
        _polite_acquired = False
        if self.polite_lock is not None:
            try:
                await self.polite_lock.acquire()
                _polite_acquired = True
            except asyncio.CancelledError:
                # ESC or task cancellation during the polite wait.
                # acquire() already released any hold; just restore IDLE
                # and re-raise so the caller can clean up.
                await self._set_status(AgentStatus.IDLE)
                raise

        try:
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

                    # Extract display text (TUI/stats expect str, not ToolResult)
                    display_text = (
                        result.text if isinstance(result, ToolResult) else result
                    )

                    # Record tool statistics
                    self.tool_stats.record(name, arguments, display_text)

                    # Emit result
                    log_tool_result(name, display_text)
                    await self.emit(
                        AgentEvent.TOOL_RESULT,
                        {
                            "name": name,
                            "result": display_text,
                        },
                    )

                    # Add tool result to messages (handles vision routing)
                    tool_msg, extra_msgs = await self._build_tool_result_content(
                        result, name, tc["id"]
                    )
                    self.messages.append(tool_msg)
                    for msg in extra_msgs:
                        self.messages.append(msg)

                    # Safe pause point - check if pause requested after
                    # each tool result (mirrors _llm_turn's tool loop).
                    # Without this, /pause during the pending-tools branch
                    # shows "Paused" in the TUI but the loop keeps running
                    # until _llm_turn's own safe point.
                    await self._wait_if_paused()

                # Now call LLM to continue
                await self._llm_turn(_polite_acquired=_polite_acquired)
                return True

            elif self.history.has_incomplete_turn():
                # Case 2: Last message is tool result, call LLM to process
                await self._llm_turn(_polite_acquired=_polite_acquired)
                return True

            return False
        finally:
            # Polite mode safety net: release the lock if still held.
            # _llm_turn normally releases after each LLM stream (before
            # tool execution), so on normal exit the lock is already free.
            # This covers error/cancel paths where _llm_turn may not have
            # had a chance to release.
            if self.polite_lock is not None and self.polite_lock.is_held():
                self.polite_lock.release()

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

    @staticmethod
    def _estimate_message_tokens(messages: list) -> int:
        """Roughly estimate the token count of a list of messages.

        Uses a chars/4 heuristic (no tokenizer dependency). Adequate for a
        threshold check given the headroom between the auto-compact threshold
        and the model's real context limit.

        Image content blocks are counted as a fixed 170 tokens (the actual
        API cost for a medium-resolution image), not as the length of the
        base64 data URI.
        """
        IMAGE_BLOCK_TOKENS = 170  # ~token cost of one image block
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            total += len(block.get("text", "")) // 4
                        elif btype == "image_url":
                            total += IMAGE_BLOCK_TOKENS
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += len(fn.get("arguments", "") or "") // 4
        return total

    def _estimate_current_context_tokens(self) -> int:
        """Estimate the true current context size at the safe point.

        prompt_tokens reflects the LLM call that just completed (the context
        before the tool results that were just added). Add the estimated
        tokens of every message appended since that stream started.
        """
        added = self.messages[self._msg_count_before_stream :]
        return self.prompt_tokens + self._estimate_message_tokens(added)

    def _save_auto_compact_snapshot(self, n: int) -> None:
        """Save pre-compact history to a dated snapshot file.

        Uses the same location and date pattern as the auto-save, with a
        ``_N`` suffix (N = per-session auto-compact count) so snapshots never
        collide. A safety net: if the journal/compact summary loses state, the
        user can reload this file. Never blocks the turn on failure.
        """
        try:
            from agent13.persistence import get_auto_save_path, save_context

            base = get_auto_save_path(session_date=self.session_date)
            path = base.parent / f"{base.stem}_{n}.ctx"
            save_context(self, path)
        except Exception:
            # Snapshot is best-effort; a failure here must not kill the turn.
            pass

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

    async def _build_tool_result_content(
        self, result: "str | ToolResult", tool_name: str, tool_call_id: str
    ) -> tuple[dict, list[dict]]:
        """Convert tool result into message(s) for the conversation.

        Handles vision routing: if the result carries images, routes them
        based on vision config (sidecar, native, or drop).

        Args:
            result: Tool result (str or ToolResult)
            tool_name: Name of the tool that produced the result
            tool_call_id: The tool call ID for the tool message

        Returns:
            (tool_message, extra_messages) where extra_messages are
            additional messages to inject after the tool message.
        """
        if isinstance(result, ToolResult):
            text = result.text
            images = result.images
            question = result.question
        else:
            text = str(result)
            images = []
            question = ""

        if not images:
            # No images — current behavior
            return (
                {"role": "tool", "tool_call_id": tool_call_id, "content": text},
                [],
            )

        # We have images. Route based on vision config.
        from agent13.config import get_config

        config = get_config()
        vision = config.vision
        # No [vision] section → assume current model has vision (native mode).
        # [vision] is only needed to nominate a sidecar for text-only models.
        if vision is None or vision.should_use_native(self.model):
            # Native: inject images inline as a user message with image_url blocks.
            # OpenAI API requires images in user role, not tool role.
            images = [resize_image_uri(uri, NATIVE_MAX_DIMENSION) for uri in images]
            content_parts = [
                {"type": "text", "text": f"[Image from tool: {tool_name}]"}
            ]
            for uri in images:
                content_parts.append({"type": "image_url", "image_url": {"url": uri}})
            return (
                {"role": "tool", "tool_call_id": tool_call_id, "content": text},
                [
                    {
                        "role": "user",
                        "content": content_parts,
                        # Mid-turn injection: no user intent. The flag keeps
                        # it inside the current turn for grouping/compaction
                        # (see message_history.is_turn_start).
                        "injected": True,
                    }
                ],
            )

        elif vision.sidecar_provider:
            # Sidecar: send each image to vision model, get text descriptions back.
            sidecar_config = config.get_provider(vision.sidecar_provider)
            if sidecar_config is None:
                # Provider not found — drop images
                return (
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": text
                        + f"\n\n[Image data omitted — sidecar provider '{vision.sidecar_provider}' not found]",
                    },
                    [],
                )

            images = [resize_image_uri(uri, SIDECAR_MAX_DIMENSION) for uri in images]
            descriptions = []
            for uri in images:
                if question:
                    prompt = question
                else:
                    prompt = (
                        f"Context: {text}. "
                        "Describe this image in detail: all visible text, "
                        "UI elements, colors, layout, and anything notable."
                    )
                desc = await describe_image(
                    sidecar_config, vision.sidecar_model or None, uri, prompt
                )
                descriptions.append(desc)

            combined = text + "\n\n" + "\n\n".join(descriptions)
            return (
                {"role": "tool", "tool_call_id": tool_call_id, "content": combined},
                [],
            )

        else:
            # Auto mode, no match, no sidecar — drop images
            return (
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": text
                    + "\n\n[Image data omitted — no vision backend available]",
                },
                [],
            )

    def _strip_images_from_messages(self) -> None:
        """Remove image_url blocks from messages after the model has seen them.

        Replaces each image_url content block with a text placeholder
        (e.g. "[image: screenshot.png]"). This prevents base64 image data
        from accumulating in context across turns. The model already
        processed the image in the current turn; future turns reference
        the text description in the tool result.

        Only affects user messages with list content (the pattern used
        by native vision injection). Tool messages and plain-string
        user messages are untouched.
        """
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # Check if this message has any image_url blocks
            has_image = any(
                isinstance(block, dict) and block.get("type") == "image_url"
                for block in content
            )
            if not has_image:
                continue
            # Replace image_url blocks with text placeholders
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    # Extract a short label from the data URI if possible
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/"):
                        # e.g. "data:image/png;base64,..." → "png"
                        media_type = url.split("/")[1].split(";")[0]
                        new_content.append(
                            {"type": "text", "text": f"[image: {media_type}]"}
                        )
                    else:
                        new_content.append({"type": "text", "text": "[image]"})
                else:
                    new_content.append(block)
            msg["content"] = new_content

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
        from agent13.tools import apply_tool_filter
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
                if not apply_tool_filter(tool_name, enabled, disabled):
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

    async def _process_item(
        self, item: QueueItem, _polite_acquired: bool = False
    ) -> None:
        """Process a single queue item.

        Args:
            item: The queue item to process
            _polite_acquired: Whether the polite lock was already acquired
                by the caller (run()). The lock is released in the finally
                below if this is True.
        """
        # Clear and load are instant housekeeping operations — skip
        # WAITING status so the bell doesn't arm/ring for them.
        if item.kind not in ("clear", "load"):
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
            elif item.kind == "compact":
                compact_prompt = (item.data or {}).get("compact_prompt", "")
                success, message = await self.compact_history(compact_prompt)
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
                mode = (item.data or {}).get("mode", "all")
                keep_turns = (item.data or {}).get("keep_turns", 0)
                if mode == "trim":
                    count = self.trim_messages(keep_turns)
                else:
                    count = self.clear_messages()
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
                await self.emit(
                    AgentEvent.MESSAGES_CLEARED,
                    {
                        "count": count,
                        "mode": mode,
                    },
                )
            elif item.kind == "load":
                # Deferred /load — safe at this boundary between items
                from agent13.persistence import load_context

                success, message, incomplete = load_context(self, item.text)
                log_journal_debug(
                    "load_context",
                    {
                        "success": success,
                        "message": message,
                        "messages_count": len(self.messages),
                        "has_tool_calls": self.history.has_tool_calls(),
                        "first_3_roles": [m.get("role") for m in self.messages[:3]]
                        if self.messages
                        else [],
                    },
                )
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

                # Reset the per-turn compact counter before this turn starts.
                self._auto_compact_iterations = 0

                # Process with LLM (may include multiple tool call rounds)
                await self._llm_turn(_polite_acquired=_polite_acquired)

                # Auto-compact: the threshold was hit mid-turn. Compact (or
                # journal) the history, nudge the model to resume the in-progress
                # task, and re-enter the turn. Bounded by
                # auto_compact_max_iterations; once the bound is reached the
                # agent pauses (handled inside _llm_turn) instead of failing,
                # so the user can /compact, /journal all, or /model to a
                # bigger-context model, then /resume.
                while self._auto_compact_triggered:
                    self._auto_compact_triggered = False
                    self._auto_compact_iterations += 1
                    # Snapshot the pre-compact history (safety net). Named with
                    # a per-session monotonic counter so snapshots never collide.
                    self._auto_compact_snapshot_count += 1
                    self._save_auto_compact_snapshot(self._auto_compact_snapshot_count)
                    if self.journal_mode:
                        success, msg = await self.journal.journal_all()
                    else:
                        success, msg = await self.compact_history("")
                    if not success:
                        self._auto_compact_failures += 1
                        break
                    self._auto_compact_failures = 0
                    # Nudge the model to finish the original task.
                    self.messages.append(
                        {"role": "user", "content": AUTO_COMPACT_CONTINUE_HINT}
                    )
                    await self._llm_turn(_polite_acquired=_polite_acquired)

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
                log_error(e, {"context": "process_item_cancelled", "item_id": item.id})
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
                llm_error = categorize_error(e) if not isinstance(e, LLMError) else e
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
        finally:
            # Polite mode: release the shared lock if it's still held.
            # _llm_turn normally releases after each LLM stream (before tool
            # execution), so on normal exit the lock is already free. This
            # is a safety net for error/cancel paths where _llm_turn may
            # not have had a chance to release.
            if self.polite_lock is not None and self.polite_lock.is_held():
                self.polite_lock.release()

        # Skip IDLE if there are queued items — go straight to WAITING
        # so the TUI (and bell) never sees "ready" between queue items.
        # Mirrors the resume path logic above.
        if self.queue.pending_count > 0:
            await self._set_status(AgentStatus.WAITING)
        else:
            await self._set_status(AgentStatus.IDLE)

    async def _llm_turn(self, _polite_acquired: bool = False) -> None:
        """Execute one LLM turn (may include multiple tool call rounds).

        A turn continues until the LLM responds without tool calls.
        Uses streaming for all phases to capture reasoning tokens.

        Args:
            _polite_acquired: Whether the polite lock was acquired by run().
                If True, the lock is released after each LLM stream (before
                tool execution) and re-acquired before the next stream. This
                frees the GPU for other agents during long-running tools.
        """
        if is_debug_enabled():
            log_tps_event("agent_llm_turn_start", {"note": "Starting new LLM turn"})
        while self._running:
            try:
                # Polite mode: acquire the lock before each LLM stream (GPU).
                # On the first iteration, run() already acquired it, so
                # is_held() is True and we skip. On subsequent iterations
                # (after tool execution), we re-acquire here.
                if _polite_acquired and self.polite_lock is not None:
                    if not self.polite_lock.is_held():
                        await self.polite_lock.acquire()

                # Stream response, capturing reasoning and tool calls
                content = ""
                reasoning = ""
                tool_calls = None
                # Tool-call ids we've already emitted TOOL_CALL_STARTED for, so
                # the early "name known" signal fires exactly once per tool call.
                _started_tool_ids: set = set()
                # Silent-stream detection: providers that buffer whole tool
                # calls (e.g. mlx-lm) stream nothing visible while generating
                # them - only payload-less keep-alive chunks. Track the last
                # time a visible token arrived so a long silence can be
                # reported to the UI as "a tool call is likely being prepared".
                # None until the first chunk arrives: prompt-processing time
                # (request sent -> first chunk) is normal waiting, not silence.
                _last_visible_t: float | None = None
                _pending_emitted = False

                if is_debug_enabled():
                    log_tps_event(
                        "agent_stream_start",
                        {"note": "Starting _stream_and_emit"},
                    )

                # Track first tokens for status transitions
                _first_reasoning = True
                _first_content = True

                # Record context size at stream start so the safe point can
                # estimate the true current context (prompt_tokens alone is
                # stale - it predates the tool results added after this call).
                self._msg_count_before_stream = len(self.messages)

                # Stream via _stream_and_emit which handles STREAM_START
                # and TOKEN_USAGE centrally (DRY).
                async for event_type, data in self._stream_and_emit(
                    self.messages,
                    source="assistant",
                ):
                    if event_type == "content":
                        content += data
                        _last_visible_t = time.time()
                        # Visible tokens are flowing: re-arm the silent-stream
                        # signal. A placeholder shown during an earlier silence
                        # was a false alarm (slow start / pre-thinking gap),
                        # but a LATER silence in the same stream (e.g.
                        # thinking -> buffered tool call) must signal again.
                        _pending_emitted = False
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
                        _last_visible_t = time.time()
                        # Re-arm the silent-stream signal (same as content).
                        _pending_emitted = False
                        # Transition to THINKING on first reasoning token
                        if _first_reasoning and data.strip():
                            _first_reasoning = False
                            await self._set_status(AgentStatus.THINKING)
                        await self.emit(
                            AgentEvent.ASSISTANT_REASONING,
                            {
                                "text": data,
                            },
                        )
                    elif event_type == "keepalive":
                        # Payload-less chunk: the stream is alive but the
                        # server is suppressing output. The first chunk only
                        # starts the clock (generation has begun); silence
                        # AFTER it means the model is almost certainly
                        # generating a (buffered) tool call - tell the UI so
                        # the user isn't left staring at nothing.
                        if _last_visible_t is None:
                            _last_visible_t = time.time()
                            continue
                        _silent_for = time.time() - _last_visible_t
                        if (
                            not _pending_emitted
                            and _silent_for > TOOL_CALL_PENDING_SILENCE
                        ):
                            _pending_emitted = True
                            if is_debug_enabled():
                                log_tps_event(
                                    "agent_tool_call_pending",
                                    {
                                        "silent_for": round(_silent_for, 1),
                                        "note": "stream alive, no visible tokens - "
                                        "provider likely buffering a tool call",
                                    },
                                )
                            await self.emit(AgentEvent.TOOL_CALL_PENDING, {})
                    elif event_type == "tool_call":
                        # Early signal: the model has committed to a tool (name
                        # known) but its arguments are still streaming. Emit once
                        # per tool-call id so the UI can show the tool name
                        # immediately instead of waiting for the full argument
                        # stream to finish (which can be long — e.g. a big
                        # write_file whose file content IS the arguments).
                        tc_id = data.get("id")
                        if tc_id is not None and tc_id not in _started_tool_ids:
                            _started_tool_ids.add(tc_id)
                            # The name is now known - any TOOL_CALL_PENDING
                            # placeholder is superseded by the named widget.
                            _pending_emitted = True
                            if is_debug_enabled():
                                log_tps_event(
                                    "agent_tool_call_started",
                                    {
                                        "name": data.get("name", ""),
                                        "id": tc_id,
                                        "note": "name known, args still streaming",
                                    },
                                )
                            await self.emit(
                                AgentEvent.TOOL_CALL_STARTED,
                                {"name": data.get("name", ""), "id": tc_id},
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

                # Polite mode: release the lock after streaming completes.
                # The GPU is no longer needed — tool execution (if any) runs
                # without the lock so other agents can use the GPU. If there
                # are no tool calls, we break below and the lock is already
                # free for the next agent.
                if _polite_acquired and self.polite_lock is not None:
                    if self.polite_lock.is_held():
                        self.polite_lock.release()

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

                        # Extract display text (TUI/stats expect str, not ToolResult)
                        display_text = (
                            result.text if isinstance(result, ToolResult) else result
                        )

                        # Record tool statistics
                        self.tool_stats.record(name, arguments, display_text)

                        # Emit result
                        log_tool_result(name, display_text)
                        await self.emit(
                            AgentEvent.TOOL_RESULT,
                            {
                                "name": name,
                                "result": display_text,
                            },
                        )

                        # Add tool result to messages (handles vision routing)
                        tool_msg, extra_msgs = await self._build_tool_result_content(
                            result, name, tc["id"]
                        )
                        self.messages.append(tool_msg)
                        for msg in extra_msgs:
                            self.messages.append(msg)

                    # Safe pause point - check if pause requested
                    await self._wait_if_paused()

                    # Check for interrupt messages at natural boundary
                    # (after all tool results are in, before next LLM call)
                    # Inject interrupt into current turn to preserve KV cache
                    if self.queue.has_interrupt:
                        interrupt_items = self.queue.pop_interrupt_items()

                        # Log the interruption
                        log_queue_interrupt(
                            self.queue.current.id if self.queue.current else None
                        )

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

                    # Auto-compact threshold check at safe point.
                    # Use the estimated CURRENT context (prompt_tokens is stale
                    # here - it predates the tool results just added). If over
                    # the threshold, either compact-and-continue (bounded) or
                    # pause so the user can intervene.
                    estimated_context = self._estimate_current_context_tokens()
                    # Publish the live context estimate so the UI can show it
                    # (with a "~") until the next LLM call reports the grounded
                    # count via TOKEN_USAGE.
                    await self.emit(
                        AgentEvent.CONTEXT_ESTIMATE,
                        {"estimated_tokens": estimated_context},
                    )
                    if (
                        self.auto_compact_threshold > 0
                        and estimated_context >= self.auto_compact_threshold
                        and self._auto_compact_failures < 3
                    ):
                        if is_debug_enabled():
                            log_tps_event(
                                "auto_compact_check",
                                {
                                    "prompt_tokens": self.prompt_tokens,
                                    "estimated_context": estimated_context,
                                    "threshold": self.auto_compact_threshold,
                                    "iterations": self._auto_compact_iterations,
                                    "max_iterations": self.auto_compact_max_iterations,
                                    "action": (
                                        "compact"
                                        if self._auto_compact_iterations
                                        < self.auto_compact_max_iterations
                                        else "pause"
                                    ),
                                },
                            )
                        if (
                            self._auto_compact_iterations
                            < self.auto_compact_max_iterations
                        ):
                            # Compact in the post-turn handler, then re-enter
                            # the turn so the model finishes its task.
                            action = "journaling" if self.journal_mode else "compacting"
                            await self.emit(
                                AgentEvent.ASSISTANT_TOKEN,
                                {
                                    "text": f"\n[Auto-compact: context at ~{estimated_context:,} tokens, {action}]\n"
                                },
                            )
                            self._auto_compact_triggered = True
                            break
                        else:
                            # Bound reached: pause at this safe point so the
                            # user can /compact, /journal all, or /model to a
                            # bigger-context model, then /resume.
                            await self.emit(
                                AgentEvent.ASSISTANT_TOKEN,
                                {
                                    "text": (
                                        f"\n[Auto-compact limit ({self.auto_compact_max_iterations}) "
                                        f"reached — context at ~{estimated_context:,} tokens. "
                                        "Pausing. You can /compact, /journal all, or /model to a "
                                        "larger-context model, then /resume.]\n"
                                    )
                                },
                            )
                            self.pause()
                            await self._wait_if_paused()
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

                        # Extract display text (TUI/stats expect str, not ToolResult)
                        display_text = (
                            result.text if isinstance(result, ToolResult) else result
                        )

                        # Record tool statistics
                        self.tool_stats.record(name, arguments, display_text)

                        # Emit result
                        log_tool_result(name, display_text)
                        await self.emit(
                            AgentEvent.TOOL_RESULT,
                            {
                                "name": name,
                                "result": display_text,
                            },
                        )

                        # Add tool result to messages (handles vision routing)
                        tool_msg, extra_msgs = await self._build_tool_result_content(
                            result, name, f"tc_reasoning_{i}"
                        )
                        self.messages.append(tool_msg)
                        for msg in extra_msgs:
                            self.messages.append(msg)

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

                # Strip image data from messages — the model has already
                # processed them this turn. Replaces image_url blocks with
                # a text placeholder to prevent context bloat on future turns.
                self._strip_images_from_messages()

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

        If polite mode is enabled, the lock is re-keyed to the new model so
        coordination tracks the model actually in use. The swap is skipped
        when the lock is held mid-turn (mirrors ``set_client``).

        Args:
            model: The model name to use
        """
        self.model = model
        self.tool_stats.reset()  # Reset stats on model change
        # Re-key the polite lock to the new model (per-model coordination).
        if self.polite_lock is not None and not self.polite_lock.is_held():
            old_interval = self.polite_lock.interval
            self.polite_lock.release()
            self.set_polite(interval=old_interval)

    def set_client(self, client: AsyncOpenAI, models: list[str] | None = None) -> None:
        """Set the OpenAI client.

        If polite mode is enabled, the lock is re-keyed to the new
        provider's base URL so coordination targets the correct backend
        after a ``/provider`` switch. The interval is preserved. The swap
        is skipped when the lock is held mid-turn (the in-flight turn
        releases it in its ``finally``; the next ``/polite`` re-keys).

        Args:
            client: The AsyncOpenAI client to use
            models: Optional list of available model names to store
        """
        self.client = client
        if models is not None:
            self.available_models = models
        # Re-key the polite lock to the new provider's base URL.
        if self.polite_lock is not None:
            new_key = str(client.base_url)
            if self.polite_lock.provider != new_key and not self.polite_lock.is_held():
                old_interval = self.polite_lock.interval
                self.polite_lock.release()
                self.set_polite(interval=old_interval, provider=new_key)

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
        self._devel_mode = enabled
        self._rebuild_tools()

    @property
    def skills_mode(self) -> bool:
        """Whether skills-mode tools are visible to the AI."""
        return self._skills_mode

    def set_skills_mode(self, enabled: bool) -> None:
        """Toggle skills mode and rebuild the built-in tool list.

        Args:
            enabled: True to show skills-group tools, False to hide them
        """
        self._skills_mode = enabled
        self._rebuild_tools()

    @property
    def polite_mode(self) -> bool:
        """Whether polite mode (multi-agent lock coordination) is enabled."""
        return self.polite_lock is not None

    def set_polite(self, interval: float, provider: Optional[str] = None) -> None:
        """Enable polite mode with the given poll interval.

        The shared lock is keyed by the backend's base URL (derived from
        ``self.client.base_url``) *and* the current model, so that two agents
        targeting the same backend and model coordinate correctly — whether
        they used a provider name or a raw URL — while agents on different
        models of the same backend run in parallel. An explicit ``provider``
        override is accepted for testing/diagnostics.

        The lock is acquired at the top of ``_process_item`` and released
        in its ``finally``. Takes effect on the next ``_process_item``.

        Args:
            interval: Poll interval N in seconds (pseudo-priority; lower is
                more aggressive). 0 yields to the event loop each iteration.
            provider: Optional explicit key for the lock filename. If None,
                ``str(self.client.base_url)`` is used.
        """
        key = provider if provider is not None else str(self.client.base_url)
        model_key = self._polite_model_key()
        # If already enabled with the same provider+model key, just update the
        # interval (avoids recreating the lock and dropping any in-flight hold).
        if (
            self.polite_lock is not None
            and self.polite_lock.provider == key
            and self.polite_lock.model == model_key
        ):
            self.polite_lock.interval = interval
            return
        self.polite_lock = PoliteLock(
            provider=key,
            interval=interval,
            model=model_key,
            emit=self.emit,
        )

    def _model_base_name(self) -> str:
        """Base model name for the polite lock key.

        A trailing thinking-level suffix (``:none``, ``:medium``, … — see
        ``_THINKING_SUFFIXES``) is stripped so agents on the same model with
        different thinking budgets coordinate on one lock. Any other colon
        suffix (e.g. OpenRouter ``meta-llama/llama-3.1:free``) is part of the
        model name and is kept.
        """
        model = self.model
        if ":" not in model:
            return model
        base, _, suffix = model.rpartition(":")
        if suffix.lower() in self._THINKING_SUFFIXES:
            return base
        return model

    def _polite_model_key(self) -> Optional[str]:
        """Model key for the polite lock filename.

        The sanitized base model name (thinking-level suffix stripped, see
        ``_model_base_name``). Returns ``None`` when no model is set
        (provider-only key).
        """
        if not self.model:
            return None
        return _sanitize_provider(self._model_base_name())

    def disable_polite(self) -> None:
        """Disable polite mode. Releases the lock if currently held.

        Safe to call when polite mode was never enabled (silent no-op).
        Takes effect on the next ``_process_item`` (a turn in flight is
        unaffected).
        """
        if self.polite_lock is not None:
            self.polite_lock.release()
            self.polite_lock = None

    def _rebuild_tools(self) -> None:
        """Rebuild ``self.tools`` from current mode flags and config filter.

        Centralizes the ``get_filtered_tools`` call that ``set_devel_mode``
        and ``set_skills_mode`` both used, so the ``enabled_tools or None``
        idiom and the devel/skills flag plumbing live in one place. Initial
        construction still calls ``get_filtered_tools`` directly (via
        repl/cli/tui) because those sites read the flags from CLI args, not
        from ``self._devel_mode`` / ``self._skills_mode``.
        """
        from agent13.tools import get_filtered_tools
        from agent13.config import get_config

        config = get_config()
        self.tools = get_filtered_tools(
            devel=self._devel_mode,
            skills=self._skills_mode,
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

    def trim_messages(self, keep_turns: int) -> int:
        """Trim message history to keep only the last N turns.

        A turn = one user message + the assistant response + any intervening
        tool calls/results. Walks backwards counting turn-start user messages
        (interrupts and image injections don't open a turn, so the cut never
        lands in the middle of a turn).

        Args:
            keep_turns: Number of turns to keep

        Returns:
            Number of messages removed
        """
        if keep_turns <= 0 or not self.messages:
            return self.clear_messages()

        user_indices = [i for i, m in enumerate(self.messages) if is_turn_start(m)]
        if len(user_indices) <= keep_turns:
            # Not enough turns to trim — nothing to do
            return 0

        cut_index = user_indices[-keep_turns]
        removed = len(self.messages[:cut_index])
        self.messages = self.messages[cut_index:]
        self.reset_token_usage()
        return removed

    async def request_clear(self, mode: str = "all", keep_turns: int = 0) -> int:
        """Request a clear/trim of message history via the queue.

        Adds a kind="clear" item to the queue so the clear happens at a
        safe boundary between items, not mid-loop. This prevents the race
        condition where /clear wipes messages while _llm_turn is iterating.

        Args:
            mode: "all" to wipe entire history, "trim" to keep last N turns.
            keep_turns: Number of turns to keep (only used when mode="trim").

        Returns:
            The queue item ID
        """
        item_id = self.queue.add(
            "",
            kind="clear",
            data={"mode": mode, "keep_turns": keep_turns},
        )
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

    async def compact_history(self, compact_prompt: str) -> tuple[bool, str]:
        """Compact the entire message history into a single user/assistant pair.

        1. Appends the compaction prompt directly to self.messages
        2. Streams the LLM response (visible to user via ASSISTANT_TOKEN events)
        3. Replaces entire history with a lightweight user/assistant pair

        The compaction prompt is appended to self.messages (not a copy) and
        streamed with the default tool_choice="auto". This preserves the KV
        cache prefix — same tools schema, same tool_choice as normal turns.
        Using a separate list or tool_choice="none" would alter the prompt
        structure and cause a full cache miss.

        The compaction prompt is transient — it elicits the summary then is
        discarded. The replacement user message is a small generic line so
        the summary reads as a natural response without re-bloating context.

        Args:
            compact_prompt: The full compaction prompt text to send to the LLM.

        Returns:
            Tuple of (success: bool, message: str) describing the outcome.
        """
        from agent13.prompts import COMPACT_REPLACEMENT_MESSAGE, DEFAULT_COMPACT_PROMPT
        from agent13.journal import _count_message_words

        if not compact_prompt:
            compact_prompt = DEFAULT_COMPACT_PROMPT

        if not self.messages:
            return False, "No messages in context"

        tokens_before = _count_message_words(self.messages)

        if is_debug_enabled():
            log_compact_start(len(self.messages), tokens_before)

        # Append compaction prompt directly to self.messages.
        # This preserves the KV cache prefix: same tools schema, same
        # tool_choice ("auto") as normal turns. Using a separate list or
        # changing tool_choice to "none" would alter the prompt structure
        # and cause a full cache miss.
        original_len = len(self.messages)
        self.messages.append({"role": "user", "content": compact_prompt})

        await self._set_status(AgentStatus.COMPACTING)

        content_parts: list[str] = []
        _first_content = True

        try:
            async for event_type, data in self._stream_and_emit(
                self.messages,
                source="compact",
            ):
                if event_type == "content":
                    content_parts.append(data)
                    if _first_content:
                        _first_content = False
                        await self._set_status(AgentStatus.PROCESSING)
                    await self.emit(
                        AgentEvent.ASSISTANT_TOKEN,
                        {"text": data},
                    )
                elif event_type == "reasoning":
                    await self.emit(
                        AgentEvent.ASSISTANT_REASONING,
                        {"text": data, "source": "compact"},
                    )

            summary = "".join(content_parts).strip()
            if not summary:
                # Remove the compaction prompt we appended
                del self.messages[original_len:]
                if self._status == AgentStatus.COMPACTING:
                    await self._set_status(AgentStatus.IDLE)
                return False, "Compaction produced no summary"

            tokens_after = len(summary.split())

            # Replace entire history with the lightweight pair
            self.messages = [
                {"role": "user", "content": COMPACT_REPLACEMENT_MESSAGE},
                {"role": "assistant", "content": summary},
            ]

            # Emit ASSISTANT_COMPLETE to signal stream end
            await self.emit(AgentEvent.ASSISTANT_COMPLETE, {"text": summary})

            # Emit JOURNAL_COMPACT for UI notification (consistent with journal)
            await self.emit(
                AgentEvent.JOURNAL_COMPACT,
                {
                    "summary": summary,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                },
            )

            # Emit TOKEN_USAGE so the TUI refreshes its Ctx counter
            await self.emit(
                AgentEvent.TOKEN_USAGE,
                {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": self.prompt_tokens,
                    "source": "compact",
                },
            )

            return True, f"Compacted {tokens_before}\u2192{tokens_after} words"

        except Exception as e:
            # Remove the compaction prompt we appended on failure
            del self.messages[original_len:]
            log_error(e, {"context": "compact_history"})
            llm_error = categorize_error(e) if not isinstance(e, LLMError) else e
            await self.emit(
                AgentEvent.ERROR,
                {
                    "message": str(llm_error),
                    "error_type": llm_error.error_type,
                    "exception": e,
                },
            )
            if self._status == AgentStatus.COMPACTING:
                await self._set_status(AgentStatus.IDLE)
            return False, str(llm_error)
