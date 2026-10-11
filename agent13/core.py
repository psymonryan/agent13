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

from agent13.context_gauge import ContextGauge
from agent13.durable_records import DurableRecords
from agent13.events import AgentEvent, AgentEventData, EventHandler
from agent13.journal import JournalManager
from agent13.polite import PoliteLock, PolitePauseRequested, _sanitize_provider
from agent13.message_history import MessageHistory, is_turn_start
from agent13.prompts import (
    DEFAULT_PROMPT,
    AUTO_CONTEXT_CONTINUE_HINT,
    AUTO_CONTEXT_CARRY_HINT,
    DEFAULT_REPORT_AND_COMPACT_PROMPT,
    DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT,
    DEGENERATE_NUDGE_PROMPT,
)
from agent13.queue import AgentQueue, QueueItem
from agent13.llm import (
    append_assistant_message,
    categorize_error,
    detect_tool_calls_in_reasoning,
    LLMError,
    reload_agents_md,
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
    log_degenerate_nudge,
    log_degenerate_stop,
    # TPS debug logging
    is_debug_enabled,
    log_tps_event,
)
from agent13.vision import (
    NATIVE_MAX_DIMENSION,
    SIDECAR_MAX_DIMENSION,
    describe_image,
    image_uri_decodable,
    resize_image_uri,
)
from tools import ToolResult

# Seconds of stream silence (no visible tokens, only keep-alive chunks) before
# core emits TOOL_CALL_PENDING. Providers that buffer whole tool calls (mlx-lm)
# send a keep-alive every ~10s while generating; 5s means the placeholder
# appears on the first keep-alive past the threshold. Normal responses start
# streaming content within ~1s of the role chunk, so they never trip this.
TOOL_CALL_PENDING_SILENCE = 5.0

# Max number of times to re-issue an LLM stream that came back empty
# (zero bytes). The first retry is the real fix: after a long pause the
# pooled HTTP connection has gone stale, and re-issuing on a fresh
# connection recovers. The extra attempt is insurance against a
# transient backend hiccup; more just delays surfacing the failure.
_MAX_EMPTY_STREAM_RETRIES = 2
# Delay before the second retry (the first is immediate - a fresh
# connection is the fix, there is nothing to wait for). Don't hammer a
# struggling backend.
_EMPTY_STREAM_RETRY_BACKOFF = 0.5

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


# Degeneration guard: the tail window and garbage threshold for the
# keep-vs-drop decision. docs_archive/degeneration_guard_plan.md.
_DEGENERATE_TAIL_CHARS = 6000
_DEGENERATE_TAIL_MIN_NON_WS = 0.01  # <1% non-whitespace in the tail = garbage wall


def _degenerate_tail_is_garbage(content: str, reasoning: str) -> bool:
    """True when the tail of a degenerate partial output is whitespace-dominated.

    The drop heuristic for the nudge path: a whitespace wall (the 09-10
    incident - a huge run of newlines) carries no information and would
    poison the context (prompt-too-long risk), so the partial is dropped
    and preserved via snapshot instead. Coherent tails (legit cut-offs, or
    the reasoning-loop deaths) keep the partial in history.
    """
    tail = (content + reasoning)[-_DEGENERATE_TAIL_CHARS:]
    if not tail:
        return False
    non_ws = sum(1 for ch in tail if not ch.isspace())
    return (non_ws / len(tail)) < _DEGENERATE_TAIL_MIN_NON_WS


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
    # Results bounded by tool output bounding (over-cap text spilled to a
    # temp file; see agent13/tool_output_bound.py). A total, not per-tool -
    # it answers "did truncation happen this session?" in /status.
    bounded: int = 0

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
        self.bounded = 0

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


class StopReason(Enum):
    """Why the agent was asked to stop.

    Distinguishes user interrupts (ESC — history needs repair so the next
    message has valid role alternation) from clean shutdowns (quit/app exit
    — history must be left untouched so a mid-turn pause survives the
    auto-save and /resume after --continue can pick it up exactly where it
    stopped, preserving the backend kv-cache prefix).
    """

    INTERRUPT = "interrupt"  # ESC — repair history at cancellation point
    QUIT = "quit"  # Clean shutdown — preserve mid-turn state for auto-save


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
        auto_context_threshold: int = 220000,
        auto_context_action: str = "report_and_compact",
        auto_context_chain: int = 3,
        auto_context_nudge_cap: int = 2,
        report_and_compact_prompt: str = "",
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
            auto_context_threshold: Token threshold for the auto-context check.
            auto_context_action: What to do at the threshold - "compact",
                "report_and_compact", or "none".
            auto_context_chain: How many times the turn may restart after a
                compact within one user turn (0 = none; ignored when action is
                "none").
            report_and_compact_prompt: Prompt text injected for the wrap-up
                turn. Resolved from prompts.yaml ("report_and_compact") by the
                caller; empty = use DEFAULT_REPORT_AND_COMPACT_PROMPT.
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
        # Skills manually loaded this session (name -> loaded content, in
        # load order). Tracked by note_skill_loaded (TUI /skillname); the
        # content is re-injected by _mechanical_restart on chain restarts
        # so a manual /skillname survives a /chain. Session state only.
        self._active_skills: dict[str, str] = {}
        self.priming_enabled = priming_enabled
        self.auto_context_threshold = auto_context_threshold
        self._auto_context_failures = 0  # Circuit breaker counter
        self._auto_context_triggered = False  # Set when threshold hit mid-turn
        self._auto_context_chain_used = 0  # Chain restarts used this turn
        # Degeneration guard (docs_archive/degeneration_guard_plan.md):
        # cap = consecutive degenerate stream ends tolerated per turn before
        # the guard gives up (0 = guard off). streak counts consecutive
        # degenerate ends THIS turn (reset on a clean tool-calls round, a
        # chain restart, or a new user turn). nudges_total is the
        # session-lifetime nudge count for /status (persisted in saves).
        self.auto_context_nudge_cap = auto_context_nudge_cap
        self._degenerate_streak = 0
        self._degenerate_nudges_total = 0
        self._degenerate_snapshot_count = (
            0  # Per-session monotonic (snapshot filenames)
        )
        self._degenerate_partial = None  # Held only while _save_degenerate_snapshot runs
        # Session-lifetime compaction total (persisted in the .ctx save).
        # Incremented on every FULL history replacement (compact_history,
        # _mechanical_restart); journal-reflection compaction is excluded
        # (per-turn housekeeping, not a context cycle). Reset only by
        # /new_session (clear mode="all").
        self._chain_count = 0
        # Original session start (ISO). Persisted in the .ctx save - survives
        # save/load so /status shows the session's true age across resumes.
        self._session_started_at = datetime.datetime.now().isoformat()
        self._in_flight_user_request = (
            ""  # Turn-start user instruction, carried across chain restarts
        )
        self._auto_context_snapshot_count = (
            0  # Per-session monotonic (snapshot filenames)
        )
        # Auto-context action (--auto-context-action / [auto_context] action)
        self.auto_context_action = auto_context_action
        self.auto_context_chain = auto_context_chain
        self.report_and_compact_prompt = (
            report_and_compact_prompt or DEFAULT_REPORT_AND_COMPACT_PROMPT
        )
        self._report_and_compact_triggered = False  # Set when threshold hit mid-turn
        # Suppresses the auto-context threshold check while the wrap-up turn
        # runs, so the threshold can't re-fire mid-wrap-up and loop.
        self._suppress_auto_context = False
        # Durable records: focus name + record paths + wrap-up capture +
        # seeding + restart intro + the compact pointer block
        # (agent13.durable_records). The _focus_name / _session_*_path
        # attributes below are forwarding properties onto it.
        self.durable_records = DurableRecords()
        # Context gauge: single store for the grounded prompt-token value
        # and the stream cursor (agent13.context_gauge). prompt_tokens and
        # _msg_count_before_stream are forwarding properties onto it.
        self._context_gauge = ContextGauge()
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
        # 1-based tool-call number within the current user turn (reset at
        # the top of _process_item). Carried in TOOL_CALL/TOOL_RESULT
        # payloads so the UIs can pair a result with its call.
        self._turn_tool_seq = 0

        # Pause/resume state (single source of truth — PauseState enum)
        self._pause_state = PauseState.RUNNING
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        # When True, run() enters PAUSING state immediately after startup
        # (used by ESC-during-polite-wait: the TUI cancels the blocked
        # acquire and restarts run(), which then blocks at
        # _wait_if_paused() on the very first iteration).
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
        # Why stop() was called — gates history repair on cancel (Fix 1).
        self._stop_reason: StopReason = StopReason.INTERRUPT

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

    @property
    def prompt_tokens(self) -> int:
        """Grounded prompt-token value - the context size of the last LLM
        call's request. Forwarded to the context gauge (the single store);
        the attribute surface is kept so TUI, /status, journal, and tests
        read and write it exactly as before."""
        return self._context_gauge.grounded

    @prompt_tokens.setter
    def prompt_tokens(self, value: int):
        self._context_gauge.record_usage(value)

    @property
    def _msg_count_before_stream(self) -> int:
        """Stream cursor (messages covered by the grounded value).
        Forwarded to the context gauge; kept as a property so the long-standing
        private-attribute surface (persistence, tests) is unchanged."""
        return self._context_gauge.cursor

    @_msg_count_before_stream.setter
    def _msg_count_before_stream(self, value: int):
        self._context_gauge.record_stream_start(value)

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
        # A stop() issued in the create_task()→run() window (a quit sequence
        # firing before this task's first line) has already set _stop_event
        # with its reason — it targets this run, so honour it. Otherwise
        # reset to INTERRUPT so a reason left by an earlier run never leaks
        # into this run's cancel handling.
        _stop_pending = self._stop_event.is_set()
        self._stop_event.clear()
        if not _stop_pending:
            self._stop_reason = StopReason.INTERRUPT

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

                # Don't start a turn while MCP is (re)connecting — the tool
                # list is partial and a turn started now would need the whole
                # history re-prompted once the connect finishes. Queued items
                # stay in the queue (visible, deletable) until it returns.
                while (
                    self._running
                    and not self._stop_event.is_set()
                    and self._mcp is not None
                    and self._mcp.connecting
                ):
                    await asyncio.sleep(0.25)
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
                        await self._acquire_polite()
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
            self._repair_history_if_interrupted()
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

    def stop(self, reason: StopReason = StopReason.INTERRUPT) -> None:
        """Signal the agent to stop processing.

        Args:
            reason: Why we're stopping. INTERRUPT (ESC) marks the history
                for repair at the cancellation point. QUIT (app exit)
                leaves history untouched so a mid-turn pause survives the
                auto-save and --continue can resume it exactly.

        Works even if called before run() has executed its first line
        (the create_task()→run() window): run() detects the pending stop
        via _stop_event and honours the recorded reason instead of
        resetting it.
        """
        self._stop_reason = reason
        self._running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused so run() can exit

    def _repair_history_if_interrupted(self) -> None:
        """Repair history on a true interrupt (ESC); preserve mid-turn on quit.

        Called from both CancelledError handlers (run() between items and
        _process_item mid-item). On a quit-path stop (StopReason.QUIT) the
        history must stay mid-turn so the auto-save preserves the exact
        resume point (kv-cache prefix) for --continue + /resume.
        """
        if self._stop_reason == StopReason.INTERRUPT:
            self.history.repair_interrupted()

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
                await self._acquire_polite(resume_status=AgentStatus.WAITING)
                _polite_acquired = True
            except asyncio.CancelledError:
                # ESC or task cancellation during the polite wait.
                # acquire() already released any hold; just restore IDLE
                # and re-raise so the caller can clean up.
                await self._set_status(AgentStatus.IDLE)
                raise

        try:
            # Drop a trailing [Interrupted] marker before dispatching: it
            # exists only to keep an in-process history API-valid after an
            # ESC. On resume the turn continues from the real last message
            # (tool result or user message) — an assistant message at the
            # tail would be an invalid last role for the next LLM call.
            if self.history.messages and self.history.messages[-1].get("interrupt"):
                self.history.messages.pop()

            # Resume path: the restored history's last genuine user message
            # is this turn's standing instruction (the else-branch capture
            # is not on this path). Captured BEFORE _check_threshold_pre_turn
            # below, which may restart (and clear) the history.
            self._capture_in_flight_user_request()

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
                    # Mid-turn resume: the turn continues into _llm_turn
                    # below, so report WAITING rather than IDLE.
                    await self._wait_if_paused(resume_status=AgentStatus.WAITING)

                # Turn-start threshold check: a loaded/restored session may be
                # over-threshold (a previous turn can end over-threshold
                # without a mid-turn hit). Compact before the resumed turn's
                # first LLM call so it starts on a compacted context.
                await self._check_threshold_pre_turn(_polite_acquired=_polite_acquired)

                # Now call LLM to continue
                await self._llm_turn(_polite_acquired=_polite_acquired)
                # Post-turn journal reflection (Stage 12 equivalent): the
                # resume path has no queue item, so _process_item's Stage 12
                # never runs for a resumed turn. Without this, the turn's
                # tool work stays un-compacted and the NEXT turn's
                # retrospective_compact fires the reflection before the
                # user's new message is processed (10-06 edmac incident:
                # 163k-token prefill, 10-min delay on a simple question).
                # Internal guards make it a no-op when not applicable.
                await self.journal.maybe_reflect_after_turn()
                return True

            elif self.history.has_incomplete_turn():
                # Case 2: Last message is tool result, call LLM to process.
                # Turn-start threshold check first (see case 1 above).
                await self._check_threshold_pre_turn(_polite_acquired=_polite_acquired)
                await self._llm_turn(_polite_acquired=_polite_acquired)
                # Post-turn journal reflection — see case 1 above.
                await self.journal.maybe_reflect_after_turn()
                return True

            return False
        finally:
            # Polite mode safety net: release the lock if still held.
            # _llm_turn normally releases after each LLM stream (before tool
            # execution), so on normal exit the lock is already free. This
            # covers error/cancel paths where _llm_turn may not have
            # had a chance to release.
            if self.polite_lock is not None and self.polite_lock.is_held():
                self.polite_lock.release()
            # Auto-context safety net: the resume path calls _llm_turn but has
            # no post-turn handler of its own, so a threshold hit during the
            # resumed turn would orphan the flags and idle over-threshold.
            # Run the handler here (guarded by the flags; a no-op if not armed).
            if self._report_and_compact_triggered or self._auto_context_triggered:
                try:
                    await self._run_auto_context_post_turn(
                        _polite_acquired=_polite_acquired
                    )
                except BaseException:
                    # Must not mask the original exception.
                    pass

    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self._status

    async def _wait_if_paused(self, resume_status: AgentStatus | None = None) -> None:
        """Wait while the agent is paused.

        This is called at safe pause points. When pause is requested,
        this will block until resume() is called.

        Args:
            resume_status: Status to report when the wait ends, for callers
                that resume *mid-turn* (the turn continues immediately after
                this returns). Omit it for the run loop's between-items pause,
                where the next state is decided by the queue. Without this a
                mid-turn resume would report IDLE while the agent is still
                working — the status bar would read "idle", and the TUI's idle
                handling would end the turn (elapsed timer reset, bell).
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
                if resume_status is not None:
                    # Mid-turn resume: the caller continues the turn, so the
                    # queue is irrelevant — report what it is about to do.
                    await self._set_status(resume_status)
                elif self.queue.pending_count > 0:
                    # Skip IDLE if there are queued items — go straight to
                    # WAITING so the user never sees "ready" between resume
                    # and processing the next item.
                    await self._set_status(AgentStatus.WAITING)
                else:
                    await self._set_status(AgentStatus.IDLE)
                await self.emit(AgentEvent.RESUMED, {})

    async def _acquire_polite(self, resume_status: AgentStatus | None = None) -> None:
        """Acquire the polite lock, honouring a pause request while waiting.

        /pause during a lock wait must PAUSE, not interrupt: acquire() raises
        PolitePauseRequested when the pause flag is set, so we go PAUSED and
        block at _wait_if_paused(), then re-enter acquire() on resume. The
        queue item (pending at the top of the loop, current mid-turn) and the
        turn state are untouched — no [Interrupted] marker, no item
        completion. Loop (not one-shot) because the user can pause again
        while we wait for the lock after resuming.

        Pass resume_status=WAITING from mid-turn call sites (the turn
        continues immediately after the wait); omit it at the top of the run
        loop, where the queue decides the next state.
        """
        while True:
            try:
                await self.polite_lock.acquire()
                return
            except PolitePauseRequested:
                await self._wait_if_paused(resume_status=resume_status)

    def _estimate_current_context_tokens(self) -> int:
        """Estimate the true current context size at the safe point.

        Thin forward to the context gauge (agent13.context_gauge) — the
        grounded value, the stream cursor, the stale-grounding (shrink)
        guard, and the chars/4 fallback all live there now. Kept as a
        method (not inlined at the two decision sites) so the auto-context
        decision path stays greppable and mockable.
        """
        return self._context_gauge.estimate(self.messages, len(self.system_prompt))

    def _invalidate_context_grounding(self) -> None:
        """Re-ground the context gauge after a model swap.

        Token counts are tokenizer-specific: identical content measured
        10.2% apart on Swift vs DeepSeek (observed 2026-09-29). The
        stale-grounding guard cannot detect a swap - the message count is
        unchanged, so the old tokenizer's grounded count would be trusted
        as-is. Re-ground to a fresh chars/4 estimate of the current
        history (system + messages) - the same value the guard's shrink
        fallback computes - and move the stream cursor to the end of the
        history. The next pre-turn check decides on the estimate until the
        next LLM call re-grounds with the new tokenizer's actual usage.
        """
        self._context_gauge.reground(self.messages, len(self.system_prompt))

    async def _estimate_post_compact_context(self) -> int:
        """Re-ground the context gauge to the post-compact context size.

        System prompt + tools schema + current (compacted) messages, using
        the gauge's chars/4 heuristic. The tools schema is the largest
        unknown, so serialize the live tool list; on failure fall back to
        system prompt + messages only. Returns the new grounded value (for
        CONTEXT_ESTIMATE emission) and moves the stream cursor to the end
        of the (rewritten) history.
        """
        tools_len = None
        try:
            tools = await self.get_all_tools()
            tools_len = len(json.dumps(tools)) // 4
        except Exception:
            pass
        return self._context_gauge.reground(
            self.messages, len(self.system_prompt), tools_len
        )

    def _save_auto_context_snapshot(self, n: int) -> None:
        """Save pre-compact history to a dated snapshot file.

        Uses the same location and date pattern as the auto-save, with a
        ``_N`` suffix (N = per-session auto-context count) so snapshots never
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

    def _save_degenerate_snapshot(self, n: int) -> None:
        """Save the history plus a degenerate partial to a snapshot file.

        Sibling of ``_save_auto_context_snapshot``: same save_context call,
        same dated location, ``_degenerate_N`` name, best-effort. Called
        whenever the guard destroys or drops the partial output (the stop
        path, and a nudge whose tail is a whitespace wall) — the snapshot
        is the evidence the dropped output would otherwise leave behind.

        The partial is temp-appended for the save and popped right after
        (no await in between, so no interleaving). It is a PLAIN assistant
        message, so the snapshot's ``incomplete_turn`` field reads False —
        this file is a post-mortem artifact, not a resume point.
        """
        partial = None
        if self._degenerate_partial is not None:
            partial = self._degenerate_partial
            self.messages.append(partial)
        try:
            from agent13.persistence import get_auto_save_path, save_context

            base = get_auto_save_path(session_date=self.session_date)
            path = base.parent / f"{base.stem}_degenerate_{n}.ctx"
            save_context(self, path)
        except Exception:
            # Snapshot is best-effort; a failure here must not kill the turn.
            pass
        finally:
            if partial is not None:
                try:
                    self.messages.remove(partial)
                except ValueError:
                    pass

    def _classify_degenerate_end(
        self, content: str, reasoning: str, finish_reason: str | None
    ) -> str:
        """Decide what to do with a no-tool-calls stream end.

        Returns:
            "none"  - normal end (caller appends the response and finishes)
            "nudge" - auto-recover (caller applies the nudge, loop continues)
            "stop"  - give up (caller snapshots, marks incomplete, breaks)

        A degenerate end is a cap-hit (``finish_reason == "length"``) or a
        whitespace-wall stop (visible content is all whitespace — the
        empty-stream retry only covers zero-byte streams). The streak counts
        CONSECUTIVE degenerate ends this turn and reaches the stop at
        ``auto_context_nudge_cap`` (2: first end nudges, second stops).
        Clean completions reset it: a tool-calls round, a chain restart,
        a new user turn. Over the auto-context threshold the normal
        wrap-up/compact/chain machinery owns the recovery (decision 2) —
        the compact replaces the history anyway, so the partial never
        poisons anything. cap <= 0 disables the guard.
        """
        if self.auto_context_nudge_cap <= 0:
            return "none"
        hit_cap = finish_reason == "length"
        garbage_wall = finish_reason == "stop" and not content.strip()
        if not (hit_cap or garbage_wall):
            return "none"
        # Priority rule: over threshold, skip the nudge and let the turn
        # end - the threshold machinery (wrap-up/compact/chain) takes over.
        estimated = self._estimate_current_context_tokens()
        if self.auto_context_threshold > 0 and estimated >= self.auto_context_threshold:
            return "none"
        self._degenerate_streak += 1
        if self._degenerate_streak >= self.auto_context_nudge_cap:
            return "stop"
        return "nudge"

    def _build_degenerate_partial(
        self, content: str, reasoning: str
    ) -> dict:
        """The partial output as a message dict (for snapshot preservation)."""
        partial = {"role": "assistant", "content": content}
        if reasoning:
            partial["reasoning_content"] = reasoning
        return partial

    def _snapshot_degenerate_partial(self, content: str, reasoning: str) -> None:
        """Snapshot the current history plus the degenerate partial.

        Called whenever the partial is about to leave the history (a dropped
        nudge, or the stop path) - the snapshot is the evidence the dropped
        output would otherwise leave behind (the 09-16 user requirement:
        "we are removing the degenerated turn it would be good to save it
        for later inspection"). Best-effort.
        """
        self._degenerate_snapshot_count += 1
        self._degenerate_partial = self._build_degenerate_partial(content, reasoning)
        self._save_degenerate_snapshot(self._degenerate_snapshot_count)
        self._degenerate_partial = None

    async def _apply_degenerate_outcome(
        self, decision: str, content: str, reasoning: str, finish_reason: str | None
    ) -> bool:
        """Act on a degenerate-end decision (mutates state; emits + logs).

        Args:
            decision: "nudge" or "stop" (from _classify_degenerate_end).
            content: The partial visible output.
            reasoning: The partial reasoning output.
            finish_reason: The terminal chunk's finish_reason (for logs/notices).

        Returns:
            True when the caller should CONTINUE the turn loop (nudge);
            False when it should break (stop - the turn is given up).
        """
        drop = _degenerate_tail_is_garbage(content, reasoning)
        if decision == "nudge":
            if drop:
                # Whitespace wall: the partial carries no information and
                # would poison the context - snapshot it, then drop it.
                self._snapshot_degenerate_partial(content, reasoning)
            else:
                # Keep the partial: the model continues from its own last
                # words ("continue exactly where you stopped").
                append_assistant_message(self.messages, content, reasoning)
                log_assistant_response(content, reasoning if reasoning else None)
            self.messages.append(
                {"role": "user", "content": DEGENERATE_NUDGE_PROMPT, "injected": True}
            )
            self._degenerate_nudges_total += 1
            if is_debug_enabled():
                log_degenerate_nudge(
                    {
                        "finish_reason": finish_reason,
                        "content_length": len(content),
                        "reasoning_length": len(reasoning),
                        "kept": not drop,
                        "streak": self._degenerate_streak,
                        "cap": self.auto_context_nudge_cap,
                        "nudges_total": self._degenerate_nudges_total,
                    }
                )
            cause = (
                "Model output was cut off (cap-hit)"
                if finish_reason == "length"
                else "Model output was empty/repetitive"
            )
            await self.emit(
                AgentEvent.DEGENERATE_NUDGE,
                {
                    "message": (
                        f"{cause} - nudged to continue "
                        f"({self._degenerate_streak}/{self.auto_context_nudge_cap})"
                    ),
                    "level": "info",
                    "action": "nudge",
                },
            )
            return True

        # Stop: the streak reached the cap. Snapshot the partial (it is
        # about to leave the history), mark the turn incomplete (the
        # empty-stream-exhaustion twin: /resume re-runs the turn from the
        # pre-stream history tail), notify loudly.
        self._degenerate_partial = {
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            self._degenerate_partial["reasoning_content"] = reasoning
        self._degenerate_snapshot_count += 1
        self._save_degenerate_snapshot(self._degenerate_snapshot_count)
        self._degenerate_partial = None
        self.mark_incomplete_turn(True)
        if is_debug_enabled():
            log_degenerate_stop(
                {
                    "finish_reason": finish_reason,
                    "content_length": len(content),
                    "reasoning_length": len(reasoning),
                    "streak": self._degenerate_streak,
                    "cap": self.auto_context_nudge_cap,
                    "nudges_total": self._degenerate_nudges_total,
                }
            )
        await self.emit(
            AgentEvent.DEGENERATE_NUDGE,
            {
                "message": (
                    "Model kept degenerating "
                    f"({self._degenerate_streak}/{self.auto_context_nudge_cap} "
                    "cut-off ends) - I stopped this turn. Snapshot saved; "
                    "/resume retries it, or rephrase to start fresh."
                ),
                "level": "warning",
                "action": "stop",
            },
        )
        return False

    async def _check_threshold_pre_turn(self, _polite_acquired: bool = False) -> None:
        """Turn-start threshold check.

        A turn can end over-threshold without the mid-turn safe point ever
        firing (a final response has no tool boundary), so when the user
        asks for the next thing — a new message, or /resume — check before
        that turn's first LLM call. If over-threshold, run the configured
        action (wrap-up and/or compact) so the new work starts on a compacted
        context. A finished turn is left alone: the compact happens when the
        user continues, not the moment the agent goes idle.

        No chain restart: the incoming user message (or the resumed turn) is
        the continuation — the caller runs the turn itself after this
        returns. action=none pauses and waits for the user to intervene
        (/compact, /model, /journal all) before the turn proceeds.
        """
        estimated_context = self._estimate_current_context_tokens()
        # Publish the live context estimate so the UI can show it (with a
        # "~") until the next LLM call reports the grounded count via
        # TOKEN_USAGE.
        await self.emit(
            AgentEvent.CONTEXT_ESTIMATE,
            {"estimated_tokens": estimated_context},
        )
        if not self._should_check_threshold(estimated_context):
            return
        action = await self._handle_threshold_check(estimated_context)
        if action == "pause_wait":
            # action=none: paused; wait for the user to intervene and
            # /resume before the turn proceeds.
            await self._wait_if_paused(resume_status=AgentStatus.WAITING)
            return
        if self._report_and_compact_triggered or self._auto_context_triggered:
            await self._run_auto_context_post_turn(
                _polite_acquired=_polite_acquired, restart=False
            )

    async def _run_auto_context_post_turn(
        self, _polite_acquired: bool = False, restart: bool = True
    ) -> None:
        """Run the pending auto-context action.

        Called from process_item and the resume path after each _llm_turn
        (the threshold was hit mid-turn; _handle_threshold_check armed the
        flags and broke out of the tool loop), and from
        _check_threshold_pre_turn before a turn's first LLM call (a previous
        turn ended over-threshold without a mid-turn hit). Runs the pending
        wrap-up (report_and_compact), compacts, then either restarts the
        turn (chain budget remaining) or finishes.

        restart=False (pre-turn): no nudge + _llm_turn restart — the incoming
        user message (or the resumed turn) is the continuation, and the
        caller runs the turn itself after this returns.

        The loop shape matters: a restarted _llm_turn can re-trip the
        threshold and set the flags again — the while picks that up; do not
        restructure into recursion.
        """
        while self._report_and_compact_triggered or self._auto_context_triggered:
            if self._report_and_compact_triggered:
                # Leave the flag set: _run_report_and_compact_turn checks it as
                # its no-op guard (a direct call without the flag does nothing)
                # and clears it itself. Clearing it here first would make that
                # guard see False and skip the wrap-up turn entirely.
                await self._run_report_and_compact_turn(
                    _polite_acquired=_polite_acquired
                )
                continue  # proceed to the pending compact
            if self._auto_context_triggered:
                self._auto_context_triggered = False
                # Snapshot the pre-compact history (safety net). Named with a
                # per-session monotonic counter so snapshots never collide.
                self._auto_context_snapshot_count += 1
                self._save_auto_context_snapshot(self._auto_context_snapshot_count)

                chat_cleared = False
                # Chain budget spent: the persistent chat footer says so.
                # The wipe + rebuild below would otherwise leave no trace
                # in the window of why the turn stopped (the toast is
                # transient).
                exhaustion_line = (
                    f"Auto-context: {self._auto_context_chain_used}/"
                    f"{self.auto_context_chain} restarts used — turn "
                    "stopped, context compacted. Send a message to "
                    f"continue. (session total: {self._chain_count})"
                    if restart
                    and self._auto_context_chain_used
                    >= self.auto_context_chain
                    else None
                )
                if self.auto_context_action == "report_and_compact":
                    # Wrap-up ran: mechanical restart (no AI call).
                    chat_cleared = True
                    await self._mechanical_restart(system_line=exhaustion_line)
                    self._auto_context_failures = 0
                else:
                    # No wrap-up: the verbose compact summary is the only
                    # carrier. Original AI-call path.
                    success, msg = await self.compact_history("")
                    if not success:
                        self._auto_context_failures += 1
                        if is_debug_enabled():
                            log_tps_event(
                                "auto_context_decision",
                                {
                                    "reason": "compact_failed",
                                    "chain": self.auto_context_chain,
                                    "chain_used": self._auto_context_chain_used,
                                },
                            )
                        await self.emit(
                            AgentEvent.NOTIFICATION,
                            {
                                "message": (
                                    f"Auto-context: compact failed — idle. {msg}"
                                ),
                                "level": "error",
                            },
                        )
                        break
                    self._auto_context_failures = 0
                if restart and self._auto_context_chain_used < self.auto_context_chain:
                    self._auto_context_chain_used += 1
                    if not chat_cleared:
                        await self.emit(AgentEvent.CHAT_CLEAR, {})
                    # Nudge the model to continue. The durable records
                    # (mechanical path) or the compact summary's embedded
                    # pointer block (compact-only path) carry the file
                    # orientation — no redundant cues here. When the turn
                    # carries a captured user instruction (set at turn
                    # start), re-inject it VERBATIM: the user's standing
                    # instruction is the intent source across chains (a
                    # mid-turn question must not be wiped out by the
                    # restart).
                    if self._in_flight_user_request:
                        nudge = AUTO_CONTEXT_CARRY_HINT.format(
                            instruction=self._in_flight_user_request
                        )
                    else:
                        nudge = AUTO_CONTEXT_CONTINUE_HINT
                    # "injected" keeps the nudge inside the turn it lands in
                    # (is_turn_start) and off the API wire (LOCAL_MSG_KEYS) -
                    # it carries no user intent. The TUI's chain-restart
                    # rebuild relies on this: the kept "last turn" must be
                    # the compacted summary pair, not the nudge.
                    self.messages.append(
                        {"role": "user", "content": nudge, "injected": True}
                    )
                    if not chat_cleared:
                        # Compact-only path: the mechanical restart already
                        # emitted CHAT_REBUILD; this path needs its own so
                        # the TUI renders the compact summary + nudge before
                        # the continuation turn streams in.
                        await self.emit(AgentEvent.CHAT_REBUILD, {})
                    await self._llm_turn(_polite_acquired=_polite_acquired)
                else:
                    if is_debug_enabled():
                        log_tps_event(
                            "auto_context_decision",
                            {
                                "reason": (
                                    "continuing" if not restart else "chain_exhausted"
                                ),
                                "chain": self.auto_context_chain,
                                "chain_used": self._auto_context_chain_used,
                            },
                        )
                    await self.emit(
                        AgentEvent.NOTIFICATION,
                        {
                            "message": (
                                "Auto-context: compacted — continuing"
                                if not restart
                                else "Auto-context: compacted — idle, over to you"
                            ),
                            "level": "info",
                        },
                    )
                    break

    async def _run_report_and_compact_turn(
        self, _polite_acquired: bool = False
    ) -> None:
        """Run the report-and-compact wrap-up turn.

        Called from the post-turn handler when the threshold was hit mid-turn
        and action = "report_and_compact". Snapshots the full history (safety
        net), injects the wrap-up prompt (with the current date and time
        appended so the model can write a dated journal heading), and runs
        ONE turn with the full tool set so the model can append its session
        journal entry while it still has full context. Auto-context is
        suppressed for the duration so the threshold can't re-fire mid-wrap-up;
        afterwards the pending compact proceeds. Fires once per chain cycle —
        the per-turn chain counter (plus suppression) replaces the old
        once-per-session latch.

        When an explicit focus name is set in state (/focus NAME,
        /wrapup NAME, --focus NAME), it is injected into the prompt so the
        name is used for the {focus} files even if it was never said in the
        conversation. The auto threshold path relies on the same state.
        """
        if not self._report_and_compact_triggered:
            # No pending wrap-up: a direct call without the flag is a no-op.
            return
        self._report_and_compact_triggered = False
        self._auto_context_snapshot_count += 1
        self._save_auto_context_snapshot(self._auto_context_snapshot_count)
        # Inject a fake assistant message first. The threshold is hit straight
        # after a tool result, and strict chat templates (Jinja) reject a bare
        # user message following a tool message: "conversation roles must
        # alternate user and assistant roles except for tool calls and
        # results". Same trick the mid-turn interrupt path uses.
        self.messages.append(
            {
                "role": "assistant",
                "content": "[Wrapping up]",
                "report_and_compact": True,
            }
        )
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [self.report_and_compact_prompt]
        if self._focus_name:
            parts.append(
                f"The user has specified the focus name for this wrap-up: "
                f'"{self._focus_name}". Use it for {{focus}} in both file '
                "names and headings; it overrides any name from the "
                "conversation."
            )
        parts.append(f"Current date and time: {now}")
        self.messages.append(
            {
                "role": "user",
                "content": "\n\n".join(parts),
                "report_and_compact": True,
            }
        )
        start_len = len(self.messages)
        self._suppress_auto_context = True
        try:
            await self._llm_turn(_polite_acquired=_polite_acquired)
            # Capture the journal path the model chose (its own append tool
            # call) so the continuation nudge can carry it into every chain
            # restart.
            self._capture_session_journal_path(self.messages[start_len:])
        finally:
            self._suppress_auto_context = False

    # ── Durable records (agent13.durable_records) ─────────────────────────
    #
    # The focus/record state and its derivations live in DurableRecords;
    # these are the thin forwards that keep the Agent attribute surface the
    # UIs, persistence and the existing tests use. History is passed in,
    # never bound. The two record readers deliberately have NO forward here:
    # patching agent._read_record_file would be a silent no-op, so tests
    # patch agent.durable_records.read_record_file instead - the seam the
    # module exists to make single.

    @property
    def _focus_name(self) -> str:
        """Explicit focus name (authoritative when set) - forwarded."""
        return self.durable_records.focus_name

    @property
    def _session_truth_path(self) -> str:
        """Current-truth record path - forwarded to the durable records."""
        return self.durable_records.truth_path

    @_session_truth_path.setter
    def _session_truth_path(self, value: str) -> None:
        self.durable_records.truth_path = value

    @property
    def _session_journal_path(self) -> str:
        """Session journal record path - forwarded to the durable records."""
        return self.durable_records.journal_path

    @_session_journal_path.setter
    def _session_journal_path(self, value: str) -> None:
        self.durable_records.journal_path = value

    @property
    def _session_focus_name(self) -> str:
        """Effective focus name (explicit, else path-derived) - forwarded."""
        return self.durable_records.name

    def set_focus_name(self, name: str) -> None:
        """Set the explicit focus name; the record paths derive from it."""
        self.durable_records.set_focus_name(name)

    async def apply_focus_name(self, name: str) -> tuple[str, str]:
        """Apply /focus [NAME] immediately (the UI command path).

        Thin async forward onto DurableRecords.apply_focus - see there for
        the semantics and the (message, level) contract the UIs present.
        """
        return self.durable_records.apply_focus(name, self.messages)

    def seed_from_records(self, force: bool = False) -> bool:
        """Seed the message history from the focus's durable records."""
        return self.durable_records.seed(self.messages, force=force)

    def _capture_session_journal_path(self, new_messages: list[dict]) -> None:
        """Capture the record paths from the wrap-up's own tool calls."""
        self.durable_records.capture_paths(new_messages)

    def _wrapup_is_current(self) -> bool:
        """True when the durable record is still current (no tool calls
        since the last wrap-up completed)."""
        return self.durable_records.is_wrapup_current(self.messages)

    async def _mechanical_restart(
        self, carry_skills: bool = True, system_line: str | None = None
    ) -> bool:
        """Clear history and inject the pre-baked intro + durable records.

        No AI call. Used by the auto-chain post-wrap-up path and the
        /chain command. The intro is merged into the records message
        (single user message, clean role alternation). When carry_skills
        is True (default) and skills were manually loaded this session,
        their content is injected as a section between the intro and the
        records (order: intro -> skills -> truth -> journal). system_line
        overrides the persistent chat footer the TUI writes with the
        CHAT_REBUILD (the chain-exhausted path passes the exhaustion
        notice, so the wipe + rebuild leaves a visible record of why the
        turn stopped). Returns True if records were injected, False if no
        record content was found.
        """
        self.clear_messages()
        await self.emit(AgentEvent.CHAT_CLEAR, {})
        # Full history replacement = one chain (session-lifetime total).
        self._chain_count += 1
        # Fresh context = a clean completion for the degeneration guard: the
        # streak counts CONSECUTIVE degenerate ends, and the restart is the
        # strongest possible reset between them.
        self._degenerate_streak = 0
        intro = self.durable_records.restart_intro()
        skills_section = (
            self._format_active_skills_section() if carry_skills else ""
        )
        prefix = intro
        if skills_section:
            prefix += "\n\n" + skills_section
        seeded = self.seed_from_records()
        if self.messages:
            # Prepend intro (+ skills) to the records message (single user
            # message).
            self.messages[0]["content"] = prefix + "\n\n" + self.messages[0]["content"]
        else:
            # No records found — intro (+ skills) stands alone.
            self.messages.append(
                {"role": "user", "content": prefix, "injected": True}
            )
        # Tell the TUI to rebuild its display from the now-populated history.
        # The CHAT_CLEAR above already wiped the display; this repopulates it
        # so the user sees the injected records instead of a blank window.
        await self.emit(
            AgentEvent.CHAT_REBUILD,
            {
                "system_line": (
                    system_line
                    or "Context compacted — showing session summary"
                )
            },
        )
        post_est = await self._estimate_post_compact_context()
        await self.emit(
            AgentEvent.CONTEXT_ESTIMATE,
            {"estimated_tokens": post_est},
        )
        return seeded

    @property
    def auto_context_chain_used(self) -> int:
        """Chain restarts consumed by the current turn (0..auto_context_chain).

        Public view of _auto_context_chain_used for UIs (TUI status bar and
        /status). Resets to 0 at the start of each user turn.
        """
        return self._auto_context_chain_used

    @property
    def chain_count(self) -> int:
        """Session-lifetime compaction total for UIs (status bar, /status).

        Counts every full history replacement (compacts + chain restarts).
        Survives save/load; reset only by /new_session.
        """
        return self._chain_count

    @property
    def session_started_at(self) -> str:
        """ISO timestamp of the original session start.

        Persisted in the .ctx save, so a session resumed days later keeps
        its true start time (survives save/load; reset by /new_session).
        """
        return self._session_started_at

    @property
    def degenerate_nudges(self) -> int:
        """Session-lifetime degeneration-guard nudge count (for /status).

        Incremented each time the guard nudges a degenerate stream end
        (cap-hit or whitespace wall). Persisted in the .ctx save so a
        resumed session keeps its total (chain_count pattern).
        """
        return self._degenerate_nudges_total

    def reset_session_counters(self) -> None:
        """Start a new session identity: compaction total to 0, start to now."""
        self._chain_count = 0
        self._session_started_at = datetime.datetime.now().isoformat()

    def _should_check_threshold(self, estimated_context: int) -> bool:
        """Gate for the auto-context threshold check at a safe point.

        Extracted as a predicate (rather than inlined in _llm_turn) so the four
        conditions are testable without copying them into the test.
        """
        return (
            self.auto_context_threshold > 0
            and estimated_context >= self.auto_context_threshold
            and self._auto_context_failures < 3
            and not self._suppress_auto_context
        )

    def _capture_in_flight_user_request(self) -> str:
        """Store this turn's user instruction from the current history.

        Used by the resume path, where the turn-start user message comes
        from the restored history (the normal path captures item.text
        explicitly where it is appended). Scans for the last genuine user
        message (is_turn_start — excludes injected nudges, interrupt
        markers, wrap-up prompts) and stores it as the standing
        instruction re-injected verbatim at each chain restart. A no-op
        when an instruction is already stored (the turn-start capture
        wins). Cleared by _reset_auto_context_turn_state at each
        user-turn start, so a manual /chain while idle carries nothing
        stale (falls back to the bare nudge).
        """
        if self._in_flight_user_request:
            return self._in_flight_user_request
        for msg in reversed(self.messages):
            if is_turn_start(msg):
                self._in_flight_user_request = str(msg.get("content", ""))
                break
        return self._in_flight_user_request

    def _reset_auto_context_turn_state(self) -> None:
        """Reset per-turn auto-context state at the start of each user turn.

        The chain budget is per-turn (chain=N means N restarts every turn, not
        N for the session). The flag clears are hygiene: a flag left set by a
        failed wrap-up or compact cannot leak into the next turn. The
        degeneration streak resets here too — a new user turn is a clean
        completion as far as the guard is concerned.
        """
        self._auto_context_chain_used = 0
        self._auto_context_triggered = False
        self._report_and_compact_triggered = False
        self._in_flight_user_request = ""
        self._degenerate_streak = 0

    async def _handle_threshold_check(self, estimated_context: int) -> str:
        """Handle the auto-context threshold being hit at a safe point.

        Called from _llm_turn when _should_check_threshold passes. Decides the
        action, emits the user notice, and arms the flags the post-turn handler
        consumes — the actual work (wrap-up turn, compact, nudge) happens there.

        Args:
            estimated_context: Estimated current context tokens.

        Returns:
            Control action for the caller:
              "break"      - leave _llm_turn's loop; the post-turn handler
                              (wrap-up and/or compact) takes over
              "pause_wait" - pause() was called (action=none); wait, then
                              continue the loop
              "continue"   - nothing applied (caller shouldn't get this for
                              an over-threshold estimate, but be safe)
        """
        # action=none: no automatic action. Pause with context untouched so the
        # user can /model, /compact, /journal all, then /resume. Chain is
        # meaningless without a compact, so it is ignored.
        if self.auto_context_action == "none":
            if is_debug_enabled():
                log_tps_event(
                    "auto_context_check",
                    {
                        "prompt_tokens": self.prompt_tokens,
                        "estimated_context": estimated_context,
                        "threshold": self.auto_context_threshold,
                        "action": self.auto_context_action,
                        "chain": self.auto_context_chain,
                        "chain_used": self._auto_context_chain_used,
                    },
                )
            await self.emit(
                AgentEvent.NOTIFICATION,
                {
                    "message": (
                        f"Auto-context: threshold reached "
                        f"(~{estimated_context:,} tokens) — paused "
                        f"(action=none). /model to a larger-context model, "
                        f"/compact, or /journal all, then /resume."
                    ),
                    "level": "warning",
                },
            )
            self.pause()
            return "pause_wait"

        if is_debug_enabled():
            log_tps_event(
                "auto_context_check",
                {
                    "prompt_tokens": self.prompt_tokens,
                    "estimated_context": estimated_context,
                    "threshold": self.auto_context_threshold,
                    "action": self.auto_context_action,
                    "chain": self.auto_context_chain,
                    "chain_used": self._auto_context_chain_used,
                },
            )

        # NOTIFICATION, never ASSISTANT_TOKEN: a notice streamed into the
        # response buffer is orphaned when the next turn's STREAM_START resets
        # the display tracker (start_response() runs twice).
        if self.auto_context_action == "report_and_compact":
            await self.emit(
                AgentEvent.NOTIFICATION,
                {
                    "message": (
                        f"Report-and-compact: context at "
                        f"~{estimated_context:,} tokens — wrapping up"
                    ),
                    "level": "warning",
                },
            )
            # Wrap-up first, then the pending compact (post-turn handler).
            self._report_and_compact_triggered = True
            self._auto_context_triggered = True
            return "break"

        # action=compact (default)
        await self.emit(
            AgentEvent.NOTIFICATION,
            {
                "message": (
                    f"Auto-context: context at ~{estimated_context:,} tokens "
                    f"— compacting"
                ),
                "level": "info",
            },
        )
        self._auto_context_triggered = True
        return "break"

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

    def _bound_tool_result_text(self, text: str, tool_name: str) -> str:
        """Bound a tool result's text per [tools] max_result_chars.

        A per-tool override (declared on the tool function via
        @tool(max_result_chars=...)) wins over the global config. 0 = off.
        Over-cap text is spilled to a temp file; the model receives
        head + marker + tail (agent13.tool_output_bound.bound_tool_result).
        """
        if not text:
            return text
        from agent13.tool_output_bound import bound_tool_result
        from agent13.tools import get_tool_max_result_chars
        from agent13.config import get_config

        cap = get_tool_max_result_chars(tool_name)
        if cap is None:
            # The real Config always sets this (default 30000); fall back to
            # the design cap if the object lacks it (a minimal test mock).
            cap = getattr(get_config(), "tools_max_result_chars", 30_000)
        bounded, _spill = bound_tool_result(text, cap, tool_name)
        if bounded is not text:
            # bound_tool_result returns the input object unchanged when no
            # bounding happened; a new string means the marker was inserted.
            # Count it even when the spill write failed - the model still
            # received truncated text.
            self.tool_stats.bounded += 1
        return bounded

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

        # Phase 1 tool output bounding: over-cap text is spilled to a temp
        # file; the model gets head + marker + tail (the marker names the
        # spill file). Applies to the text of BOTH branches - image payloads
        # pass through untouched (Phase 2).
        text = self._bound_tool_result_text(text, tool_name)

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
            # Unreadable images (corrupt/truncated) become text notes — sending
            # them would make the provider reject the whole request with a 500.
            content_parts = [
                {"type": "text", "text": f"[Image from tool: {tool_name}]"}
            ]
            for uri in images:
                resized = resize_image_uri(uri, NATIVE_MAX_DIMENSION)
                if image_uri_decodable(resized):
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": resized}}
                    )
                else:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": (
                                "[Image unreadable — file may be corrupt or "
                                "truncated; try re-fetching it]"
                            ),
                        }
                    )
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
                try:
                    desc = await describe_image(
                        sidecar_config, vision.sidecar_model or None, uri, prompt
                    )
                except Exception as e:
                    # Sidecar failure (e.g. 500 on a corrupt image) must not
                    # kill the turn — note it so the model can re-fetch.
                    log_error(
                        e, {"context": "sidecar_describe_image", "tool": tool_name}
                    )
                    desc = (
                        f"[sidecar failed to describe image: {e} — "
                        "the image may be corrupt or truncated; try re-fetching it]"
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
        # One queue item = one user turn: reset the per-turn tool-call
        # numbering so each turn's calls number from 1.
        self._turn_tool_seq = 0

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
                item_data = item.data or {}
                compact_prompt = item_data.get("compact_prompt", "")
                # A wrap-up is still current (it ran and no tool calls - major
                # work - happened since) and the user gave no explicit prompt:
                # use the short bridge summary (the files carry the details),
                # matching the auto path. An explicit prompt (--prompt / focus
                # text) always wins.
                if self._wrapup_is_current() and not item_data.get(
                    "user_specified_prompt", False
                ):
                    compact_prompt = DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT
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
            elif item.kind == "wrapup":
                # Manual /wrapup: bring the durable record up to date, then
                # STOP - no compact, no chain restart. The session ends idle
                # with the full history intact (user can /save it, or
                # /compact later to start a new chain). Same injected prompt
                # and turn machinery as the threshold-triggered wrap-up.
                # /wrapup NAME makes NAME the authoritative focus name (the
                # prompt reads it from state); bare /wrapup keeps the
                # existing name, if any. If context is left over threshold,
                # the pre-turn check handles it on the next user command.
                focus_name = (item.data or {}).get("focus_name")
                if focus_name:
                    self.set_focus_name(focus_name)
                self._report_and_compact_triggered = True
                await self._run_report_and_compact_turn(
                    _polite_acquired=_polite_acquired,
                )
                await self.emit(
                    AgentEvent.NOTIFICATION,
                    {
                        "message": (
                            "Wrap-up complete - journal + current-truth "
                            "updated. History unchanged; /save to snapshot "
                            "or /compact to start a new chain."
                        ),
                        "level": "info",
                    },
                )
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
            elif item.kind == "chain":
                # /chain - manual "restart the loop": bring the durable
                # record up to date (wrap-up if stale), clear the history,
                # and re-inject the records. No compact summary - the
                # records ARE the context. Zero model calls when the
                # wrap-up is already current. "no-skills" arg: drop
                # manually-loaded skills from this restart AND forget
                # them (they are one /skillname away if still needed).
                no_skills = bool((item.data or {}).get("no_skills", False))
                if no_skills:
                    self.clear_active_skills()
                if not self._wrapup_is_current():
                    self._report_and_compact_triggered = True
                    await self._run_report_and_compact_turn(
                        _polite_acquired=_polite_acquired,
                    )
                seeded = await self._mechanical_restart(carry_skills=not no_skills)
                suffix = ", skills dropped" if no_skills else ""
                if seeded:
                    message = f"Chain restarted - records injected{suffix}"
                else:
                    message = (
                        "Chain restarted - no records found "
                        "(set a focus name first with /focus NAME)" + suffix
                    )
                await self.emit(
                    AgentEvent.NOTIFICATION,
                    {"message": message, "level": "info"},
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
                    # /new_session: wiping the history starts a NEW session
                    # identity - compaction total resets, start time is now.
                    self.reset_session_counters()
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

                # Reset the per-turn auto-context state before this turn
                # starts (chain budget + triggered flags).
                self._reset_auto_context_turn_state()

                # Turn-start threshold check: a previous turn may have ended
                # over-threshold without a mid-turn hit (a final response has
                # no tool boundary). Compact before this turn's first LLM
                # call so the new work starts on a compacted context. Runs
                # BEFORE the user message is appended — the compact replaces
                # the whole history and would swallow it.
                await self._check_threshold_pre_turn(_polite_acquired=_polite_acquired)

                # Add user message to history
                self.messages.append({"role": "user", "content": item.text})
                # This message is the turn's standing instruction: carried
                # verbatim into every chain restart's nudge so intent
                # survives the restarts (cleared at the next turn start).
                self._in_flight_user_request = item.text

                # Process with LLM (may include multiple tool call rounds)
                await self._llm_turn(_polite_acquired=_polite_acquired)

                # Auto-context: the threshold was hit mid-turn. Run the pending
                # wrap-up (report_and_compact), compact, then either restart the
                # turn (chain budget remaining) or go idle.
                await self._run_auto_context_post_turn(
                    _polite_acquired=_polite_acquired
                )

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
                self._repair_history_if_interrupted()
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
            # Auto-context safety net: if _llm_turn was interrupted before the
            # post-turn handler ran (above), the flags armed by
            # _handle_threshold_check are orphaned and the agent would idle
            # over-threshold with no wrap-up/compact. Running the handler here
            # (guarded by the flags; a no-op on the normal path) guarantees it
            # happens. Skipped on cancel — the ESC handler restarts the loop.
            if not self._cancel_requested and (
                self._report_and_compact_triggered or self._auto_context_triggered
            ):
                try:
                    await self._run_auto_context_post_turn(
                        _polite_acquired=_polite_acquired
                    )
                except BaseException:
                    # Must not mask the original exception that triggered this
                    # finally block.
                    pass

        # Skip IDLE if there are queued items — go straight to WAITING
        # so the TUI (and bell) never sees "ready" between queue items.
        # Mirrors the resume path logic above.
        if self.queue.pending_count > 0:
            await self._set_status(AgentStatus.WAITING)
        else:
            await self._set_status(AgentStatus.IDLE)

    def _jit_repair_loaded_turn(self) -> None:
        """JIT safety net: close a loaded mid-turn history if the API requires it.

        If the user typed a new message after --continue / /load without
        /resume first, the history must be API-valid before the LLM call:
        - pending tool_calls: synthesize missing tool results + [Interrupted]
          marker so the request doesn't 400 (API requires a result for every
          tool_call);
        - trailing user message: append the [Interrupted] marker, otherwise
          the new message would be consecutive users (breaks role alternation
          on strict backends).
        A trailing tool result or a complete assistant message needs NO
        repair — tool-then-user is valid, and inserting a marker would break
        the kv-cache prefix. /resume paths clear the flag first
        (continue_incomplete_turn), so they're unaffected.

        Called at _llm_turn entry (deterministic call site), not in
        _stream_and_emit — that's an async generator shared with the
        reflection path, where pre-yield code runs lazily on first
        iteration and any future caller would silently inherit the history
        mutation.
        """
        if self._incomplete_turn_loaded:
            last = self.history.messages[-1] if self.history.messages else None
            if last is not None and (
                last.get("role") == "user"
                or (last.get("role") == "assistant" and last.get("tool_calls"))
            ):
                self.history.repair_interrupted()
            self._incomplete_turn_loaded = False

    async def _close_llm_connections(self) -> None:
        """Close the LLM client's pooled HTTP connections (best effort).

        After a long pause the server's keep-alive timeout may have closed
        the pooled connection while we were idle; the next stream is then
        issued on a stale connection and ends with zero bytes. Closing the
        pool forces the next request onto a fresh connection.

        The pool (httpcore2.AsyncConnectionPool) stays usable after
        aclose() - it opens fresh connections on demand. Degrades to a
        no-op if the client's internal structure changes (or is a mock),
        so a failed close can never break the turn.
        """
        try:
            httpx_client = getattr(self.client, "_client", None)
            transport = getattr(httpx_client, "_transport", None)
            pool = getattr(transport, "_pool", None)
            if pool is not None:
                await pool.aclose()
        except Exception:
            # Best effort - the pool (if any) is simply left as-is.
            pass

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
        self._jit_repair_loaded_turn()
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
                        await self._acquire_polite(resume_status=AgentStatus.WAITING)

                # Stream response, capturing reasoning and tool calls.
                #
                # Empty-stream retry: after a long pause the pooled HTTP
                # connection has gone stale (the server's keep-alive timeout
                # elapsed while we were idle), so the stream can end with
                # zero bytes. For a streaming request that is
                # indistinguishable from an empty completion - no exception
                # is raised - so we detect "nothing came back" and re-issue
                # on a fresh connection.
                _empty_retries = 0
                while True:
                    content = ""
                    reasoning = ""
                    tool_calls = None
                    # Terminal finish_reason for this stream attempt (the
                    # degeneration guard's cap-hit signal). Reset per retry:
                    # a stale value from an empty-stream attempt must not
                    # classify the next attempt's end.
                    _finish_reason = None
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
                    self._context_gauge.record_stream_start(len(self.messages))

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
                                        "tool_names": [
                                            tc.get("name") for tc in tool_calls
                                        ],
                                    },
                                )
                        elif event_type == "finish":
                            # Terminal finish_reason (from llm.py's
                            # choices-bearing terminal chunk - the usage-only
                            # chunk has empty choices). Missing = no signal;
                            # the guard stays conservative.
                            _finish_reason = data.get("finish_reason")
                    # Non-empty stream: break out of the retry loop.
                    if content or reasoning or tool_calls is not None:
                        break

                    # Empty stream: stale pooled connection (or a genuinely
                    # empty completion). Re-issue on a fresh connection.
                    if _empty_retries < _MAX_EMPTY_STREAM_RETRIES and self._running:
                        _empty_retries += 1
                        if is_debug_enabled():
                            log_tps_event(
                                "agent_empty_stream_retry",
                                {
                                    "attempt": _empty_retries,
                                    "note": "stream ended with zero bytes - "
                                    "retrying on a fresh connection",
                                },
                            )
                        await self.emit(
                            AgentEvent.NOTIFICATION,
                            {
                                "message": (
                                    "Empty response from the model - retrying "
                                    "on a fresh connection"
                                ),
                                "level": "info",
                            },
                        )
                        await self._close_llm_connections()
                        # First retry is immediate (a fresh connection is
                        # the fix); delay the second so we don't hammer a
                        # struggling backend.
                        if _empty_retries > 1:
                            await asyncio.sleep(_EMPTY_STREAM_RETRY_BACKOFF)
                        continue

                    # Retries exhausted: surface the empty response and
                    # leave the turn recoverable via /resume (the history
                    # may be dangling on a tool result).
                    if is_debug_enabled():
                        log_tps_event(
                            "agent_empty_stream_exhausted",
                            {
                                "retries": _empty_retries,
                                "note": "stream kept coming back empty - "
                                "turn paused, use /resume to retry",
                            },
                        )
                    retries_note = (
                        f" after {_empty_retries} retries" if _empty_retries else ""
                    )
                    await self.emit(
                        AgentEvent.NOTIFICATION,
                        {
                            "message": (
                                "The model returned an empty response"
                                f"{retries_note} - turn paused, "
                                "use /resume to retry"
                            ),
                            "level": "warning",
                        },
                    )
                    self.mark_incomplete_turn(True)
                    break

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
                    # A tool-calls round is a clean completion: it breaks the
                    # consecutive-degenerate-ends streak (plan decision 3).
                    self._degenerate_streak = 0
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
                    # seq: 1-based number of the call within the current user
                    # turn (continues across the turn's LLM iterations),
                    # carried in the TOOL_CALL and TOOL_RESULT payloads so the
                    # UIs can pair a result with its call when a batch's
                    # results render after all its calls.
                    for tc in tool_calls:
                        self._turn_tool_seq += 1
                        seq = self._turn_tool_seq
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
                                "seq": seq,
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
                                "seq": seq,
                            },
                        )

                        # Add tool result to messages (handles vision routing)
                        tool_msg, extra_msgs = await self._build_tool_result_content(
                            result, name, tc["id"]
                        )
                        self.messages.append(tool_msg)
                        for msg in extra_msgs:
                            self.messages.append(msg)

                    # Safe pause point - check if pause requested.
                    # Mid-turn resume: the loop continues to the next LLM
                    # call below, so report WAITING rather than IDLE.
                    await self._wait_if_paused(resume_status=AgentStatus.WAITING)

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

                    # Auto-context threshold check at safe point.
                    # Use the estimated CURRENT context (prompt_tokens is stale
                    # here - it predates the tool results just added). If over
                    # the threshold, run the configured action (compact, or
                    # report-and-compact) and either chain-restart or go idle;
                    # action=none pauses so the user can intervene.
                    estimated_context = self._estimate_current_context_tokens()
                    # Publish the live context estimate so the UI can show it
                    # (with a "~") until the next LLM call reports the grounded
                    # count via TOKEN_USAGE.
                    await self.emit(
                        AgentEvent.CONTEXT_ESTIMATE,
                        {"estimated_tokens": estimated_context},
                    )
                    if self._should_check_threshold(estimated_context):
                        action = await self._handle_threshold_check(estimated_context)
                        if action == "break":
                            break
                        if action == "pause_wait":
                            # Mid-turn resume: the loop continues to the next
                            # LLM call, so report WAITING rather than IDLE.
                            await self._wait_if_paused(
                                resume_status=AgentStatus.WAITING
                            )
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
                    # Same streak reset as the structured tool-calls branch:
                    # this is also a tool round (the Qwen workaround path).
                    self._degenerate_streak = 0
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

                    # Execute each detected tool call (i stays for the
                    # tc_reasoning_{i} ids; seq is the per-turn number)
                    for i, tc in enumerate(reasoning_tool_calls):
                        self._turn_tool_seq += 1
                        seq = self._turn_tool_seq
                        name = tc["name"]
                        arguments = tc["arguments"]

                        # Emit tool call event
                        log_tool_call(name, arguments)
                        await self.emit(
                            AgentEvent.TOOL_CALL,
                            {
                                "name": name,
                                "arguments": arguments,
                                "seq": seq,
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
                                "seq": seq,
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

                # Degeneration guard: classify the no-tool-calls stream end
                # (cap-hit or whitespace wall). "none" falls through to the
                # normal append below; "nudge"/"stop" hand the appends over
                # to _apply_degenerate_outcome (keep-or-drop + injected
                # nudge, or snapshot + incomplete marker) and its return
                # value drives the loop.
                decision = self._classify_degenerate_end(
                    content, reasoning, _finish_reason
                )
                if decision != "none":
                    keep_going = await self._apply_degenerate_outcome(
                        decision, content, reasoning, _finish_reason
                    )
                    if keep_going:
                        # Nudged: loop continues - the next LLM call rides
                        # the warm KV prefix (partial + nudge in history).
                        continue
                    # Stop: turn given up (the apply set the incomplete-turn
                    # marker; /resume retries it from the history tail).
                    self._strip_images_from_messages()
                    break

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

        A real model swap also invalidates the grounded prompt_tokens:
        token counts are tokenizer-specific, so the old model's grounded
        count is stale for the new tokenizer (and the stale-grounding
        guard cannot detect it - the message count is unchanged).

        Args:
            model: The model name to use
        """
        if model == self.model:
            return
        self.model = model
        self.tool_stats.reset()  # Reset stats on model change
        self._invalidate_context_grounding()
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

    def reload_agents_md(self) -> None:
        """Re-read AGENTS.md from disk, refreshing the system-prompt cache.

        AGENTS.md content is appended to the system prompt on every LLM call;
        this refreshes the cached copy so edits to AGENTS.md take effect
        without restarting. Called by the TUI /prompt command.
        """
        reload_agents_md()

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
    def active_skills(self) -> dict[str, str]:
        """Skills manually loaded this session (name -> content, load order)."""
        return dict(self._active_skills)

    def note_skill_loaded(self, name: str, content: str) -> None:
        """Track a manually loaded skill for chain-restart re-injection.

        Called by the TUI's /skillname handler with the formatted skill
        content (including any user-appended args). Dedups by name: a
        re-load replaces the stored content; first-seen order is kept.
        Empty name or content is ignored.
        """
        if not name or not content:
            return
        self._active_skills[name] = content

    def clear_active_skills(self) -> None:
        """Forget all tracked skills (the /chain no-skills path)."""
        self._active_skills.clear()

    def _format_active_skills_section(self) -> str:
        """The chain-restart skills section: a header plus the loaded skill
        blocks (each already a <skill_content> block), in load order.
        "" when no skills are tracked."""
        if not self._active_skills:
            return ""
        lines = [
            "Skills loaded this session (re-injected on chain restart; "
            "drop them from future restarts with /chain no-skills):"
        ]
        lines.extend(self._active_skills.values())
        return "\n\n".join(lines)

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
            # /pause during a lock wait must pause, not interrupt: acquire()
            # backs off (PolitePauseRequested) so the turn/queue state
            # survives and /resume re-enters the wait.
            pause_check=lambda: self._pause_state != PauseState.RUNNING,
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
        # Only the verbose DEFAULT falls through here; a non-empty
        # compact_prompt (user prompt, bridge prompt) is used as-is. The
        # durable-record pointer block is appended to the summary of a
        # DEFAULT-prompt compact only - a user-specified prompt wins clean,
        # and the bridge prompt already carries the file pointers. Compare
        # by VALUE, not identity: the REPL/TUI resolve "compaction" from
        # prompts.yaml (a separate string object, content-pinned to the
        # constant by test_default_prompts_pin.py).
        default_prompt = compact_prompt == DEFAULT_COMPACT_PROMPT

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

            if default_prompt:
                summary = self.durable_records.append_pointer(summary)

            tokens_after = len(summary.split())

            # Replace entire history with the lightweight pair
            self.messages = [
                {"role": "user", "content": COMPACT_REPLACEMENT_MESSAGE},
                {"role": "assistant", "content": summary},
            ]

            # Full history replacement = one chain (session-lifetime total).
            self._chain_count += 1

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

            # Re-ground the estimator state AND publish a fresh context
            # estimate so the TUI's Ctx counter drops to the post-compact
            # size immediately. prompt_tokens still reflects the pre-compact
            # call that produced the summary (its REQUEST was the full old
            # history), and _msg_count_before_stream points past the
            # rewritten history - left as-is, the pre-turn auto-context
            # check would compare the stale pre-compact size against the
            # threshold and fire a spurious wrap-up on the tiny post-compact
            # history (observed 2026-09-28: /compact to ~14k, next pre-turn
            # check read 259k and wrapped up). The estimate stands in until
            # the next real LLM call re-grounds with actual usage.
            post_compact_estimate = await self._estimate_post_compact_context()
            await self.emit(
                AgentEvent.CONTEXT_ESTIMATE,
                {"estimated_tokens": post_compact_estimate},
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
