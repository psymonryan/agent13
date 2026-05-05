"""Agent core class - event-driven agent implementation."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from agent13.events import AgentEvent, AgentEventData, EventHandler
from agent13.prompts import DEFAULT_PROMPT, REFLECTION_PROMPT
from agent13.queue import AgentQueue, QueueItem
from agent13.llm import (
    stream_response_with_tools,
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
    log_journal_reflection,
    log_journal_debug,
    # TPS debug logging
    is_debug_enabled,
    log_tps_event,
)

if TYPE_CHECKING:
    from agent13.mcp import MCPManager


# Configuration constants
REASONING_TOOL_CALL_NOTIFICATION_DURATION = 30.0  # seconds


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
        self.messages = messages or []
        self.tools = tools or []
        self.execute_tool = execute_tool
        self.response_format = response_format
        self.journal_mode = journal_mode
        self.send_reasoning = send_reasoning
        self.remove_reasoning = remove_reasoning
        self._devel_mode = devel_mode
        self._skills_mode = skills_mode
        self.execute_tool = execute_tool
        self.response_format = response_format
        self.journal_mode = journal_mode
        self.send_reasoning = send_reasoning
        self.remove_reasoning = remove_reasoning

        self._handlers: list[EventHandler] = []
        self._running = False
        self._stop_event = asyncio.Event()
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
            except Exception as e:
                # Don't let handler errors crash the agent
                print(
                    f"Error in event handler: {e}, event_data: {event_data}, result: {result}"
                )

    def _strip_reasoning_from_messages(self) -> None:
        """Remove reasoning_content from all assistant messages.

        Called between turns when remove_reasoning is enabled (non-journal mode)
        to reduce context usage. When remove_reasoning is off (default),
        reasoning is preserved for better multi-step continuity.
        """
        for msg in self.messages:
            if msg.get("role") == "assistant" and "reasoning_content" in msg:
                del msg["reasoning_content"]

    def _has_tool_calls(self) -> bool:
        """Check if any message in the history contains tool calls.

        Returns:
            True if any assistant message has tool_calls or any message
            has role 'tool'.
        """
        assistant_tc = sum(
            1 for m in self.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        tool_msgs = sum(
            1 for m in self.messages if m.get("role") == "tool"
        )
        result = assistant_tc > 0 or tool_msgs > 0
        log_journal_debug("has_tool_calls", {
            "messages_count": len(self.messages),
            "assistant_with_tool_calls": assistant_tc,
            "tool_messages": tool_msgs,
            "result": result,
        })
        return result

    def _find_last_user_idx(self, start: int | None = None) -> int | None:
        """Return index of the last non-interrupt user message.

        Walks backward from ``start`` (default: end of messages).
        Returns None if no non-interrupt user message is found.
        """
        begin = start if start is not None else len(self.messages) - 1
        for i in range(begin, -1, -1):
            if self.messages[i].get("role") == "user" and not self.messages[i].get(
                "interrupt"
            ):
                return i
        return None

    def _find_earliest_tool_turn(self) -> tuple[int, int] | None:
        """Find the boundary of the earliest tool-using turn.

        A tool-using turn consists of:
        - A non-interrupt user message (turn start)
        - One or more assistant messages with tool_calls + tool results
        - A final assistant message (turn conclusion)

        Returns:
            Tuple of (user_idx, end_idx) where:
            - user_idx: index of the non-interrupt user message starting the turn
            - end_idx: index of the final assistant message concluding the turn,
              or the last message if the turn lacks a concluding assistant message
            Returns None if no tool-using turn is found.
        """
        if not self.messages:
            log_journal_debug("find_earliest_tool_turn", {
                "messages_count": 0,
                "result": None,
                "reason": "no_messages",
            })
            return None

        # Step 1: Find the first assistant message with tool_calls
        first_tool_idx = None
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                first_tool_idx = i
                break

        if first_tool_idx is None:
            log_journal_debug("find_earliest_tool_turn", {
                "messages_count": len(self.messages),
                "result": None,
                "reason": "no_tool_calls_found",
            })
            return None

        # Step 2: Find the non-interrupt user message that starts this turn
        user_idx = self._find_last_user_idx(start=first_tool_idx - 1)
        if user_idx is None:
            # No user message before tool calls — unusual but handle it
            # Use the start of messages as the boundary
            user_idx = 0

        # Step 3: Walk forward from the tool_calls to find the end of the turn.
        # The turn ends when we reach a non-interrupt user message or a final
        # assistant message without tool_calls that isn't followed by more tools.
        # We need to handle multi-round tool use within a single turn:
        #   assistant(tool_calls) → tool → assistant(tool_calls) → tool → assistant(text)
        end_idx = None
        i = first_tool_idx
        while i < len(self.messages):
            msg = self.messages[i]

            if msg.get("role") == "user" and not msg.get("interrupt"):
                # We've hit the next turn — back up one
                end_idx = i - 1
                break

            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                # Final assistant message in this turn
                end_idx = i
                # Keep going to check if there's more tool use after this
                # (shouldn't happen without a user message, but be safe)
                i += 1
                continue

            i += 1

        if end_idx is None:
            # Turn doesn't have a clean end — use end of messages as boundary
            # This handles the case where the last turn has tool calls but no
            # concluding assistant text (e.g. after --continue or interrupted runs)
            end_idx = len(self.messages) - 1

        log_journal_debug("find_earliest_tool_turn", {
            "messages_count": len(self.messages),
            "result": (user_idx, end_idx),
            "first_tool_idx": first_tool_idx,
        })
        return (user_idx, end_idx)

    def _count_tool_turns(self) -> int:
        """Count the number of tool-using turns in the message history.

        A tool-using turn is a group (non-interrupt user msg through to next
        non-interrupt user msg or end) that contains at least one assistant
        message with tool_calls.

        Returns:
            Number of tool-using turn groups.
        """
        if not self.messages:
            log_journal_debug("count_tool_turns", {
                "messages_count": 0,
                "result": 0,
            })
            return 0

        count = 0
        in_tool_turn = False
        for msg in self.messages:
            if msg.get("role") == "user" and not msg.get("interrupt"):
                # Start of a new group
                in_tool_turn = False
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                if not in_tool_turn:
                    count += 1
                    in_tool_turn = True

        log_journal_debug("count_tool_turns", {
            "messages_count": len(self.messages),
            "result": count,
        })
        return count

    def _has_tool_calls_in_last_turn(self) -> bool:
        """Check if the last turn contained any tool calls.

        Looks from the last non-interrupt user message forward, so that
        tool calls before a mid-turn interrupt are still detected.

        Returns:
            True if any assistant message after the last non-interrupt
            user message has tool_calls.
        """
        if not self.messages:
            return False

        # Find the last non-interrupt user message
        last_user_idx = self._find_last_user_idx()
        if last_user_idx is None:
            return False

        # Check for tool_calls in any assistant message after the last non-interrupt user message
        for msg in self.messages[last_user_idx + 1 :]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return True

        return False

    def _has_skill_call_in_last_turn(self) -> bool:
        """Check if the last turn contained a 'skill' tool call.

        Skill tool calls load instructions that must remain in context —
        journalling would destroy them by replacing the tool result with
        a summary. This method detects such calls so journalling can be
        skipped.

        Returns:
            True if any assistant message after the last non-interrupt
            user message has a tool_call with function name 'skill'.
        """
        if not self.messages:
            return False

        last_user_idx = self._find_last_user_idx()
        if last_user_idx is None:
            return False

        for msg in self.messages[last_user_idx + 1 :]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") == "skill":
                        return True

        return False

    def _has_skill_call_in_range(self, start: int, end: int) -> bool:
        """Check if a range of messages contains a 'skill' tool call.

        Used by journal_all to skip individual turns that contain skill
        calls while still journalling other turns.

        Args:
            start: Start index (inclusive).
            end: End index (inclusive).

        Returns:
            True if any assistant message in [start, end] has a
            tool_call with function name 'skill'.
        """
        for msg in self.messages[start : end + 1]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") == "skill":
                        return True

        return False

    def _find_skill_call_ranges(self, start: int, end: int) -> list[tuple[int, int]]:
        """Find sub-ranges within [start, end] that contain skill calls.

        Each skill call range includes the assistant message with the skill
        tool_call and the corresponding tool result messages that follow it.
        These ranges must be preserved verbatim during journalling.

        Args:
            start: Start index (inclusive).
            end: End index (inclusive).

        Returns:
            List of (skill_start, skill_end) tuples, sorted by index.
        """
        skill_ranges = []
        i = start
        while i <= end:
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                has_skill = any(
                    tc.get("function", {}).get("name") == "skill"
                    for tc in msg["tool_calls"]
                )
                if has_skill:
                    skill_start = i
                    # Include this assistant message and all following
                    # tool result messages
                    skill_end = i
                    j = i + 1
                    while j <= end and self.messages[j].get("role") == "tool":
                        skill_end = j
                        j += 1
                    skill_ranges.append((skill_start, skill_end))
                    i = skill_end + 1
                    continue
            i += 1
        return skill_ranges

    def _repair_interrupted_messages(self) -> None:
        """Repair message history after an interrupt (task cancellation).

        When the agent is interrupted mid-turn via Escape (task.cancel()),
        the message history can be left in an inconsistent state:
        - Last message is 'user' (streaming was interrupted before the
          assistant response was appended) → next user message breaks
          role alternation.
        - Last message is 'assistant' with tool_calls but missing tool
          results → API requires every tool_call to have a matching result.

        This method looks at the last message and closes the turn:
        - user → append [Interrupted] assistant message
        - assistant with tool_calls → append missing tool results with
          [Interrupted] error, then append [Interrupted] assistant message
        - tool (results sent but LLM hasn't responded) → append
          [Interrupted] assistant message
        - assistant without tool_calls → already complete, do nothing
        """
        if not self.messages:
            return

        last = self.messages[-1]

        if last["role"] == "user":
            # Streaming was interrupted before assistant response was appended.
            # Close the turn so the next user message alternates correctly.
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "[Interrupted]",
                    "interrupt": True,
                }
            )

        elif last["role"] == "assistant" and last.get("tool_calls"):
            # Tool calls were issued but not all results came back.
            # Append missing tool results so every tool_call has a match.
            tool_call_ids = {tc["id"] for tc in last["tool_calls"]}
            result_ids = set()
            for msg in self.messages[self.messages.index(last) + 1 :]:
                if msg.get("role") == "tool":
                    result_ids.add(msg.get("tool_call_id"))
            for tc_id in tool_call_ids - result_ids:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "[Interrupted]",
                    }
                )
            # Close the turn
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "[Interrupted]",
                    "interrupt": True,
                }
            )

        elif last["role"] == "tool":
            # Tool results sent but LLM hasn't responded yet.
            # Close the turn.
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "[Interrupted]",
                    "interrupt": True,
                }
            )

        # assistant without tool_calls → already complete, nothing to do

    def _has_incomplete_turn(self) -> bool:
        """Check if the conversation has an incomplete turn.

        A turn is incomplete if:
        - Last message is assistant with tool_calls (tools not yet executed)
        - Last message is tool (results not yet processed by LLM)

        Returns:
            True if the turn is incomplete and needs to be resumed.
        """
        if not self.messages:
            return False

        last_msg = self.messages[-1]

        # Case 1: Assistant with pending tool calls
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            return True

        # Case 2: Tool result waiting for LLM to process
        if last_msg.get("role") == "tool":
            return True

        return False

    def _get_pending_tool_calls(self) -> list[dict] | None:
        """Get pending tool calls that need to be executed.

        Returns the tool_calls from the last assistant message if:
        - Last message is assistant with tool_calls
        - Not all tools have been executed (fewer tool results than tool_calls)

        Returns:
            List of pending tool call dicts, or None if no pending tools.
        """
        if not self.messages:
            return None

        last_msg = self.messages[-1]

        # Must be assistant with tool_calls
        if last_msg.get("role") != "assistant":
            return None

        tool_calls = last_msg.get("tool_calls")
        if not tool_calls:
            return None

        # Count how many tool results we have after this assistant message
        # (they would be right after it if any)
        tool_results = 0
        for i in range(len(self.messages) - 2, -1, -1):  # Start from second-to-last
            if self.messages[i].get("role") == "tool":
                tool_results += 1
            else:
                break  # Stop at first non-tool message

        # If we have fewer results than tool_calls, return the pending ones
        if tool_results < len(tool_calls):
            return tool_calls[tool_results:]  # Return unexecuted tools
        return None

    def _get_final_assistant_message(self) -> str | None:
        """Get the content of the final assistant message in the last turn.

        Returns:
            The content of the last assistant message, or None if not found.
        """
        if not self.messages:
            return None

        # The last message should be the final assistant response
        last_msg = self.messages[-1]
        if last_msg.get("role") == "assistant":
            return last_msg.get("content", "")
        return None

    def _get_message_groups(self) -> list[list[int]]:
        """Group messages for atomic deletion.

        Each group starts with a non-interrupt user message and includes all
        subsequent messages (interrupt user messages, tool calls, tool results,
        assistant responses) until the next non-interrupt user message.

        Interrupt user messages (marked with "interrupt": True) are kept in
        the same group as the turn they interrupted, so they are deleted
        together when retrying or compacting.

        Returns:
            List of groups, where each group is a list of message indices.
        """
        groups = []
        current_group = []

        for i, msg in enumerate(self.messages):
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

    def _compact_previous_turn(
        self,
        tool_summary: str,
        final_message: str = "",
        preserved_skills: list[dict] | None = None,
    ) -> None:
        """Compact the previous turn by replacing tool exploration with a summary.

        Finds the last non-interrupt user message and replaces everything after
        it with:
        - Preserved skill messages (if any) — verbatim, at the start
        - The tool summary (summarizing tool calls and results)
        - The original final assistant message (preserving the conclusion)

        Interrupt user messages are skipped so that the entire turn (including
        any mid-turn injected interrupts and their responses) is compacted as
        one unit.

        Args:
            tool_summary: Summary of tool exploration.
            final_message: The original final assistant response to preserve.
            preserved_skills: Skill call/result messages to preserve verbatim.
                Inserted at the start of the compacted turn (before the
                summary), since skill content was loaded before the work
                that the summary describes.
        """
        if not self.messages:
            return

        # Find the index of the last non-interrupt user message
        last_user_idx = self._find_last_user_idx()
        if last_user_idx is None:
            # No non-interrupt user message found, nothing to compact
            return

        # Combine tool summary with final message
        combined_content = (
            f"{tool_summary}\n\n{final_message}" if final_message else tool_summary
        )

        # Keep messages up to and including the last non-interrupt user message
        self.messages = self.messages[: last_user_idx + 1]

        # Insert preserved skill messages before the summary
        if preserved_skills:
            self.messages.extend(preserved_skills)

        # Then append the combined summary
        self.messages.append({"role": "assistant", "content": combined_content})

    async def _reflect_on_tool_use(
        self,
        skill_names: list[str] | None = None,
        messages: list[dict] | None = None,
    ) -> str | None:
        """Ask the LLM to summarize its tool use for context compaction.

        Makes a separate API call with a reflection prompt focused on tool calls.
        The response summarizes what tools were used and what was found.

        Sends tools with tool_choice="none" to preserve LCP cache matching
        with the main loop. Without this, omitting tools from the request
        causes the serialization order to diverge after the system prompt,
        resulting in massive KV cache misses (sim_best drops from ~0.997
        to ~0.591, adding 20+ minutes of reprocessing).

        Emits streaming events so the TUI can display "Reflecting:" feedback.

        Args:
            skill_names: Names of skills loaded this turn. If provided,
                a brief note is prepended to the reflection prompt so the
                LLM can reference skills in its summary without seeing the
                full skill content.
            messages: Messages to reflect on. Defaults to self.messages.copy().

        Returns:
            The tool use summary text, or None if reflection fails.
        """
        from agent13.llm import stream_response_with_tools

        # Build reflection prompt — add skill names if present
        reflection_prompt = REFLECTION_PROMPT
        if skill_names:
            skill_note = f"[Skills loaded this turn: {', '.join(skill_names)}]"
            reflection_prompt = f"{skill_note}\n\n{reflection_prompt}"

        # Build temporary messages for reflection API call
        temp_messages = (messages or self.messages).copy()
        temp_messages.append({"role": "user", "content": reflection_prompt})

        try:
            # Set JOURNALING status so TUI shows correct spinner
            await self._set_status(AgentStatus.JOURNALING)

            # Emit STREAM_START with source="reflection" so TUI shows "Reflecting" title
            await self.emit(AgentEvent.STREAM_START, {"source": "reflection"})

            # Stream the response, emitting tokens as reasoning
            # Include tools with tool_choice="none" to preserve LCP cache
            # matching with the main loop (which sends tools with "auto").
            # Without tools, the serialization order diverges after the system
            # prompt, causing massive cache misses (sim_best=0.591 vs 0.997).
            # Note: I rolled back to tool_choice="auto" rather than "none" as I suspect this also causes issues
            content_parts = []
            tools = await self.get_all_tools()
            async for event_type, data in stream_response_with_tools(
                self.client,
                self.model,
                temp_messages,
                self.system_prompt,
                tools,
                tool_choice="auto",
            ):
                # Ignore tool_call events (shouldn't happen with
                # tool_choice="none", but handle gracefully if they do)
                if event_type in ("tool_call", "tool_calls_complete", "token_usage"):
                    continue
                if data:
                    # Emit all tokens as reasoning (reflection is essentially thinking)
                    await self.emit(
                        AgentEvent.ASSISTANT_REASONING,
                        {
                            "text": data,
                            "source": "reflection",
                        },
                    )
                    if event_type == "content":
                        content_parts.append(data)

            # Emit final token to signal stream end
            await self.emit(AgentEvent.ASSISTANT_COMPLETE, {})

            content = "".join(content_parts)
            if not content or not content.strip():
                # Restore status from JOURNALING (caller may not expect it)
                if self._status == AgentStatus.JOURNALING:
                    await self._set_status(AgentStatus.IDLE)
                return None

            # Log the reflection for debugging
            log_journal_reflection("", content.strip(), len(self.messages))

            return content.strip()

        except Exception as e:
            log_error(e, {"context": "journal_reflection"})
            # Emit the error to the UI so user can see what went wrong
            llm_error = categorize_error(e) if not isinstance(e, LLMError) else e
            await self.emit(
                AgentEvent.ERROR,
                {
                    "message": str(llm_error),
                    "error_type": llm_error.error_type,
                    "exception": e,
                },
            )
            # Restore status from JOURNALING
            if self._status == AgentStatus.JOURNALING:
                await self._set_status(AgentStatus.IDLE)
            return None

    async def _journal_one_turn(
        self,
        token_count_messages: list | None = None,
        skill_ranges: list[tuple[int, int]] | None = None,
        **event_extras,
    ) -> tuple[bool, str | None, int, int]:
        """Reflect on tool use, compact the turn, and emit a journal event.

        This is the shared core of all journal paths: auto-journal,
        journal_last_turn, and journal_all. It performs the 5-step sequence:
        reflect → get final message → count tokens → compact → emit.

        When skill_ranges are provided, skill messages are extracted before
        reflection (so the LLM doesn't waste tokens on content that will
        be preserved verbatim), a skill-names note is added to the
        reflection prompt, and the full skill messages are reinserted at
        the start of the compacted turn.

        Args:
            token_count_messages: Messages to count tokens from.
                Defaults to self.messages[-4:] (approximate last turn).
            skill_ranges: Ranges of (skill_start, skill_end) indices
                within self.messages that contain skill calls and results.
                These are extracted before reflection and reinserted
                verbatim at the start of the compacted turn.
            **event_extras: Additional fields merged into the JOURNAL_COMPACT event
                (e.g. retrospective=True, mode="all", iteration, total_turns).

        Returns:
            Tuple of (success, summary_or_None, tokens_before, tokens_after).
        """
        # Extract skill messages and build reflection input without them
        preserved_skills: list[dict] | None = None
        skill_names: list[str] | None = None
        reflect_messages: list[dict] | None = None

        if skill_ranges:
            # Collect skill message dicts for later reinsertion
            preserved_skills = []
            skill_names = []
            for sr_start, sr_end in skill_ranges:
                for idx in range(sr_start, sr_end + 1):
                    preserved_skills.append(self.messages[idx])
                # Extract skill name from the assistant message's tool_calls
                for idx in range(sr_start, sr_end + 1):
                    msg = self.messages[idx]
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            fn_name = tc.get("function", {}).get("name", "")
                            if fn_name == "skill":
                                args = tc.get("function", {}).get("arguments", {})
                                # arguments may be a JSON string or dict
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except (json.JSONDecodeError, ValueError):
                                        args = {}
                                skill_name = args.get("name", "")
                                if skill_name and skill_name not in skill_names:
                                    skill_names.append(skill_name)

            # Build reflection messages with skill ranges removed
            reflect_messages = []
            skip_indices = set()
            for sr_start, sr_end in skill_ranges:
                for idx in range(sr_start, sr_end + 1):
                    skip_indices.add(idx)
            for idx, msg in enumerate(self.messages):
                if idx not in skip_indices:
                    reflect_messages.append(msg)

        tool_summary = await self._reflect_on_tool_use(
            skill_names=skill_names,
            messages=reflect_messages,
        )
        if not tool_summary:
            return False, None, 0, 0

        final_message = self._get_final_assistant_message() or ""

        if token_count_messages is None:
            token_count_messages = self.messages[-4:]
        tokens_before = sum(
            len(m.get("content", "").split()) for m in token_count_messages
        )
        tokens_after = len(tool_summary.split()) + len(final_message.split())

        self._compact_previous_turn(
            tool_summary, final_message, preserved_skills=preserved_skills
        )

        await self.emit(
            AgentEvent.JOURNAL_COMPACT,
            {
                "summary": tool_summary,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                **event_extras,
            },
        )

        # Emit TOKEN_USAGE so the TUI can update its Ctx counter
        # after compaction (wishlist #54). Use the LLM-reported
        # prompt_tokens as the source of truth — this is the context
        # size prior to compaction, not a word count estimate.
        await self.emit(
            AgentEvent.TOKEN_USAGE,
            {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": self.prompt_tokens,
                "source": "journal_compact",
            },
        )

        return True, tool_summary, tokens_before, tokens_after

    async def _maybe_reflect_after_turn(self) -> None:
        """Run reflection after turn completes and apply compaction immediately.

        This is Stage 12 of journal mode: proactive reflection trigger.
        Reflection, compaction, and event emission all happen immediately
        after the agent finishes its turn, via _journal_one_turn(), so the
        message history is always compacted before the user sees it.

        Skill calls in the turn are handled gracefully: skill messages are
        extracted before reflection (so the LLM doesn't process content
        that will be preserved verbatim), a skill-names note is added to
        the reflection prompt, and the full skill messages are reinserted
        at the start of the compacted turn.

        Conditions for reflection:
        - journal_mode is True
        - There are messages to compact
        - Last turn had tool calls
        - No interrupt is in progress
        """
        if not self.journal_mode:
            return
        if not self.messages:
            return
        if not self._has_tool_calls_in_last_turn():
            return
        if self.queue.has_interrupt:
            return

        # Find skill call ranges in the last turn (if any)
        last_user_idx = self._find_last_user_idx()
        skill_ranges = None
        if last_user_idx is not None:
            end_idx = len(self.messages) - 1
            ranges = self._find_skill_call_ranges(last_user_idx, end_idx)
            if ranges:
                skill_ranges = ranges

        # Reflect, compact, and emit
        await self._journal_one_turn(skill_ranges=skill_ranges)

    async def journal_last_turn(self) -> tuple[bool, str]:
        """Journal the most recent tool-using turn immediately.

        This performs retrospective compaction on the last turn, similar to
        what happens when journal_mode is enabled and a new message arrives.

        Skill calls in the turn are handled gracefully: skill messages are
        extracted before reflection and reinserted verbatim at the start
        of the compacted turn.

        Called from _process_item when a journal_last queue item is processed.
        Status management is handled by _process_item.

        Returns:
            Tuple of (success: bool, message: str) describing the outcome.
        """
        if not self.messages:
            return False, "No messages in context"

        if not self._has_tool_calls_in_last_turn():
            return False, "No tool calls in the most recent turn"

        # Find skill call ranges in the last turn (if any)
        last_user_idx = self._find_last_user_idx()
        skill_ranges = None
        if last_user_idx is not None:
            end_idx = len(self.messages) - 1
            ranges = self._find_skill_call_ranges(last_user_idx, end_idx)
            if ranges:
                skill_ranges = ranges

        # Reflect, compact, and emit
        success, _, tokens_before, tokens_after = await self._journal_one_turn(
            retrospective=True, skill_ranges=skill_ranges
        )
        if not success:
            return False, "Reflection produced no summary"

        return True, f"Compacted {tokens_before}\u2192{tokens_after} words"

    async def journal_all(self) -> tuple[bool, str]:
        """Iteratively journal all tool-using turns from earliest to latest.

        Unlike journal_last_turn which compacts only the most recent turn,
        this method finds every tool-using turn in the history and applies
        the existing per-turn reflection and compaction, one at a time,
        starting from the earliest.

        Called from _process_item when a journal_all queue item is processed.
        Status management is handled by _process_item.

        Each iteration:
        1. Finds the earliest tool-using turn boundary
        2. Temporarily truncates messages to end at that turn
        3. Runs _journal_one_turn() to reflect, compact, and emit
        4. Restores messages after the compacted turn
        5. Repeats until no tool-using turns remain

        Skill calls within a turn are handled by _journal_one_turn():
        skill messages are extracted before reflection and reinserted
        verbatim at the start of the compacted turn.

        This preserves user messages and non-tool assistant messages verbatim
        — only the tool-calling machinery gets compacted.

        Returns:
            Tuple of (success: bool, message: str) describing the outcome.
        """
        log_journal_debug("journal_all", {
            "step": "start",
            "messages_count": len(self.messages),
            "journal_mode": self.journal_mode,
            "first_3_roles": [m.get("role") for m in self.messages[:3]] if self.messages else [],
        })
        if not self.messages:
            log_journal_debug("journal_all", {
                "step": "early_return",
                "reason": "no_messages",
                "messages_count": 0,
            })
            return False, "No messages in context"

        if not self._has_tool_calls():
            log_journal_debug("journal_all", {
                "step": "early_return",
                "reason": "no_tool_calls",
                "messages_count": len(self.messages),
            })
            return False, "No tool-using turns to journal"

        total_turns = self._count_tool_turns()
        if total_turns == 0:
            log_journal_debug("journal_all", {
                "step": "early_return",
                "reason": "zero_tool_turns",
                "messages_count": len(self.messages),
            })
            return False, "No tool-using turns to journal"

        total_tokens_before = 0
        total_tokens_after = 0
        iteration = 0

        while True:
            # Find the earliest tool-using turn
            boundary = self._find_earliest_tool_turn()
            if boundary is None:
                break

            user_idx, end_idx = boundary

            # Find skill call ranges within this turn
            skill_ranges = self._find_skill_call_ranges(user_idx, end_idx)

            if skill_ranges:
                log_journal_debug("journal_all_skill_ranges", {
                    "user_idx": user_idx,
                    "end_idx": end_idx,
                    "skill_ranges": skill_ranges,
                })

            # Save messages after this turn (they'll be restored after compaction)
            tail = self.messages[end_idx + 1 :]

            # Temporarily truncate to just the turn + preceding context
            self.messages = self.messages[: end_idx + 1]

            # Messages in the turn being compacted (for token counting)
            turn_msgs = self.messages[user_idx:]

            # Reflect, compact, and emit
            # skill_ranges is passed so _journal_one_turn can extract
            # skill messages before reflection and reinsert them at the
            # start of the compacted turn.
            success, _, tokens_before, tokens_after = await self._journal_one_turn(
                token_count_messages=turn_msgs,
                retrospective=True,
                mode="all",
                iteration=iteration + 1,
                total_turns=total_turns,
                skill_ranges=skill_ranges or None,
            )

            # Restore the tail
            self.messages.extend(tail)

            if not success:
                # Reflection failed — stop iterating
                log_error(
                    RuntimeError("Reflection returned None"),
                    {
                        "context": "journal_all_iteration",
                        "iteration": iteration + 1,
                        "total_turns": total_turns,
                        "messages_count": len(self.messages),
                    },
                )
                if iteration == 0:
                    return False, "Reflection produced no summary"
                # Partial success — stop iterating but report what we did
                break

            iteration += 1
            total_tokens_before += tokens_before
            total_tokens_after += tokens_after

        if iteration == 0:
            return False, "No tool-using turns to journal"

        savings = total_tokens_before - total_tokens_after
        return True, (
            f"Journalled {iteration} turn(s): "
            f"{total_tokens_before}→{total_tokens_after} words (saved {savings})"
        )

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
                else:
                    # Nothing to do, wait briefly
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # User interrupted - repair message history and emit event
            current_id = self.queue.current.id if self.queue.current else None
            log_queue_interrupt(current_id)
            self._repair_interrupted_messages()
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
        pending_tools = self._get_pending_tool_calls()

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

        elif self._has_incomplete_turn():
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
                success, message = await self.journal_last_turn()
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
                success, message = await self.journal_all()
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
                    "has_tool_calls": self._has_tool_calls(),
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
            elif item.kind == "retry":
                # Deferred /retry — safe at this boundary between items
                groups = self._get_message_groups()
                user_text = ""
                if groups:
                    last_group = groups[-1]
                    first_msg_idx = last_group[0]
                    user_text = self.messages[first_msg_idx].get("content", "")
                    for idx in sorted(last_group, reverse=True):
                        del self.messages[idx]
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()
                await self.emit(
                    AgentEvent.RETRY_STARTED,
                    {
                        "user_text": user_text,
                    },
                )
            else:
                # Normal prompt processing
                # Retrospective compaction: journal was off during the previous
                # turn, but user has now turned it on. Apply compaction now so
                # the tool calls get summarized before the new turn starts.
                if (
                    self.journal_mode
                    and self.messages
                    and self._has_tool_calls_in_last_turn()
                    and not item.interrupt
                    and not self._has_skill_call_in_last_turn()
                ):
                    tool_summary = await self._reflect_on_tool_use()
                    if tool_summary:
                        final_message = self._get_final_assistant_message() or ""
                        # Count tokens before compaction
                        tokens_before = sum(
                            len(m.get("content", "").split())
                            for m in self.messages[-4:]  # Approximate last turn
                        )
                        tokens_after = len(tool_summary.split()) + len(
                            final_message.split()
                        )
                        self._compact_previous_turn(tool_summary, final_message)
                        await self.emit(
                            AgentEvent.JOURNAL_COMPACT,
                            {
                                "summary": tool_summary,
                                "tokens_before": tokens_before,
                                "tokens_after": tokens_after,
                                "retrospective": True,
                            },
                        )
                # Strip reasoning from previous turns when remove_reasoning is enabled.
                # In journal mode, the turn gets replaced with a summary anyway.
                # When remove_reasoning is off (default), reasoning is preserved between
                # turns for better multi-step continuity and user visibility.
                if not self.journal_mode and self.remove_reasoning:
                    self._strip_reasoning_from_messages()

                # Add user message to history
                self.messages.append({"role": "user", "content": item.text})

                # Process with LLM (may include multiple tool call rounds)
                await self._llm_turn()

                # Stage 12: Run reflection after turn completes (stores pending compaction)
                await self._maybe_reflect_after_turn()

                # Mark item as complete
                self.queue.complete_current()
                log_queue_complete(item.id, "complete")
                await self._emit_queue_update()

        except Exception as e:
            # Categorize the error for better user feedback
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

            # Pause after error so /resume works (matches _llm_turn behavior)
            # This handles errors that originate outside _llm_turn (e.g.
            # retrospective journal path). _llm_turn already sets PAUSED
            # for its own errors, so we check to avoid redundant transitions.
            if self._pause_state != PauseState.PAUSED:
                self._pause_state = PauseState.PAUSED
                self._pause_event.clear()
                await self._set_status(AgentStatus.PAUSED)
                await self.emit(AgentEvent.PAUSED, {"reason": "error"})
        # Preserve PAUSED status from error (set by either _llm_turn or
        # the except block above)
        if self._pause_state != PauseState.PAUSED:
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
                        {"note": "Starting stream_response_with_tools"},
                    )

                # Emit STREAM_START for TUI to reset per-stream timing
                await self.emit(AgentEvent.STREAM_START, {})

                # Get all tools including MCP tools
                tools = await self.get_all_tools()

                # Track first tokens for status transitions
                _first_reasoning = True
                _first_content = True

                async for event_type, data in stream_response_with_tools(
                    self.client,
                    self.model,
                    self.messages,
                    self.system_prompt,
                    tools,
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
                    elif event_type == "token_usage":
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
                    msg = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
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

                # Pause after error so /resume works
                self._pause_state = PauseState.PAUSED
                self._pause_event.clear()  # Block so _wait_if_paused will actually wait
                await self._set_status(AgentStatus.PAUSED)
                await self.emit(AgentEvent.PAUSED, {"reason": "error"})

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

    def set_client(self, client: AsyncOpenAI) -> None:
        """Set the OpenAI client.

        Args:
            client: The AsyncOpenAI client to use
        """
        self.client = client

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

    async def request_retry(self) -> int:
        """Request a retry of the last message via the queue.

        Adds a kind="retry" item with interrupt=True so it breaks into
        the current turn (if any), then deletes the last message group
        and re-adds the user message at a safe boundary.

        Returns:
            The queue item ID
        """
        item_id = self.queue.add("", interrupt=True, kind="retry")
        await self._emit_queue_update()
        return item_id

    def reset_token_usage(self) -> None:
        """Reset token usage counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
