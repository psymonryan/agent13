"""Journal manager — context compaction via LLM reflection on tool use."""

import json
from typing import Callable, Awaitable

from agent13.events import AgentEvent
from agent13.message_history import MessageHistory
from agent13.prompts import REFLECTION_PROMPT
from agent13.debug_log import (
    log_error,
    log_journal_reflection,
    log_journal_debug,
)
from agent13.llm import LLMError, categorize_error


# Type aliases for the callback functions
SetStatusFn = Callable[..., Awaitable[None]]
"""async (status) -> None — sets agent status and emits STATUS_CHANGED."""

GetStatusFn = Callable[[], object]
"""() -> AgentStatus — returns current agent status."""

EmitFn = Callable[[AgentEvent, dict], Awaitable[None]]
"""async (event, data) -> None — emits an agent event."""

StreamFn = Callable[..., object]
"""async generator (messages, **kwargs) -> (event_type, data) — streams LLM."""


def _count_message_words(messages: list[dict]) -> int:
    """Count words in messages, including tool-call arguments.

    Standard messages have 'content' (str or None). Assistant messages with
    tool calls store their payload in 'tool_calls[].function.arguments',
    which is a JSON string or dict — those words are real context tokens
    that get compacted away, so they must be counted.
    """
    total = 0
    for m in messages:
        content = m.get("content")
        if content:
            total += len(content.split())
        # Count tool-call arguments (the payload that compaction removes)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                if args:
                    total += len(args.split())
    return total


class JournalManager:
    """Context compaction via LLM reflection on tool use.

    Replaces tool-heavy turns with concise summaries to keep
    the context window manageable. Operates on a MessageHistory
    instance and communicates with the agent via callbacks.
    """

    def __init__(
        self,
        history: MessageHistory,
        stream_fn: StreamFn,
        emit_fn: EmitFn,
        set_status_fn: SetStatusFn,
        get_status_fn: GetStatusFn,
        get_prompt_tokens_fn: Callable[[], int],
        is_interrupted_fn: Callable[[], bool],
        journal_mode_fn: Callable[[], bool],
        status_journaling: object,
        status_idle: object,
    ):
        self.history = history
        self._stream = stream_fn
        self._emit = emit_fn
        self._set_status = set_status_fn
        self._get_status = get_status_fn
        self._get_prompt_tokens = get_prompt_tokens_fn
        self._is_interrupted = is_interrupted_fn
        self._journal_mode = journal_mode_fn
        self._JOURNALING = status_journaling
        self._IDLE = status_idle

    # ------------------------------------------------------------------
    # Public API — called from Agent._process_item
    # ------------------------------------------------------------------

    async def retrospective_compact(self, is_interrupt: bool = False) -> None:
        """Compact the previous turn retroactively.

        Called when journal_mode was off during the previous turn but is
        now on — applies compaction before the new turn starts.

        Args:
            is_interrupt: Whether this message was an interrupt (skip if so).

        Skips silently if any condition is not met.
        """
        if not self._journal_mode():
            return
        if not self.history.messages:
            return
        if not self.history.has_tool_calls_in_last_turn():
            return
        if is_interrupt or self._is_interrupted():
            return
        if self.history.has_skill_call_in_last_turn():
            return

        tool_summary = await self.reflect_on_tool_use()
        if tool_summary:
            final_message = self.history.get_final_assistant_message() or ""
            tokens_before = _count_message_words(self.history.messages[-4:])
            tokens_after = len(tool_summary.split()) + len(
                final_message.split()
            )
            self.history.compact(tool_summary, final_message)
            await self._emit(
                AgentEvent.JOURNAL_COMPACT,
                {
                    "summary": tool_summary,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "retrospective": True,
                },
            )

    async def maybe_reflect_after_turn(self) -> None:
        """Run reflection after turn completes and apply compaction.

        Conditions for reflection:
        - journal_mode is True
        - There are messages to compact
        - Last turn had tool calls
        - No interrupt is in progress
        """
        if not self._journal_mode():
            return
        if not self.history.messages:
            return
        if not self.history.has_tool_calls_in_last_turn():
            return
        if self._is_interrupted():
            return

        # Find skill call ranges in the last turn (if any)
        last_user_idx = self.history.find_last_user_idx()
        skill_ranges = None
        if last_user_idx is not None:
            end_idx = len(self.history.messages) - 1
            ranges = self.history.find_skill_call_ranges(
                last_user_idx, end_idx
            )
            if ranges:
                skill_ranges = ranges

        # Reflect, compact, and emit
        await self._journal_one_turn(skill_ranges=skill_ranges)

    async def journal_last_turn(self) -> tuple[bool, str]:
        """Journal the most recent tool-using turn immediately.

        Returns:
            Tuple of (success: bool, message: str) describing the outcome.
        """
        if not self.history.messages:
            return False, "No messages in context"

        if not self.history.has_tool_calls_in_last_turn():
            return False, "No tool calls in the most recent turn"

        # Find skill call ranges in the last turn (if any)
        last_user_idx = self.history.find_last_user_idx()
        skill_ranges = None
        if last_user_idx is not None:
            end_idx = len(self.history.messages) - 1
            ranges = self.history.find_skill_call_ranges(
                last_user_idx, end_idx
            )
            if ranges:
                skill_ranges = ranges

        # Reflect, compact, and emit
        success, _, tokens_before, tokens_after = (
            await self._journal_one_turn(
                retrospective=True, skill_ranges=skill_ranges
            )
        )
        if not success:
            return False, "Reflection produced no summary"

        return True, f"Compacted {tokens_before}\u2192{tokens_after} words"

    async def journal_all(self) -> tuple[bool, str]:
        """Iteratively journal all tool-using turns from earliest to latest.

        Returns:
            Tuple of (success: bool, message: str) describing the outcome.
        """
        log_journal_debug(
            "journal_all",
            {
                "step": "start",
                "messages_count": len(self.history.messages),
                "journal_mode": self._journal_mode(),
                "first_3_roles": (
                    [m.get("role") for m in self.history.messages[:3]]
                    if self.history.messages
                    else []
                ),
            },
        )
        if not self.history.messages:
            log_journal_debug(
                "journal_all",
                {
                    "step": "early_return",
                    "reason": "no_messages",
                    "messages_count": 0,
                },
            )
            return False, "No messages in context"

        if not self.history.has_tool_calls():
            log_journal_debug(
                "journal_all",
                {
                    "step": "early_return",
                    "reason": "no_tool_calls",
                    "messages_count": len(self.history.messages),
                },
            )
            return False, "No tool-using turns to journal"

        total_turns = self.history.count_tool_turns()
        if total_turns == 0:
            log_journal_debug(
                "journal_all",
                {
                    "step": "early_return",
                    "reason": "zero_tool_turns",
                    "messages_count": len(self.history.messages),
                },
            )
            return False, "No tool-using turns to journal"

        total_tokens_before = 0
        total_tokens_after = 0
        iteration = 0

        while True:
            # Find the earliest tool-using turn
            boundary = self.history.find_earliest_tool_turn()
            if boundary is None:
                break

            user_idx, end_idx = boundary

            # Find skill call ranges within this turn
            skill_ranges = self.history.find_skill_call_ranges(
                user_idx, end_idx
            )

            if skill_ranges:
                log_journal_debug(
                    "journal_all_skill_ranges",
                    {
                        "user_idx": user_idx,
                        "end_idx": end_idx,
                        "skill_ranges": skill_ranges,
                    },
                )

            # Save messages after this turn
            tail = self.history.messages[end_idx + 1 :]

            # Temporarily truncate to just the turn + preceding context
            self.history.messages = self.history.messages[: end_idx + 1]

            # Messages in the turn being compacted (for token counting)
            turn_msgs = self.history.messages[user_idx:]

            # Reflect, compact, and emit
            success, _, tokens_before, tokens_after = (
                await self._journal_one_turn(
                    token_count_messages=turn_msgs,
                    retrospective=True,
                    mode="all",
                    iteration=iteration + 1,
                    total_turns=total_turns,
                    skill_ranges=skill_ranges or None,
                )
            )

            # Restore the tail
            self.history.messages.extend(tail)

            if not success:
                log_error(
                    RuntimeError("Reflection returned None"),
                    {
                        "context": "journal_all_iteration",
                        "iteration": iteration + 1,
                        "total_turns": total_turns,
                        "messages_count": len(self.history.messages),
                    },
                )
                if iteration == 0:
                    return False, "Reflection produced no summary"
                break

            iteration += 1
            total_tokens_before += tokens_before
            total_tokens_after += tokens_after

        if iteration == 0:
            return False, "No tool-using turns to journal"

        savings = total_tokens_before - total_tokens_after
        return True, (
            f"Journalled {iteration} turn(s): "
            f"{total_tokens_before}\u2192{total_tokens_after} "
            f"words (saved {savings})"
        )

    # ------------------------------------------------------------------
    # Internal — called by public methods above
    # ------------------------------------------------------------------

    async def reflect_on_tool_use(
        self,
        skill_names: list[str] | None = None,
        messages: list[dict] | None = None,
    ) -> str | None:
        """Ask the LLM to summarize its tool use for context compaction.

        Args:
            skill_names: Names of skills loaded this turn.
            messages: Messages to reflect on. Defaults to history.messages.copy().

        Returns:
            The tool use summary text, or None if reflection fails.
        """
        # Build reflection prompt — add skill names if present
        reflection_prompt = REFLECTION_PROMPT
        if skill_names:
            skill_note = (
                f"[Skills loaded this turn: {', '.join(skill_names)}]"
            )
            reflection_prompt = f"{skill_note}\n\n{reflection_prompt}"

        # Build temporary messages for reflection API call
        temp_messages = (messages or self.history.messages).copy()
        temp_messages.append(
            {"role": "user", "content": reflection_prompt}
        )

        try:
            # Set JOURNALING status so TUI shows correct spinner
            await self._set_status(self._JOURNALING)

            content_parts = []
            async for event_type, data in self._stream(
                temp_messages,
                source="reflection",
                tool_choice="auto",
            ):
                if event_type in ("tool_call", "tool_calls_complete"):
                    continue
                if data:
                    await self._emit(
                        AgentEvent.ASSISTANT_REASONING,
                        {
                            "text": data,
                            "source": "reflection",
                        },
                    )
                    if event_type == "content":
                        content_parts.append(data)

            # Emit final token to signal stream end
            await self._emit(AgentEvent.ASSISTANT_COMPLETE, {})

            content = "".join(content_parts)
            if not content or not content.strip():
                # Restore status from JOURNALING
                if self._get_status() == self._JOURNALING:
                    await self._set_status(self._IDLE)
                return None

            log_journal_reflection(
                "", content.strip(), len(self.history.messages)
            )

            return content.strip()

        except Exception as e:
            log_error(e, {"context": "journal_reflection"})
            llm_error = (
                categorize_error(e)
                if not isinstance(e, LLMError)
                else e
            )
            await self._emit(
                AgentEvent.ERROR,
                {
                    "message": str(llm_error),
                    "error_type": llm_error.error_type,
                    "exception": e,
                },
            )
            # Restore status from JOURNALING
            if self._get_status() == self._JOURNALING:
                await self._set_status(self._IDLE)
            return None

    async def _journal_one_turn(
        self,
        token_count_messages: list | None = None,
        skill_ranges: list[tuple[int, int]] | None = None,
        **event_extras,
    ) -> tuple[bool, str | None, int, int]:
        """Reflect on tool use, compact the turn, and emit a journal event.

        Returns:
            Tuple of (success, summary_or_None, tokens_before, tokens_after).
        """
        # Extract skill messages and build reflection input without them
        preserved_skills: list[dict] | None = None
        skill_names: list[str] | None = None
        reflect_messages: list[dict] | None = None

        if skill_ranges:
            preserved_skills = []
            skill_names = []
            for sr_start, sr_end in skill_ranges:
                for idx in range(sr_start, sr_end + 1):
                    preserved_skills.append(self.history.messages[idx])
                for idx in range(sr_start, sr_end + 1):
                    msg = self.history.messages[idx]
                    if msg.get("role") == "assistant" and msg.get(
                        "tool_calls"
                    ):
                        for tc in msg["tool_calls"]:
                            fn_name = (
                                tc.get("function", {}).get("name", "")
                            )
                            if fn_name == "skill":
                                args = tc.get("function", {}).get(
                                    "arguments", {}
                                )
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except (
                                        json.JSONDecodeError,
                                        ValueError,
                                    ):
                                        args = {}
                                skill_name = args.get("name", "")
                                if (
                                    skill_name
                                    and skill_name not in skill_names
                                ):
                                    skill_names.append(skill_name)

            # Build reflection messages with skill ranges removed
            reflect_messages = []
            skip_indices = set()
            for sr_start, sr_end in skill_ranges:
                for idx in range(sr_start, sr_end + 1):
                    skip_indices.add(idx)
            for idx, msg in enumerate(self.history.messages):
                if idx not in skip_indices:
                    reflect_messages.append(msg)

        tool_summary = await self.reflect_on_tool_use(
            skill_names=skill_names,
            messages=reflect_messages,
        )
        if not tool_summary:
            return False, None, 0, 0

        final_message = (
            self.history.get_final_assistant_message() or ""
        )

        if token_count_messages is None:
            token_count_messages = self.history.messages[-4:]
        tokens_before = _count_message_words(token_count_messages)
        tokens_after = len(tool_summary.split()) + len(
            final_message.split()
        )

        self.history.compact(
            tool_summary, final_message, preserved_skills=preserved_skills
        )

        await self._emit(
            AgentEvent.JOURNAL_COMPACT,
            {
                "summary": tool_summary,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                **event_extras,
            },
        )

        # Emit TOKEN_USAGE so the TUI can update its Ctx counter
        await self._emit(
            AgentEvent.TOKEN_USAGE,
            {
                "prompt_tokens": self._get_prompt_tokens(),
                "completion_tokens": 0,
                "total_tokens": self._get_prompt_tokens(),
                "source": "journal_compact",
            },
        )

        return True, tool_summary, tokens_before, tokens_after