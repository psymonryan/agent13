"""Message history queries and compaction.

Owns the message list and provides query/mutation methods that were
previously scattered across Agent. All methods are pure functions
over the message list — no LLM calls, no events, no queue interaction.
"""

import json

from agent13.debug_log import log_journal_debug


class MessageHistory:
    """Message list wrapper with query and compaction methods.

    Owns the message list. Agent delegates here instead of maintaining
    private history methods. ``Agent.messages`` becomes a property that
    proxies to ``self.history.messages`` so existing callers are unchanged.
    """

    def __init__(self, messages: list[dict] | None = None):
        self.messages: list[dict] = messages or []

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_tool_calls(self) -> bool:
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

    def find_last_user_idx(self, start: int | None = None) -> int | None:
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

    def find_earliest_tool_turn(self) -> tuple[int, int] | None:
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
        user_idx = self.find_last_user_idx(start=first_tool_idx - 1)
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

    def count_tool_turns(self) -> int:
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

    def has_tool_calls_in_last_turn(self) -> bool:
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
        last_user_idx = self.find_last_user_idx()
        if last_user_idx is None:
            return False

        # Check for tool_calls in any assistant message after the last non-interrupt user message
        for msg in self.messages[last_user_idx + 1 :]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                return True

        return False

    def has_skill_call_in_last_turn(self) -> bool:
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

        last_user_idx = self.find_last_user_idx()
        if last_user_idx is None:
            return False

        for msg in self.messages[last_user_idx + 1 :]:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") == "skill":
                        return True

        return False

    def has_skill_call_in_range(self, start: int, end: int) -> bool:
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

    def find_skill_call_ranges(self, start: int, end: int) -> list[tuple[int, int]]:
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

    def has_incomplete_turn(self) -> bool:
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

    def get_pending_tool_calls(self) -> list[dict] | None:
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

    def get_final_assistant_message(self) -> str | None:
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

    def get_message_groups(self) -> list[list[int]]:
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

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def strip_reasoning(self) -> None:
        """Remove reasoning_content from all assistant messages.

        Called between turns when remove_reasoning is enabled (non-journal mode)
        to reduce context usage. When remove_reasoning is off (default),
        reasoning is preserved for better multi-step continuity.
        """
        for msg in self.messages:
            if msg.get("role") == "assistant" and "reasoning_content" in msg:
                del msg["reasoning_content"]

    def repair_interrupted(self) -> None:
        """Repair message history after an interrupt (task cancellation).

        When the agent is interrupted mid-turn via Escape (task.cancel()),
        the message history can be left in an inconsistent state:
        - Last message is 'user' (streaming was interrupted before the
          assistant response was appended) -> next user message breaks
          role alternation.
        - Last message is 'assistant' with tool_calls but missing tool
          results -> API requires every tool_call to have a matching result.

        This method looks at the last message and closes the turn:
        - user -> append [Interrupted] assistant message
        - assistant with tool_calls -> append missing tool results with
          [Interrupted] error, then append [Interrupted] assistant message
        - tool (results sent but LLM hasn't responded) -> append
          [Interrupted] assistant message
        - assistant without tool_calls -> already complete, do nothing
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

        # assistant without tool_calls -> already complete, nothing to do

    def is_priming_pair_at_tail(self) -> bool:
        """Check if the last two messages are a priming pair.

        A priming pair is a user message with the priming prompt followed by
        an assistant message with the priming response ("ok").

        Returns:
            True if the last two messages match the priming pair pattern.
        """
        from agent13.prompts import PRIMING_PROMPT, PRIMING_RESPONSE

        if len(self.messages) < 2:
            return False
        user_msg = self.messages[-2]
        asst_msg = self.messages[-1]
        return (
            user_msg.get("role") == "user"
            and user_msg.get("content") == PRIMING_PROMPT
            and asst_msg.get("role") == "assistant"
            and asst_msg.get("content") == PRIMING_RESPONSE
        )

    def remove_priming_pair_at_tail(self) -> bool:
        """Remove the priming pair if it's at the tail.

        Returns:
            True if a pair was removed, False otherwise.
        """
        if self.is_priming_pair_at_tail():
            self.messages.pop()
            self.messages.pop()
            return True
        return False

    def remove_all_priming_pairs(self) -> int:
        """Remove all priming pairs from anywhere in history.

        Returns:
            Number of pairs removed.
        """
        from agent13.prompts import PRIMING_PROMPT, PRIMING_RESPONSE

        removed = 0
        i = 0
        while i < len(self.messages) - 1:
            user_msg = self.messages[i]
            asst_msg = self.messages[i + 1]
            if (
                user_msg.get("role") == "user"
                and user_msg.get("content") == PRIMING_PROMPT
                and asst_msg.get("role") == "assistant"
                and asst_msg.get("content") == PRIMING_RESPONSE
            ):
                del self.messages[i : i + 2]
                removed += 1
            else:
                i += 1
        return removed

    def append_priming_pair(self) -> None:
        """Append a fresh priming pair at the tail."""
        from agent13.prompts import PRIMING_PROMPT, PRIMING_RESPONSE

        self.messages.append({"role": "user", "content": PRIMING_PROMPT})
        self.messages.append({"role": "assistant", "content": PRIMING_RESPONSE})

    def compact(
        self,
        tool_summary: str,
        final_message: str = "",
        preserved_skills: list[dict] | None = None,
    ) -> None:
        """Compact the previous turn by replacing tool exploration with a summary.

        Finds the last non-interrupt user message and replaces everything after
        it with:
        - Preserved skill messages (if any) — as text, at the start
        - The tool summary (summarizing tool calls and results)
        - The original final assistant message (preserving the conclusion)

        Interrupt user messages are skipped so that the entire turn (including
        any mid-turn injected interrupts and their responses) is compacted as
        one unit.

        Args:
            tool_summary: Summary of tool exploration.
            final_message: The original final assistant response to preserve.
            preserved_skills: Skill call/result messages to preserve.
                Converted from raw API format (assistant+tool_calls + tool
                result) to text-only assistant messages during insertion.
                This prevents find_earliest_tool_turn from re-finding the
                same turn after compaction.
        """
        if not self.messages:
            return

        # Find the index of the last non-interrupt user message
        last_user_idx = self.find_last_user_idx()
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
        # Convert from raw API format (assistant+tool_calls + tool result)
        # to text-only assistant messages. This prevents find_earliest_tool_turn
        # from re-finding the same turn after compaction (infinite loop).
        if preserved_skills:
            i = 0
            while i < len(preserved_skills):
                msg = preserved_skills[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Extract skill name from tool_calls
                    skill_name = ""
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        if fn.get("name") == "skill":
                            args = fn.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except (json.JSONDecodeError, ValueError):
                                    args = {}
                            skill_name = args.get("name", "")
                            break

                    # Find the matching tool result (next message)
                    skill_content = ""
                    if i + 1 < len(preserved_skills):
                        next_msg = preserved_skills[i + 1]
                        if next_msg.get("role") == "tool":
                            skill_content = next_msg.get("content", "")
                            i += 1  # skip tool result, consumed

                    # Emit as single text-only assistant message
                    label = f"[Skill: {skill_name}]" if skill_name else "[Skill]"
                    self.messages.append({
                        "role": "assistant",
                        "content": f"{label}\n\n{skill_content}" if skill_content else label,
                    })
                else:
                    # Non-skill message (shouldn't happen, but handle gracefully)
                    content = msg.get("content", "")
                    if content:
                        self.messages.append({
                            "role": "assistant",
                            "content": content,
                        })
                i += 1

        # Then append the combined summary
        self.messages.append({"role": "assistant", "content": combined_content})
