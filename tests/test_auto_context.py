"""Tests for the auto-context feature (action = compact | report_and_compact | none).

At the auto-context threshold, when action = "report_and_compact" the agent:
  1. emits a NOTIFICATION notice,
  2. snapshots the full history,
  3. injects the report_and_compact prompt as a user message
     (preceded by a fake assistant message for role alternation),
  4. runs ONE wrap-up turn with auto-context suppressed,
  5. goes idle (no pause).
"""

import asyncio
import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent, AgentStatus
from agent13.events import AgentEvent
from agent13.message_history import LOCAL_MSG_KEYS, is_turn_start
from agent13.prompts import (
    AUTO_CONTEXT_CARRY_HINT,
    AUTO_CONTEXT_CONTINUE_HINT,
    CHAIN_RESTART_INTRO,
    DEFAULT_REPORT_AND_COMPACT_PROMPT,
    DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT,
    resolve_report_and_compact_prompt,
)
from agent13.queue import QueueItem


def make_agent(
    auto_context_threshold=0,
    auto_context_action="compact",
    auto_context_chain=0,
    report_and_compact_prompt=None,
):
    """Create a minimal agent for testing."""
    client = MagicMock()
    agent = Agent(
        client=client,
        model="test-model",
        auto_context_threshold=auto_context_threshold,
        auto_context_action=auto_context_action,
        auto_context_chain=auto_context_chain,
        report_and_compact_prompt=report_and_compact_prompt,
    )
    agent._llm_turn = AsyncMock()
    agent._save_auto_context_snapshot = MagicMock()
    return agent


# ───────────────────────── Constructor / config ─────────────────────────


class TestAutoContextConfig:
    def test_default_action_is_compact(self):
        agent = make_agent()
        assert agent.auto_context_action == "compact"
        assert agent._auto_context_triggered is False
        assert agent._report_and_compact_triggered is False
        assert agent._auto_context_chain_used == 0

    def test_report_and_compact_via_constructor(self):
        agent = make_agent(auto_context_action="report_and_compact")
        assert agent.auto_context_action == "report_and_compact"

    def test_prompt_defaults_to_builtin(self):
        agent = make_agent()
        assert agent.report_and_compact_prompt == DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_prompt_override(self):
        agent = make_agent(report_and_compact_prompt="custom wrap-up prompt")
        assert agent.report_and_compact_prompt == "custom wrap-up prompt"


class TestAutoContextPromptResolution:
    """resolve_report_and_compact_prompt() mirrors resolve_compact_prompt()."""

    def test_none_manager_falls_back_to_default(self):
        assert (
            resolve_report_and_compact_prompt(None) == DEFAULT_REPORT_AND_COMPACT_PROMPT
        )

    def test_missing_entry_falls_back_to_default(self):
        pm = MagicMock()
        pm.prompts = {}
        assert (
            resolve_report_and_compact_prompt(pm) == DEFAULT_REPORT_AND_COMPACT_PROMPT
        )

    def test_yaml_entry_wins(self):
        pm = MagicMock()
        pm.prompts = {"report_and_compact": "my wrap-up wording"}
        assert resolve_report_and_compact_prompt(pm) == "my wrap-up wording"

    def test_default_mentions_journal_append(self):
        """The built-in default must carry the wrap-up behaviours: append a
        dated journal entry, append-only (no doc reads). No progress report —
        it is invisible once the chain clears the display."""
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT.lower()
        assert "journal" in text
        assert "append" in text
        assert "progress report" not in text


# ───────────────────────── Threshold-gate logic ─────────────────────────


class TestAutoContextGate:
    """The safe-point gate (_should_check_threshold) and the threshold dispatch
    (_handle_threshold_check). The tests call the real methods — they do NOT
    copy the gate condition into the test (the old _report_wins helper did)."""

    def test_check_gate_off_when_threshold_zero(self):
        agent = make_agent(auto_context_threshold=0)
        assert agent._should_check_threshold(10**9) is False

    def test_check_gate_off_below_threshold(self):
        agent = make_agent(auto_context_threshold=1000)
        assert agent._should_check_threshold(500) is False

    def test_check_gate_off_when_circuit_broken(self):
        agent = make_agent(auto_context_threshold=1000)
        agent._auto_context_failures = 3
        assert agent._should_check_threshold(5000) is False

    def test_check_gate_off_when_suppressed(self):
        """The wrap-up turn itself must not re-trigger."""
        agent = make_agent(
            auto_context_threshold=1000, auto_context_action="report_and_compact"
        )
        agent._suppress_auto_context = True
        assert agent._should_check_threshold(5000) is False

    def test_check_gate_on_when_over_threshold(self):
        agent = make_agent(auto_context_threshold=1000)
        assert agent._should_check_threshold(5000) is True

    @pytest.mark.asyncio
    async def test_handle_check_none_pauses_without_flags(self):
        agent = make_agent(auto_context_threshold=1000, auto_context_action="none")
        events = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture
        agent.pause = MagicMock()

        action = await agent._handle_threshold_check(5000)

        assert action == "pause_wait"
        agent.pause.assert_called_once()
        assert agent._auto_context_triggered is False
        assert agent._report_and_compact_triggered is False
        notice = next(d for k, d in events if k == AgentEvent.NOTIFICATION)
        assert "paused" in notice["message"]
        assert notice["level"] == "warning"

    @pytest.mark.asyncio
    async def test_handle_check_compact_sets_compact_flag(self):
        agent = make_agent(auto_context_threshold=1000, auto_context_action="compact")
        action = await agent._handle_threshold_check(5000)
        assert action == "break"
        assert agent._auto_context_triggered is True
        assert agent._report_and_compact_triggered is False

    @pytest.mark.asyncio
    async def test_handle_check_report_and_compact_sets_both_flags(self):
        agent = make_agent(
            auto_context_threshold=1000, auto_context_action="report_and_compact"
        )
        action = await agent._handle_threshold_check(5000)
        assert action == "break"
        assert agent._report_and_compact_triggered is True
        assert agent._auto_context_triggered is True


# ───────────────────────── Wrap-up behaviour ─────────────────────────


class TestAutoContextWrapUp:
    @pytest.mark.asyncio
    async def test_notice_is_notification_not_assistant_token(self):
        """Regression: the notice once rode ASSISTANT_TOKEN, whose buffer is
        orphaned when the wrap-up turn's STREAM_START resets the display
        tracker. It must be a NOTIFICATION instead."""
        agent = make_agent(
            auto_context_threshold=1000, auto_context_action="report_and_compact"
        )
        events = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture

        # Real threshold path: _handle_threshold_check emits the notice and
        # asks the caller to break so the post-turn handler wraps up.
        action = await agent._handle_threshold_check(5000)
        assert action == "break"
        assert agent._report_and_compact_triggered is True

        kinds = [e[0] for e in events]
        assert AgentEvent.NOTIFICATION in kinds
        assert AgentEvent.ASSISTANT_TOKEN not in kinds
        notice = next(d for k, d in events if k == AgentEvent.NOTIFICATION)
        assert "wrapping up" in notice["message"]
        assert notice.get("level") == "warning"

    @pytest.mark.asyncio
    async def test_wrap_up_injects_prompt_and_runs_one_turn(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        await agent._run_report_and_compact_turn()

        roles = [m["role"] for m in agent.messages]
        # Fake assistant precedes the prompt user message (role alternation).
        assert roles == ["assistant", "user"]
        # The prompt plus the injected date/time line (for the dated journal
        # heading).
        content = agent.messages[-1]["content"]
        assert content.startswith("WRAP-UP-PROMPT")
        assert "Current date and time:" in content
        # Both are marked so persistence/journal can recognise them.
        assert all(m.get("report_and_compact") for m in agent.messages)
        # Exactly one wrap-up turn was run.
        assert agent._llm_turn.await_count == 1
        # Snapshot taken, suppression released afterwards.
        agent._save_auto_context_snapshot.assert_called_once()
        assert agent._suppress_auto_context is False
        # The wrap-up message text now reflects that work continues after.
        assert agent.messages[0]["content"] == "[Wrapping up]"

    @pytest.mark.asyncio
    async def test_wrap_up_marker_not_a_turn_start(self):
        """F1/F2: the injected wrap-up user message carries a local marker so it
        is (a) stripped before the API wire and (b) never mistaken for a new
        user turn. Without the marker in LOCAL_MSG_KEYS the flag leaks to the
        provider; without the is_turn_start clause the wrap-up splits a turn
        in half during grouping/compaction."""
        assert "report_and_compact" in LOCAL_MSG_KEYS
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        await agent._run_report_and_compact_turn()

        wrap_user = next(
            m
            for m in agent.messages
            if m["role"] == "user" and m.get("report_and_compact")
        )
        # The wrap-up prompt is agent-injected mid-turn, like `!!`.
        assert is_turn_start(wrap_user) is False

    @pytest.mark.asyncio
    async def test_suppression_released_even_if_turn_raises(self):
        agent = make_agent(
            auto_context_threshold=1000, auto_context_action="report_and_compact"
        )
        agent.messages = []
        agent._report_and_compact_triggered = True
        agent._llm_turn = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError):
            await agent._run_report_and_compact_turn()

        assert agent._suppress_auto_context is False

    @pytest.mark.asyncio
    async def test_no_wrap_up_when_not_triggered(self):
        agent = make_agent(
            auto_context_threshold=1000, auto_context_action="report_and_compact"
        )
        agent.messages = []

        await agent._run_report_and_compact_turn()

        assert agent.messages == []
        agent._llm_turn.assert_not_awaited()

    def test_per_turn_reset_clears_flags_and_counter(self):
        """The per-turn reset clears the chain budget and any flags a failed
        wrap-up or compact might have left set (hygiene)."""
        agent = make_agent(
            auto_context_action="report_and_compact", auto_context_chain=2
        )
        # Simulate leftover state from an interrupted turn.
        agent._auto_context_chain_used = 2
        agent._auto_context_triggered = True
        agent._report_and_compact_triggered = True
        agent._reset_auto_context_turn_state()
        assert agent._auto_context_chain_used == 0
        assert agent._auto_context_triggered is False
        assert agent._report_and_compact_triggered is False


# ───────────────────────────────────────────────── Chain budget ────────────────────────────────────────────────


class TestSessionJournal:
    """The session journal: date injection, path capture from the wrap-up's own
    append tool call, and the nudge carrying the path across chain restarts."""

    @staticmethod
    def _append_tool_call(path: str) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        # Real tool calls use the tools' "filepath" parameter
                        # (a "path" key here once masked a capture bug).
                        "arguments": json.dumps(
                            {"filepath": path, "content": "## 2026-09-18 01:45\n..."}
                        ),
                    },
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_date_injected_into_wrapup_message(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        await agent._run_report_and_compact_turn()

        content = agent.messages[-1]["content"]
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        assert f"Current date and time: {today}" in content

    @pytest.mark.asyncio
    async def test_journal_path_captured_from_append_tool_call(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        async def fake_llm_turn(*args, **kwargs):
            agent.messages.append(
                self._append_tool_call("docs_archive/my_focus_journal.md")
            )

        agent._llm_turn = fake_llm_turn

        await agent._run_report_and_compact_turn()

        assert agent._session_journal_path == "docs_archive/my_focus_journal.md"

    @pytest.mark.asyncio
    async def test_journal_path_not_captured_for_other_paths(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        async def fake_llm_turn(*args, **kwargs):
            agent.messages.append(self._append_tool_call("src/foo.py"))

        agent._llm_turn = fake_llm_turn

        await agent._run_report_and_compact_turn()

        assert agent._session_journal_path == ""

    @pytest.mark.asyncio
    async def test_nudge_is_bare_continue_hint(self):
        """The nudge is the bare continue hint — no file cues. The durable
        records (mechanical path) or the compact summary's embedded pointer
        block (compact-only path) carry the file orientation."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=1,
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._session_truth_path = "docs_archive/my_focus_current_truth.md"
        agent._session_journal_path = "docs_archive/my_focus_journal.md"
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        nudges = [
            m
            for m in agent.messages
            if str(m.get("content", "")).startswith("[Context was compacted")
        ]
        assert len(nudges) == 1
        # Bare hint only — no file paths, no read-first cues.
        assert nudges[0]["content"] == AUTO_CONTEXT_CONTINUE_HINT
        # "injected" keeps the nudge out of turn-start counting so the TUI's
        # chain-restart rebuild keeps the summary pair, not the nudge.
        assert nudges[0].get("injected") is True

    @pytest.mark.asyncio
    async def test_truth_path_captured_from_wrapup_write(self):
        """The wrap-up's current-truth rewrite is captured alongside the
        journal append, using the tools' real "filepath" argument key."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            report_and_compact_prompt="WRAP-UP-PROMPT",
        )
        agent.messages = []
        agent._report_and_compact_triggered = True

        async def fake_llm_turn(*args, **kwargs):
            agent.messages.append(
                self._append_tool_call("docs_archive/my_focus_journal.md")
            )
            agent.messages.append(
                self._append_tool_call("docs_archive/my_focus_current_truth.md")
            )

        agent._llm_turn = fake_llm_turn

        await agent._run_report_and_compact_turn()

        assert agent._session_truth_path == "docs_archive/my_focus_current_truth.md"
        assert agent._session_journal_path == "docs_archive/my_focus_journal.md"

    @pytest.mark.asyncio
    async def test_nudge_plain_without_journal_path(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="compact",
            auto_context_chain=1,
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        nudges = [
            m
            for m in agent.messages
            if str(m.get("content", "")).startswith("[Context was compacted")
        ]
        assert len(nudges) == 1
        assert nudges[0]["content"] == AUTO_CONTEXT_CONTINUE_HINT
        assert nudges[0].get("injected") is True


class TestAutoCompactSummaryPrompt:
    """The auto-path compact prompt branches on the action: a completed
    report_and_compact wrap-up just wrote the journal + truth files, so the
    summary only bridges to them (short prompt); action=compact has no
    wrap-up and keeps the verbose default (compact_history("") fallback)."""

    @pytest.mark.asyncio
    async def test_report_and_compact_is_mechanical_no_ai_call(self):
        """report_and_compact is now a mechanical restart: no compact_history
        AI call. The pre-baked intro replaces the bridge summary."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        agent.compact_history.assert_not_awaited()
        assert len(agent.messages) > 0
        assert "Session restarted from durable records" in agent.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_compact_action_keeps_verbose_prompt(self):
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="compact",
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        # "" routes to the verbose DEFAULT_COMPACT_PROMPT inside
        # compact_history - pre-existing behavior, unchanged.
        agent.compact_history.assert_awaited_once_with("")


class TestManualCompactSummaryPrompt:
    """The manual /compact prompt follows the wrap-up state, DERIVED from the
    message history (no flag): a wrap-up that ran and no tool calls (major
    work) happened since keeps the durable record current, so a plain /compact
    (no explicit prompt) uses the short bridge summary - matching the auto
    path. An explicit prompt (--prompt / focus text) always wins, and a
    /compact with no current wrap-up keeps the verbose default."""

    @staticmethod
    def _wrapup_history(after=None):
        """A completed wrap-up (marker + prompt + file write + final response),
        optionally followed by `after` messages."""
        hist = [
            {
                "role": "assistant",
                "content": "[Wrapping up]",
                "report_and_compact": True,
            },
            {
                "role": "user",
                "content": "wrap-up prompt",
                "report_and_compact": True,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": "{}",
                        }
                    }
                ],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Wrap-up complete."},
        ]
        return hist + (after or [])

    def test_no_wrapup_not_current(self):
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        assert agent._wrapup_is_current() is False

    def test_wrapup_current_after_completion(self):
        """A completed wrap-up (no tool calls since) is current."""
        agent = make_agent()
        agent.messages = self._wrapup_history()
        assert agent._wrapup_is_current() is True

    def test_wrapup_not_current_after_tool_call(self):
        """A tool call after the wrap-up invalidates it (major work)."""
        agent = make_agent()
        agent.messages = self._wrapup_history(
            [
                {"role": "user", "content": "now edit the file"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": "{}",
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "content": "Done."},
            ]
        )
        assert agent._wrapup_is_current() is False

    def test_wrapup_still_current_after_plain_qa(self):
        """A simple Q&A (no tool calls) after the wrap-up keeps it current."""
        agent = make_agent()
        agent.messages = self._wrapup_history(
            [
                {"role": "user", "content": "quick question?"},
                {"role": "assistant", "content": "answer, no tools"},
            ]
        )
        assert agent._wrapup_is_current() is True

    @pytest.mark.asyncio
    async def test_manual_compact_uses_bridge_after_wrapup(self):
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent.messages = self._wrapup_history()
        item = QueueItem(
            id=1,
            text="/compact",
            kind="compact",
            data={"compact_prompt": "VERBOSE", "user_specified_prompt": False},
        )

        await agent._process_item(item)

        agent.compact_history.assert_awaited_once_with(
            DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT
        )

    @pytest.mark.asyncio
    async def test_manual_compact_uses_verbose_without_wrapup(self):
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent.messages = [{"role": "user", "content": "hi"}]
        item = QueueItem(
            id=1,
            text="/compact",
            kind="compact",
            data={"compact_prompt": "VERBOSE", "user_specified_prompt": False},
        )

        await agent._process_item(item)

        # No current wrap-up: the item's (verbose) prompt is used as-is.
        agent.compact_history.assert_awaited_once_with("VERBOSE")

    @pytest.mark.asyncio
    async def test_manual_compact_respects_explicit_prompt(self):
        """An explicit /compact prompt wins over the bridge, even with a
        current wrap-up."""
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent.messages = self._wrapup_history()
        item = QueueItem(
            id=1,
            text="/compact focus on X",
            kind="compact",
            data={"compact_prompt": "STEERED", "user_specified_prompt": True},
        )

        await agent._process_item(item)

        agent.compact_history.assert_awaited_once_with("STEERED")


class TestWrapupCommand:
    """Manual /wrapup queues a kind="wrapup" item; core runs the wrap-up
    turn (journal + truth files) and NEVER compacts or arms the chain."""

    def _make_tui(self, **kwargs):
        from ui.tui import AgentTUI

        defaults = dict(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            prompt_manager=None,
        )
        defaults.update(kwargs)
        return AgentTUI(**defaults)

    @pytest.mark.asyncio
    async def test_wrapup_queue_item_runs_wrapup_without_compact(self):
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock()
        agent.queue.complete_current = MagicMock()
        events = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture
        agent.messages = []
        item = QueueItem(id=1, text="/wrapup", kind="wrapup")

        await agent._process_item(item)

        assert agent._llm_turn.await_count == 1
        injected = [m for m in agent.messages if m.get("report_and_compact")]
        assert len(injected) == 2  # fake assistant ack + prompt message
        agent.compact_history.assert_not_awaited()
        assert agent._auto_context_triggered is False
        assert agent._report_and_compact_triggered is False
        notes = [d for e, d in events if e == AgentEvent.NOTIFICATION]
        assert any("Wrap-up complete" in n.get("message", "") for n in notes)

    @pytest.mark.asyncio
    async def test_wrapup_focus_name_injected_into_prompt(self):
        """A /wrapup given a focus name carries it into the injected
        wrap-up prompt as the explicit-name line (overrides the
        conversation heuristic)."""
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock()
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent.messages = []
        item = QueueItem(
            id=1,
            text="/wrapup my cool game",
            kind="wrapup",
            data={"focus_name": "my cool game"},
        )

        await agent._process_item(item)

        # The name became first-class state (authoritative from now on).
        assert agent._focus_name == "my cool game"
        prompt_msgs = [
            m
            for m in agent.messages
            if m.get("report_and_compact") and m["role"] == "user"
        ]
        assert len(prompt_msgs) == 1
        content = prompt_msgs[0]["content"]
        assert (
            "The user has specified the focus name for this wrap-up: "
            '"my cool game". Use it for {focus} in both file names '
            "and headings; it overrides any name from the conversation."
        ) in content
        # The line sits between the prompt and the date line.
        prompt_pos = content.find(agent.report_and_compact_prompt)
        name_pos = content.find("The user has specified the focus name")
        date_pos = content.find("Current date and time:")
        assert prompt_pos < name_pos < date_pos

    def test_wrapup_in_tui_command_list(self):
        tui = self._make_tui()
        assert "/wrapup" in tui.SLASH_COMMANDS

    @pytest.mark.asyncio
    async def test_wrapup_tui_handler_queues_item(self):
        tui = self._make_tui()
        tui.agent.add_message = AsyncMock()
        tui._handle_wrapup_command()
        await asyncio.sleep(0.01)
        tui.agent.add_message.assert_awaited_once_with("/wrapup", kind="wrapup")

    @pytest.mark.asyncio
    async def test_wrapup_tui_handler_queues_focus_name(self):
        """Rest of line (spaces kept, stripped) becomes data['focus_name']."""
        tui = self._make_tui()
        tui.agent.add_message = AsyncMock()
        tui._handle_wrapup_command("  my cool game  ")
        await asyncio.sleep(0.01)
        tui.agent.add_message.assert_awaited_once_with(
            "/wrapup my cool game",
            kind="wrapup",
            data={"focus_name": "my cool game"},
        )


class TestFocusNameState:
    """Explicit focus name: first-class core state, authoritative over the
    wrap-up's tool-call capture. /focus [NAME] sets it (and seeds an empty
    history from the focus's durable records); /wrapup NAME re-sets it."""

    def _make_tui(self, **kwargs):
        from ui.tui import AgentTUI

        defaults = dict(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            prompt_manager=None,
        )
        defaults.update(kwargs)
        return AgentTUI(**defaults)

    @staticmethod
    def _tool_call(path: str) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"filepath": path}),
                    },
                }
            ],
        }

    @staticmethod
    def _write_records(tmp_path, name, truth=None, journal=None):
        (tmp_path / "docs_archive").mkdir(exist_ok=True)
        if truth is not None:
            (tmp_path / "docs_archive" / f"{name}_current_truth.md").write_text(
                truth, encoding="utf-8"
            )
        if journal is not None:
            (tmp_path / "docs_archive" / f"{name}_journal.md").write_text(
                journal, encoding="utf-8"
            )

    # ── set_focus_name ──────────────────────────────────────────────

    def test_set_focus_name_derives_record_paths(self):
        agent = make_agent()
        agent.set_focus_name("my game")
        assert agent._focus_name == "my game"
        assert agent._session_truth_path == "docs_archive/my game_current_truth.md"
        assert agent._session_journal_path == "docs_archive/my game_journal.md"

    def test_set_focus_name_rederives_on_change(self):
        agent = make_agent()
        agent.set_focus_name("alpha")
        agent.set_focus_name("beta")
        assert agent._focus_name == "beta"
        assert agent._session_truth_path == "docs_archive/beta_current_truth.md"
        assert agent._session_journal_path == "docs_archive/beta_journal.md"

    def test_set_focus_name_ignores_blank(self):
        agent = make_agent()
        agent.set_focus_name("")
        agent.set_focus_name("   ")
        assert agent._focus_name == ""
        assert agent._session_truth_path == ""
        assert agent._session_journal_path == ""

    # ── _session_focus_name precedence ──────────────────────────────

    def test_explicit_name_wins_over_path_derived(self):
        agent = make_agent()
        agent._session_truth_path = "docs_archive/captured_current_truth.md"
        agent.set_focus_name("explicit")
        assert agent._session_focus_name == "explicit"

    def test_path_derived_fallback_when_unset(self):
        agent = make_agent()
        agent._session_journal_path = "docs_archive/derived_journal.md"
        assert agent._session_focus_name == "derived"

    # ── capture authority ───────────────────────────────────────────

    def test_capture_skipped_when_focus_name_set(self):
        agent = make_agent()
        agent.set_focus_name("explicit")
        agent._capture_session_journal_path(
            [self._tool_call("docs_archive/other_journal.md")]
        )
        # The derived paths are authoritative - the wrap-up's writes can't
        # move the record.
        assert agent._session_journal_path == "docs_archive/explicit_journal.md"
        assert agent._session_truth_path == "docs_archive/explicit_current_truth.md"

    def test_capture_still_works_when_unset(self):
        agent = make_agent()
        agent._capture_session_journal_path(
            [self._tool_call("docs_archive/captured_journal.md")]
        )
        assert agent._session_journal_path == "docs_archive/captured_journal.md"

    # ── /wrapup NAME -> state ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_wrapup_name_becomes_focus_state(self):
        agent = make_agent(auto_context_action="report_and_compact")
        agent.compact_history = AsyncMock()
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent.messages = []
        item = QueueItem(
            id=1,
            text="/wrapup my cool game",
            kind="wrapup",
            data={"focus_name": "my cool game"},
        )

        await agent._process_item(item)

        assert agent._focus_name == "my cool game"
        assert agent._session_truth_path == "docs_archive/my cool game_current_truth.md"
        assert agent._session_journal_path == "docs_archive/my cool game_journal.md"

    # ── seeding from records ────────────────────────────────────────

    def test_seed_injects_valid_user_assistant_pair(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        self._write_records(
            tmp_path,
            "my game",
            truth="# my game - current truth\nTRUTH CONTENT",
            journal=(
                "## 2026-01-01 10:00 - old - done\nold entry body\n\n"
                "## 2026-02-02 11:00 - new - done\nnew entry body"
            ),
        )
        agent.set_focus_name("my game")

        assert agent.seed_from_records() is True
        assert [m["role"] for m in agent.messages] == ["user", "assistant"]
        user_content = agent.messages[0]["content"]
        assert "TRUTH CONTENT" in user_content
        # Only the LAST journal entry is seeded, not the whole journal.
        assert "new entry body" in user_content
        assert "old entry body" not in user_content
        assert agent.messages[1]["content"] == "[Loaded records for focus my game]"

    def test_seed_noop_on_non_empty_history(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        self._write_records(tmp_path, "my game", truth="TRUTH CONTENT")
        agent.set_focus_name("my game")
        agent.messages = [{"role": "user", "content": "mid session"}]

        assert agent.seed_from_records() is False
        assert len(agent.messages) == 1

    def test_seed_noop_without_record_files(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        agent.set_focus_name("missing")

        assert agent.seed_from_records() is False
        assert agent.messages == []

    def test_seed_noop_without_name(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)

        assert agent.seed_from_records() is False
        assert agent.messages == []

    def test_seed_force_appends_to_non_empty_history(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        self._write_records(
            tmp_path,
            "my game",
            truth="TRUTH CONTENT",
            journal="## 2026-02-02 11:00 - new - done\nnew entry body",
        )
        agent.set_focus_name("my game")
        # Simulate the post-compact bridge summary pair.
        agent.messages = [
            {"role": "user", "content": "Bridge summary"},
            {"role": "assistant", "content": "Understood."},
        ]

        assert agent.seed_from_records(force=True) is True
        assert len(agent.messages) == 4
        # Original bridge pair preserved.
        assert agent.messages[0]["content"] == "Bridge summary"
        assert agent.messages[1]["content"] == "Understood."
        # Records appended as a new user/assistant pair.
        assert agent.messages[2]["role"] == "user"
        assert "TRUTH CONTENT" in agent.messages[2]["content"]
        assert "new entry body" in agent.messages[2]["content"]
        assert agent.messages[3]["role"] == "assistant"
        assert agent.messages[3]["content"] == "[Loaded records for focus my game]"

    def test_seed_force_noop_without_records(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        agent.set_focus_name("missing")
        agent.messages = [
            {"role": "user", "content": "Bridge summary"},
            {"role": "assistant", "content": "Understood."},
        ]

        assert agent.seed_from_records(force=True) is False
        assert len(agent.messages) == 2

    # ── /focus (apply_focus_name - immediate, not queued) ───────────

    @pytest.mark.asyncio
    async def test_focus_mid_session_is_set_only(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        self._write_records(
            tmp_path,
            "my game",
            truth="TRUTH",
            journal="## 2026-01-01 10:00 - x - y\nbody",
        )
        agent.messages = [{"role": "user", "content": "mid session"}]

        message, level = await agent.apply_focus_name("my game")

        assert agent._focus_name == "my game"
        assert len(agent.messages) == 1  # no seeding mid-session
        assert (message, level) == ("Focus name is now set to my game", "info")

    @pytest.mark.asyncio
    async def test_mid_turn_focus_lands_before_wrapup_prompt(
        self, tmp_path, monkeypatch
    ):
        """The motivating scenario: /focus NAME mid-turn, then the
        threshold wrap-up builds its prompt - the prompt must carry the
        new name, so record targeting follows a mid-turn /focus (the old
        queued path only applied the name after the wrap-up had run)."""
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        agent.messages = [{"role": "user", "content": "mid-turn work"}]
        agent.report_and_compact_prompt = "WRAPUP"
        agent._report_and_compact_triggered = True
        agent._save_auto_context_snapshot = MagicMock()

        async def fake_llm_turn(*args, **kwargs):
            pass

        agent._llm_turn = fake_llm_turn

        await agent.apply_focus_name("aname")
        await agent._run_report_and_compact_turn()

        wrapup_prompt = agent.messages[-1]["content"]
        assert '"aname"' in wrapup_prompt

    @pytest.mark.asyncio
    async def test_focus_empty_history_seeds_records(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        self._write_records(
            tmp_path,
            "my game",
            truth="TRUTH",
            journal="## 2026-01-01 10:00 - x - y\nbody",
        )
        agent.messages = []

        message, level = await agent.apply_focus_name("my game")

        assert agent._focus_name == "my game"
        assert [m["role"] for m in agent.messages] == ["user", "assistant"]
        assert "TRUTH" in agent.messages[0]["content"]
        assert (message, level) == (
            "Focus name set to my game - records loaded",
            "info",
        )

    @pytest.mark.asyncio
    async def test_focus_empty_history_no_records(self, tmp_path, monkeypatch):
        agent = make_agent()
        monkeypatch.chdir(tmp_path)
        agent.messages = []

        message, level = await agent.apply_focus_name("my game")

        assert agent._focus_name == "my game"
        assert agent.messages == []
        assert (message, level) == (
            "Focus name set to my game - no records found",
            "warning",
        )

    @pytest.mark.asyncio
    async def test_focus_no_arg_reports_current_name(self):
        agent = make_agent()
        agent.set_focus_name("my game")

        message, level = await agent.apply_focus_name("")

        assert (message, level) == ("Focus name: my game", "info")

    @pytest.mark.asyncio
    async def test_focus_no_arg_unset(self):
        agent = make_agent()

        message, level = await agent.apply_focus_name("")

        assert (message, level) == ("No focus name set", "warning")

    # ── UI wiring ───────────────────────────────────────────────────

    def test_focus_in_tui_command_list(self):
        tui = self._make_tui()
        assert "/focus" in tui.SLASH_COMMANDS

    @pytest.mark.asyncio
    async def test_focus_tui_handler_applies_immediately(self):
        """No arg: reports the current name via apply_focus_name('') -
        no queue item is created; the result goes to the info pane (not
        a toast), warning level = yellow."""
        tui = self._make_tui()
        tui.agent.apply_focus_name = AsyncMock(
            return_value=("No focus name set", "warning")
        )
        tui._update_info_content = MagicMock()
        tui.update_status = MagicMock()
        tui._handle_focus_command()
        await asyncio.sleep(0.01)
        tui.agent.apply_focus_name.assert_awaited_once_with("")
        tui._update_info_content.assert_called_once_with(
            "[yellow]No focus name set[/]"
        )
        tui.update_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_focus_tui_handler_applies_name(self):
        """Rest of line (spaces kept, stripped) goes to apply_focus_name;
        the result goes to the info pane (info level = green)."""
        tui = self._make_tui()
        tui.agent.apply_focus_name = AsyncMock(
            return_value=("Focus name is now set to my game", "info")
        )
        tui._update_info_content = MagicMock()
        tui.update_status = MagicMock()
        tui._handle_focus_command("  my game  ")
        await asyncio.sleep(0.01)
        tui.agent.apply_focus_name.assert_awaited_once_with("my game")
        tui._update_info_content.assert_called_once_with(
            "[green]Focus name is now set to my game[/]"
        )
        tui.update_status.assert_called_once()

    def test_repl_help_lists_focus(self):
        from agent13.repl import COMMANDS

        assert "/focus" in COMMANDS
        assert "/wrapup" in COMMANDS


class TestFocusStartupPreprocess:
    """--focus startup preprocess: on a fresh session the records become
    the session's FIRST TURN (the --read pattern) - the model reads them
    and replies a minimal ack while the user types. The TUI wiring is
    _apply_focus_startup (called from on_mount); the REPL wiring is the
    startup_focus call in run_repl."""

    def _make_tui(self, **kwargs):
        from ui.tui import AgentTUI

        defaults = dict(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            prompt_manager=None,
        )
        defaults.update(kwargs)
        return AgentTUI(**defaults)

    @staticmethod
    def _write_records(tmp_path, name, truth=None, journal=None):
        (tmp_path / "docs_archive").mkdir(exist_ok=True)
        if truth is not None:
            (tmp_path / "docs_archive" / f"{name}_current_truth.md").write_text(
                truth, encoding="utf-8"
            )
        if journal is not None:
            (tmp_path / "docs_archive" / f"{name}_journal.md").write_text(
                journal, encoding="utf-8"
            )

    @pytest.mark.asyncio
    async def test_tui_enqueues_preprocess_turn(self, tmp_path, monkeypatch):
        """Fresh session + records: the records message is enqueued as a
        prompt turn, and a dim system line says what is happening first."""
        monkeypatch.chdir(tmp_path)
        self._write_records(tmp_path, "my game", truth="TRUTH")
        tui = self._make_tui(focus_name="my game")
        tui._write_system = AsyncMock()
        tui.agent.add_message = AsyncMock()
        tui._update_info_content = MagicMock()

        await tui._apply_focus_startup()

        # Dim line BEFORE the enqueue (ordering matters for the user).
        tui._write_system.assert_awaited_once()
        dim_line = tui._write_system.await_args.args[0]
        assert "Preprocessing records for focus my game" in dim_line
        # The records message is enqueued as the first turn.
        tui.agent.add_message.assert_awaited_once()
        enqueued = tui.agent.add_message.await_args
        assert "Durable record for focus my game" in enqueued.args[0]
        assert 'Reply with "Records for focus my game loaded."' in enqueued.args[0]
        assert enqueued.kwargs.get("kind") == "prompt"
        # Nothing to report - the ack is the model's job now.
        tui._update_info_content.assert_not_called()
        assert tui.agent._focus_name == "my game"

    @pytest.mark.asyncio
    async def test_tui_no_records_reports_warning(self, tmp_path, monkeypatch):
        """Fresh session, no records: no enqueue, the warning goes to the
        info pane (the pre-preprocess behavior)."""
        monkeypatch.chdir(tmp_path)
        tui = self._make_tui(focus_name="missing")
        tui._write_system = AsyncMock()
        tui.agent.add_message = AsyncMock()
        tui._update_info_content = MagicMock()

        await tui._apply_focus_startup()

        tui._write_system.assert_not_awaited()
        tui.agent.add_message.assert_not_awaited()
        tui._update_info_content.assert_called_once_with(
            "[dim]No records found for focus 'missing' - name set[/]"
        )

    @pytest.mark.asyncio
    async def test_tui_resumed_history_enqueues_nothing(self, tmp_path, monkeypatch):
        """A resumed (non-empty) history is never double-seeded."""
        monkeypatch.chdir(tmp_path)
        self._write_records(tmp_path, "my game", truth="TRUTH")
        tui = self._make_tui(focus_name="my game")
        tui.agent.messages = [{"role": "user", "content": "resumed"}]
        tui._write_system = AsyncMock()
        tui.agent.add_message = AsyncMock()
        tui._update_info_content = MagicMock()

        await tui._apply_focus_startup()

        tui._write_system.assert_not_awaited()
        tui.agent.add_message.assert_not_awaited()
        tui._update_info_content.assert_not_called()

    # The REPL/pipemode/batch wirings are covered end-to-end by
    # tests/test_focus_preprocess_integration.py (the real REPL spawned
    # with --focus; the first mock request must carry the records) - a
    # unit test here would mirror the wiring instead of exercising it.


class TestAutoContextChain:
    """The post-turn chain loop, driven through the real _run_auto_context_post_turn
    with compact_history and _llm_turn mocked. Each restarted _llm_turn re-trips
    the threshold (re-sets _auto_context_triggered), exactly as a real over-
    threshold continuation turn would."""

    @staticmethod
    def _set_retripping_llm_turn(agent):
        """Replace _llm_turn with an async fn that re-trips the threshold,
        simulating a continuation turn whose context is still over the limit."""

        async def fake_llm_turn(*args, **kwargs):
            agent._auto_context_triggered = True

        agent._llm_turn = fake_llm_turn

    @staticmethod
    def _capture(agent) -> list:
        events: list = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture
        return events

    @staticmethod
    def _continue_hints(agent) -> list:
        return [
            m for m in agent.messages if m.get("content") == AUTO_CONTEXT_CONTINUE_HINT
        ]

    @pytest.mark.asyncio
    async def test_chain_zero_compacts_then_idles(self):
        """Control group: compact, chain=0 → one compact, no nudge, idle."""
        agent = make_agent(auto_context_threshold=1000, auto_context_action="compact")
        self._set_retripping_llm_turn(agent)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        events = self._capture(agent)
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        assert agent.compact_history.await_count == 1
        assert self._continue_hints(agent) == []
        assert sum(1 for k, _ in events if k == AgentEvent.CHAT_CLEAR) == 0
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("idle, over to you" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_chain_two_restarts_then_idles(self):
        """chain=2 with re-trips → three compacts, two nudges, idle after the 3rd."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="compact",
            auto_context_chain=2,
        )
        self._set_retripping_llm_turn(agent)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        events = self._capture(agent)
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        assert agent.compact_history.await_count == 3
        assert len(self._continue_hints(agent)) == 2
        assert sum(1 for k, _ in events if k == AgentEvent.CHAT_CLEAR) == 2
        assert agent._auto_context_chain_used == 2
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("idle, over to you" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_report_and_compact_runs_wrap_up_then_mechanical_restart(self):
        """Regression: the loop must actually RUN the wrap-up turn before the
        pending restart. The wrap-up turn is the only thing that calls _llm_turn
        here; if the loop pre-cleared _report_and_compact_triggered, the
        method's no-op guard would see False and silently skip the wrap-up.
        (Both flags are set exactly as _handle_threshold_check sets them.)"""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=0,
        )
        agent.messages = []
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        events = self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True
        # _llm_turn is the default AsyncMock (no re-trip) -> the wrap-up turn is
        # the only caller of it.

        await agent._run_auto_context_post_turn()

        # The wrap-up turn ran: it made exactly one LLM call.
        assert agent._llm_turn.await_count == 1
        # Mechanical restart: no compact_history call, intro is first message.
        assert agent.compact_history.await_count == 0
        assert "Session restarted from durable records" in agent.messages[0]["content"]
        # CHAT_REBUILD emitted after records injection
        assert sum(1 for k, _ in events if k == AgentEvent.CHAT_REBUILD) == 1
        # We went idle (chain=0).
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("idle, over to you" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_compact_failure_idle_no_nudge(self):
        """A failed compact increments the breaker, emits an error notice, and
        idles — no restart, no nudge."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="compact",
            auto_context_chain=2,
        )
        agent.compact_history = AsyncMock(return_value=(False, "boom"))
        agent.messages = []
        events = self._capture(agent)
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        assert agent.compact_history.await_count == 1
        assert agent._auto_context_failures == 1
        assert self._continue_hints(agent) == []
        error = [
            d
            for k, d in events
            if k == AgentEvent.NOTIFICATION and d.get("level") == "error"
        ]
        assert any("compact failed" in d["message"] for d in error)
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert not any("idle, over to you" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_chain_exhausted_persistent_footnote(self):
        """The chain budget is spent: the CHAT_REBUILD footer is the
        exhaustion notice — a persistent record in the wiped window of why
        the turn stopped (the toast alone is transient and was invisible
        by morning, observed 2026-10-07 edmac overnight run)."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=2,
        )
        agent.messages = []
        events = self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True
        agent._auto_context_chain_used = 2  # budget spent
        agent._chain_count = 7  # lifetime total for the footer

        await agent._run_auto_context_post_turn()

        rebuilds = [d for k, d in events if k == AgentEvent.CHAT_REBUILD]
        assert len(rebuilds) == 1
        assert rebuilds[0]["system_line"] == (
            "Auto-context: 2/2 restarts used — turn stopped, context "
            "compacted. Send a message to continue. (session total: 7)"
        )

    @pytest.mark.asyncio
    async def test_chain_not_exhausted_default_footnote(self):
        """Budget remaining: the footer stays the generic chain-restart
        line (the exhaustion notice must not appear on a normal restart)."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=5,
        )
        agent.messages = []
        events = self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        rebuilds = [d for k, d in events if k == AgentEvent.CHAT_REBUILD]
        assert len(rebuilds) == 1
        assert rebuilds[0]["system_line"] == (
            "Context compacted — showing session summary"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Session-lifetime chain count (persisted compaction total + session start)
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionChainCount:
    """_chain_count counts every FULL history replacement (compact_history,
    _mechanical_restart) and survives save/load; /new_session (clear
    mode="all") resets it together with the session start time. Journal-
    reflection compaction is excluded by design (it calls history.compact()
    directly, never these two sites)."""

    def _mock_restart_deps(self, agent, seeded=True):
        agent.clear_messages = MagicMock(return_value=0)
        agent.seed_from_records = MagicMock(return_value=seeded)
        agent._estimate_post_compact_context = AsyncMock(return_value=100)
        agent.emit = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_mechanical_restart_increments(self):
        agent = self._mock_restart_deps(make_agent())
        assert agent._chain_count == 0

        seeded = await agent._mechanical_restart()

        assert seeded is True
        assert agent._chain_count == 1

    @pytest.mark.asyncio
    async def test_mechanical_restart_increments_without_records(self):
        """The window was wiped and re-introduced - that's a chain even
        when no record content was found."""
        agent = self._mock_restart_deps(make_agent(), seeded=False)

        seeded = await agent._mechanical_restart()

        assert seeded is False
        assert agent._chain_count == 1

    @pytest.mark.asyncio
    async def test_compact_history_increments(self):
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "old history " * 200}]

        async def mock_stream(
            client, model, messages, system_prompt, tools, **kw
        ):
            yield ("content", "Compacted summary text.")
            yield (
                "token_usage",
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        with patch(
            "agent13.llm.stream_response_with_tools", side_effect=mock_stream
        ):
            ok, _ = await agent.compact_history("")

        assert ok
        assert agent._chain_count == 1

    @pytest.mark.asyncio
    async def test_compact_history_failure_does_not_increment(self):
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "old history"}]
        agent.emit = AsyncMock()

        async def mock_stream(
            client, model, messages, system_prompt, tools, **kw
        ):
            raise RuntimeError("boom")
            yield  # pragma: no cover

        with patch(
            "agent13.llm.stream_response_with_tools", side_effect=mock_stream
        ):
            ok, _ = await agent.compact_history("")

        assert not ok
        assert agent._chain_count == 0

    @pytest.mark.asyncio
    async def test_clear_all_resets_counters(self):
        """/new_session dispatch (mode="all") resets total + start time."""
        agent = make_agent()
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent._emit_queue_update = AsyncMock()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._chain_count = 20
        agent._session_started_at = "2026-09-21T14:32:00"
        item = QueueItem(
            id=1, text="", kind="clear", data={"mode": "all", "keep_turns": 0}
        )

        await agent._process_item(item)

        assert agent._chain_count == 0
        assert agent._session_started_at != "2026-09-21T14:32:00"

    @pytest.mark.asyncio
    async def test_clear_trim_does_not_reset_counters(self):
        """Trim is a history trim, not a new session - counters survive."""
        agent = make_agent()
        agent.queue.complete_current = MagicMock()
        agent.emit = AsyncMock()
        agent._emit_queue_update = AsyncMock()
        agent.messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "b"},
        ]
        agent._chain_count = 20
        agent._session_started_at = "2026-09-21T14:32:00"
        item = QueueItem(
            id=1, text="", kind="clear", data={"mode": "trim", "keep_turns": 1}
        )

        await agent._process_item(item)

        assert agent._chain_count == 20
        assert agent._session_started_at == "2026-09-21T14:32:00"

    def test_public_properties(self):
        agent = make_agent()
        agent._chain_count = 3
        assert agent.chain_count == 3
        # Fresh agent: ISO-parseable start time
        datetime.datetime.fromisoformat(agent.session_started_at)

    def test_reset_session_counters(self):
        agent = make_agent()
        agent._chain_count = 20
        agent._session_started_at = "2026-09-21T14:32:00"

        agent.reset_session_counters()

        assert agent._chain_count == 0
        assert agent._session_started_at != "2026-09-21T14:32:00"


# ────────────────────────────────────────────────────────────────────────────
# In-flight instruction carry (standing instruction across chain restarts)
# ────────────────────────────────────────────────────────────────────────────


class TestInFlightInstructionCarry:
    """The turn-start user instruction is carried VERBATIM into every chain
    restart's nudge. The user's standing instruction is the intent source
    across chains: a mid-turn question survives its restart, and the intro
    no longer commands ticket work (the former "continue on the top
    unblocked ticket" line is gone)."""

    @staticmethod
    def _capture(agent) -> list:
        events: list = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture
        return events

    @pytest.mark.asyncio
    async def test_nudge_carries_instruction_verbatim(self):
        """report_and_compact restart: the nudge is the carry hint with the
        turn-start instruction verbatim; the intro has no ticket command."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=1,
        )
        agent.messages = [
            {"role": "user", "content": "close the first ticket"},
            {"role": "assistant", "content": "on it"},
        ]
        agent._in_flight_user_request = "close the first ticket"
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        nudge = agent.messages[-1]
        assert nudge["role"] == "user"
        assert nudge.get("injected") is True
        assert nudge["content"] == AUTO_CONTEXT_CARRY_HINT.format(
            instruction="close the first ticket"
        )
        # The intro (first message) no longer commands the ticket queue.
        assert "top unblocked ticket" not in agent.messages[0]["content"]
        assert "Read the records below." in agent.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_nudge_falls_back_to_bare_hint_without_instruction(self):
        """No captured instruction (e.g. manual /chain while idle) -> the
        bare continue hint, as before."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=1,
        )
        agent.messages = [{"role": "assistant", "content": "prior turn"}]
        agent._in_flight_user_request = ""
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        assert agent.messages[-1]["content"] == AUTO_CONTEXT_CONTINUE_HINT

    @pytest.mark.asyncio
    async def test_instruction_persists_across_restarts_in_same_turn(self):
        """Re-tripping continuation turn, chain=2: BOTH restarts' nudges
        carry the original instruction (the stored state, not a re-scan of
        the post-restart history, which would find the injected intro)."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=2,
        )
        nudges: list = []

        async def fake_llm_turn(*args, **kwargs):
            if agent.messages:
                nudges.append(agent.messages[-1]["content"])
            agent._auto_context_triggered = True

        agent._llm_turn = fake_llm_turn
        agent.messages = [{"role": "user", "content": "close the first ticket"}]
        agent._in_flight_user_request = "close the first ticket"
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        self._capture(agent)
        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        carried = [c for c in nudges if "standing instruction" in c]
        assert len(carried) == 2
        assert all("close the first ticket" in c for c in carried)

    def test_capture_from_restored_history(self):
        """Resume path: the scan finds the last genuine user message of the
        restored history (the interrupted turn's instruction)."""
        agent = make_agent()
        agent.messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "close the first ticket"},
            {"role": "assistant", "content": "working on it"},
            {"role": "tool", "content": "tool output", "tool_call_id": "x"},
        ]
        assert agent._capture_in_flight_user_request() == "close the first ticket"
        # Stored: a second call is a no-op returning the same value.
        assert agent._in_flight_user_request == "close the first ticket"

    def test_capture_skips_injected_and_interrupt_messages(self):
        agent = make_agent()
        agent.messages = [
            {"role": "user", "content": "real question"},
            {"role": "user", "content": "injected nudge", "injected": True},
            {"role": "user", "content": "[Interrupted]", "interrupt": True},
        ]
        assert agent._capture_in_flight_user_request() == "real question"

    def test_capture_no_user_message_returns_empty(self):
        agent = make_agent()
        agent.messages = []
        assert agent._capture_in_flight_user_request() == ""
        assert agent._in_flight_user_request == ""

    def test_turn_state_reset_clears_instruction(self):
        agent = make_agent()
        agent._in_flight_user_request = "stale instruction"
        agent._reset_auto_context_turn_state()
        assert agent._in_flight_user_request == ""

    def test_intro_and_carry_hint_text(self):
        intro = CHAIN_RESTART_INTRO.format(truth_file="t.md", journal_file="j.md")
        assert "top unblocked ticket" not in intro
        assert intro.endswith("Read the records below.")
        assert AUTO_CONTEXT_CARRY_HINT.format(instruction="q") == (
            "[Context was compacted mid-turn. Your standing instruction for "
            "this run, verbatim:\n" '"q"\nContinue it.]'
        )


# ────────────────────────────────────────────── Pre-turn threshold ──────────────────────────────────────────────


class TestPreTurnThreshold:
    """The turn-start threshold check (the resume bug): a turn can end
    over-threshold without _llm_turn's safe-point check ever firing (a final
    response has no tool boundary), so when the user asks for the next thing
    the check runs BEFORE that turn's first LLM call. A finished turn is left
    alone — the compact happens when the user continues.
    _estimate_current_context_tokens is mocked; everything else is real."""

    @staticmethod
    def _capture(agent) -> list:
        events: list = []

        async def capture(event_type, data):
            events.append((event_type, data))

        agent.emit = capture
        return events

    @pytest.mark.asyncio
    async def test_pre_turn_over_threshold_compacts(self):
        """Over-threshold at turn start → compact runs before the turn, with
        NO nudge restart (the user's message is the continuation) and the
        chain budget untouched."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="compact",
            auto_context_chain=2,
        )
        agent._estimate_current_context_tokens = MagicMock(return_value=5000)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        events = self._capture(agent)

        await agent._check_threshold_pre_turn()

        assert agent.compact_history.await_count == 1
        # No nudge restart: the caller runs the user's turn itself.
        assert agent._llm_turn.await_count == 0
        assert agent._auto_context_chain_used == 0
        assert [
            m for m in agent.messages if m.get("content") == AUTO_CONTEXT_CONTINUE_HINT
        ] == []
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("compacting" in d["message"] for d in notices)
        assert any("compacted — continuing" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_pre_turn_under_threshold_is_noop(self):
        """Under-threshold at turn start → nothing runs."""
        agent = make_agent(auto_context_threshold=1000, auto_context_action="compact")
        agent._estimate_current_context_tokens = MagicMock(return_value=500)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        events = self._capture(agent)

        await agent._check_threshold_pre_turn()

        assert agent.compact_history.await_count == 0
        assert agent._llm_turn.await_count == 0
        assert [d for k, d in events if k == AgentEvent.NOTIFICATION] == []

    @pytest.mark.asyncio
    async def test_pre_turn_report_and_compact_runs_wrap_up(self):
        """action=report_and_compact at turn start → wrap-up turn, then
        mechanical restart (no compact AI call), no nudge restart."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=2,
        )
        agent._estimate_current_context_tokens = MagicMock(return_value=5000)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        events = self._capture(agent)

        await agent._check_threshold_pre_turn()

        # The wrap-up turn ran (one LLM call), then mechanical restart.
        assert agent._llm_turn.await_count == 1
        assert agent.compact_history.await_count == 0
        assert "Session restarted from durable records" in agent.messages[0]["content"]
        assert agent._auto_context_chain_used == 0
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("wrapping up" in d["message"] for d in notices)
        assert any("compacted — continuing" in d["message"] for d in notices)

    @pytest.mark.asyncio
    async def test_pre_turn_action_none_pauses_and_waits(self):
        """action=none at turn start → pause, wait for the user, no compact,
        no flags."""
        agent = make_agent(auto_context_threshold=1000, auto_context_action="none")
        agent._estimate_current_context_tokens = MagicMock(return_value=5000)
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.pause = MagicMock()
        agent._wait_if_paused = AsyncMock()
        agent.messages = []
        events = self._capture(agent)

        await agent._check_threshold_pre_turn()

        agent.pause.assert_called_once()
        agent._wait_if_paused.assert_awaited_once()
        assert agent.compact_history.await_count == 0
        assert agent._auto_context_triggered is False
        assert agent._report_and_compact_triggered is False
        notices = [d for k, d in events if k == AgentEvent.NOTIFICATION]
        assert any("paused" in d["message"] for d in notices)


# ───────────────────────── TOML parsing ─────────────────────────


def _parse_auto_context(toml_text: str):
    """Write TOML to a temp file and return Config.from_file() (auto-unlink)."""
    import tempfile
    from pathlib import Path

    from agent13.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_text)
        path = Path(f.name)
    try:
        return Config.from_file(path)
    finally:
        path.unlink()


class TestAutoContextConfigParsing:
    def test_default_action_is_report_and_compact(self):
        from agent13.config import Config

        assert Config().auto_context_action == "report_and_compact"

    def test_parse_report_and_compact(self):
        config = _parse_auto_context('[auto_context]\naction = "report_and_compact"\n')
        assert config.auto_context_action == "report_and_compact"

    def test_parse_compact(self):
        config = _parse_auto_context('[auto_context]\naction = "compact"\n')
        assert config.auto_context_action == "compact"

    def test_invalid_action_fails_fast(self):
        from agent13.fileio import ConfigFileError

        with pytest.raises(ConfigFileError, match="action"):
            _parse_auto_context('[auto_context]\naction = "yes"\n')

    def test_parse_none(self):
        config = _parse_auto_context('[auto_context]\naction = "none"\n')
        assert config.auto_context_action == "none"

    def test_default_chain_is_three(self):
        from agent13.config import Config

        assert Config().auto_context_chain == 3

    def test_parse_chain(self):
        config = _parse_auto_context("[auto_context]\nchain = 2\n")
        assert config.auto_context_chain == 2

    def test_chain_accepts_zero(self):
        config = _parse_auto_context("[auto_context]\nchain = 0\n")
        assert config.auto_context_chain == 0

    def test_chain_rejects_negative(self):
        from agent13.fileio import ConfigFileError

        with pytest.raises(ConfigFileError, match="chain"):
            _parse_auto_context("[auto_context]\nchain = -1\n")


class TestAgentTUIWiring:
    """Regression: the CLI passes auto_context_action / report_and_compact_prompt
    to AgentTUI in TUI mode. AgentTUI must forward them to the inner Agent (a
    kwargs mismatch crashes construction with TypeError)."""

    def _make_tui(self, **kwargs):
        from unittest.mock import MagicMock

        from ui.tui import AgentTUI

        defaults = dict(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            prompt_manager=None,
        )
        defaults.update(kwargs)
        return AgentTUI(**defaults)

    def test_tui_accepts_feature_kwargs(self):
        tui = self._make_tui(auto_context_action="report_and_compact")
        assert tui.agent.auto_context_action == "report_and_compact"

    def test_tui_forwards_explicit_prompt(self):
        tui = self._make_tui(
            auto_context_action="report_and_compact",
            report_and_compact_prompt="CUSTOM WRAP-UP",
        )
        assert tui.agent.report_and_compact_prompt == "CUSTOM WRAP-UP"

    def test_tui_resolves_prompt_when_not_given(self):
        tui = self._make_tui(
            auto_context_action="report_and_compact",
            report_and_compact_prompt="",
        )
        assert tui.agent.report_and_compact_prompt == DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_tui_defaults_to_report_and_compact(self):
        tui = self._make_tui()
        assert tui.agent.auto_context_action == "report_and_compact"
        assert tui.agent.auto_context_chain == 3

    def test_auto_context_action_tab_completes(self):
        tui = self._make_tui()
        assert tui._get_param_completions("auto_context_action", "") == [
            "compact",
            "report_and_compact",
            "none",
        ]
        assert tui._get_param_completions("auto_context_action", "re") == [
            "report_and_compact"
        ]

    def test_status_shows_auto_context_lines(self):
        tui = self._make_tui()
        captured = []
        tui._update_info_content = captured.append
        tui._handle_status_command()
        assert len(captured) == 1
        text = captured[0]
        assert "auto-context: [yellow]220,000 tokens[/]" in text
        assert "auto-action: [yellow]report_and_compact[/]" in text
        assert "auto-chain: [yellow]3[/]" in text
        assert "chains: [yellow]0/3[/]" in text

    def test_status_bar_shows_chain_used(self):
        """Bottom bar trn: segment - three segments total:turn:turns.

        trn: {chain_count}:{chain_used}:{turns} - the session-lifetime
        compaction total (persists across saves), chains THIS turn, and
        turns in history. Hidden while total-turns and chains are zero.
        """

        class _Bar:
            def __init__(self):
                self.values = []

            def update(self, v):
                self.values.append(v)

        tui = self._make_tui()
        tui._status_left = _Bar()
        tui._status_right = _Bar()
        right = tui._status_right.values

        # 0 turns, 0 chains -> no trn segment
        tui.update_status()
        assert "trn:" not in right[-1]

        # 2 turns, 0 chains, 0 lifetime -> "trn: 0:0:2"
        tui.agent.messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        tui.update_status()
        assert "trn: 0:0:2" in right[-1]

        # 2 turns, 1 chain restart this turn -> "trn: 0:1:2"
        tui.agent._auto_context_chain_used = 1
        tui.update_status()
        assert "trn: 0:1:2" in right[-1]

        # Lifetime total rides in the leftmost segment
        tui.agent._chain_count = 5
        tui.update_status()
        assert "trn: 5:1:2" in right[-1]

        # Chain fired mid-first-turn (0 turns done) -> "trn: 5:1:0"
        tui.agent.messages = []
        tui.update_status()
        assert "trn: 5:1:0" in right[-1]


# ───────────────────────────────────────────────────────────────────────── TUI commands ─────────────────────────────────────────────────────────────────────────


class TestAutoContextTuiCommands:
    """The /auto_context_* TUI command handlers: display strings + set/validation.

    Handler methods are exercised directly with _update_info_content spied on, so
    the Textual app does not need to be running.
    """

    def _make_tui(self, **kwargs):
        from ui.tui import AgentTUI

        defaults = dict(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            prompt_manager=None,
        )
        defaults.update(kwargs)
        return AgentTUI(**defaults)

    def _info(self, tui):
        """Replace _update_info_content with a spy; return the captured strings."""
        captured = []
        tui._update_info_content = captured.append
        return captured

    def test_threshold_noarg_two_lines_on_default(self):
        tui = self._make_tui()  # threshold 220000, action report_and_compact, chain 3
        captured = self._info(tui)
        tui._handle_auto_context_threshold_command("")
        assert len(captured) == 1
        assert "Auto-context: on (220,000 tokens)" in captured[0]
        assert "Action: report_and_compact (chain: 3)" in captured[0]

    def test_threshold_noarg_two_lines_on_compact(self):
        tui = self._make_tui(auto_context_action="compact", auto_context_chain=0)
        captured = self._info(tui)
        tui._handle_auto_context_threshold_command("")
        assert "Auto-context: on (220,000 tokens)" in captured[0]
        assert "Action: compact (chain: 0)" in captured[0]

    def test_threshold_noarg_action_none_ignores_chain(self):
        tui = self._make_tui(auto_context_action="none", auto_context_chain=5)
        captured = self._info(tui)
        tui._handle_auto_context_threshold_command("")
        assert "Action: none (chain ignored)" in captured[0]
        assert "chain: 5" not in captured[0]

    def test_threshold_noarg_off(self):
        tui = self._make_tui(auto_context_threshold=0)
        captured = self._info(tui)
        tui._handle_auto_context_threshold_command("")
        assert "Auto-context: off (threshold 0)" in captured[0]
        assert "Action:" not in captured[0]

    def test_threshold_set_still_works(self):
        tui = self._make_tui()
        captured = self._info(tui)
        tui._handle_auto_context_threshold_command("150k")
        assert tui.agent.auto_context_threshold == 150000
        assert "Auto-context: on (150,000 tokens)" in captured[0]

    def test_action_noarg_shows_current_and_valid(self):
        tui = self._make_tui(auto_context_action="report_and_compact")
        captured = self._info(tui)
        tui._handle_auto_context_action_command("")
        assert "Auto-context action: report_and_compact" in captured[0]
        assert "compact | report_and_compact | none" in captured[0]

    def test_action_set_valid(self):
        tui = self._make_tui()
        captured = self._info(tui)
        tui._handle_auto_context_action_command("none")
        assert tui.agent.auto_context_action == "none"
        assert "Auto-context action: none" in captured[0]

    def test_action_set_invalid_lists_the_three(self):
        tui = self._make_tui()
        captured = self._info(tui)
        tui._handle_auto_context_action_command("bogus")
        assert tui.agent.auto_context_action == "report_and_compact"  # unchanged
        assert "Usage: /auto_context_action" in captured[0]
        assert "compact" in captured[0]
        assert "report_and_compact" in captured[0]
        assert "none" in captured[0]

    def test_chain_noarg_shows_suffix(self):
        tui = self._make_tui(auto_context_chain=3)
        captured = self._info(tui)
        tui._handle_auto_context_chain_command("")
        assert "Auto-context chain: 3" in captured[0]
        assert "(restarts per turn; 0 = stop after the action)" in captured[0]

    def test_nudge_cap_noarg_shows_on_with_counter(self):
        tui = self._make_tui(auto_context_nudge_cap=2)
        tui.agent._degenerate_nudges_total = 4
        captured = self._info(tui)
        tui._handle_auto_context_nudge_cap_command("")
        assert "Degeneration guard: on (cap 2)" in captured[0]
        assert "nudges: 4" in captured[0]

    def test_nudge_cap_noarg_shows_off_at_zero(self):
        tui = self._make_tui(auto_context_nudge_cap=0)
        captured = self._info(tui)
        tui._handle_auto_context_nudge_cap_command("")
        assert "Degeneration guard: off (nudge cap 0)" in captured[0]

    def test_nudge_cap_set_and_zero_disables(self):
        tui = self._make_tui()
        captured = self._info(tui)
        tui._handle_auto_context_nudge_cap_command("3")
        assert tui.agent.auto_context_nudge_cap == 3
        assert "Degeneration guard cap: 3" in captured[-1]
        tui._handle_auto_context_nudge_cap_command("0")
        assert tui.agent.auto_context_nudge_cap == 0
        assert "Degeneration guard: off (nudge cap 0)" in captured[-1]

    def test_nudge_cap_invalid_shows_usage(self):
        tui = self._make_tui()
        captured = self._info(tui)
        tui._handle_auto_context_nudge_cap_command("bogus")
        assert tui.agent.auto_context_nudge_cap == 2  # unchanged
        assert "Usage: /auto_context_nudge_cap" in captured[0]

    @pytest.mark.asyncio
    async def test_degenerate_nudge_event_writes_system_line(self):
        """DEGENERATE_NUDGE renders as a persistent dim/yellow system footer.

        The handler defers via asyncio.create_task, so the test yields a
        loop tick before asserting.
        """
        tui = self._make_tui()
        writes = []

        async def spy(text, **kwargs):
            writes.append(text)

        tui._write_system = spy
        await tui.agent.emit(
            AgentEvent.DEGENERATE_NUDGE,
            {
                "message": "Model output was cut off - nudged",
                "level": "info",
                "action": "nudge",
            },
        )
        await asyncio.sleep(0)
        assert writes == ["[dim]Model output was cut off - nudged[/]"]
        await tui.agent.emit(
            AgentEvent.DEGENERATE_NUDGE,
            {
                "message": "I stopped this turn",
                "level": "warning",
                "action": "stop",
            },
        )
        await asyncio.sleep(0)
        assert writes[-1] == "[yellow]I stopped this turn[/]"

    @pytest.mark.asyncio
    async def test_chat_clear_event_clears_display(self):
        """CHAT_CLEAR wipes the chat display inline (not deferred)."""
        tui = self._make_tui()
        remove_spy = AsyncMock()
        tui._chat = MagicMock()
        tui._chat.remove_children = remove_spy
        tui._chat_ready = MagicMock(return_value=True)
        await tui.agent.emit(AgentEvent.CHAT_CLEAR, {})
        remove_spy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_rebuild_event_rebuilds_with_summary(self):
        """CHAT_REBUILD rebuilds the chat from current history and writes
        the system footer line."""
        tui = self._make_tui()
        rebuild_spy = AsyncMock()
        write_sys_spy = AsyncMock()
        tui._rebuild_chat = rebuild_spy
        tui._write_system = write_sys_spy
        await tui.agent.emit(
            AgentEvent.CHAT_REBUILD,
            {"system_line": "Context compacted — showing session summary"},
        )
        # on_chat_rebuild defers via create_task; yield so the task runs.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        rebuild_spy.assert_awaited_once()
        write_sys_spy.assert_awaited_once_with(
            "Context compacted — showing session summary"
        )

    @pytest.mark.asyncio
    async def test_rebuild_skips_text_injected_user_messages(self):
        """Text-only injected user messages (chain-restart nudge) are not
        rendered as user widgets; image injections (list content) keep their
        "[Image from tool: X]" trace."""
        tui = self._make_tui()
        write_user = AsyncMock()
        tui._write_user = write_user
        tui._chat = MagicMock()
        await tui._rebuild_from_messages(
            [
                {
                    "role": "user",
                    "content": "Give me a summary of our previous session",
                },
                {
                    "role": "user",
                    "content": "[Context was compacted to fit. Continue.]",
                    "injected": True,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "[Image from tool: read_file]"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,x"},
                        },
                    ],
                    "injected": True,
                },
            ]
        )
        calls = [c.args[0] for c in write_user.await_args_list]
        assert calls == [
            "Give me a summary of our previous session",
            "[Image from tool: read_file]",
        ]


class TestAutoContextReplCommand:
    """REPL surface: the /auto_context_* handlers are inline in the REPL dispatch
    loop, so the only cleanly importable seam is the COMMANDS help dict."""

    def test_help_lists_all_three_commands(self):
        from agent13.repl import COMMANDS

        assert "/auto_context_threshold" in COMMANDS
        assert "/auto_context_action" in COMMANDS
        assert "/auto_context_chain" in COMMANDS


# ───────────────────────── Mid-turn pause resume status ─────────────────────────


class TestPauseWaitResumeStatus:
    """The mid-turn pauses in _llm_turn resume as WAITING, not IDLE.

    Two callers inside _llm_turn pause mid-turn: the safe point after tool
    results, and the auto-context pause_wait branch — which continues straight
    back into the loop, skipping the _set_status(WAITING) the normal path runs.
    Both must report WAITING, otherwise the status bar reads "idle" while the
    agent is working.
    """

    @staticmethod
    def _stateful_stream(first_call_events):
        """First call yields first_call_events; later calls terminate the loop."""
        state = {"n": 0}

        async def fake_stream_and_emit(
            messages, *, source="assistant", tool_choice="auto"
        ):
            state["n"] += 1
            if state["n"] == 1:
                for ev in first_call_events:
                    yield ev
            else:
                yield ("content", "Done.")

        return fake_stream_and_emit

    @staticmethod
    def _prepare(agent):
        """Undo make_agent's mocks, wire a one-tool-call stream, spy the pause."""
        agent._llm_turn = Agent._llm_turn.__get__(agent)
        agent._running = True

        async def exec_tool(name, arguments):
            return '{"ok": true}'

        agent.execute_tool = exec_tool
        agent._stream_and_emit = TestPauseWaitResumeStatus._stateful_stream(
            [
                (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "name": "read_file",
                                "arguments": '{"filepath": "x.py"}',
                            }
                        ]
                    },
                ),
            ]
        )

        resume_statuses = []

        async def fake_wait_if_paused(resume_status=None):
            resume_statuses.append(resume_status)

        agent._wait_if_paused = fake_wait_if_paused
        return resume_statuses

    @pytest.mark.asyncio
    async def test_safe_point_resumes_as_waiting(self):
        """The post-tool safe point reports WAITING."""
        agent = make_agent(auto_context_threshold=1, auto_context_action="none")
        resume_statuses = self._prepare(agent)
        # Keep the threshold check out of the way: only the safe point runs.
        agent._should_check_threshold = lambda estimated: False

        await agent._llm_turn()

        assert resume_statuses == [AgentStatus.WAITING], (
            f"the post-tool safe point must resume as WAITING, got: {resume_statuses}"
        )

    @pytest.mark.asyncio
    async def test_pause_wait_resumes_as_waiting(self):
        """The auto-context pause_wait branch reports WAITING too."""
        agent = make_agent(auto_context_threshold=1, auto_context_action="none")
        resume_statuses = self._prepare(agent)

        # Prove the pause_wait branch ran. The gate stays true, so both
        # mid-turn callers fire: the safe point, then the pause_wait branch.
        returned = []
        real_check = agent._handle_threshold_check

        async def spy(estimated):
            result = await real_check(estimated)
            returned.append(result)
            return result

        agent._handle_threshold_check = spy

        await agent._llm_turn()

        assert returned == ["pause_wait"], f"expected a pause, got: {returned}"
        assert resume_statuses == [AgentStatus.WAITING, AgentStatus.WAITING], (
            "both mid-turn pauses must resume as WAITING (not IDLE), "
            f"got: {resume_statuses}"
        )
