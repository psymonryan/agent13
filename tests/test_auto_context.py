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
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent13.core import Agent, AgentStatus
from agent13.events import AgentEvent
from agent13.message_history import LOCAL_MSG_KEYS, is_turn_start
from agent13.prompts import (
    AUTO_CONTEXT_CONTINUE_HINT,
    DEFAULT_REPORT_AND_COMPACT_PROMPT,
    resolve_report_and_compact_prompt,
)


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
                self._append_tool_call("docs_archive/my_feature_journal.md")
            )

        agent._llm_turn = fake_llm_turn

        await agent._run_report_and_compact_turn()

        assert agent._session_journal_path == "docs_archive/my_feature_journal.md"

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
    async def test_nudge_carries_journal_path(self):
        """The continuation nudge re-announces the captured journal path so it
        survives into every chain restart."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=1,
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._session_journal_path = "docs_archive/my_feature_journal.md"
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        nudges = [
            m
            for m in agent.messages
            if str(m.get("content", "")).startswith("[Context was compacted")
        ]
        assert len(nudges) == 1
        assert "docs_archive/my_feature_journal.md" in nudges[0]["content"]
        assert "append to it at the next wrap-up" in nudges[0]["content"]
        # "injected" keeps the nudge out of turn-start counting so the TUI's
        # chain-restart rebuild keeps the summary pair, not the nudge.
        assert nudges[0].get("injected") is True

    @pytest.mark.asyncio
    async def test_nudge_carries_truth_read_first_cue(self):
        """The nudge names the current-truth file with a read-first cue so
        the new chain orients from the durable facts (and the section 6
        ticket queue) before starting new work."""
        agent = make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=1,
        )
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent.messages = []
        agent._session_truth_path = "docs_archive/my_feature_current_truth.md"
        agent._session_journal_path = "docs_archive/my_feature_journal.md"
        agent._auto_context_triggered = True

        await agent._run_auto_context_post_turn()

        nudges = [
            m
            for m in agent.messages
            if str(m.get("content", "")).startswith("[Context was compacted")
        ]
        assert len(nudges) == 1
        content = nudges[0]["content"]
        # Truth cue: read-first + the section 6 work queue
        assert "docs_archive/my_feature_current_truth.md" in content
        assert "read it in full before starting new work" in content
        assert "section 6" in content
        # Journal cue still present
        assert "docs_archive/my_feature_journal.md" in content

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
                self._append_tool_call("docs_archive/my_feature_journal.md")
            )
            agent.messages.append(
                self._append_tool_call("docs_archive/my_feature_current_truth.md")
            )

        agent._llm_turn = fake_llm_turn

        await agent._run_report_and_compact_turn()

        assert agent._session_truth_path == "docs_archive/my_feature_current_truth.md"
        assert agent._session_journal_path == "docs_archive/my_feature_journal.md"

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
    async def test_report_and_compact_runs_wrap_up_then_compact(self):
        """Regression: the loop must actually RUN the wrap-up turn before the
        pending compact. The wrap-up turn is the only thing that calls _llm_turn
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

        # The wrap-up turn ran: it made exactly one LLM call and injected the
        # fake assistant marker + the prompt before it.
        assert agent._llm_turn.await_count == 1
        roles = [m["role"] for m in agent.messages]
        assert roles[:2] == ["assistant", "user"]
        assert agent.messages[0]["content"] == "[Wrapping up]"
        # Then the pending compact ran, and we went idle (chain=0).
        assert agent.compact_history.await_count == 1
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
        compact, no nudge restart."""
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

        # The wrap-up turn ran (one LLM call), then the pending compact.
        assert agent._llm_turn.await_count == 1
        assert agent.compact_history.await_count == 1
        assert agent.messages[0]["content"] == "[Wrapping up]"
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
        """Bottom bar trn: segment - chain restarts prepended when > 0.

        trn: {chain}:{turns}; the zero chain count is hidden so the common
        case renders exactly as before (trn: {turns}).
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

        # 2 turns, 0 chains -> plain turn count (unchanged format)
        tui.agent.messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        tui.update_status()
        assert "trn: 2" in right[-1]

        # 2 turns, 1 chain restart -> "trn: 1:2"
        tui.agent._auto_context_chain_used = 1
        tui.update_status()
        assert "trn: 1:2" in right[-1]

        # Chain fired mid-first-turn (0 turns done) -> "trn: 1:0"
        tui.agent.messages = []
        tui.update_status()
        assert "trn: 1:0" in right[-1]


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

    @pytest.mark.asyncio
    async def test_chat_clear_event_rebuilds_with_summary(self):
        """Chain restart: CHAT_CLEAR rebuilds keeping the last turn (the
        compacted summary pair) with the compact footer, not a bare wipe."""
        tui = self._make_tui()
        clear_spy = AsyncMock()
        tui._clear_and_show_last_n = clear_spy
        await tui.agent.emit(AgentEvent.CHAT_CLEAR, {})
        # on_chat_clear defers via create_task; yield so the task runs.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        clear_spy.assert_awaited_once_with(
            1, system_line="Context compacted — showing session summary"
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
