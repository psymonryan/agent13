"""Resuming a turn that was paused MID-WRAP-UP.

The defect: a pause at a tool boundary inside the wrap-up, then save/quit,
then --continue + /resume. `_check_threshold_pre_turn` fired first (the
history is over threshold and `_suppress_auto_context` is False in the new
process), armed a FRESH wrap-up, and ran it on a history that already
contained the half-finished one - then the restart wiped both. Measured
2026-11-01: 2 `_llm_turn` calls, 4 wrap-up messages in the history at the
second one, and the resumed turn's instruction dropped entirely.

The fix: on the resume path, an unfinished wrap-up is FINISHED, not
restarted. The pre-turn check arms the compact alone; the post-turn handler
snapshots, restarts and nudges after the wrap-up completes.

`is_wrapup_current` cannot answer this: it is True both for a wrap-up that
finished and for one still running. `wrapup_in_progress` is the second
question on its own.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent13.core import Agent
from agent13.durable_records import DurableRecords

THRESHOLD = 100_000


def make_agent(auto_context_threshold=THRESHOLD, chain=3):
    return Agent(
        client=MagicMock(),
        model="test-model",
        auto_context_threshold=auto_context_threshold,
        auto_context_chain=chain,
    )


def _marker():
    return {"role": "assistant", "content": "[Wrapping up]", "report_and_compact": True}


def _prompt():
    return {
        "role": "user",
        "content": "WRAP-UP PROMPT ... Current date and time: 2026-11-02 08:00",
        "report_and_compact": True,
    }


def _tool_call(cid="c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "name": "edit_file", "arguments": "{}"}],
    }


def _tool_result(cid="c1"):
    return {"role": "tool", "tool_call_id": cid, "content": "journal entry appended"}


def _mid_wrapup_history():
    """Paused inside the wrap-up: the marker has no completion after it."""
    return [
        {"role": "user", "content": "fix the flaky test"},
        _tool_call("c0"),
        _tool_result("c0"),
        _marker(),
        _prompt(),
        _tool_call("c1"),
        _tool_result("c1"),
    ]


def _finished_wrapup_history():
    """The wrap-up completed, then more tool work happened."""
    return [
        {"role": "user", "content": "fix the flaky test"},
        _marker(),
        _prompt(),
        _tool_call("c1"),
        _tool_result("c1"),
        {"role": "assistant", "content": "Wrap-up done."},
        _tool_call("c2"),
        _tool_result("c2"),
    ]


# ── the predicate ─────────────────────────────────────────────────────────


class TestWrapupInProgress:
    def test_no_wrapup_marker_is_false(self):
        rec = DurableRecords()
        assert rec.wrapup_in_progress([{"role": "user", "content": "hi"}]) is False

    def test_marker_without_completion_is_true(self):
        rec = DurableRecords()
        assert rec.wrapup_in_progress(_mid_wrapup_history()) is True

    def test_marker_with_completion_is_false(self):
        rec = DurableRecords()
        assert rec.wrapup_in_progress(_finished_wrapup_history()) is False

    def test_completion_found_even_when_more_tools_follow(self):
        """A finished wrap-up followed by more work is NOT in progress."""
        rec = DurableRecords()
        history = _mid_wrapup_history() + [
            {"role": "assistant", "content": "Wrap-up done."},
            _tool_call("c9"),
            _tool_result("c9"),
        ]
        assert rec.wrapup_in_progress(history) is False

    def test_disagrees_with_is_wrapup_current_on_the_in_progress_case(self):
        """The whole reason the predicate exists: they differ here."""
        rec = DurableRecords()
        history = _mid_wrapup_history()
        assert rec.is_wrapup_current(history) is True  # "still in progress"
        assert rec.wrapup_in_progress(history) is True

    def test_agent_forwards_the_predicate(self):
        agent = make_agent()
        agent.messages = _mid_wrapup_history()
        assert agent._wrapup_in_progress() is True
        agent.messages = _finished_wrapup_history()
        assert agent._wrapup_in_progress() is False


# ── the guard in the pre-turn check ───────────────────────────────────────


class TestPreTurnGuard:
    @pytest.mark.asyncio
    async def test_resuming_into_a_wrapup_does_not_arm_a_second_one(self):
        agent = make_agent()
        agent.messages = _mid_wrapup_history()
        agent.prompt_tokens = 150_000  # over threshold
        agent._estimate_current_context_tokens = MagicMock(return_value=150_000)
        agent._handle_threshold_check = AsyncMock()

        await agent._check_threshold_pre_turn(resuming=True)

        agent._handle_threshold_check.assert_not_called()
        assert agent._report_and_compact_triggered is False
        assert agent._auto_context_triggered is True

    @pytest.mark.asyncio
    async def test_resuming_without_a_wrapup_still_runs_the_action(self):
        """A normal over-threshold resume is unchanged."""
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._estimate_current_context_tokens = MagicMock(return_value=150_000)
        agent._handle_threshold_check = AsyncMock(return_value="break")

        await agent._check_threshold_pre_turn(resuming=True)

        agent._handle_threshold_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_new_user_message_still_wraps_up_a_finished_turn(self):
        """resuming=False keeps the old behaviour: the guard is resume-only."""
        agent = make_agent()
        agent.messages = _mid_wrapup_history()
        agent._estimate_current_context_tokens = MagicMock(return_value=150_000)
        agent._handle_threshold_check = AsyncMock(return_value="break")

        await agent._check_threshold_pre_turn()  # resuming defaults False

        agent._handle_threshold_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_under_threshold_nothing_happens(self):
        agent = make_agent()
        agent.messages = _mid_wrapup_history()
        agent._estimate_current_context_tokens = MagicMock(return_value=10)
        agent._handle_threshold_check = AsyncMock()

        await agent._check_threshold_pre_turn(resuming=True)

        agent._handle_threshold_check.assert_not_called()
        assert agent._auto_context_triggered is False


# ── the whole resume, end to end through continue_incomplete_turn ─────────


class TestResumeMidWrapup:
    @pytest.mark.asyncio
    async def test_one_wrapup_not_two(self):
        """The regression: the resumed wrap-up is continued, not restarted."""
        agent = make_agent()
        agent.messages = _mid_wrapup_history()
        agent.prompt_tokens = 150_000
        agent._incomplete_turn_loaded = True

        seen = []

        async def fake_llm_turn(*args, **kwargs):
            seen.append(
                {
                    "wrapup_msgs": sum(
                        1 for m in agent.messages if m.get("report_and_compact")
                    ),
                    # The continuation is the injected message the restart
                    # appends. It carries the record pointer (or the generic
                    # continue hint when there is no record); it no longer
                    # quotes the user's message - that machinery is gone.
                    "continuation": bool(agent.messages[-1].get("injected")),
                    "quotes_seed": any(
                        "flaky test" in str(m.get("content"))
                        for m in agent.messages
                        if m.get("injected")
                    ),
                }
            )

        agent._llm_turn = fake_llm_turn
        agent._save_auto_context_snapshot = MagicMock()

        await agent.continue_incomplete_turn()

        # Call 1 = the resumed wrap-up continuing (its own pair only: 2
        # messages, not 4). Call 2 = the post-restart continuation.
        assert len(seen) == 2, seen
        assert seen[0]["wrapup_msgs"] == 2, (
            "a second wrap-up pair was injected on top of the half-finished one"
        )
        assert seen[1]["continuation"] is True, (
            "the post-restart continuation injected no message"
        )
        assert seen[1]["quotes_seed"] is False, (
            "the continuation still quotes the user's message"
        )
        assert agent._auto_context_chain_used == 1
