"""Tests for the goal window (/grind) - a deadline that survives turn ends.

Work order: docs_archive/work_order_2026-11-03_agent13_goal_loop.md (rev 5),
section 8. The failure it exists to fix: 2026-11-02, a turn ended at 01:53 with
6h 34m of a standing order still to run, and nothing re-prompted the agent.

Two layers:
  - agent13/goal.py - the pure rules (duration, tickets, preamble, answer,
    continuation text). No Agent, no I/O.
  - Agent - the state (`_goal_deadline`, the two question flags), the two
    injection points (the chain restart and the idle hook) and the budget.
"""

import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13 import goal as goal_rules
from agent13.commands import execute_grind
from agent13.core import Agent, AgentStatus
from agent13.events import AgentEvent
from agent13.message_history import is_turn_start
from agent13.persistence import load_context, save_context
from agent13.prompts import (
    AUTO_CONTEXT_CONTINUE_HINT,
    AUTO_CONTEXT_RECORD_FRESH_HINT,
    GOAL_ITEM,
    GOAL_ITEM_RECORD_FALLBACK,
    GOAL_QUESTION_ALL_BLOCKED,
    GOAL_QUESTION_INSTRUCTION,
    GOAL_QUESTION_PLAIN,
    GOAL_QUESTION_UNBLOCKED,
)

# A record with one unblocked ticket and one blocked one. Section 6 is matched
# by TEXT ("## ... open tickets"), not by its number - see truth_format.
RECORD = """\
# Current truth

## 1. Findings

### A topic
- a fact

## 6. Open tickets

- Fix the flaky test - FIRST SEEN 2026-11-01. [observed 2026-11-01]
- Rotate the API key - FIRST SEEN 2026-11-01 - BLOCKED: needs the user. \
[observed 2026-11-01]
"""

ALL_BLOCKED_RECORD = """\
# Current truth

## 6. Open tickets

- Rotate the key - FIRST SEEN 2026-11-01 - BLOCKED: needs the user.
- Ask about the box - FIRST SEEN 2026-11-01 - NEEDS USER: which box?
"""

NO_TICKETS_RECORD = """\
# Current truth

## 1. Findings

### A topic
- a fact

## 6. Open tickets
"""


def make_agent(record=RECORD, mtime=1_000_000.0):
    """A minimal agent with the record and the clock stubbed out."""
    agent = Agent(client=MagicMock(), model="test-model")
    agent._llm_turn = AsyncMock()
    agent.emit = AsyncMock()
    agent.durable_records.read_record_file = MagicMock(return_value=record)
    agent.durable_records.records_mtime = MagicMock(return_value=mtime)
    return agent


def question_turn(agent, answer, tool_calls=None):
    """Put a finished question turn in history: the flagged question + answer."""
    agent.messages = [
        {"role": "user", "content": "earlier work"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": goal_rules.question_text(1, 1, True),
            "goal_question": True,
        },
        {"role": "assistant", "content": answer, "tool_calls": tool_calls or []},
    ]
    agent._goal_question_pending = True


# ─────────────────────────────────────────────────────────────────────────────
# Duration parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestParseDuration:
    @pytest.mark.parametrize(
        "text,seconds",
        [
            ("8h", 8 * 3600),
            ("90m", 90 * 60),
            ("2h30m", 2 * 3600 + 30 * 60),
            ("1h0m", 3600),
            ("0h", 0.0),
            ("23M", 23 * 60),  # case-insensitive
        ],
    )
    def test_hours_and_minutes(self, text, seconds):
        assert goal_rules.parse_duration(text) == (seconds, "")

    @pytest.mark.parametrize("text", ["90s", "1h30s", "1.5h", "h", "m", "abc", ""])
    def test_seconds_and_garbage_are_rejected(self, text):
        """Seconds are rejected so the status bar's window can never be
        seconds: both sides of the ratio are H:MM."""
        assert goal_rules.parse_duration(text) == (None, "")

    def test_remainder_is_the_goal_text_verbatim(self):
        assert goal_rules.parse_duration("23m Try to crack the key") == (
            23 * 60,
            "Try to crack the key",
        )

    def test_remainder_keeps_internal_spacing(self):
        seconds, text = goal_rules.parse_duration("  8h  two  spaces  ")
        assert seconds == 8 * 3600
        assert text == "two  spaces"


# ─────────────────────────────────────────────────────────────────────────────
# The mechanical ticket check
# ─────────────────────────────────────────────────────────────────────────────


class TestTicketClassifier:
    def test_unblocked(self):
        bullet = "- Fix it - FIRST SEEN 2026-11-01. [observed 2026-11-01]"
        assert goal_rules.ticket_is_blocked(bullet) is False

    def test_blocked(self):
        bullet = "- Fix it - FIRST SEEN 2026-11-01 - BLOCKED: needs the key."
        assert goal_rules.ticket_is_blocked(bullet) is True

    def test_needs_user(self):
        bullet = "- Fix it - FIRST SEEN 2026-11-01 - NEEDS USER: which box?"
        assert goal_rules.ticket_is_blocked(bullet) is True

    def test_parked_in_place(self):
        """Parked-in-place is a live shape: the bullet stays where it is and
        PARKED goes into its body (truth_format.ticket_order)."""
        bullet = "- Fix it - FIRST SEEN 2026-11-01 - PARKED 2026-11-02."
        assert goal_rules.ticket_is_blocked(bullet) is True

    def test_no_first_seen_stamp_is_searched_whole(self):
        assert goal_rules.ticket_is_blocked("- Fix it - BLOCKED: no key") is True
        assert goal_rules.ticket_is_blocked("- Fix it") is False

    def test_word_boundaries(self):
        """"unblocked" contains "blocked" as a substring, and a ticket that
        says so is the opposite of blocked."""
        bullet = "- Fix it - FIRST SEEN 2026-11-01 - unblocked by the merge."
        assert goal_rules.ticket_is_blocked(bullet) is False

    def test_only_the_tail_after_the_stamp_counts(self):
        """The wrap-up writes the state after the stamp; a marker in the title
        is the ticket's subject, not its state."""
        bullet = "- Remove the BLOCKED marker - FIRST SEEN 2026-11-01. [observed]"
        assert goal_rules.ticket_is_blocked(bullet) is False

    def test_counts_from_a_record(self):
        assert goal_rules.ticket_counts(RECORD) == (1, 1, True)

    def test_counts_with_no_tickets(self):
        assert goal_rules.ticket_counts(NO_TICKETS_RECORD) == (0, 0, True)

    def test_counts_with_no_record(self):
        assert goal_rules.ticket_counts("") == (0, 0, False)


# ─────────────────────────────────────────────────────────────────────────────
# The question
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestionText:
    def test_unblocked_tickets_exist(self):
        text = goal_rules.question_text(2, 1, True)
        assert text.startswith(GOAL_QUESTION_UNBLOCKED)

    def test_all_blocked(self):
        text = goal_rules.question_text(0, 3, True)
        assert text.startswith(GOAL_QUESTION_ALL_BLOCKED)

    def test_no_tickets(self):
        assert goal_rules.question_text(0, 0, True).startswith(GOAL_QUESTION_PLAIN)

    def test_no_record(self):
        assert goal_rules.question_text(0, 0, False).startswith(GOAL_QUESTION_PLAIN)

    @pytest.mark.parametrize("counts", [(2, 1, True), (0, 3, True), (0, 0, False)])
    def test_all_three_end_with_the_same_instruction(self, counts):
        """The shape of the answer must not vary, or the parser has three
        shapes to tolerate."""
        assert goal_rules.question_text(*counts).endswith(GOAL_QUESTION_INSTRUCTION)

    def test_unresolved_for_user_items_are_injected(self):
        """The grind's own question is the gate - no new flag.

        A model that wrote "this is a scope call for you" and then answers
        "can progress: true" is contradicting itself inside one turn, and the
        question is what forces the reconciliation.
        """
        text = goal_rules.question_text(1, 1, True, ["scope call: continue?"])
        assert "FOR THE USER (unresolved):" in text
        assert "- scope call: continue?" in text
        assert "Can you make progress without an answer to this?" in text
        # The answer shape is unchanged: the parser still has one shape.
        assert text.endswith(GOAL_QUESTION_INSTRUCTION)

    def test_no_items_leaves_the_question_unchanged(self):
        plain = goal_rules.question_text(1, 1, True)
        assert goal_rules.question_text(1, 1, True, []) == plain
        assert "FOR THE USER" not in plain

    def test_every_item_is_listed(self):
        text = goal_rules.question_text(1, 1, True, ["a", "b", "c"])
        assert "- a" in text and "- b" in text and "- c" in text


# ─────────────────────────────────────────────────────────────────────────────
# The answer
# ─────────────────────────────────────────────────────────────────────────────


class TestAnswerParser:
    def test_bare_json(self):
        answer = goal_rules.parse_answer('{"can_progress": true, "next": "do X"}')
        assert answer == {"can_progress": True, "next": "do X"}

    def test_fenced_json(self):
        text = '```json\n{"can_progress": false, "next": ""}\n```'
        assert goal_rules.parse_answer(text) == {"can_progress": False, "next": ""}

    def test_json_inside_prose(self):
        text = 'Here is my answer: {"can_progress": true, "next": "step"} - done.'
        assert goal_rules.parse_answer(text)["next"] == "step"

    def test_braces_inside_a_string(self):
        text = '{"can_progress": true, "next": "fix the {x} thing"}'
        assert goal_rules.parse_answer(text)["next"] == "fix the {x} thing"

    def test_truncated_json_is_unparseable(self):
        assert goal_rules.parse_answer('{"can_progress": true, "next": "do') is None

    def test_prose_with_no_json(self):
        assert goal_rules.parse_answer("I think I can keep going.") is None

    def test_empty(self):
        assert goal_rules.parse_answer("") is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("false", False),
            ("no", False),
            ("0", False),
            ("", False),
            (None, True),
            (1, True),
            (0, False),
        ],
    )
    def test_can_progress_coercion(self, value, expected):
        assert goal_rules.can_progress({"can_progress": value}) is expected

    def test_anything_ambiguous_continues(self):
        """Continuing is the safe failure; stopping silently is the failure the
        window exists to fix."""
        assert goal_rules.can_progress(None) is True
        assert goal_rules.can_progress({}) is True

    def test_waiting_on_user_normalisation(self):
        assert goal_rules.waiting_on({"waiting_on_user": "the key"}) == ["the key"]
        assert goal_rules.waiting_on({"waiting_on_user": ["a", " b "]}) == ["a", "b"]
        assert goal_rules.waiting_on({"waiting_on_user": None}) == []
        assert goal_rules.waiting_on(None) == []


# ─────────────────────────────────────────────────────────────────────────────
# The continuation message
# ─────────────────────────────────────────────────────────────────────────────


class TestGoalItem:
    def test_goal_line_and_step(self):
        assert goal_rules.goal_item("ship it", "run the tests", True) == (
            f"Goal: ship it\n{GOAL_ITEM.format(step='run the tests')}"
        )

    def test_step_alone_when_there_is_no_goal_text(self):
        assert goal_rules.goal_item("", "run the tests", True) == (
            GOAL_ITEM.format(step="run the tests")
        )

    def test_record_fallback_with_goal_text(self):
        assert goal_rules.goal_item("ship it", "", True) == (
            f"Goal: ship it\n{GOAL_ITEM_RECORD_FALLBACK}"
        )

    def test_record_fallback_without_goal_text(self):
        assert goal_rules.goal_item("", "", True) == GOAL_ITEM_RECORD_FALLBACK

    def test_bare_goal_with_no_record(self):
        """A bare goal is still a valid instruction."""
        assert goal_rules.goal_item("ship it", "", False) == "Goal: ship it"

    def test_nothing_to_say(self):
        """No record, no goal text and no next step: the caller substitutes
        AUTO_CONTEXT_CONTINUE_HINT - an empty item is never enqueued."""
        assert goal_rules.goal_item("", "", False) == ""

    @pytest.mark.parametrize(
        "answer",
        [
            {},  # absent
            {"next": ""},  # empty
            {"next": None},  # null
            {"next": 42},  # wrong type
            {"can_progress": True},  # absent, with the decision present
        ],
    )
    def test_missing_next_is_unrecoverable(self, answer):
        assert goal_rules.next_step(answer) == ""

    def test_unparseable_answer_is_unrecoverable(self):
        assert goal_rules.next_step(goal_rules.parse_answer("no json here")) == ""

    def test_next_is_trimmed(self):
        assert goal_rules.next_step({"next": "  run the tests  "}) == "run the tests"

    def test_never_refers_to_the_deleted_exchange(self):
        """The pair is deleted, so "you said your next step would be ..." is
        false from the model's point of view."""
        text = goal_rules.goal_item("ship it", "run the tests", True)
        assert "you said" not in text.lower()
        assert "may be that or something better" not in text.lower()

    def test_the_label_carries_its_own_provenance(self):
        """The wrap-up QUOTES this line, so the quote must say whose it is.

        Live failure it prevents: edmac 2026-11-06 06:20 recorded a re-rank
        "by user direction at 06:09" quoting this message, while the history
        file has no user message before 18:23 that day.
        """
        text = goal_rules.goal_item("ship it", "run the tests", True)
        assert "your own proposal" in text
        assert "not a user instruction" in text


# ─────────────────────────────────────────────────────────────────────────────
# The dur: formatter (H:MM on both sides)
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatWindow:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0:00"),
            (2, "0:00"),
            (125, "0:02"),
            (23 * 60, "0:23"),
            (3.5 * 3600, "3:30"),
            (4 * 3600, "4:00"),
            (10 * 3600, "10:00"),
            (-5, "0:00"),
        ],
    )
    def test_hours_and_minutes_never_seconds(self, seconds, expected):
        assert goal_rules.format_window(seconds) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Goal state
# ─────────────────────────────────────────────────────────────────────────────


class TestGoalState:
    def test_no_goal_by_default(self):
        agent = make_agent()
        assert agent.goal_active is False
        assert agent.goal_remaining == 0.0
        assert agent.goal_window == 0.0

    def test_active_until_the_deadline(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        assert agent.goal_active is True
        assert 3595 < agent.goal_remaining <= 3600
        assert agent.goal_window == pytest.approx(3600, abs=1)

    def test_deadline_only_the_text_is_optional(self):
        """An empty goal text must not disarm the window."""
        agent = make_agent()
        agent.set_goal(3600)
        assert agent.goal_active is True
        assert agent._goal_text == ""

    def test_expired_window_is_inactive(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        assert agent.goal_active is False

    def test_expire_clears_the_deadline_and_the_set_at(self):
        """_goal_set_at feeds only the window length; leaving it would make
        deadline - set_at negative (a window that reads backwards)."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        agent._expire_goal()
        assert agent._goal_deadline == 0.0
        assert agent._goal_set_at == 0.0
        assert agent.goal_window == 0.0

    def test_expire_keeps_the_goal_text(self):
        """The text is what /grind re-arms and what the status line shows."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        agent._expire_goal()
        assert agent._goal_text == "ship it"

    def test_expire_is_one_shot(self):
        """The deadline is its own latch: the counter is zeroed exactly once,
        then accumulates again so the budget binds for the wind-down."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        agent._auto_context_chain_used = 2

        agent._expire_goal()
        assert agent._auto_context_chain_used == 0

        # The wind-down spends restarts; a second evaluation must not zero.
        agent._auto_context_chain_used = 1
        agent._expire_goal()
        assert agent._auto_context_chain_used == 1

    def test_expire_on_a_live_window_does_nothing(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._auto_context_chain_used = 2
        agent._expire_goal()
        assert agent.goal_active is True
        assert agent._auto_context_chain_used == 2


# ─────────────────────────────────────────────────────────────────────────────
# The /grind command
# ─────────────────────────────────────────────────────────────────────────────


class TestGrindCommand:
    def test_arms_the_window(self):
        agent = make_agent()
        result = execute_grind(agent, "4h ship the thing")
        assert result.success is True
        assert agent.goal_active is True
        assert agent._goal_text == "ship the thing"
        assert "4:00" in result.message

    def test_text_is_sticky(self):
        """A second /grind always recomputes the deadline and replaces the text
        only when text is given."""
        agent = make_agent()
        execute_grind(agent, "23m Try to crack the key")
        execute_grind(agent, "6h")
        assert agent._goal_text == "Try to crack the key"
        assert agent.goal_window == pytest.approx(6 * 3600, abs=1)

    def test_text_is_replaced_when_given(self):
        agent = make_agent()
        execute_grind(agent, "23m first goal")
        execute_grind(agent, "6h second goal")
        assert agent._goal_text == "second goal"

    def test_no_text_leaves_it_empty_but_active(self):
        agent = make_agent()
        result = execute_grind(agent, "6h")
        assert result.success is True
        assert agent._goal_text == ""
        assert agent.goal_active is True

    def test_off_stops_the_window_but_keeps_the_text(self):
        """"off" stops the clock; it does not forget what was asked for."""
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        result = execute_grind(agent, "off")
        assert result.success is True
        assert agent.goal_active is False
        assert agent._goal_deadline == 0.0
        assert agent._goal_text == "ship it"

    def test_off_says_what_it_kept(self):
        # The confirmation is the only thing distinguishing off from clear at
        # the point of use. "Grind cleared" here would be a lie.
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        result = execute_grind(agent, "off")
        assert "kept" in result.message
        assert "ship it" in result.message

    def test_off_with_no_text_says_there_was_nothing_to_keep(self):
        agent = make_agent()
        execute_grind(agent, "6h")
        result = execute_grind(agent, "off")
        assert result.success is True
        assert "no goal text" in result.message

    def test_clear_drops_the_text_and_stops_the_window(self):
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        result = execute_grind(agent, "clear")
        assert result.success is True
        assert agent.goal_active is False
        assert agent._goal_deadline == 0.0
        assert agent._goal_text == ""

    def test_clear_is_idempotent(self):
        # Nothing armed, nothing kept: still a success. An error here would
        # teach nothing and would make a scripted "reset" need a guard.
        agent = make_agent()
        result = execute_grind(agent, "clear")
        assert result.success is True
        assert agent._goal_text == ""

    def test_a_bare_arm_after_off_reuses_the_kept_goal(self):
        """The point of keeping the text: /grind 4h re-arms the same goal."""
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        execute_grind(agent, "off")
        execute_grind(agent, "4h")
        assert agent.goal_active is True
        assert agent._goal_text == "ship it"
        assert agent.goal_window == pytest.approx(4 * 3600, abs=1)

    def test_a_bare_arm_after_clear_has_no_goal_text(self):
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        execute_grind(agent, "clear")
        execute_grind(agent, "4h")
        assert agent.goal_active is True
        assert agent._goal_text == ""

    def test_show_after_off_prints_the_kept_text(self):
        # The branch that printed "goal text kept" could never be reached
        # before off kept the text - it was a promise the code did not honour.
        agent = make_agent()
        execute_grind(agent, "6h ship it")
        execute_grind(agent, "off")
        result = execute_grind(agent, "")
        assert result.success is True
        assert "No grind active" in result.message
        assert "ship it" in result.message

    def test_show_when_armed(self):
        agent = make_agent()
        execute_grind(agent, "4h ship it")
        result = execute_grind(agent, "")
        assert result.success is True
        assert "ship it" in result.message
        assert "4:00" in result.message

    def test_show_when_not_armed(self):
        agent = make_agent()
        result = execute_grind(agent, "")
        assert result.success is True
        assert "No grind active" in result.message

    @pytest.mark.parametrize("args", ["90s", "0h", "garbage", "1.5h"])
    def test_bad_duration_changes_nothing(self, args):
        agent = make_agent()
        result = execute_grind(agent, args)
        assert result.success is False
        assert agent.goal_active is False
        assert agent._goal_text == ""

    @pytest.mark.parametrize("sub", ["off", "clear"])
    def test_stopping_clears_the_question_flags(self, sub):
        """A question turn in flight must not leave a flag that stops the next
        window from asking."""
        agent = make_agent()
        execute_grind(agent, "4h ship it")
        agent._goal_question_pending = True
        agent._goal_answer_pending = True
        execute_grind(agent, sub)
        assert agent._goal_question_pending is False
        assert agent._goal_answer_pending is False

    @pytest.mark.parametrize("sub", ["off", "clear"])
    def test_stopping_zeroes_the_chain_counter(self, sub):
        agent = make_agent()
        execute_grind(agent, "4h ship it")
        agent._auto_context_chain_used = 2
        execute_grind(agent, sub)
        assert agent._auto_context_chain_used == 0


# ─────────────────────────────────────────────────────────────────────────────
# The idle hook - the state machine
# ─────────────────────────────────────────────────────────────────────────────


class TestIdleStateMachine:
    @pytest.mark.asyncio
    async def test_no_goal_goes_idle(self):
        agent = make_agent()
        agent.queue.add("hello")
        item = agent.queue.get_next()

        await agent._process_item(item)

        assert agent.queue.pending_count == 0
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_goal_asks_the_question(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.queue.add("hello")
        item = agent.queue.get_next()

        await agent._process_item(item)

        assert agent.queue.pending_count == 1
        queued = agent.queue.items[0]
        assert queued.kind == "goal_question"
        assert queued.text == goal_rules.question_text(1, 1, True)
        assert agent._goal_question_pending is True
        assert agent.status == AgentStatus.WAITING

    @pytest.mark.asyncio
    async def test_the_question_turn_hands_over_to_the_answer(self):
        """The question turn ends -> its answer is in history, unparsed. Without
        this transition the hook sees "question in flight" forever and the
        window stops silently - the failure the window exists to prevent.

        Driven end to end: a turn ends and the hook asks; the question turn
        ends and the hook answers.
        """
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.queue.add("hello")
        await agent._process_item(agent.queue.get_next())

        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].kind == "goal_question"
        assert agent._goal_question_pending is True

        agent._llm_turn = AsyncMock(
            side_effect=lambda **kw: agent.messages.append(
                {"role": "assistant", "content": '{"can_progress": true, "next": "go"}'}
            )
        )
        await agent._process_item(agent.queue.get_next())

        assert agent._goal_question_pending is False
        assert agent._goal_answer_pending is False
        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].kind == "goal"
        assert agent.queue.items[0].text == f"Goal: ship it\n{GOAL_ITEM.format(step='go')}"

    @pytest.mark.asyncio
    async def test_the_goal_item_runs_as_a_prompt(self):
        """`kind="goal"` has no branch in process_item's elif chain, so it falls
        through to normal prompt processing - deliberately. If a branch were
        ever added for it, the item would stop being a turn."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.queue.add(f"Goal: ship it\n{GOAL_ITEM.format(step='run the tests')}", kind="goal")
        item = agent.queue.get_next()

        await agent._process_item(item)

        assert agent._llm_turn.await_count == 1
        assert agent.messages[-1] == {
            "role": "user",
            "content": f"Goal: ship it\n{GOAL_ITEM.format(step='run the tests')}",
        }
        assert is_turn_start(agent.messages[-1]) is True

    @pytest.mark.asyncio
    async def test_answer_false_stops_and_keeps_the_pair(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(
            agent,
            '{"can_progress": false, "waiting_on_user": ["the API key"]}',
        )
        before = [dict(m) for m in agent.messages]

        await agent._goal_check()

        assert agent.queue.pending_count == 0
        assert agent.messages == before  # the pair stays in history
        notices = [
            c.args[1]["message"]
            for c in agent.emit.await_args_list
            if c.args[0] == AgentEvent.NOTIFICATION
        ]
        assert any("no progress without you" in n for n in notices)
        assert any("the API key" in n for n in notices)

    @pytest.mark.asyncio
    async def test_answer_true_deletes_the_pair_and_enqueues_the_goal(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')

        await agent._goal_check()

        assert agent.queue.pending_count == 1
        queued = agent.queue.items[0]
        assert queued.kind == "goal"
        assert queued.text == f"Goal: ship it\n{GOAL_ITEM.format(step='run the tests')}"
        assert agent._goal_answer_pending is False

    @pytest.mark.asyncio
    async def test_unparseable_answer_continues_with_the_record_fallback(self):
        """No retry: continuing is the safe failure."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, "I think I can keep going, boss.")

        await agent._goal_check()

        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].text == (
            f"Goal: ship it\n{GOAL_ITEM_RECORD_FALLBACK}"
        )

    @pytest.mark.asyncio
    async def test_tool_calls_in_the_answer_keep_the_pair(self):
        """A tool call means the model kept working instead of answering; the
        pair is part of that work."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(
            agent,
            '{"can_progress": true, "next": "run the tests"}',
            tool_calls=[{"id": "1", "function": {"name": "read_file"}}],
        )
        before = [dict(m) for m in agent.messages]

        await agent._goal_check()

        assert agent.messages == before
        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].text == (
            f"Goal: ship it\n{GOAL_ITEM_RECORD_FALLBACK}"
        )

    @pytest.mark.asyncio
    async def test_no_record_no_text_no_next_uses_the_continue_hint(self):
        """Never enqueue an empty item."""
        agent = make_agent(record="", mtime=0.0)
        agent.set_goal(3600)
        question_turn(agent, "no json here")

        await agent._goal_check()

        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].text == AUTO_CONTEXT_CONTINUE_HINT

    @pytest.mark.asyncio
    async def test_a_question_already_in_flight_is_not_stacked(self):
        """Stacking a second question would ask the same thing twice and lose
        the first answer. The gate is the queue: the hook only runs when it is
        empty, so a queued question suppresses it."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.queue.add("hello")
        agent.queue.add("a queued question", kind="goal_question")
        agent._goal_question_pending = True
        item = agent.queue.get_next()

        await agent._process_item(item)

        # The queued question is untouched and no second one was added.
        assert agent.queue.pending_count == 1
        assert agent.queue.items[0].kind == "goal_question"

    @pytest.mark.asyncio
    async def test_off_while_the_answer_is_in_flight_does_not_enqueue(self):
        """An answer that arrives after the user cancelled cannot start a goal
        turn. Both guards are needed: the flags are persisted, so a restart
        could otherwise resurrect a cancelled question."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        agent._goal_question_pending = False
        agent._goal_answer_pending = True
        execute_grind(agent, "off")

        # The handler was already entered when the cancel landed.
        await agent._goal_handle_answer()

        assert agent.queue.pending_count == 0
        assert agent._goal_question_pending is False
        assert agent._goal_answer_pending is False

    @pytest.mark.asyncio
    async def test_no_question_after_the_deadline(self):
        """The gate is `goal_active`: an expired window winds down to a natural
        idle, with no question and no goal item."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        agent.queue.add("hello")
        item = agent.queue.get_next()

        await agent._process_item(item)

        assert agent.queue.pending_count == 0
        assert agent.status == AgentStatus.IDLE


class TestPairDeletion:
    @pytest.mark.asyncio
    async def test_history_is_byte_identical_to_the_pre_question_state(self):
        """The model must not see the exchange it answered: the pair is deleted,
        and nothing else moves."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        pre_question = [
            {"role": "user", "content": "earlier work"},
            {"role": "assistant", "content": "ok"},
        ]
        agent.messages = [dict(m) for m in pre_question]
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        agent.messages = agent.messages[:2] + agent.messages[2:]

        await agent._goal_check()

        assert agent.messages == pre_question

    @pytest.mark.asyncio
    async def test_deletion_matches_the_flag_not_the_text(self):
        """The question is rebuilt from the record each time, so its wording can
        differ from the one in history."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.messages = [
            {"role": "user", "content": "earlier work"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "a DIFFERENT question text",
             "goal_question": True},
            {"role": "assistant", "content": '{"can_progress": true, "next": "go"}'},
        ]
        agent._goal_question_pending = True

        await agent._goal_check()

        assert agent.messages == [
            {"role": "user", "content": "earlier work"},
            {"role": "assistant", "content": "ok"},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# The chain restart
# ─────────────────────────────────────────────────────────────────────────────


class TestChainRestartMessage:
    def test_goal_text_plus_the_record_pointer(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        assert agent._goal_restart_message() == (
            f"Goal: ship it\n{AUTO_CONTEXT_RECORD_FRESH_HINT}"
        )

    def test_no_goal_text_gives_the_pointer_alone(self):
        agent = make_agent()
        agent.set_goal(3600)
        assert agent._goal_restart_message() == AUTO_CONTEXT_RECORD_FRESH_HINT

    def test_no_goal_gives_the_pointer_alone(self):
        """Regression: the no-goal path is unchanged."""
        agent = make_agent()
        assert agent._goal_restart_message() == AUTO_CONTEXT_RECORD_FRESH_HINT

    def test_no_record_gives_the_continue_hint(self):
        agent = make_agent(record="", mtime=0.0)
        agent.set_goal(3600, "ship it")
        assert agent._goal_restart_message() == AUTO_CONTEXT_CONTINUE_HINT

    def test_expired_goal_gives_the_pointer_alone(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        assert agent._goal_restart_message() == AUTO_CONTEXT_RECORD_FRESH_HINT

    def test_the_restart_does_not_quote_a_user_message(self):
        """The restart message no longer quotes the seed, so a resumed chain
        cannot re-inject its own seed as "the user's last message"."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.messages = [{"role": "user", "content": "SEED TEXT"}]
        assert "SEED TEXT" not in agent._goal_restart_message()


class TestChainBudget:
    @staticmethod
    def _arm_restart(agent):
        agent._auto_context_triggered = True
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))

    @pytest.mark.asyncio
    async def test_goal_neither_blocks_a_restart_nor_spends_the_budget(self):
        """Skipping only the comparison would saturate the counter at the cap
        and kill the budget for the rest of the session."""
        agent = make_agent()
        agent.auto_context_chain = 0
        agent.set_goal(3600, "ship it")
        agent._auto_context_chain_used = 0
        self._arm_restart(agent)

        await agent._run_auto_context_post_turn(restart=True)

        assert agent._llm_turn.await_count == 1  # the restart ran
        assert agent._auto_context_chain_used == 0  # and cost nothing

    @pytest.mark.asyncio
    async def test_goal_restarts_even_with_the_counter_at_the_cap(self):
        agent = make_agent()
        agent.auto_context_chain = 1
        agent.set_goal(3600, "ship it")
        agent._auto_context_chain_used = 1
        self._arm_restart(agent)

        await agent._run_auto_context_post_turn(restart=True)

        assert agent._llm_turn.await_count == 1
        assert agent._auto_context_chain_used == 1

    @pytest.mark.asyncio
    async def test_the_budget_binds_again_after_the_deadline(self):
        """The wind-down needs a budget: _expire_goal zeroes the counter once,
        then it accumulates from zero."""
        agent = make_agent()
        agent.auto_context_chain = 1
        agent.set_goal(3600, "ship it")
        agent._goal_deadline = time.time() - 1
        agent._auto_context_chain_used = 1  # spent during the window
        self._arm_restart(agent)

        await agent._run_auto_context_post_turn(restart=True)

        # Expiry zeroed the counter, so this restart is allowed - and counted.
        assert agent._llm_turn.await_count == 1
        assert agent._auto_context_chain_used == 1

    @pytest.mark.asyncio
    async def test_no_goal_budget_still_binds(self):
        """Regression: the no-goal path is unchanged."""
        agent = make_agent()
        agent.auto_context_chain = 1
        agent._auto_context_chain_used = 1
        self._arm_restart(agent)

        await agent._run_auto_context_post_turn(restart=True)

        assert agent._llm_turn.await_count == 0
        notices = [
            c.args[1]["message"]
            for c in agent.emit.await_args_list
            if c.args[0] == AgentEvent.NOTIFICATION
        ]
        assert any("idle, over to you" in n for n in notices)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_round_trip_keeps_the_remaining_time(self, tmp_path):
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        path = tmp_path / "session.ctx"
        save_context(agent, str(path))

        other = make_agent()
        ok, _, _ = load_context(other, str(path))

        assert ok is True
        assert other._goal_text == "ship it"
        assert other.goal_active is True
        assert other.goal_window == pytest.approx(4 * 3600, abs=2)

    def test_load_clears_both_question_flags(self, tmp_path):
        """A question in flight is not resumable: waiting forever is a deadlock
        - the hook sees the flag, does nothing, and the window stops silently."""
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        agent._goal_answer_pending = True
        path = tmp_path / "session.ctx"
        save_context(agent, str(path))

        other = make_agent()
        load_context(other, str(path))

        assert other._goal_question_pending is False
        assert other._goal_answer_pending is False

    def test_load_keeps_the_pair_in_history(self, tmp_path):
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        path = tmp_path / "session.ctx"
        save_context(agent, str(path))

        other = make_agent()
        load_context(other, str(path))

        contents = [m.get("content") for m in other.messages]
        assert goal_rules.question_text(1, 1, True) in contents

    def test_a_corrupt_deadline_does_not_fail_the_load(self, tmp_path):
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        path = tmp_path / "session.ctx"
        save_context(agent, str(path))
        text = path.read_text(encoding="utf-8").replace(
            '"goal_deadline":', '"goal_deadline": "not-a-number", "_old":'
        )
        path.write_text(text, encoding="utf-8")

        other = make_agent()
        ok, _, _ = load_context(other, str(path))

        assert ok is True
        assert other.goal_active is False


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


class _Bar:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)


def make_tui(**kwargs):
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


class TestStatusBar:
    """The dur: segment gains a denominator while a window is armed, and the
    numerator keeps the task clock's semantics (time since the last user
    message, frozen at idle)."""

    @staticmethod
    def _right(tui):
        tui._status_left = _Bar()
        tui._status_right = _Bar()
        tui.update_status()
        return tui._status_right.values[-1]

    def test_armed_with_no_task_yet(self):
        tui = make_tui()
        tui.agent.set_goal(4 * 3600, "ship it")
        assert "dur: 0:00/4:00" in self._right(tui)

    def test_armed_and_running(self):
        tui = make_tui()
        tui.agent.set_goal(4 * 3600, "ship it")
        tui.agent.task_clock.start()
        tui.agent.task_clock._bank = 91.0
        assert "dur: 0:01/4:00" in self._right(tui)

    def test_armed_and_frozen(self):
        tui = make_tui()
        tui.agent.set_goal(4 * 3600, "ship it")
        tui.agent.task_clock.start()
        tui.agent.task_clock._bank = 3.5 * 3600
        tui.agent.task_clock.stop()
        assert "dur: 3:30/4:00" in self._right(tui)

    def test_expired_drops_the_denominator(self):
        tui = make_tui()
        tui.agent.set_goal(4 * 3600, "ship it")
        tui.agent.task_clock.start()
        tui.agent.task_clock._bank = 14 * 60
        tui.agent._goal_deadline = time.time() - 1
        right = self._right(tui)
        assert "dur: 0:14" in right
        assert "/" not in right.split("dur: ")[1].split(" |")[0]

    def test_no_goal_and_no_task_shows_nothing(self):
        tui = make_tui()
        assert "dur:" not in self._right(tui)

    def test_status_line_reports_the_window(self):
        from agent13.commands import execute_status

        tui = make_tui()
        tui.agent.set_goal(4 * 3600, "ship it")
        text = execute_status(tui.agent).message
        assert "Grind: ship it" in text
        assert "4:00" in text

    def test_status_line_is_absent_without_a_window(self):
        from agent13.commands import execute_status

        tui = make_tui()
        assert "Grind:" not in execute_status(tui.agent).message

    def test_status_line_says_so_when_there_is_no_goal_text(self):
        from agent13.commands import execute_status

        tui = make_tui()
        tui.agent.set_goal(4 * 3600)
        assert "no goal text" in execute_status(tui.agent).message


# ─────────────────────────────────────────────────────────────────────────────
# The model never sees the budget
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetIsUserOnly:
    def test_no_injected_message_carries_the_remaining_time(self):
        """A visible budget invites "I'm running low, let me consolidate and
        hand back" - the failure the window exists to fix."""
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        restart = agent._goal_restart_message()
        question = goal_rules.question_text(1, 1, True)
        item = goal_rules.goal_item("ship it", "run the tests", True)
        for text in (restart, question, item):
            assert "remaining" not in text.lower()
            assert "3:5" not in text
            assert "4:00" not in text


# ─────────────────────────────────────────────────────────────────────────────
# The debug log - the rollout's only evidence
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def captured_goal_events():
    """Collect the goal events; every one of them is silent unless debug is on."""
    events = []
    with (
        patch("agent13.core.is_debug_enabled", return_value=True),
        patch(
            "agent13.core.log_tps_event",
            side_effect=lambda name, data: events.append((name, data)),
        ),
    ):
        yield events


def only(events, name):
    return [data for event_name, data in events if event_name == name]


class TestAskLogging:
    @pytest.mark.asyncio
    async def test_the_ask_is_logged_and_enqueued(self):
        """One record of what was asked and why, and the question always goes
        out: a goal set by /grind IS the intent to grind, so there is no
        log-only mode to be silently in."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        with captured_goal_events() as events:
            await agent._goal_ask()

        asks = only(events, "goal_ask")
        assert len(asks) == 1
        assert asks[0]["unblocked"] == 1
        assert asks[0]["blocked"] == 1
        assert asks[0]["has_record"] is True
        assert asks[0]["would_continue"] is True
        assert asks[0]["preamble"] == "unblocked"
        assert asks[0]["question"] == goal_rules.question_text(1, 1, True)
        assert asks[0]["goal_text"] == "ship it"
        assert agent.queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_all_blocked_preamble_is_named(self):
        agent = make_agent(record=ALL_BLOCKED_RECORD)
        agent.set_goal(3600)
        with captured_goal_events() as events:
            await agent._goal_ask()

        asks = only(events, "goal_ask")
        assert asks[0]["preamble"] == "all_blocked"
        assert asks[0]["unblocked"] == 0
        assert asks[0]["would_continue"] is False

    @pytest.mark.asyncio
    async def test_no_record_is_named_plain_and_would_continue(self):
        agent = make_agent(record="", mtime=0.0)
        agent.set_goal(3600)
        with captured_goal_events() as events:
            await agent._goal_ask()

        asks = only(events, "goal_ask")
        assert asks[0]["preamble"] == "plain"
        assert asks[0]["has_record"] is False
        assert asks[0]["would_continue"] is True

    @pytest.mark.asyncio
    async def test_nothing_is_logged_without_debug(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        with patch("agent13.core.log_tps_event") as log:
            await agent._goal_ask()
        log.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_ask_logs_the_entry_the_for_user_list_came_from(self):
        """The list is carried from the newest entry WITH a block, so it can
        be hours behind the last entry. Staleness must be a line in the log,
        not an invisible assumption."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.durable_records.for_user_state = MagicMock(
            return_value=(["a", "b"], "## 2026-11-07 23:25 - age gate")
        )
        with captured_goal_events() as events:
            await agent._goal_ask()

        ask = only(events, "goal_ask")[0]
        assert ask["for_user"] == ["a", "b"]
        assert ask["for_user_source"] == "## 2026-11-07 23:25 - age gate"
        assert isinstance(ask["for_user_source_age_s"], float)

    @pytest.mark.asyncio
    async def test_an_unparseable_source_heading_logs_a_null_age(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        agent.durable_records.for_user_state = MagicMock(
            return_value=(["a"], "## no timestamp")
        )
        with captured_goal_events() as events:
            await agent._goal_ask()

        ask = only(events, "goal_ask")[0]
        assert ask["for_user_source_age_s"] is None


class TestAnswerLogging:
    @pytest.mark.asyncio
    async def test_continue_records_the_step_and_the_item(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")
        assert len(idle) == 1
        assert idle[0]["decision"] == "continue"
        assert idle[0]["parse_ok"] is True
        assert idle[0]["carried_tools"] is False
        assert idle[0]["next"] == "run the tests"
        assert idle[0]["source"] == "step"
        assert idle[0]["item"] == f"Goal: ship it\n{GOAL_ITEM.format(step='run the tests')}"

    @pytest.mark.asyncio
    async def test_unparseable_is_distinguishable_from_no_step(self):
        """Three shapes used to collapse into "next is empty"; they need
        different fixes, so they are logged differently."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, "I think I can keep going.")
        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")[0]
        assert idle["parse_ok"] is False
        assert idle["source"] == "record"

    @pytest.mark.asyncio
    async def test_parsed_but_no_step_is_parse_ok(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true}')
        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")[0]
        assert idle["parse_ok"] is True
        assert idle["next"] == ""
        assert idle["source"] == "record"

    @pytest.mark.asyncio
    async def test_carried_tools_are_logged(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(
            agent,
            '{"can_progress": true, "next": "run the tests"}',
            tool_calls=[{"id": "1", "function": {"name": "read_file"}}],
        )
        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")[0]
        assert idle["carried_tools"] is True
        assert idle["next"] == ""

    @pytest.mark.asyncio
    async def test_stop_records_the_waiting_list(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(
            agent, '{"can_progress": false, "waiting_on_user": ["the API key"]}'
        )
        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")[0]
        assert idle["decision"] == "stop"
        assert idle["waiting_on_user"] == ["the API key"]
        assert idle["answer"].startswith('{"can_progress": false')

    @pytest.mark.asyncio
    async def test_cancelled_window_is_logged(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        agent._goal_question_pending = False
        agent._goal_answer_pending = True
        execute_grind(agent, "off")
        with captured_goal_events() as events:
            await agent._goal_handle_answer()

        idle = only(events, "goal_idle")[0]
        assert idle["decision"] == "cancelled"

    @pytest.mark.asyncio
    async def test_the_question_turns_tokens_are_attributed(self):
        """The rollout has to show what the loop COSTS, not only that it fires:
        a full round-trip at ~300k context at every turn end."""
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        question_turn(agent, '{"can_progress": true, "next": "run the tests"}')
        agent._goal_question_tokens = {"prompt": 301_000, "completion": 42}

        with captured_goal_events() as events:
            await agent._goal_check()

        idle = only(events, "goal_idle")[0]
        assert idle["question_prompt_tokens"] == 301_000
        assert idle["question_completion_tokens"] == 42


class TestExpiryLogging:
    def test_expiry_records_the_budget_handover(self):
        """The wind-down boundary: the one place the chain budget is handed
        back, and invisible in the transcript."""
        agent = make_agent()
        agent.set_goal(4 * 3600, "ship it")
        # A 9 s window that closed 1 s ago: expired, and the window length
        # still reads forwards.
        agent._goal_set_at = time.time() - 10
        agent._goal_deadline = time.time() - 1
        agent._auto_context_chain_used = 2
        agent._goal_question_pending = True

        with captured_goal_events() as events:
            agent._expire_goal()

        expired = only(events, "goal_expired")
        assert len(expired) == 1
        assert expired[0]["chain_used_before"] == 2
        assert expired[0]["chain_used_after"] == 0
        assert expired[0]["had_goal_text"] is True
        assert expired[0]["question_was_pending"] is True
        assert expired[0]["window_s"] == pytest.approx(9.0, abs=0.2)

    def test_a_live_window_logs_nothing(self):
        agent = make_agent()
        agent.set_goal(3600, "ship it")
        with captured_goal_events() as events:
            agent._expire_goal()
        assert only(events, "goal_expired") == []


class TestCommandLogging:
    @staticmethod
    def _capture():
        events = []
        return events, (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        )

    def test_arm_records_the_window_and_the_text(self):
        agent = make_agent()
        events, patches = self._capture()
        with patches[0], patches[1]:
            execute_grind(agent, "2h30m ship the cockpit")

        commands = only(events, "goal_command")
        assert len(commands) == 1
        assert commands[0]["action"] == "arm"
        assert commands[0]["raw"] == "2h30m ship the cockpit"
        assert commands[0]["goal_text"] == "ship the cockpit"
        assert commands[0]["goal_active"] is True
        assert commands[0]["goal_window_s"] == pytest.approx(9000, abs=1)

    def test_a_rejected_duration_is_recorded(self):
        """"I typed /grind 90s and nothing happened" is the question this
        answers."""
        agent = make_agent()
        events, patches = self._capture()
        with patches[0], patches[1]:
            execute_grind(agent, "90s")

        commands = only(events, "goal_command")
        assert commands[0]["action"] == "reject"
        assert commands[0]["reason"] == "no duration"
        assert commands[0]["goal_active"] is False

    def test_off_is_recorded_as_stop_with_the_text_it_kept(self):
        # The action names follow the COMMAND, not the shared effect: after the
        # fact the log is the only place off and clear can be told apart.
        agent = make_agent()
        execute_grind(agent, "2h ship it")
        events, patches = self._capture()
        with patches[0], patches[1]:
            execute_grind(agent, "off")

        commands = only(events, "goal_command")
        assert commands[0]["action"] == "stop"
        assert commands[0]["goal_active"] is False
        assert commands[0]["goal_text"] == "ship it"
        assert commands[0]["kept_text"] == "ship it"

    def test_clear_is_recorded_as_clear_with_the_text_it_dropped(self):
        agent = make_agent()
        execute_grind(agent, "2h ship it")
        events, patches = self._capture()
        with patches[0], patches[1]:
            execute_grind(agent, "clear")

        commands = only(events, "goal_command")
        assert commands[0]["action"] == "clear"
        assert commands[0]["goal_active"] is False
        assert commands[0]["goal_text"] == ""
        assert commands[0]["cleared_text"] == "ship it"

    def test_show_is_recorded(self):
        agent = make_agent()
        execute_grind(agent, "2h ship it")
        events, patches = self._capture()
        with patches[0], patches[1]:
            execute_grind(agent, "")

        commands = only(events, "goal_command")
        assert commands[0]["action"] == "show"
        assert commands[0]["goal_active"] is True
