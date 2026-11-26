"""The model's clock: a frozen start stamp, refreshed by injected messages.

The system prompt says ``Session started <date> <time>`` and is FROZEN for the
session - rewriting it mid-session would invalidate the KV-cache prefix. So
"now" reaches the model a second way: an injected assistant+user pair, appended
at a safe boundary.

Three triggers, one predicate (``Agent._datetime_nudge_due``):

- the day rolled over (fires regardless of the interval);
- ``datetime_nudge_seconds`` elapsed during a turn;
- the same interval elapsed while the session sat idle, checked at turn start.

The pair is flagged ``injected``, NOT ``interrupt``: a clock refresh is not an
interruption, so it must not emit INTERRUPT_INJECTED, flip the status bar, or
write the history marker the turn-boundary logic reads.
"""

import datetime
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.llm import build_messages_with_system

FIXED_NOW = datetime.datetime(2026, 11, 1, 14, 3)


@contextmanager
def frozen_now(value):
    """Patch ``datetime.datetime.now()`` inside core to a fixed value."""

    class _Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return value

    with patch("agent13.core.datetime.datetime", _Frozen):
        yield


def make_agent(**kwargs):
    agent = Agent(client=MagicMock(), model="test-model", **kwargs)
    agent._running = True
    return agent


def stamp(dt):
    return dt.isoformat(timespec="seconds")


class TestNudgeDue:
    """The predicate. Pure: no turn, no messages."""

    def test_not_due_immediately_after_the_session_starts(self):
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW)
            assert agent._datetime_nudge_due() is False

    def test_not_due_before_the_interval(self):
        agent = make_agent(datetime_nudge_seconds=3600)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(minutes=59))
            assert agent._datetime_nudge_due() is False

    def test_due_at_the_interval(self):
        """Measured from the LAST nudge, so the boundary itself counts."""
        agent = make_agent(datetime_nudge_seconds=3600)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(hours=1))
            assert agent._datetime_nudge_due() is True

    def test_due_again_after_each_interval(self):
        """A three-hour turn with a one-hour interval refreshes three times."""
        agent = make_agent(datetime_nudge_seconds=3600)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(hours=3))
            assert agent._datetime_nudge_due() is True
            agent._inject_datetime_nudge()
            # The injection reset the clock, so it is not due again at once.
            assert agent._datetime_nudge_due() is False

    def test_a_day_change_fires_below_the_interval(self):
        """Midnight is special: it fires even when the interval has not elapsed.

        A ten-year interval isolates the rule - only the date can trigger it.
        """
        agent = make_agent(datetime_nudge_seconds=315_360_000)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(days=1))
            assert agent._datetime_nudge_due() is True

    def test_a_backwards_clock_is_not_due(self):
        """NTP can move the wall clock back; a negative interval is not a nudge."""
        agent = make_agent(datetime_nudge_seconds=3600)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW + datetime.timedelta(minutes=30))
            assert agent._datetime_nudge_due() is False

    def test_zero_disables(self):
        agent = make_agent(datetime_nudge_seconds=0)
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(days=30))
            assert agent._datetime_nudge_due() is False

    def test_an_unparseable_stamp_is_not_due(self):
        """A corrupt save must not crash the turn loop."""
        agent = make_agent()
        agent._last_datetime_nudge_at = "not-a-timestamp"
        with frozen_now(FIXED_NOW):
            assert agent._datetime_nudge_due() is False


class TestInjection:
    """The pair itself: shape, flags, and what it does to the clock."""

    def test_shape_is_assistant_then_user(self):
        """The conversation must not end on an assistant turn.

        Providers read a trailing assistant message as a prefill to continue,
        and some reject it. The fake assistant goes FIRST, exactly as the
        ``!!`` steer does at the same boundary.
        """
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._inject_datetime_nudge()

        assert len(agent.messages) == 2
        first, second = agent.messages
        assert first["role"] == "assistant"
        assert second["role"] == "user"
        assert second["content"] == "The current datetime is: 2026-11-01 14:03"

    def test_both_messages_are_flagged_injected(self):
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._inject_datetime_nudge()
        assert all(m.get("injected") for m in agent.messages)

    def test_it_is_not_an_interrupt(self):
        """``interrupt`` would emit INTERRUPT_INJECTED and flip the status bar."""
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._inject_datetime_nudge()
        assert not any(m.get("interrupt") for m in agent.messages)

    def test_it_advances_the_clock(self):
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._inject_datetime_nudge()
        assert agent._last_datetime_nudge_at == stamp(FIXED_NOW)

    def test_the_flags_never_reach_the_wire(self):
        """``injected`` is a local key; the API must not see it."""
        agent = make_agent()
        with frozen_now(FIXED_NOW):
            agent._inject_datetime_nudge()
        api = build_messages_with_system(agent.messages, "BODY")
        for msg in api:
            assert "injected" not in msg


class TestTurnStart:
    """Condition 3: the session sat idle, then the user speaks."""

    @pytest.mark.asyncio
    async def test_the_nudge_precedes_the_user_message(self):
        agent = make_agent(datetime_nudge_seconds=3600)
        agent.queue.add("what is the status?")
        item = agent.queue.get_next()

        async def noop(_polite_acquired=False):
            return None

        agent._llm_turn = noop
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(hours=2))
            await agent._process_item(item)

        contents = [m.get("content") for m in agent.messages]
        nudge_at = next(
            i for i, c in enumerate(contents) if c and "The current datetime is:" in c
        )
        user_at = contents.index("what is the status?")
        assert nudge_at < user_at, "the model must read the time before the message"

    @pytest.mark.asyncio
    async def test_the_user_message_is_left_exactly_as_typed(self):
        """The time is a separate message, so item.text stays clean.

        That is what keeps the TUI display, the saved session and the shared
        history store free of the timestamp.
        """
        agent = make_agent(datetime_nudge_seconds=3600)
        agent.queue.add("what is the status?")
        item = agent.queue.get_next()

        async def noop(_polite_acquired=False):
            return None

        agent._llm_turn = noop
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW - datetime.timedelta(hours=2))
            await agent._process_item(item)

        assert {"role": "user", "content": "what is the status?"} in agent.messages

    @pytest.mark.asyncio
    async def test_no_nudge_when_the_clock_is_fresh(self):
        agent = make_agent(datetime_nudge_seconds=3600)
        agent.queue.add("hello")
        item = agent.queue.get_next()

        async def noop(_polite_acquired=False):
            return None

        agent._llm_turn = noop
        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW)
            await agent._process_item(item)

        assert not any(
            "The current datetime is:" in (m.get("content") or "")
            for m in agent.messages
        )


def _tool_call(call_id, name, args):
    return {"id": call_id, "name": name, "arguments": json.dumps(args)}


class TestMidTurn:
    """Conditions 1 and 2: a long turn refreshes at the tool boundary."""

    @pytest.mark.asyncio
    async def test_the_nudge_lands_at_the_tool_boundary(self):
        agent = make_agent(datetime_nudge_seconds=3600)

        async def mock_execute_tool(name, arguments):
            return json.dumps({"result": 9})

        agent._execute_tool_async = mock_execute_tool
        agent.messages.append({"role": "user", "content": "square 3"})

        call_count = 0

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield (
                    "tool_calls_complete",
                    {"tool_calls": [_tool_call("tc1", "square_number", {"x": 3})]},
                )
            else:
                yield ("content", "9")

        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(
                FIXED_NOW - datetime.timedelta(hours=2)
            )
            with patch(
                "agent13.llm.stream_response_with_tools", side_effect=mock_stream
            ):
                await agent._llm_turn()

        nudges = [
            m
            for m in agent.messages
            if "The current datetime is:" in (m.get("content") or "")
        ]
        assert len(nudges) == 1
        assert nudges[0]["role"] == "user"
        assert nudges[0].get("injected") is True

    @pytest.mark.asyncio
    async def test_no_nudge_in_a_short_turn(self):
        agent = make_agent(datetime_nudge_seconds=3600)

        async def mock_execute_tool(name, arguments):
            return json.dumps({"result": 9})

        agent._execute_tool_async = mock_execute_tool
        agent.messages.append({"role": "user", "content": "square 3"})

        call_count = 0

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield (
                    "tool_calls_complete",
                    {"tool_calls": [_tool_call("tc1", "square_number", {"x": 3})]},
                )
            else:
                yield ("content", "9")

        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(FIXED_NOW)
            with patch(
                "agent13.llm.stream_response_with_tools", side_effect=mock_stream
            ):
                await agent._llm_turn()

        assert not any(
            "The current datetime is:" in (m.get("content") or "")
            for m in agent.messages
        )


class TestWrapUpCountsAsANudge:
    """The wrap-up prompt states the time, so it refreshes the clock.

    Without this the next tool boundary would inject the same information
    minutes later. It does NOT suppress the nudge for the rest of the turn: a
    wrap-up that runs past the interval still gets a refresh.
    """

    @pytest.mark.asyncio
    async def test_the_wrapup_refreshes_the_clock(self):
        agent = make_agent(datetime_nudge_seconds=3600)

        async def noop(_polite_acquired=False):
            return None

        agent._llm_turn = noop
        agent._capture_session_journal_path = MagicMock()
        agent._check_ticket_rerank = MagicMock()
        agent._save_auto_context_snapshot = MagicMock()
        agent._report_and_compact_triggered = True

        async def noop_check(_polite_acquired=False):
            return None

        agent._check_truth_file_size = noop_check

        with frozen_now(FIXED_NOW):
            agent._last_datetime_nudge_at = stamp(
                FIXED_NOW - datetime.timedelta(hours=5)
            )
            await agent._run_report_and_compact_turn()
            # The due-check reads the wall clock, so it must run under the
            # frozen clock too. Outside the freeze it compares FIXED_NOW
            # against the real time: it passed only while the suite happened
            # to run within an hour of 14:03 on 2026-11-01, and has been red
            # ever since (the date check then fires too).
            assert agent._last_datetime_nudge_at == stamp(FIXED_NOW)
            assert agent._datetime_nudge_due() is False

    @pytest.mark.asyncio
    async def test_the_wrapup_prompt_still_carries_the_time(self):
        agent = make_agent()
        seen = []

        async def capture(_polite_acquired=False):
            seen.append(list(agent.messages))

        agent._llm_turn = capture
        agent._capture_session_journal_path = MagicMock()
        agent._check_ticket_rerank = MagicMock()
        agent._save_auto_context_snapshot = MagicMock()
        agent._report_and_compact_triggered = True

        async def noop_check(_polite_acquired=False):
            return None

        agent._check_truth_file_size = noop_check

        with frozen_now(FIXED_NOW):
            await agent._run_report_and_compact_turn()

        wrapup = [
            m for m in seen[0] if m.get("report_and_compact") and m["role"] == "user"
        ]
        assert len(wrapup) == 1
        assert "Current date and time: 2026-11-01 14:03" in wrapup[0]["content"]


class TestSystemPrompt:
    """The start stamp is frozen for the session and shows the time."""

    def test_the_start_line_carries_the_time(self):
        api = build_messages_with_system(
            [{"role": "user", "content": "hi"}],
            "BODY",
            "2026-10-31",
            "2026-10-31T14:23:07.123456",
        )
        assert api[0]["content"].startswith("Session started 2026-10-31 14:23\n")

    def test_a_date_only_still_works(self):
        """The older call shape (no time) is unchanged."""
        api = build_messages_with_system(
            [{"role": "user", "content": "hi"}], "BODY", "2026-01-15"
        )
        assert api[0]["content"].startswith("Session started 2026-01-15\n")

    def test_an_unparseable_stamp_is_passed_through(self):
        api = build_messages_with_system(
            [{"role": "user", "content": "hi"}], "BODY", None, "not-a-date"
        )
        assert api[0]["content"].startswith("Session started not-a-date\n")

    def test_the_agent_passes_its_session_start(self):
        agent = make_agent()
        api = build_messages_with_system(
            [{"role": "user", "content": "hi"}],
            "BODY",
            agent.session_date,
            agent.session_started_at,
        )
        started = agent.session_started_at[:16].replace("T", " ")
        assert api[0]["content"].startswith(f"Session started {started}\n")
