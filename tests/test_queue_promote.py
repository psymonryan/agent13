#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""Unit tests for promoting an already-queued item to steer / priority.

A bare ``!`` or ``!!`` submitted on its own is a PROMOTION, not an empty
message: the user has already queued something and is now saying "run that
next" (``!``) or "steer it into the running turn" (``!!``).

Before this, a bare ``!!`` queued an EMPTY interrupt item, which the turn
loop then injected as a blank user turn - and the REPL silently dropped it,
so the two front-ends disagreed.
"""

import pytest

from agent13.commands import promote_latest
from agent13.core import Agent
from agent13.queue import AgentQueue


class _Agent:
    """Just the attribute ``promote_latest`` reads."""

    def __init__(self):
        self.queue = AgentQueue()


def _agent(*texts):
    a = _Agent()
    for t in texts:
        a.queue.add(t)
    return a


# ── latest_pending ────────────────────────────────────────────────────────


class TestLatestPending:
    def test_none_when_empty(self):
        assert _agent().queue.latest_pending() is None

    def test_returns_most_recent_by_id_not_list_position(self):
        """The queue reorders on insert, so ``items[-1]`` is the last NORMAL
        item - not the one the user just typed."""
        a = _agent("first", "second", "third")
        a.queue.set_interrupt(a.queue.items[-1].id, True)  # "third" -> front

        assert [i.text for i in a.queue.items] == ["third", "first", "second"]
        assert a.queue.items[-1].text == "second"  # what a naive impl returns
        assert a.queue.latest_pending().text == "third"


# ── set_interrupt ─────────────────────────────────────────────────────────


class TestSetInterrupt:
    def test_promotes_and_moves_to_front(self):
        a = _agent("first", "second")
        item = a.queue.items[-1]

        assert a.queue.set_interrupt(item.id, True) is True
        assert [i.text for i in a.queue.items] == ["second", "first"]
        assert a.queue.items[0].interrupt is True
        # Interrupt implies priority, matching add()
        assert a.queue.items[0].priority is True

    def test_preserves_id(self):
        a = _agent("first", "second")
        item_id = a.queue.items[-1].id

        a.queue.set_interrupt(item_id, True)
        assert a.queue.items[0].id == item_id

    def test_idempotent(self):
        a = _agent("first", "second")
        item_id = a.queue.items[-1].id

        a.queue.set_interrupt(item_id, True)
        order = [i.text for i in a.queue.items]
        a.queue.set_interrupt(item_id, True)
        assert [i.text for i in a.queue.items] == order

    def test_unknown_id_returns_false(self):
        assert _agent("first").queue.set_interrupt(999, True) is False

    def test_demote_keeps_priority(self):
        """Demoting drops the item to the priority group, not to normal."""
        a = _agent("first", "second")
        item_id = a.queue.items[-1].id

        a.queue.set_interrupt(item_id, True)
        a.queue.set_interrupt(item_id, False)
        assert a.queue.items[0].interrupt is False
        assert a.queue.items[0].priority is True

    def test_steers_keep_submission_order(self):
        a = _Agent()
        a.queue.add("one")
        a.queue.add("two")
        third = a.queue.add("three")
        a.queue.set_interrupt(third, True)
        fourth = a.queue.add("four")
        a.queue.set_interrupt(fourth, True)

        assert [i.text for i in a.queue.items] == ["three", "four", "one", "two"]


# ── promote_latest ────────────────────────────────────────────────────────


class TestPromoteLatest:
    def test_steer_names_the_item(self):
        a = _agent("first", "second")

        assert promote_latest(a, True) == '[steer] "second"'
        assert a.queue.items[0].text == "second"
        assert a.queue.items[0].interrupt is True

    def test_priority_names_the_item(self):
        a = _agent("first", "second")

        assert promote_latest(a, False) == '[priority] "second"'
        assert a.queue.items[0].text == "second"
        assert a.queue.items[0].interrupt is False
        assert a.queue.items[0].priority is True

    def test_nothing_queued(self):
        assert promote_latest(_agent(), True) == "[steer] nothing queued"
        assert promote_latest(_agent(), False) == "[priority] nothing queued"

    def test_already_steering(self):
        a = _agent("first")
        promote_latest(a, True)
        assert promote_latest(a, True) == "[steer] already steering"

    def test_already_priority(self):
        a = _agent("first")
        promote_latest(a, False)
        assert promote_latest(a, False) == "[priority] already priority"

    def test_collapses_whitespace(self):
        a = _agent("  multi\n  line   text ")
        assert promote_latest(a, True) == '[steer] "multi line text"'

    def test_promotes_the_most_recent_only(self):
        a = _agent("first", "second", "third")

        promote_latest(a, True)
        assert a.queue.items[0].text == "third"
        assert [i.text for i in a.queue.items[1:]] == ["first", "second"]


# ── add_message refuses an empty prompt ───────────────────────────────────


class _StubAgent:
    """The attributes ``Agent.add_message`` touches."""

    def __init__(self):
        self.queue = AgentQueue()

    async def emit(self, *args, **kwargs):
        pass

    async def _emit_queue_update(self):
        pass


class TestEmptyPromptRefused:
    @pytest.mark.asyncio
    async def test_empty_prompt_queues_nothing(self):
        stub = _StubAgent()

        assert await Agent.add_message(stub, "") == 0
        assert stub.queue.pending_count == 0

    @pytest.mark.asyncio
    async def test_whitespace_prompt_queues_nothing(self):
        stub = _StubAgent()

        assert await Agent.add_message(stub, "   \n\t ") == 0
        assert stub.queue.pending_count == 0

    @pytest.mark.asyncio
    async def test_empty_non_prompt_kind_still_queued(self):
        """The guard is scoped to prompts: command turns keep working."""
        stub = _StubAgent()

        assert await Agent.add_message(stub, "", kind="compact") > 0
        assert stub.queue.pending_count == 1

    @pytest.mark.asyncio
    async def test_real_prompt_still_queued(self):
        stub = _StubAgent()

        assert await Agent.add_message(stub, "do the thing") > 0
        assert stub.queue.pending_count == 1
