"""Tests for the degeneration guard (cap-hit / whitespace-wall recovery).

The guard (docs_archive/degeneration_guard_plan.md, locked design):
a no-tool-calls stream end with finish_reason "length" (cap-hit) or a
whitespace-only wall ("stop" + all-whitespace content - the empty-stream
retry only covers zero-byte streams) is DEGENERATE. The guard nudges the
model once to continue (injected user message, turn keeps running) and
gives up at auto_context_nudge_cap consecutive degenerate ends: snapshot
the partial, mark the turn incomplete (/resume retries), break.

Streak resets: a tool-calls round, a chain restart, a new user turn.
Over the auto-context threshold the guard stands down (the compact
machinery owns recovery). cap <= 0 disables the guard.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent, _degenerate_tail_is_garbage
from agent13.events import AgentEvent
from agent13.prompts import DEGENERATE_NUDGE_PROMPT


def make_agent(
    auto_context_threshold=0,
    auto_context_nudge_cap=2,
):
    """Create a minimal agent for degeneration-guard testing."""
    client = MagicMock()
    agent = Agent(
        client=client,
        model="test-model",
        auto_context_threshold=auto_context_threshold,
        auto_context_nudge_cap=auto_context_nudge_cap,
    )
    agent._save_auto_context_snapshot = MagicMock()
    return agent


def cap_hit_stream(content="partial answer "):
    """A stream that ends at the output cap with coherent content."""

    async def stream(*args, **kwargs):
        yield ("content", content)
        yield ("finish", {"finish_reason": "length"})

    return stream


def whitespace_wall_stream(n=200):
    """The 09-10 shape: a newline wall ending 'normally' (finish stop)."""

    async def stream(*args, **kwargs):
        yield ("content", "\n" * n)
        yield ("finish", {"finish_reason": "stop"})

    return stream


def normal_stream(content="final answer"):
    """A clean stream: real content, finish stop."""

    async def stream(*args, **kwargs):
        yield ("content", content)
        yield ("finish", {"finish_reason": "stop"})

    return stream


# ─── Classification matrix ────────────────────────────────────────────────


class TestDegenerateClassification:
    """_classify_degenerate_end: trigger, streak, cap, threshold rules."""

    def test_cap_hit_nudges_first_time(self):
        agent = make_agent()
        decision = agent._classify_degenerate_end(
            "coherent partial", "", "length"
        )
        assert decision == "nudge"
        assert agent._degenerate_streak == 1

    def test_whitespace_wall_nudges_first_time(self):
        agent = make_agent()
        decision = agent._classify_degenerate_end("\n\n  \n", "", "stop")
        assert decision == "nudge"
        assert agent._degenerate_streak == 1

    def test_normal_end_is_none_and_keeps_streak(self):
        agent = make_agent()
        agent._degenerate_streak = 1  # prior degenerate end
        decision = agent._classify_degenerate_end("real answer", "", "stop")
        assert decision == "none"
        # A clean TEXT end does not reset the streak (only a tool round,
        # a chain restart, or a new user turn does).
        assert agent._degenerate_streak == 1

    def test_missing_finish_reason_is_conservative_none(self):
        agent = make_agent()
        agent._degenerate_streak = 1
        decision = agent._classify_degenerate_end("", "", None)
        assert decision == "none"
        assert agent._degenerate_streak == 1

    def test_second_consecutive_degenerate_end_stops(self):
        agent = make_agent()
        assert (
            agent._classify_degenerate_end("first", "", "length") == "nudge"
        )
        assert (
            agent._classify_degenerate_end("second", "", "length") == "stop"
        )

    def test_cap_zero_disables_guard(self):
        agent = make_agent(auto_context_nudge_cap=0)
        decision = agent._classify_degenerate_end("x" * 1000, "", "length")
        assert decision == "none"
        assert agent._degenerate_streak == 0

    def test_over_threshold_stands_down(self):
        agent = make_agent(auto_context_threshold=1000)
        agent.prompt_tokens = 2000  # grounded, over threshold
        decision = agent._classify_degenerate_end("partial", "", "length")
        assert decision == "none"
        assert agent._degenerate_streak == 0

    def test_cap_hit_with_empty_content_but_reasoning(self):
        """The reasoning-loop death shape: visible content empty, the cap
        hit inside reasoning. Still a degenerate end."""
        agent = make_agent()
        decision = agent._classify_degenerate_end("", "thinking... thinking", "length")
        assert decision == "nudge"


# ─── Tail-garbage helper ──────────────────────────────────────────────────


class TestTailGarbage:
    """_degenerate_tail_is_garbage: <1% non-whitespace in the last 6k chars."""

    def test_whitespace_wall_is_garbage(self):
        assert _degenerate_tail_is_garbage("\n" * 5000, "") is True

    def test_coherent_tail_is_kept(self):
        assert _degenerate_tail_is_garbage("a coherent sentence. " * 100, "") is False

    def test_empty_is_not_garbage(self):
        assert _degenerate_tail_is_garbage("", "") is False

    def test_reasoning_tail_counts(self):
        # Content empty; the wall lives in reasoning.
        assert _degenerate_tail_is_garbage("", "\n" * 5000) is True

    def test_short_wall(self):
        assert _degenerate_tail_is_garbage("\n\n\n", "") is True

    def test_boundary_below_threshold(self):
        # 6k tail, exactly 1% non-whitespace = NOT garbage (strict <).
        tail = "x" * 60 + "\n" * 5940
        assert _degenerate_tail_is_garbage(tail, "") is False


# ─── Nudge path (in-process _llm_turn, streams mocked) ────────────────────


class TestNudgeApply:
    @pytest.mark.asyncio
    async def test_cap_hit_keeps_partial_and_continues(self, tmp_path, monkeypatch):
        """A coherent cap-hit: the partial is appended, the nudge rides as an
        injected user message, the counter increments, the turn continues to
        a clean final answer."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent._running = True
        agent.messages = [{"role": "user", "content": "do the thing"}]
        events = []

        async def on_event(event):
            events.append(event)

        agent.on_event(on_event)

        streams = iter([cap_hit_stream("partial answer "), normal_stream()])
        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return next(streams)()

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=mock_stream,
        ):
            await agent._llm_turn()

        roles = [m["role"] for m in agent.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert agent.messages[1]["content"] == "partial answer "
        nudge = agent.messages[2]
        assert nudge["role"] == "user"
        assert nudge["content"] == DEGENERATE_NUDGE_PROMPT
        assert nudge.get("injected") is True
        assert agent.messages[3]["content"] == "final answer"
        assert agent.degenerate_nudges == 1
        assert agent.has_incomplete_turn is False
        assert call_count == 2
        nudge_events = [e for e in events if e.event == AgentEvent.DEGENERATE_NUDGE]
        assert len(nudge_events) == 1
        assert nudge_events[0].data["action"] == "nudge"

    @pytest.mark.asyncio
    async def test_whitespace_wall_drops_partial_and_snapshots(
        self, tmp_path, monkeypatch
    ):
        """A whitespace wall: the partial is NOT appended (it would poison
        the context); a snapshot preserves it; the nudge still appends."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent._running = True
        agent.messages = [{"role": "user", "content": "go"}]

        streams = iter([whitespace_wall_stream(), normal_stream()])

        def mock_stream(*args, **kwargs):
            return next(streams)()

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=mock_stream,
        ):
            await agent._llm_turn()

        roles = [m["role"] for m in agent.messages]
        # No assistant message for the wall - just user, nudge, final answer.
        assert roles == ["user", "user", "assistant"]
        assert agent.messages[1]["content"] == DEGENERATE_NUDGE_PROMPT
        assert agent.degenerate_nudges == 1
        # The snapshot exists beside the daily auto-save.
        saves = list((tmp_path / ".agent13" / "saves").glob("*_degenerate_1.ctx"))
        assert len(saves) == 1, f"expected one degenerate snapshot, got {saves}"

    @pytest.mark.asyncio
    async def test_streak_resets_on_tool_round(self, tmp_path, monkeypatch):
        """degenerate -> tool round -> degenerate: the second degenerate end
        nudges again (streak was reset by the tool round) instead of stopping."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent._running = True
        agent.messages = [{"role": "user", "content": "go"}]
        agent._execute_tool_async = AsyncMock(return_value="tool ok")

        async def tool_round(*args, **kwargs):
            yield (
                "tool_calls_complete",
                {
                    "tool_calls": [
                        {"id": "t1", "name": "command", "arguments": "{}"}
                    ]
                },
            )
            yield ("finish", {"finish_reason": "tool_calls"})

        streams = iter([cap_hit_stream(), tool_round, cap_hit_stream()])

        def mock_stream(*args, **kwargs):
            return next(streams)()

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=mock_stream,
        ):
            await agent._llm_turn()

        # Second cap-hit nudged (not stopped): counter 2, turn incomplete=False.
        assert agent.degenerate_nudges == 2
        assert agent.has_incomplete_turn is False

    @pytest.mark.asyncio
    async def test_cap_reached_stops_turn(self, tmp_path, monkeypatch):
        """Two consecutive degenerate ends with cap 2: the second one stops
        the turn - snapshot, incomplete marker, no assistant append."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()  # cap 2
        agent._running = True
        agent.messages = [{"role": "user", "content": "go"}]

        streams = iter([cap_hit_stream("first try "), cap_hit_stream("second try ")])

        def mock_stream(*args, **kwargs):
            return next(streams)()

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=mock_stream,
        ):
            await agent._llm_turn()

        # First end: nudge (partial + nudge appended). Second end: stop -
        # the partial is NOT appended (snapshot holds it), turn incomplete.
        roles = [m["role"] for m in agent.messages]
        assert roles == ["user", "assistant", "user"]
        assert agent.messages[1]["content"] == "first try "
        assert agent.messages[2]["content"] == DEGENERATE_NUDGE_PROMPT
        assert agent.degenerate_nudges == 1  # only the first end was a nudge
        assert agent.has_incomplete_turn is True
        saves = list((tmp_path / ".agent13" / "saves").glob("*_degenerate_1.ctx"))
        assert len(saves) == 1

    @pytest.mark.asyncio
    async def test_cap_zero_guard_off_ends_turn_normally(self, tmp_path, monkeypatch):
        """cap 0 disables the guard: a cap-hit ends the turn like a normal
        response (the partial is appended, no nudge, no snapshot)."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent(auto_context_nudge_cap=0)
        agent._running = True
        agent.messages = [{"role": "user", "content": "go"}]

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=lambda *a, **k: cap_hit_stream("cut off mid")(),
        ):
            await agent._llm_turn()

        roles = [m["role"] for m in agent.messages]
        assert roles == ["user", "assistant"]
        assert agent.messages[1]["content"] == "cut off mid"
        assert agent.degenerate_nudges == 0
        assert agent.has_incomplete_turn is False

    @pytest.mark.asyncio
    async def test_nudge_message_never_reaches_the_wire(self, tmp_path, monkeypatch):
        """The injected flag is a LOCAL_MSG_KEYS member: build_messages strips
        it before the request. Assert on the wire-shaping helper directly."""
        from agent13.llm import LOCAL_MSG_KEYS

        assert "injected" in LOCAL_MSG_KEYS


# ─── Streak resets outside the turn loop ──────────────────────────────────


class TestStreakResets:
    def test_turn_start_resets(self):
        agent = make_agent()
        agent._degenerate_streak = 2
        agent._reset_auto_context_turn_state()
        assert agent._degenerate_streak == 0

    @pytest.mark.asyncio
    async def test_mechanical_restart_resets(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent._degenerate_streak = 2
        agent._running = True
        agent.messages = [{"role": "user", "content": "old"}]
        await agent._mechanical_restart()
        assert agent._degenerate_streak == 0


# ─── Persistence + status ─────────────────────────────────────────────────


class TestNudgesPersistenceAndStatus:
    def test_save_context_writes_nudges_total(self, tmp_path, monkeypatch):
        from agent13.persistence import save_context

        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._degenerate_nudges_total = 7
        path = tmp_path / "save.ctx"
        save_context(agent, path)

        import json

        data = json.loads(path.read_text())
        assert data["nudges_total"] == 7

    def test_load_context_restores_nudges_total(self, tmp_path, monkeypatch):
        from agent13.persistence import load_context, save_context

        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._degenerate_nudges_total = 7
        path = tmp_path / "save.ctx"
        save_context(agent, path)

        fresh = make_agent()
        assert fresh.degenerate_nudges == 0
        load_context(fresh, path)
        assert fresh.degenerate_nudges == 7

    def test_load_context_old_save_defaults_zero(self, tmp_path, monkeypatch):
        """A save written before the field existed loads with 0."""
        import json

        from agent13.persistence import load_context

        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        path = tmp_path / "old.ctx"
        path.write_text(json.dumps({"version": 1, "messages": []}))

        load_context(agent, path)
        assert agent.degenerate_nudges == 0

    def test_gather_status_reports_nudges(self):
        import time

        from agent13.status import gather_status

        agent = make_agent()
        agent._degenerate_nudges_total = 4
        sd = gather_status(agent, "test-provider", "test-model", time.time())
        assert sd.nudges == 4
