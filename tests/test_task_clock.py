"""TaskClock: time on the current task, excluding paused time.

Two layers:

- ``TestTaskClock`` - the state machine on its own (no agent, no UI).
- ``TestTaskClockWiring`` - the agent drives it: a new user turn starts it,
  the pause path freezes/thaws it, going idle freezes the total, and the
  auto-context wrap-up + chain restart does NOT reset it (the regression
  this exists for: ``dur:`` used to be a per-turn timer, so a wrap-up made
  the status bar drop from the task total to the last turn's duration).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent13.core import Agent, AgentStatus
from agent13.timing import TaskClock


def _make_agent(**kwargs):
    return Agent(client=MagicMock(), model="test-model", **kwargs)


class TestTaskClock:
    def test_new_clock_has_no_value(self):
        clock = TaskClock()
        assert clock.has_value is False
        assert clock.is_running is False
        assert clock.is_paused is False
        assert clock.elapsed() == 0.0

    def test_start_runs(self):
        clock = TaskClock()
        clock.start()
        assert clock.is_running is True
        assert clock.has_value is True
        assert clock.elapsed() >= 0.0

    def test_start_while_running_does_not_reset(self):
        """A queued message mid-task must not restart the clock."""
        clock = TaskClock()
        clock.start()
        clock._bank = 42.0  # simulate elapsed time without sleeping
        clock.start()
        assert clock.elapsed() >= 42.0

    def test_pause_freezes(self):
        clock = TaskClock()
        clock.start()
        clock._bank = 10.0
        clock._started_at = time.time()
        clock.pause()
        assert clock.is_paused is True
        assert clock.is_running is False
        frozen = clock.elapsed()
        time.sleep(0.05)
        assert clock.elapsed() == frozen  # the pause gap is not counted

    def test_resume_continues_from_frozen_value(self):
        clock = TaskClock()
        clock.start()
        clock._bank = 10.0
        clock.pause()
        time.sleep(0.05)  # paused time - must not appear in elapsed()
        clock.resume()
        assert clock.is_running is True
        assert clock.is_paused is False
        assert 10.0 <= clock.elapsed() < 10.5

    def test_stop_freezes_the_total(self):
        clock = TaskClock()
        clock.start()
        clock._bank = 90.0
        clock._started_at = time.time()
        clock.stop()
        assert clock.is_running is False
        assert clock.has_value is True
        assert clock.elapsed() >= 90.0
        time.sleep(0.02)
        assert clock.elapsed() >= 90.0  # frozen, not growing

    def test_start_after_stop_resets(self):
        """A new task starts from zero, not from the last task's total."""
        clock = TaskClock()
        clock.start()
        clock._bank = 90.0
        clock.stop()
        clock.start()
        assert clock.elapsed() < 1.0

    def test_stop_while_paused_keeps_total(self):
        clock = TaskClock()
        clock.start()
        clock._bank = 30.0
        clock.pause()
        clock.stop()
        assert clock.is_paused is False
        assert 30.0 <= clock.elapsed() < 31.0  # total kept, not zeroed

    def test_pause_when_idle_is_noop(self):
        clock = TaskClock()
        clock.pause()
        assert clock.is_paused is False
        assert clock.has_value is False

    def test_resume_when_not_paused_is_noop(self):
        clock = TaskClock()
        clock.resume()
        assert clock.is_running is False

    def test_reset_forgets_everything(self):
        clock = TaskClock()
        clock.start()
        clock._bank = 60.0
        clock.reset()
        assert clock.has_value is False
        assert clock.elapsed() == 0.0


class TestTaskClockWiring:
    """The agent drives the clock; front-ends only read it."""

    def test_agent_owns_a_clock(self):
        agent = _make_agent()
        assert isinstance(agent.task_clock, TaskClock)

    @pytest.mark.asyncio
    async def test_idle_freezes_the_total(self):
        agent = _make_agent()
        agent.task_clock.start()
        agent.task_clock._bank = 75.0
        await agent._set_status(AgentStatus.IDLE)
        assert agent.task_clock.is_running is False
        assert agent.task_clock.elapsed() >= 75.0

    @pytest.mark.asyncio
    async def test_pause_and_resume_freeze_and_thaw(self):
        """/pause freezes the clock; /resume continues, no jump."""
        agent = _make_agent()
        agent._running = True
        agent.task_clock.start()
        agent.task_clock._bank = 20.0

        agent.pause()
        waiter = asyncio.create_task(agent._wait_if_paused(resume_status=AgentStatus.WAITING))
        await asyncio.sleep(0)  # let it reach the pause branch
        assert agent.task_clock.is_paused is True
        frozen = agent.task_clock.elapsed()
        await asyncio.sleep(0.05)
        assert agent.task_clock.elapsed() == frozen

        agent.resume()
        await asyncio.wait_for(waiter, timeout=1.0)
        assert agent.task_clock.is_running is True
        assert 20.0 <= agent.task_clock.elapsed() < 20.5

    @pytest.mark.asyncio
    async def test_resume_with_nothing_to_continue_does_not_thaw(self):
        """Resuming with an empty queue ends the task: the total stays frozen."""
        agent = _make_agent()
        agent._running = True
        agent.task_clock.start()
        agent.task_clock._bank = 20.0
        agent.pause()
        waiter = asyncio.create_task(agent._wait_if_paused())
        await asyncio.sleep(0)
        agent.resume()
        await asyncio.wait_for(waiter, timeout=1.0)
        assert agent.task_clock.is_running is False
        assert agent.task_clock.elapsed() >= 20.0

    @pytest.mark.asyncio
    async def test_wrapup_and_chain_do_not_reset_the_clock(self):
        """The regression: a wrap-up + re-chain keeps the task total."""
        agent = _make_agent(
            auto_context_threshold=1000,
            auto_context_action="report_and_compact",
            auto_context_chain=2,
        )
        agent.task_clock.start()
        agent.task_clock._bank = 600.0  # ten minutes on task so far

        # Stub the turns and the record machinery: this test is about the
        # clock, not about what the wrap-up writes.
        agent._llm_turn = AsyncMock()
        agent._save_auto_context_snapshot = MagicMock()
        agent._check_truth_file_size = AsyncMock()
        agent.durable_records.restart_intro = MagicMock(return_value="INTRO")
        agent.seed_from_records = MagicMock(return_value=False)

        agent._report_and_compact_triggered = True
        agent._auto_context_triggered = True
        agent._status = "processing"

        await agent._run_auto_context_post_turn(restart=True)

        assert agent.auto_context_chain_used == 1  # the chain did restart
        assert agent.task_clock.is_running is True
        assert agent.task_clock.elapsed() >= 600.0

    @pytest.mark.asyncio
    async def test_new_user_turn_starts_a_fresh_task(self):
        """A turn starting while the clock is frozen resets it."""
        agent = _make_agent()
        agent.task_clock.start()
        agent.task_clock._bank = 600.0
        await agent._set_status(AgentStatus.IDLE)  # task over, clock frozen
        agent.task_clock.start()  # what _process_item does for a new turn
        assert agent.task_clock.elapsed() < 1.0

    def test_new_session_resets_the_clock(self):
        agent = _make_agent()
        agent.task_clock.start()
        agent.task_clock._bank = 600.0
        agent.reset_session_counters()
        assert agent.task_clock.has_value is False
