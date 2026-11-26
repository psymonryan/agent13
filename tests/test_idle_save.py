"""Tests for the idle auto-save ([auto_save] idle_seconds / --auto-save-idle).

The daily auto-save is written on exit. With idle_seconds > 0 it is ALSO
written once per idle period: after that many seconds with nothing to do,
and only when the history changed since the last idle save.

No LLM, no queue processing: these feed Agent's idle state directly.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from agent13.config import Config, ConfigFileError
from agent13.core import Agent, AgentStatus
from agent13.events import AgentEvent
from agent13.status import gather_status


def make_agent(idle_seconds=300, messages=None):
    """Minimal idle agent with one user message (nothing to call)."""
    agent = Agent(
        client=MagicMock(),
        model="test-model",
        auto_save_idle_seconds=idle_seconds,
        messages=(
            messages
            if messages is not None
            else [{"role": "user", "content": "hello"}]
        ),
    )
    agent._status = AgentStatus.IDLE
    return agent


def capture_notifications(agent):
    """Collect NOTIFICATION messages emitted by the agent."""
    seen = []

    async def handler(event):
        if event.event == AgentEvent.NOTIFICATION:
            seen.append(event.data.get("message", ""))

    agent.on_event(handler)
    return seen


async def make_due(agent):
    """Start the idle clock, then backdate it so the save is due now."""
    await agent._maybe_idle_save()  # first pass only starts the clock
    assert agent._idle_since is not None
    agent._idle_since -= agent.auto_save_idle_seconds + 1


@pytest.fixture
def saves_dir(tmp_path, monkeypatch):
    """Project-local saves dir in a temp cwd (never the real repo)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent13.config.get_config", lambda: Config())
    return tmp_path / ".agent13" / "saves"


# ─────────────────────────────── Config parsing ───────────────────────────────


class TestConfigParsing:
    def _config(self, tmp_path, body):
        path = tmp_path / "config.toml"
        path.write_text(body)
        return Config.from_file(path)

    def test_default_is_300(self, tmp_path):
        config = self._config(tmp_path, "")
        assert config.auto_save_idle_seconds == 300

    def test_parses_idle_seconds(self, tmp_path):
        config = self._config(tmp_path, "[auto_save]\nidle_seconds = 60\n")
        assert config.auto_save_idle_seconds == 60

    def test_zero_disables(self, tmp_path):
        config = self._config(tmp_path, "[auto_save]\nidle_seconds = 0\n")
        assert config.auto_save_idle_seconds == 0

    def test_rejects_non_integer(self, tmp_path):
        with pytest.raises(ConfigFileError, match="idle_seconds"):
            self._config(tmp_path, '[auto_save]\nidle_seconds = "5m"\n')

    def test_rejects_boolean(self, tmp_path):
        with pytest.raises(ConfigFileError, match="idle_seconds"):
            self._config(tmp_path, "[auto_save]\nidle_seconds = true\n")

    def test_rejects_negative(self, tmp_path):
        with pytest.raises(ConfigFileError, match=">= 0"):
            self._config(tmp_path, "[auto_save]\nidle_seconds = -1\n")

    def test_rejects_non_table(self, tmp_path):
        with pytest.raises(ConfigFileError, match="must be a table"):
            self._config(tmp_path, 'auto_save = "on"\n')


# ──────────────────────────────── Save behaviour ──────────────────────────────


class TestIdleSave:
    def test_default_is_300(self):
        assert make_agent().auto_save_idle_seconds == 300

    @pytest.mark.asyncio
    async def test_saves_once_due(self, saves_dir):
        agent = make_agent(idle_seconds=300)
        seen = capture_notifications(agent)

        await make_due(agent)
        await agent._maybe_idle_save()

        files = list(saves_dir.glob("*.ctx"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text())
        assert saved["messages"][0]["content"] == "hello"
        assert len(seen) == 1
        assert "auto-saved" in seen[0]

    @pytest.mark.asyncio
    async def test_does_not_save_before_due(self, saves_dir):
        agent = make_agent(idle_seconds=300)
        await agent._maybe_idle_save()  # starts the clock
        await agent._maybe_idle_save()  # still short of 300s
        assert list(saves_dir.glob("*.ctx")) == []

    @pytest.mark.asyncio
    async def test_saves_only_once_per_idle_period(self, saves_dir):
        """Still idle after a save: no second write, no second toast."""
        agent = make_agent(idle_seconds=300)
        seen = capture_notifications(agent)

        await make_due(agent)
        await agent._maybe_idle_save()
        first = list(saves_dir.glob("*.ctx"))[0]
        first_mtime = first.stat().st_mtime_ns

        # Long past due again, still the same idle period.
        agent._idle_since -= 10_000
        await agent._maybe_idle_save()

        assert first.stat().st_mtime_ns == first_mtime
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_activity_rearms_and_saves_new_work(self, saves_dir):
        agent = make_agent(idle_seconds=300)
        seen = capture_notifications(agent)

        await make_due(agent)
        await agent._maybe_idle_save()
        assert len(seen) == 1

        # New work, then idle again -> a second save.
        agent.messages.append({"role": "assistant", "content": "answer"})
        agent._mark_activity()
        assert agent._idle_since is None
        await make_due(agent)
        await agent._maybe_idle_save()

        assert len(seen) == 2
        saved = json.loads(list(saves_dir.glob("*.ctx"))[0].read_text())
        assert len(saved["messages"]) == 2

    @pytest.mark.asyncio
    async def test_unchanged_history_is_not_rewritten(self, saves_dir):
        """An idle period that changed nothing (e.g. only /status) skips."""
        agent = make_agent(idle_seconds=300)
        seen = capture_notifications(agent)

        await make_due(agent)
        await agent._maybe_idle_save()
        path = list(saves_dir.glob("*.ctx"))[0]
        first_mtime = path.stat().st_mtime_ns

        agent._mark_activity()  # a command ran, history untouched
        await make_due(agent)
        await agent._maybe_idle_save()

        assert path.stat().st_mtime_ns == first_mtime
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_disabled_writes_nothing(self, saves_dir):
        agent = make_agent(idle_seconds=0)
        seen = capture_notifications(agent)

        await agent._maybe_idle_save()
        await agent._maybe_idle_save()

        assert agent._idle_since is None
        assert list(saves_dir.glob("*.ctx")) == []
        assert seen == []

    @pytest.mark.asyncio
    async def test_empty_history_writes_nothing(self, saves_dir):
        agent = make_agent(idle_seconds=300, messages=[])
        await agent._maybe_idle_save()
        await agent._maybe_idle_save()
        assert list(saves_dir.glob("*.ctx")) == []

    @pytest.mark.asyncio
    async def test_not_idle_writes_nothing(self, saves_dir):
        """A busy agent (mid-turn) never saves, even past the threshold."""
        agent = make_agent(idle_seconds=300)
        await make_due(agent)
        agent._status = AgentStatus.TOOLING
        await agent._maybe_idle_save()
        assert list(saves_dir.glob("*.ctx")) == []

    @pytest.mark.asyncio
    async def test_paused_is_not_idle(self, saves_dir):
        agent = make_agent(idle_seconds=300)
        await make_due(agent)
        agent.pause()
        await agent._maybe_idle_save()
        assert list(saves_dir.glob("*.ctx")) == []

    @pytest.mark.asyncio
    async def test_failed_write_does_not_retry_or_raise(self, saves_dir, monkeypatch):
        """A write failure is swallowed (best-effort) and not retried."""
        agent = make_agent(idle_seconds=300)
        seen = capture_notifications(agent)
        calls = []

        def boom(agent_arg, path):
            calls.append(path)
            raise OSError("disk full")

        monkeypatch.setattr("agent13.persistence.save_context", boom)

        await make_due(agent)
        await agent._maybe_idle_save()
        agent._idle_since -= 10_000
        await agent._maybe_idle_save()

        assert len(calls) == 1
        assert seen == []


# ───────────────────────────────── Status ─────────────────────────────────────


class TestStatus:
    def test_gather_status_reports_value(self):
        agent = make_agent(idle_seconds=120)
        status = gather_status(agent, "test-provider", "test-model", time.time())
        assert status.auto_save_idle_seconds == 120

    def test_gather_status_tolerates_missing_attribute(self):
        """Old/stub agents without the attribute report 0, never crash."""
        agent = make_agent(idle_seconds=120)
        del agent.auto_save_idle_seconds
        status = gather_status(agent, "test-provider", "test-model", time.time())
        assert status.auto_save_idle_seconds == 0
