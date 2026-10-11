"""Wiring/unit tests for the /chain command.

/chain is a manual "restart the loop": wrap-up (if stale) → clear history
→ inject records. No AI calls when the wrap-up is already current.

Skill reload: manually-loaded skills (TUI /skillname) are tracked on the
agent and re-injected by the mechanical restart (intro -> skills -> truth
-> journal). "/chain no-skills" drops them from the restart and forgets
them.
"""

import asyncio
import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.durable_records import DurableRecords
from agent13.events import AgentEvent
from agent13.queue import AgentQueue, QueueItem

from .test_repl import find_call, run_scenario


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Keep test history/saves/config out of the real ~/.agent13."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))


def make_agent():
    """Create a minimal agent for testing."""
    client = MagicMock()
    agent = Agent(
        client=client,
        model="test-model",
    )
    return agent


# ── REPL wiring ───────────────────────────────────────────────────────────────


class TestChainRepl:
    """REPL wiring: /chain queues a kind="chain" item."""

    async def test_bare_chain(self):
        _output, agent = await run_scenario(["/chain", "/quit"])
        call = find_call(agent, "add_message")
        assert call is not None
        _, text, _prio, _int, kind, data = call
        assert text == "/chain"
        assert kind == "chain"


# ── Core dispatch ─────────────────────────────────────────────────────────────


class TestChainDispatch:
    """Core dispatch: kind="chain" → wrap-up (if stale) → clear → seed."""

    def _mock_agent(self, agent, messages, seed_result=True):
        """Mock the agent's dependencies for dispatch testing."""
        agent.messages = messages
        agent._run_report_and_compact_turn = AsyncMock()
        agent._run_auto_context_post_turn = AsyncMock()
        agent.seed_from_records = MagicMock(return_value=seed_result)
        agent.clear_messages = MagicMock(return_value=len(messages))
        agent._set_status = AsyncMock()
        agent._emit_queue_update = AsyncMock()
        agent.queue.complete_current = MagicMock()
        events = []
        agent.emit = AsyncMock(
            side_effect=lambda evt, data: events.append((evt, data))
        )
        return events

    @pytest.mark.asyncio
    async def test_wrapup_current_skips_wrapup(self):
        """When _wrapup_is_current() is True, no wrap-up turn runs."""
        agent = make_agent()
        agent.set_focus_name("test-focus")
        messages = [
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
            {"role": "assistant", "content": "Wrap-up complete."},
        ]
        events = self._mock_agent(agent, messages)
        assert agent._wrapup_is_current() is True

        item = QueueItem(id=1, text="/chain", kind="chain")
        with patch("agent13.core.log_queue_start"), \
             patch("agent13.core.log_queue_complete"):
            await agent._process_item(item)

        # Wrap-up should NOT have run
        agent._run_report_and_compact_turn.assert_not_called()
        # History cleared
        agent.clear_messages.assert_called_once()
        # Records seeded
        agent.seed_from_records.assert_called_once()
        # CHAT_CLEAR emitted
        assert AgentEvent.CHAT_CLEAR in [e[0] for e in events]
        # CHAT_REBUILD emitted after records injection
        assert AgentEvent.CHAT_REBUILD in [e[0] for e in events]
        # Notification says records injected
        notifs = [e for e in events if e[0] == AgentEvent.NOTIFICATION]
        assert len(notifs) == 1
        assert "records injected" in notifs[0][1]["message"]

    @pytest.mark.asyncio
    async def test_wrapup_stale_runs_wrapup(self):
        """When _wrapup_is_current() is False, the wrap-up turn runs first."""
        agent = make_agent()
        agent.set_focus_name("test-focus")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        events = self._mock_agent(agent, messages)
        assert agent._wrapup_is_current() is False

        item = QueueItem(id=1, text="/chain", kind="chain")
        with patch("agent13.core.log_queue_start"), \
             patch("agent13.core.log_queue_complete"):
            await agent._process_item(item)

        # Wrap-up SHOULD have run
        agent._run_report_and_compact_turn.assert_called_once()
        # History cleared
        agent.clear_messages.assert_called_once()
        # Records seeded
        agent.seed_from_records.assert_called_once()
        # CHAT_REBUILD emitted after records injection
        assert AgentEvent.CHAT_REBUILD in [e[0] for e in events]

    @pytest.mark.asyncio
    async def test_no_records_notification(self):
        """When seed_from_records finds nothing, the notification says so."""
        agent = make_agent()
        agent.set_focus_name("test-focus")
        messages = [
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
            {"role": "assistant", "content": "Wrap-up complete."},
        ]
        events = self._mock_agent(agent, messages, seed_result=False)
        assert agent._wrapup_is_current() is True

        item = QueueItem(id=1, text="/chain", kind="chain")
        with patch("agent13.core.log_queue_start"), \
             patch("agent13.core.log_queue_complete"):
            await agent._process_item(item)

        # CHAT_REBUILD emitted even when no records found (intro stands alone)
        assert AgentEvent.CHAT_REBUILD in [e[0] for e in events]
        notifs = [e for e in events if e[0] == AgentEvent.NOTIFICATION]
        assert len(notifs) == 1
        assert "no records found" in notifs[0][1]["message"]
        assert "/focus" in notifs[0][1]["message"]


# ── Skill tracking state ─────────────────────────────────────────────────────


class TestActiveSkills:
    """Unit tests for the manually-loaded skill tracking state."""

    def test_note_and_property(self):
        agent = make_agent()
        assert agent.active_skills == {}
        agent.note_skill_loaded("humanizer", "<skill_content name=\"humanizer\">X")
        skills = agent.active_skills
        assert list(skills.keys()) == ["humanizer"]
        assert skills["humanizer"].startswith("<skill_content")
        # Property returns a copy - mutating it must not touch the state.
        skills["injected"] = "nope"
        assert agent.active_skills == {"humanizer": "<skill_content name=\"humanizer\">X"}

    def test_dedup_latest_content_order_preserved(self):
        agent = make_agent()
        agent.note_skill_loaded("a", "A1")
        agent.note_skill_loaded("b", "B1")
        agent.note_skill_loaded("a", "A2")  # re-load: content replaced
        skills = agent.active_skills
        assert list(skills.keys()) == ["a", "b"]  # first-seen order kept
        assert skills["a"] == "A2"  # latest content wins
        assert skills["b"] == "B1"

    def test_empty_name_or_content_ignored(self):
        agent = make_agent()
        agent.note_skill_loaded("", "content")
        agent.note_skill_loaded("name", "")
        agent.note_skill_loaded(None, "content")  # type: ignore[arg-type]
        assert agent.active_skills == {}

    def test_clear(self):
        agent = make_agent()
        agent.note_skill_loaded("a", "A")
        agent.note_skill_loaded("b", "B")
        agent.clear_active_skills()
        assert agent.active_skills == {}


# ── Skill injection in _mechanical_restart ──────────────────────────────────


def make_record_agent(truth="TRUTH-CONTENT", journal="JOURNAL-ENTRY"):
    """Agent with a focus name and mocked record readers/estimator, ready
    for a direct _mechanical_restart() call.

    The readers are patched on the DURABLE RECORDS object, not the agent:
    the record file reads live there now and Agent deliberately has no
    forwarding method for them (a forward would make patching
    agent._read_record_file a silent no-op that reads the real filesystem).
    """
    agent = make_agent()
    agent.set_focus_name("test-focus")
    agent.durable_records.read_record_file = MagicMock(return_value=truth)
    agent.durable_records.read_last_journal_entry = MagicMock(return_value=journal)
    agent._estimate_post_compact_context = AsyncMock(return_value=100)
    return agent


class TestMechanicalRestartSkills:
    """_mechanical_restart injects the skills section between the intro
    and the durable records (order: intro -> skills -> truth -> journal)."""

    @pytest.mark.asyncio
    async def test_skills_injected_between_intro_and_records(self):
        agent = make_record_agent()
        agent.note_skill_loaded(
            "skill_a", '<skill_content name="skill_a">A-INSTRUCTIONS'
        )
        agent.note_skill_loaded(
            "skill_b", '<skill_content name="skill_b">B-INSTRUCTIONS'
        )

        seeded = await agent._mechanical_restart()

        assert seeded is True
        content = agent.messages[0]["content"]
        i_intro = content.find("[Session restarted from durable records")
        i_skills_header = content.find("Skills loaded this session")
        i_a = content.find("A-INSTRUCTIONS")
        i_b = content.find("B-INSTRUCTIONS")
        i_truth = content.find("TRUTH-CONTENT")
        i_journal = content.find("JOURNAL-ENTRY")
        assert (
            0 <= i_intro < i_skills_header < i_a < i_b < i_truth < i_journal
        ), f"unexpected injection order:\n{content}"
        # Load order preserved (a before b).
        # The restart ack follows the injected user message.
        assert agent.messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_no_skills_no_section(self):
        """Regression: with no skills tracked the message is intro+records
        exactly as before the feature."""
        agent = make_record_agent()
        seeded = await agent._mechanical_restart()
        assert seeded is True
        content = agent.messages[0]["content"]
        assert "Skills loaded this session" not in content
        assert "TRUTH-CONTENT" in content
        assert "JOURNAL-ENTRY" in content

    @pytest.mark.asyncio
    async def test_carry_skills_false_skips_section_keeps_state(self):
        """carry_skills=False (the /chain no-skills path, after the
        dispatch has already cleared state) must not inject the section.
        Called directly with state present, it skips injection WITHOUT
        clearing the state (clearing is the dispatch's job)."""
        agent = make_record_agent()
        agent.note_skill_loaded("a", "A-INSTRUCTIONS")
        seeded = await agent._mechanical_restart(carry_skills=False)
        assert seeded is True
        content = agent.messages[0]["content"]
        assert "Skills loaded this session" not in content
        assert "A-INSTRUCTIONS" not in content
        assert agent.active_skills == {"a": "A-INSTRUCTIONS"}

    @pytest.mark.asyncio
    async def test_skills_without_records_stand_alone(self):
        """No record files (fresh focus): the injected message is
        intro + skills, no records part."""
        agent = make_record_agent(truth="", journal="")
        agent.note_skill_loaded("a", "A-INSTRUCTIONS")
        seeded = await agent._mechanical_restart()
        assert seeded is False
        content = agent.messages[0]["content"]
        assert "A-INSTRUCTIONS" in content
        assert "Skills loaded this session" in content
        assert "TRUTH-CONTENT" not in content


# ── /chain no-skills dispatch ───────────────────────────────────────────────


class TestChainNoSkillsDispatch:
    """Core dispatch: kind="chain" with data={"no_skills": True} clears the
    tracked skills, restarts without them, and says so in the
    notification."""

    def _mock_agent(self, agent, messages):
        """Mock dependencies; keep _mechanical_restart as a spy so the
        carry_skills argument is observable."""
        agent.messages = messages
        agent._run_report_and_compact_turn = AsyncMock()
        agent._mechanical_restart = AsyncMock(return_value=True)
        agent._set_status = AsyncMock()
        agent._emit_queue_update = AsyncMock()
        agent.queue.complete_current = MagicMock()
        events = []
        agent.emit = AsyncMock(
            side_effect=lambda evt, data: events.append((evt, data))
        )
        return events

    def _current_wrapup_messages(self):
        return [
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
            {"role": "assistant", "content": "Wrap-up complete."},
        ]

    @pytest.mark.asyncio
    async def test_no_skills_clears_state_and_notifies(self):
        agent = make_agent()
        agent.set_focus_name("test-focus")
        agent.note_skill_loaded("humanizer", "HUMANIZER-CONTENT")
        events = self._mock_agent(agent, self._current_wrapup_messages())

        item = QueueItem(
            id=1, text="/chain no-skills", kind="chain", data={"no_skills": True}
        )
        with patch("agent13.core.log_queue_start"), \
             patch("agent13.core.log_queue_complete"):
            await agent._process_item(item)

        assert agent.active_skills == {}, "state must be cleared"
        agent._mechanical_restart.assert_awaited_once_with(carry_skills=False)
        notifs = [e for e in events if e[0] == AgentEvent.NOTIFICATION]
        assert len(notifs) == 1
        assert "records injected" in notifs[0][1]["message"]
        assert "skills dropped" in notifs[0][1]["message"]

    @pytest.mark.asyncio
    async def test_bare_chain_keeps_state_and_carries(self):
        agent = make_agent()
        agent.set_focus_name("test-focus")
        agent.note_skill_loaded("humanizer", "HUMANIZER-CONTENT")
        events = self._mock_agent(agent, self._current_wrapup_messages())

        item = QueueItem(id=1, text="/chain", kind="chain")
        with patch("agent13.core.log_queue_start"), \
             patch("agent13.core.log_queue_complete"):
            await agent._process_item(item)

        assert agent.active_skills == {"humanizer": "HUMANIZER-CONTENT"}
        agent._mechanical_restart.assert_awaited_once_with(carry_skills=True)
        notifs = [e for e in events if e[0] == AgentEvent.NOTIFICATION]
        assert len(notifs) == 1
        assert "skills dropped" not in notifs[0][1]["message"]


# ── TUI wiring ──────────────────────────────────────────────────────────────


def _seed_skill(root: Path, name: str) -> Path:
    """Write a minimal valid SKILL.md under root/<name>/ and return root.

    SkillManager discovers from config skill_paths, the project dir and
    ~/.agent13/skills - it does NOT search the package's bundled
    default_skills. A test that needs a resolvable skill must seed one;
    otherwise it silently reads the developer's live skills collection and
    breaks the day someone renames or removes a skill there.
    """
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture skill for tests.\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return root


def _make_tui_app(skills_dir: Path | None = None):
    """Build an AgentTUI with a stub agent and a REAL SkillManager.

    skills_dir is fed into config skill_paths (the highest-priority
    discovery path) so the manager resolves a known fixture set instead of
    the developer's ~/.agent13/skills. The agent stub mirrors
    test_tui_event_guard._make_app (the known-good mount pattern); the
    only difference is the real SkillManager.
    """
    from ui.tui import AgentTUI as ChatApp

    from agent13.config import get_config
    from agent13.skills import SkillManager

    config = copy.deepcopy(get_config())
    if skills_dir is not None:
        config.skill_paths = [skills_dir]

    class FakeMCP:
        """Stands in for MCPManager - just the attributes the TUI reads."""

        connecting = False
        servers = {}
        _shutting_down = False

        def __init__(self):
            self.connect_all = AsyncMock(return_value={})

        def is_connected(self):
            return True

    with patch("ui.tui.get_config", return_value=config):
        app = ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            skill_manager=SkillManager(lambda: config),
        )

    agent = MagicMock()
    agent.durable_records = DurableRecords()
    mcp = FakeMCP()
    agent.mcp = mcp
    agent._mcp = mcp
    agent._mcp_server_configs = []
    agent._ensure_mcp = AsyncMock(return_value=mcp)
    agent.add_message = AsyncMock()
    agent.queue = AgentQueue()
    agent.is_pausing = False
    agent.is_paused = False
    agent.status = MagicMock()
    agent.status.value = "idle"
    agent.tool_stats = MagicMock()
    agent.tool_stats.total_calls = 0
    agent.auto_context_chain_used = 0
    agent.skills_mode = False

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    app.agent = agent
    return app


class TestChainTuiWiring:
    """TUI command handlers: /chain no-skills queues the data flag;
    /skillname tracks the loaded content on the agent."""

    @pytest.mark.asyncio
    async def test_chain_no_skills_queues_data(self, isolated_config):
        app = _make_tui_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_chain_command("no-skills")
            await pilot.pause(0.2)
        app.agent.add_message.assert_awaited_once_with(
            "/chain", kind="chain", data={"no_skills": True}
        )

    @pytest.mark.asyncio
    async def test_chain_dash_and_underscore_variants(self, isolated_config):
        app = _make_tui_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_chain_command("--no-skills")
            app._handle_chain_command("no_skills")
            await pilot.pause(0.2)
        calls = [c for c in app.agent.add_message.await_args_list]
        assert len(calls) == 2
        for c in calls:
            assert c.kwargs.get("data") == {"no_skills": True}

    @pytest.mark.asyncio
    async def test_chain_bare_queues_no_data_flag(self, isolated_config):
        app = _make_tui_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_chain_command("")
            await pilot.pause(0.2)
        app.agent.add_message.assert_awaited_once_with(
            "/chain", kind="chain", data={}
        )

    @pytest.mark.asyncio
    async def test_skill_command_tracks_on_agent(self, isolated_config, tmp_path):
        """Typing /<skill-name> tracks the formatted skill content (with any
        user args) on the agent for chain-restart re-injection.

        Uses a seeded fixture skill, not a real one: the manager reads the
        developer's live skills collection otherwise, so renaming a skill
        there used to break this test.
        """
        app = _make_tui_app(_seed_skill(tmp_path / "skills", "fixture-skill"))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_command("/fixture-skill extra instructions")
            await pilot.pause(0.2)

        agent = app.agent
        agent.note_skill_loaded.assert_called_once()
        name, content = agent.note_skill_loaded.call_args.args
        assert name == "fixture-skill"
        assert content.startswith('<skill_content name="fixture-skill">')
        assert content.endswith("extra instructions")
        # The skill text is also sent as a message.
        agent.add_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_command_not_tracked(self, isolated_config):
        app = _make_tui_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._handle_command("/definitely-not-a-skill")
            await pilot.pause(0.2)
        app.agent.note_skill_loaded.assert_not_called()
