#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for skill slash commands with arguments.

`/skillname <text>` must send the skill content followed by the user's text
as a single message. `/skillname` with no text must behave as before
(skill content only).

The TUI app is constructed with the real SkillManager (real on-disk skill)
and a stub agent, so `_handle_command` runs its real code path.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.skills.manager import SkillManager


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Poll until predicate() is true (bounded).

    A single ``await asyncio.sleep(0)`` is not enough on Windows: the task
    spawned by ``asyncio.create_task`` may not get scheduled in one loop
    iteration. Polling on a condition is deterministic on all platforms.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError(
                f"Timed out waiting for condition after {timeout}s"
            )
        await asyncio.sleep(0.01)

SKILL_MD = """---
name: test-skill
description: A test skill
---

# Test Skill

Do the thing.
"""


@pytest.fixture
def skill_manager(tmp_path):
    """Real SkillManager with one skill on disk.

    The config is a real (empty) Config so that ``skill_paths`` points at
    the tmp skills dir — no other search path (project .agent13/skills,
    ~/.agent13/skills) can contribute skills, keeping the test
    deterministic on any machine.
    """
    from agent13.config import Config

    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    config = Config()
    config.skill_paths = [tmp_path / "skills"]
    return SkillManager(lambda: config)


def _make_app(skill_manager):
    """Build a ChatApp with a stub agent; return (app, agent_mock)."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
            skill_manager=skill_manager,
        )
    agent = MagicMock()
    agent.add_message = AsyncMock()

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    agent.queue = MagicMock()
    agent.queue.pending_count = 0
    agent.queue.current = None
    agent.devel_mode = False
    app.agent = agent
    return app, agent


@pytest.mark.asyncio
async def test_skill_command_with_args(skill_manager):
    """Args are appended after the skill content, one message."""
    app, agent = _make_app(skill_manager)
    async with app.run_test():
        app._handle_command("/test-skill do the other thing")
        await _wait_for(lambda: agent.add_message.await_count == 1)
        sent = agent.add_message.await_args.args[0]
        assert sent.startswith("<skill_content name=\"test-skill\">")
        assert "# Test Skill" in sent
        assert "Briefly acknowledge the skill is loaded" in sent
        assert sent.rstrip().endswith("do the other thing")
        assert sent.endswith("</skill_content>\n\nBriefly acknowledge the skill is loaded. Do not summarize or explain it.\n\ndo the other thing")


@pytest.mark.asyncio
async def test_skill_command_without_args(skill_manager):
    """No args: behaviour unchanged, skill content only."""
    app, agent = _make_app(skill_manager)
    async with app.run_test():
        app._handle_command("/test-skill")
        await _wait_for(lambda: agent.add_message.await_count == 1)
        sent = agent.add_message.await_args.args[0]
        assert sent.startswith("<skill_content name=\"test-skill\">")
        assert sent.rstrip().endswith("Do not summarize or explain it.")
        assert "do the other thing" not in sent


@pytest.mark.asyncio
async def test_skill_command_whitespace_only_args(skill_manager):
    """Whitespace-only args are treated as no args."""
    app, agent = _make_app(skill_manager)
    async with app.run_test():
        app._handle_command("/test-skill   ")
        await _wait_for(lambda: agent.add_message.await_count == 1)
        sent = agent.add_message.await_args.args[0]
        assert sent.rstrip().endswith("Do not summarize or explain it.")
