#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""
Tests for the skill *tool* ratchet.

The skill tool is off by default and is ratcheted on (one-way, no off-switch)
the first time the user invokes a skill via slash command. `--skills` /
`/skills on` also enable it; `/skills off` only removes the skills *list* and
never switches the tool off.

The TUI app is constructed with the real SkillManager (real on-disk skill) and
a stub agent, so `_handle_command` / `_handle_skills_command` run their real
code path. The ratchet (`agent.set_skills_mode(True)`) is synchronous in the
skill-invocation path, so it can be asserted directly.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.skills.manager import SkillManager


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    """Poll until predicate() is true (bounded). See test_skills_slash_args."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"Timed out waiting for condition after {timeout}s")
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
    """Real SkillManager with one skill on a deterministic, isolated path."""
    from agent13.config import Config

    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

    config = Config()
    config.skill_paths = [tmp_path / "skills"]
    return SkillManager(lambda: config)


def _make_app(skill_manager, skills_mode: bool):
    """Build a ChatApp with a stub agent whose ``skills_mode`` is *skills_mode*.

    Returns ``(app, agent_mock)``. The stub agent records ``set_skills_mode()``
    calls and its ``_update_info_content`` is stubbed so we don't depend on
    widget mounting.
    """
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
    agent.skills_mode = skills_mode
    agent.set_skills_mode = MagicMock()
    agent.add_message = AsyncMock()
    agent.set_system_prompt = MagicMock()

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    agent.queue = MagicMock()
    agent.queue.pending_count = 0
    agent.queue.current = None
    agent.devel_mode = False
    app.agent = agent
    app._update_info_content = MagicMock()
    return app, agent


@pytest.mark.asyncio
async def test_ratchet_enables_tool_on_first_invocation(skill_manager):
    """Invoking a skill while the tool is off ratchets it on."""
    app, agent = _make_app(skill_manager, skills_mode=False)
    async with app.run_test():
        app._handle_command("/test-skill")
        # Ratchet is synchronous; also let the async skill-content send settle.
        await _wait_for(lambda: agent.add_message.await_count == 1)
        agent.set_skills_mode.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_ratchet_noop_when_already_enabled(skill_manager):
    """If the tool is already on (e.g. via --skills), invocation must not re-set it."""
    app, agent = _make_app(skill_manager, skills_mode=True)
    async with app.run_test():
        app._handle_command("/test-skill")
        await _wait_for(lambda: agent.add_message.await_count == 1)
        agent.set_skills_mode.assert_not_called()


@pytest.mark.asyncio
async def test_skills_off_does_not_disable_tool(skill_manager):
    """/skills off removes the list but must NOT switch the tool off."""
    app, agent = _make_app(skill_manager, skills_mode=True)
    async with app.run_test():
        app._handle_command("/skills off")
        agent.set_skills_mode.assert_not_called()
        assert app._skills_list_enabled is False


@pytest.mark.asyncio
async def test_skills_on_enables_tool_and_list(skill_manager):
    """/skills on enables both the list and the tool."""
    app, agent = _make_app(skill_manager, skills_mode=False)
    async with app.run_test():
        app._handle_command("/skills on")
        agent.set_skills_mode.assert_called_once_with(True)
        assert app._skills_list_enabled is True
