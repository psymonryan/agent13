#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""Chat widget cap - integration tests.

Drives the REAL TUI with the REAL Agent against the built-in ``fake``
provider (``FakeOpenAIClient``): no network, no mock server - the fake IS
the model. Each message is a genuine turn through the agent loop, so this
tests what the user actually experiences: a long conversation whose chat
window stops growing.

``MAX_CHAT_MESSAGES`` is monkeypatched down to a handful so the test runs
in seconds instead of minutes. The trim loop is size-independent; the real
constant is covered by ``test_tui_widget_cap.py``.
"""

import asyncio

import pytest


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Isolated config dir: no MCP servers, skills or history from the real
    ~/.agent13 can leak into the run."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text("# isolated empty config\n")
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(config_dir))


def _make_app():
    from agent13.fake import FakeOpenAIClient
    from ui.tui import AgentTUI

    return AgentTUI(
        client=FakeOpenAIClient(),
        model="fake",
        model_names=["fake"],
        provider="fake",
    )


async def _wait_turn_done(app, pilot, timeout=30.0):
    """Wait until the agent has finished the turn AND the queue is empty.

    Gates on the AGENT's state, never on a line the front-end printed (see
    the burn scar in AGENTS.md): a printed status line races the run loop.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if app.agent.is_idle and not app.agent.has_pending_work:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("turn did not finish in time")


def _texts(chat):
    out = []
    for child in chat.children:
        render = getattr(child, "render", None)
        out.append(str(render()) if callable(render) else str(child))
    return out


@pytest.mark.asyncio
async def test_long_conversation_stops_growing_the_chat(isolated_config, monkeypatch):
    """After N real turns the chat holds at most the cap, and the messages
    that were dropped are still in the agent's history."""
    import ui.tui as tui_mod

    monkeypatch.setattr(tui_mod, "MAX_CHAT_MESSAGES", 6)

    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        turns = 10
        for i in range(turns):
            await app._send_message(f"MSG-{i:02d}")
            await _wait_turn_done(app, pilot)

        texts = _texts(chat)
        assert len(texts) <= 6, f"chat grew past the cap: {len(texts)} widgets"
        assert not any("MSG-00" in t for t in texts), "the oldest turn must be gone"
        assert any(f"MSG-{turns - 1:02d}" in t for t in texts), (
            "the newest turn must still be visible"
        )

        # The cap is display-only: every turn is still in the history.
        user_msgs = [
            m for m in app.agent.messages
            if m.get("role") == "user" and "MSG-" in str(m.get("content", ""))
        ]
        assert len(user_msgs) == turns, (
            f"history lost messages: {len(user_msgs)} of {turns}"
        )


@pytest.mark.asyncio
async def test_rebuild_restores_the_full_conversation(isolated_config, monkeypatch):
    """/rebuild is the escape hatch: it materialises every message again."""
    import ui.tui as tui_mod

    monkeypatch.setattr(tui_mod, "MAX_CHAT_MESSAGES", 6)

    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        turns = 10
        for i in range(turns):
            await app._send_message(f"MSG-{i:02d}")
            await _wait_turn_done(app, pilot)

        capped = len(list(chat.children))
        assert capped <= 6

        await app._rebuild_chat()
        await pilot.pause()

        rebuilt = len(list(chat.children))
        assert rebuilt > capped, (
            f"/rebuild must show the full history ({rebuilt} vs {capped})"
        )
        texts = _texts(chat)
        assert any("MSG-00" in t for t in texts), (
            "/rebuild must bring the oldest turn back"
        )
