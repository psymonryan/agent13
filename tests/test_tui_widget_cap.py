#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "textual>=0.85.0",
#     "pytest>=7.0.0",
#     "pytest-asyncio>=0.21.0",
# ]
# ///
"""Chat widget cap - wiring tests.

The live chat used to keep every message as a widget for the life of the
session: 100 messages = 6,109 widgets, and mounting one more message cost
250 ms because the cost is O(tree size). ``MAX_CHAT_MESSAGES`` bounds the
live widgets; the message HISTORY is untouched.

These tests drive the real ``AgentTUI`` with a stub agent (same harness as
the other ``test_tui_*`` files) and assert on the chat's own children.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.durable_records import DurableRecords
from agent13.queue import AgentQueue


class FakeMCP:
    """Stands in for MCPManager - just the attributes the TUI reads."""

    def __init__(self):
        self.connecting = False
        self.connect_all = AsyncMock(return_value={})
        self.servers = {}
        self._shutting_down = False

    def is_connected(self):
        return not self.connecting


def _make_app():
    """Build an AgentTUI with a stub agent (the test_tui_* pattern)."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
        )

    agent = MagicMock()
    agent.durable_records = DurableRecords()
    # The status bar reads model/provider from the agent (single source of
    # truth), so the stub must carry real strings, not MagicMocks.
    agent.model = "test-model"
    agent.provider_name = "test"
    agent.mcp = FakeMCP()
    agent._mcp = agent.mcp
    agent._mcp_server_configs = []
    agent._ensure_mcp = AsyncMock(return_value=agent.mcp)
    agent.add_message = AsyncMock()
    agent.queue = AgentQueue()
    agent.is_pausing = False
    agent.is_paused = False
    agent.status = MagicMock()
    agent.status.value = "idle"
    agent.tool_stats = MagicMock()
    agent.tool_stats.total_calls = 0
    agent.auto_context_chain_used = 0
    agent.messages = []

    async def _run_forever():
        await asyncio.Event().wait()

    agent.run = _run_forever
    app.agent = agent
    return app


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Keep test history/saves out of the real ~/.agent13."""
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(tmp_path))


def _texts(chat):
    """The rendered text of each top-level chat child, in order."""
    out = []
    for child in chat.children:
        render = getattr(child, "render", None)
        out.append(str(render()) if callable(render) else str(child))
    return out


@pytest.mark.asyncio
async def test_live_mounts_are_capped(isolated_config):
    """Mounting past the cap drops the OLDEST messages, not the newest."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        for i in range(MAX_CHAT_MESSAGES + 5):
            await app._write_system(f"msg {i}")
        await pilot.pause()

        texts = _texts(chat)
        assert len(texts) == MAX_CHAT_MESSAGES
        assert "msg 0" not in texts, "the oldest message must be the one dropped"
        assert "msg 4" not in texts
        assert "msg 5" in texts, "the cap must keep exactly the newest N"
        assert f"msg {MAX_CHAT_MESSAGES + 4}" in texts, "the newest must survive"


@pytest.mark.asyncio
async def test_just_mounted_widget_is_never_trimmed(isolated_config):
    """The widget that triggers a trim is the newest, so it always survives."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        for i in range(MAX_CHAT_MESSAGES * 2):
            await app._write_system(f"msg {i}")
            await pilot.pause()
            assert f"msg {i}" in _texts(chat), (
                f"message {i} was trimmed by its own mount"
            )


@pytest.mark.asyncio
async def test_rebuild_bypasses_the_cap(isolated_config):
    """/rebuild is the "show me everything" path: it must not trim itself."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    app.agent.messages = [{"role": "user", "content": f"u{i}"} for i in range(60)]
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        await app._rebuild_chat()
        await pilot.pause()

        assert len(list(chat.children)) == 61, (
            "60 messages + the 'Rebuilt chat from' status line"
        )
        assert len(list(chat.children)) > MAX_CHAT_MESSAGES
        assert app._chat_cap_suspends == 0, (
            "the cap must be re-enabled after the rebuild finishes"
        )


@pytest.mark.asyncio
async def test_next_live_message_trims_after_a_rebuild(isolated_config):
    """A rebuild leaves the tree long; the next live message trims it back."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    app.agent.messages = [{"role": "user", "content": f"u{i}"} for i in range(60)]
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        await app._rebuild_chat()
        await pilot.pause()
        assert len(list(chat.children)) > MAX_CHAT_MESSAGES

        await app._write_system("after rebuild")
        await pilot.pause()

        texts = _texts(chat)
        assert len(texts) == MAX_CHAT_MESSAGES
        assert texts[-1] == "after rebuild"
        assert "u0" not in texts, "the trim takes from the front"


@pytest.mark.asyncio
async def test_cap_is_restored_when_a_rebuild_raises(isolated_config):
    """The context manager must not leak a suspended cap on an exception."""
    app = _make_app()
    app.agent.messages = [{"role": "user", "content": "u"}]
    async with app.run_test() as pilot:
        await pilot.pause()

        app._rebuild_from_messages = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            await app._rebuild_chat()

        assert app._chat_cap_suspends == 0


@pytest.mark.asyncio
async def test_rebuild_status_line_does_not_undo_the_rebuild(isolated_config):
    """/rebuild writes a status line AFTER mounting everything. That line
    goes through _safe_mount too, so a bool flag (rather than a nesting-safe
    counter) let it trim the rebuild straight back to the cap."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    app.agent.messages = [{"role": "user", "content": f"u{i}"} for i in range(60)]
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = app.query_one("#chat")

        await app._rebuild_chat()
        await pilot.pause()

        assert len(list(chat.children)) > MAX_CHAT_MESSAGES, (
            "the status line trimmed the rebuild back to the cap"
        )
        assert app._chat_cap_suspends == 0


@pytest.mark.asyncio
async def test_trim_leaves_message_history_untouched(isolated_config):
    """The cap is display-only: agent.messages must not shrink."""
    from ui.tui import MAX_CHAT_MESSAGES

    app = _make_app()
    app.agent.messages = [{"role": "user", "content": f"u{i}"} for i in range(100)]
    async with app.run_test() as pilot:
        await pilot.pause()

        for i in range(MAX_CHAT_MESSAGES + 5):
            await app._write_system(f"msg {i}")
        await pilot.pause()

        assert len(app.agent.messages) == 100


def test_cap_is_a_sane_size():
    """A cap below a screenful would make the chat useless; above ~100
    messages the mount cost is back to the old problem."""
    from ui.tui import MAX_CHAT_MESSAGES

    assert 10 <= MAX_CHAT_MESSAGES <= 100
