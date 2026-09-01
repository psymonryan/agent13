"""Native-vision image injections must stay inside the turn they arrive in.

Bug: a tool result carrying images is injected mid-turn as a user message
(``{"role": "user", "content": [text, image_url]}``). Nothing marked it as
mid-turn, so every "is this a turn start?" check — grouping, compaction,
trims, turn counters — treated the injection as a brand new turn. /retry
then deleted the injection instead of the user's prompt and crashed on
``AttributeError: 'list' object has no attribute 'startswith'`` when trying
to offer it as retry text.

Fix: injections are flagged ``"injected": True``, stripped before the API
call, and all turn-boundary checks go through ``message_history.is_turn_start``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent13.commands import execute_retry, format_history_groups
from agent13.core import Agent, AgentStatus, PauseState
from agent13.llm import build_messages_with_system
from agent13.message_history import (
    MessageHistory,
    content_to_text,
    is_injected,
    is_turn_start,
    mark_injected_messages,
)
from agent13.persistence import load_context, save_context
from tools import ToolResult

# 1x1 red PNG
PNG_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _injected_msg(text="[Image from tool: read_file]", injected=True):
    """A native-vision injection message (as core.py builds it)."""
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": PNG_URI}},
        ],
    }
    if injected:
        msg["injected"] = True
    return msg


def _history_turn():
    """Two turns; the second uses a tool that returns an image."""
    return [
        {"role": "user", "content": "first turn"},                     # 0
        {"role": "assistant", "content": "answer 1"},                   # 1
        {"role": "user", "content": "read the screenshot"},             # 2
        {                                                              # 3
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "file text"},  # 4
        _injected_msg(),                                                # 5
        {"role": "assistant", "content": "It is red."},                  # 6
    ]


def _make_agent(messages=None):
    client = MagicMock()
    agent = Agent(client=client, model="mock-model")
    agent.messages = messages if messages is not None else []
    agent._status = AgentStatus.IDLE
    agent._pause_state = PauseState.RUNNING
    return agent


@pytest.fixture
def native_vision(monkeypatch):
    """Force native vision routing (no [vision] config section)."""
    from agent13 import config as config_mod

    monkeypatch.setattr(config_mod, "get_config", lambda: SimpleNamespace(vision=None))


# ── is_turn_start / helpers ────────────────────────────────────────────────


class TestIsTurnStart:
    def test_plain_user_message_starts_turn(self):
        assert is_turn_start({"role": "user", "content": "hello"})

    def test_interrupt_does_not_start_turn(self):
        msg = {"role": "user", "content": "stop", "interrupt": True}
        assert not is_turn_start(msg)

    def test_injection_does_not_start_turn(self):
        assert not is_turn_start(_injected_msg())

    def test_non_user_roles_do_not_start_turn(self):
        assert not is_turn_start({"role": "assistant", "content": "hi"})
        assert not is_turn_start({"role": "tool", "content": "out"})
        assert not is_turn_start({"role": "system", "content": "x"})

    def test_is_injected(self):
        assert is_injected({"role": "user", "injected": True})
        assert not is_injected({"role": "user"})


class TestContentToText:
    def test_string_passthrough(self):
        assert content_to_text("hello") == "hello"

    def test_none_becomes_empty(self):
        assert content_to_text(None) == ""

    def test_list_joins_text_blocks_and_skips_images(self):
        content = [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": PNG_URI}},
            {"type": "text", "text": "b"},
        ]
        assert content_to_text(content) == "a b"

    def test_empty_list_becomes_empty(self):
        assert content_to_text([]) == ""


class TestMarkInjectedMessages:
    def test_marks_unmarked_list_content_user_message(self):
        messages = [_injected_msg(injected=False)]
        assert mark_injected_messages(messages) == 1
        assert messages[0]["injected"] is True

    def test_leaves_marked_string_and_non_user_alone(self):
        messages = [
            _injected_msg(injected=True),
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": []},
        ]
        assert mark_injected_messages(messages) == 0


# ── injection is flagged where it is created ───────────────────────────────


class TestInjectionMarking:
    @pytest.mark.asyncio
    async def test_image_result_injection_is_flagged(self, native_vision):
        agent = _make_agent()
        tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(text="file text", images=[PNG_URI]), "read_file", "t1"
        )
        assert tool_msg["role"] == "tool"
        assert len(extras) == 1
        assert extras[0]["role"] == "user"
        assert extras[0]["injected"] is True
        assert any(b.get("type") == "image_url" for b in extras[0]["content"])

    @pytest.mark.asyncio
    async def test_text_only_result_injects_nothing(self, native_vision):
        agent = _make_agent()
        _tool_msg, extras = await agent._build_tool_result_content(
            "just text", "read_file", "t1"
        )
        assert extras == []

    def test_injected_flag_never_sent_to_api(self):
        messages = _history_turn()
        api_messages = build_messages_with_system(messages)
        multimodal = [m for m in api_messages if isinstance(m.get("content"), list)]
        assert multimodal, "expected the injection in the payload"
        for msg in multimodal:
            assert "injected" not in msg
            assert "interrupt" not in msg

    def test_interrupt_flag_still_stripped(self):
        messages = [{"role": "user", "content": "x", "interrupt": True}]
        api_messages = build_messages_with_system(messages)
        assert "interrupt" not in api_messages[1]


# ── grouping / queries treat injection as mid-turn ─────────────────────────


class TestGrouping:
    def test_injection_stays_in_its_turn_group(self):
        history = MessageHistory(_history_turn())
        assert history.get_message_groups() == [[0, 1], [2, 3, 4, 5, 6]]

    def test_find_last_user_idx_skips_injection(self):
        history = MessageHistory(_history_turn())
        assert history.find_last_user_idx() == 2

    def test_count_tool_turns_counts_one(self):
        history = MessageHistory(_history_turn())
        assert history.count_tool_turns() == 1

    def test_trim_never_cuts_inside_a_turn(self):
        agent = _make_agent(_history_turn())
        removed = agent.trim_messages(1)
        # Whole second turn kept: the injection is not a cut point
        assert removed == 2
        assert agent.messages[0]["content"] == "read the screenshot"

    def test_compact_absorbs_whole_turn_including_injection(self):
        history = MessageHistory(_history_turn())
        history.compact("Summary: saw a red pixel")
        # First turn untouched; second turn collapses to prompt + summary
        assert [m["role"] for m in history.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert "read the screenshot" in history.messages[2]["content"]
        assert "red pixel" in history.messages[3]["content"]


# ── /retry ─────────────────────────────────────────────────────────────────


class TestRetry:
    def test_retry_targets_the_real_prompt(self):
        agent = _make_agent(_history_turn())
        result = execute_retry(agent)
        assert result.success
        assert result.data["user_text"] == "read the screenshot"
        # The entire turn (tool msg, injection, final answer) is gone
        assert [m["content"] for m in agent.messages] == [
            "first turn",
            "answer 1",
        ]

    def test_retry_survives_unmarked_injection_at_group_start(self):
        """Legacy sessions (saved before the flag existed) must not crash.

        Without the marker the injection still opens a group; /retry used to
        raise AttributeError on the list content. It now offers the flattened
        text instead.
        """
        agent = _make_agent(
            [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "t1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "t1", "content": "text"},
                _injected_msg(injected=False),
                {"role": "assistant", "content": "answer"},
            ]
        )
        result = execute_retry(agent)
        assert result.success
        assert "[Image from tool: read_file]" in result.data["user_text"]

    def test_retry_strips_journal_prefix(self):
        agent = _make_agent(
            [{"role": "user", "content": '[previous user message] "original ask"'}]
        )
        result = execute_retry(agent)
        assert result.data["user_text"] == "original ask"


# ── /history rendering data ────────────────────────────────────────────────


class TestHistoryDisplay:
    def test_entries_are_flat_text_and_injection_flagged(self):
        agent = _make_agent(_history_turn())
        groups = format_history_groups(agent)
        assert len(groups) == 2
        assert groups[1].first_content == "read the screenshot"
        injected = [e for e in groups[1].entries if e.is_injected]
        assert len(injected) == 1
        assert injected[0].role == "user"
        assert isinstance(injected[0].content, str)
        assert "Image from tool" in injected[0].content
        # Nothing leaks a content-block list into the UI layer
        for group in groups:
            assert isinstance(group.first_content, str)
            for entry in group.entries:
                assert isinstance(entry.content, str)


# ── loading an old session ─────────────────────────────────────────────────


class TestLegacySessionLoad:
    def test_load_context_marks_injections(self, tmp_path):
        """Old .ctx files get the flag inferred on load."""
        messages = _history_turn()
        for msg in messages:
            msg.pop("injected", None)  # simulate a pre-fix save
        agent = _make_agent(messages)
        path = tmp_path / "old.ctx"
        save_context(agent, path)

        fresh = _make_agent([])
        ok, _msg, _incomplete = load_context(fresh, path)
        assert ok
        assert any(m.get("injected") for m in fresh.messages)
        assert fresh.history.get_message_groups() == [[0, 1], [2, 3, 4, 5, 6]]
