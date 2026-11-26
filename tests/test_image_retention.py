"""Images must survive a turn end; only history replacement discards them.

Bug (work order 2026-10-19): ``_strip_images_from_messages()`` ran at turn
end and rewrote every image-bearing message in place. The prompt is
otherwise strictly append-only (527 of 532 consecutive request pairs in the
measured session changed zero messages), so that rewrite broke the
provider's prefix cache from the oldest image onward - measured 415.9s
first-token latency against a 6.8s warm median, to save 3.5% of context.

Fix: the turn-end strip is gone (both call sites) and the method with it.
Images live for the whole turn and are discarded wholesale by the
history-replacement paths (compact_history, _mechanical_restart, journal
compact), where the prefix changes by design and the re-prefill is paid
regardless.

Structural note: a tool result's image is NOT stored in the tool message.
``_build_tool_result_content`` returns the text in a ``role: tool`` message
plus the image blocks in a SEPARATE ``role: user`` message flagged
``injected`` - the OpenAI API requires images in user role. That is why the
removed strip only ever looked at user messages.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent13.core import Agent

# 1x1 red PNG (fully decodable).
PNG_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def make_agent():
    """A minimal agent: no auto-context, snapshots mocked out."""
    agent = Agent(
        client=MagicMock(),
        model="test-model",
        auto_context_threshold=0,
        auto_context_nudge_cap=2,
    )
    agent._save_auto_context_snapshot = MagicMock()
    agent._save_degenerate_snapshot = MagicMock()
    return agent


def injected_image_msg():
    """The native-vision injection core.py builds for a tool-result image."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "[Image from tool: read_file]"},
            {"type": "image_url", "image_url": {"url": PNG_URI}},
        ],
        "injected": True,
    }


def has_image_block(messages):
    """True if any message still carries an image_url content block."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


def final_stream(content="final answer"):
    async def stream(*args, **kwargs):
        yield ("content", content)
        yield ("finish", {"finish_reason": "stop"})

    return stream


def cap_hit_stream(content="partial answer "):
    async def stream(*args, **kwargs):
        yield ("content", content)
        yield ("finish", {"finish_reason": "length"})

    return stream


class TestImagesSurviveTurnEnd:
    """Regression tests for the 2026-10-19 prefix-cache break."""

    @pytest.mark.asyncio
    async def test_normal_turn_end_keeps_image(self, tmp_path, monkeypatch):
        """A final response must not rewrite the image message in history."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()
        agent._running = True
        agent.messages = [
            {"role": "user", "content": "read the screenshot"},
            injected_image_msg(),
        ]

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=final_stream(),
        ):
            await agent._llm_turn()

        assert has_image_block(agent.messages), (
            "turn end stripped the image - this reintroduces the prefix-cache "
            "break (415.9s first token, measured 2026-10-19)"
        )
        # The payload itself must be intact, not a text placeholder.
        assert PNG_URI in str(agent.messages)

    @pytest.mark.asyncio
    async def test_degenerate_stop_keeps_image(self, tmp_path, monkeypatch):
        """The give-up path must not strip images either."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent()  # nudge cap 2
        agent._running = True
        agent.messages = [
            {"role": "user", "content": "go"},
            injected_image_msg(),
        ]

        streams = iter([cap_hit_stream("first try "), cap_hit_stream("second try ")])

        def mock_stream(*args, **kwargs):
            return next(streams)()

        with patch(
            "agent13.llm.stream_response_with_tools",
            side_effect=mock_stream,
        ):
            await agent._llm_turn()

        # Prove the stop path actually ran before asserting on the image.
        assert agent.has_incomplete_turn is True
        assert has_image_block(agent.messages)


class TestCompactionDiscardsImages:
    """The boundary we now rely on: compaction replaces history wholesale."""

    @pytest.mark.asyncio
    async def test_compact_history_drops_images(self):
        agent = make_agent()
        agent.system_prompt = "S" * 400
        agent.messages = [
            {"role": "user", "content": "read the screenshot"},
            injected_image_msg(),
            {"role": "assistant", "content": "I see a red pixel."},
        ]

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            yield ("content", "Session summary: the user showed a screenshot.")
            yield (
                "token_usage",
                {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            )

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            ok, _ = await agent.compact_history("")

        assert ok
        assert not has_image_block(agent.messages), (
            "compaction must discard images - it is the only strip boundary left"
        )
