"""Integration tests for the TOOL_CALL_PENDING feature.

Only the LLM is mocked (via pytest-httpserver). Covers what the user actually
experiences with a provider that buffers whole tool calls (mlx-lm):

- The real llm layer yields keepalive events for the payload-less chunks the
  provider sends while buffering, and still delivers the tool call intact.
- The real AgentTUI mounts the dimmed "⚙ …" placeholder on
  ToolCallPendingMessage, upgrades it in place when the name arrives
  (ToolCallStartedMessage), finalizes it in place when args complete
  (ToolCallMessage) - one widget, never a duplicate.
- A false alarm (placeholder mounted, then content arrives) is cleanly
  discarded.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from openai import AsyncOpenAI
from agent13.llm import stream_response_with_tools
from ui.tui import (
    AgentTUI,
    TokenMessage,
    ToolCallMessage,
    ToolCallPendingMessage,
    ToolCallStartedMessage,
)


# =============================================================================
# LLM layer: buffered provider stream
# =============================================================================


def _chunk(payload: dict) -> str:
    base = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
    }
    base.update(payload)
    return f"data: {json.dumps(base)}\n\n"


def make_buffered_tool_call_sse() -> str:
    """SSE stream shaped exactly like the mlx-lm probe: role chunk, role-only
    keep-alives while the tool call is buffered, then name+args in ONE chunk."""
    parts = [
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        # keep-alives during buffered generation
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        _chunk(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        ),
        # the whole tool call arrives at once
        _chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path": "big.txt", "content": "line 1\\nline 2"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        _chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            }
        ),
    ]
    parts.append("data: [DONE]\n\n")
    return "".join(parts)


@pytest.fixture
def openai_client(httpserver: HTTPServer) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url=httpserver.url_for("/v1"),
        max_retries=0,
    )


class TestStreamYieldsKeepalives:
    @pytest.mark.asyncio
    async def test_buffered_stream_keepalives_then_intact_tool_call(
        self, httpserver, openai_client
    ):
        httpserver.expect_request(
            "/v1/chat/completions", method="POST"
        ).respond_with_data(
            make_buffered_tool_call_sse(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

        events = []
        async for event_type, data in stream_response_with_tools(
            openai_client,
            "test-model",
            [{"role": "user", "content": "write a big file"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "write_file", "parameters": {}},
                }
            ],
        ):
            events.append((event_type, data))

        types = [t for t, _ in events]
        assert types.count("keepalive") == 3, f"keepalive events missing: {types}"
        # The buffered tool call is delivered intact despite the keep-alives
        complete = [d for t, d in events if t == "tool_calls_complete"][0]
        tc = complete["tool_calls"][0]
        assert tc["name"] == "write_file"
        assert json.loads(tc["arguments"])["path"] == "big.txt"


# =============================================================================
# TUI: placeholder -> named -> complete, all in one widget
# =============================================================================


async def _wait_for(pilot, condition, *, steps: int = 300) -> None:
    """Poll (via pilot.pause) until condition() is true, else raise."""
    for _ in range(steps):
        if condition():
            return
        await pilot.pause()
    raise AssertionError("condition not met within polling budget")


class TestTUIPlaceholderWidget:
    @pytest.fixture
    def tui(self):
        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1/v1")
        return AgentTUI(client=client, model="test-model", model_names=[])

    @pytest.mark.asyncio
    async def test_placeholder_upgrades_to_named_then_complete(self, tui):
        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # 1. Buffered tool call detected (name unknown) -> "⚙ …" placeholder
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)
            assert tool_call_widgets()[0].content == "[dim]⚙ …[/]"

            # 2. Name becomes known -> SAME widget updated in place
            tui.post_message(ToolCallStartedMessage("write_file"))
            await _wait_for(
                pilot, lambda: "write_file" in tool_call_widgets()[0].content
            )
            assert len(tool_call_widgets()) == 1, (
                "upgrade must not duplicate the widget"
            )
            assert "…" in tool_call_widgets()[0].content

            # 3. Args complete -> SAME widget finalized with args
            tui.post_message(ToolCallMessage("write_file", {"path": "big.txt"}))
            await _wait_for(pilot, lambda: "big.txt" in tool_call_widgets()[0].content)
            assert len(tool_call_widgets()) == 1, (
                "finalize must not duplicate the widget"
            )
            assert "…" not in tool_call_widgets()[0].content

    @pytest.mark.asyncio
    async def test_placeholder_discarded_when_content_arrives(self, tui):
        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # 1. Placeholder mounted during a silent stretch
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)

            # 2. Content starts arriving instead -> false alarm, removed
            tui.post_message(
                TokenMessage(
                    "Actually just thinking",
                    is_reasoning=False,
                    generation=tui._stream_generation,
                )
            )
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 0)

            # ...and the content widget is there instead
            assert any(
                "user-message" not in set(w.classes)
                and "tool-call" not in set(w.classes)
                and "tool-result" not in set(w.classes)
                for w in chat.children
            )

    @pytest.mark.asyncio
    async def test_second_pending_message_is_ignored(self, tui):
        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # Duplicate pending signals must not stack placeholders
            tui.post_message(ToolCallPendingMessage())
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)
            await pilot.pause()
            await pilot.pause()
            assert len(tool_call_widgets()) == 1, "duplicate pending must be ignored"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_get_separate_widgets(self, tui):
        """A placeholder upgrades in place, but a SECOND tool call (parallel
        calls) must mount its own widget, not overwrite the first."""
        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # 1. Placeholder (name unknown)
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)

            # 2. First tool name -> upgrades the placeholder in place
            tui.post_message(ToolCallStartedMessage("write_file"))
            await _wait_for(
                pilot, lambda: "write_file" in tool_call_widgets()[0].content
            )
            assert len(tool_call_widgets()) == 1

            # 3. Second tool name (parallel call) -> NEW widget, first intact
            tui.post_message(ToolCallStartedMessage("read_file"))
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 2)
            assert "write_file" in tool_call_widgets()[0].content, (
                "first named widget must not be overwritten by the second call"
            )
            assert "read_file" in tool_call_widgets()[1].content

            # 4. Both complete -> each finalized in FIFO order, no duplicates
            tui.post_message(ToolCallMessage("write_file", {"path": "a.txt"}))
            tui.post_message(ToolCallMessage("read_file", {"path": "b.txt"}))
            await _wait_for(
                pilot,
                lambda: (
                    len(tool_call_widgets()) == 2
                    and "a.txt" in tool_call_widgets()[0].content
                    and "b.txt" in tool_call_widgets()[1].content
                ),
            )

    @pytest.mark.asyncio
    async def test_placeholder_remounts_after_false_alarm(self, tui):
        """The user-reported pattern: placeholder during a pre-thinking gap
        is discarded when thinking starts (false alarm); a SECOND pending
        signal during the later tool-call silence must mount a NEW
        placeholder, which the tool call then finalizes."""
        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # 1. First silence -> placeholder
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)

            # 2. Thinking starts -> false alarm, discarded
            tui.post_message(
                TokenMessage(
                    "Planning the essay...",
                    is_reasoning=True,
                    generation=tui._stream_generation,
                )
            )
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 0)

            # 3. Second silence (tool call buffering) -> placeholder AGAIN
            tui.post_message(ToolCallPendingMessage())
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)
            assert tool_call_widgets()[0].content == "[dim]⚙ …[/]"

            # 4. Tool call lands -> finalizes the second placeholder in place
            tui.post_message(ToolCallMessage("write_file", {"path": "essay.txt"}))
            await _wait_for(
                pilot,
                lambda: (
                    "essay.txt" in tool_call_widgets()[0].content
                    and "…" not in tool_call_widgets()[0].content
                ),
            )
            assert len(tool_call_widgets()) == 1
