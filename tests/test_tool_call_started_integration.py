"""Integration tests for the TOOL_CALL_STARTED feature.

Only the LLM is mocked (via pytest-httpserver). Covers:
- stream_response_with_tools yields the early ("tool_call", name) event
  BEFORE ("tool_calls_complete", ...) — the signal the core change relies on.
- The real AgentTUI shows the dimmed name-only widget the moment the tool
  name is known, updates it in place when args complete (no duplicate), then
  shows the result.
"""

import json

import pytest
from pytest_httpserver import HTTPServer

from openai import AsyncOpenAI
from agent13.llm import stream_response_with_tools
from ui.tui import (
    AgentTUI,
    ToolCallStartedMessage,
    ToolCallMessage,
    ToolResultMessage,
)


# =============================================================================
# LLM layer: early name event precedes the complete event
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


def make_tool_call_sse() -> str:
    """SSE stream: content, then a tool-call name chunk, then args, then done."""
    parts = [
        _chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {"content": "Writing"}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "square_number", "arguments": ""}}]}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": None, "type": "function", "function": {"name": None, "arguments": '{"x": 5}'}}]}, "finish_reason": None}]}),
        _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        _chunk({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}),
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


class TestStreamYieldsEarlyToolCall:
    @pytest.mark.asyncio
    async def test_name_event_precedes_complete(self, httpserver, openai_client):
        httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_data(
            make_tool_call_sse(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

        events = []
        async for event_type, data in stream_response_with_tools(
            openai_client,
            "test-model",
            [{"role": "user", "content": "square 5"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "square_number", "parameters": {}},
                }
            ],
        ):
            events.append((event_type, data))

        types = [t for t, _ in events]
        assert "tool_call" in types, f"early tool_call event missing: {types}"
        assert "tool_calls_complete" in types
        assert types.index("tool_call") < types.index("tool_calls_complete"), (
            "early name event must precede the complete event"
        )

        early = [d for t, d in events if t == "tool_call"][0]
        assert early["name"] == "square_number"
        assert early["id"] == "call_1"

        complete = [d for t, d in events if t == "tool_calls_complete"][0]
        tc = complete["tool_calls"][0]
        assert tc["name"] == "square_number"
        assert tc["arguments"] == '{"x": 5}'


# =============================================================================
# TUI: pending widget appears, updates in place, then result
# =============================================================================


async def _wait_for(pilot, condition, *, steps: int = 300) -> None:
    """Poll (via pilot.pause) until condition() is true, else raise."""
    for _ in range(steps):
        if condition():
            return
        await pilot.pause()
    raise AssertionError("condition not met within polling budget")


class TestTUIPendingWidget:
    @pytest.mark.asyncio
    async def test_pending_widget_shown_before_result(self):
        # Client is never called — we drive the TUI by posting messages directly.
        client = AsyncOpenAI(api_key="test", base_url="http://localhost:1/v1")
        tui = AgentTUI(client=client, model="test-model", model_names=[])

        async with tui.run_test() as pilot:
            chat = tui.query_one("#chat")

            def tool_call_widgets():
                return [w for w in chat.children if "tool-call" in set(w.classes)]

            # 1. Tool call started (name only) -> dimmed pending widget appears
            tui.post_message(ToolCallStartedMessage("write_file"))
            await _wait_for(pilot, lambda: len(tool_call_widgets()) == 1)
            w = tool_call_widgets()[0]
            assert "write_file" in w.content
            assert "…" in w.content, "pending widget should show the ellipsis"

            # 2. Tool call complete -> SAME widget updated in place (no duplicate)
            tui.post_message(ToolCallMessage("write_file", {"filepath": "test.py"}))
            await _wait_for(
                pilot,
                lambda: "test.py" in (
                    tool_call_widgets()[0].content if tool_call_widgets() else ""
                ),
            )
            assert len(tool_call_widgets()) == 1, (
                "widget must update in place, not duplicate"
            )
            assert "test.py" in tool_call_widgets()[0].content
            assert "…" not in tool_call_widgets()[0].content

            # 3. Result -> result widget appears
            tui.post_message(ToolResultMessage("write_file", '{"success": true}'))
            await _wait_for(
                pilot,
                lambda: any("tool-result" in set(w.classes) for w in chat.children),
            )
