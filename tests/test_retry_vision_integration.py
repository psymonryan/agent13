"""Integration: /retry after a turn that showed the model an image.

Only the LLM is mocked (pytest-httpserver). The user experience under test:

  1. Ask something that makes the model call a tool returning an image.
  2. The image is injected mid-turn as a user-role message (native vision).
  3. Press /retry — the composer must come back pre-filled with *the user's
     own prompt*, and the whole turn (including the injection) must go.

Before the fix the injection opened its own message group, so /retry offered
"[Image from tool: read_file]" as retry text — and crashed outright with
``AttributeError: 'list' object has no attribute 'startswith'``.
"""

import asyncio
import json
import time

import pytest
import pytest_httpserver
from openai import AsyncOpenAI
from werkzeug.wrappers import Response

from agent13.commands import execute_retry
from agent13.core import Agent
from agent13.message_history import is_turn_start
from tools import ToolResult

# 1x1 red PNG
PNG_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

PROMPT = "describe the screenshot please"


# ── Mock LLM ───────────────────────────────────────────────────────────────


class MockLLM:
    """Tool-calling mock: first call asks for `peek`, next answers in text.

    Records every request body so tests can assert on what the API received.
    """

    def __init__(self):
        self.requests: list[list[dict]] = []

    def chat_handler(self):
        def handler(request):
            body = request.get_json(force=True)
            messages = body.get("messages", [])
            self.requests.append(messages)

            has_image = any(isinstance(m.get("content"), list) for m in messages)
            if has_image:
                return self._sse(self._text_chunks("It is a red square."))
            return self._sse(self._tool_call_chunks())

        return handler

    @staticmethod
    def _sse(chunks):
        parts = [f"data: {json.dumps(c)}\n\n" for c in chunks]
        parts.append("data: [DONE]\n\n")
        return Response("".join(parts), content_type="text/event-stream")

    @staticmethod
    def _text_chunks(text):
        return [
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            },
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]

    @staticmethod
    def _tool_call_chunks():
        return [
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "t1",
                                    "type": "function",
                                    "function": {
                                        "name": "peek",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
            },
        ]


@pytest.fixture
def mock_llm():
    """Mock server; yields (recorded mock, base_url)."""
    llm = MockLLM()
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_json(
        {"data": [{"id": "mock-model", "object": "model", "owned_by": "test"}]}
    )
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        llm.chat_handler()
    )
    server.start()
    yield llm, server.url_for("/v1")
    server.stop()


@pytest.fixture
def native_vision(monkeypatch):
    """Empty config with image routing forced to native (inline injection)."""
    from agent13 import config as config_mod

    cfg = config_mod.Config()
    cfg.vision = None  # no [vision] section -> native injection
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    return cfg


def _make_agent(base_url):
    client = AsyncOpenAI(base_url=base_url, api_key="test-key")

    async def execute_tool(name, args):
        return ToolResult(text="screenshot bytes read", images=[PNG_URI])

    return Agent(client=client, model="mock-model", execute_tool=execute_tool)


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_after_image_turn_offers_original_prompt(mock_llm, native_vision):
    """Full loop: image turn -> /retry -> same text re-queued, turn deleted."""
    llm, base_url = mock_llm
    agent = _make_agent(base_url)

    agent.queue.add(PROMPT)
    task = asyncio.create_task(agent.run())
    try:
        assert await _wait_for(
            lambda: agent.is_idle and len(agent.messages) >= 5
        ), "turn did not finish"

        # The image was injected mid-turn as a user-role message...
        injections = [
            m for m in agent.messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert len(injections) == 1
        # ...flagged as mid-turn, so the turn is still a single group
        assert injections[0]["injected"] is True
        groups = agent.history.get_message_groups()
        assert len(groups) == 1
        assert len(groups[0]) == len(agent.messages)
        assert sum(1 for m in agent.messages if is_turn_start(m)) == 1

        # No local flags leaked into any API call
        for sent in llm.requests:
            for msg in sent:
                assert "injected" not in msg
                assert "interrupt" not in msg

        # /retry hands back the user's own words, whole turn gone
        result = execute_retry(agent)
        assert result.success, result.message
        assert result.data["user_text"] == PROMPT
        assert agent.messages == []

        # Re-queue it exactly as the user typed it
        await agent.add_message(result.data["user_text"])
        assert await _wait_for(
            lambda: agent.is_idle and len(llm.requests) >= 3
        ), "retry turn did not run"

        # First call of the retried turn carries the prompt verbatim
        # (requests: 1=turn, 2=after tool result, 3=retry, 4=retry after tool)
        retry_call = llm.requests[2]
        user_msgs = [m for m in retry_call if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[-1]["content"] == PROMPT
    finally:
        agent.stop()
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_injected_image_reaches_the_model_in_a_user_message(
    mock_llm, native_vision
):
    """The image must arrive as image blocks in a user-role message."""
    llm, base_url = mock_llm
    agent = _make_agent(base_url)

    agent.queue.add(PROMPT)
    task = asyncio.create_task(agent.run())
    try:
        assert await _wait_for(
            lambda: len(llm.requests) >= 2
        ), "model was never called again with the image"

        second_call = llm.requests[1]
        image_msgs = [m for m in second_call if isinstance(m.get("content"), list)]
        assert len(image_msgs) == 1
        assert image_msgs[0]["role"] == "user"
        assert any(b.get("type") == "image_url" for b in image_msgs[0]["content"])
        # ...and it sits right after the tool result it belongs to
        idx = second_call.index(image_msgs[0])
        assert second_call[idx - 1]["role"] == "tool"
    finally:
        agent.stop()
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
