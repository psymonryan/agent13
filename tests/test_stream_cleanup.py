"""Test that streaming responses don't leak async generators.

The OpenAI SDK's AsyncStream creates a chain of internal async generators
that are never closed by the SDK. During asyncio.shutdown_asyncgens()
(called by asyncio.run() on exit), these leaked generators cause
httpcore2 to raise RuntimeError: 'generator didn't stop after athrow()'.

close_tracked_asyncgens() proactively closes them before shutdown.
"""
import asyncio
import gc
import json
import logging

import pytest
from pytest_httpserver import HTTPServer

from openai import AsyncOpenAI
from agent13.llm import stream_response_with_tools, close_tracked_asyncgens


def make_sse_response(model="test-model"):
    """Build SSE-formatted response chunks simulating a chat completion stream."""
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
        {"choices": [{"delta": {"content": " world"}, "index": 0}]},
        {"choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": "stop"}]},
        {"choices": [{"delta": {}, "index": 0, "finish_reason": None}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    lines = []
    for chunk in chunks:
        chunk["id"] = "chatcmpl-test"
        chunk["model"] = model
        chunk["object"] = "chat.completion.chunk"
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


@pytest.fixture
def openai_client(httpserver: HTTPServer):
    return AsyncOpenAI(
        api_key="test-key",
        base_url=httpserver.url_for("/v1"),
        max_retries=0,
    )


def _setup_stream_endpoint(httpserver: HTTPServer):
    sse_body = make_sse_response()
    httpserver.expect_request(
        "/v1/chat/completions", method="POST"
    ).respond_with_data(
        sse_body,
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _count_tracked_asyncgens():
    loop = asyncio.get_event_loop()
    return len(list(loop._asyncgens))


class TestStreamCleanup:
    """Verify that streaming doesn't leak async generators."""

    @pytest.mark.asyncio
    async def test_generators_leak_after_stream(
        self, httpserver: HTTPServer, openai_client: AsyncOpenAI
    ):
        """SDK leaks internal async generators after stream consumption."""
        _setup_stream_endpoint(httpserver)

        async for event_type, data in stream_response_with_tools(
            openai_client, "test-model", [{"role": "user", "content": "hi"}]
        ):
            pass

        await openai_client.close()
        gc.collect()

        # SDK leaks ~12 internal generators (AsyncStream.__stream__,
        # SSEDecoder, httpx Response.aiter_bytes/raw, httpcore2
        # PoolByteStream, HTTP11ConnectionByteStream, etc)
        assert _count_tracked_asyncgens() > 0

    @pytest.mark.asyncio
    async def test_close_tracked_asyncgens_cleans_up(
        self, httpserver: HTTPServer, openai_client: AsyncOpenAI
    ):
        """close_tracked_asyncgens closes the leaked generators."""
        _setup_stream_endpoint(httpserver)

        async for event_type, data in stream_response_with_tools(
            openai_client, "test-model", [{"role": "user", "content": "hi"}]
        ):
            pass

        await openai_client.close()
        assert _count_tracked_asyncgens() > 0  # leaked

        await close_tracked_asyncgens()

        # Most are closed; 2-3 may remain (already-closed but still
        # weakly referenced) but they're harmless
        remaining = _count_tracked_asyncgens()
        assert remaining <= 3, f"{remaining} generators still tracked after cleanup"

    @pytest.mark.asyncio
    async def test_no_shutdown_errors_after_cleanup(
        self, httpserver: HTTPServer, openai_client: AsyncOpenAI
    ):
        """After close_tracked_asyncgens, shutdown_asyncgens is error-free."""
        _setup_stream_endpoint(httpserver)

        async for event_type, data in stream_response_with_tools(
            openai_client, "test-model", [{"role": "user", "content": "hi"}]
        ):
            pass

        await openai_client.close()
        await close_tracked_asyncgens()

        # Capture asyncio logger output during shutdown
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Handler()
        logger = logging.getLogger("asyncio")
        logger.addHandler(handler)
        try:
            await asyncio.get_event_loop().shutdown_asyncgens()
        finally:
            logger.removeHandler(handler)

        errors = [r for r in records if "generator" in r.getMessage().lower()]
        assert len(errors) == 0, (
            f"shutdown_asyncgens produced {len(errors)} errors:\n"
            + "\n".join(r.getMessage()[:200] for r in errors)
        )
