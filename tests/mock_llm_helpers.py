"""Shared mock LLM server helpers for tests.

Provides handler factories for mock OpenAI-compatible endpoints
and fixtures for spinning up mock servers with temp config.
"""

import json
from werkzeug import Request, Response


# ── Handler factories ─────────────────────────────────────────────────────


def make_models_handler():
    """Handler for GET /v1/models."""

    def handler(request: Request):
        return Response(
            json.dumps(
                {
                    "data": [
                        {"id": "mock-model", "object": "model", "owned_by": "test"}
                    ]
                }
            ),
            content_type="application/json",
        )

    return handler


def make_chat_handler():
    """Handler for POST /v1/chat/completions.

    Returns SSE streaming response matching OpenAI format.
    Matches keywords in the last user message and returns canned responses.
    """

    def handler(request: Request):
        body = request.get_json(force=True)
        messages = body.get("messages", [])

        last_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_msg = msg.get("content", "")
                break

        if "hello" in last_msg.lower():
            content = "Hello! I'm a mock assistant."
        elif "error" in last_msg.lower():
            content = "I encountered an error."
        elif "2+2" in last_msg:
            content = "4"
        elif "squared" in last_msg.lower():
            content = "25"
        elif "test passed" in last_msg.lower():
            content = "test passed"
        elif "count" in last_msg.lower():
            content = "1, 2, 3, 4, 5 — that's the count!"
        elif "done" in last_msg.lower():
            content = "done"
        else:
            content = f"Received: {last_msg[:50]}"

        # Build SSE streaming response
        sse_parts = []

        chunk_data = {
            "id": "mock-completion",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        sse_parts.append(f"data: {json.dumps(chunk_data)}\n\n")

        final_chunk = {
            "id": "mock-completion",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        sse_parts.append(f"data: {json.dumps(final_chunk)}\n\n")
        sse_parts.append("data: [DONE]\n\n")

        sse_body = "".join(sse_parts)

        return Response(
            sse_body,
            content_type="text/event-stream",
        )

    return handler
