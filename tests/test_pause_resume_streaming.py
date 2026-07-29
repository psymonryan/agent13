"""Tests for /pause and /resume commands in REPL mode.

The streaming pause/resume test is inherently timing-dependent and is
tested manually. These tests cover the deterministic edge cases.
"""

import os
import json
from .helpers import spawn_process
import pytest
import pytest_httpserver
from werkzeug.wrappers import Response


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_server():
    server = pytest_httpserver.HTTPServer()

    def models_handler(request):
        return Response(
            json.dumps({"data": [{"id": "mock-model", "object": "model", "owned_by": "test"}]}),
            content_type="application/json",
        )

    def chat_handler(request):
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        last_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_msg = msg.get("content", "")
                break
        content = "Hello! I'm a mock assistant." if "hello" in last_msg.lower() else f"Received: {last_msg[:50]}"

        sse = ""
        chunk = {"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
        sse += f"data: {json.dumps(chunk)}\n\n"
        final = {"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        sse += f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"
        return Response(sse, content_type="text/event-stream")

    server.expect_request("/v1/models").respond_with_handler(models_handler)
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(chat_handler)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def repl_env(tmp_path, mock_llm_server):
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()

    (config_dir / "config.toml").write_text(
        f'[saves]\nlocation = "central"\n\n[[providers]]\nname = "test_mock"\n'
        f'api_base = "http://localhost:{mock_llm_server.port}/v1"\napi_key = "test-key"\n'
    )

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(saves_dir)
    return env


def spawn_repl(repl_env):
    proc = spawn_process(
        "uv",
        args=["run", "agent13", "test_mock", "--repl", "--model", "mock-model"],
        env=repl_env,
        encoding="utf-8",
        timeout=30,
        dimensions=(50, 200),
        maxread=4096,
    )
    proc.timeout = 30
    proc.expect(">", timeout=15)
    return proc


# ── Tests ────────────────────────────────────────────────────────────


class TestPauseResumeEdgeCases:

    def test_pause_when_idle(self, repl_env):
        """/pause when agent is idle shows 'nothing to pause'."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/pause")
            proc.expect(r"(?i)nothing|idle|not processing", timeout=5)
        finally:
            proc.sendline("/quit")
            proc.close()

    def test_resume_when_not_paused(self, repl_env):
        """/resume when not paused shows 'not paused'."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/resume")
            proc.expect(r"(?i)not paused", timeout=5)
        finally:
            proc.sendline("/quit")
            proc.close()

    def test_completion_after_response(self, repl_env):
        """Completion message and prompt appear after normal response."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("hello")
            proc.expect(r"\[complete", timeout=20)
            proc.expect(">", timeout=5)
        finally:
            proc.sendline("/quit")
            proc.close()
