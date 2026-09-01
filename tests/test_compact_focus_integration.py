"""Integration test for /compact focus steering (LLM mocked only).

Spawns a REAL REPL process against a mock LLM server that captures every
chat request, then verifies the compaction call that actually hits the
wire contains the steering block — i.e. what the user experiences.

The mock returns a unique "Mock response N" marker per call so each
pexpect barrier is race-free (the REPL output buffer retains earlier
responses, so generic markers would match stale text).
"""

import json
import os

import pexpect  # noqa: F401 (pexpect.EOF referenced in except clauses)
import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def capturing_llm_server():
    """Mock LLM server that records every chat request body.

    Yields (server, captured) where captured is a list of request bodies
    in arrival order. Each response carries a unique "Mock response N"
    marker so tests can wait for a specific turn unambiguously.
    """
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        content = (
            f"Mock response {n}.\n"
            "## Session Intent\nCompacted summary of the work so far."
        )
        chunk = {
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
        final = {
            "id": "mock-completion",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        sse = (
            f"data: {json.dumps(chunk)}\n\n"
            f"data: {json.dumps(final)}\n\n"
            "data: [DONE]\n\n"
        )
        return Response(sse, content_type="text/event-stream")

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def repl_env(tmp_path, capturing_llm_server):
    """Temp config dir pointing at the capturing mock server."""
    server = capturing_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


def spawn_repl(env, timeout=30):
    proc = spawn_process(
        "uv",
        args=["run", "agent13", "test_mock", "--repl", "--model", "mock-model"],
        env=env,
        encoding="utf-8",
        timeout=timeout,
        dimensions=(50, 200),
        maxread=4096,
    )
    proc.timeout = timeout
    proc.expect(r">", timeout=timeout)
    return proc


def last_user_message(body):
    """Return the last user-role message content from a chat request body."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def wait_for_compact_complete(proc, timeout=30):
    """Wait for the 'Compacted N→M words' result line.

    /compact queues the compaction and returns to the prompt immediately —
    the LLM call happens asynchronously. The result line (from the
    JOURNAL_RESULT event) is only printed after compaction fully completes,
    so it's the safe barrier before asserting on captured requests.
    """
    proc.expect(r"Compacted \d+→\d+ words", timeout=timeout)


# ── Tests ──────────────────────────────────────────────────────────────────


class TestCompactFocusExperience:
    """User sends /compact <focus> and the steering reaches the LLM."""

    def test_focus_reaches_compaction_call(self, repl_env, capturing_llm_server):
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            # Establish a conversation turn first
            proc.sendline("hello")
            proc.expect("Mock response 1", timeout=15)

            # Compact with a focus
            proc.sendline("/compact work on the server side rendering feature next")
            wait_for_compact_complete(proc)

            assert len(captured) >= 2, f"expected 2+ LLM calls, got {len(captured)}"
            compact_request = captured[-1]
            prompt = last_user_message(compact_request)

            # Base compaction prompt intact...
            assert prompt.startswith("Summarize our conversation so far")
            # ...with the steering block appended
            assert "Next task: work on the server side rendering feature next" in prompt
            assert "Organize the summary sections around that task" in prompt
            assert 'Make "next steps" concrete actions toward it' in prompt
            assert "Details clearly unrelated to it can be compressed harder" in prompt
        finally:
            proc.close()

    def test_bare_compact_has_no_steering(self, repl_env, capturing_llm_server):
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("Mock response 1", timeout=15)

            proc.sendline("/compact")
            wait_for_compact_complete(proc)

            assert len(captured) >= 2
            prompt = last_user_message(captured[-1])
            assert prompt.startswith("Summarize our conversation so far")
            assert "Next task:" not in prompt
        finally:
            proc.close()

    def test_summary_is_displayed_and_history_replaced(
        self, repl_env, capturing_llm_server
    ):
        """After compaction the user sees the summary and context is fresh."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("Mock response 1", timeout=15)

            proc.sendline("/compact focus on the API next")
            # The streamed compact summary is visible to the user
            proc.expect("Mock response 2", timeout=30)
            wait_for_compact_complete(proc)

            # Next turn sends only the lightweight replacement history,
            # not the pre-compact conversation
            proc.sendline("continue")
            proc.expect("Mock response 3", timeout=15)

            next_turn = captured[-1]
            user_msgs = [
                m.get("content", "")
                for m in next_turn.get("messages", [])
                if m.get("role") == "user"
            ]
            assert "Give me a summary of our previous session" in user_msgs
            assert "hello" not in user_msgs
        finally:
            proc.close()
