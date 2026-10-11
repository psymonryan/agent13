"""Integration test for the --focus startup preprocess turn (LLM mocked).

Spawns a REAL REPL process with --focus NAME against a mock LLM server that
captures every chat request, then verifies the user's actual experience:

  1. The records (current truth + last journal entry + ack instruction)
     hit the wire as the FIRST request - with NO user input at all.
  2. The model's ack streams into the chat while the input loop is live.
  3. A prompt typed afterwards is processed as the SECOND turn, with the
     records still in the history and the ack carried as an assistant
     message (the warm-context ordering).
"""

import json
import os
import time

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

FOCUS_NAME = "preprocess_focus"
TRUTH_MARKER = "PREPROCESS-TRUTH-MARKER-12345"
JOURNAL_MARKER = "PREPROCESS-JOURNAL-MARKER-67890"
ACK_INSTRUCTION = f'Reply with "Records for focus {FOCUS_NAME} loaded."'


@pytest.fixture
def preprocess_llm_server():
    """Mock LLM server: plain-text replies with unique "Mock response N"
    markers (the text_only pattern), capturing every request body."""
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        usage = {
            "prompt_tokens": 1_000,
            "completion_tokens": 10,
            "total_tokens": 1_010,
        }
        content = f"Mock response {n}. Plain text, no tool calls."
        parts = [
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            },
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]

        sse = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
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
def preprocess_env(tmp_path, preprocess_llm_server):
    """Temp config dir pointing at the mock server, temp cwd carrying the
    focus's record files.

    Threshold 200_000 sits far above the mock's reported usage (1_000), so
    NO auto-context machinery can fire - the captured requests are exactly
    the preprocess turn and the user's turn, nothing else.
    """
    server = preprocess_llm_server[0]

    # The focus's durable records in the temp cwd.
    docs = tmp_path / "docs_archive"
    docs.mkdir()
    (docs / f"{FOCUS_NAME}_current_truth.md").write_text(
        f"# truth\n\n{TRUTH_MARKER}\n", encoding="utf-8"
    )
    (docs / f"{FOCUS_NAME}_journal.md").write_text(
        f"# journal\n\n## 2026-01-01 00:00 - old entry\n\n## 2026-01-02 00:00 - last entry\n\n{JOURNAL_MARKER}\n",
        encoding="utf-8",
    )

    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 200_000
action = "report_and_compact"
chain = 0
"""
    (config_dir / "config.toml").write_text(config_content)

    # The REPL must run in the temp cwd so it finds docs_archive/.
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env, tmp_path


def spawn_repl(env, timeout=30, extra_args=None):
    args = [
        "run",
        "agent13",
        "test_mock",
        "--repl",
        "--model",
        "mock-model",
        "--focus",
        FOCUS_NAME,
    ]
    if extra_args:
        args += extra_args
    proc = spawn_process(
        "uv",
        args=args,
        env=env,
        encoding="utf-8",
        timeout=timeout,
        dimensions=(50, 200),
        maxread=200_000,
    )
    proc.timeout = timeout
    proc.expect(r">", timeout=timeout)
    return proc


def all_user_messages(body):
    return [
        m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user"
    ]


def last_user_message(body):
    users = all_user_messages(body)
    return users[-1] if users else ""


class TestFocusPreprocessExperience:
    def test_records_hit_the_wire_before_any_user_input(
        self, preprocess_env, preprocess_llm_server, monkeypatch
    ):
        """The preprocess turn runs WITHOUT user input: request 1 carries
        the records + ack instruction; the ack streams to the chat; a
        prompt typed afterwards is turn 2 on a history that still holds
        the records and the ack."""
        _, captured = preprocess_llm_server
        env, tmp = preprocess_env
        # The spawned REPL inherits this process's cwd - it must run in the
        # temp dir to find docs_archive/<focus>_*.
        monkeypatch.chdir(tmp)
        proc = spawn_repl(env)

        try:
            # The preprocess turn completes on its own - no sendline yet.
            # Gate on the WIRE, not the pty: the REPL sits in INPUT mode
            # (output frozen) until a message is dispatched, so the ack is
            # buffered, not rendered.
            deadline = time.monotonic() + 30
            while not captured and time.monotonic() < deadline:
                time.sleep(0.1)
            assert captured, "the preprocess turn never hit the wire"
            assert len(captured) == 1, (
                f"expected exactly 1 LLM call (the preprocess turn) before "
                f"any user input, got {len(captured)}"
            )
            first = captured[0]
            last = last_user_message(first)
            assert "Durable record for focus " + FOCUS_NAME in last
            assert TRUTH_MARKER in last
            # The LAST journal entry only (not the whole journal file).
            assert JOURNAL_MARKER in last
            assert "old entry" not in last
            assert ACK_INSTRUCTION in last

            # Now the user types: turn 2, on the warm history. The dispatch
            # switches the REPL to streaming, which also flushes the
            # buffered preprocess output - "Mock response 2" is the gate.
            proc.sendline("hello")
            proc.expect("Mock response 2", timeout=30)
            proc.expect(r">", timeout=10)

            assert len(captured) == 2
            second = captured[1]
            assert last_user_message(second) == "hello"
            # The records are still in the history...
            users = all_user_messages(second)
            assert any(TRUTH_MARKER in u for u in users)
            # ...and the preprocess ack was carried as an assistant message.
            assistant_texts = [
                m.get("content", "")
                for m in second.get("messages", [])
                if m.get("role") == "assistant"
            ]
            assert any("Mock response 1" in a for a in assistant_texts)
        finally:
            proc.close()
