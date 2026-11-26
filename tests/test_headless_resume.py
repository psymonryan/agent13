"""Integration test: headless.py's /resume continues an incomplete turn.

The harness used to print "INCOMPLETE_TURN: use /resume to continue" and then
refuse the very command it recommended: its /resume handler only did
agent.resume() for a *pause*, never request_continue_incomplete(). The REPL
and the TUI both handled the incomplete-turn case, so only headless (and any
pexpect test built on it) silently did nothing.

LLM mocked only. Spawns the real headless.py against a mock server, with a
real .ctx on disk, and asserts the resumed turn actually reaches the wire.
"""

import json
import os
import time

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

REAL_REQUEST = "fix the flaky test in tests/test_foo.py"


def _sse(parts):
    return "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"


def _text_response(text="Resumed and answered."):
    return _sse(
        [
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        ]
    )


@pytest.fixture
def capturing_server():
    captured = []

    def handler(request: Request):
        captured.append(request.get_json(force=True))
        return Response(_text_response(), content_type="text/event-stream")

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        handler
    )
    server.start()
    yield server, captured
    server.stop()


def _write_incomplete_save(saves_dir):
    """A .ctx whose last message is a tool result (an incomplete turn)."""
    saves_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "version": 1,
        "model": "mock-model",
        "session_date": "2026-11-02",
        "messages": [
            {"role": "user", "content": REAL_REQUEST},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "read_file", "arguments": "{}"}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        ],
        "token_usage": {"prompt": 10, "completion": 5},
        "saved_at": "2026-11-02T08:00:00",
        "incomplete_turn": True,
        # The persisted anchor: the resume path restores this instead of
        # scanning the history.
        "in_flight_user_request": REAL_REQUEST,
        "turn_user_message_at": 1_700_000_000.0,
    }
    path = saves_dir / "2026-11-02.ctx"
    path.write_text(json.dumps(ctx))
    return path


def _env(tmp_path, server):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"
"""
    )
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


def test_headless_resume_continues_an_incomplete_turn(capturing_server, tmp_path):
    """/resume on a loaded incomplete turn must actually run the turn."""
    server, captured = capturing_server
    _write_incomplete_save(tmp_path / "saves")
    env = _env(tmp_path, server)

    proc = spawn_process(
        "uv",
        args=[
            "run",
            "headless.py",
            "test_mock",
            "--model",
            "mock-model",
            "--continue",
        ],
        env=env,
        encoding="utf-8",
        timeout=60,
        dimensions=(50, 200),
        maxread=8192,
    )
    proc.timeout = 60
    try:
        proc.expect("READY", timeout=60)
        proc.sendline("/resume")
        # The fix: the harness now signals run() to continue the turn rather
        # than reporting NOT_PAUSED.
        proc.expect("CONTINUING", timeout=30)
        # Gate on the AGENT, not on the line the harness printed: CONTINUING
        # only proves the command was accepted, and /quit sent now races the
        # run loop that has to pick up the pending continue. On macOS the race
        # happened to be lost by /quit; on Windows it won and the turn never
        # reached the model (observed 2026-11-02). Poll the capture list.
        deadline = time.time() + 30
        while not captured and time.time() < deadline:
            time.sleep(0.1)
        proc.sendline("/quit")
        proc.expect("DONE", timeout=30)
    finally:
        proc.close(force=True)

    assert captured, "the resumed turn never reached the model"
    last = [
        m.get("content", "")
        for m in captured[-1].get("messages", [])
        if m.get("role") == "user"
    ][-1]
    assert REAL_REQUEST in last, (
        f"the resumed turn must replay the restored instruction, got: {last[:120]!r}"
    )


def test_headless_resume_still_reports_not_paused_when_nothing_to_do(
    capturing_server, tmp_path
):
    """A fresh session has no incomplete turn: /resume stays NOT_PAUSED."""
    server, captured = capturing_server
    env = _env(tmp_path, server)

    proc = spawn_process(
        "uv",
        args=["run", "headless.py", "test_mock", "--model", "mock-model"],
        env=env,
        encoding="utf-8",
        timeout=60,
        dimensions=(50, 200),
        maxread=8192,
    )
    proc.timeout = 60
    try:
        proc.expect("READY", timeout=60)
        proc.sendline("/resume")
        proc.expect("NOT_PAUSED", timeout=30)
        proc.sendline("/quit")
        proc.expect("DONE", timeout=30)
    finally:
        proc.close(force=True)

    assert not captured, "nothing should have been sent to the model"
