"""Integration tests for the degeneration guard (LLM mocked only).

Spawns a REAL REPL process against a mock LLM server and verifies the
guard end to end:

  1. A cap-hit stream (finish_reason "length") with coherent content is
     nudged: the partial is kept in history, the injected nudge rides the
     wire as the next request's last user message, the model recovers,
     and /status reports the session nudge counter.
  2. Consecutive degenerate ends stop at nudge_cap: the turn ends with
     the stop notice, a <save>_degenerate_N.ctx snapshot exists, and the
     nudge counter counts only actual nudges (not the stop).
"""

import json
import os

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from agent13.prompts import DEGENERATE_NUDGE_PROMPT

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

# A distinctive substring of the injected nudge (recognizes the guard's
# request on the wire without pinning the whole prompt text).
NUDGE_NEEDLE = "cut off before finishing"
PARTIAL_NEEDLE = "partial answer from the model"


def _sse(parts: list) -> str:
    return "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"


def last_user_message(body) -> str:
    users = [
        m.get("content", "")
        for m in body.get("messages", [])
        if m.get("role") == "user"
    ]
    return users[-1] if users else ""


def make_degenerate_server(mode: str):
    """Mock chat handler.

    mode="recovered": the nudged retry answers cleanly (finish stop).
    mode="always": every answer is another cap-hit (the stop path).
    """
    captured: list = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        last = last_user_message(body)
        if NUDGE_NEEDLE in last:
            if mode == "recovered":
                content, finish = f"Recovered fine (call {n}).", "stop"
            else:
                content, finish = f"{PARTIAL_NEEDLE} again {n}", "length"
        elif "degenerate" in last:
            # The test prompt itself; answer with a coherent cap-hit.
            content, finish = f"{PARTIAL_NEEDLE} {n}", "length"
        else:
            content, finish = f"Mock response {n}.", "stop"
        return Response(
            _sse(
                [
                    {
                        "id": "mock-completion",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "mock-completion",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish}
                        ],
                        "usage": {
                            "prompt_tokens": 500,
                            "completion_tokens": 10,
                            "total_tokens": 510,
                        },
                    },
                ]
            ),
            content_type="text/event-stream",
        )

    return chat_handler, captured


@pytest.fixture
def degenerate_llm_server(request):
    mode = getattr(request, "param", "recovered")
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    chat_handler, captured = make_degenerate_server(mode)
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def repl_env(tmp_path, monkeypatch, degenerate_llm_server):
    """Config with a high threshold (auto-context never trips) and cap 2.

    monkeypatch.chdir BEFORE the spawn: the REPL inherits the pytest
    process's cwd (the §3.21 trap) - the test's own tmp dir keeps the
    daily auto-save out of the real repo's .agent13/saves.
    """
    monkeypatch.chdir(tmp_path)
    server = degenerate_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 200000
action = "report_and_compact"
chain = 0
nudge_cap = 2
"""
    )
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
        maxread=8192,
    )
    proc.timeout = timeout
    proc.expect(r">", timeout=timeout)
    return proc


class TestDegenerateNudgeExperience:
    @pytest.mark.parametrize("degenerate_llm_server", ["recovered"], indirect=True)
    def test_cap_hit_nudged_then_recovers(self, repl_env, degenerate_llm_server):
        """Cap-hit -> nudge on the wire -> clean answer; nudges: 1."""
        _, captured = degenerate_llm_server
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("degenerate please")
            # The recovered final answer streams to the terminal.
            proc.expect("Recovered fine", timeout=30)

            assert len(captured) == 2, (
                f"expected 2 LLM calls (cap-hit + nudged retry), got {len(captured)}"
            )
            # The nudge is the second request's last user message,
            # verbatim (the injected message is exactly the prompt).
            assert last_user_message(captured[1]) == DEGENERATE_NUDGE_PROMPT
            # ...and the kept partial precedes it in history.
            contents = [m.get("content", "") for m in captured[1]["messages"]]
            assert any(PARTIAL_NEEDLE in c for c in contents)
            # The injected flag is local-only bookkeeping.
            for payload in captured:
                for m in payload.get("messages", []):
                    assert "injected" not in m

            proc.sendline("/status")
            proc.expect(r"nudges:\s+1", timeout=15)
        finally:
            proc.close()

    @pytest.mark.parametrize(
        "degenerate_llm_server", ["always"], indirect=True
    )
    def test_consecutive_degenerate_ends_stop_at_cap(
        self, tmp_path, repl_env, degenerate_llm_server
    ):
        """cap=2: first end nudges, second stops the turn with a snapshot;
        the counter counts nudges only."""
        _, captured = degenerate_llm_server
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("degenerate please")
            # The stop notice (warning level) reaches the terminal.
            proc.expect("I stopped this turn", timeout=30)

            # Two calls: the initial cap-hit (nudged) and the nudged turn
            # whose cap-hit hit the cap (stop - no third call).
            assert len(captured) == 2
            assert last_user_message(captured[1]) == DEGENERATE_NUDGE_PROMPT

            # The stop path snapshotted the dropped partial.
            saves = sorted(
                (tmp_path / "saves").glob("*_degenerate_*.ctx")
            )
            assert len(saves) == 1, f"expected one snapshot, got {saves}"

            proc.sendline("/status")
            proc.expect(r"nudges:\s+1", timeout=15)
        finally:
            proc.close()
