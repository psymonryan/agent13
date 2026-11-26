"""Integration tests for context-management flags in batch (`-p`) mode.

Regression for the silent-no-op bug where `run_batch_with_display` built the
Agent with neither `journal_mode` nor any `auto_context_*`, so `-p --journal`
and `--auto-context-threshold` did nothing while REPL/pipe/TUI honoured them.

Only the LLM is mocked: the real CLI runs as a subprocess and every chat
request is captured, so these tests see exactly what the user would get.
Each flag is checked against a control run without it - the flag must be what
changes the wire traffic, not something else in the run.
"""

import json
import os
import subprocess
import sys

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .mock_llm_helpers import make_models_handler

# Needles identifying the extra calls on the wire.
REFLECTION_NEEDLE = "tersely reflect on each one"  # prompts.REFLECTION_PROMPT
COMPACT_NEEDLE = "Summarize our conversation"  # prompts.DEFAULT_COMPACT_PROMPT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _last_user_message(body: dict) -> str:
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def _has_tool_result(body: dict) -> bool:
    return any(m.get("role") == "tool" for m in body.get("messages", []))


def _sse(parts: list[dict]) -> Response:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
    return Response(body, content_type="text/event-stream")


def _text_response(text: str, prompt_tokens: int = 50) -> Response:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 5,
        "total_tokens": prompt_tokens + 5,
    }
    return _sse(
        [
            {
                "id": "mock-completion",
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
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]
    )


def _tool_call_response(prompt_tokens: int) -> Response:
    """A `write_file` call, so the turn reaches the post-tool safe point.

    The auto-context threshold check lives after tool execution; a plain-text
    turn ends without ever running it.
    """
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 10,
        "total_tokens": prompt_tokens + 10,
    }
    return _sse(
        [
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
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
                                        "arguments": (
                                            '{"path": "notes.md", "content": "x"}'
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
                "usage": usage,
            },
        ]
    )


@pytest.fixture
def capturing_llm_server(request):
    """Mock LLM that records every chat request body.

    Turn shape: the first request (the batch prompt) gets a tool call; the
    follow-up carrying the tool result gets plain text. Reflection and compact
    calls are recognised by their last user message and answered with text so
    their turns end immediately.

    `prompt_tokens` is an indirect parameter: the auto-context test reports a
    size over its threshold, the journal test reports a small one so the
    threshold (config default 220_000) can never trip and muddy the result.
    """
    prompt_tokens = getattr(request, "param", 50)
    captured: list[dict] = []

    def chat_handler(req: Request):
        body = req.get_json(force=True)
        captured.append(body)
        last = _last_user_message(body)
        if REFLECTION_NEEDLE in last or COMPACT_NEEDLE in last:
            return _text_response("summary text")
        if _has_tool_result(body):
            return _text_response("done")
        return _tool_call_response(prompt_tokens)

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


def _run_batch(tmp_path, server, *extra_args):
    """Run the real CLI in batch mode against the mock server."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir(exist_ok=True)
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

    # Run the real entry point with the venv interpreter: `uv run` from a temp
    # cwd cannot find the script, and the agent must run IN the temp dir so
    # the mock's write_file tool call lands there.
    return subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "agent13.py"),
            "test_mock",
            "--model",
            "mock-model",
            "-p",
            "Write the notes file.",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
        cwd=str(tmp_path),
    )


def _request_needles(captured) -> list[str]:
    return [_last_user_message(body) for body in captured]


class TestBatchJournalFlag:
    """`-p --journal` must turn on post-turn reflection (it was a no-op)."""

    @pytest.mark.parametrize("capturing_llm_server", [50], indirect=True)
    def test_journal_flag_produces_a_reflection_request(
        self, tmp_path, capturing_llm_server
    ):
        server, captured = capturing_llm_server
        result = _run_batch(tmp_path, server, "--journal")

        assert result.returncode == 0, (
            f"CLI failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        needles = _request_needles(captured)
        assert any(REFLECTION_NEEDLE in n for n in needles), (
            "--journal did not produce a reflection request; requests were: "
            f"{needles}"
        )

    @pytest.mark.parametrize("capturing_llm_server", [50], indirect=True)
    def test_without_the_flag_no_reflection(self, tmp_path, capturing_llm_server):
        """Control: the same run without --journal makes no reflection call."""
        server, captured = capturing_llm_server
        result = _run_batch(tmp_path, server)

        assert result.returncode == 0, (
            f"CLI failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        needles = _request_needles(captured)
        assert not any(REFLECTION_NEEDLE in n for n in needles), (
            f"unexpected reflection request: {needles}"
        )


class TestBatchAutoContextFlags:
    """`--auto-context-*` must reach the agent (they were dropped)."""

    @pytest.mark.parametrize("capturing_llm_server", [40_000], indirect=True)
    def test_threshold_flag_triggers_the_compact(
        self, tmp_path, capturing_llm_server
    ):
        """40k reported against a 20k threshold: the compact call must hit
        the wire. The default threshold (220k) would never trip."""
        server, captured = capturing_llm_server
        result = _run_batch(
            tmp_path,
            server,
            "--auto-context-threshold",
            "20000",
            "--auto-context-action",
            "compact",
        )

        assert result.returncode == 0, (
            f"CLI failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        needles = _request_needles(captured)
        assert any(COMPACT_NEEDLE in n for n in needles), (
            "--auto-context-threshold did not trigger a compact; requests "
            f"were: {needles}"
        )

    @pytest.mark.parametrize("capturing_llm_server", [40_000], indirect=True)
    def test_without_the_flag_no_compact(self, tmp_path, capturing_llm_server):
        """Control: same traffic, default threshold - no compact request."""
        server, captured = capturing_llm_server
        result = _run_batch(tmp_path, server)

        assert result.returncode == 0, (
            f"CLI failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        needles = _request_needles(captured)
        assert not any(COMPACT_NEEDLE in n for n in needles), (
            f"unexpected compact request: {needles}"
        )
