"""Integration test: /pause → /quit → --continue → /resume cycle.

Only the LLM is mocked (pytest-httpserver). Tests what the user actually
experiences across the full cycle, in a real REPL process:

1. User asks for sequential tool calls; model issues one, then pauses
   mid-turn after the tool result (trailing `tool` message, no response).
2. /quit auto-saves — the save MUST keep the mid-turn state intact
   (no [Interrupted] marker, incomplete_turn=true).
3. Fresh `--continue` session: /resume continues the turn — pending work
   executes with the identical message prefix (kv-cache friendly) — and
   the model finishes.
"""

import json
import os

import pytest
import pytest_httpserver
from werkzeug.wrappers import Response
from werkzeug import Request

from .helpers import spawn_process


# ── Mock LLM: scripted two-phase tool-call conversation ──────────────────────


def _chunk(delta: dict, finish=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_scripted_chat_handler(captured, counter):
    """Chat handler implementing the scripted cycle.

    Phase 1 (first request): tool call square_number(x=4), no content.
    Phase 2 (subsequent requests): plain content reply.
    """

    def handler(request: Request):
        body = request.get_json(force=True)
        captured.append(body)
        n = counter["n"]
        counter["n"] += 1

        if n == 0:
            # Phase 1: assistant requests a tool call
            sse = (
                _chunk({"role": "assistant", "content": ""})
                + _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_resume_1",
                                "type": "function",
                                "function": {
                                    "name": "square_number",
                                    "arguments": '{"x": 4}',
                                },
                            }
                        ]
                    }
                )
                + _chunk({}, finish="tool_calls")
                + _chunk(
                    {},
                )
                + "data: [DONE]\n\n"
            )
            # usage chunk (no choices) for token accounting
            usage = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            sse += f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n"
            return Response(sse, content_type="text/event-stream")

        # Phase 2: plain reply
        sse = (
            _chunk({"role": "assistant", "content": "The square of 4 is 16."})
            + _chunk({}, finish="stop")
            + "data: [DONE]\n\n"
        )
        return Response(sse, content_type="text/event-stream")

    return handler


@pytest.fixture
def cycle_server():
    server = pytest_httpserver.HTTPServer()
    captured = []
    counter = {"n": 0}
    server.expect_request("/v1/models").respond_with_handler(
        lambda r: Response(
            json.dumps(
                {"data": [{"id": "mock-model", "object": "model", "owned_by": "test"}]}
            ),
            content_type="application/json",
        )
    )
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        make_scripted_chat_handler(captured, counter)
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def cycle_env(tmp_path, cycle_server):
    server = cycle_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()

    # saves location = "local" (default): auto-saves go under
    # <cwd>/.agent13/saves/YYYY-MM-DD.ctx — cwd is the repo root, so point
    # AGENT13_SAVES_DIR at tmp for isolation.
    (config_dir / "config.toml").write_text(
        f'[[providers]]\nname = "test_mock"\n'
        f'api_base = "http://localhost:{server.port}/v1"\napi_key = "test-key"\n'
    )

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(saves_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env, saves_dir


def spawn_repl(env):
    proc, banner = _spawn(env)
    return proc, banner


def spawn_repl_with_continue(env):
    proc, banner = _spawn(env, extra_args=["--continue"])
    return proc, banner


def _spawn(env, extra_args=None):
    proc = spawn_process(
        "uv",
        args=["run", "agent13", "test_mock", "--repl", "--model", "mock-model"]
        + (extra_args or []),
        env=env,
        encoding="utf-8",
        timeout=60,
        dimensions=(50, 200),
        maxread=8192,
    )
    proc.timeout = 60
    proc.expect(r">", timeout=30)
    banner = proc.before or ""
    return proc, banner


class TestPauseQuitContinueResumeCycle:
    def test_full_cycle(self, cycle_env, cycle_server):
        env, saves_dir = cycle_env
        _, captured = cycle_server

        # ── Session 1: turn → tool call → /pause mid-turn → /quit ──
        proc, _banner1 = spawn_repl(env)
        try:
            proc.sendline("square 4 for me")
            # Phase 1 response streams: tool call is issued and executed
            proc.expect(r"square_number", timeout=20)
            # Tool result arrives, then the model would continue — but we
            # pause before its next LLM stream (safe point is hit after the
            # tool result, before the next stream; either the model reply
            # already happened or pause holds it — both must save validly).
            proc.sendline("/pause")
            proc.expect(r"(?i)paus", timeout=10)
            # Give the loop a beat to reach the safe point / reply, then quit.
            import time

            time.sleep(0.5)
            proc.sendline("/quit")
            proc.expect(r"Session saved|Goodbye", timeout=20)
        finally:
            try:
                proc.close()
            except Exception:
                pass

        # ── Verify the auto-save kept the mid-turn state intact ──
        saves = list(saves_dir.glob("*.ctx"))
        assert saves, "no auto-save file written"
        data = json.loads(saves[0].read_text())
        msgs = data["messages"]

        # No [Interrupted] marker may appear anywhere
        assert not any(m.get("content") == "[Interrupted]" for m in msgs), (
            f"quit corrupted mid-turn history: {json.dumps(msgs[-3:], indent=1)}"
        )

        # The turn is incomplete: either pending tool_calls or a trailing
        # tool result the model hasn't processed yet.
        assert data["incomplete_turn"] is True
        last = msgs[-1]
        assert last["role"] in ("assistant", "tool"), f"unexpected tail: {last}"

        # ── Session 2 spawns WITH --continue ──
        proc2, banner2 = spawn_repl_with_continue(env)
        try:
            assert "resumed session" in banner2.lower(), (
                f"--continue did not resume: {banner2!r}"
            )
            proc2.sendline("/resume")
            # The continuation streams a normal assistant reply
            proc2.expect(r"square of 4 is 16", timeout=30)
            # Turn completes — back to input prompt on empty Enter
            proc2.sendline("")
            proc2.expect(r">", timeout=20)
        finally:
            proc2.sendline("/quit")
            try:
                proc2.close()
            except Exception:
                pass

        # Exactly two LLM requests total, and session 2's continuation
        # reused the same message prefix (kv-cache friendly): request 2
        # starts with request 1's full message list.
        assert len(captured) == 2, f"expected 2 LLM calls, saw {len(captured)}"
        msgs1 = captured[0]["messages"]
        msgs2 = captured[1]["messages"]
        # The trailing tool result from phase 1 is still last (no marker),
        # and phase 2 sends the identical prefix.
        assert msgs1 == msgs2[: len(msgs1)]
