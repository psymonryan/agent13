"""Integration test: /pause during a polite lock wait pauses, not interrupts.

The 09-28 incident: two agents waited on the shared polite lock; the user
pressed /pause on the waiting agent. The TUI special-cased the polite-wait
overlay and CANCELLED the agent task. The cancel landed as an ESC-style
interrupt: [Interrupted] marker, item completed "interrupted" — and the
marker made the save look *complete* (the old detectors had no marker case),
so --continue + /resume found nothing to resume. Work unreachable.

Fix under test (the full user-facing path, real REPL processes, real file
lock, only the LLM mocked):

1. Agent A holds the lock for a long turn (8 s mock stream). Gated
   server-side: A's LLM request must arrive (the lock is acquired before
   the LLM call, so a request arriving proves the lock is held).
2. Agent B's turn starts; while A holds the lock, B's LLM request must
   NOT arrive (gate: B is genuinely waiting on the lock).
3. /pause on B -> B goes PAUSED (REPL prints "[paused]"). No interrupt,
   no marker, queue item untouched.
4. /resume on B -> B re-enters the lock wait.
5. A finishes and releases the lock -> B acquires it and its turn
   COMPLETES normally.
6. B's save: no [Interrupted] marker, incomplete_turn=false.

Timing note (why B is spawned BEFORE A's turn starts): a cold
`uv run agent13` REPL takes 7-13 s to reach its prompt under load —
LONGER than A's 8 s hold. The original design (spawn B after A's turn
begins) let the contention window close before B's turn even started:
B arrived post-release, acquired the free lock legitimately, and the
"B bypassed the lock" assertion fired falsely. Pre-spawning B and
gating B's dispatch on A's server-side request (proof A holds) makes
the contention deterministic.

pty input can be lost under load (observed on slow CI machines), so every
command send is gated on its visible effect and re-sent on timeout.
"""

import json
import os
import time

import pexpect
import pytest
import pytest_httpserver
from werkzeug import Request
from werkzeug.wrappers import Response

from .helpers import spawn_process

# How long agent A's turn (and thus its lock hold) lasts.
HOLD_SECONDS = 8.0


def _chunk(delta: dict, finish=None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_chat_handler(captured, counter):
    """Chat handler: 'work-a' streams slowly (holds A's turn open), the
    rest reply immediately."""

    def handler(request: Request):
        body = request.get_json(force=True)
        captured.append(body)
        counter["n"] += 1

        last_user = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                last_user = msg["content"]
                break

        if "work-a" in last_user:
            time.sleep(HOLD_SECONDS)
            content = "Reply to: work-a"
        else:
            content = f"Reply to: {last_user[:50]}"

        sse = (
            _chunk({"role": "assistant", "content": ""})
            + _chunk({"content": content})
            + _chunk({}, finish="stop")
            + "data: [DONE]\n\n"
        )
        return Response(sse, content_type="text/event-stream")

    return handler


def _requests_with(captured, needle):
    return [
        b
        for b in captured
        if any(
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and needle in m["content"]
            for m in b.get("messages", [])
        )
    ]


@pytest.fixture
def pause_server():
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
        make_chat_handler(captured, counter)
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def pause_env(tmp_path, pause_server):
    """Shared config dir (the lock lives there) + one saves dir per REPL."""
    server = pause_server[0]
    config_dir = tmp_path / "shared-config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f'[[providers]]\nname = "test_mock"\n'
        f'api_base = "http://localhost:{server.port}/v1"\napi_key = "test-key"\n'
    )

    def make_env(saves_name):
        saves_dir = tmp_path / saves_name
        saves_dir.mkdir()
        env = os.environ.copy()
        env["AGENT13_CONFIG_DIR"] = str(config_dir)
        env["AGENT13_SAVES_DIR"] = str(saves_dir)
        env["AGENT13_NO_UPDATE_CHECK"] = "1"
        return env, saves_dir

    return make_env


def spawn_repl(env):
    proc = spawn_process(
        "uv",
        args=[
            "run",
            "agent13",
            "test_mock",
            "--repl",
            "--model",
            "mock-model",
            "--polite",
            "0.05",
        ],
        env=env,
        encoding="utf-8",
        timeout=60,
        dimensions=(50, 200),
        maxread=8192,
    )
    proc.timeout = 60
    proc.expect(r">", timeout=60)
    return proc


def send_until(proc, line, pattern, timeout=45, label=""):
    """Send a line and wait for its visible effect; re-send on timeout.

    pty input can be lost under load (a slow `uv run` startup plus a busy
    machine has dropped the first line); re-sending is safe here because
    every gated line is idempotent for the test's assertions.
    """
    deadline = time.time() + timeout
    while True:
        proc.sendline(line)
        remaining = deadline - time.time()
        if remaining <= 0:
            pytest.fail(f"{label}: {line!r} never produced {pattern!r}")
        try:
            proc.expect(pattern, timeout=remaining)
            return
        except pexpect.TIMEOUT:
            continue


def wait_for_requests(captured, needle, timeout=30, label=""):
    """Server-side gate: wait until an LLM request carrying `needle` arrives."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _requests_with(captured, needle):
            return
        time.sleep(0.2)
    pytest.fail(f"{label}: no LLM request carrying {needle!r} arrived")


class TestPauseDuringPoliteLockWait:
    def test_pause_during_lock_wait_then_resume(self, pause_env, pause_server):
        _, captured = pause_server
        env_a, _saves_a = pause_env("saves-a")
        env_b, saves_b = pause_env("saves-b")

        # ── Spawn BOTH REPLs before any turn starts ──
        # A cold `uv run agent13` startup takes 7-13 s under load — longer
        # than A's 8 s lock hold. If B were spawned after A's turn began,
        # B would routinely arrive after A's release and the contention
        # window would be gone (false "bypass" failures). With B already
        # at its prompt, dispatching it lands squarely mid-hold.
        proc_a = spawn_repl(env_a)
        proc_b = spawn_repl(env_b)
        try:
            # ── A starts a long turn ──
            # "[processing]" is written at dispatch; the server-side gate
            # below proves A acquired the lock (acquire happens before the
            # LLM call).
            send_until(proc_a, "work-a", r"\[processing\]", label="A dispatch")
            wait_for_requests(captured, "work-a", label="A lock+LLM")

            # ── B starts a turn; the lock is busy, so B must wait ──
            send_until(proc_b, "work-b", r"\[processing\]", label="B dispatch")
            # While A holds the lock (8 s), B's LLM request must NOT arrive.
            time.sleep(2.5)
            assert not _requests_with(captured, "work-b"), (
                "B bypassed the polite lock — its LLM call arrived while A "
                "held the lock"
            )

            # ── /pause while B waits on the lock ──
            send_until(proc_b, "/pause", r"Pausing at next safe point", label="B pause")

            # B is now paused. Give it time to (incorrectly) interrupt itself
            # — the pre-fix code cancelled the task within the same beat.
            #
            # Note: the REPL freezes output_ctrl while at the input prompt, so
            # the "[paused]" status line (written via output_ctrl by the status
            # handler when B reaches the PAUSED state) is buffered now and
            # flushed on /resume. We verify it there, below.
            time.sleep(2.0)

            # ── /resume: B re-enters the lock wait ──
            # /resume calls output_ctrl.unfreeze(), which flushes the buffered
            # "[paused]" line just before printing "Resumed". Send once —
            # /resume is not idempotent (a second send prints "Not paused").
            proc_b.sendline("/resume")
            proc_b.expect(r"Resumed", timeout=30)
            # The flushed "[paused]" is in proc.before (output consumed up to
            # "Resumed"). Its presence proves the pause registered as PAUSED —
            # the pre-fix code cancelled the task here (an interrupt) and never
            # reached the PAUSED state, so no [paused] line would exist.
            assert "[paused]" in proc_b.before, (
                "pause during lock wait did not register as PAUSED "
                f"(no [paused] flushed on resume); tail: {proc_b.before[-200:]!r}"
            )

            # ── A finishes, releases the lock ──
            proc_a.expect("Reply to: work-a", timeout=45)

            # ── THE point of the test: B's turn survived the pause and
            # completes once the lock frees ──
            proc_b.expect("Reply to: work-b", timeout=45)

            # ── B quits; the save must be clean ──
            proc_b.sendline("/quit")
            proc_b.expect("Session saved|Goodbye", timeout=20)
        finally:
            for p in (proc_b, proc_a):
                try:
                    p.sendline("/quit")
                except Exception:
                    pass
                try:
                    p.close()
                except Exception:
                    pass

        # ── B's saved history: no marker, turn complete ──
        saves = list(saves_b.glob("*.ctx"))
        assert saves, "no auto-save file written for B"
        data = json.loads(saves[-1].read_text())
        msgs = data["messages"]
        assert not any(
            m.get("content") == "[Interrupted]" for m in msgs
        ), f"pause during lock wait must not interrupt: {json.dumps(msgs[-3:], indent=1)}"
        assert not any(m.get("interrupt") for m in msgs)
        assert data["incomplete_turn"] is False
        assert msgs[-1]["role"] == "assistant"
        assert "Reply to: work-b" in msgs[-1].get("content", "")

        # ── B's LLM request carried the original prompt, no marker ──
        b_requests = _requests_with(captured, "work-b")
        assert b_requests, "B never made an LLM call"
        for b in b_requests:
            assert not any(
                m.get("content") == "[Interrupted]" for m in b["messages"]
            ), "marker leaked into the LLM request"
