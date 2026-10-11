"""Integration tests for tool output bounding (Phase 1 - text spill).

Spawns a REAL REPL process against a mock LLM server that captures every
chat request. The mock drives a turn in which a real `command` tool call
prints a ~62k-char blob. Because the `command` tool wraps its output in a
JSON envelope (the blob lands in one escaped `stdout` line), this exercises
the char-cut-in-place path: the envelope stays structurally intact and only
the MIDDLE of the payload is elided. The mock then recovers the middle with
a grep-style `command` on the spill file - the design's recovery path for a
single-line (JSON) payload, since read_file line-slicing can't sub-slice one
line.

A second test proves the config axis: [tools] max_result_chars = 0 leaves
the blob on the wire untouched.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

BLOB_LINES = 2000
MIDMARK_INDEX = 999  # 0-based index of the MIDMARK line in the blob

# The full turn is 3 LLM calls + 2 real `command` tool executions. On Windows
# each command runs through PowerShell, and the CI machine's aggressive AV
# makes PowerShell cold-start slow (see Burn Scars: "PowerShell cold start on
# Windows can take seconds"). Give the turn far more room there than on Unix,
# where the same flow finishes in a few seconds.
IS_WINDOWS = sys.platform == "win32"
# 300s on Windows: the full turn (3 LLM calls + 2 real PowerShell tool
# executions) has exceeded 180s on the CI host under AV load.
FLOW_TIMEOUT = 300 if IS_WINDOWS else 60


def make_blob() -> str:
    """The ~62k-char blob the `command` tool prints (2000 lines).

    Sized to sit between the bounding cap (30k) and the command tool's own
    stdout cap (100k) so the bounding layer - not the command tool's
    truncation - is what's under test.
    """
    lines = [f"line {i:04d} " + "x" * 20 for i in range(BLOB_LINES)]
    lines[MIDMARK_INDEX] = "MIDMARK " + "x" * 12
    return "\n".join(lines)


BLOB = make_blob()
# The exact command the mock LLM asks the REPL to run. Single quotes only
# inside the -c code so it survives both POSIX sh and PowerShell.
BLOB_COMMAND = (
    f"{sys.executable} -c "
    f"\"import sys; lines=[f'line {{i:04d}} ' + 'x'*20 for i in range({BLOB_LINES})]; "
    f"lines[{MIDMARK_INDEX}] = 'MIDMARK ' + 'x'*12; sys.stdout.write('\\n'.join(lines))\""
)


def extract_command(spill_path: str) -> str:
    """A grep-style command that pulls just the MIDMARK out of the spill file.

    Returns a SMALL result (the one matching token) so it doesn't itself trip
    the bounding cap - demonstrating the design's "grep/search for free"
    recovery path for a single-line JSON payload.
    """
    return (
        f"{sys.executable} -c "
        f"\"import re; s=open(r'{spill_path}').read(); "
        f"m=re.search(r'MIDMARK [x]+', s); print(m.group())\""
    )


@pytest.fixture
def capturing_llm_server():
    """Mock LLM server that captures every chat request and drives:

    1. user turn -> `command` tool call printing the blob
    2. bounded command result (JSON envelope, marker names the spill file)
       -> `command` tool call grepping the middle out of the spill file
    3. extraction result (the middle) -> final text
    """
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        last = body.get("messages", [{}])[-1]
        content = str(last.get("content", ""))

        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }

        if last.get("role") == "tool" and "MIDMARK" in content:
            final = "got the middle"
        elif last.get("role") == "tool" and "chars elided" in content:
            m = re.search(r"full output saved to (\S+) for reference", content)
            assert m, "marker must name the spill file"
            return tool_call_response(
                "command",
                json.dumps({"command": extract_command(m.group(1))}),
                usage,
            )
        else:
            return tool_call_response(
                "command", json.dumps({"command": BLOB_COMMAND}), usage
            )

        parts = [
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"{final} ({n})"},
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


def tool_call_response(name: str, arguments: str, usage: dict) -> Response:
    """A stream that is a single tool call (keeps the turn going)."""
    parts = [
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
                                "function": {"name": name, "arguments": arguments},
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
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": usage,
        },
    ]
    sse = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
    return Response(sse, content_type="text/event-stream")


def make_repl_env(tmp_path, server, max_result_chars: str) -> dict:
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[tools]
max_result_chars = {max_result_chars}
"""
    (config_dir / "config.toml").write_text(config_content)
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


def spawn_repl(env, timeout=FLOW_TIMEOUT):
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


def tool_message_contents(body):
    """All tool-role message contents from a chat request body."""
    return [
        m.get("content", "")
        for m in body.get("messages", [])
        if m.get("role") == "tool"
    ]


def _repl_tail(proc):
    """Best-effort live output tail. On a Windows pipe hang the single
    in-flight expect() never completes, so proc.before is stale (last
    completed read); proc.buffer is the live accumulator - prefer it."""
    v = getattr(proc, "buffer", None)
    if not v:
        v = getattr(proc, "before", "") or ""
    if not v:
        return ""
    if not isinstance(v, str):
        v = v.decode("utf-8", "replace")
    return v[-2000:]


def _repl_stall_diagnostic(n, expected_calls, label, pattern, how, tail):
    """Where did the turn stall? The LLM-call count disambiguates dropped
    input (0) vs a mid-tool stall (1..expected-1) vs a render stall (>=expected)."""
    if n >= expected_calls:
        stage = (
            f"all {expected_calls} LLM calls completed - the turn finished "
            f"server-side, but the final reply ({pattern!r}) never reached the "
            f"REPL screen (rendering stall)"
        )
    else:
        stage = {
            0: "no LLM calls - the input line may not have reached the REPL",
            1: (
                "1 LLM call - the blob `command` was requested; stalled "
                "running it (PowerShell cold start) or its result never came back"
            ),
            2: (
                "2 LLM calls - the second tool call was requested; stalled "
                "running it or its result never came back"
            ),
        }.get(n, f"{n} LLM calls - stalled mid-turn")
    return (
        f"[{label}] expected {pattern!r} on the REPL: {how}.\n"
        f"  LLM calls captured: {n} / {expected_calls}\n"
        f"  stage: {stage}\n"
        f"  --- last {len(tail)} chars the REPL printed ---\n{tail}"
    )


def expect_repl_output(proc, pattern, captured, expected_calls, label,
                       watchdog_timeout=150):
    """Wait for `pattern` on the REPL; on timeout/EOF fail with the
    diagnostics needed to work out WHERE the turn stalled (no retries).

    On Windows (PopenSpawn, pipe-based) pexpect's expect(timeout=) is inert -
    it cannot interrupt a blocking pipe read, so a stalled turn hangs forever.
    A daemon watchdog enforces the deadline: it captures the LLM-call count +
    the live REPL output tail, kills the process, and the expect() below
    unblocks with EOF. The intelligence is the LLM-call count, which
    disambiguates dropped input vs a mid-tool stall vs a render stall.
    """
    import threading

    import pexpect

    state = {"done": False, "timed_out": False, "diag": None}

    def _watchdog():
        import time
        deadline = time.monotonic() + watchdog_timeout
        while time.monotonic() < deadline:
            if state["done"]:
                return
            time.sleep(1)
        if state["done"]:
            return
        state["timed_out"] = True
        state["diag"] = _repl_stall_diagnostic(
            len(captured), expected_calls, label, pattern,
            f"watchdog deadline ({watchdog_timeout}s) - on Windows pipes "
            f"pexpect's expect(timeout=) is inert, so a stalled turn hangs",
            _repl_tail(proc),
        )
        try:
            proc.terminate(force=True)
        except Exception:  # noqa: BLE001, S110 - best-effort kill; diag already captured
            pass

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()
    try:
        try:
            proc.expect(pattern, timeout=FLOW_TIMEOUT)
            return
        except pexpect.EOF:
            how = "the REPL process exited (EOF)"
        except pexpect.TIMEOUT:
            how = f"timed out after {FLOW_TIMEOUT}s"
        if not state["timed_out"]:
            # Unix real-timeout / genuine EOF - the watchdog did not fire.
            state["diag"] = _repl_stall_diagnostic(
                len(captured), expected_calls, label, pattern, how,
                _repl_tail(proc),
            )
        # else: the watchdog already built a more specific diag (it killed us).
    finally:
        state["done"] = True
        t.join(timeout=2)
    raise AssertionError(state["diag"])


class TestToolOutputBoundExperience:
    def test_blob_spilled_and_middle_readable(self, tmp_path, capturing_llm_server):
        """End to end: 62k-char command output -> JSON envelope with the
        payload char-cut in place (head + marker + tail), spill file carries
        the full envelope, and a grep-style recovery returns the middle."""
        server, captured = capturing_llm_server
        env = make_repl_env(tmp_path, server, "30000")
        proc = spawn_repl(env)

        spill_path = None
        try:
            proc.sendline("dump the blob")
            expect_repl_output(proc, "got the middle", captured, 3, "blob_spilled")

            assert len(captured) == 3, (
                f"expected 3 LLM calls (turn + extract + final), "
                f"got {len(captured)}"
            )

            # Request 2: the command's tool result is bounded - the JSON
            # envelope is intact, only the middle of the payload is elided.
            tools2 = tool_message_contents(captured[1])
            assert len(tools2) == 1
            bounded = tools2[0]
            assert "chars elided" in bounded
            assert len(bounded) < 40_000, "bounded content must be far under the blob"
            # the elided middle is NOT on the wire
            assert "MIDMARK" not in bounded
            # JSON structure + payload head AND tail survive the char cut
            assert bounded.lstrip().startswith("{")
            assert "line 0000 " in bounded
            assert "line 1999 " in bounded

            # The spill file carries the FULL envelope (byte-identical); its
            # stdout field is the untruncated blob.
            m = re.search(r"full output saved to (\S+) for reference", bounded)
            assert m
            spill_path = m.group(1)
            spill = Path(spill_path)
            assert spill.exists()
            envelope = json.loads(spill.read_text(encoding="utf-8"))
            assert envelope["stdout"] == BLOB
            assert "MIDMARK" in envelope["stdout"]

            # Request 3: carries both tool messages (the bounded blob result
            # + the extraction result). The last one is the grep-style
            # extraction, which returned the middle.
            tools3 = tool_message_contents(captured[2])
            assert len(tools3) == 2
            assert "MIDMARK" not in tools3[0]  # the blob result is still elided
            assert "MIDMARK" in tools3[-1]
        finally:
            proc.terminate(force=True)
            if spill_path:
                Path(spill_path).unlink(missing_ok=True)

    def test_config_off_leaves_blob_untouched(self, tmp_path, capturing_llm_server):
        """[tools] max_result_chars = 0 disables bounding: the full envelope
        (with the blob in stdout) reaches the wire, no marker, no spill file."""
        server, captured = capturing_llm_server
        env = make_repl_env(tmp_path, server, "0")
        proc = spawn_repl(env)

        try:
            proc.sendline("dump the blob")
            # With bounding off the marker never appears, so the mock's
            # extract branch never fires: request 2 is the full envelope
            # (stdout carries MIDMARK) -> the mock's MIDMARK branch -> final.
            expect_repl_output(proc, "got the middle", captured, 2, "config_off")

            assert len(captured) == 2
            tools2 = tool_message_contents(captured[1])
            assert "chars elided" not in tools2[0]
            # the full blob is in the envelope's stdout field
            envelope = json.loads(tools2[0])
            assert envelope["stdout"] == BLOB
        finally:
            proc.terminate(force=True)
