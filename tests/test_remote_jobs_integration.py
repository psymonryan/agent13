"""Integration tests for detached remote jobs (spec item 2).

Spawns a REAL REPL against a mock LLM that issues command(mode="detach")
then command(mode="poll") tool calls. The ssh transport is REAL
(localhost), so this tests the full user-experienced path: detach ->
job_id -> poll -> status/exit code/log tail back on the wire.

Skips if sshd is not running on localhost:22.
"""

import json
import re
import time

import pytest_httpserver
from werkzeug import Request

from .mock_llm_helpers import make_models_handler
from .test_remote_transfer_integration import (  # noqa: F401 (localhost_ssh is a pytest fixture)
    _final_chunk,
    _repl_env,
    _spawn_repl,
    _sse,
    _text_chunk,
    _tool_call_chunk,
    _tool_messages,
    localhost_ssh,
)


JOB_SCRIPT = "echo job-start; sleep 1; echo job-end; exit 0"


def test_detach_and_poll(localhost_ssh, tmp_path):
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        if n == 1:
            chunks = [
                _tool_call_chunk(
                    json.dumps(
                        {
                            "command": JOB_SCRIPT,
                            "remote": "localhost",
                            "mode": "detach",
                        }
                    ),
                    "command",
                    n,
                ),
                _final_chunk("tool_calls"),
            ]
        elif n in (2, 3):
            # Simulate LLM thinking time so the job (sleep 1) finishes
            # before the second poll.
            if n == 2:
                time.sleep(1.5)
            # Recover the job_id from the detach result riding in this
            # request's history (the mock LLM can't know it in advance).
            job_id = None
            for m in body.get("messages", []):
                if m.get("role") == "tool":
                    jm = re.search(
                        r'"job_id": "(\d{8}T\d{6}Z-[0-9a-f]{6})"',
                        m.get("content", ""),
                    )
                    if jm:
                        job_id = jm.group(1)
            chunks = [
                _tool_call_chunk(
                    json.dumps(
                        {
                            "command": "",
                            "remote": "localhost",
                            "mode": "poll",
                            "job_id": job_id,
                        }
                    ),
                    "command",
                    n,
                ),
                _final_chunk("tool_calls"),
            ]
        else:
            chunks = [_text_chunk("Job done."), _final_chunk("stop")]
        return _sse(chunks)

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()

    try:
        env = _repl_env(server, tmp_path, localhost_ssh)
        proc = _spawn_repl(env)
        proc.sendline("run the job")
        proc.expect(r"Job done", timeout=90)
        proc.sendline("/quit")
        proc.close()
    finally:
        server.stop()

    tools = _tool_messages(captured)
    detach_results = [t for t in tools if '"job_id"' in t and '"hint"' in t]
    assert detach_results, f"no detach result on the wire: {tools}"
    detach = detach_results[0]
    assert '"success": true' in detach
    jm = re.search(r'"job_id": "(\d{8}T\d{6}Z-[0-9a-f]{6})"', detach)
    assert jm, f"no valid job_id in detach result: {detach}"

    poll_results = [t for t in tools if '"log_tail"' in t]
    assert len(poll_results) >= 2, f"expected 2 polls on the wire: {tools}"
    last_poll = poll_results[-1]
    # The job (sleep 1) must be finished by the second poll.
    assert '"status": "exited"' in last_poll, last_poll
    assert '"job_exit_code": 0' in last_poll, last_poll
    assert "job-start" in last_poll
    assert "job-end" in last_poll
