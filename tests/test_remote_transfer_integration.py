"""Integration tests for remote file transfer (spec item 1).

Spawns a REAL REPL against a mock LLM that issues remote_put / remote_get
tool calls. The ssh + sftp transport is REAL (localhost), so this tests
the full user-experienced path: tool schema -> harness -> ssh/sftp ->
verified result back on the wire.

Skips if sshd is not running on localhost:22.
"""

import json
import os
import socket
import subprocess

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def localhost_ssh():
    """Real key-based ssh to localhost, self-contained.

    Strategy (no PATH shims — sandboxed children can't exec scripts from
    the pytest temp dir):
      1. Generate a temp key at a FREE default identity path
         (~/.ssh/id_ecdsa etc.) so plain ssh offers it.
      2. Append its pub to ~/.ssh/authorized_keys.
      3. Refresh the localhost known_hosts entry (accept-new).
    All user state (key files, authorized_keys, known_hosts) is restored
    on teardown. Skips if sshd is not running on localhost:22.
    """
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("localhost", 22))
    except OSError:
        pytest.skip("sshd not running on localhost:22")
    finally:
        s.close()

    ssh_dir = os.path.expanduser("~/.ssh")
    os.makedirs(ssh_dir, exist_ok=True)

    # Pick a free default identity path (ssh offers these automatically).
    key_path = None
    for name in ("id_ecdsa", "id_ed25519", "id_rsa"):
        candidate = os.path.join(ssh_dir, name)
        if not os.path.exists(candidate):
            key_path = candidate
            break
    if key_path is None:
        pytest.skip("no free default ssh identity path in ~/.ssh")

    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", key_path],
        check=True,
        capture_output=True,
        timeout=30,
    )
    pub = open(key_path + ".pub").read().strip()

    auth_path = os.path.join(ssh_dir, "authorized_keys")
    auth_orig = open(auth_path).read() if os.path.exists(auth_path) else None
    with open(auth_path, "a") as f:
        f.write(pub + "\n")

    kh_path = os.path.join(ssh_dir, "known_hosts")
    kh_orig = open(kh_path).read() if os.path.exists(kh_path) else None
    # OpenSSH-for-Windows `ssh-keygen -R <host>` hangs (not just exits) when
    # there is no <host> entry to remove, so only run it when one is present.
    if kh_orig is not None and "localhost" in kh_orig:
        subprocess.run(["ssh-keygen", "-R", "localhost"], capture_output=True, timeout=30)

    # Verify ssh to localhost works, with hard timeouts: in some environments
    # (notably nested SSH on Windows) `ssh localhost` hangs instead of failing,
    # which would deadlock the entire test run. Skip if it doesn't complete.
    try:
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=5",
                "localhost",
                "true",
            ],
            capture_output=True,
            timeout=20,
        )
        check = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
            capture_output=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("ssh to localhost hangs in this environment (nested SSH?)")
    if check.returncode != 0:
        pytest.skip(f"ssh to localhost not working: {check.stderr.decode()[:120]}")

    try:
        yield {}
    finally:
        for p in (key_path, key_path + ".pub"):
            if os.path.exists(p):
                os.unlink(p)
        if auth_orig is not None:
            with open(auth_path, "w") as f:
                f.write(auth_orig)
        elif os.path.exists(auth_path):
            os.unlink(auth_path)
        if kh_orig is not None:
            with open(kh_path, "w") as f:
                f.write(kh_orig)
        elif os.path.exists(kh_path):
            os.unlink(kh_path)


def _tool_call_chunk(args: dict, name: str, n: int):
    return {
        "id": "mock-completion",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call_{n}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }


def _text_chunk(content: str):
    return {
        "id": "mock-completion",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def _final_chunk(finish: str):
    return {
        "id": "mock-completion",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }


def _sse(chunks):
    return Response(
        "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n",
        content_type="text/event-stream",
    )


def _repl_env(server, tmp_path, extra_env):
    config_dir = tmp_path / "agent13-config"
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
    env.update(extra_env)
    return env


def _spawn_repl(env, timeout=90):
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


def _tool_messages(captured):
    """All role=tool message contents across captured chat requests."""
    out = []
    for body in captured:
        for m in body.get("messages", []):
            if m.get("role") == "tool":
                out.append(m.get("content", ""))
    return out


# ─── Test: verified binary round-trip through the real tool loop ─────────────


def test_transfer_round_trip(localhost_ssh, tmp_path):
    src = tmp_path / "blob.bin"
    data = os.urandom(65536)
    src.write_bytes(data)
    dst = tmp_path / "out.bin"
    remote_path = "/tmp/agent13-it/blob.bin"

    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        captured.append(request.get_json(force=True))

        if n == 1:
            chunks = [
                _tool_call_chunk(
                    json.dumps(
                        {
                            "local_path": str(src),
                            "remote_path": remote_path,
                            "remote": "localhost",
                        }
                    ),
                    "remote_put",
                    n,
                ),
                _final_chunk("tool_calls"),
            ]
        elif n == 2:
            chunks = [
                _tool_call_chunk(
                    json.dumps(
                        {
                            "remote_path": remote_path,
                            "local_path": str(dst),
                            "remote": "localhost",
                        }
                    ),
                    "remote_get",
                    n,
                ),
                _final_chunk("tool_calls"),
            ]
        else:
            chunks = [_text_chunk("Transfer complete."), _final_chunk("stop")]
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
        proc.sendline("transfer please")
        proc.expect(r"Transfer complete", timeout=90)
        proc.sendline("/quit")
        proc.close()
    finally:
        server.stop()

    tools = _tool_messages(captured)
    put_results = [t for t in tools if '"operation": "put"' in t]
    get_results = [t for t in tools if '"operation": "get"' in t]
    assert put_results, f"no remote_put tool result on the wire: {tools}"
    assert get_results, f"no remote_get tool result on the wire: {tools}"
    assert '"verified": true' in put_results[0]
    assert '"verified": true' in get_results[0]
    # The user-visible outcome: the local file is byte-identical.
    assert dst.read_bytes() == data
