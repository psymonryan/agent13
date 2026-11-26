"""Integration test for the idle auto-save (LLM mocked only).

Spawns a REAL REPL against a mock LLM server and verifies the feature end
to end: after a turn completes and the session sits idle past
``--auto-save-idle``, the daily auto-save appears ON THE DISK while the
REPL is still running (the exit save is not what we are testing), and it
contains the turn's messages. A second test pins that 0 disables it.
"""

import json
import os
import time

import pytest
import pytest_httpserver

from .helpers import spawn_process
from .mock_llm_helpers import make_chat_handler, make_models_handler

IDLE_SECONDS = 2


def _config_text(port: int, idle_seconds: int) -> str:
    return f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{port}/v1"
api_key = "test-key"

[auto_context]
threshold = 200000
action = "report_and_compact"
chain = 0

[auto_save]
idle_seconds = {idle_seconds}
"""


@pytest.fixture
def llm_server():
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        make_chat_handler()
    )
    server.start()
    yield server
    server.stop()


def _env(tmp_path, server, idle_seconds: int):
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(_config_text(server.port, idle_seconds))
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


def _spawn(env, timeout=60):
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


def _wait_for_auto_save(saves_dir, deadline_s=30):
    """Poll for an auto-save file; return its parsed JSON (None on timeout)."""
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        files = sorted(saves_dir.glob("*.ctx"))
        if files:
            try:
                return json.loads(files[0].read_text())
            except (json.JSONDecodeError, OSError):
                pass  # mid-write; try again
        time.sleep(0.5)
    return None


class TestIdleSaveExperience:
    def test_idle_save_written_while_running(self, tmp_path, monkeypatch, llm_server):
        """Turn ends -> session idles -> the auto-save lands before /quit."""
        monkeypatch.chdir(tmp_path)  # the REPL inherits the test's cwd
        saves_dir = tmp_path / "saves"
        proc = _spawn(_env(tmp_path, llm_server, IDLE_SECONDS))
        try:
            proc.sendline("hello")
            proc.expect("Hello! I'm a mock assistant.", timeout=30)

            saved = _wait_for_auto_save(saves_dir)
            assert saved is not None, (
                f"no auto-save appeared within 30s of idling "
                f"(files: {list(saves_dir.glob('*'))})"
            )
            contents = [m.get("content", "") for m in saved["messages"]]
            assert any("hello" in c for c in contents)
            assert any("mock assistant" in c for c in contents)
            # A real save, not a stub: the resume-relevant fields are there.
            assert saved["incomplete_turn"] is False
            assert saved["model"] == "mock-model"

            # /status reports the live setting.
            proc.sendline("/status")
            proc.expect(r"idle-save:\s+2s idle", timeout=15)
        finally:
            proc.close()

    def test_zero_disables_the_save(self, tmp_path, monkeypatch, llm_server):
        """idle_seconds = 0 -> nothing is written while the session idles."""
        monkeypatch.chdir(tmp_path)
        saves_dir = tmp_path / "saves"
        proc = _spawn(_env(tmp_path, llm_server, 0))
        try:
            proc.sendline("hello")
            proc.expect("Hello! I'm a mock assistant.", timeout=30)

            # Well past the 2s threshold the other test relies on.
            time.sleep(6)
            assert list(saves_dir.glob("*.ctx")) == []
        finally:
            proc.close()
