"""Integration tests for /devel pin - the real user experience.

Spawns a REAL REPL process (mock LLM server) in an isolated config dir and
exercises the full pin lifecycle across process restarts: pin, auto-apply on
startup, unpin. The pin is keyed to the pytest working directory; isolation
comes from the temp AGENT13_CONFIG_DIR (pins.toml never touches the user's
real config).
"""

import os
import pexpect  # for pexpect.EOF only
import pytest
import time

from .helpers import spawn_process


@pytest.fixture
def devel_env(tmp_path, mock_llm_server):
    """Isolated config dir + mock provider, like conftest's mock_provider_env."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{mock_llm_server.port}/v1"
api_key = "test-key"
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    env["UV_PROJECT"] = os.getcwd()

    # Ensure no leftover pin for this project dir from a previous test
    pins_file = config_dir / "pins.toml"
    if pins_file.exists():
        pins_file.unlink()

    return env, config_dir


class _StartupError(Exception):
    def __init__(self, proc):
        self.before = proc.before or ""
        super().__init__(f"REPL exited before prompt. output:\n{self.before!r}")


def spawn_repl(env):
    """Spawn a REPL, wait for the first prompt, return the process."""
    proc = spawn_process(
        "uv",
        args=["run", "agent13", "test_mock", "--repl", "--model", "mock-model"],
        env=env,
        encoding="utf-8",
        timeout=60,
    )
    proc.timeout = 30
    for attempt in range(2):
        try:
            proc.expect(r">", timeout=30)
            return proc
        except pexpect.EOF:
            if attempt == 0:
                proc = spawn_process(
                    "uv",
                    args=[
                        "run",
                        "agent13",
                        "test_mock",
                        "--repl",
                        "--model",
                        "mock-model",
                    ],
                    env=env,
                    encoding="utf-8",
                    timeout=60,
                )
                proc.timeout = 30
                time.sleep(1)
                continue
            raise _StartupError(proc)
    raise _StartupError(proc)


def send_command(proc, command):
    """Send a /command and return the output until the next prompt."""
    proc.sendline(command)
    proc.expect(r">", timeout=30)
    return proc.before or ""


def close_repl(proc):
    try:
        send_command(proc, "/quit")
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
    proc.close(force=True)


class TestDevelPinExperience:
    """Full pin lifecycle across REPL restarts."""

    def test_pin_auto_applies_on_restart(self, devel_env):
        env, _ = devel_env

        # Run 1: off by default, enable, pin
        proc = spawn_repl(env)
        try:
            out = send_command(proc, "/devel status")
            assert "Devel mode: off" in out
            assert "Pinned: no" in out
            out = send_command(proc, "/devel on")
            assert "Devel mode enabled" in out
            out = send_command(proc, "/devel pin")
            assert "Pinned devel mode 'on' for this project" in out
            out = send_command(proc, "/devel status")
            assert "Pinned: on" in out
        finally:
            close_repl(proc)

        # Run 2: pin auto-applies on startup without --devel
        proc = spawn_repl(env)
        try:
            out = send_command(proc, "/devel status")
            assert "Devel mode: on" in out
            assert "Pinned: on" in out
            # Unpin: session state stays on, persistence removed
            out = send_command(proc, "/devel unpin")
            assert "Removed devel pin for this project" in out
        finally:
            close_repl(proc)

        # Run 3: no pin left, back to default off
        proc = spawn_repl(env)
        try:
            out = send_command(proc, "/devel status")
            assert "Devel mode: off" in out
            assert "Pinned: no" in out
        finally:
            close_repl(proc)

    def test_pin_off_persists(self, devel_env):
        env, config_dir = devel_env

        # Pin devel OFF for this project
        proc = spawn_repl(env)
        try:
            out = send_command(proc, "/devel off")
            assert "Devel mode disabled" in out
            out = send_command(proc, "/devel pin")
            assert "Pinned devel mode 'off' for this project" in out
        finally:
            close_repl(proc)

        pins_file = config_dir / "pins.toml"
        assert pins_file.exists()
        assert "false" in pins_file.read_text()

        # Restart: pinned-off still applies, unpin is possible
        proc = spawn_repl(env)
        try:
            out = send_command(proc, "/devel status")
            assert "Devel mode: off" in out
            assert "Pinned: off" in out
            out = send_command(proc, "/devel unpin")
            assert "Removed devel pin for this project" in out
        finally:
            close_repl(proc)
