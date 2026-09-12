"""Integration tests for the built-in offline fake provider.

Spawns agent13 as a subprocess with the ``fake`` provider and verifies the
NDJSON pipe protocol runs end-to-end with NO network and NO mock LLM server -
the fake *is* the model. Mirrors ``test_pipemode_integration.py`` but needs
none of the mock-server plumbing, which is the whole point of the fake.
"""

import json
import os
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def fake_provider_env(tmp_path):
    """Isolated config dir so MCP servers / skills from a real config can't
    leak into the run. The fake needs no provider entry in config at all."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text("# isolated empty config\n")
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    env.pop("AGENT13_FAKE_RESPONSE", None)  # force the deterministic default
    return env


def _run_fake(lines, env, timeout=60, model="anything"):
    """Run agent13 with the fake provider; feed NDJSON lines; return
    (exit_code, parsed_events, stderr)."""
    cmd = ["uv", "run", "agent13.py", "fake"]
    if model:
        cmd += ["--model", model]
    cmd += ["--io-format", "json"]
    stdin_data = "\n".join(lines) + "\n"
    result = subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=REPO_ROOT,
    )
    events = []
    for line in result.stdout.strip().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return result.returncode, events, result.stderr


def _turn(text: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }
    )


class TestFakeProviderProtocol:
    def test_full_protocol_no_network(self, fake_provider_env):
        """system -> assistant -> result{success}, exit 0, provider='fake'."""
        code, events, _ = _run_fake([_turn("hello")], fake_provider_env)
        assert code == 0, f"Expected exit 0, got {code}"
        assert events[0]["type"] == "system"
        assert events[0]["subtype"] == "session_start"
        assert events[0]["data"]["provider"] == "fake"
        assert events[0]["data"]["model"] == "anything"

        assistant = [e for e in events if e["type"] == "assistant"]
        assert len(assistant) == 1

        result = events[-1]
        assert result["type"] == "result"
        assert result["subtype"] == "success"
        assert result["is_error"] is False

    def test_result_usage_is_deterministic(self, fake_provider_env):
        """Usage fields present; output_tokens == word count of the default
        reply, so the run is byte-stable across invocations."""
        _, events, _ = _run_fake([_turn("hello")], fake_provider_env)
        usage = events[-1]["usage"]
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            assert field in usage
        assert usage["output_tokens"] == 10  # len(DEFAULT_RESPONSE.split())

    def test_deterministic_across_runs(self, fake_provider_env):
        """Same input, two fresh processes -> identical result text."""
        _, events1, _ = _run_fake([_turn("hello")], fake_provider_env)
        _, events2, _ = _run_fake([_turn("hello")], fake_provider_env)
        r1 = [e for e in events1 if e["type"] == "result"][-1]
        r2 = [e for e in events2 if e["type"] == "result"][-1]
        assert r1["result"] == r2["result"]
        assert "fake provider" in r1["result"]

    def test_all_stdout_is_json(self, fake_provider_env):
        """Every non-empty stdout line is valid JSON (stdout discipline)."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "fake",
                "--model",
                "anything",
                "--io-format",
                "json",
            ],
            input=_turn("hello") + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            env=fake_provider_env,
            cwd=REPO_ROOT,
        )
        for i, line in enumerate(result.stdout.strip().splitlines()):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(
                        f"Line {i + 1} is not valid JSON: {e}\n{line[:200]}"
                    )

    def test_no_model_defaults_to_fake(self, fake_provider_env):
        """Without --model, the provider defaults the model to 'fake' and
        still runs a successful turn."""
        code, events, _ = _run_fake([_turn("hello")], fake_provider_env, model=None)
        assert code == 0
        assert events[0]["data"]["model"] == "fake"
        assert events[-1]["subtype"] == "success"


class TestFakeProviderEnv:
    def test_env_override_reaches_pipe(self, fake_provider_env):
        env = dict(fake_provider_env)
        env["AGENT13_FAKE_RESPONSE"] = "custom reply from env"
        _, events, _ = _run_fake([_turn("hello")], env)
        result = [e for e in events if e["type"] == "result"][-1]
        assert result["result"] == "custom reply from env"
