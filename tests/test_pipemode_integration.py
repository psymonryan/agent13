"""Integration tests for pipe mode (--io-format json).

Spawns agent13 as a subprocess with a mock LLM server and verifies the
NDJSON protocol end-to-end: system event, assistant events, result events,
multi-turn, malformed input, EOF exit, and stdout discipline.
"""

import json
import subprocess
import os

import pytest


def _run_pipe(
    lines: list[str],
    env: dict,
    timeout: int = 60,
) -> tuple[int, list[dict], str]:
    """Run agent13 in pipe mode, feed lines to stdin, collect stdout events.

    Returns (exit_code, parsed_events, stderr).
    """
    stdin_data = "\n".join(lines) + "\n"
    result = subprocess.run(
        [
            "uv",
            "run",
            "agent13.py",
            "test_mock",
            "--model",
            "mock-model",
            "--io-format",
            "json",
        ],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    events = []
    for line in result.stdout.strip().splitlines():
        if line.strip():
            events.append(json.loads(line))

    return result.returncode, events, result.stderr


def _make_turn(text: str) -> str:
    """Build a NDJSON turn prompt line."""
    return json.dumps(
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }
    )


class TestPipeModeProtocol:
    """Verify the NDJSON output protocol."""

    def test_system_event_first(self, mock_provider_env):
        """First stdout line must be a system event with valid session_id."""
        _, events, _ = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        assert len(events) >= 2, f"Expected at least system + result, got {len(events)}"
        assert events[0]["type"] == "system"
        assert events[0]["subtype"] == "session_start"
        assert len(events[0]["session_id"]) == 36  # UUID format
        assert events[0]["data"]["session_id"] == events[0]["session_id"]
        assert events[0]["data"]["model"] == "mock-model"

    def test_result_event_last(self, mock_provider_env):
        """Last stdout event must be a result event."""
        _, events, _ = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        assert events[-1]["type"] == "result"
        assert events[-1]["is_error"] is False
        assert events[-1]["subtype"] == "success"
        assert "usage" in events[-1]
        assert "duration_ms" in events[-1]
        assert "num_turns" in events[-1]

    def test_session_id_consistent(self, mock_provider_env):
        """All events must share the same session_id."""
        _, events, _ = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        session_ids = {e["session_id"] for e in events}
        assert len(session_ids) == 1, f"Multiple session_ids: {session_ids}"

    def test_usage_fields_present(self, mock_provider_env):
        """Result event must have all usage sub-fields."""
        _, events, _ = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        usage = events[-1]["usage"]
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            assert field in usage, f"Missing usage field: {field}"
            assert isinstance(usage[field], int)


class TestPipeModeTurns:
    """Verify turn processing and multi-turn behaviour."""

    def test_single_turn_produces_response(self, mock_provider_env):
        """A valid turn should produce assistant events with text."""
        _, events, _ = _run_pipe(
            [_make_turn("Say hello")],
            mock_provider_env,
        )
        assistant_events = [e for e in events if e["type"] == "assistant"]
        assert len(assistant_events) >= 1, "Expected at least one assistant event"

        # Check for text content
        has_text = False
        for ev in assistant_events:
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "text" and "Hello" in block.get("text", ""):
                    has_text = True
        assert has_text, f"No 'Hello' in assistant events: {assistant_events}"

    def test_multi_turn_preserves_conversation(self, mock_provider_env):
        """Two turns should both be processed; conversation state preserved."""
        turn1 = _make_turn("Say hello")
        turn2 = _make_turn("What is 2+2? Answer with just the number.")

        _, events, _ = _run_pipe([turn1, turn2], mock_provider_env)

        # Should have at least: system, result1, result2
        results = [e for e in events if e["type"] == "result"]
        assert len(results) >= 2, f"Expected 2 results, got {len(results)}"
        assert results[0]["is_error"] is False
        assert results[1]["is_error"] is False

        # Second turn's result should contain "4"
        assert "4" in results[1]["result"], (
            f"Expected '4' in second turn result, got: {results[1]['result']}"
        )

    def test_malformed_json_produces_invalid_input(self, mock_provider_env):
        """Malformed JSON line should produce invalid_input result, not crash."""
        _, events, _ = _run_pipe(
            ["not valid json", _make_turn("hello")],
            mock_provider_env,
        )
        invalid = [e for e in events if e.get("subtype") == "invalid_input"]
        assert len(invalid) == 1, f"Expected 1 invalid_input, got {len(invalid)}"
        assert invalid[0]["is_error"] is True

        # Process should continue and handle the valid turn
        results = [e for e in events if e["type"] == "result"]
        assert len(results) >= 2

    def test_missing_type_produces_invalid_input(self, mock_provider_env):
        """Missing 'type' field should produce invalid_input result."""
        bad = json.dumps({"message": {"role": "user", "content": []}})
        _, events, _ = _run_pipe([bad], mock_provider_env)
        invalid = [e for e in events if e.get("subtype") == "invalid_input"]
        assert len(invalid) == 1
        assert "type" in invalid[0]["error"]["message"]

    def test_empty_line_skipped(self, mock_provider_env):
        """Empty lines should be silently skipped (no result event)."""
        _, events, _ = _run_pipe(
            ["", _make_turn("hello"), ""],
            mock_provider_env,
        )
        # Empty lines produce no events; just system + assistant + result
        results = [e for e in events if e["type"] == "result"]
        assert len(results) == 1, (
            f"Expected 1 result (empty lines skipped), got {len(results)}"
        )


class TestPipeModeLifecycle:
    """Verify process lifecycle: EOF exit, stdout discipline."""

    def test_eof_exits_zero(self, mock_provider_env):
        """Closing stdin (EOF) should cause clean exit with code 0."""
        exit_code, _, _ = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        assert exit_code == 0, f"Expected exit 0, got {exit_code}"

    def test_stdout_all_json(self, mock_provider_env):
        """Every non-empty line on stdout must be valid JSON."""
        stdin_data = _make_turn("hello") + "\n"
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test_mock",
                "--model",
                "mock-model",
                "--io-format",
                "json",
            ],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        for i, line in enumerate(result.stdout.strip().splitlines()):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Line {i + 1} is not valid JSON: {e}\n"
                    f"Content: {line[:200]}"
                )

    def test_stderr_has_diagnostics(self, mock_provider_env):
        """Diagnostics go to stderr, not stdout."""
        _, _, stderr = _run_pipe(
            [_make_turn("hello")],
            mock_provider_env,
        )
        # stderr should have the session end message
        assert "Pipe session ended" in stderr


