"""Tests for CLI batch mode (-p flag).

Batch mode runs a single prompt and exits. These tests use subprocess
to test the real CLI behavior.
"""

import subprocess
import os
import pytest

# Skip all tests if no test provider available
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/.agent13/config.toml")),
    reason="No config file found",
)


def has_test_provider():
    """Check if 'test' provider is configured."""
    config_path = os.path.expanduser("~/.agent13/config.toml")
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        return 'name = "test"' in f.read()


# Skip if test provider not configured
requires_test_provider = pytest.mark.skipif(
    not has_test_provider(),
    reason="'test' provider not configured in ~/.agent13/config.toml",
)


class TestBatchModeBasic:
    """Basic batch mode tests."""

    @requires_test_provider
    def test_batch_mode_exits_after_processing(self):
        """Batch mode should process prompt and exit (not hang)."""
        # Run CLI with -p flag
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "Say 'hello'",
            ],
            capture_output=True,
            text=True,
            timeout=240,  # 120s doubled - test provider may need model switch time
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should exit cleanly (not timeout)
        assert result.returncode == 0, (
            f"CLI exited with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Should have some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0, (
            "No output from batch mode"
        )

    @requires_test_provider
    def test_batch_mode_produces_response(self):
        """Batch mode should produce actual output from the LLM."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "What is 2+2? Answer with just the number.",
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should contain '4' somewhere in output
        combined_output = result.stdout + result.stderr
        assert "4" in combined_output, f"Expected '4' in output, got: {combined_output}"


class TestBatchModeRegression:
    """Regression tests for batch mode bugs."""

    @requires_test_provider
    def test_batch_does_not_exit_immediately(self):
        """Regression test: batch should process, not exit immediately.

        Bug: Agent.run() sets status to IDLE at startup before any message
        is processed. If batch mode listens for IDLE to signal completion,
        it would exit immediately without processing.

        This test verifies batch mode actually waits for processing.
        """
        # Use a prompt that requires some processing
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "Count from 1 to 5",
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should complete successfully
        assert result.returncode == 0

        # Should have actual content (numbers 1-5 would indicate real processing)
        combined = result.stdout + result.stderr
        # At minimum, should have some response content beyond startup messages
        assert len(combined) > 50, (
            f"Output too short, batch may have exited early: {combined}"
        )

    @requires_test_provider
    def test_batch_with_tool_call(self):
        """Batch mode should handle tool calls before exiting."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "What is 5 squared? Use the square_number tool.",
            ],
            capture_output=True,
            text=True,
            timeout=60,  # Tool calls may take longer
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should complete successfully
        assert result.returncode == 0

        # Should contain 25 (result of 5 squared)
        combined = result.stdout + result.stderr
        assert "25" in combined, f"Expected '25' in output, got: {combined}"

    @requires_test_provider
    def test_batch_pretty_off_mode(self):
        """Batch mode with --pretty off should still process correctly."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "--pretty",
                "off",
                "-p",
                "Say 'test passed'",
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "test passed" in combined.lower() or "passed" in combined.lower(), (
            f"Expected response in output: {combined}"
        )


class TestBatchModeExitCodes:
    """Test batch mode exit codes."""

    @requires_test_provider
    def test_batch_success_returns_zero(self):
        """Successful batch execution should return 0."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "test", "--model", "devstral2", "-p", "Hi"],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0

    def test_batch_invalid_provider_exits_nonzero(self):
        """Invalid provider should exit with error."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "nonexistent_provider_xyz", "-p", "Hi"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode != 0

    def test_batch_no_provider_shows_error(self):
        """Missing provider argument should show error."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "-p", "Hi"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # Should either exit with error or show usage
        # (provider is required unless --list-providers)
        assert (
            result.returncode != 0
            or "required" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )


class TestBatchModeTiming:
    """Tests for batch mode timing and async behavior."""

    @requires_test_provider
    def test_batch_completes_within_reasonable_time(self):
        """Batch mode should complete within reasonable time.

        If batch mode hangs or exits too early (regression bug),
        this test will catch it.
        """
        import time

        start = time.time()

        result = subprocess.run(
            [
                "uv",
                "run",
                "agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "Say 'done'",
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        elapsed = time.time() - start

        # Should complete within reasonable time (not hang)
        assert elapsed < 30, f"Batch took too long: {elapsed}s"

        # Should take at least a little time to process (not exit immediately)
        # This catches the regression where batch exits on initial IDLE
        assert elapsed > 0.5, (
            f"Batch exited too quickly ({elapsed}s), may not have processed"
        )

        assert result.returncode == 0
