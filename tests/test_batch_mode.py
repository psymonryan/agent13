"""Tests for CLI batch mode (-p flag).

Batch mode runs a single prompt and exits. These tests use subprocess
to test the real CLI behavior with a mock LLM server.
"""

import subprocess
import os
import time


class TestBatchModeBasic:
    """Basic batch mode tests."""

    def test_batch_mode_exits_after_processing(self, mock_provider_env):
        """Batch mode should process prompt and exit (not hang)."""
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "-p", "Say 'hello'",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
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

    def test_batch_mode_produces_response(self, mock_provider_env):
        """Batch mode should produce actual output from the LLM."""
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "-p",
                "What is 2+2? Answer with just the number.",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should contain '4' somewhere in output
        combined_output = result.stdout + result.stderr
        assert "4" in combined_output, f"Expected '4' in output, got: {combined_output}"


class TestBatchModeRegression:
    """Regression tests for batch mode bugs."""

    def test_batch_does_not_exit_immediately(self, mock_provider_env):
        """Regression test: batch should process, not exit immediately.

        Bug: Agent.run() sets status to IDLE at startup before any message
        is processed. If batch mode listens for IDLE to signal completion,
        it would exit immediately without processing.

        This test verifies batch mode actually waits for processing.
        """
        # Use a prompt that requires some processing
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "-p", "Count from 1 to 5",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should complete successfully
        assert result.returncode == 0

        # Should have actual content beyond startup messages
        combined = result.stdout + result.stderr
        assert len(combined) > 50, (
            f"Output too short, batch may have exited early: {combined}"
        )

    def test_batch_pretty_off_mode(self, mock_provider_env):
        """Batch mode with --pretty off should still process correctly."""
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "--pretty", "off",
                "-p", "Say 'test passed'",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "test passed" in combined.lower() or "passed" in combined.lower(), (
            f"Expected response in output: {combined}"
        )


class TestBatchModeExitCodes:
    """Test batch mode exit codes."""

    def test_batch_success_returns_zero(self, mock_provider_env):
        """Successful batch execution should return 0."""
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "-p", "Hi",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0

    def test_batch_invalid_provider_exits_nonzero(self):
        """Invalid provider should exit with error."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "nonexistent_provider_xyz", "-p", "Hi"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode != 0

    def test_batch_no_provider_shows_error(self):
        """Missing provider argument should show error."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "-p", "Hi"],
            capture_output=True,
            text=True,
            timeout=30,
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

    def test_batch_completes_within_reasonable_time(self, mock_provider_env):
        """Batch mode should complete within reasonable time.

        If batch mode hangs or exits too early (regression bug),
        this test will catch it.
        """
        start = time.time()

        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "-p", "Say 'done'",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
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


class TestBatchModeReadFlag:
    """Tests for --read CLI flag in batch mode."""

    def test_read_flag_injects_file(self, mock_provider_env, tmp_path):
        """--read should inject file contents into context before the prompt."""
        # Create a temp file to read
        read_file = tmp_path / "context.txt"
        read_file.write_text("THE_SECRET_ANSWER_IS_42")

        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "--read", str(read_file),
                "-p", "What is the secret answer in the file?",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        assert result.returncode == 0, (
            f"CLI exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_read_flag_works_in_repl_mode(self, mock_provider_env, tmp_path):
        """--read should work in REPL mode (files injected at startup)."""
        # Create a temp file to read
        read_file = tmp_path / "repl_context.txt"
        read_file.write_text("REPL_TEST_CONTENT_123")

        # Run REPL with --read, send a prompt and quit
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "--read", str(read_file),
                "--repl",
            ],
            input="What was in the file?\n/quit\n",
            capture_output=True,
            text=True,
            timeout=30,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Should exit cleanly (REPL with /quit)
        assert result.returncode == 0, (
            f"CLI exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_multiple_read_flags(self, mock_provider_env, tmp_path):
        """Multiple --read flags should all be processed (not just the last)."""
        # Create multiple temp files
        read_file1 = tmp_path / "file1.txt"
        read_file1.write_text("CONTENT_FILE_1")
        read_file2 = tmp_path / "file2.txt"
        read_file2.write_text("CONTENT_FILE_2")

        # Use --read twice (this was broken before the fix)
        result = subprocess.run(
            [
                "uv", "run", "agent13.py", "test_mock",
                "--model", "mock-model",
                "--read", str(read_file1),
                "--read", str(read_file2),
                "-p", "List the files you have in context.",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        assert result.returncode == 0, (
            f"CLI exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
