"""Tests for agent13.py unified entry point.

Tests:
- Provider resolution (--list-providers, provider arg)
- Model selection (--model)
- Batch mode (-p flag)
- Help output
- Exit codes
"""

import subprocess
import os


class TestAgent13Help:
    """Test help and usage output."""

    def test_help_flag(self):
        result = subprocess.run(
            ["./agent13.py", "--help"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "agent13" in result.stdout.lower()

    def test_no_provider_shows_error(self):
        """No provider should show error."""
        result = subprocess.run(
            ["./agent13.py"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0


class TestAgent13ProviderList:
    """Test --list-providers flag."""

    def test_list_providers(self):
        result = subprocess.run(
            ["./agent13.py", "--list-providers"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Available providers:" in result.stdout


class TestAgent13ModelSelection:
    """Test --model flag."""

    def test_model_flag_lists_models(self, mock_provider_env):
        """--model with no value should list models."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "test_mock", "--model"],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0
        assert "Available models:" in result.stdout

    def test_model_selection_by_number(self, mock_provider_env):
        """--model 1 should select first model."""
        result = subprocess.run(
            ["uv", "run", "agent13.py", "test_mock", "--model", "1", "--model"],
            capture_output=True,
            text=True,
            timeout=60,
            env=mock_provider_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0


class TestAgent13BatchMode:
    """Test batch mode (-p flag)."""

    def test_batch_mode_exits_after_processing(self, mock_provider_env):
        """Batch mode should process and exit."""
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
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_batch_mode_produces_response(self, mock_provider_env):
        """Batch mode should produce LLM output."""
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
        assert result.returncode == 0
        # Should contain "4" somewhere in output
        assert "4" in result.stdout or "4" in result.stderr

    def test_batch_invalid_provider_exits_nonzero(self):
        """Invalid provider should exit with error."""
        result = subprocess.run(
            ["./agent13.py", "nonexistent_provider", "-p", "hello"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_batch_no_provider_exits_nonzero(self):
        """No provider with -p should show error."""
        result = subprocess.run(
            ["./agent13.py", "-p", "hello"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0


class TestAgent13Import:
    """Test that agent13 can be imported as a module."""

    def test_import_run_batch(self):
        """run_batch should be importable from agent13."""
        from agent13 import run_batch

        assert callable(run_batch)

    def test_import_batch_module(self):
        """agent.batch module should be importable."""
        from agent13.batch import run_batch as rb

        assert callable(rb)
