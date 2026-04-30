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
import pytest


def has_test_provider():
    """Check if 'test' provider is configured."""
    config_path = os.path.expanduser("~/.agent13/config.toml")
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        return 'name = "test"' in f.read()


requires_test_provider = pytest.mark.skipif(
    not has_test_provider(), reason="'test' provider not configured"
)


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

    @requires_test_provider
    def test_model_flag_lists_models(self):
        """--model with no value should list models."""
        result = subprocess.run(
            ["./agent13.py", "test", "--model"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Available models:" in result.stdout

    @requires_test_provider
    def test_model_selection_by_number(self):
        """--model 1 should select first model."""
        result = subprocess.run(
            ["./agent13.py", "test", "--model", "1", "--model"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0


class TestAgent13BatchMode:
    """Test batch mode (-p flag)."""

    @requires_test_provider
    def test_batch_mode_exits_after_processing(self):
        """Batch mode should process and exit."""
        result = subprocess.run(
            ["./agent13.py", "test", "--model", "devstral2", "-p", "Say 'hello'"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    @requires_test_provider
    def test_batch_mode_produces_response(self):
        """Batch mode should produce LLM output."""
        result = subprocess.run(
            [
                "./agent13.py",
                "test",
                "--model",
                "devstral2",
                "-p",
                "What is 2+2? Answer with just the number.",
            ],
            capture_output=True,
            text=True,
            timeout=60,
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
