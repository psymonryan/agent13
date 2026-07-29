"""Unit tests for agent13.models — shared model selection utilities.

Tests: resolve_model_selection, fetch_models, print_model_list.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.models import (
    fetch_models,
    print_model_list,
    resolve_from_list,
    resolve_model_selection,
)


# ─── resolve_from_list (shared generic) ───────────────────────────────


class TestResolveFromList:
    """Shared selection logic used by /model and /provider."""

    def test_numeric_selection(self):
        """Numeric string selects by 1-indexed position."""
        items = ["alpha", "beta", "gamma"]
        assert resolve_from_list(items, "2") == "beta"

    def test_numeric_out_of_range(self):
        """Number beyond list length prints error, returns None."""
        items = ["alpha", "beta"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_from_list(items, "99")
        assert result is None
        assert "out of range" in mock_out.getvalue()

    def test_numeric_zero(self):
        """Number 0 is out of range (1-indexed)."""
        items = ["alpha"]
        with patch("sys.stdout", new_callable=io.StringIO):
            assert resolve_from_list(items, "0") is None

    def test_exact_match(self):
        """Exact name match returns the item."""
        items = ["alpha", "beta", "gamma"]
        assert resolve_from_list(items, "beta") == "beta"

    def test_partial_match_single(self):
        """Partial case-insensitive match with one hit returns it."""
        items = ["openai/gpt-4o", "anthropic/claude-3"]
        assert resolve_from_list(items, "gpt") == "openai/gpt-4o"

    def test_partial_match_multiple(self):
        """Partial match with multiple hits prints ambiguity, returns None."""
        items = ["gpt-4o", "gpt-4o-mini", "claude-3"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_from_list(items, "gpt")
        assert result is None
        assert "Ambiguous" in mock_out.getvalue()

    def test_no_match(self):
        """No match prints error, returns None."""
        items = ["alpha", "beta"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_from_list(items, "nonexistent")
        assert result is None
        assert "No item matching" in mock_out.getvalue()

    def test_empty_choice(self):
        """Empty string returns None without error."""
        assert resolve_from_list(["alpha"], "") is None

    def test_empty_list(self):
        """Empty list returns None for any choice."""
        assert resolve_from_list([], "anything") is None

    def test_custom_label(self):
        """Custom label appears in error messages."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            resolve_from_list(["alpha"], "nope", label="provider")
        assert "No provider matching" in mock_out.getvalue()

    def test_custom_output(self):
        """Errors go to specified output stream."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            resolve_from_list(["alpha"], "nope", output=__import__("sys").stderr)
        assert "No item matching" in mock_err.getvalue()

    def test_case_insensitive(self):
        """Partial match is case-insensitive."""
        items = ["OpenAI/GPT-4o", "Anthropic/Claude-3"]
        assert resolve_from_list(items, "gpt") == "OpenAI/GPT-4o"

# ── resolve_model_selection ──────────────────────────────────────


class TestResolveModelSelection:
    """User selects a model by name or number."""

    def test_exact_match(self):
        """Exact name match returns the model."""
        names = ["alpha", "beta", "gamma"]
        assert resolve_model_selection(names, "beta") == "beta"

    def test_numeric_selection(self):
        """Numeric string selects by 1-indexed position."""
        names = ["alpha", "beta", "gamma"]
        assert resolve_model_selection(names, "2") == "beta"

    def test_numeric_out_of_range(self):
        """Number beyond list length prints error, returns None."""
        names = ["alpha", "beta"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_model_selection(names, "99")
        assert result is None
        assert "out of range" in mock_out.getvalue()

    def test_numeric_zero(self):
        """Number 0 is out of range (1-indexed)."""
        names = ["alpha"]
        with patch("sys.stdout", new_callable=io.StringIO):
            result = resolve_model_selection(names, "0")
        assert result is None

    def test_partial_match_single(self):
        """Partial case-insensitive match with one hit returns it."""
        names = ["openai/gpt-4o", "anthropic/claude-3"]
        assert resolve_model_selection(names, "gpt") == "openai/gpt-4o"

    def test_partial_match_multiple(self):
        """Partial match with multiple hits prints ambiguity, returns None."""
        names = ["gpt-4o", "gpt-4o-mini", "claude-3"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_model_selection(names, "gpt")
        assert result is None
        assert "Ambiguous" in mock_out.getvalue()

    def test_no_match(self):
        """No match prints error, returns None."""
        names = ["alpha", "beta"]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            result = resolve_model_selection(names, "nonexistent")
        assert result is None
        assert "No model matching" in mock_out.getvalue()

    def test_empty_choice(self):
        """Empty string returns None without error."""
        names = ["alpha"]
        assert resolve_model_selection(names, "") is None

    def test_empty_model_list(self):
        """Empty model list returns None for any choice."""
        assert resolve_model_selection([], "anything") is None

    def test_use_stderr(self):
        """Errors go to stderr when use_stderr=True."""
        names = ["alpha"]
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            result = resolve_model_selection(names, "nope", use_stderr=True)
        assert result is None
        assert "No model matching" in mock_err.getvalue()

    def test_case_insensitive_partial(self):
        """Partial match is case-insensitive."""
        names = ["OpenAI/GPT-4o", "Anthropic/Claude-3"]
        assert resolve_model_selection(names, "gpt") == "OpenAI/GPT-4o"


# ── fetch_models ────────────────────────────────────────────────


class TestFetchModels:
    """Fetching models from API."""

    @pytest.mark.asyncio
    async def test_returns_sorted_model_ids(self):
        """fetch_models returns sorted list of model IDs."""
        mock_models = MagicMock()
        mock_models.data = [
            MagicMock(id="gamma"),
            MagicMock(id="alpha"),
            MagicMock(id="beta"),
        ]
        mock_client = AsyncMock()
        mock_client.models.list.return_value = mock_models

        result = await fetch_models(mock_client)

        assert result == ["alpha", "beta", "gamma"]

    @pytest.mark.asyncio
    async def test_raises_on_api_error(self):
        """fetch_models raises RuntimeError on API failure."""
        mock_client = AsyncMock()
        mock_client.models.list.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="Failed to fetch models"):
            await fetch_models(mock_client)


# ── print_model_list ────────────────────────────────────────────


class TestPrintModelList:
    """Printing model list for user display."""

    def test_prints_numbered_list(self):
        """print_model_list outputs numbered models."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_model_list(["alpha", "beta", "gamma"])

        output = mock_out.getvalue()
        assert "1. alpha" in output
        assert "2. beta" in output
        assert "3. gamma" in output

    def test_marks_current_model(self):
        """Current model is marked with *."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_model_list(["alpha", "beta", "gamma"], current="beta")

        output = mock_out.getvalue()
        assert "beta *" in output

    def test_no_current_model(self):
        """No model marked when current is None."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_model_list(["alpha", "beta"], current=None)

        output = mock_out.getvalue()
        assert "*" not in output

    def test_empty_list(self):
        """Empty list prints header only."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            print_model_list([])

        output = mock_out.getvalue()
        assert "Available models" in output
