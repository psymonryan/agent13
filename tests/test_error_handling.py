"""Tests for error handling in the agent and TUI."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio

from agent13.llm import (
    LLMError,
    NetworkError,
    APIKeyError,
    RateLimitError_,
    PermissionError_,
    ModelError,
    categorize_error,
)
from agent13.core import Agent
from agent13.events import AgentEvent


class TestErrorCategorization:
    """Tests for error categorization."""

    def test_categorize_api_connection_error(self):
        """Test that APIConnectionError is categorized as NetworkError."""
        from openai import APIConnectionError
        import httpx

        # APIConnectionError requires a request
        request = MagicMock(spec=httpx.Request)
        error = APIConnectionError(request=request)
        result = categorize_error(error)

        assert isinstance(result, NetworkError)
        assert result.error_type == "network"

    def test_categorize_authentication_error(self):
        """Test that AuthenticationError is categorized as APIKeyError."""
        from openai import AuthenticationError
        import httpx

        # Create a mock response with headers
        response = MagicMock(spec=httpx.Response)
        response.status_code = 401
        response.headers = {}

        error = AuthenticationError("Invalid API key", response=response, body={})
        result = categorize_error(error)

        assert isinstance(result, APIKeyError)
        assert result.error_type == "auth"
        assert "Authentication failed" in str(result)

    def test_categorize_rate_limit_error(self):
        """Test that RateLimitError is categorized correctly."""
        from openai import RateLimitError
        import httpx

        response = MagicMock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {}

        error = RateLimitError("Rate limit exceeded", response=response, body={})
        result = categorize_error(error)

        assert isinstance(result, RateLimitError_)
        assert result.error_type == "rate_limit"

    def test_categorize_permission_denied_error(self):
        """Test that PermissionDeniedError is categorized as PermissionError_."""
        from openai import PermissionDeniedError
        import httpx

        response = MagicMock(spec=httpx.Response)
        response.status_code = 403
        response.headers = {}

        # Test with nested error message (OpenRouter style)
        error = PermissionDeniedError(
            "Error code: 403",
            response=response,
            body={"error": {"message": "Key limit exceeded (total limit)"}},
        )
        result = categorize_error(error)

        assert isinstance(result, PermissionError_)
        assert result.error_type == "permission"
        assert "Key limit exceeded" in str(result)

    def test_categorize_permission_denied_error_simple(self):
        """Test PermissionDeniedError with simple string body."""
        from openai import PermissionDeniedError
        import httpx

        response = MagicMock(spec=httpx.Response)
        response.status_code = 403
        response.headers = {}

        error = PermissionDeniedError(
            "Error code: 403 - Permission denied",
            response=response,
            body=None,
        )
        result = categorize_error(error)

        assert isinstance(result, PermissionError_)
        assert result.error_type == "permission"

    def test_categorize_bad_request_error(self):
        """Test that BadRequestError is categorized as ModelError."""
        from openai import BadRequestError
        import httpx

        response = MagicMock(spec=httpx.Response)
        response.status_code = 400
        response.headers = {}

        error = BadRequestError("Invalid model", response=response, body={})
        result = categorize_error(error)

        assert isinstance(result, ModelError)
        assert result.error_type == "model"

    def test_categorize_generic_api_error(self):
        """Test that generic APIError is categorized as LLMError."""
        from openai import APIError
        import httpx

        request = MagicMock(spec=httpx.Request)
        error = APIError("Something went wrong", request=request, body={})
        result = categorize_error(error)

        assert isinstance(result, LLMError)
        assert result.error_type == "api"

    def test_categorize_unknown_error(self):
        """Test that unknown errors are wrapped in LLMError."""
        error = ValueError("Something unexpected")
        result = categorize_error(error)

        assert isinstance(result, LLMError)
        assert result.error_type == "unknown"
        assert "Unexpected error" in str(result)

    def test_llm_error_passed_through(self):
        """Test that LLMError subclasses are passed through unchanged."""
        original = NetworkError("Network issue")
        result = categorize_error(original)

        # Should return the same object
        assert result is original


class TestAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_emits_error_on_api_failure(self):
        """Test that agent emits ERROR event on API failure."""
        from openai import AsyncOpenAI

        client = MagicMock(spec=AsyncOpenAI)
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("API failed"))

        agent = Agent(client=client, model="test-model")

        errors = []

        @agent.on_event
        async def on_error(event):
            if event.event == AgentEvent.ERROR:
                errors.append(event)

        # Add a message
        await agent.add_message("Hello")

        # Run agent briefly
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)  # Let it process
        agent.stop()
        await task

        # Should have received an error
        assert len(errors) > 0
        assert errors[0].event == AgentEvent.ERROR
        assert "API failed" in errors[0].message

    @pytest.mark.asyncio
    async def test_agent_includes_error_type_on_network_error(self):
        """Test that agent includes error_type for network errors."""
        from openai import AsyncOpenAI, APIConnectionError
        import httpx

        client = MagicMock(spec=AsyncOpenAI)
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        request = MagicMock(spec=httpx.Request)
        client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=request)
        )

        agent = Agent(client=client, model="test-model")

        errors = []

        @agent.on_event
        async def on_error(event):
            if event.event == AgentEvent.ERROR:
                errors.append(event)

        await agent.add_message("Hello")

        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)
        agent.stop()
        await task

        assert len(errors) > 0
        assert errors[0].data.get("error_type") == "network"


class TestTUIErrorDisplay:
    """Tests for TUI error display."""

    def test_error_message_includes_error_type(self):
        """Test that ErrorMessage includes error_type."""
        from ui.tui import ErrorMessage

        msg = ErrorMessage("Test error", "network")
        assert msg.message == "Test error"
        assert msg.error_type == "network"

    def test_error_message_default_error_type(self):
        """Test that ErrorMessage defaults to 'unknown' error_type."""
        from ui.tui import ErrorMessage

        msg = ErrorMessage("Test error")
        assert msg.error_type == "unknown"


class TestCleanExit:
    """Tests for clean exit handling."""

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        """Test that main() handles KeyboardInterrupt gracefully."""
        from ui import tui

        # Mock async_main to raise KeyboardInterrupt when called
        # This avoids creating an unawaited coroutine
        async def raise_keyboard_interrupt():
            raise KeyboardInterrupt()

        monkeypatch.setattr(tui, "async_main", raise_keyboard_interrupt)

        # Should not raise
        with pytest.raises(SystemExit) as exc_info:
            tui.main()

        # Should exit with code 0
        assert exc_info.value.code == 0

    def test_main_handles_eof_error(self, monkeypatch):
        """Test that main() handles EOFError gracefully."""
        from ui import tui

        # Mock async_main to raise EOFError when called
        # This avoids creating an unawaited coroutine
        async def raise_eof_error():
            raise EOFError()

        monkeypatch.setattr(tui, "async_main", raise_eof_error)

        with pytest.raises(SystemExit) as exc_info:
            tui.main()

        assert exc_info.value.code == 0
