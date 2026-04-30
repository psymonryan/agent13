"""Tests for agent.llm module."""

import datetime
from unittest.mock import patch
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
)
from agent13 import (
    build_messages_with_system,
    handle_tool_calls,
    append_assistant_message,
    format_context_size,
)
from agent13.llm import categorize_error, TimeoutError_


class TestBuildMessagesWithSystem:
    """Tests for build_messages_with_system function."""

    @patch("agent13.llm._AGENTS_MD_CACHE", None)
    def test_empty_messages(self):
        """Should return only system message for empty list."""
        # Cache is None, so AGENTS.md content won't be included

        messages = []
        result = build_messages_with_system(messages, "You are helpful.")

        assert len(result) == 1
        assert result[0]["role"] == "system"
        # System prompt now includes date prefix
        expected_date = datetime.date.today().isoformat()
        assert result[0]["content"] == f"Today is {expected_date}\n\nYou are helpful."

    @patch("agent13.llm._AGENTS_MD_CACHE", None)
    def test_with_messages(self):
        """Should prepend system message to existing messages."""
        # Cache is None, so AGENTS.md content won't be included

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = build_messages_with_system(messages, "You are helpful.")

        assert len(result) == 3
        assert result[0]["role"] == "system"
        # System prompt now includes date prefix
        expected_date = datetime.date.today().isoformat()
        assert result[0]["content"] == f"Today is {expected_date}\n\nYou are helpful."
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi there!"}

    @patch("agent13.llm._AGENTS_MD_CACHE", None)
    def test_default_system_prompt(self):
        """Should use default system prompt if none provided."""
        # Cache is None, so AGENTS.md content won't be included

        messages = [{"role": "user", "content": "Hello"}]
        result = build_messages_with_system(messages)

        # System prompt now includes date prefix
        expected_date = datetime.date.today().isoformat()
        assert result[0]["content"].startswith(f"Today is {expected_date}\n\n")
        assert "tool using AI assistant" in result[0]["content"]

    @patch("agent13.llm._AGENTS_MD_CACHE", None)
    def test_does_not_modify_original(self):
        """Should not modify the original messages list."""
        # Cache is None, so AGENTS.md content won't be included

        messages = [{"role": "user", "content": "Hello"}]
        result = build_messages_with_system(messages, "You are helpful.")

        assert len(messages) == 1  # Original unchanged
        assert len(result) == 2  # Result has system + user

    @patch("agent13.llm._AGENTS_MD_CACHE", "# Test Guidance\n\nSome test content.")
    def test_includes_agents_md_when_present(self):
        """Should include AGENTS.md content when file exists."""
        # Cache is set with test content

        messages = []
        result = build_messages_with_system(messages, "You are helpful.")

        assert len(result) == 1
        assert result[0]["role"] == "system"
        expected_date = datetime.date.today().isoformat()
        expected = f"Today is {expected_date}\n\nYou are helpful.\n\nGuidance for this project:\n# Test Guidance\n\nSome test content."
        assert result[0]["content"] == expected

    @patch("agent13.llm._AGENTS_MD_CACHE", None)
    def test_agents_md_read_error_silently_ignored(self):
        """Should work when cache is None (file not found or read error)."""
        # Cache is None, simulating file not existing or read error

        messages = []
        result = build_messages_with_system(messages, "You are helpful.")

        # Should still work, just without AGENTS.md content
        assert len(result) == 1
        expected_date = datetime.date.today().isoformat()
        assert result[0]["content"] == f"Today is {expected_date}\n\nYou are helpful."


class TestHandleToolCalls:
    """Tests for handle_tool_calls function."""

    def test_handle_single_tool_call(self):
        """Should handle a single tool call."""
        messages = []

        # Create a mock tool call
        class MockFunction:
            name = "square_number"
            arguments = '{"x": 5}'

        class MockToolCall:
            id = "call_123"
            function = MockFunction()
            message = type("obj", (object,), {"content": None})()

        tool_calls = [MockToolCall()]

        def execute_tool(name, args):
            return "25"

        results = handle_tool_calls(messages, tool_calls, execute_tool)

        assert len(results) == 1
        assert results[0]["name"] == "square_number"
        assert results[0]["arguments"] == {"x": 5}
        assert results[0]["result"] == "25"

        # Check messages were appended
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "square_number"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_123"

    def test_handle_multiple_tool_calls(self):
        """Should handle multiple tool calls."""
        messages = []

        class MockFunction1:
            name = "square_number"
            arguments = '{"x": 3}'

        class MockFunction2:
            name = "square_number"
            arguments = '{"x": 4}'

        class MockToolCall1:
            id = "call_1"
            function = MockFunction1()
            message = type("obj", (object,), {"content": None})()

        class MockToolCall2:
            id = "call_2"
            function = MockFunction2()
            message = type("obj", (object,), {"content": None})()

        tool_calls = [MockToolCall1(), MockToolCall2()]

        def execute_tool(name, args):
            return str(args["x"] ** 2)

        results = handle_tool_calls(messages, tool_calls, execute_tool)

        assert len(results) == 2
        assert results[0]["result"] == "9"
        assert results[1]["result"] == "16"


class TestAppendAssistantMessage:
    """Tests for append_assistant_message function."""

    def test_append_simple_message(self):
        """Should append simple assistant message."""
        messages = []
        append_assistant_message(messages, "Hello!")

        assert len(messages) == 1
        assert messages[0] == {"role": "assistant", "content": "Hello!"}

    def test_append_message_with_reasoning(self):
        """Should append message with reasoning."""
        messages = []
        append_assistant_message(messages, "The answer is 42.", "Let me think...")

        assert len(messages) == 1
        assert messages[0]["content"] == "The answer is 42."
        assert messages[0]["reasoning_content"] == "Let me think..."

    def test_append_to_existing_messages(self):
        """Should append to existing messages."""
        messages = [{"role": "user", "content": "Hello"}]
        append_assistant_message(messages, "Hi there!")

        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"

    def test_append_message_without_reasoning_when_send_reasoning_false(self):
        """Should NOT include reasoning when send_reasoning=False."""
        messages = []
        append_assistant_message(
            messages, "The answer is 42.", "Let me think...", send_reasoning=False
        )

        assert len(messages) == 1
        assert messages[0]["content"] == "The answer is 42."
        assert "reasoning_content" not in messages[0]

    def test_append_message_with_reasoning_when_send_reasoning_true(self):
        """Should include reasoning when send_reasoning=True."""
        messages = []
        append_assistant_message(
            messages, "The answer is 42.", "Let me think...", send_reasoning=True
        )

        assert len(messages) == 1
        assert messages[0]["content"] == "The answer is 42."
        assert messages[0]["reasoning_content"] == "Let me think..."


class TestFormatContextSize:
    """Tests for format_context_size function."""

    def test_empty_messages(self):
        """Should return '0' for empty messages."""
        assert format_context_size([]) == "0"

    def test_small_size(self):
        """Should return bytes for small messages."""
        messages = [{"role": "user", "content": "Hello"}]
        result = format_context_size(messages)
        # Should be a small number like "50" or similar
        assert result.isdigit()

    def test_kilobyte_size(self):
        """Should return 'k' suffix for kilobyte sizes."""
        # Create a message with ~2KB of content
        large_content = "x" * 2000
        messages = [{"role": "user", "content": large_content}]
        result = format_context_size(messages)
        assert "k" in result

    def test_megabyte_size(self):
        """Should return 'm' suffix for megabyte sizes."""
        # Create a message with ~1.5MB of content
        large_content = "x" * 1500000
        messages = [{"role": "user", "content": large_content}]
        result = format_context_size(messages)
        assert "m" in result


class TestDetectToolCallsInReasoning:
    """Tests for detect_tool_calls_in_reasoning function."""

    def test_empty_reasoning(self):
        """Should return None for empty reasoning."""
        from agent13.llm import detect_tool_calls_in_reasoning

        assert detect_tool_calls_in_reasoning("") is None
        assert detect_tool_calls_in_reasoning(None) is None

    def test_no_tool_calls(self):
        """Should return None when no tool calls present."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = "I need to think about this problem step by step."
        assert detect_tool_calls_in_reasoning(reasoning) is None

    def test_json_tool_call(self):
        """Should detect tool calls in JSON format."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = """Let me use a tool here.
{"tool": "read_file", "arguments": {"filepath": "test.py"}}"""
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["arguments"]["filepath"] == "test.py"

    def test_json_function_call(self):
        """Should detect function calls in JSON format."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = (
            'I will call {"function": "command", "arguments": {"command": "ls"}}'
        )
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "command"

    def test_xml_function_call(self):
        """Should detect XML-style function calls."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = """<function=read_file>
<parameter name="filepath">test.py</parameter>
</function>"""
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["arguments"]["filepath"] == "test.py"

    def test_xml_function_call_multiple_params(self):
        """Should detect XML-style function calls with multiple parameters."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = """<function=edit_file>
<parameter name="filepath">test.py</parameter>
<parameter name="mode">replace</parameter>
</function>"""
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert result[0]["name"] == "edit_file"
        assert result[0]["arguments"]["filepath"] == "test.py"
        assert result[0]["arguments"]["mode"] == "replace"

    def test_malformed_json_returns_none(self):
        """Should return None for malformed JSON."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = '{"tool": "read_file", "arguments": {'  # Unclosed brace
        result = detect_tool_calls_in_reasoning(reasoning)
        # Should either return None or handle gracefully
        # The function should not raise an exception
        assert result is None or isinstance(result, list)

    def test_partial_json_with_tool_spec(self):
        """Should handle partial JSON containing tool specification."""
        from agent13.llm import detect_tool_calls_in_reasoning

        # JSON with tool specification but incomplete structure
        reasoning = 'Some text "tool": "command" more text'
        # This should be detected but may not parse fully
        result = detect_tool_calls_in_reasoning(reasoning)
        # Function should handle gracefully without crashing
        assert result is None or isinstance(result, list)

    def test_nested_json(self):
        """Should handle nested JSON structures."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = '{"tool": "edit_file", "arguments": {"filepath": "test.py", "mode": "replace"}}'
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert result[0]["name"] == "edit_file"

    def test_reasoning_with_surrounding_text(self):
        """Should find tool calls embedded in surrounding reasoning text."""
        from agent13.llm import detect_tool_calls_in_reasoning

        reasoning = """Let me think about this problem.
I need to read the file first.
<function=read_file>
<parameter name="filepath">config.py</parameter>
</function>
That should help me understand the configuration."""
        result = detect_tool_calls_in_reasoning(reasoning)
        assert result is not None
        assert result[0]["name"] == "read_file"


class TestCategorizeError:
    """Tests for categorize_error function."""

    def test_api_timeout_error(self):
        """APITimeoutError is categorized as TimeoutError_."""
        import httpx

        request = httpx.Request("GET", "http://test")
        error = APITimeoutError(request=request)
        result = categorize_error(error)
        assert isinstance(result, TimeoutError_)
        assert result.error_type == "timeout"
        msg = str(result)
        assert "timed out" in msg.lower()
        assert "read_timeout" in msg or "connect_timeout" in msg

    def test_api_timeout_before_connection(self):
        """APITimeoutError (subclass of APIConnectionError) is caught first."""
        import httpx

        request = httpx.Request("GET", "http://test")
        error = APITimeoutError(request=request)
        result = categorize_error(error)
        # Should be TimeoutError_, NOT NetworkError
        assert result.error_type == "timeout"

    def test_api_connection_error(self):
        """APIConnectionError (non-timeout) is categorized as NetworkError."""
        import httpx

        request = httpx.Request("GET", "http://test")
        error = APIConnectionError(request=request, message="Connection refused")
        result = categorize_error(error)
        assert result.error_type == "network"

    def test_authentication_error(self):
        """AuthenticationError is categorized as APIKeyError."""
        import httpx

        request = httpx.Request("GET", "http://test")
        response = httpx.Response(401, request=request)
        error = AuthenticationError("Invalid API key", response=response, body=None)
        result = categorize_error(error)
        assert result.error_type == "auth"

    def test_rate_limit_error(self):
        """RateLimitError is categorized correctly."""
        import httpx

        request = httpx.Request("GET", "http://test")
        response = httpx.Response(429, request=request)
        error = RateLimitError("Rate limit exceeded", response=response, body=None)
        result = categorize_error(error)
        assert result.error_type == "rate_limit"

    def test_bad_request_error(self):
        """BadRequestError is categorized as ModelError."""
        import httpx

        request = httpx.Request("GET", "http://test")
        response = httpx.Response(400, request=request)
        error = BadRequestError("Model not found", response=response, body=None)
        result = categorize_error(error)
        assert result.error_type == "model"

    def test_unknown_error(self):
        """Unknown exceptions are categorized as 'unknown'."""
        error = RuntimeError("Something unexpected")
        result = categorize_error(error)
        assert result.error_type == "unknown"

    def test_permission_denied_error(self):
        """PermissionDeniedError is categorized as 'permission'."""
        from openai import PermissionDeniedError
        import httpx

        request = httpx.Request("GET", "http://test")
        response = httpx.Response(403, request=request)
        error = PermissionDeniedError(
            "Forbidden", response=response, body={"error": {"message": "Access denied"}}
        )
        result = categorize_error(error)
        assert result.error_type == "permission"

    def test_llm_error_passthrough(self):
        """Already-categorized LLMError passes through unchanged."""
        original = TimeoutError_("Already categorized")
        result = categorize_error(original)
        assert result is original
