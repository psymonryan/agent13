"""LLM helpers for streaming and tool calls."""

import datetime
import json
import re
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)

from agent13.debug_log import (
    log_api_request,
    log_api_response,
    log_api_hash,
    log_stream_start,
    log_stream_chunk,
    log_stream_end,
)
from agent13.prompts import DEFAULT_PROMPT

# Cache for AGENTS.md content - read once at module load time
_AGENTS_MD_CACHE: str | None = None


def _load_agents_md_cache() -> None:
    """Load AGENTS.md into module-level cache. Called once at import time."""
    global _AGENTS_MD_CACHE
    agents_path = Path.cwd() / "AGENTS.md"
    if agents_path.exists():
        try:
            _AGENTS_MD_CACHE = agents_path.read_text(encoding="utf-8").strip()
        except (IOError, OSError):
            _AGENTS_MD_CACHE = None


# Load cache at module import time
_load_agents_md_cache()


class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(self, message: str, error_type: str, original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class NetworkError(LLMError):
    """Network/connection error."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "network", original_error)


class APIKeyError(LLMError):
    """Authentication error."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "auth", original_error)


class RateLimitError_(LLMError):
    """Rate limit error."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "rate_limit", original_error)


class PermissionError_(LLMError):
    """Permission denied error (HTTP 403)."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "permission", original_error)


class TimeoutError_(LLMError):
    """Request timeout error (read or connect)."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "timeout", original_error)


class ModelError(LLMError):
    """Model-related error (not found, bad request, etc.)."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message, "model", original_error)


def categorize_error(error: Exception) -> LLMError:
    """Categorize an OpenAI exception into our error types.

    Args:
        error: The exception from OpenAI API

    Returns:
        An appropriate LLMError subclass
    """
    # If already an LLMError, return it unchanged
    if isinstance(error, LLMError):
        return error

    if isinstance(error, APITimeoutError):
        return TimeoutError_(
            "Request timed out — the model may be thinking or the server is slow. "
            "Try increasing read_timeout or connect_timeout in ~/.agent13/config.toml.",
            error,
        )
    elif isinstance(error, APIConnectionError):
        return NetworkError(
            f"Network error: {error.message if hasattr(error, 'message') else str(error)}",
            error,
        )
    elif isinstance(error, AuthenticationError):
        return APIKeyError("Authentication failed. Check your API key.", error)
    elif isinstance(error, PermissionDeniedError):
        # Extract the actual error message from the response body if available
        # OpenRouter returns {'error': {'message': 'Key limit exceeded...'}}
        error_msg = str(error)
        if hasattr(error, "body") and error.body:
            if isinstance(error.body, dict):
                nested = error.body.get("error", {})
                if isinstance(nested, dict) and "message" in nested:
                    error_msg = nested["message"]
        return PermissionError_(error_msg, error)
    elif isinstance(error, RateLimitError):
        return RateLimitError_("Rate limit exceeded. Please wait and try again.", error)
    elif isinstance(error, BadRequestError):
        return ModelError(
            f"Bad request: {error.message if hasattr(error, 'message') else str(error)}",
            error,
        )
    elif isinstance(error, APIError):
        return LLMError(
            f"API error: {error.message if hasattr(error, 'message') else str(error)}",
            "api",
            error,
        )
    else:
        return LLMError(f"Unexpected error: {str(error)}", "unknown", error)


async def stream_response(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str = None,
    response_format: dict = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """Stream a response from the model.

    Yields tuples of (token_type, content) where token_type is either
    "content" or "reasoning".

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: Conversation messages
        system_prompt: Optional system prompt to prepend
        response_format: Optional response format (e.g., {"type": "json_object"})

    Yields:
        Tuples of (token_type, content)
    """
    # Build messages with system prompt
    api_messages = build_messages_with_system(messages, system_prompt)

    api_params = {"model": model, "messages": api_messages, "stream": True}

    if response_format:
        api_params["response_format"] = response_format

    # Log API request with structural fingerprint
    api_messages_built = build_messages_with_system(messages, system_prompt)
    role_sequence = "".join(m.get("role", "?")[0] for m in api_messages_built)
    log_api_request(
        model, len(messages), 0, {"stream": True, "role_seq": role_sequence}
    )
    log_stream_start(model)

    stream = await client.chat.completions.create(**api_params)

    chunk_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta

        # Handle reasoning content (for models that support it)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            chunk_count += 1
            yield ("reasoning", delta.reasoning_content)

        # Handle regular content
        if delta.content:
            chunk_count += 1
            yield ("content", delta.content)

        # Log chunk summary every 100 tokens
        if chunk_count > 0 and chunk_count % 100 == 0:
            log_stream_chunk(chunk_count)

    # Log stream end
    log_stream_end(chunk_count)


async def stream_response_complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str = None,
    response_format: dict = None,
) -> tuple[str, str]:
    """Stream a response and return the complete content and reasoning.

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: Conversation messages
        system_prompt: Optional system prompt to prepend
        response_format: Optional response format

    Returns:
        Tuple of (content, reasoning)
    """
    content = ""
    reasoning = ""

    async for token_type, token in stream_response(
        client, model, messages, system_prompt, response_format
    ):
        if token_type == "content":
            content += token
        elif token_type == "reasoning":
            reasoning += token

    return content, reasoning


async def get_initial_response(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str = None,
    tools: list[dict] = None,
) -> tuple[ChatCompletionMessage, dict | None]:
    """Get initial response from model (non-streaming).

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: Conversation messages
        system_prompt: Optional system prompt to prepend
        tools: Optional list of tool schemas

    Returns:
        Tuple of (response message, token usage dict or None)
        Token usage dict contains: prompt_tokens, completion_tokens, total_tokens
    """
    api_messages = build_messages_with_system(messages, system_prompt)

    params = {
        "model": model,
        "messages": api_messages,
    }

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    # Log API request
    log_api_request(model, len(messages), len(tools) if tools else 0, {"stream": False})

    response = await client.chat.completions.create(**params)

    # Log API response
    finish_reason = response.choices[0].finish_reason if response.choices else None
    tokens = None
    if hasattr(response, "usage") and response.usage:
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    log_api_response(tokens, finish_reason)

    return response.choices[0].message, tokens


def build_messages_with_system(
    messages: list[dict],
    system_prompt: str = None,
) -> list[dict]:
    """Build messages list with system prompt prepended.

    Args:
        messages: Conversation messages
        system_prompt: System prompt text (uses default if None)

    Returns:
        New list with system message first, followed by conversation
    """
    if system_prompt is None:
        system_prompt = DEFAULT_PROMPT

    # Inject current date at the start of the system prompt
    current_date = datetime.date.today().isoformat()
    system_prompt = f"Today is {current_date}\n\n{system_prompt}"

    # Append cached AGENTS.md content if available
    if _AGENTS_MD_CACHE:
        system_prompt += f"\n\nGuidance for this project:\n{_AGENTS_MD_CACHE}"

    # Strip non-standard keys before sending to LLM API
    # (e.g. "interrupt" is a local flag for turn-boundary logic, not an API field)
    clean_messages = [
        {k: v for k, v in msg.items() if k != "interrupt"} for msg in messages
    ]

    return [{"role": "system", "content": system_prompt}] + clean_messages


def detect_tool_calls_in_reasoning(reasoning_content: str) -> list[dict] | None:
    """Detect and extract tool calls from reasoning content for models like Qwen.

    Some models (e.g., Qwen) occasionally place tool call information in the reasoning
    field instead of using the standard tool_calls format. This function attempts to
    detect and parse such cases.

    Args:
        reasoning_content: The reasoning content to analyze

    Returns:
        List of extracted tool calls in standard format, or None if no tool calls found
    """
    if not reasoning_content:
        return None

    # Pattern for tool calls that might appear in reasoning
    # This looks for patterns like <function=name> or JSON tool specifications
    tool_call_patterns = [
        r"<function=.*?>",  # XML-style function call
        r"<parameter name=.*?>",  # Parameter pattern
        r'"tool":\s*"[^"]+"',  # JSON-like tool specification
        r'"function":\s*"[^"]+"',  # Function specification
    ]

    # Check if any tool call patterns are present
    has_tool_pattern = any(
        re.search(p, reasoning_content, re.DOTALL | re.IGNORECASE)
        for p in tool_call_patterns
    )
    if not has_tool_pattern:
        return None

    # Try to extract tool call information
    # This is a heuristic approach - different models may have different formats
    tool_calls = []

    try:
        # Try to parse as JSON first (some models might embed JSON in reasoning)
        open_brace_pos = reasoning_content.find("{")
        if open_brace_pos != -1:
            # Try to find the matching closing brace
            brace_count = 1
            close_brace_pos = open_brace_pos + 1
            while close_brace_pos < len(reasoning_content) and brace_count > 0:
                if reasoning_content[close_brace_pos] == "{":
                    brace_count += 1
                elif reasoning_content[close_brace_pos] == "}":
                    brace_count -= 1
                close_brace_pos += 1

            if brace_count == 0:  # Found matching closing brace
                json_str = reasoning_content[open_brace_pos:close_brace_pos]
                # Check if this JSON contains tool/function info
                if '"tool":' in json_str or '"function":' in json_str:
                    try:
                        tool_data = json.loads(json_str)
                        if isinstance(tool_data, dict):
                            # Extract tool call information
                            tool_name = tool_data.get("tool") or tool_data.get(
                                "function"
                            )
                            if tool_name:
                                tool_calls.append(
                                    {
                                        "name": tool_name,
                                        "arguments": tool_data.get("arguments", {}),
                                    }
                                )
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Fall through to XML parsing

        # If no JSON found, try to extract from XML-like patterns
        if not tool_calls:
            # Look for tool names in function= pattern
            tool_name_match = re.search(
                r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>", reasoning_content
            )
            if tool_name_match:
                tool_name = tool_name_match.group(1)

                # Try to extract arguments from parameter tags
                arguments = {}
                param_matches = re.findall(
                    r'<parameter name="([^"]+)">(.*?)</parameter>',
                    reasoning_content,
                    re.DOTALL,
                )
                for param_name, param_value in param_matches:
                    arguments[param_name] = param_value.strip()

                if tool_name:
                    tool_calls.append({"name": tool_name, "arguments": arguments})

    except Exception as e:
        # Log the error for debugging and return None to let normal processing continue
        if __debug__:
            from agent13.debug_log import is_debug_enabled, log_event

            if is_debug_enabled():
                log_event(
                    "detect_tool_calls_error",
                    {"error": str(e), "error_type": type(e).__name__},
                )
        return None

    return tool_calls if tool_calls else None


def handle_tool_calls(
    messages: list[dict],
    tool_calls: list,
    execute_tool_func: callable,
) -> list[dict]:
    """Execute tool calls and append results to messages.

    Args:
        messages: Message list to modify (appends assistant message and tool results)
        tool_calls: List of tool calls from the assistant message
        execute_tool_func: Function to execute tools (name, arguments) -> result

    Returns:
        List of tool call info dicts: [{"id": ..., "name": ..., "result": ...}]
    """
    # Add assistant message with tool calls
    messages.append(
        {
            "role": "assistant",
            "content": tool_calls[0].message.content
            if hasattr(tool_calls[0], "message")
            else "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
    )

    results = []

    # Execute each tool call
    for tool_call in tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        result = execute_tool_func(name, arguments)

        # Add tool result to messages
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )

        results.append(
            {"id": tool_call.id, "name": name, "arguments": arguments, "result": result}
        )

    return results


def append_assistant_message(
    messages: list[dict], content: str, reasoning: str = "", send_reasoning: bool = True
) -> None:
    """Append an assistant message to the message list.

    Args:
        messages: Message list to modify
        content: The assistant's response content
        reasoning: Optional reasoning content
        send_reasoning: If True, include reasoning_content in message history.
                       Default True to preserve reasoning within a turn for
                       better multi-step reasoning continuity.
    """
    message = {"role": "assistant", "content": content}
    if reasoning and send_reasoning:
        message["reasoning_content"] = reasoning
    messages.append(message)


async def stream_response_with_tools(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str = None,
    tools: list[dict] = None,
    tool_choice: str = "auto",
) -> AsyncGenerator[tuple[str, str | dict], None]:
    """Stream a response from the model, handling tool calls.

    This is like stream_response but also yields tool call information.
    Yields tuples of (event_type, data) where:
    - ("content", str): Regular content token
    - ("reasoning", str): Reasoning token
    - ("tool_call", dict): Tool call info with name and arguments chunk
    - ("tool_calls_complete", dict): All tool calls ready with {"tool_calls": [...]}

    Args:
        client: AsyncOpenAI client
        model: Model name
        messages: Conversation messages
        system_prompt: Optional system prompt to prepend
        tools: Optional list of tool schemas
        tool_choice: Tool choice mode - "auto" (default), "none", "required",
                     or {"type": "function", "function": {"name": "..."}}.
                     Use "none" to include tools in the API request for LCP
                     cache matching while preventing the model from calling them.

    Yields:
        Tuples of (event_type, data)
    """
    api_messages = build_messages_with_system(messages, system_prompt)

    api_params = {
        "model": model,
        "messages": api_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if tools:
        api_params["tools"] = tools
        api_params["tool_choice"] = tool_choice

    # Log API request with structural fingerprint for cache analysis
    # Role sequence and tool name list reveal if prompt structure changes
    # between calls (which would invalidate KV cache / LCP similarity)
    role_sequence = "".join(m.get("role", "?")[0] for m in api_messages)
    tool_names = (
        [t.get("function", {}).get("name", "?") for t in tools] if tools else []
    )
    log_api_request(
        model,
        len(messages),
        len(tools) if tools else 0,
        {"stream": True, "role_seq": role_sequence, "tool_names": tool_names},
    )
    # Per-component hashes for cache divergence analysis
    # Compare consecutive api_hash entries to find where LCP prefix breaks
    log_api_hash(
        system_prompt=api_messages[0]["content"],
        tools=tools,
        messages=api_messages,
    )
    log_stream_start(model)

    stream = await client.chat.completions.create(**api_params)

    # Accumulate tool calls by index (OpenAI uses index for continuation chunks)
    tool_calls_accum = {}  # index -> {"id": str, "name": str, "arguments": str}
    chunk_count = 0
    last_chunk = None

    async for chunk in stream:
        last_chunk = chunk
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        # Handle reasoning content (for models that support it)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            chunk_count += 1
            yield ("reasoning", delta.reasoning_content)

        # Handle regular content
        if delta.content:
            chunk_count += 1
            yield ("content", delta.content)

        # Handle tool call chunks
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                tc_index = tc_chunk.index
                tc_id = tc_chunk.id
                tc_func = tc_chunk.function

                # Initialize new tool call at this index
                if tc_index not in tool_calls_accum:
                    tool_calls_accum[tc_index] = {
                        "id": tc_id or f"tc_{tc_index}",
                        "name": "",
                        "arguments": "",
                    }

                # Update ID if provided (first chunk has it)
                if tc_id and tool_calls_accum[tc_index]["id"] != tc_id:
                    tool_calls_accum[tc_index]["id"] = tc_id

                # Accumulate name and arguments
                if tc_func:
                    if tc_func.name:
                        tool_calls_accum[tc_index]["name"] = tc_func.name
                        yield (
                            "tool_call",
                            {
                                "name": tc_func.name,
                                "id": tool_calls_accum[tc_index]["id"],
                            },
                        )

                    if tc_func.arguments:
                        tool_calls_accum[tc_index]["arguments"] += tc_func.arguments

        # Log chunk summary every 100 tokens
        if chunk_count > 0 and chunk_count % 100 == 0:
            log_stream_chunk(chunk_count)

    # Log stream end
    log_stream_end(chunk_count)

    # Yield token usage if available (from stream_options: include_usage)
    if last_chunk is not None and hasattr(last_chunk, "usage") and last_chunk.usage:
        yield (
            "token_usage",
            {
                "prompt_tokens": last_chunk.usage.prompt_tokens,
                "completion_tokens": last_chunk.usage.completion_tokens,
                "total_tokens": last_chunk.usage.total_tokens,
            },
        )

    # If we have tool calls, yield the complete list
    if tool_calls_accum:
        yield ("tool_calls_complete", {"tool_calls": list(tool_calls_accum.values())})


def format_context_size(messages: list[dict]) -> str:
    """Calculate and format the size of messages for display.

    Args:
        messages: Conversation messages

    Returns:
        Human-readable string like "2.3k" or "45"
    """
    if not messages:
        return "0"

    size_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))

    if size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f}m"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f}k"
    else:
        return f"{size_bytes:.0f}"
