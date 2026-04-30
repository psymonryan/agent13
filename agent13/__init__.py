"""Agent package - event-driven agent architecture."""

# Eagerly load environment variables on import
from agent13.config import load_environment

load_environment()


def _get_version() -> str:
    """Get version from installed metadata, falling back to pyproject.toml."""
    try:
        from importlib.metadata import version

        return version("agent13")
    except Exception:
        pass
    try:
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE
        )
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


__version__ = _get_version()

from agent13.events import AgentEvent, AgentEventData, EventHandler  # noqa: E402
from agent13.queue import AgentQueue, ItemStatus, QueueItem  # noqa: E402
from agent13.history import History, get_default_history_path  # noqa: E402
from agent13.prompts import PromptManager  # noqa: E402
from agent13.snippets import SnippetManager  # noqa: E402
from agent13.core import Agent, AgentStatus, PauseState, SpinnerSpeed  # noqa: E402
from agent13.llm import (  # noqa: E402
    stream_response,
    stream_response_complete,
    stream_response_with_tools,
    get_initial_response,
    build_messages_with_system,
    handle_tool_calls,
    append_assistant_message,
    format_context_size,
)
from agent13.tools import (  # noqa: E402
    tool,
    execute_tool,
    get_tools,
    get_tool_names,
    TOOLS,
    name_matches,
    get_filtered_tools,
    get_tool_groups,
)
from agent13.config import (  # noqa: E402
    Config,
    ProviderConfig,
    create_client,
    get_config,
    get_provider,
    resolve_provider_arg,
    reset_config,
)
from agent13.skills import SkillManager, SkillInfo, SkillMetadata, ensure_default_skills  # noqa: E402
from agent13.context import skill_manager_ctx  # noqa: E402
from agent13.batch import run_batch  # noqa: E402
from agent13.debug_log import (  # noqa: E402
    init_debug,
    is_debug_enabled,
    log_event,
    log_error,
    log_session_end,
    log_user_message,
    log_api_request,
    log_api_hash,
    log_api_response,
    log_tool_call,
    log_tool_result,
    log_stream_start,
    log_stream_chunk,
    log_stream_end,
    log_queue_start,
    log_queue_complete,
    log_queue_interrupt,
    log_assistant_response,
    log_journal_reflection,
    truncate_for_log,
    # TPS debug logging
    log_tps_event,
    log_tps_token_usage,
    log_tps_first_token,
    log_tps_stream_start,
    log_tps_stream_end,
    log_tps_timing_reset,
    log_tps_tool_call,
    log_tps_calculation,
)

__all__ = [
    "__version__",
    # Events
    "AgentEvent",
    "AgentEventData",
    "EventHandler",
    # Queue
    "AgentQueue",
    "ItemStatus",
    "QueueItem",
    # History and Prompts
    "History",
    "get_default_history_path",
    "PromptManager",
    "SnippetManager",
    # Core
    "Agent",
    "AgentStatus",
    "PauseState",
    "SpinnerSpeed",
    # LLM helpers
    "stream_response",
    "stream_response_complete",
    "stream_response_with_tools",
    "get_initial_response",
    "build_messages_with_system",
    "handle_tool_calls",
    "append_assistant_message",
    "format_context_size",
    # Tools
    "tool",
    "execute_tool",
    "get_tools",
    "get_tool_names",
    "TOOLS",
    "name_matches",
    "get_filtered_tools",
    "get_tool_groups",
    "get_tool_names",
    "TOOLS",
    # Config
    "Config",
    "ProviderConfig",
    "create_client",
    "get_config",
    "get_provider",
    "resolve_provider_arg",
    "load_environment",
    "reset_config",
    # Debug logging
    "init_debug",
    "is_debug_enabled",
    "log_event",
    "log_error",
    "log_session_end",
    "log_user_message",
    "log_api_request",
    "log_api_hash",
    "log_api_response",
    "log_tool_call",
    "log_tool_result",
    "log_stream_start",
    "log_stream_chunk",
    "log_stream_end",
    "log_queue_start",
    "log_queue_complete",
    "log_queue_interrupt",
    "log_assistant_response",
    "log_journal_reflection",
    "truncate_for_log",
    # TPS debug logging
    "log_tps_event",
    "log_tps_token_usage",
    "log_tps_first_token",
    "log_tps_stream_start",
    "log_tps_stream_end",
    "log_tps_timing_reset",
    "log_tps_tool_call",
    "log_tps_calculation",
    # Skills
    "SkillManager",
    "SkillInfo",
    "SkillMetadata",
    "ensure_default_skills",
    # Context
    "skill_manager_ctx",
    # Batch
    "run_batch",
]
