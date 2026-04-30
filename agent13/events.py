"""Agent event system for event-driven architecture."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Union

from typing import TypeAlias


class AgentEvent(Enum):
    """Event types emitted by the Agent during processing."""

    # Lifecycle
    STARTED = "started"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"  # User cancelled the current operation
    PAUSED = "paused"  # Agent paused at a safe point
    RESUMED = "resumed"  # Agent resumed from pause

    # Queue
    QUEUE_UPDATE = "queue_update"
    ITEM_STARTED = "item_started"  # A queued item is now being processed

    # Messages
    USER_MESSAGE = "user_message"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_REASONING = "assistant_reasoning"
    ASSISTANT_COMPLETE = "assistant_complete"

    # Tools
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Status
    STATUS_CHANGE = "status_change"
    ERROR = "error"
    NOTIFICATION = "notification"  # User notification with optional duration

    # Model
    MODEL_CHANGE = "model_change"

    # Token usage
    TOKEN_USAGE = "token_usage"

    # Journal
    JOURNAL_COMPACT = "journal_compact"  # History compacted via journal mode
    JOURNAL_RESULT = "journal_result"  # Journal command completed (success/failure)

    # Interrupt injection
    INTERRUPT_INJECTED = "interrupt_injected"  # Interrupt message injected mid-turn

    # Streaming
    STREAM_START = "stream_start"  # Emitted at start of each LLM stream

    # MCP events
    MCP_SERVER_STARTED = "mcp_server_started"  # Server is being connected
    MCP_SERVER_READY = "mcp_server_ready"  # Server connected, tools available
    MCP_SERVER_ERROR = "mcp_server_error"  # Server connection failed
    MCP_SERVER_STDERR = "mcp_server_stderr"  # Server stderr output

    # Deferred commands (processed at safe boundary between items)
    MESSAGES_CLEARED = "messages_cleared"  # /clear completed at safe boundary
    CONTEXT_LOADED = "context_loaded"  # /load completed at safe boundary
    RETRY_STARTED = "retry_started"  # /retry completed at safe boundary


@dataclass
class AgentEventData:
    """Data payload for agent events."""

    event: AgentEvent
    data: dict[str, Any] = field(default_factory=dict)

    # Convenience properties for common event data
    @property
    def text(self) -> str | None:
        """Get text field from data."""
        return self.data.get("text")

    @property
    def name(self) -> str | None:
        """Get name field from data (for tool events)."""
        return self.data.get("name")

    @property
    def status(self) -> str | None:
        """Get status field from data."""
        return self.data.get("status")

    @property
    def model(self) -> str | None:
        """Get model field from data."""
        return self.data.get("model")

    @property
    def count(self) -> int | None:
        """Get count field from data."""
        return self.data.get("count")

    @property
    def message(self) -> str | None:
        """Get message field from data (for errors)."""
        return self.data.get("message")

    @property
    def exception(self) -> Exception | None:
        """Get exception field from data (for errors)."""
        return self.data.get("exception")

    @property
    def server_name(self) -> str | None:
        """Get server_name field from data (for MCP events)."""
        return self.data.get("server_name")

    @property
    def transport(self) -> str | None:
        """Get transport field from data (for MCP events)."""
        return self.data.get("transport")

    @property
    def tool_count(self) -> int | None:
        """Get tool_count field from data (for MCP events)."""
        return self.data.get("tool_count")

    @property
    def error(self) -> str | None:
        """Get error field from data (for MCP events)."""
        return self.data.get("error")

    @property
    def line(self) -> str | None:
        """Get line field from data (for MCP stderr events)."""
        return self.data.get("line")

    @property
    def summary(self) -> str | None:
        """Get summary field from data (for journal compact events)."""
        return self.data.get("summary")


# Type alias for event handler functions
# Handlers can be sync or async, receive AgentEventData, return None or awaitable
EventHandler: TypeAlias = Callable[
    [AgentEventData], Union[None, Coroutine[Any, Any, None]]
]
