# Agent13 Architecture

This document describes the architecture, design decisions, and implementation details of the Agent13 AI assistant framework.

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Core Components](#core-components)
4. [Event System](#event-system)
5. [Tool System](#tool-system)
6. [LLM Integration](#llm-integration)
7. [Queue and Message Flow](#queue-and-message-flow)
8. [User Interfaces](#user-interfaces)
9. [Configuration](#configuration)
10. [MCP Integration](#mcp-integration)
11. [Skills System](#skills-system)
12. [Sandbox Security](#sandbox-security)
13. [Testing Strategy](#testing-strategy)
14. [Extensibility Points](#extensibility-points)

***

## Overview

Agent13 is an event-driven AI agent framework built on top of the OpenAI-compatible API. It provides:

- **Multiple UI modes**: TUI (full-featured, Textual-based) and headless (for testing and scripting)
- **Tool execution**: Both synchronous and asynchronous tools with automatic discovery and group-based visibility
- **MCP support**: Model Context Protocol for external tool servers (stdio and HTTP transports)
- **Queue management**: Three-level priority (normal, priority, interrupt) with safe-boundary processing
- **Streaming responses**: Real-time token display with reasoning token support
- **Context management**: Journal-based compaction, save/load, and reasoning token control

The system is designed to be modular, testable, and extensible while maintaining a small codebase footprint.

***

## Design Principles

### Single Source of Truth

Values are derived from existing sources rather than duplicated. For example, tool definitions come from decorated functions, not a separate registry. State machines use enums rather than scattered booleans - `PauseState(RUNNING/PAUSING/PAUSED)` replaces the previous `_paused`/`_pausing` booleans which admitted invalid combinations.

### Minimize Context Usage

Tools and prompts are kept lean. Every token consumes context window. Fewer, general-purpose tools are preferred over many specialized ones. The `@tool(groups=["devel"])` pattern hides developer tools from the LLM by default, reducing context noise.

### Event-Driven Architecture

Components communicate through events, not direct coupling. UIs subscribe to agent events; agents don't know about UIs. This keeps the system modular and testable.

### Fail Clearly

When something goes wrong, errors are surfaced directly. No hidden failures behind fallbacks or silent retries. LLM errors are categorised into specific types (`NetworkError`, `APIKeyError`, `RateLimitError_`, etc.) rather than swallowed.

### Composition Over Inheritance

Behavior is built by combining small pieces, not through deep class hierarchies. The `@tool` decorator pattern exemplifies this: simple registration, no base class required. The `PauseState` enum composes pause behavior rather than inheriting from a pausable base class.

### Enums Over Booleans for State Machines

When a concept has 3+ states (e.g. `RUNNING/PAUSING/PAUSED`), use an enum, not scattered booleans. Booleans admit invalid combinations; enums make illegal states unrepresentable. Single source of truth: the enum lives in core, UIs read from it.

***

## Core Components

### Agent (`agent13/core.py`)

The `Agent` class is the central orchestrator. Key responsibilities:

- **Message management**: Maintains conversation history with compaction and journal summaries
- **Tool execution**: Routes tool calls to registered handlers (local or MCP)
- **Event emission**: Broadcasts state changes to subscribers
- **Queue processing**: Handles priority, interrupt, and deferred messages
- **LLM interaction**: Manages the streaming request/response cycle
- **Context management**: Save/load conversation state, clear history at safe boundaries
- **Devel mode**: Controls visibility of developer-group tools

```text
┌───────────────────────────────────────────────────┐
│                        Agent                      │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Messages │  │  Queue   │  │ Event Handlers  │  │
│  └──────────┘  └──────────┘  └─────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  Tools   │  │   MCP    │  │ Prompt Manager  │  │
│  └──────────┘  └──────────┘  └─────────────────┘  │
└───────────────────────────────────────────────────┘
```

### Agent Status

The agent has distinct states managed by the `AgentStatus` enum:

| Status         | Meaning                            |
| -------------- | ---------------------------------- |
| `INITIALISING` | Agent created, not yet running     |
| `IDLE`         | No active processing               |
| `WAITING`      | Waiting for queue item             |
| `THINKING`     | LLM is generating reasoning tokens |
| `PROCESSING`   | LLM is generating content tokens   |
| `TOOLING`      | Executing tool calls               |
| `JOURNALING`   | Compacting context via journal     |
| `PAUSED`       | Paused at a safe point             |

### Pause State

Pause behavior is managed by the `PauseState` enum (single source of truth):

| State     | Meaning                                 |
| --------- | --------------------------------------- |
| `RUNNING` | Normal operation                        |
| `PAUSING` | Pause requested, waiting for safe point |
| `PAUSED`  | Paused at safe boundary                 |

This replaces the previous `_paused`/`_pausing` booleans which could admit the invalid state `(_pausing=True, _paused=True)`.

***

## Event System

### Events (`agent13/events.py`)

All events are defined in the `AgentEvent` enum. Event data is carried in `AgentEventData`, which wraps a `data: dict` with convenience properties.

| Event                 | When Emitted                             |
| --------------------- | ---------------------------------------- |
| `STARTED`             | Agent `run()` begins                     |
| `STOPPED`             | Agent `run()` ends                       |
| `INTERRUPTED`         | User cancelled current operation         |
| `PAUSED`              | Agent paused at safe point               |
| `RESUMED`             | Agent resumed from pause                 |
| `QUEUE_UPDATE`        | Queue contents change                    |
| `ITEM_STARTED`        | A queued item begins processing          |
| `USER_MESSAGE`        | User message added to queue              |
| `ASSISTANT_TOKEN`     | Content token from LLM stream            |
| `ASSISTANT_REASONING` | Reasoning token from LLM stream          |
| `ASSISTANT_COMPLETE`  | Response finished (no tool calls)        |
| `TOOL_CALL`           | Tool execution starts                    |
| `TOOL_RESULT`         | Tool execution completes                 |
| `STATUS_CHANGE`       | Agent status transitions                 |
| `ERROR`               | Error occurs                             |
| `NOTIFICATION`        | User notification with optional duration |
| `MODEL_CHANGE`        | Model switched at runtime                |
| `TOKEN_USAGE`         | Token usage stats from stream            |
| `JOURNAL_COMPACT`     | History compacted via journal mode       |
| `JOURNAL_RESULT`      | Journal command completed                |
| `INTERRUPT_INJECTED`  | Interrupt message injected mid-turn      |
| `STREAM_START`        | Start of each LLM stream                 |
| `MCP_SERVER_STARTED`  | MCP server being connected               |
| `MCP_SERVER_READY`    | MCP server connected, tools available    |
| `MCP_SERVER_ERROR`    | MCP server connection failed             |
| `MCP_SERVER_STDERR`   | MCP server stderr output                 |
| `MESSAGES_CLEARED`    | `/clear` completed at safe boundary      |
| `CONTEXT_LOADED`      | `/load` completed at safe boundary       |
| `RETRY_STARTED`       | `/retry` completed at safe boundary      |

### Event Data

`AgentEventData` is a dataclass that wraps a `data: dict[str, Any]` with convenience properties for common fields:

```python
@dataclass
class AgentEventData:
    event: AgentEvent
    data: dict[str, Any] = field(default_factory=dict)

    # Convenience properties
    @property
    def text(self) -> str | None: ...       # For token events
    @property
    def name(self) -> str | None: ...       # For tool events
    @property
    def status(self) -> str | None: ...     # For status events
    @property
    def model(self) -> str | None: ...      # For model change events
    @property
    def message(self) -> str | None: ...    # For error events
    @property
    def exception(self) -> Exception | None: ...  # For error events
    @property
    def server_name(self) -> str | None: ...     # For MCP events
    @property
    def summary(self) -> str | None: ...         # For journal events
```

### Subscribing to Events

```python
@agent.on_event
async def handler(event: AgentEventData):
    if event.event == AgentEvent.ASSISTANT_TOKEN:
        print(event.text, end='', flush=True)
```

Handlers can be sync or async. They receive `AgentEventData`, not raw `(agent, event, data)` triples.

***

## Tool System

### Tool Discovery (`tools/__init__.py`)

Tools are auto-discovered from the `tools/` directory. Any function decorated with `@tool` is automatically registered:

```python
from tools import tool

@tool
def read_file(filepath: str) -> dict:
    """Read a file with smart structure view."""
    # Implementation
```

### Tool Registration

The `@tool` decorator:

1. Extracts the function signature and type hints
2. Parses the docstring for description and parameter docs
3. Generates JSON Schema for parameters
4. Registers the function in the appropriate registry (sync or async)
5. Stores group membership for visibility filtering
6. Optionally stores a timeout override

```python
def tool(
    func: Callable | None = None,
    *,
    is_async: bool = False,
    timeout: float | None = None,
    groups: list[str] | None = None,
) -> Callable:
```

### Tool Groups and Visibility

Tools can be assigned to groups via `@tool(groups=["devel"])`. The visibility system works in two layers:

1. **Group filter**: Tools in the `"devel"` group are hidden from the LLM unless `devel_mode=True`. This keeps developer-facing tools (like the TUI viewer) out of the context window during normal use.
2. **Config filter**: The global `enabled_tools` and `disabled_tools` settings in `config.toml` provide whitelist/blacklist filtering. If `enabled_tools` is non-empty, only matching tools pass (whitelist). Otherwise, `disabled_tools` acts as a blacklist.

```python
def get_filtered_tools(
    devel: bool = False,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
) -> list[dict]:
```

Pattern matching supports glob wildcards (`tui_*`) and regex (`re:^tui.*`), both case-insensitive.

### Tool Execution (`tools/__init__.py`)

```python
async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments."""
```

Tools are executed based on their type:

- **Async tools**: Awaited directly
- **Sync tools**: Run in executor to avoid blocking the event loop

Arguments are coerced from JSON types to Python types using the function's type hints. Missing or invalid arguments produce actionable error messages rather than silent failures.

### Built-in Tools

| Tool            | Purpose                                          |
| --------------- | ------------------------------------------------ |
| `command`       | Execute shell commands with sandboxing           |
| `read_file`     | Read files with structure awareness              |
| `write_file`    | Create new files                                 |
| `edit_file`     | Modify existing files (line-based and AST-based) |
| `skill`         | Load specialized skill modules                   |
| `square_number` | Example/test tool                                |
| `tui_viewer`    | TUI screenshot/interaction tools (devel group)   |

***

## LLM Integration

### Request Flow

The LLM integration follows a loop pattern to handle tool calls:

```text
┌────────────────────────────────────────────────────────────┐
│                      _llm_turn() Loop                      │
│                                                            │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────┐  │
│  │ Build Request │───▶│Stream Response│───▶│Check Tools │  │
│  └───────────────┘    └───────────────┘    └──────┬─────┘  │
│                                                   │        │
│         ┌─────────────────────┐                   │        │
│         │                     │                   │        │
│         ▼                     │                   │        │
│  ┌───────────────┐    ┌──────────────┐            │        │
│  │ Execute Tools │───▶│ Add Results  │────────────┘        │
│  └───────────────┘    └──────────────┘                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Each iteration:

1. Builds the API request with full message history and tool definitions
2. Streams the response, emitting reasoning and content tokens as they arrive
3. Detects tool calls in the response
4. Executes tools and appends results to messages
5. Loops back if tools were called (the LLM sees the tool results and may call more tools or respond)

### Multiple HTTP Requests Per Turn

**Important**: Each iteration of the tool loop makes a separate HTTP request to the LLM backend. This is required by the OpenAI-compatible API design:

1. The API is stateless - every request must contain full conversation history
2. Each `client.chat.completions.create()` call is independent
3. The backend tracks each request as a separate "task"

For local LLM backends (like llama.cpp), this appears as multiple tasks in the logs:

```text
# First request: User message → LLM decides to call tool
task 127438 | prompt eval = 3804ms / 1 token
            | eval time = 4048ms / 68 tokens (tool call)

# Second request: Tool result → LLM response
task 127507 | f_keep = 0.975 (97.5% KV cache retained!)
            | n_tokens = 3870
```

**Why this is still efficient**:

The backend's KV cache reuse (`f_keep = 0.975` in the example) shows it retains most context between requests. The "new task" IDs are HTTP request bookkeeping, not wasted computation. The backend recognizes similar prefixes and reuses cached computations.

**Alternative approaches not taken**:

1. **Persistent connections**: OpenAI API doesn't support session state
2. **Batched tool calls**: Would require API changes
3. **Custom protocol**: Would break compatibility with OpenAI ecosystem

### Streaming (`agent13/llm.py`)

```python
async def stream_response_with_tools(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    system_prompt: str = None,
    tools: list[dict] = None,
    tool_choice: str = "auto",
) -> AsyncGenerator[tuple[str, str | dict], None]:
```

Yields tuples of `(event_type, data)` as they occur:

| Event Type              | Data                                                                    | When                                        |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------- |
| `"content"`             | `str`                                                                   | Regular content token                       |
| `"reasoning"`           | `str`                                                                   | Reasoning token (chain-of-thought)          |
| `"tool_call"`           | `dict`                                                                  | Tool call chunk (name + arguments fragment) |
| `"tool_calls_complete"` | `{"tool_calls": [...]}`                                                 | All tool calls fully accumulated            |
| `"token_usage"`         | `{"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}` | Usage stats from final chunk                |

### Error Handling

LLM errors are categorized into specific types:

| Error Type         | Cause                       |
| ------------------ | --------------------------- |
| `NetworkError`     | Connection failures         |
| `APIKeyError`      | Authentication issues       |
| `RateLimitError_`  | Rate limiting               |
| `PermissionError_` | Permission/access denied    |
| `TimeoutError_`    | Read or connect timeout     |
| `ModelError`       | Invalid model or parameters |

All inherit from `LLMError`, which carries an `error_type` string and optional `original_error`.

***

## Queue and Message Flow

### Queue (`agent13/queue.py`)

The `AgentQueue` manages pending messages with three priority levels:

```text
┌────────────────────────────────────────────────┐
│              AgentQueue                        │
│                                                │
│  Interrupt Items (inserts in running loop)     │
│  ┌───────┐ ┌───────┐ ┌───────┐                 │
│  │ !!msg │ │ !!msg │ │ !!msg │                 │
│  └───────┘ └───────┘ └───────┘                 │
│                                                │
│  Priority Items (jumpt to top of 'next' queue) │
│  ┌───────┐ ┌───────┐                           │
│  │ !msg  │ │ !msg  │                           │
│  └───────┘ └───────┘                           │
│                                                │
│  Normal Items (Adds to end of command queue)   │
│  ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │ msg  │ │ msg  │ │ msg  │                    │
│  └──────┘ └──────┘ └──────┘                    │
└────────────────────────────────────────────────┘
```

### QueueItem

```python
@dataclass
class QueueItem:
    id: int
    text: str
    priority: bool = False
    interrupt: bool = False
    kind: str = "prompt"  # "prompt", "journal_last", "journal_all", "clear", "load", "retry"
    status: ItemStatus = ItemStatus.PENDING
    data: dict = None      # Optional metadata (e.g. {"clear_widgets": True})
```

### Item Status

```python
class ItemStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
```

### Message Priority Syntax

- `message!` - Priority message (jumps queue, processed next)
- `!!message` - Interrupt message (stops current processing, breaks agent loop)

### Deferred Operations

Operations that modify message history (`/clear`, `/load`, `/retry`) are deferred via queue items with special `kind` values. They execute at safe boundaries between items, preventing race conditions where history is wiped mid-turn.

***

## User Interfaces

### TUI (`ui/tui.py`)

The Textual-based TUI provides the full interactive experience:

```text
┌──────────────────────────────────────────────────────────┐
│ Status: thinking | Queue: 3 | Tokens: 1234/5678          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  User: What files are in this project?                   │
│                                                          │
│  Assistant: I'll check the project structure...          │
│  [Tool: command] ls -la                                  │
│  [Result: total 42...]                                   │
│                                                          │
│  The project contains:                                   │
│  - agent13/ (core library)                               │
│  - ui/ (interfaces)                                      │
│  - tools/ (tool implementations)                         │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐   │
│ │ > _                                                │   │
│ └────────────────────────────────────────────────────┘   │
│ [Tab] autocomplete  [Esc] interrupt  [Enter] send        │
└──────────────────────────────────────────────────────────┘
```

### Event Handling

The TUI subscribes to agent events:

```python
@agent.on_event
async def on_token(event: AgentEventData):
    if event.event == AgentEvent.ASSISTANT_TOKEN:
        # Update streaming display
```

### Commands

| Command                        | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `/help`                        | Show available commands                           |
| `/quit`, `/exit`               | Exit the application                              |
| `/clear`                       | Clear message history (deferred to safe boundary) |
| `/model`                       | Switch model                                      |
| `/provider`                    | Switch provider                                   |
| `/queue`                       | Show queue status                                 |
| `/pause`, `/resume`            | Control processing                                |
| `/sandbox`                     | Set sandbox mode                                  |
| `/mcp`                         | Manage MCP servers                                |
| `/tools`                       | List available tools                              |
| `/prompt`                      | Manage system prompts                             |
| `/skills`                      | List available skills                             |
| `/journal`                     | Context compaction                                |
| `/save`, `/load`               | Save/load conversation context                    |
| `/retry`                       | Retry last LLM turn (deferred)                    |
| `/devel`                       | Toggle devel mode (show/hide devel-group tools)   |
| `/history`                     | Browse conversation history                       |
| `/delete`                      | Delete messages from history                      |
| `/snippet`                     | Manage text snippets                              |
| `/spinner`                     | Control spinner animation                         |
| `/prioritise`, `/deprioritise` | Change queue item priority                        |
| `/remove-reasoning`            | Strip reasoning tokens from last turn             |
| `/list`                        | List providers or models                          |
| `/tool-response`               | Control tool result display format                |
| `/pretty`                      | Toggle pretty-printing                            |

### Headless (`headless.py`)

Minimal interface for testing and scripting:

```bash
printf "hello\n/quit\n" | uv run headless.py test --model devstral2
```

Emits raw events to stdout for debugging. Accepts input from stdin.

***

## Configuration

Follows Mistral-like standard (so if you are coming from Mistral Vibe, your config items should be pretty similar)

### Config File (`~/.agent13/config.toml`)

```toml
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"
model = "gpt-4"                    # optional default model
read_timeout = 2400                # optional, seconds (default 2400=40min; increase for slow responding models)
connect_timeout = 30               # optional, seconds (default 30)

[[providers]]
name = "local"
api_base = "http://localhost:8080/v1"
api_key_env_var = "LOCAL_API_KEY"

# MCP servers
[[mcp_servers]]
name = "filesystem"
transport = "stdio"
command = "mcp-filesystem"
args = ["/path/to/allowed/dir"]

[[mcp_servers]]
name = "remote"
transport = "http"
url = "http://mcp-server:8080"

# Global tool filtering (applies to ALL tools: builtin + MCP)
# enabled_tools = ["read_*"]        # whitelist (empty = all pass)
# disabled_tools = ["square_number"] # blacklist (applied only if enabled_tools empty)

# Skills
# skill_paths = ["/path/to/custom/skills"]  # additional skill search paths
# include_skills = true                       # include skills in system prompt
```

### Environment Files

Loaded in order (later overrides earlier):

1. `~/.env`
2. `./.env` (project-local)

### Provider Resolution (`agent13/config.py`)

```python
def resolve_provider_arg(provider_arg: str) -> tuple[str, str, float, float]:
    """Resolve provider name or URL to (base_url, api_key, read_timeout, connect_timeout)."""
```

Accepts:

- Provider name from config: `openrouter`
- Direct URL: `http://localhost:8080/v1`

Returns a 4-tuple including the configured timeouts.

### Client Creation

```python
def create_client(
    base_url: str,
    api_key: str,
    read_timeout: float = 2400.0,
    connect_timeout: float = 30.0,
) -> AsyncOpenAI:
```

Creates an `AsyncOpenAI` client with separate connect and read timeouts. The default 2400s read timeout accommodates reasoning models that may think for long periods between tokens.

***

## MCP Integration

### MCP Manager (`agent13/mcp.py`)

The MCP (Model Context Protocol) system allows external tool servers. It uses a reconnect-per-operation pattern for reliability - each tool call reconnects to the server, executes, and disconnects.

```text
┌────────────────────────────────────────────────┐
│                 MCPManager                     │
│                                                │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ Stdio Server │  │ HTTP Server  │            │
│  │ (local proc) │  │ (remote)     │            │
│  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                    │
│         ▼                 ▼                    │
│  ┌──────────────────────────────────────────┐  │
│  │        Tool Registry                     │  │
│  │  - server_a: tool_1, tool_2              │  │
│  │  - server_b: tool_3                      │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

### Server Configuration

```toml
[[mcp_servers]]
name = "filesystem"
transport = "stdio"
command = "mcp-filesystem"
args = ["/path/to/allowed/dir"]
env = { DEBUG = "1" }                # optional env vars
enabled_tools = ["read_*"]           # per-server whitelist
disabled_tools = []                  # per-server blacklist
connect_timeout = 240.0              # optional
tool_timeout = 60.0                  # optional
retry_attempts = 3                   # optional
retry_delay = 1.0                    # optional

[[mcp_servers]]
name = "remote"
transport = "http"
url = "http://mcp-server:8080"
```

### Tool Calling

MCP tools are transparently integrated with local tools. The agent's `get_all_tools()` method merges both, applying the global config filter to MCP tools:

```python
async def get_all_tools(self) -> list[dict]:
    """Get combined built-in and MCP tools.

    Built-in tools are already filtered via get_filtered_tools() at
    init time and when set_devel_mode() is called.  MCP tools are
    filtered per-server at registration time.  This method also
    applies the global config enabled_tools/disabled_tools filter
    to MCP tools (which weren't filtered at registration).
    """
```

MCP tools are identified by the `mcp://` prefix in their name (e.g. `mcp://filesystem/read_file`). The agent routes these to the MCP manager for execution.

### Server Events

The MCP manager emits events for server lifecycle:

| Event                | When                                     |
| -------------------- | ---------------------------------------- |
| `MCP_SERVER_STARTED` | Server connection initiated              |
| `MCP_SERVER_READY`   | Server connected, tools available        |
| `MCP_SERVER_ERROR`   | Connection failed                        |
| `MCP_SERVER_STDERR`  | Server stderr output (captured via pipe) |

Stderr from MCP subprocess servers is captured using a pipe + reader thread (`StderrCapture`) and emitted line-by-line as events.

***

## Skills System

Skills are specialized capability modules loaded on demand:

```python
@tool
async def skill(name: str) -> dict:
    """Load a specialized skill by name. Returns instructions and bundled resources."""
```

Skills provide:

- Domain-specific instructions (loaded into the conversation)
- Bundled resources (scripts, templates)
- Workflow guidance
- Optional tool restrictions (`allowed_tools` limits which tools the skill can use)

### Skill Structure

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```text
~/.agent13/skills/
├── code-review/
│   ├── SKILL.md           # Instructions + frontmatter
│   └── templates/
│       └── review.md
└── git-workflow/
    ├── SKILL.md
    └── scripts/
        └── branch.sh
```

### Skill Frontmatter (`SKILL.md`)

```yaml
---
name: code-review
description: Review code for quality, security, and best practices
license: MIT
compatibility: ">=0.1.0"
allowed-tools: read_file edit_file command
user-invocable: true
---

# Code Review Skill

Instructions for the AI when this skill is loaded...
```

### Skill Discovery (`agent13/skills/manager.py`)

Skills are discovered from three search paths (highest priority first):

1. **Project skills**: `.agent13/skills/` in the current working directory
2. **Global skills**: `~/.agent13/skills/`
3. **Default skills**: Bundled with the package (`agent13/default_skills/`)

The `SkillManager` scans these directories for `SKILL.md` files, parses the frontmatter using `SkillMetadata` (Pydantic model), and creates `SkillInfo` objects.

### Skill Models (`agent13/skills/models.py`)

- `SkillMetadata`: Parsed frontmatter with validation (name pattern, description length, etc.)
- `SkillInfo`: Complete skill info including filesystem path, used at runtime

***

## Sandbox Security

### Sandbox Modes (`agent13/sandbox.py`)

| Mode                 | File Write  | File Read   | Network |
| -------------------- | ----------- | ----------- | ------- |
| `permissive-open`    | Project dir | Anywhere    | Allowed |
| `permissive-closed`  | Project dir | Anywhere    | Blocked |
| `restrictive-open`   | Project dir | Project dir | Allowed |
| `restrictive-closed` | Project dir | Project dir | Blocked |
| `none`               | Anywhere    | Anywhere    | Allowed |

The default mode is `permissive-open` on all platforms. Override with the `sandbox_mode` config option or `--sandbox` CLI flag.

### Implementation

Uses macOS Seatbelt sandboxing (`sandbox-exec`):

```python
async def run_sandboxed_async(
    command: str,
    mode: SandboxMode,
    timeout: float,
    max_output: int,
    project_dir: Path
) -> dict:
```

Sandbox profiles are loaded from `~/.agent13/sandbox/` and parsed to extract allowed read/write paths. The profiles are cached per mode for performance.

### User Control

Sandbox mode is controlled exclusively by the user:

```text
/sandbox permissive-closed
```

Tools cannot override the sandbox mode. The `SandboxCapabilities` dataclass describes what each mode allows, providing a single source of truth for UI display.

***

## Testing Strategy

### Test Structure

```text
tests/
├── conftest.py                    # Shared fixtures
├── test_core.py                   # Agent core functionality
├── test_events.py                 # Event emission/handling
├── test_queue.py                  # Queue management
├── test_llm.py                    # LLM interaction
├── test_tools.py                  # Tool execution
├── test_coercion.py               # Argument type coercion
├── test_tool_groups.py            # Tool group visibility
├── test_config.py                 # Configuration loading
├── test_mcp.py                    # MCP integration
├── test_sandbox.py                # Sandbox security
├── test_skills.py                 # Skill discovery/loading
├── test_prompts.py                # Prompt management
├── test_journal.py                # Journal compaction
├── test_interrupt_consistency.py  # Interrupt message repair
├── test_pause_resume_bugs.py      # Pause/resume state machine
├── test_token_usage.py            # Token usage tracking
├── test_reasoning_collapse.py     # Reasoning token handling
├── test_edit_file_*.py            # Edit file modes (line, fuzzy, AST, snapshot)
├── test_tui_commands.py           # TUI slash commands
├── test_tui_scroll.py             # TUI scroll behavior
├── test_security.py               # Security audit
├── test_snippets.py               # Snippet management
└── test_batch_mode.py             # Batch/headless mode
```

### Testing Workflow

1. **Manual testing with headless.py**:

```bash
printf "hello\n/quit\n" | uv run headless.py test --model devstral2
```

1. **Interactive testing with pexpect**:

```python
child = pexpect.spawn('uv run ./agent13.py test', encoding='utf-8')
child.expect('Agent started')
```

1. **Automated tests**:

```bash
uv run pytest tests/ -v
uv run pytest tests/test_core.py -v
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_core.py -v

# Lint check
flake8 agent13/ ui/ tools/ tests/

# Format check
ruff check agent13/ ui/ tools/ tests/
```

***

## Extensibility Points

1. **Custom tools**: Add `@tool` decorated functions in `tools/`. Use `groups=["devel"]` for developer-only tools.
2. **Custom events**: Add to `AgentEvent` enum in `agent13/events.py`.
3. **Custom UIs**: Subscribe to agent events via `@agent.on_event`.
4. **Custom MCP servers**: Add to `config.toml` under [[mcp_servers]].
5. **Custom skills**: Add skill directories with `SKILL.md` to `~/.agent13/skills/` or project `.agent13/skills/`.
6. **Tool filtering**: Use `enabled_tools`/`disabled_tools` in config with glob or regex patterns.
7. **Sandbox profiles**: Add custom profiles in `~/.agent13/sandbox/`.
