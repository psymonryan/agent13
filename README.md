# README

![Agent13 - built for tight spaces](./images/agent13maxing.png)

> Named after the agent from *Get Smart* who always seemed to end up in the tightest places - a mailbox, a fridge, a grandfather clock. Agent13 is the AI that still opererates when there is a VRAM squeeze

Agent13 is a self-coding AI agent with tools designed to work the way the AI expects them to work, and to run on hardware that bigger agents won't touch. It was built 100% by itself (after an initial bootstrap using Mistral Vibe) using local models (Qwen-3.5-27B, GLM-5.X, Kimi-K2.5).

## ✨ Why agent13?

**Tool design that works.**\
\
Most agents fix poor tool-use by adding more instructions to the system prompt - longer lists of "do this, don't do that." Agent13 takes the opposite approach: every tool was refined by watching how the AI actually used it, and then redesigned when it failed. The result is near-100% tool-use success with many open-weight models, other agents I tried gave me 50%+ failure rates when I used them outside of their model ecosystem (eg: Mistral Vibe when using GLM, Qwen, MiniMax, Kimi rather than devstral)

**Runs where others won't.**\
\
Designed to run with 24 GB VRAM setups. Works with any OpenAI-compatible API - Llama-swap, Ollama, LM Studio, oMLX, llama-server, or cloud providers. Models that go silent for 10+ minutes are handled gracefully with configurable read timeouts. (eg: If you just swapped model and it is taking a while to load)

**Clean architecture, no surprises.**\
\
Event-driven throughout. Library-first design - import `from agent13 import Agent` for scripting. Headless mode with structured output for CI and testing / self-debugging. Tool groups, MCP servers, skills, and sandbox modes and more.

## Features

- **Near-100% tool-use success** - Tools designed around AI expectations, refined through real usage
- **Low VRAM native** - Runs on 24 GB GPUs, any OpenAI-compatible API
- **Reconfigure Mid-Flight** - While the agents turn is running, give it hints, steer its direction, change its model or provider, pause it and save session for later, all designed to minimise time wasted doing extra prompt processing.
- **Event-driven architecture** - Clean separation between agent core, tools, and UI
- **TUI interface** - Rich terminal UI with queue management, streaming output, and priority commands
- **Auto-discovered tools** - Add your own tools as functions and decorate them with `@tool`and they're automatically registered
- **Skills system** - Reusable instruction sets that extend agent capabilities
- **MCP integration** - Connect to Model Context Protocol servers for extended tool access
- **Multiple modes** - Interactive TUI, batch mode, headless, or importable library
- **Sandbox mode** - Safe tool execution with configurable profiles (5 modes) - (note: command tool implements sandboxing only under macos atm.)
- **Devel mode** - Toggle developer tools on/off at runtime
- **No telemetry** - No tracking, no analytics, no phoning home.

## 🚀 Quick Start

### Installation

Clone from source (not yet on PyPI):

```bash
git clone https://github.com/psymonryan/agent13
cd agent13
uv sync
```

Directly Install:

```bash
uv tool install -e .
```

Or build the wheel and install

```text
uv build
uv tool install dist/agent13-x.x.x-py3-none-any.whl
```

### Uninstall

```text
uv tool uninstall agent13
```

### Configuration

Create `~/.agent13/config.toml`:

```toml
[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = "OPENAI_API_KEY"

# For slower providers that need longer timeouts
[[providers]]
name = "laptop"
api_base = "http://laptop.local.home:8012/v1"
api_key_env_var = "OPENAI_API_KEY"
read_timeout = 2400  # 40 min for long load and response times
```

Set your API key in `~/.env`:

```bash
OPENAI_API_KEY=your_key_here
```

### First Run

```bash
# Interactive TUI (will prompt you for model name/number)
agent13 local

# Batch mode (single prompt, exits after processing)
agent13 local -p "Write a Python script to parse logs"

# List available providers
agent13 --list-providers

# Run on local provider and select model 5 (or give model name)
agent13 local --model 5
```

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [TUI Mode](#tui-mode)
  - [Batch Mode](#batch-mode)
  - [Headless Mode](#headless-mode)
  - [Library Mode](#library-mode)
- [TUI Reference](#-tui-reference)
  - [Keybindings](#keybindings)
  - [Slash Commands](#slash-commands)
- [Tools](#-tools)
  - [Built-in Tools](#built-in-tools)
  - [Adding Tools](#adding-tools)
  - [Tool Groups](#tool-groups)
  - [Tool Filtering](#tool-filtering)
- [Skills](#-skills)
  - [Default Skills](#default-skills)
  - [Creating Skills](#creating-skills)
  - [Skill Discovery Paths](#skill-discovery-paths)
- [Configuration](#-configuration)
  - [Provider Configuration](#provider-configuration)
  - [MCP Server Configuration](#mcp-server-configuration)
  - [Tool Filtering Configuration](#tool-filtering-configuration)
  - [Environment Variables](#environment-variables)
- [Architecture](#-architecture)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🖥️ Usage

### TUI Mode

The primary interface. Run with a provider name from your config:

```bash
agent13 local
```

Features available in TUI mode:

- **Streaming responses** - See tokens as they arrive
- **Queue management** - Multiple prompts in flight, cancel anytime
- **Priority commands** - Start your message with `!` to insert at the front of the queue to execute when the agent finishes its current task or `!!` to interrupt the agent loop and insert your prompt without cancelling.
- **Info pane** - Context stats, model info, queue status
- **Session auto-save** - Conversations saved on exit, resume with `--continue`

### Batch Mode

One-shot processing for scripting and automation:

```bash
agent13 local -p "Explain the event system"
```

Options:

- `--model <name>` - Select a specific model
- `--pretty on|off` - Enable/disable markdown rendering (default: on)
- `--tool-response raw|json` - Tool output format (default: raw for most models)
- `--sandbox <mode>` - Set sandbox mode for this session
- `--skills` - Include discovered skills in the system prompt
- `--journal` - Enable journal mode (progressive context compaction)
- `--send-reasoning` - Include reasoning tokens in message history
- `--remove-reasoning` - Strip reasoning tokens between turns
- `--devel` - Show devel-group tools to the AI (so agent can run itself in tui mode when testing)
- `--continue` - Resume the previous session (TUI mode)
- `--spinner fast|slow|off` - Control the spinner animation (TUI mode)

### Headless Mode

Minimal event printer for debugging and scripting (used by the Agent when self modifying):

```bash
printf "hello\n/quit\n" | uv run headless.py local --model devstral2
```

### Library Mode

Import agent13 as a Python library:

```python
from agent13 import Agent, AgentEvent, AgentEventData

agent = Agent(client, model="devstral")

@agent.on_event
async def handler(event: AgentEventData):
    if event.event == AgentEvent.ASSISTANT_TOKEN:
        print(event.text, end="")

await agent.add_message("Hello!")
await agent.run()
```

## 🎛️ TUI Reference

### Keybindings

| Key                       | Action                                                          |
| ------------------------- | --------------------------------------------------------------- |
| `Esc`                     | Interrupt current processing                                    |
| `Ctrl+C`                  | Clear input if not empty; interrupt if processing; quit if idle |
| `Ctrl+D` / `Ctrl+Q`       | Force quit                                                      |
| `Shift+Up` / `Shift+Down` | Scroll chat                                                     |
| `Ctrl+O`                  | Toggle reasoning collapse                                       |
| `Ctrl+Y`                  | Copy full markdown of selected message                          |
| `Ctrl+Shift+C`            | Copy rendered selection                                         |
| `Enter`                   | Submit input / close panes                                      |

### Slash Commands

| Command                               | Description                                       |
| ------------------------------------- | ------------------------------------------------- |
| `/help`                               | Show help                                         |
| `/quit` `/exit`                       | Exit the application                              |
| `/clear`                              | Clear conversation                                |
| `/history`                            | Show input history                                |
| `/delete`                             | Delete from history, queue, or saves              |
| `/model`                              | Switch model                                      |
| `/list`                               | List models                                       |
| `/tools`                              | List active tools                                 |
| `/skills`                             | List discovered skills                            |
| `/prompt`                             | Switch system prompt                              |
| `/sandbox`                            | Change sandbox mode                               |
| `/provider`                           | Switch provider                                   |
| `/queue`                              | Show queue status                                 |
| `/pause` `/resume`                    | Pause/resume processing                           |
| `/retry`                              | Retry last interaction                            |
| `/prioritise` `/deprioritise`         | Change queue item priority                        |
| `/mcp connect\|disconnect\|reload`    | Manage MCP servers                                |
| `/journal on\|off\|last\|all\|status` | Journal mode control                              |
| `/devel on\|off\|status`              | Toggle devel tool visibility                      |
| `/remove-reasoning on\|off`           | Strip reasoning between turns                     |
| `/save` `/load`                       | Save/load conversation context (-y to force save) |
| `/snippet`                            | Manage text snippets                              |
| `/spinner fast\|slow\|off`            | Spinner style                                     |
| `/pretty on\|off`                     | Markdown rendering                                |
| `/tool-response raw\|json`            | Tool output format                                |

## 🛠️ Tools

Tools are Python functions decorated with `@tool`:

```python
from tools import tool

@tool
def read_file(filepath: str, offset: int = None, limit: int = None) -> str:
    """Read a file. Optional offset/limit for partial reads."""
    ...
```

### Built-in Tools

| Tool            | Description                                   |
| --------------- | --------------------------------------------- |
| `command`       | Execute shell commands with sandboxing        |
| `read_file`     | Read file contents (raw, skim, or line-based) |
| `write_file`    | Write content to files                        |
| `edit_file`     | Line-based and AST-based file editing         |
| `skill`         | Load specialized skills                       |
| `square_number` | Demo tool                                     |
| `tui_viewer`    | TUI testing tools (devel group)               |

### Adding Tools

1. Create `tools/my_tool.py`
2. Decorate functions with `@tool`
3. Include docstrings (used for tool descriptions)
4. Restart agent13 to pick up new tools

Tools are auto-discovered from the `tools/` package directory.

### Tool Groups

Tools can be assigned to groups that control visibility:

```python
@tool(groups=["devel"])
def tui_launch(provider: str, model: str) -> str:
    """Launch the TUI in a headless PTY."""
    ...
```

The `devel` group is hidden by default. Enable with `--devel` flag or `/devel on` in TUI.

### Tool Filtering

Control which tools are active using patterns in `config.toml`:

```toml
# Whitelist: only these tools are active
enabled_tools = ["read_*", "edit_*"]

# Blacklist: these tools are disabled (applied when enabled_tools is empty)
disabled_tools = ["square_number", "re:^tui_.*$"]
```

Patterns support:

- **Exact names**: `read_file`
- **Glob patterns**: `read_*`, `tui_*`
- **Regex** (prefix with `re:`): `re:^tui.*$`

## 📚 Skills

Skills are reusable instruction sets in `SKILL.md` files:

```markdown
---
name: code-review
description: Perform automated code reviews
allowed-tools:
  - read_file
  - edit_file
user-invocable: true
---

# Code Review Skill

This skill helps analyze code quality and suggest improvements.
```

### Default Skills

Agent13 ships with these default skills (copied to `~/.agent13/skills/` on first run):

- `manage-skills` - Create and manage new skills (or fix downloaded skills)
- `get-new-skill` - Find and adapt skills from external sources
- `humanizer` - Remove AI writing patterns
- `context7` - Context lookup for libraries and APIs

### Creating Skills

1. Create `~/.agent13/skills/my-skill/SKILL.md`
2. Add YAML frontmatter with metadata
3. Write instructions in markdown below
4. Reference with `/skill my-skill` in TUI (if `user-invocable: true`)

### Skill Discovery Paths

Skills are discovered from three locations:

1. **Project skills**: `.agent13/skills/` in your project directory
2. **Global skills**: `~/.agent13/skills/`
3. **Bundled defaults**: `agent13/default_skills/` (copied to global on first run)

## ⚙️ Configuration

### Provider Configuration

```toml
[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = "OPENAI_API_KEY"
model = "qwen-3.6-27b"        # optional default model
read_timeout = 600            # optional, seconds (default 600=10min; use 2400 for reasoning models)
connect_timeout = 30          # optional, seconds (default 30)
```

Provider names can also be URLs:

```bash
agent13 http://localhost:8012/v1 --model devstral2
```

### MCP Server Configuration

```toml
# HTTP transport (remote server)
[[mcp_servers]]
name = "my_server"
transport = "http"
url = "http://localhost:8000/mcp"

# stdio transport (local process)
[[mcp_servers]]
name = "web_research"
transport = "stdio"
command = "uvx"
args = ["web-research-assistant"]
env = { "SEARXNG_BASE_URL" = "http://searxng/search" }

# Per-server tool filtering
enabled_tools = []
disabled_tools = []
```

Supported transports: `http`, `stdio`

MCP tools are named using the pattern `mcp://server_name/tool_name` and appear alongside built-in tools.

### Tool Filtering Configuration

Global tool filtering applies to all tools (built-in + MCP):

```toml
# Whitelist (if non-empty, only matching tools are active)
enabled_tools = ["read_*", "edit_*", "mcp://web_research/*"]

# Blacklist (applied only when enabled_tools is empty)
disabled_tools = ["square_number"]
```

### Environment Variables

API keys are loaded from `~/.env` then `./.env` (local overrides global):

```bash
# ~/.env
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

## 🏗️ Architecture

```text
agent13/
├─ __init__.py    # Public exports (Agent, events, queue, config, etc.)
├─ __main__.py    # Entry point (agent13 command)
├─ core.py        # Agent class, event emission, status management
├─ events.py      # AgentEvent enum (28 events), AgentEventData
├─ queue.py       # Prompt queue with priorities and deferred ops
├─ llm.py         # Streaming, tool calls, message building
├─ tools.py       # Tool discovery and execution
├─ config.py      # Provider config, create_client() factory
├─ prompts.py     # System prompt management
├─ sandbox.py     # Sandbox profiles and capabilities
├─ mcp.py         # MCP server manager
├─ batch.py       # Batch-mode runner for scripting
├─ persistence.py # Context save/load
├─ history.py     # prompt_toolkit history integration
├─ snippets.py    # Snippet management
├─ context.py     # Context management utilities
├─ config_paths.py # Config directory resolution
├─ yaml_store.py  # YAML persistence
├─ debug_log.py   # Debug logging infrastructure
├─ models.py      # Model listing and selection
├─ cli.py         # Command-line argument parsing and main()
├─ skills/
│  ├─ __init__.py
│  ├─ manager.py  # Skill discovery and loading
│  ├─ models.py   # SkillMetadata, SkillInfo
│  └─ parser.py   # SKILL.md frontmatter parser
└─ default_skills/ # Bundled skills (context7, humanizer, etc.)

ui/
├─ tui.py         # Textual TUI (async, full-featured)
└─ display.py     # Rich display helpers

tools/
└─ *.py           # Auto-discovered tool implementations
```

**Event flow:** User input → Queue → Agent → LLM stream → Tool calls → Events → UI update

For the full architecture document, see ARCHITECTURE.md.

## 🛠️ Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
flake8 agent13/ ui/ tools/ tests/

# Format
ruff format .

# Interactive development
uv run agent13.py local_provider_name
```

### Testing

Agent13 includes TUI testing tools for headless UI verification:

```python
from tools.tui_viewer import tui_launch, tui_type, tui_screenshot

tui_launch(provider="test", model="devstral2")
tui_type("hello world")
tui_press("enter")
screenshot = tui_screenshot()
```

See ARCHITECTURE.md for the full testing strategy.

## 🐛 Troubleshooting

**ReadTimeout on reasoning models** - Add `read_timeout = 2400` to provider config in `~/.agent13/config.toml`.

**Debug log** - Check `~/.agent13/debug.log` for detailed session events and API request details.

**Stale environment** - `rm -rf .uv_cache && uv sync` to force dependency refresh.

**Session recovery** - Use `agent13 local --continue` to resume from last auto-saved session.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

Agent13 was bootstrapped using Mistral Vibe and then built by itself using local models: Qwen-3.5-27B, GLM-5, GLM-5.1, Kimi-K2.5 on llama-swap/llama-server, then oMLX. Typically features were started with Qwen-3.5-27B then when it got tricky, I would swap to Kimi or GLM-5.1 running on openrouter.

Inspired by the need for a lightweight, controllable agent that respects VRAM constraints whilst remaining usable for long sessions.
