# Agent13 User Guide

Complete reference for using Agent13 - configuration, tools, skills, TUI commands, and advanced features.

## Table of Contents

- [Quick Start](#quick-start)
- [Running Agent13](#running-agent13)
  - [TUI Mode](#tui-mode)
  - [Batch Mode](#batch-mode)
  - [Headless Mode](#headless-mode)
  - [Library Mode](#library-mode)
- [Command-Line Options](#command-line-options)
- [TUI Reference](#tui-reference)
  - [Keybindings](#keybindings)
  - [Slash Commands](#slash-commands)
  - [Input Features](#input-features)
- [Configuration](#configuration)
  - [Provider Configuration](#provider-configuration)
  - [MCP Server Configuration](#mcp-server-configuration)
  - [Tool Filtering](#tool-filtering)
  - [Clipboard Configuration](#clipboard-configuration)
  - [Environment Variables](#environment-variables)
- [Tools](#tools)
  - [Built-in Tools](#built-in-tools)
  - [Tool Groups and Devel Mode](#tool-groups-and-devel-mode)
  - [Adding Custom Tools](#adding-custom-tools)
- [Skills](#skills)
  - [Skill Format](#skill-format)
  - [Default Skills](#default-skkills)
  - [Creating Skills](#creating-skills)
  - [Managing Skills](#managing-skills)
- [Queue and Priority](#queue-and-priority)
- [Sandbox Modes](#sandbox-modes)
- [Session Management](#session-management)
- [Journal Mode](#journal-mode)
- [MCP Integration](#mcp-integration)
- [Troubleshooting](#troubleshooting)
- [Updates](#updates)
- [Clipboard Configuration](#clipboard-configuration)

## Quick Start

### Install

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install package directly from github:

```
uv tool install https://github.com/psymonryan/agent13/releases/download/v0.1.9/agent13-0.1.9-py3-none-any.whl
```

Or install from source (for hacking on the agent itself):

```bash
git clone https://github.com/psymonryan/agent13
cd agent13
uv sync
uv run agent13.py      # run from source
# or
uv tool install -e .   # install as editable tool
```

### Uninstall

```text
uv tool uninstall agent13
```

### Update

```text
agent13 --upgrade
```

Or inside a running session:

```text
/upgrade
```

### Configuration

`~/.agent13/config.toml`: # Sample is created for you on first run

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
agent13 local -p "Write a Python script to add all the numbers from 1 to 100"

# List available providers
agent13 --list-providers

# Run on local provider and select model 5 (or give model name)
agent13 local --model 5
```

## Running Agent13

### TUI Mode

The primary interface - a full-featured terminal UI (using Textual library) with streaming, queue management, and slash commands:

```bash
agent13 local
agent13 openrouter --model devstral2
```

The TUI provides:

- **Streaming token output** - See responses arrive in real time
- **Reasoning display** - Collapsible view of model reasoning tokens
- **Queue management** - Submit multiple prompts, manage queue priorities, or bypass queue with an !!interrupt command for "inflight steering"
- **Info pane** - Context stats, model info, queue depth, token usage
- **Tab completion** - Slash commands, model names, file paths
- **Auto-save on exit** - Conversations saved to `~/.agent13/saves/`
- Manual save of entire session (-y to force over an old one)

### Batch Mode

One-shot processing for scripting and automation:

```bash
agent13 local -p "Explain the event system" --pretty off
```

Batch mode processes the prompt, prints the response, and exits. Useful for:

- CI/CD pipelines
- Shell script integration
- Quick one-off queries
- Easy markdown output for copy and paste when rendering unwanted

### Headless Mode (for self improvement)

Minimal event printer for debugging:

```bash
printf "hello\n/quit\n" | uv run headless.py local --model devstral2
```

Outputs structured events:

```text
READY
EVENT: STARTED
USER: hello
ASSISTANT: Hello! How can I help?
EVENT: STOPPED
```

### Library Mode

Import agent13 as a Python library for custom integrations:

```python
import asyncio
from openai import AsyncOpenAI
from agent13 import Agent, AgentEvent, AgentEventData, create_client

async def main():
    client = create_client("http://localhost:8012/v1", "your-api-key")
    agent = Agent(client, model="devstral2")

    @agent.on_event
    async def handler(event: AgentEventData):
        if event.event == AgentEvent.ASSISTANT_TOKEN:
            print(event.text, end="", flush=True)
        elif event.event == AgentEvent.ASSISTANT_COMPLETE:
            print()  # newline after response

    await agent.add_message("Hello!")
    await agent.run()

asyncio.run(main())
```

The `Agent` constructor accepts:

- `client: AsyncOpenAI` - API client
- `model: str` - Model name
- `system_prompt: str` - System prompt text
- `messages: list[dict]` - Initial message history
- `tools: list[dict]` - Tool schemas for function calling
- `execute_tool: Callable` - Tool execution function (sync or async)
- `journal_mode: bool` - Enable context compaction
- `send_reasoning: bool` - Include reasoning in message history
- `remove_reasoning: bool` - Strip reasoning between turns
- `devel_mode: bool` - Show devel-group tools

## Command-Line Options

| Option                      | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `provider`                  | Provider name from config or OpenAI-compatible URL          |
| `--list-providers`          | List available providers and exit                           |
| `--version`                 | Show version and exit                                       |
| `-p`, `--prompt`            | Run in batch mode with this prompt                          |
| `--model`                   | Select model by name or number; with no value, lists models |
| `--system-prompt`           | System prompt to use                                        |
| `--sandbox`                 | Sandbox mode (permissive-open, permissive-closed, etc.)     |
| `--pretty on\|off`          | Enable/disable markdown rendering (default: on)             |
| `--debug`                   | Enable debug logging                                        |
| `--tool-response raw\|json` | Tool response format (default: raw)                         |
| `--mcp`                     | Connect to MCP servers on startup (TUI only)                |
| `--skills`                  | Include discovered skills in the system prompt              |
| `--journal`                 | Enable journal mode (context compaction)                    |
| `--send-reasoning`          | Include reasoning_content in message history                |
| `--remove-reasoning`        | Strip reasoning tokens between turns                        |
| `-c`, `--continue`          | Continue from last auto-saved session                       |
| `--devel`                   | Enable devel mode (show devel-group tools)                  |
| `--spinner fast\|slow\|off` | Spinner style (default: fast)                               |
| `--upgrade`                 | Check for updates and apply, then exit                      |
| `--clipboard osc52\|system` | Clipboard method for this session (default: osc52)          |

## TUI Reference

### Keybindings

| Key                       | Action                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- |
| `Esc`                     | Interrupt current processing                                                          |
| `Ctrl+C`                  | Clear input field; quit if input is empty                                             |
| `Ctrl+D` / `Ctrl+Q`       | Force quit                                                                            |
| `Shift+Up` / `Shift+Down` | Scroll chat history                                                                   |
| `Ctrl+O`                  | Toggle collapse of most recent reasoning block                                        |
| `Ctrl+Y`                  | Copy full markdown of message containing selection (Select any text with mouse first) |
| `Ctrl+Shift+C`            | Copy rendered text selection                                                          |
| `Enter`                   | Submit input; close panes/modals on empty input                                       |
| `Tab`                     | Autocomplete (slash commands, model names, file paths)                                |
| Mouse drag select         | Auto copies selection to paste buffer                                                 |

### Slash Commands

Commands are typed in the input field with a `/` prefix:

#### Session Control

| Command             | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `/help`             | Show available commands                                |
| `/quit` `/exit`     | Exit the application                                   |
| `/clear`            | Clear conversation context                             |
| `/clear all`        | Clear conversation and reset UI widgets                |
| `/save [name] [-y]` | Save conversation context to file (optional overwrite) |
| `/load [name]`      | Load conversation context from file                    |

#### Model and Provider

| Command            | Description                             |
| ------------------ | --------------------------------------- |
| `/model [name]`    | Switch model (tab-complete for list)    |
| `/model`           | List available models                   |
| `/provider [name]` | Switch provider (tab-complete for list) |
| `/prompt [name]`   | Switch system prompt                    |

#### Queue Management

| Command              | Description                    |
| -------------------- | ------------------------------ |
| `/queue`             | Show queue status              |
| `/pause`             | Pause processing               |
| `/resume`            | Resume processing              |
| `/retry`             | Retry last interaction         |
| `/prioritise [id]`   | Promote queue item to priority |
| `/deprioritise [id]` | Demote queue item to normal    |

#### Display

| Command                     | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `/pretty on\|off`           | Toggle markdown rendering                                      |
| `/tool-response raw\|json`  | Toggle tool output format api request                          |
| `/spinner fast\|slow\|off`  | Change spinner style for slow connections (eg from your phone) |
| `/remove-reasoning on\|off` | Strip reasoning between turns                                  |

#### Tools and Skills

| Command                  | Description                                  |
| ------------------------ | -------------------------------------------- |
| `/tools`                 | List active tools                            |
| `/skills`                | List discovered skills                       |
| `/sandbox [mode]`        | Change sandbox mode (tab-complete for modes) |
| `/devel on\|off\|status` | Toggle devel tool visibility                 |

#### MCP

| Command           | Description                       |
| ----------------- | --------------------------------- |
| `/mcp connect`    | Connect to configured MCP servers |
| `/mcp disconnect` | Disconnect from MCP servers       |
| `/mcp reload`     | Reload MCP server connections     |

#### Journal

| Command            | Description                       |
| ------------------ | --------------------------------- |
| `/journal on\|off` | Enable/disable journal mode       |
| `/journal last`    | Journal the last turn             |
| `/journal all`     | Journal all turns that used tools |
| `/journal status`  | Show journal mode status          |

#### Other

| Command                      | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `/history`                   | Show input history                                         |
| `/delete [h\|q\|s] num`      | Delete items from history queue, or saves                  |
| `/snippet`                   | Manage text snippets (saved user messages)                 |
| `/upgrade [--copy]`          | Check for updates and apply (or copy command to clipboard) |
| `/clipboard [osc52\|system]` | Show or set clipboard method (default: osc52)              |

#### Skill Commands

Each installed skill appears as a slash command. For example, a skill named `code-review` can be invoked with:

```text
/code-review
```

This sends the skill's content as a message to the agent.

### Input Features

- **Tab completion** - Type `/` then `Tab` to see slash commands; complete model names, file paths
- **Priority messages** - Append `!` for priority, `!!` for interrupt
- **Multi-line input** - Use `^J`
- **Snippet expansion** - Save frequent prompts as snippets, invoke as `/snippet-name`

## Configuration

Agent13 is configured via `~/.agent13/config.toml`. On first run, a default config is created from the bundled template.

### Provider Configuration

```toml
[[providers]]
name = "local"                           # Short name used on command line
api_base = "http://localhost:8012/v1"    # OpenAI-compatible API endpoint
api_key_env_var = "OPENAI_API_KEY"       # Env var name for the API key
model = "qwen3-27b"                      # Optional default model
read_timeout = 2400                      # Optional: seconds to wait for tokens (default 2400=40min)
connect_timeout = 30                     # Optional: seconds for initial connection (default 30)
```

You can also use a URL directly as the provider:

```bash
agent13 http://localhost:8012/v1 --model devstral2
```

**Timeout tips:**

- Default `read_timeout` is 600 seconds (10 minutes)
- Reasoning models (DeepSeek-R1, GLM-5.1) may need `read_timeout = 2400` (40 minutes)
- If you see `ReadTimeout` errors, increase this value

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

**Transport types:**

- `http` - Connect to a remote HTTP server
- `stdio` - Launch a local process and communicate via stdin/stdout

**Key fields:**

- `name` - Short alias (used in tool names as `mcp://name/tool`)
- `transport` - Transport type
- `url` - Base URL for HTTP transport
- `command` / `args` - Command to run for stdio transport
- `env` - Environment variables for stdio transport
- `enabled_tools` / `disabled_tools` - Per-server tool filtering (same pattern syntax as global)

MCP tools appear alongside built-in tools and are named with the `mcp://` prefix to distinguish them.

### Tool Filtering

Control which tools are active using patterns in `config.toml`:

```toml
# Whitelist: if non-empty, ONLY matching tools are active
enabled_tools = ["read_*", "edit_*"]

# Blacklist: applied when enabled_tools is empty
disabled_tools = ["square_number", "re:^tui_.*$"]
```

**Pattern syntax:**

- **Exact name**: `read_file` - matches exactly
- **Glob**: `read_*`, `tui_*` - shell-style wildcards
- **Regex** (prefix `re:`): `re:^tui.*$` - full match against tool name

Tool filtering applies to both built-in and MCP tools.

### Clipboard Configuration

Control how text is copied to the clipboard:

```toml
[clipboard]
method = "osc52"    # "osc52" (terminal escape sequence, default) or "system" (OS clipboard command)
```

**Methods:**

| Method   | How it works                                         | Best for                                                                                    |
| -------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `osc52`  | OSC 52 terminal escape sequence                      | SSH sessions, modern terminals (Alacritty, Ghostty, Kitty, iTerm2, Windows Terminal v1.18+) |
| `system` | OS commands: `pbcopy`, `xclip`/`wl-copy`, `clip.exe` | tmux, screen, older terminals, Windows conhost/PowerShell                                   |

**When to switch:** If mouse select or Ctrl+Y doesn't copy to your clipboard, you likely need `system`. Type `/clipboard system` in the TUI — it persists to your config file.

**Switching at runtime:**

- `/clipboard` — show current method
- `/clipboard system` — switch to OS clipboard commands
- `/clipboard osc52` — switch back to terminal escape sequence

**CLI override:**

```bash
agent13 --clipboard system studio
```

### Environment Variables

API keys are loaded from `~/.env` then `./.env` (local overrides global):

```bash
# ~/.env (global)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# ./.env (project-local - loaded after ~/.env, so values here override)
OPENAI_API_KEY=sk-project-key
```

## Tools

### Built-in Tools

| Tool            | Description                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| `command`       | Execute shell commands with sandbox enforcement                                                                 |
| `read_file`     | Read file contents - skim (symbols), raw (lines), or offset/limit                                               |
| `write_file`    | Write content to files (fails if exists unless `overwrite=True`)                                                |
| `edit_file`     | Line-based editing (replace, append, prepend, delete, rollback) and AST-based edits                             |
| `skill`         | Load a specialized skill by name                                                                                |
| `square_number` | Demo/example tool (used for testing)                                                                            |
| `tui_viewer`    | TUI testing tools: launch, screenshot, type, press, wait, quit (devel group - wont load unless you use --devel) |
| `self_update`   | Check for updates, apply upgrade, or copy install command to clipboard                                          |

### Tool Groups and Devel Mode

Tools can be assigned to groups that control their visibility to the AI:

```python
@tool(groups=["devel"])
def tui_launch(provider: str, model: str) -> str:
    """Launch the TUI in a headless PTY."""
    ...
```

The `devel` group is hidden by default. Enable with:

```bash
# Command line
agent13 local --devel

# TUI slash command
/devel on
```

When devel mode is off, the AI cannot see or use tools in the `devel` group. This keeps the tool list lean for normal usage.

### Adding Custom Tools

1. Create a Python file in the `tools/` directory (e.g., `tools/my_tool.py`)
2. Decorate functions with `@tool`:

```python
from tools import tool

@tool
def my_tool(name: str, count: int = 5) -> str:
    """Do something useful.

    Args are automatically converted from JSON to Python types.
    Docstrings become the tool description shown to the AI.
    """
    return f"Processed {name} {count} times"
```

3. Optional: assign groups and timeouts:

```python
@tool(groups=["devel"], timeout=30.0)
def slow_tool(data: str) -> str:
    """A tool that might take a while."""
    ...
```

4. Restart agent13 or use `/mcp reload` to pick up new tools

Tools are auto-discovered from the `tools/` package directory. Each `.py` file (except `__init__.py` and files starting with `_`) is imported and scanned for `@tool`-decorated functions.

## Skills

Skills are reusable instruction sets that extend the agent's capabilities. They're defined in `SKILL.md` files with YAML frontmatter, following the [agentskills.io](https://agentskills.io/specification) specification.

### Finding and Creating Skills

Type `/get-new-skill` then ask agent13 to look for the skill you need. Use `/manage-skills` to create a new skill, improve an existing skill, or validate skill structure for standards compliance. (both of these slash commands are skills themselves)

For the full specification including frontmatter fields and best practices, see [agentskills.io](https://agentskills.io/specification).

### Default Skills

Agent13 ships with these skills (copied to `~/.agent13/skills/` on first run):

| Skill           | Description                                 |
| --------------- | ------------------------------------------- |
| `manage-skills` | Create and manage skills                    |
| `get-new-skill` | Find and adapt skills from external sources |
| `humanizer`     | Remove AI writing patterns from text        |
| `context7`      | Context lookup for libraries and APIs       |

### Managing Skills

- **List skills**: `/skills` in TUI
- **Invoke skill**: `/skill-name` or ask the AI to use the `skill` tool
- **Skill paths**: Skills are discovered from:
  1. Project directory: `.agent13/skills/`
  2. Global directory: `~/.agent13/skills/`
  3. Bundled defaults: `agent13/default_skills/` (auto-copied to global on first run)
     
     ## Queue and Priority

Agent13 uses a message queue to manage multiple prompts:

- **Normal priority** - Messages processed in order
- **Priority** (`!` suffix) - Inserts command at the top of the queue until agent finishes the current turn
- **Interrupt** (`!!` suffix) - Leaves agent running but inserts the command immediately into the existing turn. (useful to steer agent with extra knowledge while it is running - no need to stop it and restart the turn)

```text
Fix the bug in main.py!       # Priority - goes ahead of normal items
URGENT: Stop and answer this!!  # Interrupt - pauses current work
```

Queue commands:

- `/queue` - View current queue
- `/prioritise [id]` - Promote an item
- `/deprioritise [id]` - Demote an item
- `/pause` - Pause processing
- `/resume` - Resume processing

## Sandbox Modes

Sandbox modes control what tools can do - file access and network permissions:

| Mode                 | File Write   | File Read    | Network | Use Case                                |
| -------------------- | ------------ | ------------ | ------- | --------------------------------------- |
| `permissive-open`    | Project only | Anywhere     | ✅       | Default - safe writes, open reads       |
| `permissive-closed`  | Project only | Anywhere     | ❌       | No network - safe for offline work      |
| `restrictive-open`   | Project only | Project only | ✅       | Locked reads - sandboxed exploration    |
| `restrictive-closed` | Project only | Project only | ❌       | Maximum restriction - fully locked down |
| `none`               | Anywhere     | Anywhere     | ✅       | No restrictions - full access           |

**Default**: `permissive-open` on all platforms.

Set sandbox mode:

```bash
# Command line
agent13 local --sandbox restrictive-closed

# TUI
/sandbox restrictive-closed
```

## Session Management

Agent13 auto-saves your conversation on exit:

```bash
# Resume last session from global ~/.agent13 folder
agent13 local --continue
```

Manual save/load: (this saves in current working directory. ie: ./agent13/saves)

```text
/save my-session     # Save current context
/load my-session     # Load a saved context
```

Saves are stored in `~/.agent13/saves/`.

## Journal Mode

Journal mode enables context compaction - after each turn, a journal summary replaces the tool calls, with intention / result for each tool (this is far more effective that trying to use 'entire session' compaction.

```bash
# Enable on startup
agent13 local --journal

# Or in TUI
/journal on
```

Journal commands:

- `/journal on` - Enable journal mode
- `/journal off` - Disable journal mode
- `/journal last` - Convert the last turn into a journal summary
- `/journal all` - Iterate over all turns, journalling each
- `/journal status` - Show current status

## MCP Integration

MCP (Model Context Protocol) servers extend Agent13 with additional tools:

```bash
# Connect on startup
agent13 local --mcp

# Or in TUI
/mcp connect
```

MCP tools appear with the `mcp://` prefix:

- `mcp://web_research/web_search` - Web search via SearXNG
- `mcp://fetch_server/fetch` - URL fetching

Configure MCP servers in `~/.agent13/config.toml`:

```toml
[[mcp_servers]]
name = "web_research"
transport = "stdio"
command = "uvx"
args = ["web-research-assistant"]
env = { "SEARXNG_BASE_URL" = "http://searxng/search" }
```

MCP servers use a "reconnect-per-operation" pattern - they connect when needed and handle disconnections gracefully.

## Snippets

Snippets are reusable text templates you can insert into your input. Create them in `.agent13/snippets.yaml` in your project or home directory:

```toml
document_journey: document our findings, please create a new document that covers,
  the initial goals, all the problems we investigated and what the findings were,
  what we tried, what worked and what didnt work and how we got around it. Document
  enough detail such that if another AI was to read this, it would not need to replicate
  any step of the journey you just took. Write tersely for an AI rather than a human.
  Call the document docs_archive/featureX_journey.md
erudite: For this session you are a terse, erudite, dotpoint documenter. You never
  pad text, you just give the minimum as dotpoints, you dont even introduce or summarise
  what you are saying or writing to a file. Now just say "I understand"
square: square all the numbers from 1.06 to 3.06, step 1.0 for me (do one at a time
  rather than all in parallel)
```

Use `/snippet` in the TUI to see available snippets and insert one. Snippets can use `{placeholder}` syntax - you'll be prompted to fill in values before insertion.

## Reasoning Token Control

Agent13 can include or exclude reasoning tokens (chain-of-thought) in message history:

- `--send-reasoning` - Include the model's reasoning tokens the api call (defaults to on) - currently it does nothing, this was used as an experiment and will be removed.
- `--remove-reasoning` - Strip reasoning tokens between turns. Reduces context size and focuses on final answers. Recommended for most use cases.

Toggle reasoning visibility in the TUI with `Ctrl+O` to collapse/expand reasoning content.

## Prompt Management

Customize the system prompt with `--system-prompt`:

```bash
agent13 local --system-prompt my-prompt
```

The prompt name references a file in `~/.agent13/prompts/` (without the `.md` extension). You can also add custom content that gets appended to the default prompt using the `custom_additions` mechanism in the prompt file.

Use `/prompt` in the TUI to see the current system prompt.

## History Management

Use `/history` to view the full conversation history with message IDs. Use `/delete <id>` to remove specific messages from the conversation:

```text
/history        # List all messages with IDs
/delete 5       # Remove message 5
/delete 3-7     # Remove messages 3 through 7
```

Deleted messages are removed from the context sent to the model. This is useful for cleaning up off-topic tangents or mistakes.

## Updates

Agent13 checks GitHub releases for new versions on startup (once per day by default). When an update is available, you'll see a notification like:

```
⬆ Update available: v0.1.9 (you have 0.1.8). Type /upgrade to apply. Or run: uv tool install --force <wheel-url>
Disable these notifications: set check_enabled = false in [updates] section of ~/.agent13/config.toml
```

### Applying Updates

| Method              | How                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------- |
| `/upgrade`          | In the TUI — checks, downloads wheel from GitHub, installs, then tells you to restart |
| `/upgrade --copy`   | Copies the manual install command to clipboard instead of running it                  |
| `agent13 --upgrade` | Non-interactive — check and apply from the command line, then exit                    |
| Manual              | Run the `uv tool install --force <url>` command from the notification                 |

### Update Configuration

```toml
[updates]
check_enabled = true           # set false to disable startup notifications
check_interval_hours = 24     # minimum hours between checks
```

## Troubleshooting

### ReadTimeout on Slow Models / Providers

Models like DeepSeek-R1 and GLM-5.1 can go silent for 10+ minutes while loading with a local provider. Add to your provider config:

```toml
read_timeout = 2400  # 40 minutes
```

### Provider Unreachable

Check your API server is running:

```bash
curl http://localhost:8012/v1/models
```

Verify the URL in `~/.agent13/config.toml` matches your server.

### Model Doesn't Use Tools

Not all models support tool/function calling. Try a model known for good tool support (see [Getting Started](GETTING_STARTED.md) for recommendations).

### Clipboard Not Working

If mouse select, Ctrl+Y, or `/upgrade --copy` doesn't copy to your clipboard:

1. Type `/clipboard system` in the TUI to switch to OS clipboard commands
2. This persists to your config — you only need to do it once
3. On Linux, ensure `xclip` (X11) or `wl-copy` (Wayland) is installed

See [Clipboard Configuration](#clipboard-configuration) for details.

### Debug Logging

Enable debug logging to see detailed session events:

```bash
agent13 local --debug
```

Or check the debug log after a session:

```bash
cat ~/.agent13/debug.log
```

### Stale Environment

If dependencies are stale or broken:

```bash
uv sync
```

### Session Recovery

If the TUI crashes or you accidentally close it:

```bash
agent13 local --continue
```

This resumes from the last auto-saved session.
