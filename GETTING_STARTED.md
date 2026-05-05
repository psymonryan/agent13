# Getting Started with agent13

A step-by-step guide to get agent13 running for the first time.

## Prerequisites

- **Python 3.11+** - agent13 requires Python 3.11 or later
- **uv** - Modern Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **An OpenAI-compatible API** - Any server that speaks the OpenAI chat completions API with tool calling support

### Compatible API Servers

Agent13 works with any OpenAI-compatible endpoint:

- **llama-server** - Local inference via llama.cpp
- **llama-swap** - Multi-model swap server for llama.cpp / mlx / proxied requests
- **vLLM** - High-throughput local inference
- **OpenRouter** - Cloud proxy to many models
- **Ollama** - Local model runner (with OpenAI-compatible endpoint) - based on llama.cpp (which tbh I'd prefer)
- **Any OpenAI API** - Direct connection to OpenAI (never actually tried this)

## Step 1: Install

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install package directly from github:

```
uv tool install https://github.com/psymonryan/agent13/releases/download/v0.1.11/agent13-0.1.11-py3-none-any.whl
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

## Step 2: Set Up Your API Key

Agent13 looks for API keys in environment variables. Create `~/.env`:

```bash
# For local servers (key can be anything)
OPENAI_API_KEY=local

# For OpenRouter
OPENROUTER_API_KEY=sk-or-v3-...

# For OpenAI directly
OPENAI_API_KEY=sk-...
```

Agent13 loads `~/.env` on startup, then `./.env` (local overrides global).

## Step 3: Create Your Config

Create `~/.agent13/config.toml` with at least one provider:

```toml
# Local llama-server
[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = "OPENAI_API_KEY"

# Remote server with longer timeout for reasoning models
[[providers]]
name = "remote"
api_base = "http://myserver:8012/v1"
api_key_env_var = "OPENAI_API_KEY"
read_timeout = 2400  # 40 minutes for deep thinking

# OpenRouter
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"
```

Verify your providers:

```bash
agent13 --list-providers
```

You should see your providers listed with their API base URLs.

## Step 4: Choose a Model

Agent13 needs a model that supports **tool calling** (function calling). Not all models support this.

List available models for a provider:

```bash
agent13 local --model
```

This fetches the model list from your API server. Select a model by name or number:

```bash
# Run with a specific model
agent13 local --model qwen3-27b

# Or just run without --model to pick interactively
```

### Recommended Models

For best results with Agent13, use models with strong tool-calling support:

| Model        | Provider            | Notes                                                                   |
| ------------ | ------------------- | ----------------------------------------------------------------------- |
| Qwen-3.6-27B | Local               | Excellent tool calling, good context handling                           |
| devstral2    | Local               | Strong coding and tool use, but probably superceded by Qwen now         |
| GLM-5.1      | Local/Remote        | Good reasoning, excellent coding, not everyone can fit this one locally |
| Kimi-K2.5    | Local               | Very smart reasoning model - but GLM writes better code (IMO)           |
| GPT-4o       | OpenAI / openrouter | Cloud model, reliable tool calling (never used it myself ;-) )          |

> **Tip:** If you're using a reasoning model (DeepSeek-R1, GLM-5.1), add `read_timeout = 2400` to your provider config. These models can go silent for 10+ minutes while loading into VRAM from a slow disk.

## Step 5: Run Agent13

### Interactive TUI (recommended)

```bash
agent13 local
```

This opens the Textual TUI with streaming output, queue management, and slash commands.

### Batch Mode

For one-shot processing:

```bash
agent13 local -p "Explain the event system in agent13"
```

Batch mode processes the prompt and exits.

### From Source

If you cloned from source:

```bash
# TUI mode
uv run agent13.py local

# Batch mode
uv run agent13.py local -p "Hello"
```

## Step 6: Your First Conversation

Once the TUI is running:

1. **Type a message** and press `Enter`
2. **Watch the streaming response** appear token by token
3. **Observe tool calls** - the agent may read files, run commands, etc.
4. **Use slash commands** - type `/help` to see available commands

Try these prompts to explore:

```text
What tools do you have available?
Read the README.md file and summarize it.
List the files in the current directory.
I dont like the status bar colours, change them for me.
```

## Step 7: Explore Features

### Priority Messages

Start your message with `!` to mark it as priority (cuts ahead in the queue):

```text
!Fix the bug in main.py
```

Start with `!!` for an interrupt (inserts command into running agent loop without stopping):

```text
!!I see you are looking in the wrong place, the docs actually live in ~/mydocs
```

### Provider and model swapping (mid-flight)

I'm running out of context on this laptop, let me swap to a different provider even though the model is half way through working on a problem.

```bash
/pause  # pause the model at the next safe point
/provider mybiggermachine  # swap to bigger VRAM machine
/model smartermodel  # Change models if you like
```

### Session Continuation

Agent13 auto-saves your session on exit. Resume with:

```bash
agent13 local --continue
```

### Enable Skills

Skills extend the agent with specialized instructions:

```bash
# Enable skills on startup (this costs a small amount of context)
agent13 local --skills
```

Or toggle at runtime with `/skills` in the TUI. (more efficient since agent doesnt have to pick and load)

### MCP Servers

Connect to MCP servers for additional tools:

```bash
# Connect on startup
agent13 local --mcp
```

Or use `/mcp connect` in the TUI.

## Step 8: Customize

### Add a Custom Tool

Create `tools/my_tool.py`:

```python
from tools import tool

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"
```

Restart agent13 and the tool is automatically available.

### Skills

Type `/get-new-skill` then ask agent13 to look for the skill you need. Use `/manage-skills` to create a new skill, improve an existing skill, or validate skill structure for standards compliance. (both of these slash commands are skills themselves)

For the full skill specification, see [agentskills.io](https://agentskills.io/specification).

### Configure Sandbox Mode

Control what tools can do:

```bash
# On the command line
agent13 local --sandbox permissive-closed

# Or in the TUI
/sandbox permissive-closed
```

Available modes: `permissive-open`, `permissive-closed`, `restrictive-open`, `restrictive-closed`, `none`

## Common Issues

### "Provider is unreachable"

Check that your API server is running and the URL in `config.toml` is correct:

```bash
curl http://localhost:8012/v1/models
```

### "No providers configured"

Create `~/.agent13/config.toml` with at least one [[providers]] entry.

### ReadTimeout errors

Loading / switching models can take a long time. Add to your provider config:

```toml
read_timeout = 2400  # 40 minutes
```

### Model doesn't use tools

Not all models support tool calling. Try a different model from the list (`agent13 local --model`).

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for detailed usage documentation
- Read ARCHITECTURE.md to understand how the system works
- Check AGENTS.md for development guidelines
