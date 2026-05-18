# agent13

<p align="center">
<a href="https://github.com/psymonryan/agent13/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/network/members"><img alt="GitHub forks" src="https://img.shields.io/github/forks/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat"></a>
</p>

**A self-coding AI agent that runs where others won't — built for tight VRAM, any OpenAI-compatible API.**

![Agent13 - built for tight spaces](./images/agent13maxing.png)

> Named after the agent from Get Smart who always seemed to end up in the tightest places — a mailbox, a fridge, a grandfather clock, and now a GPU?

## How is this Agent Harness any different?

### AI preferred Tools

Most AI agents fix tool-use issues by adding more instructions: *"do this, don't do that, make sure you always... etc etc"*

agent13 takes the opposite approach: *every tool was refined by watching how models actually used them, then modifying the tool to suit the AI's expectations.*

After applying this approach, tool-use success across open-weight models (Qwen, GLM, Kimi, Devstral) went from around 50% to near 95%, which equates to an effective 2x speedup when the agent is trying to get things done!  (Which for local models is a huge time saver)  - PS. If you dont like this agent, then tell your agent to steal the edit tools from this agent. :grinning:

### First Class support for Local Providers

Local-first, provider-flexible. Runs on 24 GB GPUs with any OpenAI-compatible endpoint (llama-server, Ollama, LM Studio, vLLM, OpenRouter).

### Incremental Compaction

So-called *full history compaction* just doesnt seem to work, especially with small context local setups. The agent ends up throwing away too much information, and 'auto-compaction' is a royal pain, as soon as the agent starts making headway, compaction starts and it forgets what it is doing!

agent13 uses incremental journalling (use --journal to enable), so for every turn where tools are used, the agent reflects on what it does and rewords the sometimes extremely token heavy tool responses into what it was trying to do and what it learned.

There is no point keeping a 20k token file read, if the agent was just checking how things are structured.

This approach means that the context stays small and the agent doesnt lose information on each step it has taken and what it was attempting.

Also, and perhaps more critically, this approach keeps the kv-cache snapshots valid, since we are only ever modifying the 'end' of the context being sent to the api.

## Features

**Reconfigurable mid-flight.** While the agent is processing, you can: change models, switch providers, pause/resume, save session for later, or inject interrupt prompts with the `!!`  prefix. Eg:

**Inflight Steering**, without cancelling the agents turn, you can inject interrupt prompts with the `!!`  prefix. Eg:

```
> !!Oh, and I forgot to mention that the doco you need lives in ~/mydocs
```

This means when you see the agent struggling with something, or you forgot to tell it something, you can provide this information without breaking/cancelling the current turn and experiencing loss of work or worse, another round of prompt processing.

**TUI interface.** Full-featured Textual-based terminal UI: streaming responses, queue management (multiple prompts in flight), priority commands, info pane with context stats, session auto-save, and markdown rendering. Non-blocking input throughout.

**CLI interface.** Run one-shot prompts from the command line or scripts, with pretty mode (rich output) on or off. Ideal for automation, CI pipelines, and shell scripting.

**Skills and MCP** As you would expect.

**Sandbox mode.** Five security profiles from unrestricted to macOS Seatbelt sandboxing. Tools run isolated by default; escalate only when needed. Configurable per-session (`--sandbox`) or per-tool. (Note: command tool sandboxing currently macOS-only via Seatbelt.)

**Devel mode.** Toggle developer tools on/off at runtime. Hidden tools (TUI viewer, testing utilities) shown with `--devel` flag or `/devel on` in TUI. agent13 also comes with 'self development' tools, so if you ask the agent to change itself, it has tests and tools that help it change itself.

**No telemetry.** No tracking, no analytics, no phoning home.

**Mobile friendly.** Works over Turmux/Termius and similar mobile SSH clients.

## Why agent13?

**agent13 is for you if:**

- You run local models and need an agent that respects VRAM constraints
- You want to switch models/providers without restarting
- You need zero telemetry for client work
- You want to add custom tools as simple Python functions
- You're comfortable with terminal interfaces

**Consider alternatives if:**

- You need a polished GUI (try Cursor)
- You're all-in on Anthropic's ecosystem (Claude Code)
- You want managed infrastructure (cloud agents)

## Quick start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install agent13

**From GitHub release** (recommended — latest stable):

```bash
uv tool install https://github.com/psymonryan/agent13/releases/download/v0.1.13/agent13-0.1.13-py3-none-any.whl
```

**From source** (for development):

```bash
git clone https://github.com/psymonryan/agent13
cd agent13
uv sync
uv run agent13.py      # run from source
# or
uv tool install -e .   # install as editable
```

### 3. Set up your API key

Agent13 loads API keys from environment variables. Create `~/.env`:

```bash
# For local servers (key can be anything)
OPENAI_API_KEY=local

# For OpenRouter
OPENROUTER_API_KEY=sk-or-v3-...

# For OpenAI directly
OPENAI_API_KEY=sk-...
```

Agent13 loads `~/.env` first, then `./.env` (local overrides global).

### 4. Create your config

Modify the default provided `~/.agent13/config.toml`:

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
read_timeout = 2400  # 40 minutes for super slow loading

# OpenRouter
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"
```

Verify providers:

```bash
agent13 --list-providers
```

See [USER_GUIDE.md](USER_GUIDE.md) for advanced configuration (MCP servers, tool filtering, clipboard, timeouts, environment variables).

### 5. Run the TUI

```bash
# Interactive TUI (prompts for model selection)
agent13 local
```

On first run with a provider, agent13 lists available models:

```
  Available models:
    1. qwen-3.5-27b
    2. devstral2
    3. glm-5.1

  Select model (number or name, or 'q' to quit): _
```

```bash
# Batch mode (single prompt, exits after)
agent13 local --model 3 -p "Write a Python script to sum numbers 1 to 100"

# Specify model directly
agent13 local --model qwen-3.5-27b
```

### Debugging

Use: `uv run ./utils/analyse-debug.py` to explore the possibly huge debug.log (you need to first run with --devel option)

## Commands

| Command                                                     | Description                                       |
| ----------------------------------------------------------- | ------------------------------------------------- |
| `agent13 <provider>`                                        | Launch TUI with specified provider                |
| `agent13 <provider> -p "prompt"`                            | Batch mode with single prompt                     |
| `agent13 --list-providers`                                  | List configured providers                         |
| `agent13 --version`                                         | Show version and exit                             |
| `agent13 --update`                                          | Check for updates and install                     |
| `agent13 --model`                                           | List available models (no value) or select one    |
| `printf "prompt\n/quit\n" \| uv run headless.py <provider>` | Headless mode used by agent13 when self-modifying |

All commands accept `--help` for full option listing.

## Options

| Option                      | Description                                | Default               |
| --------------------------- | ------------------------------------------ | --------------------- |
| `--list-providers`          | List providers from config and exit        | —                     |
| `--version`                 | Show version number and exit               | —                     |
| `-p, --prompt <text>`       | Batch mode with this prompt                | —                     |
| `--model <name>`            | Select model (number or name)              | prompts interactively |
| `--system-prompt <name>`    | System prompt to use                       | default               |
| `--sandbox <mode>`          | Set sandbox mode for session               | permissive-open       |
| `--pretty on\|off`          | Enable/disable markdown rendering          | on                    |
| `--debug`                   | Enable debug mode                          | off                   |
| `--tool-response raw\|json` | Tool output format                         | raw                   |
| `--mcp`                     | Connect to MCP servers on startup          | off                   |
| `--skills`                  | Include discovered skills in system prompt | off                   |
| `--journal`                 | Enable journal mode (context compaction)   | off                   |
| `--send-reasoning`          | Include reasoning tokens in history        | off                   |
| `--remove-reasoning`        | Strip reasoning tokens between turns       | off                   |
| `-c, --continue`            | Resume previous session                    | —                     |
| `--devel`                   | Show devel-group tools to AI               | off                   |
| `--spinner fast\|slow\|off` | Control spinner animation                  | fast                  |
| `--upgrade`                 | Check for updates, install, exit           | —                     |
| `--clipboard osc52\|system` | Clipboard method                           | osc52                 |

## Key bindings

| Key                 | Action                                         |
| ------------------- | ---------------------------------------------- |
| `Enter`             | Submit message                                 |
| `Ctrl+J`            | Insert new line                                |
| `Ctrl+B`            | History previous (prefix-matched)              |
| `Ctrl+F`            | History next (prefix-matched)                  |
| `Esc`               | Interrupt agent                                |
| `Ctrl+C`            | Clear input → interrupt → quit                 |
| `Ctrl+D` / `Ctrl+Q` | Force quit                                     |
| Mouse scroll        | Scroll chat (disables auto-scroll)             |
| Mouse select        | Select text in chat (auto-copies to clipboard) |
| `Ctrl+Y`            | Copy full markdown of selected message         |
| `Ctrl+O`            | Toggle collapse on most recent reasoning block |
| `Tab` / `Shift+Tab` | Cycle completions (commands, files, params)    |

## Compatible API Servers

Agent13 works with any OpenAI-compatible endpoint that supports tool calling:

| Server           | Type  | Notes                                           |
| ---------------- | ----- | ----------------------------------------------- |
| **llama-server** | local | llama.cpp, most reliable tool calling           |
| **llama-swap**   | local | Multi-model swap, supports llama.cpp/mlx/proxy  |
| **vLLM**         | local | High-throughput inference                       |
| **Ollama**       | local | OpenAI-compatible endpoint, tool support varies |
| **LM Studio**    | local | User-friendly, OpenAI-compatible                |
| **oMLX**         | local | Apple Silicon native                            |
| **OpenRouter**   | cloud | Proxy to many models                            |
| **OpenAI API**   | cloud | Direct connection                               |

## Documentation

| Document                                 | Description                                            |
| ---------------------------------------- | ------------------------------------------------------ |
| [ARCHITECTURE.md](ARCHITECTURE.md)       | Event-driven architecture, code structure, tool design |
| [USER_GUIDE.md](USER_GUIDE.md)           | Full usage guide, all features in detail               |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Step-by-step setup walkthrough                         |
| [CONTRIBUTING.md](CONTRIBUTING.md)       | How to contribute, dev setup, PR process               |
| [CHANGELOG.md](CHANGELOG.md)             | Release history and changes                            |
| [AGENTS.md](AGENTS.md)                   | AI agent instructions (for self-coding context)        |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Feedback

Agent13 does not collect telemetry, so if something is confusing, useful, annoying, or missing, let me know.

- **Bug reports**: https://github.com/psymonryan/agent13/issues
- **Discussions**: https://github.com/psymonryan/agent13/discussions
- **Source**: https://github.com/psymonryan/agent13

## Credits

Agent13 was bootstrapped using Mistral Vibe and then built by itself using local models: Qwen-3.5-27B, GLM-5, GLM-5.1, Kimi-K2.5 on llama-swap/llama-server, then oMLX. Features were typically started with Qwen-3.5-27B; when things got tricky, swapped to Kimi or GLM-5.1 on OpenRouter.

Inspired by the need for a lightweight, controllable agent that fits within VRAM constraints while remaining usable for long sessions.

Built 100% by itself (after initial bootstrap) under frustrated (at times) human guidance. :sweat_smile:
