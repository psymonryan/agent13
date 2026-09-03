# agent13

<p align="center">
<a href="https://github.com/psymonryan/agent13/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/network/members"><img alt="GitHub forks" src="https://img.shields.io/github/forks/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/psymonryan/agent13?style=flat"></a>
     <a href="https://github.com/psymonryan/agent13/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat"></a>
</p>

**A self-coding AI agent that runs where others won't - built for tight VRAM, any OpenAI-compatible API.**

![Agent13 - built for tight spaces](./images/agent13maxing.png)

> Named after the agent from Get Smart who always seemed to end up in the tightest places - a mailbox, a fridge, a grandfather clock, and now a GPU?

- **Runs well in low VRAM environments** - works with any OpenAI-compatible API: llama-server, Ollama, vLLM, LM Studio, OpenRouter
- **Incremental compaction or full compaction with steering** ; journals tool responses so context stays small without losing information, or compacts the entire message thread
- **No telemetry** - privacy focussed
- **Screen Reader Friendly** - Has both a REPL mode and the option to also to split output to a separate file which can then be tailed into a text to speech engine such as Linux-Speakup

## How is this different?

### AI preferred Tools

Most AI agents fix tool-use issues by adding more instructions: *"do this, don't do that, make sure you always... etc etc"*

agent13 takes the opposite approach: *every tool was refined by watching how models actually used them, then modifying the tool to suit the AI's expectations.*

After applying this approach, tool-use success across open-weight models (Qwen, GLM, Kimi, Devstral) went from around 50% to near 95%, this is a real speedup when the agent is trying to get things done!  - PS. If you dont like this agent, then tell your agent to steal the edit tools from this agent. :grinning:

### Is agent13 for you?

**agent13 is for you if:**

- You run local models and need an agent that keeps VRAM context low
- You want to switch models/providers without restarting (mid session)
- You care about privacy
- You're comfortable with terminal interfaces
- You use a 'screen reader' and don't want cursor commands messing up your text to speech engine.

**Consider alternatives if:**

- You need a more polished GUI (but hey, this one's not bad!)
- You're all-in on Anthropic's ecosystem (Claude Code)
- You want managed infrastructure

## Features

**Reconfigurable mid-flight.** While the agent is processing, you can: change models, switch providers, pause/resume, save session for later, or inject interrupt prompts with the `!!`  prefix. Eg:

**Inflight Steering**, without cancelling the agents turn, you can inject interrupt prompts with the `!!`  prefix. Eg:

```
> !!Oh, and I forgot to mention that the doco you need lives in ~/mydocs
```

This means when you see the agent struggling with something, or you forgot to tell it something, you can provide this information without breaking/cancelling the current turn and experiencing loss of work or worse, another round of full prompt processing.

**TUI interface.** Full-featured Textual-based terminal UI: streaming responses, queue management (multiple prompts in flight), priority commands, info pane with context stats, session auto-save, and markdown rendering. Non-blocking input throughout.

**CLI interface.** Run one-shot prompts from the command line or scripts, with pretty mode (rich output) on or off. Ideal for automation, CI pipelines, and shell scripting.

**Skills and MCP** As you would expect.

**Sandbox mode.** Five security profiles from unrestricted to macOS Seatbelt sandboxing. Tools run isolated by default; escalate only when needed. Configurable per-session (`--sandbox`) or per-tool. (Note: command tool sandboxing currently macOS-only via Seatbelt.)

**Devel mode.** Toggle developer tools on/off at runtime. Hidden tools (TUI viewer, testing utilities) shown with `--devel` flag or `/devel on` in TUI. agent13 also comes with 'self development' tools, so if you ask the agent to change itself, it has tests and tools that help it change itself. So if you want to make your own mods, turn this on and let it self-modify.

**Mobile friendly.** Works over Turmux/Termius and similar mobile SSH clients. (you can slow down the activity spinner or turn it off)

**Tools that work WITH the Agent.** Automatically correcting for known LLM shortcomings with feedback, rollback and hinting on errors to help the agents next try.

## Quick start

Requires Python 3.11+. The steps below install and configure everything else.

### 1. Install uv

Install uv using [the official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) - it has the command for your platform.

If your terminal says `uv` is not recognised after installing, close and reopen it - uv adds itself to your PATH, and already-open terminals don't pick that up.

### 2. Install agent13

**From GitHub release** (recommended - latest stable):

```bash
uv tool install https://github.com/psymonryan/agent13/releases/download/v0.4.1/agent13-0.4.1-py3-none-any.whl
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

### 3. Create your config

```bash
agent13 --list-providers
```

On first run this creates your starter config at `~/.agent13/config.toml` (and a starter `~/.env` in your home directory), then lists the providers in it.

The starter config includes a few example providers. Open `~/.agent13/config.toml` and delete the `[[providers]]` blocks you don't use - each one looks like this:

```toml
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"
```

Running your own server (llama.cpp, vLLM, Ollama, ...)? Add a block pointing at it - the `name` is what you'll type on the command line:

```toml
[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = "OPENAI_API_KEY"
```

(The starter config also has an example MCP server - it stays dormant unless you pass `--mcp`.)

See [USER_GUIDE.md](USER_GUIDE.md) for advanced configuration (MCP servers, tool filtering, clipboard, timeouts, environment variables).

### 4. Set up your API key

Open `~/.env` (auto created in the previous step) and replace the `dummy` values with your real keys.

A provider is any server that speaks the OpenAI API. If you don't run your own, the easiest option is [OpenRouter](https://openrouter.ai) - create a key at [openrouter.ai/keys](https://openrouter.ai/keys) and paste it after `OPENROUTER_API_KEY=`.

agent13 loads `~/.env` first, then `./.env` in your working directory (local overrides global).

### 5. Choose a model

agent13 needs a model that supports tool calling (function calling); not all models do. List what your provider offers (replace `openrouter` with your provider name):

```bash
agent13 openrouter --model
```

Pick by name or number, or run without `--model` to choose interactively. On OpenRouter, skip models with a `:batch` suffix - those only work with its batch API, not the TUI. Good starting points:

| Model        | Provider            | Notes                                                                   |
| ------------ | ------------------- | ----------------------------------------------------------------------- |
| Qwen-3.8-27B | Local               | Excellent tool calling, good performance over the large context size    |
| GLM-5.3      | Local/Remote        | Good reasoning, excellent coding, not everyone can fit this one locally |
| GPT-4o       | OpenAI / openrouter | Cloud model, reliable tool calling (never used it myself)               |

### 6. Run the TUI

```bash
# Interactive TUI (prompts for model selection)
agent13 openrouter
```

On first run with a provider, agent13 lists available models:

```
  Available models:
    1. qwen-3.8-27B
    2. GLM-5.3-Flash-Next

  Select model (number or name, or 'q' to quit): _
```

Batch mode runs in your terminal (not the TUI) for one-shot prompts:

```bash
# Single prompt, exits after
agent13 openrouter --model 1 -p "Write a Python script to sum numbers 1 to 100"

# With MCP tools (servers auto-disconnect on exit)
agent13 openrouter --mcp -p "Use the deep_research skill to investigate X and write report.md"

# Specify model directly
agent13 openrouter --model qwen-3.8-27B
```

### 7. First conversation

Once the TUI is up:

1. Type a message and press `Enter`
2. Watch the response stream in token by token
3. Watch the agent call tools (read files, run commands, etc.)
4. Type `/help` to see the slash commands

Try these to explore:

```text
What tools do you have available?
Read the README.md file and summarize it.
List the files in the current directory.
I dont like the status bar colours, change them for me.
```

For the full reference (slash commands, config keys, modes) and troubleshooting, see [USER_GUIDE.md](USER_GUIDE.md).

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

| Option                      | Description                                                          | Default               |
| --------------------------- | -------------------------------------------------------------------- | --------------------- |
| `--list-providers`          | List providers from config and exit                                  | -                     |
| `--version`                 | Show version number and exit                                         | -                     |
| `-p, --prompt <text>`       | Batch mode with this prompt                                          | -                     |
| `--model <name>`            | Select model (number or name)                                        | prompts interactively |
| `--system-prompt <name>`    | System prompt to use                                                 | default               |
| `--sandbox <mode>`          | Set sandbox mode for session                                         | permissive-open       |
| `--pretty on\|off`          | Enable/disable markdown rendering                                    | on                    |
| `--debug`                   | Enable debug mode                                                    | off                   |
| `--tool-response raw\|json` | Tool output format                                                   | raw                   |
| `--mcp`                     | Connect to MCP servers on startup                                    | off                   |
| `--skills`                  | Include discovered skills in system prompt                           | off                   |
| `--journal`                 | Enable journal mode (context compaction)                             | off                   |
| `--remove-reasoning`        | Strip reasoning tokens between turns                                 | off                   |
| `-c, --continue`            | Resume previous session                                              | -                     |
| `--devel`                   | Show devel-group tools to AI                                         | off                   |
| `--spinner fast\|slow\|off` | Control spinner animation                                            | fast                  |
| `--upgrade`                 | Check for updates, install, exit                                     | -                     |
| `--clipboard osc52\|system` | Clipboard method                                                     | osc52                 |
| `--bell N\|off`             | Bell threshold in seconds (0=always ring, off=disable)               | off                   |
| `--polite N\|off`           | Polite mode: wait for shared provider lock (N=poll interval seconds) | 10                    |
| `--read FILE`               | Read file(s) into user message before processing                     | -                     |
| `--repl`                    | Run in REPL mode (readline-based, no TUI)                            | off                   |
| `--output FILE`             | Write REPL chat transcript to file (implies --repl)                  | -                     |

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

| Document                           | Description                                            |
| ---------------------------------- | ------------------------------------------------------ |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Event-driven architecture, code structure, tool design |
| [USER_GUIDE.md](USER_GUIDE.md)     | Full usage guide, all features in detail               |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, dev setup, PR process               |
| [CHANGELOG.md](CHANGELOG.md)       | Release history and changes                            |
| [AGENTS.md](AGENTS.md)             | AI agent instructions (for self-coding context)        |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Feedback

Agent13 does not collect telemetry, so if something is confusing, useful, annoying, or missing, let me know.

- **Bug reports**: https://github.com/psymonryan/agent13/issues
- **Discussions**: https://github.com/psymonryan/agent13/discussions
- **Source**: https://github.com/psymonryan/agent13

## Credits

Agent13 was bootstrapped using Mistral Vibe and then built by itself using local models: Qwen-3.5-27B, GLM-5, GLM-5.1, Kimi-K2.5 on llama-swap/llama-server, then oMLX. Features were typically started with Qwen-3.5-27B; when things got tricky, swapped to Kimi or GLM-5.1 on OpenRouter.

Inspired by the need for a lightweight, controllable agent that fits within VRAM constraints while remaining usable for long sessions.

Built 100% by itself (after initial bootstrap) under frustrated (at times) human guidance. :sweat_smile:
