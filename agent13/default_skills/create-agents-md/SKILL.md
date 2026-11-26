---
name: create-agents-md
description: "What: Create concise, AI-agent-optimized AGENTS.md files for repositories. When: Use when setting up new projects or standardizing documentation for AI agents."
license: MIT
compatibility: Requires read_file, write_file, grep, and basic Markdown knowledge
metadata:
  version: 1.0.0
  author: Agent13
  source: "trimmed from the original 494-line version"
  created: 2024-08-01
  updated: 2024-08-01
  category: documentation
  tags:
    - documentation
    - agents
    - best-practices
    - templates
    - project-setup
---

# Create AGENTS.md Skill

## Getting Started

This skill helps create concise, bullet-point AGENTS.md files optimized for AI agents. The format is designed to be:
- Extremely terse (no explanations)
- Focused on project-specific details
- Token-efficient
- Actionable

## Core Concepts

### AGENTS.md Purpose
AGENTS.md files provide project-specific guidance to AI agents, complementing generic knowledge. They should contain:
- Project-specific workflows
- Repository structure
- Build and deployment instructions
- Coding standards
- Testing procedures
- Troubleshooting tips

### Key Principles
1. **Terse format**: Only bullet points, no explanations
2. **Non-obvious only**: For each bullet ask "would an experienced developer already know this?" — if yes, cut it
3. **Skip empty sections**: If nothing non-obvious to say for a section, omit it entirely — don't pad with generic content
4. **Length target**: Aim for 30-60 lines. Over 80 lines means obvious content has crept in
5. **Actionable**: Focus on what to do, not why
6. **Token-efficient**: Minimize content while maximizing usefulness
7. **Project-specific**: Omit generic knowledge (PEP 8, ES6+, "check logs", "set memory limits" — agents already know this)

## Step-by-Step Guide

### Step 1: Gather Repository Information

Read key files to understand the project:

```bash
read_file(path="readme-sdr.txt")
read_file(path="package.json")
read_file(path="pyproject.toml")
read_file(path="src/index.html")
read_file(path="server.py")
```

### Step 2: Identify Non-Obvious Content

From the files you've read, extract only:
- Custom scripts, tools, or deployment commands
- Non-standard ports, paths, or configurations
- Version-specific gotchas or workarounds
- Project-specific conventions that differ from norms

Skip anything an experienced developer would already know. If a section has nothing non-obvious, omit it entirely.

### Step 3: Create AGENTS.md Content

Only include sections with non-obvious, project-specific content. Skip sections entirely if there's nothing to say. Use whatever section names fit the project — don't force the standard template.

```markdown
# AGENTS.md

## Setup
- Project-specific prerequisites and gotchas
- Non-obvious environment variables or configs

## Code structure
- Non-standard file locations
- Deprecated or confusing paths

## Commands
- Custom scripts or tools
- Non-obvious flags or arguments

## Gotchas
- Things that trip up even experienced developers
- Version-specific issues
```

### Step 4: Write the File

```python
write_file(
    path="AGENTS.md",
    content="""# AGENTS.md

## Setup
- Python 3.11 required (3.12 breaks SQLAlchemy)
- `uv sync && cp .env.example .env`
- DB: Postgres on port 5433 (not 5432)

## Code structure
- `app/api/v2/` — new API, v1 is deprecated
- `scripts/migrate.py` — runs on deploy, not automatically

## Gotchas
- `SECRET_KEY` must be 32+ bytes or JWT silently fails
- Tests require running Redis (no mock fallback)
"""
)
```

## Examples

### Example 1: Python Project

```markdown
# AGENTS.md

## Dev environment
- Python 3.11 required (3.12 breaks SQLAlchemy)
- `uv sync && cp .env.example .env`
- Run: `uv run uvicorn app.main:app --reload`
- DB: Postgres on port 5433 (not 5432)

## Code structure
- `app/api/v2/` — new API, v1 is deprecated
- `app/models/` — SQLAlchemy, NOT Pydantic for DB models
- `scripts/migrate.py` — runs on deploy, not automatically

## Gotchas
- `SECRET_KEY` must be 32+ bytes or JWT silently fails
- Tests require running Redis (no mock fallback)
```

### Example 2: Docker Swarm Project

```markdown
# AGENTS.md

## Setup
- `source env.lab3` before ANY swctl command
- Docker host: 192.168.9.50 (node: blue)

## swctl
- `./swctl <service> deploy` — auto-routes to traefik stack if YAML has traefik labels
- `./swctl <service> exec -c "cmd"` — non-interactive (our addition)
- `stop/start/restart` require swarmctl on node blue

## Volumes
- `poolM1` — NVMe (configs, app data)
- `pool4TBwd` — HDD (media, databases)
```

## Best Practices

### The Obvious Test

For each bullet point, ask: **"Would an experienced developer already know this?"**

If yes, **cut it**. Examples of obvious content to always omit:

- "PEP 8 compliant", "ES6+", "Prettier formatting" — agents know language standards
- "Set CPU/memory limits" — generic best practice
- "Check logs when debugging" — obvious
- "Docker Configs are mounted read-only" — obvious Docker behaviour
- "Run tests before committing" — obvious
- "Use descriptive variable names" — obvious

### What to Include

- Project-specific gotchas and non-obvious behaviour
- Custom scripts, tools, or workflows unique to this project
- Non-standard port numbers, paths, or configurations
- Dependencies on specific versions or environments
- Gotchas that trip up even experienced developers

### Validation Checklist

- [ ] Under 60 lines (over 80 = too verbose)
- [ ] Every section has non-obvious content (no padding)
- [ ] Would an experienced developer learn something from each bullet?
- [ ] No generic advice ("check logs", "set limits", "follow PEP 8")
- [ ] Only bullet points, no paragraphs
- [ ] Exact commands and file paths (not vague descriptions)


## Next Steps

1. **Analyze the repository**: Read key files to understand structure
2. **Identify project-specific details**: What makes this project unique?
3. **Create AGENTS.md**: Write concise, bullet-point format
4. **Validate**: Check against the validation checklist
5. **Review**: Ensure all critical information is included
6. **Update**: Keep AGENTS.md current with project changes
