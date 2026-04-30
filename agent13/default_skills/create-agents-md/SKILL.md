---
name: create-agents-md
description: "What: Create concise, AI-agent-optimized AGENTS.md files for repositories. When: Use when setting up new projects or standardizing documentation for AI agents."
license: MIT
compatibility: Requires read_file, write_file, grep, and basic Markdown knowledge
metadata:
  version: 1.0.0
  author: Agent13
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
2. **Project-specific**: Omit generic knowledge agents already have
3. **Actionable**: Focus on what to do, not why
4. **Token-efficient**: Minimize content while maximizing usefulness
5. **Consistent structure**: Use standard sections for easy navigation

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

### Step 2: Identify Key Sections

Based on the repository, determine which sections are needed:

- **Dev environment**: Setup instructions
- **Build**: Build commands
- **Code structure**: Key files and directories
- **Style**: Coding standards
- **Testing**: How to test
- **Commits**: Commit message format
- **PRs**: Pull request guidelines
- **Security**: Security considerations
- **Performance**: Performance notes
- **Deployment**: Deployment instructions
- **Troubleshooting**: Common issues
- **Docs**: Documentation location

### Step 3: Create AGENTS.md Content

Write the AGENTS.md file with the following structure:

```markdown
# AGENTS.md

## Dev environment
- Required tools and versions
- Setup commands
- Run commands

## Build
- Build commands
- Optimization commands

## Code structure
- Backend files
- Frontend files
- Data files

## Style
- Language-specific standards
- Framework conventions

## Testing
- Manual testing instructions
- Automated test commands

## Commits
- Commit message format
- Allowed types

## PRs
- PR title format
- Required information
- Quality checks

## Security
- Validation requirements
- Security best practices

## Performance
- Performance considerations
- Optimization notes

## Deployment
- Build instructions
- Run instructions

## Troubleshooting
- Common issues
- Debugging steps

## Docs
- Documentation location
- Architecture decisions
```

### Step 4: Write the File

```python
write_file(
    path="AGENTS.md",
    content="""# AGENTS.md

## Dev environment
- Python 3.11+, Node.js 18+
- `poetry install`, `npm install`
- Run: `RELEASE_MODE=src poetry run python server.py`
- Debug: `DEBUG_MODE=True RELEASE_MODE=src poetry run python server.py`

## Build
- `npm run build`, `npm run minify-html`, `npm run optimize-css`

## Code structure
- Backend: `server.py`, `itmz_processor.py`, `websocket_lib.py`
- Frontend: `src/js/*.js`, `src/css/map.css`
- Data: `maps/*.itmz`

## Style
- Python: PEP 8, type hints
- JS: ES6+, RequireJS
- CSS: rem units, mobile-first

## Testing
- Manual: Browser at `http://localhost`
- No automated tests

## Commits
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, style, refactor

## PRs
- Title: `<type>(<scope>): <subject>`
- Include: problem, testing, screenshots
- Check: no console errors

## Security
- Validate WebSocket messages
- Secure cookies
- Sanitize HTML

## Performance
- Large maps: virtual scrolling needed
- Caching: 5 min cache headers

## Deployment
- Build: `npm run build`
- Run: `RELEASE_MODE=dist poetry run python server.py`

## Troubleshooting
- WebSocket: check port 80
- Build: `rm -rf node_modules && npm install`

## Docs
- Docs in `docs/` directory
- Architecture decisions documented
"""
)
```

## Examples

### Example 1: Python Project

```markdown
# AGENTS.md

## Dev environment
- Python 3.9+
- `pip install -r requirements.txt`
- Run: `python app.py`

## Build
- No build required

## Code structure
- `app.py`: Main application
- `utils/`: Utility functions
- `tests/`: Test cases

## Style
- PEP 8 compliant
- Type hints required
- 100 character line limit

## Testing
- `pytest tests/`
- Coverage: 90%

## Commits
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, refactor
```

### Example 2: JavaScript Project

```markdown
# AGENTS.md

## Dev environment
- Node.js 16+
- `npm install`
- Run: `npm start`

## Build
- `npm run build`
- `npm run test`

## Code structure
- `src/`: Source files
- `public/`: Static assets
- `tests/`: Test files

## Style
- ES6+
- Prettier formatting
- 80 character line limit

## Testing
- Jest tests
- Coverage: 85%

## Commits
- Conventional commits
- Types: feat, fix, docs
```

## Common Patterns

### Dev Environment Section

Always include:
- Required tools and versions
- Setup commands (install dependencies)
- Run commands (development mode)
- Debug commands (if different)

Do not include:
- License

```markdown
## Dev environment
- Python 3.10+, Node.js 18+
- `pip install -r requirements.txt`, `npm install`
- Run: `python main.py`
- Debug: `DEBUG=True python main.py`
```

### Build Section

Include all build-related commands:

```markdown
## Build
- `npm run build`
- `npm run lint`
- `npm run test:coverage`
```

### Code Structure Section

Organize by component:

```markdown
## Code structure
- Backend: `server.py`, `api/`
- Frontend: `src/`, `public/`
- Tests: `tests/`
- Config: `config/`
```

### Style Section

Language-specific standards:

```markdown
## Style
- Python: PEP 8, 100 chars
- JS: ES6+, Prettier
- CSS: BEM methodology
```

### Testing Section

Both manual and automated:

```markdown
## Testing
- Manual: `npm start`
- Automated: `npm test`
- Coverage: 90%
```

### Commits Section

Standardize commit messages:

```markdown
## Commits
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, style, refactor, test
```

### PRs Section

Pull request requirements:

```markdown
## PRs
- Title: `<type>(<scope>): <subject>`
- Include: problem, solution, testing
- Check: no console errors, tests pass
```

## Best Practices

### What to Include

- **Do include**: Project-specific workflows, repository structure, build commands, coding standards
- **Do include**: Testing procedures, commit/PR guidelines, security considerations
- **Do include**: Performance notes, deployment instructions, troubleshooting tips

### What to Omit

- **Don't include**: Generic knowledge (agents already know Python, Git, etc.)
- **Don't include**: Explanations (be terse, just bullet points)
- **Don't include**: Vague or non-actionable items
- **Don't include**: Outdated or irrelevant information

### Formatting Tips

1. **Use bullet points**: Never use paragraphs
2. **Be specific**: Include exact commands and file paths
3. **Keep it short**: One line per item
4. **Use consistent structure**: Standard sections make navigation easier
5. **Update regularly**: Keep AGENTS.md in sync with the project

### Content Organization

Group related items together:
- Tools and versions together
- Commands together
- File paths together
- Standards together

## Troubleshooting

### Common Issues

**Issue: AGENTS.md too verbose**
**Cause**: Including explanations or generic knowledge
**Solution**: Remove explanations, focus only on project-specific details

**Issue: AGENTS.md missing critical information**
**Cause**: Not reviewing key repository files
**Solution**: Read README, package.json, and other config files first

**Issue: Inconsistent formatting**
**Cause**: Mixing bullet points with paragraphs
**Solution**: Convert all content to bullet points

**Issue: Outdated information**
**Cause**: Not updating AGENTS.md with project changes
**Solution**: Review and update AGENTS.md during major project changes

### Validation Checklist

Before finalizing AGENTS.md:

- [ ] Only bullet points, no paragraphs
- [ ] Project-specific information only
- [ ] Exact commands and file paths
- [ ] Standard sections used
- [ ] No generic knowledge included
- [ ] Consistent formatting
- [ ] Up-to-date with current project state

## References

### Templates

Use these templates as starting points:

**Python Project Template:**
```markdown
# AGENTS.md

## Dev environment
- Python X.X+
- `pip install -r requirements.txt`
- Run: `python main.py`

## Build
- `python setup.py sdist`

## Code structure
- `main.py`: Entry point
- `src/`: Source code
- `tests/`: Test cases

## Style
- PEP 8
- Type hints
- 100 char limit

## Testing
- `pytest tests/`

## Commits
- `<type>(<scope>): <subject>`
```

**JavaScript Project Template:**
```markdown
# AGENTS.md

## Dev environment
- Node.js X.X+
- `npm install`
- Run: `npm start`

## Build
- `npm run build`

## Code structure
- `src/`: Source
- `public/`: Assets
- `tests/`: Tests

## Style
- ES6+
- Prettier

## Testing
- `npm test`

## Commits
- Conventional commits
```

### Related Skills

- [skills](skills): Skill creation and management
- [documentation](documentation): General documentation best practices
- [templates](templates): Template creation and usage

## Next Steps

1. **Analyze the repository**: Read key files to understand structure
2. **Identify project-specific details**: What makes this project unique?
3. **Create AGENTS.md**: Write concise, bullet-point format
4. **Validate**: Check against the validation checklist
5. **Review**: Ensure all critical information is included
6. **Update**: Keep AGENTS.md current with project changes
