---
name: get-new-skill
version: "1.0.0"

description: |
  Find, download, and adapt new Agent Skills for agent13.
  
  What: Workflow for searching for skills online, downloading them, and modifying them to fit agent13's conventions and needs.
  When: Use when you need a new skill that doesn't exist yet, or when adapting external skills for agent13.

author: Agent13 manage-skills skill
license: "MIT"

keywords: ["skill-acquisition", "skill-adaptation", "workflow"]
requirements: "None"
---

# Get New Skill: Find and Adapt Skills

## Overview

This skill provides a systematic workflow for finding, downloading, and adapting Agent Skills to work with agent13. It covers searching online, using templates, and modifying skills to match agent13's conventions.

## Workflow

### Step 1: Understand the Need

Before searching, clarify what you need:

```python
# Ask user for clarification if needed
ask_user_question(
    questions=[{
        "question": "What type of skill do you need?",
        "header": "Skill Type",
        "options": [
            {"label": "Code manipulation", "description": "Refactoring, analysis, generation"},
            {"label": "Documentation", "description": "Creating docs, READMEs, guides"},
            {"label": "System operations", "description": "Docker, deployment, configuration"},
            {"label": "Data processing", "description": "Parsing, transformation, analysis"}
        ]
    }]
)
```

### Step 2: Search for Existing Skills

**Option A: Search skills.sh (Quick)**
```python
# Quick search for skills by name
fetch_server_fetch(
    url="https://skills.sh/?q=[skill-name-to-search]"
)
```

**Option B: Search skillsdirectory.com (Detailed)**
```python
# Search with detailed query terms
fetch_server_fetch(
    url="https://www.skillsdirectory.com/skills?focus=search&q=[search+terms]"
)
```

**Option C: Search agentskills.io ecosystem**
```python
# Use web search to find relevant skills
searxng_search_metasearch_web(
    query="agentskills.io [skill topic]",
    language="en"
)
```

**Option D: Check agent13's existing skills**
```python
# List current skills
bash("ls -1 ~/.agent13/skills/")

# Search for similar patterns in existing skills
grep(pattern="[keyword]", path="~/.agent13/skills/")
```

**Option E: Use templates**
```python
# Browse available templates
bash("ls -1 ~/.agent13/skills/skills/assets/templates/")
```

### Step 3: Download or Create

**If found online:**
```python
# Fetch the skill content
searxng_search_fetch_web_content(
    url="[skill-url]",
    format="markdown"
)

# Create skill directory
bash("mkdir -p ~/.agent13/skills/[skill-name]")

# Write the skill file
write_file(
    path="~/.agent13/skills/[skill-name]/SKILL.md",
    content="[downloaded-content]"
)
```

**If using template:**
```python
# Read template
read_file(path="~/.agent13/skills/skills/assets/templates/[template-name].yml")

# Create skill directory
bash("mkdir -p ~/.agent13/skills/[skill-name]")

# Copy and customize template
write_file(
    path="~/.agent13/skills/[skill-name]/SKILL.md",
    content="[customized-content]"
)
```

**If creating from scratch:**
```python
# Create skill directory
bash("mkdir -p ~/.agent13/skills/[skill-name]")

# Start with minimal structure
write_file(
    path="~/.agent13/skills/[skill-name]/SKILL.md",
    content="""---
name: [skill-name]
version: "1.0.0"
description: |
  Brief description of what this skill does.
  
  What: [what it does]
  When: [when to use it]

author: Agent13 manage-skills skill
license: "MIT"
keywords: ["keyword1", "keyword2"]
requirements: "None"
---

# [Skill Name]

## Overview

[Brief overview]

## Usage

[Usage instructions]

## Examples

[Examples]
"""
)
```

### Step 4: Adapt for agent13

**Key adaptations to make:**

1. **Focus on agent13 tools** - Ensure examples use agent13's actual tools:
   - `task`, `grep`, `read_file`, `write_file`, `search_replace`
   - `bash`, `ask_user_question`, `todo`
   - Avoid generic tool references

2. **Make it practical** - Remove meta-content about "how to use AI"
   - Focus on concrete tasks and workflows
   - Provide actionable patterns
   - Include real examples

3. **Keep it concise** - Trim unnecessary content:
   - Remove verbose explanations
   - Focus on essential information
   - Use tables and lists for quick reference

4. **agent13-specific conventions**:
   - Use `license: MIT` for open skills
   - Include `author: "Agent13"`
   - Follow agent13's file paths (`~/.agent13/skills/`)

**Example adaptation:**
```python
# Read the downloaded skill
read_file(path="~/.agent13/skills/[skill-name]/SKILL.md")

# Modify to be agent13-appropriate
search_replace(
    file_path="~/.agent13/skills/[skill-name]/SKILL.md",
    content="[generic tool reference]",
    new_text="[agent13-specific tool]"
)

# Trim down if too long
# Remove sections, simplify examples, etc.
```

### Step 5: Validate

**Basic validation:**
```python
# Check skill name matches directory
bash("test '[skill-name]' = \"$(grep '^name:' ~/.agent13/skills/[skill-name]/SKILL.md | cut -d' ' -f2)\" && echo '✓ Name matches'")

# Verify YAML frontmatter
bash("head -20 ~/.agent13/skills/[skill-name]/SKILL.md")

# Check skill is visible
bash("ls -1 ~/.agent13/skills/")
```

**Content validation:**
```python
# Read and review
read_file(path="~/.agent13/skills/[skill-name]/SKILL.md")

# Check for:
# - Clear description with "what" and "when"
# - Practical examples using agent13 tools
# - Appropriate length (not too verbose)
# - agent13-specific conventions
```

### Step 6: Test (Optional)

If the skill is complex, test it:
```python
# Try using the skill on a simple task
# Verify it provides helpful guidance
# Check if examples work as expected
```

## Common Patterns

### Pattern 1: Adapt External Skill

```python
# 1. Search and find
fetch_server_fetch(url="https://skills.sh/?q=[skill-name]")
# or
fetch_server_fetch(url="https://www.skillsdirectory.com/skills?focus=search&q=[terms]")
# or
searxng_search_metasearch_web(query="agentskills.io [topic]")

# 2. Download
searxng_search_fetch_web_content(url="[url]", format="markdown")

# 3. Create
bash("mkdir -p ~/.agent13/skills/[name]")
write_file(path="~/.agent13/skills/[name]/SKILL.md", content="[content]")

# 4. Adapt
read_file(path="~/.agent13/skills/[name]/SKILL.md")
# Modify for agent13 tools, trim content, etc.

# 5. Validate
bash("test '[name]' = \"$(grep '^name:' ~/.agent13/skills/[name]/SKILL.md | cut -d' ' -f2)\"")
```

### Pattern 2: Create from Template

```python
# 1. Browse templates
bash("ls -1 ~/.agent13/skills/skills/assets/templates/")

# 2. Read template
read_file(path="~/.agent13/skills/skills/assets/templates/basic-skill.yml")

# 3. Create and customize
bash("mkdir -p ~/.agent13/skills/[name]")
write_file(path="~/.agent13/skills/[name]/SKILL.md", content="[customized]")

# 4. Validate
bash("ls -1 ~/.agent13/skills/")
```

### Pattern 3: Create from Scratch

```python
# 1. Clarify need
ask_user_question(questions=[...])

# 2. Create structure
bash("mkdir -p ~/.agent13/skills/[name]")

# 3. Write content
write_file(path="~/.agent13/skills/[name]/SKILL.md", content="[full-skill]")

# 4. Validate
bash("test '[name]' = \"$(grep '^name:' ~/.agent13/skills/[name]/SKILL.md | cut -d' ' -f2)\"")
```

## agent13-Specific Guidelines

### Tool References

Always use agent13's actual tools:
- `task` - Delegate to subagents
- `grep` - Search for patterns
- `read_file` - Read file contents
- `write_file` - Create files
- `search_replace` - Modify files
- `bash` - Execute commands
- `ask_user_question` - Get user input
- `todo` - Track tasks

### Content Style

- **Be practical**: Focus on actionable workflows
- **Be concise**: Remove unnecessary verbosity
- **Be specific**: Use concrete examples
- **Be agent13-focused**: Reference agent13 tools and conventions

### Common Adaptations

When adapting external skills, typically:
1. Replace generic tool names with agent13 tools
2. Remove "how to prompt AI" meta-content
3. Trim down examples to be more concise
4. Add agent13-specific paths and conventions
5. Simplify overly complex explanations

## Quick Reference

| Step | Action | Tools |
|------|--------|-------|
| 1. Understand need | Clarify requirements | ask_user_question |
| 2. Search | Find existing skills | fetch_server_fetch (skills.sh, skillsdirectory.com), searxng_search_metasearch_web, grep |
| 3. Download/Create | Get or create skill | searxng_search_fetch_web_content, write_file |
| 4. Adapt | Modify for agent13 | read_file, search_replace |
| 5. Validate | Verify correctness | bash, read_file |
| 6. Test | Try it out | (manual) |

## When to Use This Skill

Use get-new-skill when:
- User requests a skill that doesn't exist
- You find a skill online that looks useful
- You need to create a skill from scratch
- You want to adapt an external skill for agent13
- User asks "can you get me a skill for X?"

## Example: Creating supercoder

This skill was created using this workflow:
1. User asked for "using-superpowers" skill
2. Couldn't find existing one
3. Created from scratch using template
4. Adapted to focus on agent13 tool orchestration
5. Trimmed down to be concise
6. Renamed from "powerup" to "supercoder"
7. Validated and tested
