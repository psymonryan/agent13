---
name: manage-skills
description: Create, validate, and manage Agent Skills following the agentskills.io specification. Use when creating new skills, improving existing ones, or understanding skill structure and best practices.
license: Commercial
compatibility: Requires basic Markdown knowledge and familiarity with file system operations. Works with any agent supporting the agentskills.io format.
metadata:
  version: 1.0.0
  author: Agent13
  created: 2024-07-15
  updated: 2024-07-15
  category: development
  tags:
    - skills
    - development
    - best-practices
    - documentation
    - templates
---

# Agent Skills Creation and Management

This skill provides comprehensive guidance for creating, validating, and managing Agent Skills that follow the [agentskills.io](https://agentskills.io/) specification.

## Getting Started

### What are Agent Skills?

Agent Skills are folders containing instructions, scripts, and resources that agents can use to extend their capabilities. They provide:

- **Procedural knowledge**: Step-by-step instructions for specific tasks
- **Domain expertise**: Specialized knowledge for specific domains
- **Reusable workflows**: Consistent, repeatable processes
- **Context**: Organizational-specific information and best practices

### When to Use This Skill

Use this skill when:
- Creating a new Agent Skill from scratch
- Improving or updating an existing skill
- Understanding the agentskills.io specification
- Validating skill structure and compliance
- Looking for templates and examples
- Need best practices for skill development

## Script Design Philosophy

Keep scripts simple. Let the AI do the thinking.

- **Scripts do**: Data extraction, format normalization, deterministic processing, tedious repetitive tasks
- **AI does**: Classification, correlation, judgment, interpretation, templating, final formatting
- **The contract**: Scripts return clean, structured data; AI processes that data using SKILL.md templates

## Skill Structure

### Basic Structure

Every skill must have at least a `SKILL.md` file:

```
my-skill/
└── SKILL.md          # Required
```

### Recommended Structure

For comprehensive skills, use this structure:

```
my-skill/
├── SKILL.md          # Required - Main instructions
├── README.md         # Optional - Quick overview
├── assets/           # Optional - Templates and resources
│   ├── templates/    # Configuration templates
│   └── images/       # Diagrams and screenshots
├── references/       # Optional - Detailed documentation
│   ├── REFERENCE.md # Technical reference
│   └── best-practices.md
└── scripts/          # Optional - Utility scripts
    ├── validate.sh   # Validation script
    └── generate.py   # Generation tool
```

## SKILL.md Format

### YAML Frontmatter (Required)

The `SKILL.md` file must start with YAML frontmatter:

```yaml
---
name: skill-name
description: What this skill does and when to use it.
---
```

### Required Frontmatter Fields

| Field | Requirements | Example |
|-------|-------------|---------|
| `name` | 1-64 chars, lowercase alphanumeric + hyphens (no underscores), no leading/trailing hyphens | `name: docker-swarm` |
| `description` | 1-1024 chars, describes what and when to use | `description: Manage Docker Swarm services with best practices.` |

### Optional Frontmatter Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `license` | License information | `license: MIT` |
| `compatibility` | Environment requirements | `compatibility: Requires Docker Swarm, bash, jq` |
| `metadata` | Additional metadata | `metadata:\n  version: 1.0.0\n  author: Your Name` |

### Frontmatter Examples

**Minimal valid frontmatter:**
```yaml
---
name: example
description: An example skill demonstrating the basic structure.
---
```

**Comprehensive frontmatter:**
```yaml
---
name: data-analysis
description: Perform statistical analysis and data visualization. Use when analyzing datasets, creating visualizations, or generating reports from data.
license: Commercial
compatibility: Requires Python 3.8+, pandas, matplotlib, numpy
metadata:
  version: 2.1.0
  author: Data Team
  created: 2024-01-10
  updated: 2024-06-15
  category: data-science
  tags:
    - data
    - analysis
    - visualization
    - pandas
    - python
---
```

## Skill Content Best Practices

### Progressive Disclosure

Structure your skill for efficient context usage:

1. **Metadata** (~100 tokens): Loaded at startup for all skills
2. **Instructions** (< 5000 tokens): Loaded when skill is activated
3. **Resources** (as needed): Loaded only when referenced

### Recommended Sections

Organize your `SKILL.md` with these sections:

1. **Getting Started**: Quick overview and prerequisites
2. **Core Concepts**: Key ideas and terminology
3. **Step-by-Step Instructions**: Detailed workflow
4. **Examples**: Input/output examples
5. **Common Patterns**: Reusable solutions
6. **Best Practices**: Recommendations and pitfalls
7. **Troubleshooting**: Error handling and debugging
8. **References**: Additional resources

### Writing Style Tips

- **Be specific**: Include keywords that help agents identify when to use the skill
- **Provide examples**: Show both inputs and expected outputs
- **Document edge cases**: Include error handling and special scenarios
- **Use clear headings**: Help agents navigate to relevant sections
- **Keep it concise**: Focus on essential information
- **Use relative paths**: Reference other files in your skill

## Creating a New Skill

### Step 1: Plan Your Skill

1. **Define the purpose**: What specific task or domain does this skill address?
2. **Identify the audience**: Who will use this skill?
3. **Determine scope**: What's included vs. what's out of scope?
4. **List prerequisites**: What tools, knowledge, or access is required?

### Step 2: Create the Directory

```bash
mkdir -p .vibe/skills/my-new-skill
cd .vibe/skills/my-new-skill
```

### Step 3: Create SKILL.md

Start with a basic template:

```bash
echo "---\nname: my-new-skill\ndescription: Brief description of what this skill does and when to use it.\n---\n\n# My New Skill\n\n## Getting Started\n\n## Usage\n\n## Examples\n" > SKILL.md
```

### Step 4: Add Content

Fill in the sections with:
- Clear instructions
- Practical examples
- Common patterns
- Best practices
- Troubleshooting tips

### Step 5: Add Optional Components

- **Templates**: Add reusable configuration files to `assets/templates/`
- **References**: Add detailed documentation to `references/`
- **Scripts**: Add utility tools to `scripts/`
- **README**: Add quick overview and usage examples

### Step 6: Validate Your Skill

Run the validation script:

```bash
cd .vibe/skills/skills
python3 scripts/validate-skill.py ../my-new-skill
```

Fix any errors before proceeding.

## Skill Naming Conventions

### Valid Names

- `docker-swarm`
- `data-analysis`
- `code-review`
- `pdf-processing`

### Invalid Names

- `DockerSwarm` (uppercase not allowed)
- `get_new_skill` (underscores not allowed, use hyphens)
- `-docker` (leading hyphen)
- `docker--swarm` (consecutive hyphens)
- `docker swarm` (spaces not allowed)

### Naming Tips

- Use lowercase letters only
- Separate words with hyphens
- Keep it descriptive but concise
- Match the directory name exactly
- Use consistent naming across your organization

## Directory Structure Best Practices

### assets/ Directory

Store templates, configuration files, and static resources:

```
assets/
├── templates/
│   ├── web-service.yml
│   ├── database-config.json
│   └── docker-compose.example.yml
└── images/
    ├── architecture-diagram.png
    └── workflow.png
```

### references/ Directory

Store detailed documentation and reference material:

```
references/
├── REFERENCE.md
├── api-specification.md
├── security-considerations.md
└── troubleshooting.md
```

### scripts/ Directory

Store utility scripts and tools:

```
scripts/
├── validate.sh
├── generate-config.py
├── deploy.sh
└── backup.sh
```

## Content Organization

### Progressive Disclosure Pattern

```markdown
# Skill Name

## Getting Started
- Quick overview
- Prerequisites
- Basic usage

## Core Concepts
- Key ideas
- Terminology
- Architecture overview

## Step-by-Step Guide
1. Step one
2. Step two
3. Step three

## Examples
### Example 1: Basic Usage
```bash
# Input
# Output
```

### Example 2: Advanced Configuration
```yaml
# Configuration
```

## Best Practices
- Do this
- Avoid that
- Consider these tradeoffs

## Troubleshooting
### Common Error: X
**Cause:** Y
**Solution:** Z

### Common Error: A
**Cause:** B
**Solution:** C

## References
- [Official Documentation](url)
- [Related Skills](url)
- [Community Resources](url)
```

## Validation Checklist

Run the validation script first:

```bash
cd .vibe/skills/skills
python3 scripts/validate-skill.py ../your-skill-name
```

### Required Elements

- [ ] SKILL.md file exists
- [ ] Valid YAML frontmatter
- [ ] `name` field present and valid
- [ ] `description` field present and descriptive
- [ ] Skill name matches directory name
- [ ] Description includes what and when to use

### Optional Elements (Recommended)

- [ ] `license` field
- [ ] `compatibility` field
- [ ] `metadata` with version, author, etc.
- [ ] `assets/` directory with templates
- [ ] `references/` directory with documentation
- [ ] `scripts/` directory with utilities
- [ ] README.md with quick overview

### Content Quality

- [ ] Clear, concise instructions
- [ ] Practical examples with inputs/outputs
- [ ] Common patterns documented
- [ ] Best practices included
- [ ] Troubleshooting guide
- [ ] References to additional resources

## Common Mistakes to Avoid

### Frontmatter Errors

1. **Missing required fields**: Always include `name` and `description`
2. **Invalid skill name**: Check naming conventions
3. **Poor description**: Include both what and when to use
4. **Incorrect indentation**: YAML is space-sensitive

### Content Issues

1. **Too much detail**: Keep main SKILL.md focused, move details to references/
2. **No examples**: Always include practical examples
3. **Vague instructions**: Be specific about steps and expected results
4. **Missing edge cases**: Document error handling and special scenarios

### Structural Problems

1. **Deep nesting**: Keep file references one level deep
2. **Large files**: Split content into multiple files when needed
3. **No organization**: Use clear section headings
4. **Missing references**: Link to related resources

## Tools for Skill Development

### File Operations

- `read_file`: Read existing skill files
- `write_file`: Create new skill files
- `search_replace`: Modify existing content
- `bash`: Execute validation scripts

### Validation

Run the validation script:

```bash
cd .vibe/skills/skills
python3 scripts/validate-skill.py ../your-skill-name
```

### License Preferences

**Default License: Commercial**

Unless explicitly specified as open source, all skills should use `license: Commercial`.

**When to use different licenses:**
- `Commercial`: Default for internal/proprietary skills
- `MIT`, `Apache-2.0`, etc.: Only for explicitly open-sourced skills
- `Proprietary`: For highly sensitive internal-only skills

**License field examples:**
```yaml
# Default (recommended)
license: Commercial

# Open source alternatives (use only when approved)
license: MIT
license: Apache-2.0
license: GPL-3.0

# For highly sensitive internal skills
license: Proprietary
```

## Examples from Existing Skills

### Example 1: Minimal Valid Skill

```yaml
---
name: hello-world
description: A simple skill that demonstrates the basic structure. Use when learning how to create skills.
---

# Hello World Skill

## Getting Started

This is a minimal example skill.

## Usage

Simply say "Hello, World!"

## Examples

```
Hello, World!
```
```

## Best Practices for Skill Authors

### 1. Start Small

Begin with a minimal skill and expand as needed. It's easier to add content than to remove it.

### 2. Focus on One Thing

Each skill should address a specific task or domain. Avoid creating "kitchen sink" skills.

### 3. Include Examples

Always provide practical examples showing inputs and expected outputs.

### 4. Document Edge Cases

Include information about error handling, special scenarios, and limitations.

### 5. Use Templates

Create reusable templates in the `assets/templates/` directory for common configurations.

### 6. Keep It Updated

Regularly review and update your skills as best practices evolve.

### 7. Get Feedback

Test your skills with others and incorporate their feedback.

### 8. Document Dependencies

Clearly state any prerequisites or requirements in the `compatibility` field.

## References

### Official Documentation

- [Agent Skills Specification](https://agentskills.io/specification)
- [Agent Skills GitHub](https://github.com/agentskills/agentskills)
- [Example Skills](https://github.com/anthropics/skills)

### Tools and Utilities

- [skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref): Reference library for validation
- [YAML Lint](https://yamllint.com/): Validate YAML frontmatter
- [Markdown Lint](https://github.com/DavidAnson/markdownlint): Validate Markdown formatting

### Related Skills

- [documentation](https://github.com/anthropics/skills/tree/main/skills/documentation): Documentation best practices
- [templates](https://github.com/anthropics/skills/tree/main/skills/templates): Template creation and management
- [validation](https://github.com/anthropics/skills/tree/main/skills/validation): Validation techniques

## Troubleshooting

### Skill Not Recognized

**Cause**: Skill name doesn't match directory name
**Solution**: Ensure the `name` field matches the directory name exactly

### Validation Errors

**Cause**: Invalid YAML frontmatter
**Solution**: Check YAML indentation and syntax

### Description Too Long

**Cause**: Description exceeds 1024 characters
**Solution**: Make the description more concise or split into multiple sentences

### Skill Not Found

**Cause**: Skill not in expected location
**Solution**: Place skills in `.vibe/skills/` directory

For more information, see the [Agent Skills Specification](https://agentskills.io/specification).
