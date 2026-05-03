# Agent Skills Creation Skill

Quickly create, validate, and manage Agent Skills following the [agentskills.io](https://agentskills.io/) specification.

## Features

- **Comprehensive guidance**: Step-by-step instructions for skill creation
- **Templates**: Ready-to-use skill templates
- **Validation**: Checklists and tools for skill validation
- **Best practices**: Proven patterns for skill development
- **Examples**: Real-world skill examples and patterns

## Quick Start

### Create a New Skill

```bash
# Create skill directory
mkdir -p .vibe/skills/my-skill

# Copy template
cp .vibe/skills/skills/assets/templates/basic-skill.yml .vibe/skills/my-new-skill/SKILL.md

# Edit the template
vim .vibe/skills/my-skill/SKILL.md
```

### Validate Your Skill

```bash
# Check YAML frontmatter
python3 -c "import yaml; yaml.safe_load(open('SKILL.md'))"

# Validate skill name matches directory
test "${PWD##*/}" = "$(grep '^name:' SKILL.md | cut -d' ' -f2)"
```

### Use the Skill

```bash
# Ask the agent to:
# - Create a new skill
# - Improve an existing skill
# - Validate skill structure
# - Find best practices
# - Understand skill format
```

## Skill Structure

### Basic (Required)

```
my-skill/
└── SKILL.md
```

### Recommended

```
my-skill/
├── SKILL.md          # Main instructions
├── README.md         # Quick overview
├── assets/           # Templates and resources
│   └── templates/    # Configuration templates
├── references/       # Detailed documentation
│   └── REFERENCE.md # Technical reference
└── scripts/          # Utility scripts
    └── validate.sh   # Validation tool
```

## Frontmatter Template

```yaml
---
name: my-skill
description: What this skill does and when to use it. Include keywords for task identification.
license: MIT
compatibility: Requires tool1, tool2, etc.
metadata:
  version: 1.0.0
  author: Your Name
  created: YYYY-MM-DD
  updated: YYYY-MM-DD
  category: development
  tags:
    - tag1
    - tag2
---
```

## Common Patterns

### 1. Skill with Templates

```
my-skill/
├── SKILL.md
├── assets/
│   └── templates/
│       ├── config.yml
│       └── script.sh
└── references/
    └── best-practices.md
```

### 2. Skill with Validation

```
my-skill/
├── SKILL.md
├── scripts/
│   ├── validate.sh
│   └── generate.py
└── references/
    └── troubleshooting.md
```

### 3. Comprehensive Skill

```
my-skill/
├── SKILL.md
├── README.md
├── assets/
│   ├── templates/
│   └── images/
├── references/
│   ├── REFERENCE.md
│   └── api.md
└── scripts/
    ├── validate.sh
    └── deploy.sh
```

## Validation Checklist

✅ **Required Elements**

- [ ] SKILL.md exists
- [ ] Valid YAML frontmatter
- [ ] `name` field (1-64 chars, lowercase, no leading/trailing hyphens)
- [ ] `description` field (1-1024 chars, includes what and when)
- [ ] Skill name matches directory name

✅ **Optional Elements (Recommended)**

- [ ] `license` field
- [ ] `compatibility` field
- [ ] `metadata` (version, author, etc.)
- [ ] `assets/` directory
- [ ] `references/` directory
- [ ] `scripts/` directory

✅ **Content Quality**

- [ ] Clear instructions
- [ ] Practical examples
- [ ] Common patterns
- [ ] Best practices
- [ ] Troubleshooting
- [ ] References

## Tools Used

- `read_file`: Read skill files and templates
- `write_file`: Create new skill files
- `search_replace`: Modify existing content
- `bash`: Execute validation scripts
- `grep`: Search for patterns in skill content

## Examples

### Minimal Valid Skill

```yaml
---
name: hello-world
description: A simple skill demonstrating the basic structure. Use when learning skill creation.
---

# Hello World

## Usage

Say "Hello, World!"

## Example
```

Hello, World!

```

```

### Skill with Templates

See `assets/templates/basic-skill-with-templates.yml` for a complete example.

## References

- [Agent Skills Specification](https://agentskills.io/specification)
- [Example Skills](https://github.com/anthropics/skills)
- [skills-ref Validation](https://github.com/agentskills/agentskills/tree/main/skills-ref)

## Best Practices

1. **Start small**: Begin with minimal structure, expand as needed
2. **Focus on one thing**: Each skill should address a specific task
3. **Include examples**: Always show inputs and expected outputs
4. **Document edge cases**: Include error handling and special scenarios
5. **Use templates**: Create reusable configurations in `assets/templates/`
6. **Keep updated**: Regularly review and improve your skills
7. **Get feedback**: Test with others and incorporate suggestions

## Troubleshooting

### YAML Validation Error

**Cause**: Invalid YAML syntax in frontmatter
**Solution**: Check indentation and quotes

### Skill Not Found

**Cause**: Skill not in `.agent13/skills/` directory
**Solution**: Move skill to correct location

### Description Too Long

**Cause**: Description exceeds 1024 characters
**Solution**: Make more concise or split into multiple sentences

## Next Steps

1. **Create your first skill**: Use templates in `assets/templates/`
2. **Validate**: Check all required elements
3. **Test**: Use with your agent
4. **Improve**: Incorporate feedback
5. **Share**: Contribute to the community

For detailed guidance, see `SKILL.md`.