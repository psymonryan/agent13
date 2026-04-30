# Skills Skill Reference Guide

This reference guide provides detailed technical information about creating and managing Agent Skills following the agentskills.io specification.

## Table of Contents

- [Skill Specification](#skill-specification)
- [YAML Frontmatter](#yaml-frontmatter)
- [Directory Structure](#directory-structure)
- [Documentation Standards](#documentation-standards)
- [Validation Rules](#validation-rules)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

## Skill Specification

### Core Requirements

1. **Skill Directory**: Each skill must be in its own directory under `.vibe/skills/`
2. **Skill Name**: Must match the directory name exactly
3. **Required Files**: `SKILL.md` (main documentation)
4. **Optional Files**: `README.md`, any additional documentation

### Skill Naming Rules

- **Length**: 1-64 characters
- **Characters**: Lowercase alphanumeric and hyphens only (`a-z`, `0-9`, `-`)
- **Format**: No leading/trailing hyphens
- **No spaces**: Use hyphens instead of spaces
- **Examples**: `my-skill`, `data-processing`, `api-integration`

### Description Requirements

- **Length**: 1-1024 characters
- **Content**: Must include:
  - **What**: What the skill does
  - **When**: When to use the skill
- **Format**: Can be multi-line using `|` syntax in YAML

## YAML Frontmatter

### Required Fields

```yaml
---
name: "skill-name"  # Must match directory name
version: "1.0.0"  # Semantic versioning

description: |
  What: Description of what the skill does
  When: Description of when to use the skill

author: "Author Name or Organization"
license: "License Type"  # e.g., MIT, Apache-2.0, GPL-3.0
---
```

### Optional Fields

```yaml
keywords: ["tag1", "tag2", "tag3"]  # Array of strings
requirements: "Prerequisites or dependencies"  # String
homepage: "https://example.com"  # URL string
repository: "https://github.com/example/repo"  # URL string
maintainers: ["name1", "name2"]  # Array of strings
contributors: ["name1", "name2"]  # Array of strings
```

### YAML Validation Rules

1. **Valid YAML syntax**: Must parse correctly
2. **Required fields present**: All required fields must be present
3. **Field types correct**: Each field must have the correct type
4. **String length limits**: Respect character limits
5. **No trailing whitespace**: Clean YAML formatting

## Directory Structure

### Standard Structure

```
.vibe/skills/skill-name/
├── SKILL.md          # Required: Main skill documentation
├── README.md         # Optional: Quick reference/usage guide
├── assets/           # Optional: Skill assets
│   ├── templates/    # Optional: Reusable templates
│   │   ├── template1.yml
│   │   └── template2.md
│   ├── images/       # Optional: Images and diagrams
│   └── data/         # Optional: Data files
├── references/       # Optional: Detailed reference material
│   ├── REFERENCE.md
│   └── api-docs.md
└── scripts/          # Optional: Utility scripts
    ├── validate.sh
    └── generate.py
```

### Directory Purpose

- **assets/**: Store any files needed by the skill (templates, images, data)
- **assets/templates/**: Reusable template files for documentation or code
- **references/**: Detailed technical documentation and references
- **scripts/**: Utility scripts for validation, generation, or management

## Documentation Standards

### Progressive Disclosure Pattern

Skills should follow a progressive disclosure pattern:

1. **Quick Start** (README.md): Immediate usage examples
2. **Overview** (SKILL.md): High-level explanation
3. **Reference** (SKILL.md): Detailed technical reference
4. **Examples** (SKILL.md): Complete use cases
5. **Best Practices** (SKILL.md): Recommendations
6. **Troubleshooting** (SKILL.md): Common issues and solutions

### Markdown Formatting

- **Headings**: Use `#` for main sections, `##` for subsections
- **Code Blocks**: Use triple backticks with language specification
- **Lists**: Use consistent indentation
- **Tables**: Use pipe format for complex data
- **Links**: Use descriptive text for links
- **Images**: Use relative paths when possible

### Documentation Sections

Recommended sections in SKILL.md:

1. **Overview**: High-level description
2. **Prerequisites**: Requirements and dependencies
3. **Usage Examples**: Practical examples
4. **Reference**: Detailed technical reference
5. **Best Practices**: Recommendations and guidelines
6. **Troubleshooting**: Common issues and solutions
7. **Advanced Usage**: Complex scenarios
8. **References**: External resources

## Validation Rules

### Skill Name Validation

```python
import re

def validate_skill_name(name):
    # Must match directory name
    if not re.match(r'^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$', name):
        return False
    # Must be 1-64 characters
    if len(name) < 1 or len(name) > 64:
        return False
    return True
```

### Description Validation

```python
def validate_description(description):
    # Must be 1-1024 characters
    if len(description) < 1 or len(description) > 1024:
        return False
    # Must contain "What" and "When"
    if "What:" not in description and "When:" not in description:
        return False
    return True
```

### YAML Frontmatter Validation

```python
import yaml

def validate_yaml_frontmatter(content):
    try:
        # Extract frontmatter
        frontmatter_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return False, "No YAML frontmatter found"
        
        frontmatter = frontmatter_match.group(1)
        data = yaml.safe_load(frontmatter)
        
        # Check required fields
        required_fields = ['name', 'version', 'description', 'author', 'license']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate field types
        if not isinstance(data['name'], str):
            return False, "name must be a string"
        if not isinstance(data['version'], str):
            return False, "version must be a string"
        if not isinstance(data['description'], str):
            return False, "description must be a string"
        if not isinstance(data['author'], str):
            return False, "author must be a string"
        if not isinstance(data['license'], str):
            return False, "license must be a string"
        
        return True, "Valid YAML frontmatter"
        
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {str(e)}"
```

### Complete Skill Validation

```python
def validate_skill(directory):
    """Validate a complete skill directory"""
    
    # Check directory exists
    if not os.path.isdir(directory):
        return False, "Directory does not exist"
    
    # Extract skill name from directory
    skill_name = os.path.basename(directory)
    
    # Validate skill name
    if not validate_skill_name(skill_name):
        return False, f"Invalid skill name: {skill_name}"
    
    # Check SKILL.md exists
    skill_file = os.path.join(directory, "SKILL.md")
    if not os.path.isfile(skill_file):
        return False, "SKILL.md not found"
    
    # Read and validate SKILL.md
    with open(skill_file, 'r') as f:
        content = f.read()
    
    # Validate YAML frontmatter
    is_valid, message = validate_yaml_frontmatter(content)
    if not is_valid:
        return False, message
    
    # Parse frontmatter
    frontmatter_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
    frontmatter = yaml.safe_load(frontmatter_match.group(1))
    
    # Check name matches directory
    if frontmatter['name'] != skill_name:
        return False, f"Skill name in frontmatter ({frontmatter['name']}) does not match directory name ({skill_name})"
    
    # Validate description
    if not validate_description(frontmatter['description']):
        return False, "Invalid description"
    
    return True, "Skill is valid"
```

## Best Practices

### Skill Organization

1. **Consistent Naming**: Use consistent naming conventions across skills
2. **Modular Design**: Keep skills focused on single responsibilities
3. **Document Dependencies**: Clearly document all prerequisites
4. **Version Management**: Use semantic versioning for skill versions
5. **License Clarity**: Clearly specify license and attribution

### Documentation Quality

1. **Clear Structure**: Use consistent section organization
2. **Progressive Disclosure**: Start with overview, drill down to details
3. **Practical Examples**: Provide real-world usage examples
4. **Error Handling**: Document common errors and solutions
5. **Visual Aids**: Use diagrams and code examples where helpful

### Template Usage

1. **Reusable Components**: Create templates for common patterns
2. **Placeholder Documentation**: Clearly document all placeholders
3. **Versioned Templates**: Include version information in templates
4. **Template Examples**: Show how to use each template
5. **Template Validation**: Validate templates before use

### Maintenance

1. **Regular Updates**: Keep skills and documentation current
2. **Deprecation Notice**: Clearly mark deprecated features
3. **Change Log**: Maintain a change log for significant updates
4. **Community Feedback**: Incorporate user feedback and issues
5. **Testing**: Test skills in various scenarios

## Common Patterns

### Pattern 1: Configuration Management

```yaml
---
name: "config-manager"
version: "1.0.0"

description: |
  What: Manages application configuration files
  When: Use when you need to read, write, or validate configuration files

author: "Organization"
license: "MIT"
---

## Configuration Formats

### JSON Configuration

```yaml
skill:
  name: config-manager
  action: read_config
  parameters:
    file: "config.json"
    format: "json"
```

### YAML Configuration

```yaml
skill:
  name: config-manager
  action: read_config
  parameters:
    file: "config.yaml"
    format: "yaml"
```

### Environment Variables

```yaml
skill:
  name: config-manager
  action: export_env
  parameters:
    config: "production"
    output: "env-file"
```
```

### Pattern 2: API Integration

```yaml
---
name: "api-client"
version: "1.0.0"

description: |
  What: Provides client for REST API interactions
  When: Use when you need to call external REST APIs

author: "Organization"
license: "MIT"
---

## API Operations

### GET Request

```yaml
skill:
  name: api-client
  action: get
  parameters:
    url: "https://api.example.com/data"
    headers:
      Authorization: "Bearer {{TOKEN}}"
    query_params:
      limit: 10
      offset: 0
```

### POST Request

```yaml
skill:
  name: api-client
  action: post
  parameters:
    url: "https://api.example.com/data"
    headers:
      Content-Type: "application/json"
      Authorization: "Bearer {{TOKEN}}"
    body:
      name: "John Doe"
      email: "john@example.com"
```

### Authentication

```yaml
skill:
  name: api-client
  action: authenticate
  parameters:
    client_id: "{{CLIENT_ID}}"
    client_secret: "{{CLIENT_SECRET}}"
    scope: "read write"
```
```

### Pattern 3: Data Processing

```yaml
---
name: "data-processor"
version: "1.0.0"

description: |
  What: Processes and transforms data
  When: Use when you need to clean, transform, or analyze data

author: "Organization"
license: "MIT"
---

## Data Transformations

### CSV to JSON

```yaml
skill:
  name: data-processor
  action: transform
  parameters:
    input: "data.csv"
    output: "data.json"
    format:
      from: "csv"
      to: "json"
```

### Data Cleaning

```yaml
skill:
  name: data-processor
  action: clean
  parameters:
    file: "raw-data.csv"
    operations:
      - remove_empty_rows
      - trim_whitespace
      - standardize_dates
```

### Aggregation

```yaml
skill:
  name: data-processor
  action: aggregate
  parameters:
    file: "sales-data.csv"
    group_by: "region"
    metrics:
      - sum: "revenue"
      - avg: "units_sold"
```
```

## Validation Checklist

Use this checklist to validate your skill:

- [ ] Skill name follows naming rules (1-64 chars, lowercase alphanumeric + hyphens)
- [ ] Skill name matches directory name exactly
- [ ] SKILL.md exists in the skill directory
- [ ] YAML frontmatter is present and valid
- [ ] All required frontmatter fields are present
- [ ] Description includes "What" and "When"
- [ ] Description is 1-1024 characters
- [ ] Author and license are specified
- [ ] Documentation follows progressive disclosure pattern
- [ ] Usage examples are provided
- [ ] Best practices are documented
- [ ] Troubleshooting section is included
- [ ] Optional directories (assets, references, scripts) are organized properly
- [ ] Templates (if any) are documented and validated
- [ ] All placeholders in templates are documented

## Tools and Utilities

### Validation Script

```bash
#!/bin/bash

# Validate a skill directory
validate_skill() {
    local dir="$1"
    local skill_name=$(basename "$dir")
    
    echo "Validating skill: $skill_name"
    
    # Check directory exists
    if [ ! -d "$dir" ]; then
        echo "ERROR: Directory does not exist"
        return 1
    fi
    
    # Check SKILL.md exists
    if [ ! -f "$dir/SKILL.md" ]; then
        echo "ERROR: SKILL.md not found"
        return 1
    fi
    
    # Validate YAML frontmatter (using python)
    python3 -c "
import yaml
import re

with open('$dir/SKILL.md', 'r') as f:
    content = f.read()

# Extract frontmatter
match = re.search(r'---\\n(.*?)\\n---', content, re.DOTALL)
if not match:
    print('ERROR: No YAML frontmatter found')
    exit(1)

try:
    data = yaml.safe_load(match.group(1))
    required = ['name', 'version', 'description', 'author', 'license']
    for field in required:
        if field not in data:
            print(f'ERROR: Missing required field: {field}')
            exit(1)
    
    if data['name'] != '$skill_name':
        print(f'ERROR: Skill name mismatch: {data[\"name\"]} != $skill_name')
        exit(1)
    
    print('SUCCESS: Skill is valid')
except yaml.YAMLError as e:
    print(f'ERROR: YAML syntax error: {e}')
    exit(1)
"
}

# Usage: validate_skill ".vibe/skills/my-skill"
```

### Skill Generator

```python
#!/usr/bin/env python3

import os
import yaml
from datetime import datetime

def generate_skill(skill_name, author, description, license_type="MIT"):
    """Generate a new skill directory structure"""
    
    # Create directory
    skill_dir = f".vibe/skills/{skill_name}"
    os.makedirs(skill_dir, exist_ok=True)
    
    # Create SKILL.md
    skill_content = f"""---
name: "{skill_name}"
version: "1.0.0"

description: |
  {description}

author: "{author}"
license: "{license_type}"

# Optional fields
keywords: ["{skill_name.replace('-', ' ')}"]
requirements: "None"

# End of YAML frontmatter ---
---

# {skill_name.replace('-', ' ').title()} Skill

## Overview

This skill provides functionality for {description.split('What:')[1].split('When:')[0].strip()}.

## Prerequisites

List any prerequisites or dependencies here.

## Usage Examples

### Basic Usage

```yaml
skill:
  name: {skill_name}
  action: example_action
  parameters:
    param1: value1
```

## Reference

### Actions

- `example_action`: Description of what this action does

### Parameters

- `param1`: Description of parameter 1

## Best Practices

Include best practices for using this skill.

## Troubleshooting

Common issues and solutions.
"""
    
    with open(f"{skill_dir}/SKILL.md", "w") as f:
        f.write(skill_content)
    
    # Create README.md
    readme_content = f"""# {skill_name.replace('-', ' ').title()}

Quick reference for the {skill_name} skill.

## Quick Start

```yaml
skill:
  name: {skill_name}
  action: example_action
  parameters:
    param1: value1
```

## See Also

- [Full Documentation](SKILL.md)
"""
    
    with open(f"{skill_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    # Create assets directory
    os.makedirs(f"{skill_dir}/assets/templates", exist_ok=True)
    os.makedirs(f"{skill_dir}/references", exist_ok=True)
    os.makedirs(f"{skill_dir}/scripts", exist_ok=True)
    
    print(f"Generated skill: {skill_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a new skill")
    parser.add_argument("--name", required=True, help="Skill name")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--description", required=True, help="Skill description")
    parser.add_argument("--license", default="MIT", help="License type")
    
    args = parser.parse_args()
    
    generate_skill(args.name, args.author, args.description, args.license)
```

## License

This reference guide is licensed under the MIT License.

Copyright (c) 2024 Agent13

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
