#!/usr/bin/env uv run

# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "pyyaml==6.0.1",
# ]
# ///

"""
Skill Validation Script
Validates a skill directory against the agentskills.io specification.
"""

import os
import re
import sys
import yaml
from typing import Tuple, Dict, Any


def validate_skill_name(name: str) -> Tuple[bool, str]:
    """Validate skill name according to specification."""

    # Must be 1-64 characters
    if len(name) < 1:
        return False, "Skill name must be at least 1 character"
    if len(name) > 64:
        return False, "Skill name must be at most 64 characters"

    # Must match pattern: lowercase alphanumeric + hyphens
    if not re.match(r'^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$', name):
        return False, "Skill name must contain only lowercase alphanumeric characters and hyphens"

    # No leading or trailing hyphens (covered by regex)
    if name.startswith('-') or name.endswith('-'):
        return False, "Skill name cannot start or end with a hyphen"

    return True, "Valid skill name"


def validate_description(description: str) -> Tuple[bool, str]:
    """Validate skill description."""

    # Must be 1-1024 characters
    if len(description) < 1:
        return False, "Description must be at least 1 character"
    if len(description) > 1024:
        return False, "Description must be at most 1024 characters"

    # Must contain "What" and "When"
    if "What:" not in description and "When:" not in description:
        return False, "Description must include 'What:' and 'When:' sections"

    return True, "Valid description"


def validate_yaml_frontmatter(content: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate YAML frontmatter in skill documentation."""

    try:
        # Extract frontmatter
        frontmatter_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return False, "No YAML frontmatter found", {}

        frontmatter_str = frontmatter_match.group(1)
        data = yaml.safe_load(frontmatter_str)

        if not isinstance(data, dict):
            return False, "YAML frontmatter must be a dictionary", {}

        # Check required fields (per agentskills.io spec: name, description are required)
        required_fields = ['name', 'description']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}", {}

        # Validate field types for required and optional fields
        type_errors = []
        if not isinstance(data['name'], str):
            type_errors.append("name must be a string")
        if not isinstance(data['description'], str):
            type_errors.append("description must be a string")

        # Optional fields (if present, must be correct type)
        if 'license' in data and not isinstance(data['license'], str):
            type_errors.append("license must be a string")
        if 'compatibility' in data and not isinstance(data['compatibility'], str):
            type_errors.append("compatibility must be a string")
        if 'metadata' in data and not isinstance(data['metadata'], dict):
            type_errors.append("metadata must be a dictionary")

        if type_errors:
            return False, "; ".join(type_errors), {}

        return True, "Valid YAML frontmatter", data

    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {str(e)}", {}


def validate_skill_directory(directory: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate a complete skill directory."""

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'skill_name': '',
        'frontmatter': {},
        'checks': {}
    }

    # Check directory exists
    if not os.path.isdir(directory):
        results['valid'] = False
        results['errors'].append(f"Directory does not exist: {directory}")
        return results, results

    # Extract skill name from directory
    skill_name = os.path.basename(os.path.normpath(directory))
    results['skill_name'] = skill_name

    # Validate skill name
    is_valid, message = validate_skill_name(skill_name)
    results['checks']['skill_name'] = {'valid': is_valid, 'message': message}
    if not is_valid:
        results['valid'] = False
        results['errors'].append(message)

    # Check SKILL.md exists
    skill_file = os.path.join(directory, "SKILL.md")
    if not os.path.isfile(skill_file):
        results['valid'] = False
        results['errors'].append("SKILL.md not found")
        results['checks']['skill_file_exists'] = {'valid': False, 'message': "SKILL.md not found"}
    else:
        results['checks']['skill_file_exists'] = {'valid': True, 'message': "SKILL.md exists"}

    # Read and validate SKILL.md if it exists
    if os.path.isfile(skill_file):
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Validate YAML frontmatter
            is_valid, message, frontmatter = validate_yaml_frontmatter(content)
            results['checks']['yaml_frontmatter'] = {'valid': is_valid, 'message': message}
            results['frontmatter'] = frontmatter

            if not is_valid:
                results['valid'] = False
                results['errors'].append(message)
            else:
                # Check name matches directory
                if frontmatter['name'] != skill_name:
                    error_msg = f"Skill name in frontmatter ({frontmatter['name']}) does not match directory name ({skill_name})"
                    results['valid'] = False
                    results['errors'].append(error_msg)
                    results['checks']['name_match'] = {'valid': False, 'message': error_msg}
                else:
                    results['checks']['name_match'] = {'valid': True, 'message': "Skill name matches directory"}

                # Validate description
                is_valid, message = validate_description(frontmatter['description'])
                results['checks']['description'] = {'valid': is_valid, 'message': message}
                if not is_valid:
                    results['valid'] = False
                    results['errors'].append(message)

        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Error reading SKILL.md: {str(e)}")
            results['checks']['read_error'] = {'valid': False, 'message': f"Error reading file: {str(e)}"}

    # Check for README.md (optional but recommended)
    readme_file = os.path.join(directory, "README.md")
    if os.path.isfile(readme_file):
        results['checks']['readme_exists'] = {'valid': True, 'message': "README.md exists (recommended)"}
    else:
        results['warnings'].append("README.md not found (recommended)")
        results['checks']['readme_exists'] = {'valid': False, 'message': "README.md not found (recommended)"}

    # Check for optional directories
    optional_dirs = ['assets', 'references', 'scripts']
    for dir_name in optional_dirs:
        dir_path = os.path.join(directory, dir_name)
        if os.path.isdir(dir_path):
            results['checks'][f'{dir_name}_exists'] = {'valid': True, 'message': f"{dir_name}/ exists"}
        else:
            results['checks'][f'{dir_name}_exists'] = {'valid': False, 'message': f"{dir_name}/ not found (optional)"}

    return results


def print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results in a readable format."""

    print("=" * 60)
    print(f"Skill Validation Results: {results['skill_name']}")
    print("=" * 60)

    # Print overall status
    status = "✓ PASS" if results['valid'] else "✗ FAIL"
    print(f"\nOverall Status: {status}")

    # Print errors
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ✗ {error}")

    # Print warnings
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")

    # Print detailed checks
    print("\nDetailed Checks:")
    for check_name, check_result in results['checks'].items():
        status_icon = "✓" if check_result['valid'] else "✗"
        print(f"  {status_icon} {check_name}: {check_result['message']}")

    # Print frontmatter info if available
    if results['frontmatter']:
        print("\nFrontmatter:")
        for key, value in results['frontmatter'].items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""

    if len(sys.argv) != 2:
        print("Usage: python validate-skill.py <skill-directory>")
        print("Example: python validate-skill.py .vibe/skills/swarm")
        sys.exit(1)

    skill_dir = sys.argv[1]
    results = validate_skill_directory(skill_dir)
    print_validation_results(results)

    sys.exit(0 if results['valid'] else 1)


if __name__ == "__main__":
    main()
