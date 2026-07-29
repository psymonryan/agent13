"""Shared YAML load/save utilities for config files."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict only if file is missing.

    A missing file is normal (first run) and returns {}. A file that
    exists but cannot be parsed raises YAMLError — callers should let
    this propagate so the user sees the error (fail-fast strategy).

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dict, or empty dict if file is missing.

    Raises:
        yaml.YAMLError: If file exists but contains invalid YAML.
        ValueError: If file exists but top-level structure is not a dict.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping/dict at top level of {path}, "
            f"got {type(data).__name__}"
        )
    return data


def save_yaml(path: Path, data: dict) -> None:
    """Save a dict to a YAML file, creating parent dirs as needed.

    Args:
        path: Path to YAML file.
        data: Dict to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
