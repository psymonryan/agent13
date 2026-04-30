"""Shared YAML load/save utilities for config files."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict on missing or corrupt file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dict, or empty dict if file missing/unreadable.
    """
    if path.exists():
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def save_yaml(path: Path, data: dict) -> None:
    """Save a dict to a YAML file, creating parent dirs as needed.

    Args:
        path: Path to YAML file.
        data: Dict to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
