"""Configuration path resolution for agent13.

Paths can be overridden via environment variables:
- AGENT13_CONFIG_DIR: Override default ~/.agent13 directory
"""

from pathlib import Path
import os


def get_config_dir() -> Path:
    """Return the configuration directory path.

    Default: ~/.agent13
    Override: Set AGENT13_CONFIG_DIR environment variable
    """
    env_dir = os.environ.get("AGENT13_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / ".agent13"


def get_config_file() -> Path:
    """Return the path to config.toml."""
    return get_config_dir() / "config.toml"


def get_global_env_file() -> Path:
    """Return the path to global .env file (~/.env)."""
    return Path.home() / ".env"


def get_local_env_file() -> Path:
    """Return the path to local .env file (./env in current directory)."""
    return Path.cwd() / ".env"


def ensure_config_dir() -> Path:
    """Ensure the config directory exists and return it."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_saves_dir() -> Path:
    """Return the saves directory path (~/.agent13/saves/).

    Creates the directory if it doesn't exist.
    """
    saves_dir = get_config_dir() / "saves"
    saves_dir.mkdir(parents=True, exist_ok=True)
    return saves_dir


def get_skills_dir() -> Path:
    """Return the global skills directory path (~/.agent13/skills/).

    Creates the directory if it doesn't exist.
    """
    skills_dir = get_config_dir() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


def get_history_dir() -> Path:
    """Return the history directory path (~/.agent13/).

    Note: History files are stored directly in ~/.agent13/ with naming
    pattern: history-{project}-{date}
    """
    return ensure_config_dir()


def get_history_path(project_name: str | None = None, suffix: str = "") -> Path:
    """Get the history file path for a project.

    Args:
        project_name: Project identifier. If None, uses basename of cwd.
                      Falls back to "global" if no cwd available.
        suffix: Optional suffix (e.g., "_test" for pytest).

    Returns:
        Path like ~/.agent13/history-{project}{suffix}-{YYYY-MM-DD}
    """
    if project_name is None:
        cwd = Path.cwd()
        project_name = cwd.name if cwd else "global"

    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    return get_history_dir() / f"history-{project_name}{suffix}-{today}"


def get_prompts_file() -> Path:
    """Return the path to prompts.yaml (~/.agent13/prompts.yaml)."""
    return get_config_dir() / "prompts.yaml"


def get_snippets_file() -> Path:
    """Return the path to snippets.yaml (~/.agent13/snippets.yaml)."""
    return get_config_dir() / "snippets.yaml"
