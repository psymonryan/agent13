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


def _ensure_dir(path: Path) -> Path:
    """Ensure a directory exists at *path*, return it.

    Handles the case where a stale regular file occupies the path (e.g.
    left behind by an uninstall).  ``mkdir(exist_ok=True)`` raises
    ``FileExistsError`` in that situation because the path is not a
    directory.

    **Symlinks are never touched** — a symlink (even a broken one) is
    treated as an intentional user configuration, e.g. linking
    ``~/.agent13/skills`` to a shared drive.  If the symlink target is
    unreachable the mkdir will still fail, but that is a mount/network
    problem, not something we should silently destroy.
    """
    if path.is_symlink():
        return path
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_config_dir() -> Path:
    """Ensure the config directory exists and return it."""
    return _ensure_dir(get_config_dir())


def get_global_saves_dir() -> Path:
    """Return the global saves directory path (~/.agent13/saves/).

    Creates the directory if it doesn't exist. This is the "central" save
    location used for auto-saves when ``[saves] location = "central"`` in
    config, and the default home for all cross-project saves.
    """
    return _ensure_dir(get_config_dir() / "saves")


def records_dir_for(project_dir: Path) -> Path:
    """Return the durable-record directory for *project_dir*.

    ``<project>/records/`` - in the project root, NOT under ``.agent13/``.
    ``.agent13`` is gitignored in every project because it holds local state
    (saves, skills, locks); a durable record is the opposite - it must be
    versioned, so it lives where git can see it. Holds a focus's
    ``<name>_current_truth.md`` and ``<name>_journal.md``.

    Takes the project directory explicitly so a caller that walks many
    projects (the migration script) does not get the cwd's directory for
    every one of them.
    """
    return Path(project_dir) / "records"


def get_records_dir(create: bool = True) -> Path:
    """Return the durable-record directory for the CURRENT project (cwd).

    Created on demand unless *create* is False - a caller that only asks a
    question about the path (a predicate, a dry run) must not leave a
    directory behind.
    """
    path = records_dir_for(Path.cwd())
    return _ensure_dir(path) if create else path


def get_skills_dir() -> Path:
    """Return the global skills directory path (~/.agent13/skills/).

    Creates the directory if it doesn't exist.
    """
    return _ensure_dir(get_config_dir() / "skills")


def get_history_dir() -> Path:
    """Return the history directory path (~/.agent13/history/).

    History files are stored in ~/.agent13/history/ with naming
    pattern: history-{project}-{date}.

    The directory is created on demand. If ~/.agent13/history is itself a
    symlink (e.g. to a shared drive) it is followed and never modified;
    see _ensure_dir().
    """
    return _ensure_dir(get_config_dir() / "history")


def get_locks_dir() -> Path:
    """Return the locks directory path (~/.agent13/locks/).

    Polite-mode lock files live here. Created on demand.
    """
    return _ensure_dir(get_config_dir() / "locks")


def get_history_path(project_name: str | None = None, suffix: str = "") -> Path:
    """Get the history file path for a project.

    Args:
        project_name: Project identifier. If None, uses basename of cwd.
                      Falls back to "global" if no cwd available.
        suffix: Optional suffix (e.g., "_test" for pytest).

    Returns:
        Path like ~/.agent13/history/history-{project}{suffix}-{YYYY-MM-DD}
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
