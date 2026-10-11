"""Per-project pin storage - single file: ``~/.agent13/pins.toml``.

A pin makes a session setting persistent per project directory so it
auto-applies on every startup in that directory.  Each pin type lives in
its own TOML section, keyed by the resolved absolute project path:

    # Per-project pins
    # Managed by /sandbox, /devel, /reasoning and /toolview pin|unpin

    [sandbox]
    "/abs/path" = "permissive-closed"

    [devel]
    "/abs/path" = true

Currently four pin types exist:

- ``sandbox`` - pinned sandbox mode, managed by ``/sandbox pin`` /
  ``/sandbox unpin`` (typed helpers in :mod:`agent13.sandbox`)
- ``devel`` - pinned devel mode, managed by ``/devel pin`` /
  ``/devel unpin`` (typed helpers below; ``False`` is a valid pin value,
  meaning "devel pinned off")
- ``reasoning`` - pinned default for the reasoning widgets (``true`` =
  collapsed), managed by ``/reasoning pin`` / ``/reasoning unpin``
- ``toolview`` - pinned default for the tool call/result widgets (``true`` =
  collapsed), managed by ``/toolview pin`` / ``/toolview unpin``

An explicit CLI flag (``--sandbox``, ``--devel``, ``--reasoning``,
``--toolview``) always overrides a pin.
"""

import tomllib
from pathlib import Path
from typing import Optional, Union

SECTION_SANDBOX = "sandbox"
SECTION_DEVEL = "devel"
SECTION_REASONING = "reasoning"
SECTION_TOOLVIEW = "toolview"

_PinValue = Union[str, bool]


def _get_pins_file() -> Path:
    """Return the path to the pins file."""
    from agent13.config_paths import get_config_dir

    return get_config_dir() / "pins.toml"


def load_pins() -> dict[str, dict]:
    """Load all pins from the pins file.

    Returns:
        Dict mapping section name -> {absolute project path -> value}.
        Sandbox values are strings, devel values are bools.
        Empty dict if the file doesn't exist or is invalid.
    """
    pins_file = _get_pins_file()
    if not pins_file.exists():
        return {}

    try:
        with open(pins_file, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}

    return {
        section: {k: v for k, v in values.items() if isinstance(v, (str, bool))}
        for section, values in data.items()
        if isinstance(values, dict)
    }


def save_pins(pins: dict[str, dict]) -> None:
    """Write all pins (every section) to the pins file.

    Args:
        pins: Dict mapping section name -> {absolute project path -> value}.
              Empty sections are omitted.
    """
    from agent13.config_paths import ensure_config_dir

    pins_file = _get_pins_file()
    ensure_config_dir()

    lines = [
        "# Per-project pins - persistent per-project settings",
        "# Managed by /sandbox, /devel, /reasoning and /toolview pin|unpin",
    ]
    for section in sorted(pins):
        values = pins[section]
        if not values:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for path in sorted(values):
            value = values[path]
            escaped = path.replace("\\", "\\\\")
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = f'"{value}"'
            lines.append(f'"{escaped}" = {rendered}')

    pins_file.write_text("\n".join(lines) + "\n")


def get_pinned(
    section: str,
    project_dir: Optional[Path] = None,
) -> Optional[_PinValue]:
    """Get the pinned value for a project directory in a section.

    Args:
        section: Pin section name (e.g. ``SECTION_DEVEL``).
        project_dir: The project directory. Defaults to cwd.

    Returns:
        The pinned value, or None if no pin exists.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    key = str(project_dir.resolve())
    return load_pins().get(section, {}).get(key)


def set_pinned(
    section: str,
    value: _PinValue,
    project_dir: Optional[Path] = None,
) -> None:
    """Pin a value for a project directory in a section.

    Args:
        section: Pin section name (e.g. ``SECTION_DEVEL``).
        value: The value to pin (str or bool).
        project_dir: The project directory. Defaults to cwd.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    pins = load_pins()
    pins.setdefault(section, {})[str(project_dir.resolve())] = value
    save_pins(pins)


def remove_pin(section: str, project_dir: Optional[Path] = None) -> bool:
    """Remove a pin for a project directory in a section.

    Args:
        section: Pin section name (e.g. ``SECTION_DEVEL``).
        project_dir: The project directory. Defaults to cwd.

    Returns:
        True if a pin was removed, False if no pin existed.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    key = str(project_dir.resolve())
    pins = load_pins()
    section_pins = pins.get(section, {})
    if key not in section_pins:
        return False
    del section_pins[key]
    if section_pins:
        pins[section] = section_pins
    else:
        pins.pop(section, None)
    save_pins(pins)
    return True


# ── Boolean pins (typed helpers) ─────────────────────────────────────────────
# Sandbox pins keep their typed helpers in agent13.sandbox because the value
# is the SandboxMode enum; every other pin type here is a plain bool.


def _get_pinned_bool(
    section: str, project_dir: Optional[Path] = None
) -> Optional[bool]:
    """Get a bool pin for a project directory (None = no pin, or non-bool value)."""
    value = get_pinned(section, project_dir)
    return value if isinstance(value, bool) else None


def get_pinned_devel(project_dir: Optional[Path] = None) -> Optional[bool]:
    """Get the pinned devel mode for a project directory.

    Args:
        project_dir: The project directory. Defaults to cwd.

    Returns:
        True/False if devel mode is pinned, None if no pin exists.
    """
    return _get_pinned_bool(SECTION_DEVEL, project_dir)


def pin_devel(enabled: bool, project_dir: Optional[Path] = None) -> None:
    """Pin a devel mode for a project directory.

    Args:
        enabled: The devel mode to pin (on or off).
        project_dir: The project directory. Defaults to cwd.
    """
    set_pinned(SECTION_DEVEL, enabled, project_dir)


def unpin_devel(project_dir: Optional[Path] = None) -> bool:
    """Remove the devel pin for a project directory.

    Args:
        project_dir: The project directory. Defaults to cwd.

    Returns:
        True if a pin was removed, False if no pin existed.
    """
    return remove_pin(SECTION_DEVEL, project_dir)


# ── UI display pins ──────────────────────────────────────────────────────────
# The pinned bool means "collapsed": True = the widget family starts folded,
# False = it starts expanded. See /reasoning and /toolview.


def get_pinned_reasoning(project_dir: Optional[Path] = None) -> Optional[bool]:
    """Get the pinned reasoning-widget default (True = collapsed)."""
    return _get_pinned_bool(SECTION_REASONING, project_dir)


def pin_reasoning(collapsed: bool, project_dir: Optional[Path] = None) -> None:
    """Pin the reasoning-widget default for a project directory."""
    set_pinned(SECTION_REASONING, collapsed, project_dir)


def unpin_reasoning(project_dir: Optional[Path] = None) -> bool:
    """Remove the reasoning pin. True if a pin was removed, False if none."""
    return remove_pin(SECTION_REASONING, project_dir)


def get_pinned_toolview(project_dir: Optional[Path] = None) -> Optional[bool]:
    """Get the pinned tool-widget default (True = collapsed)."""
    return _get_pinned_bool(SECTION_TOOLVIEW, project_dir)


def pin_toolview(collapsed: bool, project_dir: Optional[Path] = None) -> None:
    """Pin the tool-widget default for a project directory."""
    set_pinned(SECTION_TOOLVIEW, collapsed, project_dir)


def unpin_toolview(project_dir: Optional[Path] = None) -> bool:
    """Remove the toolview pin. True if a pin was removed, False if none."""
    return remove_pin(SECTION_TOOLVIEW, project_dir)
