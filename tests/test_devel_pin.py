"""Wiring tests for the shared per-project pins store and /devel pin.

Covers agent13.pins (file format, section isolation, devel typed helpers),
the sandbox pin delegation (enum in/out, public API unchanged), and the TUI
/devel handler's pin/unpin/status branches.
"""

import tomllib
from unittest.mock import MagicMock, patch

import pytest

from agent13.pins import (
    SECTION_DEVEL,
    SECTION_SANDBOX,
    get_pinned,
    get_pinned_devel,
    load_pins,
    pin_devel,
    remove_pin,
    set_pinned,
    unpin_devel,
)
from agent13.sandbox import (
    SandboxMode,
    get_pinned_sandbox_mode,
    pin_sandbox_mode,
    unpin_sandbox_mode,
)


@pytest.fixture
def pins_env(tmp_path, monkeypatch):
    """Isolated config dir + temp project dir for pin operations."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(project_dir)
    return project_dir


@pytest.fixture
def pins_file(pins_env):
    from agent13.config_paths import get_config_dir

    return get_config_dir() / "pins.toml"


class TestDevelPinStore:
    """agent13.pins devel typed helpers and file format."""

    def test_no_pin_returns_none(self, pins_env):
        assert get_pinned_devel() is None
        assert get_pinned_devel(pins_env) is None

    def test_pin_and_get_true(self, pins_env, pins_file):
        pin_devel(True)
        assert get_pinned_devel() is True
        assert pins_file.exists()
        with open(pins_file, "rb") as f:
            data = tomllib.load(f)
        assert data[SECTION_DEVEL][str(pins_env.resolve())] is True

    def test_pin_off_is_distinct_from_no_pin(self, pins_env):
        """A pinned-off project returns False, not None."""
        pin_devel(False)
        assert get_pinned_devel() is False
        unpin_devel()
        assert get_pinned_devel() is None

    def test_unpin_removes_pin(self, pins_env):
        pin_devel(True)
        assert unpin_devel() is True
        assert unpin_devel() is False  # second unpin: no pin existed
        assert get_pinned_devel() is None

    def test_overwrite_pin(self, pins_env):
        pin_devel(True)
        pin_devel(False)
        assert get_pinned_devel() is False

    def test_project_isolation(self, pins_env, tmp_path):
        """A pin for one project dir doesn't leak to another."""
        other = tmp_path / "other-proj"
        other.mkdir()
        pin_devel(True)
        assert get_pinned_devel(pins_env) is True
        assert get_pinned_devel(other) is None

    def test_invalid_file_returns_empty(self, pins_env, pins_file):
        pins_file.write_text("not [valid toml ===")
        assert get_pinned_devel() is None
        assert load_pins() == {}


class TestSectionIsolation:
    """Sandbox and devel pins coexist in one file, independent of each other."""

    def test_both_sections_in_file(self, pins_env, pins_file):
        pin_sandbox_mode(SandboxMode.RESTRICTIVE_OPEN)
        pin_devel(True)
        with open(pins_file, "rb") as f:
            data = tomllib.load(f)
        key = str(pins_env.resolve())
        assert data[SECTION_SANDBOX][key] == "restrictive-open"
        assert data[SECTION_DEVEL][key] is True

    def test_unpin_devel_leaves_sandbox(self, pins_env):
        pin_sandbox_mode(SandboxMode.OFF)
        pin_devel(True)
        unpin_devel()
        assert get_pinned_devel() is None
        assert get_pinned_sandbox_mode() is SandboxMode.OFF

    def test_empty_section_omitted_on_save(self, pins_env, pins_file):
        pin_devel(True)
        unpin_devel()
        with open(pins_file, "rb") as f:
            data = tomllib.load(f)
        assert SECTION_DEVEL not in data

    def test_generic_helpers(self, pins_env, pins_file):
        set_pinned(SECTION_DEVEL, True, pins_env)
        assert get_pinned(SECTION_DEVEL, pins_env) is True
        assert remove_pin(SECTION_DEVEL, pins_env) is True
        assert remove_pin(SECTION_DEVEL, pins_env) is False
        # File still parseable after the last pin is removed
        with open(pins_file, "rb") as f:
            tomllib.load(f)


class TestSandboxPinDelegation:
    """sandbox.py pin API is unchanged: SandboxMode in and out."""

    def test_pin_get_roundtrip(self, pins_env):
        pin_sandbox_mode(SandboxMode.PERMISSIVE_CLOSED)
        assert get_pinned_sandbox_mode() is SandboxMode.PERMISSIVE_CLOSED

    def test_unpin(self, pins_env):
        pin_sandbox_mode(SandboxMode.OFF)
        assert unpin_sandbox_mode() is True
        assert unpin_sandbox_mode() is False
        assert get_pinned_sandbox_mode() is None

    def test_no_pin_returns_none(self, pins_env):
        assert get_pinned_sandbox_mode() is None

    def test_invalid_stored_value_returns_none(self, pins_env, pins_file):
        pin_sandbox_mode(SandboxMode.OFF)
        # Corrupt the stored mode string
        content = pins_file.read_text().replace('"off"', '"bogus-mode"')
        pins_file.write_text(content)
        assert get_pinned_sandbox_mode() is None


def _make_tui_app(agent_devel_mode: bool):
    """Build a real AgentTUI with a stub agent (see test_skills_slash_args)."""
    from ui.tui import AgentTUI as ChatApp

    with patch("ui.tui.get_config", return_value=MagicMock()):
        app = ChatApp(
            client=MagicMock(),
            model="test-model",
            model_names=["test-model"],
            provider="test",
        )
    agent = MagicMock()
    agent.devel_mode = agent_devel_mode
    app.agent = agent
    return app, agent


@pytest.mark.asyncio
class TestTuiDevelHandler:
    """TUI /devel handler pin/unpin/status branches (real handler code)."""

    async def test_pin_on(self, pins_env):
        app, agent = _make_tui_app(True)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("pin")
        assert get_pinned_devel() is True
        assert "Pinned devel mode 'on'" in captured[0]

    async def test_pin_off(self, pins_env):
        app, agent = _make_tui_app(False)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("pin")
        assert get_pinned_devel() is False
        assert "Pinned devel mode 'off'" in captured[0]

    async def test_unpin_with_pin(self, pins_env):
        pin_devel(True)
        app, agent = _make_tui_app(True)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("unpin")
        assert get_pinned_devel() is None
        assert "Removed devel pin" in captured[0]

    async def test_unpin_without_pin(self, pins_env):
        app, agent = _make_tui_app(False)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("unpin")
        assert get_pinned_devel() is None
        assert "No devel pin exists" in captured[0]

    async def test_status_shows_pinned(self, pins_env):
        """Status branch reads the pin store - verify via the info content."""
        pin_devel(True)
        app, agent = _make_tui_app(True)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("")  # no args = status
        assert captured, "handler did not update info content"
        assert "Pinned: [yellow]on[/]" in captured[0]
        assert "Devel mode: on" in captured[0]

    async def test_status_no_pin(self, pins_env):
        app, agent = _make_tui_app(False)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("status")
        assert "Pinned: [dim]no[/]" in captured[0]

    async def test_usage_text_lists_pin(self, pins_env):
        app, agent = _make_tui_app(False)
        captured = []
        app._update_info_content = lambda text: captured.append(text)
        app._handle_devel_command("bogus")
        assert "/devel pin" in captured[0]
        assert "/devel unpin" in captured[0]
