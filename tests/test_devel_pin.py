"""Wiring tests for the shared per-project pins store, /devel and /sandbox.

Covers agent13.pins (file format, section isolation, devel typed helpers),
the sandbox pin delegation (enum in/out, public API unchanged), and the
shared /devel and /sandbox handlers' pin/unpin/status branches.
"""

import tomllib
from unittest.mock import MagicMock

import pytest

from agent13.commands import execute_devel, execute_sandbox
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


def _make_agent(devel_mode: bool):
    """Stub agent carrying the only state /devel touches."""
    agent = MagicMock()
    agent.devel_mode = devel_mode
    return agent


class TestDevelCommandHandler:
    """/devel shared handler: pin/unpin/status/usage branches."""

    def test_pin_on(self, pins_env):
        result = execute_devel(_make_agent(True), "pin")
        assert result.success
        assert get_pinned_devel() is True
        assert "Pinned devel mode 'on'" in result.message

    def test_pin_off(self, pins_env):
        result = execute_devel(_make_agent(False), "pin")
        assert get_pinned_devel() is False
        assert "Pinned devel mode 'off'" in result.message

    def test_unpin_with_pin(self, pins_env):
        pin_devel(True)
        result = execute_devel(_make_agent(True), "unpin")
        assert result.success
        assert get_pinned_devel() is None
        assert "Removed devel pin" in result.message

    def test_unpin_without_pin(self, pins_env):
        result = execute_devel(_make_agent(False), "unpin")
        assert not result.success
        assert "No devel pin exists" in result.message

    def test_status_shows_pinned(self, pins_env):
        """The status branch reads the pin store."""
        pin_devel(True)
        result = execute_devel(_make_agent(True), "")  # no args = status
        assert result.success
        assert "Devel mode: on" in result.message
        assert "Pinned: on" in result.message

    def test_status_no_pin(self, pins_env):
        result = execute_devel(_make_agent(False), "status")
        assert "Devel mode: off" in result.message
        assert "Pinned: no" in result.message

    def test_usage_text_lists_pin(self, pins_env):
        result = execute_devel(_make_agent(False), "bogus")
        assert not result.success
        assert "/devel pin" in result.message
        assert "/devel unpin" in result.message

    def test_on_off_set_the_flag_and_refresh_the_status_bar(self, pins_env):
        agent = _make_agent(False)
        result = execute_devel(agent, "on")
        agent.set_devel_mode.assert_called_once_with(True)
        assert result.refresh_status is True
        assert "Devel mode enabled" in result.message

        agent = _make_agent(True)
        result = execute_devel(agent, "off")
        agent.set_devel_mode.assert_called_once_with(False)
        assert result.refresh_status is True
        assert "Devel mode disabled" in result.message


class TestSandboxCommandHandler:
    """/sandbox shared handler: status, set, pin, unpin, bad mode."""

    @pytest.fixture(autouse=True)
    def _restore_session_mode(self):
        """The session override is a process-wide global - put it back.

        Leaving it set would sandbox every later test in the same run.
        """
        from tools import security

        saved = security._session_sandbox_mode
        yield
        security._session_sandbox_mode = saved

    def test_status_lists_the_four_sources(self, pins_env):
        result = execute_sandbox(None, "")
        assert result.success
        for label in (
            "Sandbox Configuration:",
            "Current mode:",
            "Session override:",
            "Config default:",
            "Pinned:",
        ):
            assert label in result.message

    def test_set_mode(self, pins_env):
        from tools.security import get_session_sandbox_mode

        result = execute_sandbox(None, "off")
        assert result.success
        assert "Sandbox mode set to: off" in result.message
        assert get_session_sandbox_mode() is SandboxMode.OFF

    def test_invalid_mode_says_error_and_usage(self, pins_env):
        result = execute_sandbox(None, "bogus")
        assert not result.success
        assert "Error" in result.message
        assert "/sandbox pin" in result.message

    def test_pin_and_unpin(self, pins_env):
        execute_sandbox(None, "restrictive-open")
        result = execute_sandbox(None, "pin")
        assert result.success
        assert get_pinned_sandbox_mode() is SandboxMode.RESTRICTIVE_OPEN

        result = execute_sandbox(None, "unpin")
        assert result.success
        assert get_pinned_sandbox_mode() is None

        result = execute_sandbox(None, "unpin")
        assert not result.success
        assert "No sandbox pin exists" in result.message
