"""Widget folding: the /reasoning and /toolview display settings.

Covers the three pieces that carry the feature's rules, none of which has an
obvious home in an existing file:

- ``cli._resolve_widget_collapse`` - the precedence rule (flag > pin > config)
- ``[ui]`` config parsing - defaults, explicit values, fail-fast on a typo,
  and the fresh-install mirror in default_config.toml
- ``commands.execute_collapse`` - the verbs, pin/unpin, and status output
  shared by the TUI and the REPL

The widget behaviour itself lives in tests/test_reasoning_collapse.py; the
REPL's folded output in tests/test_display.py.
"""

import tomllib
from unittest.mock import patch

import pytest

from agent13.cli import _resolve_widget_collapse
from agent13.commands import execute_collapse
from agent13.config import Config
from agent13.pins import (
    SECTION_REASONING,
    SECTION_TOOLVIEW,
    get_pinned_reasoning,
    get_pinned_toolview,
    pin_reasoning,
    pin_toolview,
    unpin_reasoning,
)


@pytest.fixture
def pins_env(tmp_path, monkeypatch):
    """Isolated pin store + a project dir to pin against."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.setenv("AGENT13_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(project_dir)
    return project_dir


def _parse_ui(toml_text: str) -> Config:
    """Config.from_file over a throwaway TOML file."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as f:
        f.write(toml_text)
        path = Path(f.name)
    try:
        return Config.from_file(path)
    finally:
        path.unlink()


# =============================================================================
# Precedence: CLI flag > per-project pin > [ui] config
# =============================================================================


class TestResolveWidgetCollapse:
    """The rule /devel and /sandbox already follow, in one function."""

    def test_flag_collapsed_beats_pin_and_config(self):
        assert _resolve_widget_collapse("collapsed", False, False) is True

    def test_flag_expanded_beats_pin(self):
        """The flag is the only way to override a pin for a single run."""
        assert _resolve_widget_collapse("expanded", True, True) is False

    def test_pin_beats_config(self):
        assert _resolve_widget_collapse(None, False, True) is False
        assert _resolve_widget_collapse(None, True, False) is True

    def test_config_used_with_no_flag_and_no_pin(self):
        assert _resolve_widget_collapse(None, None, True) is True
        assert _resolve_widget_collapse(None, None, False) is False

    def test_pin_false_is_not_treated_as_absent(self):
        """An explicit "expand here" pin must beat a collapsed config."""
        assert _resolve_widget_collapse(None, False, True) is False


# =============================================================================
# [ui] config parsing
# =============================================================================


class TestUiConfigParsing:
    def test_defaults_are_collapsed(self):
        config = Config()
        assert config.reasoning_collapsed is True
        assert config.toolview_collapsed is True

    def test_explicit_values_parse(self):
        config = _parse_ui(
            "[ui]\nreasoning_collapsed = false\ntoolview_collapsed = true\n"
        )
        assert config.reasoning_collapsed is False
        assert config.toolview_collapsed is True

    def test_absent_keys_keep_defaults(self):
        config = _parse_ui("[ui]\ncursor_blink = true\n")
        assert config.reasoning_collapsed is True
        assert config.toolview_collapsed is True

    @pytest.mark.parametrize("key", ["reasoning_collapsed", "toolview_collapsed"])
    def test_non_bool_fails_fast(self, key):
        """A typo must surface at startup, never be silently dropped."""
        from agent13.fileio import ConfigFileError

        with pytest.raises(ConfigFileError, match=key):
            _parse_ui(f'[ui]\n{key} = "yes"\n')

    def test_cursor_blink_stays_lenient(self):
        """The older [ui] key keeps its isinstance-skip parsing."""
        config = _parse_ui("[ui]\ncursor_blink = \"yes\"\n")
        assert config.cursor_blink is False

    def test_shipped_default_config_mirrors_the_keys(self):
        """default_config.toml is the fresh-install seed.

        A [ui] key added to Config without it gives new installs the built-in
        default with no way to see the setting - the drift this pins.
        """
        from agent13.config import DEFAULT_CONFIG_FILE

        shipped = tomllib.loads(DEFAULT_CONFIG_FILE.read_text(encoding="utf-8"))
        ui = shipped.get("ui", {})
        defaults = Config()
        for key in ("reasoning_collapsed", "toolview_collapsed"):
            assert key in ui, f"default_config.toml [ui] is missing '{key}'"
            assert ui[key] == getattr(defaults, key), (
                f"default_config.toml [ui] '{key}' = {ui[key]!r} disagrees with "
                f"Config().{key} = {getattr(defaults, key)!r}"
            )


# =============================================================================
# execute_collapse - the shared /reasoning and /toolview implementation
# =============================================================================


class _Session:
    """Records the session changes execute_collapse applies."""

    def __init__(self, collapsed: bool):
        self.collapsed = collapsed
        self.set_calls: list[bool] = []

    def set(self, value: bool) -> None:
        self.set_calls.append(value)
        self.collapsed = value


class TestCollapseVerbs:
    @pytest.mark.parametrize("kind", ["reasoning", "toolview"])
    def test_hide_folds_for_the_session(self, kind):
        s = _Session(False)
        result = execute_collapse(kind, "hide", s.collapsed, s.set)
        assert result.success is True
        assert s.collapsed is True
        assert result.data["collapsed"] is True
        assert "collapsed for this session" in result.message

    @pytest.mark.parametrize("kind", ["reasoning", "toolview"])
    def test_show_expands_for_the_session(self, kind):
        s = _Session(True)
        result = execute_collapse(kind, "show", s.collapsed, s.set)
        assert result.success is True
        assert s.collapsed is False
        assert result.data["collapsed"] is False
        assert "expanded for this session" in result.message

    def test_verbs_are_case_and_space_insensitive(self):
        s = _Session(False)
        assert execute_collapse("toolview", "  HIDE ", s.collapsed, s.set).success

    def test_unknown_verb_fails_with_usage(self):
        s = _Session(True)
        result = execute_collapse("toolview", "fold", s.collapsed, s.set)
        assert result.success is False
        assert s.set_calls == [], "a rejected verb must not change the setting"
        for verb in ("hide", "show", "pin", "unpin"):
            assert verb in result.message

    def test_no_session_change_on_pin_verbs(self):
        s = _Session(True)
        with patch("agent13.pins.set_pinned"):
            execute_collapse("reasoning", "pin", s.collapsed, s.set)
        assert s.set_calls == []


class TestCollapsePins:
    def test_pin_captures_the_current_effective_value(self, pins_env):
        s = _Session(True)
        result = execute_collapse("reasoning", "pin", s.collapsed, s.set)
        assert result.success is True
        assert get_pinned_reasoning() is True
        assert "collapsed" in result.message

    def test_pin_of_expanded_is_pinned_as_expanded(self, pins_env):
        s = _Session(False)
        execute_collapse("toolview", "pin", s.collapsed, s.set)
        assert get_pinned_toolview() is False

    def test_pin_is_per_kind(self, pins_env):
        execute_collapse("reasoning", "pin", True, _Session(True).set)
        assert get_pinned_reasoning() is True
        assert get_pinned_toolview() is None

    def test_unpin_removes_and_reports(self, pins_env):
        pin_toolview(True)
        result = execute_collapse("toolview", "unpin", True, _Session(True).set)
        assert result.success is True
        assert get_pinned_toolview() is None

    def test_unpin_without_a_pin_fails_and_changes_nothing(self, pins_env):
        result = execute_collapse("toolview", "unpin", True, _Session(True).set)
        assert result.success is False
        assert "No toolview pin exists" in result.message

    def test_pin_helpers_round_trip_and_isolate_sections(self, pins_env):
        pin_reasoning(True)
        pin_toolview(False)
        assert get_pinned_reasoning() is True
        assert get_pinned_toolview() is False
        assert unpin_reasoning() is True
        assert get_pinned_reasoning() is None
        assert get_pinned_toolview() is False, "unpinning one kind leaves the other"
        assert unpin_reasoning() is False

    def test_helpers_use_the_documented_sections(self, pins_env):
        from agent13.pins import load_pins

        pin_reasoning(True)
        pin_toolview(True)
        sections = load_pins()
        assert list(sections) == [SECTION_REASONING, SECTION_TOOLVIEW]

    def test_non_bool_stored_value_reads_as_no_pin(self, pins_env):
        """A hand-edited pins.toml must not crash startup."""
        from agent13.pins import set_pinned

        set_pinned(SECTION_TOOLVIEW, "collapsed")
        assert get_pinned_toolview() is None


class TestCollapseStatus:
    def _status(self, kind, collapsed, pinned, cfg_default):
        from agent13.config import Config

        fake = Config()
        fake.reasoning_collapsed = cfg_default
        fake.toolview_collapsed = cfg_default
        with patch("agent13.config.get_config", return_value=fake):
            return execute_collapse(kind, "", collapsed, _Session(collapsed).set)

    def test_status_reports_effective_pin_and_config(self, pins_env):
        pin_toolview(True)
        result = self._status("toolview", False, True, True)
        assert result.success is True
        assert "expanded" in result.message
        assert "Pinned for this project: collapsed" in result.message
        assert "Config default: collapsed" in result.message

    def test_status_says_no_when_unpinned(self, pins_env):
        result = self._status("toolview", True, None, True)
        assert "Pinned for this project: no" in result.message

    def test_reasoning_status_points_at_the_api_control(self, pins_env):
        """The two axes are easy to confuse; the status says which is which."""
        result = self._status("reasoning", True, None, True)
        assert "/remove-api-reasoning" in result.message

    def test_toolview_status_does_not_mention_the_api(self, pins_env):
        result = self._status("toolview", True, None, True)
        assert "remove-api-reasoning" not in result.message

    def test_status_changes_nothing(self, pins_env):
        s = _Session(True)
        from agent13.config import Config

        with patch("agent13.config.get_config", return_value=Config()):
            execute_collapse("reasoning", "", True, s.set)
        assert s.set_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
