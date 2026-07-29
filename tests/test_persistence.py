"""Tests for agent13.persistence — save/load context, auto-save locations, listing."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from agent13.config import Config
from agent13.persistence import (
    _ensure_ctx_stem,
    find_latest_auto_save,
    get_auto_save_dir,
    get_auto_save_path,
    get_saves_dir,
    list_all_saves,
    list_saves,
    load_context,
    resolve_save_path,
    save_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(saves_location="central"):
    """Create a Config with the given saves_location."""
    return Config(saves_location=saves_location)


def _touch(path: Path, mtime: float | None = None):
    """Create an empty .ctx file and optionally set its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    if mtime is not None:
        import os

        os.utime(path, (mtime, mtime))
    return path


def _make_agent_stub(messages=None, model="test-model", system_prompt=""):
    """Minimal Agent-like object for save/load tests."""

    class _Stub:
        def __init__(self):
            self.messages = messages or []
            self.model = model
            self.system_prompt = system_prompt
            self.session_date = None
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self._incomplete = False

        def mark_incomplete_turn(self, val):
            self._incomplete = val

    return _Stub()


# ---------------------------------------------------------------------------
# get_saves_dir
# ---------------------------------------------------------------------------


class TestGetSavesDir:
    def test_returns_project_local_dir(self, monkeypatch, tmp_path):
        """get_saves_dir returns ./.agent13/saves/ relative to cwd."""
        monkeypatch.chdir(tmp_path)
        result = get_saves_dir()
        assert result == tmp_path / ".agent13" / "saves"
        assert result.is_dir()

    def test_creates_dir_if_missing(self, monkeypatch, tmp_path):
        """Directory is created on first call."""
        monkeypatch.chdir(tmp_path)
        result = get_saves_dir()
        assert result.exists()


# ---------------------------------------------------------------------------
# get_auto_save_dir
# ---------------------------------------------------------------------------


class TestGetAutoSaveDir:
    def test_central_mode(self, monkeypatch, tmp_path):
        """Central config returns global saves dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir",
            lambda: tmp_path / "global_saves",
        )
        result = get_auto_save_dir()
        assert result == tmp_path / "global_saves"

    def test_local_mode(self, monkeypatch, tmp_path):
        """Local config returns project-local saves dir."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )
        result = get_auto_save_dir()
        assert result == tmp_path / ".agent13" / "saves"


# ---------------------------------------------------------------------------
# get_auto_save_path
# ---------------------------------------------------------------------------


class TestGetAutoSavePath:
    def test_uses_project_name_and_date(self, monkeypatch, tmp_path):
        """Path includes project name and today's date."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir",
            lambda: tmp_path / "global_saves",
        )
        today = datetime.now().strftime("%Y-%m-%d")
        result = get_auto_save_path("myproject")
        assert result.name == f"myproject-{today}.ctx"
        assert "global_saves" in str(result)

    def test_uses_cwd_name_when_no_project(self, monkeypatch, tmp_path):
        """Falls back to cwd name when no project_name given."""
        project_dir = tmp_path / "cool_project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir",
            lambda: tmp_path / "global_saves",
        )
        today = datetime.now().strftime("%Y-%m-%d")
        result = get_auto_save_path()
        assert result.name == f"cool_project-{today}.ctx"

    def test_local_mode_uses_project_dir(self, monkeypatch, tmp_path):
        """Local mode puts auto-save in ./.agent13/saves/ without project prefix."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )
        today = datetime.now().strftime("%Y-%m-%d")
        result = get_auto_save_path("myproject")
        # Local mode: no project prefix (directory is project-specific)
        assert result == tmp_path / ".agent13" / "saves" / f"{today}.ctx"

    def test_dashed_date_format(self, monkeypatch, tmp_path):
        """Date format is YYYY-MM-DD (dashed), not YYYYMMDD."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir",
            lambda: tmp_path / "global_saves",
        )
        result = get_auto_save_path("p")
        # Should contain dashes in the date portion
        stem = result.stem  # e.g. "p-2026-05-25"
        parts = stem.split("-")
        assert len(parts) == 4  # ["p", "2026", "05", "25"]

    def test_uses_session_date_when_provided(self, monkeypatch, tmp_path):
        """Session date overrides today in the filename."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir",
            lambda: tmp_path / "global_saves",
        )
        result = get_auto_save_path("myproject", session_date="2026-01-15")
        assert result.name == "myproject-2026-01-15.ctx"

    def test_local_mode_uses_session_date(self, monkeypatch, tmp_path):
        """Local mode also respects session_date."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )
        result = get_auto_save_path(session_date="2026-01-15")
        assert result == tmp_path / ".agent13" / "saves" / "2026-01-15.ctx"


# ---------------------------------------------------------------------------
# find_latest_auto_save
# ---------------------------------------------------------------------------


class TestFindLatestAutoSave:
    def test_returns_none_when_empty(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        auto_dir = tmp_path / "global_saves"
        auto_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: auto_dir
        )
        assert find_latest_auto_save("proj") is None

    def test_returns_most_recent_by_mtime(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        auto_dir = tmp_path / "global_saves"
        auto_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: auto_dir
        )
        _touch(auto_dir / "proj-2026-05-01.ctx", mtime=1000)
        _touch(auto_dir / "proj-2026-05-10.ctx", mtime=2000)
        _touch(auto_dir / "proj-2026-05-20.ctx", mtime=3000)
        result = find_latest_auto_save("proj")
        assert result.name == "proj-2026-05-20.ctx"

    def test_filters_by_project_name(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        auto_dir = tmp_path / "global_saves"
        auto_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: auto_dir
        )
        _touch(auto_dir / "other-2026-05-25.ctx", mtime=5000)
        _touch(auto_dir / "proj-2026-05-10.ctx", mtime=3000)
        result = find_latest_auto_save("proj")
        assert result.name == "proj-2026-05-10.ctx"

    def test_local_mode_searches_local_dir(self, monkeypatch, tmp_path):
        """Local mode: auto-saves use date-only naming."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )
        local_dir = tmp_path / ".agent13" / "saves"
        # Local mode: date-only filename (no project prefix)
        _touch(local_dir / "2026-05-25.ctx", mtime=9000)
        result = find_latest_auto_save("proj")
        assert result is not None
        assert result.name == "2026-05-25.ctx"

    def test_fallback_from_central_to_local(self, monkeypatch, tmp_path):
        """Configured for central but save is in local dir — should find it."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        # Central dir is empty
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )
        # Local dir has the save (date-only format for local mode)
        local_dir = tmp_path / ".agent13" / "saves"
        _touch(local_dir / "2026-05-25.ctx", mtime=9000)

        result = find_latest_auto_save("proj")
        assert result is not None
        assert result.name == "2026-05-25.ctx"

    def test_fallback_from_local_to_central(self, monkeypatch, tmp_path):
        """Configured for local but save is in central dir — should find it."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )
        # Local dir is empty (just the dir, no matching files)
        # (get_saves_dir creates it automatically)
        # Central dir has the save
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        _touch(global_dir / "proj-2026-06-01.ctx", mtime=9000)
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )

        result = find_latest_auto_save("proj")
        assert result is not None
        assert result.name == "proj-2026-06-01.ctx"

    def test_no_fallback_when_primary_has_match(self, monkeypatch, tmp_path):
        """Primary location match should be returned, not fallback."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )
        # Both locations have saves; primary (central) should win
        _touch(global_dir / "proj-2026-05-20.ctx", mtime=3000)
        local_dir = tmp_path / ".agent13" / "saves"
        _touch(local_dir / "proj-2026-05-25.ctx", mtime=9000)

        result = find_latest_auto_save("proj")
        assert result is not None
        assert result.name == "proj-2026-05-20.ctx"  # central, not local

    def test_returns_none_when_both_locations_empty(self, monkeypatch, tmp_path):
        """No saves in either location — return None."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )
        # Local dir exists but is empty too
        (tmp_path / ".agent13" / "saves").mkdir(parents=True, exist_ok=True)

        result = find_latest_auto_save("proj")
        assert result is None


# ---------------------------------------------------------------------------
# list_saves
# ---------------------------------------------------------------------------


class TestListSaves:
    def test_empty_dir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert list_saves() == []

    def test_returns_only_ctx_files(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        saves_dir = get_saves_dir()
        _touch(saves_dir / "one.ctx")
        _touch(saves_dir / "two.txt")
        _touch(saves_dir / "three.ctx")
        result = list_saves()
        names = [p.name for p in result]
        assert "one.ctx" in names
        assert "three.ctx" in names
        assert "two.txt" not in names


# ---------------------------------------------------------------------------
# list_all_saves
# ---------------------------------------------------------------------------


class TestListAllSaves:
    def test_manual_then_auto_central(self, monkeypatch, tmp_path):
        """Central mode: manual saves first, then auto-saves from global dir."""
        # Create a fake project directory so Path.cwd().name returns "proj"
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        monkeypatch.chdir(proj_dir)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )

        saves_dir = get_saves_dir()
        _touch(saves_dir / "manual-a.ctx", mtime=100)
        _touch(saves_dir / "manual-b.ctx", mtime=200)
        _touch(global_dir / "proj-2026-05-01.ctx", mtime=300)
        _touch(global_dir / "proj-2026-05-20.ctx", mtime=400)

        result = list_all_saves()
        names = [p.name for p in result]
        # Manual saves first (mtime desc), then auto (mtime desc)
        assert names == [
            "manual-b.ctx",
            "manual-a.ctx",
            "proj-2026-05-20.ctx",
            "proj-2026-05-01.ctx",
        ]

    def test_manual_then_auto_local(self, monkeypatch, tmp_path):
        """Local mode: all .ctx in same dir, sorted by mtime desc.

        In local mode list_saves() returns ALL .ctx files (auto-saves included).
        The auto_saves list is deduped away, so everything ends up in the
        manual group sorted by mtime.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("local")
        )

        saves_dir = get_saves_dir()
        _touch(saves_dir / "named-save.ctx", mtime=500)
        _touch(saves_dir / "older-save.ctx", mtime=100)
        _touch(saves_dir / "proj-2026-05-25.ctx", mtime=300)

        result = list_all_saves()
        names = [p.name for p in result]
        # All in same dir → all sorted by mtime descending
        assert names == [
            "named-save.ctx",
            "proj-2026-05-25.ctx",
            "older-save.ctx",
        ]

    def test_only_auto_saves(self, monkeypatch, tmp_path):
        """No manual saves, only auto-saves."""
        # Create a fake project directory so Path.cwd().name returns "proj"
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        monkeypatch.chdir(proj_dir)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )
        get_saves_dir()  # ensure dir exists

        _touch(global_dir / "proj-2026-05-10.ctx", mtime=100)
        _touch(global_dir / "proj-2026-05-20.ctx", mtime=200)

        result = list_all_saves()
        names = [p.name for p in result]
        assert names == ["proj-2026-05-20.ctx", "proj-2026-05-10.ctx"]

    def test_only_manual_saves(self, monkeypatch, tmp_path):
        """No auto-saves, only manual."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )

        saves_dir = get_saves_dir()
        _touch(saves_dir / "a.ctx", mtime=50)
        _touch(saves_dir / "b.ctx", mtime=150)

        result = list_all_saves()
        names = [p.name for p in result]
        assert names == ["b.ctx", "a.ctx"]

    def test_empty(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "agent13.config.get_config", lambda: _make_config("central")
        )
        global_dir = tmp_path / "global_saves"
        global_dir.mkdir()
        monkeypatch.setattr(
            "agent13.persistence.get_global_saves_dir", lambda: global_dir
        )
        get_saves_dir()
        assert list_all_saves() == []


# ---------------------------------------------------------------------------
# save_context / load_context round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadContext:
    def test_round_trip(self, monkeypatch, tmp_path):
        """Save then load restores messages."""
        path = tmp_path / "test.ctx"
        agent = _make_agent_stub(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            model="gpt-4",
            system_prompt="be nice",
        )
        save_context(agent, path)
        assert path.exists()

        agent2 = _make_agent_stub()
        ok, msg, incomplete = load_context(agent2, path)
        assert ok is True
        assert incomplete is False
        assert agent2.messages == agent.messages
        assert agent2.system_prompt == "be nice"

    def test_session_date_round_trip(self, tmp_path):
        """Save then load restores session_date."""
        path = tmp_path / "test.ctx"
        agent = _make_agent_stub(
            messages=[{"role": "user", "content": "hello"}],
        )
        agent.session_date = "2026-01-15"
        save_context(agent, path)

        agent2 = _make_agent_stub()
        ok, msg, _ = load_context(agent2, path)
        assert ok is True
        assert agent2.session_date == "2026-01-15"

    def test_session_date_missing_in_old_save(self, tmp_path):
        """Old save files without session_date don't break load."""
        import json

        path = tmp_path / "old.ctx"
        path.write_text(json.dumps({
            "version": 1,
            "model": "test",
            "system_prompt": "",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        agent = _make_agent_stub()
        agent.session_date = "2026-07-29"
        ok, msg, _ = load_context(agent, path)
        assert ok is True
        # session_date should remain unchanged (not overwritten with None)
        assert agent.session_date == "2026-07-29"

    def test_load_missing_file(self, tmp_path):
        agent = _make_agent_stub()
        ok, msg, incomplete = load_context(agent, tmp_path / "nope.ctx")
        assert ok is False
        assert "not found" in msg

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.ctx"
        path.write_text("not json{{{")
        agent = _make_agent_stub()
        ok, msg, _ = load_context(agent, path)
        assert ok is False
        assert "Invalid JSON" in msg

    def test_load_future_version(self, tmp_path):
        path = tmp_path / "future.ctx"
        path.write_text(json.dumps({"version": 999, "messages": []}))
        agent = _make_agent_stub()
        ok, msg, _ = load_context(agent, path)
        assert ok is False
        assert "newer than supported" in msg

    def test_load_missing_messages_field(self, tmp_path):
        path = tmp_path / "nomsg.ctx"
        path.write_text(json.dumps({"version": 1}))
        agent = _make_agent_stub()
        ok, msg, _ = load_context(agent, path)
        assert ok is False
        assert "missing" in msg

    def test_incomplete_turn_flag(self, tmp_path):
        path = tmp_path / "inc.ctx"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                    "incomplete_turn": True,
                }
            )
        )
        agent = _make_agent_stub()
        ok, _, incomplete = load_context(agent, path)
        assert ok is True
        assert incomplete is True
        assert agent._incomplete is True

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "test.ctx"
        agent = _make_agent_stub(messages=[{"role": "user", "content": "test"}])
        save_context(agent, path)
        assert path.exists()

    def test_token_usage_round_trip(self, tmp_path):
        path = tmp_path / "tokens.ctx"
        agent = _make_agent_stub()
        agent.prompt_tokens = 1234
        agent.completion_tokens = 5678
        save_context(agent, path)

        agent2 = _make_agent_stub()
        ok, _, _ = load_context(agent2, path)
        assert ok is True
        assert agent2.prompt_tokens == 1234
        assert agent2.completion_tokens == 5678


# ---------------------------------------------------------------------------
# Config: saves_location
# ---------------------------------------------------------------------------


class TestSavesLocationConfig:
    def test_default_is_local(self):
        config = Config()
        assert config.saves_location == "local"

    def test_from_file_local(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[[providers]]\nname = "test"\napi_base = "http://localhost:8012/v1"\n\n'
            '[saves]\nlocation = "local"\n'
        )
        config = Config.from_file(config_file)
        assert config.saves_location == "local"

    def test_from_file_central(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[[providers]]\nname = "test"\napi_base = "http://localhost:8012/v1"\n\n'
            '[saves]\nlocation = "central"\n'
        )
        config = Config.from_file(config_file)
        assert config.saves_location == "central"

    def test_no_section_uses_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[[providers]]\nname = "test"\napi_base = "http://localhost:8012/v1"\n'
        )
        config = Config.from_file(config_file)
        assert config.saves_location == "local"

    def test_invalid_value_keeps_default(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[[providers]]\nname = "test"\napi_base = "http://localhost:8012/v1"\n\n'
            '[saves]\nlocation = "invalid"\n'
        )
        config = Config.from_file(config_file)
        assert config.saves_location == "local"


# ---------------------------------------------------------------------------
# _ensure_ctx_stem
# ---------------------------------------------------------------------------


class TestEnsureCtxStem:
    """Tests for _ensure_ctx_stem — prevents double .ctx extension."""

    def test_strips_ctx_suffix(self):
        assert _ensure_ctx_stem("mywork.ctx") == "mywork"

    def test_passes_through_without_suffix(self):
        assert _ensure_ctx_stem("mywork") == "mywork"

    def test_only_strips_trailing_ctx(self):
        assert _ensure_ctx_stem("ctx") == "ctx"

    def test_empty_string(self):
        assert _ensure_ctx_stem("") == ""

    def test_double_ctx_gets_one_stripped(self):
        """If somehow passed 'foo.ctx.ctx', strips one layer."""
        assert _ensure_ctx_stem("foo.ctx.ctx") == "foo.ctx"


# ---------------------------------------------------------------------------
# resolve_save_path
# ---------------------------------------------------------------------------


class TestResolveSavePath:
    """Tests for resolve_save_path — spaces, tildes, absolute paths."""

    def test_bare_name(self, monkeypatch, tmp_path):
        """Bare name joins with saves dir and adds .ctx."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("mycontext")
        assert path == tmp_path / "saves" / "mycontext.ctx"

    def test_name_with_spaces(self, monkeypatch, tmp_path):
        """Names with spaces are preserved, not split."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("my context name")
        assert path == tmp_path / "saves" / "my context name.ctx"

    def test_strips_ctx_suffix(self, monkeypatch, tmp_path):
        """Existing .ctx suffix is not doubled."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("mycontext.ctx")
        assert path.name == "mycontext.ctx"
        assert path.name != "mycontext.ctx.ctx"

    def test_absolute_path(self, monkeypatch, tmp_path):
        """Absolute paths are used directly, not joined to saves dir."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("/tmp/foo/bar")
        assert path == Path("/tmp/foo/bar.ctx")

    def test_absolute_path_preserves_existing_extension(self, monkeypatch, tmp_path):
        """Absolute path with a non-.ctx extension keeps it and appends .ctx.

        Regression: with_suffix() replaced the extension, so
        /tmp/session.backup silently became /tmp/session.ctx.
        Bare names preserve the extension via _ensure_ctx_stem; absolute
        paths must behave the same way.
        """
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("/tmp/session.backup")
        assert path == Path("/tmp/session.backup.ctx")
        assert path.name == "session.backup.ctx"

    def test_absolute_path_with_ctx(self, monkeypatch, tmp_path):
        """Absolute path with .ctx suffix is preserved."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        path = resolve_save_path("/tmp/foo/bar.ctx")
        assert path == Path("/tmp/foo/bar.ctx")

    def test_tilde_expanded(self, monkeypatch, tmp_path):
        """Tilde paths are expanded."""
        monkeypatch.setenv("AGENT13_SAVES_DIR", str(tmp_path / "saves"))
        home = str(tmp_path / "home")
        # On Windows, Path.expanduser() prefers USERPROFILE over HOME, so set
        # both to keep the test deterministic across platforms.
        monkeypatch.setenv("HOME", home)
        monkeypatch.setenv("USERPROFILE", home)
        path = resolve_save_path("~/sessions/foo")
        assert str(path).startswith(home)
        assert path.name == "foo.ctx"

    def test_empty_raises(self):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError):
            resolve_save_path("")
