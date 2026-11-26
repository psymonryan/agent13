"""Tests for the default-skills hash-manifest sync (skills/defaults.py).

The sync must never clobber local edits: bundled skills install when
missing, update when the installed bytes still match the manifest (we
shipped them, the user didn't touch them), and are skipped with a report
otherwise. /skills refresh re-runs it; -y overwrites modified copies.
"""

import pytest

from agent13.skills.defaults import (
    DefaultSkillsReport,
    SyncPaths,
    dir_hash,
    load_manifest,
    plan_sync,
    read_version,
    save_manifest,
    sync_default_skills,
)
from agent13.skills.manager import SkillManager
from agent13.commands import execute_skills


def make_skill(root, name, version="1.0.0", body="original body", extra_file=None):
    """Write a skill dir under root and return its path."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = f"""---
name: {name}
description: test skill {name}
metadata:
  version: {version}
---

{body}
"""
    (d / "SKILL.md").write_text(fm, encoding="utf-8")
    if extra_file:
        (d / extra_file).write_text("extra", encoding="utf-8")
    return d


def make_paths(tmp_path):
    """SyncPaths pointing entirely at tmp dirs."""
    return SyncPaths(
        skills_dir=tmp_path / "skills",
        bundled_dir=tmp_path / "bundled",
        manifest_path=tmp_path / "config" / ".skills-defaults.json",
    )


class TestDirHash:
    def test_deterministic_and_content_sensitive(self, tmp_path):
        a = make_skill(tmp_path, "s1")
        b = make_skill(tmp_path / "copy", "s1")
        assert dir_hash(a) == dir_hash(b)
        (b / "SKILL.md").write_text("changed", encoding="utf-8")
        assert dir_hash(a) != dir_hash(b)

    def test_ignores_platform_noise(self, tmp_path):
        a = make_skill(tmp_path, "s1")
        h1 = dir_hash(a)
        (a / ".DS_Store").write_bytes(b"junk")
        cache = a / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_bytes(b"junk")
        assert dir_hash(a) == h1


class TestReadVersion:
    def test_metadata_version(self, tmp_path):
        d = make_skill(tmp_path, "s1", version="2.3.1")
        assert read_version(d) == "2.3.1"

    def test_missing_version_reads_empty(self, tmp_path):
        d = tmp_path / "s1"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: s1\ndescription: x\n---\nbody\n")
        assert read_version(d) == ""

    def test_unparseable_reads_empty(self, tmp_path):
        d = tmp_path / "s1"
        d.mkdir()
        (d / "SKILL.md").write_text("no frontmatter at all")
        assert read_version(d) == ""


class TestPlanSync:
    """The pure decision table."""

    BUNDLED = {"s": {"hash": "H2", "version": "2.0"}}

    def test_not_installed_copies(self):
        assert plan_sync(self.BUNDLED, {}, {}) == {"s": "copy"}

    def test_tracked_and_changed_updates(self):
        manifest = {"s": {"hash": "H1", "version": "1.0"}}
        installed = {"s": "H1"}
        assert plan_sync(self.BUNDLED, manifest, installed) == {"s": "update"}

    def test_tracked_but_locally_modified_skips(self):
        manifest = {"s": {"hash": "H1", "version": "1.0"}}
        installed = {"s": "HUSER"}  # user edited it
        assert plan_sync(self.BUNDLED, manifest, installed) == {"s": "skip_modified"}

    def test_legacy_identical_adopts(self):
        installed = {"s": "H2"}  # matches the BUNDLE, no manifest entry
        assert plan_sync(self.BUNDLED, {}, installed) == {"s": "adopt"}

    def test_legacy_different_skips(self):
        installed = {"s": "HOLD"}  # an old bundled version, no manifest entry
        assert plan_sync(self.BUNDLED, {}, installed) == {"s": "skip_modified"}

    def test_in_sync_is_ok(self):
        manifest = {"s": {"hash": "H2", "version": "2.0"}}
        installed = {"s": "H2"}
        assert plan_sync(self.BUNDLED, manifest, installed) == {"s": "ok"}


class TestSync:
    def test_empty_dir_installs_everything(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha")
        make_skill(p.bundled_dir, "beta", version="0.9.0")

        report = sync_default_skills(p)

        assert report.copied == ["alpha", "beta"]
        assert (p.skills_dir / "alpha" / "SKILL.md").is_file()
        manifest = load_manifest(p.manifest_path)
        assert set(manifest) == {"alpha", "beta"}
        assert manifest["beta"]["version"] == "0.9.0"
        assert manifest["alpha"]["hash"] == dir_hash(p.bundled_dir / "alpha")

    def test_missing_skills_dir_is_created(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha")
        report = sync_default_skills(p)
        assert report.copied == ["alpha"]
        assert (p.skills_dir / "alpha" / "SKILL.md").is_file()

    def test_user_skills_untouched_and_new_default_still_installs(self, tmp_path):
        """The old seed-once rule is gone: one existing (non-bundled) skill
        must not block a new default from installing."""
        p = make_paths(tmp_path)
        make_skill(p.skills_dir, "users-own")
        make_skill(p.bundled_dir, "alpha")

        report = sync_default_skills(p)

        assert report.copied == ["alpha"]
        assert (p.skills_dir / "users-own" / "SKILL.md").is_file()
        assert set(load_manifest(p.manifest_path)) == {"alpha"}

    def test_bundled_change_updates_untouched_install(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)  # first install

        # Bundle ships a new version (content + version bump)
        make_skill(p.bundled_dir, "alpha", version="1.1.0", body="new body")

        report = sync_default_skills(p)

        assert report.updated == [("alpha", "1.0.0", "1.1.0")]
        assert "new body" in (p.skills_dir / "alpha" / "SKILL.md").read_text()
        assert load_manifest(p.manifest_path)["alpha"]["version"] == "1.1.0"

    def test_update_removes_stale_files(self, tmp_path):
        """An update wipes the old copy, so files the new version dropped
        don't linger."""
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", extra_file="stale.txt")
        sync_default_skills(p)
        assert (p.skills_dir / "alpha" / "stale.txt").exists()

        # v2 drops the file the bundle shipped in v1
        make_skill(p.bundled_dir, "alpha", version="1.1.0", body="new body")
        (p.bundled_dir / "alpha" / "stale.txt").unlink()
        sync_default_skills(p)

        assert not (p.skills_dir / "alpha" / "stale.txt").exists()

    def test_locally_modified_skill_is_never_touched(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)

        # User edits their copy; bundle ships an update
        (p.skills_dir / "alpha" / "SKILL.md").write_text("my tweaks")
        make_skill(p.bundled_dir, "alpha", version="2.0.0", body="shipped")

        report = sync_default_skills(p)

        assert report.skipped_modified == ["alpha"]
        assert report.updated == []
        assert (p.skills_dir / "alpha" / "SKILL.md").read_text() == "my tweaks"
        assert load_manifest(p.manifest_path)["alpha"]["version"] == "1.0.0"

    def test_legacy_identical_install_is_adopted_not_copied(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        # Pre-manifest install: user copied the bundle by hand
        make_skill(p.skills_dir, "alpha", version="1.0.0")

        report = sync_default_skills(p)

        assert report.adopted == ["alpha"]
        assert report.copied == []
        assert load_manifest(p.manifest_path)["alpha"]["version"] == "1.0.0"

    def test_legacy_different_install_is_reported_not_touched(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="2.0.0", body="shipped v2")
        # Pre-manifest install of an OLDER bundled version
        make_skill(p.skills_dir, "alpha", version="1.0.0", body="old v1")

        report = sync_default_skills(p)

        assert report.skipped_modified == ["alpha"]
        assert "old v1" in (p.skills_dir / "alpha" / "SKILL.md").read_text()

    def test_corrupt_manifest_reads_as_empty(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        make_skill(p.skills_dir, "alpha", version="1.0.0")  # identical
        p.manifest_path.parent.mkdir(parents=True)
        p.manifest_path.write_text("{not json", encoding="utf-8")

        report = sync_default_skills(p)

        assert report.adopted == ["alpha"]  # legacy rules, no crash

    def test_manifest_entry_for_removed_default_is_preserved(self, tmp_path):
        """A default that leaves the bundle stays installed and tracked;
        if it ever returns, the user's untouched copy can still be updated."""
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha")
        sync_default_skills(p)
        # Bundle drops alpha
        (p.bundled_dir / "alpha").rename(tmp_path / "alpha-gone")

        report = sync_default_skills(p)

        assert report == DefaultSkillsReport()
        assert (p.skills_dir / "alpha" / "SKILL.md").is_file()
        assert set(load_manifest(p.manifest_path)) == {"alpha"}

    def test_force_overwrites_modified_skill(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)
        (p.skills_dir / "alpha" / "SKILL.md").write_text("my tweaks")
        make_skill(p.bundled_dir, "alpha", version="2.0.0", body="shipped")

        report = sync_default_skills(p, force=True)

        assert report.updated == [("alpha", "1.0.0", "2.0.0")]
        assert "shipped" in (p.skills_dir / "alpha" / "SKILL.md").read_text()
        assert load_manifest(p.manifest_path)["alpha"]["version"] == "2.0.0"

    def test_second_run_is_silent(self, tmp_path):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha")
        sync_default_skills(p)
        assert sync_default_skills(p) == DefaultSkillsReport()

    def test_no_bundled_dir_is_a_noop(self, tmp_path):
        p = make_paths(tmp_path)  # bundled_dir never created
        assert sync_default_skills(p) == DefaultSkillsReport()
        assert not p.manifest_path.exists()


class TestRender:
    def test_empty_renders_silent(self):
        assert DefaultSkillsReport().render() == ""

    def test_skipped_line_names_the_escape_hatch(self):
        r = DefaultSkillsReport(skipped_modified=["a", "b"])
        text = r.render()
        assert "a, b" in text
        assert "/skills refresh -y" in text

    def test_updated_line_shows_versions(self):
        r = DefaultSkillsReport(updated=[("retro", "1.0.0", "1.1.0")])
        assert "retro (1.0.0 -> 1.1.0)" in r.render()


class TestSkillsRefreshCommand:
    """'/skills refresh' through the shared command handler."""

    def _agent_with_manager(self, tmp_path):
        from types import SimpleNamespace

        def mock_config():
            cfg = SimpleNamespace()
            cfg.skill_paths = [tmp_path / "skills"]
            cfg.include_skills = False
            return cfg

        sm = SkillManager(mock_config)
        return SimpleNamespace(skill_manager=sm)

    def _run_refresh(self, monkeypatch, tmp_path, args):
        p = make_paths(tmp_path)
        monkeypatch.setattr("agent13.skills.defaults.default_paths", lambda: p)
        agent = self._agent_with_manager(tmp_path)
        result = execute_skills(agent, args)
        return result, p, agent

    def test_refresh_updates_and_reports(self, tmp_path, monkeypatch):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)
        make_skill(p.bundled_dir, "alpha", version="1.1.0", body="new body")

        result, p, agent = self._run_refresh(monkeypatch, tmp_path, "refresh")

        assert result.success
        assert "alpha (1.0.0 -> 1.1.0)" in result.message
        assert "new body" in (p.skills_dir / "alpha" / "SKILL.md").read_text()

    def test_rescan_makes_new_content_visible_to_the_session(
        self, tmp_path, monkeypatch
    ):
        """The manager caches at construction; refresh must rescan so the
        running session serves the refreshed content without a restart."""
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)
        agent = self._agent_with_manager(tmp_path)
        assert "original body" in agent.skill_manager.load_skill_content("alpha")

        make_skill(p.bundled_dir, "alpha", version="1.1.0", body="new body")
        monkeypatch.setattr("agent13.skills.defaults.default_paths", lambda: p)
        execute_skills(agent, "refresh")

        assert "new body" in agent.skill_manager.load_skill_content("alpha")

    def test_refresh_without_y_warns_about_modified(self, tmp_path, monkeypatch):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)
        (p.skills_dir / "alpha" / "SKILL.md").write_text("my tweaks")
        make_skill(p.bundled_dir, "alpha", version="2.0.0", body="shipped")

        result, p, _ = self._run_refresh(monkeypatch, tmp_path, "refresh")

        assert result.success
        assert "alpha" in result.message
        assert "-y" in result.message
        assert (p.skills_dir / "alpha" / "SKILL.md").read_text() == "my tweaks"

    def test_refresh_with_y_overwrites_modified(self, tmp_path, monkeypatch):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha", version="1.0.0")
        sync_default_skills(p)
        (p.skills_dir / "alpha" / "SKILL.md").write_text("my tweaks")
        make_skill(p.bundled_dir, "alpha", version="2.0.0", body="shipped")

        result, p, _ = self._run_refresh(monkeypatch, tmp_path, "refresh -y")

        assert result.success
        assert "shipped" in (p.skills_dir / "alpha" / "SKILL.md").read_text()

    def test_refresh_when_in_sync_says_so(self, tmp_path, monkeypatch):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "alpha")
        sync_default_skills(p)

        result, _, _ = self._run_refresh(monkeypatch, tmp_path, "refresh")

        assert result.success
        assert "up to date" in result.message

    def test_refresh_installs_brand_new_default(self, tmp_path, monkeypatch):
        p = make_paths(tmp_path)
        make_skill(p.bundled_dir, "fresh-skill")

        result, p, _ = self._run_refresh(monkeypatch, tmp_path, "refresh")

        assert result.success
        assert "fresh-skill" in result.message
        assert (p.skills_dir / "fresh-skill" / "SKILL.md").is_file()

    def test_usage_still_lists_refresh(self):
        from agent13.commands import _SKILLS_USAGE

        assert "refresh" in _SKILLS_USAGE


class TestManifestRoundTrip:
    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "m.json"
        manifest = {"a": {"hash": "H1", "version": "1.0.0"}}
        save_manifest(path, manifest)
        assert load_manifest(path) == manifest

    def test_load_missing_is_empty(self, tmp_path):
        assert load_manifest(tmp_path / "nope.json") == {}

    def test_load_non_object_is_empty(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_manifest(path) == {}


@pytest.mark.parametrize(
    "report,expect_empty",
    [
        (DefaultSkillsReport(), True),
        (DefaultSkillsReport(copied=["a"]), False),
        (DefaultSkillsReport(adopted=["a"]), False),
        (DefaultSkillsReport(skipped_modified=["a"]), False),
    ],
)
def test_report_truthiness(report, expect_empty):
    assert bool(report) is not expect_empty
