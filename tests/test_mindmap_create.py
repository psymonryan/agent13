"""Wiring tests for ``create_map``'s codec half and the check module.

The archive rules are asserted on the BYTES - entry order, compression, dates -
because those are what the original app is sensitive to, and none of them is a
default: a bare ``ZipInfo`` is STORED and stamped 1980-01-01.
"""

import hashlib
import plistlib
import time
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from agent13.mindmap import (
    DEFAULT_STYLE_PATH,
    MAPDATA,
    STYLE,
    MindMap,
    MindMapError,
    STORE,
    default_style,
    read_style,
)
from agent13.mindmap_check import check
from tests.test_mindmap import _child_xml, _topic, _uuid, make_map, simple_map


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create(path: Path, **kwargs) -> MindMap:
    mind_map = MindMap.create(path, **kwargs)
    mind_map.save(create=True)
    return mind_map


def _topic_no_position(text: str, uuid: str, **attrs) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return (
        f'<topic uuid="{uuid}" text="{text}" '
        f'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"{extra}/>'
    )


@pytest.fixture(autouse=True)
def _clean_state():
    STORE.forget_all()
    yield
    STORE.forget_all()


# ── The created archive ─────────────────────────────────────────────────────


class TestCreatedArchive:
    def test_two_entries_in_order(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        with zipfile.ZipFile(path) as zf:
            assert [i.filename for i in zf.infolist()] == [MAPDATA, STYLE]

    def test_no_directory_entries(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        with zipfile.ZipFile(path) as zf:
            assert not any(info.is_dir() for info in zf.infolist())

    def test_both_entries_are_deflated(self, tmp_path):
        """A fresh ZipInfo is STORED, so this fails unless it is set explicitly."""
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        with zipfile.ZipFile(path) as zf:
            assert [i.compress_type for i in zf.infolist()] == [
                zipfile.ZIP_DEFLATED,
                zipfile.ZIP_DEFLATED,
            ]

    def test_entries_are_stamped_now(self, tmp_path):
        """A fresh ZipInfo is stamped 1980-01-01, which becomes the archive's date."""
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        with zipfile.ZipFile(path) as zf:
            assert all(i.date_time[0] == time.localtime().tm_year for i in zf.infolist())

    def test_style_is_byte_identical_to_the_default(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        with zipfile.ZipFile(path) as zf:
            assert zf.read(STYLE) == DEFAULT_STYLE_PATH.read_bytes()

    def test_header_matches_our_app(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path)
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(MAPDATA).decode("utf-8")
        assert raw.startswith('<iThoughts version="4.0" app="com.ithoughtsx.web"')
        assert 'app-version="1.0"' in raw
        assert 'modified="' in raw
        assert "<?xml" not in raw  # our app writes no declaration

    def test_root_topic(self, tmp_path):
        path = tmp_path / "x.itmz"
        mind_map = _create(path, root_text="IPSec FY27")
        root = mind_map.root_topic
        assert root.get("text") == "IPSec FY27"
        assert root.get("position") == "{0, 0}"
        assert root.get("uuid") == root.get("uuid").upper()
        assert root.get("created") and root.get("modified")

    def test_relationships_present_and_empty(self, tmp_path):
        path = tmp_path / "x.itmz"
        mind_map = _create(path)
        relationships = mind_map.root.find("relationships")
        assert relationships is not None
        assert len(relationships) == 0

    def test_no_manifest_preview_or_display_state(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        assert names == {MAPDATA, STYLE}


class TestRefusesToOverwrite:
    def test_existing_path_is_refused(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path)
        before = _sha(path)
        with pytest.raises(MindMapError, match="already exists"):
            MindMap.create(path).save(create=True)
        assert _sha(path) == before

    def test_error_names_the_alternative(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path)
        with pytest.raises(MindMapError) as excinfo:
            MindMap.create(path).save(create=True)
        assert "edit_map" in str(excinfo.value)
        assert "never overwrites" in str(excinfo.value)


class TestSaveCreate:
    def test_plain_save_still_raises_on_a_missing_path(self, tmp_path):
        """The reason the parameter exists: version_of reads the file."""
        path = tmp_path / "x.itmz"
        with pytest.raises(FileNotFoundError):
            MindMap.create(path).save()

    def test_create_registers_the_baseline(self, tmp_path):
        path = tmp_path / "x.itmz"
        mind_map = _create(path)
        assert STORE.baseline(path) == mind_map.version

    def test_create_then_edit_immediately(self, tmp_path):
        """The trap: a 0-second-old file is inside the quiet window."""
        path = tmp_path / "x.itmz"
        _create(path)
        opened = MindMap.open(path)
        opened.apply_ops(
            [{"op": "rename", "topic": opened.address_of(opened.root_topic), "text": "Renamed"}]
        )
        new_version = opened.save(expect_short=opened.version.short, quiet_seconds=20)
        assert new_version.short != ""


# ── Outline ─────────────────────────────────────────────────────────────────


class TestOutline:
    def test_outline_becomes_children_of_the_root(self, tmp_path):
        path = tmp_path / "x.itmz"
        mind_map = MindMap.create(path)
        created = mind_map.add_outline(mind_map.root_topic, "a\n\tb\n\tc")
        assert [el.get("text") for el in created] == ["a", "b", "c"]
        assert mind_map.children(mind_map.root_topic)[0].get("text") == "a"
        assert [el.get("text") for el in mind_map.children(created[0])] == ["b", "c"]

    def test_addresses_resolve_after_a_reopen(self, tmp_path):
        path = tmp_path / "x.itmz"
        mind_map = MindMap.create(path)
        created = mind_map.add_outline(mind_map.root_topic, "a\n\tb")
        addresses = [mind_map.address_of(el) for el in created]
        mind_map.save(create=True)

        reopened = MindMap.open(path)
        assert [reopened.resolve(addr).get("text") for addr in addresses] == ["a", "b"]

    def test_empty_outline_is_refused_like_add_branch(self, tmp_path):
        """One outline dialect: parse_outline rejects empty text, so this does too.

        create_map guards with ``outline.strip()``, so an omitted outline is a
        no-op rather than an error.
        """
        mind_map = MindMap.create(tmp_path / "x.itmz")
        with pytest.raises(MindMapError, match="outline is empty"):
            mind_map.add_outline(mind_map.root_topic, "")


# ── Style templates ─────────────────────────────────────────────────────────


class TestStyleTemplates:
    def test_default_style_is_the_bundled_file(self):
        assert DEFAULT_STYLE_PATH.exists()
        assert default_style() == DEFAULT_STYLE_PATH.read_bytes()
        assert b'name="Outline"' in default_style()

    def test_read_style_lifts_the_member(self, tmp_path):
        source = make_map(tmp_path / "src.itmz", "", extra=[(STYLE, b"<style>mine</style>")])
        assert read_style(source) == b"<style>mine</style>"

    def test_missing_template_names_the_default(self, tmp_path):
        with pytest.raises(MindMapError) as excinfo:
            read_style(tmp_path / "nope.itmz")
        assert "Omit style_from" in str(excinfo.value)

    def test_not_a_zip(self, tmp_path):
        bad = tmp_path / "x.itmz"
        bad.write_bytes(b"nope")
        with pytest.raises(MindMapError, match="not a readable .itmz"):
            read_style(bad)

    def test_no_style_member(self, tmp_path):
        plain = make_map(tmp_path / "plain.itmz", "")
        with pytest.raises(MindMapError, match="no style.xml"):
            read_style(plain)

    def test_reads_only_the_style_member(self, tmp_path):
        """The 49 MB trap: opening the template would parse its whole mapdata."""
        source = make_map(
            tmp_path / "big.itmz",
            '<topic uuid="%s" position="{0, 0}" text="Root"/>' % _uuid(1),
            extra=[(STYLE, b"<style>mine</style>")],
        )
        calls = []
        real_read = zipfile.ZipFile.read

        def spy(self, name, *args, **kwargs):
            calls.append(name)
            return real_read(self, name, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "read", spy), mock.patch.object(
            MindMap, "open", side_effect=AssertionError("parsed the template")
        ):
            assert read_style(source) == b"<style>mine</style>"
        assert calls == [STYLE]


# ── the check's rules ───────────────────────────────────────────────────────


class TestCheckGoodMap:
    def test_created_map_is_ok(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        report = check(path)
        assert report.ok
        assert report.errors == []
        assert report.warnings == []
        assert report.topic_count == 1
        assert report.has_style
        assert "OK (1 topic, style.xml present)" in report.render()

    def test_existing_map_without_style_warns(self, tmp_path):
        report = check(simple_map(tmp_path / "m.itmz"))
        assert report.ok
        assert any("no style.xml" in w for w in report.warnings)
        assert "OK with 1 warning" in report.render()


class TestCheckErrors:
    def test_no_position(self, tmp_path):
        path = make_map(tmp_path / "m.itmz", _topic_no_position("Loose", _uuid(1)))
        report = check(path)
        assert not report.ok
        assert any("has no position" in e for e in report.errors)
        # DoD 9: the message says what the app DOES with a missing position,
        # not just that it is missing.
        assert any("treats a missing position as {0, 0}" in e for e in report.errors)
        assert _uuid(1)[:8] in report.render()

    def test_malformed_position(self, tmp_path):
        xml = (
            f'<topic uuid="{_uuid(1)}" position="middle" text="Odd" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml))
        assert any("malformed position" in e for e in report.errors)

    def test_malformed_uuid(self, tmp_path):
        xml = (
            '<topic uuid="NOT-A-UUID" position="{0, 0}" text="Odd" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml))
        assert any("malformed uuid" in e for e in report.errors)

    def test_no_uuid(self, tmp_path):
        xml = (
            '<topic position="{0, 0}" text="Nameless" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml))
        assert any("has no uuid" in e for e in report.errors)

    def test_duplicate_uuid(self, tmp_path):
        report = check(
            make_map(tmp_path / "m.itmz", _topic("One", _uuid(1)) + _topic("Two", _uuid(1)))
        )
        assert any("duplicate uuid" in e for e in report.errors)
        assert any("2 topics" in e for e in report.errors)

    def test_not_a_zip(self, tmp_path):
        bad = tmp_path / "x.itmz"
        bad.write_bytes(b"nope")
        report = check(bad)
        assert any("not a ZIP archive" in e for e in report.errors)

    def test_no_mapdata(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("other.txt", b"hi")
        assert any("no mapdata.xml" in e for e in check(path).errors)

    def test_malformed_xml(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(MAPDATA, b"<iThoughts><topics>")
        assert any("not well-formed" in e for e in check(path).errors)

    def test_wrong_root(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(MAPDATA, b"<other><topics/></other>")
        assert any("expected <iThoughts>" in e for e in check(path).errors)

    def test_no_topics_element(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(MAPDATA, b'<iThoughts version="4.0" app="x" modified="y"/>')
        assert any("no <topics>" in e for e in check(path).errors)

    def test_missing_file_is_an_error_not_a_raise(self, tmp_path):
        report = check(tmp_path / "nope.itmz")
        assert not report.ok
        assert "cannot read the file" in report.render()


class TestCheckWarningsAndNotes:
    def test_missing_header_attributes(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                MAPDATA,
                f"<iThoughts><topics>{_topic('R', _uuid(1))}</topics></iThoughts>",
            )
        report = check(path)
        assert any("header has no version, app, modified" in w for w in report.warnings)

    def test_stale_manifest_warns_but_stays_ok(self, tmp_path):
        manifest = plistlib.dumps({"TopicCount": 9})
        path = make_map(
            tmp_path / "m.itmz",
            _topic("R", _uuid(1)),
            extra=[("manifest.plist", manifest)],
        )
        report = check(path)
        assert report.ok
        assert any("manifest TopicCount 9" in w for w in report.warnings)

    def test_directory_entries_warn(self, tmp_path):
        path = make_map(
            tmp_path / "m.itmz", _topic("R", _uuid(1)), extra=[("assets/", b"")]
        )
        assert any("directory entr" in w for w in check(path).warnings)

    def test_lowercase_uuids_are_a_note_not_a_warning(self, tmp_path):
        xml = (
            f'<topic uuid="{_uuid(0xAB).lower()}" position="{{0, 0}}" text="R" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml, extra=[(STYLE, b"<style/>")]))
        assert report.ok
        assert report.warnings == []
        assert any("lowercase" in n for n in report.notes)
        assert "OK (" in report.render()

    def test_child_order_disagreeing_with_position_warns(self, tmp_path):
        """DoD 8: the rule that would have caught the reported bug - a topic
        moved in XML, drawn where it used to be."""
        kids = (
            _child_xml("A", _uuid(10), 120)
            + _child_xml("B", _uuid(11), 0)
            + _child_xml("C", _uuid(12), 60)
        )
        report = check(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        # A WARNING: the file opens, it just does not look like what it says.
        assert report.ok
        assert any("child order disagrees with the position order" in w for w in report.warnings)
        assert any(w.startswith("1 single-column parent where") for w in report.warnings)

    def test_two_sided_parent_is_silent(self, tmp_path):
        """DoD 8, third clause: in a two-sided layout x picks the column and y
        only orders within it, so no rule about y can judge it.

        Modelled on Life.itmz's "Life": the right column ascends, the left
        descends. A rule that ignored x would call this a disagreement.
        """
        kids = (
            _child_xml("R1", _uuid(10), 0, x=80)
            + _child_xml("R2", _uuid(11), 60, x=80)
            + _child_xml("L1", _uuid(12), 60, x=-80)
            + _child_xml("L2", _uuid(13), 0, x=-80)
        )
        report = check(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        assert not any("disagrees" in w for w in report.warnings)

    def test_same_side_but_different_x_is_silent(self, tmp_path):
        """The predicate is exactly "one x", NOT "one x sign".

        Measured: the equal-x bucket (4,436 parents) has 0 disagreements, the
        same-sign bucket (9,588) has 2. Loosening the scope to the sign would
        re-admit those two, so the narrower predicate is the one to keep.
        """
        kids = _child_xml("A", _uuid(10), 120, x=80) + _child_xml("B", _uuid(11), 0, x=120)
        report = check(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        assert not any("disagrees" in w for w in report.warnings)

    def test_agreeing_child_order_is_silent(self, tmp_path):
        """DoD 8, second half: the app's own invariant, so it must not fire."""
        kids = (
            _child_xml("A", _uuid(10), 0)
            + _child_xml("B", _uuid(11), 60)
            + _child_xml("C", _uuid(12), 120)
        )
        report = check(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        assert not any("disagrees" in w for w in report.warnings)

    def test_equal_positions_are_not_a_disagreement(self, tmp_path):
        """Ties keep XML order, so a map of {0, 0} topics is consistent."""
        kids = _child_xml("A", _uuid(10), 0) + _child_xml("B", _uuid(11), 0)
        report = check(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        assert not any("disagrees" in w for w in report.warnings)

    def test_missing_asset_is_not_reported(self, tmp_path):
        """Cross-file consistency is itmz-check.py's job, not ours."""
        xml = (
            f'<topic uuid="{_uuid(1)}" position="{{0, 0}}" text="R" att-id="DEADBEEF" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml))
        assert report.ok
        assert not any("asset" in f for f in report.errors + report.warnings + report.notes)


# ── The trailer every tool appends ──────────────────────────────────────────


class TestTrailer:
    def test_clean_map_says_nothing(self, tmp_path):
        path = tmp_path / "x.itmz"
        _create(path, style=default_style())
        assert check(path).trailer() == ""

    def test_errors_and_warnings_are_listed_with_counts(self, tmp_path):
        path = make_map(tmp_path / "m.itmz", _topic_no_position("Loose", _uuid(1)))
        trailer = check(path).trailer()
        assert trailer.splitlines()[0] == "check: 1 error, 1 warning"
        assert "  error: topic 00000001 has no position" in trailer
        assert "  warning: no style.xml" in trailer

    def test_notes_are_excluded(self, tmp_path):
        """Lowercase uuids are ours; reporting them would be noise on most maps."""
        xml = (
            f'<topic uuid="{_uuid(0xAB).lower()}" position="{{0, 0}}" text="R" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        report = check(make_map(tmp_path / "m.itmz", xml, extra=[(STYLE, b"<style/>")]))
        assert report.notes
        assert report.trailer() == ""

    def test_capped_with_a_remainder_line(self, tmp_path):
        xml = "".join(_topic_no_position(f"t{n}", _uuid(n)) for n in range(1, 101))
        trailer = check(make_map(tmp_path / "m.itmz", xml)).trailer()
        assert trailer.count("  error: ") == 5
        assert "  ... and 95 more" in trailer

    def test_extra_line_only_when_there_are_findings(self, tmp_path):
        clean = _create(tmp_path / "clean.itmz", style=default_style())
        assert check(clean.path).trailer(extra="NOTE") == ""

        dirty = make_map(tmp_path / "dirty.itmz", _topic_no_position("L", _uuid(1)))
        assert check(dirty).trailer(extra="NOTE").endswith("  NOTE")
