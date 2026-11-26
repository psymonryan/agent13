"""Integration tests for the mind-map tools, through the real tool surface.

Only the LLM is absent - these call ``execute_tool`` exactly as the agent does,
so the schema, argument coercion, sandbox gate and error shapes are all real.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent13.mindmap import STORE
from tests.test_mindmap import _topic, _uuid, make_map, simple_map
from tools import execute_tool, get_tools

ITHOUGHTSX = Path.home() / "Resilio/work/src/ithoughtsx.com"
CHECKER = ITHOUGHTSX / "utils/itmz-check.py"
SAMPLE = ITHOUGHTSX / "samples/Life.itmz"


def _text(out) -> str:
    """Success returns a ToolResult (raw text); errors return a JSON string."""
    return out.text if hasattr(out, "text") else out


def _error(out) -> str:
    return json.loads(out)["error"] if isinstance(out, str) else out["error"]


@pytest.fixture(autouse=True)
def _clean_state():
    STORE.forget_all()
    yield
    STORE.forget_all()


@pytest.fixture
def map_path(tmp_path):
    return simple_map(tmp_path / "project.itmz")


# ── Schema ──────────────────────────────────────────────────────────────────


class TestSchema:
    def test_both_tools_are_discovered(self):
        names = {s["function"]["name"] for s in get_tools()}
        assert {"read_map", "edit_map"} <= names

    def test_read_map_schema(self):
        schema = next(s for s in get_tools() if s["function"]["name"] == "read_map")
        params = schema["function"]["parameters"]["properties"]
        assert set(params) == {"filepath", "topic", "depth", "query", "text_offset"}
        assert schema["function"]["parameters"]["required"] == ["filepath"]

    def test_edit_map_schema(self):
        schema = next(s for s in get_tools() if s["function"]["name"] == "edit_map")
        params = schema["function"]["parameters"]["properties"]
        assert set(params) == {"filepath", "ops", "expect_version"}
        assert params["ops"]["type"] == "array"

    def test_docstring_carries_the_carve_out(self):
        """The model needs the rule where it chooses, not only in the order."""
        schema = next(s for s in get_tools() if s["function"]["name"] == "read_map")
        description = schema["function"]["description"].lower()
        assert "structure" in description
        assert "diff" in description


# ── read_map ────────────────────────────────────────────────────────────────


class TestReadMapTool:
    async def test_returns_the_tree(self, map_path):
        out = _text(await execute_tool("read_map", {"filepath": str(map_path)}))
        assert "root: [" in out
        assert "text:\n  Root" in out
        assert "Alpha" in out

    async def test_drill_by_address(self, map_path):
        out = _text(
            await execute_tool(
                "read_map", {"filepath": str(map_path), "topic": _uuid(2)[:8]}
            )
        )
        assert f"topic: {_uuid(2)[:8]}" in out
        assert "text:\n  Alpha" in out
        assert "One" in out

    async def test_query(self, map_path):
        out = _text(
            await execute_tool("read_map", {"filepath": str(map_path), "query": "beta"})
        )
        assert "Root  (no children)" in out
        assert "    Beta     [drill for the full text]" in out

    async def test_missing_file_names_the_directory(self, tmp_path):
        out = await execute_tool("read_map", {"filepath": str(tmp_path / "nope.itmz")})
        assert str(tmp_path) in _error(out)

    async def test_not_a_map(self, tmp_path):
        bad = tmp_path / "x.itmz"
        bad.write_bytes(b"nope")
        out = await execute_tool("read_map", {"filepath": str(bad)})
        assert "not a .itmz archive" in _error(out)

    async def test_sandbox_refusal_is_surfaced(self, map_path):
        with patch(
            "tools.mindmap.validate_path_for_read",
            return_value=(False, "denied by sandbox"),
        ):
            out = await execute_tool("read_map", {"filepath": str(map_path)})
        assert _error(out) == "denied by sandbox"
        assert json.loads(out)["sandbox_mode"]

    async def test_text_offset_pages_through_the_tool(self, tmp_path):
        """The whole point of the work order: a long topic is quotable in full.

        The schema test proves the argument exists; this proves it is plumbed
        through to the codec and that the footer's offset is usable as given.
        """
        text = "\n".join(f"line {i:03d} " + "x" * 60 for i in range(360))  # two windows
        path = make_map(
            tmp_path / "long.itmz", _topic(text.replace("\n", "&#10;"), _uuid(2))
        )
        first = _text(
            await execute_tool(
                "read_map", {"filepath": str(path), "topic": _uuid(2)[:8], "depth": 0}
            )
        )
        offset = int(
            first.split("text continues at offset ")[1]
            .split(" of ")[0]
            .replace(",", "")
        )
        second = _text(
            await execute_tool(
                "read_map",
                {
                    "filepath": str(path),
                    "topic": _uuid(2)[:8],
                    "depth": 0,
                    "text_offset": offset,
                },
            )
        )
        assert "text continues at offset" not in second  # the tail fits
        # The block indents every line by two spaces; the text itself is intact.
        assert all(f"  {line}" in second for line in text[offset:].split("\n"))


# ── edit_map ────────────────────────────────────────────────────────────────


class TestEditMapTool:
    async def test_requires_a_baseline(self, map_path):
        out = await execute_tool(
            "edit_map",
            {
                "filepath": str(map_path),
                "ops": [{"op": "rename", "topic": _uuid(5)[:8], "text": "X"}],
            },
        )
        assert "call read_map" in _error(out).lower()
        assert "Nothing was written" in _error(out)

    async def test_read_then_edit_cycle(self, map_path):
        await execute_tool("read_map", {"filepath": str(map_path)})
        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(map_path),
                    "ops": [
                        {
                            "op": "add_branch",
                            "parent": _uuid(2)[:8],
                            "outline": "Docker\n\t-icc\nKubernetes",
                        }
                    ],
                },
            )
        )
        assert "wrote" in out
        assert "added 3 topics" in out
        assert "new version:" in out

        # The created addresses are in the result, so no re-read is needed.
        # Stop at the check trailer: its lines are indented too.
        report = out.split("\ncheck:")[0]
        addresses = [
            line.split()[0]
            for line in report.splitlines()
            if line.startswith("  ") and len(line.split()) > 1
        ]
        assert len(addresses) == 3

        after = _text(
            await execute_tool(
                "read_map", {"filepath": str(map_path), "topic": _uuid(2)[:8]}
            )
        )
        assert "Docker" in after and "Kubernetes" in after

    async def test_explicit_expect_version_survives_a_restart(self, map_path):
        """No in-process baseline: the version from read_map's output is enough."""
        first = _text(await execute_tool("read_map", {"filepath": str(map_path)}))
        version = next(
            line.split(":", 1)[1].strip()
            for line in first.splitlines()
            if line.startswith("version:")
        )
        STORE.forget_all()  # simulate a chain restart

        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(map_path),
                    "ops": [{"op": "rename", "topic": _uuid(5)[:8], "text": "Renamed"}],
                    "expect_version": version,
                },
            )
        )
        assert "wrote" in out

    async def test_stale_version_is_refused(self, map_path):
        first = _text(await execute_tool("read_map", {"filepath": str(map_path)}))
        stale = next(
            line.split(":", 1)[1].strip()
            for line in first.splitlines()
            if line.startswith("version:")
        )
        # Someone else changes the file.
        other = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(map_path),
                    "ops": [{"op": "rename", "topic": _uuid(6)[:8], "text": "Theirs"}],
                    "expect_version": stale,
                },
            )
        )
        assert "wrote" in other

        out = await execute_tool(
            "edit_map",
            {
                "filepath": str(map_path),
                "ops": [{"op": "rename", "topic": _uuid(5)[:8], "text": "Mine"}],
                "expect_version": stale,
            },
        )
        assert "changed since it was read" in _error(out)
        assert "Nothing was written" in _error(out)

    async def test_a_bad_op_writes_nothing(self, map_path):
        await execute_tool("read_map", {"filepath": str(map_path)})
        before = map_path.read_bytes()
        out = await execute_tool(
            "edit_map",
            {
                "filepath": str(map_path),
                "ops": [
                    {"op": "rename", "topic": _uuid(5)[:8], "text": "Changed"},
                    {"op": "nonsense"},
                ],
            },
        )
        assert "op 1" in _error(out)
        assert map_path.read_bytes() == before

    async def test_sandbox_refusal_is_surfaced(self, map_path):
        await execute_tool("read_map", {"filepath": str(map_path)})
        with patch(
            "tools.mindmap.validate_path_for_write", return_value=(False, "denied")
        ):
            out = await execute_tool(
                "edit_map",
                {
                    "filepath": str(map_path),
                    "ops": [{"op": "rename", "topic": _uuid(5)[:8], "text": "X"}],
                },
            )
        assert _error(out) == "denied"


# ── The real checker ────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (CHECKER.exists() and SAMPLE.exists()), reason="ithoughtsx samples not present"
)
class TestRealSample:
    """Round-trip a REAL map and run the real integrity checker on the result.

    The sample is copied to tmp_path first: the checker is the release gate in
    the ithoughtsx repo, and nothing here may write to it.
    """

    def test_round_trip_passes_the_checker(self, tmp_path):
        target = tmp_path / "Life.itmz"
        shutil.copy2(SAMPLE, target)
        old = time.time() - 3600
        os.utime(target, (old, old))

        from agent13.mindmap import MindMap

        mm = MindMap.open(target)
        parent = mm.children(mm.root_topic)[0]
        mm.apply_ops(
            [
                {
                    "op": "add_branch",
                    "parent": mm.address_of(parent),
                    "outline": "Added\n\tChild",
                }
            ]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        result = subprocess.run(
            [sys.executable, str(CHECKER), str(target)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
        assert "RESULT: clean" in result.stdout

    def test_every_entry_except_mapdata_is_byte_identical(self, tmp_path):
        target = tmp_path / "Life.itmz"
        shutil.copy2(SAMPLE, target)
        old = time.time() - 3600
        os.utime(target, (old, old))

        with zipfile.ZipFile(SAMPLE) as zf:
            before = [
                (i.filename, i.compress_type, zf.read(i.filename))
                for i in zf.infolist()
            ]

        from agent13.mindmap import MindMap

        mm = MindMap.open(target)
        mm.apply_ops(
            [{"op": "rename", "topic": mm.address_of(mm.root_topic), "text": "Life"}]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        with zipfile.ZipFile(target) as zf:
            after = [
                (i.filename, i.compress_type, zf.read(i.filename))
                for i in zf.infolist()
            ]

        assert [x[0] for x in before] == [x[0] for x in after]
        assert [x[1] for x in before] == [x[1] for x in after]
        differing = [x[0] for x, y in zip(before, after) if x[2] != y[2]]
        assert differing == ["mapdata.xml"]

    async def test_query_finds_a_note_only_match(self, tmp_path):
        """A real map's notes are searchable, and the hit says it was a note.

        The needle is DERIVED from the map, not hard-coded: a word that occurs
        in some topic's note and in NO topic's text. A literal would silently
        stop testing the note path the day the sample is refreshed and the word
        moves into a topic title.
        """
        from agent13.mindmap import MindMap

        mm = MindMap.open(SAMPLE)
        all_text = "\n".join((el.get("text") or "").lower() for el in mm._order)
        needle = address = None
        for el in mm._order:
            for word in sorted(
                set(re.findall(r"[a-z]{6,}", (el.get("note") or "").lower()))
            ):
                if word not in all_text:
                    needle, address = word, mm.address_of(el)
                    break
            if needle:
                break
        assert needle, "sample has no note-only word to search for"

        out = _text(
            await execute_tool("read_map", {"filepath": str(SAMPLE), "query": needle})
        )
        assert f"matches for {needle!r}" in out
        assert address in out
        # The note's own context is the indented line under the hit, not a
        # "note:" prefix - the hit line carries ancestors, the line below it
        # carries whichever field matched.
        assert any(
            line.startswith("    ") and needle in line for line in out.splitlines()
        ), "a note-only hit must show the note's context"


# ── create_map ──────────────────────────────────────────────────────────────


class _Cfg:
    """Stands in for Config: the tool reads one attribute."""

    def __init__(self, template: str):
        self.tools_mindmap_style_template = template


class TestCreateMapTool:
    async def test_creates_a_map_and_returns_addresses(self, tmp_path):
        path = tmp_path / "new.itmz"
        out = _text(
            await execute_tool(
                "create_map", {"filepath": str(path), "root_text": "IPSec FY27"}
            )
        )
        assert f"created {path}" in out
        assert "  root " in out
        assert "new version: " in out
        assert path.exists()

    async def test_outline_becomes_children_of_the_root(self, tmp_path):
        path = tmp_path / "new.itmz"
        out = _text(
            await execute_tool(
                "create_map", {"filepath": str(path), "outline": "Tunnels\n\tSite A"}
            )
        )
        tree = _text(
            await execute_tool("read_map", {"filepath": str(path), "depth": 2})
        )
        assert "Tunnels" in out and "Site A" in out
        assert "Tunnels" in tree and "Site A" in tree

    async def test_refuses_an_existing_path(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        before = path.read_bytes()
        out = await execute_tool("create_map", {"filepath": str(path)})
        assert "already exists" in _error(out)
        assert "edit_map" in _error(out)
        assert path.read_bytes() == before

    async def test_creates_missing_directories_and_says_so(self, tmp_path):
        path = tmp_path / "a" / "b" / "new.itmz"
        out = _text(await execute_tool("create_map", {"filepath": str(path)}))
        assert "created the missing directory" in out
        assert path.exists()

    async def test_create_then_edit_immediately(self, tmp_path):
        """The note_own_write regression: a 0-second-old file is in the quiet window."""
        path = tmp_path / "new.itmz"
        out = _text(
            await execute_tool(
                "create_map", {"filepath": str(path), "root_text": "Root"}
            )
        )
        address = out.split("  root ")[1].split()[0]
        edited = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(path),
                    "ops": [{"op": "rename", "topic": address, "text": "Renamed"}],
                },
            )
        )
        assert "renamed" in edited.lower()

    async def test_style_from_lifts_that_maps_style(self, tmp_path):
        source = make_map(
            tmp_path / "src.itmz", "", extra=[("style.xml", b"<style>mine</style>")]
        )
        path = tmp_path / "new.itmz"
        await execute_tool(
            "create_map", {"filepath": str(path), "style_from": str(source)}
        )
        with zipfile.ZipFile(path) as zf:
            assert zf.read("style.xml") == b"<style>mine</style>"

    async def test_style_from_a_map_without_style(self, tmp_path):
        source = simple_map(tmp_path / "src.itmz")
        out = await execute_tool(
            "create_map",
            {"filepath": str(tmp_path / "new.itmz"), "style_from": str(source)},
        )
        assert "no style.xml" in _error(out)

    async def test_style_from_a_missing_file(self, tmp_path):
        out = await execute_tool(
            "create_map",
            {
                "filepath": str(tmp_path / "new.itmz"),
                "style_from": str(tmp_path / "nope.itmz"),
            },
        )
        assert "no style template" in _error(out)

    async def test_config_template_overrides_the_default(self, tmp_path):
        source = make_map(
            tmp_path / "src.itmz", "", extra=[("style.xml", b"<style>config</style>")]
        )
        path = tmp_path / "new.itmz"
        with patch("agent13.config.get_config", return_value=_Cfg(str(source))):
            await execute_tool("create_map", {"filepath": str(path)})
        with zipfile.ZipFile(path) as zf:
            assert zf.read("style.xml") == b"<style>config</style>"

    async def test_empty_config_falls_back_to_the_bundled_default(self, tmp_path):
        from agent13.mindmap import DEFAULT_STYLE_PATH

        path = tmp_path / "new.itmz"
        with patch("agent13.config.get_config", return_value=_Cfg("")):
            await execute_tool("create_map", {"filepath": str(path)})
        with zipfile.ZipFile(path) as zf:
            assert zf.read("style.xml") == DEFAULT_STYLE_PATH.read_bytes()

    async def test_sandbox_refusal_is_surfaced(self, tmp_path):
        with patch(
            "tools.mindmap.validate_path_for_write",
            return_value=(False, "denied by sandbox"),
        ):
            out = await execute_tool(
                "create_map", {"filepath": str(tmp_path / "x.itmz")}
            )
        assert "denied by sandbox" in _error(out)

    async def test_schema(self):
        schema = next(s for s in get_tools() if s["function"]["name"] == "create_map")
        params = schema["function"]["parameters"]["properties"]
        assert set(params) == {"filepath", "root_text", "outline", "style_from"}
        assert schema["function"]["parameters"]["required"] == ["filepath"]

    async def test_description_stays_short(self):
        """The description is re-sent on every request: keep it to a screenful."""
        schema = next(s for s in get_tools() if s["function"]["name"] == "create_map")
        assert len(schema["function"]["description"].splitlines()) <= 16


# ── the check trailer (§4.1) ──────────────────────────────────────────────────


def _loose_topic(n: int) -> str:
    """A topic with no position - the rule that fails the release gate."""
    return (
        f'<topic uuid="{_uuid(n)}" text="Loose {n}" '
        'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
    )


class TestCheckTrailer:
    async def test_a_clean_map_appends_nothing(self, tmp_path):
        path = tmp_path / "new.itmz"
        out = _text(await execute_tool("create_map", {"filepath": str(path)}))
        assert "check:" not in out
        read = _text(await execute_tool("read_map", {"filepath": str(path)}))
        assert "check:" not in read

    async def test_read_map_appends_the_error_with_the_address(self, tmp_path):
        path = make_map(tmp_path / "m.itmz", _loose_topic(1))
        out = _text(await execute_tool("read_map", {"filepath": str(path)}))
        assert "check: 1 error" in out
        assert "has no position" in out
        assert _uuid(1)[:8] in out

    async def test_read_map_appends_on_a_query_result(self, tmp_path):
        path = make_map(tmp_path / "m.itmz", _loose_topic(1))
        out = _text(
            await execute_tool("read_map", {"filepath": str(path), "query": "Loose"})
        )
        assert "check: 1 error" in out

    async def test_read_map_appends_on_the_over_budget_refusal(self, tmp_path):
        """The refusal is where a check matters most: a huge map we cannot read."""
        children = "".join(
            f'<topic uuid="{_uuid(n)}" position="{{0, 0}}" text="T{n}"/>'
            for n in range(2, 450)
        ) + _loose_topic(999)
        xml = (
            f'<topic uuid="{_uuid(1)}" position="{{0, 0}}" text="Root">'
            f"{children}</topic>"
        )
        path = make_map(tmp_path / "big.itmz", xml)
        out = _error(
            await execute_tool("read_map", {"filepath": str(path), "depth": 2})
        )
        assert "budget" in out
        assert "check: 1 error" in out
        assert "has no position" in out

    async def test_edit_map_appends_after_the_version_line(self, tmp_path):
        path = make_map(tmp_path / "m.itmz", _loose_topic(1))
        await execute_tool("read_map", {"filepath": str(path)})
        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(path),
                    "ops": [{"op": "rename", "topic": _uuid(1)[:8], "text": "Fixed"}],
                },
            )
        )
        assert out.index("new version:") < out.index("check: 1 error")
        assert "may pre-date this write" in out

    async def test_a_clean_edit_appends_no_note(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(path),
                    "ops": [{"op": "rename", "topic": _uuid(2)[:8], "text": "Renamed"}],
                },
            )
        )
        assert "check:" not in out
        assert "may pre-date" not in out

    async def test_lowercase_uuids_are_not_reported(self, tmp_path):
        xml = (
            f'<topic uuid="{_uuid(0xAB).lower()}" position="{{0, 0}}" text="R" '
            'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"/>'
        )
        path = make_map(tmp_path / "m.itmz", xml, extra=[("style.xml", b"<style/>")])
        out = _text(await execute_tool("read_map", {"filepath": str(path)}))
        assert "check:" not in out

    async def test_a_hundred_defects_are_capped(self, tmp_path):
        xml = "".join(_loose_topic(n) for n in range(1, 101))
        path = make_map(tmp_path / "m.itmz", xml)
        out = _text(await execute_tool("read_map", {"filepath": str(path)}))
        assert out.count("  error: ") == 5
        assert "... and 95 more" in out

    async def test_check_map_is_a_tool_now(self):
        """The 2026-11-25 link work order reversed the old 'no fourth tool'
        rule: link breakage happens OUTSIDE the tool (a folder rename), so
        check-without-write is a real request. The trailer still rides along
        on every touch - both surfaces exist."""
        assert "check_map" in {s["function"]["name"] for s in get_tools()}


# ── the real journal (the 49 MB template) ───────────────────────────────────

JOURNAL = Path.home() / "Resilio/IPSec/maps/IPSec-journal-FY27.itmz"


@pytest.mark.skipif(
    not JOURNAL.exists(), reason="the IPSec journal is not on this machine"
)
class TestRealJournal:
    async def test_style_from_the_journal_is_the_bundled_default(self, tmp_path):
        """The bundled style IS this journal's style.xml, and it lifts in one read.

        Opening the template would parse 49 MB and 1,791 topics; the unit test
        pins that only one member is read, and this pins the result on the real
        file.
        """
        from agent13.mindmap import DEFAULT_STYLE_PATH

        path = tmp_path / "from-journal.itmz"
        await execute_tool(
            "create_map", {"filepath": str(path), "style_from": str(JOURNAL)}
        )
        with zipfile.ZipFile(path) as zf:
            assert zf.read("style.xml") == DEFAULT_STYLE_PATH.read_bytes()

    async def test_a_map_created_from_it_passes_our_own_check(self, tmp_path):
        path = tmp_path / "from-journal.itmz"
        await execute_tool(
            "create_map",
            {
                "filepath": str(path),
                "root_text": "IPSec FY27",
                "outline": "Tunnels\n\tSite A\n\tSite B\nKeys\n\tPSK rotation",
                "style_from": str(JOURNAL),
            },
        )
        out = _text(await execute_tool("read_map", {"filepath": str(path)}))
        assert "6 topics" in out
        assert "check:" not in out


# ── check_map: on-demand validation ──────────────────────────────────────────


class TestCheckMapTool:
    async def test_discovered(self):
        names = {s["function"]["name"] for s in get_tools()}
        assert "check_map" in names

    async def test_clean_map_reports_links_summary(self, tmp_path):
        target = tmp_path / "a.pdf"
        target.write_text("x")
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("A", _uuid(2), link=f"file://{target}")
            + _topic("B", _uuid(3), link="https://example.com"),
        )
        path = make_map(tmp_path / "m.itmz", topics, extra=[("style.xml", b"<style/>")])
        out = _text(await execute_tool("check_map", {"filepath": str(path)}))
        assert "OK (" in out
        assert "links: 1 local checked, all resolve, 1 web not checked" in out

    async def test_broken_link_reported_with_title_and_path(self, tmp_path):
        """The work order's acceptance case, through the tool: rename a
        linked file outside the tool, check_map reports exactly that link."""
        target = tmp_path / "plan.pdf"
        target.write_text("x")
        topics = _topic(
            "Root",
            _uuid(1),
            _topic(
                "Attendees",
                _uuid(2),
                _topic("Yarra", _uuid(3), link=f"file://{target}"),
            ),
        )
        path = make_map(tmp_path / "m.itmz", topics, extra=[("style.xml", b"<style/>")])
        assert "broken link" not in _text(
            await execute_tool("check_map", {"filepath": str(path)})
        )

        target.rename(tmp_path / "plan-renamed.pdf")
        out = _text(await execute_tool("check_map", {"filepath": str(path)}))
        assert '"Yarra"' in out
        assert _uuid(3)[:8] in out
        assert "(under Root > Attendees)" in out
        assert str(target) in out

    async def test_validation_creates_no_file(self, tmp_path):
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link=f"file://{tmp_path}/gone.pdf")
        )
        path = make_map(tmp_path / "m.itmz", topics, extra=[("style.xml", b"<style/>")])
        before = sorted(os.listdir(tmp_path))
        await execute_tool("check_map", {"filepath": str(path)})
        assert sorted(os.listdir(tmp_path)) == before

    async def test_missing_file_is_a_report_not_a_raise(self, tmp_path):
        out = _text(
            await execute_tool("check_map", {"filepath": str(tmp_path / "nope.itmz")})
        )
        assert "cannot read" in out

    async def test_sandbox_refusal_is_surfaced(self, map_path):
        with patch(
            "tools.mindmap.validate_path_for_read",
            return_value=(False, "denied by sandbox"),
        ):
            out = await execute_tool("check_map", {"filepath": str(map_path)})
        assert _error(out) == "denied by sandbox"
        assert json.loads(out)["sandbox_mode"]

    async def test_edit_map_save_reports_a_broken_link(self, tmp_path):
        """Requirement 1: the save completes, and the broken link rides the
        check trailer on the edit result."""
        path = simple_map(tmp_path / "project.itmz")
        await execute_tool("read_map", {"filepath": str(path)})
        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(path),
                    "ops": [
                        {
                            "op": "set_link",
                            "topic": _uuid(5)[:8],
                            "url": f"file://{tmp_path}/gone.pdf",
                        }
                    ],
                },
            )
        )
        assert "wrote" in out  # the save completed
        assert "broken link on" in out
        assert str(tmp_path / "gone.pdf") in out

    async def test_edit_map_save_is_quiet_when_links_resolve(self, tmp_path):
        target = tmp_path / "ok.pdf"
        target.write_text("x")
        path = simple_map(tmp_path / "project.itmz")
        await execute_tool("read_map", {"filepath": str(path)})
        out = _text(
            await execute_tool(
                "edit_map",
                {
                    "filepath": str(path),
                    "ops": [
                        {
                            "op": "set_link",
                            "topic": _uuid(5)[:8],
                            "url": f"file://{target}",
                        }
                    ],
                },
            )
        )
        assert "wrote" in out
        assert "broken link" not in out
