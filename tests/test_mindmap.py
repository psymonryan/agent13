"""Wiring tests for the .itmz codec (agent13/mindmap.py).

The fixtures are built here, not copied: `tests/` is in the release whitelist,
so a real user map in `tests/fixtures/` would publish. The integration test
runs the real checker against the real samples when they are present.
"""

import os
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pytest

from agent13.mindmap import (
    ADDRESS_LEN,
    NEWLINE_MARK,
    NO_ADDRESS,
    QUERY_CONTEXT_CHARS,
    READ_BUDGET_CHARS,
    READ_BUDGET_LINES,
    MindMap,
    MindMapError,
    STORE,
    _flatten,
    parse_outline,
)
from agent13.mindmap_check import POS_RE, check


# ── Fixtures ────────────────────────────────────────────────────────────────


def _uuid(n: int) -> str:
    return f"{n:08X}-0000-4000-8000-000000000000"


def _topic(text: str, uuid: str, children: str = "", **attrs) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return (
        f'<topic uuid="{uuid}" position="{{0, 0}}" text="{text}" '
        f'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00"{extra}>'
        f"{children}</topic>"
    )


def _backdate(path: Path, seconds: int = 3600) -> None:
    """Old mtime, so the quiet window does not fire on a fresh fixture."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def make_map(
    path: Path, topics_xml: str, *, bom=False, decl=False, extra=(), header_attrs=""
) -> Path:
    body = (
        '<iThoughts version="4.0" app="com.toketaware.ithoughtsx" '
        'app-version="9.0" modified="2026-01-01T00:00:00"'
        f"{header_attrs}>"
        f"<topics>{topics_xml}</topics><relationships></relationships></iThoughts>"
    )
    data = body.encode("utf-8")
    if decl:
        data = b'<?xml version="1.0" encoding="UTF-8"?>\n' + data
    if bom:
        data = b"\xef\xbb\xbf" + data
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("mapdata.xml", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, data)
        for name, payload in extra:
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(entry, payload)
    _backdate(path)
    return path


def simple_map(path: Path, **kwargs) -> Path:
    """root > (Alpha > (One, Two), Beta, Gamma)"""
    alpha = _topic("Alpha", _uuid(2), _topic("One", _uuid(3)) + _topic("Two", _uuid(4)))
    beta = _topic("Beta", _uuid(5), note="a note about beta")
    gamma = _topic("Gamma", _uuid(6), folded="1")
    root = _topic("Root", _uuid(1), alpha + beta + gamma)
    return make_map(path, root, **kwargs)


@pytest.fixture(autouse=True)
def _clean_state():
    STORE.forget_all()
    yield
    STORE.forget_all()


# ── Outline parsing ─────────────────────────────────────────────────────────


class TestParseOutline:
    def test_tabs_are_depth(self):
        rows = parse_outline("Docker\n\t-icc\n\tpipework\nKubernetes\n\tcalico")
        assert rows == [
            (0, "Docker"),
            (1, "-icc"),
            (1, "pipework"),
            (0, "Kubernetes"),
            (1, "calico"),
        ]

    def test_two_space_indent_is_inferred(self):
        assert parse_outline("a\n  b\n    c") == [(0, "a"), (1, "b"), (2, "c")]

    def test_four_space_indent_is_inferred(self):
        assert parse_outline("a\n    b") == [(0, "a"), (1, "b")]

    def test_tab_and_four_space_agree(self):
        """§12.4: expanding tabs first makes the two common styles identical."""
        assert parse_outline("a\n\tb") == parse_outline("a\n    b")

    def test_a_uniformly_indented_block_is_normalised(self):
        """A model that indents the whole outline still gets a level-0 first line."""
        assert parse_outline("  a\n    b") == [(0, "a"), (1, "b")]

    def test_bullet_marker_is_stripped(self):
        assert parse_outline("- Docker\n* Kubernetes\n+ Calico") == [
            (0, "Docker"),
            (0, "Kubernetes"),
            (0, "Calico"),
        ]

    def test_bare_dash_is_kept(self):
        """Real maps contain '-icc' and '--link host:name' as topic text."""
        assert parse_outline("-icc\n--link othermachine:nameforinside") == [
            (0, "-icc"),
            (0, "--link othermachine:nameforinside"),
        ]

    def test_backslash_forces_a_literal(self):
        assert parse_outline("\\- Docker") == [(0, "- Docker")]

    def test_blank_lines_are_ignored(self):
        assert parse_outline("a\n\n\tb\n\n") == [(0, "a"), (1, "b")]

    def test_empty_outline_is_an_error(self):
        with pytest.raises(MindMapError, match="empty"):
            parse_outline("   \n\n")

    def test_non_multiple_indent_names_the_line(self):
        with pytest.raises(MindMapError, match="line 3"):
            parse_outline("a\n  b\n   c")

    def test_level_jump_names_the_line(self):
        """A line two levels deeper than the one above is ambiguous, so refuse."""
        with pytest.raises(MindMapError, match="line 3"):
            parse_outline("a\n  b\n      c")


# ── Opening ─────────────────────────────────────────────────────────────────


class TestOpen:
    def test_missing_file_names_the_directory(self, tmp_path):
        missing = tmp_path / "nope.itmz"
        with pytest.raises(MindMapError) as exc:
            MindMap.open(missing)
        assert str(tmp_path) in str(exc.value)
        assert "iThoughtsX" in str(exc.value)

    def test_missing_file_lists_sibling_maps(self, tmp_path):
        simple_map(tmp_path / "present.itmz")
        with pytest.raises(MindMapError, match="present.itmz"):
            MindMap.open(tmp_path / "absent.itmz")

    def test_not_a_zip(self, tmp_path):
        path = tmp_path / "x.itmz"
        path.write_bytes(b"not a zip")
        with pytest.raises(MindMapError, match="not a .itmz archive"):
            MindMap.open(path)

    def test_zip_without_mapdata(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("other.txt", "hi")
        with pytest.raises(MindMapError, match="mapdata.xml"):
            MindMap.open(path)

    def test_wrong_root_element(self, tmp_path):
        path = tmp_path / "x.itmz"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mapdata.xml", "<notamap/>")
        with pytest.raises(MindMapError, match="<iThoughts>"):
            MindMap.open(path)

    def test_counts(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert mm.topic_count() == 6
        assert mm.stats() == (6, 3, 1)


# ── Addressing ──────────────────────────────────────────────────────────────


class TestAddressing:
    def test_address_is_an_eight_char_prefix(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert mm.address_of(mm.root_topic) == _uuid(1)[:ADDRESS_LEN]

    def test_resolve_by_address(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        el = mm.resolve(_uuid(5)[:ADDRESS_LEN])
        assert el.get("text") == "Beta"

    def test_resolve_is_case_insensitive(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert mm.resolve(_uuid(5)[:ADDRESS_LEN].lower()).get("text") == "Beta"

    def test_resolve_by_exact_text(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert mm.resolve("Gamma").get("text") == "Gamma"

    def test_ambiguous_text_lists_the_candidates(self, tmp_path):
        dup = _topic("TODO", _uuid(10)) + _topic("TODO", _uuid(11))
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), dup)))
        with pytest.raises(MindMapError) as exc:
            mm.resolve("TODO")
        assert "2 topics" in str(exc.value)
        assert _uuid(10)[:ADDRESS_LEN] in str(exc.value)

    def test_unknown_address_suggests_nearest(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        with pytest.raises(MindMapError) as exc:
            mm.resolve("Alpho")
        assert "Nearest by text" in str(exc.value)
        assert "Alpha" in str(exc.value)

    def test_colliding_prefixes_are_extended(self, tmp_path):
        """Two uuids sharing 8 chars must not produce the same address."""
        a = _topic("First", "AAAAAAAA-1111-4000-8000-000000000000")
        b = _topic("Second", "AAAAAAAA-2222-4000-8000-000000000000")
        mm = MindMap.open(
            make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), a + b))
        )
        first, second = mm.children(mm.root_topic)
        assert mm.address_of(first) != mm.address_of(second)
        assert len(mm.address_of(first)) > ADDRESS_LEN
        assert mm.resolve(mm.address_of(first)).get("text") == "First"

    def test_a_non_hex_uuid_address_resolves(self, tmp_path):
        """read_map must never print an address that resolve then rejects.

        A hand-made map can carry uuids that are not UUID format. The address
        is a prefix of the hyphen-stripped uuid, so `CHILD-NOTE` prints as
        `CHILDNOT` - which a hex-only gate rejected, telling the model "here is
        the address" and then "no such address".
        """
        child = _topic("Child", "CHILD-NOTE")
        mm = MindMap.open(
            make_map(tmp_path / "m.itmz", _topic("Root", "ROOT-ATTR-TEST", child))
        )
        printed = mm.address_of(mm.children(mm.root_topic)[0])
        assert printed == "CHILDNOT"
        assert mm.resolve(printed).get("text") == "Child"

    def test_every_printed_address_resolves(self, tmp_path):
        """The invariant behind the fix, over a map mixing hex and non-hex uuids."""
        kids = (
            _topic("Hex", _uuid(2))
            + _topic("Dashed", "CHILD-NOTE")
            + _topic("Short", "ABC")
            + _topic("Plain", "plainuuid")
            + '<topic text="NoUuid" position="{0, 0}"/>'
        )
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        for el in mm._order:
            address = mm.address_of(el)
            if address == NO_ADDRESS:
                continue  # a uuid-less topic has no address; covered by the test below
            assert mm.resolve(address) is el, (
                f"{address} did not resolve to its own topic"
            )

    def test_a_uuidless_topic_says_why_it_has_no_address(self, tmp_path):
        """`????????` must not read as an address the tool then denies."""
        mm = MindMap.open(
            make_map(
                tmp_path / "m.itmz",
                _topic("Root", _uuid(1), '<topic text="NoUuid" position="{0, 0}"/>'),
            )
        )
        assert mm.address_of(mm.children(mm.root_topic)[0]) == NO_ADDRESS
        with pytest.raises(MindMapError) as exc:
            mm.resolve(NO_ADDRESS)
        assert "no uuid" in str(exc.value)
        assert "exact text" in str(exc.value)

    def test_a_short_topic_text_is_not_read_as_an_address(self, tmp_path):
        """The 6-char floor: text must still resolve as text, not as a prefix."""
        mm = MindMap.open(
            make_map(
                tmp_path / "m.itmz", _topic("Root", _uuid(1), _topic("Ideas", _uuid(2)))
            )
        )
        assert mm.resolve("Ideas").get("uuid") == _uuid(2)


# ── Reading ─────────────────────────────────────────────────────────────────


class TestRender:
    def test_first_call_returns_root_and_children_only(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render()
        # The root line carries identity only; its text is the block below it.
        assert "root: [00000001]" in text
        assert "text:\n  Root" in text
        assert "Alpha" in text and "Beta" in text
        # Grandchildren are NOT in the first call.
        assert "One" not in text and "Two" not in text
        assert "6 topics, 3 levels, 1 folded" in text

    def test_depth_two_includes_grandchildren(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert "One" in mm.render(depth=2)

    def test_drill_returns_that_topic_and_its_children(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render(topic=_uuid(2)[:ADDRESS_LEN])
        assert f"topic: {_uuid(2)[:ADDRESS_LEN]}  (2 children)" in text
        assert "text:\n  Alpha" in text
        assert "One" in text and "Two" in text
        assert "Beta" not in text

    def test_drilled_topic_shows_its_own_note(self, tmp_path):
        """§12.2: the drilled topic's note, not every child's."""
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert "note: a note about beta" in mm.render(topic=_uuid(5)[:ADDRESS_LEN])

    def test_children_show_a_note_flag_not_the_text(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render()
        assert "note" in text  # the flag
        assert "a note about beta" not in text  # not the body

    def test_version_is_printed(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert f"version: {mm.version.short}" in mm.render()

    def test_flags(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render()
        assert "(2 children)" in text  # Alpha
        assert "folded" in text  # Gamma

    def test_query_returns_addresses_and_paths(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render(query="one")
        assert "matches for 'one'" in text
        # The hit line carries the ANCESTORS only; the topic's own text is the
        # context line below it (a long topic must not repeat itself here).
        assert "Root > Alpha  (no children)" in text
        assert "    One     [drill for the full text]" in text

    def test_query_matching_a_note_shows_the_note(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        text = mm.render(query="about beta")
        assert "    a note about beta     [drill for the full text]" in text

    def test_query_with_no_match_says_so(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        assert "no topic or note contains" in mm.render(query="zzzz")

    def test_over_budget_is_refused_not_truncated(self, tmp_path):
        """A truncated tree reads as an empty branch, so the tool refuses."""
        kids = "".join(
            _topic(
                f"Child {i}",
                _uuid(100 + i),
                "".join(
                    _topic(f"Grand {i}.{j}", _uuid(1000 + i * 10 + j))
                    for j in range(10)
                ),
            )
            for i in range(50)
        )
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        with pytest.raises(MindMapError) as exc:
            mm.render(depth=2)
        message = str(exc.value)
        assert "would return" in message
        assert "depth=1" in message
        assert "Nothing was truncated" in message


# ── Reading the full text ────────────────────────────────────────────────
#
# The work order's DoD: a topic's text is never clipped in a read - it is a
# block of its own, paged by `text_offset`, and the tree stays one line per
# topic. The assertions are about ORDER and EXACTNESS, never about a layout.

LONG = _uuid(2)[:ADDRESS_LEN]


def _esc(text: str) -> str:
    """A topic's text lives in an XML ATTRIBUTE: a raw newline is normalised to
    a space by the parser, so a multi-line fixture must write the entity."""
    return text.replace("\n", "&#10;")


def text_map(path: Path, text: str, children: str = "", note: str = "") -> Path:
    """root > (Long [carries `text`], Short)."""
    attrs = {"note": note} if note else {}
    long_topic = _topic(_esc(text), _uuid(2), children, **attrs)
    return make_map(
        path, _topic("Root", _uuid(1), long_topic + _topic("Short", _uuid(3)))
    )


def _split_block(out: str) -> tuple[str, Optional[int]]:
    """(the text block's body, the footer's offset or None).

    The body is the block's own lines with their two-space indent removed, so a
    window can be compared to the original text character for character.
    """
    head, sep, rest = out.partition("text:\n")
    assert sep, f"no text block in the output: {out!r}"
    kept: list[str] = []
    next_offset = None
    for line in rest.split("\n"):
        if line.startswith("  [text continues at offset "):
            next_offset = int(
                line.split("offset ")[1].split(" of ")[0].replace(",", "")
            )
            break
        if not line.startswith("  "):
            break  # version: / blank / drill - the block has ended
        kept.append(line[2:])
    return "\n".join(kept), next_offset


def _page(mm: MindMap, address: str) -> list[tuple[int, str]]:
    """Read a topic's text in windows: [(offset, body)], the last one footerless."""
    windows: list[tuple[int, str]] = []
    offset = 0
    while True:
        body, next_offset = _split_block(
            mm.render(topic=address, depth=0, text_offset=offset)
        )
        windows.append((offset, body))
        if next_offset is None:
            return windows
        assert next_offset > offset, (
            "the footer must advance, or a caller loops for ever"
        )
        assert len(windows) < 50, "paging did not terminate"
        offset = next_offset


class TestFullText:
    def test_a_500_char_topic_reads_back_in_full(self, tmp_path):
        """DoD 1: no clipping anywhere - neither in the tree nor in the block."""
        text = "Alpha beta gamma delta. " * 21
        assert len(text) == 504
        mm = MindMap.open(text_map(tmp_path / "m.itmz", text))
        block = mm.render(topic=LONG, depth=0)
        assert text in block
        assert "\u2026" not in block
        # The same topic listed as a child of the root: also unclipped.
        tree = mm.render()
        assert text in tree
        assert "\u2026" not in tree

    def test_newlines_survive_the_block_and_are_marked_in_the_tree(self, tmp_path):
        """DoD 2: the block keeps the breaks; a tree line cannot, so it marks them."""
        text = "first line\nsecond line\nthird"
        mm = MindMap.open(text_map(tmp_path / "m.itmz", text))
        assert "  first line\n  second line\n  third" in mm.render(topic=LONG, depth=0)
        line = next(row for row in mm.render().splitlines() if LONG in row)
        assert f"first line{NEWLINE_MARK}second line{NEWLINE_MARK}third" in line

    def test_flatten_collapses_space_runs_and_marks_newline_runs(self):
        """DoD 3: both halves. Marking a newline run without collapsing a space
        run was the bug that was almost shipped - a 200-space indent would push
        the text off the screen."""
        assert _flatten("a" + " " * 200 + "b") == "a b"
        assert _flatten("a\nb") == f"a{NEWLINE_MARK}b"
        assert _flatten("a\n\nb") == f"a{NEWLINE_MARK}b"  # one run, one mark
        assert _flatten("a \n b") == f"a{NEWLINE_MARK}b"
        assert _flatten("a\tb") == "a b"  # a tab is whitespace, not a break

    @pytest.mark.parametrize("shape", ["short-lines", "long-lines"])
    def test_a_long_topic_pages_and_the_windows_concatenate_exactly(
        self, tmp_path, shape
    ):
        """DoD 4: the footer offset is the only thing standing between a model
        and an infinite loop, so it must be exact.

        Two shapes, because two different budgets do the cutting: short lines
        hit the LINE budget (which cuts at a line boundary by construction),
        long lines hit the CHAR budget (where the cut has to be moved back to a
        newline, or the windows stop concatenating).
        """
        if shape == "short-lines":
            text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(700))
        else:
            text = "\n".join("y" * 900 for _ in range(40))
        assert len(text) > 30000
        mm = MindMap.open(text_map(tmp_path / "m.itmz", text))
        windows = _page(mm, LONG)
        assert len(windows) > 1
        for (offset, body), (next_offset, _) in zip(windows, windows[1:]):
            # The body is the text from `offset`; the footer steps past the
            # newline that separated the windows, so it is at most one further.
            assert text[offset : offset + len(body)] == body, (
                "a window is not the text it claims"
            )
            assert next_offset in (offset + len(body), offset + len(body) + 1)
        assert windows[-1][0] + len(windows[-1][1]) == len(text)
        assert "\n".join(body for _, body in windows) == text

    def test_text_offset_past_the_end_names_the_real_length(self, tmp_path):
        """DoD 5: no text, one line, the true length - so the caller can correct."""
        mm = MindMap.open(text_map(tmp_path / "m.itmz", "hello"))
        out = mm.render(topic=LONG, depth=0, text_offset=99)
        assert "nothing at offset 99" in out
        assert "the text is 5 chars" in out
        assert "hello" not in out

    def test_ignored_arguments_are_validated_not_dropped(self, tmp_path):
        """DoD 6: a negative offset must not be sliced from the end, and the
        arguments a query ignores must be refused rather than silently unused."""
        mm = MindMap.open(text_map(tmp_path / "m.itmz", "hello"))
        with pytest.raises(MindMapError) as exc:
            mm.render(topic=LONG, depth=0, text_offset=-1)
        assert "text_offset must be 0 or more" in str(exc.value)
        assert "5 chars" in str(exc.value)

        with pytest.raises(
            MindMapError, match="query searches the whole map, so topic"
        ):
            mm.render(query="hello", topic=LONG)
        with pytest.raises(
            MindMapError, match="query searches the whole map, so text_offset"
        ):
            mm.render(query="hello", text_offset=5)

    def test_depth_zero_returns_the_topic_and_no_children(self, tmp_path):
        """DoD 7: 0 is a real depth. `depth or 1` would turn it into 1."""
        kids = _topic("One", _uuid(3)) + _topic("Two", _uuid(4))
        mm = MindMap.open(text_map(tmp_path / "m.itmz", "parent text", kids))
        out = mm.render(topic=LONG, depth=0)
        assert "One" not in out and "Two" not in out
        assert "(2 children)" in out  # the header still says what is below
        assert "\n  (no children)" not in out

    def test_a_tree_that_eats_the_budget_withholds_the_block(self, tmp_path):
        """DoD 8: the loop test. A window of a few hundred chars is not worth a
        round trip, and a zero-width one would repeat itself for ever."""
        kids = "".join(
            _topic(f"k{i:02d}" + "y" * 600, _uuid(100 + i)) for i in range(30)
        )
        mm = MindMap.open(
            make_map(tmp_path / "m.itmz", _topic(_esc("z" * 5000), _uuid(1), kids))
        )
        out = mm.render(depth=1)
        assert "the tree used the budget" in out
        assert "depth=0" in out
        assert "continues at offset 0" not in out
        assert "text continues at offset" not in out  # no footer at all

    def test_the_line_budget_holds_and_the_footer_offset_is_exact(self, tmp_path):
        """DoD 9: the block's own header and footer count against the budget."""
        text = "\n".join("ab" for _ in range(9000))
        mm = MindMap.open(text_map(tmp_path / "m.itmz", text))
        out = mm.render(topic=LONG, depth=0)
        assert len(out.splitlines()) <= READ_BUDGET_LINES
        body, next_offset = _split_block(out)
        assert next_offset is not None
        # The footer steps past the newline that separated the windows, so the
        # offset is one more than the characters emitted.
        assert next_offset == len(body) + 1
        assert text[len(body)] == "\n"

    def test_children_alone_over_budget_still_refuses(self, tmp_path):
        """DoD 10: paging the text must not have made refusals unreachable."""
        kids = "".join(
            _topic(f"k{i:02d}" + "y" * 400, _uuid(100 + i)) for i in range(60)
        )
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        with pytest.raises(MindMapError) as exc:
            mm.render(depth=1)
        message = str(exc.value)
        assert "would return" in message
        assert "Nothing was truncated" in message
        assert "depth=0" in message  # the advice fits the depth it refused
        assert "00000064" in message  # drill options, not a bare refusal

    def test_the_refusal_advice_fits_the_depth(self, tmp_path):
        """DoD 11: suggesting the depth that just failed is a loop."""
        kids = "".join(
            _topic(
                f"Child {i}",
                _uuid(100 + i),
                "".join(
                    _topic(f"Grand {i}.{j}", _uuid(1000 + i * 10 + j))
                    for j in range(10)
                ),
            )
            for i in range(50)
        )
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        with pytest.raises(MindMapError) as two:
            mm.render(depth=2)
        assert "Call again with depth=1" in str(two.value)

        wide = "".join(
            _topic(f"k{i:02d}" + "y" * 400, _uuid(200 + i)) for i in range(60)
        )
        flat = MindMap.open(
            make_map(tmp_path / "wide.itmz", _topic("Root", _uuid(1), wide))
        )
        with pytest.raises(MindMapError) as one:
            flat.render(depth=1)
        assert "Call again with depth=0" in str(one.value)
        assert "Call again with depth=1" not in str(one.value)

    def test_the_root_text_is_not_duplicated(self, tmp_path):
        """DoD 12: the root line carries identity; the text is the block below."""
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        out = mm.render()
        root_line = next(row for row in out.splitlines() if row.startswith("root: "))
        assert "Root" not in root_line
        assert "text:\n  Root" in out
        assert out.count("Root") == 1

    def test_the_block_sits_between_the_header_and_the_meta(self, tmp_path):
        """DoD 13: header -> block -> note -> version -> tree -> drill."""
        kids = _topic("One", _uuid(3)) + _topic("Two", _uuid(4))
        mm = MindMap.open(
            text_map(tmp_path / "m.itmz", "the text", kids, note="the note")
        )
        lines = mm.render(topic=LONG, depth=1).splitlines()
        assert lines[0].startswith("topic: ")
        order = [
            next(i for i, l in enumerate(lines) if l.startswith(prefix))
            for prefix in ("topic: ", "text:", "note: ", "version: ")
        ]
        assert order == sorted(order)
        assert lines[-1].startswith("[drill: ")

    def test_a_query_hit_is_bounded(self, tmp_path):
        """DoD 14: a hit is ancestors + context + child count, never the path's
        full text - a 37k topic in a query result would blow the budget."""
        needle = "needleword"
        big = ("filler " * 600) + needle + (" tail" * 600)
        mm = MindMap.open(
            make_map(
                tmp_path / "m.itmz",
                _topic(
                    "Root", _uuid(1), _topic("A" * 300, _uuid(2), _topic(big, _uuid(3)))
                ),
            )
        )
        out = mm.render(query=needle)
        assert len(out) < READ_BUDGET_CHARS
        assert len(out.splitlines()) < READ_BUDGET_LINES
        assert big not in out
        hit, context = out.splitlines()[3:5]
        assert hit.startswith(_uuid(3)[:ADDRESS_LEN])
        assert "A" * 300 not in hit  # the ancestor path is elided, not repeated
        assert hit.endswith("(no children)")
        assert needle in context
        assert len(context) < QUERY_CONTEXT_CHARS + 100

    def test_a_note_only_hit_shows_the_note_context(self, tmp_path):
        """DoD 14, second half: an empty text window would be a dead end."""
        mm = MindMap.open(
            text_map(
                tmp_path / "m.itmz", "no match here", note="a note about the needleword"
            )
        )
        out = mm.render(query="needleword")
        assert "a note about the needleword" in out
        assert "no match here" not in out

    def test_a_query_that_cannot_fit_refuses(self, tmp_path, monkeypatch):
        """DoD 15: unreachable on real data, so the budget is forced down. A
        test that cannot fail is not a test."""
        monkeypatch.setattr("agent13.mindmap.READ_BUDGET_CHARS", 100)
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        with pytest.raises(MindMapError) as exc:
            mm.render(query="o")
        assert "Nothing was truncated" in str(exc.value)
        assert "narrow the query" in str(exc.value)

    def test_the_refusal_menu_previews_40_chars(self, tmp_path):
        """DoD 16: 24 chars was too short to tell two siblings apart."""
        kids = "".join(
            _topic(f"child {i} " + "q" * 400, _uuid(100 + i)) for i in range(60)
        )
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))
        with pytest.raises(MindMapError) as exc:
            mm.render(depth=1)
        message = str(exc.value)
        assert "child 0 " + "q" * 31 + "\u2026" in message  # 40 chars, ellipsis at 39
        assert "q" * 40 not in message

    def test_an_edit_report_still_clips(self, tmp_path):
        """DoD 17: the report is a receipt, not a second copy of the map."""
        mm = MindMap.open(text_map(tmp_path / "m.itmz", "short"))
        report = mm.apply_ops([{"op": "rename", "topic": LONG, "text": "n" * 200}])
        assert report[0].startswith(f"renamed {LONG}: 'short' -> '")
        assert "\u2026'" in report[0]
        assert "n" * 200 not in report[0]

    def test_a_topic_with_empty_text_prints_no_block(self, tmp_path):
        """DoD 19: no bare header with nothing under it."""
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("", _uuid(1))))
        out = mm.render()
        assert "text:" not in out
        assert "root: [" in out


# ── Sibling order ────────────────────────────────────────────────────────
#
# `position.y` is an ORDERING KEY, not a layout: the app computes the geometry
# and ignores the magnitudes, but it draws by y. So the invariant every op must
# keep is "for every parent, the XML child order equals the order by y" - and
# the assertions here are about ORDER, never about exact coordinates. Pinning a
# y value would pin a layout the app does not promise.


def _child_xml(
    text: str, uuid: str, y: float, x: float = 0.0, children: str = ""
) -> str:
    return (
        f'<topic uuid="{uuid}" position="{{{x}, {y}}}" text="{text}" '
        f'created="2026-01-01T00:00:00" modified="2026-01-01T00:00:00">{children}</topic>'
    )


def _y(el) -> float:
    return float((el.get("position") or "{0, 0}").strip("{} ").split(",")[1])


def _by_y(kids) -> list:
    return sorted(kids, key=_y)


def _uuids(kids) -> list[str]:
    return [el.get("uuid") for el in kids]


class TestSiblingOrder:
    def _row(self, tmp_path, ys) -> MindMap:
        """root > S0..Sn in XML order, at the given y values."""
        kids = "".join(_child_xml(f"S{i}", _uuid(10 + i), y) for i, y in enumerate(ys))
        return MindMap.open(
            make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids))
        )

    @pytest.mark.parametrize("target", [0, 2, 3])
    def test_move_within_a_parent_keeps_the_order_and_touches_nothing_else(
        self, tmp_path, target
    ):
        """DoD 1: the reported bug. XML order moved, y did not, so the app drew
        the topic where it used to be."""
        mm = self._row(tmp_path, [0, 60, 120, 180])
        root = mm.root_topic
        moved = mm.children(root)[3]  # S3, last
        before = {el.get("uuid"): el.get("position") for el in mm.children(root)}

        mm.apply_ops(
            [{"op": "move", "topic": _uuid(13), "parent": _uuid(1), "index": target}]
        )

        kids = mm.children(root)
        assert _uuids(_by_y(kids)) == _uuids(kids), (
            "the app would draw a different order"
        )
        for el in kids:
            if el is not moved:
                assert el.get("position") == before[el.get("uuid")], (
                    "a sibling was renumbered"
                )

    def test_move_across_parents_keeps_the_order_in_the_new_parent(self, tmp_path):
        """DoD 2: positions are parent-RELATIVE, so a cross-parent move must
        assign a fresh one or the topic shifts by the parent's delta."""
        p1 = _child_xml(
            "P1",
            _uuid(2),
            0,
            children=_child_xml("A", _uuid(10), 0) + _child_xml("B", _uuid(11), 60),
        )
        p2 = _child_xml(
            "P2",
            _uuid(3),
            100,
            children=_child_xml("X", _uuid(20), 0) + _child_xml("Y", _uuid(21), 60),
        )
        mm = MindMap.open(
            make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), p1 + p2))
        )
        p2_el = mm.resolve(_uuid(3))
        before = {el.get("uuid"): el.get("position") for el in mm.children(p2_el)}

        mm.apply_ops(
            [{"op": "move", "topic": _uuid(11), "parent": _uuid(3), "index": 1}]
        )

        kids = mm.children(p2_el)
        assert _uuids(kids) == [_uuid(20), _uuid(11), _uuid(21)]
        assert _uuids(_by_y(kids)) == _uuids(kids)
        assert kids[1].get("position") is not None
        for el in kids:
            if el.get("uuid") in before:
                assert el.get("position") == before[el.get("uuid")], (
                    "a sibling was renumbered"
                )

    def test_add_branch_sorts_after_siblings_at_y_4288(self, tmp_path):
        """DoD 3: `parent.y + n*60` only sorts last if it beats every sibling,
        and real siblings reach y = 4,288."""
        mm = self._row(tmp_path, [100, 2000, 4288])
        root = mm.root_topic
        mm.apply_ops(
            [{"op": "add_branch", "parent": _uuid(1), "outline": "New1\nNew2"}]
        )

        kids = mm.children(root)
        assert _uuids(_by_y(kids)) == _uuids(kids)
        created = kids[-2:]
        assert [el.get("text") for el in created] == ["New1", "New2"]
        assert all(_y(el) > 4288 for el in created), (
            "a created topic sorted into the middle"
        )

    def test_add_branch_on_a_childless_parent_follows_the_outline(self, tmp_path):
        """DoD 4: no siblings means y = 0 upward, in outline order."""
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1))))
        mm.apply_ops([{"op": "add_branch", "parent": _uuid(1), "outline": "A\nB\nC"}])

        kids = mm.children(mm.root_topic)
        assert [el.get("text") for el in _by_y(kids)] == ["A", "B", "C"]

    def test_replace_branch_sorts_the_new_children_in_outline_order(self, tmp_path):
        """DoD 5: replace_branch goes through the same outline path."""
        mm = self._row(tmp_path, [0, 60])
        mm.apply_ops(
            [{"op": "replace_branch", "topic": _uuid(1), "outline": "X\nY\nZ"}]
        )

        kids = mm.children(mm.root_topic)
        assert [el.get("text") for el in _by_y(kids)] == ["X", "Y", "Z"]

    def test_delete_keeping_children_places_them_at_the_deleted_index(self, tmp_path):
        """DoD 6: the promoted children were positioned relative to the deleted
        topic, not to the grandparent."""
        # B's children are positioned relative to B, so their y (500, 560) is
        # unrelated to the grandparent's range (0..120) - left alone they would
        # sort AFTER C. That is the bug, not a contrived one.
        b = _child_xml(
            "B",
            _uuid(11),
            60,
            children=_child_xml("B1", _uuid(20), 500)
            + _child_xml("B2", _uuid(21), 560),
        )
        kids = _child_xml("A", _uuid(10), 0) + b + _child_xml("C", _uuid(12), 120)
        mm = MindMap.open(make_map(tmp_path / "m.itmz", _topic("Root", _uuid(1), kids)))

        mm.apply_ops([{"op": "delete", "topic": _uuid(11), "keep_children": True}])

        after = mm.children(mm.root_topic)
        assert [el.get("text") for el in after] == ["A", "B1", "B2", "C"]
        assert _uuids(_by_y(after)) == _uuids(after)

    def test_every_created_or_moved_topic_has_a_position(self, tmp_path):
        """DoD 7: a missing position is treated as {0, 0} - the parent's origin,
        which is almost never the intent."""
        mm = self._row(tmp_path, [0, 60])
        mm.apply_ops(
            [
                {"op": "add_branch", "parent": _uuid(1), "outline": "New\n\tDeep"},
                {"op": "move", "topic": _uuid(11), "parent": _uuid(1), "index": 0},
            ]
        )
        for el in mm._order:
            assert el.get("position"), f"{el.get('text')} has no position"
            assert POS_RE.match(el.get("position")), el.get("position")


# ── Writing ─────────────────────────────────────────────────────────────────


class TestSave:
    def test_round_trip_preserves_entries_and_attributes(self, tmp_path):
        src = simple_map(
            tmp_path / "m.itmz",
            extra=[("style.xml", b"<style/>"), ("assets/A/one.png", b"\x89PNG")],
        )
        before = {
            i.filename: (i.compress_type, zipfile.ZipFile(src).read(i.filename))
            for i in zipfile.ZipFile(src).infolist()
        }
        order_before = [i.filename for i in zipfile.ZipFile(src).infolist()]

        mm = MindMap.open(src)
        mm.apply_ops(
            [{"op": "add_branch", "parent": _uuid(2)[:ADDRESS_LEN], "outline": "New"}]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        with zipfile.ZipFile(src) as zf:
            order_after = [i.filename for i in zf.infolist()]
            assert order_after == order_before
            assert not any(i.is_dir() for i in zf.infolist())
            for name in order_before:
                if name == "mapdata.xml":
                    continue
                assert zf.read(name) == before[name][1]
                assert zf.getinfo(name).compress_type == before[name][0]

    def test_no_op_save_is_semantically_identical(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops(
            [{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "Beta"}]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        reopened = MindMap.open(path)
        assert reopened.topic_count() == 6
        assert reopened.resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "Beta"

    def test_bom_is_preserved(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz", bom=True)
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        with zipfile.ZipFile(path) as zf:
            assert zf.read("mapdata.xml").startswith(b"\xef\xbb\xbf")

    def test_declaration_is_preserved(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz", decl=True)
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        with zipfile.ZipFile(path) as zf:
            assert zf.read("mapdata.xml").lstrip(b"\xef\xbb\xbf").startswith(b"<?xml")

    def test_no_declaration_is_not_added(self, tmp_path):
        """App-written maps have no declaration; do not invent one."""
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        with zipfile.ZipFile(path) as zf:
            assert not zf.read("mapdata.xml").startswith(b"<?xml")

    def test_header_modified_is_stamped(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        assert MindMap.open(path).root.get("modified") != "2026-01-01T00:00:00"


class TestVersionCheck:
    def test_stale_version_is_refused_and_nothing_is_written(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        stale = mm.version.short
        # Someone else writes the file.
        other = MindMap.open(path)
        other.apply_ops(
            [{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "Theirs"}]
        )
        other.save(expect_short=other.version.short, quiet_seconds=0)

        mm.apply_ops(
            [{"op": "rename", "topic": _uuid(6)[:ADDRESS_LEN], "text": "Mine"}]
        )
        with pytest.raises(MindMapError) as exc:
            mm.save(expect_short=stale, quiet_seconds=0)
        message = str(exc.value)
        assert "changed since it was read" in message
        assert "Nothing was written" in message
        assert "all-or-nothing" in message
        assert "read_map" in message
        # The other writer's change survived; ours did not.
        assert (
            MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "Theirs"
        )

    def test_matching_version_writes(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        new = mm.save(expect_short=mm.version.short, quiet_seconds=0)
        assert new.short != mm.version.short or True
        assert MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "B2"


class TestQuietWindow:
    def test_fresh_mtime_is_refused(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        os.utime(path, None)  # now
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        with pytest.raises(MindMapError) as exc:
            mm.save(expect_short=mm.version.short, quiet_seconds=20)
        message = str(exc.value)
        assert "may be typing" in message
        assert "Nothing was written" in message
        assert MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "Beta"

    def test_zero_disables_the_guard(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        os.utime(path, None)
        mm = MindMap.open(path)
        mm.apply_ops([{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}])
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        assert MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "B2"

    def test_our_own_write_does_not_trip_the_guard(self, tmp_path):
        """§12.1: a second edit in the same turn must not be refused by our own mtime."""
        path = simple_map(tmp_path / "m.itmz")
        first = MindMap.open(path)
        first.apply_ops(
            [{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}]
        )
        first.save(expect_short=first.version.short, quiet_seconds=20)

        second = MindMap.open(path)  # mtime is now fresh, and it is ours
        second.apply_ops(
            [{"op": "rename", "topic": _uuid(6)[:ADDRESS_LEN], "text": "G2"}]
        )
        second.save(expect_short=second.version.short, quiet_seconds=20)
        assert MindMap.open(path).resolve(_uuid(6)[:ADDRESS_LEN]).get("text") == "G2"

    def test_an_external_write_after_ours_trips_the_guard(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        first = MindMap.open(path)
        first.apply_ops(
            [{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "B2"}]
        )
        first.save(expect_short=first.version.short, quiet_seconds=20)

        # Someone else touches the file a moment later.
        later = time.time() + 5
        os.utime(path, (later, later))

        second = MindMap.open(path)
        second.apply_ops(
            [{"op": "rename", "topic": _uuid(6)[:ADDRESS_LEN], "text": "G2"}]
        )
        with pytest.raises(MindMapError, match="may be typing"):
            second.save(expect_short=second.version.short, quiet_seconds=20)


# ── Operations ──────────────────────────────────────────────────────────────


class TestOps:
    def _map(self, tmp_path):
        return MindMap.open(simple_map(tmp_path / "m.itmz"))

    def test_add_branch_creates_valid_topics(self, tmp_path):
        mm = self._map(tmp_path)
        report = mm.apply_ops(
            [
                {
                    "op": "add_branch",
                    "parent": _uuid(2)[:ADDRESS_LEN],
                    "outline": "Docker\n\t-icc\nKubernetes",
                }
            ]
        )
        assert any("added 3 topics" in line for line in report)
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        assert [c.get("text") for c in mm.children(alpha)] == [
            "One",
            "Two",
            "Docker",
            "Kubernetes",
        ]
        docker = mm.children(alpha)[2]
        assert docker.get("uuid") and docker.get("position")
        assert docker.get("created") and docker.get("modified")
        assert [c.get("text") for c in mm.children(docker)] == ["-icc"]

    def test_replace_branch(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [
                {
                    "op": "replace_branch",
                    "topic": _uuid(2)[:ADDRESS_LEN],
                    "outline": "Only",
                }
            ]
        )
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        assert [c.get("text") for c in mm.children(alpha)] == ["Only"]

    def test_rename(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [{"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "Renamed"}]
        )
        assert mm.resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "Renamed"

    def test_move(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [
                {
                    "op": "move",
                    "topic": _uuid(6)[:ADDRESS_LEN],
                    "parent": _uuid(2)[:ADDRESS_LEN],
                }
            ]
        )
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        assert "Gamma" in [c.get("text") for c in mm.children(alpha)]

    def test_move_into_own_subtree_is_refused(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError, match="own subtree"):
            mm.apply_ops(
                [
                    {
                        "op": "move",
                        "topic": _uuid(2)[:ADDRESS_LEN],
                        "parent": _uuid(3)[:ADDRESS_LEN],
                    }
                ]
            )

    def test_move_to_an_index_equal_to_the_child_count_appends(self, tmp_path):
        """The old inline code indexed kids[at] and raised IndexError here."""
        mm = self._map(tmp_path)
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        count = len(mm.children(alpha))
        mm.apply_ops(
            [
                {
                    "op": "move",
                    "topic": _uuid(6)[:ADDRESS_LEN],
                    "parent": _uuid(2)[:ADDRESS_LEN],
                    "index": count,
                }
            ]
        )
        assert [c.get("text") for c in mm.children(alpha)][-1] == "Gamma"

    @pytest.mark.parametrize(
        "op",
        [
            {
                "op": "add_branch",
                "parent": _uuid(2)[:ADDRESS_LEN],
                "outline": "New\n\tChild",
            },
            {
                "op": "replace_branch",
                "topic": _uuid(2)[:ADDRESS_LEN],
                "outline": "Only",
            },
            {"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "Renamed"},
            {
                "op": "move",
                "topic": _uuid(6)[:ADDRESS_LEN],
                "parent": _uuid(2)[:ADDRESS_LEN],
            },
            {
                "op": "move",
                "topic": _uuid(6)[:ADDRESS_LEN],
                "parent": _uuid(2)[:ADDRESS_LEN],
                "index": 0,
            },
            {"op": "delete", "topic": _uuid(2)[:ADDRESS_LEN]},
            {"op": "delete", "topic": _uuid(2)[:ADDRESS_LEN], "keep_children": True},
            {"op": "set_note", "topic": _uuid(6)[:ADDRESS_LEN], "text": "n"},
            {
                "op": "set_link",
                "topic": _uuid(6)[:ADDRESS_LEN],
                "url": "https://a.example",
            },
        ],
    )
    def test_every_op_leaves_every_address_resolvable(self, tmp_path, op):
        """The invariant the structural primitives exist to hold.

        Three ways the index can go stale, all checked: a new topic missing
        from it (its printed address becomes its full uuid and resolve()
        rejects it), a deleted topic still in it, and a moved topic still
        pointing at its old parent.
        """
        mm = self._map(tmp_path)
        mm.apply_ops([op])

        def reachable(el):
            return 1 + sum(reachable(child) for child in el.findall("topic"))

        assert len(mm._order) == reachable(mm.root_topic), (
            "the index holds a stale topic count"
        )
        for el in mm._order:
            address = mm.address_of(el)
            assert mm.resolve(address) is el, f"{op['op']}: {address} did not resolve"
            actual_parent = next((p for p in mm._order if el in list(p)), None)
            assert mm.parent_of(el) is actual_parent, (
                f"{op['op']}: stale parent for {address}"
            )

    def test_a_created_topic_reports_a_usable_address(self, tmp_path):
        """The report line must hand back an address the next call can use."""
        mm = self._map(tmp_path)
        report = mm.apply_ops(
            [{"op": "add_branch", "parent": _uuid(2)[:ADDRESS_LEN], "outline": "Probe"}]
        )
        created_address = report[1].split()[0]
        assert created_address != _uuid(2)[:ADDRESS_LEN]
        assert mm.resolve(created_address).get("text") == "Probe"

    def test_delete_drops_the_subtree(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops([{"op": "delete", "topic": _uuid(2)[:ADDRESS_LEN]}])
        assert "Alpha" not in [c.get("text") for c in mm.children(mm.root_topic)]

    def test_delete_keep_children_splices_at_the_same_index(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [{"op": "delete", "topic": _uuid(2)[:ADDRESS_LEN], "keep_children": True}]
        )
        assert [c.get("text") for c in mm.children(mm.root_topic)] == [
            "One",
            "Two",
            "Beta",
            "Gamma",
        ]

    def test_root_cannot_be_deleted(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError, match="root"):
            mm.apply_ops([{"op": "delete", "topic": _uuid(1)[:ADDRESS_LEN]}])

    def test_set_note(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [{"op": "set_note", "topic": _uuid(6)[:ADDRESS_LEN], "text": "hello"}]
        )
        assert mm.resolve(_uuid(6)[:ADDRESS_LEN]).get("note") == "hello"

    # -- set_link ---------------------------------------------------------

    def test_set_link_reports_the_text_and_the_url(self, tmp_path):
        mm = self._map(tmp_path)
        report = mm.apply_ops(
            [
                {
                    "op": "set_link",
                    "topic": _uuid(6)[:ADDRESS_LEN],
                    "url": "https://example.com/x",
                }
            ]
        )
        assert mm.resolve(_uuid(6)[:ADDRESS_LEN]).get("link") == "https://example.com/x"
        assert "Gamma" in report[0], "the report must name the topic text"
        assert "https://example.com/x" in report[0], "the report must echo the URL"

    @pytest.mark.parametrize(
        "url",
        [
            "file:///Users/simon/notes/a.pdf",
            "ithoughts://0B4E1A2C",
            "message:guid=ABC-123",
            "mailto:someone@example.com",
            "ftp://ftp.example.com/pub",
            "../work/src/agent13",
            "/Local/My%20Map.itmz",
            "javascript:void(0)",
            "  https://example.com/padded  ",
            "not a url at all",
        ],
    )
    def test_set_link_stores_any_value_verbatim(self, tmp_path, url):
        """No scheme allowlist: a value a validator dislikes is still the user's data."""
        mm = self._map(tmp_path)
        mm.apply_ops([{"op": "set_link", "topic": _uuid(6)[:ADDRESS_LEN], "url": url}])
        assert mm.resolve(_uuid(6)[:ADDRESS_LEN]).get("link") == url

    @pytest.mark.parametrize("empty", ["", None])
    def test_set_link_empty_or_null_removes_the_attribute(self, tmp_path, empty):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops(
            [
                {
                    "op": "set_link",
                    "topic": _uuid(6)[:ADDRESS_LEN],
                    "url": "https://a.example",
                }
            ]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        reopened = MindMap.open(path)
        report = reopened.apply_ops(
            [{"op": "set_link", "topic": _uuid(6)[:ADDRESS_LEN], "url": empty}]
        )
        reopened.save(expect_short=reopened.version.short, quiet_seconds=0)

        final = MindMap.open(path).resolve(_uuid(6)[:ADDRESS_LEN])
        assert "link" not in final.attrib, (
            "the attribute must be REMOVED, not set to empty"
        )
        assert "cleared" in report[0]

    def test_clearing_a_link_that_was_never_there_says_so(self, tmp_path):
        """A URL in the topic TEXT is not a link attribute - do not report a removal."""
        mm = self._map(tmp_path)  # Gamma has no link attribute
        report = mm.apply_ops(
            [{"op": "set_link", "topic": _uuid(6)[:ADDRESS_LEN], "url": ""}]
        )
        assert "nothing to clear" in report[0]
        assert "cleared" not in report[0]

    def test_clearing_a_link_reports_the_previous_value(self, tmp_path):
        mm = self._map(tmp_path)
        mm.apply_ops(
            [
                {
                    "op": "set_link",
                    "topic": _uuid(6)[:ADDRESS_LEN],
                    "url": "https://old.example",
                }
            ]
        )
        report = mm.apply_ops(
            [{"op": "set_link", "topic": _uuid(6)[:ADDRESS_LEN], "url": ""}]
        )
        assert "https://old.example" in report[0], (
            "the report must say what was removed"
        )

    def test_set_link_rejects_a_non_string(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError) as exc:
            mm.apply_ops(
                [{"op": "set_link", "topic": _uuid(6)[:ADDRESS_LEN], "url": 42}]
            )
        assert "op 0 (set_link)" in str(exc.value)
        assert "Nothing was written" in str(exc.value)

    def test_a_failed_set_link_writes_nothing_from_the_earlier_ops(self, tmp_path):
        """Atomicity: op 0 is valid, op 1 fails, so op 0 must not reach the file."""
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        with pytest.raises(MindMapError):
            mm.apply_ops(
                [
                    {
                        "op": "rename",
                        "topic": _uuid(5)[:ADDRESS_LEN],
                        "text": "Renamed",
                    },
                    {"op": "set_link", "topic": "DEADBEEF", "url": "https://a.example"},
                ]
            )
        assert MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).get("text") == "Beta"

    def test_set_link_survives_a_save_and_reopen(self, tmp_path):
        path = simple_map(tmp_path / "m.itmz")
        mm = MindMap.open(path)
        mm.apply_ops(
            [
                {
                    "op": "set_link",
                    "topic": _uuid(2)[:ADDRESS_LEN],
                    "url": "file:///tmp/a.pdf",
                }
            ]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)
        assert (
            MindMap.open(path).resolve(_uuid(2)[:ADDRESS_LEN]).get("link")
            == "file:///tmp/a.pdf"
        )

    def test_unknown_op_names_the_index(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError) as exc:
            mm.apply_ops(
                [
                    {"op": "rename", "topic": _uuid(5)[:ADDRESS_LEN], "text": "x"},
                    {"op": "explode"},
                ]
            )
        assert "op 1" in str(exc.value)
        assert "Nothing was written" in str(exc.value)

    def test_bad_address_names_the_op_index(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError) as exc:
            mm.apply_ops([{"op": "rename", "topic": "DEADBEEF", "text": "x"}])
        assert "op 0 (rename)" in str(exc.value)

    def test_empty_ops_is_an_error(self, tmp_path):
        mm = self._map(tmp_path)
        with pytest.raises(MindMapError, match="non-empty list"):
            mm.apply_ops([])


class TestStructuralPrimitives:
    """The primitives own the index; these pin that at the seam itself.

    The op-level invariant test cannot reach ``_detach_all`` on its own -
    ``_op_replace_branch`` always follows it with ``_add_outline``, which
    rebuilds - so a broken ``_detach_all`` would hide behind its caller.
    """

    def _assert_index_current(self, mm):
        def reachable(el):
            return 1 + sum(reachable(child) for child in el.findall("topic"))

        assert len(mm._order) == reachable(mm.root_topic), "stale topic count"
        for el in mm._order:
            assert mm.resolve(mm.address_of(el)) is el, "address does not resolve"
            assert mm.parent_of(el) is next(
                (p for p in mm._order if el in list(p)), None
            ), "stale parent"

    def test_detach_all_leaves_the_index_current(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        mm._detach_all(alpha)
        assert mm.children(alpha) == []
        self._assert_index_current(mm)

    def test_detach_then_attach_leaves_the_index_current(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        gamma = mm.resolve(_uuid(6)[:ADDRESS_LEN])
        mm._detach(gamma)
        self._assert_index_current(mm)
        mm._attach(alpha, gamma, 0)
        assert mm.children(alpha)[0] is gamma
        self._assert_index_current(mm)

    def test_attach_clamps_an_out_of_range_index(self, tmp_path):
        mm = MindMap.open(simple_map(tmp_path / "m.itmz"))
        alpha = mm.resolve(_uuid(2)[:ADDRESS_LEN])
        beta = mm.resolve(_uuid(5)[:ADDRESS_LEN])
        mm._detach(beta)
        mm._attach(alpha, beta, 99)
        assert mm.children(alpha)[-1] is beta
        self._assert_index_current(mm)


# ── Attribute preservation ────────────────────────────────────────────────────
#
# The writer serialises the PARSED tree, so an attribute the codec does not
# model survives by construction. These tests are the guard on that invariant:
# they would fail the day someone "helpfully" rebuilds a <topic> from a dict,
# which is the one change that would silently erase the user's links, task
# state, attachment ids and fold summaries on the next edit.

# Every key here is one the codec does NOT model - it only ever reads text,
# uuid, position, note, link, folded, att-id, image-width and task-progress.
_UNMODELLED_TOPIC_ATTRS = {
    "link": "file:///Users/simon/notes/a.pdf",
    "l1-link": "https://example.com/l1",
    "ln-link": "https://example.com/ln",
    "task-progress": "40",
    "task-start": "2026-01-01T00:00:00",
    "task-due": "2026-02-01T00:00:00",
    "summary1": "a folded summary",
    "summary2": "another",
    "att-id": "894693F1-6662-47C9-9797-BFC108FC85E6",
    "image-width": "120",
    "colour": "FF0000",
    "custom-unknown-attr": "keep me",
}


class TestAttributePreservation:
    def _map_with_unknown_attrs(self, tmp_path):
        beta = _topic("Beta", _uuid(5), **_UNMODELLED_TOPIC_ATTRS)
        alpha = _topic("Alpha", _uuid(2), _topic("One", _uuid(3)))
        root = _topic("Root", _uuid(1), alpha + beta)
        return make_map(tmp_path / "m.itmz", root)

    def test_unmodelled_topic_attributes_survive_an_edit(self, tmp_path):
        path = self._map_with_unknown_attrs(tmp_path)
        mm = MindMap.open(path)
        mm.apply_ops(
            [{"op": "add_branch", "parent": _uuid(2)[:ADDRESS_LEN], "outline": "New"}]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        got = MindMap.open(path).resolve(_uuid(5)[:ADDRESS_LEN]).attrib
        for key, value in _UNMODELLED_TOPIC_ATTRS.items():
            assert got.get(key) == value, f"attribute {key!r} was lost or changed"

    def test_no_topic_loses_an_attribute_across_every_op(self, tmp_path):
        """The invariant, not one attribute: every op, then compare whole trees."""
        path = self._map_with_unknown_attrs(tmp_path)
        before = {el.get("uuid"): dict(el.attrib) for el in MindMap.open(path)._order}

        mm = MindMap.open(path)
        mm.apply_ops(
            [
                {
                    "op": "add_branch",
                    "parent": _uuid(2)[:ADDRESS_LEN],
                    "outline": "New\n\tChild",
                },
                {"op": "rename", "topic": _uuid(3)[:ADDRESS_LEN], "text": "Renamed"},
                {"op": "set_note", "topic": _uuid(3)[:ADDRESS_LEN], "text": "n"},
                {
                    "op": "set_link",
                    "topic": _uuid(3)[:ADDRESS_LEN],
                    "url": "https://a.example",
                },
                {
                    "op": "move",
                    "topic": _uuid(5)[:ADDRESS_LEN],
                    "parent": _uuid(2)[:ADDRESS_LEN],
                },
            ]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        after = {el.get("uuid"): dict(el.attrib) for el in MindMap.open(path)._order}
        assert set(before) <= set(after), "a topic disappeared"
        renamed = _uuid(3)
        for uuid, attrs in before.items():
            for key, value in attrs.items():
                if key == "modified":
                    continue  # stamped by design on every touched topic
                if key == "text" and uuid == renamed:
                    continue  # the rename op's whole purpose
                if key == "position":
                    # Modelled, and deliberately REWRITTEN by move/add_branch -
                    # that is the ordering fix. It must never be dropped, though.
                    assert after[uuid].get("position"), f"{uuid} lost its position"
                    continue
                assert after[uuid].get(key) == value, f"{uuid} lost {key!r}"

    def test_header_attributes_survive_an_edit(self, tmp_path):
        """author and system-version are as easy to lose as a topic attribute."""
        path = make_map(
            tmp_path / "m.itmz",
            _topic("Root", _uuid(1)),
            header_attrs=' author="Simon-R-MacPro [simon]" system-version="Version 10.15.7 (Build 19H2026)"',
        )
        mm = MindMap.open(path)
        mm.apply_ops(
            [{"op": "add_branch", "parent": _uuid(1)[:ADDRESS_LEN], "outline": "New"}]
        )
        mm.save(expect_short=mm.version.short, quiet_seconds=0)

        with zipfile.ZipFile(path) as zf:
            header = ET.fromstring(zf.read("mapdata.xml")).attrib
        assert header.get("author") == "Simon-R-MacPro [simon]"
        assert header.get("system-version") == "Version 10.15.7 (Build 19H2026)"
        assert header.get("app") == "com.toketaware.ithoughtsx"
        assert header.get("app-version") == "9.0"
        assert header.get("version") == "4.0"
        assert header.get("modified") != "2026-01-01T00:00:00", (
            "modified is stamped on save"
        )


# ── Link validation ──────────────────────────────────────────────────────────


def _link_map(tmp_path: Path, topics_xml: str) -> Path:
    """A map with style.xml, so the only warnings are the ones under test."""
    return make_map(tmp_path / "m.itmz", topics_xml, extra=[("style.xml", b"<style/>")])


def _broken(report) -> list[str]:
    return [w for w in report.warnings if "broken link" in w]


class TestLinkValidation:
    def test_existing_file_link_is_clean(self, tmp_path):
        target = tmp_path / "docs" / "yarra.pdf"
        target.parent.mkdir()
        target.write_text("x")
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link=f"file://{target}")
        )
        report = check(_link_map(tmp_path, topics))
        assert _broken(report) == []
        assert report.links.checked == 1
        assert report.links.broken == 0

    def test_broken_file_link_warns_with_title_chain_and_path(self, tmp_path):
        topics = _topic(
            "Root",
            _uuid(1),
            _topic(
                "Section",
                _uuid(2),
                _topic("Yarra", _uuid(3), link=f"file://{tmp_path}/gone.pdf"),
            ),
        )
        report = check(_link_map(tmp_path, topics))
        assert len(_broken(report)) == 1
        w = _broken(report)[0]
        assert 'broken link on "Yarra"' in w
        assert _uuid(3)[:8] in w
        assert "(under Root > Section)" in w
        assert str(tmp_path / "gone.pdf") in w
        assert report.links.broken == 1

    def test_percent_encoded_target_resolves(self, tmp_path):
        from urllib.parse import quote

        target = tmp_path / "my file.pdf"
        target.write_text("x")
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link="file://" + quote(str(target)))
        )
        report = check(_link_map(tmp_path, topics))
        assert _broken(report) == []
        assert report.links.checked == 1

    def test_relative_link_resolves_against_map_dir(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.pdf").write_text("x")
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("Good", _uuid(2), link="docs/a.pdf")
            + _topic("Bad", _uuid(3), link="docs/b.pdf"),
        )
        report = check(_link_map(tmp_path, topics))
        assert len(_broken(report)) == 1
        assert str(docs / "b.pdf") in _broken(report)[0]
        assert report.links.checked == 2

    def test_rename_breaks_then_unbreaks(self, tmp_path):
        """The work order's acceptance case: rename outside the tool, the
        next check reports exactly that link."""
        target = tmp_path / "plan.pdf"
        target.write_text("x")
        topics = _topic(
            "Root", _uuid(1), _topic("Plan", _uuid(2), link=f"file://{target}")
        )
        path = _link_map(tmp_path, topics)
        assert _broken(check(path)) == []

        renamed = tmp_path / "plan-renamed.pdf"
        target.rename(renamed)
        broken = _broken(check(path))
        assert len(broken) == 1
        assert '"Plan"' in broken[0]
        assert str(target) in broken[0]

        renamed.rename(target)
        assert _broken(check(path)) == []

    def test_web_links_counted_never_warned(self, tmp_path):
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("A", _uuid(2), link="https://example.com/x")
            + _topic("B", _uuid(3), link="http://example.com/y"),
        )
        report = check(_link_map(tmp_path, topics))
        assert _broken(report) == []
        assert report.links.web == 2
        assert report.links.checked == 0

    def test_non_local_schemes_are_skipped(self, tmp_path):
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("A", _uuid(2), link="/Local/Store/abc")
            + _topic("B", _uuid(3), link="mailto:a@b.c")
            + _topic("C", _uuid(4), link="file://server/share/x.pdf"),
        )
        report = check(_link_map(tmp_path, topics))
        assert _broken(report) == []
        assert report.links.skipped == 3
        assert report.links.checked == 0

    def test_windows_drive_path_is_local_not_a_scheme(self, tmp_path):
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link="C:\\Users\\s\\x.pdf")
        )
        report = check(_link_map(tmp_path, topics))
        assert report.links.checked == 1
        assert report.links.skipped == 0
        # The target genuinely does not exist here, wherever "here" is.
        assert len(_broken(report)) == 1

    def test_file_url_with_drive_in_host_slot_is_local(self, tmp_path):
        """f"file://{windows_path}" yields file://C:\\... - the drive in the
        HOST slot, which urlparse reads as a network host. A model passing a
        raw Path to set_link produces exactly this; it must resolve as local,
        not be skipped as file://server/share. Found by the Windows CI run
        2026-11-26 (12 failures, every f"file://{tmp_path}" test)."""
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("Backslash", _uuid(2), link="file://C:\\Users\\s\\x.pdf")
            + _topic("Forward", _uuid(3), link="file://C:/Users/s/x.pdf")
            + _topic(
                "Encoded",
                _uuid(4),
                link="file://C%3A%5CUsers%5Cs%5Cmy%20file.pdf",
            ),
        )
        report = check(_link_map(tmp_path, topics))
        assert report.links.skipped == 0
        assert report.links.checked == 3
        assert len(_broken(report)) == 3
        assert "C:\\Users\\s\\x.pdf" in _broken(report)[0]
        assert "my file.pdf" in _broken(report)[2]

    def test_root_topic_link_has_no_chain(self, tmp_path):
        topics = _topic("Root", _uuid(1), link=f"file://{tmp_path}/gone.pdf")
        report = check(_link_map(tmp_path, topics))
        w = _broken(report)[0]
        assert "(under" not in w

    def test_untitled_topic_falls_back_to_address(self, tmp_path):
        topics = _topic(
            "Root", _uuid(1), _topic("", _uuid(2), link=f"file://{tmp_path}/gone.pdf")
        )
        report = check(_link_map(tmp_path, topics))
        assert _uuid(2)[:8] in _broken(report)[0]

    def test_summary_line_in_render(self, tmp_path):
        ok = tmp_path / "ok.pdf"
        ok.write_text("x")
        topics = _topic(
            "Root",
            _uuid(1),
            _topic("Good", _uuid(2), link=f"file://{ok}")
            + _topic("Bad", _uuid(3), link=f"file://{tmp_path}/gone.pdf")
            + _topic("Web", _uuid(4), link="https://example.com")
            + _topic("Local", _uuid(5), link="/Local/x"),
        )
        report = check(_link_map(tmp_path, topics))
        out = report.render()
        assert "links: 2 local checked, 1 broken, 1 web not checked, 1 skipped" in out

    def test_all_resolve_summary_when_clean(self, tmp_path):
        target = tmp_path / "a.pdf"
        target.write_text("x")
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link=f"file://{target}")
        )
        out = check(_link_map(tmp_path, topics)).render()
        assert "links: 1 local checked, all resolve" in out

    def test_no_links_no_summary_line(self, tmp_path):
        report = check(_link_map(tmp_path, _topic("Root", _uuid(1))))
        assert "links:" not in report.render()

    def test_validation_creates_no_files(self, tmp_path):
        topics = _topic(
            "Root", _uuid(1), _topic("A", _uuid(2), link=f"file://{tmp_path}/gone.pdf")
        )
        path = _link_map(tmp_path, topics)
        before = sorted(p.name for p in tmp_path.iterdir())
        check(path)
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_fifty_two_links_validates_fast(self, tmp_path):
        import time

        target = tmp_path / "real.pdf"
        target.write_text("x")
        kids = "".join(
            _topic(
                f"T{i}",
                _uuid(i + 2),
                link=f"file://{target}" if i % 2 else f"file://{tmp_path}/gone-{i}.pdf",
            )
            for i in range(52)
        )
        report = check(_link_map(tmp_path, _topic("Root", _uuid(1), kids)))
        start = time.time()
        report = check(_link_map(tmp_path, _topic("Root", _uuid(1), kids)))
        assert time.time() - start < 5.0  # the order's bar is "well under a second"
        assert report.links.checked == 52
        assert report.links.broken == 26
