"""iThoughts (.itmz) mind-map codec.

Pure format work - parse, address, render, mutate, write - with no dependency
on the tool framework or on agent state, so it can be tested on its own.
``tools/mindmap.py`` is the thin ``@tool`` wrapper.

The rules encoded here are not preferences; each one is a bug already shipped
in the ithoughtsx repo (work order 2026-11-10, section 2):

* ``.itmz`` is a ZIP. Every entry other than ``mapdata.xml`` is copied
  byte-for-byte, in the same order, with the same ``compress_type``.
* Never emit ZIP directory entries - the original renders a blank page for an
  archive containing them (valid CRCs, no crash).
* Every topic attribute is preserved: the parsed tree is mutated in place and
  a ``<topic>`` is never rebuilt from a dict.
* Every topic needs a ``uuid`` and a ``position``; ``note`` is an attribute
  with ``&#10;`` for newlines.
* A save is not byte-identical to the file opened. Only ``mapdata.xml`` may
  differ, and only semantically.

Two behaviours that are easy to get wrong and are pinned by tests:

* **Addressing is stateless.** A topic is addressed by an 8-character prefix
  of its own ``uuid``, extended only on collision. Session handles die at a
  chain restart, which is routine in this harness (22 restarts in 79 h on one
  project), and a summary then carries a handle nothing can resolve.
* **The version baseline is stateless too.** ``read_map`` prints the version;
  ``edit_map`` accepts it back. The in-process baseline is a convenience, not
  the mechanism.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import stat
import time
import uuid as uuid_mod
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

# ── Tunables ────────────────────────────────────────────────────────────────

ADDRESS_LEN = 8  # uuid prefix length; extended only on collision
NO_ADDRESS = "????????"  # printed for a topic with no uuid: it has no address
QUIET_SECONDS_DEFAULT = 20  # §12.3: covers the measured 9 s AND 20 s autosaves
INDENT_TAB_WIDTH = 4  # §12.4: expand tabs before measuring the indent unit
READ_BUDGET_CHARS = 20000
READ_BUDGET_LINES = 400
NOTE_PREVIEW_CHARS = 400  # §12.2: the drilled topic's own note only
QUERY_LIMIT = 20
TEXT_PREVIEW_CHARS = 70
TEXT_WINDOW_FLOOR = 2000  # below this many chars left, the text block is withheld
QUERY_PATH_CHARS = 120  # ancestor path cap in a query hit
QUERY_CONTEXT_CHARS = 200  # context window around a query match
NEWLINE_MARK = " \u23ce "  # marks a newline inside a one-line tree entry
# The y step when a topic is placed before the first or after the last sibling.
# Real sibling gaps measure 28-50 px (median 42); the number is a courtesy, not
# a layout - the app re-spaces the branch and only the ORDER is load-bearing.
SIBLING_GAP = 60.0

MAPDATA = "mapdata.xml"
STYLE = "style.xml"
BOM = b"\xef\xbb\xbf"

# The bundled default look for a created map. Package data, so it must be
# listed in agent13.spec and pyproject's wheel artifacts or it vanishes from
# the built artifact only.
DEFAULT_STYLE_PATH = Path(__file__).with_name("mindmap_styles") / "outline.style.xml"

_POS_RE = re.compile(r"^\{\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\}$")
# Address-SHAPED, not hex: an address is a prefix of the hyphen-stripped uuid,
# whatever that uuid contains. Requiring hex made `resolve` reject the very
# address `address_of` had just printed for a map whose uuids are not UUID
# format (`CHILD-NOTE` -> `CHILDNOT`), so read_map said "here is the address"
# and the next edit said "no such address". The 6-char floor is what keeps a
# short topic TEXT from being read as an address prefix.
_ADDR_RE = re.compile(r"^[^\s]{6,32}$")


def _hex_key(uuid_text: str) -> str:
    """A uuid with its hyphens removed, lowercased.

    Addresses are prefixes of THIS, so an extended address never contains a
    hyphen and stays a plain hex string the model can copy.
    """
    return (uuid_text or "").replace("-", "").lower()


class MindMapError(Exception):
    """A refusal or a malformed input, with a message written for the model."""


# ── Version + baselines ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Version:
    """What the file looked like when it was read."""

    size: int
    mtime_ns: int
    sha256: str

    @property
    def short(self) -> str:
        return self.sha256[:8]


def version_of(path: Path) -> Version:
    data = path.read_bytes()
    st = path.stat()
    return Version(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=hashlib.sha256(data).hexdigest())


# ── Style templates ─────────────────────────────────────────────────────────


def default_style() -> bytes:
    """The bundled default ``style.xml``."""
    return DEFAULT_STYLE_PATH.read_bytes()


def read_style(path: Path) -> bytes:
    """The ``style.xml`` member of an .itmz, verbatim.

    Reads ONLY that member. ``MindMap.open`` would read every entry and parse
    the whole mapdata - 49 MB and 1,791 topics on a real journal - to fetch
    1 KB of style.
    """
    if not path.exists():
        raise MindMapError(
            f"no style template at {path}. Omit style_from to use the bundled "
            "default style."
        )
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.read(STYLE)
    except zipfile.BadZipFile:
        raise MindMapError(
            f"{path} is not a readable .itmz - a .itmz is a ZIP archive. Omit "
            "style_from to use the bundled default style."
        )
    except KeyError:
        raise MindMapError(
            f"{path} has no {STYLE} - it is not a styled iThoughts map. Omit "
            "style_from to use the default style."
        )


def _zip_info(name: str) -> zipfile.ZipInfo:
    """A fresh archive entry: DEFLATE, dated now.

    Neither is the default, and ``_write_archive`` copies both from the entry
    it is handed: a bare ``ZipInfo`` is STORED and stamped 1980-01-01.
    """
    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    return info

# ── Outline parsing ─────────────────────────────────────────────────────────


def _outline_text(text: str) -> str:
    """Apply the two text rules of §3.3.

    A leading backslash forces a literal; otherwise one leading bullet marker
    (``- ``, ``* ``, ``+ ``) is decoration and is stripped. The space is
    required, because real maps contain topics whose text begins with ``-``
    (``-icc``, ``--link othermachine:nameforinside``).
    """
    if text.startswith("\\"):
        return text[1:]
    for marker in ("- ", "* ", "+ "):
        if text.startswith(marker):
            return text[2:]
    return text


def parse_outline(outline: str) -> list[tuple[int, str]]:
    """Parse an indented outline into ``[(level, text)]``.

    Tabs are expanded to ``INDENT_TAB_WIDTH`` spaces *before* measuring (§12.4),
    so a tab-indented line and a 4-space-indented line are the same level. The
    unit is the shortest non-zero indent; a line whose indent is not a multiple
    of it is an error naming the line, and so is a jump of more than one level
    - silently flattening either would change the tree without saying so.
    """
    expanded = outline.replace("\t", " " * INDENT_TAB_WIDTH)
    rows: list[tuple[int, int, str]] = []
    for lineno, raw in enumerate(expanded.splitlines(), 1):
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        rows.append((lineno, indent, raw.strip()))

    if not rows:
        raise MindMapError(
            "the outline is empty - nothing to add. Pass one topic per line, "
            "indented with tabs (or spaces) for depth."
        )

    indents = [i for _, i, _ in rows if i > 0]
    unit = min(indents) if indents else 0

    levels: list[int] = []
    for lineno, indent, text in rows:
        if unit and indent % unit:
            raise MindMapError(
                f"outline line {lineno}: an indent of {indent} spaces is not a "
                f"multiple of the {unit}-space step used by the rest of the "
                f"outline ({text!r}). Use consistent indentation."
            )
        levels.append(indent // unit if unit else 0)

    base = min(levels)
    levels = [level - base for level in levels]

    for idx in range(1, len(levels)):
        if levels[idx] - levels[idx - 1] > 1:
            raise MindMapError(
                f"outline line {rows[idx][0]}: this line is {levels[idx] - levels[idx - 1]} "
                f"levels deeper than the line above ({rows[idx][2]!r}). Indent one "
                "level at a time - a bigger jump has no unambiguous tree."
            )

    return [(levels[i], _outline_text(rows[i][2])) for i in range(len(rows))]


# ── The map ─────────────────────────────────────────────────────────────────


class VersionStore:
    """Per-path version baselines and own-write stamps for one process.

    Process-level by nature, not per-instance: ``edit_map`` reads the baseline
    BEFORE opening the map (so a baseline the open created itself cannot
    satisfy the guard), and that read has to survive from the ``read_map`` call
    to the ``edit_map`` call. It is also NOT the mechanism - the version string
    in the tool result is, so a chain restart costs a re-read, not correctness.

    An object rather than two module dicts so a test can own one outright, and
    so ``forget_all()`` names the operation instead of naming its caller
    (it used to be ``_reset_state_for_tests``, a private function whose only
    public purpose was to be imported by tests).
    """

    # 1 ms, for filesystem timestamp granularity
    OWN_WRITE_SLACK_NS = 1_000_000

    def __init__(self) -> None:
        self._baselines: dict[str, Version] = {}
        self._own_writes: dict[str, int] = {}

    def remember(self, path: Path, version: Version) -> None:
        self._baselines[str(path)] = version

    def baseline(self, path: Path) -> Optional[Version]:
        return self._baselines.get(str(path))

    def note_own_write(self, path: Path, mtime_ns: int) -> None:
        """Record the mtime our own write produced.

        Deliberately NOT expired by wall clock - the ithoughtsx app shipped
        exactly that bug (a phantom "file changed on disk" dialog whenever no
        tick ran for >60 s).
        """
        self._own_writes[str(path)] = mtime_ns

    def is_own_write(self, path: Path, mtime_ns: int) -> bool:
        recorded = self._own_writes.get(str(path))
        return recorded is not None and mtime_ns <= recorded + self.OWN_WRITE_SLACK_NS

    def forget_all(self) -> None:
        """Drop every baseline and own-write stamp (a chain restart, or a test)."""
        self._baselines.clear()
        self._own_writes.clear()


# The process default. Tools use this one; tests may construct their own.
STORE = VersionStore()


class MindMap:
    """A parsed .itmz. Mutate, then ``save()`` - one atomic write per call."""

    def __init__(
        self,
        path: Path,
        entries: list[tuple[zipfile.ZipInfo, bytes]],
        root: ET.Element,
        version: Version,
        store: Optional[VersionStore] = None,
    ):
        self.path = path
        self._entries = entries
        self.root = root
        self.version = version
        self._store = store if store is not None else STORE
        self._topics_el = root.find("topics")
        if self._topics_el is None:
            raise MindMapError(
                f"{path} has no <topics> element - it is not an iThoughts map "
                "(or it is corrupt)."
            )
        self._index: dict[str, ET.Element] = {}
        self._parent: dict[ET.Element, ET.Element] = {}
        self._order: list[ET.Element] = []
        self._build_index()

    # -- opening ------------------------------------------------------------

    @classmethod
    def open(cls, path: Path, store: Optional[VersionStore] = None) -> "MindMap":
        if not path.exists():
            raise MindMapError(_missing_map_message(path))
        if not path.is_file():
            raise MindMapError(f"{path} is not a file.")
        try:
            with zipfile.ZipFile(path) as zf:
                entries = [(info, zf.read(info.filename)) for info in zf.infolist() if not info.is_dir()]
        except zipfile.BadZipFile:
            raise MindMapError(
                f"{path} is not a .itmz archive (a .itmz is a ZIP containing "
                "mapdata.xml)."
            )
        raw = next((data for info, data in entries if info.filename == MAPDATA), None)
        if raw is None:
            raise MindMapError(f"{path} has no {MAPDATA} entry - not an iThoughts map.")

        bom = raw.startswith(BOM)
        if bom:
            raw = raw[len(BOM):]
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise MindMapError(f"{path}: {MAPDATA} is not well-formed XML ({exc}).")
        if root.tag != "iThoughts":
            raise MindMapError(
                f"{path}: root element is <{root.tag}>, expected <iThoughts>."
            )

        version = version_of(path)
        store = store if store is not None else STORE
        store.remember(path, version)
        return cls(path, entries, root, version, store)

    @classmethod
    def create(
        cls,
        path: Path,
        root_text: str = "Central Topic",
        style: bytes = b"",
        store: Optional[VersionStore] = None,
    ) -> "MindMap":
        """A NEW map in memory, ready for ``save(create=True)``.

        The header is the one our own app writes for File > New
        (``app/js/io/opening.js newMap()``), plus the empty ``<relationships>``
        the original app writes and our app omits. ``style`` is the
        ``style.xml`` bytes, copied verbatim; empty means the map renders with
        the app's own defaults, which is what our File > New produces.
        """
        now = _now_iso()
        root = ET.Element(
            "iThoughts",
            {
                "version": "4.0",
                "app": "com.ithoughtsx.web",
                "app-version": "1.0",
                "modified": now,
            },
        )
        topics = ET.SubElement(root, "topics")
        ET.SubElement(
            topics,
            "topic",
            {
                "uuid": str(uuid_mod.uuid4()).upper(),
                "position": "{0, 0}",
                "text": root_text,
                "created": now,
                "modified": now,
            },
        )
        ET.SubElement(root, "relationships")

        entries = [(_zip_info(MAPDATA), b""), (_zip_info(STYLE), style)]
        return cls(path, entries, root, Version(size=0, mtime_ns=0, sha256=""), store)

    # -- indexing -----------------------------------------------------------

    def _build_index(self) -> None:
        self._index.clear()
        self._parent.clear()
        self._order = []

        def walk(el: ET.Element, parent: Optional[ET.Element]) -> None:
            self._order.append(el)
            if parent is not None:
                self._parent[el] = parent
            u = _hex_key(el.get("uuid") or "")
            if u:
                self._index[u] = el
            for child in el.findall("topic"):
                walk(child, el)

        for top in self._topics_el.findall("topic"):
            walk(top, None)

    @property
    def root_topic(self) -> ET.Element:
        tops = self._topics_el.findall("topic")
        if not tops:
            raise MindMapError(f"{self.path} has no topics.")
        return tops[0]

    def children(self, el: ET.Element) -> list[ET.Element]:
        return el.findall("topic")

    def parent_of(self, el: ET.Element) -> Optional[ET.Element]:
        return self._parent.get(el)

    def topic_count(self) -> int:
        return len(self._order)

    # -- addressing ---------------------------------------------------------

    def address_of(self, el: ET.Element) -> str:
        """8-char uuid prefix, extended only as far as uniqueness requires."""
        u = _hex_key(el.get("uuid") or "")
        if not u:
            return NO_ADDRESS
        length = ADDRESS_LEN
        while length < len(u):
            prefix = u[:length]
            if sum(1 for other in self._index if other.startswith(prefix)) == 1:
                return prefix.upper()
            length += 1
        return u.upper()

    def resolve(self, ref: str) -> ET.Element:
        """Resolve an address, or a topic's exact text.

        Text is accepted because a model will sometimes paste the text it just
        read instead of the address; a unique match is unambiguous, several are
        an error that lists them.
        """
        ref = (ref or "").strip()
        if not ref:
            raise MindMapError("no topic given - pass the address shown in [brackets] by read_map.")
        if ref == NO_ADDRESS:
            raise MindMapError(
                f"{NO_ADDRESS} is not an address - read_map prints it for a topic that has "
                "no uuid, so it cannot be addressed. Use that topic's exact text instead, "
                "or skip it."
            )

        # An EXACT uuid always resolves, whatever its length or shape: a uuid
        # shorter than the address floor (`ABC`) or a full hyphenated uuid
        # pasted back from elsewhere would otherwise fall through to the text
        # match and fail. This is the "read_map printed it, so it resolves"
        # guarantee at its cheapest - one dict lookup.
        exact_key = _hex_key(ref)
        if exact_key in self._index:
            return self._index[exact_key]

        if _ADDR_RE.match(ref):
            key = _hex_key(ref)
            hits = [u for u in self._index if u.startswith(key)]
            if len(hits) == 1:
                return self._index[hits[0]]
            if len(hits) > 1:
                listing = ", ".join(self.address_of(self._index[u]) for u in hits[:5])
                raise MindMapError(
                    f"address {ref!r} is ambiguous - it matches {len(hits)} topics "
                    f"({listing}). Use a longer prefix."
                )

        exact = [el for el in self._order if (el.get("text") or "").lower() == ref.lower()]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            listing = ", ".join(f"{self.address_of(el)} {el.get('text')!r}" for el in exact[:5])
            raise MindMapError(
                f"{ref!r} matches {len(exact)} topics with that exact text ({listing}). "
                "Use the address instead."
            )

        raise MindMapError(self._unknown_topic_message(ref))

    def _unknown_topic_message(self, ref: str) -> str:
        texts = [(el.get("text") or "") for el in self._order]
        nearest = difflib.get_close_matches(ref, texts, n=3, cutoff=0.3)
        lines = [f"no topic matches {ref!r}."]
        if nearest:
            lines.append("Nearest by text:")
            for text in nearest:
                el = next(e for e in self._order if (e.get("text") or "") == text)
                lines.append(f"  {self.address_of(el)}  {_preview(text)}")
        else:
            lines.append("Call read_map to see the addresses.")
        return "\n".join(lines)

    def path_of(self, el: ET.Element) -> str:
        parts = []
        cur: Optional[ET.Element] = el
        while cur is not None:
            parts.append(cur.get("text") or "")
            cur = self._parent.get(cur)
        return " > ".join(reversed(parts))

    # -- reading ------------------------------------------------------------

    def stats(self) -> tuple[int, int, int]:
        """(topics, levels, folded)."""
        levels = 0
        for el in self._order:
            depth = 1
            cur = self._parent.get(el)
            while cur is not None:
                depth += 1
                cur = self._parent.get(cur)
            levels = max(levels, depth)
        folded = sum(1 for el in self._order if el.get("folded") in ("1", "true"))
        return len(self._order), levels, folded

    def render(
        self,
        topic: Optional[str] = None,
        depth: int = 1,
        query: str = "",
        text_offset: int = 0,
    ) -> str:
        if query:
            if topic:
                raise MindMapError(
                    "query searches the whole map, so topic is not used. Drop "
                    "topic, or drop query and read the topic itself."
                )
            if text_offset:
                raise MindMapError(
                    "query searches the whole map, so text_offset is not used. "
                    "Drop text_offset, or drop query and read the topic's own "
                    "text with depth=0."
                )
            return self._render_query(query)

        el = self.root_topic if not topic else self.resolve(topic)

        if text_offset < 0:
            raise MindMapError(
                f"text_offset must be 0 or more, got {text_offset}. This topic's "
                f"text is {len(el.get('text') or ''):,} chars - pass an offset "
                "within that."
            )

        # 0 is a real value (the topic, no children), so `depth or 1` would
        # silently turn it into 1.
        depth = 1 if depth is None else int(depth)
        depth = max(0, depth)

        nodes = self._collect(el, depth)
        header, meta, body = self._format_parts(
            el, nodes, drilled=bool(topic), show_children=depth >= 1
        )
        tree = "\n".join(header + meta + body)

        if len(nodes) > READ_BUDGET_LINES or len(tree) > READ_BUDGET_CHARS:
            raise MindMapError(self._budget_refusal(el, topic, depth, len(nodes), len(tree)))

        block = self._text_block(
            el,
            text_offset,
            remaining_chars=READ_BUDGET_CHARS - len(tree),
            remaining_lines=READ_BUDGET_LINES - len(header + meta + body),
        )
        return "\n".join(header + block + meta + body)

    def _collect(self, el: ET.Element, depth: int) -> list[tuple[int, ET.Element]]:
        """``depth`` counts the levels BELOW ``el``: 1 = its children, 0 = none."""
        out: list[tuple[int, ET.Element]] = [(0, el)]
        if depth >= 1:
            for child in self.children(el):
                for level, node in self._collect(child, depth - 1):
                    out.append((level + 1, node))
        return out

    def _format_parts(
        self,
        el: ET.Element,
        nodes: list[tuple[int, ET.Element]],
        drilled: bool,
        show_children: bool,
    ) -> tuple[list[str], list[str], list[str]]:
        """(header, meta, body) - the text block goes between header and meta.

        The header carries identity and structure only. A topic's own text is a
        block of its own, so a long one is never squeezed onto a header line and
        a short one is never printed twice.
        """
        total, levels, folded = self.stats()
        header: list[str] = []
        if drilled:
            kids = len(self.children(el))
            header.append(f"topic: {self.address_of(el)}  {_child_note(kids)}")
        else:
            header.append(f"map: {self.path}  ({total} topics, {levels} levels, {folded} folded)")
            header.append(f"root: [{self.address_of(el)}]")

        meta: list[str] = []
        note = el.get("note")
        if note:
            meta.append(f"note: {_note_preview(note)}")
        meta.append(f"version: {self.version.short}")
        meta.append("")

        body: list[str] = []
        for level, node in nodes[1:]:
            indent = "  " * level
            addr = self.address_of(node)
            text = _flatten(node.get("text") or "")
            flags = _flags(node)
            suffix = f"  ({', '.join(flags)})" if flags else ""
            body.append(f"{indent}{addr}  {text}{suffix}")

        if show_children and not nodes[1:]:
            body.append("  (no children)")

        first_child = self.children(el)
        if first_child:
            example = self.address_of(first_child[0])
            if body:
                body.append("")
            body.append(
                f'[drill: read_map(path, topic="{example}")  |  depth=2 for two levels'
                f'  |  query="text" to search the whole map]'
            )
        return header, meta, body

    def _text_block(
        self,
        el: ET.Element,
        offset: int,
        remaining_chars: int,
        remaining_lines: int,
    ) -> list[str]:
        """The topic's own text, as a block paged by ``offset``.

        Bounded by BOTH budgets. Below ``TEXT_WINDOW_FLOOR`` chars the block is
        withheld rather than emitted as a window too small to be worth a round
        trip - and a zero-width window would repeat itself for ever, because the
        footer's offset would not advance.
        """
        text = el.get("text") or ""
        if not text:
            return []

        addr = self.address_of(el)
        if offset >= len(text):
            return [f"text: nothing at offset {offset:,} - the text is {len(text):,} chars"]

        more = len(text) - offset
        withheld = [
            f"text: the tree used the budget - call read_map(path, "
            f'topic="{addr}", depth=0) to read this topic\'s text'
        ]
        if more > remaining_chars and remaining_chars < TEXT_WINDOW_FLOOR:
            return withheld

        chunk = text[offset : offset + max(0, remaining_chars)]
        # Cut at a LINE boundary so consecutive windows concatenate exactly: the
        # join separator supplies the newline, and the footer offset steps past
        # it. A single line longer than the budget is the one case that has to
        # cut mid-line - the offset is still exact, so the caller can continue.
        if chunk and offset + len(chunk) < len(text):
            cut = chunk.rfind("\n")
            if cut > 0:
                chunk = chunk[:cut]
        # The block's own "text:" line and its footer line count against the
        # budget too, or a window can push the result two lines over.
        budget_lines = max(0, remaining_lines - 2)
        lines = chunk.split("\n")
        if len(lines) > budget_lines:
            chunk = "\n".join(lines[:budget_lines])
        if not chunk:
            return withheld

        block = ["text:"]
        block += [f"  {line}" for line in chunk.split("\n")]
        end = offset + len(chunk)
        if end < len(text) and text[end] == "\n":
            end += 1
        if end < len(text):
            block.append(
                f"  [text continues at offset {end:,} of {len(text):,} - call "
                f'read_map(path, topic="{addr}", depth=0, text_offset={end})]'
            )
        return block

    def _budget_refusal(self, el: ET.Element, topic: Optional[str], depth: int, count: int, chars: int) -> str:
        where = f"depth={depth} on {self.address_of(el)}" if topic else f"depth={depth} from the root"
        kids = self.children(el)[:8]
        options = ", ".join(f"{self.address_of(k)} {_preview(k.get('text') or '', 40)}" for k in kids)
        if depth > 1:
            advice = "Call again with depth=1, or drill into one child"
        else:
            advice = (
                "Call again with depth=0 to read this topic's own text, or drill "
                "into one child"
            )
        return (
            f"{where} would return {count:,} topics (~{chars // 1000}k chars), over the "
            f"{READ_BUDGET_LINES}-line / {READ_BUDGET_CHARS // 1000}k-char budget. Nothing was "
            f"truncated - a truncated tree reads as an empty branch. {advice}: {options}"
        )

    def _ancestor_path(self, el: ET.Element) -> str:
        """Ancestors only, elided from the START so the parent survives."""
        parts: list[str] = []
        cur = self._parent.get(el)
        while cur is not None:
            parts.append(_flatten(cur.get("text") or ""))
            cur = self._parent.get(cur)
        if not parts:
            return "(root)"
        path = " > ".join(reversed(parts))
        if len(path) > QUERY_PATH_CHARS:
            path = "\u2026" + path[-(QUERY_PATH_CHARS - 1) :]
        return path

    def _hit_context(self, el: ET.Element, needle: str, in_note: bool) -> str:
        """~QUERY_CONTEXT_CHARS around the match, from whichever field matched."""
        if in_note:
            return _note_preview(el.get("note") or "", needle, QUERY_CONTEXT_CHARS)
        return _context_window(el.get("text") or "", needle, QUERY_CONTEXT_CHARS)

    def _render_query(self, query: str) -> str:
        needle = query.lower()
        hits: list[tuple[ET.Element, bool]] = []
        for el in self._order:
            if needle in (el.get("text") or "").lower():
                hits.append((el, False))
            elif needle in (el.get("note") or "").lower():
                hits.append((el, True))
        if not hits:
            return (
                f"no topic or note contains {query!r} ({self.topic_count():,} topics "
                f"searched). Try a shorter or different word."
            )
        lines = [f'matches for {query!r}: {len(hits)} of {self.topic_count():,} topics']
        lines.append(f"version: {self.version.short}")
        lines.append("")
        for el, in_note in hits[:QUERY_LIMIT]:
            kids = len(self.children(el))
            lines.append(f"{self.address_of(el)}  {self._ancestor_path(el)}  {_child_note(kids)}")
            lines.append(f"    {self._hit_context(el, needle, in_note)}     [drill for the full text]")
        if len(hits) > QUERY_LIMIT:
            lines.append("")
            lines.append(f"[{len(hits) - QUERY_LIMIT} more matches - narrow the query]")

        text = "\n".join(lines)
        # Unreachable with the caps above, and here so that it cannot regress:
        # a query result used to be the one read path with no budget at all.
        if len(text) > READ_BUDGET_CHARS or len(lines) > READ_BUDGET_LINES:
            raise MindMapError(
                f"the query {query!r} matched {len(hits):,} topics and the result is "
                f"over the {READ_BUDGET_LINES}-line / {READ_BUDGET_CHARS // 1000}k-char "
                "budget. Nothing was truncated - narrow the query."
            )
        return text

    # -- editing ------------------------------------------------------------

    def apply_ops(self, ops: list[dict]) -> list[str]:
        """Apply a batch. Returns the human-readable report lines.

        All-or-nothing is structural: the caller writes only if this returns,
        and a failed op raises before any write happens.
        """
        if not isinstance(ops, list) or not ops:
            raise MindMapError(
                "ops must be a non-empty list of operations. Each op is an object "
                'with "op" plus its arguments, e.g. {"op": "add_branch", '
                '"parent": "4F76A79F", "outline": "Docker\\n\\t-icc"}.'
            )
        report: list[str] = []
        for index, op in enumerate(ops):
            if not isinstance(op, dict):
                raise MindMapError(f"op {index}: expected an object, got {type(op).__name__}.")
            name = op.get("op") or op.get("kind") or op.get("action")
            handler = {
                "add_branch": self._op_add_branch,
                "replace_branch": self._op_replace_branch,
                "rename": self._op_rename,
                "move": self._op_move,
                "delete": self._op_delete,
                "set_note": self._op_set_note,
                "set_link": self._op_set_link,
            }.get(name)
            if handler is None:
                raise MindMapError(
                    f"op {index}: unknown op {name!r}. Valid ops: add_branch, "
                    "replace_branch, rename, move, delete, set_note, set_link.\n"
                    "Nothing was written."
                )
            try:
                report.extend(handler(index, op))
            except MindMapError as exc:
                message = str(exc)
                if "Nothing was written" not in message:
                    message += "\nNothing was written."
                raise MindMapError(f"op {index} ({name}): {message}")
        return report

    def _op_add_branch(self, index: int, op: dict) -> list[str]:
        parent = self.resolve(op.get("parent") or "")
        outline = op.get("outline")
        if not isinstance(outline, str):
            raise MindMapError('"outline" must be a string of indented lines.')
        created = self.add_outline(parent, outline)
        lines = [f"added {len(created)} topics under {self.address_of(parent)} {_preview(parent.get('text') or '')}:"]
        for el in created:
            lines.append(f"  {self.address_of(el)}  {_preview(el.get('text') or '')}")
        return lines

    def _op_replace_branch(self, index: int, op: dict) -> list[str]:
        topic = self.resolve(op.get("topic") or "")
        outline = op.get("outline")
        if not isinstance(outline, str):
            raise MindMapError('"outline" must be a string of indented lines.')
        removed = len(self.children(topic))
        self._detach_all(topic)
        created = self._add_outline(topic, parse_outline(outline))
        lines = [
            f"replaced {removed} children of {self.address_of(topic)} "
            f"{_preview(topic.get('text') or '')} with {len(created)}:"
        ]
        for el in created:
            lines.append(f"  {self.address_of(el)}  {_preview(el.get('text') or '')}")
        return lines

    def _op_rename(self, index: int, op: dict) -> list[str]:
        topic = self.resolve(op.get("topic") or "")
        text = op.get("text")
        if not isinstance(text, str) or not text.strip():
            raise MindMapError('"text" must be a non-empty string.')
        old = topic.get("text") or ""
        topic.set("text", text)
        topic.set("modified", _now_iso())
        return [f"renamed {self.address_of(topic)}: {_preview(old)!r} -> {_preview(text)!r}"]

    def _op_move(self, index: int, op: dict) -> list[str]:
        topic = self.resolve(op.get("topic") or "")
        if topic is self.root_topic:
            raise MindMapError("the root topic cannot be moved.")
        new_parent = self.resolve(op.get("parent") or "")
        cur: Optional[ET.Element] = new_parent
        while cur is not None:
            if cur is topic:
                raise MindMapError("cannot move a topic into its own subtree.")
            cur = self._parent.get(cur)
        old_parent = self.parent_of(topic)
        if old_parent is None:
            raise MindMapError("that topic has no parent.")
        self._detach(topic)
        self._attach(new_parent, topic, op.get("index"))
        # The app draws by position, so an XML move without a position is a
        # no-op on screen - the reported bug this op used to have.
        self._place_among(new_parent, topic, self.children(new_parent).index(topic))
        return [
            f"moved {self.address_of(topic)} {_preview(topic.get('text') or '')} "
            f"under {self.address_of(new_parent)}"
        ]

    def _op_delete(self, index: int, op: dict) -> list[str]:
        topic = self.resolve(op.get("topic") or "")
        if topic is self.root_topic:
            raise MindMapError("the root topic cannot be deleted.")
        parent = self.parent_of(topic)
        if parent is None:
            raise MindMapError("that topic has no parent.")
        keep = bool(op.get("keep_children"))
        kids = self.children(topic)
        at = list(parent).index(topic)
        self._detach(topic)
        if keep:
            # Children keep their order and land at the deleted topic's index -
            # and their positions are recomputed, because they were relative to
            # the deleted topic, not to the grandparent.
            for offset, child in enumerate(kids):
                self._attach(parent, child, at + offset)
                self._place_among(parent, child, at + offset)
        kept = f", {len(kids)} children kept in place" if keep and kids else ""
        return [f"deleted {self.address_of(topic)} {_preview(topic.get('text') or '')}{kept}"]

    def _op_set_note(self, index: int, op: dict) -> list[str]:
        topic = self.resolve(op.get("topic") or "")
        text = op.get("text")
        if not isinstance(text, str):
            raise MindMapError('"text" must be a string ("" clears the note).')
        if text:
            topic.set("note", text)
        elif "note" in topic.attrib:
            del topic.attrib["note"]
        topic.set("modified", _now_iso())
        return [f"note {'set' if text else 'cleared'} on {self.address_of(topic)}"]

    def _op_set_link(self, index: int, op: dict) -> list[str]:
        """Set or clear a topic's ``link``.

        NO scheme allowlist, by decision: real maps carry ``ithoughts://``,
        ``file:///``, ``message:``, ``mailto:``, ``ftp:``, relative paths and
        ``/Local/...``, and a value a validator dislikes is still the user's
        data. The only check is that it is a string; anything non-empty is
        stored VERBATIM (not stripped, not normalised), and ``""``/``null``
        removes the attribute.
        """
        topic = self.resolve(op.get("topic") or "")
        url = op.get("url")
        if url is not None and not isinstance(url, str):
            raise MindMapError(
                f'"url" must be a string, got {type(url).__name__}. Use "" or '
                "null to remove the link."
            )
        url = url or ""
        had = topic.get("link")
        if url:
            topic.set("link", url)
        elif "link" in topic.attrib:
            del topic.attrib["link"]
        topic.set("modified", _now_iso())
        text = _preview(topic.get("text") or "")
        if url:
            return [f"link set on {self.address_of(topic)} {text}: {url}"]
        if had is None:
            # A URL in the topic TEXT is not a link attribute. Saying "cleared"
            # here let a model report having removed a link that never existed.
            return [f"no link on {self.address_of(topic)} {text} - nothing to clear"]
        return [f"link cleared on {self.address_of(topic)} {text} (was {had})"]

    # -- structural mutation primitives -------------------------------------
    #
    # Every change to the SHAPE of the tree goes through one of these, and each
    # one rebuilds the index before returning. The rule exists because the
    # index is what `address_of` and `resolve` read: a handler that mutated the
    # tree directly left a just-created topic out of `_index`, so `address_of`
    # printed its full 32-char uuid as the "address" and `resolve` then
    # rejected it - "here is the address" followed by "no such address".
    # Attribute-only ops (rename, set_note, set_link) do not come through here.

    def _unlink(self, el: ET.Element) -> None:
        """Detach from the parent WITHOUT rebuilding - for use inside a loop."""
        parent = self._parent.get(el)
        if parent is not None:
            parent.remove(el)

    def _detach(self, el: ET.Element) -> None:
        """Remove one topic from its parent, then rebuild."""
        self._unlink(el)
        self._build_index()

    def _detach_all(self, parent: ET.Element) -> None:
        """Remove every child of ``parent``, rebuilding ONCE.

        A per-child rebuild here costs N x the whole-tree walk (50 children on
        a real map is ~150 ms), so the loop unlinks and rebuilds at the end.
        """
        for child in self.children(parent):
            self._unlink(child)
        self._build_index()

    def _place_among(self, parent: ET.Element, topic: ET.Element, index: int) -> None:
        """Give ``topic`` a position that SORTS correctly among ``parent``'s children.

        ``position.y`` is an ordering key, not a layout: the app computes the
        geometry and ignores the magnitudes (y = 0, 10, 500, 510 renders as four
        evenly spaced rows). What it does not ignore is the ORDER - it draws by
        ``position.y``, so a topic whose XML order says "third" and whose y says
        "first" appears first. A missing position counts as ``{0, 0}``, which
        sorts at the parent's origin, so every topic this tool creates or moves
        gets one.

        The magnitudes below are a courtesy: in an outline map the app re-spaces
        the branch anyway, and in a free-form map a midpoint is the coordinate a
        user would expect. **No other sibling is ever renumbered** - that would
        rewrite a layout the user may have tuned, and it is not needed to fix the
        order. Repeated midpoints in one gap converge after ~6 halvings, because
        ``_format_position`` writes 6 decimals; when the neighbours share a y the
        midpoint is that same value and the XML order breaks the tie, which is
        correct - the topic was inserted at that index.

        ``topic`` must already be a child of ``parent``; ``index`` is where it
        sits (or should sit).
        """
        kids = self.children(parent)
        at = kids.index(topic) if topic in kids else max(0, min(index, len(kids)))
        others = kids[:at] + kids[at + 1 :]

        xs = [_parse_position(k.get("position"))[0] for k in others]
        # An actual sibling's x, not an average: a coordinate the user already
        # has beats one we invent.
        x = sorted(xs)[len(xs) // 2] if xs else 0.0

        if not others:
            y = 0.0
        elif at == 0:
            y = _parse_position(others[0].get("position"))[1] - SIBLING_GAP
        elif at >= len(others):
            y = _parse_position(others[-1].get("position"))[1] + SIBLING_GAP
        else:
            prev_y = _parse_position(others[at - 1].get("position"))[1]
            next_y = _parse_position(others[at].get("position"))[1]
            y = (prev_y + next_y) / 2.0

        topic.set("position", _format_position(x, y))

    def _attach(self, parent: ET.Element, el: ET.Element, index: Optional[int] = None) -> None:
        """Insert ``el`` under ``parent`` at child position ``index``, then rebuild.

        ``index`` is a CHILD position, clamped to the current child count, so
        an index equal to the count appends instead of raising - the old
        inline code indexed ``kids[at]`` and crashed on that boundary.
        """
        kids = self.children(parent)
        if isinstance(index, int) and not isinstance(index, bool):
            pos = max(0, min(index, len(kids)))
        else:
            pos = len(kids)
        if pos < len(kids):
            parent.insert(list(parent).index(kids[pos]), el)
        else:
            parent.append(el)
        self._build_index()

    def add_outline(self, parent: ET.Element, outline: str) -> list[ET.Element]:
        """Parse an outline and add it under ``parent``. Returns the new topics.

        The public half of ``_add_outline``: ``create_map`` and the
        ``add_branch`` op share one outline dialect, so they share this call.
        """
        return self._add_outline(parent, parse_outline(outline))

    def _add_outline(self, parent: ET.Element, rows: list[tuple[int, str]]) -> list[ET.Element]:
        created: list[ET.Element] = []
        stack: list[tuple[int, ET.Element]] = [(-1, parent)]
        for level, text in rows:
            while stack and stack[-1][0] >= level:
                stack.pop()
            owner = stack[-1][1] if stack else parent
            el = ET.Element("topic")
            el.set("uuid", str(uuid_mod.uuid4()).upper())
            el.set("text", text)
            stamp = _now_iso()
            el.set("created", stamp)
            el.set("modified", stamp)
            owner.append(el)
            # The REAL index among this owner's children, not a running counter:
            # an append only sorts last if its y beats every sibling's, and
            # siblings can sit at y = 4,288.
            self._place_among(owner, el, len(self.children(owner)) - 1)
            created.append(el)
            stack.append((level, el))
        self._build_index()
        return created

    # -- writing ------------------------------------------------------------

    def save(
        self,
        expect_short: str = "",
        quiet_seconds: int = QUIET_SECONDS_DEFAULT,
        create: bool = False,
    ) -> Version:
        """Version-check, quiet-check, write atomically. Returns the new version.

        ``expect_short`` is the version the caller believes it read - the
        ``version:`` line from ``read_map``, or the in-process baseline. It is
        compared as a hash PREFIX so a short value is enough and the check
        survives a chain restart.

        ``create=True`` is for a path that does not exist yet. It skips the
        ``version_of`` CALL - which reads the file and raises on a missing one,
        before any comparison could happen - and the quiet check, because there
        is no file to clobber and no user typing in it. It REFUSES an existing
        path, so "never overwrites" is a property of this method rather than of
        its caller. The tail below is unchanged either way.
        """
        if create:
            if self.path.exists():
                raise MindMapError(
                    f"{self.path} already exists. Use edit_map to change it, or "
                    "delete it first. create_map never overwrites."
                )
        else:
            current = version_of(self.path)

            if expect_short and not current.sha256.startswith(expect_short.lower()):
                raise MindMapError(
                    f"the map changed since it was read ({expect_short} -> {current.short}, "
                    f"mtime {_fmt_mtime(current.mtime_ns)}). Nothing was written. The batch "
                    "is all-or-nothing, so retrying is safe: call "
                    f'read_map("{self.path}") again, recompute the ops against the new '
                    "tree, then retry."
                )

            if quiet_seconds > 0 and not self._store.is_own_write(self.path, current.mtime_ns):
                age = time.time() - current.mtime_ns / 1e9
                if age < quiet_seconds:
                    raise MindMapError(
                        f"the map was written {int(age)} s ago - the app autosaves 10-20 s "
                        "after an edit, so the user may be typing in it (or a sync touched "
                        f"the file). Nothing was written. Wait ~{quiet_seconds} s and retry, "
                        "or re-read first."
                    )

        self.root.set("modified", _now_iso())
        payload = self._serialise_mapdata()
        self._write_archive(payload)
        new_version = version_of(self.path)
        self._store.note_own_write(self.path, new_version.mtime_ns)
        self._store.remember(self.path, new_version)
        self.version = new_version
        return new_version

    def _serialise_mapdata(self) -> bytes:
        original = next((data for info, data in self._entries if info.filename == MAPDATA), b"")
        had_bom = original.startswith(BOM)
        body = original[len(BOM):] if had_bom else original
        had_decl = body.lstrip().startswith(b"<?xml")

        xml = ET.tostring(self.root, encoding="utf-8")
        if had_decl:
            xml = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml
        if had_bom:
            xml = BOM + xml
        return xml

    def _write_archive(self, mapdata: bytes) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            with zipfile.ZipFile(tmp, "w") as zf:
                for info, data in self._entries:
                    # A fresh ZipInfo carrying the original's metadata: entry
                    # order, compress_type and timestamps are all preserved, and
                    # no directory entries are ever written.
                    out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                    out.compress_type = info.compress_type
                    out.external_attr = info.external_attr
                    out.internal_attr = info.internal_attr
                    out.create_system = info.create_system
                    zf.writestr(out, mapdata if info.filename == MAPDATA else data)
            try:
                os.chmod(tmp, stat.S_IMODE(self.path.stat().st_mode))
            except OSError:
                pass
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


# ── Small helpers ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _fmt_mtime(mtime_ns: int) -> str:
    return datetime.fromtimestamp(mtime_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")


def _preview(text: str, limit: int = TEXT_PREVIEW_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _flatten(text: str) -> str:
    """One line: every whitespace run collapses, newline-bearing runs are marked.

    ``_preview`` cannot do this - it maps a newline to a plain space, so a break
    in a topic's text would be invisible in the tree. The tree is one line per
    topic, so a raw newline would read as a topic of its own.
    """
    return re.sub(r"\s+", lambda m: NEWLINE_MARK if "\n" in m.group(0) else " ", text)


def _context_window(text: str, needle: str, limit: int = QUERY_CONTEXT_CHARS) -> str:
    """~``limit`` chars around the first match, with an ellipsis at any cut end."""
    flat = " ".join(text.split())
    at = flat.lower().find(needle)
    if at < 0:
        return flat[:limit]
    start = max(0, at - (limit - len(needle)) // 2)
    end = min(len(flat), start + limit)
    start = max(0, end - limit)
    out = flat[start:end]
    if start > 0:
        out = "\u2026" + out
    if end < len(flat):
        out = out + "\u2026"
    return out


def _note_preview(note: str, needle: Optional[str] = None, limit: int = NOTE_PREVIEW_CHARS) -> str:
    flat = " ".join(note.split())
    if needle:
        at = flat.lower().find(needle)
        if at > 60:
            flat = "…" + flat[at - 60:]
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + f"… ({len(note):,} chars total)"


def _child_note(count: int) -> str:
    if count == 0:
        return "(no children)"
    return f"({count} children)"


def _flags(el: ET.Element) -> list[str]:
    flags: list[str] = []
    kids = len(el.findall("topic"))
    if kids:
        flags.append(f"{kids} children")
    if el.get("folded") in ("1", "true"):
        flags.append("folded")
    if el.get("note"):
        flags.append("note")
    if el.get("link"):
        flags.append("link")
    if el.get("att-id"):
        flags.append("attachment")
    if el.get("image-width"):
        flags.append("image")
    progress = el.get("task-progress")
    if progress:
        flags.append(f"task {progress}%")
    return flags


def _parse_position(value: Optional[str]) -> tuple[float, float]:
    match = _POS_RE.match(value or "")
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def _format_position(x: float, y: float) -> str:
    return f"{{{x:.6f}, {y:.6f}}}"


def _missing_map_message(path: Path) -> str:
    parent = path.parent
    if parent.is_dir():
        others = sorted(p.name for p in parent.glob("*.itmz"))[:5]
        hint = f" Maps in that directory: {', '.join(others)}." if others else ""
        return (
            f"no map at {path}. The directory {parent} exists.{hint} Create one in "
            "iThoughtsX (File > New) and save it there, or check the path."
        )
    return (
        f"no map at {path} - the directory {parent} does not exist either. Check the "
        "path, or create the map in iThoughtsX and save it there."
    )
