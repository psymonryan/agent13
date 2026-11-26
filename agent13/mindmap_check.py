"""Diagnosis for .itmz maps: the format rules our own writer can break.

Deliberately NOT the release gate. ``utils/itmz-check.py`` in the ithoughtsx
repo has 22 fail-level checks, and most of them diagnose a map that came from
somewhere else - CRC, compression, writer signature, ``preview.png``,
``display_state``, manifest consistency, assets - because our writer replaces
``mapdata.xml`` and copies every other entry byte-for-byte. Those stay there.

What is left is what ``create_map`` and ``edit_map`` can actually violate: the
header, the topic tree, and the two attributes every topic must carry. A
position-less topic is the rule that fails the ithoughtsx release gate, and it
is the one an agent-built map is most likely to break.

Topic LINKS are checked too, and that scope is deliberately broader: a link
breaks from OUTSIDE the tool (a folder rename invalidates every file:// link
at once), the tool owns the map format, and it is where a fix can be offered.
Only local targets are resolved - ``file://`` URLs (percent-decoded) and
relative paths (against the map's own directory). Web links are counted but
NEVER fetched: a bookmark map would mean thousands of requests, and sites
answer bots with 403/429, so a "broken" verdict would be noise. ``/Local/...``
and other schemes are skipped.

Pure: a path in, a report out. No tools, no agent state, no network.
"""

from __future__ import annotations

import plistlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from agent13.mindmap import MAPDATA, STYLE

# Both rules are copied from utils/itmz-check.py (:53-54) rather than invented:
# a checker that disagrees with the release gate is worse than no checker.
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
POS_RE = re.compile(r"^\{\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\}$")

HEADER_ATTRS = ("version", "app", "modified")

# Per severity, in a trailer. A map where every topic lacks a position must not
# emit one line per topic.
TRAILER_LIMIT = 5


@dataclass
class LinkStats:
    """What the link pass saw. Broken targets are NOT counted here only -
    each one also lands in ``warnings`` (with title, chain and path) so the
    save trailer reports it; these counters feed the summary line."""

    checked: int = 0  # local targets resolved (file:// + relative)
    broken: int = 0  # subset of checked that does not exist
    web: int = 0  # http(s) - counted, never fetched
    skipped: int = 0  # /Local/..., other schemes, file:// on a non-local host

    @property
    def any_seen(self) -> bool:
        return bool(self.checked or self.web or self.skipped)

    def summary(self) -> str:
        parts = [f"{self.checked} local checked"]
        parts.append(f"{self.broken} broken" if self.broken else "all resolve")
        if self.web:
            parts.append(f"{self.web} web not checked")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        return ", ".join(parts)


@dataclass
class CheckReport:
    """Findings from one map.

    ``errors`` mean the app will not open it, ``warnings`` mean it opens but
    something is off, ``notes`` are informational and do not change the verdict.
    """

    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    topic_count: int = 0
    has_style: bool = False
    links: LinkStats = field(default_factory=LinkStats)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        """The full report, for an explicit check. Notes included."""
        if self.errors:
            head = f"{self.path}: {_plural(len(self.errors), 'error')}"
        elif self.warnings:
            head = f"{self.path}: OK with {_plural(len(self.warnings), 'warning')}"
        else:
            style = "style.xml present" if self.has_style else "no style.xml"
            head = f"{self.path}: OK ({_plural(self.topic_count, 'topic')}, {style})"
        lines = [head]
        if self.links.any_seen:
            lines.append(f"  links: {self.links.summary()}")
        lines += [f"  error: {message}" for message in self.errors]
        lines += [f"  warning: {message}" for message in self.warnings]
        lines += [f"  note: {message}" for message in self.notes]
        return "\n".join(lines)

    def trailer(self, extra: str = "", limit: int = TRAILER_LIMIT) -> str:
        """The findings as a trailer for a tool result, or "" when there are none.

        Every tool that touches a map appends this unconditionally, so a clean
        map - 9 of 12 real ones - leaves the output byte-for-byte unchanged.

        Notes are excluded: the only one is lowercase uuids, which our own app
        writes, so it would fire on most real maps and is noise in a read.

        Capped, because a map where every topic lacks a position must not emit
        one line per topic.
        """
        if not self.errors and not self.warnings:
            return ""

        counts = []
        if self.errors:
            counts.append(_plural(len(self.errors), "error"))
        if self.warnings:
            counts.append(_plural(len(self.warnings), "warning"))

        shown_errors = self.errors[:limit]
        shown_warnings = self.warnings[:limit]
        hidden = (len(self.errors) - len(shown_errors)) + (
            len(self.warnings) - len(shown_warnings)
        )

        lines = [f"check: {', '.join(counts)}"]
        lines += [f"  error: {message}" for message in shown_errors]
        lines += [f"  warning: {message}" for message in shown_warnings]
        if hidden:
            lines.append(f"  ... and {hidden} more")
        if extra:
            lines.append(f"  {extra}")
        return "\n".join(lines)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _address(uuid_text: str) -> str:
    """The 8-char address a reader would use."""
    return (uuid_text or "").replace("-", "").upper()[:8]


def _label(topic: ET.Element) -> str:
    """How to name a topic in a finding: its address, else its text."""
    uuid_text = topic.get("uuid") or ""
    if uuid_text:
        return _address(uuid_text)
    text = " ".join((topic.get("text") or "").split())
    return f'"{text[:40]}"' if text else "(untitled topic)"


def _pos(topic: ET.Element) -> tuple[float, float]:
    """A topic's ``(x, y)``, or ``(0, 0)`` when it has none.

    ``(0, 0)`` is not a guess: the app treats a missing position as ``{0, 0}``,
    so a position-less topic sorts at its parent's origin. That is also why the
    ordering rule below can run on a map that already has errors.
    """
    value = topic.get("position") or ""
    if not POS_RE.match(value):  # POS_RE is the release gate's, groups and all
        return 0.0, 0.0
    x, y = value.strip("{} ").split(",")
    return float(x), float(y)


def _pos_y(topic: ET.Element) -> float:
    return _pos(topic)[1]


# ── Link validation ──────────────────────────────────────────────────────────
# Only LOCAL targets are resolved. Web links are counted, never fetched: a
# bookmark map would mean thousands of requests on every save, and sites
# answer bots with 403/429, so a "broken" verdict would be guesswork.


def _walk_with_chain(parent_el: ET.Element, chain: tuple[str, ...] = ()):
    """Yield ``(topic, ancestor_titles)`` in document order. The chain for a
    topic is the titles of everything ABOVE it (root included, topic itself
    excluded); the map root's own chain is empty."""
    for kid in parent_el.findall("topic"):
        title = " ".join((kid.get("text") or "").split())
        yield kid, chain
        yield from _walk_with_chain(kid, chain + (title or "(untitled)",))


def _link_label(topic: ET.Element) -> str:
    """Title first (that is how a human knows the topic), address second
    (that is how edit_map reaches it)."""
    text = " ".join((topic.get("text") or "").split())[:60]
    addr = _address(topic.get("uuid") or "")
    if text and addr:
        return f'"{text}" [{addr}]'
    if text:
        return f'"{text}"'
    return _label(topic)


def _classify_link(link: str, base: Path) -> tuple[str, Optional[Path]]:
    """``link`` -> (kind, local_target).

    kinds: ``local`` (target is resolved, caller checks existence), ``web``
    (http/https - never fetched), ``skipped`` (/Local/..., other schemes,
    file:// on a non-local host).
    """
    low = link.lower()
    if low.startswith("http://") or low.startswith("https://"):
        return "web", None
    if low.startswith("file:"):
        parsed = urlparse(link)
        # Decode the host BEFORE classifying: quote() on a Windows path
        # encodes the colon and backslashes, so the drive can arrive as
        # C%3A%5C... and must be recognised decoded.
        netloc = unquote(parsed.netloc)
        if re.match(r"^[A-Za-z]:", netloc):
            # file://C:/... or file://C:\\... - the drive landed in the HOST
            # slot, which is what f"file://{windows_path}" produces (and what
            # a model passes to set_link). The app writes the proper
            # file:///C:/... form, but links are stored verbatim, so resolve
            # both. A real network host is never one letter + a colon.
            return "local", Path(netloc + unquote(parsed.path))
        if netloc.lower() not in ("", "localhost"):
            return "skipped", None  # file://server/share - a network location
        raw = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", raw):  # file:///C:/Users/... (Windows drive)
            raw = raw[1:]
        return "local", Path(raw)
    if link.startswith("/Local/") or link.startswith("/local/"):
        return "skipped", None  # the app's internal store
    # Any other scheme (mailto:, x-apple-:, ...) - RFC 3986 shape, so
    # "mailto:a@b.c" (no //) is caught too. A single letter before the colon
    # is a Windows drive path (C:\\Users), not a scheme - that stays local.
    if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", link) and not re.match(
        r"^[A-Za-z]:[\\/]", link
    ):
        return "skipped", None
    # No scheme: a relative path (set_link stores these verbatim). An absolute
    # path also lands here - pathlib makes ``base / "/abs"`` just "/abs".
    return "local", base / link


def _check_links(report: CheckReport, topics_el: ET.Element, map_path: Path) -> None:
    base = map_path.parent
    for topic, chain in _walk_with_chain(topics_el):
        link = (topic.get("link") or "").strip()
        if not link:
            continue
        kind, target = _classify_link(link, base)
        if kind == "web":
            report.links.web += 1
            continue
        if kind == "skipped":
            report.links.skipped += 1
            continue
        report.links.checked += 1
        if target.exists():
            continue
        report.links.broken += 1
        where = f" (under {' > '.join(chain)})" if chain else ""
        report.warnings.append(f"broken link on {_link_label(topic)}{where}: {target}")


def _manifest_topic_count(raw: bytes) -> Optional[int]:
    """``TopicCount`` from manifest.plist, or None when it cannot be read.

    An unreadable manifest is a cross-file defect this tool cannot cause, so it
    is silent here - ``itmz-check.py`` reports it.
    """
    try:
        value = plistlib.loads(raw).get("TopicCount")
    except Exception:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check(path: Path) -> CheckReport:
    """Check one .itmz. A bad map is reported, never raised."""
    report = CheckReport(path=path)

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(infos):
                report.warnings.append(
                    f"{len(infos) - len(names)} ZIP directory entr(y/ies) - the original "
                    "app renders a blank page for an archive containing them"
                )
            if MAPDATA not in names:
                report.errors.append(f"no {MAPDATA} entry - not an iThoughts map")
                return report
            raw = archive.read(MAPDATA)
            report.has_style = STYLE in names
            manifest = (
                archive.read("manifest.plist") if "manifest.plist" in names else None
            )
    except zipfile.BadZipFile:
        report.errors.append(
            "not a ZIP archive - a .itmz is a ZIP containing mapdata.xml"
        )
        return report
    except OSError as exc:
        report.errors.append(f"cannot read the file: {exc}")
        return report

    if not report.has_style:
        report.warnings.append(
            "no style.xml - the map will render with the app's defaults"
        )

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        report.errors.append(f"{MAPDATA} is not well-formed XML ({exc})")
        return report
    if root.tag != "iThoughts":
        report.errors.append(f"root element is <{root.tag}>, expected <iThoughts>")
        return report

    missing = [name for name in HEADER_ATTRS if not root.get(name)]
    if missing:
        report.warnings.append(f"header has no {', '.join(missing)}")

    topics_el = root.find("topics")
    if topics_el is None:
        report.errors.append("no <topics> element")
        return report

    topics = list(topics_el.iter("topic"))
    report.topic_count = len(topics)

    counts: dict[str, int] = {}
    lowercase = 0
    for topic in topics:
        uuid_text = topic.get("uuid") or ""
        if not uuid_text:
            report.errors.append(f"topic {_label(topic)} has no uuid")
        elif not UUID_RE.match(uuid_text):
            report.errors.append(
                f"topic {_address(uuid_text)} has a malformed uuid {uuid_text!r}"
            )
        else:
            key = uuid_text.lower()
            counts[key] = counts.get(key, 0) + 1
            if uuid_text != uuid_text.upper():
                lowercase += 1

        position = topic.get("position")
        if position is None:
            report.errors.append(
                f"topic {_label(topic)} has no position (the app treats a missing "
                "position as {0, 0}, which sorts it at the parent's origin)"
            )
        elif not POS_RE.match(position):
            report.errors.append(
                f"topic {_label(topic)} has a malformed position {position!r}"
            )

    for key, count in counts.items():
        if count > 1:
            report.errors.append(f"duplicate uuid {_address(key)} ({count} topics)")

    # Links: local targets resolved against the map's own directory (a folder
    # rename breaks them from outside the tool - the case that motivated the
    # check). Broken ones are warnings, never errors: the map opens fine.
    _check_links(report, topics_el, path)

    # The app lays out by position.y, so a parent whose XML child order differs
    # from its position order LOOKS different in the app than it reads here.
    # A warning, not an error: the file opens, it just does not look like what
    # the XML says. This is the rule that would have caught the reported bug
    # (a topic moved in XML, drawn where it used to be) before the user opened
    # the map.
    #
    # Scoped to SINGLE-COLUMN parents - children all sharing one x - because
    # that is the only place y is the ordering key. In a two-sided or vertical
    # layout x picks the column and y only orders within it: Life.itmz's "Life"
    # has its right column ascending (-2467 -> 1865) while the left descends
    # (100 -> -100). Measured across 22 real maps, the equal-x bucket (4,436
    # parents) has 0 disagreements and every one of the 9 that disagree is a
    # two-sided or vertical parent. The predicate is exactly "one x", NOT "one
    # x sign": the same-sign bucket (9,588 parents) has 2 disagreements.
    # Ties keep XML order, so an app-written single-column map never trips it.
    disagreeing = 0
    for parent in [topics_el] + topics:
        kids = parent.findall("topic")
        if len(kids) < 2:
            continue
        if len({_pos(k)[0] for k in kids}) > 1:
            continue  # two-sided or vertical: y is not the ordering key here
        if sorted(kids, key=_pos_y) != kids:
            disagreeing += 1
    if disagreeing:
        report.warnings.append(
            f"{_plural(disagreeing, 'single-column parent')} where the child order "
            "disagrees with the position order - the app lays out by position, so "
            "it will show a different order"
        )

    if lowercase:
        report.notes.append(
            f"{_plural(lowercase, 'uuid')} in lowercase (our own app writes them; "
            "the app accepts them)"
        )

    if manifest is not None:
        count = _manifest_topic_count(manifest)
        if count is not None and count != len(topics):
            report.warnings.append(
                f"manifest TopicCount {count} != {len(topics)} topics - stale manifest; "
                "the app regenerates it on its next save"
            )

    return report
