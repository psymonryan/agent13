"""The durable record's format: topics, facts, and the ticket section.

The record is a contract between a prompt that WRITES it and code that READS
it. This module owns the reading half, and the prompt's counting command is
composed from ``TICKET_HEADING`` (see DEFAULT_REPORT_AND_COMPACT_PROMPT), so a
rename cannot desync the two.

Shape, as the wrap-up prompt defines it::

    ## 3. Findings          <- a section
    ### Topic name          <- a topic: the only capped unit
    - one fact              <- a fact: one claim, uncapped
    ## 6. Open tickets      <- everything from here is the work queue

Parsing happens ONCE (``parse``). The index builder's lossless guard and its
renderer used to walk the text separately, with two state machines that had to
agree by inspection; a disagreement would have let the guard pass while the
render dropped content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TICKET_HEADING = "open tickets"  # matched case-insensitively in a "## " heading

# The literal the wrap-up prompt writes in "h. TICKET CHANGES" and the rank
# detector greps for. A contract between the prompt and the code, like
# TICKET_HEADING: the prompt line is composed from this constant, so a rename
# cannot desync the two.
RERANK_TOKEN = "RE-RANKED"


def is_ticket_section(line: str) -> bool:
    """True for the "## ... Open tickets" heading.

    Matched by TEXT, not by number: keying on a literal "## 6." would silently
    drop the whole ticket list from the injected context if a record numbered
    its sections differently - the work queue would vanish with no error.
    """
    return line.startswith("## ") and TICKET_HEADING in line.lower()


@dataclass(frozen=True)
class Topic:
    """One "### " heading and the span it owns.

    No fact is sampled: a positional sample cannot say what a topic holds. The
    first fact is the oldest content and the last is often the tail of a
    sentence whose head is three bullets up, so the index shows the heading and
    the line count and reads as an unopened box - which is the truth.

    ``number`` is the topic's ADDRESS ("3.2"), assigned once when the topic is
    created and never changed. It is the handle a handoff nominates, so it must
    survive every rewrite the record gets: a line number drifts with every edit
    above it, and a heading gets paraphrased. "" on a record that has not been
    migrated - the index then prints no address and the model falls back to
    reading by line number.
    """

    line_no: int  # 1-based, so it can be used as read_file(offset=...)
    heading: str  # the LABEL, with any address prefix removed
    line_count: int  # heading through the last non-blank line of the span
    number: str  # "3.2", or "" when the record carries no addresses


@dataclass(frozen=True)
class RecordShape:
    """What the record contains, parsed once."""

    topics: list[Topic]
    ticket_line: int | None  # 0-based index of the ticket heading; None if absent
    flat_facts: list[int]  # 1-based line numbers of "- " facts under no topic


def _span_end(lines: list[str], heading_index: int) -> int:
    """Index of the first line after the heading that starts with "#".

    One stop rule for the whole span, at any heading level: a "### " topic ends
    at the next "### " OR the next "## " section. Stopping at the next heading
    matters for every reader of the span - without it an empty topic would be
    shown with the FOLLOWING topic's fact.
    """
    j = heading_index + 1
    while j < len(lines) and not lines[j].startswith("#"):
        j += 1
    return j


def _line_count(lines: list[str], heading_index: int, end: int) -> int:
    """Heading through the last NON-BLANK line of the span; 1 for an empty topic.

    Trailing blank lines are not counted: the count is a read_file window, and
    a window that ends on a blank line is a window with a wasted line.
    """
    last = heading_index
    for j in range(heading_index, end):
        if lines[j].strip():
            last = j
    return last - heading_index + 1


_TOPIC_ADDRESS = re.compile(r"^(\d+)\.(\d+)\s+(\S.*)$")


def split_address(text: str) -> tuple[str, str]:
    """("3.2", "THE OBJECT STATE MACHINE") from a "### " heading body.

    A record that has not been migrated carries no address: the whole body is
    the label and the number is "". The pattern needs whitespace after the
    digits, so "### 1.5x speed" and "### 2026-11-04 ..." stay labels.
    """
    match = _TOPIC_ADDRESS.match(text)
    if match:
        return f"{match.group(1)}.{match.group(2)}", match.group(3).strip()
    return "", text


def parse(text: str) -> RecordShape:
    """Parse the record's structure. Pure; no I/O, no model call."""
    lines = text.split("\n")
    topics: list[Topic] = []
    flat_facts: list[int] = []
    ticket_line: int | None = None
    under_topic = False

    for i, line in enumerate(lines):
        if is_ticket_section(line):
            ticket_line = i
            break  # everything from here is the work queue, not topics
        if line.startswith("### "):
            under_topic = True
            end = _span_end(lines, i)
            number, heading = split_address(line[4:])
            topics.append(Topic(i + 1, heading, _line_count(lines, i, end), number))
        elif line.startswith("## "):
            under_topic = False
        elif line.startswith("- ") and not under_topic:
            flat_facts.append(i + 1)

    return RecordShape(topics, ticket_line, flat_facts)


def count_topics(text: str) -> int:
    """Topics before the ticket section - the capped unit."""
    return len(parse(text).topics)


def count_facts(text: str) -> int:
    """Every "- " line in the file, ticket bullets included (facts are uncapped)."""
    return sum(1 for line in text.split("\n") if line.startswith("- "))


# --- assigning addresses (the one-time migration) ---------------------------
#
# The address is assigned ONCE and never re-sequenced. That is the whole point:
# a handoff nominates an address, and the next session may read it hours later,
# across a chain restart, with the record rewritten in between. A positional
# scheme would silently re-point every number after an insertion.

_SECTION = re.compile(r"^##\s+(\d+)\.")


def assign_addresses(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(new text, [(address, heading)]) - gaps filled, nothing renumbered.

    A topic that already has an address keeps it, so a second run is a no-op.
    A new topic takes the next free number in its section, so a deleted number
    is never handed out again by this function and gaps are expected.

    A "### " heading outside a numbered "## N." section gets no address: there
    is no section number to compose one from, and inventing a section is worse
    than leaving the topic addressable by line number alone.
    """
    lines = text.split("\n")
    section = ""
    used: dict[str, set[int]] = {}
    assigned: list[tuple[str, str]] = []

    for i, line in enumerate(lines):
        if is_ticket_section(line):
            break  # the ticket section's "### " sub-headings are not topics
        if line.startswith("## "):
            match = _SECTION.match(line)
            section = match.group(1) if match else ""
            continue
        if not line.startswith("### "):
            continue
        body = line[4:]
        existing = _TOPIC_ADDRESS.match(body)
        if existing:
            used.setdefault(existing.group(1), set()).add(int(existing.group(2)))
            continue
        if not section:
            continue
        taken = used.setdefault(section, set())
        item = max(taken) + 1 if taken else 1
        taken.add(item)
        address = f"{section}.{item}"
        lines[i] = f"### {address} {body}"
        assigned.append((address, body))

    return "\n".join(lines), assigned


# --- the ranked ticket order (section 6) ------------------------------------
#
# A ticket's identity must survive the edits a wrap-up makes to its bullet: the
# rank marker moves between tickets, and the tail after the title is rewritten
# every entry. What stays stable is the title.

_RANK_MARKER = re.compile(r"^\s*\(RANK\s+(\d+)\)\s*", re.IGNORECASE)


def _ticket_id(bullet: str) -> tuple[str, int | None]:
    """(identity, marker_rank) for one ticket bullet.

    ``**`` is stripped FIRST: real bullets are ``- **(RANK 1)** **title**``, so
    the marker sits inside the bold run and a literal "leading (RANK n)" match
    finds nothing. Measured on a pair of real records: without this the demoted
    ticket's 1 -> 6 move disappears, because the marker makes the same ticket
    look like two different tickets.

    The tail after the first " - " is cut because it changes between commits
    (``- THE NEW GOAL'S CORE QUESTION`` -> ``- the UI ...`` -> ``- PARKED ...``)
    while the title is stable.
    """
    s = bullet[2:].replace("**", "")
    match = _RANK_MARKER.match(s)
    marker = int(match.group(1)) if match else None
    if match:
        s = s[match.end():]
    for sep in (" - ", " \u2014 "):
        if sep in s:
            s = s.split(sep)[0]
            break
    s = re.sub(r"\s+", " ", s).strip().rstrip(".,;:")
    return s[:60], marker


def ticket_order(text: str) -> list[tuple[str, int | None, int]]:
    """(identity, marker_rank, position) per ticket bullet in section 6.

    Stops at the next "## " or "### " heading: "### Parked" is not ranked, and
    a parked ticket appearing in the order would read as a re-rank on every
    park. Returns [] when the record has no ticket section.

    A ticket parked IN PLACE - bullet kept, "PARKED" in its body - still counts
    as ranked; only the "### Parked" heading is excluded. That is what the live
    records do, and it is harmless: ``rank_moves`` compares common tickets
    only, so a park that leaves the bullet where it was registers no move.
    """
    shape = parse(text)
    if shape.ticket_line is None:
        return []
    lines = text.split("\n")
    out: list[tuple[str, int | None, int]] = []
    for line in lines[shape.ticket_line + 1:]:
        if line.startswith(("## ", "### ")):
            break
        if line.startswith("- "):
            identity, marker = _ticket_id(line)
            out.append((identity, marker, len(out) + 1))
    return out


def ticket_bullets(text: str) -> list[str]:
    """The full text of each ticket bullet in section 6, in order.

    ``ticket_order`` returns identities, which cut the bullet at the first
    " - " - right for ranking (the tail is rewritten every entry) and wrong for
    reading a ticket's STATE, because the BLOCKED / NEEDS USER / PARKED marker
    lives in that tail. Same walk, same stop rule, raw lines.

    "### Parked" and "### Deferred" are excluded by the stop rule, so a ticket
    moved under a sub-heading leaves the work queue. A ticket parked IN PLACE
    does not - the caller has to read its body (see agent13/goal.py).
    """
    shape = parse(text)
    if shape.ticket_line is None:
        return []
    out: list[str] = []
    for line in text.split("\n")[shape.ticket_line + 1:]:
        if line.startswith(("## ", "### ")):
            break
        if line.startswith("- "):
            out.append(line)
    return out


def encoding_of(order: list[tuple[str, int | None, int]]) -> str:
    """How the order expresses rank: "position", "markers", "mixed", or "none".

    Logged so a reader can judge a noisy event. A "mixed" record is the defect
    the prompt's "Rank IS position" rule removes; it is reported, never
    resolved silently.
    """
    if not order:
        return "none"
    marked = sum(1 for _, marker, _ in order if marker is not None)
    if marked == 0:
        return "position"
    if marked == len(order):
        return "markers"
    return "mixed"


def rank_moves(
    before: list[tuple[str, int | None, int]],
    after: list[tuple[str, int | None, int]],
) -> list[tuple[str, int, int]]:
    """(ticket, old_rank, new_rank) for tickets whose EFFECTIVE rank changed.

    Effective rank per ticket = its (RANK n) marker when it has one, else its
    position among the COMMON tickets. Marker-else-position is the whole
    correction: a measured wrap-up expressed a new ranking with markers while
    leaving bullet order untouched, so a position-only reader saw no change in
    the record that FIXED the bug.

    Ranks are computed among the COMMON tickets only, so adding a ticket at the
    top or closing one does not register as a move for every other ticket.

    Deliberately more sensitive than precise: a false positive costs a log
    line, a false negative costs an hour of work on the wrong ticket.
    """
    ids_before = [identity for identity, _, _ in before]
    ids_after = [identity for identity, _, _ in after]
    common_before = [t for t in ids_before if t in ids_after]
    common_after = [t for t in ids_after if t in ids_before]

    def effective(order, common):
        ranks: dict[str, int] = {}
        for identity, marker, _position in order:
            if identity in common:
                ranks[identity] = (
                    marker if marker is not None else common.index(identity) + 1
                )
        return ranks

    old = effective(before, common_before)
    new = effective(after, common_after)
    return [(t, old[t], new[t]) for t in common_before if old[t] != new[t]]


def _bold_block(entry_text: str, marker: str) -> str:
    """One journal entry's ``<marker>`` section, heading line included.

    The section runs from the bold heading to the next bold section heading
    (or the next entry). The heading line's own remainder is part of it: the
    delta is often on the SAME line as the heading - observed live 2026-10-30
    23:17::

        **h. TICKET CHANGES** - RE-RANKED: Make A work rank 2 -> rank 1

    Skipping the rest of the heading line reported that as unreported.

    ONE walker for every bold section, so the readers of ``h.`` and ``b.``
    cannot disagree about where a section ends.
    """
    lines = entry_text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            continue  # the entry's own heading; its sections follow it
        if not stripped.lower().startswith(marker):
            continue
        block = [stripped]
        for following in lines[i + 1 :]:
            following = following.strip()
            if following.startswith("**") or following.startswith("## "):
                break
            block.append(following)
        return "\n".join(block).strip()
    return ""


def _after_heading(line: str, marker: str) -> str:
    """The text after a bold heading's closing ``**`` ("" when it has none)."""
    rest = line.strip()[len(marker) :]
    if "**" in rest:
        rest = rest.split("**", 1)[1]
    return rest.strip()


def h_block_text(entry_text: str) -> str:
    """One journal entry's h. block; "" when it has none."""
    return _bold_block(entry_text, "**h.")


def h_block_reports_rerank(entry_text: str) -> bool:
    """True when one journal entry's h. block reports a re-rank.

    Takes the ENTRY, not the whole journal: ``DurableRecords.read_last_journal_entry``
    already owns "which entry is the last one", and a second walk here would be
    two state machines that have to agree by inspection.

    Scoped to the h. block, not the entry: a measured failure said
    "Re-ranked the tickets" in ``e.`` while its ``h.`` said "- none", so a
    whole-entry boolean reported the failure as compliant.

    Matched at the START of a line, not as a substring: "- none; nothing
    re-ranked" contains the token but reports nothing, and counting it as
    reported is the same inversion in the other direction.
    """
    block = h_block_text(entry_text)
    if not block:
        return False
    for line in block.split("\n"):
        text = line.strip()
        if text.lower().startswith("**h."):
            text = _after_heading(text, "**h.")
        text = text.lstrip("-*_ ").lstrip("*_ ").strip()
        if text.upper().startswith(RERANK_TOKEN):
            return True
    return False
