"""Which topics the seed should carry in full.

The index tells the model what exists; this decides what it should already
have. Pure functions over two strings - no I/O, no model call, no Agent - so
the same record and the same query always produce the same list.

The query is the previous session's own statement of what to do next (the last
journal entry's ``b.`` block) unioned with the rank-1 ticket. The score is the
sum of ``1/df`` over the query keys a topic contains, where ``df`` is the number
of topics containing that key.

Why the weighting is the whole correction: an UNWEIGHTED match on the rank-1
ticket's keys hits 24 of 39 topics - 71% of the record - which is no selection
at all. Weighted, the top topic scores ~40 against a second of ~7, and the same
topic is the one the model fetched by hand.

Two numbers, and they do different jobs. The THRESHOLD (1.0) only prevents the
floor case: on a real record 18 topics clear it, so it is not what limits the
load. The BUDGET is the primary control - the fill stops at 40,000 chars - and
both are logged, because the threshold alone cannot be tuned from the log.

The DF CEILING is what stops the selector scoring MARKUP. A query is a
handoff's prose, so it carries the record's own structural vocabulary ("CHECK",
"SESSION", "TABLE", "FIRST") alongside the subject, and a topic that shares
only those words scores as if it were relevant. A key in more than ~8 of 39
topics cannot say which topic matters, whatever it means - so the ceiling is a
statistic and needs no stop-list, which could not transfer to another field.

Nothing scoring >= 1.0 loads NOTHING. A weak match puts a wrong fact in
context, which is worse than an unopened box: the index's own line already says
a topic you never fetch is a topic you are guessing about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent13.truth_format import parse

# The budget for the topics carried in full. The ticket section and the journal
# entry are NOT part of it - this bounds only what the selector adds.
DEFAULT_BUDGET = 40_000

# Below this, nothing is loaded. See the module docstring: on a real record this
# admits ~18 topics, so it guards the floor, not the size.
DEFAULT_THRESHOLD = 1.0

# A key appearing in more topics than this is structural vocabulary, not
# subject: it cannot say WHICH topic matters. A statistic, not a stop-list, so
# it transfers to any record and any language.
DF_CEILING = 8

# A single topic larger than this is injected as its head and tail with an
# explicit elision marker rather than dropped - the top-scoring topic is the one
# the query named, so losing it entirely is the worst outcome.
OVERSIZE_TOPIC = 60_000
OVERSIZE_KEEP = 20_000

_BACKTICK = re.compile(r"`([^`]{4,})`")
_HEX = re.compile(r"0x[0-9a-fA-F]{4,}")
_CAPS = re.compile(r"\b[A-Z][A-Z0-9_\-]{3,}\b")


@dataclass(frozen=True)
class Selected:
    """One topic the seed carries in full."""

    heading: str
    line_no: int
    line_count: int
    score: float
    text: str  # the topic's span, or its head+tail when oversized


def _backticked_is_identifier(span: str) -> bool:
    """True when a backticked span names something, rather than quoting prose.

    Real records backtick prose as well as identifiers, and the prose spans are
    the most common keys of all - measured on edmac's record ``' -> '`` had df 26
    and ``' is the '`` df 25. They are not identifiers and they change which
    topics the budget loads.

    Two conditions, both needed:
    * at least one alphanumeric - rejects ``' -> '``;
    * no whitespace UNLESS it carries a digit - keeps ``Port 8111`` (a real
      identifier with a space) and rejects ``' is the '`` (prose).
    """
    stripped = span.strip()
    if not any(c.isalnum() for c in stripped):
        return False
    if any(c.isspace() for c in stripped) and not any(c.isdigit() for c in stripped):
        return False
    return True


def query_keys(query: str) -> set[str]:
    """The identifiers a query names: backticked, hex, and ALL-CAPS tokens."""
    keys = {k for k in _BACKTICK.findall(query) if _backticked_is_identifier(k)}
    keys.update(_HEX.findall(query))
    keys.update(_CAPS.findall(query))
    return {k for k in keys if len(k) >= 4}


def _topic_spans(truth: str):
    """[(heading, line_no, line_count, text)] for every topic, in file order."""
    shape = parse(truth)
    lines = truth.split("\n")
    out = []
    for i, topic in enumerate(shape.topics):
        end = (
            shape.topics[i + 1].line_no - 1
            if i + 1 < len(shape.topics)
            else (shape.ticket_line if shape.ticket_line is not None else len(lines))
        )
        out.append(
            (
                topic.heading,
                topic.line_no,
                topic.line_count,
                "\n".join(lines[topic.line_no - 1 : end]),
            )
        )
    return out


def b_block(journal_entry: str) -> str:
    """The last entry's ``b. NEXT SESSION, FIRST ACTION`` block, or "".

    The previous session's own statement of what to do next - already read for
    the seed, so using it as the query costs no extra file read.
    """
    if not journal_entry:
        return ""
    start = journal_entry.find("b. NEXT SESSION")
    if start < 0:
        return ""
    rest = journal_entry[start:]
    for stop in ("\nc.", "\n**c.", "\nd.", "\n**d."):
        at = rest.find(stop)
        if at > 0:
            return rest[:at]
    return rest


def build_query(journal_entry: str, truth: str) -> str:
    """The b. block unioned with the rank-1 ticket bullet.

    Falls back to the first ~1,000 chars of the entry when there is no b.
    block, and to the entry alone when the record has no ticket section.
    """
    from agent13.truth_format import ticket_bullets

    block = b_block(journal_entry)
    if not block:
        block = (journal_entry or "")[:1000]
    bullets = ticket_bullets(truth) if truth else []
    rank1 = bullets[0] if bullets else ""
    return (block + "\n" + rank1).strip()


def select_by_address(
    truth: str, addresses: list[str]
) -> tuple[list[Selected], list[str], list[str]]:
    """(topics in nomination order, not_found, over_budget).

    The handoff's ``load`` list, resolved against the record's addresses. An
    address either resolves or it does not - that is the whole reason the
    handle is a number and not a heading, which gets paraphrased into silence.

    **Every nominated topic is carried.** The prompt already caps the
    nomination at six, and six topics is ~30k tokens at a 1M context, so an
    aggregate char budget had exactly one effect: it broke the promise. On the
    first live run it silently dropped the topic the model nominated SECOND
    (3.11, 55,692 chars) while reporting it as ``missing`` - indistinguishable
    from an invented address. A single topic over ``OVERSIZE_TOPIC`` is
    injected head+tail with the elision marker, never discarded.

    The two failure lists are kept apart because they are different bugs: a
    ``not_found`` address is the model inventing a handle, an ``over_budget``
    one is the harness dropping a real topic. ``over_budget`` is empty by
    construction now; it stays as the instrument that would catch a budget
    creeping back.
    """
    shape = parse(truth)
    by_number = {topic.number: topic for topic in shape.topics if topic.number}
    spans = {row[1]: row for row in _topic_spans(truth)}

    out: list[Selected] = []
    not_found: list[str] = []
    over_budget: list[str] = []
    for address in addresses:
        topic = by_number.get(address)
        if topic is None:
            not_found.append(address)
            continue
        heading, line_no, line_count, text = spans[topic.line_no]
        if len(text) > OVERSIZE_TOPIC:
            text = _truncate(text)
        out.append(Selected(heading, line_no, line_count, 0.0, text))
    return out, not_found, over_budget


def _truncate(text: str) -> str:
    """Head + elision marker + tail, for a topic too large to inject whole."""
    head = text[:OVERSIZE_KEEP]
    tail = text[-OVERSIZE_KEEP:]
    dropped = len(text) - 2 * OVERSIZE_KEEP
    return f"{head}\n[... {dropped} chars elided ...]\n{tail}"


def select_topics(
    truth: str,
    query: str,
    budget: int = DEFAULT_BUDGET,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Selected]:
    """The topics to carry in full, best first. Deterministic.

    The top-scoring topic is always included (it is the one the query named),
    then the rest fill in descending score order until ``budget`` is reached.
    """
    spans = _topic_spans(truth)
    if not spans or not query:
        return []

    keys = query_keys(query)
    if not keys:
        return []

    df = {k: sum(1 for _, _, _, text in spans if k in text) for k in keys}
    # A key in more than DF_CEILING topics is structural vocabulary, not
    # subject: it cannot say WHICH topic matters, whatever it means. This is a
    # statistic, so it transfers to any record and any language - a hand-tuned
    # stop-list cannot (measured over 14 journal blocks, a deny-list changed
    # the selected SET zero times).
    keys = {k for k in keys if df[k] <= DF_CEILING}
    if not keys:
        return []

    scored = []
    for heading, line_no, line_count, text in spans:
        score = sum(1.0 / df[k] for k in keys if k in text and df[k])
        if score >= threshold:
            scored.append((score, line_no, heading, line_count, text))

    # Descending score, then file order: a tie must not depend on dict order.
    scored.sort(key=lambda row: (-row[0], row[1]))

    out: list[Selected] = []
    used = 0
    for i, (score, line_no, heading, line_count, text) in enumerate(scored):
        if len(text) > OVERSIZE_TOPIC:
            text = _truncate(text)
        if i > 0 and used + len(text) > budget:
            break
        used += len(text)
        out.append(Selected(heading, line_no, line_count, round(score, 2), text))
    return out
