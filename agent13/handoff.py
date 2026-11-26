"""The handoff block: what the wrap-up tells the next session, mechanically.

The wrap-up's final message ends with a fenced JSON block::

    {"load": ["3.2", "3.7"], "for_user": ["scope call: continue?"]}

Two fields, and both have a reader:

* ``load`` - the topic ADDRESSES the next action needs. The seed carries them
  in full, so the next session starts with the facts it is about to use
  instead of guessing from the previous session's vocabulary.
* ``for_user`` - what the user must act on or know. Lifted to the top of the
  seed and injected into the goal question, because the summary that produced
  it is otherwise discarded at the rebuild.

Why the block is written LAST and captured separately: the 04:16 scope call on
2026-11-05 appeared in the wrap-up's closing summary, not in the journal entry,
and the summary is thrown away. A block written as part of the journal entry
would be written in "record" mode, before the summarising that produced the
insight. So the model emits it last and the harness appends it to the entry
afterwards - written last, persisted append-only, read mechanically.

Pure functions over strings: no I/O, no model call, so every case here is a
plain assertion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# The cap on carried for_user items. An unanswered item that accumulates
# forever makes the section wallpaper, and wallpaper is read as noise.
MAX_FOR_USER = 3

# A fence is ```json or ```, with the body up to the closing fence.
_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Handoff:
    """One parsed handoff block."""

    load: list[str] = field(default_factory=list)
    for_user: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)  # keys the schema rejects
    raw: str = ""  # the fenced block verbatim, for appending to the journal

    @property
    def empty(self) -> bool:
        return not self.load and not self.for_user


def _clean_list(value) -> list[str]:
    """A list of non-empty stripped strings; anything else becomes []."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_block(text: str) -> Handoff | None:
    """The LAST fenced JSON block in ``text``, or None when there is none.

    LAST, not first: the model may quote an earlier block while reasoning, and
    the one that counts is the one it ended with. A block that does not parse
    is None - the caller falls back to the selector and logs it, rather than
    failing the wrap-up.
    """
    if not text:
        return None
    for match in reversed(list(_FENCE.finditer(text))):
        body = match.group(1).strip()
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        known = {"load", "for_user"}
        return Handoff(
            load=_clean_list(data.get("load")),
            for_user=_clean_list(data.get("for_user"))[:MAX_FOR_USER],
            unknown=sorted(set(data) - known),
            raw=match.group(0),
        )
    return None


def render_block(handoff: Handoff) -> str:
    """The block as it is appended to the journal entry.

    Re-rendered from the parsed fields rather than copied, so what the seed
    later reads is exactly what was validated - a stray key or a malformed
    list cannot survive into the record.
    """
    payload = {"load": handoff.load, "for_user": handoff.for_user}
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def for_user_items(handoff: Handoff | None) -> list[str]:
    """The carried items, capped. [] when there is no block."""
    if handoff is None:
        return []
    return handoff.for_user[:MAX_FOR_USER]


def render_for_user(items: list[str]) -> str:
    """The seed's FOR THE USER block, or "" when there is nothing to carry.

    Lifted above the index rather than left inside the journal entry: if the
    point is that the next session must act on them, burying them in a 9k-char
    entry defeats it.
    """
    if not items:
        return ""
    lines = ["FOR THE USER (unresolved):"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)
