"""The goal window's pure rules.

A goal window is a deadline that survives turn ends: when the agent would go
idle inside the window, it asks itself whether it can make progress without the
user and, if so, starts another turn. The mechanism lives in core.py; every
rule that can be decided without an Agent lives here, so it is testable in
isolation and visible in one place.

Why the model never sees the remaining time: a visible budget invites "I'm
running low, let me consolidate and hand back", which is the failure the window
exists to prevent (measured 2026-11-02 01:53 - a turn ended with 6h 34m of the
window still to run). The model still gets the wall clock from the datetime
nudge, which is what journal headings need.

The command is ``/grind``; the state is a goal (``_goal_text``,
``_goal_deadline``). A goal is not time-bounded, a work window is - and "goal"
reads as a definition of done, which is the judgement the model got wrong.
"""

from __future__ import annotations

import json
import re

from agent13 import truth_format
from agent13.prompts import (
    GOAL_ITEM,
    GOAL_ITEM_RECORD_FALLBACK,
    GOAL_QUESTION_ALL_BLOCKED,
    GOAL_QUESTION_FOR_USER,
    GOAL_QUESTION_INSTRUCTION,
    GOAL_QUESTION_PLAIN,
    GOAL_QUESTION_UNBLOCKED,
)

# "8h", "90m", "2h30m" - hours and minutes only. Seconds are rejected so the
# status bar's window can never be seconds: the numerator is H:MM too, and a
# mixed pair reads as a misprint (see ui/tui.py).
_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?$", re.IGNORECASE)


def parse_duration(text: str) -> tuple[float | None, str]:
    """(seconds, remainder) for "<duration> [goal text]".

    Returns (None, "") when the first token is not a duration, so the caller
    can tell "no duration given" from "a zero window" - ``0h`` parses to 0.0
    and the command rejects it, because a window that closes on arrival is a
    typo, not a request.
    """
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None, ""
    match = _DURATION.match(parts[0])
    if not match or (match.group(1) is None and match.group(2) is None):
        return None, ""
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    remainder = parts[1].strip() if len(parts) > 1 else ""
    return float(hours * 3600 + minutes * 60), remainder


def format_window(seconds: float) -> str:
    """H:MM - hours and minutes, never seconds.

    Both sides of the status bar's ratio use this. The numerator used to be
    m:ss with unbounded minutes, so a 3.5-hour task rendered "210:00" next to a
    "4:00" window; hours-and-minutes on both sides is the only reading that
    cannot be mistaken for the other.
    """
    total_minutes = max(0, int(seconds // 60))
    return f"{total_minutes // 60}:{total_minutes % 60:02d}"


# --- the mechanical ticket check -------------------------------------------
#
# The record is only rewritten at a wrap-up, so this can be hours stale. It
# therefore SHAPES the question and never ends the run: ending a live window on
# a snapshot would be the mechanism causing the failure it exists to prevent.

_BLOCKED = re.compile(r"\b(?:BLOCKED|PARKED)\b|\bNEEDS\s+USER\b", re.IGNORECASE)
_FIRST_SEEN = re.compile(r"FIRST\s+SEEN", re.IGNORECASE)


def ticket_is_blocked(bullet: str) -> bool:
    """True when a ticket bullet marks itself unworkable.

    Searched in the tail after the FIRST SEEN stamp, because that is where the
    wrap-up writes the state ("- ... - FIRST SEEN 2026-10-30 - BLOCKED: ...").
    A bullet with no stamp is searched whole: the marker is the only evidence
    there is, and missing it would read a blocked ticket as available work.

    Word boundaries matter: "unblocked" contains "blocked" as a substring, and
    a ticket that says so is the opposite of blocked. PARKED is included
    because parked-in-place is a live shape - the bullet stays where it is and
    the word goes into its body (truth_format.ticket_order's docstring).
    """
    match = _FIRST_SEEN.search(bullet)
    tail = bullet[match.end():] if match else bullet
    return bool(_BLOCKED.search(tail))


def classify_tickets(bullets: list[str]) -> tuple[int, int]:
    """(unblocked, blocked) counts for the record's open tickets."""
    blocked = sum(1 for bullet in bullets if ticket_is_blocked(bullet))
    return len(bullets) - blocked, blocked


def ticket_counts(truth_text: str) -> tuple[int, int, bool]:
    """(unblocked, blocked, has_record) from the current-truth text."""
    if not truth_text:
        return 0, 0, False
    unblocked, blocked = classify_tickets(truth_format.ticket_bullets(truth_text))
    return unblocked, blocked, True


def preamble_name(unblocked: int, blocked: int) -> str:
    """Which preamble the mechanical check selects.

    One home for the rule: ``question_text`` picks its text from this name and
    the log records it, so the two cannot describe different questions.
    """
    if unblocked:
        return "unblocked"
    if blocked:
        return "all_blocked"
    return "plain"


def question_text(
    unblocked: int, blocked: int, has_record: bool, for_user: list[str] | None = None
) -> str:
    """The question turn's user message, shaped by the mechanical check.

    All three preambles end with the same JSON instruction: the shape of the
    answer must not vary, or the parser has three shapes to tolerate.

    ``for_user`` is the last handoff's unresolved items. They are injected
    into the SAME question rather than asked separately: the model must
    reconcile "this is a scope call for you" with "can you make progress",
    and a separate question would let it answer both without noticing.
    """
    preamble = {
        "unblocked": GOAL_QUESTION_UNBLOCKED,
        "all_blocked": GOAL_QUESTION_ALL_BLOCKED,
        "plain": GOAL_QUESTION_PLAIN,
    }[preamble_name(unblocked, blocked)]
    parts = [preamble]
    if for_user:
        items = "\n".join(f"- {item}" for item in for_user)
        parts.append(GOAL_QUESTION_FOR_USER.format(items=items))
    parts.append(GOAL_QUESTION_INSTRUCTION)
    return "\n\n".join(parts)


# --- the answer -------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _first_json_object(text: str) -> str | None:
    """The first balanced {...} run in the text, or None.

    A brace counter rather than a regex: the answer contains nested arrays and
    strings with braces in them, and a greedy match would swallow the prose
    that follows.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_answer(text: str) -> dict | None:
    """The answer JSON, tolerating the shapes models actually emit.

    A fenced block, JSON inside prose and ``"true"`` as a string are all
    normal. Returns None when nothing parses; the caller treats that as
    ``can_progress: true``, because continuing is the safe failure and stopping
    silently is the failure this whole mechanism exists to fix.
    """
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    for chunk in candidates:
        blob = _first_json_object(chunk)
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def can_progress(answer: dict | None) -> bool:
    """The decision, with every ambiguous shape resolving to True."""
    if not isinstance(answer, dict):
        return True
    value = answer.get("can_progress")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "no", "0", "")
    if value is None:
        return True
    return bool(value)


def next_step(answer: dict | None) -> str:
    """The bare step, or "" when it is missing or unusable."""
    if not isinstance(answer, dict):
        return ""
    value = answer.get("next")
    if not isinstance(value, str):
        return ""
    return value.strip()


def waiting_on(answer: dict | None) -> list[str]:
    """What the model says it needs, normalised to a list of strings."""
    if not isinstance(answer, dict):
        return []
    value = answer.get("waiting_on_user")
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


# --- the continuation message ----------------------------------------------


def goal_item(goal_text: str, step: str, has_record: bool) -> str:
    """The continuation message, or "" when there is nothing to say.

    Two lines, no attribution and no hedge. "You said your next step would be
    ..." is false from the model's point of view - the question and answer were
    deleted, so its history contains no trace of having said it - and a hedge
    invites meta-reasoning about why the harness is hedging, when the model has
    already told us the step it wants to take.

    The caller substitutes AUTO_CONTEXT_CONTINUE_HINT for "", so an empty
    message is never enqueued.
    """
    if step:
        body = GOAL_ITEM.format(step=step)
    elif has_record:
        body = GOAL_ITEM_RECORD_FALLBACK
    else:
        body = ""
    if goal_text:
        return f"Goal: {goal_text}\n{body}" if body else f"Goal: {goal_text}"
    return body
