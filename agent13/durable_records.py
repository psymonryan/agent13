"""Durable records - the focus name, its record paths, and the record files.

The focus/record cluster lived on ``Agent`` as eight private methods and
three fields, with the ``--focus`` startup pattern duplicated four times
(cli / repl / pipemode / tui) and a fifth variant in the ``/focus`` command.
Every feature touching the records (wrap-up, compact, /chain, /focus,
persistence, /status, the TUI status bar) reached into Agent's private state.
This module owns it: the name, the paths, the wrap-up's path capture, the
seeding of records into a history, the chain-restart intro, the compact
pointer block, wrap-up currency, and the ``.ctx`` field.

Two rules keep the traps of the surface it replaces out of this one
(``agent13.context_gauge`` is the precedent, 5702ef5):

* **History is passed in, never bound.** The methods that need the message
  list take it as an argument. A stored reference goes stale the moment the
  history is replaced - and it is replaced constantly (compact, chain
  restart, /clear, load_context). The journal's ``stream_fn`` binding trap
  is the same failure with a callable in it.
* **No Agent reference, no events, no LLM calls.** Pure state plus file
  reads, so the whole surface is testable by constructing one object.

Prompt text lives in ``agent13.prompts`` (the ``default_prompts.yaml``
mirror discipline applies there, not here); this module only formats it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent13.config_paths import get_records_dir
from agent13.truth_format import parse

from agent13.prompts import (
    CHAIN_RESTART_INTRO,
    COMPACT_DURABLE_RECORD_HEADER,
    COMPACT_DURABLE_RECORD_HEADER_BARE,
    COMPACT_DURABLE_RECORD_JOURNAL_LINE,
    COMPACT_DURABLE_RECORD_TRUTH_LINE,
    COMPACT_DURABLE_RECORD_TRUTH_LINE_INDEXED,
)

# Header of the user message seed() injects. Core-internal (not
# yaml-exposed): it describes the shape of the record files, which is this
# module's own knowledge.
#
# The mid-session edit rule. Measured 2026-10-22: one project's truth file took
# 38 mid-session edits in a day, 33 of them additive, net +31,718 chars - and
# that growth is what pushed the file past the 30,000-char read cap and made
# the wrap-up rewrite blind in the middle. The post-wrap-up size check
# (6dd2117) treats the symptom at the handoff; this removes the cause.
SEED_PREAMBLE = (
    "Durable record for focus {name} - the state at the last handoff. "
    "Here is the current truth before starting new work; its "
    "section 6 tickets are the work queue (top unblocked ticket first). "
    "Mid-session, correct a bullet in place when a fact changes; do not "
    "add new bullets - additions belong in the handoff rewrite."
)

# Instruction appended to the seed message on the --focus startup path,
# turning the records load into a real turn (the --read pattern: contents
# + "Reply with ... and nothing else."). The model's reply replaces the
# mechanical "[Loaded records ...]" ack of seed() and warms the KV prefix
# on the records while the user types their first prompt. Core-internal.
SEED_ACK_INSTRUCTION = (
    'Reply with "Records for focus {name} loaded." and nothing else.'
)

# ── topic index injection (config: [auto_context] truth_index, default off) ──
#
# The record is injected into message 0 of every new context, so the whole
# file is paid for on every context. Once the record holds topics, injecting
# an index instead costs a fixed ~3-12k chars and lets the model fetch a topic
# on demand. The index is built ONLY when it is provably lossless (see
# build_truth_index), so an unmigrated or half-migrated record is injected
# unchanged and behaves exactly as it does today.
TRUTH_INDEX_PREAMBLE = (
    "TOPIC INDEX of {path} - the facts are NOT loaded.\n"
    "Each line is one topic: its file line number, the number of lines it spans,\n"
    "and its heading. [LOADED] marks a topic whose facts are already present\n"
    "below. Before you act on any other topic, read it with\n"
    "read_file(offset=L, limit=N) using those two numbers, or grep the file for\n"
    "any term. A topic you never fetch is a topic you are guessing about.\n"
)


def _collapse_parked(ticket_lines: list[str]) -> list[str]:
    """One line per parked ticket: title, date and the reason's first sentence.

    Parked tickets are inspected only when unparked, and their bodies are the
    bulk of the section. The cut is the first sentence boundary, capped, so a
    bullet that carries no full stop still ends somewhere sane.
    """
    out: list[str] = []
    in_parked = False
    for line in ticket_lines:
        if line.startswith("### "):
            in_parked = "parked" in line.lower()
            out.append(line)
            continue
        if not in_parked or not line.startswith("- "):
            if in_parked and not line.strip():
                continue  # blank lines inside a collapsed section are noise
            out.append(line)
            continue
        cut = len(line)
        for sep in (". ", ".** "):
            at = line.find(sep)
            if 0 < at < cut:
                cut = at + 1  # keep the full stop, drop the rest
        if cut > 240:
            cut = 240
        out.append(line[:cut].rstrip() + ("\u2026" if cut < len(line) else ""))
    return out


def build_truth_index_parts(
    truth: str, loaded: frozenset[int] | set[int] = frozenset()
) -> tuple[str, str] | None:
    """(topic list, ticket section), or None when an index would be lossy.

    Split so the seed can put the topics it carries in full BETWEEN the two:
    the topic list says what exists, the loaded topics are the ones that matter
    now, and the ticket section is the work queue.

    ``loaded`` is the set of topic LINE NUMBERS the seed carries in full, and
    the ``[LOADED]`` marker is applied HERE, where the line format lives. The
    caller used to re-parse the rendered lines with a regex and a
    ``split(") ")`` to recover a heading it already had, which made the
    format's exact spacing an undeclared interface: changing it dropped every
    marker with no error. Line numbers are also exact where a heading is not -
    two topics may share a heading, and only the selected one is loaded.

    Returns None, meaning "inject the file as it is", when an index would LOSE
    something:

    * no ``### `` topics at all - an index of nothing is worse than the file;
    * no ticket section - we cannot tell where the work queue starts, and
      dropping it would hide every open ticket;
    * a ``- `` fact outside the ticket section that sits under no topic - it
      would be dropped, so the model would never see it.

    The last rule is why the guard is not simply "any topic present". A record
    can be half-migrated - one section in topics, the rest still flat - and an
    index of that would silently hide the flat sections.

    A topic line carries its file line number, the number of lines it spans and
    its heading - and NO sampled fact. A positional sample cannot say what a
    topic holds: the first fact is the oldest content and the last is often the
    tail of a sentence whose head is three bullets up. Heading plus count reads
    as an unopened box, which is the truth. The two numbers together are the
    model's ``read_file(offset=L, limit=N)``.
    """
    shape = parse(truth)
    if not shape.topics or shape.ticket_line is None or shape.flat_facts:
        return None

    lines = truth.split("\n")
    by_line = {topic.line_no: topic for topic in shape.topics}
    out: list[str] = []
    for i, line in enumerate(lines[: shape.ticket_line]):
        topic = by_line.get(i + 1)
        if topic is not None:
            address = f"{topic.number:<5s}" if topic.number else ""
            marker = " [LOADED]" if topic.line_no in loaded else ""
            out.append(
                f"{address}L{topic.line_no:<5d} ({topic.line_count} lines){marker} {topic.heading}"
            )
        elif line.startswith("## "):
            out.append(line)
    topic_list = "\n".join(out)
    tickets = "\n".join(_collapse_parked(lines[shape.ticket_line :]))
    return topic_list, tickets


def build_truth_index(truth: str) -> str | None:
    """The whole index: topic list then ticket section, or None when lossy."""
    parts = build_truth_index_parts(truth)
    if parts is None:
        return None
    topic_list, tickets = parts
    return f"{topic_list}\n{tickets}"


# Suffixes of the two record file names; the focus name is the stem with the
# suffix removed.
TRUTH_SUFFIX = "_current_truth"
JOURNAL_SUFFIX = "_journal"


def _record_path(name: str, suffix: str) -> str:
    """The absolute path of a record file in the project's records dir.

    Resolved, so the derived path and a captured one (see ``capture_paths``)
    are the same string - a symlinked project directory would otherwise give
    two spellings of one file.
    """
    return str((get_records_dir() / f"{name}{suffix}.md").resolve())


def is_record_path(path: str) -> bool:
    """True when *path* is a file directly inside the project's records dir.

    The wrap-up's path capture uses this to tell a record write from any
    other file the model touched. Containment, not a substring: "records"
    appears in ordinary paths, so a substring test would capture the wrong
    file. A ``~``-relative path is expanded first - the file tools do the
    same.
    """
    if not path:
        return False
    try:
        Path(path).expanduser().resolve().relative_to(
            get_records_dir(create=False).resolve()
        )
    except (ValueError, OSError):
        return False
    return True


def entry_heading_time(entry_or_heading: str) -> datetime | None:
    """The timestamp of a journal entry's '## YYYY-MM-DD HH:MM' heading.

    None when there is no heading or it does not parse - the caller logs null
    rather than guessing at an age. Accepts either the whole entry or just its
    heading line.
    """
    for line in entry_or_heading.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("## "):
            continue
        try:
            return datetime.strptime(stripped[3:19].strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    return None


def heading_age_seconds(entry_or_heading: str) -> float | None:
    """Seconds since a heading's timestamp; None when it does not parse.

    NOT clamped at zero: a heading can run ahead of the clock (observed five
    minutes ahead on edmac 2026-11-08), and a negative age is the only signal
    that the record's clock and the box's disagree.
    """
    stamp = entry_heading_time(entry_or_heading)
    if stamp is None:
        return None
    return round((datetime.now() - stamp).total_seconds(), 1)


class DurableRecords:
    """Focus name + record paths + everything derived from them.

    Constructed empty by ``Agent.__init__``; state arrives through
    :meth:`set_focus_name` (user input: ``/focus``, ``/wrapup NAME``,
    ``--focus``), :meth:`capture_paths` (the wrap-up's own file writes) and
    :meth:`load` (the ``.ctx`` save).
    """

    def __init__(self) -> None:
        # Explicit focus name, set by the user (/focus NAME, /wrapup NAME,
        # --focus NAME). When set it is authoritative: the record paths
        # derive from it and the wrap-up's tool-call capture cannot
        # overwrite them. "" = unset - paths are captured from the wrap-up's
        # file writes and the name derives from them.
        self.focus_name = ""
        # Current-truth path (<project>/records/*_current_truth.md).
        self.truth_path = ""
        # Session journal path (<project>/records/*_journal.md).
        self.journal_path = ""
        # Inject a topic index instead of the whole truth file. Default False
        # so the first run is the control; set from config in Agent.__init__.
        # Only takes effect when build_truth_index() can be lossless.
        self.truth_index = False

    # ── name ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Effective focus name: the explicit name when set (authoritative),
        else derived from the captured record file names
        (<name>_current_truth.md / <name>_journal.md); "" when neither is
        recognizable."""
        if self.focus_name:
            return self.focus_name
        for path, suffix in (
            (self.truth_path, TRUTH_SUFFIX),
            (self.journal_path, JOURNAL_SUFFIX),
        ):
            if not path:
                continue
            stem = Path(path).stem
            if stem.endswith(suffix) and len(stem) > len(suffix):
                return stem[: -len(suffix)]
        return ""

    def set_focus_name(self, name: str) -> None:
        """Set the explicit focus name; the record paths derive from it.

        User input is authoritative: an explicitly-set name takes
        precedence over the path-derived name, and the wrap-up's tool-call
        capture can no longer overwrite the record paths. Setting a new name
        re-derives both paths (e.g. /wrapup NAME after /focus OTHER moves
        the record to NAME's files).
        """
        name = (name or "").strip()
        if not name:
            return
        self.focus_name = name
        self.truth_path = _record_path(name, TRUTH_SUFFIX)
        self.journal_path = _record_path(name, JOURNAL_SUFFIX)

    # ── the two command paths ─────────────────────────────────────────────

    def apply_focus(self, name: str, messages: list[dict]) -> tuple[str, str]:
        """Apply ``/focus [NAME]`` immediately (the UI command path).

        Runs at call time - not at the next queue slot - so a mid-turn
        ``/focus NAME`` lands before a threshold-triggered wrap-up builds its
        prompt, and record targeting (wrap-up files, the compact pointer
        block, restart seeding) follows the new name. On an empty message
        history the focus's durable records are seeded into ``messages`` (a
        mid-turn history is never empty, so seeding only happens while
        idle); mid-session it is set-only. No name: report the current one.

        Returns ``(message, level)`` for the UI to present - level is
        "info" (shown green/bright) or "warning" (shown yellow). The TUI
        puts it in the info pane like the other state-reporting commands;
        the REPL prints it to the console. No event is emitted, so the
        presentation is the UI's to choose.
        """
        name = (name or "").strip()
        if not name:
            current = self.name
            if current:
                return f"Focus name: {current}", "info"
            return "No focus name set", "warning"
        self.set_focus_name(name)
        seeded = self.seed(messages)
        if seeded:
            return f"Focus name set to {name} - records loaded", "info"
        if not messages:
            return f"Focus name set to {name} - no records found", "warning"
        return f"Focus name is now set to {name}", "info"

    def startup_focus(
        self, name: str, messages: list[dict]
    ) -> tuple[str, str, str | None]:
        """Apply ``--focus NAME`` at startup (the one wiring for 4 UIs).

        Sets the authoritative name. On an EMPTY history (fresh session)
        the records become the session's FIRST TURN - the --read pattern:
        this returns the seed message (records + ack instruction) as the
        third value and the UI enqueues it via
        ``agent.add_message(msg, kind="prompt")``. The model reads the
        records and replies a minimal ack while the user types their first
        prompt; that prompt queues behind it and is answered on a warm
        context. On a resumed (non-empty) history nothing is enqueued -
        seeding a resumed session would double-inject the records.

        Call it AFTER any resume load, so a resumed history is never
        double-seeded.

        Returns ``(message, level, enqueue_msg)``. ``message`` is "" when
        the records were enqueued (or there is nothing to say); each UI
        keeps its own channel for a non-empty message (stderr, the debug
        log, the TUI info pane). ``enqueue_msg`` is the records message to
        enqueue, or None when there is nothing to enqueue (no name, a
        resumed history, or no record content).
        """
        name = (name or "").strip()
        if not name:
            return "", "info", None
        self.set_focus_name(name)
        if not messages:
            # Fresh session: the records become the first turn.
            enqueue_msg = self.build_seed_message()
            if enqueue_msg:
                return "", "info", enqueue_msg
            return f"No records found for focus '{name}' - name set", "warning", None
        # Non-empty history (a resumed session): seed() would skip without
        # looking, so nothing is known about the records and nothing is
        # claimed. The former wiring printed "no records found" here, which
        # was a false report on every --focus resume.
        return "", "info", None

    # ── capture from the wrap-up ──────────────────────────────────────────

    def capture_paths(self, new_messages: list[dict]) -> None:
        """Capture the journal and current-truth paths from the wrap-up's
        own tool calls.

        The wrap-up prompt tells the model to append to
        records/{intent_or_focus}_journal.md and rewrite
        records/{focus}_current_truth.md. The paths the model chose are
        carried across chain boundaries (the restart intro names them, the
        compact pointer block appends them), so each later wrap-up touches
        the same files. Last matching write wins.

        Skipped entirely when an explicit focus name is set: the derived
        paths are authoritative and the wrap-up's file writes can't move the
        record.
        """
        if self.focus_name:
            return
        for msg in new_messages:
            for call in msg.get("tool_calls") or []:
                try:
                    fn = call["function"]
                    if fn["name"] not in ("write_file", "edit_file"):
                        continue
                    # The file tools' parameter is "filepath" (write_file/
                    # edit_file signatures). Reading "path" here always
                    # returned "" - the nudge never carried the journal path.
                    path = json.loads(fn["arguments"]).get("filepath", "")
                except (KeyError, TypeError, ValueError):
                    continue
                if not is_record_path(path):
                    continue
                # Stored resolved: the model may write the "~" form the
                # prompt shows it, and every reader here opens the string
                # directly (Path.read_text does not expand "~").
                resolved = str(Path(path).expanduser().resolve())
                if resolved.endswith(f"{JOURNAL_SUFFIX}.md"):
                    self.journal_path = resolved
                elif resolved.endswith(f"{TRUTH_SUFFIX}.md"):
                    self.truth_path = resolved

    # ── record file reads ─────────────────────────────────────────────────

    def read_record_file(self, path: str) -> str:
        """Read a record file (relative to cwd); "" when absent or
        unreadable.

        An instance method (not a module function) so a test can patch it on
        the object: ``agent.durable_records.read_record_file = MagicMock(...)``
        - the seam the seeding tests are written against.
        """
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def read_last_journal_entry(self, path: str) -> str:
        """Return the LAST entry of a journal file - from its last dated
        '## ' heading to EOF; "" when the file is absent, unreadable, or has
        no recognizable entry."""
        content = self.read_record_file(path)
        if not content:
            return ""
        lines = content.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("## "):
                return "\n".join(lines[i:]).strip()
        return ""

    def read_journal_entries(self, path: str) -> list[str]:
        """Every entry of a journal file, oldest first; [] when unreadable.

        ONE read of the file, split on its '## ' headings. The journal is
        megabytes (edmac's is 2.74 MB) and the goal ask runs at every turn
        end, so a read per entry is not an option.
        """
        content = self.read_record_file(path)
        if not content:
            return []
        entries: list[str] = []
        current: list[str] = []
        for line in content.splitlines():
            if line.startswith("## "):
                if current:
                    entries.append("\n".join(current).strip())
                current = [line]
            elif current:
                current.append(line)
            # Anything before the first heading is a title, not an entry.
        if current:
            entries.append("\n".join(current).strip())
        return entries

    def last_entry_with_handoff(self, path: str) -> str:
        """The newest journal entry carrying a handoff block; "" when none.

        An entry whose block parses with ``for_user: []`` WINS. An empty list
        is a deliberate claim, and skipping it would turn "nothing
        outstanding" into "look further back and resurrect an answered item".
        Only entries with NO block are skipped - which is the hole this
        closes: the agent appends a manual entry at the end of most turns,
        and those carry no block, so reading only the last entry returned []
        at 7 of 7 goal asks overnight while three items were open.
        """
        from agent13.handoff import parse_block

        for entry in reversed(self.read_journal_entries(path)):
            if parse_block(entry) is not None:
                return entry
        return ""

    def for_user_state(self) -> tuple[list[str], str]:
        """(items, source heading) from the newest entry that HAS a block.

        One journal read, and the ONE function both the seed and the goal
        question call, so they cannot disagree about which entry is the
        source. The heading is the entry's own '## ' line, so a caller can
        log how stale the carried list is.
        """
        from agent13.handoff import for_user_items, parse_block

        entry = self.last_entry_with_handoff(self.journal_path)
        if not entry:
            return [], ""
        heading = entry.split("\n", 1)[0].strip()
        return for_user_items(parse_block(entry)), heading

    def records_mtime(self) -> float:
        """Newest mtime across the record files, or 0.0 when unknown.

        Answers one question: *does a durable record exist to point at?* 0.0
        means "no record", which is the branch that falls back to
        AUTO_CONTEXT_CONTINUE_HINT instead of the record pointer (see
        core.py's _goal_restart_message). It must consider BOTH files: the
        pointer names the journal's "b. NEXT SESSION, FIRST ACTION", and a
        wrap-up whose reconcile finds nothing to correct leaves the truth file
        untouched while still appending the journal. Statting only the truth
        file would read that routine shape as "no record".

        0.0 sorts below any real timestamp, so an unset path or a missing
        file falls through to the hint - the safe branch.
        """
        newest = 0.0
        for path in (self.truth_path, self.journal_path):
            if not path:
                continue
            try:
                newest = max(newest, Path(path).stat().st_mtime)
            except OSError:
                continue
        return newest

    # ── seeding ───────────────────────────────────────────────────────────

    def _indexed_truth(
        self,
        truth: str,
        journal_entry: str,
        for_user: tuple[int, str] = (0, ""),
    ) -> str:
        """Preamble + topic list + the loaded topics + ticket section.

        Falls back to the file unchanged (and WITHOUT the preamble, which says
        the facts are not loaded) when an index would lose something.

        The loaded set comes from the wrap-up's own handoff block first: it
        names the addresses the next action needs, and it is the only source
        that knows what the next action IS. Failing that, ``truth_select``
        scores the last journal entry's b. block unioned with the rank-1
        ticket - the previous session's statement of what to do next. Failing
        that, nothing loads: a weak match puts a wrong fact in context, which
        is worse than an unopened box.

        ``for_user`` is (count, source heading) from ``for_user_state()`` -
        passed in rather than re-read here, so the logged count and the list
        printed at the top of the seed always describe the same entry.
        """
        from agent13.handoff import parse_block
        from agent13.truth_select import build_query, select_by_address, select_topics

        block = parse_block(journal_entry)
        nominated = list(block.load) if block else []
        selected, not_found, over_budget = select_by_address(truth, nominated)
        if selected:
            source = "json"
            query = ""
        else:
            query = build_query(journal_entry, truth)
            selected = select_topics(truth, query)
            source = "matched" if selected else "floor"

        # Built AFTER selection: the marker is applied by the renderer, which
        # needs to know what was selected. A lossy record returns the file
        # unchanged, and the selection above is then simply discarded.
        parts = build_truth_index_parts(truth, {s.line_no for s in selected})
        if parts is None:
            return truth
        topic_list, tickets = parts

        body = [topic_list]
        if selected:
            # s.text is the topic's own span, heading included - re-emitting it
            # verbatim is what makes the loaded copy byte-identical to the file.
            body.append("\n\n".join(s.text for s in selected))
        body.append(tickets)
        preamble = TRUTH_INDEX_PREAMBLE.format(path=self.truth_path)
        self._log_seed(
            selected,
            len(preamble) + len("\n".join(body)),
            query,
            source=source,
            nominated=nominated,
            not_found=not_found,
            over_budget=over_budget,
            for_user=for_user[0],
            for_user_source=for_user[1],
        )
        return preamble + "\n".join(body)

    def _log_seed(
        self,
        selected,
        total: int,
        query: str,
        source: str = "matched",
        nominated: list[str] | None = None,
        not_found: list[str] | None = None,
        over_budget: list[str] | None = None,
        for_user: int = 0,
        for_user_source: str = "",
    ) -> None:
        """One tps_truth_seed event per seed - the only way to tune the numbers.

        Both controls are logged: the THRESHOLD only guards the floor (a real
        record has ~18 topics above it), so the BUDGET is what actually limits
        the load and the two cannot be tuned from one number.

        ``source`` is the question the rollout turns on: ``json`` means the
        wrap-up's own nomination was sufficient, ``matched`` means the selector
        had to guess, ``floor`` means nothing was carried at all.

        ``not_found`` and ``over_budget`` are kept APART on purpose: the first
        is the model inventing a handle, the second is the harness dropping a
        real topic, and they need different fixes. Conflating them hid a
        dropped topic behind a plausible-looking address on the first live run.

        ``for_user_source`` names the entry the carried list came from: the
        list can be hours old, and an invisible assumption is what the
        walk-back exists to stop.
        """
        from agent13.debug_log import is_debug_enabled, log_tps_event
        from agent13.truth_select import DEFAULT_BUDGET, DEFAULT_THRESHOLD

        if not is_debug_enabled():
            return
        log_tps_event(
            "truth_seed",
            {
                "source": source,
                "nominated": nominated or [],
                "not_found": not_found or [],
                "over_budget": over_budget or [],
                "for_user_carried": for_user,
                "for_user_source": for_user_source,
                "topics": [
                    {"heading": s.heading, "score": s.score, "chars": len(s.text)}
                    for s in selected
                ],
                "loaded_chars": sum(len(s.text) for s in selected),
                "seed_chars": total,
                "floor_hit": not selected,
                "threshold": DEFAULT_THRESHOLD,
                "budget": DEFAULT_BUDGET,
                "query_chars": len(query),
            },
        )

    def unresolved_for_user(self) -> list[str]:
        """The carried ``for_user`` items, capped. [] when none.

        Read from the journal rather than held in memory: the block is
        persisted append-only, so a chain restart, a resume or a fresh process
        all see the same list.

        The source is the newest entry that HAS a block, not simply the last
        entry: most turns append a manual entry, and those carry no block.
        """
        return self.for_user_state()[0]

    def capture_handoff(self, text: str) -> bool:
        """Append the wrap-up's handoff block to the journal entry.

        The final message is discarded at rebuild, so the block is captured
        when the turn completes and appended to the entry the model just
        wrote. Written LAST (so it carries the insight the summary produced),
        persisted append-only (so it survives any rebuild), read mechanically
        by the seed builder (so no prose parsing).

        Returns True when a block was found and appended. A wrap-up that
        emitted none is not an error - the seed falls back to the selector and
        ``tps_handoff_json`` records the miss.
        """
        from agent13.debug_log import is_debug_enabled, log_tps_event
        from agent13.handoff import parse_block, render_block

        block = parse_block(text)
        if is_debug_enabled():
            log_tps_event(
                "handoff_json",
                {
                    "parsed": block is not None,
                    "load_count": len(block.load) if block else 0,
                    "for_user_count": len(block.for_user) if block else 0,
                    "unknown_keys": block.unknown if block else [],
                },
            )
        if block is None or not self.journal_path:
            return False
        try:
            with open(self.journal_path, "a", encoding="utf-8") as handle:
                handle.write("\n\n" + render_block(block) + "\n")
        except OSError:
            return False
        return True

    def _record_parts(self) -> list[str] | None:
        """The seed message body: preamble + current truth + LAST journal
        entry. None when there is no focus name or no record content.

        Shared by seed() (mechanical injection) and build_seed_message()
        (the --focus startup preprocess turn) so the two paths cannot
        diverge on what counts as "records found".
        """
        name = self.name
        if not name:
            return None
        truth = self.read_record_file(self.truth_path)
        journal_entry = self.read_last_journal_entry(self.journal_path)
        if not truth and not journal_entry:
            return None
        parts = [SEED_PREAMBLE.format(name=name)]
        # Lifted above the index rather than left inside the journal entry: if
        # the point is that the next session must act on them, burying them in
        # a 9k-char entry defeats it.
        from agent13.handoff import render_for_user

        # ONE call for both the printed list and the logged count: the seed
        # must not print items from one entry and log a count from another.
        pending_items, pending_source = self.for_user_state()
        pending = render_for_user(pending_items)
        if pending:
            parts.append(pending)
        if truth:
            if self.truth_index:
                truth = self._indexed_truth(
                    truth,
                    journal_entry,
                    for_user=(len(pending_items), pending_source),
                )
            parts.append(f"Current truth ({self.truth_path}):\n\n{truth}")
        if journal_entry:
            parts.append(
                f"Last journal entry ({self.journal_path}):\n\n{journal_entry}"
            )
        return parts

    def build_seed_message(self) -> str:
        """The records as ONE user message with an ack instruction - the
        payload the ``--focus`` startup path enqueues as the session's
        first turn (the --read pattern: file contents + ``Reply with ...
        and nothing else.``).

        The model's reply replaces the mechanical "[Loaded records ...]"
        ack of seed(): a real turn grounds the model AND warms the KV
        prefix on the records while the user types. "" when there is no
        focus name or no record content (the caller then has nothing to
        enqueue).
        """
        parts = self._record_parts()
        if not parts:
            return ""
        parts.append(SEED_ACK_INSTRUCTION.format(name=self.name))
        return "\n\n".join(parts)

    def seed(self, messages: list[dict], force: bool = False) -> bool:
        """Mechanically inject the records into a message history - the
        ``/focus`` command path and the auto-chain post-compact re-injection.

        By default only acts when the history is empty (fresh session, or
        right after /new_session). With force=True, appends to an existing
        history (used by the auto-chain post-compact path to re-inject
        records alongside the bridge summary). Injects the records as a
        user message followed by a wire-visible assistant ack - the
        standard valid user/assistant pair, zero model calls. Returns True
        when records were injected, False when no record content was found
        (or the history was non-empty without force).

        The ``--focus`` STARTUP path does not come through here: it
        enqueues build_seed_message() as a real turn instead (see
        startup_focus).
        """
        if messages and not force:
            return False
        parts = self._record_parts()
        if not parts:
            return False
        name = self.name
        messages.append({"role": "user", "content": "\n\n".join(parts)})
        messages.append(
            {
                "role": "assistant",
                "content": f"[Loaded records for focus {name}]",
            }
        )
        return True

    def restart_intro(self) -> str:
        """The pre-baked intro message for a mechanical chain restart.

        Names both record files so the post-restart context carries its own
        pointers even when no records were found to inject.
        """
        return CHAIN_RESTART_INTRO.format(
            truth_file=self.truth_path or "unknown",
            journal_file=self.journal_path or "unknown",
        )

    # ── compact pointer block ─────────────────────────────────────────────

    def append_pointer(self, summary: str) -> str:
        """Append the durable-record pointer block to a verbose compact
        summary, when the record paths are known.

        The 09-29 games incident: the verbose compact (328,500 prompt ->
        4,857-token summary) dropped file-resident knowledge - the summary
        named no record files, the new 28k context carried no pointer, and
        the agent re-struggled ~1 h before the next wrap-up's file reads
        revealed the fix. The paths are captured facts (the wrap-up's own
        file writes), so the block is appended mechanically: a model-written
        pointer is exactly what dropped.
        """
        if not (self.truth_path or self.journal_path):
            return summary
        focus = self.name
        lines = [
            COMPACT_DURABLE_RECORD_HEADER.format(focus=focus)
            if focus
            else COMPACT_DURABLE_RECORD_HEADER_BARE
        ]
        if self.truth_path:
            lines.append(
                (
                    COMPACT_DURABLE_RECORD_TRUTH_LINE_INDEXED
                    if self.truth_index
                    else COMPACT_DURABLE_RECORD_TRUTH_LINE
                ).format(truth_path=self.truth_path)
            )
        if self.journal_path:
            lines.append(
                COMPACT_DURABLE_RECORD_JOURNAL_LINE.format(
                    journal_path=self.journal_path
                )
            )
        return summary + "\n\n" + "\n".join(lines)

    # ── wrap-up currency ──────────────────────────────────────────────────

    def wrapup_in_progress(self, messages: list[dict]) -> bool:
        """True when a wrap-up STARTED but never produced its final response.

        The distinction that matters on resume: ``is_wrapup_current`` answers
        "is the record current?", which is True both for a wrap-up that
        finished and for one that is still running. Resuming needs the second
        question on its own - a wrap-up interrupted by a pause/quit must be
        FINISHED, not restarted, because starting a second one runs it on top
        of the half-finished first and the restart then wipes both.

        Derived from the history like its sibling: the wrap-up injects a
        "[Wrapping up]" assistant message marked report_and_compact, and the
        turn's final response is the first assistant message after it with no
        tool_calls. No completion -> still in progress.
        """
        marker_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("report_and_compact") and msg.get("role") == "assistant":
                marker_idx = i
        if marker_idx == -1:
            return False
        for msg in messages[marker_idx + 1 :]:
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                return False
        return True

    def is_wrapup_current(self, messages: list[dict]) -> bool:
        """Return True if a wrap-up ran and no tool calls (major work) have
        happened since its completion - i.e., the durable record (journal +
        truth) is still current.

        Derived from the message history (no separate flag): the wrap-up
        injects a "[Wrapping up]" assistant message marked
        report_and_compact=True. A simple Q&A (no tool calls) after the
        wrap-up keeps the record current; any tool call (file changes,
        commands) invalidates it. A compact replaces the history with the
        summary pair, erasing the marker, so the next /compact reverts to
        the verbose default automatically.
        """
        # Find the last wrap-up marker (the injected "[Wrapping up]"
        # assistant message).
        marker_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("report_and_compact") and msg.get("role") == "assistant":
                marker_idx = i
        if marker_idx == -1:
            return False
        # Find the wrap-up's completion: the first assistant message after
        # the marker with no tool calls (the wrap-up turn's final response).
        # The wrap-up's own file-write tool calls sit between the marker and
        # this completion and must not count as "major work since".
        completion_idx = -1
        for i in range(marker_idx + 1, len(messages)):
            msg = messages[i]
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                completion_idx = i
                break
        if completion_idx == -1:
            # The wrap-up turn is still in progress (no final response yet) -
            # the files are being written; treat the record as current.
            return True
        # Any tool call (or tool result) after the completion means major
        # work happened since the wrap-up - the files no longer reflect it.
        for msg in messages[completion_idx + 1 :]:
            if msg.get("tool_calls") or msg.get("role") == "tool":
                return False
        return True

    # ── .ctx persistence ──────────────────────────────────────────────────

    def save(self) -> dict:
        """The fields this module writes into the ``.ctx`` save.

        Additive: ``focus_name`` is absent in saves written before focus
        persistence, and the record paths are not saved - they derive from
        the name (capture is in-session only).
        """
        return {"focus_name": self.focus_name}

    def load(self, context: dict) -> None:
        """Restore the focus name from a ``.ctx`` save.

        An explicit focus already set on this object (e.g. the ``--focus``
        CLI flag, applied before the continue-load) is authoritative and
        wins over the saved value.
        """
        saved_focus = context.get("focus_name", "")
        if saved_focus and not self.focus_name:
            self.set_focus_name(saved_focus)


def startup_focus(
    records: DurableRecords, name: str, messages: list[dict]
) -> tuple[str, str, str | None]:
    """Apply ``--focus NAME`` at startup - the one wiring for 4 UIs.

    cli / repl / pipemode / tui each did the same three steps by hand (set
    the authoritative name, load the records into a fresh history, report
    when nothing was found), which is four copies of one rule about record
    targeting. Each UI keeps its own channel for the report - stderr, the
    debug log, the TUI info pane - and its own enqueue mechanics (the TUI
    and REPL loops are already running; batch passes the message into
    run_batch, whose completion waits would otherwise desynchronize), so
    this returns ``(message, level, enqueue_msg)`` instead of acting: the
    UI enqueues ``enqueue_msg`` (when not None) via
    ``agent.add_message(msg, kind="prompt")`` and presents ``message``.

    Call it AFTER any resume load, so a resumed history is never
    double-seeded. See :meth:`DurableRecords.startup_focus` for the full
    contract.
    """
    return records.startup_focus(name, messages)
