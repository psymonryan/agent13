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
from pathlib import Path

from agent13.prompts import (
    CHAIN_RESTART_INTRO,
    COMPACT_DURABLE_RECORD_HEADER,
    COMPACT_DURABLE_RECORD_HEADER_BARE,
    COMPACT_DURABLE_RECORD_JOURNAL_LINE,
    COMPACT_DURABLE_RECORD_TRUTH_LINE,
)

# Header of the user message seed() injects. Core-internal (not
# yaml-exposed): it describes the shape of the record files, which is this
# module's own knowledge.
SEED_PREAMBLE = (
    "Durable record for focus {name} - the state at the last handoff. "
    "Here is the current truth in full before starting new work; its "
    "section 6 tickets are the work queue (top unblocked ticket first)."
)

# Instruction appended to the seed message on the --focus startup path,
# turning the records load into a real turn (the --read pattern: contents
# + "Reply with ... and nothing else."). The model's reply replaces the
# mechanical "[Loaded records ...]" ack of seed() and warms the KV prefix
# on the records while the user types their first prompt. Core-internal.
SEED_ACK_INSTRUCTION = (
    'Reply with "Records for focus {name} loaded." and nothing else.'
)

# Suffixes of the two record file names; the focus name is the stem with the
# suffix removed.
TRUTH_SUFFIX = "_current_truth"
JOURNAL_SUFFIX = "_journal"


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
        # Current-truth path (docs_archive/*_current_truth.md).
        self.truth_path = ""
        # Session journal path (docs_archive/*_journal.md).
        self.journal_path = ""

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
        self.truth_path = f"docs_archive/{name}{TRUTH_SUFFIX}.md"
        self.journal_path = f"docs_archive/{name}{JOURNAL_SUFFIX}.md"

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
        docs_archive/{intent_or_focus}_journal.md and rewrite
        docs_archive/{focus}_current_truth.md. The paths the model chose are
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
                if "docs_archive" not in path:
                    continue
                if path.endswith("_journal.md"):
                    self.journal_path = path
                elif path.endswith("_current_truth.md"):
                    self.truth_path = path

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

    # ── seeding ───────────────────────────────────────────────────────────

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
        if truth:
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
                COMPACT_DURABLE_RECORD_TRUTH_LINE.format(truth_path=self.truth_path)
            )
        if self.journal_path:
            lines.append(
                COMPACT_DURABLE_RECORD_JOURNAL_LINE.format(
                    journal_path=self.journal_path
                )
            )
        return summary + "\n\n" + "\n".join(lines)

    # ── wrap-up currency ──────────────────────────────────────────────────

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
