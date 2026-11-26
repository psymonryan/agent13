"""Tests for agent13.durable_records - the focus/record module.

Pure state-feed tests: construct one DurableRecords, feed it the same
messages/lists the Agent feeds it, assert on the result. No Agent, no LLM,
no events - the point of the extraction is that the whole record surface is
decidable this way (the tests/test_context_gauge.py precedent).
"""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agent13.config_paths import get_records_dir
from agent13.durable_records import (
    DurableRecords,
    entry_heading_time,
    heading_age_seconds,
    startup_focus,
)


@pytest.fixture(autouse=True)
def _project_cwd(tmp_path, monkeypatch):
    """The records directory is ``<cwd>/records`` - run every test in
    a temp project so none of them touch the repo's own."""
    monkeypatch.chdir(tmp_path)


def _mem(filename: str) -> str:
    """The absolute path of a record file in the current project's memory
    directory."""
    return str((get_records_dir() / filename).resolve())


def _records(name=None, truth=None, journal=None):
    """DurableRecords with the record readers stubbed (no filesystem)."""
    records = DurableRecords()
    if name:
        records.set_focus_name(name)
    records.read_record_file = MagicMock(return_value=truth or "")
    records.read_last_journal_entry = MagicMock(return_value=journal or "")
    return records


# ── name authority ────────────────────────────────────────────────────────


class TestName:
    def test_explicit_name_is_authoritative(self):
        records = DurableRecords()
        records.truth_path = "docs_archive/captured_current_truth.md"
        records.set_focus_name("explicit")
        assert records.name == "explicit"

    def test_derived_from_truth_path(self):
        records = DurableRecords()
        records.truth_path = "docs_archive/auto_context_current_truth.md"
        assert records.name == "auto_context"

    def test_derived_from_journal_path(self):
        records = DurableRecords()
        records.journal_path = "docs_archive/games_journal.md"
        assert records.name == "games"

    def test_unrecognizable_path_yields_no_name(self):
        records = DurableRecords()
        records.truth_path = "docs_archive/notes.md"
        assert records.name == ""

    def test_bare_suffix_yields_no_name(self):
        """"_current_truth.md" has no stem left - no name is invented."""
        records = DurableRecords()
        records.truth_path = "docs_archive/_current_truth.md"
        assert records.name == ""

    def test_no_state_yields_no_name(self):
        assert DurableRecords().name == ""


class TestSetFocusName:
    def test_derives_both_paths(self):
        records = DurableRecords()
        records.set_focus_name("my game")
        assert records.focus_name == "my game"
        assert records.truth_path == _mem("my game_current_truth.md")
        assert records.journal_path == _mem("my game_journal.md")

    def test_rederives_on_change(self):
        records = DurableRecords()
        records.set_focus_name("alpha")
        records.set_focus_name("beta")
        assert records.truth_path == _mem("beta_current_truth.md")
        assert records.journal_path == _mem("beta_journal.md")

    def test_ignores_blank(self):
        records = DurableRecords()
        records.set_focus_name("")
        records.set_focus_name("   ")
        records.set_focus_name(None)
        assert records.name == ""
        assert records.truth_path == ""
        assert records.journal_path == ""

    def test_strips_whitespace(self):
        records = DurableRecords()
        records.set_focus_name("  auto_context  ")
        assert records.focus_name == "auto_context"


# ── /focus (apply_focus) ──────────────────────────────────────────────────


class TestApplyFocus:
    def test_no_name_reports_current(self):
        records = _records(name="auto_context")
        assert records.apply_focus("", []) == ("Focus name: auto_context", "info")

    def test_no_name_no_focus_warns(self):
        assert DurableRecords().apply_focus("  ", []) == (
            "No focus name set",
            "warning",
        )

    def test_empty_history_seeds_records(self):
        records = _records(truth="TRUTH", journal="## 2026-01-01 x\nENTRY")
        messages = []

        message, level = records.apply_focus("my game", messages)

        assert message == "Focus name set to my game - records loaded"
        assert level == "info"
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_empty_history_no_records_warns(self):
        records = _records()
        messages = []

        message, level = records.apply_focus("missing", messages)

        assert message == "Focus name set to missing - no records found"
        assert level == "warning"
        assert messages == []
        # The name is set even when there is nothing to seed.
        assert records.focus_name == "missing"

    def test_mid_session_is_set_only(self):
        records = _records(truth="TRUTH")
        messages = [{"role": "user", "content": "mid session"}]

        message, level = records.apply_focus("my game", messages)

        assert message == "Focus name is now set to my game"
        assert level == "info"
        assert len(messages) == 1


# ── --focus at startup (the collapsed wiring) ─────────────────────────────


class TestStartupFocus:
    def test_blank_name_is_a_noop(self):
        records = _records(truth="TRUTH")
        messages = []

        message, level, enqueue_msg = startup_focus(records, "  ", messages)

        assert (message, level, enqueue_msg) == ("", "info", None)
        assert records.focus_name == ""
        assert messages == []

    def test_fresh_history_returns_the_seed_message_to_enqueue(self):
        """The records become the session's FIRST TURN: the message is
        returned for the UI to enqueue, the history is NOT touched here."""
        records = _records(truth="TRUTH", journal="## 2026-01-01 x\nENTRY")
        messages = []

        message, level, enqueue_msg = startup_focus(records, "my game", messages)

        assert (message, level) == ("", "info")
        assert enqueue_msg is not None
        assert "Durable record for focus my game" in enqueue_msg
        assert "TRUTH" in enqueue_msg
        assert 'Reply with "Records for focus my game loaded."' in enqueue_msg
        # The history is untouched - the enqueue is the UI's job.
        assert messages == []
        assert records.focus_name == "my game"

    def test_missing_records_reports_warning(self):
        records = _records()
        messages = []

        message, level, enqueue_msg = startup_focus(records, "missing", messages)

        assert message == "No records found for focus 'missing' - name set"
        assert level == "warning"
        assert enqueue_msg is None
        assert records.focus_name == "missing"

    def test_resumed_history_is_not_double_seeded(self):
        """A resumed (non-empty) history is left alone - the name is set,
        nothing is enqueued, and nothing is printed about records that were
        never looked for."""
        records = _records(truth="TRUTH")
        messages = [{"role": "user", "content": "resumed"}]

        message, _level, enqueue_msg = startup_focus(records, "my game", messages)

        assert message == ""
        assert enqueue_msg is None
        assert len(messages) == 1


class TestBuildSeedMessage:
    def test_truth_only(self):
        records = _records(name="my game", truth="TRUTH")
        msg = records.build_seed_message()
        assert msg.startswith("Durable record for focus my game - ")
        assert f"Current truth ({_mem('my game_current_truth.md')})" in msg
        assert "TRUTH" in msg
        assert 'Reply with "Records for focus my game loaded."' in msg
        # No journal section when there is no journal entry.
        assert "Last journal entry" not in msg

    def test_journal_only(self):
        records = _records(name="my game", journal="## 2026-01-01 x\nENTRY")
        msg = records.build_seed_message()
        assert f"Last journal entry ({_mem('my game_journal.md')})" in msg
        assert "ENTRY" in msg
        assert "Current truth" not in msg

    def test_no_records_is_empty(self):
        assert _records(name="my game").build_seed_message() == ""

    def test_no_name_is_empty(self):
        assert _records(truth="TRUTH").build_seed_message() == ""

    def test_matches_the_mechanical_seed_body(self):
        """Both paths share _record_parts, so the user-message body the
        model sees is identical; only the ack differs (real turn vs
        mechanical string)."""
        records = _records(name="my game", truth="TRUTH", journal="## e\nJ")
        seed_messages = []
        assert records.seed(seed_messages)
        mechanical_body = seed_messages[0]["content"]
        preprocess_body = records.build_seed_message()
        assert preprocess_body.startswith(mechanical_body)
        assert preprocess_body[len(mechanical_body) :].strip() == (
            'Reply with "Records for focus my game loaded." and nothing else.'
        )


# ── wrap-up path capture ──────────────────────────────────────────────────


def _write_call(name, filepath):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": name,
                    # json.dumps, NOT an f-string. A Windows path is full of
                    # backslashes, which are JSON escape characters: the
                    # interpolated form produced invalid JSON ("\U" in
                    # C:\Users\...), json.loads raised, capture_paths
                    # swallowed it as a malformed call, and the captured path
                    # came back "" - three tests failed only on Windows.
                    "arguments": json.dumps({"filepath": filepath}),
                },
            }
        ],
    }


class TestCapturePaths:
    def test_captures_write_and_edit(self):
        records = DurableRecords()
        records.capture_paths(
            [
                _write_call("write_file", _mem("games_journal.md")),
                _write_call("edit_file", _mem("games_current_truth.md")),
            ]
        )
        assert records.journal_path == _mem("games_journal.md")
        assert records.truth_path == _mem("games_current_truth.md")

    def test_last_write_wins(self):
        records = DurableRecords()
        records.capture_paths(
            [
                _write_call("write_file", _mem("first_journal.md")),
                _write_call("write_file", _mem("second_journal.md")),
            ]
        )
        assert records.journal_path == _mem("second_journal.md")

    def test_skipped_when_explicit_name_set(self):
        """The derived paths are authoritative - the wrap-up's writes cannot
        move the record."""
        records = DurableRecords()
        records.set_focus_name("explicit")
        records.capture_paths([_write_call("write_file", _mem("other.md"))])
        assert records.journal_path == _mem("explicit_journal.md")
        assert records.truth_path == _mem("explicit_current_truth.md")

    def test_ignores_other_tools_and_dirs(self):
        records = DurableRecords()
        records.capture_paths(
            [
                _write_call("command", "docs_archive/games_journal.md"),
                _write_call("write_file", "/tmp/games_journal.md"),
            ]
        )
        assert records.journal_path == ""

    def test_tolerates_malformed_calls(self):
        """No exception on the shapes the wire can actually produce."""
        records = DurableRecords()
        records.capture_paths(
            [
                {"role": "assistant", "content": "no tool calls"},
                {"role": "assistant", "tool_calls": None},
                {"role": "assistant", "tool_calls": [{}]},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "write_file", "arguments": "{not json"}}
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "docs_archive/x_journal.md"}',
                            }
                        }
                    ],
                },
            ]
        )
        assert records.journal_path == ""


# ── seeding ───────────────────────────────────────────────────────────────


class TestSeed:
    def test_injects_valid_user_assistant_pair(self):
        records = _records(name="my game", truth="TRUTH", journal="JOURNAL ENTRY")
        messages = []

        assert records.seed(messages) is True
        assert [m["role"] for m in messages] == ["user", "assistant"]
        content = messages[0]["content"]
        assert "Durable record for focus my game" in content
        assert "TRUTH" in content
        assert "JOURNAL ENTRY" in content
        assert messages[1]["content"] == "[Loaded records for focus my game]"

    def test_names_the_record_files(self):
        """The seeded message carries the paths, not just the content."""
        records = _records(name="my game", truth="TRUTH")
        messages = []
        records.seed(messages)
        assert _mem("my game_current_truth.md") in messages[0]["content"]

    def test_noop_on_non_empty_history(self):
        records = _records(name="my game", truth="TRUTH")
        messages = [{"role": "user", "content": "mid session"}]

        assert records.seed(messages) is False
        assert len(messages) == 1

    def test_force_appends_to_existing_history(self):
        records = _records(name="my game", truth="TRUTH")
        messages = [
            {"role": "user", "content": "Bridge summary"},
            {"role": "assistant", "content": "Understood."},
        ]

        assert records.seed(messages, force=True) is True
        assert len(messages) == 4
        assert messages[0]["content"] == "Bridge summary"
        assert "TRUTH" in messages[2]["content"]

    def test_noop_without_name(self):
        records = _records(truth="TRUTH")
        messages = []
        assert records.seed(messages) is False
        assert messages == []

    def test_noop_without_record_content(self):
        records = _records(name="missing")
        messages = []
        assert records.seed(messages) is False
        assert messages == []

    def test_seed_reads_the_captured_paths(self):
        """Capture (not just an explicit name) is enough to seed: the
        wrap-up's file writes set the paths the readers are called with."""
        records = DurableRecords()
        records.read_record_file = MagicMock(return_value="TRUTH")
        records.read_last_journal_entry = MagicMock(return_value="JOURNAL ENTRY")
        records.capture_paths(
            [
                _write_call("write_file", _mem("g_journal.md")),
                _write_call("edit_file", _mem("g_current_truth.md")),
            ]
        )
        messages = []

        assert records.seed(messages) is True
        # read_record_file is called for the truth AND, by the for_user
        # walk-back, for the journal - so assert membership, not the last call.
        records.read_record_file.assert_any_call(_mem("g_current_truth.md"))
        records.read_last_journal_entry.assert_called_with(_mem("g_journal.md"))
        # The name derived from the captured file names carries through.
        assert "[Loaded records for focus g]" == messages[1]["content"]


# ── restart intro ─────────────────────────────────────────────────────────


class TestRestartIntro:
    def test_names_both_files(self):
        records = DurableRecords()
        records.set_focus_name("auto_context")
        intro = records.restart_intro()
        assert _mem("auto_context_current_truth.md") in intro
        assert _mem("auto_context_journal.md") in intro

    def test_unknown_when_no_paths(self):
        """A fresh session still gets an intro - it names "unknown" rather
        than crashing."""
        intro = DurableRecords().restart_intro()
        assert "unknown" in intro


# ── compact pointer block ─────────────────────────────────────────────────


class TestAppendPointer:
    def test_no_paths_summary_untouched(self):
        records = DurableRecords()
        assert records.append_pointer("SUMMARY") == "SUMMARY"

    def test_both_paths_with_focus_and_cues(self):
        records = DurableRecords()
        records.set_focus_name("auto_context")
        out = records.append_pointer("SUMMARY")
        assert out.startswith("SUMMARY")
        assert "Durable record (focus auto_context):" in out
        assert _mem("auto_context_current_truth.md") in out
        assert _mem("auto_context_journal.md") in out
        # The cues ride with the paths.
        assert "work queue (top unblocked ticket first)" in out
        assert "LAST entry holds the state at handoff" in out

    def test_journal_only(self):
        records = DurableRecords()
        records.journal_path = "docs_archive/games_journal.md"
        out = records.append_pointer("SUMMARY")
        assert "Durable record (focus games):" in out
        assert "docs_archive/games_journal.md" in out
        assert "current_truth" not in out

    def test_unrecognizable_name_bare_header(self):
        records = DurableRecords()
        records.truth_path = "docs_archive/notes.md"
        out = records.append_pointer("SUMMARY")
        assert "Durable record:" in out
        assert "(focus" not in out
        assert "docs_archive/notes.md" in out

    def test_truth_line_plain_when_index_off(self):
        """Index off: the seed carried the facts, so "read it in full"."""
        records = DurableRecords()
        records.truth_path = "docs_archive/x_current_truth.md"
        records.truth_index = False
        out = records.append_pointer("SUMMARY")
        assert "read it in full before starting new work" in out
        assert "the facts are NOT loaded" not in out

    def test_truth_line_indexed_when_index_on(self):
        """Index on: the seed carried a topic index, not the facts - saying
        "read it in full" would contradict the index preamble, and the L
        numbers are a snapshot, so the heading grep is the stable address."""
        records = DurableRecords()
        records.truth_path = "docs_archive/x_current_truth.md"
        records.truth_index = True
        out = records.append_pointer("SUMMARY")
        assert "the facts are NOT loaded" in out
        assert "grep '^### <heading>'" in out
        assert "read it in full" not in out
        # The work-queue half survives in both modes.
        assert "top unblocked ticket first" in out


# ── record mtime (the nudge's recency test) ─────────────────────────────────


class TestRecordsMtime:
    def test_unknown_without_paths(self):
        assert DurableRecords().records_mtime() == 0.0

    def test_unknown_when_file_missing(self, tmp_path):
        records = DurableRecords()
        records.truth_path = str(tmp_path / "nope.md")
        records.journal_path = str(tmp_path / "also_nope.md")
        assert records.records_mtime() == 0.0

    def test_newest_of_both_files(self, tmp_path):
        """The journal counts: a wrap-up whose reconcile finds nothing to
        correct leaves the truth file untouched but still appends the
        journal, and the hint points at the journal's b. section."""
        truth = tmp_path / "t.md"
        journal = tmp_path / "j.md"
        truth.write_text("# t\n")
        journal.write_text("# j\n")
        os.utime(truth, (1_000_000, 1_000_000))
        os.utime(journal, (2_000_000, 2_000_000))
        records = DurableRecords()
        records.truth_path = str(truth)
        records.journal_path = str(journal)
        assert records.records_mtime() == 2_000_000

    def test_truth_alone_when_journal_absent(self, tmp_path):
        truth = tmp_path / "t.md"
        truth.write_text("# t\n")
        os.utime(truth, (1_500_000, 1_500_000))
        records = DurableRecords()
        records.truth_path = str(truth)
        records.journal_path = str(tmp_path / "missing.md")
        assert records.records_mtime() == 1_500_000


# ── wrap-up currency ──────────────────────────────────────────────────────


def _marker():
    return {
        "role": "assistant",
        "content": "[Wrapping up]",
        "report_and_compact": True,
    }


class TestIsWrapupCurrent:
    def test_no_marker(self):
        assert DurableRecords().is_wrapup_current(
            [{"role": "user", "content": "hi"}]
        ) is False

    def test_wrapup_completed_no_tools_since(self):
        messages = [
            {"role": "user", "content": "/wrapup"},
            _marker(),
            {"role": "assistant", "content": "did the files", "tool_calls": []},
            {"role": "user", "content": "quick question"},
            {"role": "assistant", "content": "short answer"},
        ]
        assert DurableRecords().is_wrapup_current(messages) is True

    def test_tool_call_after_completion_invalidates(self):
        messages = [
            _marker(),
            {"role": "assistant", "content": "done", "tool_calls": []},
            {"role": "user", "content": "now fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "command"}}],
            },
            {"role": "tool", "content": "ok"},
        ]
        assert DurableRecords().is_wrapup_current(messages) is False

    def test_wrapup_own_file_writes_do_not_count(self):
        """The wrap-up's tool calls sit between the marker and its final
        response - they are the record being written, not work since."""
        messages = [
            _marker(),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "written"},
            {"role": "assistant", "content": "Record updated."},
        ]
        assert DurableRecords().is_wrapup_current(messages) is True

    def test_in_progress_wrapup_counts_current(self):
        messages = [
            _marker(),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "written"},
        ]
        assert DurableRecords().is_wrapup_current(messages) is True

    def test_empty_history(self):
        assert DurableRecords().is_wrapup_current([]) is False


# ── .ctx save/load ────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_round_trip(self):
        records = DurableRecords()
        records.set_focus_name("auto_context")
        restored = DurableRecords()
        restored.load(records.save())
        assert restored.focus_name == "auto_context"
        assert restored.truth_path == _mem("auto_context_current_truth.md")

    def test_old_save_without_the_field(self):
        records = DurableRecords()
        records.load({"version": 1, "messages": []})
        assert records.focus_name == ""

    def test_explicit_name_wins_over_saved(self):
        """The --focus CLI flag is applied before the continue-load and is
        authoritative."""
        records = DurableRecords()
        records.set_focus_name("cli_focus")
        records.load({"focus_name": "saved_focus"})
        assert records.focus_name == "cli_focus"
        assert records.truth_path == _mem("cli_focus_current_truth.md")

    def test_saved_name_kept_when_nothing_explicit(self):
        records = DurableRecords()
        records.load({"focus_name": "saved_focus"})
        assert records.name == "saved_focus"

    def test_save_is_json_serialisable(self):
        import json

        json.dumps(DurableRecords().save())


# ── the record readers (real filesystem) ──────────────────────────────────


class TestRecordReaders:
    def test_read_missing_file_is_empty(self, tmp_path):
        assert DurableRecords().read_record_file(str(tmp_path / "nope.md")) == ""

    def test_read_empty_path_is_empty(self):
        assert DurableRecords().read_record_file("") == ""

    def test_read_last_entry_is_the_latest_heading(self, tmp_path):
        path = tmp_path / "f_journal.md"
        path.write_text(
            "# journal\n\n## 2026-01-01 10:00 - old\nold body\n\n"
            "## 2026-02-02 11:00 - new\nnew body\n",
            encoding="utf-8",
        )
        last = DurableRecords().read_last_journal_entry(str(path))
        assert last.startswith("## 2026-02-02 11:00 - new")
        assert "new body" in last
        assert "old body" not in last

    def test_journal_without_headings_is_empty(self, tmp_path):
        path = tmp_path / "f_journal.md"
        path.write_text("just prose, no dated entry\n", encoding="utf-8")
        assert DurableRecords().read_last_journal_entry(str(path)) == ""

    def test_seed_reads_the_real_files(self, tmp_path, monkeypatch):
        """End-to-end through the real readers (no mocks): the focus name
        derives the paths and the files are found there.

        The autouse fixture has chdir'd into tmp_path, so the records
        directory is the temp project's, not the repo's.
        """
        records_dir = get_records_dir()
        (records_dir / "my game_current_truth.md").write_text(
            "TRUTH", encoding="utf-8"
        )
        (records_dir / "my game_journal.md").write_text(
            "## 2026-01-01 10:00 - STALE\nSTALE\n\n## 2026-02-02 11:00 - new\nnew\n",
            encoding="utf-8",
        )
        records = DurableRecords()
        records.set_focus_name("my game")
        messages = []

        assert records.seed(messages) is True
        content = messages[0]["content"]
        assert "TRUTH" in content
        assert "new" in content
        # Not "old": the memory path is embedded in the content, and a
        # tmp_path containing "folders" would match it.
        assert "STALE" not in content


# ── for_user walk-back ───────────────────────────────────────────────────────
#
# The headings and handoff blocks below are VERBATIM from edmac's journal
# (~/Resilio/work/projects/edmac/docs_archive/edmac_journal.md). The entry
# bodies are elided: the mechanism under test is which entry's block wins, not
# what the entry says.
#
# The failure being pinned, measured 2026-11-07 23:24 - 2026-11-08 08:05: the
# agent appends a manual entry at the end of most turns (10 of 13 journal
# writes overnight), none of them carries a block, and the goal ask fires
# seconds later - so reading only the LAST entry returned [] at 7 of 7 asks
# while three items were open the whole time.

EDMAC_0350 = """\
## 2026-11-06 03:50 - ride 1731: the +0x110 watchpoint premise was wrong; LINK1 now passes, wall is LINK2

**b. NEXT SESSION, FIRST ACTION** - find what sets the state-6 field.

```json
{"load": ["3.7", "3.6", "3.9", "3.16", "4.5", "3.8"], "for_user": ["Standing ask (first raised 2026-10-24): the custom key-binding list - the CONTROLS screen works and persists, but your layout is not set.", "Cleanup left the repo-root PNGs (1.8 GB, 537 files) and res/ (1.1 GB) untouched - the tool does not manage them and 16 repo-root shots are cited in the journal, so deleting the rest needs your keep-list decision.", "The open-galaxy ticket is now closed as a tutorial question: the map's selection is gated to the scenario's 2 locations, so the real game (SOLO PLAY) is the only path to an open galaxy - your 22:40 ranking is therefore satisfied only through that ticket, which is now rank 1."]}
```
"""

EDMAC_0545 = """\
## 2026-11-06 05:45 - rides 1737-1743: guard131 moves the first gate object to state 8; the gate still fails on the SECOND slot, which is never enqueued

**b. NEXT SESSION, FIRST ACTION** - find what sets `[obj+0x2d0]`.
"""

EDMAC_0615 = """\
## 2026-11-06 06:15 - ride 1745: the last gate's second object is NEVER ENQUEUED - the online load is server-data-blocked, measured

**b. NEXT SESSION, FIRST ACTION** - build the level-1 standalone launcher.
"""

EDMAC_1125 = """\
## 2026-11-07 23:25 - age gate: the octree message is a teardown race - partial

**b. NEXT SESSION, FIRST ACTION** - build the dylib inline hook.

```json
{"load": ["3.4", "4.6", "4.5", "4.2", "1.6", "3.3"], "for_user": ["The age-ticket A/B needs a dylib inline hook, not lldb - an attach brakes the load 20-30x and timing is the variable; next session builds that first.", "Rank 1 still needs you for one ride: a jump attempt to a refused in-range system with a breakpoint armed at 0x100efaf69.", "Leave the Studio session unlocked for unattended rides - a locked screen breaks input injection."]}
```
"""

EDMAC_0045 = """\
## 2026-11-08 00:45 - age gate: it is not a threshold, and it is the producer bug - NEGATIVE

**b. NEXT SESSION, FIRST ACTION** - write the age-gate A/B.
"""


def _journal_of(*entries):
    return "\n\n".join(entries) + "\n"


def _journal_records(journal):
    """DurableRecords whose journal file is `journal` (truth is irrelevant)."""
    records = DurableRecords()
    records.journal_path = "docs_archive/edmac_journal.md"
    records.read_record_file = MagicMock(return_value=journal)
    return records


class TestForUserWalkBack:
    """The source is the newest entry that HAS a block, not the last entry."""

    def test_the_last_entry_with_a_block_wins(self):
        """The 06:08 accident: two manual entries after the block."""
        records = _journal_records(_journal_of(EDMAC_0350, EDMAC_0545, EDMAC_0615))

        items = records.unresolved_for_user()

        assert len(items) == 3
        assert items[0].startswith("Standing ask (first raised 2026-10-24)")

    def test_the_overnight_case(self):
        """The case that failed seven times in a row: the ask fires seconds
        after a manual entry lands, and the entry before it has the items."""
        records = _journal_records(_journal_of(EDMAC_1125, EDMAC_0045))

        items = records.unresolved_for_user()

        assert len(items) == 3
        assert items[1].startswith("Rank 1 still needs you for one ride")

    def test_a_deliberate_empty_list_wins_over_an_earlier_block(self):
        """An empty list is a claim, not an omission - resurrecting an
        answered item is the opposite error."""
        empty = (
            "## 2026-11-08 09:00 - done\n\n"
            '```json\n{"load": [], "for_user": []}\n```\n'
        )
        records = _journal_records(_journal_of(EDMAC_1125, empty))

        assert records.unresolved_for_user() == []

    def test_no_block_anywhere_is_empty(self):
        records = _journal_records(_journal_of(EDMAC_0545, EDMAC_0615))
        assert records.unresolved_for_user() == []

    def test_a_malformed_block_falls_through_to_an_earlier_one(self):
        broken = (
            "## 2026-11-08 09:00 - broken\n\n"
            "```json\n{\"load\": [\"3.1\"], \"for_user\": [\n```\n"
        )
        records = _journal_records(_journal_of(EDMAC_1125, broken))

        items = records.unresolved_for_user()

        assert len(items) == 3
        assert items[0].startswith("The age-ticket A/B")

    def test_the_cap_holds_across_the_walk(self):
        many = (
            "## 2026-11-08 09:00 - many\n\n"
            '```json\n{"load": [], "for_user": ["a", "b", "c", "d", "e"]}\n```\n'
        )
        records = _journal_records(_journal_of(EDMAC_1125, many))
        assert records.unresolved_for_user() == ["a", "b", "c"]

    def test_one_read_of_the_journal_per_call(self):
        """The journal is megabytes and the ask runs at every turn end."""
        records = _journal_records(_journal_of(EDMAC_0350, EDMAC_0545))

        records.unresolved_for_user()

        assert records.read_record_file.call_count == 1

    def test_the_source_heading_names_the_entry_that_was_read(self):
        records = _journal_records(_journal_of(EDMAC_0350, EDMAC_0545, EDMAC_0615))

        items, source = records.for_user_state()

        assert len(items) == 3
        assert source.startswith("## 2026-11-06 03:50")

    def test_no_source_heading_when_there_is_no_block(self):
        records = _journal_records(_journal_of(EDMAC_0545))
        assert records.for_user_state() == ([], "")

    def test_entries_are_split_oldest_first(self, tmp_path):
        path = tmp_path / "f_journal.md"
        path.write_text(
            "# journal\n\n## 2026-01-01 10:00 - old\nold\n\n## 2026-02-02 11:00 - new\nnew\n",
            encoding="utf-8",
        )

        entries = DurableRecords().read_journal_entries(str(path))

        assert entries[0].startswith("## 2026-01-01 10:00")
        assert entries[-1].startswith("## 2026-02-02 11:00")
        assert "old" in entries[0] and "new" not in entries[0]

    def test_a_sub_heading_does_not_split_an_entry(self, tmp_path):
        """'### Parked' is inside the entry, not a new one."""
        path = tmp_path / "f_journal.md"
        path.write_text(
            "## 2026-01-01 10:00 - one\n\n### Parked\n- x\n", encoding="utf-8"
        )
        assert len(DurableRecords().read_journal_entries(str(path))) == 1


class TestHeadingAge:
    """The age of the carried list is logged, so staleness is visible."""

    def test_a_heading_timestamp_is_parsed(self):
        assert entry_heading_time("## 2026-11-08 00:45 - x") == datetime(
            2026, 11, 8, 0, 45
        )

    def test_a_whole_entry_is_accepted(self):
        assert entry_heading_time(EDMAC_0350) == datetime(2026, 11, 6, 3, 50)

    def test_an_unparseable_heading_is_none(self):
        assert entry_heading_time("## no timestamp here") is None
        assert heading_age_seconds("## no timestamp here") is None
        assert heading_age_seconds("") is None

    def test_age_is_seconds_since_the_heading(self):
        age = heading_age_seconds("## 2020-01-01 00:00 - x")
        assert age is not None and age > 0
