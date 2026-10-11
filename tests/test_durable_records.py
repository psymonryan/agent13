"""Tests for agent13.durable_records - the focus/record module.

Pure state-feed tests: construct one DurableRecords, feed it the same
messages/lists the Agent feeds it, assert on the result. No Agent, no LLM,
no events - the point of the extraction is that the whole record surface is
decidable this way (the tests/test_context_gauge.py precedent).
"""

from unittest.mock import MagicMock

from agent13.durable_records import DurableRecords, startup_focus


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
        assert records.truth_path == "docs_archive/my game_current_truth.md"
        assert records.journal_path == "docs_archive/my game_journal.md"

    def test_rederives_on_change(self):
        records = DurableRecords()
        records.set_focus_name("alpha")
        records.set_focus_name("beta")
        assert records.truth_path == "docs_archive/beta_current_truth.md"
        assert records.journal_path == "docs_archive/beta_journal.md"

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
        assert "Current truth (docs_archive/my game_current_truth.md)" in msg
        assert "TRUTH" in msg
        assert 'Reply with "Records for focus my game loaded."' in msg
        # No journal section when there is no journal entry.
        assert "Last journal entry" not in msg

    def test_journal_only(self):
        records = _records(name="my game", journal="## 2026-01-01 x\nENTRY")
        msg = records.build_seed_message()
        assert "Last journal entry (docs_archive/my game_journal.md)" in msg
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
                    "arguments": f'{{"filepath": "{filepath}"}}',
                },
            }
        ],
    }


class TestCapturePaths:
    def test_captures_write_and_edit(self):
        records = DurableRecords()
        records.capture_paths(
            [
                _write_call("write_file", "docs_archive/games_journal.md"),
                _write_call("edit_file", "docs_archive/games_current_truth.md"),
            ]
        )
        assert records.journal_path == "docs_archive/games_journal.md"
        assert records.truth_path == "docs_archive/games_current_truth.md"

    def test_last_write_wins(self):
        records = DurableRecords()
        records.capture_paths(
            [
                _write_call("write_file", "docs_archive/first_journal.md"),
                _write_call("write_file", "docs_archive/second_journal.md"),
            ]
        )
        assert records.journal_path == "docs_archive/second_journal.md"

    def test_skipped_when_explicit_name_set(self):
        """The derived paths are authoritative - the wrap-up's writes cannot
        move the record."""
        records = DurableRecords()
        records.set_focus_name("explicit")
        records.capture_paths([_write_call("write_file", "docs_archive/other.md")])
        assert records.journal_path == "docs_archive/explicit_journal.md"
        assert records.truth_path == "docs_archive/explicit_current_truth.md"

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
        assert "docs_archive/my game_current_truth.md" in messages[0]["content"]

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
                _write_call("write_file", "docs_archive/g_journal.md"),
                _write_call("edit_file", "docs_archive/g_current_truth.md"),
            ]
        )
        messages = []

        assert records.seed(messages) is True
        records.read_record_file.assert_called_with("docs_archive/g_current_truth.md")
        records.read_last_journal_entry.assert_called_with("docs_archive/g_journal.md")
        # The name derived from the captured file names carries through.
        assert "[Loaded records for focus g]" == messages[1]["content"]


# ── restart intro ─────────────────────────────────────────────────────────


class TestRestartIntro:
    def test_names_both_files(self):
        records = DurableRecords()
        records.set_focus_name("auto_context")
        intro = records.restart_intro()
        assert "docs_archive/auto_context_current_truth.md" in intro
        assert "docs_archive/auto_context_journal.md" in intro

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
        assert "docs_archive/auto_context_current_truth.md" in out
        assert "docs_archive/auto_context_journal.md" in out
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
        assert restored.truth_path == "docs_archive/auto_context_current_truth.md"

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
        assert records.truth_path == "docs_archive/cli_focus_current_truth.md"

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
        derives the paths, the files are found relative to cwd."""
        monkeypatch.chdir(tmp_path)
        archive = tmp_path / "docs_archive"
        archive.mkdir()
        (archive / "my game_current_truth.md").write_text("TRUTH", encoding="utf-8")
        (archive / "my game_journal.md").write_text(
            "## 2026-01-01 10:00 - old\nold\n\n## 2026-02-02 11:00 - new\nnew\n",
            encoding="utf-8",
        )
        records = DurableRecords()
        records.set_focus_name("my game")
        messages = []

        assert records.seed(messages) is True
        content = messages[0]["content"]
        assert "TRUTH" in content
        assert "new" in content
        assert "old" not in content
