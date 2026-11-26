"""Topic index injection: build_truth_index + the _record_parts wiring.

The record is injected into message 0 of every new context, so the whole file
is paid for on every context. The index replaces it with one line per topic
plus the ticket section, and the model fetches a topic on demand.

The load-bearing property is that the index is only built when it is
LOSSLESS. A record that is unmigrated, half-migrated, or missing its ticket
section is injected unchanged - otherwise facts would vanish from the model's
view with no error anywhere.
"""

from unittest.mock import MagicMock, patch

from agent13.durable_records import (
    DurableRecords,
    TRUTH_INDEX_PREAMBLE,
    build_truth_index,
    build_truth_index_parts,
)
from agent13.truth_format import parse

# A migrated record: topics with facts, then the ticket section.
RECORD = """# demo - current truth (updated 2026-10-28 10:00)

## 1. Environment & access

### Host
- Runs on localhost   [observed 2026-10-28]
- Port 8111   [observed 2026-10-28]

### Credentials
- Key lives in ~/.env   [observed 2026-10-28]

## 3. Findings

### Command registry
- 48 commands, one catalogue   [agent13/commands.py]
- Handlers are (agent, args)   [agent13/commands.py]

## 6. Open tickets

- Fix the thing - FIRST SEEN 2026-10-28 10:00
- Another ticket - FIRST SEEN 2026-10-27 09:00

### Parked

- Parked thing - parked 2026-10-26: not now
"""


def _records(truth, journal="", index=True):
    """DurableRecords with the record readers stubbed (no filesystem)."""
    records = DurableRecords()
    records.set_focus_name("demo")
    records.read_record_file = MagicMock(return_value=truth)
    records.read_last_journal_entry = MagicMock(return_value=journal)
    # The for_user walk-back reads ENTRIES, not just the last one; a fixture's
    # journal is a single entry.
    records.read_journal_entries = MagicMock(
        return_value=[journal] if journal else []
    )
    records.truth_index = index
    return records


class TestBuildTruthIndex:
    def test_index_has_one_line_per_topic(self):
        index = build_truth_index(RECORD)
        topic_lines = [ln for ln in index.split("\n") if ln.startswith("L")]
        headings = [
            ln
            for ln in RECORD.split("\n")
            if ln.startswith("### ") and "Parked" not in ln
        ]
        assert len(topic_lines) == len(headings) == 3

    def test_loaded_line_numbers_are_marked(self):
        """The marker is applied by the renderer, keyed by line number.

        The caller no longer knows the line format: it passes the numbers and
        the renderer, which owns the format, does the marking.
        """
        host = next(t for t in parse(RECORD).topics if t.heading == "Host")
        topic_list, _ = build_truth_index_parts(RECORD, {host.line_no})
        marked = [ln for ln in topic_list.split("\n") if "[LOADED]" in ln]
        assert len(marked) == 1
        assert "Host" in marked[0]

    def test_a_duplicate_heading_marks_only_the_loaded_topic(self):
        """Two topics may share a heading; only the selected one is loaded.

        Marking by heading - the old way - marked both, so the index claimed
        facts were in context when only one topic's were.
        """
        dup = RECORD.replace("### Credentials", "### Host")
        hosts = [t for t in parse(dup).topics if t.heading == "Host"]
        assert len(hosts) == 2
        topic_list, _ = build_truth_index_parts(dup, {hosts[1].line_no})
        marked = [ln for ln in topic_list.split("\n") if "[LOADED]" in ln]
        assert len(marked) == 1
        assert f"L{hosts[1].line_no}" in marked[0]

    def test_no_loaded_set_marks_nothing(self):
        topic_list, _ = build_truth_index_parts(RECORD)
        assert "[LOADED]" not in topic_list

    def test_a_numbered_record_prints_its_addresses(self):
        """The address is the handle a handoff nominates (section.item)."""
        numbered = RECORD.replace("### Host", "### 1.1 Host").replace(
            "### Credentials", "### 1.2 Credentials"
        ).replace("### Command registry", "### 3.1 Command registry")
        index = build_truth_index(numbered)
        assert "1.1  L" in index and "Host" in index
        assert "3.1  L" in index and "Command registry" in index

    def test_an_unnumbered_record_prints_no_address(self):
        """Backward compatibility: an unmigrated record still renders.

        The model then reads by line number, which is what it did before
        addresses existed. Printing a bogus address would be worse than none.
        """
        index = build_truth_index(RECORD)
        assert "1.1" not in index
        assert "L" in index  # the line number is still there

    def test_index_keeps_the_queue_bullets_verbatim(self):
        index = build_truth_index(RECORD)
        assert "- Fix the thing - FIRST SEEN 2026-10-28 10:00" in index
        assert "- Another ticket - FIRST SEEN 2026-10-27 09:00" in index

    def test_parked_collapses_to_one_line_per_ticket(self):
        """Parked tickets are inspected only when unparked, so their bodies are
        dropped - but every one is still listed."""
        record = RECORD.replace(
            "- Parked thing - parked 2026-10-26: not now",
            "- Parked thing - parked 2026-10-26: not now. " + "body " * 200,
        )
        index = build_truth_index(record)
        parked = [ln for ln in index.split("\n") if "Parked thing" in ln]
        assert len(parked) == 1
        assert "not now." in parked[0]
        assert "body body" not in parked[0]
        assert parked[0].endswith("\u2026")

    def test_index_finds_tickets_by_heading_not_number(self):
        """A record numbered differently must still keep its work queue.

        Keying on the literal "## 6." would drop the whole ticket list from the
        injected context - silently, since nothing errors.
        """
        renumbered = RECORD.replace("## 6. Open tickets", "## 7. Open tickets")
        index = build_truth_index(renumbered)
        assert index is not None
        assert "Fix the thing" in index

    def test_the_topic_line_carries_no_fact(self):
        """A positional sample cannot say what a topic holds.

        The first fact is the oldest content and the last is often the tail of
        a sentence whose head is three bullets up. Heading + count reads as an
        unopened box, which is the truth.
        """
        index = build_truth_index(RECORD)
        assert "Runs on localhost" not in index
        assert "Port 8111" not in index
        assert "Key lives in ~/.env" not in index

    def test_the_topic_line_carries_its_line_count(self):
        # Host: heading + 2 facts + a trailing blank = 3 lines.
        index = build_truth_index(RECORD)
        host = next(ln for ln in index.split("\n") if "Host" in ln and ln.startswith("L"))
        assert "(3 lines)" in host

    def test_line_count_is_a_valid_read_window(self):
        """L and N together must be a window that covers the topic.

        This is the check the two numbers exist for: the model is told to call
        read_file(offset=L, limit=N), so a pair that does not land on the
        heading and end on a non-blank line is worse than no count at all.
        """
        import re

        index = build_truth_index(RECORD)
        lines = RECORD.split("\n")
        windows = re.findall(r"^L(\d+)\s+\((\d+) lines\)", index, re.MULTILINE)
        assert windows, "no topic lines in the index"
        for offset, count in windows:
            start, n = int(offset), int(count)
            window = lines[start - 1 : start - 1 + n]
            assert window[0].startswith("### "), f"window at L{start} misses the heading"
            assert window[-1].strip(), f"window at L{start} ends on a blank line"

    def test_empty_topic_does_not_steal_the_next_fact(self):
        """A topic with no facts must emit no fact line.

        The scan for a topic's first fact must stop at the next heading, or an
        empty topic is displayed with the following topic's first fact.
        """
        record = RECORD.replace(
            "### Credentials\n- Key lives in ~/.env   [observed 2026-10-28]",
            "### Credentials\n",
        )
        index = build_truth_index(record)
        lines = index.split("\n")
        at = next(
            i for i, ln in enumerate(lines) if "Credentials" in ln and ln.startswith("L")
        )
        # Next line is the next section heading, not a fact
        assert lines[at + 1].startswith("## ")

    def test_flat_record_falls_back(self):
        """No topics at all -> no index (an index of nothing is worse)."""
        flat = "# demo\n\n## 1. Environment\n\n- one fact\n- another fact\n"
        assert build_truth_index(flat) is None

    def test_half_migrated_record_falls_back(self):
        """One topic anywhere must not turn on indexing for the flat rest.

        This is the dangerous case: a partly migrated record would lose every
        fact in its still-flat sections, with no error.
        """
        half = (
            "# demo\n\n"
            "## 1. Environment\n\n"
            "- flat fact one\n"
            "- flat fact two\n\n"
            "## 3. Findings\n\n"
            "### A topic\n"
            "- a fact\n\n"
            "## 6. Open tickets\n\n"
            "- a ticket\n"
        )
        assert build_truth_index(half) is None

    def test_no_ticket_section_falls_back(self):
        """Without a ticket heading we cannot tell where the work queue is."""
        no_tickets = RECORD[: RECORD.index("## 6. Open tickets")]
        assert build_truth_index(no_tickets) is None

    def test_index_carries_no_facts_at_all(self):
        """The index is a table of contents: headings and counts, no content."""
        index = build_truth_index(RECORD)
        for fact in ("Runs on localhost", "Port 8111", "48 commands, one catalogue"):
            assert fact not in index
        # ... and the topics themselves are all present
        for topic in ("Host", "Credentials", "Command registry"):
            assert topic in index


class TestTruthIndexWiring:
    def test_index_off_by_default(self):
        assert DurableRecords().truth_index is False

    def test_seed_uses_the_index_when_enabled(self):
        records = _records(RECORD, index=True)
        messages = []
        assert records.seed(messages) is True
        content = messages[0]["content"]
        assert "TOPIC INDEX of" in content
        assert "the facts are NOT loaded" in content
        # No journal entry -> no query -> nothing scores -> the floor case.
        # The index is a table of contents: headings and counts, no facts.
        assert "Host" in content
        assert "Port 8111" not in content
        assert "Runs on localhost" not in content
        # The ticket section is the work queue - always carried
        assert "Fix the thing" in content

    def test_seed_carries_the_topic_the_next_action_names(self):
        """The whole point: the b. block names a topic, and its facts arrive.

        The query is the previous session's own statement of what to do next,
        so the topic it names is already in context - no fetch, no guessing.
        """
        journal = (
            "## 2026-10-28 12:00 - entry\n\n"
            "**b. NEXT SESSION, FIRST ACTION**\n"
            "- Check the `Port 8111` listener on the `Host`.\n\n"
            "**c. STATE AT HANDOFF**\n- nothing\n"
        )
        records = _records(RECORD, journal=journal, index=True)
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert "Port 8111" in content  # the Host topic, in full
        assert "[LOADED]" in content
        # A topic the query did not name stays unopened
        assert "48 commands, one catalogue" not in content

    def test_the_loaded_marker_survives_an_address_prefix(self):
        """A numbered topic line starts with "3.1", not "L".

        The marker used to be applied to lines starting with "L"; once the
        address is printed first that test silently stopped matching and every
        loaded topic lost its [LOADED] marker - the index would claim a topic
        was unopened while its facts sat right below.
        """
        numbered = RECORD.replace("### Host", "### 1.1 Host").replace(
            "### Credentials", "### 1.2 Credentials"
        ).replace("### Command registry", "### 3.1 Command registry")
        journal = (
            "## 2026-10-28 12:00 - entry\n\n"
            "**b. NEXT SESSION, FIRST ACTION**\n"
            "- Check the `Port 8111` listener on the `Host`.\n\n"
            "**c. STATE AT HANDOFF**\n- nothing\n"
        )
        records = _records(numbered, journal=journal, index=True)
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert "1.1  L" in content and "[LOADED] Host" in content
        assert "[LOADED] Credentials" not in content

    def test_the_seed_is_logged_with_both_controls(self):
        """One tps_truth_seed per seed - the only way to tune the numbers.

        Both controls ride in the event: the threshold guards the floor, the
        budget limits the load, and one number cannot be tuned from the other.
        """
        from unittest.mock import patch

        journal = (
            "**b. NEXT SESSION, FIRST ACTION**\n- Check `Port 8111`.\n\n**c. X**\n"
        )
        records = _records(RECORD, journal=journal, index=True)
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.seed([])

        seeds = [d for n, d in events if n == "truth_seed"]
        assert len(seeds) == 1
        assert seeds[0]["floor_hit"] is False
        assert seeds[0]["threshold"] == 1.0
        assert seeds[0]["budget"] == 40_000
        assert [t["heading"] for t in seeds[0]["topics"]] == ["Host"]
        assert seeds[0]["loaded_chars"] > 0

    def test_the_floor_case_is_logged(self):
        records = _records(RECORD, index=True)  # no journal -> no query
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.seed([])
        seeds = [d for n, d in events if n == "truth_seed"]
        assert seeds[0]["floor_hit"] is True
        assert seeds[0]["topics"] == []

    def test_seed_injects_the_whole_file_when_disabled(self):
        records = _records(RECORD, index=False)
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert "TOPIC INDEX" not in content
        assert "Port 8111" in content

    def test_no_preamble_when_the_index_falls_back(self):
        """The preamble says the facts are NOT loaded - never prepend it to the
        full file, or the model is told the opposite of what it is given."""
        flat = "# demo\n\n## 1. Environment\n\n- one fact\n- another fact\n"
        records = _records(flat, index=True)
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert TRUTH_INDEX_PREAMBLE.split("{")[0].strip() not in content
        assert "one fact" in content

    def test_seed_preamble_does_not_promise_the_full_file(self):
        """SEED_PREAMBLE must not say "in full" - it would contradict the index."""
        assert "in full" not in TRUTH_INDEX_PREAMBLE
        records = _records(RECORD, index=True)
        messages = []
        records.seed(messages)
        assert "in full" not in messages[0]["content"]


# A numbered record, so a handoff block can nominate by address.
NUMBERED_RECORD = RECORD.replace("### Host", "### 1.1 Host").replace(
    "### Credentials", "### 1.2 Credentials"
).replace("### Command registry", "### 3.1 Command registry")


def _handoff_journal(load, for_user=None):
    """A journal entry whose last line is a handoff block."""
    import json

    payload = {"load": load, "for_user": for_user or []}
    return (
        "## 2026-11-05 09:00 - entry\n\n"
        "**b. NEXT SESSION, FIRST ACTION**\n"
        "- Work the command registry.\n\n"
        "**c. STATE AT HANDOFF**\n- nothing\n\n"
        "```json\n" + json.dumps(payload) + "\n```\n"
    )


class TestHandoffSeed:
    def test_the_nominated_topic_loads(self):
        """The whole point: the wrap-up names an address, the facts arrive.

        The selector would have to guess from the b. block's vocabulary; the
        block states it outright, and it is the only source that knows what
        the next action IS.
        """
        records = _records(
            NUMBERED_RECORD, journal=_handoff_journal(["3.1"]), index=True
        )
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert "48 commands, one catalogue" in content
        assert "[LOADED] Command registry" in content
        # A topic the block did not name stays unopened
        assert "Runs on localhost" not in content

    def test_the_source_is_logged_as_json(self):
        records = _records(
            NUMBERED_RECORD, journal=_handoff_journal(["3.1"]), index=True
        )
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.seed([])
        seed = [d for n, d in events if n == "truth_seed"][0]
        assert seed["source"] == "json"
        assert seed["nominated"] == ["3.1"]
        assert seed["not_found"] == [] and seed["over_budget"] == []

    def test_an_address_that_does_not_exist_is_a_mismatch_not_a_crash(self):
        """A stale address must fall through, not load the wrong topic.

        This is why the handle is a number and not a heading: a number either
        resolves or it does not, and the miss is visible in the log.
        """
        records = _records(
            NUMBERED_RECORD, journal=_handoff_journal(["9.9"]), index=True
        )
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            messages = []
            records.seed(messages)
        seed = [d for n, d in events if n == "truth_seed"][0]
        assert seed["not_found"] == ["9.9"]
        assert seed["over_budget"] == []
        # Nothing resolved, so the selector took over rather than loading junk.
        assert seed["source"] in ("matched", "floor")

    def test_no_block_falls_back_to_the_selector(self):
        journal = (
            "## 2026-11-05 09:00 - entry\n\n"
            "**b. NEXT SESSION, FIRST ACTION**\n- Check `Port 8111`.\n\n"
            "**c. X**\n- nothing\n"
        )
        records = _records(RECORD, journal=journal, index=True)
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.seed([])
        seed = [d for n, d in events if n == "truth_seed"][0]
        assert seed["source"] == "matched"
        assert seed["nominated"] == []

    def test_for_user_is_lifted_above_the_index(self):
        """Burying it in a 9k-char journal entry defeats the point.

        The item is what the next session must act on, so it goes above the
        index - not left inside the entry at the bottom of the seed.
        """
        records = _records(
            NUMBERED_RECORD,
            journal=_handoff_journal(["3.1"], ["scope call: continue?"]),
            index=True,
        )
        messages = []
        records.seed(messages)
        content = messages[0]["content"]
        assert "FOR THE USER (unresolved):" in content
        assert "scope call: continue?" in content
        assert content.index("FOR THE USER") < content.index("TOPIC INDEX")

    def test_no_for_user_block_when_there_is_nothing(self):
        records = _records(
            NUMBERED_RECORD, journal=_handoff_journal(["3.1"]), index=True
        )
        messages = []
        records.seed(messages)
        assert "FOR THE USER" not in messages[0]["content"]

    def test_the_carried_count_is_logged(self):
        records = _records(
            NUMBERED_RECORD,
            journal=_handoff_journal(["3.1"], ["a", "b"]),
            index=True,
        )
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.seed([])
        seed = [d for n, d in events if n == "truth_seed"][0]
        assert seed["for_user_carried"] == 2


class TestCaptureHandoff:
    """The block is written last and persisted append-only.

    The final message is discarded at rebuild, so a block that is not appended
    to the journal is lost - and it is the only record of what the wrap-up
    judged the next session would need.
    """

    def _records(self, tmp_path):
        journal = tmp_path / "demo_journal.md"
        journal.write_text("## 2026-11-05 09:00 - entry\n\n**b. X**\n- do it\n")
        records = DurableRecords()
        records.journal_path = str(journal)
        return records, journal

    def test_appends_the_block_to_the_entry(self, tmp_path):
        records, journal = self._records(tmp_path)
        text = 'Done.\n\n```json\n{"load": ["3.2"], "for_user": ["ask"]}\n```\n'
        assert records.capture_handoff(text) is True
        content = journal.read_text()
        assert content.startswith("## 2026-11-05 09:00")  # nothing overwritten
        assert '"load": ["3.2"]' in content
        # Inside the last entry, so read_last_journal_entry sees it
        assert "```json" in records.read_last_journal_entry(str(journal))

    def test_no_block_writes_nothing(self, tmp_path):
        records, journal = self._records(tmp_path)
        before = journal.read_text()
        assert records.capture_handoff("just prose") is False
        assert journal.read_text() == before

    def test_the_block_is_re_rendered_not_copied(self, tmp_path):
        # A stray key cannot survive into the record: what the seed later
        # reads is exactly what was validated.
        records, journal = self._records(tmp_path)
        text = '```json\n{"load": ["3.2"], "next": "do it"}\n```'
        records.capture_handoff(text)
        assert "next" not in journal.read_text()

    def test_unresolved_for_user_reads_it_back(self, tmp_path):
        records, _ = self._records(tmp_path)
        records.capture_handoff('```json\n{"for_user": ["a", "b"]}\n```')
        assert records.unresolved_for_user() == ["a", "b"]

    def test_unresolved_for_user_is_empty_without_a_block(self, tmp_path):
        records, _ = self._records(tmp_path)
        assert records.unresolved_for_user() == []

    def test_the_handoff_json_event_is_logged(self, tmp_path):
        records, _ = self._records(tmp_path)
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.capture_handoff('```json\n{"load": ["3.2"]}\n```')
        event = [d for n, d in events if n == "handoff_json"][0]
        assert event["parsed"] is True
        assert event["load_count"] == 1

    def test_a_missing_block_is_logged_as_unparsed(self, tmp_path):
        """A model that stops emitting the block must be visible, not silent."""
        records, _ = self._records(tmp_path)
        events = []
        with (
            patch("agent13.debug_log.is_debug_enabled", return_value=True),
            patch(
                "agent13.debug_log.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            records.capture_handoff("no block here")
        event = [d for n, d in events if n == "handoff_json"][0]
        assert event["parsed"] is False
