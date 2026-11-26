"""A wrap-up re-rank must be reported, and a silent one must be detectable.

Two halves, and the second is the one that was got wrong first:

- the PROMPT makes a re-rank a ticket change that must be cited
  (``agent13/prompts.py``, pinned here only through ``RERANK_TOKEN``);
- the DETECTOR logs a rank change the wrap-up did not report (this file).

The detector is log-only. It never nudges, rewrites or adds a turn - a
post-wrap-up agent that rewrote the record deleted six verified facts on
2026-10-27.

History, because the first version of this detector was INVERTED and the
fixtures below are what caught it:

- Rev 1 read POSITION only, so a correction that expressed the new ranking with
  ``(RANK n)`` markers while leaving bullet order untouched produced no event at
  all. The record that FIXED the bug was invisible.
- Rev 1 tested ``reported`` over the whole turn, so a failure whose ``e.`` says
  "Re-ranked the tickets" while its ``h.`` says "- none" read as reported. The
  failure was MISFILING, not silence.

Both are regression tests against the synthetic records in ``tests/fixtures/``
(``rank_before`` -> ``rank_after_bug`` -> ``rank_after_fix``). The fixtures are
synthetic on purpose: they carry no real project's content, and each one
documents the single property it exercises. The numbers they produce are the
numbers a real record produced - 8 moves for the inversion, 2 for the marker-only
correction - so the tests still assert measured behaviour rather than a guess.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.prompts import DEFAULT_REPORT_AND_COMPACT_PROMPT
from agent13.truth_format import (
    RERANK_TOKEN,
    _ticket_id,
    encoding_of,
    h_block_reports_rerank,
    rank_moves,
    ticket_order,
)

FIXTURES = Path(__file__).parent / "fixtures"

# A minimal record: three ranked tickets and one parked one. The parked bullet
# is under "### Parked" so it is excluded from the order.
TRUTH = """# probe - current truth

## 1. Environment
### Repo
- a fact

## 6. Open tickets
- A ticket - FIRST SEEN 2026-10-30 10:00
- B ticket - FIRST SEEN 2026-10-30 10:00
- C ticket - FIRST SEEN 2026-10-30 10:00
### Parked
- an old ticket - parked 2026-10-30: reason
"""

SWAPPED = TRUTH.replace(
    "- A ticket - FIRST SEEN 2026-10-30 10:00\n"
    "- B ticket - FIRST SEEN 2026-10-30 10:00",
    "- B ticket - FIRST SEEN 2026-10-30 10:00\n"
    "- A ticket - FIRST SEEN 2026-10-30 10:00",
)


def record(name):
    """One synthetic record: "before", "after_bug", or "after_fix"."""
    return (FIXTURES / f"rank_{name}.md").read_text(encoding="utf-8")


def journal(name):
    """One synthetic journal entry: "silent", "reported", or "parked"."""
    return (FIXTURES / f"journal_{name}.md").read_text(encoding="utf-8")


def make_agent(tmp_path, truth_text=TRUTH, journal_text=""):
    """An agent with the wrap-up armed, records on disk, snapshots mocked."""
    docs = tmp_path / "docs_archive"
    docs.mkdir(parents=True, exist_ok=True)
    truth_path = docs / "probe_current_truth.md"
    journal_path = docs / "probe_journal.md"
    truth_path.write_text(truth_text, encoding="utf-8")
    journal_path.write_text(journal_text, encoding="utf-8")

    agent = Agent(client=MagicMock(), model="test-model", auto_context_threshold=0)
    agent._save_auto_context_snapshot = MagicMock()
    agent.durable_records.truth_path = str(truth_path)
    agent.durable_records.journal_path = str(journal_path)
    agent._report_and_compact_triggered = True
    agent._running = True
    return agent, truth_path, journal_path


def stub_turn(agent, truth_path, journal_path, new_truth=None, entry=None):
    """Replace the wrap-up turn with a recorder that rewrites the records."""

    async def fake(_polite_acquired=False):
        if new_truth is not None:
            truth_path.write_text(new_truth, encoding="utf-8")
        if entry is not None:
            with open(journal_path, "a", encoding="utf-8") as fh:
                fh.write(entry)

    agent._llm_turn = fake


@contextmanager
def captured_events():
    """Collect log_tps_event calls; the check is silent unless debug is on."""
    events = []
    with (
        patch("agent13.core.is_debug_enabled", return_value=True),
        patch(
            "agent13.core.log_tps_event",
            side_effect=lambda name, data: events.append((name, data)),
        ),
    ):
        yield events


def rank_events(events):
    return [data for name, data in events if name == "wrapup_rank_changed"]


class TestTicketOrder:
    """Reading section 6: what is ranked, and what a ticket's identity is."""

    def test_stops_at_the_parked_heading(self):
        """"### Parked" is not ranked - a parked ticket in the order would read
        as a re-rank on every park."""
        order = ticket_order(TRUTH)
        assert [identity for identity, _, _ in order] == [
            "A ticket",
            "B ticket",
            "C ticket",
        ]

    def test_ignores_non_bullet_lines(self):
        """Prose and headings inside the section are not tickets."""
        text = TRUTH.replace(
            "- B ticket - FIRST SEEN 2026-10-30 10:00",
            "Some prose about B.\n- B ticket - FIRST SEEN 2026-10-30 10:00",
        )
        assert len(ticket_order(text)) == 3

    def test_no_ticket_section_returns_empty(self):
        assert ticket_order("# probe\n\n## 1. Environment\n- a fact\n") == []

    def test_positions_are_one_based(self):
        assert [pos for _, _, pos in ticket_order(TRUTH)] == [1, 2, 3]

    def test_id_strips_bold_then_the_marker(self):
        """Real bullets are ``- **(RANK 1)** **title**``: the marker sits inside
        the bold run, so stripping the marker before ``**`` matches nothing."""
        identity, marker = _ticket_id("- **(RANK 1)** **Make the export stream** - x")
        assert identity == "Make the export stream"
        assert marker == 1

    def test_id_cuts_the_tail_after_the_separator(self):
        """The tail is rewritten every entry; the title is stable."""
        identity, _ = _ticket_id(
            "- **Make the first-run setup reliable** - the gate is `boot`"
        )
        assert identity == "Make the first-run setup reliable"

    def test_id_without_a_marker(self):
        identity, marker = _ticket_id("- Make the docs build cleanly - x")
        assert identity == "Make the docs build cleanly"
        assert marker is None


class TestRankMoves:
    """Effective rank = the marker when present, else position among common."""

    def test_a_swap_is_two_moves(self):
        moves = rank_moves(ticket_order(TRUTH), ticket_order(SWAPPED))
        assert sorted(moves) == [("A ticket", 1, 2), ("B ticket", 2, 1)]

    def test_an_insert_at_the_top_is_not_a_move(self):
        """Ranks are computed among COMMON tickets, so a new top ticket does
        not register as a move for every other ticket."""
        text = TRUTH.replace(
            "- A ticket - FIRST SEEN 2026-10-30 10:00",
            "- NEW ticket - FIRST SEEN 2026-10-30 11:00\n"
            "- A ticket - FIRST SEEN 2026-10-30 10:00",
        )
        assert rank_moves(ticket_order(TRUTH), ticket_order(text)) == []

    def test_a_close_is_not_a_move(self):
        text = TRUTH.replace("- B ticket - FIRST SEEN 2026-10-30 10:00\n", "")
        assert rank_moves(ticket_order(TRUTH), ticket_order(text)) == []

    def test_a_three_way_reorder(self):
        text = TRUTH.replace(
            "- A ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- B ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- C ticket - FIRST SEEN 2026-10-30 10:00",
            "- C ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- A ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- B ticket - FIRST SEEN 2026-10-30 10:00",
        )
        assert sorted(rank_moves(ticket_order(TRUTH), ticket_order(text))) == [
            ("A ticket", 1, 2),
            ("B ticket", 2, 3),
            ("C ticket", 3, 1),
        ]

    def test_identical_is_empty(self):
        assert rank_moves(ticket_order(TRUTH), ticket_order(TRUTH)) == []

    def test_a_marker_only_change_is_a_move(self):
        """The 17:55 shape: bullet order untouched, ranking expressed with
        markers. A position-only reader sees nothing here - that was Rev 1."""
        text = TRUTH.replace(
            "- A ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- B ticket - FIRST SEEN 2026-10-30 10:00",
            "- (RANK 2) A ticket - FIRST SEEN 2026-10-30 10:00\n"
            "- (RANK 1) B ticket - FIRST SEEN 2026-10-30 10:00",
        )
        assert sorted(rank_moves(ticket_order(TRUTH), ticket_order(text))) == [
            ("A ticket", 1, 2),
            ("B ticket", 2, 1),
        ]

    def test_encoding_is_reported(self):
        assert encoding_of([]) == "none"
        assert encoding_of(ticket_order(TRUTH)) == "position"
        marked = TRUTH.replace("- A ticket", "- (RANK 1) A ticket")
        assert encoding_of(ticket_order(marked)) == "mixed"


class TestRecordFixtures:
    """The failure and the fix, on records built to reproduce them.

    These are the numbers a real record produced; they are assertions, not
    predictions. Rev 1's version of this class asserted outcomes nobody ran,
    and both were wrong.

    The fixtures are synthetic - ``rank_before`` -> ``rank_after_bug`` ->
    ``rank_after_fix`` - because a test must not carry another project's
    content, and because a fixture built for one property documents that
    property. The structure is what matters: ten tickets, an inversion that
    moves eight of them, and a correction that moves two while leaving the
    bullet order untouched.
    """

    def test_the_inverted_priority(self):
        """The wrap-up inverted an agreed priority: the search index to rank 1,
        the ingest pipeline demoted to rank 6."""
        moves = rank_moves(
            ticket_order(record("before")), ticket_order(record("after_bug"))
        )
        assert len(moves) == 8
        by_ticket = {t: (old, new) for t, old, new in moves}
        assert by_ticket["Make the search index rebuild incrementally"] == (2, 1)
        ingest = next(t for t in by_ticket if t.startswith("Make the ingest pipeline"))
        assert by_ticket[ingest] == (1, 6)

    def test_the_marker_only_correction(self):
        """The correction expressed the new ranking with markers and left the
        bullet order alone - invisible to a position-only reader."""
        moves = rank_moves(
            ticket_order(record("after_bug")), ticket_order(record("after_fix"))
        )
        assert len(moves) == 2
        by_ticket = {t: (old, new) for t, old, new in moves}
        assert by_ticket["Make the session store survive a restart"] == (4, 2)
        assert by_ticket["Make the first-run setup reliable"] == (5, 1)

    def test_the_correction_leaves_the_bullet_order_alone(self):
        """The premise of the test above: if the correction had reordered the
        bullets, a position-only reader would have seen it and the marker rule
        would be untested."""
        before = [t for t, _, _ in ticket_order(record("after_bug"))]
        after = [t for t, _, _ in ticket_order(record("after_fix"))]
        assert before == after

    @pytest.mark.parametrize("name", ["before", "after_bug", "after_fix"])
    def test_the_parser_is_never_silently_empty(self, name):
        """A detector that parses nothing never fires and looks healthy."""
        assert len(ticket_order(record(name))) == 10

    @pytest.mark.parametrize("name", ["after_bug", "after_fix"])
    def test_the_fixtures_are_mixed_encoding(self, name):
        """Both post-incident records carry a partial marker set, which is the
        defect the prompt's "Rank IS position" rule removes. Reported, never
        resolved silently."""
        assert encoding_of(ticket_order(record(name))) == "mixed"


class TestReportedScope:
    """``reported`` is scoped to the h. block, and matched at line start."""

    def test_a_re_rank_in_e_but_not_h_is_not_reported(self):
        """``e.`` says "Re-ranked the tickets"; ``h.`` says "- none". Rev 1 read
        the whole entry and called this compliant."""
        assert h_block_reports_rerank(journal("silent")) is False

    def test_a_re_rank_in_h_is_reported(self):
        assert h_block_reports_rerank(journal("reported")) is True

    def test_a_parked_note_is_not_reported(self):
        """``e.`` mentions a re-rank; ``h.`` records a park. The token is not in
        the block, so nothing was reported."""
        assert h_block_reports_rerank(journal("parked")) is False

    def test_a_negation_is_not_reported(self):
        """A substring test would call this reported - the same inversion in
        the other direction."""
        entry = (
            "## 2026-10-30 20:00 - x\n\n"
            "**h. TICKET CHANGES**\n"
            "- none; no ticket was re-ranked this entry.\n"
        )
        assert h_block_reports_rerank(entry) is False

    def test_a_bold_marker_is_reported(self):
        entry = (
            "## 2026-10-30 20:00 - x\n\n"
            "**h. TICKET CHANGES**\n"
            "- **RE-RANKED**: A -> 2, B -> 1.\n"
        )
        assert h_block_reports_rerank(entry) is True

    def test_the_delta_on_the_heading_line_is_reported(self):
        """Observed live 2026-10-30 23:17: the model put the delta on the SAME
        line as the bold heading, and a parser that skipped the rest of that
        line reported a compliant wrap-up as unreported."""
        entry = (
            "## 2026-10-30 23:17 - x\n\n"
            "**h. TICKET CHANGES** - RE-RANKED: Make A work rank 2 -> rank 1; "
            "Make B work rank 1 -> rank 2. Reason: user agreement this session.\n"
        )
        assert h_block_reports_rerank(entry) is True

    def test_an_empty_heading_line_is_not_reported(self):
        entry = "## 2026-10-30 20:00 - x\n\n**h. TICKET CHANGES**\n"
        assert h_block_reports_rerank(entry) is False

    def test_no_h_block_is_not_reported(self):
        """A wrap-up that writes no h. block reports nothing."""
        entry = "## 2026-10-30 20:00 - x\n\n**e. DECISIONS**\n- RE-RANKED: A -> 2.\n"
        assert h_block_reports_rerank(entry) is False


class TestCoreHook:
    """The post-wrap-up check: log and stop."""

    @pytest.mark.asyncio
    async def test_unreported_rerank_logs(self, tmp_path):
        agent, truth_path, journal_path = make_agent(tmp_path)
        stub_turn(
            agent,
            truth_path,
            journal_path,
            new_truth=SWAPPED,
            entry="## 2026-10-30 20:00 - x\n\n**h. TICKET CHANGES**\n- none\n",
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        logged = rank_events(events)
        assert len(logged) == 1
        assert logged[0]["reported"] is False
        assert {m["ticket"] for m in logged[0]["moved"]} == {"A ticket", "B ticket"}
        assert logged[0]["before_n"] == 3
        assert logged[0]["after_n"] == 3

    @pytest.mark.asyncio
    async def test_reported_rerank_logs(self, tmp_path):
        agent, truth_path, journal_path = make_agent(tmp_path)
        stub_turn(
            agent,
            truth_path,
            journal_path,
            new_truth=SWAPPED,
            entry=(
                "## 2026-10-30 20:00 - x\n\n"
                "**h. TICKET CHANGES**\n"
                "- RE-RANKED: B -> 1, A -> 2.\n"
            ),
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        logged = rank_events(events)
        assert len(logged) == 1
        assert logged[0]["reported"] is True

    @pytest.mark.asyncio
    async def test_e_only_mention_is_not_reported(self, tmp_path):
        """The 16:40 failure end to end: the re-rank is filed as a rationale."""
        agent, truth_path, journal_path = make_agent(tmp_path)
        stub_turn(
            agent,
            truth_path,
            journal_path,
            new_truth=SWAPPED,
            entry=(
                "## 2026-10-30 20:00 - x\n\n"
                "**e. DECISIONS + RATIONALE**\n"
                "- Re-ranked the tickets: B to rank 1.\n\n"
                "**h. TICKET CHANGES**\n"
                "- none\n"
            ),
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        logged = rank_events(events)
        assert len(logged) == 1
        assert logged[0]["reported"] is False

    @pytest.mark.asyncio
    async def test_no_change_logs_nothing(self, tmp_path):
        agent, truth_path, journal_path = make_agent(tmp_path)
        stub_turn(agent, truth_path, journal_path, new_truth=TRUTH)
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert rank_events(events) == []

    @pytest.mark.asyncio
    async def test_no_truth_path_logs_nothing(self, tmp_path):
        """A focus-less session's first handoff has no before_order - a silent
        no-op, documented on the method."""
        agent, truth_path, journal_path = make_agent(tmp_path)
        agent.durable_records.truth_path = ""
        stub_turn(agent, truth_path, journal_path, new_truth=SWAPPED)
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert rank_events(events) == []


class TestMisattribution:
    """A wrap-up that cites the harness's own step as user direction.

    The live failure (edmac 2026-11-06 06:20): ``h.`` recorded
    "RE-RANKED ... by user direction at 06:09" quoting the continuation
    message, while the history file has no user message before 18:23 that day.
    The prompt fix is the deliverable; this detector says whether it held.
    """

    STEP = "Next step (your own proposal, not a user instruction): Build the launcher"

    def _agent(self, tmp_path, entry):
        agent, truth_path, journal_path = make_agent(tmp_path)
        stub_turn(agent, truth_path, journal_path, new_truth=TRUTH, entry=entry)
        agent._goal_last_item = self.STEP
        return agent

    def _logged(self, events):
        return [d for n, d in events if n == "wrapup_rerank_misattributed"]

    @pytest.mark.asyncio
    async def test_a_step_quoted_as_user_direction_logs(self, tmp_path):
        agent = self._agent(
            tmp_path,
            "## 2026-11-06 06:20 - x\n\n"
            "**h. TICKET CHANGES**\n"
            "- RE-RANKED: Make the game stand alone 5 -> 1, by user direction "
            "at 2026-11-06 06:09 (\"Next step (your own proposal, not a user "
            "instruction): Build the launcher\").\n",
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        logged = self._logged(events)
        assert len(logged) == 1
        assert logged[0]["step_prefix"].startswith("Next step (your own proposal")

    @pytest.mark.asyncio
    async def test_an_honest_self_attribution_does_not_log(self, tmp_path):
        """The legal form the prompt now offers."""
        agent = self._agent(
            tmp_path,
            "## 2026-11-06 06:20 - x\n\n"
            "**h. TICKET CHANGES**\n"
            "- RE-RANKED: Make the game stand alone 5 -> 1, by my own judgement: "
            "the launcher is the only path to the open galaxy.\n",
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert self._logged(events) == []

    @pytest.mark.asyncio
    async def test_a_user_citation_that_does_not_quote_the_step_does_not_log(
        self, tmp_path
    ):
        agent = self._agent(
            tmp_path,
            "## 2026-11-06 06:20 - x\n\n"
            "**h. TICKET CHANGES**\n"
            "- RE-RANKED: A 3 -> 1, by user direction at 18:23 "
            "(\"make the launcher first\").\n",
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert self._logged(events) == []

    @pytest.mark.asyncio
    async def test_no_injected_step_logs_nothing(self, tmp_path):
        """No goal window ran, so there is nothing to misattribute."""
        agent = self._agent(
            tmp_path,
            "## 2026-11-06 06:20 - x\n\n"
            "**h. TICKET CHANGES**\n- RE-RANKED: A 3 -> 1, by user direction.\n",
        )
        agent._goal_last_item = ""
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert self._logged(events) == []

    @pytest.mark.asyncio
    async def test_a_quote_outside_the_h_block_does_not_log(self, tmp_path):
        """Scoped to h.: the step quoted in e. is not a rank attribution."""
        agent = self._agent(
            tmp_path,
            "## 2026-11-06 06:20 - x\n\n"
            "**e. DECISIONS**\n"
            "- by user direction we build the launcher: Next step (your own "
            "proposal, not a user instruction): Build the launcher\n\n"
            "**h. TICKET CHANGES**\n- none\n",
        )
        with captured_events() as events:
            await agent._run_report_and_compact_turn()

        assert self._logged(events) == []


class TestTokenIsAnInterface:
    """``RE-RANKED`` is a contract between the prompt and the detector.

    Same pattern as ``TICKET_HEADING``: the prompt line is composed from the
    constant, so a rename cannot desync the two. The PROSE is deliberately not
    pinned - its failure mode is model non-compliance, which no test can check,
    and a prose pin freezes wording so every improvement becomes a test edit.
    """

    def test_the_prompt_carries_the_token(self):
        assert RERANK_TOKEN in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_the_prompt_builds_the_rerank_line_from_it(self):
        assert f"tickets {RERANK_TOKEN} (old rank -> new rank" in (
            DEFAULT_REPORT_AND_COMPACT_PROMPT
        )
