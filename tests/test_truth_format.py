"""The record's READ half, tested directly.

``agent13/truth_format.py`` is a contract between a prompt that writes the
record and code that reads it. Until now it was covered only INDIRECTLY:
``test_truth_index`` drives ``build_truth_index``, which parses through
``parse``, and ``test_goal`` drives ``ticket_bullets`` through
``goal.ticket_counts``. Both exercise the happy path of a well-formed record,
so the failure modes that matter - a renamed ticket heading, a section with no
topics, a topic with no facts - were never asserted anywhere.

Each test below pins one property that fails SILENTLY in the product:

- a renamed ticket heading must not drop the work queue quietly (the index
  refuses to build instead);
- a topic with no facts must not borrow the next topic's first fact;
- ``ticket_bullets`` must keep the tail that ``ticket_order`` cuts, because
  BLOCKED / PARKED live in that tail;
- ``count_facts`` must count ticket bullets, because the cap is on topics.

The record fixtures are synthetic and small on purpose: each one documents the
single shape it exercises.
"""

from agent13.truth_format import (
    RecordShape,
    assign_addresses,
    count_facts,
    count_topics,
    h_block_text,
    is_ticket_section,
    parse,
    split_address,
    ticket_bullets,
    ticket_order,
)

# Line numbers are load-bearing (they are what read_file(offset=...) is given),
# so every assertion below refers to this numbering:
#
#    1  # demo - current truth
#    3  ## 1. Environment
#    5  - a flat fact under no topic        <- under NO topic
#    7  ### Host
#    8  - runs on localhost
#    9  - port 8111
#   11  ### Empty topic                     <- no facts of its own
#   13  ### Next topic
#   14  - its fact
#   16  ## 6. Open tickets
#   18  - Ticket A - FIRST SEEN 2026-10-30
#   19  - Ticket B - PARKED                 <- parked IN PLACE, still queued
#   21  ### Parked
#   23  - Ticket C - parked under a heading <- moved out of the queue
ANNOTATED = """# demo - current truth

## 1. Environment

- a flat fact under no topic

### Host
- runs on localhost
- port 8111

### Empty topic

### Next topic
- its fact

## 6. Open tickets

- Ticket A - FIRST SEEN 2026-10-30
- Ticket B - PARKED

### Parked

- Ticket C - parked under a heading
"""


class TestIsTicketSection:
    """The heading is matched by TEXT, never by its number."""

    def test_matches_the_standard_heading(self):
        assert is_ticket_section("## 6. Open tickets")

    def test_matches_a_different_section_number(self):
        # Keying on "## 6." would drop the whole work queue from a record that
        # numbered its sections differently - with no error anywhere.
        assert is_ticket_section("## 9. Open tickets")

    def test_is_case_insensitive(self):
        assert is_ticket_section("## 6. OPEN TICKETS")

    def test_matches_a_decorated_heading(self):
        assert is_ticket_section("## 6. Open tickets (work queue)")

    def test_rejects_a_sub_heading(self):
        # "### Open tickets" is not a section, so it must not end the topics.
        assert not is_ticket_section("### Open tickets")

    def test_rejects_a_renamed_heading(self):
        # The consequence is asserted in TestCountTopics: the section's topics
        # get counted and the index refuses to build.
        assert not is_ticket_section("## 6. Work queue")

    def test_rejects_the_words_in_a_bullet(self):
        assert not is_ticket_section("- see open tickets")


class TestParse:
    """One walk, and the shape everything else is derived from."""

    def test_topics_carry_their_file_line_number(self):
        # 1-based, because the index hands it to read_file(offset=...).
        assert [t.line_no for t in parse(ANNOTATED).topics] == [7, 11, 13]

    def test_topic_heading_has_the_marker_stripped(self):
        assert [t.heading for t in parse(ANNOTATED).topics] == [
            "Host",
            "Empty topic",
            "Next topic",
        ]

    def test_a_topic_carries_no_sampled_fact(self):
        # A positional sample cannot say what a topic holds - the first fact is
        # the oldest content and the last is often a sentence tail. The index
        # shows the heading and the count and reads as an unopened box.
        host = parse(ANNOTATED).topics[0]
        assert not hasattr(host, "last_fact")
        assert not hasattr(host, "first_fact")

    def test_line_count_runs_from_the_heading_to_the_last_non_blank_line(self):
        # Host: heading (L7) + two facts (L8, L9) + a trailing blank (L10) = 3.
        host = parse(ANNOTATED).topics[0]
        assert host.line_count == 3

    def test_line_count_is_one_for_an_empty_topic(self):
        # Heading only: the window is the heading, not the blank lines after it.
        empty = parse(ANNOTATED).topics[1]
        assert empty.line_count == 1

    def test_line_count_stops_at_the_next_section_heading(self):
        # A topic's span ends at the next "## " as well as the next "### ".
        text = "### Host\n- one\n- two\n\n## 2. Next\n- flat\n"
        assert parse(text).topics[0].line_count == 3

    def test_a_fact_under_no_topic_is_flat(self):
        # 1-based line numbers; this is the lossless guard's input.
        assert parse(ANNOTATED).flat_facts == [5]

    def test_a_section_heading_ends_the_current_topic(self):
        # "## 1. Environment" resets under_topic, so line 5 is flat rather than
        # belonging to whatever topic preceded it.
        shape = parse("## 1. Environment\n- flat\n### Host\n- under a topic\n")
        assert shape.flat_facts == [2]
        assert shape.topics[0].line_count == 2

    def test_ticket_line_is_the_zero_based_heading_index(self):
        # 0-based because it slices the line list; the topics are 1-based
        # because they are file offsets. Two conventions, both deliberate.
        assert parse(ANNOTATED).ticket_line == 15

    def test_parsing_stops_at_the_ticket_heading(self):
        # Everything from section 6 on is the work queue, not topics. Its
        # "### Parked" must not be counted as a topic.
        shape = parse(ANNOTATED)
        assert len(shape.topics) == 3
        assert "Parked" not in [t.heading for t in shape.topics]

    def test_ticket_bullets_are_not_flat_facts(self):
        # They sit under no topic, but they are the work queue - counting them
        # as flat facts would make every record fail the lossless guard.
        assert parse(ANNOTATED).flat_facts == [5]

    def test_no_ticket_section_yields_none(self):
        assert parse("## 1. Environment\n### Host\n- a fact\n").ticket_line is None

    def test_empty_text_yields_an_empty_shape(self):
        assert parse("") == RecordShape(topics=[], ticket_line=None, flat_facts=[])


class TestCountTopics:
    """The capped unit: topics BEFORE the ticket section."""

    def test_counts_only_topics_before_the_ticket_section(self):
        assert count_topics(ANNOTATED) == 3

    def test_a_renamed_ticket_heading_does_not_hide_its_topics(self):
        # The failure this pins: rename "## 6. Open tickets" and the section is
        # no longer recognised, so its "### " headings are counted as ordinary
        # topics. Nothing is silently dropped - and ticket_line is None, so
        # build_truth_index refuses to build and the record is injected
        # verbatim. Loud fallback, not a quiet loss.
        renamed = (
            "## 1. Environment\n### Host\n- a fact\n\n"
            "## 6. Work queue\n\n### Still open\n- Ticket A - FIRST SEEN 2026-10-30\n"
        )
        assert count_topics(renamed) == 2
        assert parse(renamed).ticket_line is None

    def test_empty_text_counts_zero(self):
        assert count_topics("") == 0


class TestCountFacts:
    """Facts are uncapped, so this is a report, not a limit."""

    def test_counts_every_bullet_including_the_ticket_section(self):
        # 5, 8, 9, 14 (topics) + 18, 19 (tickets) + 23 (under "### Parked").
        assert count_facts(ANNOTATED) == 7

    def test_counts_a_ticket_parked_in_place(self):
        # The bullet stays in the queue, so it is still a fact.
        text = "## 6. Open tickets\n- A - PARKED\n- B - FIRST SEEN 2026-10-30\n"
        assert count_facts(text) == 2

    def test_ignores_indented_and_star_bullets(self):
        assert count_facts("- one\n  - indented\n* star\n- two\n") == 2

    def test_empty_text_counts_zero(self):
        assert count_facts("") == 0


class TestTicketBullets:
    """The work queue as raw lines - the tail survives, which is the point."""

    def test_returns_the_full_bullet_including_its_tail(self):
        bullets = ticket_bullets(ANNOTATED)
        assert bullets == [
            "- Ticket A - FIRST SEEN 2026-10-30",
            "- Ticket B - PARKED",
        ]

    def test_keeps_the_tail_that_ticket_order_cuts(self):
        # ticket_order cuts at the first " - " because the tail is rewritten
        # every entry and the title is stable. That cut is right for RANKING
        # and wrong for reading a ticket's STATE, which is why the two walks
        # exist separately.
        assert "PARKED" in ticket_bullets(ANNOTATED)[1]
        assert "PARKED" not in ticket_order(ANNOTATED)[1][0]

    def test_stops_at_a_sub_heading(self):
        # A ticket moved under "### Parked" leaves the work queue.
        assert len(ticket_bullets(ANNOTATED)) == 2
        assert not any("Ticket C" in b for b in ticket_bullets(ANNOTATED))

    def test_stops_at_the_next_section(self):
        text = (
            "## 6. Open tickets\n- A - FIRST SEEN 2026-10-30\n\n"
            "## 7. Notes\n- not a ticket\n"
        )
        assert ticket_bullets(text) == ["- A - FIRST SEEN 2026-10-30"]

    def test_agrees_with_ticket_order_on_a_flat_queue(self):
        # Two separate walks of the same section. They must not drift: the
        # goal window reads STATES from one and the rank detector reads ORDER
        # from the other, and a disagreement would silently drop a ticket from
        # one of the two views.
        text = (
            "## 6. Open tickets\n"
            "- **(RANK 1)** **A ticket** - FIRST SEEN 2026-10-30\n"
            "- **(RANK 2)** **B ticket** - FIRST SEEN 2026-10-30\n"
            "- C ticket - PARKED\n"
        )
        assert len(ticket_bullets(text)) == len(ticket_order(text)) == 3

    def test_no_ticket_section_yields_an_empty_list(self):
        assert ticket_bullets("## 1. Environment\n### Host\n- a fact\n") == []

    def test_empty_text_yields_an_empty_list(self):
        assert ticket_bullets("") == []


# --- topic addresses (section.item) -----------------------------------------
#
# The address is the handle a handoff nominates, so two properties are
# load-bearing and both fail SILENTLY:
#
# * an address, once assigned, must never change - the next session may read
#   the nomination hours later, with the record rewritten in between;
# * the migration must be idempotent - a re-run that renumbered would re-point
#   every nomination made before it.

NUMBERED = """# demo - current truth

## 1. Environment

### 1.1 Host
- runs on localhost

### 1.2 Empty topic

## 3. Findings

### 3.1 The object state machine
- a fact
"""


class TestSplitAddress:
    def test_splits_a_numbered_heading(self):
        assert split_address("3.2 THE OBJECT STATE MACHINE") == (
            "3.2",
            "THE OBJECT STATE MACHINE",
        )

    def test_an_unnumbered_heading_is_all_label(self):
        assert split_address("Host") == ("", "Host")

    def test_a_decimal_that_is_not_an_address_stays_a_label(self):
        # "1.5x" has no whitespace after the digits, so it is a label - not an
        # address of section 1, item 5. Renumbering it would corrupt the topic.
        assert split_address("1.5x speed") == ("", "1.5x speed")

    def test_a_date_heading_stays_a_label(self):
        assert split_address("2026-11-04 ride log") == ("", "2026-11-04 ride log")


class TestParseAddress:
    def test_parse_splits_the_address_from_the_label(self):
        topics = parse(NUMBERED).topics
        assert [t.number for t in topics] == ["1.1", "1.2", "3.1"]
        assert topics[0].heading == "Host"

    def test_an_unnumbered_record_has_empty_addresses(self):
        # The index must still render an unmigrated record: the model then
        # reads by line number, which is what it did before addresses existed.
        topics = parse(ANNOTATED).topics
        assert [t.number for t in topics] == ["", "", ""]
        assert topics[0].heading == "Host"


class TestAssignAddresses:
    def test_assigns_per_section_starting_at_one(self):
        text = "## 1. Env\n### A\n- x\n### B\n- y\n\n## 3. Findings\n### C\n- z\n"
        _, assigned = assign_addresses(text)
        assert assigned == [("1.1", "A"), ("1.2", "B"), ("3.1", "C")]

    def test_is_idempotent(self):
        once, assigned = assign_addresses(NUMBERED)
        assert assigned == []  # already numbered: nothing to do
        twice, again = assign_addresses(once)
        assert again == [] and twice == once

    def test_never_renumbers_an_existing_address(self):
        # The gap at 1.2 is deliberate: a deleted topic leaves its number
        # behind, and the next new topic takes the NEXT free number rather
        # than sliding into the hole.
        text = "## 1. Env\n### 1.1 A\n- x\n### 1.5 B\n- y\n### C\n- z\n"
        out, assigned = assign_addresses(text)
        assert assigned == [("1.6", "C")]
        assert "### 1.1 A" in out and "### 1.5 B" in out

    def test_leaves_the_ticket_sections_sub_headings_alone(self):
        # "### Parked" sits under "## 6. Open tickets". It is not a topic, and
        # numbering it would put it in the address space the handoff loads by.
        text = "## 1. Env\n### A\n- x\n\n## 6. Open tickets\n- T - FIRST SEEN\n\n### Parked\n- P\n"
        out, assigned = assign_addresses(text)
        assert assigned == [("1.1", "A")]
        assert "### Parked" in out

    def test_a_topic_outside_a_numbered_section_gets_no_address(self):
        text = "## Notes\n### A\n- x\n"
        out, assigned = assign_addresses(text)
        assert assigned == [] and out == text

    def test_assigning_does_not_change_any_other_line(self):
        text = "## 1. Env\n\n### A\n- x\n- y\n\n## 2. Conv\n### B\n- z\n"
        out, _ = assign_addresses(text)
        assert out.replace("### 1.1 A", "### A").replace("### 2.1 B", "### B") == text


# ── the h. section ──────────────────────────────────────────────────────────
#
# The re-rank detector reads h. through this one walker, so the "where does the
# section end" rule cannot drift between the boolean and the text.


class TestHBlock:
    ENTRY = (
        "## 2026-11-07 23:25 - age gate: the octree message is a teardown race\n"
        "\n"
        "**a. GOAL STATUS**\n"
        "- rank 1: BLOCKED\n"
        "\n"
        "**c. STATE AT HANDOFF**\n"
        "- nothing running\n"
    )

    def test_no_h_block_is_empty(self):
        assert h_block_text(self.ENTRY) == ""

    def test_an_empty_entry_is_empty(self):
        assert h_block_text("") == ""

    def test_h_block_text_includes_the_heading_line_delta(self):
        entry = (
            "## 2026-10-30 23:17 - x\n\n"
            "**h. TICKET CHANGES** - RE-RANKED: A 2 -> 1\n"
            "- none else\n\n"
            "**i. NEXT**\n- y\n"
        )
        block = h_block_text(entry)
        assert block.startswith("**h. TICKET CHANGES** - RE-RANKED")
        assert "- none else" in block
        assert "NEXT" not in block
