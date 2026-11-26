"""The selector: which topics the seed carries in full.

Pure functions over two strings, so every case here is a plain assertion - no
Agent, no I/O, no model.

The two numbers do different jobs and the tests keep them apart:

- the THRESHOLD only guards the floor (a real record has ~18 topics above 1.0,
  so it is not what limits the load);
- the BUDGET is the primary control - the fill stops at 40,000 chars.

The weighting is the whole correction: an unweighted match on a real rank-1
ticket hit 24 of 39 topics (71% of the record), which is no selection at all.
"""

from agent13.truth_select import (
    DEFAULT_BUDGET,
    DEFAULT_THRESHOLD,
    DF_CEILING,
    b_block,
    build_query,
    query_keys,
    select_by_address,
    select_topics,
)

# Two topics that share one rare key, one that shares only a common word, and a
# ticket section. `0xdeadbeef` appears in ONE topic, so it scores 1.0 there.
RECORD = """# demo - current truth

## 1. Environment

### Alpha
- the `0xdeadbeef` handler lives here
- ALPHA THING is true

### Beta
- BETA THING is separate
- nothing else

### Gamma
- GAMMA THING only

## 6. Open tickets

- Fix the `0xdeadbeef` path - FIRST SEEN 2026-11-04 10:00
"""


class TestQueryKeys:
    def test_backticked_hex_and_caps_are_keys(self):
        keys = query_keys("run `0xdeadbeef` then check ALPHA THING")
        assert "0xdeadbeef" in keys
        assert "ALPHA" in keys
        assert "THING" in keys

    def test_a_backticked_prose_span_is_not_a_key(self):
        """Real prose backticks yield `' -> '` (df 26) and `' is the '` (df 25).

        Those are not identifiers, and they change which topics the budget
        loads - measured on edmac's record: 2 topics with them, 3 without.
        """
        keys = query_keys("the ` -> ` arrow and ` is the ` copula")
        assert " -> " not in keys
        assert " is the " not in keys

    def test_a_backticked_identifier_with_a_space_survives(self):
        """`Port 8111` is a real identifier - the prose rule must not eat it.

        The first cut of this rule rejected any span containing whitespace,
        which silently dropped every two-word identifier in the record.
        """
        assert "Port 8111" in query_keys("check `Port 8111` now")
        assert "~/.../netLog*.log" in query_keys("see `~/.../netLog*.log`")

    def test_short_tokens_are_dropped(self):
        assert query_keys("`ab` ABC") == set()

    def test_an_empty_query_yields_no_keys(self):
        assert query_keys("") == set()


class TestBBlock:
    def test_extracts_the_b_block(self):
        entry = "## e\n\n**b. NEXT SESSION, FIRST ACTION**\n- do the thing\n\n**c. X**\n- y\n"
        assert "do the thing" in b_block(entry)
        assert "- y" not in b_block(entry)

    def test_no_b_block_yields_empty(self):
        assert b_block("## e\n\n**c. X**\n- y\n") == ""

    def test_empty_entry_yields_empty(self):
        assert b_block("") == ""

    def test_query_falls_back_to_the_entry_head(self):
        entry = "## e\n\n" + "x" * 2000
        assert len(build_query(entry, RECORD)) < 1200

    def test_query_unions_the_rank_one_ticket(self):
        entry = "**b. NEXT SESSION, FIRST ACTION**\n- ALPHA\n\n**c. X**\n"
        assert "Fix the" in build_query(entry, RECORD)


class TestSelectTopics:
    def test_the_rare_key_wins(self):
        selected = select_topics(RECORD, "`0xdeadbeef`")
        assert [s.heading for s in selected] == ["Alpha"]
        assert selected[0].score == 1.0

    def test_a_common_key_alone_cannot_clear_the_threshold(self):
        # THING is in all three topics -> 1/3 each, under 1.0 -> nothing loads.
        assert select_topics(RECORD, "THING") == []

    def test_nothing_scoring_loads_nothing(self):
        """A weak match puts a wrong fact in context - worse than an unopened
        box, which the index's own line already tells the model how to open."""
        assert select_topics(RECORD, "ZZZZ NOTHING HERE") == []

    def test_an_empty_query_loads_nothing(self):
        assert select_topics(RECORD, "") == []

    def test_deterministic(self):
        first = select_topics(RECORD, "`0xdeadbeef` ALPHA BETA")
        second = select_topics(RECORD, "`0xdeadbeef` ALPHA BETA")
        assert [(s.heading, s.score) for s in first] == [
            (s.heading, s.score) for s in second
        ]

    def test_ties_break_on_file_order(self):
        # ALPHA and BETA each contain one df-1 key -> equal scores, so the
        # order must come from the file, not from dict iteration.
        selected = select_topics(RECORD, "ALPHA BETA")
        assert [s.heading for s in selected] == ["Alpha", "Beta"]

    def test_the_top_topic_is_always_included(self):
        # Even when it alone exceeds the budget.
        selected = select_topics(RECORD, "`0xdeadbeef`", budget=1)
        assert [s.heading for s in selected] == ["Alpha"]

    def test_the_budget_stops_the_fill(self):
        text = "`0xdeadbeef` ALPHA BETA"
        everything = select_topics(RECORD, text, budget=DEFAULT_BUDGET)
        assert len(everything) == 2
        squeezed = select_topics(RECORD, text, budget=len(everything[0].text) + 1)
        assert len(squeezed) == 1

    def test_a_record_with_no_ticket_section_still_selects(self):
        flat = RECORD[: RECORD.index("## 6. Open tickets")]
        assert [s.heading for s in select_topics(flat, "`0xdeadbeef`")] == ["Alpha"]

    def test_an_oversized_top_topic_is_kept_as_head_and_tail(self):
        from agent13 import truth_select

        big = "### Huge\n" + "- " + "x" * 500 + "\n" * 1
        big = "### Huge\n" + ("- filler line\n" * 5000) + "- the `0xdeadbeef` tail\n"
        record = big + "\n## 6. Open tickets\n- t - FIRST SEEN 2026-11-04\n"
        assert len(big) > truth_select.OVERSIZE_TOPIC
        selected = select_topics(record, "`0xdeadbeef`")
        assert len(selected) == 1
        assert "chars elided" in selected[0].text
        assert selected[0].text.startswith("### Huge")
        assert "0xdeadbeef" in selected[0].text

    def test_the_selected_text_is_the_topic_span_verbatim(self):
        selected = select_topics(RECORD, "`0xdeadbeef`")
        assert selected[0].text in RECORD
        assert selected[0].line_count == 3


class TestThresholdIsNotTheOnlyControl:
    def test_both_controls_are_named(self):
        # The log carries both; tuning from the threshold alone is impossible
        # because a real record has ~18 topics above it.
        assert DEFAULT_THRESHOLD == 1.0
        assert DEFAULT_BUDGET == 40_000


# A record whose ten topics all share one structural word, plus one topic that
# owns a rare key. This is the shape a real query has: the handoff's prose
# carries "CHECK"/"SESSION"/"TABLE" alongside the subject.
MARKUP_RECORD = (
    "## 1. Environment\n\n"
    + "".join(
        f"### T{i}\n- CHECK the thing\n- filler {i}\n\n" for i in range(10)
    )
    + "### The real one\n- the `0xdeadbeef` handler\n\n"
    + "## 6. Open tickets\n\n- t - FIRST SEEN 2026-11-04\n"
)


class TestSelectByAddress:
    """The handoff's `load` list, resolved against the record's addresses."""

    NUMBERED = RECORD.replace("### Alpha", "### 1.1 Alpha").replace(
        "### Beta", "### 1.2 Beta"
    ).replace("### Gamma", "### 1.3 Gamma")

    def test_resolves_an_address_to_its_topic(self):
        selected, not_found, over_budget = select_by_address(self.NUMBERED, ["1.2"])
        assert [s.heading for s in selected] == ["Beta"]
        assert not_found == [] and over_budget == []

    def test_nomination_order_is_preserved(self):
        # The wrap-up wrote them in priority order; the seed must honour it.
        selected, _, _ = select_by_address(self.NUMBERED, ["1.3", "1.1"])
        assert [s.heading for s in selected] == ["Gamma", "Alpha"]

    def test_an_unknown_address_is_reported_not_guessed(self):
        """A stale address must not resolve to whatever now sits there.

        This is the property the whole numbering scheme exists for: a number
        either resolves or it does not.
        """
        selected, not_found, _ = select_by_address(self.NUMBERED, ["9.9", "1.1"])
        assert [s.heading for s in selected] == ["Alpha"]
        assert not_found == ["9.9"]

    def test_an_unnumbered_record_resolves_nothing(self):
        selected, not_found, _ = select_by_address(RECORD, ["1.1"])
        assert selected == [] and not_found == ["1.1"]

    def test_every_nominated_topic_is_carried(self):
        """No aggregate budget: the prompt already caps the nomination at six.

        On the first live run the budget silently dropped the topic the model
        nominated SECOND and reported it as `missing` - indistinguishable from
        an invented address, and it broke the promise the block makes.
        """
        selected, not_found, over_budget = select_by_address(
            self.NUMBERED, ["1.1", "1.2", "1.3"]
        )
        assert [s.heading for s in selected] == ["Alpha", "Beta", "Gamma"]
        assert not_found == [] and over_budget == []

    def test_an_oversized_topic_is_truncated_not_dropped(self):
        from agent13 import truth_select

        big = "### 1.1 Huge\n" + ("- filler line\n" * 6000) + "- the tail fact\n"
        record = big + "\n## 6. Open tickets\n\n- t - FIRST SEEN 2026-11-05\n"
        assert len(big) > truth_select.OVERSIZE_TOPIC
        selected, not_found, over_budget = select_by_address(record, ["1.1"])
        assert len(selected) == 1
        assert "chars elided" in selected[0].text
        assert not_found == [] and over_budget == []

    def test_the_text_is_the_topic_span_verbatim(self):
        selected, _, _ = select_by_address(self.NUMBERED, ["1.1"])
        assert selected[0].text in self.NUMBERED
        assert selected[0].text.startswith("### 1.1 Alpha")


class TestDfCeiling:
    def test_a_key_in_more_than_eight_topics_cannot_select(self):
        """Markup alone must load nothing.

        Before the ceiling, a query of only structural vocabulary scored 1.0
        in every topic that shared it and the selector confidently loaded one -
        worse than loading nothing, because the model may treat it as covered.
        """
        assert select_topics(MARKUP_RECORD, "CHECK") == []

    def test_the_ceiling_leaves_a_rare_key_working(self):
        selected = select_topics(MARKUP_RECORD, "CHECK `0xdeadbeef`")
        assert [s.heading for s in selected] == ["The real one"]
        # Only the rare key contributed: 1/1, not 1/1 + 1/10.
        assert selected[0].score == 1.0

    def test_the_ceiling_is_a_statistic_not_a_word_list(self):
        # The rule is "appears in more than DF_CEILING topics", whatever the
        # word means - so it transfers to a record in another language.
        assert DF_CEILING == 8

    def test_a_key_at_the_ceiling_still_counts(self):
        # df == 8 is under the ceiling: the boundary is "more than", not "at".
        # Observable only alongside a rare key - 1/8 alone is far below the
        # threshold, which is why the ceiling and the threshold are separate.
        record = (
            "## 1. Environment\n\n"
            + "".join(f"### T{i}\n- CHECK the thing\n\n" for i in range(7))
            + "### The real one\n- CHECK plus `0xdeadbeef`\n\n"
            + "## 6. Open tickets\n\n- t - FIRST SEEN 2026-11-04\n"
        )
        selected = select_topics(record, "CHECK `0xdeadbeef`")
        assert [s.heading for s in selected] == ["The real one"]
        assert selected[0].score == 1.12  # 1/1 for the rare key + 1/8 for CHECK
