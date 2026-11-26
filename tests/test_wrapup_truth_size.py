"""The wrap-up truth file: one item cap, and a size check that only logs.

History, because the file has been through four regimes. A 150-line cap, then a
27,000-char budget, then a 38-item / 700-char pair, then a post-wrap-up nudge
that rewrote the file whenever it came out too big. Measured on 2026-10-27 from
the same starting file: the char budget was met by deleting six verified facts,
the nudge's own wording triggered 14 whole-file python rewrites and destroyed
another, and the per-item cap turned the reconcile into 26 round-trips.

The regime is gone. What remains:

- One cap on TOPIC count (``TRUTH_TOPIC_CAP``) - navigability, not size.
- A CONCISION rule for length, since no cap bounds it any more.
- ``_check_truth_file_size`` logs ``wrapup_truth_oversize`` with
  ``nudged: False`` and adds no turn.

The truncation test in that check is still tied to ``bound_tool_result``: it is
what would tell us the upstream guarantee (the post-compaction rebuild injects
the file whole, uncapped) had broken.

Note on fixtures: ``bound_tool_result`` returns text unchanged when
``len(text) <= HEAD_CHARS + TAIL_CHARS`` (30,000) *regardless of the cap*, so a
truncation fixture must exceed 30,000 chars - a small patched cap alone does
not trigger it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.prompts import DEFAULT_REPORT_AND_COMPACT_PROMPT, TRUTH_TOPIC_CAP
from agent13.tool_output_bound import HEAD_CHARS, TAIL_CHARS, bound_tool_result

# The real default; HEAD_CHARS + TAIL_CHARS is the floor below which nothing is
# ever elided, so the cap must not be patched below it.
CAP = 30_000
FLOOR = HEAD_CHARS + TAIL_CHARS


def make_agent(truth_path=""):
    """A minimal agent with the wrap-up armed and snapshots mocked out."""
    agent = Agent(client=MagicMock(), model="test-model", auto_context_threshold=0)
    agent._save_auto_context_snapshot = MagicMock()
    agent.durable_records.truth_path = truth_path
    agent._report_and_compact_triggered = True
    agent._running = True
    return agent


def write_truth(tmp_path, lines, chars_per_line):
    """Write a truth file of exactly `lines` lines; return its path."""
    path = tmp_path / "docs_archive" / "focus_current_truth.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("x" * chars_per_line for _ in range(lines)))
    return str(path)


def stub_llm_turn(agent, on_call=None):
    """Replace _llm_turn with a recorder; returns the call list."""
    calls = []

    async def fake(_polite_acquired=False):
        calls.append(1)
        if on_call:
            on_call(len(calls))

    agent._llm_turn = fake
    return calls


def nudge_messages(agent):
    """The injected size-nudge user messages."""
    return [
        m
        for m in agent.messages
        if m.get("role") == "user" and "The truth file is" in str(m.get("content"))
    ]


def patched_cap(cap=CAP):
    """Patch the global tool-result cap deterministically."""
    return patch(
        "agent13.config.get_config",
        return_value=SimpleNamespace(tools_max_result_chars=cap),
    )


class TestPromptText:
    """Test 1: the prompt names the call and keeps the load-bearing rules."""

    def test_write_file_only_for_a_new_file(self):
        """The prompt no longer tells the model to rewrite: the truth file is
        edited in place, and write_file is scoped to creating it."""
        assert (
            "write_file is allowed on this file ONLY if it does not exist"
            in DEFAULT_REPORT_AND_COMPACT_PROMPT
        )

    def test_keeps_journal_and_append(self):
        # Mirrors tests/test_auto_context.py::test_default_mentions_journal_append
        # - the journal append rule must survive this edit.
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT.lower()
        assert "journal" in text
        assert "append" in text

    def test_states_the_topic_cap(self):
        """One cap: the topic count. Facts are uncapped, and there is no
        per-topic character cap - a per-topic cap plus a topic cap deadlocks
        at the limit (split one -> 39 topics; merge two -> an over-size topic).
        """
        assert f"<= {TRUTH_TOPIC_CAP} TOPICS" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "chars per item" not in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "chars per topic" not in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_a_new_topic_is_not_forced_under_a_wrong_heading(self):
        """The placement rule, inverted 2026-11-04.

        The old text called appending "the normal case" and made a new topic
        cost a MERGE. That is what made a grab-bag structural: a fact about a
        new subsystem got filed under the nearest heading instead of getting
        its own. The rule is now "add a topic when no heading fits", which is
        the uncommon case - so the prompt must NOT claim it is the normal one.
        """
        assert "do not force the" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "fact under a heading it does not fit" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "that is the normal case" not in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_a_merge_must_rename_the_heading(self):
        """A merge destroys a navigation key, so it has to buy one back.

        Merge two topics under one of the old names and every fact about the
        other now sits under a heading that does not name it - the index stub
        then points at the wrong subsystem, which is the failure the index
        work exists to fix.
        """
        assert "RENAME the merged topic" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "names BOTH sets of facts" in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_the_merge_picks_related_and_stale_topics(self):
        """Relatedness keeps the heading honest; staleness keeps the damage
        small. Either alone is wrong: related-only spends two hot keys,
        stale-only cannot produce an honest heading."""
        assert "a reader would look for TOGETHER" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "STALEST" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "backstop, not a" in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_states_the_split_trigger(self):
        """One number for one rule: the old "~15 facts" is deleted, not kept
        beside the new one, because two numbers for one rule cannot be
        followed - and the old one was never acted on."""
        assert "~40" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "lines is a split candidate" in DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "~15 facts" not in DEFAULT_REPORT_AND_COMPACT_PROMPT
        # The correction against a per-topic character cap survives.
        assert "NO hard cap on a topic" in DEFAULT_REPORT_AND_COMPACT_PROMPT

    def test_states_the_numbering_rule(self):
        """The address is a HANDLE the next session loads by.

        Two properties, both load-bearing and both silent when violated: the
        address never changes (a shifted number loads the wrong topic), and a
        new topic takes the next free number rather than sliding into a gap.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "Every topic heading carries its ADDRESS" in text
        assert "NEVER changes" in text
        assert "Do not renumber" in text
        assert "next free number" in text
        # The format template must show the address, or the model writes none.
        assert "### <s>.<n> <TOPIC" in text
        assert "### <TOPIC>" not in text

    def test_states_the_handoff_block(self):
        """The block is the machine interface, and it must be LAST.

        The 04:16 scope call appeared in the closing summary, not the journal
        entry, and the summary is discarded at rebuild - so the block is
        written after the prose and captured separately.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "as the very last thing in your final message" in text
        assert '"load"' in text and '"for_user"' in text
        assert "CARRY FORWARD" in text
        # The example must parse as a real block, or the model copies a shape
        # the parser rejects.
        from agent13.handoff import parse_block

        start = text.index("4. THEN")
        block = parse_block(text[start : start + 400])
        assert block is not None
        assert block.load == ["3.2", "3.7"]

    def test_the_handoff_block_has_exactly_two_fields(self):
        """A field with no reader is a second source of truth with no resolver.

        `next` and `goal` were in an earlier draft: `next` restated the b.
        block's action line and `goal` restated the agent's own goal state,
        and neither had a consumer anywhere in the seed or the log.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "exactly two fields" in text
        assert '"next"' not in text
        assert '"goal"' not in text

    def test_heading_is_a_label_not_the_first_fact(self):
        """The heading is a LABEL - never restate it in the bullet below.

        The 10-28 migration derived each heading from the first bold span of
        the bullet that opened the topic, then kept the bullet intact: 16 of
        25 topics restated their heading in the first fact. The record
        was cleaned by hand, but RECONCILE reads the record in full at every
        wrap-up, so without a rule the model copies the shape it reads. The
        EXAMPLE is the load-bearing part - a rule alone is what the prompt
        already lacked. The wrong form shown is the heading + separator +
        restatement, NOT the bold restatement: the latter is also observed but
        would read as discouraging bold in facts generally, which the record
        relies on for emphasis. And not the post-strip residue ("- = path") -
        no model writes that, so it teaches nothing.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "The heading is a LABEL, not the first fact" in text
        assert "restate it in the bullet below" in text
        assert '"### netLog" + "- netLog = `~/.../netLog*.log`"' in text
        assert '"- `~/.../netLog*.log`, newest per run."' in text
        assert "- **netLog**" not in text

    def test_tells_the_model_to_verify_with_the_count_commands(self):
        """The cap is only useful if the model checks it before finishing.

        The topic count must be SCOPED to before the ticket section: that
        section's own "### " sub-headings are not topics, and a bare
        `grep -c '^### '` counts them - so a record at the cap reads as over
        it and the model merges topics to fix a non-problem.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert (
            "awk 'tolower($0) ~ /^## .*open tickets/{exit} /^### /{n++} END{print n+0}'"
            in text
        )
        assert "grep -c '^- '" in text

    def test_prompt_forbids_regeneration(self):
        """The 2026-10-27 failure was a rewrite under another name: 14 python
        heredocs each embedding the file's content. Naming only write_file
        leaves that path open, so every route must be closed - and the rule
        now lives in the prompt, since the nudge that carried it is gone."""
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        for phrase in ("no write_file", "no heredoc", "no python string literal"):
            assert phrase in text.lower()
        assert "NEVER DELETE" in text

    def test_no_archive_in_the_prompt(self):
        """Nothing leaves the truth: facts are corrected in place, and a merge
        joins topics rather than content. The archive is retired - with topics
        capped and facts free, nothing needs to leave to keep the load small."""
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "truth_archive" not in text
        assert "NEVER DELETE" in text
        assert "append-and-correct" in text

    def test_prompt_makes_no_claim_about_what_the_next_context_loads(self):
        """The prompt must not assert a runtime configuration.

        Whether the next context gets the topic index or the whole file is
        [auto_context] truth_index, which defaults to false. A prompt that says
        "the next context loads the index only" is false in the control run -
        and the model would write facts believing they are not loaded.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT.lower()
        assert "next context loads" not in text
        assert "topic index" not in text
        assert "costs nothing" not in text
        # The old clause asserted the opposite ("injects this file in full").
        # "in full" on its own is legitimate - RECONCILE tells the model to
        # READ the file in full.
        assert "injects this file in full" not in text

    def test_no_size_target_in_the_prompt(self):
        """The size regime is gone: no char budget, no per-item cap, no wc -c.
        Length is bounded by the topic count and steered by the concision rule.
        """
        text = DEFAULT_REPORT_AND_COMPACT_PROMPT
        assert "chars per item" not in text
        assert "wc -c" not in text
        assert "CONCISION" in text


class TestBudgetDriftAlarm:
    """Test 3: one number, two places - the prompt's literal and the constant.
    The prompt is not an f-string (it carries a {focus} placeholder), so a test
    is the only thing that can catch drift."""

    def test_prompt_literal_matches_constant(self):
        assert f"<= {TRUTH_TOPIC_CAP} TOPICS" in DEFAULT_REPORT_AND_COMPACT_PROMPT


class TestSizeCheck:
    """The post-wrap-up check logs and never acts - no nudge turn at all."""

    @pytest.mark.asyncio
    async def test_under_both_limits_does_not_log(self, tmp_path, monkeypatch):
        """100 lines, well under the cap - one call, no event."""
        monkeypatch.chdir(tmp_path)
        truth = write_truth(tmp_path, lines=100, chars_per_line=20)

        agent = make_agent(truth)
        calls = stub_llm_turn(agent)

        with patched_cap():
            await agent._run_report_and_compact_turn()

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_oversize_never_adds_a_turn(self, tmp_path, monkeypatch):
        """The whole point of the change: an oversized file is logged, not
        repaired. 200 lines of 200 chars is over the read cap, and the old
        regime answered that with a second turn that rewrote the file."""
        monkeypatch.chdir(tmp_path)
        truth = write_truth(tmp_path, lines=200, chars_per_line=200)
        assert len(open(truth).read()) > FLOOR  # actually elidable

        agent = make_agent(truth)
        calls = stub_llm_turn(agent)

        with patched_cap():
            await agent._run_report_and_compact_turn()

        assert len(calls) == 1, "the wrap-up turn only - the nudge is gone"
        assert nudge_messages(agent) == []

    @pytest.mark.asyncio
    async def test_no_truth_path_does_nothing(self, tmp_path, monkeypatch):
        """No focus and no captured path - no extra turn, no error."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent("")  # truth_path unset
        calls = stub_llm_turn(agent)

        with patched_cap():
            await agent._run_report_and_compact_turn()

        assert len(calls) == 1
        assert nudge_messages(agent) == []

    @pytest.mark.asyncio
    async def test_missing_file_does_nothing(self, tmp_path, monkeypatch):
        """A path that does not exist is not an error."""
        monkeypatch.chdir(tmp_path)
        agent = make_agent("docs_archive/nope_current_truth.md")
        calls = stub_llm_turn(agent)

        with patched_cap():
            await agent._run_report_and_compact_turn()

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_logs_oversize_event(self, tmp_path, monkeypatch):
        """The retro counts how often this fires, so the event must exist."""
        monkeypatch.chdir(tmp_path)
        truth = write_truth(tmp_path, lines=200, chars_per_line=200)
        agent = make_agent(truth)
        calls = stub_llm_turn(agent)

        events = []

        with (
            patched_cap(),
            patch("agent13.core.is_debug_enabled", return_value=True),
            patch(
                "agent13.core.log_tps_event",
                side_effect=lambda name, data: events.append((name, data)),
            ),
        ):
            await agent._run_report_and_compact_turn()

        oversize = [d for n, d in events if n == "wrapup_truth_oversize"]
        assert len(oversize) == 1
        assert oversize[0]["lines"] == 200
        assert oversize[0]["items"] == 0
        assert oversize[0]["cap"] == CAP
        assert oversize[0]["truncated"] is True
        # False distinguishes the log-only regime from the old nudge in the
        # retro's counts, and the turn count proves nothing followed the log.
        assert oversize[0]["nudged"] is False
        assert len(calls) == 1


class TestBounderBoundary:
    """Test 4: the invariant is tied to the bounder, not a copied constant."""

    def test_at_cap_no_marker(self):
        text = "x" * CAP
        bounded, spill = bound_tool_result(text, CAP, "read_file")
        assert bounded is text
        assert spill is None

    def test_one_char_over_inserts_marker(self):
        text = "x" * (CAP + 1)
        bounded, _spill = bound_tool_result(text, CAP, "read_file")
        assert bounded is not text
        assert "chars elided" in bounded


class TestCapHelper:
    """Test 8: one cap-resolution rule, shared by both callers."""

    def test_global_cap_when_no_override(self):
        with patched_cap(1234):
            assert Agent._tool_result_cap("read_file") == 1234

    def test_per_tool_override_wins(self):
        with (
            patched_cap(1234),
            patch.dict("tools._tool_max_result_chars", {"read_file": 500}),
        ):
            assert Agent._tool_result_cap("read_file") == 500

    def test_read_path_and_size_check_resolve_same_cap(self):
        """The size check and the read path must agree on the cap."""
        agent = make_agent()
        with patched_cap(CAP):
            assert Agent._tool_result_cap("read_file") == CAP
            text = "x" * (CAP + 1)
            bounded = agent._bound_tool_result_text(text, "read_file")
            assert bounded is not text, (
                "the read path must bound at the same cap the size check uses"
            )
