"""The handoff block: the wrap-up's machine-readable closing statement.

Two fields, both with a reader, and both fail SILENTLY when they go wrong:

* ``load`` - the addresses the next action needs. A block that does not parse
  means the seed silently falls back to guessing from vocabulary, which is the
  behaviour the block exists to replace.
* ``for_user`` - what the user must act on. The summary that produces it is
  discarded at rebuild, so an item that is not carried is lost with no error
  anywhere.

The parsing is deliberately forgiving about SHAPE (a bare fence, prose around
it) and strict about CONTENT (only the two fields survive).
"""

from agent13.handoff import (
    MAX_FOR_USER,
    Handoff,
    for_user_items,
    parse_block,
    render_block,
    render_for_user,
)


class TestParseBlock:
    def test_parses_a_fenced_block(self):
        text = 'Done.\n\n```json\n{"load": ["3.2"], "for_user": ["ask about scope"]}\n```\n'
        block = parse_block(text)
        assert block.load == ["3.2"]
        assert block.for_user == ["ask about scope"]

    def test_a_bare_fence_is_accepted(self):
        # Models drop the "json" tag often enough that rejecting it would lose
        # the block on a large fraction of wrap-ups.
        block = parse_block('```\n{"load": ["1.1"]}\n```')
        assert block.load == ["1.1"]

    def test_the_LAST_block_wins(self):
        """The model may quote an earlier block while reasoning.

        The one that counts is the one it ended with - that is the whole point
        of "the block is the last thing you emit".
        """
        text = (
            'Earlier I wrote:\n```json\n{"load": ["9.9"]}\n```\n'
            'Final:\n```json\n{"load": ["3.2"]}\n```\n'
        )
        assert parse_block(text).load == ["3.2"]

    def test_no_block_is_none(self):
        assert parse_block("just prose, no fence") is None
        assert parse_block("") is None

    def test_a_malformed_block_is_none(self):
        # Not an exception: the caller falls back to the selector and logs it.
        assert parse_block("```json\n{not json}\n```") is None

    def test_a_json_array_is_not_a_block(self):
        assert parse_block('```json\n["3.2"]\n```') is None

    def test_missing_fields_default_to_empty(self):
        block = parse_block('```json\n{"load": ["3.2"]}\n```')
        assert block.for_user == []

    def test_for_user_is_capped(self):
        items = ", ".join(f'"item {i}"' for i in range(6))
        block = parse_block(f'```json\n{{"for_user": [{items}]}}\n```')
        assert len(block.for_user) == MAX_FOR_USER == 3

    def test_unknown_keys_are_recorded_not_kept(self):
        """`next` and `goal` were in an earlier draft and have no reader.

        A field with no consumer is a second source of truth with no resolver,
        so it is dropped - but recorded, so a model drifting back to them is
        visible in the log rather than silent.
        """
        block = parse_block('```json\n{"load": ["3.2"], "next": "do it"}\n```')
        assert block.unknown == ["next"]
        assert not hasattr(block, "next")

    def test_non_string_items_are_stringified_and_blank_dropped(self):
        block = parse_block('```json\n{"for_user": ["", "  ", 42]}\n```')
        assert block.for_user == ["42"]

    def test_a_non_list_field_is_empty(self):
        block = parse_block('```json\n{"load": "3.2"}\n```')
        assert block.load == []


class TestRenderBlock:
    def test_round_trips_through_parse(self):
        original = Handoff(load=["3.2", "3.7"], for_user=["a", "b"])
        again = parse_block(render_block(original))
        assert again.load == ["3.2", "3.7"]
        assert again.for_user == ["a", "b"]

    def test_renders_only_the_two_fields(self):
        # Re-rendered from the parsed fields, so a stray key or a malformed
        # list cannot survive into the record.
        text = render_block(Handoff(load=["3.2"], for_user=[], unknown=["next"]))
        assert "next" not in text
        assert '"load": ["3.2"]' in text

    def test_empty_lists_render(self):
        text = render_block(Handoff(load=[], for_user=[]))
        assert '"load": []' in text and '"for_user": []' in text


class TestForUser:
    def test_items_are_capped(self):
        block = Handoff(for_user=["a", "b", "c", "d"])
        assert for_user_items(block) == ["a", "b", "c"]

    def test_no_block_yields_nothing(self):
        assert for_user_items(None) == []

    def test_render_is_empty_when_there_is_nothing(self):
        assert render_for_user([]) == ""

    def test_render_lists_every_item(self):
        text = render_for_user(["scope call: continue?", "box is down"])
        assert text.startswith("FOR THE USER (unresolved):")
        assert "- scope call: continue?" in text
        assert "- box is down" in text
