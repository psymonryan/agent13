"""Tests for agent13.commands shared slash-command logic.

Covers the helpers that don't have an obvious home elsewhere:
``_normalize_idx`` (negative-index normalization) and ``_parse_index_spec``
(spec parsing for /delete h N | /delete q N).  These were previously only
exercised indirectly through TUI integration tests.
"""

from agent13.commands import _DELETE_USAGE, _normalize_idx, _parse_index_spec


# ---------------------------------------------------------------------------
# _normalize_idx
# ---------------------------------------------------------------------------


class TestNormalizeIdx:
    """Negative-index normalization for 1-based indexing.

    Mirrors Python's negative-index ergonomics on a 1-based scale:
    ``-1`` -> ``total``, ``-2`` -> ``total - 1``, etc.  Positive indices
    pass through unchanged.
    """

    def test_positive_passes_through(self):
        assert _normalize_idx(1, 5) == 1
        assert _normalize_idx(3, 5) == 3
        assert _normalize_idx(5, 5) == 5

    def test_negative_one_is_last(self):
        assert _normalize_idx(-1, 5) == 5
        assert _normalize_idx(-1, 1) == 1

    def test_negative_n_is_n_from_end(self):
        assert _normalize_idx(-2, 5) == 4
        assert _normalize_idx(-3, 5) == 3
        assert _normalize_idx(-5, 5) == 1

    def test_zero_unchanged(self):
        # Zero is a degenerate "index" — _parse_index_spec validates bounds
        # separately.  _normalize_idx itself should not special-case it.
        assert _normalize_idx(0, 5) == 0

    def test_out_of_range_negative_wraps_below_one(self):
        # -6 on a 5-item collection would be 0 — caller must reject.
        assert _normalize_idx(-6, 5) == 0


# ---------------------------------------------------------------------------
# _parse_index_spec — 'last' keyword
# ---------------------------------------------------------------------------


class TestParseIndexSpecLastKeyword:
    def test_last_returns_total(self):
        assert _parse_index_spec("last", 5) == ([5], None)

    def test_last_case_insensitive(self):
        assert _parse_index_spec("LAST", 3) == ([3], None)
        assert _parse_index_spec("Last", 3) == ([3], None)

    def test_last_on_empty_is_error(self):
        indices, error = _parse_index_spec("last", 0)
        assert indices == []
        assert error == "No items to delete"


# ---------------------------------------------------------------------------
# _parse_index_spec — single index
# ---------------------------------------------------------------------------


class TestParseIndexSpecSingle:
    def test_positive_single(self):
        assert _parse_index_spec("1", 5) == ([1], None)
        assert _parse_index_spec("5", 5) == ([5], None)

    def test_negative_single(self):
        assert _parse_index_spec("-1", 5) == ([5], None)
        assert _parse_index_spec("-2", 5) == ([4], None)
        assert _parse_index_spec("-5", 5) == ([1], None)

    def test_out_of_range_positive(self):
        indices, error = _parse_index_spec("6", 5)
        assert indices == []
        assert "out of range" in error

    def test_out_of_range_negative(self):
        # -6 on 5 items wraps to 0, which is < 1
        indices, error = _parse_index_spec("-6", 5)
        assert indices == []
        assert "out of range" in error

    def test_invalid_int(self):
        indices, error = _parse_index_spec("abc", 5)
        assert indices == []
        assert "Invalid index" in error


# ---------------------------------------------------------------------------
# _parse_index_spec — ranges
# ---------------------------------------------------------------------------


class TestParseIndexSpecRange:
    def test_basic_range(self):
        assert _parse_index_spec("1:3", 5) == ([1, 2, 3], None)
        assert _parse_index_spec("2:4", 5) == ([2, 3, 4], None)

    def test_open_start(self):
        assert _parse_index_spec(":3", 5) == ([1, 2, 3], None)

    def test_open_end(self):
        assert _parse_index_spec("3:", 5) == ([3, 4, 5], None)

    def test_negative_start(self):
        # -2 on 5 items is 4
        assert _parse_index_spec("-2:5", 5) == ([4, 5], None)

    def test_negative_end(self):
        # -1 on 5 items is 5
        assert _parse_index_spec("1:-1", 5) == ([1, 2, 3, 4, 5], None)

    def test_both_negative(self):
        # -3:-1 on 5 items is 3:5
        assert _parse_index_spec("-3:-1", 5) == ([3, 4, 5], None)

    def test_last_in_range(self):
        assert _parse_index_spec("last:last", 5) == ([5], None)
        assert _parse_index_spec("4:last", 5) == ([4, 5], None)
        assert _parse_index_spec("last:5", 5) == ([5], None)

    def test_start_after_end_is_error(self):
        indices, error = _parse_index_spec("4:2", 5)
        assert indices == []
        assert "start" in error and "end" in error

    def test_invalid_range_format(self):
        # Multiple colons
        indices, error = _parse_index_spec("1:2:3", 5)
        assert indices == []
        assert "Invalid range format" in error

    def test_invalid_start(self):
        indices, error = _parse_index_spec("abc:3", 5)
        assert indices == []
        assert "Invalid start index" in error

    def test_invalid_end(self):
        indices, error = _parse_index_spec("1:xyz", 5)
        assert indices == []
        assert "Invalid end index" in error

    def test_range_out_of_bounds(self):
        indices, error = _parse_index_spec("1:6", 5)
        assert indices == []
        assert "out of range" in error

        indices, error = _parse_index_spec("0:3", 5)
        assert indices == []
        assert "out of range" in error


# ---------------------------------------------------------------------------
# _DELETE_USAGE constant
# ---------------------------------------------------------------------------


class TestDeleteUsage:
    """Lock in the usage string so all four /delete branches stay consistent."""

    def test_usage_mentions_all_three_targets(self):
        assert "/delete h N" in _DELETE_USAGE
        assert "/delete q N" in _DELETE_USAGE
        assert "/delete s NAME" in _DELETE_USAGE

    def test_usage_mentions_index_formats(self):
        assert "last" in _DELETE_USAGE
        assert "range" in _DELETE_USAGE
        assert "-1" in _DELETE_USAGE  # negative index example
