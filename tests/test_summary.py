"""The folded one-line "what": extraction and cell-aware truncation.

Covers ui/summary.py, which both the TUI headers and the REPL lines import so
the two surfaces always agree. Pure functions - no app, no screen.
"""

import pytest

from ui.summary import (
    DEFAULT_BUDGET,
    MAX_BUDGET,
    MIN_BUDGET,
    budget_for,
    first_line,
    tool_what,
    truncate_line,
)


# =============================================================================
# tool_what
# =============================================================================


class TestToolWhat:
    def test_command_uses_the_command_string(self):
        assert tool_what("command", {"command": "git log --oneline -5"}) == (
            "git log --oneline -5"
        )

    def test_command_collapses_newlines(self):
        assert tool_what("command", {"command": "ls\npwd"}) == "ls pwd"

    def test_file_tools_use_the_filepath(self):
        args = {"filepath": "src/foo.py", "content": "x" * 200}
        for name in ("read_file", "write_file", "edit_file"):
            assert tool_what(name, args) == "src/foo.py"

    def test_unknown_tool_falls_back_to_first_short_string(self):
        args = {"query": "find me", "options": {"a": 1}}
        assert tool_what("mystery_tool", args) == "find me"

    def test_unknown_tool_with_only_long_values_is_empty(self):
        assert tool_what("mystery_tool", {"blob": "x" * 200}) == ""

    def test_no_arguments_is_empty(self):
        assert tool_what("command", None) == ""
        assert tool_what("command", {}) == ""

    def test_empty_string_value_is_skipped(self):
        assert tool_what("command", {"command": "   "}) == ""

    def test_unknown_tool_falls_back_to_first_short_scalar(self):
        assert tool_what("square_number", {"x": 5}) == "5"

    def test_scalar_fallback_renders_bools_plainly(self):
        assert tool_what("mystery_tool", {"flag": True}) == "true"
        assert tool_what("mystery_tool", {"flag": False}) == "false"

    def test_scalar_fallback_renders_floats(self):
        assert tool_what("mystery_tool", {"p": 0.5}) == "0.5"

    def test_string_argument_beats_scalar(self):
        args = {"flag": True, "query": "find me"}
        assert tool_what("mystery_tool", args) == "find me"

    def test_scalar_is_last_resort_after_long_strings(self):
        args = {"payload": "x" * 200, "x": 7}
        assert tool_what("mystery_tool", args) == "7"

    def test_only_structured_values_stay_empty(self):
        assert tool_what("mystery_tool", {"items": [1, 2], "meta": None}) == ""

    def test_non_finite_float_is_skipped(self):
        assert tool_what("mystery_tool", {"n": float("nan")}) == ""


# =============================================================================
# first_line
# =============================================================================


class TestFirstLine:
    def test_first_non_empty_line(self):
        assert first_line("let me check the log\nthen the test") == (
            "let me check the log"
        )

    def test_leading_blank_lines_are_skipped(self):
        assert first_line("\n\n  the real start  \nmore") == "the real start"

    def test_internal_whitespace_is_collapsed(self):
        assert first_line("a   b\tc") == "a b c"

    def test_empty_is_empty(self):
        assert first_line("") == ""
        assert first_line("   \n  ") == ""


# =============================================================================
# truncate_line
# =============================================================================


class TestTruncateLine:
    def test_short_text_is_unchanged(self):
        assert truncate_line("short", 10) == "short"

    def test_exact_fit_is_unchanged(self):
        assert truncate_line("12345", 5) == "12345"

    def test_long_text_is_truncated_with_ellipsis(self):
        out = truncate_line("a" * 50, 10)
        assert out.endswith("…")
        assert len(out) == 10  # 9 chars + ellipsis

    def test_zero_budget_is_empty(self):
        assert truncate_line("abc", 0) == ""

    def test_wide_characters_count_as_two_cells(self):
        # 中 is 2 cells. Budget 3 -> one wide char (2) + ellipsis (1).
        out = truncate_line("中中中", 3)
        assert out == "中…"


# =============================================================================
# budget_for
# =============================================================================


class TestBudgetFor:
    def test_non_numeric_available_falls_back_to_default(self):
        assert budget_for("Thoughts: ", object()) == DEFAULT_BUDGET
        assert budget_for("Thoughts: ", 0) == DEFAULT_BUDGET

    def test_small_available_is_floored(self):
        assert budget_for("Thoughts: ", 20) == MIN_BUDGET

    def test_huge_available_is_capped(self):
        assert budget_for("Thoughts: ", 500) == MAX_BUDGET

    def test_normal_available_subtracts_overhead(self):
        # "Thoughts: " is 10 cells, the " ▶" indicator is 2.
        assert budget_for("Thoughts: ", 60) == 60 - 10 - 2

    def test_no_indicator_leaves_the_tail_space(self):
        assert budget_for("Thoughts: ", 60, indicator=False) == 60 - 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
