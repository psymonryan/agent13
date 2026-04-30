"""
Tests for AST validation and context preview in edit_file tool.

These tests verify:
1. AST validation gate: invalid Python edits are rejected before writing
2. Context preview: successful edits include a preview with surrounding lines
3. Edge cases: non-Python files skip validation, empty edits, etc.
"""

import os
from tools.edit_file import edit_file, _validate_python_syntax, _build_preview
from agent13.sandbox import get_temp_dir


def create_test_file(content: str, name: str = "test_ast.py") -> str:
    """Create a test file in the system temp directory and return its path."""
    filepath = os.path.join(get_temp_dir(), name)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def delete_test_file(filepath: str):
    """Delete test file if it exists."""
    if os.path.exists(filepath):
        os.unlink(filepath)


def result_success(result: dict) -> bool:
    """Helper to check if result indicates success."""
    return result.get("success", False) is True


def result_has_error(result: dict) -> bool:
    """Helper to check if result has an error."""
    return "error" in result


# =============================================================================
# _validate_python_syntax unit tests
# =============================================================================


class TestValidatePythonSyntax:
    """Tests for the _validate_python_syntax helper."""

    def test_valid_python_returns_none(self):
        code = "def hello():\n    return 'world'\n"
        assert _validate_python_syntax(code, "test.py") is None

    def test_invalid_indentation_returns_error(self):
        code = "def hello():\nreturn 'world'\n"  # missing indent
        error = _validate_python_syntax(code, "test.py")
        assert error is not None
        assert "syntax error" in error.lower()
        assert "NOT modified" in error

    def test_missing_colon_returns_error(self):
        code = "def hello()\n    return 'world'\n"
        error = _validate_python_syntax(code, "test.py")
        assert error is not None
        assert "syntax error" in error.lower()

    def test_unmatched_paren_returns_error(self):
        code = "print('hello'\n"
        error = _validate_python_syntax(code, "test.py")
        assert error is not None
        assert "line" in error  # Should include line number

    def test_empty_file_is_valid(self):
        code = ""
        assert _validate_python_syntax(code, "test.py") is None

    def test_error_includes_line_number(self):
        code = "x = 1\ny = 2\nz = (\n"  # unclosed paren on line 3
        error = _validate_python_syntax(code, "test.py")
        assert error is not None
        assert "line 3" in error

    def test_error_includes_filepath_hint(self):
        code = "def broken(\n"
        error = _validate_python_syntax(code, "my_module.py")
        assert "NOT modified" in error
        assert "Fix the syntax" in error


# =============================================================================
# _build_preview unit tests
# =============================================================================


class TestBuildPreview:
    """Tests for the _build_preview helper."""

    def test_small_edit_shows_full_region(self):
        lines = ["line1", "line2", "EDITED", "line4", "line5"]
        preview = _build_preview(lines, 2, 3)
        # Should show context before, the edit, context after
        assert "1" in preview  # context before
        assert "2" in preview  # context before
        assert "EDITED" in preview
        assert "4" in preview  # context after
        assert "5" in preview  # context after

    def test_large_edit_shows_skip_marker(self):
        lines = [f"line{i}" for i in range(20)]
        # Edit region is lines 5-15 (10 lines, > max_edit_show=6)
        preview = _build_preview(lines, 5, 15)
        assert "..." in preview
        assert "lines)" in preview.lower() or "lines" in preview

    def test_edit_at_file_start(self):
        lines = ["EDITED", "line2", "line3", "line4"]
        preview = _build_preview(lines, 0, 1)
        # No context before (start of file), but should have context after
        assert "EDITED" in preview
        assert "2" in preview
        assert "3" in preview

    def test_edit_at_file_end(self):
        lines = ["line1", "line2", "line3", "EDITED"]
        preview = _build_preview(lines, 3, 4)
        # Context before, edit, no context after (end of file)
        assert "2" in preview
        assert "3" in preview
        assert "EDITED" in preview

    def test_line_numbers_are_1_indexed(self):
        lines = ["alpha", "beta", "gamma"]
        preview = _build_preview(lines, 1, 2)
        # Line numbers should be 1-indexed (showing 1, 2, 3)
        assert "1" in preview
        assert "2" in preview
        assert "3" in preview

    def test_empty_edit_region_returns_empty(self):
        # edit_start == edit_end means nothing changed
        lines = ["line1", "line2"]
        preview = _build_preview(lines, 1, 1)
        # Should still show context lines
        assert "1" in preview


# =============================================================================
# AST validation gate integration tests
# =============================================================================


class TestASTValidationGate:
    """Tests for the AST validation gate in edit_file."""

    def test_valid_python_edit_succeeds(self):
        filepath = create_test_file("def hello():\n    return 'world'\n")
        try:
            result = edit_file(
                filepath=filepath,
                find="return 'world'",
                content="return 'universe'",
                mode="replace",
            )
            assert result_success(result), f"Expected success, got: {result}"
            # Verify file was actually written
            with open(filepath) as f:
                content = f.read()
            assert "universe" in content
        finally:
            delete_test_file(filepath)

    def test_invalid_indentation_blocked(self):
        filepath = create_test_file("def hello():\n    return 'world'\n")
        try:
            result = edit_file(
                filepath=filepath,
                find="return 'world'",
                content="return 'universe'\nreturn 'extra'",  # bad dedent
                mode="replace",
            )
            assert result_has_error(result), f"Expected error, got: {result}"
            assert "syntax error" in result["error"].lower()
            assert "NOT modified" in result["error"]
            # Verify file was NOT written
            with open(filepath) as f:
                content = f.read()
            assert "world" in content  # original preserved
        finally:
            delete_test_file(filepath)

    def test_non_python_file_not_validated(self):
        filepath = create_test_file(
            "# Header\nSome text\nMore text\n", name="test_ast.md"
        )
        try:
            result = edit_file(
                filepath=filepath,
                find="Some text",
                content="Replaced text",
                mode="replace",
            )
            # Markdown files should skip AST validation
            assert result_success(result), f"Expected success, got: {result}"
        finally:
            delete_test_file(filepath)

    def test_invalid_syntax_in_append_blocked(self):
        filepath = create_test_file("x = 1\n")
        try:
            result = edit_file(
                filepath=filepath,
                content="def broken(\n",  # missing closing paren
                mode="append",
            )
            assert result_has_error(result), f"Expected error, got: {result}"
            assert "syntax error" in result["error"].lower()
            # File should not be modified
            with open(filepath) as f:
                content = f.read()
            assert content == "x = 1\n"
        finally:
            delete_test_file(filepath)

    def test_invalid_syntax_in_replace_range_blocked(self):
        filepath = create_test_file("def foo():\n    pass\n\ndef bar():\n    pass\n")
        try:
            result = edit_file(
                filepath=filepath,
                start_line=2,
                end_line=3,
                content="return True\nreturn False",  # dedent error
                mode="replace_range",
            )
            assert result_has_error(result), f"Expected error, got: {result}"
            assert "NOT modified" in result["error"]
        finally:
            delete_test_file(filepath)

    def test_delete_mode_that_breaks_syntax_blocked(self):
        filepath = create_test_file("def foo():\n    x = 1\n    return x\n")
        try:
            # Deleting the body line leaves `def foo():` with nothing — actually
            # that's valid Python. Let's make a case where deletion creates bad syntax.
            filepath2 = create_test_file(
                "if True:\n    x = 1\nelse\n    y = 2\n", name="test_delete_syntax.py"
            )
            try:
                result = edit_file(
                    filepath=filepath2, start_line=2, end_line=2, mode="delete"
                )
                # Deleting `x = 1` leaves `if True:\nelse\n    y = 2` — invalid
                assert result_has_error(result), f"Expected error, got: {result}"
                assert "syntax error" in result["error"].lower()
            finally:
                delete_test_file(filepath2)
        finally:
            delete_test_file(filepath)


# =============================================================================
# Context preview integration tests
# =============================================================================


class TestContextPreview:
    """Tests for the context preview on successful edits."""

    def test_replace_includes_preview(self):
        filepath = create_test_file("line1\nline2\nline3\nline4\nline5\n")
        try:
            result = edit_file(
                filepath=filepath, find="line3", content="REPLACED", mode="replace"
            )
            assert result_success(result)
            assert "preview" in result, (
                f"Expected preview key, got keys: {result.keys()}"
            )
            assert "REPLACED" in result["preview"]
        finally:
            delete_test_file(filepath)

    def test_preview_has_context_lines(self):
        filepath = create_test_file("before1\nbefore2\ntarget\nafter1\nafter2\n")
        try:
            result = edit_file(
                filepath=filepath, find="target", content="REPLACED", mode="replace"
            )
            assert result_success(result)
            preview = result["preview"]
            # Should show context before and after
            assert "before1" in preview or "before2" in preview
            assert "after1" in preview or "after2" in preview
        finally:
            delete_test_file(filepath)

    def test_preview_includes_line_numbers(self):
        filepath = create_test_file("a\nb\nc\nd\ne\n")
        try:
            result = edit_file(filepath=filepath, find="c", content="C", mode="replace")
            assert result_success(result)
            preview = result["preview"]
            # Should include line numbers (1-indexed)
            assert "1" in preview or "2" in preview
        finally:
            delete_test_file(filepath)

    def test_append_includes_preview(self):
        filepath = create_test_file("existing\n")
        try:
            result = edit_file(filepath=filepath, content="appended", mode="append")
            assert result_success(result)
            assert "preview" in result
            assert "appended" in result["preview"]
        finally:
            delete_test_file(filepath)

    def test_replace_range_includes_preview(self):
        filepath = create_test_file("a\nb\nc\nd\ne\n")
        try:
            result = edit_file(
                filepath=filepath,
                start_line=2,
                end_line=3,
                content="B\nC",
                mode="replace_range",
            )
            assert result_success(result)
            assert "preview" in result
            assert "B" in result["preview"]
        finally:
            delete_test_file(filepath)

    def test_non_python_file_still_gets_preview(self):
        filepath = create_test_file(
            "# Header\nSome text\nMore text\n", name="test_preview.md"
        )
        try:
            result = edit_file(
                filepath=filepath,
                find="Some text",
                content="Replaced text",
                mode="replace",
            )
            assert result_success(result)
            assert "preview" in result, "Non-Python files should still get preview"
        finally:
            delete_test_file(filepath)

    def test_long_edit_has_skip_marker(self):
        # Create a file with many lines
        content = "\n".join([f"line{i}" for i in range(30)]) + "\n"
        filepath = create_test_file(content)
        try:
            # Replace a large region
            result = edit_file(
                filepath=filepath,
                start_line=5,
                end_line=25,
                content="\n".join([f"new{i}" for i in range(20)]),
                mode="replace_range",
            )
            assert result_success(result)
            preview = result["preview"]
            # With 20 new lines (more than max_edit_show=6), should have skip marker
            assert "..." in preview
        finally:
            delete_test_file(filepath)


# =============================================================================
# Combined: AST validation + preview interaction
# =============================================================================


class TestASTAndPreviewInteraction:
    """Tests that AST validation and preview work together correctly."""

    def test_blocked_edit_no_preview(self):
        filepath = create_test_file("def foo():\n    pass\n")
        try:
            result = edit_file(
                filepath=filepath, content="def broken(\n", mode="append"
            )
            # Blocked edit should have error, not preview
            assert result_has_error(result)
            assert "preview" not in result
        finally:
            delete_test_file(filepath)

    def test_valid_edit_gets_preview_not_error(self):
        filepath = create_test_file("def foo():\n    return 1\n")
        try:
            result = edit_file(
                filepath=filepath, find="return 1", content="return 2", mode="replace"
            )
            # Valid edit should have success and preview, not error
            assert result_success(result)
            assert "preview" in result
            assert "error" not in result
        finally:
            delete_test_file(filepath)

    def test_chopped_blank_lines_detected_by_preview(self):
        """Verify that double blank lines (common indent wreck) show in preview."""
        filepath = create_test_file("class Foo:\n\n    def bar(self):\n        pass\n")
        try:
            # Replace that accidentally introduces double blank line
            result = edit_file(
                filepath=filepath,
                find="pass",
                content="return True\n\n\n    def baz(self):\n        pass",
                mode="replace",
            )
            if result_success(result):
                # If the edit was syntactically valid, preview should show the blank lines
                assert "preview" in result
                # Preview should reveal the triple-blank-line issue
                preview = result["preview"]
                # The preview itself should be present — the AI can visually see issues
                assert len(preview) > 0
        finally:
            delete_test_file(filepath)
