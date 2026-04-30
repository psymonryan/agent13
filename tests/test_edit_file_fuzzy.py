"""
Tests for fuzzy matching fallback in edit_file tool.

These tests verify that when an exact match fails, the tool provides
helpful error messages showing the closest fuzzy match.
"""

import os
import pytest
from tools.edit_file import edit_file
from agent13.sandbox import get_temp_dir


def create_test_file(content: str, name: str = "test_fuzzy.py") -> str:
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
    return "error" in result or result.get("success") is False


class TestFuzzyMatchBasic:
    """Basic fuzzy matching scenarios."""

    def test_single_char_difference_shows_fuzzy_match(self):
        """Missing single character should show fuzzy match suggestion."""
        filepath = create_test_file('print("Hello, world")\n', "test1.py")

        try:
            # Search for text with extra '!'
            result = edit_file(
                filepath=filepath,
                find='print("Hello, world!")',
                content='print("Goodbye!")',
            )

            assert result_has_error(result)
            error = result.get("error", "")
            assert "fuzzy" in error.lower() or "similar" in error.lower()
            assert "96" in error or "97" in error or "98" in error  # ~97% similarity
        finally:
            delete_test_file(filepath)

    def test_wrong_indentation_shows_normalized_match(self):
        """Wrong indentation should show normalized fuzzy match."""
        content = """def greet():
    print("Hello")
    return True
"""
        filepath = create_test_file(content, "test2.py")

        try:
            # Search with extra indentation
            result = edit_file(
                filepath=filepath,
                find='        print("Hello")',  # 8 spaces instead of 4
                content='    print("Goodbye")',
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should find it via normalized matching
            assert "similar" in error.lower() or "fuzzy" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_typo_shows_fuzzy_match(self):
        """Typo in search text should show fuzzy match."""
        filepath = create_test_file("def hello_world():\n    pass\n", "test3.py")

        try:
            # Search with typo (helo instead of hello)
            result = edit_file(
                filepath=filepath, find="def helo_world():", content="def goodbye():"
            )

            assert result_has_error(result)
            error = result.get("error", "")
            assert "similar" in error.lower() or "fuzzy" in error.lower()
            # Should have high similarity despite typo
            assert "9" in error  # 90%+ similarity
        finally:
            delete_test_file(filepath)

    def test_exact_match_succeeds_no_fuzzy(self):
        """Exact match should succeed without fuzzy matching."""
        filepath = create_test_file("x = 1\ny = 2\n", "test4.py")

        try:
            result = edit_file(filepath=filepath, find="x = 1", content="x = 10")

            assert result_success(result)
            message = result.get("message", "")
            assert "fuzzy" not in message.lower()

            # Verify the edit happened
            with open(filepath) as f:
                assert f.read() == "x = 10\ny = 2\n"
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchMultiLine:
    """Multi-line fuzzy matching scenarios."""

    def test_multiline_with_small_difference(self):
        """Multi-line search with small difference should show fuzzy match."""
        content = """def calculate(a, b):
    result = a + b
    return result
"""
        filepath = create_test_file(content, "test5.py")

        try:
            # Search with 'results' instead of 'result'
            result = edit_file(
                filepath=filepath,
                find="""def calculate(a, b):
    result = a + b
    return results""",
                content="def add(x, y):\n    return x + y",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            assert "similar" in error.lower() or "fuzzy" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_multiline_wrong_indentation(self):
        """Multi-line with wrong indentation should find via normalization."""
        content = """class MyClass:
    def method(self):
        x = 1
        return x
"""
        filepath = create_test_file(content, "test6.py")

        try:
            # Search with different indentation
            result = edit_file(
                filepath=filepath,
                find="""class MyClass:
  def method(self):
    x = 1
    return x""",
                content="class NewClass:\n    pass",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should still find it via normalized comparison
            assert "similar" in error.lower() or "normalized" in error.lower()
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchLineScoping:
    """Fuzzy matching with line number scoping."""

    def test_fuzzy_match_respects_start_line(self):
        """Fuzzy match should only search from start_line onwards."""
        content = """# First occurrence
target_text = "old"

# Second occurrence
target_text = "new"
"""
        filepath = create_test_file(content, "test7.py")

        try:
            # Search for typo of "old", scoped to lines 1-3
            result = edit_file(
                filepath=filepath,
                find='target_text = "odl"',  # typo
                content='target_text = "replaced"',
                start_line=1,
                end_line=3,
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should find "old" as closest match (not "new")
            assert "similar" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_fuzzy_match_respects_end_line(self):
        """Fuzzy match should only search up to end_line."""
        content = """# Header
x = 1
y = 2
z = 3
# Footer
"""
        filepath = create_test_file(content, "test8.py")

        try:
            # Search for typo, scoped to lines 2-4
            result = edit_file(
                filepath=filepath,
                find="w = 2",  # typo of y = 2
                content="y = 20",
                start_line=2,
                end_line=4,
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should find y = 2 within scope
            assert "similar" in error.lower()
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchNoMatch:
    """Cases where no fuzzy match should be found."""

    def test_completely_different_text_no_match(self):
        """Completely different text should not find fuzzy match."""
        filepath = create_test_file('print("Hello")\n', "test9.py")

        try:
            result = edit_file(
                filepath=filepath, find="def completely_different():", content="x = 1"
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should not find any similar text
            assert "not found" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_below_threshold_no_match(self):
        """Text below similarity threshold should not be suggested."""
        filepath = create_test_file("x = 123\n", "test10.py")

        try:
            # Very different search text (should be below 90% threshold)
            result = edit_file(
                filepath=filepath, find="abcdefghijklmnopqrstuvwxyz", content="y = 456"
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should not suggest because similarity is too low
            assert "similar" not in error.lower() or "no similar" in error.lower()
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchDeleteMode:
    """Fuzzy matching in delete mode."""

    def test_delete_with_fuzzy_match_suggestion(self):
        """Delete mode should show fuzzy match when text not found."""
        content = """def hello():
    print("Hello")
    return True
"""
        filepath = create_test_file(content, "test11.py")

        try:
            # Try to delete with typo
            result = edit_file(
                filepath=filepath,
                find='print("Helo")',  # typo: missing 'l'
                mode="delete",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            assert "similar" in error.lower() or "fuzzy" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_delete_exact_match_succeeds(self):
        """Delete mode with exact match should succeed."""
        filepath = create_test_file("x = 1\ny = 2\nz = 3\n", "test12.py")

        try:
            result = edit_file(filepath=filepath, find="y = 2", mode="delete")

            assert result_success(result)

            with open(filepath) as f:
                assert f.read() == "x = 1\nz = 3\n"
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchAppendPrepend:
    """Fuzzy matching in append/prepend modes."""

    def test_append_with_wrong_anchor_shows_fuzzy(self):
        """Append mode should show fuzzy match when anchor not found."""
        content = """def process():
    data = load()
    return data
"""
        filepath = create_test_file(content, "test13.py")

        try:
            # Try to append after line with typo
            result = edit_file(
                filepath=filepath,
                find="data = lod()",  # typo
                content="    # Added comment",
                mode="append",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            assert "similar" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_prepend_with_correct_anchor_succeeds(self):
        """Prepend mode with correct anchor should succeed."""
        content = """class User:
    def __init__(self):
        self.name = ""
"""
        filepath = create_test_file(content, "test14.py")

        try:
            result = edit_file(
                filepath=filepath,
                find="    def __init__(self):",
                content='    """Initialize user."""\n',
                mode="prepend",
            )

            assert result_success(result)
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchEdgeCases:
    """Edge cases for fuzzy matching."""

    def test_empty_search_text(self):
        """Empty search text should fail gracefully."""
        filepath = create_test_file("x = 1\n", "test15.py")

        try:
            result = edit_file(filepath=filepath, find="", content="y = 2")

            assert result_has_error(result)
        finally:
            delete_test_file(filepath)

    def test_single_character_difference(self):
        """Single character difference should have high similarity."""
        filepath = create_test_file("x = 1\n", "test16.py")

        try:
            result = edit_file(
                filepath=filepath,
                find="x = 2",  # only '2' is different
                content="x = 10",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should have high similarity
            assert "similar" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_whitespace_only_difference(self):
        """Whitespace-only difference should match via normalization."""
        filepath = create_test_file(
            "def  hello():\n    pass\n", "test17.py"
        )  # double space

        try:
            result = edit_file(
                filepath=filepath,
                find="def hello():",  # single space
                content="def greet():",
            )

            # Should either succeed via normalization or show high similarity
            error = result.get("error", "")
            if not result_success(result):
                assert (
                    "normalized" in error.lower()
                    or "100" in error
                    or "similar" in error.lower()
                )
        finally:
            delete_test_file(filepath)


class TestFuzzyMatchErrorMessages:
    """Quality of error messages."""

    def test_error_includes_similarity_percentage(self):
        """Error message should include similarity percentage."""
        filepath = create_test_file("hello_world = True\n", "test18.py")

        try:
            result = edit_file(
                filepath=filepath,
                find="hello_word = True",  # typo
                content="goodbye = False",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should include percentage like "95.2%"
            assert "%" in error
        finally:
            delete_test_file(filepath)

    def test_error_includes_line_numbers(self):
        """Error message should include line numbers of closest match."""
        content = """x = 1
y = 2
target_line = "value"
z = 3
"""
        filepath = create_test_file(content, "test19.py")

        try:
            result = edit_file(
                filepath=filepath,
                find='target_line = "valu"',  # typo
                content="replaced = True",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should mention line number where match was found
            assert "line" in error.lower()
        finally:
            delete_test_file(filepath)

    def test_error_includes_debugging_tips(self):
        """Error message should include debugging tips."""
        filepath = create_test_file("some_code()\n", "test20.py")

        try:
            result = edit_file(
                filepath=filepath,
                find="some_cod()",  # typo
                content="other_code()",
            )

            assert result_has_error(result)
            error = result.get("error", "")
            # Should include helpful tips
            assert "tip" in error.lower() or "check" in error.lower()
        finally:
            delete_test_file(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
