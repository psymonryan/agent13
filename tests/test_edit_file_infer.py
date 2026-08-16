"""Tests for greedy anchor inference — replace mode with no 'find' parameter.

When the model provides only 'content' (no 'find'), the tool tries to locate
the old text by greedily matching from both ends of content, growing char by
char until each anchor uniquely matches in the file.
"""

import tempfile
import os

from tools.edit_file import (
    edit_file,
    _greedy_anchor_match,
    _find_all_substr,
)


def create_test_file(content: str, name: str = "test_infer.txt") -> str:
    fd, path = tempfile.mkstemp(suffix=f"_{name}")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def delete_test_file(filepath: str):
    if os.path.exists(filepath):
        os.unlink(filepath)


# =============================================================================
# Unit tests for helper functions
# =============================================================================


class TestFindAllSubstr:
    def test_single_match(self):
        assert _find_all_substr("hello world", "world", 0, 11) == [6]

    def test_multiple_matches(self):
        assert _find_all_substr("aaa", "a", 0, 3) == [0, 1, 2]

    def test_no_match(self):
        assert _find_all_substr("hello", "x", 0, 5) == []

    def test_respects_start_end(self):
        assert _find_all_substr("abab", "a", 1, 4) == [2]


class TestGreedyAnchorMatch:
    def test_basic_unique_match(self):
        text = "hello cruel world"
        content = "hello new world"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        start, end, err = result
        assert err == ""
        assert start == 0
        assert end == len(text)

    def test_ambiguous_start_grows_to_unique(self):
        # "import" appears twice but "import os" is unique;
        # end "sys" also exists in file so end anchor finds it
        text = "import os\nimport sys"
        content = "import os\nimport sys"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        start, end, err = result
        assert err == ""
        assert start == 0
        assert end == len(text)

    def test_start_not_found(self):
        text = "abc"
        content = "xyz"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        _, _, err = result
        assert "not found" in err

    def test_end_not_found_after_start(self):
        text = "hello world"
        content = "hello xyzzy"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        _, _, err = result
        assert "end of content not found" in err

    def test_ambiguous_both_directions(self):
        text = "aba"
        content = "a"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        _, _, err = result
        assert "ambiguous" in err

    def test_empty_content(self):
        result = _greedy_anchor_match("hello", "")
        assert result is not None
        _, _, err = result
        assert "empty" in err

    def test_search_region_constraint(self):
        text = "aaa bbb aaa"
        content = "aaa"
        # Search only in the "bbb aaa" region
        result = _greedy_anchor_match(text, content, 4, 11)
        assert result is not None
        start, end, err = result
        assert err == ""
        assert start == 8  # second "aaa"

    def test_single_char_unique(self):
        text = "abc"
        content = "aXc"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        start, end, err = result
        assert err == ""
        assert start == 0
        assert end == 3

    def test_preserves_unchanged_boundaries_replaces_middle(self):
        text = "prefix MIDDLE suffix"
        content = "prefix CHANGED suffix"
        result = _greedy_anchor_match(text, content)
        assert result is not None
        start, end, err = result
        assert err == ""
        # Should span the entire text
        assert text[start:end] == "prefix MIDDLE suffix"


# =============================================================================
# Integration tests via edit_file tool
# =============================================================================


class TestInferReplaceBasic:
    def test_middle_change_succeeds(self):
        path = create_test_file(
            "def hello():\n    print('hello')\n    return None\n\ndef world():\n    pass\n"
        )
        try:
            result = edit_file(
                path,
                content="def hello():\n    print('hi')\n    return None",
            )
            assert result["success"] is True
            assert result["inferred"] is True
            assert result["replacements"] == 1
            with open(path) as f:
                assert "print('hi')" in f.read()
        finally:
            delete_test_file(path)

    def test_single_line_change(self):
        path = create_test_file("x = 1\ny = 2\nz = 3\n")
        try:
            result = edit_file(path, content="y = 42")
            assert result["success"] is True
            assert result["inferred"] is True
            with open(path) as f:
                content = f.read()
            assert "y = 42" in content
            assert "x = 1" in content
            assert "z = 3" in content
        finally:
            delete_test_file(path)

    def test_expanding_block(self):
        path = create_test_file("def foo():\n    return 1\n\ndef bar():\n    pass\n")
        try:
            result = edit_file(
                path,
                content="def foo():\n    x = 1\n    return x\n\ndef bar():",
            )
            assert result["success"] is True
            assert result["inferred"] is True
            with open(path) as f:
                content = f.read()
            assert "x = 1" in content
            assert "return x" in content
            assert "def bar():" in content
        finally:
            delete_test_file(path)

    def test_shrinking_block(self):
        path = create_test_file(
            "def foo():\n    x = 1\n    y = 2\n    return x\n\ndef bar():\n    pass\n"
        )
        try:
            result = edit_file(
                path,
                content="def foo():\n    return x\n\ndef bar():",
            )
            assert result["success"] is True
            assert result["inferred"] is True
            with open(path) as f:
                content = f.read()
            assert "y = 2" not in content
            assert "def bar():" in content
        finally:
            delete_test_file(path)

    def test_includes_original_lines(self):
        path = create_test_file("hello world\nfoo bar\n")
        try:
            result = edit_file(path, content="hello earth\nfoo bar")
            assert result["success"] is True
            assert "original_lines" in result
            assert "hello world" in result["original_lines"]
        finally:
            delete_test_file(path)

    def test_includes_preview(self):
        path = create_test_file("line1\nold middle\nline3\n")
        try:
            result = edit_file(path, content="line1\nnew middle\nline3")
            assert result["success"] is True
            assert "preview" in result
            assert "new middle" in result["preview"]
        finally:
            delete_test_file(path)


class TestInferReplaceFailures:
    def test_content_not_in_file(self):
        path = create_test_file("aaa bbb ccc\n")
        try:
            result = edit_file(path, content="zzz yyy xxx")
            assert result["success"] is False
            assert "find" in result["error"].lower()
        finally:
            delete_test_file(path)

    def test_ambiguous_short_content(self):
        path = create_test_file("aba\n")
        try:
            result = edit_file(path, content="a")
            assert result["success"] is False
            assert "ambiguous" in result["error"]
        finally:
            delete_test_file(path)

    def test_changed_boundary_fails_gracefully(self):
        """When model changes the last char to something not in file,
        end anchor can't be found and inference fails gracefully."""
        path = create_test_file("hello world\n")
        try:
            # 'z' at end of content doesn't exist anywhere in file
            result = edit_file(path, content="hello worldz")
            assert result["success"] is False
            assert "find" in result["error"].lower()
        finally:
            delete_test_file(path)

    def test_error_message_is_actionable(self):
        path = create_test_file("hello\n")
        try:
            result = edit_file(path, content="xyz")
            assert result["success"] is False
            err = result["error"]
            assert "find" in err
            assert "content" in err
        finally:
            delete_test_file(path)


class TestInferReplaceWithLineRange:
    def test_line_range_disambiguates(self):
        # "foo ... bar" appears twice; scope to first line to pick the right one
        path = create_test_file("foo ABC bar\nfoo XYZ bar\n")
        try:
            result = edit_file(
                path, content="foo DEF bar", start_line=1, end_line=1
            )
            assert result["success"] is True
            assert result["inferred"] is True
            with open(path) as f:
                lines = f.read().splitlines()
            assert lines[0] == "foo DEF bar"
            assert lines[1] == "foo XYZ bar"
        finally:
            delete_test_file(path)


class TestInferReplaceSnapshotRollback:
    def test_inferred_edit_can_be_rolled_back(self):
        path = create_test_file("def foo():\n    return 1\n\ndef bar():\n    pass\n")
        original = open(path).read()
        try:
            result = edit_file(
                path, content="def foo():\n    return 42\n\ndef bar():"
            )
            assert result["success"] is True
            assert "snapshot_id" in result
            rb = edit_file(path, mode="rollback")
            assert rb["success"] is True
            assert open(path).read() == original
        finally:
            delete_test_file(path)


class TestInferDoesNotBreakNormalReplace:
    def test_normal_find_content_still_works(self):
        path = create_test_file("hello world\n")
        try:
            result = edit_file(path, find="hello", content="goodbye")
            assert result["success"] is True
            assert "inferred" not in result
            with open(path) as f:
                assert f.read() == "goodbye world\n"
        finally:
            delete_test_file(path)

    def test_both_missing_still_errors(self):
        path = create_test_file("hello\n")
        try:
            result = edit_file(path, mode="replace")
            assert "error" in result
            assert "requires" in result["error"]
        finally:
            delete_test_file(path)

    def test_find_without_content_still_errors(self):
        path = create_test_file("hello\n")
        try:
            result = edit_file(path, find="hello", mode="replace")
            assert "error" in result
            assert "content" in result["error"]
        finally:
            delete_test_file(path)
