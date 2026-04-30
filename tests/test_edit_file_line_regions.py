"""Tests for Philosophy C line number behavior in append/prepend modes.

Either start_line or end_line can anchor an insertion; both can specify
a region. append inserts after the bottom edge, prepend before the top edge.
"""

import os
import pytest

from tools.edit_file import edit_file
from agent13.sandbox import get_temp_dir


@pytest.fixture
def tmp():
    """Create a temp file with numbered lines in sandbox-allowed dir."""
    path = os.path.join(get_temp_dir(), "test_regions.txt")
    content = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
    with open(path, "w") as f:
        f.write(content)
    yield path
    os.unlink(path)


class TestAppendEitherLine:
    """append mode: either start_line or end_line anchors the insertion."""

    def test_append_after_start_line(self, tmp):
        """append with start_line only inserts after that line."""
        result = edit_file(tmp, content="new line", mode="append", start_line=5)
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[4] == "line 5"
        assert lines[5] == "new line"
        assert lines[6] == "line 6"

    def test_append_after_end_line(self, tmp):
        """append with end_line only inserts after that line."""
        result = edit_file(tmp, content="new line", mode="append", end_line=5)
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[4] == "line 5"
        assert lines[5] == "new line"
        assert lines[6] == "line 6"

    def test_append_both_uses_end_line(self, tmp):
        """append with both uses end_line (bottom of region)."""
        result = edit_file(
            tmp, content="new line", mode="append", start_line=3, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        # Should be after line 5 (end_line), not after line 3 (start_line)
        assert lines[4] == "line 5"
        assert lines[5] == "new line"
        assert lines[6] == "line 6"

    def test_append_both_description_mentions_end(self, tmp):
        """append with both lines mentions end_line in description."""
        result = edit_file(
            tmp, content="new line", mode="append", start_line=3, end_line=5
        )
        assert result["success"]
        assert "5" in result["message"]


class TestPrependEitherLine:
    """prepend mode: either start_line or end_line anchors the insertion."""

    def test_prepend_before_start_line(self, tmp):
        """prepend with start_line only inserts before that line."""
        result = edit_file(tmp, content="new line", mode="prepend", start_line=5)
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[4] == "new line"
        assert lines[5] == "line 5"

    def test_prepend_before_end_line(self, tmp):
        """prepend with end_line only inserts before that line (NOT at file start)."""
        result = edit_file(tmp, content="new line", mode="prepend", end_line=5)
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        # Must NOT be at start of file — must be before line 5
        assert lines[0] == "line 1"
        assert lines[4] == "new line"
        assert lines[5] == "line 5"

    def test_prepend_both_uses_start_line(self, tmp):
        """prepend with both uses start_line (top of region)."""
        result = edit_file(
            tmp, content="new line", mode="prepend", start_line=3, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        # Should be before line 3 (start_line), not before line 5 (end_line)
        assert lines[2] == "new line"
        assert lines[3] == "line 3"

    def test_prepend_both_description_mentions_start(self, tmp):
        """prepend with both lines mentions start_line in description."""
        result = edit_file(
            tmp, content="new line", mode="prepend", start_line=3, end_line=5
        )
        assert result["success"]
        assert "3" in result["message"]


class TestRegionSemantics:
    """start_line/end_line define a region; append/prepend choose an edge."""

    def test_append_after_region_bottom(self, tmp):
        """append with region 3-5 inserts after line 5."""
        result = edit_file(
            tmp, content="after region", mode="append", start_line=3, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[5] == "after region"
        assert lines[6] == "line 6"

    def test_prepend_before_region_top(self, tmp):
        """prepend with region 3-5 inserts before line 3."""
        result = edit_file(
            tmp, content="before region", mode="prepend", start_line=3, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[2] == "before region"
        assert lines[3] == "line 3"

    def test_single_line_region_append(self, tmp):
        """append with start_line==end_line inserts after that line."""
        result = edit_file(
            tmp, content="after line 5", mode="append", start_line=5, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[4] == "line 5"
        assert lines[5] == "after line 5"

    def test_single_line_region_prepend(self, tmp):
        """prepend with start_line==end_line inserts before that line."""
        result = edit_file(
            tmp, content="before line 5", mode="prepend", start_line=5, end_line=5
        )
        assert result["success"]
        with open(tmp) as f:
            lines = f.read().splitlines()
        assert lines[4] == "before line 5"
        assert lines[5] == "line 5"
