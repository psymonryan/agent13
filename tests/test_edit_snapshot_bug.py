"""Test that demonstrates phantom snapshot bug in edit_file.

When editing a Python file with invalid syntax:
1. Snapshot is saved BEFORE syntax validation
2. Syntax validation fails
3. File is NOT written
4. But snapshot is already in history

This creates "phantom snapshots" that were never actually applied.
"""

import os
import tempfile
from pathlib import Path
import pytest
from tools.edit_file import edit_file, _snapshots, _snapshot_counter


def resolved(path: str) -> str:
    """Resolve a path the same way edit_file does — for snapshot key lookups."""
    return str(Path(path).expanduser().resolve())


@pytest.fixture(autouse=True)
def clear_snapshots():
    """Clear snapshot state between tests."""
    _snapshots.clear()
    _snapshot_counter.clear()
    yield
    _snapshots.clear()
    _snapshot_counter.clear()


@pytest.fixture
def python_file():
    """Create a temporary Python file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def hello():\n    print('hello')\n")
        filepath = f.name
    yield filepath
    os.unlink(filepath)


class TestPhantomSnapshotBug:
    """Test that failed edits don't create phantom snapshots."""

    def test_failed_syntax_validation_creates_phantom_snapshot(self, python_file):
        """BUG: Failed edit still saves a snapshot, even though file wasn't modified."""
        # Read original content
        with open(python_file, 'r') as f:
            original_content = f.read()

        # Edit that will fail syntax validation (invalid Python)
        result = edit_file(
            filepath=python_file,
            find="def hello():",
            content="def hello(:",  # Invalid syntax - missing closing paren
            mode="replace"
        )

        # Edit should fail
        assert result["success"] is False
        assert "syntax" in result.get("error", "").lower() or "invalid" in result.get("error", "").lower()

        # File should NOT be modified
        with open(python_file, 'r') as f:
            current_content = f.read()
        assert current_content == original_content

        # BUG: Snapshot was still saved even though edit failed!
        # There should be NO snapshots since the edit never actually happened
        assert resolved(python_file) not in _snapshots or len(_snapshots.get(resolved(python_file), {})) == 0

    def test_successful_edit_saves_snapshot(self, python_file):
        """Verify that successful edits DO save snapshots (correct behavior)."""
        result = edit_file(
            filepath=python_file,
            find="def hello():",
            content="def hello_world():",
            mode="replace"
        )

        assert result["success"] is True

        # Snapshot should exist
        assert resolved(python_file) in _snapshots
        assert len(_snapshots[resolved(python_file)]) == 1

    def test_rollback_after_failed_edit_no_phantom_snapshot(self, python_file):
        """FIXED: Failed edit does NOT create phantom snapshot."""
        # First, make a successful edit
        result1 = edit_file(
            filepath=python_file,
            find="def hello():",
            content="def hello_world():",
            mode="replace"
        )
        assert result1["success"] is True
        snapshot_after_first_edit = result1["snapshot_id"]

        # Now attempt an edit that fails syntax validation
        result2 = edit_file(
            filepath=python_file,
            find="def hello_world():",
            content="def invalid(:",  # Invalid syntax
            mode="replace"
        )
        assert result2["success"] is False

        # FIXED: Only 1 snapshot exists (the successful edit)
        assert len(_snapshots[resolved(python_file)]) == 1

        # Rollback should restore to the state before the first edit
        result3 = edit_file(
            filepath=python_file,
            mode="rollback",
            snapshot_id=snapshot_after_first_edit
        )
        assert result3["success"] is True

        # File should be restored to original state
        with open(python_file, 'r') as f:
            content = f.read()
        assert "def hello():" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
