"""Tests for snapshot path consistency and write_file snapshots.

Tests three fixes:
1. Path normalization (.resolve()) — edit with relative path, rollback with absolute
2. write_file overwrite creates snapshots for rollback
3. Cross-tool rollback — write_file overwrites, edit_file rolls back
"""

import os
from pathlib import Path
import pytest
from tools.edit_file import edit_file, _snapshots, _snapshot_counter
from tools.write_file import write_file
from agent13.sandbox import get_temp_dir


def resolved(path: str) -> str:
    """Resolve a path the same way edit_file/write_file does — for snapshot key lookups."""
    return str(Path(path).expanduser().resolve())


@pytest.fixture(autouse=True)
def clear_snapshots():
    """Clear snapshot state between tests."""
    _snapshots.clear()
    _snapshot_counter.clear()
    yield
    _snapshots.clear()
    _snapshot_counter.clear()


def create_test_file(content: str, name: str = "test_path.py") -> str:
    """Create a temp test file in sandbox-allowed dir and return its absolute path."""
    filepath = os.path.join(get_temp_dir(), name)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def cleanup_test_file(filepath: str):
    """Remove test file if it exists."""
    if os.path.exists(filepath):
        os.unlink(filepath)


class TestPathNormalization:
    """edit_file normalizes paths via .resolve() so relative and absolute
    paths share the same snapshot key."""

    def test_edit_relative_rollback_absolute(self):
        """Edit with relative path, rollback with absolute path — same snapshot."""
        abs_path = create_test_file("hello\n", "path_rel.py")
        rel_path = os.path.basename(abs_path)
        reset_cwd = os.getcwd()
        os.chdir(get_temp_dir())
        try:
            r = edit_file(rel_path, find="hello", content="world")
            assert r["success"] is True
            assert r["snapshot_id"] == 0
        finally:
            os.chdir(reset_cwd)

        # Rollback using absolute path — should find the snapshot
        r2 = edit_file(abs_path, mode="rollback")
        assert r2["success"] is True
        with open(abs_path) as f:
            assert "hello" in f.read()
        cleanup_test_file(abs_path)

    def test_edit_absolute_rollback_relative(self):
        """Edit with absolute path, rollback with relative path — same snapshot."""
        abs_path = create_test_file("foo\n", "path_abs.py")
        rel_path = os.path.basename(abs_path)

        r = edit_file(abs_path, find="foo", content="bar")
        assert r["success"] is True

        reset_cwd = os.getcwd()
        os.chdir(get_temp_dir())
        try:
            r2 = edit_file(rel_path, mode="rollback")
            assert r2["success"] is True
        finally:
            os.chdir(reset_cwd)

        with open(abs_path) as f:
            assert "foo" in f.read()
        cleanup_test_file(abs_path)

    def test_snapshot_key_is_resolved(self):
        """Internal: snapshot dict key is the resolved absolute path."""
        abs_path = create_test_file("x\n", "path_key.py")
        rel_path = os.path.basename(abs_path)
        reset_cwd = os.getcwd()
        os.chdir(get_temp_dir())
        try:
            edit_file(rel_path, find="x", content="y")
        finally:
            os.chdir(reset_cwd)

        # The snapshot should be stored under the resolved absolute path
        assert resolved(abs_path) in _snapshots
        assert rel_path not in _snapshots
        cleanup_test_file(abs_path)

    def test_mixed_paths_share_snapshot_history(self):
        """Multiple edits using mixed path formats share one snapshot history."""
        abs_path = create_test_file("v0\n", "path_mixed.py")
        rel_path = os.path.basename(abs_path)

        # Edit 1: absolute path
        edit_file(abs_path, find="v0", content="v1")
        # Edit 2: relative path
        reset_cwd = os.getcwd()
        os.chdir(get_temp_dir())
        try:
            edit_file(rel_path, find="v1", content="v2")
        finally:
            os.chdir(reset_cwd)

        # Both edits should be in the same snapshot history
        assert len(_snapshots[resolved(abs_path)]) == 2

        # Rollback to snapshot 0 (before edit 1) using absolute path
        r = edit_file(abs_path, mode="rollback", snapshot_id=0)
        assert r["success"] is True
        with open(abs_path) as f:
            assert "v0" in f.read()
        cleanup_test_file(abs_path)


class TestWriteFileSnapshots:
    """write_file with overwrite=True creates a snapshot before overwriting."""

    def test_overwrite_creates_snapshot(self):
        """Overwriting a file with write_file saves a snapshot."""
        fp = create_test_file("original\n", "wf_snap1.py")
        try:
            r = write_file(fp, "replaced\n", overwrite=True)
            assert r["success"] is True
            assert "snapshot_id" in r
            assert r["snapshot_id"] == 0
        finally:
            cleanup_test_file(fp)

    def test_create_does_not_create_snapshot(self):
        """Creating a new file (no overwrite) does NOT create a snapshot."""
        fp = os.path.join(get_temp_dir(), "wf_snap2.py")
        try:
            r = write_file(fp, "new file\n")
            assert r["success"] is True
            assert "snapshot_id" not in r
            assert fp not in _snapshots or len(_snapshots.get(fp, {})) == 0
        finally:
            cleanup_test_file(fp)

    def test_overwrite_rollback_restores_original(self):
        """Rollback after write_file overwrite restores original content."""
        fp = create_test_file("original content\n", "wf_snap3.py")
        try:
            # Overwrite with write_file
            write_file(fp, "new content\n", overwrite=True)
            assert "new content" in open(fp).read()

            # Rollback via edit_file
            r = edit_file(fp, mode="rollback")
            assert r["success"] is True
            assert "original content" in open(fp).read()
        finally:
            cleanup_test_file(fp)

    def test_write_then_edit_shared_snapshot_history(self):
        """write_file overwrite and edit_file share the same snapshot history."""
        fp = create_test_file("v0\n", "wf_snap4.py")
        try:
            # write_file overwrite → snapshot 0
            r1 = write_file(fp, "v1\n", overwrite=True)
            assert r1["snapshot_id"] == 0

            # edit_file replace → snapshot 1
            r2 = edit_file(fp, find="v1", content="v2")
            assert r2["snapshot_id"] == 1

            # Rollback to snapshot 0 → should restore "v0" (pre-write_file)
            r3 = edit_file(fp, mode="rollback", snapshot_id=0)
            assert r3["success"] is True
            assert "v0" in open(fp).read()

            # Rollback (undo the undo) → should restore "v2"
            r4 = edit_file(fp, mode="rollback")
            assert r4["success"] is True
            assert "v2" in open(fp).read()
        finally:
            cleanup_test_file(fp)

    def test_write_file_overwrite_path_normalization(self):
        """write_file overwrite with relative path, rollback with absolute path."""
        abs_path = create_test_file("original\n", "wf_snap5.py")
        rel_path = os.path.basename(abs_path)
        reset_cwd = os.getcwd()
        os.chdir(get_temp_dir())
        try:
            write_file(rel_path, "overwritten\n", overwrite=True)
        finally:
            os.chdir(reset_cwd)

        # Rollback using absolute path
        r = edit_file(abs_path, mode="rollback")
        assert r["success"] is True
        assert "original" in open(abs_path).read()
        cleanup_test_file(abs_path)

    def test_multiple_overwrites_increment_snapshot_id(self):
        """Multiple write_file overwrites produce incrementing snapshot IDs."""
        fp = create_test_file("v0\n", "wf_snap6.py")
        try:
            r1 = write_file(fp, "v1\n", overwrite=True)
            assert r1["snapshot_id"] == 0
            r2 = write_file(fp, "v2\n", overwrite=True)
            assert r2["snapshot_id"] == 1
            r3 = write_file(fp, "v3\n", overwrite=True)
            assert r3["snapshot_id"] == 2

            # Rollback to snapshot 1 → "v1"
            r = edit_file(fp, mode="rollback", snapshot_id=1)
            assert r["success"] is True
            assert "v1" in open(fp).read()
        finally:
            cleanup_test_file(fp)


class TestRollbackWriteFailure:
    """Rollback write failure doesn't corrupt snapshot state."""

    def test_rollback_write_failure_cleans_up_snapshot(self):
        """If write fails during rollback, the undo snapshot is removed."""
        fp = create_test_file("original\n", "wf_fail1.py")
        try:
            # Make an edit to create a snapshot
            edit_file(fp, find="original", content="modified")

            # Make the file read-only to force write failure
            # (rollback takes a snapshot of current state, then tries to write)
            os.chmod(fp, 0o444)

            r = edit_file(fp, mode="rollback")
            assert r["success"] is False
            assert "Failed to write" in r["error"]

            # Restore permissions and verify file is still "modified"
            os.chmod(fp, 0o644)
            assert "modified" in open(fp).read()
        finally:
            os.chmod(fp, 0o644)
            cleanup_test_file(fp)
