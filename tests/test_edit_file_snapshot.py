"""
Tests for snapshot/rollback in edit_file tool.

These tests verify:
1. Every edit result includes a snapshot_id
2. Sequential edits produce incrementing snapshot IDs
3. Bare rollback (mode="rollback") undoes the most recent edit
4. rollback with snapshot_id=-1 is same as bare rollback
5. rollback with explicit snapshot_id restores that specific snapshot
6. Undo is undoable - rolling back snapshots current state first
7. FIFO cap evicts oldest snapshots but IDs remain stable
8. Error: no snapshots for a file
9. Error: snapshot_id not found (evicted or never existed)
10. Error: invalid mode string rejected
"""

import os
from tools.edit_file import (
    edit_file,
    _snapshots,
    _snapshot_counter,
    MAX_SNAPSHOTS_PER_FILE,
)
from agent13.sandbox import get_temp_dir


def create_test_file(content: str, name: str = "test_snap.py") -> str:
    """Create a temp test file in sandbox-allowed dir and return its path."""
    filepath = os.path.join(get_temp_dir(), name)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def reset_snapshots():
    """Clear all snapshot state between tests."""
    _snapshots.clear()
    _snapshot_counter.clear()


def cleanup_test_file(filepath: str):
    """Remove test file if it exists."""
    if os.path.exists(filepath):
        os.unlink(filepath)


class TestSnapshotInResults:
    """Every edit result includes a snapshot_id."""

    def test_replace_includes_snapshot_id(self):
        fp = create_test_file("hello\nworld\n", "snap_1.py")
        reset_snapshots()
        try:
            r = edit_file(fp, find="hello", content="HELLO")
            assert r["success"] is True
            assert "snapshot_id" in r
            assert r["snapshot_id"] == 0
        finally:
            cleanup_test_file(fp)

    def test_append_includes_snapshot_id(self):
        fp = create_test_file("hello\n", "snap_2.py")
        reset_snapshots()
        try:
            r = edit_file(fp, mode="append", content="world")
            assert r["success"] is True
            assert "snapshot_id" in r
        finally:
            cleanup_test_file(fp)

    def test_delete_includes_snapshot_id(self):
        fp = create_test_file("hello\nworld\n", "snap_3.py")
        reset_snapshots()
        try:
            r = edit_file(fp, find="hello", mode="delete")
            assert r["success"] is True
            assert "snapshot_id" in r
        finally:
            cleanup_test_file(fp)

    def test_sequential_edits_increment_snapshot_id(self):
        fp = create_test_file("a\nb\nc\n", "snap_4.py")
        reset_snapshots()
        try:
            r1 = edit_file(fp, find="a", content="A")
            r2 = edit_file(fp, find="b", content="B")
            assert r1["snapshot_id"] == 0
            assert r2["snapshot_id"] == 1
        finally:
            cleanup_test_file(fp)


class TestBareRollback:
    """mode='rollback' undoes the most recent edit."""

    def test_rollback_restores_previous_content(self):
        fp = create_test_file("original\n", "snap_5.py")
        reset_snapshots()
        try:
            edit_file(fp, find="original", content="modified")
            r = edit_file(fp, mode="rollback")
            assert r["success"] is True
            with open(fp) as f:
                assert "original" in f.read()
        finally:
            cleanup_test_file(fp)

    def test_rollback_returns_snapshot_id(self):
        fp = create_test_file("x\n", "snap_6.py")
        reset_snapshots()
        try:
            edit_file(fp, find="x", content="y")
            r = edit_file(fp, mode="rollback")
            assert r["success"] is True
            assert "snapshot_id" in r
        finally:
            cleanup_test_file(fp)

    def test_snapshot_id_minus_one_same_as_bare(self):
        """snapshot_id=-1 resolves to the latest snapshot, same as bare rollback."""
        fp = create_test_file("x\n", "snap_7.py")
        reset_snapshots()
        try:
            edit_file(fp, find="x", content="y")
            # Both bare and -1 should restore to the same state
            r1 = edit_file(fp, mode="rollback")
            assert r1["success"] is True
            with open(fp) as f:
                after_bare = f.read()
            # Undo the undo to get back to "y" state
            edit_file(fp, mode="rollback")
            with open(fp) as f:
                assert "y" in f.read()
            # Now use snapshot_id=-1 to undo again
            r2 = edit_file(fp, mode="rollback", snapshot_id=-1)
            assert r2["success"] is True
            with open(fp) as f:
                after_minus_one = f.read()
            # Both bare and -1 restored to the previous state
            assert after_bare == after_minus_one
        finally:
            cleanup_test_file(fp)


class TestExplicitRollback:
    """rollback with explicit snapshot_id restores that specific snapshot."""

    def test_rollback_to_snapshot_zero(self):
        fp = create_test_file("v0\n", "snap_8.py")
        reset_snapshots()
        try:
            edit_file(fp, find="v0", content="v1")
            edit_file(fp, find="v1", content="v2")
            r = edit_file(fp, mode="rollback", snapshot_id=0)
            assert r["success"] is True
            with open(fp) as f:
                assert "v0" in f.read()
        finally:
            cleanup_test_file(fp)

    def test_rollback_to_middle_snapshot(self):
        fp = create_test_file("v0\n", "snap_9.py")
        reset_snapshots()
        try:
            edit_file(fp, find="v0", content="v1")
            edit_file(fp, find="v1", content="v2")
            edit_file(fp, find="v2", content="v3")
            r = edit_file(fp, mode="rollback", snapshot_id=1)
            assert r["success"] is True
            with open(fp) as f:
                assert "v1" in f.read()
        finally:
            cleanup_test_file(fp)


class TestUndoIsUndoable:
    """Rollback snapshots current state before restoring, enabling undo-the-undo."""

    def test_rollback_then_rollback_changes_file(self):
        fp = create_test_file("a\n", "snap_10.py")
        reset_snapshots()
        try:
            edit_file(fp, find="a", content="b")
            # Rollback: b -> a
            edit_file(fp, mode="rollback")
            with open(fp) as f:
                after_first = f.read()
            assert "a" in after_first
            # Undo the undo: a -> b
            r = edit_file(fp, mode="rollback")
            assert r["success"] is True
            with open(fp) as f:
                after_second = f.read()
            assert "b" in after_second
        finally:
            cleanup_test_file(fp)


class TestFIFOEviction:
    """FIFO cap evicts oldest but IDs remain stable."""

    def test_eviction_removes_oldest_snapshots(self):
        fp = create_test_file("v0\n", "snap_11.py")
        reset_snapshots()
        try:
            for i in range(MAX_SNAPSHOTS_PER_FILE + 2):
                edit_file(fp, find=f"v{i}", content=f"v{i + 1}")
            # After 12 edits with cap=10, snapshots 0 and 1 should be evicted
            keys = sorted(_snapshots[fp].keys())
            assert 0 not in keys
            assert 1 not in keys
            assert len(keys) <= MAX_SNAPSHOTS_PER_FILE
        finally:
            cleanup_test_file(fp)

    def test_ids_stable_after_eviction(self):
        """Snapshot IDs don't shift when older ones are evicted - rollback still works."""
        fp = create_test_file("v0\n", "snap_12.py")
        reset_snapshots()
        try:
            for i in range(MAX_SNAPSHOTS_PER_FILE + 2):
                edit_file(fp, find=f"v{i}", content=f"v{i + 1}")
            # Snapshot 2 should still contain "v2"
            r = edit_file(fp, mode="rollback", snapshot_id=2)
            assert r["success"] is True
            with open(fp) as f:
                assert "v2" in f.read()
        finally:
            cleanup_test_file(fp)


class TestRollbackErrors:
    """Error handling for rollback mode."""

    def test_no_snapshots_for_file(self):
        fp = create_test_file("x\n", "snap_13.py")
        reset_snapshots()
        try:
            r = edit_file(fp, mode="rollback")
            assert r["success"] is False
            assert "No snapshots" in r["error"]
        finally:
            cleanup_test_file(fp)

    def test_snapshot_id_not_found(self):
        fp = create_test_file("x\n", "snap_14.py")
        reset_snapshots()
        try:
            edit_file(fp, find="x", content="y")
            r = edit_file(fp, mode="rollback", snapshot_id=999)
            assert r["success"] is False
            assert "not found" in r["error"]
        finally:
            cleanup_test_file(fp)

    def test_invalid_mode_with_colon(self):
        """The 'rollback:N' colon syntax should be rejected as invalid mode."""
        fp = create_test_file("x\n", "snap_15.py")
        reset_snapshots()
        try:
            r = edit_file(fp, mode="rollback:3")
            assert "error" in r
            assert "Invalid mode" in r["error"]
        finally:
            cleanup_test_file(fp)

    def test_evicted_snapshot_gives_not_found_error(self):
        fp = create_test_file("v0\n", "snap_16.py")
        reset_snapshots()
        try:
            for i in range(MAX_SNAPSHOTS_PER_FILE + 2):
                edit_file(fp, find=f"v{i}", content=f"v{i + 1}")
            # Snapshot 0 was evicted
            r = edit_file(fp, mode="rollback", snapshot_id=0)
            assert r["success"] is False
            assert "not found" in r["error"]
        finally:
            cleanup_test_file(fp)
