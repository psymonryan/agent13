"""Tests for tools.security module."""

import tempfile
from pathlib import Path

from tools.security import (
    set_session_sandbox_mode,
    get_session_sandbox_mode,
    get_current_capabilities,
    validate_path_for_read,
    validate_path_for_write,
)
from agent13.sandbox import SandboxMode, SANDBOX_CAPABILITIES


class TestSessionSandboxMode:
    """Tests for session sandbox mode management."""

    def setup_method(self):
        """Reset session mode before each test."""
        set_session_sandbox_mode(None)

    def teardown_method(self):
        """Reset session mode after each test."""
        set_session_sandbox_mode(None)

    def test_set_and_get_session_mode(self):
        """Should set and get session sandbox mode."""
        set_session_sandbox_mode(SandboxMode.RESTRICTIVE_OPEN)
        assert get_session_sandbox_mode() == SandboxMode.RESTRICTIVE_OPEN

    def test_reset_session_mode(self):
        """Should be able to reset session mode to None."""
        set_session_sandbox_mode(SandboxMode.PERMISSIVE_CLOSED)
        assert get_session_sandbox_mode() == SandboxMode.PERMISSIVE_CLOSED

        set_session_sandbox_mode(None)
        assert get_session_sandbox_mode() is None


class TestValidatePathForRead:
    """Tests for validate_path_for_read function."""

    def setup_method(self):
        """Reset session mode before each test."""
        set_session_sandbox_mode(None)

    def teardown_method(self):
        """Reset session mode after each test."""
        set_session_sandbox_mode(None)

    def test_path_traversal_blocked(self):
        """Path traversal should always be blocked."""
        is_valid, error = validate_path_for_read("../secret.txt")
        assert is_valid is False
        assert "Path traversal" in error

    def test_path_traversal_various_patterns(self):
        """Various path traversal patterns should be blocked."""
        patterns = [
            "../secret.txt",
            "foo/../../secret.txt",
            "./../secret.txt",
        ]
        for pattern in patterns:
            is_valid, error = validate_path_for_read(pattern)
            assert is_valid is False, f"Should block: {pattern}"
            assert "Path traversal" in error

    def test_project_path_allowed_in_restrictive_mode(self):
        """Paths within project should be allowed in restrictive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello")

            set_session_sandbox_mode(SandboxMode.RESTRICTIVE_CLOSED)

            # Use an explicit cwd
            is_valid, error = validate_path_for_read("test.txt", cwd=Path(tmpdir))
            assert is_valid is True
            assert error == ""

    def test_outside_project_blocked_in_restrictive_mode(self):
        """Paths outside project should be blocked in restrictive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.RESTRICTIVE_CLOSED)

            # Try to read from outside the cwd (use /etc/passwd which is definitely outside)
            is_valid, error = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
            assert is_valid is False
            assert "Read access denied" in error
            assert "restrictive-closed" in error

    def test_outside_project_allowed_in_none_mode(self):
        """Paths outside project should be allowed in none mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            other_dir = Path(tmpdir) / "other"
            other_dir.mkdir()

            set_session_sandbox_mode(SandboxMode.NONE)

            # In none mode, outside paths are allowed (but traversal is still blocked)
            is_valid, error = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
            assert is_valid is True
            assert error == ""

    def test_error_message_includes_mode_info(self):
        """Error message should include mode info for user guidance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.RESTRICTIVE_OPEN)

            is_valid, error = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
            assert is_valid is False
            assert "restrictive-open" in error
            # Error message format changed - no longer includes file_read=project
            assert "/sandbox none" in error


class TestValidatePathForWrite:
    """Tests for validate_path_for_write function."""

    def setup_method(self):
        """Reset session mode before each test."""
        set_session_sandbox_mode(None)

    def teardown_method(self):
        """Reset session mode after each test."""
        set_session_sandbox_mode(None)

    def test_path_traversal_blocked(self):
        """Path traversal should always be blocked."""
        is_valid, error = validate_path_for_write("../secret.txt")
        assert is_valid is False
        assert "Path traversal" in error

    def test_project_path_allowed_in_default_mode(self):
        """Paths within project should be allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

            is_valid, error = validate_path_for_write("test.txt", cwd=Path(tmpdir))
            assert is_valid is True
            assert error == ""

    def test_outside_project_blocked_in_default_mode(self):
        """Paths outside project and allowed directories should be blocked for write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

            # Try to write to a path NOT in allowed list (/etc is not allowed)
            is_valid, error = validate_path_for_write(
                "/etc/secret.txt", cwd=Path(tmpdir)
            )
            assert is_valid is False
            assert "Write access denied" in error

    def test_outside_project_allowed_in_none_mode(self):
        """Paths outside project should be allowed for write in none mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.NONE)

            is_valid, error = validate_path_for_write("/tmp/test.txt", cwd=Path(tmpdir))
            assert is_valid is True
            assert error == ""

    def test_write_blocked_outside_allowed_paths(self):
        """Write to paths not in allowed list should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in [
                SandboxMode.PERMISSIVE_OPEN,
                SandboxMode.PERMISSIVE_CLOSED,
                SandboxMode.RESTRICTIVE_OPEN,
                SandboxMode.RESTRICTIVE_CLOSED,
            ]:
                set_session_sandbox_mode(mode)
                # /etc is NOT in the allowed write paths for any mode
                is_valid, error = validate_path_for_write(
                    "/etc/test.txt", cwd=Path(tmpdir)
                )
                assert is_valid is False, f"Write should be blocked in {mode.value}"
                assert "Write access denied" in error


class TestGetCurrentCapabilities:
    """Tests for get_current_capabilities function."""

    def setup_method(self):
        """Reset session mode before each test."""
        set_session_sandbox_mode(None)

    def teardown_method(self):
        """Reset session mode after each test."""
        set_session_sandbox_mode(None)

    def test_returns_correct_capabilities_for_mode(self):
        """Should return correct capabilities for each mode."""
        for mode in SandboxMode:
            set_session_sandbox_mode(mode)
            caps = get_current_capabilities()
            expected = SANDBOX_CAPABILITIES[mode]
            assert caps == expected


class TestIntegrationWithSandboxModes:
    """Integration tests for security with different sandbox modes."""

    def setup_method(self):
        """Reset session mode before each test."""
        set_session_sandbox_mode(None)

    def teardown_method(self):
        """Reset session mode after each test."""
        set_session_sandbox_mode(None)

    def test_permissive_open_read_anywhere(self):
        """Permissive-open should allow reading outside project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

            # file_read=anywhere in permissive-open
            is_valid, _ = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
            assert is_valid is True

    def test_permissive_open_write_limited(self):
        """Permissive-open should allow write in project and /tmp but not arbitrary paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

            # Project path is allowed
            is_valid, _ = validate_path_for_write("test.txt", cwd=Path(tmpdir))
            assert is_valid is True

            # /tmp is explicitly allowed in the sandbox profile
            is_valid, _ = validate_path_for_write("/tmp/test.txt", cwd=Path(tmpdir))
            assert is_valid is True

            # But arbitrary paths like /etc are not allowed
            is_valid, _ = validate_path_for_write("/etc/test.txt", cwd=Path(tmpdir))
            assert is_valid is False

    def test_restrictive_read_project_only(self):
        """Restrictive modes should only allow read in project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in [SandboxMode.RESTRICTIVE_OPEN, SandboxMode.RESTRICTIVE_CLOSED]:
                set_session_sandbox_mode(mode)
                # Try to read outside the cwd (use /etc/passwd which is definitely outside)
                is_valid, _ = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
                assert is_valid is False, f"Read should be blocked in {mode.value}"

    def test_none_mode_allows_all(self):
        """None mode should allow all paths (except traversal)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_session_sandbox_mode(SandboxMode.NONE)

            is_valid, _ = validate_path_for_read("/etc/passwd", cwd=Path(tmpdir))
            assert is_valid is True

            is_valid, _ = validate_path_for_write("/tmp/test.txt", cwd=Path(tmpdir))
            assert is_valid is True

            # But traversal still blocked
            is_valid, error = validate_path_for_read("../secret.txt", cwd=Path(tmpdir))
            assert is_valid is False
            assert "Path traversal" in error
