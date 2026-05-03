"""Tests for the clipboard module."""

from unittest.mock import patch, MagicMock

from agent13.clipboard import copy_via_system, copy_to_clipboard, VALID_METHODS


class TestValidMethods:
    def test_valid_methods(self):
        assert VALID_METHODS == ("osc52", "system")


class TestCopyViaSystem:
    def test_success(self):
        """Should return True when clipboard command succeeds."""
        with patch("agent13.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = copy_via_system("test text")
        assert result is True

    def test_failure(self):
        """Should return False when clipboard command fails."""
        with patch("agent13.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = copy_via_system("test text")
        assert result is False

    def test_timeout(self):
        """Should return False on timeout."""
        import subprocess
        with patch("agent13.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="pbcopy", timeout=5
            )
            result = copy_via_system("test text")
        assert result is False

    def test_os_error(self):
        """Should return False on OSError."""
        with patch("agent13.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("not found")
            result = copy_via_system("test text")
        assert result is False


class TestCopyToClipboard:
    def test_system_method(self):
        """Should delegate to copy_via_system for system method."""
        with patch("agent13.clipboard.copy_via_system") as mock_system:
            mock_system.return_value = True
            result = copy_to_clipboard("test", method="system")
        mock_system.assert_called_once_with("test")
        assert result is True

    def test_osc52_with_handler(self):
        """Should call osc52_handler for osc52 method."""
        handler = MagicMock()
        result = copy_to_clipboard("test", method="osc52", osc52_handler=handler)
        handler.assert_called_once_with("test")
        assert result is True

    def test_osc52_without_handler(self):
        """Should return False for osc52 with no handler."""
        result = copy_to_clipboard("test", method="osc52", osc52_handler=None)
        assert result is False

    def test_osc52_handler_exception(self):
        """Should return False if osc52_handler raises."""
        handler = MagicMock(side_effect=RuntimeError("no terminal"))
        result = copy_to_clipboard("test", method="osc52", osc52_handler=handler)
        assert result is False
