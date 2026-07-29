"""Tests for file injection: @filename expansion and --read message building."""

import os

import pytest

from agent13.file_injection import (
    expand_file_mentions,
    build_read_message,
    DEFAULT_MAX_EMBED_BYTES,
)


class TestExpandFileMentions:
    """Tests for @filename inline expansion."""

    def test_no_at_token(self):
        """Text without @ is returned unchanged."""
        assert expand_file_mentions("hello world") == "hello world"

    def test_empty_string(self):
        assert expand_file_mentions("") == ""

    def test_email_not_expanded(self, tmp_path):
        """@ in email addresses must not trigger expansion."""
        text = "contact me at user@example.com"
        assert expand_file_mentions(text) == text

    def test_underscore_before_at_not_expanded(self, tmp_path):
        """@ preceded by underscore is not a file anchor."""
        text = "some_var@not_a_file"
        assert expand_file_mentions(text) == text

    def test_nonexistent_file_stays_literal(self):
        """@nonexistent stays as literal text."""
        result = expand_file_mentions("@this_file_does_not_exist_xyz.txt")
        assert result == "@this_file_does_not_exist_xyz.txt"

    def test_real_file_expanded(self, tmp_path):
        """@file that exists gets inlined with <file> tags."""
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = expand_file_mentions(f"look at @{f}")
        assert '<file path="' in result
        assert "hello world" in result
        assert "</file>" in result

    def test_relative_path(self, tmp_path):
        """Relative @path resolves against base_dir."""
        f = tmp_path / "rel.txt"
        f.write_text("relative content")
        result = expand_file_mentions("@rel.txt", base_dir=str(tmp_path))
        assert "relative content" in result

    def test_multiple_files(self, tmp_path):
        """Multiple @file tokens in one message."""
        f1 = tmp_path / "a.txt"
        f1.write_text("content A")
        f2 = tmp_path / "b.txt"
        f2.write_text("content B")
        result = expand_file_mentions(f"see @{f1} and @{f2}")
        assert "content A" in result
        assert "content B" in result

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        """@~/file expands ~ to home directory."""
        f = tmp_path / "tilde_test.txt"
        f.write_text("tilde content")
        # Mock expanduser to map ~/anything to tmp_path/anything
        original_expanduser = os.path.expanduser

        def mock_expanduser(path):
            if path.startswith("~/"):
                return str(tmp_path) + "/" + path[2:]
            return original_expanduser(path)
        monkeypatch.setattr("os.path.expanduser", mock_expanduser)
        result = expand_file_mentions("@~/tilde_test.txt")
        assert "tilde content" in result

    def test_binary_file_not_expanded(self, tmp_path):
        """Binary files are left as literal @path."""
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        result = expand_file_mentions(f"@{f}")
        # Binary file should stay literal
        assert f"@{f}" in result

    def test_large_file_not_expanded(self, tmp_path):
        """Files over the size cap are left as literal @path."""
        f = tmp_path / "large.txt"
        f.write_text("x" * (DEFAULT_MAX_EMBED_BYTES + 1))
        result = expand_file_mentions(f"@{f}")
        assert f"@{f}" in result

    def test_directory_not_expanded(self, tmp_path):
        """@dir (directory) stays literal."""
        result = expand_file_mentions(f"@{tmp_path}")
        assert f"@{tmp_path}" in result

    def test_crlf_file_expanded(self, tmp_path):
        """Windows CRLF line endings must not be treated as binary."""
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"a\r\nb\r\nc\r\n")
        result = expand_file_mentions(f"@{f}")
        assert '<file path="' in result
        assert "a\r\nb\r\nc\r\n" in result

    def test_quoted_path_with_spaces(self, tmp_path):
        """@'path with spaces.txt' works."""
        f = tmp_path / "with spaces.txt"
        f.write_text("spaced content")
        result = expand_file_mentions(f"@'{f}'", base_dir=str(tmp_path))
        assert "spaced content" in result

    def test_text_around_at_file(self, tmp_path):
        """Normal text before and after @file is preserved."""
        f = tmp_path / "mid.txt"
        f.write_text("middle")
        result = expand_file_mentions(f"before @{f} after")
        assert "before" in result
        assert "middle" in result
        assert "after" in result


class TestBuildReadMessage:
    """Tests for --read message building."""

    def test_includes_acknowledgement(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file content")
        msg = build_read_message([str(f)])
        assert "files read" in msg.lower()

    def test_includes_file_contents(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("the content")
        msg = build_read_message([str(f)])
        assert "the content" in msg
        assert '<file path=' in msg

    def test_missing_file_error(self):
        msg = build_read_message(["/nonexistent/path.txt"])
        assert "<file_error" in msg
        assert "not found" in msg.lower()

    def test_directory_error(self, tmp_path):
        msg = build_read_message([str(tmp_path)])
        assert "<file_error" in msg
        assert "not a file" in msg.lower() or "directory" in msg.lower()

    def test_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("content A")
        f2 = tmp_path / "b.txt"
        f2.write_text("content B")
        msg = build_read_message([str(f1), str(f2)])
        assert "content A" in msg
        assert "content B" in msg

    def test_binary_file_error(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        msg = build_read_message([str(f)])
        assert "<file_error" in msg
        assert "binary" in msg.lower()

    def test_large_file_error(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("x" * (DEFAULT_MAX_EMBED_BYTES + 1))
        msg = build_read_message([str(f)])
        assert "<file_error" in msg
        assert "too large" in msg.lower()

    def test_empty_list(self):
        """No files — just the header and acknowledgement."""
        msg = build_read_message([])
        assert "files read" in msg.lower()
        assert "<file" not in msg


class TestPathAnchor:
    """Tests for _is_path_anchor logic (via expand_file_mentions behavior)."""

    def test_at_start_of_string(self, tmp_path):
        """@ at position 0 is always an anchor."""
        f = tmp_path / "start.txt"
        f.write_text("start")
        result = expand_file_mentions(f"@{f}")
        assert "start" in result

    def test_at_after_space(self, tmp_path):
        """@ after a space is an anchor."""
        f = tmp_path / "spaced.txt"
        f.write_text("spaced")
        result = expand_file_mentions(f"see @{f}")
        assert "spaced" in result

    def test_at_after_newline(self, tmp_path):
        """@ after a newline is an anchor."""
        f = tmp_path / "lined.txt"
        f.write_text("lined")
        result = expand_file_mentions(f"first line\n@{f}")
        assert "lined" in result


class TestPathExtraction:
    """Tests for path candidate extraction (cross-platform)."""

    def test_colon_is_path_char(self):
        """Colon must be allowed in paths for Windows drive letters (C:\\)."""
        from agent13.file_injection import _is_path_char

        assert _is_path_char(":")

    def test_windows_drive_path_extraction(self):
        """Windows drive-letter paths (C:\\Users\\...) extract fully."""
        from agent13.file_injection import _extract_path_candidate

        path = r"C:\Users\Admin\hello.txt"
        candidate, end = _extract_path_candidate(path, 0)
        assert candidate == path
        assert end == len(path)

    def test_line_reference_stays_literal(self, tmp_path):
        """@file.txt:42 (line reference) stays literal when file doesn't exist."""
        result = expand_file_mentions("@nonexistent.txt:42")
        assert result == "@nonexistent.txt:42"

    def test_colon_in_filename_expands(self, tmp_path):
        """Colons in filenames work on POSIX systems."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Colon in filenames not allowed on Windows")

        f = tmp_path / "file:with:colons.txt"
        try:
            f.write_text("colon content")
        except OSError:
            pytest.skip("Filesystem does not support colons in filenames")

        result = expand_file_mentions(f"@{f}")
        assert "colon content" in result
