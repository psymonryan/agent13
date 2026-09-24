"""Unit tests for remote_transfer (spec item 1: binary-exact file transfer).

The sftp subprocess and the remote hash/dir probes are mocked; the local
hashing, quoting, and result assembly are tested for real.
"""

import hashlib
import os

import pytest

from agent13.remote_transfer import (
    _normalize_remote_path,
    _parent_dir,
    _ps_quote,
    _run_sftp,
    _sftp_quote,
    _sftp_wire_path,
    _sha256_file,
    remote_get,
    remote_put,
)
from agent13.remote_exec import clear_remote_shell_cache


# ─── Pure helpers ─────────────────────────────────────────────────────────────


class TestSha256File:
    def test_known_content(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        assert _sha256_file(str(p)) == hashlib.sha256(b"hello world").hexdigest()

    def test_binary_content(self, tmp_path):
        p = tmp_path / "b.bin"
        data = bytes(range(256)) * 100
        p.write_bytes(data)
        assert _sha256_file(str(p)) == hashlib.sha256(data).hexdigest()


class TestQuoting:
    def test_sftp_quote_plain(self):
        assert _sftp_quote("/tmp/file.bin") == '"/tmp/file.bin"'

    def test_sftp_quote_spaces(self):
        assert _sftp_quote("/tmp/my file.bin") == '"/tmp/my file.bin"'

    def test_sftp_quote_embedded_quote_and_backslash(self):
        assert _sftp_quote('/tmp/a"b\\c') == '"/tmp/a\\"b\\\\c"'

    def test_ps_quote_plain(self):
        assert _ps_quote("C:\\tmp\\f.bin") == "'C:\\tmp\\f.bin'"

    def test_ps_quote_embedded_single_quote(self):
        assert _ps_quote("C:\\User's\\f.bin") == "'C:\\User''s\\f.bin'"

    def test_parent_dir_posix(self):
        assert _parent_dir("/tmp/a/b/f.bin", "posix") == "/tmp/a/b"

    def test_parent_dir_posix_no_dir_part(self):
        assert _parent_dir("f.bin", "posix") == ""

    def test_parent_dir_powershell_backslash(self):
        assert _parent_dir("C:\\tmp\\f.bin", "powershell") == "C:\\tmp"

    def test_parent_dir_powershell_slash(self):
        assert _parent_dir("C:/tmp/f.bin", "powershell") == "C:/tmp"

    def test_parent_dir_powershell_no_dir_part(self):
        assert _parent_dir("f.bin", "powershell") == ""


# ─── sftp subprocess ──────────────────────────────────────────────────────────


class TestRunSftp:
    @pytest.mark.asyncio
    async def test_sftp_binary_missing(self):
        from unittest.mock import patch

        async def mock_create(*args, **kwargs):
            raise FileNotFoundError("sftp")

        with patch(
            "agent13.remote_transfer.asyncio.create_subprocess_exec",
            mock_create,
        ):
            result = await _run_sftp("myhost", "put a b")

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "sftp not found" in result["stderr"]

    @pytest.mark.asyncio
    async def test_subsystem_unavailable_is_actionable(self):
        from unittest.mock import patch, AsyncMock, MagicMock

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.pid = 999
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"sftp: subsystem request failed on channel 0\n")
        )

        async def mock_create(*args, **kwargs):
            return mock_proc

        with patch(
            "agent13.remote_transfer.asyncio.create_subprocess_exec",
            mock_create,
        ):
            result = await _run_sftp("myhost", "put a b")

        assert result["success"] is False
        assert "subsystem" in result["stderr"]
        assert "sshd_config" in result["stderr"]


# ─── remote_put / remote_get flows ────────────────────────────────────────────


class TestRemotePut:
    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_local_file_missing(self, tmp_path):
        result = await remote_put(
            host="myhost",
            local_path=str(tmp_path / "nope.bin"),
            remote_path="/tmp/x.bin",
            remote_shell="posix",
        )
        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "Local file not found" in result["stderr"]
        assert result["operation"] == "put"

    @pytest.mark.asyncio
    async def test_success_verified(self, tmp_path, monkeypatch):
        src = tmp_path / "blob.bin"
        data = os.urandom(1000)
        src.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()

        monkeypatch.setattr("agent13.remote_transfer._ensure_remote_dir", _ok_async())
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", _ok_async())
        monkeypatch.setattr(
            "agent13.remote_transfer._remote_sha256", _ok_async(sha256=expected)
        )

        result = await remote_put(
            host="myhost",
            local_path=str(src),
            remote_path="/tmp/deep/dir/blob.bin",
            remote_shell="posix",
        )

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["verified"] is True
        assert result["bytes"] == 1000
        assert result["local_sha256"] == expected
        assert result["remote_sha256"] == expected
        assert result["output_encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_hash_mismatch_fails_loudly(self, tmp_path, monkeypatch):
        src = tmp_path / "blob.bin"
        src.write_bytes(b"data")

        monkeypatch.setattr("agent13.remote_transfer._ensure_remote_dir", _ok_async())
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", _ok_async())
        monkeypatch.setattr(
            "agent13.remote_transfer._remote_sha256", _ok_async(sha256="0" * 64)
        )

        result = await remote_put(
            host="myhost",
            local_path=str(src),
            remote_path="/tmp/x.bin",
            remote_shell="posix",
        )

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert result["verified"] is False
        assert "SHA-256 mismatch" in result["stderr"]

    @pytest.mark.asyncio
    async def test_sftp_failure_propagates(self, tmp_path, monkeypatch):
        src = tmp_path / "blob.bin"
        src.write_bytes(b"data")

        monkeypatch.setattr("agent13.remote_transfer._ensure_remote_dir", _ok_async())
        monkeypatch.setattr(
            "agent13.remote_transfer._run_sftp",
            _ok_async(
                success=False,
                exit_code=13,
                stderr="sftp: put read: No such file",
            ),
        )

        result = await remote_put(
            host="myhost",
            local_path=str(src),
            remote_path="/tmp/x.bin",
            remote_shell="posix",
        )

        assert result["success"] is False
        assert result["exit_code"] == 13
        assert "No such file" in result["stderr"]

    @pytest.mark.asyncio
    async def test_dir_creation_failure(self, tmp_path, monkeypatch):
        src = tmp_path / "blob.bin"
        src.write_bytes(b"data")

        monkeypatch.setattr(
            "agent13.remote_transfer._ensure_remote_dir",
            _ok_async(success=False, exit_code=1, stderr="mkdir: permission denied"),
        )

        result = await remote_put(
            host="myhost",
            local_path=str(src),
            remote_path="/tmp/x.bin",
            remote_shell="posix",
        )

        assert result["success"] is False
        assert "Failed to create remote directory" in result["stderr"]


class TestRemoteGet:
    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_success_verified(self, tmp_path, monkeypatch):
        data = b"x" * 64
        expected = hashlib.sha256(data).hexdigest()
        local_dst = tmp_path / "sub" / "out.bin"

        async def fake_sftp(host, batch_command, timeout=None):
            # Simulate the transfer landing the file locally.
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            local_dst.write_bytes(data)
            return _ok()

        monkeypatch.setattr(
            "agent13.remote_transfer._remote_sha256", _ok_async(sha256=expected)
        )
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", fake_sftp)

        result = await remote_get(
            host="myhost",
            remote_path="/tmp/x.bin",
            local_path=str(local_dst),
            remote_shell="posix",
        )

        assert result["success"] is True
        assert result["verified"] is True
        assert result["bytes"] == 64
        assert result["local_sha256"] == expected
        assert result["remote_sha256"] == expected
        assert local_dst.read_bytes() == data

    @pytest.mark.asyncio
    async def test_remote_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent13.remote_transfer._remote_sha256",
            _ok_async(success=False, stderr="sha256sum: /tmp/x.bin: No such file"),
        )

        result = await remote_get(
            host="myhost",
            remote_path="/tmp/x.bin",
            local_path=str(tmp_path / "out.bin"),
            remote_shell="posix",
        )

        assert result["success"] is False
        assert "Remote file not readable" in result["stderr"]

    @pytest.mark.asyncio
    async def test_hash_mismatch_fails_loudly(self, tmp_path, monkeypatch):
        local_dst = tmp_path / "out.bin"

        async def fake_sftp(host, batch_command, timeout=None):
            local_dst.write_bytes(b"something-else")
            return _ok()

        monkeypatch.setattr(
            "agent13.remote_transfer._remote_sha256", _ok_async(sha256="ab" * 32)
        )
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", fake_sftp)

        result = await remote_get(
            host="myhost",
            remote_path="/tmp/x.bin",
            local_path=str(local_dst),
            remote_shell="posix",
        )

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "SHA-256 mismatch" in result["stderr"]


class TestPathNormalization:
    """The path-semantics trap: sftp and PowerShell speak different dialects.

    The harness must accept every spelling and translate per-consumer,
    so agents never have to know the remote's dialect.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/abs/path/file.bin",
            "relative/file.bin",
            "/C:/weird-but-posix/file.bin",  # posix: no drive concept, as-is
        ],
    )
    def test_posix_unchanged(self, path):
        assert _normalize_remote_path(path, "posix") == path

    @pytest.mark.parametrize(
        "spelling",
        [
            "C:\\EDDeploy\\x.zip",  # native backslash
            "C:/EDDeploy/x.zip",  # native slash
            "/C:/EDDeploy/x.zip",  # sftp wire form
            "/C:\\EDDeploy\\x.zip",  # sftp wire form, backslashes
        ],
    )
    def test_windows_all_spellings_canonicalize(self, spelling):
        assert _normalize_remote_path(spelling, "powershell") == "C:\\EDDeploy\\x.zip"

    def test_windows_relative_unchanged(self):
        # Both consumers resolve relative paths against home - consistent.
        assert _normalize_remote_path("tab5_evidence.zip", "powershell") == (
            "tab5_evidence.zip"
        )

    def test_wire_path_windows(self):
        assert (
            _sftp_wire_path("C:\\EDDeploy\\x.zip", "powershell") == "/C:/EDDeploy/x.zip"
        )

    def test_wire_path_windows_from_canonical(self):
        p = _normalize_remote_path("/C:/EDDeploy/x.zip", "powershell")
        assert _sftp_wire_path(p, "powershell") == "/C:/EDDeploy/x.zip"

    @pytest.mark.parametrize(
        "path", ["/abs/file.bin", "relative/file.bin", "/C:/file.bin"]
    )
    def test_wire_path_posix_unchanged(self, path):
        assert _sftp_wire_path(path, "posix") == path

    def test_wire_path_relative_unchanged(self):
        assert _sftp_wire_path("x.zip", "powershell") == "x.zip"


class TestWindowsPathFlow:
    """End-to-end: every consumer gets the dialect it needs."""

    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    def test_sftp_quote_escapes_windows_local_path(self):
        # Windows local paths contain backslashes; sftp batch interprets
        # backslashes as C escapes inside double quotes, so they must be
        # doubled. Platform-independent: must hold on macOS too (this is
        # what the Windows CI caught — tmp_path has backslashes only there).
        assert _sftp_quote(r"C:\Users\admin\temp\x.zip") == r'"C:\\Users\\admin\\temp\\x.zip"'
        # \n in a path (C:\notes\file.txt) must not become a newline
        assert _sftp_quote(r"C:\notes\file.txt") == r'"C:\\notes\\file.txt"'
        assert "\n" not in _sftp_quote(r"C:\notes\file.txt")
        # quotes are escaped too
        assert _sftp_quote(r'C:\a"b.txt') == r'"C:\\a\"b.txt"'

    @pytest.mark.asyncio
    async def test_put_windows_path_translates_per_consumer(
        self, tmp_path, monkeypatch
    ):
        src = tmp_path / "x.zip"
        src.write_bytes(b"data")
        seen = {}

        async def fake_ensure_dir(host, shell, dir_path):
            seen["dir"] = dir_path
            return _ok()

        async def fake_sftp(host, batch_command, timeout=None):
            seen["sftp"] = batch_command
            return _ok()

        async def fake_sha(host, shell, path):
            seen["probe"] = path
            return _ok(sha256=hashlib.sha256(b"data").hexdigest())

        monkeypatch.setattr(
            "agent13.remote_transfer._ensure_remote_dir", fake_ensure_dir
        )
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", fake_sftp)
        monkeypatch.setattr("agent13.remote_transfer._remote_sha256", fake_sha)

        # The agent passes the sftp wire form (what it learned from an error).
        result = await remote_put(
            host="myhost",
            local_path=str(src),
            remote_path="/C:/EDDeploy/x.zip",
            remote_shell="powershell",
        )

        assert result["success"] is True
        # PowerShell consumers get the native form...
        assert seen["dir"] == "C:\\EDDeploy"
        assert seen["probe"] == "C:\\EDDeploy\\x.zip"
        # ...the sftp protocol gets the wire form (local path sftp-escaped:
        # inside double quotes sftp interprets backslashes as C escapes)...
        assert seen["sftp"] == "put " + _sftp_quote(str(src)) + ' "/C:/EDDeploy/x.zip"'
        # ...and the result echoes the canonical native form.
        assert result["remote_path"] == "C:\\EDDeploy\\x.zip"

    @pytest.mark.asyncio
    async def test_get_windows_native_path(self, tmp_path, monkeypatch):
        local_dst = tmp_path / "out.zip"
        data = b"zipdata"
        seen = {}

        async def fake_sha(host, shell, path):
            seen["probe"] = path
            return _ok(sha256=hashlib.sha256(data).hexdigest())

        async def fake_sftp(host, batch_command, timeout=None):
            seen["sftp"] = batch_command
            local_dst.write_bytes(data)
            return _ok()

        monkeypatch.setattr("agent13.remote_transfer._remote_sha256", fake_sha)
        monkeypatch.setattr("agent13.remote_transfer._run_sftp", fake_sftp)

        # The agent passes the native form (what it was told).
        result = await remote_get(
            host="myhost",
            remote_path="C:\\EDDeploy\\x.zip",
            local_path=str(local_dst),
            remote_shell="powershell",
        )

        assert result["success"] is True
        assert seen["probe"] == "C:\\EDDeploy\\x.zip"
        assert seen["sftp"] == "get \"/C:/EDDeploy/x.zip\" " + _sftp_quote(str(local_dst))
        assert result["remote_path"] == "C:\\EDDeploy\\x.zip"


def _ok(**overrides):
    """A successful probe/sftp result with optional overrides."""
    base = {
        "success": True,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "sha256": None,
    }
    base.update(overrides)
    return base


def _ok_async(**overrides):
    """Async stand-in for the mocked probes (they are awaited)."""
    result = _ok(**overrides)

    async def _fn(*a, **k):
        return result

    return _fn
