"""Tests for the self-update checker."""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from agent13.updater import (
    _parse_version,
    _is_newer,
    _should_check,
    _read_last_check,
    _write_last_check,
    _find_wheel_asset,
    _build_manual_command,
    _find_scripts_dir,
    _rename_locked_scripts_dir,
    _restore_renamed_scripts_dir,
    cleanup_old_scripts_dir,
    check_for_update,
    perform_update,
)
from agent13.clipboard import copy_via_system


class TestParseVersion:
    def test_simple_version(self):
        assert _parse_version("0.1.8") == (0, 1, 8)

    def test_version_with_v_prefix(self):
        assert _parse_version("v0.2.0") == (0, 2, 0)

    def test_two_part_version(self):
        assert _parse_version("1.0") == (1, 0)

    def test_empty_string(self):
        assert _parse_version("") == ()

    def test_non_numeric_suffix(self):
        # "0.1.8a" -> (0, 1) — stops at "8a" which isn't a pure int
        assert _parse_version("0.1.8a") == (0, 1)


class TestIsNewer:
    def test_newer_patch(self):
        assert _is_newer("0.1.9", "0.1.8") is True

    def test_newer_minor(self):
        assert _is_newer("0.2.0", "0.1.8") is True

    def test_newer_major(self):
        assert _is_newer("1.0.0", "0.1.8") is True

    def test_same_version(self):
        assert _is_newer("0.1.8", "0.1.8") is False

    def test_older_version(self):
        assert _is_newer("0.1.7", "0.1.8") is False

    def test_empty_remote(self):
        assert _is_newer("", "0.1.8") is False

    def test_v_prefix_remote(self):
        assert _is_newer("v0.2.0", "0.1.8") is True


class TestFindWheelAsset:
    def test_finds_wheel(self):
        assets = [
            {"name": "agent13-0.1.9.tar.gz",
             "browser_download_url": "https://example.com/sdist.tar.gz"},
            {"name": "agent13-0.1.9-py3-none-any.whl",
             "browser_download_url": "https://example.com/wheel.whl"},
        ]
        assert _find_wheel_asset(assets) == "https://example.com/wheel.whl"

    def test_no_wheel(self):
        assets = [
            {"name": "agent13-0.1.9.tar.gz",
             "browser_download_url": "https://example.com/sdist.tar.gz"},
        ]
        assert _find_wheel_asset(assets) is None

    def test_empty_assets(self):
        assert _find_wheel_asset([]) is None


class TestBuildManualCommand:
    def test_builds_command(self):
        url = "https://github.com/o/r/releases/download/v0.1.9/a-0.1.9-py3-none-any.whl"
        assert _build_manual_command(url) == f"uv tool install --force {url}"


class TestThrottle:
    def test_should_check_no_state_file(self, tmp_path):
        """Should check when no state file exists."""
        with patch("agent13.updater._LAST_CHECK_FILE", tmp_path / "nope.json"):
            assert _should_check(24) is True

    def test_should_check_recent(self, tmp_path):
        """Should NOT check if last check was recent."""
        state_file = tmp_path / "last_update_check.json"
        now = datetime.now(timezone.utc)
        state_file.write_text(json.dumps({"last_check": now.isoformat()}))
        with patch("agent13.updater._LAST_CHECK_FILE", state_file):
            assert _should_check(24) is False

    def test_should_check_expired(self, tmp_path):
        """Should check if last check was long ago."""
        state_file = tmp_path / "last_update_check.json"
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        state_file.write_text(json.dumps({"last_check": old.isoformat()}))
        with patch("agent13.updater._LAST_CHECK_FILE", state_file):
            assert _should_check(24) is True

    def test_write_and_read_last_check(self, tmp_path):
        """Write then read should roundtrip correctly."""
        state_file = tmp_path / "last_update_check.json"
        now = datetime.now(timezone.utc)
        with patch("agent13.updater._LAST_CHECK_FILE", state_file):
            with patch("agent13.updater.get_config_dir", return_value=tmp_path):
                _write_last_check(now)
            result = _read_last_check()
        assert result is not None
        # Compare ISO format strings to avoid microsecond drift
        assert result.isoformat().startswith(now.strftime("%Y-%m-%dT%H"))


class TestCheckForUpdate:
    def test_returns_none_when_throttled(self, tmp_path):
        """Should return None if checked recently."""
        state_file = tmp_path / "last_update_check.json"
        now = datetime.now(timezone.utc)
        state_file.write_text(json.dumps({"last_check": now.isoformat()}))
        with patch("agent13.updater._LAST_CHECK_FILE", state_file):
            assert check_for_update(24) is None

    def test_returns_none_when_up_to_date(self, tmp_path):
        """Should return None if remote version is not newer."""
        state_file = tmp_path / "last_update_check.json"
        with (
            patch("agent13.updater._LAST_CHECK_FILE", state_file),
            patch("agent13.updater.get_config_dir", return_value=tmp_path),
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.8", "html_url": "",
                "wheel_url": "",
            }
            assert check_for_update(0) is None  # interval=0 forces check

    def test_returns_dict_when_update_available(self, tmp_path):
        """Should return a dict if remote version is newer."""
        state_file = tmp_path / "last_update_check.json"
        with (
            patch("agent13.updater._LAST_CHECK_FILE", state_file),
            patch("agent13.updater.get_config_dir", return_value=tmp_path),
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.2.0", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            result = check_for_update(0)  # interval=0 forces check
        assert result is not None
        assert result["remote_tag"] == "0.2.0"
        assert result["local_version"] == "0.1.8"
        assert result["wheel_url"] == "https://example.com/wheel.whl"
        assert "uv tool install --force" in result["manual_cmd"]

    def test_returns_none_when_fetch_fails(self, tmp_path):
        """Should return None gracefully if GitHub is unreachable."""
        state_file = tmp_path / "last_update_check.json"
        with (
            patch("agent13.updater._LAST_CHECK_FILE", state_file),
            patch("agent13.updater.fetch_latest_release", return_value=None),
        ):
            assert check_for_update(0) is None


class TestPerformUpdate:
    def test_already_up_to_date(self):
        """Should report already on latest when no newer version."""
        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.9"),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            success, msg = perform_update()
        assert success is True
        assert "Already on latest" in msg

    def test_no_wheel_asset(self):
        """Should fail gracefully when no wheel on the release."""
        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "",
            }
            success, msg = perform_update()
        assert success is False
        assert "no wheel asset" in msg.lower()

    def test_fetch_fails(self):
        """Should fail when GitHub is unreachable."""
        with (
            patch("agent13.updater.fetch_latest_release", return_value=None),
        ):
            success, msg = perform_update()
        assert success is False
        assert "Could not reach" in msg

    def test_download_then_install_success(self):
        """Should download wheel and install via uv tool."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            mock_run.return_value = MagicMock(returncode=0)
            success, msg = perform_update()
        assert success is True
        assert "0.1.9" in msg
        assert "successfully" in msg.lower()
        # Verify it used uv tool install --force <wheel_path>
        args = mock_run.call_args[0][0]
        assert args[0] == "uv"
        assert args[1] == "tool"
        assert args[2] == "install"
        assert args[3] == "--force"
        assert args[4].endswith(".whl")

    def test_download_fails(self):
        """Should fail and suggest manual command when download fails."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            success, msg = perform_update()
        assert success is False
        assert "Failed to download" in msg
        assert "uv tool install --force" in msg

    def test_install_fails_with_manual_fallback(self):
        """Should suggest manual command when install subprocess fails."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            mock_run.return_value = MagicMock(
                returncode=1, stderr="permission denied"
            )
            success, msg = perform_update()
        assert success is False
        assert "Install failed" in msg
        assert "uv tool install --force" in msg

    def test_timeout(self):
        """Should handle subprocess timeout with manual fallback."""
        import subprocess

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/wheel.whl",
            }
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="uv", timeout=120
            )
            success, msg = perform_update()
        assert success is False
        assert "timed out" in msg.lower()
        assert "uv tool install --force" in msg


class TestCopyToClipboard:
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


class TestFindScriptsDir:
    """Tests for _find_scripts_dir helper."""

    def test_returns_dir_of_sys_executable(self):
        """Should return the directory containing sys.executable."""
        with (
            patch("agent13.updater.sys.executable", "/opt/uv/tools/agent13/Scripts/python"),
            patch("agent13.updater.os.path.isdir", return_value=True),
        ):
            result = _find_scripts_dir()
            assert result == "/opt/uv/tools/agent13/Scripts"

    def test_returns_none_when_not_a_dir(self):
        """Should return None when the directory doesn't exist."""
        with (
            patch("agent13.updater.sys.executable", "/nonexistent/python"),
            patch("agent13.updater.os.path.isdir", return_value=False),
        ):
            assert _find_scripts_dir() is None


class TestRenameLockedScriptsDir:
    """Tests for _rename_locked_scripts_dir Windows helper."""

    def test_returns_none_on_posix(self):
        """Should return None on non-Windows (no rename needed)."""
        with patch("agent13.updater.os.name", "posix"):
            assert _rename_locked_scripts_dir() is None

    def test_renames_dir_to_temp_on_windows(self):
        """Should rename Scripts dir to temp location on Windows."""
        with (
            patch("agent13.updater.os.name", "nt"),
            patch("agent13.updater._find_scripts_dir", return_value=r"C:\Scripts"),
            patch("agent13.updater.os.path.exists", return_value=False),
            patch("agent13.updater.os.rename") as mock_rename,
        ):
            result = _rename_locked_scripts_dir()
        # Result should be in temp dir with agent13-scripts- prefix
        assert result is not None
        assert "agent13-scripts-" in result
        assert result.endswith(".old")
        mock_rename.assert_called_once()

    def test_returns_none_on_rename_failure(self):
        """Should return None if os.rename fails."""
        with (
            patch("agent13.updater.os.name", "nt"),
            patch("agent13.updater._find_scripts_dir", return_value=r"C:\Scripts"),
            patch("agent13.updater.os.path.exists", return_value=False),
            patch("agent13.updater.os.rename", side_effect=OSError("denied")),
        ):
            result = _rename_locked_scripts_dir()
        assert result is None

    def test_returns_none_when_dir_not_found(self):
        """Should return None if _find_scripts_dir returns None."""
        with (
            patch("agent13.updater.os.name", "nt"),
            patch("agent13.updater._find_scripts_dir", return_value=None),
        ):
            result = _rename_locked_scripts_dir()
        assert result is None


class TestRestoreRenamedScriptsDir:
    """Tests for _restore_renamed_scripts_dir rollback helper."""

    def test_restores_dir_from_temp(self):
        """Should rename temp dir back to Scripts path."""
        with (
            patch("agent13.updater.os.path.exists", return_value=False),
            patch("agent13.updater.os.rename") as mock_rename,
        ):
            _restore_renamed_scripts_dir("/tmp/agent13-scripts-123.old", r"C:\Scripts")
        mock_rename.assert_called_once_with(
            "/tmp/agent13-scripts-123.old",
            r"C:\Scripts",
        )

    def test_removes_new_dir_before_restore(self):
        """Should remove new Scripts dir before restoring from temp."""
        with (
            patch("agent13.updater.os.path.exists", return_value=True),
            patch("agent13.updater.shutil.rmtree") as mock_rmtree,
            patch("agent13.updater.os.rename") as mock_rename,
        ):
            _restore_renamed_scripts_dir("/tmp/agent13-scripts-123.old", r"C:\Scripts")
        mock_rmtree.assert_called_once_with(r"C:\Scripts")
        mock_rename.assert_called_once()


class TestCleanupOldScriptsDir:
    """Tests for cleanup_old_scripts_dir startup helper."""

    def test_noop_on_posix(self):
        """Should do nothing on non-Windows."""
        with patch("agent13.updater.os.name", "posix"):
            cleanup_old_scripts_dir()  # Should not raise

    def test_removes_stale_temp_dirs_on_windows(self):
        """Should remove agent13-scripts-*.old dirs in temp on Windows."""
        with (
            patch("agent13.updater.os.name", "nt"),
            patch("agent13.updater.os.listdir", return_value=[
                "agent13-scripts-999.old",
                "other-file.txt",
            ]),
            patch("agent13.updater.os.path.exists", return_value=True),
            patch("agent13.updater.shutil.rmtree") as mock_rmtree,
        ):
            cleanup_old_scripts_dir()
        assert mock_rmtree.call_count == 1

    def test_noop_when_no_stale_dirs(self):
        """Should do nothing when no stale dirs exist."""
        with (
            patch("agent13.updater.os.name", "nt"),
            patch("agent13.updater.os.listdir", return_value=["other-file.txt"]),
        ):
            cleanup_old_scripts_dir()  # Should not raise


class TestPerformUpdateWindowsDirRename:
    """Tests for perform_update using the Windows Scripts dir rename."""

    def test_rename_called_on_windows(self):
        """Should rename Scripts dir before install on Windows."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
            patch("agent13.updater._rename_locked_scripts_dir", return_value="/tmp/agent13-scripts-123.old"),
            patch("agent13.updater._find_scripts_dir", return_value=r"C:\Scripts"),
            patch("agent13.updater.os.path.exists", return_value=False),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/agent13-0.1.9-py3-none-any.whl",
            }
            mock_run.return_value = MagicMock(returncode=0)
            success, msg = perform_update()
        assert success is True
        assert "0.1.9" in msg

    def test_rollback_on_install_failure_windows(self):
        """Should restore from temp dir when install fails on Windows."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
            patch("agent13.updater._rename_locked_scripts_dir", return_value="/tmp/agent13-scripts-123.old"),
            patch("agent13.updater._find_scripts_dir", return_value=r"C:\Scripts"),
            patch("agent13.updater._restore_renamed_scripts_dir") as mock_restore,
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/agent13-0.1.9-py3-none-any.whl",
            }
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Access is denied"
            )
            success, msg = perform_update()
        assert success is False
        mock_restore.assert_called_once_with(
            "/tmp/agent13-scripts-123.old", r"C:\Scripts"
        )

    def test_no_rename_on_posix(self):
        """Should not try to rename Scripts dir on POSIX."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
            patch("agent13.updater._rename_locked_scripts_dir", return_value=None) as mock_rename,
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/agent13-0.1.9-py3-none-any.whl",
            }
            mock_run.return_value = MagicMock(returncode=0)
            success, msg = perform_update()
        assert success is True
        mock_rename.assert_called_once()

    def test_entrypoint_copy_failure_treated_as_success(self):
        """Should treat entrypoint copy failure as success on Windows."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-wheel-bytes"

        with (
            patch("agent13.updater.fetch_latest_release") as mock_fetch,
            patch("agent13.updater.__version__", "0.1.8"),
            patch("agent13.updater.httpx.get", return_value=mock_response),
            patch("agent13.updater.subprocess.run") as mock_run,
            patch("agent13.updater._rename_locked_scripts_dir", return_value="/tmp/agent13-scripts-123.old"),
            patch("agent13.updater._find_scripts_dir", return_value=r"C:\Scripts"),
            patch("agent13.updater.os.path.exists", return_value=False),
            patch("agent13.updater.os.name", "nt"),
        ):
            mock_fetch.return_value = {
                "tag_name": "0.1.9", "html_url": "",
                "wheel_url": "https://example.com/agent13-0.1.9-py3-none-any.whl",
            }
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="error: Failed to install entrypoint\n"
                "  Caused by: failed to copy file: os error 32",
            )
            success, msg = perform_update()
        assert success is True
        assert "0.1.9" in msg
        assert "successfully" in msg
