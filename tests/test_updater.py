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
        assert "restart" in msg.lower()
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
