"""Integration tests for corrupt/foreign-encoded config files.

Spawns the REAL CLI (agent13.py --list-providers) in a subprocess with an
isolated config dir, exactly what a user experiences. Verifies:

- BOM / UTF-16 / CRLF files are repaired transparently (no user action)
- Genuinely invalid TOML exits 1 with ONE clean actionable line -
  no Python traceback, no PyInstaller error

Unlike wiring tests (test_config_encoding.py), these exercise the full
process: entry point, CLI wrapper, config load, and exit code.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

GOOD_TOML = """[[providers]]
name = "testp"
api_base = "http://localhost:9999/v1"
api_key_env_var = "TEST_KEY"
"""


def run_cli(config_dir: Path, args=None):
    """Run agent13.py in a subprocess with an isolated config dir."""
    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    env["HOME"] = str(config_dir)  # isolate .env discovery too
    cmd = [sys.executable, str(REPO_ROOT / "agent13.py")] + (
        args or ["--list-providers"]
    )
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
    )


class TestForeignEncodingsRepaired:
    """Files a text editor (Notepad) may produce parse without user action."""

    @pytest.mark.parametrize(
        "encode",
        [
            lambda t: t.replace("\n", "\r\n").encode("utf-8"),  # CRLF
            lambda t: b"\xef\xbb\xbf" + t.encode("utf-8"),  # UTF-8 BOM
            lambda t: t.encode("utf-16"),  # UTF-16 LE + BOM
            lambda t: t.encode("utf-16-be"),  # UTF-16 BE + BOM
            lambda t: t.encode("utf-16-le"),  # UTF-16 LE, no BOM
            lambda t: t.encode("utf-16-be"),  # UTF-16 BE, no BOM
        ],
        ids=[
            "crlf",
            "utf8-bom",
            "utf16-le",
            "utf16-be",
            "utf16-le-nobom",
            "utf16-be-nobom",
        ],
    )
    def test_lists_providers(self, tmp_path, encode):
        (tmp_path / "config.toml").write_bytes(encode(GOOD_TOML))
        r = run_cli(tmp_path)
        assert r.returncode == 0, r.stderr
        assert "testp" in r.stdout


class TestCorruptConfigFailsCleanly:
    """Invalid content exits 1 with one actionable line, no traceback."""

    def test_invalid_toml(self, tmp_path):
        (tmp_path / "config.toml").write_text("[[providers]]\nname = \n")
        r = run_cli(tmp_path)
        assert r.returncode == 1
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined
        assert "not valid TOML" in combined
        assert "re-save it as UTF-8" in combined

    def test_unrecognized_encoding(self, tmp_path):
        (tmp_path / "config.toml").write_bytes(b"\x01\x02\x03\xff\xfe\xfd garbage")
        r = run_cli(tmp_path)
        assert r.returncode == 1
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined
        assert "re-save it as UTF-8" in combined

    def test_error_goes_to_stderr(self, tmp_path):
        (tmp_path / "config.toml").write_text("[[providers]]\nname = \n")
        r = run_cli(tmp_path)
        assert "not valid TOML" in r.stderr
        assert r.stdout == ""


class TestCorruptEnvNotFatal:
    """A corrupt .env warns but does not block startup."""

    def test_utf16_env_still_starts(self, tmp_path):
        (tmp_path / "config.toml").write_text(GOOD_TOML)
        (tmp_path / ".env").write_bytes(
            b"\xff\xfe" + "TEST_KEY=abc\n".encode("utf-16-le")
        )
        r = run_cli(tmp_path)
        assert r.returncode == 0, r.stderr
        assert "testp" in r.stdout
        assert "Traceback" not in r.stdout + r.stderr
