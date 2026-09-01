"""Wiring tests for robust reading of user-editable config files.

Covers read_text_robust (BOM / UTF-16 / ANSI handling), Config.from_file
error wrapping, .env loading with corrupt files, and yaml_store fallbacks.

Integration tests (real CLI process) live in test_config_encoding_integration.py.
"""

import os
import locale

import pytest

from agent13.fileio import ConfigFileError, read_text_robust
from agent13 import config as config_mod
from agent13.config import Config, reset_config, load_environment
from agent13.yaml_store import load_yaml

TOML = """[[providers]]
name = "testp"
api_base = "http://localhost:9999/v1"
api_key_env_var = "TEST_KEY"
"""


# ---------------------------------------------------------------------------
# read_text_robust
# ---------------------------------------------------------------------------


class TestReadTextRobust:
    def test_plain_utf8(self, tmp_path):
        p = tmp_path / "f.txt"
        # newline="" keeps \n as \n on Windows (write_text would translate to \r\n)
        p.write_text("hello = 1\n", encoding="utf-8", newline="")
        assert read_text_robust(p) == "hello = 1\n"

    def test_crlf_passes_through(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"a = 1\r\nb = 2\r\n")
        assert read_text_robust(p) == "a = 1\r\nb = 2\r\n"

    def test_utf8_bom_stripped(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"\xef\xbb\xbf" + "a = 1\n".encode())
        assert read_text_robust(p) == "a = 1\n"

    def test_utf16_le_bom(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes("a = 1\n".encode("utf-16"))  # LE + BOM on all platforms
        assert read_text_robust(p) == "a = 1\n"

    def test_utf16_be_bom(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes("a = 1\n".encode("utf-16-be"))
        assert read_text_robust(p) == "a = 1\n"

    def test_utf16_le_no_bom(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes("a = 1\n".encode("utf-16-le"))
        assert read_text_robust(p) == "a = 1\n"

    def test_utf16_be_no_bom(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes("a = 1\n".encode("utf-16-be"))
        assert read_text_robust(p) == "a = 1\n"

    def test_ansi_falls_back_to_os_encoding(self, tmp_path, monkeypatch):
        # Simulate a Windows cp1252 file on any platform
        monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "cp1252")
        p = tmp_path / "f.txt"
        p.write_bytes("name = caf\xe9\n".encode("cp1252"))
        assert read_text_robust(p) == "name = café\n"

    def test_unrecognized_encoding_raises_config_file_error(
        self, tmp_path, monkeypatch
    ):
        # Bytes that are neither UTF-8 nor cp1252-decodable
        monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "cp1252")
        p = tmp_path / "f.txt"
        p.write_bytes(b"\x81\x81\x81 invalid")
        with pytest.raises(ConfigFileError) as exc_info:
            read_text_robust(p)
        msg = str(exc_info.value)
        assert str(p) in msg
        assert "re-save it as UTF-8" in msg

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            read_text_robust(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# Config.from_file
# ---------------------------------------------------------------------------


class TestConfigFromFile:
    def test_utf8_bom_config_parses(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_bytes(b"\xef\xbb\xbf" + TOML.encode())
        cfg = Config.from_file(p)
        assert cfg.providers[0].name == "testp"

    def test_utf16_config_parses(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_bytes(TOML.encode("utf-16"))
        cfg = Config.from_file(p)
        assert cfg.providers[0].name == "testp"

    def test_invalid_toml_raises_config_file_error(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text("[[providers]]\nname = \n")
        with pytest.raises(ConfigFileError) as exc_info:
            Config.from_file(p)
        msg = str(exc_info.value)
        assert "not valid TOML" in msg
        assert str(p) in msg
        assert "re-save it as UTF-8" in msg

    def test_structural_error_still_value_error(self, tmp_path):
        # Valid TOML, wrong shape - stays a plain ValueError (not wrapped)
        p = tmp_path / "config.toml"
        p.write_text("providers = 42\n")
        with pytest.raises(ValueError):
            Config.from_file(p)


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


class TestLoadEnvironment:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        """Point env-file helpers at a temp dir and reset module state."""
        self.env_dir = tmp_path / "envhome"
        self.env_dir.mkdir()
        monkeypatch.setattr(
            config_mod, "get_global_env_file", lambda: self.env_dir / ".env"
        )
        monkeypatch.setattr(
            config_mod, "get_local_env_file", lambda: self.env_dir / "local.env"
        )
        monkeypatch.setattr(config_mod, "ensure_default_env", lambda: None)
        reset_config()
        yield
        reset_config()

    def test_utf16_env_file_loads(self, monkeypatch):
        (self.env_dir / ".env").write_bytes(
            "TEST_KEY_FROM_ENV=secret123\n".encode("utf-16")
        )
        monkeypatch.delenv("TEST_KEY_FROM_ENV", raising=False)
        load_environment()
        assert os.environ.get("TEST_KEY_FROM_ENV") == "secret123"

    def test_bom_env_file_loads(self, monkeypatch):
        (self.env_dir / ".env").write_bytes(
            b"\xef\xbb\xbf" + "TEST_KEY_BOM=abc\n".encode()
        )
        monkeypatch.delenv("TEST_KEY_BOM", raising=False)
        load_environment()
        assert os.environ.get("TEST_KEY_BOM") == "abc"

    def test_corrupt_env_warns_but_continues(self, monkeypatch, capsys):
        # Force a strict fallback encoding so the bytes are undecodable on
        # every platform (latin-1-based locales would decode them silently)
        monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "cp1252")
        (self.env_dir / ".env").write_bytes(b"\x81\x81\x81 garbage")
        load_environment()  # must not raise
        err = capsys.readouterr().err
        assert "warning:" in err
        assert "re-save it as UTF-8" in err

    def test_local_env_overrides_global(self, monkeypatch):
        (self.env_dir / ".env").write_text("SHARED=global\nONLY_GLOBAL=1\n")
        (self.env_dir / "local.env").write_text("SHARED=local\n")
        for k in ("SHARED", "ONLY_GLOBAL"):
            monkeypatch.delenv(k, raising=False)
        load_environment()
        assert os.environ.get("SHARED") == "local"
        assert os.environ.get("ONLY_GLOBAL") == "1"


# ---------------------------------------------------------------------------
# yaml_store.load_yaml
# ---------------------------------------------------------------------------


class TestLoadYamlRobust:
    def test_utf8_bom_yaml(self, tmp_path):
        p = tmp_path / "f.yaml"
        p.write_bytes(b"\xef\xbb\xbf" + "a: 1\n".encode())
        assert load_yaml(p) == {"a": 1}

    def test_utf16_yaml(self, tmp_path):
        p = tmp_path / "f.yaml"
        p.write_bytes("a: 1\n".encode("utf-16"))
        assert load_yaml(p) == {"a": 1}

    def test_missing_returns_empty(self, tmp_path):
        assert load_yaml(tmp_path / "nope.yaml") == {}

    def test_invalid_yaml_still_raises(self, tmp_path):
        import yaml

        p = tmp_path / "f.yaml"
        p.write_text("a: [unclosed\n")
        with pytest.raises(yaml.YAMLError):
            load_yaml(p)
