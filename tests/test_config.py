"""Tests for configuration system."""

import os
from pathlib import Path
from unittest import mock

import pytest

from agent13.config import (
    Config,
    ProviderConfig,
    get_config,
    get_provider,
    create_client,
    load_environment,
    reset_config,
    resolve_provider_arg,
)
from agent13.config_paths import (
    get_config_dir,
    get_config_file,
    get_global_env_file,
    get_local_env_file,
    ensure_config_dir,
)


class TestConfigPaths:
    """Tests for config_paths module."""

    def test_get_config_dir_default(self):
        """Default config dir is ~/.agent13."""
        result = get_config_dir()
        assert result == Path.home() / ".agent13"

    def test_get_config_dir_env_override(self):
        """AGENT13_CONFIG_DIR overrides default."""
        with mock.patch.dict(os.environ, {"AGENT13_CONFIG_DIR": "/custom/path"}):
            result = get_config_dir()
            expected = Path("/custom/path").expanduser().resolve()
            assert result == expected

    def test_get_config_file(self):
        """Config file is config.toml in config dir."""
        with mock.patch.dict(os.environ, {"AGENT13_CONFIG_DIR": "/custom"}):
            result = get_config_file()
            expected_dir = Path("/custom").expanduser().resolve()
            assert result == expected_dir / "config.toml"

    def test_get_global_env_file(self):
        """Global env file is ~/.env."""
        result = get_global_env_file()
        assert result == Path.home() / ".env"

    def test_get_local_env_file(self):
        """Local env file is ./.env."""
        result = get_local_env_file()
        assert result == Path.cwd() / ".env"

    def test_ensure_config_dir(self, tmp_path):
        """ensure_config_dir creates directory if needed."""
        with mock.patch.dict(
            os.environ, {"AGENT13_CONFIG_DIR": str(tmp_path / "newdir")}
        ):
            result = ensure_config_dir()
            assert result.exists()
            assert result.is_dir()


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_basic(self):
        """Basic provider config creation."""
        provider = ProviderConfig(
            name="test",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
        )
        assert provider.name == "test"
        assert provider.api_base == "https://api.example.com/v1"
        assert provider.api_key_env_var == "TEST_API_KEY"

    def test_provider_config_no_key(self):
        """Provider can have no API key (for local servers)."""
        provider = ProviderConfig(
            name="local",
            api_base="http://localhost:8012/v1",
            api_key_env_var="",
        )
        assert provider.api_key_env_var == ""

    def test_get_api_key_from_env(self):
        """get_api_key reads from environment."""
        provider = ProviderConfig(
            name="test",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
        )
        with mock.patch.dict(os.environ, {"TEST_API_KEY": "secret123"}):
            assert provider.get_api_key() == "secret123"

    def test_get_api_key_missing_env(self):
        """get_api_key returns None if env var not set."""
        provider = ProviderConfig(
            name="test",
            api_base="https://api.example.com/v1",
            api_key_env_var="MISSING_KEY",
        )
        # Ensure the env var is not set
        os.environ.pop("MISSING_KEY", None)
        assert provider.get_api_key() is None

    def test_get_api_key_empty_env_var(self):
        """get_api_key returns None if api_key_env_var is empty."""
        provider = ProviderConfig(
            name="local",
            api_base="http://localhost:8012/v1",
            api_key_env_var="",
        )
        assert provider.get_api_key() is None

    def test_read_timeout_default(self):
        """ProviderConfig defaults read_timeout to 2400.0."""
        provider = ProviderConfig(
            name="test",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
        )
        assert provider.read_timeout == 2400.0

    def test_read_timeout_custom(self):
        """ProviderConfig accepts custom read_timeout."""
        provider = ProviderConfig(
            name="slow",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
            read_timeout=2400.0,
        )
        assert provider.read_timeout == 2400.0

    def test_connect_timeout_default(self):
        """ProviderConfig defaults connect_timeout to 30.0."""
        provider = ProviderConfig(
            name="test",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
        )
        assert provider.connect_timeout == 30.0

    def test_connect_timeout_custom(self):
        """ProviderConfig accepts custom connect_timeout."""
        provider = ProviderConfig(
            name="slow",
            api_base="https://api.example.com/v1",
            api_key_env_var="TEST_API_KEY",
            connect_timeout=60.0,
        )
        assert provider.connect_timeout == 60.0


class TestConfig:
    """Tests for Config class."""

    def test_config_empty(self):
        """Empty config has no providers."""
        config = Config()
        assert config.providers == []

    def test_config_from_file_valid(self, tmp_path):
        """Load valid TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"

[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = ""
""")
        config = Config.from_file(config_file)
        assert len(config.providers) == 2
        assert config.providers[0].name == "openrouter"
        assert config.providers[1].name == "local"

    def test_config_from_file_missing(self, tmp_path):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_file(tmp_path / "missing.toml")

    def test_config_from_file_or_empty_missing(self, tmp_path):
        """from_file_or_empty returns empty config if file missing."""
        config = Config.from_file_or_empty(tmp_path / "missing.toml")
        assert config.providers == []

    def test_config_get_provider(self, tmp_path):
        """get_provider finds provider by name."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"
""")
        config = Config.from_file(config_file)
        provider = config.get_provider("openrouter")
        assert provider is not None
        assert provider.api_base == "https://openrouter.ai/api/v1"

        assert config.get_provider("missing") is None

    def test_config_read_timeout_from_file(self, tmp_path):
        """read_timeout is parsed from TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"

[[providers]]
name = "slow-reasoner"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
read_timeout = 2400
""")
        config = Config.from_file(config_file)
        assert len(config.providers) == 2
        # Default timeout for first provider
        assert config.providers[0].read_timeout == 2400.0
        # Custom timeout for second provider
        assert config.providers[1].read_timeout == 2400.0

    def test_config_read_timeout_invalid(self, tmp_path):
        """Non-numeric read_timeout raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "bad"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
read_timeout = "not_a_number"
""")
        with pytest.raises(ValueError, match="read_timeout"):
            Config.from_file(config_file)

    def test_config_validate_duplicate_names(self, tmp_path):
        """Duplicate provider names raise ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "openrouter"
api_base = "https://openrouter.ai/api/v1"
api_key_env_var = "OPENROUTER_API_KEY"

[[providers]]
name = "openrouter"
api_base = "https://other.example.com/v1"
api_key_env_var = "OTHER_KEY"
""")
        with pytest.raises(ValueError, match="Duplicate provider name"):
            Config.from_file(config_file)

    def test_config_validate_invalid_name(self, tmp_path):
        """Invalid provider name raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "invalid name!"
api_base = "https://api.example.com/v1"
api_key_env_var = "KEY"
""")
        with pytest.raises(ValueError, match="must be alphanumeric"):
            Config.from_file(config_file)

    def test_config_validate_invalid_url(self, tmp_path):
        """Invalid URL raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "not a url"
api_key_env_var = "KEY"
""")
        with pytest.raises(ValueError, match="invalid api_base"):
            Config.from_file(config_file)

    def test_config_missing_name(self, tmp_path):
        """Missing name field raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
api_base = "https://api.example.com/v1"
api_key_env_var = "KEY"
""")
        with pytest.raises(ValueError, match="missing 'name' field"):
            Config.from_file(config_file)

    def test_config_missing_api_base(self, tmp_path):
        """Missing api_base field raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_key_env_var = "KEY"
""")
        with pytest.raises(ValueError, match="missing 'api_base' field"):
            Config.from_file(config_file)


class TestEnvironmentLoading:
    """Tests for environment variable loading."""

    def test_load_environment_global(self, tmp_path):
        """load_environment loads global .env file."""
        global_env = tmp_path / ".env"
        global_env.write_text("GLOBAL_VAR=global_value\n")

        with mock.patch("agent13.config.get_global_env_file", return_value=global_env):
            with mock.patch(
                "agent13.config.get_local_env_file", return_value=tmp_path / "nope"
            ):
                reset_config()
                load_environment()
                assert os.environ.get("GLOBAL_VAR") == "global_value"
                # Cleanup
                os.environ.pop("GLOBAL_VAR", None)

    def test_load_environment_local_overrides(self, tmp_path):
        """Local .env overrides global .env."""
        global_env = tmp_path / "global.env"
        global_env.write_text("TEST_VAR=global\n")
        local_env = tmp_path / "local.env"
        local_env.write_text("TEST_VAR=local\n")

        with mock.patch("agent13.config.get_global_env_file", return_value=global_env):
            with mock.patch(
                "agent13.config.get_local_env_file", return_value=local_env
            ):
                reset_config()
                load_environment()
                assert os.environ.get("TEST_VAR") == "local"
                # Cleanup
                os.environ.pop("TEST_VAR", None)

    def test_load_environment_idempotent(self, tmp_path):
        """load_environment only loads once."""
        global_env = tmp_path / ".env"
        global_env.write_text("ONCE_VAR=value\n")

        with mock.patch("agent13.config.get_global_env_file", return_value=global_env):
            with mock.patch(
                "agent13.config.get_local_env_file", return_value=tmp_path / "nope"
            ):
                reset_config()
                load_environment()
                load_environment()  # Second call should be no-op
                assert os.environ.get("ONCE_VAR") == "value"
                # Cleanup
                os.environ.pop("ONCE_VAR", None)


class TestGlobalConfig:
    """Tests for global config functions."""

    def test_get_config_no_file(self, tmp_path):
        """get_config returns empty config if no file exists."""
        with mock.patch(
            "agent13.config.get_config_file", return_value=tmp_path / "missing.toml"
        ):
            with mock.patch("agent13.config.ensure_default_config"):
                reset_config()
                config = get_config()
                assert config.providers == []

    def test_get_provider_from_global(self, tmp_path):
        """get_provider uses global config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            reset_config()
            provider = get_provider("test")
            assert provider is not None
            assert provider.api_base == "https://api.example.com/v1"

    def test_reset_config(self, tmp_path):
        """reset_config clears cached config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = ""
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            reset_config()
            config1 = get_config()
            assert len(config1.providers) == 1

            # Change the file
            config_file.write_text("""
[[providers]]
name = "other"
api_base = "https://other.example.com/v1"
api_key_env_var = ""
""")
            # Without reset, cached config is returned
            config2 = get_config()
            assert config2.providers[0].name == "test"

            # After reset, new config is loaded
            reset_config()
            config3 = get_config()
            assert config3.providers[0].name == "other"


class TestResolveProviderArg:
    """Tests for resolve_provider_arg function."""

    def test_resolve_url(self):
        """Direct URL returns the URL with OPENAI_API_KEY."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            reset_config()
            base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                "http://localhost:8012/v1"
            )
            assert base_url == "http://localhost:8012/v1"
            assert api_key == "test-key"
            assert read_timeout == 2400.0
            assert connect_timeout == 30.0

    def test_resolve_url_https(self):
        """HTTPS URL works too."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            reset_config()
            base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                "https://api.example.com/v1"
            )
            assert base_url == "https://api.example.com/v1"
            assert api_key == "test-key"
            assert read_timeout == 2400.0
            assert connect_timeout == 30.0

    def test_resolve_provider_name(self, tmp_path):
        """Provider name looks up config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            with mock.patch.dict(os.environ, {"TEST_KEY": "secret-key"}):
                reset_config()
                base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                    "test"
                )
                assert base_url == "https://api.example.com/v1"
                assert api_key == "secret-key"
                assert read_timeout == 2400.0
                assert connect_timeout == 30.0

    def test_resolve_provider_no_key(self, tmp_path):
        """Provider with no api_key_env_var uses 'none'."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "local"
api_base = "http://localhost:8012/v1"
api_key_env_var = ""
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            reset_config()
            base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                "local"
            )
            assert base_url == "http://localhost:8012/v1"
            assert api_key == "none"
            assert read_timeout == 2400.0
            assert connect_timeout == 30.0

    def test_resolve_provider_not_found(self, tmp_path):
        """Unknown provider name raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            reset_config()
            with pytest.raises(ValueError, match="Provider 'unknown' not found"):
                resolve_provider_arg("unknown")

    def test_resolve_provider_missing_key(self, tmp_path):
        """Provider with missing env var raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = "MISSING_KEY"
""")
        # Ensure MISSING_KEY is not set
        os.environ.pop("MISSING_KEY", None)
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            reset_config()
            with pytest.raises(ValueError, match="MISSING_KEY not found"):
                resolve_provider_arg("test")

    def test_resolve_url_missing_key(self):
        """URL without OPENAI_API_KEY raises ValueError."""
        # Ensure OPENAI_API_KEY is not set
        os.environ.pop("OPENAI_API_KEY", None)
        reset_config()
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            resolve_provider_arg("http://localhost:8012/v1")

    def test_resolve_provider_with_read_timeout(self, tmp_path):
        """Provider with custom read_timeout returns it."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "slow-model"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
read_timeout = 2400
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            with mock.patch.dict(os.environ, {"TEST_KEY": "secret-key"}):
                reset_config()
                base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                    "slow-model"
                )
                assert base_url == "https://api.example.com/v1"
                assert api_key == "secret-key"
                assert read_timeout == 2400.0
                assert connect_timeout == 30.0

    def test_resolve_provider_default_read_timeout(self, tmp_path):
        """Provider without read_timeout uses default 2400."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[[providers]]
name = "test"
api_base = "https://api.example.com/v1"
api_key_env_var = "TEST_KEY"
""")
        with mock.patch("agent13.config.get_config_file", return_value=config_file):
            with mock.patch.dict(os.environ, {"TEST_KEY": "secret-key"}):
                reset_config()
                base_url, api_key, read_timeout, connect_timeout = resolve_provider_arg(
                    "test"
                )
                assert read_timeout == 2400.0
                assert connect_timeout == 30.0


class TestCreateClient:
    """Tests for create_client function."""

    def test_create_client_default_timeout(self):
        """create_client with default timeout uses 2400s read, 30s connect."""
        client = create_client("https://api.example.com/v1", "test-key")
        # SDK appends trailing slash to base_url
        assert str(client.base_url).startswith("https://api.example.com/v1")
        # Verify timeout is set
        timeout = client._client.timeout
        assert timeout.read == 2400.0
        assert timeout.connect == 30.0

    def test_create_client_custom_timeout(self):
        """create_client with custom read_timeout and connect_timeout passes them through."""
        client = create_client(
            "https://api.example.com/v1",
            "test-key",
            read_timeout=4800.0,
            connect_timeout=60.0,
        )
        assert str(client.base_url).startswith("https://api.example.com/v1")
        timeout = client._client.timeout
        assert timeout.read == 4800.0
        assert timeout.connect == 60.0

    def test_create_client_with_placeholder_key(self):
        """create_client works with placeholder API key (local servers use 'none')."""
        client = create_client("http://localhost:8012/v1", "none")
        assert str(client.base_url).startswith("http://localhost:8012/v1")
        timeout = client._client.timeout
        assert timeout.read == 2400.0
