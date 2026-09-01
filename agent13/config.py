"""Configuration management for agent13.

Supports:
- TOML-based provider configuration (~/.agent13/config.toml)
- Environment variable loading (~/.env then ./.env)
- Direct URL/key initialization (no config required)
"""

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional, Literal
import os
import re
import sys
import urllib.parse

import httpx

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from dotenv import load_dotenv

from agent13.fileio import ConfigFileError, read_text_robust
from agent13.config_paths import (
    get_config_file,
    get_global_env_file,
    get_local_env_file,
    ensure_config_dir,
)

# Default config bundled with the package
DEFAULT_CONFIG_FILE = Path(__file__).parent / "default_config.toml"


def ensure_default_config() -> None:
    """Copy default config to user's config directory if it doesn't exist.

    This provides a starter config for new users.
    """
    config_file = get_config_file()

    # If config already exists, don't overwrite
    if config_file.exists():
        return

    # Check if we have a default config to copy
    if not DEFAULT_CONFIG_FILE.exists():
        return

    # Ensure config directory exists
    ensure_config_dir()

    # Copy default config
    try:
        config_file.write_text(DEFAULT_CONFIG_FILE.read_text())
    except OSError as e:
        # Log warning but don't fail
        import logging

        logging.getLogger(__name__).warning("Failed to copy default config: %s", e)


# Default .env bundled with the package
DEFAULT_ENV_FILE = Path(__file__).parent / "default_env"


def ensure_default_env() -> None:
    """Copy default .env to the user's home directory if none exists.

    Mirrors ensure_default_config(): provides a starter .env for new users.
    Only ~/.env is created (never ./.env); existing files are never overwritten.
    """
    env_file = get_global_env_file()

    # If .env already exists, don't overwrite
    if env_file.exists():
        return

    # Check if we have a default .env to copy
    if not DEFAULT_ENV_FILE.exists():
        return

    # Copy default .env
    try:
        env_file.write_text(DEFAULT_ENV_FILE.read_text())
    except OSError as e:
        # Log warning but don't fail
        import logging

        logging.getLogger(__name__).warning("Failed to copy default .env: %s", e)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.

    Attributes:
        name: Human-readable provider name (e.g., "openrouter", "local")
        api_base: Base URL for the API (e.g., "https://openrouter.ai/api/v1")
        api_key_env_var: Environment variable name for the API key
                         Empty string means no key required (e.g., local servers)
        read_timeout: Read timeout in seconds for the OpenAI client.
                      Controls how long to wait for tokens during streaming.
                      Default 2400s (40 minutes). Reasoning models (GLM, o1,
                      DeepSeek-R1) may think for long periods between tokens.
        connect_timeout: Connection timeout in seconds for the OpenAI client.
                         Controls how long to wait for initial connection.
                         Default 30s. Increase for slow or remote servers.
    """

    name: str
    api_base: str
    api_key_env_var: str = ""
    read_timeout: float = 2400.0
    connect_timeout: float = 30.0

    def get_api_key(self) -> Optional[str]:
        """Get the API key from the environment variable.

        Returns None if api_key_env_var is empty or the variable is not set.
        """
        if not self.api_key_env_var:
            return None
        return os.environ.get(self.api_key_env_var)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        name: Server name (no spaces or slashes)
        transport: "stdio" or "http"
        url: URL for HTTP transport
        command: Command for stdio transport
        args: Arguments for stdio transport
        env: Environment variables for stdio transport
        enabled_tools: Whitelist of tools (empty = all enabled)
        disabled_tools: Tools to exclude
        connect_timeout: Connection timeout in seconds
        tool_timeout: Tool execution timeout in seconds
        retry_attempts: Number of connection retry attempts
        retry_delay: Initial retry delay in seconds
    """

    name: str
    transport: Literal["stdio", "http"]
    url: Optional[str] = None
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    connect_timeout: float = 240.0
    tool_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []

        if not self.name or "/" in self.name or " " in self.name:
            errors.append(f"Invalid server name: {self.name}")

        if self.transport not in ("stdio", "http"):
            errors.append(f"Invalid transport: {self.transport}")

        if self.transport == "http":
            if not self.url:
                errors.append("HTTP transport requires 'url'")
            elif not self.url.startswith(("http://", "https://")):
                errors.append(f"Invalid URL scheme: {self.url}")

        if self.transport == "stdio":
            if not self.command:
                errors.append("stdio transport requires 'command'")

        # Check for overlapping enabled/disabled
        overlap = set(self.enabled_tools) & set(self.disabled_tools)
        if overlap:
            errors.append(f"Tools in both enabled and disabled: {overlap}")

        return errors


@dataclass
class VisionConfig:
    """Configuration for vision/image support.

    Controls how images returned by tools are routed to a vision model:
    - sidecar: a separate vision model (e.g. oMLX/Gemma) describes the image
    - native: images are injected inline as image_url content blocks
    - auto: checks the current model name against patterns to decide

    Attributes:
        mode: Routing strategy - "sidecar", "native", or "auto"
        sidecar_provider: Name of existing ProviderConfig for the sidecar
        sidecar_model: Specific model to request from sidecar (empty = server default)
        native_vision_patterns: Glob patterns for auto-detecting vision-capable models
    """

    mode: str = "sidecar"
    sidecar_provider: str = ""
    sidecar_model: str = ""
    native_vision_patterns: list[str] = field(default_factory=list)

    def is_native_vision(self, model_name: str) -> bool:
        """Check if a model name matches any native vision pattern (glob)."""
        from fnmatch import fnmatch

        lower = model_name.lower()
        return any(fnmatch(lower, p.lower()) for p in self.native_vision_patterns)

    def should_use_native(self, model_name: str) -> bool:
        """Determine routing for auto mode.

        Returns:
            True if images should be injected inline (native),
            False if they should go through the sidecar.
        """
        if self.mode == "native":
            return True
        if self.mode == "sidecar":
            return False
        # auto
        return self.is_native_vision(model_name)


@dataclass
class Config:
    """Agent configuration loaded from TOML file.

    Attributes:
        providers: List of available LLM providers
        mcp_servers: List of MCP server configurations
        skill_paths: Additional paths to search for skills (highest priority)
        include_skills: Whether to include skills in system prompt
        update_check_enabled: Whether to check for updates on startup
        update_check_interval_hours: Minimum hours between update checks
        clipboard_method: Clipboard method - "osc52" or "system"
        saves_location: Auto-save location - "central" (global) or "local" (project dir)
        cursor_blink: Whether the input cursor blinks (default false for tmux activity silence)
    """

    providers: list[ProviderConfig] = field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    skill_paths: list[Path] = field(default_factory=list)
    include_skills: bool = False
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    update_check_enabled: bool = True
    update_check_interval_hours: float = 24
    clipboard_method: str = "osc52"
    saves_location: str = "local"
    bell_threshold: float = 0.0  # Seconds before bell on long turns (0 = always)
    bell_enabled: bool = True  # Whether bell is active
    bell_command: str = ""  # External command to run instead of terminal bell
    cursor_blink: bool = False  # Whether input cursor blinks (false = tmux-friendly)
    auto_compact_threshold: int = 0  # Token threshold for auto-compact (0 = disabled)
    auto_compact_max_iterations: int = 3  # Max compact-and-continue cycles per turn
    vision: Optional[VisionConfig] = None  # Vision/image routing config

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from a TOML file.

        Args:
            path: Path to config file. Defaults to ~/.agent13/config.toml

        Returns:
            Config object with loaded providers

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if path is None:
            path = get_config_file()

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # read_text_robust transparently handles BOMs / UTF-16 / ANSI files
        # (Notepad on Windows can save any of these). Raises ConfigFileError
        # with an actionable message if the bytes are undecodable.
        try:
            data = tomllib.loads(read_text_robust(path))
        except tomllib.TOMLDecodeError as e:
            raise ConfigFileError(
                f"{path} is not valid TOML: {e}. "
                f"If you edited it in a text editor, re-save it as UTF-8 "
                f"(in Notepad: Save As -> Encoding -> UTF-8)."
            ) from e

        config = cls()

        # Parse providers
        providers_data = data.get("providers", [])
        if not isinstance(providers_data, list):
            raise ValueError("'providers' must be a list")

        for i, provider_data in enumerate(providers_data):
            if not isinstance(provider_data, dict):
                raise ValueError(f"Provider {i} must be a table/dict")

            name = provider_data.get("name")
            if not name:
                raise ValueError(f"Provider {i} missing 'name' field")
            if not isinstance(name, str):
                raise ValueError(f"Provider {i} 'name' must be a string")

            api_base = provider_data.get("api_base")
            if not api_base:
                raise ValueError(f"Provider '{name}' missing 'api_base' field")
            if not isinstance(api_base, str):
                raise ValueError(f"Provider '{name}' 'api_base' must be a string")

            api_key_env_var = provider_data.get("api_key_env_var", "")
            if not isinstance(api_key_env_var, str):
                raise ValueError(
                    f"Provider '{name}' 'api_key_env_var' must be a string"
                )

            read_timeout = provider_data.get("read_timeout", 2400.0)
            if not isinstance(read_timeout, (int, float)):
                raise ValueError(f"Provider '{name}' 'read_timeout' must be a number")

            connect_timeout = provider_data.get("connect_timeout", 30.0)
            if not isinstance(connect_timeout, (int, float)):
                raise ValueError(
                    f"Provider '{name}' 'connect_timeout' must be a number"
                )

            config.providers.append(
                ProviderConfig(
                    name=name,
                    api_base=api_base,
                    api_key_env_var=api_key_env_var,
                    read_timeout=float(read_timeout),
                    connect_timeout=float(connect_timeout),
                )
            )

        # Parse MCP servers
        mcp_data = data.get("mcp_servers", [])
        if not isinstance(mcp_data, list):
            raise ValueError("'mcp_servers' must be a list")

        for i, server_data in enumerate(mcp_data):
            if not isinstance(server_data, dict):
                raise ValueError(f"MCP server {i} must be a table/dict")

            config_obj = MCPServerConfig(
                name=server_data.get("name"),
                transport=server_data.get("transport"),
                url=server_data.get("url"),
                command=server_data.get("command"),
                args=server_data.get("args", []),
                env=server_data.get("env", {}),
                enabled_tools=server_data.get("enabled_tools", []),
                disabled_tools=server_data.get("disabled_tools", []),
                connect_timeout=server_data.get("connect_timeout", 300.0),
                tool_timeout=server_data.get("tool_timeout", 60.0),
                retry_attempts=server_data.get("retry_attempts", 3),
                retry_delay=server_data.get("retry_delay", 1.0),
            )

            # Validate and collect errors
            errors = config_obj.validate()
            if errors:
                raise ValueError(
                    f"MCP server '{config_obj.name}' config errors: {errors}"
                )

            config.mcp_servers.append(config_obj)

        # Parse skill_paths (additional paths to search for skills)
        skill_paths_data = data.get("skill_paths", [])
        if isinstance(skill_paths_data, list):
            for path_str in skill_paths_data:
                if isinstance(path_str, str):
                    config.skill_paths.append(Path(path_str).expanduser())

        # Parse include_skills flag
        config.include_skills = data.get("include_skills", False)

        # Parse global tool filters
        enabled = data.get("enabled_tools", [])
        if isinstance(enabled, list):
            config.enabled_tools = [str(p) for p in enabled]
        disabled = data.get("disabled_tools", [])
        if isinstance(disabled, list):
            config.disabled_tools = [str(p) for p in disabled]

        # Parse [updates] section
        updates_data = data.get("updates", {})
        if isinstance(updates_data, dict):
            config.update_check_enabled = updates_data.get(
                "check_enabled", True
            )
            config.update_check_interval_hours = float(
                updates_data.get("check_interval_hours", 24)
            )

        # Parse [clipboard] section
        clipboard_data = data.get("clipboard", {})
        if isinstance(clipboard_data, dict):
            method = clipboard_data.get("method", "osc52")
            if isinstance(method, str) and method in ("osc52", "system"):
                config.clipboard_method = method

        # Parse [saves] section
        saves_data = data.get("saves", {})
        if isinstance(saves_data, dict):
            loc = saves_data.get("location", "local")
            if isinstance(loc, str) and loc in ("central", "local"):
                config.saves_location = loc

        # Parse [bell] section
        bell_data = data.get("bell", {})
        if isinstance(bell_data, dict):
            bt = bell_data.get("threshold", 0)
            if isinstance(bt, (int, float)):
                config.bell_threshold = float(bt)
            be = bell_data.get("enabled", True)
            if isinstance(be, bool):
                config.bell_enabled = be
            bc = bell_data.get("command", "")
            if isinstance(bc, str):
                config.bell_command = bc

        # Parse [auto_compact] section
        auto_compact_data = data.get("auto_compact", {})
        if isinstance(auto_compact_data, dict):
            ac_threshold = auto_compact_data.get("threshold", 0)
            if isinstance(ac_threshold, (int, float)) and ac_threshold >= 0:
                config.auto_compact_threshold = int(ac_threshold)
            ac_max = auto_compact_data.get("max_iterations", 3)
            if isinstance(ac_max, (int, float)) and ac_max >= 1:
                config.auto_compact_max_iterations = int(ac_max)

        # Parse [ui] section
        ui_data = data.get("ui", {})
        if isinstance(ui_data, dict):
            cb = ui_data.get("cursor_blink", False)
            if isinstance(cb, bool):
                config.cursor_blink = cb

        # Parse [vision] section
        vision_data = data.get("vision")
        if isinstance(vision_data, dict):
            mode = vision_data.get("mode", "sidecar")
            if not isinstance(mode, str) or mode not in ("sidecar", "native", "auto"):
                raise ValueError(f"[vision] 'mode' must be 'sidecar', 'native', or 'auto', got '{mode}'")
            sidecar_provider = vision_data.get("sidecar_provider", "")
            if not isinstance(sidecar_provider, str):
                raise ValueError("[vision] 'sidecar_provider' must be a string")
            sidecar_model = vision_data.get("sidecar_model", "")
            if not isinstance(sidecar_model, str):
                raise ValueError("[vision] 'sidecar_model' must be a string")
            patterns = vision_data.get("native_vision_patterns", [])
            if not isinstance(patterns, list):
                raise ValueError("[vision] 'native_vision_patterns' must be a list")
            config.vision = VisionConfig(
                mode=mode,
                sidecar_provider=sidecar_provider,
                sidecar_model=sidecar_model,
                native_vision_patterns=[str(p) for p in patterns],
            )

        config.validate()
        return config

    @classmethod
    def from_file_or_empty(cls, path: Optional[Path] = None) -> "Config":
        """Load config from file, or return empty config if file doesn't exist.

        This is useful for graceful degradation when config is optional.
        """
        if path is None:
            path = get_config_file()

        # Ensure default config exists before checking
        ensure_default_config()

        if not path.exists():
            return cls()

        return cls.from_file(path)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider by name.

        Args:
            name: Provider name to look up

        Returns:
            ProviderConfig if found, None otherwise
        """
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        seen_names = set()
        for provider in self.providers:
            # Check for duplicate names
            if provider.name in seen_names:
                raise ValueError(f"Duplicate provider name: {provider.name}")
            seen_names.add(provider.name)

            # Validate name format (alphanumeric, underscore, hyphen)
            if not re.match(r"^[a-zA-Z0-9_-]+$", provider.name):
                raise ValueError(
                    f"Provider name '{provider.name}' must be alphanumeric with underscores/hyphens only"
                )

            # Validate api_base is a valid URL
            try:
                parsed = urllib.parse.urlparse(provider.api_base)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(
                        f"Provider '{provider.name}' has invalid api_base URL: {provider.api_base}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Provider '{provider.name}' has invalid api_base: {e}"
                )


# Global config instance (loaded lazily)
_config: Optional[Config] = None
_environment_loaded = False


def load_environment() -> None:
    """Load environment variables from .env files.

    Loads in order (later overrides earlier):
    1. ~/.env (global)
    2. ./.env (local, overrides global)

    This is safe to call multiple times - it will only load once.
    """
    global _environment_loaded

    if _environment_loaded:
        return

    # Ensure a starter ~/.env exists (mirrors ensure_default_config)
    ensure_default_env()

    # A corrupt .env is not fatal (it only holds API keys) - warn and
    # continue rather than blocking startup. read_text_robust transparently
    # repairs BOM / UTF-16 / ANSI files that Notepad may have produced.
    for env_path, override in (
        (get_global_env_file(), False),
        (get_local_env_file(), True),
    ):
        if not env_path.exists():
            continue
        try:
            text = read_text_robust(env_path)
        except ConfigFileError as e:
            print(f"warning: {e}", file=sys.stderr)
            continue
        load_dotenv(stream=StringIO(text), override=override)

    _environment_loaded = True


def get_config() -> Config:
    """Get the global configuration, loading if necessary.

    Returns an empty config if no config file exists.
    """
    global _config

    if _config is None:
        load_environment()
        # Corrupt/invalid config raises ConfigFileError (or ValueError for
        # structural problems) - the CLI catches it and exits cleanly.
        _config = Config.from_file_or_empty()

    return _config


def get_provider(name: str) -> Optional[ProviderConfig]:
    """Get a provider by name from the global config.

    Args:
        name: Provider name to look up

    Returns:
        ProviderConfig if found, None otherwise
    """
    return get_config().get_provider(name)


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config, _environment_loaded
    _config = None
    _environment_loaded = False


def resolve_provider_arg(provider_arg: str) -> tuple[str, str, float, float]:
    """Resolve a provider argument to (base_url, api_key, read_timeout, connect_timeout).

    Handles both direct URLs and provider names from config.

    Args:
        provider_arg: Either a URL (e.g., "http://localhost:8012/v1")
                     or a provider name from config (e.g., "studio", "openrouter")

    Returns:
        Tuple of (base_url, api_key, read_timeout, connect_timeout)
        read_timeout is the configured read timeout in seconds for the
        OpenAI client (default 2400.0 for direct URLs).
        connect_timeout is the configured connect timeout in seconds
        (default 30.0 for direct URLs).

    Raises:
        ValueError: If provider name not found in config
        ValueError: If required API key environment variable not set
    """
    # Check if it looks like a URL
    if provider_arg.startswith("http://") or provider_arg.startswith("https://"):
        # Direct URL - use OPENAI_API_KEY
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return provider_arg, api_key, 2400.0, 30.0

    # Must be a provider name - look up in config
    provider = get_provider(provider_arg)
    if not provider:
        available = [p.name for p in get_config().providers]
        if available:
            raise ValueError(
                f"Provider '{provider_arg}' not found in config. "
                f"Available providers: {', '.join(available)}"
            )
        else:
            raise ValueError(
                f"Provider '{provider_arg}' not found in config. "
                f"No providers configured in ~/.agent13/config.toml"
            )

    # Get API key from environment
    api_key = provider.get_api_key()
    if provider.api_key_env_var and not api_key:
        raise ValueError(f"{provider.api_key_env_var} not found in environment")

    # Use placeholder if no key required (local servers)
    if not api_key:
        api_key = "none"

    return provider.api_base, api_key, provider.read_timeout, provider.connect_timeout


def create_client(
    base_url: str,
    api_key: str,
    read_timeout: float = 2400.0,
    connect_timeout: float = 30.0,
):
    """Create an AsyncOpenAI client with configured timeout.

    Args:
        base_url: API base URL
        api_key: API key string
        read_timeout: Read timeout in seconds (default 2400s = 40 min).
                      Reasoning models (GLM, o1, DeepSeek-R1) may think for
                      long periods between tokens.
        connect_timeout: Connection timeout in seconds (default 30s).
                         Increase for slow or remote servers.

    Returns:
        Configured AsyncOpenAI client instance
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=600.0,
            pool=600.0,
        ),
    )


def resolve_provider_selection(
    choice: str, use_stderr: bool = False
) -> str | None:
    """Resolve a provider selection by number, name, or URL.

    Args:
        choice: User's choice (number like "1", name/partial name, or URL)
        use_stderr: If True, print errors to stderr instead of stdout

    Returns:
        Resolved provider name/URL, or None if ambiguous/not found
    """
    # URLs pass through directly (resolve_provider_arg handles them)
    if choice.startswith("http://") or choice.startswith("https://"):
        return choice

    from agent13.models import resolve_from_list

    import sys

    output = sys.stderr if use_stderr else sys.stdout
    names = get_provider_names()
    return resolve_from_list(names, choice, label="provider", output=output)


def get_provider_names() -> list[str]:
    """Get list of configured provider names.

    Returns:
        Sorted list of provider names from config.
    """
    return sorted(p.name for p in get_config().providers)


def print_provider_list(current: str = "") -> None:
    """Print a numbered list of configured providers.

    Args:
        current: Currently active provider name (shown with *).
    """
    names = get_provider_names()
    if not names:
        print("  No providers configured in ~/.agent13/config.toml")
        return
    print("\nAvailable providers:")
    for i, name in enumerate(names, 1):
        marker = " *" if name == current else ""
        print(f"  {i}. {name}{marker}")
    print()
    print("  Use /provider <name> to switch")
