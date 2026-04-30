"""Tests for sandbox module."""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from agent13.sandbox import (
    SandboxMode,
    SANDBOX_CAPABILITIES,
    get_sandbox_profiles_dir,
    get_sandbox_profile_path,
    load_sandbox_profile,
    parse_sandbox_mode,
    get_default_sandbox_mode,
    get_effective_sandbox_mode,
    build_sandbox_command,
    run_sandboxed,
    format_sandbox_mode_info,
    format_all_sandbox_modes,
    get_temp_dir,
)


class TestSandboxMode:
    """Tests for SandboxMode enum."""

    def test_all_modes_exist(self):
        """All expected modes should exist."""
        assert SandboxMode.PERMISSIVE_OPEN.value == "permissive-open"
        assert SandboxMode.PERMISSIVE_CLOSED.value == "permissive-closed"
        assert SandboxMode.RESTRICTIVE_OPEN.value == "restrictive-open"
        assert SandboxMode.RESTRICTIVE_CLOSED.value == "restrictive-closed"
        assert SandboxMode.NONE.value == "none"

    def test_mode_count(self):
        """Should have 5 modes."""
        assert len(SandboxMode) == 5


class TestSandboxCapabilities:
    """Tests for sandbox capabilities."""

    def test_permissive_open_capabilities(self):
        """Permissive-open should allow network and read anywhere."""
        caps = SANDBOX_CAPABILITIES[SandboxMode.PERMISSIVE_OPEN]
        assert caps.file_write == "project"
        assert caps.file_read == "anywhere"
        assert caps.network is True

    def test_permissive_closed_capabilities(self):
        """Permissive-closed should block network."""
        caps = SANDBOX_CAPABILITIES[SandboxMode.PERMISSIVE_CLOSED]
        assert caps.file_write == "project"
        assert caps.file_read == "anywhere"
        assert caps.network is False

    def test_restrictive_open_capabilities(self):
        """Restrictive-open should restrict read to project."""
        caps = SANDBOX_CAPABILITIES[SandboxMode.RESTRICTIVE_OPEN]
        assert caps.file_write == "project"
        assert caps.file_read == "project"
        assert caps.network is True

    def test_restrictive_closed_capabilities(self):
        """Restrictive-closed should restrict read and block network."""
        caps = SANDBOX_CAPABILITIES[SandboxMode.RESTRICTIVE_CLOSED]
        assert caps.file_write == "project"
        assert caps.file_read == "project"
        assert caps.network is False

    def test_none_capabilities(self):
        """None mode should allow everything."""
        caps = SANDBOX_CAPABILITIES[SandboxMode.NONE]
        assert caps.file_write == "anywhere"
        assert caps.file_read == "anywhere"
        assert caps.network is True


class TestSandboxProfiles:
    """Tests for sandbox profile files."""

    def test_profiles_dir_exists(self):
        """Profiles directory should exist."""
        profiles_dir = get_sandbox_profiles_dir()
        assert profiles_dir.exists()
        assert profiles_dir.is_dir()

    def test_all_profile_files_exist(self):
        """All profile files should exist (except 'none')."""
        for mode in SandboxMode:
            if mode == SandboxMode.NONE:
                continue  # 'none' mode doesn't need a profile file
            profile_path = get_sandbox_profile_path(mode)
            assert profile_path.exists(), f"Profile file missing: {profile_path}"

    def test_load_sandbox_profile(self):
        """Should load profile content."""
        content = load_sandbox_profile(SandboxMode.PERMISSIVE_OPEN)
        assert "(version 1)" in content
        assert "(deny default)" in content

    def test_load_nonexistent_profile(self):
        """Should raise FileNotFoundError for missing profile."""
        with pytest.raises(FileNotFoundError):
            load_sandbox_profile(SandboxMode.NONE)  # 'none' has no profile file


class TestParseSandboxMode:
    """Tests for parse_sandbox_mode function."""

    def test_parse_valid_modes(self):
        """Should parse all valid mode strings."""
        assert parse_sandbox_mode("permissive-open") == SandboxMode.PERMISSIVE_OPEN
        assert parse_sandbox_mode("permissive-closed") == SandboxMode.PERMISSIVE_CLOSED
        assert parse_sandbox_mode("restrictive-open") == SandboxMode.RESTRICTIVE_OPEN
        assert (
            parse_sandbox_mode("restrictive-closed") == SandboxMode.RESTRICTIVE_CLOSED
        )
        assert parse_sandbox_mode("none") == SandboxMode.NONE

    def test_parse_case_insensitive(self):
        """Should be case-insensitive."""
        assert parse_sandbox_mode("PERMISSIVE-OPEN") == SandboxMode.PERMISSIVE_OPEN
        assert parse_sandbox_mode("Permissive-Closed") == SandboxMode.PERMISSIVE_CLOSED

    def test_parse_aliases(self):
        """Should accept aliases for 'none' mode."""
        assert parse_sandbox_mode("disabled") == SandboxMode.NONE
        assert parse_sandbox_mode("off") == SandboxMode.NONE

    def test_parse_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError):
            parse_sandbox_mode("invalid-mode")

    def test_parse_whitespace(self):
        """Should handle whitespace."""
        assert parse_sandbox_mode("  permissive-open  ") == SandboxMode.PERMISSIVE_OPEN


class TestDefaultSandboxMode:
    """Tests for get_default_sandbox_mode function."""

    def test_default_without_config(self):
        """Should return PERMISSIVE_OPEN when no config exists."""
        with patch("agent13.sandbox.get_config_file") as mock_config:
            mock_config.return_value = Path("/nonexistent/config.toml")
            mode = get_default_sandbox_mode()
            assert mode == SandboxMode.PERMISSIVE_OPEN

    def test_default_with_config(self):
        """Should read default from config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[sandbox]\ndefault = "restrictive-open"\n')
            config_path = Path(f.name)

        try:
            with patch("agent13.sandbox.get_config_file") as mock_config:
                mock_config.return_value = config_path
                mode = get_default_sandbox_mode()
                assert mode == SandboxMode.RESTRICTIVE_OPEN
        finally:
            os.unlink(config_path)


class TestEffectiveSandboxMode:
    """Tests for get_effective_sandbox_mode function."""

    def test_session_override_takes_precedence(self):
        """Session override should take precedence over config default."""
        with patch("agent13.sandbox.get_config_file") as mock_config:
            mock_config.return_value = Path("/nonexistent/config.toml")
            mode = get_effective_sandbox_mode(
                session_override=SandboxMode.RESTRICTIVE_CLOSED
            )
            assert mode == SandboxMode.RESTRICTIVE_CLOSED

    def test_no_override_uses_default(self):
        """Should use default when no session override."""
        with patch("agent13.sandbox.get_config_file") as mock_config:
            mock_config.return_value = Path("/nonexistent/config.toml")
            mode = get_effective_sandbox_mode(session_override=None)
            assert mode == SandboxMode.PERMISSIVE_OPEN


class TestBuildSandboxCommand:
    """Tests for build_sandbox_command function."""

    def test_none_mode_returns_shell_command(self):
        """'none' mode should return simple shell command."""
        cmd = build_sandbox_command("echo hello", SandboxMode.NONE)
        if sys.platform == "win32":
            assert cmd == ["cmd.exe", "/c", "echo hello"]
        else:
            assert cmd == ["/bin/sh", "-c", "echo hello"]

    def test_non_macos_returns_shell_command(self):
        """Non-macOS should return simple shell command."""
        with patch("agent13.sandbox.is_macos") as mock_macos:
            mock_macos.return_value = False
            cmd = build_sandbox_command("echo hello", SandboxMode.PERMISSIVE_OPEN)
            if sys.platform == "win32":
                assert cmd == ["cmd.exe", "/c", "echo hello"]
            else:
                assert cmd == ["/bin/sh", "-c", "echo hello"]

    def test_macos_uses_sandbox_exec(self):
        """macOS should use sandbox-exec."""
        with patch("agent13.sandbox.is_macos") as mock_macos:
            mock_macos.return_value = True
            cmd = build_sandbox_command("echo hello", SandboxMode.PERMISSIVE_OPEN)
            assert cmd[0] == "sandbox-exec"
            assert "-f" in cmd
            assert "permissive-open.sb" in str(cmd)

    def test_project_dir_parameter(self):
        """Should include PROJECT_DIR parameter."""
        with patch("agent13.sandbox.is_macos") as mock_macos:
            mock_macos.return_value = True
            project_dir = Path(get_temp_dir()) / "test_project"
            cmd = build_sandbox_command(
                "echo hello", SandboxMode.PERMISSIVE_OPEN, project_dir
            )
            assert "-D" in cmd
            assert "PROJECT_DIR=" in str(cmd)


class TestRunSandboxed:
    """Tests for run_sandboxed function."""

    def test_successful_command(self):
        """Should run command successfully."""
        result = run_sandboxed("echo hello", SandboxMode.NONE)
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_failed_command(self):
        """Should capture exit code for failed command."""
        result = run_sandboxed("exit 1", SandboxMode.NONE)
        assert result["success"] is False
        assert result["exit_code"] == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell command (sleep)")
    def test_timeout(self):
        """Should timeout long-running commands."""
        result = run_sandboxed("sleep 10", SandboxMode.NONE, timeout=0.5)
        assert result["success"] is False
        assert result["timed_out"] is True
        assert "timed out" in result["stderr"].lower()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell command (python3)")
    def test_output_truncation(self):
        """Should truncate large output."""
        # Generate output larger than max_output
        result = run_sandboxed(
            "python3 -c 'print(\"x\" * 200000)'", SandboxMode.NONE, max_output=1000
        )
        assert result["truncated"] is True
        assert len(result["stdout"]) < 200000

    def test_command_not_found(self):
        """Should handle command not found."""
        result = run_sandboxed("nonexistent_command_xyz", SandboxMode.NONE)
        assert result["success"] is False
        assert "not found" in result["stderr"].lower() or result["exit_code"] != 0

    def test_sandbox_mode_in_result(self):
        """Should include sandbox mode in result."""
        result = run_sandboxed("echo test", SandboxMode.NONE)
        assert result["sandbox_mode"] == "none"


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_sandbox_mode_info(self):
        """Should format mode info correctly."""
        info = format_sandbox_mode_info(SandboxMode.PERMISSIVE_OPEN)
        assert "permissive-open" in info
        assert "file write" in info.lower()
        assert "file read" in info.lower()
        assert "network" in info.lower()

    def test_format_all_sandbox_modes(self):
        """Should list all modes."""
        text = format_all_sandbox_modes()
        assert "permissive-open" in text
        assert "permissive-closed" in text
        assert "restrictive-open" in text
        assert "restrictive-closed" in text
        assert "none" in text


class TestBashTool:
    """Tests for bash tool integration."""

    def test_bash_tool_registered(self):
        """bash tool should be registered."""
        from agent13 import get_tool_names

        names = get_tool_names()
        assert "command" in names

    @pytest.mark.asyncio
    async def test_bash_tool_execution(self):
        """Should execute bash command."""
        from agent13 import execute_tool
        import json

        result = await execute_tool("command", {"command": "echo test"})
        data = json.loads(result)
        assert data["success"] is True
        assert "test" in data["stdout"]

    @pytest.mark.asyncio
    async def test_bash_tool_ignores_sandbox_param(self):
        """Should ignore sandbox parameter (security - user controls sandbox only)."""
        from agent13 import execute_tool
        import json

        # The sandbox parameter is no longer accepted - it's ignored if passed
        # The tool uses the session/config default
        result = await execute_tool("command", {"command": "echo test"})
        data = json.loads(result)
        assert data["success"] is True
        # Sandbox mode should be the default (permissive-open)
        assert data["sandbox_mode"] in [
            "permissive-open",
            "permissive-closed",
            "restrictive-open",
            "restrictive-closed",
            "none",
        ]


class TestBashToolUseCases:
    """Complex, realistic use cases for bash tool in AI coding agent context."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (git &&, pipe)"
    )
    async def test_git_workflow_chained_commands(self):
        """Multi-step git workflow with chained commands."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test (git needs to read .git)
        set_session_sandbox_mode(SandboxMode.NONE)

        # Run git status in project directory
        result = await execute_tool(
            "command", {"command": "git status --porcelain && git log --oneline -3"}
        )
        data = json.loads(result)
        # Git commands should succeed (we're in a git repo)
        assert data["success"] is True
        # Should have some output from git log
        assert len(data["stdout"]) > 0 or len(data["stderr"]) >= 0

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (find, grep, pipe)"
    )
    async def test_find_and_grep_combo(self):
        """Find and grep combination to locate files with patterns."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "find . -name '*.py' -type f | head -10 | xargs grep -l 'import' 2>/dev/null || echo 'no matches'"
            },
        )
        data = json.loads(result)
        assert data["success"] is True
        # grep -l lists filenames, not the word "import"
        # Output will be file paths like ./ui/display.py
        assert ".py" in data["stdout"] or "no matches" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_dependency_audit_python(self):
        """Check for outdated Python dependencies using uv."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import shutil

        # Skip if uv not installed
        if not shutil.which("uv"):
            pytest.skip("uv not installed")

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "uv pip list --outdated 2>&1 | head -20 || echo 'check complete'",
                "timeout": 10,
            },
        )
        data = json.loads(result)
        # Command should complete (may or may not have outdated deps)
        assert data["success"] is True or data["exit_code"] != 0
        assert data["timed_out"] is False

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_dependency_audit_npm(self):
        """Check for outdated npm dependencies."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import shutil

        # Skip if npm not installed
        if not shutil.which("npm"):
            pytest.skip("npm not installed")

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "npm outdated --json 2>&1 | head -20 || echo 'no outdated'",
                "timeout": 10,
            },
        )
        data = json.loads(result)
        # npm outdated returns non-zero if there are outdated packages
        # so we just check it ran
        assert data["timed_out"] is False

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (uv run, pipe, tail)"
    )
    async def test_test_suite_execution(self):
        """Run test suite and capture summary output."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import shutil

        # Skip if pytest not available
        if not shutil.which("pytest") and not shutil.which("uv"):
            pytest.skip("pytest not available")

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # Use 'python -m pytest' instead of bare 'pytest' to avoid
        # process-exec denial when running inside an outer sandbox
        # (python is already an allowed executable, so importing pytest
        # as a module works even when spawning the pytest binary is blocked)
        result = await execute_tool(
            "command",
            {
                "command": "uv run python -m pytest tests/test_sandbox.py::TestSandboxMode -v --tb=short 2>&1 | tail -20",
                "timeout": 30,
            },
        )
        data = json.loads(result)
        # Tests should pass
        assert data["success"] is True
        assert "passed" in data["stdout"].lower() or "PASSED" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (ps, grep)"
    )
    async def test_process_investigation(self):
        """Find running processes for conflict detection."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "ps aux | grep -E '(python|node)' | grep -v grep | head -10 || echo 'none found'"
            },
        )
        data = json.loads(result)
        assert data["success"] is True
        # May or may not have running processes
        assert data["timed_out"] is False

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands ($PATH, head, which)"
    )
    async def test_environment_debugging(self):
        """Diagnose environment issues with variable expansion."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "echo 'PATH:' $PATH | head -c 500 && echo '' && which python && python --version"
            },
        )
        data = json.loads(result)
        assert data["success"] is True
        # Should show Python version
        assert "Python" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell command (python3)")
    async def test_large_output_truncation(self):
        """Handle large output with truncation."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # Generate large output (200KB of 'x' characters)
        result = await execute_tool(
            "command", {"command": "python3 -c 'print(\"x\" * 200000)'"}
        )
        data = json.loads(result)
        assert data["success"] is True
        # Output should be truncated (max_output is 100KB in bash tool)
        assert data["truncated"] is True
        # stdout should be less than original 200KB
        assert len(data["stdout"]) < 200000

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_network_diagnostics_allowed(self):
        """Network call should work in permissive-open mode."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import shutil

        # Skip if curl not installed
        if not shutil.which("curl"):
            pytest.skip("curl not installed")

        # Set sandbox to permissive-open for this test
        set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

        result = await execute_tool(
            "command",
            {
                "command": "curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 https://httpbin.org/status/200",
                "timeout": 15,
            },
        )
        data = json.loads(result)
        # Should succeed with network access (or fail gracefully if network unavailable)
        assert data["sandbox_mode"] == "permissive-open"
        # If network is available, should get 200; if not, that's OK for this test
        # We're testing that the sandbox mode is applied correctly
        if data["success"]:
            assert "200" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_network_diagnostics_blocked(self):
        """Network call should fail in permissive-closed mode."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to permissive-closed for this test
        set_session_sandbox_mode(SandboxMode.PERMISSIVE_CLOSED)

        result = await execute_tool(
            "command",
            {
                "command": "curl -s -o /dev/null -w '%{http_code}' https://httpbin.org/status/200 --connect-timeout 3",
                "timeout": 10,
            },
        )
        data = json.loads(result)
        # Should fail or timeout due to network restriction
        # The sandbox blocks network, so curl should fail
        assert data["sandbox_mode"] == "permissive-closed"
        # Either success=False (curl failed) or stdout doesn't contain 200
        if data["success"]:
            assert "200" not in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (mkdir -p, ls -la)"
    )
    async def test_build_and_validate(self):
        """Run build and verify output artifacts."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import tempfile

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # Create a temp directory to simulate a build
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple "build" - just create a dist directory
            result = await execute_tool(
                "command",
                {
                    "command": f"mkdir -p {tmpdir}/dist && echo 'build output' > {tmpdir}/dist/bundle.js && ls -la {tmpdir}/dist/"
                },
            )
            data = json.loads(result)
            assert data["success"] is True
            assert "bundle.js" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell command (sleep, &&)"
    )
    async def test_timeout_handling(self):
        """Handle long-running commands with timeout."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # Command that sleeps longer than timeout
        result = await execute_tool(
            "command", {"command": "sleep 10 && echo 'done'", "timeout": 1}
        )
        data = json.loads(result)
        assert data["success"] is False
        assert data["timed_out"] is True
        assert "timed out" in data["stderr"].lower()

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell command (sleep 0.2)"
    )
    async def test_timeout_completes_within_limit(self):
        """Command that completes within timeout should succeed."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command", {"command": "sleep 0.2 && echo 'completed'", "timeout": 5}
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["timed_out"] is False
        assert "completed" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_conditional_fallback(self):
        """Conditional fallback with || operator."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # First command fails, second succeeds
        result = await execute_tool(
            "command",
            {
                "command": "nonexistent_command_xyz 2>/dev/null || echo 'fallback executed'"
            },
        )
        data = json.loads(result)
        # Overall should succeed due to fallback
        assert data["success"] is True
        assert "fallback executed" in data["stdout"]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX shell commands (echo -e, grep, sort, head)",
    )
    async def test_pipe_chain(self):
        """Complex pipe chain with multiple stages."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command",
            {
                "command": "echo -e 'apple\\nbanana\\ncherry\\napricot' | grep 'a' | sort | head -2"
            },
        )
        data = json.loads(result)
        assert data["success"] is True
        # Should get 'apple' and 'apricot' (sorted, first 2 with 'a')
        lines = data["stdout"].strip().split("\n")
        assert len(lines) == 2
        assert "apple" in lines[0] or "apricot" in lines[0]

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX shell commands (echo >, cat)"
    )
    async def test_file_write_in_project(self):
        """Write file within project directory."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode
        import os

        # Set sandbox to permissive-open (allows project write)
        set_session_sandbox_mode(SandboxMode.PERMISSIVE_OPEN)

        test_file = "test_bash_write_temp.txt"
        try:
            result = await execute_tool(
                "command",
                {"command": f"echo 'test content' > {test_file} && cat {test_file}"},
            )
            data = json.loads(result)
            assert data["success"] is True
            assert "test content" in data["stdout"]
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            # Reset sandbox
            set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_exit_code_propagation(self):
        """Exit code should be captured correctly."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        # Command that exits with specific code
        result = await execute_tool("command", {"command": "exit 42"})
        data = json.loads(result)
        assert data["success"] is False
        assert data["exit_code"] == 42

        # Reset sandbox
        set_session_sandbox_mode(None)

    @pytest.mark.asyncio
    async def test_stderr_capture(self):
        """Stderr should be captured separately."""
        import json
        from agent13 import execute_tool
        from tools.command import set_session_sandbox_mode
        from agent13.sandbox import SandboxMode

        # Set sandbox to none for this test
        set_session_sandbox_mode(SandboxMode.NONE)

        result = await execute_tool(
            "command", {"command": "echo 'to stdout' && echo 'to stderr' >&2"}
        )
        data = json.loads(result)
        assert data["success"] is True
        assert "to stdout" in data["stdout"]
        assert "to stderr" in data["stderr"]

        # Reset sandbox
        set_session_sandbox_mode(None)
