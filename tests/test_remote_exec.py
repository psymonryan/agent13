"""Tests for remote_exec module (command v2, Phase 2)."""

import asyncio
import base64
import pytest

from agent13.remote_exec import (
    _validate_host,
    _validate_remote_shell,
    _validate_b64_payload,
    build_remote_argv,
    clear_remote_shell_cache,
    detect_remote_shell,
    run_remote_command,
    _remote_shell_cache,
)


class TestValidateHost:
    """Tests for _validate_host."""

    def test_simple_host(self):
        assert _validate_host("myhost") == "myhost"

    def test_user_at_host(self):
        assert _validate_host("user@myhost") == "user@myhost"

    def test_host_with_dots_dashes_underscores(self):
        assert _validate_host("my-host_1.example.com") == "my-host_1.example.com"

    def test_host_with_whitespace(self):
        assert _validate_host("  myhost  ") == "myhost"

    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="Empty host"):
            _validate_host("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty host"):
            _validate_host("   ")

    def test_host_with_shell_metacharacters_raises(self):
        for bad in ["host;rm", "host|cat", "host$(cmd)", "host`cmd`", "host&&x"]:
            with pytest.raises(ValueError, match="Invalid host"):
                _validate_host(bad)

    def test_host_with_port_flag_raises(self):
        with pytest.raises(ValueError, match="Invalid host"):
            _validate_host("-p 2222 host")

    def test_host_with_spaces_raises(self):
        with pytest.raises(ValueError, match="Invalid host"):
            _validate_host("host with spaces")


class TestValidateRemoteShell:
    """Tests for _validate_remote_shell."""

    def test_none_returns_none(self):
        assert _validate_remote_shell(None) is None

    def test_auto_returns_none(self):
        assert _validate_remote_shell("auto") is None

    def test_empty_returns_none(self):
        assert _validate_remote_shell("") is None

    def test_posix_aliases(self):
        for alias in ("posix", "sh", "bash"):
            assert _validate_remote_shell(alias) == "posix"

    def test_powershell_aliases(self):
        for alias in ("powershell", "ps"):
            assert _validate_remote_shell(alias) == "powershell"

    def test_case_insensitive(self):
        assert _validate_remote_shell("POWERSHELL") == "powershell"
        assert _validate_remote_shell("Sh") == "posix"

    def test_whitespace_stripped(self):
        assert _validate_remote_shell("  posix  ") == "posix"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid remote_shell"):
            _validate_remote_shell("zsh")


class TestValidateB64Payload:
    """Tests for _validate_b64_payload."""

    def test_valid_b64(self):
        b64 = base64.b64encode(b"hello world").decode("ascii")
        assert _validate_b64_payload(b64) == b64

    def test_b64_with_padding(self):
        b64 = base64.b64encode(b"hi").decode("ascii")  # aGk=
        assert _validate_b64_payload(b64) == b64

    def test_b64_with_plus_and_slash(self):
        # Find input that produces + and / in base64
        b64 = base64.b64encode(b"\xfb\xff\xfe").decode("ascii")
        assert _validate_b64_payload(b64) == b64

    def test_b64_with_newline_raises(self):
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_b64_payload("aGk=\n")

    def test_b64_with_space_raises(self):
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_b64_payload("aGk =")


class TestBuildRemoteArgv:
    """Tests for build_remote_argv."""

    def test_posix_argv(self):
        argv = build_remote_argv("myhost", "posix")
        assert argv[0] == "ssh"
        assert argv[1] == "myhost"
        # The remote command should contain the fixed safe template
        assert "sh -s" in argv[2]
        assert "base64 -d" in argv[2]

    def test_powershell_argv(self):
        argv = build_remote_argv("myhost", "powershell")
        assert argv[0] == "ssh"
        assert argv[1] == "myhost"
        # The remote command should contain the PowerShell decode template
        assert "FromBase64String" in argv[2]
        assert "powershell" in argv[2]
        assert "-NoProfile" in argv[2]
        assert "-NonInteractive" in argv[2]

    def test_user_at_host(self):
        argv = build_remote_argv("user@myhost", "posix")
        assert argv[1] == "user@myhost"

    def test_invalid_host_raises(self):
        with pytest.raises(ValueError, match="Invalid host"):
            build_remote_argv("host;rm", "posix")

    def test_none_shell_raises(self):
        with pytest.raises(ValueError, match="remote_shell must be"):
            build_remote_argv("myhost", None)

    def test_posix_template_is_fixed_constant(self):
        """The POSIX remote command is a fixed safe constant — no
        interpolation of the payload. The payload travels on stdin."""
        argv1 = build_remote_argv("host1", "posix")
        argv2 = build_remote_argv("host2", "posix")
        # Same remote command regardless of host
        assert argv1[2] == argv2[2]

    def test_powershell_template_is_fixed_constant(self):
        """The PowerShell remote command is a fixed safe constant."""
        argv1 = build_remote_argv("host1", "powershell")
        argv2 = build_remote_argv("host2", "powershell")
        assert argv1[2] == argv2[2]


class TestDetectRemoteShell:
    """Tests for detect_remote_shell (with mocked ssh probe)."""

    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_posix_detection(self):
        """uname returns non-empty stdout => posix."""
        async def mock_probe(host, remote_cmd, timeout):
            assert remote_cmd == "uname"
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "Linux\n",
                "stderr": "",
            }

        from unittest.mock import patch
        with patch("agent13.remote_exec._run_ssh_probe", mock_probe):
            result = await detect_remote_shell("myhost")
        assert result == "posix"
        assert _remote_shell_cache["myhost"] == "posix"

    @pytest.mark.asyncio
    async def test_powershell_detection(self):
        """uname fails => powershell."""
        async def mock_probe(host, remote_cmd, timeout):
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "'uname' is not recognized...",
            }

        from unittest.mock import patch
        with patch("agent13.remote_exec._run_ssh_probe", mock_probe):
            result = await detect_remote_shell("winhost")
        assert result == "powershell"
        assert _remote_shell_cache["winhost"] == "powershell"

    @pytest.mark.asyncio
    async def test_cache_prevents_reprobe(self):
        """Second call for the same host uses the cache."""
        call_count = 0

        async def mock_probe(host, remote_cmd, timeout):
            nonlocal call_count
            call_count += 1
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "Linux\n",
                "stderr": "",
            }

        from unittest.mock import patch
        with patch("agent13.remote_exec._run_ssh_probe", mock_probe):
            await detect_remote_shell("myhost")
            await detect_remote_shell("myhost")
        assert call_count == 1  # Only one probe

    @pytest.mark.asyncio
    async def test_ssh_not_found(self):
        """ssh binary missing => powershell fallback (probe fails)."""
        async def mock_probe(host, remote_cmd, timeout):
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "ssh not found on this machine",
            }

        from unittest.mock import patch
        with patch("agent13.remote_exec._run_ssh_probe", mock_probe):
            result = await detect_remote_shell("myhost")
        # Probe failed => assume powershell (conservative)
        assert result == "powershell"


class TestRunRemoteCommand:
    """Tests for run_remote_command (with mocked subprocess)."""

    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_ssh_child_runs_in_own_process_group(self):
        """Regression: ssh must be spawned with start_new_session=True so a
        timeout _kill_process_tree() cannot SIGKILL the agent's own group."""
        from unittest.mock import patch, AsyncMock, MagicMock

        captured = {}

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        async def fake_read(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read

        async def mock_create_subprocess_exec(*args, **kwargs):
            captured["kwargs"] = kwargs
            return mock_proc

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            await run_remote_command(
                host="myhost",
                command="echo hi",
                remote_shell="posix",
                timeout=10,
            )

        assert captured["kwargs"].get("start_new_session") is True, (
            "ssh must run in its own process group (start_new_session=True), "
            "otherwise a timeout killpg() SIGKILLs the agent"
        )

    @pytest.mark.asyncio
    async def test_posix_command_sends_b64_on_stdin(self):
        """The base64 payload is written to ssh's stdin, not the argv."""
        captured = {}

        class MockStdin:
            def write(self, data):
                captured["stdin"] = data

            def close(self):
                captured["stdin_closed"] = True

        class MockProcess:
            returncode = 0
            stdin = MockStdin()
            pid = 12345

            async def wait(self):
                return 0

            @property
            def stdout(self):
                return self._stdout

            @property
            def stderr(self):
                return self._stderr

        from unittest.mock import patch, AsyncMock, MagicMock

        expected_b64 = base64.b64encode(b"echo hello").decode("ascii")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)

        # Mock stdin
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        # First read returns data, second returns b"" (EOF)
        out_calls = {"n": 0}

        async def fake_read_out2(n):
            out_calls["n"] += 1
            if out_calls["n"] == 1:
                return b"hello from remote\n"
            return b""

        err_calls = {"n": 0}

        async def fake_read_err2(n):
            err_calls["n"] += 1
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out2
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err2

        async def mock_create_subprocess_exec(*args, **kwargs):
            captured["argv"] = list(args)
            captured["kwargs"] = kwargs
            return mock_proc

        with patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            result = await run_remote_command(
                host="myhost",
                command="echo hello",
                remote_shell="posix",
                timeout=10,
            )

        # Verify the payload was written to stdin as base64
        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert written == expected_b64.encode("ascii")
        mock_proc.stdin.close.assert_called_once()

        # Verify argv does NOT contain the payload
        argv = captured["argv"]
        assert argv[0] == "ssh"
        assert argv[1] == "myhost"
        # The remote command is the fixed template, not the payload
        assert "echo hello" not in argv[2]
        assert expected_b64 not in argv[2]

        # Verify result
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello from remote" in result["stdout"]
        assert result["remote"] == "myhost"
        assert result["remote_shell"] == "posix"

    @pytest.mark.asyncio
    async def test_powershell_command(self):
        """PowerShell path: base64 on stdin, fixed PS template in argv."""
        from unittest.mock import patch, AsyncMock, MagicMock

        expected_b64 = base64.b64encode(b"Write-Output 'hi'").decode("ascii")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        out_calls = {"n": 0}

        async def fake_read_out(n):
            out_calls["n"] += 1
            if out_calls["n"] == 1:
                return b"hi\r\n"
            return b""

        async def fake_read_err(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err

        captured = {}

        async def mock_create_subprocess_exec(*args, **kwargs):
            captured["argv"] = list(args)
            return mock_proc

        with patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            result = await run_remote_command(
                host="winhost",
                command="Write-Output 'hi'",
                remote_shell="powershell",
                timeout=10,
            )

        # Verify base64 on stdin
        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert written == expected_b64.encode("ascii")

        # Verify PS template in argv
        argv = captured["argv"]
        assert "FromBase64String" in argv[2]
        assert "Write-Output" not in argv[2]

        assert result["success"] is True
        assert result["remote_shell"] == "powershell"

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        """Non-zero exit code is propagated."""
        from unittest.mock import patch, AsyncMock, MagicMock

        mock_proc = MagicMock()
        mock_proc.returncode = 42
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=42)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        async def fake_read_out(n):
            return b""

        async def fake_read_err(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            result = await run_remote_command(
                host="myhost",
                command="exit 42",
                remote_shell="posix",
                timeout=10,
            )

        assert result["success"] is False
        assert result["exit_code"] == 42

    @pytest.mark.asyncio
    async def test_invalid_host_returns_error(self):
        """Invalid host returns an error dict, not an exception."""
        # _validate_host raises ValueError — run_remote_command should
        # let it propagate (the tool layer will catch it)
        with pytest.raises(ValueError, match="Invalid host"):
            await run_remote_command(
                host="host;rm",
                command="echo hi",
                remote_shell="posix",
            )

    @pytest.mark.asyncio
    async def test_ssh_not_found(self):
        """ssh binary missing returns a clean error."""
        from unittest.mock import patch

        async def mock_create_subprocess_exec(*args, **kwargs):
            raise FileNotFoundError("ssh")

        with patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            result = await run_remote_command(
                host="myhost",
                command="echo hi",
                remote_shell="posix",
                timeout=10,
            )

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "ssh not found" in result["stderr"]
        assert result["remote"] == "myhost"

    @pytest.mark.asyncio
    async def test_auto_detect_uses_cache(self):
        """Auto-detect probes once, then uses the cache."""
        from unittest.mock import patch, AsyncMock, MagicMock

        probe_calls = {"n": 0}

        async def mock_probe(host, remote_cmd, timeout):
            probe_calls["n"] += 1
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "Linux\n",
                "stderr": "",
            }

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        async def fake_read_out(n):
            return b""

        async def fake_read_err(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch("agent13.remote_exec._run_ssh_probe", mock_probe), \
             patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            await run_remote_command(host="myhost", command="echo 1", timeout=10)
            await run_remote_command(host="myhost", command="echo 2", timeout=10)

        assert probe_calls["n"] == 1  # Only one probe for the session

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Large output is truncated."""
        from unittest.mock import patch, AsyncMock, MagicMock

        big_output = b"x" * 200000

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        out_calls = {"n": 0}

        async def fake_read_out(n):
            out_calls["n"] += 1
            if out_calls["n"] == 1:
                return big_output
            return b""

        async def fake_read_err(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch("agent13.remote_exec.asyncio.create_subprocess_exec", mock_create_subprocess_exec):
            result = await run_remote_command(
                host="myhost",
                command="yes",
                remote_shell="posix",
                timeout=10,
                max_output=1000,
            )

        assert result["truncated"] is True
        assert len(result["stdout"]) <= 1000 + 100  # Allow for truncation message
        assert "truncated" in result["stdout"]


class TestEncodingGuarantees:
    """Spec item 3: CRLF normalization + status/output_encoding fields."""

    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @staticmethod
    def _mock_process(returncode=0, stdout=b"", stderr=b""):
        from unittest.mock import AsyncMock, MagicMock

        mock_proc = MagicMock()
        mock_proc.returncode = returncode
        mock_proc.pid = 12345
        mock_proc.wait = AsyncMock(return_value=returncode)
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        async def fake_read_out(n):
            # sleep(0) so the read suspends like a real StreamReader —
            # a never-suspending read would spin the pump synchronously
            # and starve the event loop's timers.
            await asyncio.sleep(0)
            return stdout

        async def fake_read_err(n):
            await asyncio.sleep(0)
            return stderr

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read_out
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read_err
        return mock_proc

    @pytest.mark.asyncio
    async def test_powershell_payload_crlf_normalized(self):
        """LF-only scripts arrive CRLF on a powershell remote (spec item 3)."""
        import base64

        from unittest.mock import patch

        mock_proc = self._mock_process()

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            await run_remote_command(
                host="myhost",
                command="line1\nline2\n",
                remote_shell="powershell",
                timeout=10,
            )

        sent_b64 = mock_proc.stdin.write.call_args[0][0].decode("ascii")
        decoded = base64.b64decode(sent_b64).decode("utf-8")
        assert decoded == "line1\r\nline2\r\n"

    @pytest.mark.asyncio
    async def test_posix_payload_newlines_untouched(self):
        """POSIX payloads keep their original line endings."""
        import base64

        from unittest.mock import patch

        mock_proc = self._mock_process()

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            await run_remote_command(
                host="myhost",
                command="line1\nline2\n",
                remote_shell="posix",
                timeout=10,
            )

        sent_b64 = mock_proc.stdin.write.call_args[0][0].decode("ascii")
        decoded = base64.b64decode(sent_b64).decode("utf-8")
        assert decoded == "line1\nline2\n"

    @pytest.mark.asyncio
    async def test_result_has_status_and_output_encoding(self):
        """Every remote result carries status + output_encoding (spec item 3)."""
        from unittest.mock import patch

        mock_proc = self._mock_process(returncode=3, stdout=b"out")

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            result = await run_remote_command(
                host="myhost", command="exit 3", remote_shell="posix", timeout=10
            )

        assert result["status"] == "completed"
        assert result["output_encoding"] == "utf-8"
        assert result["exit_code"] == 3
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout_is_distinct_status(self):
        """A timeout is status='timeout', not a completed run (spec item 3)."""
        from unittest.mock import patch, MagicMock

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345

        async def blocking_wait():
            await asyncio.sleep(60)

        mock_proc.wait = blocking_wait
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()

        async def fake_read(n):
            return b""

        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = fake_read
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = fake_read

        async def mock_create_subprocess_exec(*args, **kwargs):
            return mock_proc

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            result = await run_remote_command(
                host="myhost", command="sleep 99", remote_shell="posix", timeout=0.1
            )

        assert result["status"] == "timeout"
        assert result["timed_out"] is True
        assert result["success"] is False
        assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_ssh_missing_is_error_status(self):
        """ssh binary missing => status='error' (spec item 3)."""
        from unittest.mock import patch

        async def mock_create_subprocess_exec(*args, **kwargs):
            raise FileNotFoundError("ssh")

        with patch(
            "agent13.remote_exec.asyncio.create_subprocess_exec",
            mock_create_subprocess_exec,
        ):
            result = await run_remote_command(
                host="myhost", command="echo hi", remote_shell="posix", timeout=10
            )

        assert result["status"] == "error"
        assert result["output_encoding"] == "utf-8"
        assert "ssh not found" in result["stderr"]


class TestTransportCleanup:
    """run_remote_command must close pipe + subprocess transports on every
    path (same rationale as TestTransportCleanup in test_sandbox.py:
    unclosed pipe transports warn at GC and raise
    ValueError("I/O operation on closed pipe") on Windows).

    ssh is stood in for by a local command via a patched build_remote_argv,
    so real asyncio transports exist to assert on. The stand-in is
    sys.executable (not shell builtins like echo/sleep) because
    create_subprocess_exec runs without a shell on Windows.
    """

    @staticmethod
    def _assert_transports_closed(proc):
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                t = stream._transport
                assert t is not None and t._closing, "pipe transport left open"
        assert proc._transport is not None and proc._transport._closed, (
            "subprocess transport left open"
        )

    @staticmethod
    def _patches(argv, processes):
        from unittest.mock import AsyncMock, patch

        real = asyncio.create_subprocess_exec

        async def capture(*args, **kwargs):
            proc = await real(*args, **kwargs)
            processes.append(proc)
            return proc

        return (
            patch(
                "agent13.remote_exec.detect_remote_shell",
                new=AsyncMock(return_value="posix"),
            ),
            patch("agent13.remote_exec.build_remote_argv", return_value=argv),
            patch("asyncio.create_subprocess_exec", capture),
        )

    @staticmethod
    def _py(code):
        import sys

        return [sys.executable, "-c", code]

    @pytest.mark.asyncio
    async def test_normal_path_closes_transports(self):
        processes = []
        p1, p2, p3 = self._patches(self._py("print('hi')"), processes)
        start = len(processes)
        with p1, p2, p3:
            result = await run_remote_command(
                host="myhost", command="echo hi", timeout=10
            )
        assert result["success"]
        # processes[start] is the command itself (on Windows the timeout
        # kill may append a taskkill after it).
        self._assert_transports_closed(processes[start])

    @pytest.mark.asyncio
    async def test_cancel_path_closes_transports_and_kills(self):
        processes = []
        p1, p2, p3 = self._patches(
            self._py("import time; time.sleep(30)"), processes
        )
        start = len(processes)
        with p1, p2, p3:
            task = asyncio.create_task(
                run_remote_command(host="myhost", command="sleep 30", timeout=60)
            )
            # Poll until the spawn has completed: a cancel before spawn
            # leaves no process to assert on (slow CI machines).
            for _ in range(100):
                if len(processes) > start:
                    break
                await asyncio.sleep(0.1)
            assert len(processes) > start, "spawn did not complete before cancel"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        proc = processes[start]
        self._assert_transports_closed(proc)
        assert proc.returncode is not None, "ssh not killed on cancel"

    @pytest.mark.asyncio
    async def test_timeout_path_closes_transports(self):
        processes = []
        p1, p2, p3 = self._patches(
            self._py("import time; time.sleep(30)"), processes
        )
        start = len(processes)
        with p1, p2, p3:
            result = await run_remote_command(
                host="myhost", command="sleep 30", timeout=1
            )
        assert result["timed_out"]
        self._assert_transports_closed(processes[start])
