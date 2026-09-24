"""Unit tests for remote_jobs (spec item 2: detached long-running remote jobs).

The ssh transport (run_remote_command) is mocked; job_id handling, script
construction, and poll-output parsing are tested for real.
"""

import base64
import re

import pytest

from agent13.remote_jobs import (
    JOB_ID_RE,
    TAIL_BYTES,
    TTL_DAYS,
    _build_detach_script,
    _build_poll_script,
    detach_remote_job,
    job_dir,
    new_job_id,
    parse_poll_output,
    poll_remote_job,
    validate_job_id,
)
from agent13.remote_exec import clear_remote_shell_cache


# ─── job_id ───────────────────────────────────────────────────────────────────


class TestJobId:
    def test_new_job_id_format(self):
        jid = new_job_id()
        assert JOB_ID_RE.match(jid)
        assert re.match(r"^\d{8}T\d{6}Z-", jid)

    def test_new_job_id_unique(self):
        ids = {new_job_id() for _ in range(50)}
        assert len(ids) == 50

    def test_validate_accepts_good(self):
        assert validate_job_id("20260920T120000Z-abc123") == "20260920T120000Z-abc123"

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "garbage",
            "20260920T120000Z",
            "20260920T120000Z-abc",
            "20260920T120000Z-ABCDEF",  # uppercase hex not allowed
            "20260920T120000Z-abc1234",  # too long
            "  20260920T120000Z-abc123  ",  # whitespace stripped -> ok below
        ],
    )
    def test_validate_rejects_bad(self, bad):
        if bad.strip() == "20260920T120000Z-abc123":
            # whitespace-only padding is tolerated (stripped)
            assert validate_job_id(bad) == "20260920T120000Z-abc123"
            return
        with pytest.raises(ValueError):
            validate_job_id(bad)

    def test_validate_error_message_teaches(self):
        with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
            validate_job_id("nope")


# ─── job dir ──────────────────────────────────────────────────────────────────


class TestJobDir:
    def test_posix(self):
        assert (
            job_dir("posix", "20260920T120000Z-abc123")
            == "/tmp/agent13-jobs/20260920T120000Z-abc123"
        )

    def test_powershell(self):
        assert (
            job_dir("powershell", "20260920T120000Z-abc123")
            == "$env:TEMP\\agent13-jobs\\20260920T120000Z-abc123"
        )


# ─── script construction ──────────────────────────────────────────────────────


class TestDetachScript:
    def test_posix_script_shape(self):
        script = _build_detach_script("posix", "20260920T120000Z-abc123", "echo hi")
        assert "nohup" in script
        assert "mkdir -p" in script
        assert "base64 -d" in script
        assert "echo $? > status" in script
        assert f"-mtime +{TTL_DAYS}" in script  # TTL cleanup
        # The user script is embedded as base64 (zero quoting layers)
        b64 = re.search(r"printf %s '([A-Za-z0-9+/=]+)'", script).group(1)
        assert base64.b64decode(b64).decode("utf-8") == "echo hi"

    def test_powershell_script_shape(self):
        script = _build_detach_script(
            "powershell", "20260920T120000Z-abc123", "Write-Output hi\n"
        )
        assert "Register-ScheduledTask" in script
        assert "wrapper.ps1" in script
        assert "FromBase64String" in script
        assert "AddDays(-{0})".format(TTL_DAYS) in script  # TTL cleanup
        b64 = re.search(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", script).group(1)
        decoded = base64.b64decode(b64).decode("utf-8")
        assert decoded == "Write-Output hi\r\n"  # CRLF-normalized (spec item 3)

    def test_posix_script_survives_weird_characters(self):
        nasty = "echo 'a $b `c` \"d\" $(e) > f 2>&1 | g'\nexit 7"
        script = _build_detach_script("posix", "20260920T120000Z-abc123", nasty)
        b64 = re.search(r"printf %s '([A-Za-z0-9+/=]+)'", script).group(1)
        assert base64.b64decode(b64).decode("utf-8") == nasty


class TestPollScript:
    def test_posix_markers(self):
        script = _build_poll_script("posix", "20260920T120000Z-abc123")
        assert "STATUS=not_found" in script
        assert "STATUS=exited" in script
        assert "STATUS=running" in script
        assert "TAIL_BEGIN" in script
        assert "TAIL_END" in script
        assert f"tail -c {TAIL_BYTES}" in script
        assert f"-mtime +{TTL_DAYS}" in script  # TTL cleanup

    def test_powershell_markers(self):
        script = _build_poll_script("powershell", "20260920T120000Z-abc123")
        assert "STATUS=not_found" in script
        assert "STATUS=exited" in script
        assert "STATUS=running" in script
        assert "TAIL_BEGIN" in script
        assert "TAIL_END" in script
        assert "Unregister-ScheduledTask" in script  # TTL task sweep
        assert str(TAIL_BYTES) in script


# ─── poll output parsing ──────────────────────────────────────────────────────


class TestParsePollOutput:
    def test_running(self):
        out = "STATUS=running\nTAIL_BEGIN\nlog line\nTAIL_END\n"
        p = parse_poll_output(out)
        assert p["status"] == "running"
        assert p["exit_code"] is None
        assert p["log_tail"] == "log line"

    def test_exited_with_code(self):
        out = "STATUS=exited\nEXIT_CODE=5\nTAIL_BEGIN\nline1\nline2\nTAIL_END\n"
        p = parse_poll_output(out)
        assert p["status"] == "exited"
        assert p["exit_code"] == 5
        assert p["log_tail"] == "line1\nline2"

    def test_exited_zero(self):
        out = "STATUS=exited\nEXIT_CODE=0\nTAIL_BEGIN\nTAIL_END\n"
        p = parse_poll_output(out)
        assert p["status"] == "exited"
        assert p["exit_code"] == 0
        assert p["log_tail"] == ""

    def test_not_found(self):
        p = parse_poll_output("STATUS=not_found\n")
        assert p["status"] == "not_found"

    def test_crlf_output(self):
        out = "STATUS=exited\r\nEXIT_CODE=2\r\nTAIL_BEGIN\r\nlog\r\nTAIL_END\r\n"
        p = parse_poll_output(out)
        assert p["status"] == "exited"
        assert p["exit_code"] == 2
        assert p["log_tail"] == "log"

    def test_garbage_defaults_unknown(self):
        p = parse_poll_output("???")
        assert p["status"] == "unknown"
        assert p["exit_code"] is None
        assert p["log_tail"] == ""


# ─── detach / poll flows (transport mocked) ───────────────────────────────────


class TestDetachRemoteJob:
    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_success_returns_job_metadata(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "PID 4321\n",
                "stderr": "",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await detach_remote_job(
            host="myhost", command="sleep 5", remote_shell="posix"
        )

        assert result["success"] is True
        assert JOB_ID_RE.match(result["job_id"])
        assert result["pid"] == 4321
        assert result["log_path"].endswith(f"{result['job_id']}/log")
        assert result["status_path"].endswith(f"{result['job_id']}/status")
        assert 'mode="poll"' in result["hint"]
        assert result["output_encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_success_powershell_returns_task_name(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "TASK agent13-job-20260920T120000Z-abc123\n",
                "stderr": "",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await detach_remote_job(
            host="myhost", command="Write-Output hi", remote_shell="powershell"
        )

        assert result["success"] is True
        assert result["pid"] is None
        assert result["task_name"] == "agent13-job-20260920T120000Z-abc123"

    @pytest.mark.asyncio
    async def test_failure_returns_stderr(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "mkdir: permission denied",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await detach_remote_job(
            host="myhost", command="sleep 5", remote_shell="posix"
        )

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "permission denied" in result["stderr"]
        assert result["pid"] is None


class TestPollRemoteJob:
    def setup_method(self):
        clear_remote_shell_cache()

    def teardown_method(self):
        clear_remote_shell_cache()

    @pytest.mark.asyncio
    async def test_exited_job(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "STATUS=exited\nEXIT_CODE=5\nTAIL_BEGIN\nline1\nline2\nTAIL_END\n",
                "stderr": "",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await poll_remote_job(
            host="myhost", job_id="20260920T120000Z-abc123", remote_shell="posix"
        )

        assert result["success"] is True
        assert result["status"] == "exited"
        assert result["job_exit_code"] == 5
        assert result["log_tail"] == "line1\nline2"
        assert result["exit_code"] == 0  # the poll itself succeeded

    @pytest.mark.asyncio
    async def test_not_found_gets_note(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "STATUS=not_found\n",
                "stderr": "",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await poll_remote_job(
            host="myhost", job_id="20260920T120000Z-abc123", remote_shell="posix"
        )

        assert result["status"] == "not_found"
        assert "TTL" in result["note"]

    @pytest.mark.asyncio
    async def test_probe_failure(self, monkeypatch):
        from agent13 import remote_jobs

        async def fake_run_remote_command(**kwargs):
            return {
                "success": False,
                "exit_code": 255,
                "stdout": "",
                "stderr": "ssh: connect failed",
            }

        monkeypatch.setattr(remote_jobs, "run_remote_command", fake_run_remote_command)
        result = await poll_remote_job(
            host="myhost", job_id="20260920T120000Z-abc123", remote_shell="posix"
        )

        assert result["success"] is False
        assert result["status"] == "poll_failed"
        assert "connect failed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_bad_job_id_raises(self):
        with pytest.raises(ValueError, match="Invalid job_id"):
            await poll_remote_job(host="myhost", job_id="nope")
