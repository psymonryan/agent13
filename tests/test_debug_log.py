"""Tests for debug_log memory-growth diagnostics (log_rss).

log_rss() emits a ``mem_rss`` event carrying peak RSS, sampled on every API
request so a long debug log shows whether memory climbs toward an OS
memory-pressure SIGKILL.
"""

import json
import sys

import pytest

import agent13.debug_log as dl

skip_windows = pytest.mark.skipif(sys.platform == "win32", reason="resource module is POSIX-only; log_rss no-ops on Windows")


@pytest.fixture(autouse=True)
def _reset_debug(tmp_path):
    """Point debug logging at a temp file; restore global state afterwards."""
    old_enabled = dl._debug_enabled
    old_file = dl._log_file
    dl._log_file = tmp_path / "debug.log"
    dl._debug_enabled = True
    yield
    dl._debug_enabled = old_enabled
    dl._log_file = old_file


def _events():
    if not dl._log_file.exists():
        return []
    return [json.loads(line) for line in dl._log_file.read_text().splitlines() if line.strip()]


@skip_windows
def test_log_rss_writes_mem_rss_event():
    dl.log_rss(note="t1")
    memrss = [e for e in _events() if e["event"] == "mem_rss"]
    assert memrss, "expected a mem_rss event"
    assert memrss[-1]["data"]["peak_rss_mb"] > 0
    assert memrss[-1]["data"]["note"] == "t1"


@skip_windows
def test_log_rss_without_note_omits_key():
    dl.log_rss()
    memrss = [e for e in _events() if e["event"] == "mem_rss"]
    assert memrss, "expected a mem_rss event"
    assert "note" not in memrss[-1]["data"]


def test_log_rss_noop_when_disabled():
    dl._debug_enabled = False
    dl.log_rss(note="x")
    assert _events() == []


@skip_windows
def test_init_debug_writes_session_start_and_baseline_rss(tmp_path):
    dl._debug_enabled = False
    dl._log_file = tmp_path / "debug.log"
    dl.init_debug(log_dir=tmp_path)
    evs = _events()
    assert evs[0]["event"] == "session_start"
    assert any(
        e["event"] == "mem_rss" and e["data"].get("note") == "session_start"
        for e in evs
    ), "expected a session_start mem_rss baseline right after session_start"
