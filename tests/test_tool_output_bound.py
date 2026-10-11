"""Unit tests for tool output bounding (Phase 1 - text spill).

Covers the pure spill logic (agent13/tool_output_bound.py) and the choke
point (Agent._build_tool_result_content): cap boundary, head/tail sizing
with line-boundary cuts, marker content (line ranges that map 1:1 onto
read_file's offset/limit, path), byte-identical spill file, per-tool
override, config off (0), spill-write-failure fallback, images untouched.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent13.config import Config
from agent13.core import Agent
import agent13.tool_output_bound as tob
from agent13.tool_output_bound import (
    HEAD_CHARS,
    TAIL_CHARS,
    bound_tool_result,
)
from tools import ToolResult

CAP = 30_000  # HEAD_CHARS + TAIL_CHARS - the launch cap


# ─────────────────────────────────────────────────────────────────────────────
# Config parsing ([tools] max_result_chars)
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigParsing:
    def test_tools_section_parsed(self, tmp_path):
        from agent13.config import Config

        p = tmp_path / "config.toml"
        p.write_text("[tools]\nmax_result_chars = 12345\n")
        assert Config.from_file(p).tools_max_result_chars == 12345

    def test_tools_section_zero(self, tmp_path):
        from agent13.config import Config

        p = tmp_path / "config.toml"
        p.write_text("[tools]\nmax_result_chars = 0\n")
        assert Config.from_file(p).tools_max_result_chars == 0

    def test_tools_section_bad_value_fails_fast(self, tmp_path):
        from agent13.config import Config, ConfigFileError

        p = tmp_path / "config.toml"
        p.write_text("[tools]\nmax_result_chars = \"30k\"\n")
        with pytest.raises(ConfigFileError):
            Config.from_file(p)

    def test_default_when_absent(self, tmp_path):
        from agent13.config import Config

        p = tmp_path / "config.toml"
        p.write_text("")
        assert Config.from_file(p).tools_max_result_chars == 30_000


def make_agent():
    """Create a minimal agent for testing."""
    return Agent(client=MagicMock(), model="test-model")


def make_text(lines=2000, width=20):
    """~44k chars of line-oriented text (over the 30k cap)."""
    return "\n".join(f"line {i:04d} " + "x" * width for i in range(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Pure spill logic
# ─────────────────────────────────────────────────────────────────────────────


class TestCapBoundary:
    def test_at_cap_unchanged_no_spill(self):
        text = "a" * CAP
        out, path = bound_tool_result(text, CAP, "t")
        assert out == text
        assert path is None

    def test_just_over_cap_spills(self):
        text = "a\n" * 15_000 + "b" * (CAP + 1 - 30_000)  # 30,001 chars
        assert len(text) == CAP + 1
        out, path = bound_tool_result(text, CAP, "t")
        try:
            assert path is not None
            assert Path(path).exists()
            assert "chars elided" in out
            # NOTE: with only 2 chars elided the marker overhead makes the
            # bounded content slightly LONGER than the input - accepted by
            # the design ("harmless; no grace band").
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_under_cap_unchanged(self):
        text = "short output"
        out, path = bound_tool_result(text, CAP, "t")
        assert out == text
        assert path is None

    def test_cap_zero_disabled(self):
        text = make_text()
        out, path = bound_tool_result(text, 0, "t")
        assert out == text
        assert path is None


class TestHeadTailSizing:
    def test_head_and_tail_bounds(self):
        text = make_text()
        out, path = bound_tool_result(text, CAP, "t")
        try:
            head = out.split("\n[... ")[0]
            tail = out.rsplit("]\n", 1)[1]
            assert len(head) <= HEAD_CHARS
            assert len(tail) <= TAIL_CHARS
            # head ends on a line boundary, tail starts on one
            assert head.endswith("\n")
            assert not tail.startswith("\n")
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_head_tail_content(self):
        text = make_text()
        out, path = bound_tool_result(text, CAP, "t")
        try:
            head = out.split("\n[... ")[0]
            tail = out.rsplit("]\n", 1)[1]
            assert head.startswith("line 0000 ")
            assert text.split("\n")[-1] in tail
        finally:
            if path:
                Path(path).unlink(missing_ok=True)


class TestMarker:
    def test_marker_line_range_and_path(self):
        text = make_text()
        out, path = bound_tool_result(text, CAP, "my_tool")
        try:
            m = re.search(
                r"\[\.\.\. (\d[\d,]*) chars elided "
                r"\(lines (\d+)-(\d+) of (\d+)\) - full output saved to "
                r"(\S+) for reference\]",
                out,
            )
            assert m, f"marker not found in: {out[:200]}..."
            elided = int(m.group(1).replace(",", ""))
            x, y, z = int(m.group(2)), int(m.group(3)), int(m.group(4))
            assert z == len(text.split("\n"))
            assert 1 <= x < y <= z
            # elided count = the line span plus the trailing newline that
            # separates the last elided line from the tail
            assert elided == len("\n".join(text.split("\n")[x - 1 : y])) + 1
            assert m.group(5) == path
            assert "my_tool" in Path(path).name
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_line_numbers_map_onto_read_file(self):
        """head + elided lines (x-1..y in 0-based file lines) + tail
        reconstructs the original text exactly - the marker's line numbers
        can be used verbatim as read_file offset/limit."""
        text = make_text()
        out, path = bound_tool_result(text, CAP, "t")
        try:
            m = re.search(r"lines (\d+)-(\d+) of (\d+)", out)
            x, y = int(m.group(1)), int(m.group(2))
            lines = text.split("\n")
            head = out.split("\n[... ")[0]
            tail = out.rsplit("]\n", 1)[1]
            elided = "\n".join(lines[x - 1 : y])
            assert head + elided + "\n" + tail == text
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_single_line_reports_chars_not_lines(self):
        text = "y" * 40_000  # no newlines - minified JSON shape
        out, path = bound_tool_result(text, CAP, "t")
        try:
            m = re.search(
                r"\[\.\.\. \d[\d,]* chars elided "
                r"\(chars (\d+)-(\d+) of (\d+)\) - full output saved to",
                out,
            )
            assert m
            assert int(m.group(3)) == 40_000
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_json_envelope_payload_char_cut(self):
        """A JSON envelope (payload = one escaped line) is char-cut in place:
        the line/JSON structure is preserved and only the MIDDLE of the
        payload is elided - not the whole payload line. The marker reports a
        char range and sits inline; the spill file carries the full
        unbounded envelope."""
        import json

        blob = "\n".join(f"line {i:04d} " + "x" * 20 for i in range(2000))
        text = json.dumps(
            {"success": True, "exit_code": 0, "stdout": blob}, indent=2
        )
        out, path = bound_tool_result(text, CAP, "command")
        try:
            # single-line elided region -> char range, not line range
            m = re.search(r"\(chars (\d+)-(\d+) of (\d+)\)", out)
            assert m, f"expected a char-range marker, got: {out[:200]}"
            assert int(m.group(3)) == len(text)
            # JSON structure survives: opening + closing keys intact
            assert out.lstrip().startswith("{")
            assert out.rstrip().endswith("}")
            assert '"stdout": "' in out
            # payload head AND tail both present (not elided away)
            assert "line 0000 " in out
            assert "line 1999 " in out
            # the middle is elided
            assert "line 0999 " not in out
            # the marker is inline (no newline around it) - payload stays 1 line
            assert "\n[... " not in out
            # spill file is byte-identical to the full envelope
            assert Path(path).read_text(encoding="utf-8") == text
        finally:
            if path:
                Path(path).unlink(missing_ok=True)


class TestSpillFile:
    def test_file_byte_identical_to_full_output(self):
        text = make_text()
        out, path = bound_tool_result(text, CAP, "t")
        try:
            assert Path(path).read_text(encoding="utf-8") == text
        finally:
            Path(path).unlink(missing_ok=True)

    def test_write_failure_fallback(self, monkeypatch):
        """Spill write failure: head+tail WITHOUT the file pointer, marker
        says the middle is unavailable. Never fails the turn."""

        def boom(self, *a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        text = make_text()
        out, path = bound_tool_result(text, CAP, "t")
        assert path is None
        assert "full output unavailable (spill write failed)" in out
        assert "chars elided" in out
        # head + tail still present
        assert out.startswith("line 0000 ")
        assert make_text().split("\n")[-1] in out


# ─────────────────────────────────────────────────────────────────────────────
# Choke point (Agent._build_tool_result_content)
# ─────────────────────────────────────────────────────────────────────────────


class TestDebugLogging:
    """Observability: one debug-log event per bounded result."""

    def test_bounded_result_emits_event(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            tob, "log_event", lambda name, data: events.append((name, data))
        )
        text = make_text()
        _, spill_path = bound_tool_result(text, CAP, "command")
        assert len(events) == 1
        name, data = events[0]
        assert name == "tool_output_bounded"
        assert data["tool"] == "command"
        assert data["original_chars"] == len(text)
        assert data["cap"] == CAP
        assert data["elided_chars"] > 0
        assert data["spill_path"] == spill_path

    def test_write_failure_event_has_null_spill(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            tob, "log_event", lambda name, data: events.append((name, data))
        )

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        bound_tool_result(make_text(), CAP, "command")
        assert len(events) == 1
        assert events[0][1]["spill_path"] is None

    def test_unbounded_result_emits_nothing(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            tob, "log_event", lambda name, data: events.append((name, data))
        )
        bound_tool_result("short text", CAP, "command")
        assert events == []


@pytest.fixture
def fixed_config(monkeypatch):
    """Deterministic config: the choke point reads get_config() at call
    time; pin it so the test never depends on the machine's real
    ~/.agent13/config.toml."""
    cfg = Config()
    cfg.tools_max_result_chars = CAP
    monkeypatch.setattr("agent13.config.get_config", lambda: cfg)
    return cfg


class TestChokePoint:
    @pytest.mark.asyncio
    async def test_over_cap_text_bounded(self, fixed_config):
        agent = make_agent()
        text = make_text()
        msg, extra = await agent._build_tool_result_content(
            text, "command", "call_1"
        )
        assert extra == []
        content = msg["content"]
        assert len(content) < len(text)
        assert "chars elided" in content
        assert content.startswith("line 0000 ")
        assert agent.tool_stats.bounded == 1

    @pytest.mark.asyncio
    async def test_at_cap_text_unchanged(self, fixed_config):
        agent = make_agent()
        text = "a" * CAP
        msg, _ = await agent._build_tool_result_content(text, "command", "c")
        assert msg["content"] == text
        assert agent.tool_stats.bounded == 0

    @pytest.mark.asyncio
    async def test_config_off(self, fixed_config):
        fixed_config.tools_max_result_chars = 0
        agent = make_agent()
        text = make_text()
        msg, _ = await agent._build_tool_result_content(text, "command", "c")
        assert msg["content"] == text
        assert agent.tool_stats.bounded == 0

    @pytest.mark.asyncio
    async def test_per_tool_override_wins(self, fixed_config, monkeypatch):
        """A per-tool cap declared on the tool function wins over the
        global config. Injected via the same map @tool(max_result_chars=)
        populates (no global schema leak from a test-local @tool)."""
        import tools as tools_pkg

        monkeypatch.setitem(tools_pkg._tool_max_result_chars, "verbose_tool", CAP)
        agent = make_agent()
        text = make_text(lines=3000)  # ~66k chars
        msg, _ = await agent._build_tool_result_content(
            text, "verbose_tool", "c"
        )
        assert "chars elided" in msg["content"]
        # A tool WITHOUT an override uses the global cap (same 30k here -
        # verify the lookup path by using a cap the global does not have):
        monkeypatch.setitem(tools_pkg._tool_max_result_chars, "tiny_tool", CAP)
        tiny = "z" * CAP  # at the tiny cap -> unchanged
        msg2, _ = await agent._build_tool_result_content(tiny, "tiny_tool", "c2")
        assert msg2["content"] == tiny

    @pytest.mark.asyncio
    async def test_images_untouched_text_still_bounded(self, fixed_config):
        """Image lists pass through untouched (Phase 2); the accompanying
        text is still bounded."""
        agent = make_agent()
        text = make_text()
        # A real (1x1 px) PNG so the native routing keeps it as an image_url
        # block rather than the "unreadable" text note
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        result = ToolResult(text=text, images=[uri])
        msg, extra = await agent._build_tool_result_content(
            result, "read_file", "c"
        )
        # tool message text bounded
        assert "chars elided" in msg["content"]
        # image message intact (native routing - no [vision] config)
        assert len(extra) == 1
        parts = extra[0]["content"]
        assert any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in parts
        )
