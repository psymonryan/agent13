"""Corrupt/truncated images must not kill the agent turn in either vision mode.

Bug (2026-09-20): a PNG scp'd while the remote side was still writing it
had an intact header but cut-off pixel data.

- Native mode: PIL's lazy decode raised ``OSError: image file is
  truncated`` at ``resize`` and, once that was fixed, the provider
  rejected the whole request with a 500 — either way the turn died.
- Sidecar mode: the sidecar vision model 500'd on the image and
  ``describe_image``'s exception propagated up and killed the turn.

Fix: native mode fully decodes each image before injection (unreadable
ones become text notes); sidecar mode catches ``describe_image``
failures and turns them into text notes. The model sees an actionable
note and can re-fetch the file.
"""

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from agent13.core import Agent, AgentStatus, PauseState
from tools import ToolResult


def _png_uri(w: int = 100, h: int = 50) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _truncated_png_uri() -> str:
    """Header intact, pixel data cut off — the 2026-09-20 failure case."""
    full = io.BytesIO()
    Image.new("RGB", (2000, 1500), (200, 30, 30)).save(full, format="PNG")
    data = full.getvalue()[: len(full.getvalue()) // 2]
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _make_agent():
    client = MagicMock()
    agent = Agent(client=client, model="mock-model")
    agent.messages = []
    agent._status = AgentStatus.IDLE
    agent._pause_state = PauseState.RUNNING
    return agent


@pytest.fixture
def native_vision(monkeypatch):
    """Force native vision routing (no [vision] config section)."""
    from agent13 import config as config_mod

    monkeypatch.setattr(config_mod, "get_config", lambda: SimpleNamespace(vision=None))


@pytest.fixture
def sidecar_vision(monkeypatch):
    """Force sidecar vision routing with a dummy provider config."""
    from agent13 import config as config_mod

    vision = SimpleNamespace(
        sidecar_provider="testmlx",
        sidecar_model="Qwen3-VL-4B",
        should_use_native=lambda model: False,
    )
    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(
            vision=vision,
            get_provider=lambda name: SimpleNamespace(),
        ),
    )


# ── native mode ─────────────────────────────────────────────────────────────


class TestNativeCorruptImage:
    @pytest.mark.asyncio
    async def test_truncated_image_becomes_text_note(self, native_vision):
        agent = _make_agent()
        tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(text="Image file: /tmp/x.png", images=[_truncated_png_uri()]),
            "read_file",
            "t1",
        )
        assert len(extras) == 1
        blocks = extras[0]["content"]
        assert not any(b.get("type") == "image_url" for b in blocks), (
            "truncated image must not be injected — the provider would 500"
        )
        notes = [b["text"] for b in blocks if b.get("type") == "text"]
        assert any("unreadable" in n for n in notes)

    @pytest.mark.asyncio
    async def test_valid_image_still_injected(self, native_vision):
        agent = _make_agent()
        _tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(text="Image file: /tmp/y.png", images=[_png_uri()]),
            "read_file",
            "t1",
        )
        assert len(extras) == 1
        assert any(b.get("type") == "image_url" for b in extras[0]["content"])

    @pytest.mark.asyncio
    async def test_mixed_images_only_valid_injected(self, native_vision):
        agent = _make_agent()
        _tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(
                text="two images",
                images=[_png_uri(), _truncated_png_uri()],
            ),
            "read_file",
            "t1",
        )
        blocks = extras[0]["content"]
        assert sum(1 for b in blocks if b.get("type") == "image_url") == 1
        assert any("unreadable" in b.get("text", "") for b in blocks)


# ── sidecar mode ────────────────────────────────────────────────────────────


class TestSidecarCorruptImage:
    @pytest.mark.asyncio
    async def test_sidecar_failure_becomes_text_note(self, sidecar_vision, monkeypatch):
        """The 10:02 failure: sidecar 500s on the truncated image."""
        from agent13 import core as core_mod

        async def failing_describe(provider, model, uri, prompt):
            raise RuntimeError("Error code: 500 - Internal server error")

        monkeypatch.setattr(core_mod, "describe_image", failing_describe)
        agent = _make_agent()
        tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(text="Image file: /tmp/x.png", images=[_truncated_png_uri()]),
            "read_file",
            "t1",
        )
        assert extras == []
        assert "sidecar failed to describe image" in tool_msg["content"]
        assert "500" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_sidecar_success_includes_description(
        self, sidecar_vision, monkeypatch
    ):
        from agent13 import core as core_mod

        async def ok_describe(provider, model, uri, prompt):
            return "A red square on a white background."

        monkeypatch.setattr(core_mod, "describe_image", ok_describe)
        agent = _make_agent()
        tool_msg, extras = await agent._build_tool_result_content(
            ToolResult(text="Image file: /tmp/y.png", images=[_png_uri()]),
            "read_file",
            "t1",
        )
        assert extras == []
        assert "A red square" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_sidecar_failure_keeps_original_text(
        self, sidecar_vision, monkeypatch
    ):
        from agent13 import core as core_mod

        async def failing_describe(provider, model, uri, prompt):
            raise RuntimeError("boom")

        monkeypatch.setattr(core_mod, "describe_image", failing_describe)
        agent = _make_agent()
        tool_msg, _extras = await agent._build_tool_result_content(
            ToolResult(
                text="Image file: /tmp/x.png (1015808 bytes)",
                images=[_truncated_png_uri()],
            ),
            "read_file",
            "t1",
        )
        assert tool_msg["content"].startswith("Image file: /tmp/x.png")
