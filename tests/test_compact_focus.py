"""Wiring tests for /compact focus steering.

/compact [focus text] appends a steering block to the base compaction
prompt so the summary's sections and next steps focus on what the user
wants done after compaction. /compact --prompt <name> keeps the existing
named-prompt swap behavior.
"""

from agent13.prompts import (
    COMPACT_STEERING_TEMPLATE,
    DEFAULT_COMPACT_PROMPT,
    PromptManager,
)

from .test_repl import find_call, run_scenario


def _pm(tmp_path, yaml_text=None):
    """PromptManager backed by a temp file (hermetic — no user config)."""
    path = tmp_path / "prompts.yaml"
    if yaml_text is not None:
        path.write_text(yaml_text)
    return PromptManager(str(path))


def _compact_call(agent):
    """Return the add_message call for /compact, or None."""
    return find_call(agent, "add_message")


# ── Base prompt content ────────────────────────────────────────────────────


class TestDefaultCompactPrompt:
    """The default prompt preserves acquired knowledge, not just decisions."""

    def test_includes_gained_by_doing_bullet(self):
        assert "New knowledge gained by doing" in DEFAULT_COMPACT_PROMPT
        assert "how to connect to hosts/services" in DEFAULT_COMPACT_PROMPT

    def test_includes_rediscover_heuristic(self):
        assert "re-discover it from scratch" in DEFAULT_COMPACT_PROMPT

    def test_steering_template_has_focus_placeholder(self):
        rendered = COMPACT_STEERING_TEMPLATE.format(focus="work on SSR next")
        assert "Next task: work on SSR next" in rendered
        assert "Organize the summary sections around that task" in rendered
        assert "compressed harder" in rendered


# ── Bare /compact (no args) ────────────────────────────────────────────────


class TestBareCompact:
    async def test_uses_base_prompt_without_steering(self, tmp_path):
        _output, agent = await run_scenario(
            ["/compact", "/quit"], prompt_manager=_pm(tmp_path)
        )

        call = _compact_call(agent)
        assert call is not None
        _, text, _prio, _int, kind, data = call
        assert text == "/compact"
        assert kind == "compact"
        assert data["compact_prompt"] == DEFAULT_COMPACT_PROMPT
        assert "Next task:" not in data["compact_prompt"]


# ── /compact [focus text] ──────────────────────────────────────────────────


class TestCompactFocus:
    async def test_focus_appended_to_base_prompt(self, tmp_path):
        focus = "lets work on the server side rendering feature next"
        _output, agent = await run_scenario(
            [f"/compact {focus}", "/quit"], prompt_manager=_pm(tmp_path)
        )

        call = _compact_call(agent)
        assert call is not None
        _, text, _prio, _int, kind, data = call
        assert text == f"/compact {focus}"
        assert kind == "compact"
        prompt = data["compact_prompt"]
        # Base prompt intact, steering block appended after it
        assert prompt.startswith(DEFAULT_COMPACT_PROMPT)
        assert prompt == DEFAULT_COMPACT_PROMPT + COMPACT_STEERING_TEMPLATE.format(
            focus=focus
        )

    async def test_focus_uses_custom_compaction_prompt_when_present(self, tmp_path):
        """If the user has a custom 'compaction' prompt, steer that one."""
        custom = "My custom compaction instructions."
        pm = _pm(tmp_path, yaml_text=f"compaction: {custom!r}\n")
        _output, agent = await run_scenario(
            ["/compact focus on the API next", "/quit"], prompt_manager=pm
        )

        call = _compact_call(agent)
        assert call is not None
        prompt = call[5]["compact_prompt"]
        assert prompt.startswith(custom)
        assert "Next task: focus on the API next" in prompt


# ── /compact --prompt <name> (existing behavior) ───────────────────────────


class TestCompactPromptFlag:
    async def test_named_prompt_swapped_without_steering(self, tmp_path):
        pm = _pm(tmp_path, yaml_text="mycompact: Swap in this prompt.\n")
        _output, agent = await run_scenario(
            ["/compact --prompt mycompact", "/quit"], prompt_manager=pm
        )

        call = _compact_call(agent)
        assert call is not None
        _, text, _prio, _int, kind, data = call
        assert text == "/compact --prompt mycompact"
        assert kind == "compact"
        assert data["compact_prompt"] == "Swap in this prompt."
        assert "Next task:" not in data["compact_prompt"]

    async def test_missing_name_shows_usage(self, tmp_path):
        output, agent = await run_scenario(
            ["/compact --prompt", "/quit"], prompt_manager=_pm(tmp_path)
        )

        assert _compact_call(agent) is None
        assert "Usage: /compact --prompt <name>" in output

    async def test_unknown_name_shows_not_found(self, tmp_path):
        output, agent = await run_scenario(
            ["/compact --prompt nosuchprompt", "/quit"], prompt_manager=_pm(tmp_path)
        )

        assert _compact_call(agent) is None
        assert "Prompt 'nosuchprompt' not found" in output
