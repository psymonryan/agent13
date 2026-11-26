"""Wiring tests for context-management flags in batch (`-p`) mode.

Regression for the silent-no-op bug where `run_batch_with_display` built the
Agent with neither `journal_mode` nor any `auto_context_*`, so `-p --journal`
did nothing while REPL/pipe/TUI honoured it.

These tests cover the seam (signature + Agent construction). The user-visible
behaviour is covered in test_batch_context_integration.py.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from agent13.cli import run_batch_with_display
from agent13.core import Agent

CONTEXT_KWARGS = (
    "journal_mode",
    "auto_context_threshold",
    "auto_context_action",
    "auto_context_chain",
    "auto_context_nudge_cap",
    "report_and_compact_prompt",
)


class TestBatchContextSignature:
    """The batch entry point must accept every context-management kwarg."""

    @pytest.mark.parametrize("name", CONTEXT_KWARGS)
    def test_kwarg_is_accepted(self, name):
        assert name in inspect.signature(run_batch_with_display).parameters, (
            f"run_batch_with_display must accept {name} so the flag works in "
            "batch mode"
        )

    @pytest.mark.parametrize("name", CONTEXT_KWARGS)
    def test_default_matches_the_agent_constructor(self, name):
        """A caller that omits the kwarg gets the Agent's own default."""
        batch_default = inspect.signature(run_batch_with_display).parameters[
            name
        ].default
        agent_default = inspect.signature(Agent.__init__).parameters[name].default
        assert batch_default == agent_default


class TestBatchContextReachesTheAgent:
    """run_batch_with_display must pass the kwargs into the Agent it builds."""

    async def _run(self, **kwargs):
        captured = {}

        async def fake_run_batch(agent, prompt, **rest):
            captured["agent"] = agent

        with patch("agent13.cli.run_batch", fake_run_batch):
            await run_batch_with_display(
                client=MagicMock(),
                model="test-model",
                prompt="ignored by stub",
                pretty=False,
                **kwargs,
            )
        return captured["agent"]

    async def test_explicit_values_land_on_the_agent(self):
        agent = await self._run(
            journal_mode=True,
            auto_context_threshold=1234,
            auto_context_action="compact",
            auto_context_chain=7,
            auto_context_nudge_cap=9,
            report_and_compact_prompt="PROMPT",
        )
        assert agent.journal_mode is True
        assert agent.auto_context_threshold == 1234
        assert agent.auto_context_action == "compact"
        assert agent.auto_context_chain == 7
        assert agent.auto_context_nudge_cap == 9
        assert agent.report_and_compact_prompt == "PROMPT"

    async def test_omitted_values_keep_the_agent_defaults(self):
        agent = await self._run()
        assert agent.journal_mode is False
        assert agent.auto_context_threshold == 220000
        assert agent.auto_context_action == "report_and_compact"
        assert agent.auto_context_chain == 3
        assert agent.auto_context_nudge_cap == 2
