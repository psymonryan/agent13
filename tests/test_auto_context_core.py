"""Tests for the auto-context threshold feature (config, circuit breaker,
context estimation, snapshot naming, continuation hint).

NOTE: the threshold-gate, post-turn dispatch, and max-iteration-bound test
classes that lived here before were removed as part of the auto-context rework
(action + chain replace max_iterations; the journal branch of the auto path is
gone). Gate behaviour is covered by TestAutoContextGate in test_auto_context.py;
the chain / restart-after-compact behaviour is covered by TestAutoContextChain
(added with the Stage 3 dispatch rework).
"""

from unittest.mock import MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.events import AgentEvent
from agent13.prompts import AUTO_CONTEXT_CONTINUE_HINT


def make_agent(auto_context_threshold=0, journal_mode=False):
    """Create a minimal agent for testing."""
    client = MagicMock()
    agent = Agent(
        client=client,
        model="test-model",
        auto_context_threshold=auto_context_threshold,
        journal_mode=journal_mode,
    )
    return agent


class TestAutoContextConfig:
    def test_default_disabled(self):
        agent = make_agent()
        assert agent.auto_context_threshold == 0
        assert agent._auto_context_failures == 0
        assert agent._auto_context_triggered is False

    def test_threshold_set(self):
        agent = make_agent(auto_context_threshold=150000)
        assert agent.auto_context_threshold == 150000

    def test_threshold_runtime_change(self):
        agent = make_agent(auto_context_threshold=0)
        agent.auto_context_threshold = 100000
        assert agent.auto_context_threshold == 100000
        agent.auto_context_threshold = 0
        assert agent.auto_context_threshold == 0


def _parse_auto_context(toml_text: str):
    """Write TOML to a temp file and return Config.from_file() (auto-unlink)."""
    import tempfile

    from pathlib import Path

    from agent13.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_text)
        path = Path(f.name)
    try:
        return Config.from_file(path)
    finally:
        path.unlink()


class TestAutoContextThresholdParsing:
    """Threshold default (ON by default) + fail fast on invalid TOML values.

    The threshold is enabled by default (220000) to suit large-context models
    (e.g. Qwen 3.8, ~260k). A bad value must raise ConfigFileError at startup
    - never be silently dropped.
    """

    def test_config_default_is_220000(self):
        from agent13.config import Config

        assert Config().auto_context_threshold == 220000

    def test_parse_valid_threshold(self):
        config = _parse_auto_context("[auto_context]\nthreshold = 220000\n")
        assert config.auto_context_threshold == 220000

    def test_parse_zero_disables(self):
        config = _parse_auto_context("[auto_context]\nthreshold = 0\n")
        assert config.auto_context_threshold == 0

    def test_invalid_string_threshold_fails_fast(self):
        from agent13.fileio import ConfigFileError

        with pytest.raises(ConfigFileError, match="threshold"):
            _parse_auto_context('[auto_context]\nthreshold = "220k"\n')

    def test_negative_threshold_fails_fast(self):
        from agent13.fileio import ConfigFileError

        with pytest.raises(ConfigFileError, match="threshold"):
            _parse_auto_context("[auto_context]\nthreshold = -5\n")


class TestAutoContextCircuitBreaker:
    def test_failures_increment(self):
        agent = make_agent(auto_context_threshold=1000)
        agent._auto_context_failures = 0
        agent._auto_context_failures += 1
        assert agent._auto_context_failures == 1
        agent._auto_context_failures += 1
        assert agent._auto_context_failures == 2

    def test_failures_reset_on_success(self):
        agent = make_agent(auto_context_threshold=1000)
        agent._auto_context_failures = 2
        agent._auto_context_failures = 0
        assert agent._auto_context_failures == 0

    def test_circuit_breaker_at_3(self):
        agent = make_agent(auto_context_threshold=1000)
        agent._auto_context_failures = 3
        # The threshold check should not trigger when failures >= 3
        assert agent._auto_context_failures >= 3


class TestContextEstimation:
    """The stale-token fix: estimate includes tool results added after the stream."""

    def test_estimate_equals_prompt_tokens_with_no_added(self):
        agent = make_agent()
        agent.prompt_tokens = 3000
        agent._msg_count_before_stream = len(agent.messages)  # nothing added
        assert agent._estimate_current_context_tokens() == 3000

    def test_estimate_includes_added_tool_results(self):
        agent = make_agent()
        agent.prompt_tokens = 3000
        agent._msg_count_before_stream = 0
        # Simulate an assistant tool-call msg + a large tool result added after
        # the stream. ~40k chars of tool output ≈ 10k tokens.
        agent.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        agent.messages.append(
            {"role": "tool", "content": "x" * 40000, "tool_call_id": "t1"}
        )
        est = agent._estimate_current_context_tokens()
        # prompt_tokens (3k) + ~10k from the tool result
        assert est > 10000
        assert agent.prompt_tokens < 10000  # stale count alone would miss it

    def test_stale_count_misses_but_estimate_triggers(self):
        """The exact bug scenario: batched tool results jump over the threshold."""
        agent = make_agent(auto_context_threshold=10000)
        agent.prompt_tokens = 3000  # stale: under threshold
        agent._msg_count_before_stream = 0
        agent.messages.append(
            {"role": "tool", "content": "y" * 60000, "tool_call_id": "t1"}
        )
        # prompt_tokens alone: 3000 < 10000 -> would NOT trigger
        assert agent.prompt_tokens < agent.auto_context_threshold
        # estimate: 3000 + 15000 = 18000 >= 10000 -> triggers
        assert agent._estimate_current_context_tokens() >= agent.auto_context_threshold

    def test_estimate_counts_tool_call_arguments(self):
        agent = make_agent()
        agent.prompt_tokens = 0
        agent._msg_count_before_stream = 0
        agent.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "command", "arguments": "z" * 8000},
                    }
                ],
            }
        )
        # 8000 chars of args ≈ 2000 tokens
        assert agent._estimate_current_context_tokens() >= 2000


class TestContextEstimateEmission:
    """The safe point emits CONTEXT_ESTIMATE with the estimated context size."""

    @pytest.mark.asyncio
    async def test_emits_estimate_at_safe_point(self):
        from unittest.mock import patch

        agent = make_agent(
            auto_context_threshold=0
        )  # disabled; emission is independent
        agent._running = True
        agent.prompt_tokens = 3000

        async def mock_execute_tool(name, arguments):
            return "x" * 40000  # large tool result (~10k tokens)

        agent._execute_tool_async = mock_execute_tool

        call_count = 0

        async def mock_stream(client, model, messages, system_prompt, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield (
                    "tool_calls_complete",
                    {
                        "tool_calls": [
                            {"id": "t1", "name": "command", "arguments": "{}"}
                        ]
                    },
                )
            else:
                yield ("content", "done")

        agent.messages.append({"role": "user", "content": "go"})

        estimates = []

        async def on_event(event):
            if event.event == AgentEvent.CONTEXT_ESTIMATE:
                estimates.append(event.data.get("estimated_tokens"))

        agent.on_event(on_event)

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            await agent._llm_turn()

        # Emitted once (round 1 has tool calls; round 2 is a final answer).
        assert len(estimates) == 1
        # estimate = 3000 (prompt) + ~10000 (40000-char tool result)
        assert estimates[0] > 10000
        # The stale prompt_tokens alone would have missed this
        assert agent.prompt_tokens < 10000


class TestPostCompactContextEstimate:
    """compact_history emits a fresh CONTEXT_ESTIMATE (post-compact size) so
    the TUI's Ctx counter drops immediately - not the stale pre-compact
    TOKEN_USAGE, which lingers until the next LLM call reports usage."""

    @pytest.mark.asyncio
    async def test_emits_fresh_estimate_not_stale_usage(self):
        agent = make_agent()
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [{"role": "user", "content": "old history " * 200}]
        agent.prompt_tokens = 220000  # stale pre-compact grounded count

        events = []

        async def on_event(event):
            if event.event in (
                AgentEvent.CONTEXT_ESTIMATE,
                AgentEvent.TOKEN_USAGE,
            ):
                events.append((event.event, event.data))

        agent.on_event(on_event)

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            yield (
                "content",
                "This is the compacted session summary. " * 5,  # 300 chars
            )
            yield (
                "token_usage",
                {
                    "prompt_tokens": 220000,
                    "completion_tokens": 100,
                    "total_tokens": 220100,
                },
            )

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            ok, _ = await agent.compact_history("")

        assert ok

        estimates = [d for e, d in events if e == AgentEvent.CONTEXT_ESTIMATE]
        stale = [
            d
            for e, d in events
            if e == AgentEvent.TOKEN_USAGE and d.get("source") == "compact"
        ]
        # No stale source=compact TOKEN_USAGE (the pre-compact prompt_tokens)
        assert not stale
        # A fresh estimate was published
        assert len(estimates) == 1
        est = estimates[0]["estimated_tokens"]
        # ~system prompt (1000) + compacted pair (~75) - far below the
        # pre-compact 220k, and above the system prompt floor
        assert 1000 < est < 2000

    @pytest.mark.asyncio
    async def test_estimate_includes_tools_schema(self):
        agent = make_agent()
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [{"role": "user", "content": "hi"}]
        big_schema = "x" * 8000  # ~2000 tokens of tool schema
        agent.tools = [
            {
                "type": "function",
                "function": {"name": "big", "description": big_schema},
            }
        ]

        est = await agent._estimate_post_compact_context()
        # 1000 (system) + ~2000 (tools) + a few (messages)
        assert est >= 3000

    @pytest.mark.asyncio
    async def test_estimate_survives_tool_fetch_failure(self):
        agent = make_agent()
        agent.system_prompt = "S" * 4000
        agent.messages = [{"role": "user", "content": "hi"}]

        async def boom():
            raise RuntimeError("mcp down")

        agent.get_all_tools = boom
        est = await agent._estimate_post_compact_context()
        # Falls back to system prompt + messages only
        assert est >= 1000


class TestAutoContextSnapshot:
    """The pre-compact snapshot file naming."""

    def test_snapshot_filename_pattern(self, tmp_path):
        agent = make_agent()
        agent.session_date = "2026-08-19"

        with patch("agent13.persistence.get_auto_save_path") as mock_path:
            mock_path.return_value = tmp_path / "2026-08-19.ctx"
            with patch("agent13.persistence.save_context") as mock_save:
                agent._save_auto_context_snapshot(2)

        expected = tmp_path / "2026-08-19_2.ctx"
        mock_save.assert_called_once()
        # save_context(agent, path)
        assert mock_save.call_args[0][1] == expected

    def test_snapshot_swallows_errors(self):
        agent = make_agent()
        with patch(
            "agent13.persistence.save_context", side_effect=OSError("disk full")
        ):
            # Must not raise
            agent._save_auto_context_snapshot(1)

    def test_snapshot_count_is_monotonic(self):
        agent = make_agent()
        assert agent._auto_context_snapshot_count == 0
        agent._auto_context_snapshot_count += 1
        agent._auto_context_snapshot_count += 1
        assert agent._auto_context_snapshot_count == 2


class TestAutoContextContinueHint:
    """The continuation hint constant."""

    def test_hint_is_defined(self):
        assert isinstance(AUTO_CONTEXT_CONTINUE_HINT, str)
        assert "Continue" in AUTO_CONTEXT_CONTINUE_HINT

    def test_hint_mentions_outstanding_items(self):
        assert "outstanding items" in AUTO_CONTEXT_CONTINUE_HINT.lower()


class TestConfigParsing:
    def test_auto_context_threshold_from_config(self):
        from agent13.config import Config

        config = Config()
        assert config.auto_context_threshold == 220000

    def test_auto_context_threshold_parse(self):
        """Test that the TOML parsing works."""
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"

[auto_context]
threshold = 150000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = Path(f.name)

        try:
            config = Config.from_file(path)
            assert config.auto_context_threshold == 150000
        finally:
            path.unlink()

    def test_auto_context_threshold_absent(self):
        """Test that missing [auto_context] section defaults to 220000 (enabled)."""
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = Path(f.name)

        try:
            config = Config.from_file(path)
            assert config.auto_context_threshold == 220000
        finally:
            path.unlink()
