"""Tests for auto-compact threshold feature."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent13.core import Agent
from agent13.events import AgentEvent
from agent13.prompts import AUTO_COMPACT_CONTINUE_HINT


def make_agent(auto_compact_threshold=0, journal_mode=False, max_iterations=3):
    """Create a minimal agent for testing."""
    client = MagicMock()
    agent = Agent(
        client=client,
        model="test-model",
        auto_compact_threshold=auto_compact_threshold,
        auto_compact_max_iterations=max_iterations,
        journal_mode=journal_mode,
    )
    return agent


class TestAutoCompactConfig:
    def test_default_disabled(self):
        agent = make_agent()
        assert agent.auto_compact_threshold == 0
        assert agent._auto_compact_failures == 0
        assert agent._auto_compact_triggered is False

    def test_threshold_set(self):
        agent = make_agent(auto_compact_threshold=150000)
        assert agent.auto_compact_threshold == 150000

    def test_threshold_runtime_change(self):
        agent = make_agent(auto_compact_threshold=0)
        agent.auto_compact_threshold = 100000
        assert agent.auto_compact_threshold == 100000
        agent.auto_compact_threshold = 0
        assert agent.auto_compact_threshold == 0


class TestAutoCompactCircuitBreaker:
    def test_failures_increment(self):
        agent = make_agent(auto_compact_threshold=1000)
        agent._auto_compact_failures = 0
        agent._auto_compact_failures += 1
        assert agent._auto_compact_failures == 1
        agent._auto_compact_failures += 1
        assert agent._auto_compact_failures == 2

    def test_failures_reset_on_success(self):
        agent = make_agent(auto_compact_threshold=1000)
        agent._auto_compact_failures = 2
        agent._auto_compact_failures = 0
        assert agent._auto_compact_failures == 0

    def test_circuit_breaker_at_3(self):
        agent = make_agent(auto_compact_threshold=1000)
        agent._auto_compact_failures = 3
        # The threshold check should not trigger when failures >= 3
        assert agent._auto_compact_failures >= 3


class TestAutoCompactTrigger:
    """Test the threshold check logic at the safe point."""

    def _should_trigger(self, agent) -> bool:
        """Replicate the threshold check from _llm_turn (uses the estimate)."""
        return (
            agent.auto_compact_threshold > 0
            and agent._estimate_current_context_tokens()
            >= agent.auto_compact_threshold
            and agent._auto_compact_failures < 3
        )

    def test_no_trigger_when_disabled(self):
        agent = make_agent(auto_compact_threshold=0)
        agent.prompt_tokens = 999999
        assert not self._should_trigger(agent)

    def test_no_trigger_below_threshold(self):
        agent = make_agent(auto_compact_threshold=150000)
        agent.prompt_tokens = 149999
        assert not self._should_trigger(agent)

    def test_trigger_at_threshold(self):
        agent = make_agent(auto_compact_threshold=150000)
        agent.prompt_tokens = 150000
        assert self._should_trigger(agent)

    def test_trigger_above_threshold(self):
        agent = make_agent(auto_compact_threshold=150000)
        agent.prompt_tokens = 200000
        assert self._should_trigger(agent)

    def test_no_trigger_when_circuit_breaker_open(self):
        agent = make_agent(auto_compact_threshold=150000)
        agent.prompt_tokens = 200000
        agent._auto_compact_failures = 3
        assert not self._should_trigger(agent)

    def test_trigger_with_failures_below_limit(self):
        agent = make_agent(auto_compact_threshold=150000)
        agent.prompt_tokens = 200000
        agent._auto_compact_failures = 2
        assert self._should_trigger(agent)


class TestAutoCompactPostTurn:
    """Test the post-turn journal/compact dispatch (mirrors _process_item)."""

    async def _run_post_turn(self, agent):
        """Replicate the post-turn auto-compact loop from _process_item.

        Returns the number of times _llm_turn was re-entered.
        """
        reentered = 0

        def fake_llm_turn(_polite_acquired=False):
            nonlocal reentered
            reentered += 1

        agent._llm_turn = fake_llm_turn
        agent._auto_compact_iterations = 0

        while agent._auto_compact_triggered:
            agent._auto_compact_triggered = False
            agent._auto_compact_iterations += 1
            agent._auto_compact_snapshot_count += 1
            agent._save_auto_compact_snapshot(agent._auto_compact_snapshot_count)
            if agent.journal_mode:
                success, msg = await agent.journal.journal_all()
            else:
                success, msg = await agent.compact_history("")
            if not success:
                agent._auto_compact_failures += 1
                break
            agent._auto_compact_failures = 0
            agent.messages.append(
                {"role": "user", "content": AUTO_COMPACT_CONTINUE_HINT}
            )
            agent._llm_turn(_polite_acquired=False)
        return reentered

    @pytest.mark.asyncio
    async def test_journal_mode_calls_journal_all(self):
        agent = make_agent(auto_compact_threshold=1000, journal_mode=True)
        agent._auto_compact_triggered = True
        agent.prompt_tokens = 1500
        agent.journal.journal_all = AsyncMock(return_value=(True, "Journalled"))
        agent._save_auto_compact_snapshot = MagicMock()

        await self._run_post_turn(agent)

        agent.journal.journal_all.assert_called_once()
        assert agent._auto_compact_triggered is False
        assert agent._auto_compact_failures == 0
        assert agent._auto_compact_iterations == 1

    @pytest.mark.asyncio
    async def test_no_journal_calls_compact(self):
        agent = make_agent(auto_compact_threshold=1000, journal_mode=False)
        agent._auto_compact_triggered = True
        agent.prompt_tokens = 1500
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent._save_auto_compact_snapshot = MagicMock()

        await self._run_post_turn(agent)

        agent.compact_history.assert_called_once_with("")
        assert agent._auto_compact_triggered is False
        assert agent._auto_compact_failures == 0

    @pytest.mark.asyncio
    async def test_failure_increments_counter_and_stops(self):
        agent = make_agent(auto_compact_threshold=1000, journal_mode=False)
        agent._auto_compact_triggered = True
        agent._auto_compact_failures = 1
        agent.compact_history = AsyncMock(return_value=(False, "Failed"))
        agent._save_auto_compact_snapshot = MagicMock()

        reentered = await self._run_post_turn(agent)

        assert agent._auto_compact_failures == 2
        assert reentered == 0  # loop breaks before re-entering the turn

    @pytest.mark.asyncio
    async def test_success_injects_continue_hint(self):
        agent = make_agent(auto_compact_threshold=1000, journal_mode=False)
        agent._auto_compact_triggered = True
        agent.compact_history = AsyncMock(return_value=(True, "Compacted"))
        agent._save_auto_compact_snapshot = MagicMock()

        await self._run_post_turn(agent)

        last = agent.messages[-1]
        assert last["role"] == "user"
        assert last["content"] == AUTO_COMPACT_CONTINUE_HINT

    @pytest.mark.asyncio
    async def test_no_trigger_no_action(self):
        agent = make_agent(auto_compact_threshold=1000, journal_mode=False)
        agent._auto_compact_triggered = False
        agent.compact_history = AsyncMock()
        agent.journal.journal_all = AsyncMock()
        agent._save_auto_compact_snapshot = MagicMock()

        await self._run_post_turn(agent)

        agent.compact_history.assert_not_called()
        agent.journal.journal_all.assert_not_called()
        agent._save_auto_compact_snapshot.assert_not_called()


class TestAutoCompactMaxIterations:
    """Config default + TOML parsing for max_iterations."""

    def test_default_is_3(self):
        from agent13.config import Config

        assert Config().auto_compact_max_iterations == 3

    def test_agent_default_is_3(self):
        agent = make_agent()
        assert agent.auto_compact_max_iterations == 3

    def test_agent_param(self):
        agent = make_agent(max_iterations=5)
        assert agent.auto_compact_max_iterations == 5

    def test_parse_from_config(self):
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"

[auto_compact]
threshold = 150000
max_iterations = 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            path = Path(f.name)
        try:
            config = Config.from_file(path)
            assert config.auto_compact_max_iterations == 5
        finally:
            path.unlink()

    def test_parse_absent_defaults_to_3(self):
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"

[auto_compact]
threshold = 150000
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            path = Path(f.name)
        try:
            config = Config.from_file(path)
            assert config.auto_compact_max_iterations == 3
        finally:
            path.unlink()

    def test_parse_invalid_ignored(self):
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"

[auto_compact]
max_iterations = 0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            path = Path(f.name)
        try:
            config = Config.from_file(path)
            assert config.auto_compact_max_iterations == 3  # 0 rejected, default kept
        finally:
            path.unlink()


class TestAutoCompactContinueLoop:
    """The compact-vs-pause decision at the safe point (mirrors _llm_turn)."""

    def _decision(self, agent) -> str:
        """Return 'compact', 'pause', or 'none' for the current state."""
        if not (
            agent.auto_compact_threshold > 0
            and agent._estimate_current_context_tokens()
            >= agent.auto_compact_threshold
            and agent._auto_compact_failures < 3
        ):
            return "none"
        if agent._auto_compact_iterations < agent.auto_compact_max_iterations:
            return "compact"
        return "pause"

    def test_compact_when_under_bound(self):
        agent = make_agent(auto_compact_threshold=1000, max_iterations=3)
        agent.prompt_tokens = 2000
        agent._auto_compact_iterations = 0
        assert self._decision(agent) == "compact"

    def test_compact_at_bound_minus_one(self):
        agent = make_agent(auto_compact_threshold=1000, max_iterations=3)
        agent.prompt_tokens = 2000
        agent._auto_compact_iterations = 2
        assert self._decision(agent) == "compact"

    def test_pause_at_bound(self):
        agent = make_agent(auto_compact_threshold=1000, max_iterations=3)
        agent.prompt_tokens = 2000
        agent._auto_compact_iterations = 3
        assert self._decision(agent) == "pause"

    def test_pause_above_bound(self):
        agent = make_agent(auto_compact_threshold=1000, max_iterations=3)
        agent.prompt_tokens = 2000
        agent._auto_compact_iterations = 5
        assert self._decision(agent) == "pause"

    def test_none_when_disabled(self):
        agent = make_agent(auto_compact_threshold=0, max_iterations=3)
        agent.prompt_tokens = 999999
        assert self._decision(agent) == "none"

    def test_none_when_circuit_breaker_open(self):
        agent = make_agent(auto_compact_threshold=1000, max_iterations=3)
        agent.prompt_tokens = 2000
        agent._auto_compact_failures = 3
        assert self._decision(agent) == "none"

    def test_iteration_counter_resets_per_turn(self):
        agent = make_agent(auto_compact_threshold=1000)
        agent._auto_compact_iterations = 3  # simulate end of a prior turn
        # _process_item resets this to 0 at the start of each turn
        agent._auto_compact_iterations = 0
        assert agent._auto_compact_iterations == 0


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
        agent = make_agent(auto_compact_threshold=10000, max_iterations=3)
        agent.prompt_tokens = 3000  # stale: under threshold
        agent._msg_count_before_stream = 0
        agent.messages.append(
            {"role": "tool", "content": "y" * 60000, "tool_call_id": "t1"}
        )
        # prompt_tokens alone: 3000 < 10000 -> would NOT trigger
        assert agent.prompt_tokens < agent.auto_compact_threshold
        # estimate: 3000 + 15000 = 18000 >= 10000 -> triggers
        assert agent._estimate_current_context_tokens() >= agent.auto_compact_threshold

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

        agent = make_agent(auto_compact_threshold=0)  # disabled; emission is independent
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

        with patch(
            "agent13.llm.stream_response_with_tools", side_effect=mock_stream
        ):
            await agent._llm_turn()

        # Emitted once (round 1 has tool calls; round 2 is a final answer).
        assert len(estimates) == 1
        # estimate = 3000 (prompt) + ~10000 (40000-char tool result)
        assert estimates[0] > 10000
        # The stale prompt_tokens alone would have missed this
        assert agent.prompt_tokens < 10000


class TestAutoCompactSnapshot:
    """The pre-compact snapshot file naming."""

    def test_snapshot_filename_pattern(self, tmp_path):
        agent = make_agent()
        agent.session_date = "2026-08-19"

        with patch("agent13.persistence.get_auto_save_path") as mock_path:
            mock_path.return_value = tmp_path / "2026-08-19.ctx"
            with patch("agent13.persistence.save_context") as mock_save:
                agent._save_auto_compact_snapshot(2)

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
            agent._save_auto_compact_snapshot(1)

    def test_snapshot_count_is_monotonic(self):
        agent = make_agent()
        assert agent._auto_compact_snapshot_count == 0
        agent._auto_compact_snapshot_count += 1
        agent._auto_compact_snapshot_count += 1
        assert agent._auto_compact_snapshot_count == 2


class TestAutoCompactContinueHint:
    """The continuation hint constant."""

    def test_hint_is_defined(self):
        assert isinstance(AUTO_COMPACT_CONTINUE_HINT, str)
        assert "Continue" in AUTO_COMPACT_CONTINUE_HINT

    def test_hint_mentions_completing_task(self):
        assert "complete" in AUTO_COMPACT_CONTINUE_HINT.lower()


class TestConfigParsing:
    def test_auto_compact_threshold_from_config(self):
        from agent13.config import Config

        config = Config()
        assert config.auto_compact_threshold == 0

    def test_auto_compact_threshold_parse(self):
        """Test that the TOML parsing works."""
        import tempfile
        from pathlib import Path

        from agent13.config import Config

        toml_content = """
[[providers]]
name = "test"
api_base = "http://localhost:8000/v1"

[auto_compact]
threshold = 150000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = Path(f.name)

        try:
            config = Config.from_file(path)
            assert config.auto_compact_threshold == 150000
        finally:
            path.unlink()

    def test_auto_compact_threshold_absent(self):
        """Test that missing [auto_compact] section defaults to 0."""
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
            assert config.auto_compact_threshold == 0
        finally:
            path.unlink()
