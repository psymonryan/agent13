"""Tests for the auto-context threshold feature (config, circuit breaker,
context estimation, snapshot naming, continuation hint).

NOTE: the threshold-gate, post-turn dispatch, and max-iteration-bound test
classes that lived here before were removed as part of the auto-context rework
(action + chain replace max_iterations; the journal branch of the auto path is
gone). Gate behaviour is covered by TestAutoContextGate in test_auto_context.py;
the chain / restart-after-compact behaviour is covered by TestAutoContextChain
(added with the Stage 3 dispatch rework).
"""

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestStaleGroundingGuard:
    """Stale-grounding guard: prompt_tokens describes the request that
    contained messages[0:cursor]. If the history has since shrunk below
    that point (a compact rewrote the messages), the number is dead and
    must not drive the estimate. If the history grew - including
    tool-heavy history that tokenizes far worse than chars/4 - the
    grounded request is a prefix of the live history and the count is
    trusted."""

    def test_stale_precompact_count_falls_back_to_estimate(self):
        """The 2026-09-28 incident: /compact left a 2-message history (~14k)
        while prompt_tokens still carried 259213 (the compact call's
        request). The estimate must be the small post-compact size."""
        agent = make_agent()
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [
            {"role": "user", "content": "Session summary request"},
            {"role": "assistant", "content": "Compacted session summary. " * 20},
        ]
        agent.prompt_tokens = 259213  # stale pre-compact grounded count
        agent._msg_count_before_stream = 126  # points past the rewritten history

        est = agent._estimate_current_context_tokens()
        # Far below the stale 259k; above the system-prompt floor
        assert est < 10000
        assert est >= 1000

    def test_fresh_grounding_unchanged(self):
        """A grounded count consistent with the current history passes
        through untouched - the guard must not distort normal operation."""
        agent = make_agent()
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [{"role": "user", "content": "hello world " * 100}]
        agent.prompt_tokens = 1500  # ~1000 system + ~275 history + ~225 tools
        agent._msg_count_before_stream = len(agent.messages)

        assert agent._estimate_current_context_tokens() == 1500

    def test_image_session_not_falsely_stale(self):
        """A grounded count that includes a large base64 image payload is
        legitimate: the ceiling counts the payload at REAL size, so the
        guard must not classify it as stale. (The chars/4 message estimate
        alone says ~170 tokens for the image and would falsely trip,
        masking the true size and suppressing auto-context.)"""
        agent = make_agent()
        agent.system_prompt = "S" * 4000
        b64 = "A" * 400_000  # ~100k tokens of base64 payload
        agent.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]
        # Real API count: ~1000 system + ~100k image + a little
        agent.prompt_tokens = 102_000
        agent._msg_count_before_stream = len(agent.messages)

        est = agent._estimate_current_context_tokens()
        # Grounded value passes through - not clipped to the ~170-token
        # estimate of the image block
        assert est == 102_000

    def test_tool_heavy_history_keeps_grounded_count(self):
        """The 2026-09-29 misfire: tool-result-heavy history tokenizes up to
        ~3.5x worse than chars/4, so a GENUINE grounded count far exceeds
        the chars/4 size of the same messages. The history has not shrunk
        below the grounding point, so the count is trusted - the old
        ceiling test vetoed it and the threshold never fired all day."""
        agent = make_agent()
        agent.system_prompt = "S" * 4000
        # ~40k chars of tool-heavy content -> chars/4 ~= 10k, but the real
        # request charged 237k for this content
        agent.messages = [
            {"role": "user", "content": "please continue"},
            {"role": "assistant", "content": "x" * 20_000},
            {"role": "tool", "content": "y" * 20_000},
        ]
        agent.prompt_tokens = 237_227
        agent._msg_count_before_stream = len(agent.messages)

        est = agent._estimate_current_context_tokens()
        assert est == 237_227


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
    async def test_regrounds_estimator_state(self):
        """After a compact, prompt_tokens and the stream cursor are reset to
        the POST-compact size - the compact call's own token_usage (the
        pre-compact request) must not leak into the next pre-turn check,
        which would compare 259k against the threshold on a 2-message
        history and fire a spurious wrap-up (observed 2026-09-28 22:37)."""
        agent = make_agent(auto_context_threshold=180_000)
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [{"role": "user", "content": "old history " * 200}]
        agent.prompt_tokens = 259_213  # stale pre-compact grounded count
        agent._msg_count_before_stream = 100  # cursor past the rewrite

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            yield (
                "content",
                "This is the compacted session summary. " * 5,  # 300 chars
            )
            yield (
                "token_usage",
                {
                    "prompt_tokens": 259_213,
                    "completion_tokens": 100,
                    "total_tokens": 259_313,
                },
            )

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            ok, _ = await agent.compact_history("")

        assert ok
        # Grounded count reset to the post-compact estimate (not 259k)
        assert agent.prompt_tokens < 5000
        # Cursor reset to the rewritten history length (added=[] next check)
        assert agent._msg_count_before_stream == 2
        # The value the pre-turn check reads is post-compact: below the
        # estimate ceiling AND below the threshold
        est = agent._estimate_current_context_tokens()
        assert est < 5000
        assert est < agent.auto_context_threshold

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

    @pytest.mark.asyncio
    async def test_wrapup_not_current_after_compact(self):
        """A compact replaces the history with the summary pair, erasing the
        wrap-up marker - so _wrapup_is_current() reverts to False and the next
        plain /compact uses the verbose default (no separate flag)."""
        agent = make_agent()
        agent.system_prompt = "S" * 4000
        agent.messages = [
            {
                "role": "assistant",
                "content": "[Wrapping up]",
                "report_and_compact": True,
            },
            {
                "role": "user",
                "content": "wrap-up prompt",
                "report_and_compact": True,
            },
            {"role": "assistant", "content": "Wrap-up complete."},
        ]
        assert agent._wrapup_is_current() is True

        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            yield ("content", "This is the compacted session summary. " * 5)
            yield (
                "token_usage",
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                },
            )

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            ok, _ = await agent.compact_history("")

        assert ok
        # The compact replaced the history - the wrap-up marker is gone.
        assert agent._wrapup_is_current() is False


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


class TestDurableRecordPointer:
    """A verbose compact summary carries the durable-record pointer block
    when the record paths are known (captured from the wrap-up's own file
    writes). The 09-29 games incident: the model-written verbose summary
    named no record files, the new context had no pointer, and the agent
    re-struggled ~1 h re-discovering file-resident knowledge. The block is
    appended mechanically - a model-written pointer is exactly what dropped.
    """

    SUMMARY_TEXT = "Model-written summary of the session."

    async def _compact(self, agent, compact_prompt=""):
        async def mock_stream(client, model, messages, system_prompt, tools, **kw):
            yield ("content", self.SUMMARY_TEXT)
            yield (
                "token_usage",
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 10,
                    "total_tokens": 1010,
                },
            )

        with patch("agent13.llm.stream_response_with_tools", side_effect=mock_stream):
            ok, _ = await agent.compact_history(compact_prompt)
        assert ok
        # The history is the replacement pair; the summary is the assistant msg
        return agent.messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_verbose_summary_carries_pointer_block(self):
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]
        agent._session_truth_path = "docs_archive/auto_context_current_truth.md"
        agent._session_journal_path = "docs_archive/auto_context_journal.md"

        summary = await self._compact(agent)

        # The model-written summary is intact, with the block appended after
        assert summary.startswith(self.SUMMARY_TEXT)
        assert "Durable record (focus auto_context):" in summary
        assert "docs_archive/auto_context_current_truth.md" in summary
        assert "docs_archive/auto_context_journal.md" in summary
        # The cues ride with the paths
        assert "work queue (top unblocked ticket first)" in summary
        assert "LAST entry holds the state at handoff" in summary

    @pytest.mark.asyncio
    async def test_no_paths_no_block(self):
        """Fresh restore with no in-session wrap-up: no captured paths, so
        the summary is untouched (out of scope)."""
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]

        summary = await self._compact(agent)

        assert summary == self.SUMMARY_TEXT

    @pytest.mark.asyncio
    async def test_user_specified_prompt_untouched(self):
        """A user-specified compact prompt wins clean - no pointer block."""
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]
        agent._session_truth_path = "docs_archive/auto_context_current_truth.md"
        agent._session_journal_path = "docs_archive/auto_context_journal.md"

        summary = await self._compact(agent, compact_prompt="My custom prompt.")

        assert summary == self.SUMMARY_TEXT

    @pytest.mark.asyncio
    async def test_bridge_prompt_untouched(self):
        """The short bridge summary prompt already instructs the model to
        name both files - no mechanical block on top."""
        from agent13.prompts import DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT

        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]
        agent._session_truth_path = "docs_archive/auto_context_current_truth.md"
        agent._session_journal_path = "docs_archive/auto_context_journal.md"

        summary = await self._compact(
            agent, compact_prompt=DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT
        )

        assert summary == self.SUMMARY_TEXT

    @pytest.mark.asyncio
    async def test_journal_only_pointer(self):
        """Only the journal path known: block carries the journal line and
        the project name derived from the journal file name."""
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]
        agent._session_journal_path = "docs_archive/games_journal.md"

        summary = await self._compact(agent)

        assert "Durable record (focus games):" in summary
        assert "docs_archive/games_journal.md" in summary
        assert "current_truth" not in summary

    @pytest.mark.asyncio
    async def test_unrecognizable_names_bare_header(self):
        """Paths that don't follow the <focus>_current_truth/_journal
        convention: bare header, no invented project name."""
        agent = make_agent()
        agent.messages = [{"role": "user", "content": "history " * 100}]
        agent._session_truth_path = "docs_archive/notes.md"

        summary = await self._compact(agent)

        assert "Durable record:" in summary
        assert "(project" not in summary
        assert "docs_archive/notes.md" in summary


class TestModelSwapGrounding:
    """A /model swap must invalidate the grounded prompt_tokens: counts are
    tokenizer-specific (10.2% measured Swift vs DeepSeek, 2026-09-29), and
    the stale-grounding guard cannot detect a swap - the message count is
    unchanged, so the old tokenizer's grounded count would be trusted
    as-is (e.g. a 237k count driving auto-context decisions on a history
    the new tokenizer charges ~213k for)."""

    def test_set_model_regrounds_to_estimate(self):
        agent = make_agent(auto_context_threshold=180_000)
        agent.system_prompt = "S" * 4000  # ~1000 tokens
        agent.messages = [{"role": "user", "content": "history " * 200}]
        agent.prompt_tokens = 237_227  # the old model's grounded count
        agent._msg_count_before_stream = 1

        agent.set_model("other-model")

        # Re-grounded to the fresh estimate (system + chars/4 of messages),
        # not the stale 237k
        assert agent.prompt_tokens < 5000
        # Cursor at the end of the history (added=[] at the next check)
        assert agent._msg_count_before_stream == 1

    def test_set_model_same_model_keeps_grounding(self):
        agent = make_agent()
        agent.system_prompt = "S" * 4000
        agent.messages = [{"role": "user", "content": "history " * 200}]
        agent.prompt_tokens = 237_227
        agent._msg_count_before_stream = 1

        agent.set_model("test-model")  # the current model

        assert agent.prompt_tokens == 237_227
        assert agent._msg_count_before_stream == 1

    def test_estimate_not_stale_after_swap(self):
        """The pre-turn check's input after a swap is the fresh estimate
        (below the threshold) - not the stale grounded count, which would
        fire a spurious wrap-up on a tiny history."""
        agent = make_agent(auto_context_threshold=180_000)
        agent.system_prompt = "S" * 4000
        agent.messages = [{"role": "user", "content": "history " * 200}]
        agent.prompt_tokens = 237_227
        agent._msg_count_before_stream = 1

        agent.set_model("other-model")

        est = agent._estimate_current_context_tokens()
        assert est < 5000
        assert est < agent.auto_context_threshold


class TestPostTurnDecisionLog:
    """The post-turn handler's terminal decision is logged (reason + chain +
    chain_used) so a budget-exhausted idle is never silent in debug.log - the
    09-28 02:48 "silent stall" was by design (chain 5/5 spent) but the TUI-only
    idle notification was never logged, so the log went quiet and the idle
    looked like a silent no-op. The restart path (which loops back into
    _llm_turn) is not a terminal decision and logs nothing."""

    async def _decisions(self, agent, restart, compact_result):
        agent._auto_context_triggered = True
        agent._report_and_compact_triggered = False
        calls = []

        def fake_log(event_type, data=None):
            calls.append((event_type, data))

        with (
            patch("agent13.core.is_debug_enabled", return_value=True),
            patch("agent13.core.log_tps_event", side_effect=fake_log),
            patch.object(agent, "_save_auto_context_snapshot"),
            patch.object(
                agent, "compact_history", new=AsyncMock(return_value=compact_result)
            ),
        ):
            await agent._run_auto_context_post_turn(restart=restart)

        return [d for et, d in calls if et == "auto_context_decision"]

    @pytest.mark.asyncio
    async def test_chain_exhausted_logs_decision(self):
        agent = make_agent(auto_context_threshold=180_000)
        agent.auto_context_chain = 5
        agent._auto_context_chain_used = 5  # budget spent (the 09-28 case)

        decisions = await self._decisions(
            agent, restart=True, compact_result=(True, "ok")
        )

        assert len(decisions) == 1
        assert decisions[0]["reason"] == "chain_exhausted"
        assert decisions[0]["chain"] == 5
        assert decisions[0]["chain_used"] == 5

    @pytest.mark.asyncio
    async def test_pre_turn_logs_continuing(self):
        agent = make_agent(auto_context_threshold=180_000)
        agent.auto_context_chain = 5
        agent._auto_context_chain_used = 0

        decisions = await self._decisions(
            agent, restart=False, compact_result=(True, "ok")
        )

        assert len(decisions) == 1
        assert decisions[0]["reason"] == "continuing"
        assert decisions[0]["chain_used"] == 0

    @pytest.mark.asyncio
    async def test_compact_failed_logs_decision(self):
        agent = make_agent(auto_context_threshold=180_000)
        agent.auto_context_action = "compact"
        agent.auto_context_chain = 5
        agent._auto_context_chain_used = 1

        decisions = await self._decisions(
            agent, restart=True, compact_result=(False, "boom")
        )

        assert len(decisions) == 1
        assert decisions[0]["reason"] == "compact_failed"
