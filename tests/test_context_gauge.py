"""Tests for the context gauge (agent13.context_gauge).

The gauge is the single store for the grounded prompt-token value and the
stream cursor, and owns the stale-grounding (shrink) guard + chars/4
fallback. These tests feed it plain data — no Agent, no LLM — which is the
point of the extraction: the grounding state machine (three incidents:
09-28 spurious wrap-up, 09-29 threshold misfire, 09-30 model swap) is now
testable in isolation.
"""

from unittest.mock import MagicMock

from agent13.context_gauge import ContextGauge, estimate_message_tokens
from agent13.core import Agent


# ─────────────────────────── estimate_message_tokens ───────────────────────────


class TestEstimateMessageTokens:
    def test_empty_list_is_zero(self):
        assert estimate_message_tokens([]) == 0

    def test_str_content_is_chars_over_4(self):
        assert estimate_message_tokens([{"role": "user", "content": "x" * 400}]) == 100

    def test_none_content_is_zero(self):
        assert estimate_message_tokens([{"role": "assistant", "content": None}]) == 0

    def test_tool_call_arguments_counted(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "' + "x" * 400 + '"}',
                    }
                }
            ],
        }
        # '{"path": "' + 400 x's + '"}' = 412 chars -> 103
        assert estimate_message_tokens([msg]) == 103

    def test_text_blocks_in_list_content(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "y" * 800}],
        }
        assert estimate_message_tokens([msg]) == 200

    def test_image_block_is_fixed_170_not_base64_length(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100000}},
            ],
        }
        assert estimate_message_tokens([msg]) == 170 + 1


# ───────────────────────────────── the gauge ─────────────────────────────────


class TestContextGauge:
    def test_initial_state_is_ungrounded(self):
        gauge = ContextGauge()
        assert gauge.grounded == 0
        assert gauge.cursor == 0

    def test_estimate_ungrounded_is_chars_over_4_of_history(self):
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 4000}]
        # Ungrounded (cursor=0, grounded=0): the estimate is the chars/4 of
        # the messages — the system prompt only enters via the shrink
        # fallback (preserved from the original estimator).
        assert gauge.estimate(messages, system_len=400) == 1000

    def test_ground_then_estimate_adds_only_new_messages(self):
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 4000}]
        gauge.record_stream_start(len(messages))
        gauge.record_usage(150_000)
        # The grounded request covered all of `messages`; nothing added yet.
        assert gauge.estimate(messages, system_len=400) == 150_000
        # A tool result lands after the grounded request.
        messages.append({"role": "tool", "content": "y" * 8000})
        assert gauge.estimate(messages, system_len=400) == 150_000 + 2000

    def test_cursor_advances_per_stream(self):
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 4000}]
        gauge.record_stream_start(1)
        gauge.record_usage(100_000)
        messages.append({"role": "tool", "content": "y" * 4000})
        # Second stream: the grounded request now covers 2 messages.
        gauge.record_stream_start(2)
        gauge.record_usage(120_000)
        assert gauge.estimate(messages, system_len=400) == 120_000

    def test_shrink_below_cursor_falls_back_to_chars_over_4(self):
        """The 09-28 incident: a compact rewrites the history to a 2-message
        pair BELOW the cursor (the grounded request had 100 messages) — the
        grounded value must not drive the decision."""
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 4000}]
        gauge.record_stream_start(100)
        gauge.record_usage(259_213)
        # Compact rewrites the history to a 2-message summary pair.
        messages[:] = [
            {"role": "user", "content": "summary seed"},
            {"role": "assistant", "content": "summary text " * 10},
        ]
        # Fallback: system + chars/4 of the NEW history, not 259k.
        expected = 400 // 4 + estimate_message_tokens(messages)
        assert gauge.estimate(messages, system_len=400) == expected
        assert gauge.estimate(messages, system_len=400) < 5000

    def test_growth_past_cursor_is_trusted(self):
        """The 09-29 lesson: a genuine grounded count must not be vetoed
        just because chars/4 of the same content is much smaller."""
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 4000}]
        gauge.record_stream_start(1)
        gauge.record_usage(237_000)
        # History grows (the normal flow) — the grounded count is trusted.
        messages.append({"role": "tool", "content": "y" * 4000})
        assert gauge.estimate(messages, system_len=400) == 237_000 + 1000

    def test_record_usage_zero_ungrounds_without_moving_cursor(self):
        """reset_token_usage() semantics: the plain assignment set the value
        to 0 and left the cursor alone."""
        gauge = ContextGauge()
        gauge.record_stream_start(1)
        gauge.record_usage(99_000)
        gauge.record_usage(0)
        assert gauge.grounded == 0
        assert gauge.cursor == 1

    def test_reground_sets_value_and_cursor(self):
        gauge = ContextGauge()
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        value = gauge.reground(messages, system_len=400)
        assert value == 100 + 200
        assert gauge.grounded == value
        assert gauge.cursor == 2
        # After a reground, added=[] — the estimate IS the regrounded value.
        assert gauge.estimate(messages, system_len=400) == value

    def test_reground_includes_tools_len(self):
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "a" * 400}]
        value = gauge.reground(messages, system_len=400, tools_len=2000)
        assert value == 100 + 100 + 2000

    def test_reground_without_tools_len(self):
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "a" * 400}]
        assert gauge.reground(messages, system_len=400, tools_len=None) == 200
        assert gauge.reground(messages, system_len=400) == 200

    def test_reground_after_shrink_restores_normal_flow(self):
        """The compact_history / _mechanical_restart path: rewrite below the
        cursor, reground, then append — the estimate tracks the new history."""
        gauge = ContextGauge()
        messages = [{"role": "user", "content": "x" * 8000}]
        gauge.record_stream_start(1)
        gauge.record_usage(200_000)
        messages[:] = [{"role": "assistant", "content": "summary " * 20}]
        value = gauge.reground(messages, system_len=400)
        messages.append({"role": "user", "content": "nudge " * 10})
        # "nudge " * 10 = 60 chars -> 15 tokens added past the reground.
        assert gauge.estimate(messages, system_len=400) == value + 15


# ─────────────────────── Agent forwarding properties ───────────────────────


def make_agent():
    return Agent(client=MagicMock(), model="test-model")


class TestAgentForwarding:
    def test_prompt_tokens_property_roundtrip(self):
        agent = make_agent()
        assert agent.prompt_tokens == 0
        agent.prompt_tokens = 123_456
        assert agent.prompt_tokens == 123_456
        # Same store the gauge methods see.
        assert agent._context_gauge.grounded == 123_456

    def test_msg_count_before_stream_property_roundtrip(self):
        agent = make_agent()
        assert agent._msg_count_before_stream == 0
        agent._msg_count_before_stream = 7
        assert agent._msg_count_before_stream == 7
        assert agent._context_gauge.cursor == 7

    def test_estimate_forwards_to_gauge(self):
        agent = make_agent()
        agent.system_prompt = "S" * 400
        agent.messages = [{"role": "user", "content": "x" * 4000}]
        agent.prompt_tokens = 150_000
        agent._msg_count_before_stream = 1
        assert agent._estimate_current_context_tokens() == 150_000

    def test_invalidate_context_grounding_regrounds_gauge(self):
        agent = make_agent()
        agent.system_prompt = "S" * 400
        agent.messages = [{"role": "user", "content": "x" * 4000}]
        agent.prompt_tokens = 250_000
        agent._msg_count_before_stream = 99
        agent._invalidate_context_grounding()
        assert agent.prompt_tokens == 100 + 1000
        assert agent._msg_count_before_stream == 1

    def test_reset_token_usage_ungrounds(self):
        agent = make_agent()
        agent.prompt_tokens = 99_000
        agent._msg_count_before_stream = 3
        agent.reset_token_usage()
        assert agent.prompt_tokens == 0
        # Cursor untouched (the shrink guard handles the rewritten history).
        assert agent._msg_count_before_stream == 3
