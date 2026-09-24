"""Integration tests for auto-context (LLM mocked only).

Spawns a REAL REPL process against a mock LLM server that captures every
chat request, then verifies the auto-context behavior end to end:

  1. The wrap-up prompt reaches the wire as the last user message,
     preceded by a fake assistant message (role alternation), with the
     full un-compacted conversation still visible.
  2. report_and_compact fires once per chain cycle: chain=N means N+1
     actions and N continue-hints per user turn (the chain gates
     restarts, never the action).
  3. The chain budget is per-turn: a second user message gets a fresh
     budget and runs the full cycle again.
  4. action=compact, chain=0: one compact request, no continue-hint,
     agent back at the prompt.
  5. action=none: no wrap-up or compact hits the server; the agent
     pauses with the notice.
"""

import json
import os

import pytest
import pytest_httpserver
from werkzeug import Request, Response

from .helpers import spawn_process
from .mock_llm_helpers import make_models_handler

# Needles identifying the three auto-context injections on the wire.
WRAPUP_NEEDLE = "CONTEXT IS NEARLY FULL"  # DEFAULT_REPORT_AND_COMPACT_PROMPT
COMPACT_NEEDLE = "Summarize our conversation"  # DEFAULT_COMPACT_PROMPT
HINT_NEEDLE = "Continue work on the outstanding items"  # AUTO_CONTEXT_CONTINUE_HINT


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def capturing_llm_server():
    """Mock LLM server that records every chat request body.

    Mock design notes (the first two details are non-obvious and cost two
    false starts in the original build - do not remove):

    1. Every request whose last user message is a real user message or the
       auto-context continue hint returns a `write_file` tool call. This
       fixture exercises the MID-TURN safe point (the check after tool
       execution); a plain-text (final-response) turn trips the separate
       turn-start check on the NEXT user message instead — see
       text_only_llm_server. A restarted (chained) turn must therefore get a
       tool call, or it ends without re-tripping mid-turn and the chain
       cycle can never run its second action.
    2. Every response carries a usage chunk with `prompt_tokens: 200`
       (threshold 100) - the safe-point estimate is `prompt_tokens + tokens
       appended since stream start`, so the check re-trips at the next safe
       point.

    Wrap-up and compact calls (detected by their last user message) return
    plain text so their turns end immediately.

    Each response carries a unique "Mock response N" marker so tests can
    wait for a specific turn unambiguously.
    """
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        users = [
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "user"
        ]
        last = users[-1] if users else ""
        ends_turn = WRAPUP_NEEDLE in last or COMPACT_NEEDLE in last

        # Report a prompt_tokens count over the configured threshold (100)
        # so the safe-point estimate trips the auto-context path. The
        # estimate = prompt_tokens + tokens appended since stream start.
        # Compact calls report a small count: after a compact the context is
        # genuinely small, so the next turn's pre-turn estimate must not trip
        # (a constant 200 would make every post-compact state look
        # over-threshold and fire the pre-turn check spuriously).
        usage = {
            "prompt_tokens": 50 if COMPACT_NEEDLE in last else 200,
            "completion_tokens": 10,
            "total_tokens": 60 if COMPACT_NEEDLE in last else 210,
        }
        parts = []
        if not ends_turn:
            # A tool call, so the turn reaches the safe point AFTER tool
            # execution where the threshold check lives. Without a tool call
            # the turn simply ends and the check never runs.
            parts.append(
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "write_file",
                                            "arguments": '{"path": "report.md", "content": "notes"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            parts.append(
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                    "usage": usage,
                }
            )
        else:
            content = f"Mock response {n}. Progress report placeholder."
            parts.append(
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            parts.append(
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                }
            )

        sse = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
        return Response(sse, content_type="text/event-stream")

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def repl_env(tmp_path, capturing_llm_server, request):
    """Temp config dir pointing at the capturing mock server.

    Threshold is tiny so a single short message trips it. Indirect
    parameter is an (action, chain) tuple.
    """
    server = capturing_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    action, chain = getattr(request, "param", ("compact", 0))
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 100
action = "{action}"
chain = {chain}
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


@pytest.fixture
def text_only_llm_server():
    """Mock LLM server that returns plain text (no tool calls) for every
    request — turns end with a final response, so the threshold can only be
    tripped by the TURN-START check on the next user message (the
    resume-bug path), never by the mid-turn safe point. Usage carries
    prompt_tokens: 200 (threshold 100) so the pre-turn estimate is over the
    limit after the first turn.

    Each response carries a unique "Mock response N" marker so tests can
    wait for a specific turn unambiguously.
    """
    captured = []
    call_count = 0

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        usage = {
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "total_tokens": 210,
        }
        content = f"Mock response {n}. Plain text, no tool calls."
        parts = [
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ]

        sse = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
        return Response(sse, content_type="text/event-stream")

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def text_only_repl_env(tmp_path, text_only_llm_server, request):
    """Same as repl_env but pointing at the text-only mock server."""
    server = text_only_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    action, chain = getattr(request, "param", ("compact", 0))
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 100
action = "{action}"
chain = {chain}
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


def spawn_repl(env, timeout=30, extra_args=None):
    args = ["run", "agent13", "test_mock", "--repl", "--model", "mock-model"]
    if extra_args:
        args += extra_args
    proc = spawn_process(
        "uv",
        args=args,
        env=env,
        encoding="utf-8",
        timeout=timeout,
        dimensions=(50, 200),
        maxread=8192,
    )
    proc.timeout = timeout
    proc.expect(r">", timeout=timeout)
    return proc


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def all_user_messages(body):
    """All user-role message contents from a chat request body."""
    return [
        m.get("content", "")
        for m in body.get("messages", [])
        if m.get("role") == "user"
    ]


def last_user_message(body):
    """The last user-role message content from a chat request body."""
    users = all_user_messages(body)
    return users[-1] if users else ""


def count_requests_ending_with(captured, needle):
    """Count requests whose LAST user message contains `needle`.

    Counting only the last user message isolates the injection itself
    (a wrap-up prompt, a compact prompt, a continue hint) from the same
    text still sitting earlier in the history of later requests within
    the same chain cycle.
    """
    return sum(1 for b in captured if needle in last_user_message(b))


def roles_in_order(body):
    """Message roles in wire order (assistant entries with tool_calls count)."""
    return [m.get("role") for m in body.get("messages", [])]


def has_local_key(body, key):
    """True if any message in a wire payload carries the given local flag key.

    Local flags (``LOCAL_MSG_KEYS``) are stripped by
    ``build_messages_with_system`` before the request goes out; a hit here
    means a local bookkeeping field leaked onto the provider wire (F1)."""
    return any(key in m for m in body.get("messages", []))


# ────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "repl_env",
    [pytest.param(("report_and_compact", 0), id="rac-chain0")],
    indirect=True,
)
class TestAutoContextExperience:
    """action=report_and_compact, chain=0: wrap-up, then compact, then idle."""

    def test_wrapup_prompt_reaches_wire_and_compact_does_not(
        self, repl_env, capturing_llm_server
    ):
        """Wrap-up prompt sent on the wire with the full un-compacted
        conversation; the compaction happens only AFTER the wrap-up."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            # Terminal state: with chain=0 the idle notice fires after the
            # post-wrap-up compact, so every LLM call of the turn has
            # completed by the time we assert.
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 3, (
                f"expected exactly 3 LLM calls (turn + wrap-up + compact), got {len(captured)}"
            )
            wrapup = captured[1]
            users = all_user_messages(wrapup)

            # The built-in wrap-up prompt is the last user message...
            assert WRAPUP_NEEDLE in users[-1]
            assert "journal" in users[-1]
            # ...preceded by the fake assistant marker (role alternation).
            assert roles_in_order(wrapup)[-2:] == ["assistant", "user"]

            # No premature compaction: the wrap-up request still contains
            # the full original conversation, and the compact request
            # comes after it.
            assert any("hello" in u for u in users)
            assert COMPACT_NEEDLE in last_user_message(captured[2])

            # F1: the report_and_compact marker is local bookkeeping and must
            # never reach the provider. Check every captured payload, not just
            # the wrap-up - the marker rides both injected wrap-up messages.
            for payload in captured:
                assert not has_local_key(payload, "report_and_compact"), (
                    "local marker 'report_and_compact' leaked onto the wire: "
                    f"{[list(m.keys()) for m in payload.get('messages', [])]}"
                )
        finally:
            proc.close()

    def test_config_key_only_no_flag(self, repl_env, capturing_llm_server):
        """Feature activates from config.toml alone (no CLI flag)."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 1
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 1
        finally:
            proc.close()


@pytest.mark.parametrize(
    "repl_env",
    [pytest.param(("report_and_compact", 1), id="rac-chain1")],
    indirect=True,
)
class TestAutoContextChain:
    """action=report_and_compact, chain=1: two actions, one restart, per turn."""

    def test_fires_once_per_chain_cycle(self, repl_env, capturing_llm_server):
        """chain=1 -> exactly 2 wrap-up prompts, 2 compact requests,
        1 continue-hint on the wire, then idle.

        The N+1 rule: the chain gates restarts, never the action. The
        restarted turn gets a tool call from the mock, re-trips the
        threshold at its safe point, and runs the second action before the
        exhausted budget sends the agent idle.
        """
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            # The idle notice fires only when the chain budget is exhausted,
            # i.e. after the last compact of the cycle.
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 6, (
                "expected exactly 6 LLM calls "
                "(turn, wrap-up, compact, restart, wrap-up, compact), "
                f"got {len(captured)}"
            )
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 2
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 2
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 1
        finally:
            proc.close()

    def test_per_turn_reset(self, repl_env, capturing_llm_server):
        """After the chain idles, a second user message gets a fresh budget."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)
            assert len(captured) == 6

            # Second turn: the chain counter resets when the new user message
            # is processed, so the full cycle (action, restart, action) runs
            # again.
            proc.sendline("second message")
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 12
            # A second continue-hint proves the budget was fresh: with the
            # turn-1 counter (1 of 1 restarts used) still in effect, the
            # second turn would have compacted once and gone idle without
            # restarting.
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 2
        finally:
            proc.close()


@pytest.mark.parametrize(
    "repl_env", [pytest.param(("compact", 0), id="compact-chain0")], indirect=True
)
class TestCompactAction:
    """Control group: action=compact, chain=0 -> one compact, then idle."""

    def test_compact_chain_zero_compacts_then_idles(
        self, repl_env, capturing_llm_server
    ):
        """action=compact, chain=0 -> one compact request, zero
        continue-hints, agent back at the prompt."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 2
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 1
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 0
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 0
            # Idle (not paused): the agent is back at the prompt.
            proc.expect(r">", timeout=10)
        finally:
            proc.close()


@pytest.mark.parametrize(
    "repl_env", [pytest.param(("none", 0), id="none-chain0")], indirect=True
)
class TestNoneAction:
    """action=none: the threshold pauses without touching the context."""

    def test_none_pauses_without_compacting(self, repl_env, capturing_llm_server):
        """No compact or wrap-up requests hit the server; the pause
        notice appears in the output."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect(r"paused \(action=none\)", timeout=30)

            # Only the original turn reached the wire.
            assert len(captured) == 1
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 0
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 0
        finally:
            proc.close()


class TestAutoContextFlagOverride:
    def test_cli_flag_enables_when_config_off(self, tmp_path, capturing_llm_server):
        """--report-and-compact overrides a compact config value."""
        server = capturing_llm_server[0]
        config_dir = tmp_path / "agent13-config"
        config_dir.mkdir()

        config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 100
action = "compact"
chain = 0
"""
        (config_dir / "config.toml").write_text(config_content)

        env = os.environ.copy()
        env["AGENT13_CONFIG_DIR"] = str(config_dir)
        env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
        env["AGENT13_NO_UPDATE_CHECK"] = "1"

        _, captured = capturing_llm_server
        proc = spawn_repl(env, extra_args=["--report-and-compact"])

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 1
        finally:
            proc.close()


@pytest.mark.parametrize(
    "text_only_repl_env",
    [pytest.param(("compact", 0), id="compact-chain0")],
    indirect=True,
)
class TestPreTurnThreshold:
    """A turn that ends with a tool-call-free final response over the
    threshold is LEFT ALONE (a finished turn is a natural stopping point);
    the compact fires at the START of the next turn, before its first LLM
    call. The mid-turn safe point can never fire here — the mock returns no
    tool calls."""

    def test_final_response_turn_compacts_at_next_turn_start(
        self, text_only_repl_env, text_only_llm_server
    ):
        """Turn 1 (final response) → no compact. Turn 2 → compact before the
        turn, then the turn itself: 3 requests total."""
        _, captured = text_only_llm_server
        proc = spawn_repl(text_only_repl_env)

        try:
            proc.sendline("hello")
            proc.expect("Mock response 1", timeout=30)
            proc.expect(r">", timeout=10)
            # Turn 1 alone: the finished turn is left over-threshold.
            assert len(captured) == 1
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 0

            proc.sendline("world")
            proc.expect("Mock response 3", timeout=30)
            # Turn 2: the pre-turn compact (request 2) ran before the turn
            # (request 3); no wrap-up, no continue-hint.
            assert len(captured) == 3
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 1
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 0
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 0
            # Idle (not paused): the agent is back at the prompt.
            proc.expect(r">", timeout=10)
        finally:
            proc.close()
