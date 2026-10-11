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

# Needles identifying the auto-context injections on the wire.
WRAPUP_NEEDLE = "CONTEXT IS NEARLY FULL"  # DEFAULT_REPORT_AND_COMPACT_PROMPT
COMPACT_NEEDLE = "Summarize our conversation"  # DEFAULT_COMPACT_PROMPT
# Short bridge summary the auto path sends instead of the verbose compact
# prompt when action = report_and_compact (DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT).
SUMMARY_NEEDLE = "Write a SHORT bridge summary"
HINT_NEEDLE = "Continue work on the outstanding items"  # AUTO_CONTEXT_CONTINUE_HINT
CARRY_NEEDLE = "standing instruction for this run"  # AUTO_CONTEXT_CARRY_HINT


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
    2. Every in-turn response carries a usage chunk with
       `prompt_tokens: 40_000` (threshold 20_000) - the safe-point estimate
       is `prompt_tokens + tokens appended since stream start`, so the check
       re-trips at the next safe point. The claimed size must stay below the
       stale-grounding ceiling (raw content chars/4 + system + 50k tools
       slack, ~50k for these tiny conversations) or the estimator will
       (correctly) treat the claim as stale and ignore it.

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
        # A compact call ends the turn too; with action = report_and_compact
        # it carries the SHORT bridge summary instead of the verbose prompt.
        ends_turn = (
            WRAPUP_NEEDLE in last or COMPACT_NEEDLE in last or SUMMARY_NEEDLE in last
        )

        # Report a prompt_tokens count over the configured threshold
        # (20_000) so the safe-point estimate trips the auto-context path.
        # The estimate = prompt_tokens + tokens appended since stream start.
        # The threshold itself sits ABOVE the post-compact baseline (system
        # prompt + tools schema + compact pair, ~5-10k with the real toolset)
        # - a threshold below that baseline would make the pre-turn check
        # fire on every turn, because compact_history re-grounds
        # prompt_tokens to that honest post-compact estimate. And the claimed
        # in-turn size stays below the stale-grounding ceiling (~50k for
        # these tiny conversations) so the estimator trusts it.
        is_compact_call = COMPACT_NEEDLE in last or SUMMARY_NEEDLE in last
        usage = {
            "prompt_tokens": 50 if is_compact_call else 40_000,
            "completion_tokens": 10,
            "total_tokens": 60 if is_compact_call else 40_010,
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

    Threshold (20_000) sits above the post-compact baseline (system + tools
    schema + compact pair, ~5-10k) and below the mock's in-turn reported
    size (40_000), so only the mock's usage - never the re-grounded
    post-compact estimate - can trip the checks. Indirect parameter is an
    (action, chain) tuple.
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
threshold = 20_000
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
    prompt_tokens: 40_000 (threshold 20_000) so the pre-turn estimate is
    over the limit after the first turn.

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
            "prompt_tokens": 40_000,
            "completion_tokens": 10,
            "total_tokens": 40_010,
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
threshold = 20_000
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
        conversation; the restart is mechanical (no compact AI call)."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            # Terminal state: with chain=0 the idle notice fires after the
            # post-wrap-up mechanical restart, so every LLM call of the turn
            # has completed by the time we assert.
            proc.expect("idle, over to you", timeout=30)

            # 2 LLM calls: the original turn + the wrap-up. The restart
            # (clear + intro + seed) is mechanical — no third call.
            assert len(captured) == 2, (
                f"expected exactly 2 LLM calls (turn + wrap-up), got {len(captured)}"
            )
            wrapup = captured[1]
            users = all_user_messages(wrapup)

            # The built-in wrap-up prompt is the last user message...
            assert WRAPUP_NEEDLE in users[-1]
            assert "journal" in users[-1]
            # ...preceded by the fake assistant marker (role alternation).
            assert roles_in_order(wrapup)[-2:] == ["assistant", "user"]

            # No premature compaction: the wrap-up request still contains
            # the full original conversation.
            assert any("hello" in u for u in users)

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
            # The restart is mechanical — no compact AI call on the wire.
            assert count_requests_ending_with(captured, SUMMARY_NEEDLE) == 0
        finally:
            proc.close()


class TestManualWrapupCompact:
    """A manual /wrapup refreshes the durable record, so a subsequent plain
    /compact sends the SHORT bridge summary (not the verbose prompt) -
    matching the auto path. The action setting is irrelevant: the bridge
    selection is driven by the _wrapup_ran_since_compact flag, not the
    action."""

    def test_manual_wrapup_then_compact_uses_bridge(
        self, repl_env, capturing_llm_server
    ):
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/wrapup")
            proc.expect("Mock response 1", timeout=30)
            proc.sendline("/compact")
            proc.expect("Mock response 2", timeout=30)

            assert len(captured) == 2, (
                f"expected exactly 2 LLM calls (wrap-up + compact), "
                f"got {len(captured)}"
            )
            # The wrap-up turn (call 1) carries the wrap-up prompt...
            assert WRAPUP_NEEDLE in last_user_message(captured[0])
            # ...and the manual /compact (call 2) sends the SHORT bridge
            # summary, not the verbose compact prompt.
            compact_last = last_user_message(captured[1])
            assert SUMMARY_NEEDLE in compact_last
            assert COMPACT_NEEDLE not in compact_last
        finally:
            proc.close()


@pytest.fixture
def pointer_llm_server(tmp_path):
    """Mock LLM server for the durable-record pointer scenario.

    Drives: a /wrapup whose model writes BOTH record files (so
    _capture_session_journal_path captures the paths), a plain turn with a
    tool call (invalidates the wrap-up's currency -> the next /compact must
    use the VERBOSE prompt), the /compact itself, and one turn after it (to
    observe the compacted summary on the wire).

    The record paths are absolute tmp paths so the wrap-up's write_file
    calls land in the test's tmp dir, never the repo. Usage stays tiny
    (prompt_tokens: 50) so no auto-context check can fire.
    """
    journal_path = str(tmp_path / "docs_archive" / "pointer_journal.md")
    truth_path = str(tmp_path / "docs_archive" / "pointer_current_truth.md")

    captured = []
    call_count = 0
    wrapup_calls = 0

    usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 65}

    def sse(parts):
        return Response(
            "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    def text_response(content):
        return sse(
            [
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"content": content}, "finish_reason": None}
                    ],
                },
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            ]
        )

    def tool_call_response(name, arguments):
        return sse(
            [
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
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "mock-completion",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": usage,
                },
            ]
        )

    def chat_handler(request: Request):
        nonlocal call_count, wrapup_calls
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

        if WRAPUP_NEEDLE in last:
            # The wrap-up's model writes the journal, then the truth file,
            # then gives its final response.
            wrapup_calls += 1
            if wrapup_calls == 1:
                return tool_call_response(
                    "write_file",
                    json.dumps({"filepath": journal_path, "content": "journal"}),
                )
            if wrapup_calls == 2:
                return tool_call_response(
                    "write_file",
                    json.dumps({"filepath": truth_path, "content": "truth"}),
                )
            return text_response(f"Mock response {n}. wrapped.")
        if COMPACT_NEEDLE in last:
            return text_response(f"Mock compact summary {n}.")
        # Plain turn: one tool call (invalid "path" arg -> harmless error,
        # no file written) so the turn reaches a tool result; the tool call
        # also invalidates the wrap-up's currency for the later /compact.
        # "This turn" = messages after the last user message: the wrap-up's
        # own tool results sit earlier in history and must not count.
        msgs = body.get("messages", [])
        last_user_idx = max(
            (i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1
        )
        if not any(m.get("role") == "tool" for m in msgs[last_user_idx + 1 :]):
            return tool_call_response(
                "write_file", json.dumps({"path": "report.md", "content": "notes"})
            )
        return text_response(f"Mock response {n}.")

    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured, journal_path, truth_path
    server.stop()


@pytest.fixture
def pointer_repl_env(tmp_path, pointer_llm_server):
    """Config for the pointer scenario: the threshold is set so high that
    no auto-context check can fire (the mock's usage is tiny anyway)."""
    server = pointer_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 1000000
action = "report_and_compact"
chain = 0
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


class TestDurableRecordPointerIntegration:
    """A verbose compact (no current wrap-up) carries the durable-record
    pointer block in the summary: the 09-29 games incident - the summary
    named no record files, the new context had no pointer, and the agent
    re-struggled ~1 h. The block is appended mechanically from the paths
    captured at the wrap-up."""

    def test_verbose_compact_summary_carries_pointer(
        self, pointer_repl_env, pointer_llm_server
    ):
        _, captured, journal_path, truth_path = pointer_llm_server
        proc = spawn_repl(pointer_repl_env)

        try:
            # 1. /wrapup: the model writes both record files (paths get
            #    captured), then a final response (R1-R3).
            proc.sendline("/wrapup")
            proc.expect("Mock response 3", timeout=30)
            # 2. A plain turn with a tool call (R4-R5) - invalidates the
            #    wrap-up's currency, so the next /compact is VERBOSE.
            proc.sendline("hello")
            proc.expect("Mock response 5", timeout=30)
            # 3. /compact: verbose prompt (R6) -> summary + pointer block.
            proc.sendline("/compact")
            proc.expect("Mock compact summary 6", timeout=30)
            # 4. One more turn (R7-R8) to observe the compacted summary on
            #    the wire (it gets a tool call, then a final response).
            proc.sendline("next")
            proc.expect("Mock response 8", timeout=30)

            assert len(captured) == 8, f"expected 8 LLM calls, got {len(captured)}"

            # R6: the /compact sent the VERBOSE prompt (the wrap-up is no
            # longer current - a tool call happened after it), not the
            # short bridge summary.
            compact_last = last_user_message(captured[5])
            assert COMPACT_NEEDLE in compact_last
            assert SUMMARY_NEEDLE not in compact_last

            # R7: the compacted summary on the wire carries the pointer
            # block - project name, both record paths, and the cues.
            summaries = [
                m.get("content", "")
                for m in captured[6].get("messages", [])
                if m.get("role") == "assistant"
            ]
            assert any("Mock compact summary 6" in s for s in summaries)
            summary = next(s for s in summaries if "Mock compact summary 6" in s)
            assert "Durable record (focus pointer):" in summary
            assert journal_path in summary
            assert truth_path in summary
            assert "work queue (top unblocked ticket first)" in summary
            assert "LAST entry holds the state at handoff" in summary
        finally:
            proc.close()


@pytest.fixture
def model_swap_llm_server():
    """Text-only mock server advertising TWO models. Every response carries
    prompt_tokens: 40_000 (threshold 20_000 in the env fixture), so the
    grounded count after turn 1 sits OVER the threshold - turn 2's
    pre-turn check can only be kept quiet by the model-swap
    re-grounding. Text-only turns end with a final response, so the
    threshold can only be tripped by the TURN-START check on the next
    user message."""
    captured = []
    call_count = 0

    def models_handler(request: Request):
        return Response(
            json.dumps(
                {
                    "data": [
                        {"id": "mock-model", "object": "model", "owned_by": "test"},
                        {"id": "mock-model-2", "object": "model", "owned_by": "test"},
                    ]
                }
            ),
            content_type="application/json",
        )

    def chat_handler(request: Request):
        nonlocal call_count
        call_count += 1
        n = call_count
        body = request.get_json(force=True)
        captured.append(body)

        usage = {
            "prompt_tokens": 40_000,
            "completion_tokens": 10,
            "total_tokens": 40_010,
        }
        content = f"Mock response {n}."
        parts = [
            {
                "id": "mock-completion",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
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
    server.expect_request("/v1/models").respond_with_handler(models_handler)
    server.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_handler
    )
    server.start()
    yield server, captured
    server.stop()


@pytest.fixture
def model_swap_repl_env(tmp_path, model_swap_llm_server):
    """Config for the model-swap scenario: threshold 20_000 sits below the
    mock's grounded 40_000, so a STALE grounded count would fire the
    pre-turn check on the very next turn."""
    server = model_swap_llm_server[0]
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()
    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 20_000
action = "report_and_compact"
chain = 0
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
    env["AGENT13_NO_UPDATE_CHECK"] = "1"
    return env


class TestModelSwapIntegration:
    """A /model swap must invalidate the grounded prompt_tokens (counts are
    tokenizer-specific). Threshold 20_000, turn 1 grounds at 40_000: without
    the swap re-grounding, turn 2's pre-turn check reads the stale 40_000
    and fires a spurious wrap-up on a 2-message history (the 09-28 08:26
    incident class, via a model swap)."""

    def test_no_spurious_wrapup_after_swap(
        self, model_swap_repl_env, model_swap_llm_server
    ):
        _, captured = model_swap_llm_server
        proc = spawn_repl(model_swap_repl_env)

        try:
            proc.sendline("hello")
            proc.expect("Mock response 1", timeout=30)
            proc.sendline("/model mock-model-2")
            proc.expect("Model set to: mock-model-2", timeout=10)
            proc.sendline("second")
            proc.expect("Mock response 2", timeout=30)

            # No wrap-up fired between the turns: exactly 2 LLM calls
            assert len(captured) == 2, (
                f"expected 2 LLM calls (no wrap-up), got {len(captured)}"
            )
            # Turn 2 reached the NEW model...
            assert captured[1].get("model") == "mock-model-2"
            # ...on the user's message, not a wrap-up prompt
            second_last = last_user_message(captured[1])
            assert "second" in second_last
            assert WRAPUP_NEEDLE not in second_last
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

            assert len(captured) == 4, (
                "expected exactly 4 LLM calls "
                "(turn, wrap-up, restart, wrap-up), "
                f"got {len(captured)}"
            )
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 2
            # Restarts are mechanical — no compact AI calls on the wire.
            assert count_requests_ending_with(captured, SUMMARY_NEEDLE) == 0
            # The restart nudge carries the turn's user instruction verbatim
            # (standing-instruction design) — the bare hint no longer appears.
            assert count_requests_ending_with(captured, CARRY_NEEDLE) == 1
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 0
        finally:
            proc.close()

    def test_per_turn_reset(self, repl_env, capturing_llm_server):
        """After the chain idles, a second user message gets a fresh budget."""
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)
            assert len(captured) == 4

            # Second turn: the chain counter resets when the new user message
            # is processed, so the full cycle (action, restart, action) runs
            # again.
            proc.sendline("second message")
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 8
            # A second carry nudge proves the budget was fresh: with the
            # turn-1 counter (1 of 1 restarts used) still in effect, the
            # second turn would have wrapped up once and gone idle without
            # restarting.
            assert count_requests_ending_with(captured, CARRY_NEEDLE) == 2
            assert count_requests_ending_with(captured, HINT_NEEDLE) == 0
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
        """--auto-context-action overrides a compact config value."""
        server = capturing_llm_server[0]
        config_dir = tmp_path / "agent13-config"
        config_dir.mkdir()

        config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 20_000
action = "compact"
chain = 0
"""
        (config_dir / "config.toml").write_text(config_content)

        env = os.environ.copy()
        env["AGENT13_CONFIG_DIR"] = str(config_dir)
        env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
        env["AGENT13_NO_UPDATE_CHECK"] = "1"

        _, captured = capturing_llm_server
        proc = spawn_repl(
            env, extra_args=["--auto-context-action", "report_and_compact"]
        )

        try:
            proc.sendline("hello")
            proc.expect("idle, over to you", timeout=30)
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 1
        finally:
            proc.close()

    def test_cli_flag_disables_when_config_on(self, tmp_path, capturing_llm_server):
        """--auto-context-action none disables a report_and_compact config."""
        server = capturing_llm_server[0]
        config_dir = tmp_path / "agent13-config"
        config_dir.mkdir()

        config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{server.port}/v1"
api_key = "test-key"

[auto_context]
threshold = 20_000
action = "report_and_compact"
chain = 0
"""
        (config_dir / "config.toml").write_text(config_content)

        env = os.environ.copy()
        env["AGENT13_CONFIG_DIR"] = str(config_dir)
        env["AGENT13_SAVES_DIR"] = str(tmp_path / "saves")
        env["AGENT13_NO_UPDATE_CHECK"] = "1"

        _, captured = capturing_llm_server
        proc = spawn_repl(env, extra_args=["--auto-context-action", "none"])

        try:
            proc.sendline("hello")
            proc.expect(r"paused \(action=none\)", timeout=30)

            # Only the original turn reached the wire.
            assert len(captured) == 1
            assert count_requests_ending_with(captured, WRAPUP_NEEDLE) == 0
            assert count_requests_ending_with(captured, COMPACT_NEEDLE) == 0
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


@pytest.mark.parametrize(
    "repl_env",
    [pytest.param(("report_and_compact", 1), id="rac-chain1")],
    indirect=True,
)
class TestInFlightInstructionCarry:
    """Mid-turn threshold hit on a user QUESTION: the mechanical restart
    carries the question VERBATIM into the nudge (the standing-instruction
    design), so the post-restart turn continues the question instead of
    drifting to the ticket queue. The injected intro no longer commands
    ticket work."""

    QUESTION = "what is the multiplexing rule for agents"

    def test_question_survives_mechanical_restart(
        self, repl_env, capturing_llm_server
    ):
        _, captured = capturing_llm_server
        proc = spawn_repl(repl_env)

        try:
            proc.sendline(self.QUESTION)
            # chain=1: the idle notice fires when the budget exhausts, i.e.
            # after the LAST LLM call of the cycle (turn, wrap-up, restart,
            # wrap-up).
            proc.expect("idle, over to you", timeout=30)

            assert len(captured) == 4, (
                "expected exactly 4 LLM calls "
                f"(turn, wrap-up, restart, wrap-up), got {len(captured)}"
            )

            restart_req = captured[2]
            last = last_user_message(restart_req)

            # The nudge carries the question VERBATIM...
            assert CARRY_NEEDLE in last
            assert self.QUESTION in last
            # ...not the bare hint.
            assert HINT_NEEDLE not in last

            # The injected intro orients on the records but no longer
            # commands the ticket queue.
            intro = all_user_messages(restart_req)[0]
            assert "Session restarted from durable records" in intro
            assert "top unblocked ticket" not in intro

            # F1: the injected flag is local bookkeeping, never on the wire.
            for payload in captured:
                assert not has_local_key(payload, "injected")
        finally:
            proc.close()
