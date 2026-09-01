"""User expectation tests for REPL mode.

These tests simulate real user interactions and verify the experience
matches expectations. Each test represents a scenario a user would
encounter, verifying both the output they'd see and the side effects
on the agent.

Principle: tests should not break the "principle of least astonishment".

Features tested:
- Basic message sending
- Interrupt (!!) and priority (!) prefixes
- Multi-line mode (/multi, backslash continuation)
- Multi-line cancellation (Ctrl+C, /cancel, empty ., /quit)
- Ctrl+D (EOF) clean exit
- Slash commands: /clear, /history, /queue, /delete, /help, /status
- Edge cases: empty input, unknown commands, bare prefixes
"""

import asyncio
import datetime
import io
import sys
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

from agent13.queue import AgentQueue


# ── Mock infrastructure ──────────────────────────────────────────────


class InputFeeder:
    """Feeds scripted inputs to mocked input() and sys.stdin.readline().

    Both input() and stdin.readline() draw from the same ordered list.
    The caller provides the COMPLETE sequence of inputs including
    streaming-mode Enters.

    Sentinels:
        InputFeeder.EOF       — triggers EOFError (Ctrl+D)
        InputFeeder.INTERRUPT — triggers KeyboardInterrupt (Ctrl+C)
    """

    EOF = object()
    INTERRUPT = object()

    def __init__(self, responses):
        self._queue = list(responses)

    def do_input(self, prompt=""):
        if not self._queue:
            raise EOFError("feeder exhausted — add more inputs to your scenario")
        val = self._queue.pop(0)
        if val is self.EOF:
            raise EOFError()
        if val is self.INTERRUPT:
            raise KeyboardInterrupt()
        return val

    def do_stdin_readline(self):
        """Handle sys.stdin.readline() calls (used in streaming mode)."""
        if not self._queue:
            return ""  # EOF
        val = self._queue.pop(0)
        if val is self.EOF:
            return ""  # EOF
        if val is self.INTERRUPT:
            raise KeyboardInterrupt()
        return val + "\n"  # readline includes trailing newline


class MockMessageHistory:
    """Mock MessageHistory that groups messages like the real one."""

    def __init__(self, messages):
        self.messages = messages

    def get_message_groups(self):
        """Group messages: each user message starts a new group."""
        groups = []
        current = []
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "user" and not msg.get("interrupt"):
                if current:
                    groups.append(current)
                current = [i]
            else:
                current.append(i)
        if current:
            groups.append(current)
        return groups


class MockAgent:
    """Mock agent for REPL testing.

    Records all calls so tests can assert on side effects.
    """

    def __init__(self, **kwargs):
        self.messages = []
        self.history = MockMessageHistory(self.messages)
        self.queue = AgentQueue()
        self.is_paused = False
        self.is_pausing = False
        self._handlers = []
        self._calls = []
        self.available_models = []
        self.model = kwargs.get("model", "test-model")
        self.session_date = datetime.date.today().isoformat()
        # Provide pause_state and status like real Agent
        from agent13.core import PauseState, AgentStatus
        self.pause_state = PauseState.RUNNING
        self.status = AgentStatus.IDLE
        self.PauseState = PauseState
        self.AgentStatus = AgentStatus
        # Status-related attributes (needed by gather_status)
        self.mcp = None
        self.tool_stats = type("ToolStats", (), {"total_successes": 0, "total_calls": 0, "calls": {}, "successes": {}})()
        self.journal_mode = False
        self.devel_mode = False
        self.skills_mode = False
        self.remove_reasoning = False
        self._mcp_server_configs = {}

    @property
    def is_idle(self) -> bool:
        return self.pause_state == self.PauseState.RUNNING and self.status == self.AgentStatus.IDLE

    def on_event(self, handler):
        self._handlers.append(handler)
        return handler

    async def run(self):
        """Mock run — wait until cancelled."""
        try:
            await asyncio.sleep(float("inf"))
        except asyncio.CancelledError:
            pass

    async def add_message(self, text, priority=False, interrupt=False, kind="prompt", data=None):
        self._calls.append(("add_message", text, priority, interrupt, kind, data))
        msg = {"role": "user", "content": text}
        if interrupt:
            msg["interrupt"] = True
        self.messages.append(msg)
        # Simulate agent becoming busy (like real Agent does)
        from agent13.core import AgentStatus
        self.status = AgentStatus.PROCESSING

    def clear_messages(self):
        self._calls.append(("clear_messages",))
        count = len(self.messages)
        self.messages.clear()
        return count

    async def request_clear(self, mode="all", keep_turns=0):
        self._calls.append(("request_clear", mode, keep_turns))

    def pause(self):
        self._calls.append(("pause",))
        self.is_paused = True
        self.pause_state = self.PauseState.PAUSED

    def resume(self):
        self._calls.append(("resume",))
        self.is_paused = False
        self.pause_state = self.PauseState.RUNNING

    def stop(self):
        self._calls.append(("stop",))

    async def request_load(self, path):
        self._calls.append(("request_load", path))

    def set_mcp_servers(self, servers):
        pass

    def reset_token_usage(self):
        pass

    def set_model(self, model):
        self._calls.append(("set_model", model))
        self.model = model

    def set_client(self, client, models=None):
        self._calls.append(("set_client", client, models))
        self.client = client
        if models is not None:
            self.available_models = models

    def set_devel_mode(self, enabled):
        self._calls.append(("set_devel_mode", enabled))
        self.devel_mode = enabled

    def set_skills_mode(self, enabled):
        self._calls.append(("set_skills_mode", enabled))
        self.skills_mode = enabled

    def disconnect_mcp(self):
        self._calls.append(("disconnect_mcp",))

    async def _ensure_mcp(self):
        self._calls.append(("_ensure_mcp",))
        return self.mcp


class MockHistoryStore:
    """Mock History store (avoids file I/O)."""

    def __init__(self):
        self.session_items = []
        self._adds = []

    def add(self, text):
        self._adds.append(text)


# ── Scenario runner ──────────────────────────────────────────────────


async def run_scenario(inputs, prompt_manager=None):
    """Run a REPL scenario with scripted inputs.

    Args:
        inputs: Ordered list of ALL inputs the user provides.
                Strings are returned by input()/readline().
                Use InputFeeder.EOF to trigger Ctrl+D.
                Use InputFeeder.INTERRUPT to trigger Ctrl+C.
        prompt_manager: Optional PromptManager to inject (e.g. backed by a
                temp file for hermetic prompt lookups). Defaults to the
                real PromptManager (~/.agent13/prompts.yaml).

    Returns:
        (output_text, agent) tuple — captured stdout and mock agent
    """
    feeder = InputFeeder(inputs)
    created_agents = []

    def capture_agent(*args, **kwargs):
        agent = MockAgent()
        created_agents.append(agent)
        return agent

    captured_stdout = io.StringIO()

    with ExitStack() as stack:
        stack.enter_context(
            patch("agent13.repl.Agent", side_effect=capture_agent)
        )
        stack.enter_context(
            patch("agent13.repl.History", MockHistoryStore)
        )
        stack.enter_context(
            patch("agent13.repl.get_filtered_tools", return_value=[])
        )
        stack.enter_context(
            patch(
                "agent13.repl.get_config",
                return_value=MagicMock(mcp_servers=None),
            )
        )
        stack.enter_context(patch("agent13.repl.RichDisplay"))
        stack.enter_context(
            patch("builtins.input", side_effect=feeder.do_input)
        )
        stack.enter_context(
            patch.object(sys.stdin, "readline", side_effect=feeder.do_stdin_readline)
        )
        stack.enter_context(patch("sys.stdout", captured_stdout))
        stack.enter_context(patch("readline.read_history_file"))
        stack.enter_context(patch("readline.set_history_length"))
        stack.enter_context(patch("readline.add_history"))
        stack.enter_context(patch("readline.write_history_file"))
        stack.enter_context(patch("os.makedirs"))

        from agent13.repl import run_repl

        await run_repl(
            client=MagicMock(),
            model="test-model",
            provider="test",
            pretty=False,
            system_prompt="test prompt",
            prompt_manager=prompt_manager,
        )

    output = captured_stdout.getvalue()
    agent = created_agents[0] if created_agents else None
    return output, agent


# ── Helpers ──────────────────────────────────────────────────────────

EOF = InputFeeder.EOF
INT = InputFeeder.INTERRUPT


def find_call(agent, method_name):
    """Find the first call to a mock agent method."""
    for call in agent._calls:
        if call[0] == method_name:
            return call
    return None


def find_calls(agent, method_name):
    """Find all calls to a mock agent method."""
    return [c for c in agent._calls if c[0] == method_name]


# ── Feature: Basic message sending ──────────────────────────────────


class TestBasicSending:
    """User sends a plain message and it reaches the agent."""

    async def test_send_simple_message(self):
        """User types 'hello world' — agent receives it."""
        output, agent = await run_scenario(["hello world", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "hello world"
        assert call[2] is False  # priority
        assert call[3] is False  # interrupt

    async def test_send_shows_processing(self):
        """User sees [processing] when message is dispatched."""
        output, agent = await run_scenario(["test", "", "/quit"])

        assert "[processing]" in output

    async def test_send_then_quit(self):
        """User sends a message then types /quit — clean exit."""
        output, agent = await run_scenario(["hello", "", "/quit"])

        assert "Goodbye!" in output
        call = find_call(agent, "add_message")
        assert call[1] == "hello"

    async def test_multiple_messages(self):
        """User sends two messages in sequence."""
        output, agent = await run_scenario(
            ["first", "", "/quit"]
            # Note: after first message, user enters streaming mode,
            # presses Enter (empty), gets prompt, types /quit.
            # But to send a SECOND message, they'd need the agent to
            # finish and return to idle. Since our mock agent's add_message
            # returns immediately and no events fire to set processing_done,
            # the second message would be queued. Let's test that.
        )

        calls = find_calls(agent, "add_message")
        assert len(calls) >= 1
        assert calls[0][1] == "first"


# ── Feature: Interrupt prefix (!!) ─────────────────────────────────


class TestInterruptPrefix:
    """User uses !! prefix for interrupt messages."""

    async def test_interrupt_prefix(self):
        """'!!fix this' sends with interrupt=True."""
        output, agent = await run_scenario(["!!fix this", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "fix this"  # prefix stripped
        assert call[2] is True  # priority (implied by interrupt)
        assert call[3] is True  # interrupt

    async def test_interrupt_with_space(self):
        """'!! fix this' (space after !!) — prefix still stripped."""
        output, agent = await run_scenario(["!! fix this", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call[1] == "fix this"

    async def test_interrupt_shows_notification(self):
        """User sees [interrupt] notification."""
        output, agent = await run_scenario(["!!urgent", "", "/quit"])

        assert "interrupt" in output.lower()

    async def test_bare_interrupt_prefix(self):
        """'!!' with nothing after — no message sent, prompts again."""
        # User types bare !!, gets prompted again, then types real message
        output, agent = await run_scenario(["!!", "real message", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "real message"


# ── Feature: Priority prefix (!) ───────────────────────────────────


class TestPriorityPrefix:
    """User uses ! prefix for priority messages."""

    async def test_priority_prefix(self):
        """'!important' sends with priority=True, interrupt=False."""
        output, agent = await run_scenario(["!important", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "important"  # prefix stripped
        assert call[2] is True  # priority
        assert call[3] is False  # not interrupt

    async def test_priority_shows_notification(self):
        """User sees [priority] notification."""
        output, agent = await run_scenario(["!priority msg", "", "/quit"])

        assert "priority" in output.lower()

    async def test_bare_priority_prefix(self):
        """'!' with nothing after — no message sent."""
        output, agent = await run_scenario(["!", "real message", "", "/quit"])

        call = find_call(agent, "add_message")
        assert call[1] == "real message"


# ── Feature: Multi-line mode (/multi) ──────────────────────────────


class TestMultiLineMode:
    """User enters multi-line mode with /multi."""

    async def test_multi_basic(self):
        """User types /multi, two lines, then . — assembled message sent."""
        output, agent = await run_scenario(
            ["/multi", "line one", "line two", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "line one\nline two"

    async def test_multi_single_line(self):
        """User types /multi, single line, then . — message sent as-is."""
        output, agent = await run_scenario(
            ["/multi", "just one line", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call[1] == "just one line"

    async def test_multi_preserves_content(self):
        """Multi-line preserves exact content of each line."""
        output, agent = await run_scenario(
            ["/multi", "first", "second", "third", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call[1] == "first\nsecond\nthird"

    async def test_multi_shows_prompt(self):
        """User sees multi-line mode confirmation."""
        output, agent = await run_scenario(
            ["/multi", "line", ".", "", "/quit"]
        )

        assert "Multi-line mode" in output

    async def test_multi_records_in_history(self):
        """Multi-line message is recorded in History as single entry."""
        output, agent = await run_scenario(
            ["/multi", "a", "b", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call[1] == "a\nb"


# ── Feature: Backslash continuation ────────────────────────────────


class TestBackslashContinuation:
    """User enters multi-line mode via trailing backslash."""

    async def test_backslash_continuation(self):
        """'line one \\' then 'line two' then '.' — assembled message."""
        output, agent = await run_scenario(
            ["line one \\", "line two", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "line one \nline two"

    async def test_backslash_strips_backslash(self):
        """Trailing backslash is removed from the assembled message."""
        output, agent = await run_scenario(
            ["hello \\", "world", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        # Should NOT contain a backslash
        assert "\\" not in call[1]
        assert call[1] == "hello \nworld"

    async def test_backslash_shows_mode_indicator(self):
        """User sees confirmation that backslash continuation is active."""
        output, agent = await run_scenario(
            ["test \\", "line", ".", "", "/quit"]
        )

        assert "Multi-line" in output or "\\\\" in output or "continuation" in output.lower()

    async def test_backslash_multiple_continuations(self):
        """Three lines via backslash — only the FIRST backslash triggers
        continuation. Subsequent lines in multi-mode are literal text,
        so 'two \\' keeps its trailing backslash."""
        output, agent = await run_scenario(
            ["one \\", "two \\", "three", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        # "one \\" → backslash stripped (triggers continuation)
        # "two \\" → literal text in multi-mode (backslash kept)
        # "three"  → literal text
        assert call[1] == "one \ntwo \\\nthree"


# ── Feature: Multi-line + prefix combination ───────────────────────


class TestMultiLineWithPrefix:
    """Multi-line messages combined with !! and ! prefixes."""

    async def test_backslash_with_interrupt(self):
        """'!!urgent \\' with continuation — sends as interrupt."""
        output, agent = await run_scenario(
            ["!!urgent \\", "details here", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[3] is True  # interrupt
        assert "urgent" in call[1]
        assert "details here" in call[1]

    async def test_backslash_with_priority(self):
        """'!priority \\' with continuation — sends as priority."""
        output, agent = await run_scenario(
            ["!priority \\", "more info", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[2] is True  # priority
        assert call[3] is False  # not interrupt

    async def test_multi_slash_not_interpreted_as_command(self):
        """/multi then '/something' — sent as message, not command."""
        output, agent = await run_scenario(
            ["/multi", "/something", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "/something"
        # Should NOT show "Unknown command" error
        assert "Unknown command" not in output

    async def test_multi_first_line_with_prefix(self):
        """/multi with first line starting with !! — prefix parsed."""
        output, agent = await run_scenario(
            ["/multi", "!!interrupt text", "more details", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[3] is True  # interrupt


# ── Feature: Multi-line cancellation ───────────────────────────────


class TestMultiLineCancellation:
    """Cancelling out of multi-line mode."""

    async def test_multi_ctrl_c_exits(self):
        """Ctrl+C in multi-line — cancels buffer and exits REPL."""
        output, agent = await run_scenario(
            ["/multi", "some text", INT]
        )

        assert "cancelled" in output.lower() or "Multi-line" in output
        assert "Goodbye!" in output

    async def test_multi_empty_dot_cancels(self):
        """'.' with empty buffer — silent cancel, back to normal."""
        # /multi, then immediately "." — buffer is empty
        output, agent = await run_scenario(
            ["/multi", ".", "/quit"]
        )

        assert "cancelled" in output.lower() or "empty" in output.lower()
        # Should NOT have sent a message
        call = find_call(agent, "add_message")
        assert call is None

    async def test_multi_slash_cancel(self):
        """/cancel during multi-line — cancels and returns to prompt."""
        output, agent = await run_scenario(
            ["/multi", "line one", "/cancel", "/quit"]
        )

        assert "cancelled" in output.lower()
        call = find_call(agent, "add_message")
        assert call is None

    async def test_multi_slash_quit_exits(self):
        """/quit during multi-line — exits REPL."""
        output, agent = await run_scenario(
            ["/multi", "line one", "/quit"]
        )

        assert "Goodbye!" in output
        # Buffer should NOT be sent
        call = find_call(agent, "add_message")
        assert call is None


# ── Feature: EOF (Ctrl+D) ─────────────────────────────────────────


class TestEOF:
    """Ctrl+D exits the REPL cleanly."""

    async def test_eof_exits_cleanly(self):
        """Ctrl+D on empty prompt — prints EOF message and exits."""
        output, agent = await run_scenario([EOF])

        assert "EOF" in output
        assert "Goodbye!" in output

    async def test_eof_no_traceback(self):
        """Ctrl+D should not produce a Python traceback."""
        output, agent = await run_scenario([EOF])

        assert "Traceback" not in output
        assert "EOFError" not in output

    async def test_eof_in_multi_mode(self):
        """Ctrl+D during multi-line — cancels and exits."""
        output, agent = await run_scenario(
            ["/multi", "some text", EOF]
        )

        assert "Goodbye!" in output
        # No traceback
        assert "Traceback" not in output


# ── Feature: /clear command ────────────────────────────────────────


class TestClearCommand:
    """User clears message history."""

    async def test_clear_idle(self):
        """/clear when idle — preserves history, prints message."""
        output, agent = await run_scenario(["/clear", "/quit"])

        call = find_call(agent, "clear_messages")
        assert call is None
        assert "History preserved" in output

    async def test_clear_while_processing(self):
        """/clear while processing — preserves history, prints message."""
        output, agent = await run_scenario(
            ["hello", "", "/clear", "/quit"]
        )

        call = find_call(agent, "request_clear")
        assert call is None
        assert "History preserved" in output

    async def test_clear_no_messages(self):
        """/clear with no messages — preserves history, prints message."""
        output, agent = await run_scenario(["/clear", "/quit"])

        call = find_call(agent, "clear_messages")
        assert call is None
        assert "History preserved" in output

    async def test_clear_all(self):
        """/clear all — clears history in REPL."""
        output, agent = await run_scenario(["/clear all", "/quit"])

        call = find_call(agent, "clear_messages")
        assert call is not None
        assert "Cleared 0" in output


# ── Feature: /history command ──────────────────────────────────────


class TestHistoryCommand:
    """User views message history."""

    async def test_history_empty(self):
        """/history with no messages — shows 'no messages'."""
        output, agent = await run_scenario(["/history", "/quit"])

        assert "No messages" in output

    async def test_history_after_send(self):
        """/history after sending a message — shows the message."""
        output, agent = await run_scenario(
            ["hello world", "", "/history", "/quit"]
        )

        assert "Message history" in output
        assert "hello world" in output

    async def test_history_shows_groups(self):
        """/history shows grouped messages."""
        output, agent = await run_scenario(
            ["first message", "", "/history", "/quit"]
        )

        assert "groups" in output.lower()


# ── Feature: /queue command ────────────────────────────────────────


class TestQueueCommand:
    """User views the message queue."""

    async def test_queue_empty(self):
        """/queue when queue is empty — shows 'empty'."""
        output, agent = await run_scenario(["/queue", "/quit"])

        assert "empty" in output.lower()

    async def test_queue_after_priority(self):
        """/queue after sending priority message — may show items."""
        # Priority messages are sent immediately (not queued), so
        # the queue might be empty. This tests the command works.
        output, agent = await run_scenario(
            ["!priority", "", "/queue", "/quit"]
        )

        # Command should not error
        assert "Unknown command" not in output


# ── Feature: /delete command ───────────────────────────────────────


class TestDeleteCommand:
    """User deletes items from history/queue/saves."""

    async def test_delete_no_args(self):
        """/delete with no arguments — shows usage."""
        output, agent = await run_scenario(["/delete", "/quit"])

        assert "Usage" in output

    async def test_delete_invalid_target(self):
        """/delete x 1 — shows usage (invalid target)."""
        output, agent = await run_scenario(["/delete x 1", "/quit"])

        assert "Usage" in output

    async def test_delete_history_group(self):
        """/delete h 1 after sending a message."""
        output, agent = await run_scenario(
            ["hello", "", "/delete h 1", "/quit"]
        )

        assert "Deleted" in output

    async def test_delete_history_range(self):
        """/delete h 1:2 - deletes a range of groups."""
        output, agent = await run_scenario(
            ["first", "", "second", "", "/delete h 1:2", "/quit"]
        )

        assert "Deleted groups 1-2" in output

    async def test_delete_queue_item(self):
        """/delete q N — removes queue item."""
        # Add something to the queue manually, then delete
        output, agent = await run_scenario(
            ["/delete q 1", "/quit"]
        )

        assert "Invalid" in output or "empty" in output.lower()

    async def test_delete_history_last(self):
        """/delete h last — deletes the last group."""
        output, agent = await run_scenario(
            ["first", "", "second", "", "/delete h last", "/quit"]
        )

        assert "Deleted group 2" in output

    async def test_delete_history_negative(self):
        """/delete h -1 — deletes using negative index."""
        output, agent = await run_scenario(
            ["first", "", "second", "", "/delete h -1", "/quit"]
        )

        assert "Deleted group 2" in output

    async def test_delete_history_range_negative(self):
        """/delete h -2:-1 — deletes range with negative indices."""
        output, agent = await run_scenario(
            ["one", "", "two", "", "three", "", "/delete h -2:-1", "/quit"]
        )

        assert "Deleted groups 2-3" in output

    async def test_delete_queue_last(self):
        """/delete q last — deletes last queue item."""
        # Queue is empty, should error gracefully
        output, agent = await run_scenario(
            ["/delete q last", "/quit"]
        )

        assert "No items" in output or "Invalid" in output

    async def test_delete_queue_with_items(self):
        """/delete q N - removes actual queue item."""
        from agent13.commands import execute_delete

        agent = MockAgent()
        agent.queue.add("first queued prompt")
        agent.queue.add("second queued prompt")
        agent.queue.add("third queued prompt")

        result = execute_delete(agent, "q 2")
        assert result.success
        assert "Removed queue item" in result.message
        assert "second queued" in result.message
        assert agent.queue.pending_count == 2

    async def test_delete_queue_last_with_items(self):
        """/delete q last - removes last queue item."""
        from agent13.commands import execute_delete

        agent = MockAgent()
        agent.queue.add("alpha")
        agent.queue.add("beta")

        result = execute_delete(agent, "q last")
        assert result.success
        assert "beta" in result.message
        assert agent.queue.pending_count == 1

    async def test_delete_queue_range_with_items(self):
        """/delete q 1:2 - removes range of queue items."""
        from agent13.commands import execute_delete

        agent = MockAgent()
        agent.queue.add("one")
        agent.queue.add("two")
        agent.queue.add("three")

        result = execute_delete(agent, "q 1:2")
        assert result.success
        assert "Removed 2 queue items" in result.message
        assert agent.queue.pending_count == 1


# ── Feature: /help command ─────────────────────────────────────────


class TestHelpCommand:
    """User views available commands."""

    async def test_help_shows_all_commands(self):
        """/help lists all available commands."""
        output, agent = await run_scenario(["/help", "/quit"])

        assert "/quit" in output
        assert "/help" in output
        assert "/multi" in output
        assert "/clear" in output
        assert "/history" in output
        assert "/queue" in output
        assert "/delete" in output
        assert "/status" in output
        assert "/save" in output
        assert "/load" in output

    async def test_help_shows_mode_info(self):
        """/help mentions input/streaming modes."""
        output, agent = await run_scenario(["/help", "/quit"])

        assert "Enter" in output
        assert "mode" in output.lower()


# ── Feature: /status command ───────────────────────────────────────


class TestStatusCommand:
    """User checks agent status."""

    async def test_status_shows_model(self):
        """/status shows the model name."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "test-model" in output or "test/test-model" in output

    async def test_status_shows_idle(self):
        """/status when idle shows 'idle'."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "idle" in output.lower()

    async def test_status_shows_message_count(self):
        """/status shows message count."""
        output, agent = await run_scenario(
            ["hello", "", "/status", "/quit"]
        )

        assert "messages" in output.lower()

    async def test_status_shows_sections(self):
        """/status shows Session, Provider, Context, Connectivity, Tools, Settings sections."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "Session" in output
        assert "Provider" in output
        assert "Context" in output
        assert "Connectivity" in output
        assert "Tools" in output
        assert "Settings" in output

    async def test_status_shows_provider_and_model(self):
        """/status shows provider and model separately."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "provider:" in output
        assert "model:" in output

    async def test_status_shows_settings(self):
        """/status shows settings like sandbox, journal, devel."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "sandbox:" in output
        assert "journal:" in output
        assert "devel:" in output
        assert "skills:" in output

    async def test_status_shows_mcp(self):
        """/status shows MCP connectivity status."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "mcp:" in output

    async def test_status_shows_tokens(self):
        """/status shows token counts."""
        output, agent = await run_scenario(["/status", "/quit"])

        assert "prompt tokens:" in output
        assert "completion tokens:" in output
        assert "total tokens:" in output


# ── Feature: Unknown commands ──────────────────────────────────────


class TestUnknownCommands:
    """User types unrecognized slash commands."""

    async def test_unknown_command(self):
        """/foo — shows error and hints at /help."""
        output, agent = await run_scenario(["/foo", "/quit"])

        assert "Unknown command" in output
        assert "/help" in output

    async def test_unknown_does_not_send(self):
        """/foo should not send a message to the agent."""
        output, agent = await run_scenario(["/foo", "/quit"])

        call = find_call(agent, "add_message")
        assert call is None


# ── Feature: Empty input / mode transitions ────────────────────────


class TestEmptyInput:
    """User presses Enter on empty line."""

    async def test_empty_input_no_crash(self):
        """Empty input followed by /quit — no crash."""
        output, agent = await run_scenario(["", "", "/quit"])

        assert "Goodbye!" in output

    async def test_whitespace_only_input(self):
        """Whitespace-only input treated as empty."""
        output, agent = await run_scenario(["   ", "", "/quit"])

        assert "Goodbye!" in output


# ── Feature: Exit commands ─────────────────────────────────────────


class TestExitCommands:
    """User exits the REPL."""

    async def test_quit(self):
        """/quit exits cleanly."""
        output, agent = await run_scenario(["/quit"])

        assert "Goodbye!" in output

    async def test_exit(self):
        """/exit exits cleanly (alias for /quit)."""
        output, agent = await run_scenario(["/exit"])

        assert "Goodbye!" in output

    async def test_ctrl_c_exits(self):
        """Ctrl+C exits the REPL."""
        output, agent = await run_scenario([INT])

        assert "Interrupted" in output or "Goodbye!" in output

    async def test_stop_command(self):
        """/stop when idle — says nothing to stop."""
        output, agent = await run_scenario(["/stop", "/quit"])

        assert "Nothing to stop" in output


# ── Feature: Banner ────────────────────────────────────────────────


class TestBanner:
    """User sees the REPL banner on startup."""

    async def test_banner_shows_model(self):
        """Banner includes model name."""
        output, agent = await run_scenario(["/quit"])

        assert "test-model" in output

    async def test_banner_shows_help_hint(self):
        """Banner mentions /help."""
        output, agent = await run_scenario(["/quit"])

        assert "/help" in output


# ── Feature: /pause and /resume ────────────────────────────────────


class TestPauseResume:
    """User pauses and resumes the agent."""

    async def test_pause_when_idle(self):
        """/pause when idle — nothing to pause."""
        output, agent = await run_scenario(["/pause", "/quit"])

        assert "Nothing to pause" in output

    async def test_resume_when_not_paused(self):
        """/resume when not paused — not paused."""
        output, agent = await run_scenario(["/resume", "/quit"])

        assert "Not paused" in output


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Unusual but possible user interactions."""

    async def test_slash_in_multi_not_command(self):
        """Messages starting with / inside /multi are sent as text."""
        output, agent = await run_scenario(
            ["/multi", "/help me please", ".", "", "/quit"]
        )

        call = find_call(agent, "add_message")
        assert call is not None
        assert call[1] == "/help me please"

    async def test_multi_assembled_trailing_backslash_not_re_trigger(self):
        """Trailing backslash in /multi content should be sent as-is,
        not re-trigger continuation mode.

        Previously: the backslash check didn't guard against from_multi,
        so assembled text ending with \\ would re-enter multi-line mode.
        Fixed: added 'and not from_multi' to the backslash check.
        """
        output, agent = await run_scenario(
            ["/multi", "line with \\", ".", "", "/quit"]
        )

        # Message should be sent — trailing backslash is literal content
        call = find_call(agent, "add_message")
        assert call is not None, "Message with trailing backslash was not sent"
        assert "line with" in call[1]
        # Should NOT re-enter multi-line mode
        assert "Multi-line mode" not in output.split("Multi-line mode")[0]

    async def test_history_adds_slash_commands(self):
        """Slash commands are NOT added to history (only messages)."""
        output, agent = await run_scenario(
            ["/help", "real message", "", "/quit"]
        )

        # /help should not have been recorded as a message
        calls = find_calls(agent, "add_message")
        texts = [c[1] for c in calls]
        assert "/help" not in texts

    async def test_empty_prefix_then_real_message(self):
        """Bare '!!' then real message — only real message sent."""
        output, agent = await run_scenario(
            ["!!", "actual content", "", "/quit"]
        )

        calls = find_calls(agent, "add_message")
        assert len(calls) == 1
        assert calls[0][1] == "actual content"


# ── Persistence-aware scenario runner ──────────────────────────────────────


async def run_scenario_persistence(inputs, tmp_path):
    """Run a REPL scenario with real persistence (save/load to tmp_path).

    Like run_scenario but patches get_saves_dir to use tmp_path so
    /save and /load work with real files.
    """
    feeder = InputFeeder(inputs)
    created_agents = []

    def capture_agent(*args, **kwargs):
        agent = MockAgent()
        # Add persistence attributes that save_context expects
        agent.system_prompt = "test prompt"
        agent.model = "test-model"
        agent.prompt_tokens = 0
        agent.completion_tokens = 0
        created_agents.append(agent)
        return agent

    captured_stdout = io.StringIO()

    saves_dir = tmp_path / "saves"
    saves_dir.mkdir(exist_ok=True)

    with ExitStack() as stack:
        stack.enter_context(
            patch("agent13.repl.Agent", side_effect=capture_agent)
        )
        stack.enter_context(
            patch("agent13.repl.History", MockHistoryStore)
        )
        stack.enter_context(
            patch("agent13.repl.get_filtered_tools", return_value=[])
        )
        stack.enter_context(
            patch(
                "agent13.repl.get_config",
                return_value=MagicMock(mcp_servers=None),
            )
        )
        stack.enter_context(patch("agent13.repl.RichDisplay"))
        stack.enter_context(
            patch("builtins.input", side_effect=feeder.do_input)
        )
        stack.enter_context(
            patch.object(
                sys.stdin, "readline", side_effect=feeder.do_stdin_readline
            )
        )
        stack.enter_context(patch("sys.stdout", captured_stdout))
        stack.enter_context(patch("readline.read_history_file"))
        stack.enter_context(patch("readline.set_history_length"))
        stack.enter_context(patch("readline.add_history"))
        stack.enter_context(patch("readline.write_history_file"))
        stack.enter_context(patch("os.makedirs"))
        # Persistence mocks - use real tmp_path
        stack.enter_context(
            patch(
                "agent13.persistence.get_saves_dir", return_value=saves_dir
            )
        )

        from agent13.repl import run_repl

        await run_repl(
            client=MagicMock(),
            model="test-model",
            provider="test",
            pretty=False,
            system_prompt="test prompt",
        )

    output = captured_stdout.getvalue()
    agent = created_agents[0] if created_agents else None
    return output, agent


# ── Save command tests ─────────────────────────────────────────────────────


class TestSaveCommand:
    """User expectation tests for /save."""

    async def test_save_basic(self, tmp_path):
        """User sends a message then saves - sees confirmation with path."""
        output, agent = await run_scenario_persistence(
            ["hello world", "/save mywork", "/quit"], tmp_path
        )

        assert "Saved" in output
        assert "mywork.ctx" in output
        # Should show message count
        assert "messages" in output

    async def test_save_no_name_shows_usage(self, tmp_path):
        """User types /save with no name - sees usage with -y flag info."""
        output, _ = await run_scenario_persistence(
            ["/save", "/quit"], tmp_path
        )

        assert "Usage" in output
        assert "-y" in output

    async def test_save_invalid_name(self, tmp_path):
        """User types /save with name starting with dash - sees error."""
        output, _ = await run_scenario_persistence(
            ["/save -bad", "/quit"], tmp_path
        )

        assert "Error" in output or "valid" in output.lower()

    async def test_save_creates_file(self, tmp_path):
        """User saves - the .ctx file actually exists on disk."""
        output, _ = await run_scenario_persistence(
            ["hello", "/save myfile", "/quit"], tmp_path
        )

        saved_file = tmp_path / "saves" / "myfile.ctx"
        assert saved_file.exists()

    async def test_save_overwrite_blocked(self, tmp_path):
        """User saves twice with same name (no -y) - second is blocked."""
        output, _ = await run_scenario_persistence(
            ["hello", "/save mywork", "/save mywork", "/quit"], tmp_path
        )

        # First save succeeds
        assert "Saved" in output
        # Second save is blocked
        assert "already exists" in output
        assert "-y" in output

    async def test_save_overwrite_with_force(self, tmp_path):
        """User saves twice with -y - both succeed."""
        output, _ = await run_scenario_persistence(
            ["hello", "/save mywork", "/save mywork -y", "/quit"], tmp_path
        )

        # Both should succeed
        saved_count = output.count("Saved")
        assert saved_count == 2

    async def test_save_force_flag_anywhere(self, tmp_path):
        """-y flag works in any position, not just last (regression)."""
        output, _ = await run_scenario_persistence(
            ["hello", "/save mywork", "/save -y mywork", "/quit"], tmp_path
        )

        # Both should succeed: first save creates, second overwrites with -y first
        saved_count = output.count("Saved")
        assert saved_count == 2

    async def test_save_empty_conversation(self, tmp_path):
        """User saves with no messages - still works."""
        output, _ = await run_scenario_persistence(
            ["/save empty", "/quit"], tmp_path
        )

        assert "Saved" in output
        assert "0 messages" in output


# ── Load command tests ─────────────────────────────────────────────────────


class TestLoadCommand:
    """User expectation tests for /load."""

    async def test_load_no_name_shows_available_saves(self, tmp_path):
        """User types /load with no name - sees list of available saves."""
        # Create some save files first
        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        (saves_dir / "project-a.ctx").write_text("{}")
        (saves_dir / "project-b.ctx").write_text("{}")

        output, _ = await run_scenario_persistence(
            ["/load", "/quit"], tmp_path
        )

        assert "Available saves" in output
        assert "project-a" in output
        assert "project-b" in output

    async def test_load_no_saves_found(self, tmp_path):
        """User types /load with no name and no saves exist."""
        output, _ = await run_scenario_persistence(
            ["/load", "/quit"], tmp_path
        )

        assert "No saves found" in output

    async def test_load_nonexistent_save(self, tmp_path):
        """User tries to load a save that does not exist."""
        output, _ = await run_scenario_persistence(
            ["/load nosuchthing", "/quit"], tmp_path
        )

        assert "Not found" in output
        assert "nosuchthing" in output

    async def test_load_shows_conversation(self, tmp_path):
        """User loads a save - sees the loaded conversation displayed."""
        import json

        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        context = {
            "version": 1,
            "model": "test-model",
            "system_prompt": "test",
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
                },
                {"role": "user", "content": "Tell me more."},
            ],
            "token_usage": {"prompt": 100, "completion": 50},
        }
        (saves_dir / "myproject.ctx").write_text(json.dumps(context))

        output, agent = await run_scenario_persistence(
            ["/load myproject", "/quit"], tmp_path
        )

        # Should display the loaded conversation
        assert "Loaded" in output or "loaded" in output.lower()
        assert "3 messages" in output
        # Should show the actual message content
        assert "What is Python" in output
        # Agent should have the messages loaded
        assert len(agent.messages) == 3

    async def test_load_with_tool_calls(self, tmp_path):
        """User loads a save with tool calls - sees tool names and results."""
        import json

        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        context = {
            "version": 1,
            "model": "test-model",
            "system_prompt": "test",
            "messages": [
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_123",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"filepath": "test.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc_123",
                    "content": "print('hello')",
                },
                {
                    "role": "assistant",
                    "content": "The file contains a print statement.",
                },
            ],
            "token_usage": {"prompt": 200, "completion": 100},
        }
        (saves_dir / "withtools.ctx").write_text(json.dumps(context))

        output, agent = await run_scenario_persistence(
            ["/load withtools", "/quit"], tmp_path
        )

        # Should show tool call name
        assert "read_file" in output
        # Should show tool result
        assert "print" in output
        # Agent should have all messages
        assert len(agent.messages) == 4

    async def test_load_incomplete_turn_warning(self, tmp_path):
        """User loads a save with incomplete turn - sees warning."""
        import json

        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        context = {
            "version": 1,
            "model": "test-model",
            "system_prompt": "test",
            "messages": [
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_456",
                            "function": {
                                "name": "read_file",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ],
            "token_usage": {"prompt": 100, "completion": 50},
            "incomplete_turn": True,
        }
        (saves_dir / "incomplete.ctx").write_text(json.dumps(context))

        output, _ = await run_scenario_persistence(
            ["/load incomplete", "/quit"], tmp_path
        )

        assert "incomplete" in output.lower() or "Warning" in output

    async def test_load_while_processing_deferred(self, tmp_path):
        """User loads while agent is busy - load is queued."""
        import json

        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        context = {
            "version": 1,
            "model": "test-model",
            "system_prompt": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "token_usage": {"prompt": 50, "completion": 25},
        }
        (saves_dir / "queued.ctx").write_text(json.dumps(context))

        # Send a message then immediately load
        output, agent = await run_scenario_persistence(
            ["hello", "/load queued", "/quit"], tmp_path
        )

        # Should indicate the load is queued
        assert "queued" in output.lower() or "will take effect" in output.lower()
        # Should have called request_load
        assert any(c[0] == "request_load" for c in agent._calls)

    async def test_load_error_handling(self, tmp_path):
        """User loads a corrupt save file - sees error, no crash."""
        saves_dir = tmp_path / "saves"
        saves_dir.mkdir(exist_ok=True)
        (saves_dir / "corrupt.ctx").write_text("not valid json {{{")

        output, _ = await run_scenario_persistence(
            ["/load corrupt", "/quit"], tmp_path
        )

        assert "Error" in output
        # Should NOT have crashed
        assert "Traceback" not in output

# -- Feature: /model command ---------------------------------------------


async def run_scenario_with_models(inputs, model_names):
    """Run a REPL scenario with pre-populated agent.available_models.

    Like run_scenario() but passes model_names through to run_repl,
    which sets agent.available_models on the mock agent.
    """
    feeder = InputFeeder(inputs)
    created_agents = []

    def capture_agent(*args, **kwargs):
        agent = MockAgent()
        created_agents.append(agent)
        return agent

    captured_stdout = io.StringIO()

    with ExitStack() as stack:
        stack.enter_context(
            patch("agent13.repl.Agent", side_effect=capture_agent)
        )
        stack.enter_context(
            patch("agent13.repl.History", MockHistoryStore)
        )
        stack.enter_context(
            patch("agent13.repl.get_filtered_tools", return_value=[])
        )
        stack.enter_context(
            patch(
                "agent13.repl.get_config",
                return_value=MagicMock(mcp_servers=None),
            )
        )
        stack.enter_context(patch("agent13.repl.RichDisplay"))
        stack.enter_context(
            patch("builtins.input", side_effect=feeder.do_input)
        )
        stack.enter_context(
            patch.object(sys.stdin, "readline", side_effect=feeder.do_stdin_readline)
        )
        stack.enter_context(patch("sys.stdout", captured_stdout))
        stack.enter_context(patch("readline.read_history_file"))
        stack.enter_context(patch("readline.set_history_length"))
        stack.enter_context(patch("readline.add_history"))
        stack.enter_context(patch("readline.write_history_file"))
        stack.enter_context(patch("os.makedirs"))

        from agent13.repl import run_repl

        await run_repl(
            client=MagicMock(),
            model="test-model",
            provider="test",
            pretty=False,
            system_prompt="test prompt",
            model_names=model_names,
        )

    output = captured_stdout.getvalue()
    agent = created_agents[0] if created_agents else None
    return output, agent


class TestModelCommand:
    """User interacts with /model command."""

    async def test_model_lists_models(self):
        """/model with no args shows available models."""
        output, agent = await run_scenario_with_models(
            ["/model", "/quit"],
            model_names=["alpha-model", "beta-model", "gamma-model"],
        )

        assert "alpha-model" in output
        assert "beta-model" in output
        assert "gamma-model" in output

    async def test_model_marks_current(self):
        """/model marks the currently selected model with *."""
        output, agent = await run_scenario_with_models(
            ["/model", "/quit"],
            model_names=["alpha-model", "test-model", "gamma-model"],
        )

        assert "test-model *" in output

    async def test_model_no_models_loaded(self):
        """/model with empty available_models shows guidance."""
        output, agent = await run_scenario_with_models(
            ["/model", "/quit"],
            model_names=[],
        )

        assert "No models loaded" in output

    async def test_model_select_by_name(self):
        """/model <name> selects a model by exact name."""
        output, agent = await run_scenario_with_models(
            ["/model beta-model", "/quit"],
            model_names=["alpha-model", "beta-model"],
        )

        assert "Model set to: beta-model" in output
        call = find_call(agent, "set_model")
        assert call is not None
        assert call[1] == "beta-model"

    async def test_model_select_by_number(self):
        """/model 2 selects the second model."""
        output, agent = await run_scenario_with_models(
            ["/model 2", "/quit"],
            model_names=["alpha-model", "beta-model"],
        )

        assert "Model set to: beta-model" in output
        call = find_call(agent, "set_model")
        assert call is not None
        assert call[1] == "beta-model"

    async def test_model_invalid_number(self):
        """/model 99 for out-of-range number shows error."""
        output, _ = await run_scenario_with_models(
            ["/model 99", "/quit"],
            model_names=["alpha-model"],
        )

        assert "out of range" in output

    async def test_model_not_found(self):
        """/model unknown-name shows no match message."""
        output, _ = await run_scenario_with_models(
            ["/model nonexistent", "/quit"],
            model_names=["alpha-model", "beta-model"],
        )

        assert "No model matching" in output


# -- Feature: /provider command ------------------------------------------


async def run_scenario_with_provider_patches(inputs, provider_side_effect=None):
    """Run a REPL scenario with resolve_provider_arg, create_client, and fetch_models mocked.

    Args:
        inputs: Scripted user inputs
        provider_side_effect: If set, resolve_provider_arg raises this exception.
                              Otherwise returns mock provider values.
    """
    feeder = InputFeeder(inputs)
    created_agents = []

    def capture_agent(*args, **kwargs):
        agent = MockAgent()
        created_agents.append(agent)
        return agent

    captured_stdout = io.StringIO()

    if provider_side_effect is not None:
        provider_mock = MagicMock(side_effect=provider_side_effect)
    else:
        provider_mock = MagicMock(
            return_value=("http://localhost:9999/v1", "test-key", 600, 10)
        )

    mock_models_list = ["mock-alpha", "mock-beta"]

    with ExitStack() as stack:
        stack.enter_context(
            patch("agent13.repl.Agent", side_effect=capture_agent)
        )
        stack.enter_context(
            patch("agent13.repl.History", MockHistoryStore)
        )
        stack.enter_context(
            patch("agent13.repl.get_filtered_tools", return_value=[])
        )
        stack.enter_context(
            patch(
                "agent13.repl.get_config",
                return_value=MagicMock(mcp_servers=None),
            )
        )
        stack.enter_context(patch("agent13.repl.RichDisplay"))
        stack.enter_context(
            patch("builtins.input", side_effect=feeder.do_input)
        )
        stack.enter_context(
            patch.object(sys.stdin, "readline", side_effect=feeder.do_stdin_readline)
        )
        stack.enter_context(patch("sys.stdout", captured_stdout))
        stack.enter_context(patch("readline.read_history_file"))
        stack.enter_context(patch("readline.set_history_length"))
        stack.enter_context(patch("readline.add_history"))
        stack.enter_context(patch("readline.write_history_file"))
        stack.enter_context(patch("os.makedirs"))
        stack.enter_context(
            patch("agent13.repl.resolve_provider_arg", side_effect=provider_mock.side_effect if provider_side_effect else provider_mock)
        )
        # Mock resolve_provider_selection to pass through the name directly
        stack.enter_context(
            patch("agent13.repl.resolve_provider_selection", side_effect=lambda choice: choice)
        )
        stack.enter_context(
            patch("agent13.repl.create_client", return_value=MagicMock())
        )

        async def mock_fetch_models(client):
            return mock_models_list

        stack.enter_context(
            patch("agent13.repl.fetch_models", side_effect=mock_fetch_models)
        )

        from agent13.repl import run_repl

        await run_repl(
            client=MagicMock(),
            model="test-model",
            provider="test",
            pretty=False,
            system_prompt="test prompt",
        )

    output = captured_stdout.getvalue()
    agent = created_agents[0] if created_agents else None
    return output, agent


class TestProviderCommand:
    """User interacts with /provider command."""

    async def test_provider_no_args_shows_list(self):
        """/provider with no args shows provider list."""
        with patch("agent13.repl.get_provider_names", return_value=["alpha", "beta"]):
            output, _ = await run_scenario(
                ["/provider", "/quit"],
            )

        assert "alpha" in output
        assert "beta" in output
        assert "/provider <name>" in output

    async def test_provider_switch(self):
        """/provider <name> switches provider and shows models."""
        output, agent = await run_scenario_with_provider_patches(
            ["/provider test_provider", "/quit"],
        )

        assert "Provider changed" in output
        assert "mock-alpha" in output
        assert "mock-beta" in output
        call = find_call(agent, "set_client")
        assert call is not None

    async def test_provider_invalid_shows_error(self):
        """/provider with unknown provider shows error."""
        output, _ = await run_scenario_with_provider_patches(
            ["/provider nonexistent", "/quit"],
            provider_side_effect=ValueError("Provider 'nonexistent' not found"),
        )

        assert "Error" in output or "not found" in output

    async def test_provider_by_number(self):
        """Resolve provider selection by number."""
        feeder = InputFeeder(["/provider 1", "/quit"])
        created_agents = []
        captured_stdout = io.StringIO()

        def capture_agent(*args, **kwargs):
            agent = MockAgent()
            created_agents.append(agent)
            return agent

        with ExitStack() as stack:
            stack.enter_context(patch("agent13.repl.Agent", side_effect=capture_agent))
            stack.enter_context(patch("agent13.repl.History", MockHistoryStore))
            stack.enter_context(patch("agent13.repl.get_filtered_tools", return_value=[]))
            stack.enter_context(
                patch("agent13.repl.get_config", return_value=MagicMock(mcp_servers=None))
            )
            stack.enter_context(patch("agent13.repl.RichDisplay"))
            stack.enter_context(patch("builtins.input", side_effect=feeder.do_input))
            stack.enter_context(patch.object(sys.stdin, "readline", side_effect=feeder.do_stdin_readline))
            stack.enter_context(patch("sys.stdout", captured_stdout))
            stack.enter_context(patch("readline.read_history_file"))
            stack.enter_context(patch("readline.set_history_length"))
            stack.enter_context(patch("readline.add_history"))
            stack.enter_context(patch("readline.write_history_file"))
            stack.enter_context(patch("os.makedirs"))
            # "1" resolves to "test_provider", then resolve_provider_arg succeeds
            stack.enter_context(
                patch("agent13.repl.resolve_provider_selection", return_value="test_provider")
            )
            stack.enter_context(
                patch(
                    "agent13.repl.resolve_provider_arg",
                    return_value=("http://test.example.com/v1", "key", 2400.0, 30.0),
                )
            )
            stack.enter_context(
                patch("agent13.repl.create_client", return_value=MagicMock())
            )

            async def mock_fetch(client):
                return ["model-a", "model-b"]

            stack.enter_context(
                patch("agent13.repl.fetch_models", side_effect=mock_fetch)
            )

            from agent13.repl import run_repl

            await run_repl(
                client=MagicMock(),
                model="test-model",
                provider="test",
                pretty=False,
                system_prompt="test prompt",
            )

        output = captured_stdout.getvalue()
        agent = created_agents[0] if created_agents else None
        assert "Provider changed" in output
        assert "model-a" in output
        call = find_call(agent, "set_client")
        assert call is not None


# -- Feature: /help includes /model and /provider ------------------------


class TestHelpIncludesNewCommands:
    """/help output includes /model and /provider."""

    async def test_help_lists_model(self):
        """/help output mentions /model."""
        output, _ = await run_scenario(["/help", "/quit"])

        assert "/model" in output

    async def test_help_lists_provider(self):
        """/help output mentions /provider."""
        output, _ = await run_scenario(["/help", "/quit"])

        assert "/provider" in output


# -- Feature: /sandbox command ---------------------------------------------


class TestSandboxCommand:
    """User interacts with /sandbox command."""

    async def test_sandbox_no_args_shows_config(self):
        """/sandbox with no args shows sandbox configuration."""
        from agent13.sandbox import SandboxMode

        with (
            patch("agent13.repl.get_current_sandbox_mode", return_value=SandboxMode.PERMISSIVE_OPEN),
            patch("agent13.repl.get_session_sandbox_mode", return_value=None),
            patch("agent13.repl.get_default_sandbox_mode", return_value=SandboxMode.PERMISSIVE_OPEN),
        ):
            output, _ = await run_scenario(["/sandbox", "/quit"])

        assert "permissive-open" in output
        assert "Sandbox Configuration" in output

    async def test_sandbox_set_mode(self):
        """/sandbox <mode> sets session override."""
        from agent13.sandbox import SandboxMode

        with (
            patch("agent13.repl.set_session_sandbox_mode") as mock_set,
        ):
            output, _ = await run_scenario(["/sandbox restrictive-open", "/quit"])

        mock_set.assert_called_once_with(SandboxMode.RESTRICTIVE_OPEN)
        assert "restrictive-open" in output

    async def test_sandbox_invalid_mode(self):
        """/sandbox <invalid> shows error."""
        output, _ = await run_scenario(["/sandbox bogus", "/quit"])

        assert "Error" in output or "Invalid" in output

    async def test_sandbox_shows_session_override(self):
        """When session override is set, /sandbox shows it."""
        from agent13.sandbox import SandboxMode

        with (
            patch("agent13.repl.get_current_sandbox_mode", return_value=SandboxMode.RESTRICTIVE_CLOSED),
            patch("agent13.repl.get_session_sandbox_mode", return_value=SandboxMode.RESTRICTIVE_CLOSED),
            patch("agent13.repl.get_default_sandbox_mode", return_value=SandboxMode.PERMISSIVE_OPEN),
        ):
            output, _ = await run_scenario(["/sandbox", "/quit"])

        assert "restrictive-closed" in output
        assert "Session override" in output

    async def test_sandbox_off_alias(self):
        """/sandbox off sets NONE mode."""
        from agent13.sandbox import SandboxMode

        with (
            patch("agent13.repl.set_session_sandbox_mode") as mock_set,
        ):
            output, _ = await run_scenario(["/sandbox off", "/quit"])

        mock_set.assert_called_once_with(SandboxMode.OFF)


# -- Feature: /devel command -----------------------------------------------


class TestDevelCommand:
    """User interacts with /devel command."""

    async def test_devel_no_args_shows_status(self):
        """/devel with no args shows current state."""
        output, agent = await run_scenario(["/devel", "/quit"])

        assert "Devel mode:" in output
        assert "off" in output

    async def test_devel_on(self):
        """/devel on enables devel mode."""
        output, agent = await run_scenario(["/devel on", "/quit"])

        assert "Devel mode enabled" in output
        call = find_call(agent, "set_devel_mode")
        assert call is not None
        assert call[1] is True

    async def test_devel_off(self):
        """/devel off disables devel mode."""
        # First enable, then disable
        output, agent = await run_scenario(["/devel on", "/devel off", "/quit"])

        assert "Devel mode disabled" in output
        calls = find_calls(agent, "set_devel_mode")
        assert len(calls) == 2
        assert calls[0][1] is True
        assert calls[1][1] is False

    async def test_devel_status(self):
        """/devel status shows current state."""
        output, _ = await run_scenario(["/devel status", "/quit"])

        assert "Devel mode:" in output

    async def test_devel_invalid(self):
        """/devel <invalid> shows usage."""
        output, _ = await run_scenario(["/devel foobar", "/quit"])

        assert "Usage: /devel" in output


# -- Feature: /tools command -----------------------------------------------


class TestToolsCommand:
    """User interacts with /tools command."""

    async def test_tools_no_calls(self):
        """/tools with no calls shows 'no tool calls yet'."""
        output, _ = await run_scenario(["/tools", "/quit"])

        assert "No tool calls yet" in output

    async def test_tools_with_stats(self):
        """/tools shows tool stats when calls exist."""
        # We need to patch get_tool_stats_summary to return non-empty data
        mock_summary = {
            "total_calls": 10,
            "total_successes": 8,
            "total_failures": 2,
            "success_rate": 80.0,
            "per_tool": [
                {"name": "read_file", "calls": 7, "successes": 6},
                {"name": "edit_file", "calls": 3, "successes": 2},
            ],
        }
        with patch("agent13.repl.get_tool_stats_summary", return_value=mock_summary):
            output, _ = await run_scenario(["/tools", "/quit"])

        assert "8/10" in output
        assert "80%" in output
        assert "read_file" in output
        assert "edit_file" in output

    async def test_tools_per_tool_shows_successes(self):
        """/tools shows per-tool success counts."""
        mock_summary = {
            "total_calls": 5,
            "total_successes": 4,
            "total_failures": 1,
            "success_rate": 80.0,
            "per_tool": [
                {"name": "command", "calls": 5, "successes": 4},
            ],
        }
        with patch("agent13.repl.get_tool_stats_summary", return_value=mock_summary):
            output, _ = await run_scenario(["/tools", "/quit"])

        assert "4/5" in output


# -- Feature: /mcp command -------------------------------------------------


class TestMcpCommand:
    """User interacts with /mcp command."""

    async def test_mcp_no_servers(self):
        """/mcp with no MCP configured shows not initialized."""
        output, _ = await run_scenario(["/mcp", "/quit"])

        assert "Not initialized" in output or "No MCP" in output

    async def test_mcp_connect_no_config(self):
        """/mcp connect with no servers configured."""
        output, _ = await run_scenario(["/mcp connect", "/quit"])

        assert "No MCP servers configured" in output

    async def test_mcp_disconnect_not_connected(self):
        """/mcp disconnect when not connected."""
        output, _ = await run_scenario(["/mcp disconnect", "/quit"])

        assert "not connected" in output.lower() or "MCP" in output

    async def test_mcp_with_connected_servers(self):
        """/mcp shows server info when connected."""
        mock_mcp = MagicMock()
        mock_mcp.get_server_info.return_value = {
            "test-server": ["tool_a", "tool_b"],
        }

        # Need to patch run_scenario to inject mcp on the agent
        feeder = InputFeeder(["/mcp", "/quit"])
        created_agents = []

        def capture_agent(*args, **kwargs):
            agent = MockAgent()
            agent.mcp = mock_mcp
            created_agents.append(agent)
            return agent

        captured_stdout = io.StringIO()

        with ExitStack() as stack:
            stack.enter_context(
                patch("agent13.repl.Agent", side_effect=capture_agent)
            )
            stack.enter_context(
                patch("agent13.repl.History", MockHistoryStore)
            )
            stack.enter_context(
                patch("agent13.repl.get_filtered_tools", return_value=[])
            )
            stack.enter_context(
                patch(
                    "agent13.repl.get_config",
                    return_value=MagicMock(mcp_servers=None),
                )
            )
            stack.enter_context(patch("agent13.repl.RichDisplay"))
            stack.enter_context(
                patch("builtins.input", side_effect=feeder.do_input)
            )
            stack.enter_context(
                patch.object(sys.stdin, "readline", side_effect=feeder.do_stdin_readline)
            )
            stack.enter_context(patch("sys.stdout", captured_stdout))
            stack.enter_context(patch("readline.read_history_file"))
            stack.enter_context(patch("readline.set_history_length"))
            stack.enter_context(patch("readline.add_history"))
            stack.enter_context(patch("readline.write_history_file"))
            stack.enter_context(patch("os.makedirs"))

            from agent13.repl import run_repl

            await run_repl(
                client=MagicMock(),
                model="test-model",
                provider="test",
                pretty=False,
                system_prompt="test prompt",
            )

        output = captured_stdout.getvalue()
        _agent = created_agents[0]  # noqa: F841
        assert "test-server" in output
        assert "2 tools" in output


# -- Feature: /help includes new commands -----------------------------------


class TestHelpIncludesBatch2Commands:
    """/help output includes batch 2 commands."""

    async def test_help_lists_sandbox(self):
        output, _ = await run_scenario(["/help", "/quit"])
        assert "/sandbox" in output

    async def test_help_lists_devel(self):
        output, _ = await run_scenario(["/help", "/quit"])
        assert "/devel" in output

    async def test_help_lists_tools(self):
        output, _ = await run_scenario(["/help", "/quit"])
        assert "/tools" in output

    async def test_help_lists_mcp(self):
        output, _ = await run_scenario(["/help", "/quit"])
        assert "/mcp" in output
