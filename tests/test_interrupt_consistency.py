"""Tests for interrupt consistency — Escape should leave agent in a clean state.

Bug #1: After Escape (task cancellation) during streaming, the conversation
history can be left in an inconsistent state. The next user message may break
role alternation (user after user, or assistant after assistant with no close).

Bug #3: After Escape, the agent can end up in a zombie "paused" state where
the status bar says "paused" but /resume says "Not paused".

These tests verify that interrupt (task cancellation) produces a consistent
state where the user can immediately type new instructions.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent13.core import Agent, AgentEvent, AgentStatus


class MockClient:
    """Minimal mock AsyncOpenAI client."""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()


def _make_tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Create a tool_call dict matching the OpenAI format."""
    return {
        "id": call_id,
        "type": "function",
        "name": name,
        "arguments": json.dumps(args),
    }


class TestInterruptMessageConsistency:
    """Bug #1: After interrupt, message history must be in a consistent state.

    When Escape cancels the agent task mid-stream, the partial assistant
    response must be properly closed so that the next user message doesn't
    break role alternation. The !! interrupt path already does this correctly
    (appending [Interrupted] + user message pairs at line 1414-1423), but
    the Escape/interrupt path (task cancellation) bypasses that mechanism.

    Expected behavior: after interrupt, the last message in self.messages
    should be a complete assistant turn (either with [Interrupted] appended,
    or the partial message should be removed entirely).
    """

    @pytest.mark.asyncio
    async def test_messages_role_alternation_after_interrupt_during_streaming(self):
        """After interrupt during streaming, messages must maintain role alternation.

        The API requires user/assistant/user/assistant alternation. If the
        agent was streaming a response when interrupted, the partial assistant
        content is lost (it's in a local variable, not yet appended to
        self.messages). The last message is the user message. When the user
        sends a new message after the interrupt, it would be user after user.

        Expected: after interrupt, the agent should append a synthetic
        [Interrupted] assistant message to close the turn, so the next
        user message correctly alternates.
        """
        client = MockClient()

        # Create a streaming response that yields content tokens slowly
        # so we can interrupt mid-stream
        async def slow_stream(*args, **kwargs):
            """Yield content tokens with delays so interrupt can fire mid-stream."""
            yield "content", "I will "
            yield "content", "delete all "
            await asyncio.sleep(10)  # Give us time to interrupt
            yield "content", "the tables"  # Won't reach this

        with patch("agent13.core.stream_response_with_tools", slow_stream):
            agent = Agent(client=client, model="test-model")

            events = []

            @agent.on_event
            async def handler(event):
                events.append(event)

            # Add a message and start processing
            await agent.add_message("Refactor the database")
            task = asyncio.create_task(agent.run())

            # Wait for streaming to start
            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ASSISTANT_TOKEN for e in events):
                    break

            # Now cancel the task (simulating Escape/interrupt)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Check message history consistency
            # The user message "Refactor the database" was appended (line 1213).
            # The partial streaming content was in a local variable and is LOST.
            # So the last message is role="user" with no assistant response.
            #
            # BUG: When the user sends a new message, it will be:
            #   user: "Refactor the database"
            #   user: "No wait, don't delete anything"
            # This breaks role alternation and causes API 500 errors.
            #
            # EXPECTED: The agent should have appended a synthetic
            # [Interrupted] assistant message to close the turn:
            #   user: "Refactor the database"
            #   assistant: "[Interrupted]"
            # Then the next user message correctly alternates.

            # Verify the bug exists: last message should be assistant, not user
            last_role = agent.messages[-1]["role"] if agent.messages else None
            assert last_role == "assistant", (
                f"After interrupt during streaming, last message role should be "
                f"'assistant' (with [Interrupted] marker), got '{last_role}'. "
                f"Messages: {[m['role'] for m in agent.messages]}"
            )

            # Verify the [Interrupted] marker is present
            last_content = agent.messages[-1].get("content", "")
            assert "[Interrupted]" in last_content, (
                f"After interrupt, last assistant message should contain "
                f"'[Interrupted]', got: '{last_content[:100]}'"
            )

    @pytest.mark.asyncio
    async def test_messages_role_alternation_after_interrupt_during_tooling(self):
        """After interrupt during tool execution, messages must maintain consistency.

        When the agent has made tool calls and is executing them, the assistant
        message with tool_calls is already in self.messages. If interrupted
        mid-tool-batch, some tool results may be missing. The API requires
        every tool_call to have a matching tool result.

        Expected: after interrupt, any incomplete tool calls should have
        synthetic error results appended, and the turn should be closed
        with [Interrupted] so the next user message alternates correctly.
        """
        client = MockClient()

        # Create a streaming response that yields tool calls, then a slow tool
        async def stream_with_tool_calls(*args, **kwargs):
            yield "content", "Let me check the files"
            yield (
                "tool_calls_complete",
                {
                    "tool_calls": [
                        _make_tool_call("command", {"command": "sleep 60"}, "call_1"),
                        _make_tool_call("command", {"command": "ls"}, "call_2"),
                    ]
                },
            )

        # Mock tool execution that takes a long time
        async def slow_tool_exec(*args, **kwargs):
            await asyncio.sleep(10)
            return "done"

        with patch("agent13.core.stream_response_with_tools", stream_with_tool_calls):
            with patch.object(Agent, "_execute_tool_async", slow_tool_exec):
                agent = Agent(client=client, model="test-model")

                events = []

                @agent.on_event
                async def handler(event):
                    events.append(event)

                await agent.add_message("Check the files")
                task = asyncio.create_task(agent.run())

                # Wait for tooling to start
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if any(e.event == AgentEvent.TOOL_CALL for e in events):
                        break

                # Interrupt during tool execution
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Check: the assistant message with tool_calls is in messages
                # but we may be missing tool results for some/all calls.
                # Every tool_call_id must have a matching tool result.
                tool_call_ids = set()
                tool_result_ids = set()
                for msg in agent.messages:
                    if msg["role"] == "assistant" and "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            tool_call_ids.add(tc["id"])
                    if msg["role"] == "tool":
                        tool_result_ids.add(msg.get("tool_call_id"))

                missing = tool_call_ids - tool_result_ids
                assert len(missing) == 0, (
                    f"After interrupt during tooling, all tool_calls should have "
                    f"matching tool results. Missing results for: {missing}. "
                    f"Messages: {[m['role'] for m in agent.messages]}"
                )

    @pytest.mark.asyncio
    async def test_next_user_message_alternates_after_interrupt(self):
        """After interrupt, adding a new user message should not break alternation.

        This is the end-to-end test: interrupt the agent, then simulate the
        user sending a new message. The resulting message history should have
        valid role alternation (no two consecutive user messages).
        """
        client = MockClient()

        async def slow_stream(*args, **kwargs):
            yield "content", "I will "
            await asyncio.sleep(10)

        with patch("agent13.core.stream_response_with_tools", slow_stream):
            agent = Agent(client=client, model="test-model")

            events = []

            @agent.on_event
            async def handler(event):
                events.append(event)

            await agent.add_message("Hello")
            task = asyncio.create_task(agent.run())

            # Wait for streaming to start
            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ASSISTANT_TOKEN for e in events):
                    break

            # Interrupt
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Simulate user sending a new message after interrupt
            # (this is what happens when the user types a new prompt)
            agent.messages.append({"role": "user", "content": "Wait, stop"})

            # Check role alternation
            roles = [m["role"] for m in agent.messages]
            for i in range(1, len(roles)):
                if roles[i] == "user" and roles[i - 1] == "user":
                    pytest.fail(
                        f"Consecutive user messages at positions {i - 1} and {i}. "
                        f"Role sequence: {roles}. "
                        f"After interrupt, the agent should close the partial "
                        f"turn with [Interrupted] so the next user message "
                        f"alternates correctly."
                    )


class TestInterruptStateConsistency:
    """Bug #3: After interrupt, agent should be in 'ready' state, not 'paused'.

    Escape is a hard stop — the user wants to redirect, not resume. After
    interrupt, the agent should be in IDLE/ready state, never PAUSED.
    The zombie state happens when the error path sets _paused=True, then
    the interrupt clears the TUI's _paused flag but not the agent's.
    """

    @pytest.mark.asyncio
    async def test_agent_not_paused_after_interrupt(self):
        """After interrupt, agent.is_paused should be False.

        The user hit Escape to stop and redirect — they don't intend to
        resume. The agent should be in a clean, ready state.
        """
        client = MockClient()

        async def slow_stream(*args, **kwargs):
            yield "content", "Working"
            await asyncio.sleep(10)

        with patch("agent13.core.stream_response_with_tools", slow_stream):
            agent = Agent(client=client, model="test-model")

            events = []

            @agent.on_event
            async def handler(event):
                events.append(event)

            await agent.add_message("Hello")
            task = asyncio.create_task(agent.run())

            # Wait for streaming to start
            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ASSISTANT_TOKEN for e in events):
                    break

            # Interrupt
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # After interrupt, agent should NOT be paused
            assert not agent.is_paused, (
                f"After interrupt, agent should not be paused. "
                f"Got is_paused={agent.is_paused}. "
                f"Escape means 'stop and redirect', not 'pause to resume later'."
            )

    @pytest.mark.asyncio
    async def test_agent_status_not_paused_after_interrupt(self):
        """After interrupt, agent status should not be PAUSED."""
        client = MockClient()

        async def slow_stream(*args, **kwargs):
            yield "content", "Working"
            await asyncio.sleep(10)

        with patch("agent13.core.stream_response_with_tools", slow_stream):
            agent = Agent(client=client, model="test-model")

            events = []

            @agent.on_event
            async def handler(event):
                events.append(event)

            await agent.add_message("Hello")
            task = asyncio.create_task(agent.run())

            for _ in range(50):
                await asyncio.sleep(0.02)
                if any(e.event == AgentEvent.ASSISTANT_TOKEN for e in events):
                    break

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert agent.status != AgentStatus.PAUSED, (
                f"After interrupt, status should not be PAUSED. "
                f"Got status={agent.status}."
            )
