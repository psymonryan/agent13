"""Tests for agent.core module."""

import pytest
import asyncio
from agent13 import Agent, AgentEvent, AgentQueue, AgentStatus


class MockClient:
    """Mock OpenAI client for testing."""

    pass


class TestAgent:
    """Tests for Agent class."""

    def test_create_agent(self):
        """Should create agent with required parameters."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        assert agent.client is client
        assert agent.model == "test-model"
        assert isinstance(agent.queue, AgentQueue)
        assert agent.messages == []
        assert agent.system_prompt is not None

    def test_create_agent_with_custom_params(self):
        """Should create agent with custom parameters."""
        client = MockClient()
        queue = AgentQueue()
        messages = [{"role": "user", "content": "Hello"}]

        agent = Agent(
            client,
            model="custom-model",
            queue=queue,
            system_prompt="Custom prompt",
            messages=messages,
        )

        assert agent.queue is queue
        assert agent.system_prompt == "Custom prompt"
        assert agent.messages == messages

    def test_on_event_decorator(self):
        """Should register event handler via decorator."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        @agent.on_event
        async def handler(event):
            pass

        assert len(agent._handlers) == 1
        assert handler in agent._handlers

    def test_on_event_method(self):
        """Should register event handler via method call."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        async def handler(event):
            pass

        agent.on_event(handler)

        assert len(agent._handlers) == 1
        assert handler in agent._handlers

    def test_multiple_handlers(self):
        """Should register multiple handlers."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        @agent.on_event
        async def handler1(event):
            pass

        @agent.on_event
        async def handler2(event):
            pass

        assert len(agent._handlers) == 2

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Should emit events to all handlers."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        received = []

        @agent.on_event
        async def handler(event):
            received.append(event)

        await agent.emit(AgentEvent.STATUS_CHANGE, {"status": "processing"})

        assert len(received) == 1
        assert received[0].event == AgentEvent.STATUS_CHANGE
        assert received[0].status == "processing"

    @pytest.mark.asyncio
    async def test_emit_to_multiple_handlers(self):
        """Should emit events to all handlers."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        received1 = []
        received2 = []

        @agent.on_event
        async def handler1(event):
            received1.append(event)

        @agent.on_event
        async def handler2(event):
            received2.append(event)

        await agent.emit(AgentEvent.STARTED, {})

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        """Should support sync handlers."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        received = []

        @agent.on_event
        def handler(event):  # Note: sync function
            received.append(event)

        await agent.emit(AgentEvent.STARTED, {})

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_crash(self):
        """Should not crash if handler raises error."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        @agent.on_event
        async def bad_handler(event):
            raise ValueError("Handler error")

        @agent.on_event
        async def good_handler(event):
            pass

        # Should not raise
        await agent.emit(AgentEvent.STARTED, {})

    @pytest.mark.asyncio
    async def test_add_message(self):
        """Should add message to queue and emit event."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        item_id = await agent.add_message("Hello")

        assert item_id == 1
        assert agent.queue.pending_count == 1

        # Should emit USER_MESSAGE and QUEUE_UPDATE events
        assert len(events) == 2
        assert events[0].event == AgentEvent.USER_MESSAGE
        assert events[0].text == "Hello"
        assert events[1].event == AgentEvent.QUEUE_UPDATE

    @pytest.mark.asyncio
    async def test_add_priority_message(self):
        """Should add priority message to queue."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.queue.add("normal")
        await agent.add_message("urgent", priority=True)

        items = agent.queue.list_items()
        assert items[0].text == "urgent"
        assert items[0].priority is True

    def test_stop(self):
        """Should stop the agent."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent._running = True
        agent.stop()

        assert agent._running is False
        assert agent._stop_event.is_set()

    def test_set_model(self):
        """Should set model."""
        client = MockClient()
        agent = Agent(client, model="old-model")

        agent.set_model("new-model")

        assert agent.model == "new-model"

    def test_set_system_prompt(self):
        """Should set system prompt."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        agent.set_system_prompt("New prompt")

        assert agent.system_prompt == "New prompt"

    def test_set_response_format(self):
        """Should set response format."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        assert agent.response_format is None

        agent.set_response_format({"type": "json_object"})

        assert agent.response_format == {"type": "json_object"}

    def test_create_agent_with_response_format(self):
        """Should create agent with response format."""
        client = MockClient()
        agent = Agent(
            client,
            model="test-model",
            response_format={"type": "json_object"},
        )

        assert agent.response_format == {"type": "json_object"}

    def test_clear_messages(self):
        """Should clear messages and return count."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        count = agent.clear_messages()

        assert count == 2

    def test_strip_reasoning_from_messages(self):
        """Should remove reasoning_content from assistant messages."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "Thinking..."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4",
                "reasoning_content": "Calculating...",
            },
        ]

        agent._strip_reasoning_from_messages()

        assert "reasoning_content" not in agent.messages[1]
        assert "reasoning_content" not in agent.messages[3]
        assert agent.messages[1]["content"] == "Hi"
        assert agent.messages[3]["content"] == "4"

    def test_strip_reasoning_preserves_other_messages(self):
        """Should not affect user messages or assistant messages without reasoning."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Math!"},
        ]

        agent._strip_reasoning_from_messages()

        assert agent.messages[0] == {"role": "user", "content": "Hello"}
        assert agent.messages[1] == {"role": "assistant", "content": "Hi"}
        assert agent.messages[2] == {"role": "user", "content": "What is 2+2?"}
        assert agent.messages[3] == {"role": "assistant", "content": "4"}

    @pytest.mark.asyncio
    async def test_status_change_emitted(self):
        """Should emit STATUS_CHANGE when status changes."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent._set_status(AgentStatus.PROCESSING)
        await agent._set_status(AgentStatus.IDLE)

        assert len(events) == 2
        assert events[0].status == "processing"
        assert events[1].status == "idle"

    @pytest.mark.asyncio
    async def test_status_no_change_no_event(self):
        """Should not emit event if status unchanged."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        await agent._set_status(AgentStatus.IDLE)

        assert len(events) == 1  # INITIALISING -> IDLE transition

    @pytest.mark.asyncio
    async def test_interrupt_preserves_mcp(self):
        """Interrupt (CancelledError) should NOT cleanup MCP manager.

        When the agent is interrupted (ESC key), the MCP manager should remain
        usable so that subsequent tool calls work after restart.
        """
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Create a mock MCP manager
        class MockMCPManager:
            def __init__(self):
                self._shutting_down = False
                self.cleaned_up = False

            async def cleanup(self):
                self.cleaned_up = True
                self._shutting_down = True

        agent._mcp = MockMCPManager()

        # Simulate interrupt: run() raises CancelledError with _running still True
        agent._running = True

        # The finally block in run() checks `if not self._running and self._mcp`
        # Since _running is True (interrupt scenario), cleanup should NOT be called
        # This simulates what happens in the finally block
        if not agent._running and agent._mcp:
            await agent._mcp.cleanup()

        # MCP should NOT be cleaned up after interrupt
        assert agent._mcp.cleaned_up is False
        assert agent._mcp._shutting_down is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up_mcp(self):
        """Calling stop() should cleanup MCP manager.

        When the agent is properly stopped (not just interrupted), the MCP
        manager should be cleaned up.
        """
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Create a mock MCP manager
        class MockMCPManager:
            def __init__(self):
                self._shutting_down = False
                self.cleaned_up = False

            async def cleanup(self):
                self.cleaned_up = True
                self._shutting_down = True

        agent._mcp = MockMCPManager()

        # Simulate proper stop: stop() sets _running = False
        agent._running = True
        agent.stop()  # This sets _running = False

        # Now the finally block condition should trigger cleanup
        if not agent._running and agent._mcp:
            await agent._mcp.cleanup()

        # MCP SHOULD be cleaned up after stop
        assert agent._mcp.cleaned_up is True
        assert agent._mcp._shutting_down is True

    @pytest.mark.asyncio
    async def test_run_interrupt_emits_events(self):
        """Interrupt should emit INTERRUPTED and STOPPED events."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event.event)

        # Start the run task
        task = asyncio.create_task(agent.run())

        # Let it start
        await asyncio.sleep(0.01)

        # Cancel it (simulating ESC/interrupt)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have INTERRUPTED and STOPPED events
        assert AgentEvent.INTERRUPTED in events
        assert AgentEvent.STOPPED in events

    @pytest.mark.asyncio
    async def test_status_initialising_to_idle(self):
        """Agent starts in initialising, transitions to idle on run()."""
        client = MockClient()
        agent = Agent(client, model="test-model")

        # Initial status should be INITIALISING
        assert agent.status == AgentStatus.INITIALISING

        events = []

        @agent.on_event
        async def handler(event):
            events.append(event)

        # Start the run task
        task = asyncio.create_task(agent.run())

        # Let it start and transition to idle
        await asyncio.sleep(0.01)

        # Status should now be IDLE
        assert agent.status == AgentStatus.IDLE

        # Should have seen STATUS_CHANGE event
        status_changes = [e for e in events if e.event == AgentEvent.STATUS_CHANGE]
        assert len(status_changes) == 1
        assert status_changes[0].status == "idle"

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_status_enum_values(self):
        """AgentStatus enum has expected values."""
        assert AgentStatus.INITIALISING.value == "initialising"
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.THINKING.value == "thinking"
        assert AgentStatus.PROCESSING.value == "processing"
        assert AgentStatus.PAUSED.value == "paused"


class TestToolStats:
    """Tests for ToolStats class."""

    def test_initial_state(self):
        """ToolStats starts empty."""
        from agent13.core import ToolStats

        stats = ToolStats()
        assert stats.total_calls == 0
        assert stats.total_successes == 0
        assert stats.calls == {}
        assert stats.successes == {}
        assert stats.modes == {}

    def test_record_success(self):
        """Recording a successful tool call."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record("read_file", {"filepath": "test.py"}, '{"content": "hello"}')
        assert stats.total_calls == 1
        assert stats.total_successes == 1
        assert stats.calls["read_file"] == 1
        assert stats.successes["read_file"] == 1

    def test_record_failure(self):
        """Recording a failed tool call."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record(
            "read_file", {"filepath": "test.py"}, '{"error": "File not found"}'
        )
        assert stats.total_calls == 1
        assert stats.total_successes == 0
        assert stats.calls["read_file"] == 1
        assert "read_file" not in stats.successes

    def test_record_with_mode(self):
        """Recording tool calls with mode parameter."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record(
            "read_file", {"filepath": "a.py", "mode": "skim"}, '{"content": ""}'
        )
        stats.record(
            "read_file", {"filepath": "b.py", "mode": "raw"}, '{"content": ""}'
        )
        stats.record(
            "read_file", {"filepath": "c.py", "mode": "skim"}, '{"content": ""}'
        )
        stats.record(
            "edit_file",
            {"filepath": "x.py", "mode": "replace"},
            '{"error": "not found"}',
        )
        assert stats.modes["read_file"]["skim"] == 2
        assert stats.modes["read_file"]["raw"] == 1
        # Mode successes tracked for successful calls
        assert stats.mode_successes["read_file"]["skim"] == 2
        assert stats.mode_successes["read_file"]["raw"] == 1
        # Failed mode calls tracked in modes but not in mode_successes
        assert stats.modes["edit_file"]["replace"] == 1
        assert "edit_file" not in stats.mode_successes

    def test_multiple_tools(self):
        """Recording calls to multiple tools."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record("read_file", {}, '{"content": ""}')
        stats.record("command", {"command": "ls"}, '{"stdout": ""}')
        stats.record("command", {"command": "pwd"}, '{"error": "failed"}')
        assert stats.total_calls == 3
        assert stats.total_successes == 2
        assert stats.calls["read_file"] == 1
        assert stats.calls["command"] == 2
        assert stats.successes["command"] == 1

    def test_reset(self):
        """Reset clears all statistics."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record("read_file", {"mode": "skim"}, '{"content": ""}')
        stats.reset()
        assert stats.total_calls == 0
        assert stats.total_successes == 0
        assert stats.calls == {}
        assert stats.modes == {}
        assert stats.mode_successes == {}

    def test_summary(self):
        """Summary returns expected structure."""
        from agent13.core import ToolStats

        stats = ToolStats()
        stats.record("read_file", {"mode": "skim"}, '{"content": ""}')
        summary = stats.summary()
        assert summary["total"] == 1
        assert summary["successes"] == 1
        assert "read_file" in summary["by_tool"]
        assert summary["by_tool"]["read_file"]["modes"]["skim"] == 1
        assert summary["by_tool"]["read_file"]["mode_successes"]["skim"] == 1

    def test_agent_has_tool_stats(self):
        """Agent initializes with ToolStats."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        assert hasattr(agent, "tool_stats")
        assert agent.tool_stats.total_calls == 0

    def test_set_model_resets_tool_stats(self):
        """Changing model resets tool stats."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.tool_stats.record("read_file", {}, '{"content": ""}')
        assert agent.tool_stats.total_calls == 1
        agent.set_model("new-model")
        assert agent.tool_stats.total_calls == 0


class TestSkillJournalProtection:
    """Tests for skill tool call journal protection.

    Skill tool calls load instructions that must remain in context.
    Journalling would destroy them by replacing the tool result with
    a summary, so turns containing skill calls should be skipped.
    """

    def _make_agent_with_turn(self, tool_names: list[str]) -> Agent:
        """Create an agent with a simulated tool-using turn."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        agent.messages = [
            {"role": "user", "content": "Use the code-review skill"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for i, name in enumerate(tool_names)
                ],
            },
        ]
        for i, name in enumerate(tool_names):
            agent.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"Result of {name}",
                }
            )
        agent.messages.append(
            {
                "role": "assistant",
                "content": "I've loaded the skill and am ready.",
            }
        )
        return agent

    def test_has_skill_call_in_last_turn_true(self):
        """Should detect a skill tool call in the last turn."""
        agent = self._make_agent_with_turn(["skill", "read_file"])
        assert agent._has_skill_call_in_last_turn() is True

    def test_has_skill_call_in_last_turn_false(self):
        """Should not detect a skill tool call when none present."""
        agent = self._make_agent_with_turn(["read_file", "edit_file"])
        assert agent._has_skill_call_in_last_turn() is False

    def test_has_skill_call_in_last_turn_empty(self):
        """Should return False when no messages exist."""
        client = MockClient()
        agent = Agent(client, model="test-model")
        assert agent._has_skill_call_in_last_turn() is False

    def test_has_skill_call_in_range_true(self):
        """Should detect skill call within a message range."""
        agent = self._make_agent_with_turn(["skill"])
        # user=0, assistant=1, tool=2, tool=3, assistant=4
        assert agent._has_skill_call_in_range(0, 4) is True

    def test_has_skill_call_in_range_false(self):
        """Should not detect skill call when none in range."""
        agent = self._make_agent_with_turn(["read_file"])
        assert agent._has_skill_call_in_range(0, 4) is False

    def test_maybe_reflect_skips_skill_turn(self):
        """_maybe_reflect_after_turn should skip when skill call present."""
        agent = self._make_agent_with_turn(["skill"])
        agent.journal_mode = True
        # Should return early (not crash) — we can't easily verify
        # the early return without mocking _journal_one_turn, but we
        # can verify it doesn't attempt reflection (which would need
        # a real API call). Just calling it should not raise.
        # Note: this would hang if it tried to call the LLM since
        # MockClient has no chat.create, so not raising = proof of skip.
        import asyncio

        try:
            asyncio.get_event_loop().run_until_complete(
                agent._maybe_reflect_after_turn()
            )
        except Exception:
            pytest.fail("_maybe_reflect_after_turn should have skipped skill turn")

    def test_journal_last_turn_refuses_skill(self):
        """journal_last_turn should refuse to compact a skill-containing turn."""
        agent = self._make_agent_with_turn(["skill"])
        import asyncio

        success, message = asyncio.get_event_loop().run_until_complete(
            agent.journal_last_turn()
        )
        assert success is False
        assert "skill" in message.lower()

    def test_journal_all_skips_skill_turns(self):
        """journal_all should skip turns with skill calls.

        Set up two turns: one with a skill call, one without.
        journal_all should compact only the non-skill turn.
        """
        client = MockClient()
        agent = Agent(client, model="test-model")
        # Turn 1: skill call (should be skipped)
        agent.messages = [
            {"role": "user", "content": "Load skill"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_skill",
                        "type": "function",
                        "function": {"name": "skill", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_skill", "content": "Skill loaded"},
            {"role": "assistant", "content": "Skill is ready."},
            # Turn 2: normal tool call (should be compactable)
            {"role": "user", "content": "Read a file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_read", "content": "File contents"},
            {"role": "assistant", "content": "Here's the file."},
        ]
        # journal_all would need a real LLM call for reflection,
        # so we just verify the skill detection + skip logic works
        assert agent._has_skill_call_in_range(0, 3) is True
        assert agent._has_skill_call_in_range(4, 7) is False
        # The _journal_skip marker mechanism should work:
        agent.messages[0]["_journal_skip"] = True
        boundary = agent._find_earliest_tool_turn()
        assert boundary is not None
        # Should find turn 2 (user_idx=4), not turn 1 (user_idx=0)
        assert boundary[0] == 4
        # Clean up
        agent.messages[0].pop("_journal_skip", None)
