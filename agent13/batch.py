"""Pure batch orchestration - suitable for library use.

This module provides a display-agnostic way to run the agent with a single prompt.
All output is handled via callbacks, making it suitable for:
- Headless operation
- Library use
- Custom display implementations
"""

import asyncio
from typing import Callable, Awaitable, Optional
from agent13 import Agent, AgentEvent, AgentEventData


async def run_batch(
    agent: Agent,
    prompt: str,
    *,
    on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    on_reasoning: Optional[Callable[[str], Awaitable[None]]] = None,
    on_tool_call: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    on_tool_result: Optional[Callable[[str], Awaitable[None]]] = None,
    on_error: Optional[Callable[[str], Awaitable[None]]] = None,
    on_complete: Optional[Callable[[], Awaitable[None]]] = None,
    on_status_change: Optional[Callable[[str], Awaitable[None]]] = None,
) -> None:
    """Run agent with a single prompt.

    No display dependency - all output via callbacks.
    Suitable for library use.

    Args:
        agent: Configured Agent instance
        prompt: The prompt to send
        on_token: Called for each response token
        on_reasoning: Called for reasoning tokens
        on_tool_call: Called when a tool is invoked (name, arguments)
        on_tool_result: Called when a tool returns (result)
        on_error: Called on errors (message)
        on_complete: Called when response is complete
        on_status_change: Called when agent status changes (status string)
    """
    # Track state
    processing_done = asyncio.Event()
    work_started = False

    @agent.on_event
    async def on_item_started(event: AgentEventData):
        nonlocal work_started  # noqa: F824
        if event.event == AgentEvent.ITEM_STARTED:
            work_started = True

    @agent.on_event
    async def on_status(event: AgentEventData):
        nonlocal work_started  # noqa: F824
        if event.event != AgentEvent.STATUS_CHANGE:
            return
        status = event.data.get("status")
        if on_status_change:
            await on_status_change(status)
        if status == "idle" and work_started:
            processing_done.set()

    @agent.on_event
    async def on_token_event(event: AgentEventData):
        if event.event != AgentEvent.ASSISTANT_TOKEN:
            return
        text = event.text or ""
        if on_token:
            await on_token(text)

    @agent.on_event
    async def on_reasoning_event(event: AgentEventData):
        if event.event != AgentEvent.ASSISTANT_REASONING:
            return
        text = event.text or ""
        if on_reasoning:
            await on_reasoning(text)

    @agent.on_event
    async def on_complete_event(event: AgentEventData):
        if event.event != AgentEvent.ASSISTANT_COMPLETE:
            return
        if on_complete:
            await on_complete()

    @agent.on_event
    async def on_tool_call_event(event: AgentEventData):
        if event.event != AgentEvent.TOOL_CALL:
            return
        name = event.data.get("name", "")
        arguments = event.data.get("arguments", {})
        if on_tool_call:
            await on_tool_call(name, arguments)

    @agent.on_event
    async def on_tool_result_event(event: AgentEventData):
        if event.event != AgentEvent.TOOL_RESULT:
            return
        result = event.data.get("result", "")
        if on_tool_result:
            await on_tool_result(result)

    @agent.on_event
    async def on_error_event(event: AgentEventData):
        if event.event != AgentEvent.ERROR:
            return
        message = event.message or "Unknown error"
        if on_error:
            await on_error(message)

    # Start agent in background
    agent_task = asyncio.create_task(agent.run())

    try:
        # Wait for agent to start
        await asyncio.sleep(0.1)

        # Add the prompt
        await agent.add_message(prompt)

        # Wait for processing to complete
        await processing_done.wait()

    finally:
        # Stop agent and clean up
        agent.stop()
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
