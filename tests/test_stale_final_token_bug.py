"""Test: stale is_final token discarded by generation check, causing widget reuse.

Bug: When a queued item starts processing, on_item_started increments
_stream_generation BEFORE the previous turn's is_final TokenMessage is
processed by _process_tokens. The generation check in on_token_message
then discards the is_final sentinel, so _finalize_streaming() never runs.
The old _streaming_content_widget survives, and the next turn's content
tokens append to it instead of creating a fresh widget.

Symptom: "the new response is getting appended to the previous response"
and "the question is displayed below the response."

This test simulates the exact race using the real on_token_message discard
logic and _handle_token widget state transitions, without requiring a full
Textual App instance.
"""

import asyncio
import pytest

from ui.tui import TokenMessage


class StreamingWidgetSim:
    """Simulates StreamingMessage — tracks appends and finalization."""

    def __init__(self, title: str | None = None):
        self.title = title
        self.content = ""
        self.finalized = False

    async def append(self, text: str) -> None:
        if not text:
            return
        self.content += text

    async def finalize(self) -> None:
        self.finalized = True


class TUIStateSimulator:
    """Simulates the relevant TUI state machine for token processing.

    Mirrors the real logic from AgentTUI:
    - post_message: Textual's message queue (stage 1, async delay)
    - on_token_message: generation-based discard (the bug)
    - _token_queue + _process_tokens: sequential processing (stage 2)
    - _handle_token: widget creation, append, finalize on is_final
    - on_item_started: increments _stream_generation

    The two-stage pipeline is critical to reproducing the race:
    - post_message() enqueues into Textual's message queue (non-blocking)
    - Textual later calls on_token_message() which enqueues into _token_queue
    - Meanwhile, on_item_started() runs synchronously via emit() in the agent task

    In the real code, on_item_started can run BETWEEN post_message(is_final)
    and on_token_message(is_final), causing the generation to increment
    before the discard check sees the is_final.
    """

    def __init__(self):
        self._stream_generation = 0
        self._token_queue: asyncio.Queue = asyncio.Queue()
        self._streaming_content_widget: StreamingWidgetSim | None = None
        self._streaming = False
        self._processor_task: asyncio.Task | None = None
        self.mounted_widgets: list[StreamingWidgetSim] = []
        # Stage 1: Textual's message queue. post_message enqueues here,
        # and a background task drains it into on_token_message.
        self._textual_msg_queue: asyncio.Queue[TokenMessage] = asyncio.Queue()
        self._textual_drain_task: asyncio.Task | None = None

    async def start(self):
        self._processor_task = asyncio.create_task(self._process_tokens())
        # Stage 1 drainer: simulates Textual processing its message queue
        self._textual_drain_task = asyncio.create_task(self._drain_textual_queue())

    async def stop(self):
        await self._textual_msg_queue.put(None)  # stop drainer
        if self._textual_drain_task:
            await self._textual_drain_task
        await self._token_queue.put(None)  # stop processor
        if self._processor_task:
            await self._processor_task

    async def _drain_textual_queue(self):
        """Simulates Textual's message loop draining post_message queue.

        Each message yields control (await asyncio.sleep(0)) before being
        dispatched to on_token_message, matching how Textual's event loop
        processes messages between other async tasks.
        """
        while True:
            message = await self._textual_msg_queue.get()
            if message is None:
                break
            # Yield control — this allows the agent task (emit/on_item_started)
            # to run between post_message and on_token_message
            await asyncio.sleep(0)
            self.on_token_message(message)

    # ── post_message (stage 1: Textual's queue, non-blocking) ──

    def post_message(self, message: TokenMessage) -> None:
        """Simulates Textual's post_message — non-blocking enqueue."""
        self._textual_msg_queue.put_nowait(message)

    # ── on_token_message (the real discard logic) ──

    def on_token_message(self, message: TokenMessage) -> None:
        """Exact replica of AgentTUI.on_token_message."""
        # Discard stale tokens from previous streaming sessions,
        # but always allow is_final through so the old widget gets finalized.
        if message.generation != self._stream_generation and not message.is_final:
            return
        self._token_queue.put_nowait(message)

    # ── on_item_started (increments generation, runs via emit) ──

    def on_item_started(self):
        """Simulates on_item_started incrementing _stream_generation.

        In the real code, this runs synchronously inside emit() in the
        agent's task. It does NOT yield control — it increments the
        generation and (in the real code) awaits _write_user().
        """
        self._stream_generation += 1

    # ── _process_tokens (sequential consumer) ──

    async def _process_tokens(self):
        while True:
            message = await self._token_queue.get()
            if message is None:
                break
            await self._handle_token(message)

    # ── _handle_token (widget management) ──

    async def _handle_token(self, message: TokenMessage):
        """Simulates _handle_token — the real widget state transitions."""
        if message.is_final:
            self._streaming = False
            await self._finalize_streaming()
        else:
            if self._streaming_content_widget is None:
                await self._start_content_stream(message.text)
                self._streaming = True
            elif not self._streaming:
                # Widget exists but stream ended — start fresh
                await self._streaming_content_widget.finalize()
                self._streaming_content_widget = None
                await self._start_content_stream(message.text)
                self._streaming = True
            else:
                # Append to existing widget
                await self._streaming_content_widget.append(message.text)

    async def _start_content_stream(self, first_token: str):
        self._streaming_content_widget = StreamingWidgetSim(title="Agent")
        self.mounted_widgets.append(self._streaming_content_widget)
        await self._streaming_content_widget.append(first_token)

    async def _finalize_streaming(self):
        if self._streaming_content_widget:
            await self._streaming_content_widget.finalize()
            self._streaming_content_widget = None


@pytest.mark.asyncio
async def test_is_final_not_discarded_across_turns():
    """is_final from a previous generation must not be discarded.

    Simulates the race:
    1. Turn 1 streams content tokens (gen 0)
    2. Turn 1 emits ASSISTANT_COMPLETE → is_final posted via post_message (gen 0)
    3. Agent loop immediately picks up queued item → on_item_started (gen→1)
    4. is_final (gen 0) arrives at on_token_message → DISCARDED (bug!)
    5. Turn 2 content tokens (gen 1) → append to stale widget

    After fix: is_final is NOT discarded, old widget finalized, fresh
    widget created for Turn 2.
    """
    sim = TUIStateSimulator()
    await sim.start()

    gen0 = sim._stream_generation  # 0

    # ── Turn 1: stream content (goes through post_message → on_token_message) ──
    sim.post_message(TokenMessage("Hello ", generation=gen0))
    sim.post_message(TokenMessage("world", generation=gen0))
    # Let the processor handle these
    await asyncio.sleep(0.05)

    assert sim._streaming_content_widget is not None
    assert sim._streaming_content_widget.content == "Hello world"
    turn1_widget = sim._streaming_content_widget

    # ── Turn 1 completes: ASSISTANT_COMPLETE posts is_final via post_message ──
    # In the real code, on_assistant_complete does:
    #   self.post_message(TokenMessage("", is_final=True, generation=gen))
    # This is non-blocking — the message goes to Textual's queue, NOT _token_queue yet.
    sim.post_message(TokenMessage("", is_final=True, generation=gen0))

    # ── RACE: agent loop continues immediately (emit returned) ──
    # The agent picks up the next queued item and calls on_item_started,
    # which increments _stream_generation. This runs BEFORE Textual's
    # message loop drains the is_final from post_message.
    #
    # We do NOT yield control here — on_item_started runs synchronously
    # in the agent's task before the Textual drainer gets a chance to run.
    sim.on_item_started()  # gen is now 1
    gen1 = sim._stream_generation

    # Now yield control — Textual's drainer processes the is_final,
    # but on_token_message discards it because gen 0 != gen 1
    await asyncio.sleep(0.05)

    # ── Turn 2: content tokens arrive (gen 1) ──
    sim.post_message(TokenMessage("Second ", generation=gen1))
    sim.post_message(TokenMessage("response", generation=gen1))
    await asyncio.sleep(0.05)

    await sim.stop()

    # ── Assertions ──

    # The old widget should have been finalized (is_final processed)
    assert turn1_widget.finalized, (
        "Turn 1's widget was never finalized — is_final was discarded "
        "by the generation check"
    )

    # A NEW widget should have been created for Turn 2
    assert len(sim.mounted_widgets) == 2, (
        f"Expected 2 mounted widgets (one per turn), got {len(sim.mounted_widgets)}. "
        "If only 1 widget exists, Turn 2's content was appended to Turn 1's widget."
    )

    # Turn 2's widget should be separate and contain only Turn 2's content
    turn2_widget = sim.mounted_widgets[1]
    assert turn2_widget is not turn1_widget, (
        "Turn 2 reused Turn 1's widget — content was appended to previous response"
    )
    assert turn2_widget.content == "Second response", (
        f"Turn 2 widget content should be 'Second response', "
        f"got '{turn2_widget.content}'"
    )

    # Turn 1's widget should still have only Turn 1's content
    assert turn1_widget.content == "Hello world", (
        f"Turn 1 widget content was corrupted: '{turn1_widget.content}'"
    )


@pytest.mark.asyncio
async def test_stale_is_final_after_interrupt_is_safe():
    """A stale is_final arriving after an interrupt should be a safe no-op.

    After interrupt: widgets are finalized, generation incremented.
    A late is_final from the old generation should not cause issues.
    """
    sim = TUIStateSimulator()
    await sim.start()

    gen0 = sim._stream_generation

    # Stream some content
    sim.post_message(TokenMessage("Working ", generation=gen0))
    sim.post_message(TokenMessage("on something", generation=gen0))
    await asyncio.sleep(0.05)

    # Interrupt: finalize widgets and increment generation (like _interrupt_agent_loop)
    await sim._finalize_streaming()
    sim.on_item_started()  # gen→1

    # Late is_final from gen 0 arrives
    sim.post_message(TokenMessage("", is_final=True, generation=gen0))
    await asyncio.sleep(0.05)

    # No widget should exist — is_final should be a no-op
    assert sim._streaming_content_widget is None, (
        "Stale is_final after interrupt should not create or affect widgets"
    )

    await sim.stop()
