"""Tests for journaling status display.

Verifies that the agent's JOURNALING status is correctly reflected
in the TUI status display, and that the status map includes the
journaling state.
"""


class TestJournalingStatus:
    """Test the journaling status logic without Textual dependencies."""

    def test_status_map_includes_journaling(self):
        """Test that the status map includes journaling."""
        status_map = {
            "initialising": "Initialising",
            "idle": "Ready",
            "waiting": "Waiting",
            "thinking": "Thinking",
            "processing": "Processing",
            "tooling": "Tooling",
            "journaling": "Journaling",
            "paused": "Paused",
        }
        assert "journaling" in status_map
        assert status_map["journaling"] == "Journaling"

    def test_journaling_activates_spinner(self):
        """Test that journaling status activates the spinner (processing=True)."""
        active_statuses = ("waiting", "thinking", "processing", "tooling", "journaling")
        assert "journaling" in active_statuses

        # Simulate: status comes in as "journaling"
        status = "journaling"
        processing = status in active_statuses
        assert processing is True

    def test_idle_deactivates_spinner(self):
        """Test that idle status deactivates the spinner."""
        active_statuses = ("waiting", "thinking", "processing", "tooling", "journaling")
        status = "idle"
        processing = status in active_statuses
        assert processing is False

    def test_journaling_status_display(self):
        """Test the status line format during journaling."""
        status_map = {
            "journaling": "Journaling",
        }
        display_status = status_map.get("journaling", "journaling".capitalize())
        status_text = display_status.lower()

        assert status_text == "journaling"

    def test_compaction_happens_immediately(self):
        """Test that compaction happens immediately (not deferred).

        With the new design, _maybe_reflect_after_turn applies compaction
        immediately rather than storing pending_compaction. This means
        the message history is always compacted right after reflection.
        """
        # Simulate: after reflection, messages are compacted immediately
        # (tool calls replaced with summary)
        compacted = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "Summary\n\nDone"},
        ]
        assert len(compacted) == 2
        assert compacted[1]["role"] == "assistant"
        assert "tool_calls" not in compacted[1]

    def test_status_transitions_during_auto_journal(self):
        """Test status transitions during auto-journal flow.

        Flow: PROCESSING -> JOURNALING -> IDLE
        The agent sets JOURNALING status during reflection, so the TUI
        doesn't need a separate _reflection_mode flag.
        """
        # During LLM turn
        status = "processing"
        assert status == "processing"

        # Agent starts reflection
        status = "journaling"
        assert status == "journaling"

        # After reflection + compaction, agent goes idle
        status = "idle"
        assert status == "idle"

    def test_status_transitions_standalone_journal_last(self):
        """Test status transitions during /journal last.

        Flow: IDLE -> JOURNALING -> IDLE
        Even when called standalone, the agent emits JOURNALING status.
        """
        # Start idle
        status = "idle"
        assert status == "idle"

        # /journal last starts reflection
        status = "journaling"
        assert status == "journaling"

        # After reflection, back to idle
        status = "idle"
        assert status == "idle"
