"""Failing test: ESC during PAUSING leaves status stuck on 'pausing'.

Bug: When /pause is issued and the TUI is in 'pausing' state, pressing ESC
should clear the pause and update the status bar. But _interrupt_agent_loop
does NOT call update_status() after clearing the pause state, so the status
bar can remain stuck showing "pausing".

Root cause:
The TUI's _interrupt_agent_loop (ui/tui.py lines 5071-5153):
  1. Cancels the agent task
  2. Calls agent.resume() if is_pausing/is_paused → clears pause_state
  3. Stops and restarts the agent loop
  4. Does NOT call self.update_status()

Compare with _handle_pause_command (line 3599) and _handle_resume_command
(line 3651) which both call self.update_status() explicitly.

Without an explicit update_status() call, the status bar refresh depends
on async events/reactive triggers that may not fire after resume() clears
is_pausing. Specifically:
- The cancelled task emits STATUS_CHANGE("idle") which sets processing=False
- The new run() task also emits STATUS_CHANGE("idle")
- If processing is already False, the reactive watch_processing doesn't
  fire (False→False is no change), so update_status() is NOT called
- The status bar keeps its last rendered text ("pausing")

Fix: Add self.update_status() after the resume() call in
_interrupt_agent_loop (around line 5143 in ui/tui.py).
"""

import inspect

from ui.tui import AgentTUI


class TestEscDuringPausingStatusStuck:
    """ESC during PAUSING should refresh the status bar, but doesn't.

    _interrupt_agent_loop clears the pause state via resume() but never
    calls update_status() to refresh the display.
    """

    def test_interrupt_loop_calls_update_status(self):
        """_interrupt_agent_loop MUST call update_status() after clearing pause state.

        This test FAILS because _interrupt_agent_loop does not call
        update_status(). After ESC during PAUSING:

        1. resume() clears is_pausing (pause_state → RUNNING)
        2. But update_status() is NOT called
        3. The status bar keeps its last rendered text
        4. If the last text was 'pausing' (set by _handle_pause_command),
           it stays stuck

        Both _handle_pause_command (line 3599) and _handle_resume_command
        (line 3651) call update_status() explicitly. _interrupt_agent_loop
        is the odd one out.

        Fix: add self.update_status() after the resume() call in
        _interrupt_agent_loop (ui/tui.py ~line 5143):

            if self.agent.is_paused or self.agent.is_pausing:
                self.agent.resume()
            self.update_status()  # Refresh status bar after clearing pause
        """
        source = inspect.getsource(AgentTUI._interrupt_agent_loop)

        assert "self.update_status()" in source, (
            "_interrupt_agent_loop does not call update_status().\n"
            "After ESC during PAUSING, resume() clears the pause state, "
            "but the status bar is never refreshed. The display stays "
            "stuck showing 'pausing'.\n\n"
            "Fix: add self.update_status() after the resume() call "
            "(around line 5143 in ui/tui.py):\n\n"
            "    if self.agent.is_paused or self.agent.is_pausing:\n"
            "        self.agent.resume()\n"
            "    self.update_status()  # <-- ADD THIS\n"
        )

    def test_pause_command_calls_update_status(self):
        """Sanity check: _handle_pause_command DOES call update_status().

        This confirms the pattern: commands that change pause state
        should call update_status() to refresh the display.
        """
        source = inspect.getsource(AgentTUI._handle_pause_command)
        assert "self.update_status()" in source

    def test_resume_command_calls_update_status(self):
        """Sanity check: _handle_resume_command DOES call update_status().

        This confirms the pattern: commands that change pause state
        should call update_status() to refresh the display.
        """
        source = inspect.getsource(AgentTUI._handle_resume_command)
        assert "self.update_status()" in source
