"""Unit tests for BellManager (shared bell logic between TUI and REPL)."""

import asyncio
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from agent13.bell import BellManager


class TestBellManagerValidation:
    """Tests for validate_command static method."""

    def test_valid_command(self):
        """A known executable validates successfully."""
        exe = os.path.basename(sys.executable)
        assert BellManager.validate_command(f"{exe} hello") is True

    def test_invalid_command(self):
        """A nonexistent executable fails validation."""
        assert BellManager.validate_command("nonexistent_xyz_123") is False

    def test_empty_command(self):
        assert BellManager.validate_command("") is False

    def test_whitespace_only(self):
        assert BellManager.validate_command("   ") is False

    def test_command_with_args(self):
        exe = os.path.basename(sys.executable)
        assert BellManager.validate_command(f"{exe} --flag value") is True


class TestBellManagerInit:
    """Tests for constructor threshold/command parsing."""

    def test_defaults(self):
        bm = BellManager()
        assert bm.enabled is True
        assert bm.threshold == 0.0
        assert bm.command == ""

    def test_threshold_off_string(self):
        bm = BellManager(threshold="off")
        assert bm.enabled is False

    def test_threshold_off_string_case_insensitive(self):
        bm = BellManager(threshold="OFF")
        assert bm.enabled is False

    def test_threshold_numeric(self):
        bm = BellManager(threshold=30.0)
        assert bm.enabled is True
        assert bm.threshold == 30.0

    def test_threshold_numeric_string(self):
        bm = BellManager(threshold="15")
        assert bm.enabled is True
        assert bm.threshold == 15.0

    def test_threshold_invalid_string_falls_back(self):
        bm = BellManager(threshold="abc")
        assert bm.enabled is True
        assert bm.threshold == 0.0

    def test_threshold_none_uses_defaults(self):
        bm = BellManager(threshold=None)
        assert bm.enabled is True
        assert bm.threshold == 0.0

    def test_invalid_command_warns_and_clears(self, capsys):
        bm = BellManager(command="nonexistent_xyz_123")
        captured = capsys.readouterr()
        assert "not executable" in captured.err
        assert bm.command == ""

    def test_valid_command_kept(self):
        exe = os.path.basename(sys.executable)
        bm = BellManager(command=f"{exe} --flag")
        assert bm.command == f"{exe} --flag"


class TestBellManagerStatus:
    """Tests for status_text()."""

    def test_status_off(self):
        bm = BellManager(enabled=False)
        assert bm.status_text() == "off"

    def test_status_always(self):
        bm = BellManager(threshold=0)
        assert bm.status_text() == "always"

    def test_status_threshold(self):
        bm = BellManager(threshold=30)
        assert bm.status_text() == "30s"


class TestBellManagerRingFallback:
    """Tests that ring() calls fallback_ring when no command set."""

    def test_ring_calls_fallback(self):
        called = []

        def fake_bell():
            called.append(True)

        bm = BellManager(fallback_ring=fake_bell)
        bm.ring()
        assert called == [True]

    def test_ring_no_fallback_no_error(self):
        bm = BellManager(fallback_ring=None)
        bm.ring()  # Should not raise

    def test_ring_with_command_uses_subprocess(self):
        exe = os.path.basename(sys.executable)
        bm = BellManager(command=f"{exe} -c pass")
        with patch("agent13.bell.subprocess.Popen") as mock_popen:
            bm.ring()
            mock_popen.assert_called_once()


class TestBellManagerTurnLifecycle:
    """Tests for on_turn_start / on_turn_end / on_pause."""

    def test_threshold_zero_rings_on_turn_end(self):
        """When threshold=0, bell rings immediately on turn end."""
        called = []
        bm = BellManager(threshold=0, fallback_ring=lambda: called.append(True))
        bm.on_turn_start()
        assert called == []  # No ring on start
        bm.on_turn_end()
        assert called == [True]  # Rings on end

    def test_disabled_does_not_ring(self):
        called = []
        bm = BellManager(enabled=False, fallback_ring=lambda: called.append(True))
        bm.on_turn_start()
        bm.on_turn_end()
        assert called == []

    def test_pause_cancels_and_disarms(self):
        """After pause, on_turn_end should not ring (not armed)."""
        called = []
        bm = BellManager(threshold=0, fallback_ring=lambda: called.append(True))
        bm.on_turn_start()
        bm.on_pause()
        bm.on_turn_end()
        assert called == []  # Disarmed by pause

    def test_startup_idle_no_ring(self):
        """on_turn_end without prior on_turn_start should not ring."""
        called = []
        bm = BellManager(threshold=0, fallback_ring=lambda: called.append(True))
        bm.on_turn_end()
        assert called == []

    @pytest.mark.asyncio
    async def test_cancel_clears_timer(self):
        """cancel() should cancel any pending timer."""
        bm = BellManager(threshold=100)
        bm.on_turn_start()
        assert bm._task is not None
        bm.cancel()
        # Task should be cancelled (cancelled tasks set to None by _cancel_timer)
        assert bm._task is None


class TestBellManagerThresholdTimer:
    """Tests for threshold timer (async)."""

    @pytest.mark.asyncio
    async def test_threshold_timer_rings_after_delay(self):
        """Bell rings after threshold seconds if turn is still active."""
        called = []
        bm = BellManager(threshold=0.05, fallback_ring=lambda: called.append(True))
        bm.on_turn_start()
        assert bm._task is not None
        # Wait long enough for timer to fire
        await asyncio.sleep(0.1)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_threshold_timer_cancelled_on_turn_end(self):
        """Timer is cancelled when turn ends before threshold."""
        called = []
        bm = BellManager(threshold=1.0, fallback_ring=lambda: called.append(True))
        bm.on_turn_start()
        bm.on_turn_end()  # Should cancel timer, NOT ring (threshold != 0)
        await asyncio.sleep(0.1)
        assert called == []  # No ring because threshold != 0

    @pytest.mark.asyncio
    async def test_only_one_timer_per_turn(self):
        """Starting twice doesn't create two timers."""
        bm = BellManager(threshold=1.0)
        bm.on_turn_start()
        task1 = bm._task
        bm.on_turn_start()  # Second call
        assert bm._task is task1  # Same task, not replaced


class TestBellManagerSetters:
    """Tests for runtime setters used by /bell and /bell-command."""

    def test_set_threshold_enables(self):
        bm = BellManager(enabled=False)
        bm.set_threshold(10)
        assert bm.enabled is True
        assert bm.threshold == 10.0

    @pytest.mark.asyncio
    async def test_set_threshold_zero_cancels_timer(self):
        bm = BellManager(threshold=100)
        bm.on_turn_start()
        assert bm._task is not None
        bm.set_threshold(0)
        assert bm._task is None

    @pytest.mark.asyncio
    async def test_disable_cancels_timer(self):
        bm = BellManager(threshold=100)
        bm.on_turn_start()
        assert bm._task is not None
        bm.disable()
        assert bm.enabled is False
        assert bm._task is None

    def test_set_command_valid(self):
        exe = os.path.basename(sys.executable)
        bm = BellManager()
        assert bm.set_command(f"{exe} --beep") is True
        assert bm.command == f"{exe} --beep"

    def test_set_command_invalid(self):
        bm = BellManager()
        assert bm.set_command("nonexistent_xyz") is False
        assert bm.command == ""

    def test_clear_command(self):
        exe = os.path.basename(sys.executable)
        bm = BellManager(command=f"{exe} --beep")
        bm.clear_command()
        assert bm.command == ""