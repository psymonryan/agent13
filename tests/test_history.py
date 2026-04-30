"""Tests for history management."""

import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from agent13.history import History, get_history_path, get_default_history_path

TODAY = datetime.now().strftime("%Y-%m-%d")
NEXT_DAY = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.fixture
def temp_file():
    """Create a temporary file for history testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".history") as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for history testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestGetHistoryPath:
    """Tests for get_history_path function."""

    def test_returns_path_with_project_and_date(self):
        """Test that path includes project name and date.

        When running under pytest, path includes _test suffix.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        path = get_history_path("myproject")
        # Under pytest, path has _test suffix
        assert f"history-myproject_test-{today}" in path

    def test_returns_path_in_agent13_directory(self):
        """Test that path is in ~/.agent13 directory."""
        path = get_history_path("test")
        assert ".agent13" in path

    def test_path_is_expanded(self):
        """Test that ~ is expanded to home directory."""
        path = get_history_path("test")
        assert "~" not in path
        assert os.path.isabs(path)

    def test_default_uses_cwd_basename(self):
        """Test that default path uses basename of cwd.

        When running under pytest, path includes _test suffix.
        """
        path = get_default_history_path()
        cwd_basename = os.path.basename(os.getcwd())
        # Under pytest, path has _test suffix
        assert f"history-{cwd_basename}_test-" in path


class TestHistoryWithProject:
    """Tests for History class with project-based storage."""

    def test_create_history_with_project(self, temp_dir):
        """Test creating history with explicit project name."""
        project = "test-project"
        with patch.object(
            History,
            "_get_path",
            return_value=os.path.join(temp_dir, f"history-{project}-{TODAY}"),
        ):
            h = History(project)
            assert h.project_name == project

    def test_add_command(self, temp_dir):
        """Test adding a command to history."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first command")
            assert len(h.session_items) == 1
            assert h.session_items[0][1] == "first command"

    def test_add_multiple_commands(self, temp_dir):
        """Test adding multiple commands."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.add("third")
            # Session items are in order added (oldest first)
            assert len(h.session_items) == 3
            assert [cmd for _, cmd in h.session_items] == ["first", "second", "third"]

    def test_add_empty_command(self, temp_dir):
        """Test that empty commands are ignored."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("")
            h.add("   ")
            assert len(h.session_items) == 0

    def test_add_duplicate_consecutive(self, temp_dir):
        """Test that consecutive duplicate commands are ignored."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command")
            h.add("command")
            assert len(h.session_items) == 1

    def test_navigate_up(self, temp_dir):
        """Test navigating up (towards older)."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.add("third")
            # Start at end (after third)
            # up() should go to third (most recent in session)
            assert h.up() == "third"
            assert h.up() == "second"
            assert h.up() == "first"

    def test_navigate_up_at_start(self, temp_dir):
        """Test navigating up when already at oldest."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("only")
            h.up()  # Now at "only"
            # Should stay at "only"
            assert h.up() == "only"

    def test_navigate_down(self, temp_dir):
        """Test navigating down (towards newer)."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.add("third")
            # Navigate up to oldest
            h.up()  # third
            h.up()  # second
            h.up()  # first
            # Navigate down
            assert h.down() == "second"
            assert h.down() == "third"
            # At newest, down returns None
            assert h.down() is None

    def test_navigate_down_at_end(self, temp_dir):
        """Test navigating down when at newest."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("only")
            # At end, down returns None
            assert h.down() is None

    def test_reset(self, temp_dir):
        """Test resetting navigation position."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.up()  # Move to second
            h.reset()
            # After reset, up should go to second (most recent)
            assert h.up() == "second"

    def test_get_all_returns_file_items(self, temp_dir):
        """Test that get_all returns file items (newest first)."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            # get_all returns file_items (newest first)
            all_cmds = h.get_all()
            assert all_cmds == ["second", "first"]

    def test_get_with_timestamps(self, temp_dir):
        """Test getting commands with timestamps."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command")
            items = h.get_with_timestamps()
            assert len(items) == 1
            assert items[0][1] == "command"
            assert isinstance(items[0][0], datetime)

    def test_clear_session(self, temp_dir):
        """Test clearing session history only."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command")
            h.clear_session()
            assert len(h.session_items) == 0
            # File items should still exist
            assert len(h.file_items) == 1

    def test_clear_all(self, temp_dir):
        """Test clearing all history."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command")
            h.clear()
            assert len(h.session_items) == 0
            assert len(h.file_items) == 0

    def test_persistence(self, temp_dir):
        """Test that history persists across instances."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h1 = History("test")
            h1.add("first")
            h1.add("second")

            # Create new instance - should load from file
            h2 = History("test")
            # file_items should have the commands (newest first)
            assert len(h2.file_items) == 2
            assert h2.get_all() == ["second", "first"]
            # session_items is now seeded from file history for immediate navigation
            assert len(h2.session_items) == 2
            # up() returns newest first when navigating back
            assert h2.up() == "second"  # newest (first up)
            assert h2.up() == "first"  # oldest (second up)

    def test_len(self, temp_dir):
        """Test __len__ returns session count."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            assert len(h) == 2

    def test_repr(self, temp_dir):
        """Test string representation."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command")
            repr_str = repr(h)
            assert "test" in repr_str
            assert "session=1" in repr_str
            assert "file=1" in repr_str

    def test_empty_history_file(self, temp_dir):
        """Test loading from non-existent file."""
        path = os.path.join(temp_dir, f"history-nonexistent-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            assert len(h.session_items) == 0
            assert len(h.file_items) == 0


class TestConcurrentAccess:
    """Tests for concurrent access to history files."""

    def test_concurrent_writes(self, temp_dir):
        """Test that concurrent writes don't lose data."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")

        def writer(commands):
            with patch.object(History, "_get_path", return_value=path):
                h = History("test")
                for cmd in commands:
                    h.add(cmd)
                    time.sleep(0.001)

        threads = [
            threading.Thread(target=writer, args=([f"t1-{i}" for i in range(5)],)),
            threading.Thread(target=writer, args=([f"t2-{i}" for i in range(5)],)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Load and check
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            # Should have all 10 commands
            assert len(h.file_items) >= 10

    def test_concurrent_read_write(self, temp_dir):
        """Test that reads during writes don't block or corrupt."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        results = []

        def writer():
            with patch.object(History, "_get_path", return_value=path):
                h = History("test")
                for i in range(10):
                    h.add(f"cmd-{i}")
                    time.sleep(0.01)

        def reader():
            with patch.object(History, "_get_path", return_value=path):
                time.sleep(0.05)  # Let writer start
                h = History("test")
                results.append(len(h.file_items))

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Reader should have seen some commands
        assert len(results) == 1
        assert results[0] >= 0  # Could be any number depending on timing


class TestDateRollover:
    """Tests for date rollover behavior."""

    def test_path_computed_fresh(self, temp_dir):
        """Test that path is computed fresh on each access."""
        # Create history with a fixed path
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("command1")

            # Simulate date change by patching _get_path to return a new path
            new_path = os.path.join(temp_dir, f"history-test-{NEXT_DAY}")
            with patch.object(h, "_get_path", return_value=new_path):
                h.add("command2")

                # First command should be in old file
                assert os.path.exists(path)
                # Second command should be in new file
                assert os.path.exists(new_path)


class TestProjectScoping:
    """Tests for project-based scoping."""

    def test_different_projects_separate_files(self, temp_dir):
        """Test that different projects use separate files."""
        path1 = os.path.join(temp_dir, f"history-project1-{TODAY}")
        path2 = os.path.join(temp_dir, f"history-project2-{TODAY}")

        with patch.object(History, "_get_path", return_value=path1):
            h1 = History("project1")
            h1.add("from project1")

        with patch.object(History, "_get_path", return_value=path2):
            h2 = History("project2")
            h2.add("from project2")

        # Each should only see their own commands
        with patch.object(History, "_get_path", return_value=path1):
            h1_reload = History("project1")
            assert h1_reload.get_all() == ["from project1"]

        with patch.object(History, "_get_path", return_value=path2):
            h2_reload = History("project2")
            assert h2_reload.get_all() == ["from project2"]

    def test_same_project_shares_history(self, temp_dir):
        """Test that same project shares history across instances."""
        path = os.path.join(temp_dir, f"history-myproject-{TODAY}")

        with patch.object(History, "_get_path", return_value=path):
            h1 = History("myproject")
            h1.add("from instance1")

        with patch.object(History, "_get_path", return_value=path):
            h2 = History("myproject")
            h2.add("from instance2")

            # Should see both commands (newest first)
            all_cmds = h2.get_all()
            assert "from instance2" in all_cmds
            assert "from instance1" in all_cmds


class TestSlashCommandBlacklist:
    """Test that slash commands are excluded from history."""

    def test_slash_command_not_added(self, temp_dir):
        """Test that slash commands are not added to history."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("/help")
            h.add("/quit")
            h.add("/model devstral")
            assert len(h.session_items) == 0

    def test_slash_command_with_leading_spaces_not_added(self, temp_dir):
        """Test that slash commands with leading spaces are not added."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("   /help")
            assert len(h.session_items) == 0

    def test_normal_command_added_after_slash(self, temp_dir):
        """Test that normal commands are still added after slash commands."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("/quit")
            h.add("hello world")
            h.add("/help")
            h.add("another command")
            assert len(h.session_items) == 2
            assert h.session_items[0][1] == "hello world"
            assert h.session_items[1][1] == "another command"


class TestPrefixNavigation:
    """Test prefix-based history navigation."""

    def test_start_prefix_navigation(self, temp_dir):
        """Test starting prefix navigation mode."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("git status")
            h.add("git log")
            h.add("git diff")
            h.add("ls -la")
            h.add("cat file.txt")

            # Start prefix mode with "git"
            h.start_prefix_navigation("git")
            assert h.in_prefix_mode()
            assert h.get_prefix() == "git"

    def test_prefix_navigation_finds_matches(self, temp_dir):
        """Test that prefix navigation finds matching commands."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("git status")
            h.add("git log")
            h.add("git diff")
            h.add("ls -la")
            h.add("cat file.txt")

            # Start prefix mode with "git"
            h.start_prefix_navigation("git")

            # Should find git commands (newest to oldest)
            assert h.up_with_prefix() == "git diff"
            assert h.up_with_prefix() == "git log"
            assert h.up_with_prefix() == "git status"

    def test_prefix_navigation_down(self, temp_dir):
        """Test navigating down in prefix mode."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("git status")
            h.add("git log")
            h.add("git diff")

            h.start_prefix_navigation("git")

            # Go up twice
            h.up_with_prefix()  # git diff
            h.up_with_prefix()  # git log

            # Go down
            assert h.down_with_prefix() == "git diff"

            # Go down again returns None (back to prefix)
            assert h.down_with_prefix() is None

    def test_prefix_navigation_no_matches(self, temp_dir):
        """Test prefix navigation with no matches."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("git status")
            h.add("ls -la")

            h.start_prefix_navigation("xyz")
            assert h.up_with_prefix() is None

    def test_prefix_mode_resets_on_reset(self, temp_dir):
        """Test that reset() clears prefix mode."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("git status")

            h.start_prefix_navigation("git")
            assert h.in_prefix_mode()

            h.reset()
            assert not h.in_prefix_mode()
            assert h.get_prefix() is None

    def test_prefix_navigation_preserves_original_prefix(self, temp_dir):
        """Test that the original prefix is preserved during navigation.

        This is the key bug fix: when navigating with Ctrl+B, the prefix
        should be the originally typed text, not the current buffer content.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("/quit")  # This should NOT be in history due to blacklist
            h.add("git status")
            h.add("git log")
            h.add("ls -la")

            # Start prefix mode with "git"
            h.start_prefix_navigation("git")

            # Navigate up - should get git log
            assert h.up_with_prefix() == "git log"

            # Navigate up again - should get git status
            assert h.up_with_prefix() == "git status"

            # The prefix should still be "git", not changed
            assert h.get_prefix() == "git"

            # Navigate down - should go back to git log
            assert h.down_with_prefix() == "git log"

            # Navigate down again - now return None (back to showing prefix)
            assert h.down_with_prefix() is None
            assert h.get_prefix() == "git"

    def test_slash_commands_not_matched_in_prefix_mode(self, temp_dir):
        """Test that slash commands don't appear even in prefix mode.

        Since slash commands are blacklisted, they shouldn't appear
        when searching with '/' as prefix.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("/quit")
            h.add("/help")
            h.add("normal command")

            # Try to search for slash commands
            h.start_prefix_navigation("/")
            assert h.up_with_prefix() is None  # No matches

    def test_prefix_navigation_does_not_affect_normal_navigation(self, temp_dir):
        """Test that prefix navigation doesn't break normal up/down."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.add("third")

            # Do some prefix navigation
            h.start_prefix_navigation("s")
            h.up_with_prefix()  # Gets "second"

            # Reset and use normal navigation
            h.reset()
            assert h.up() == "third"
            assert h.up() == "second"
            assert h.up() == "first"

    def test_ctrl_b_twice_continues_navigation(self, temp_dir):
        """Test that calling up_with_prefix multiple times continues navigation.

        This tests the bug fix: when Ctrl+B is pressed twice, the second press
        should continue navigating through matches, NOT restart with the
        currently displayed text as a new prefix.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("hello world")
            h.add("hello there")
            h.add("goodbye")

            # First Ctrl+B: start prefix mode with "hel"
            h.start_prefix_navigation("hel")
            result1 = h.up_with_prefix()
            assert result1 == "hello there"  # Most recent match

            # Second Ctrl+B: should continue, not restart
            # The key insight: we're ALREADY in prefix mode, so we just call up_with_prefix again
            # We do NOT call start_prefix_navigation again with "hello there"
            assert h.in_prefix_mode()  # Still in prefix mode
            result2 = h.up_with_prefix()
            assert result2 == "hello world"  # Next older match

            # Third Ctrl+B: at oldest, should stay there
            result3 = h.up_with_prefix()
            assert result3 == "hello world"  # Stays at oldest

    def test_in_prefix_mode_prevents_restart(self, temp_dir):
        """Test that checking in_prefix_mode() prevents restarting navigation.

        This verifies the actual code pattern used in the TUI handler.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("hello world")
            h.add("hello there")
            h.add("goodbye")

            # Simulate TUI handler behavior
            def handle_ctrl_b(prefix):
                if prefix:
                    if not h.in_prefix_mode():
                        h.start_prefix_navigation(prefix)
                    return h.up_with_prefix()
                return h.up()

            # First call with "hel"
            result1 = handle_ctrl_b("hel")
            assert result1 == "hello there"

            # Second call - prefix is now the matched text, but we should NOT restart
            result2 = handle_ctrl_b("hello there")
            assert result2 == "hello world"

            # Third call
            result3 = handle_ctrl_b("hello world")
            assert result3 == "hello world"  # At oldest, stays there

    def test_empty_buffer_starts_prefix_mode(self, temp_dir):
        """Test that Ctrl+B with empty buffer starts prefix mode with empty prefix.

        This tests the bug fix: when Ctrl+B is pressed with an empty buffer,
        it should start prefix mode with an empty prefix (matches all items),
        not use normal up() navigation.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("first")
            h.add("second")
            h.add("third")

            # Empty buffer should still start prefix mode
            h.start_prefix_navigation("")  # empty prefix
            assert h.in_prefix_mode()
            assert h.get_prefix() == ""

            # Should match all items
            assert h.up_with_prefix() == "third"
            assert h.up_with_prefix() == "second"
            assert h.up_with_prefix() == "first"

    def test_input_change_restarts_prefix_mode(self, temp_dir):
        """Test that changing input restarts prefix mode.

        This tests the bug fix: when user types something different after
        navigating, prefix mode should restart with the new input.
        """
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("square number 1")
            h.add("square number 2")
            h.add("test command")

            # Start with "square"
            h.start_prefix_navigation("square")
            result1 = h.up_with_prefix()
            assert result1 == "square number 2"

            # Continue navigating (buffer shows the match)
            result2 = h.up_with_prefix()
            assert result2 == "square number 1"

            # Now user changes input to "test" - should restart
            # This simulates the TUI handler's check
            new_prefix = "test"
            saved_prefix = h.get_prefix()
            current_match = (
                h._prefix_matches[h._prefix_match_idx][1]
                if h._prefix_match_idx >= 0
                else None
            )

            # Buffer doesn't match saved prefix or current match
            assert new_prefix != saved_prefix
            assert new_prefix != current_match

            # Handler would restart prefix mode
            h.start_prefix_navigation(new_prefix)
            result3 = h.up_with_prefix()
            assert result3 == "test command"

    def test_prefix_navigation_deduplicates(self, temp_dir):
        """Test that prefix navigation deduplicates history items."""
        path = os.path.join(temp_dir, f"history-test-{TODAY}")
        with patch.object(History, "_get_path", return_value=path):
            h = History("test")
            h.add("cmd one")
            h.add("cmd two")
            h.add("cmd one")  # duplicate - should be deduplicated
            h.add("cmd three")

            h.start_prefix_navigation("cmd")

            # Should have 3 unique matches, not 4
            assert len(h._prefix_matches) == 3

            # Navigate - should get each unique value once
            result1 = h.up_with_prefix()
            result2 = h.up_with_prefix()
            result3 = h.up_with_prefix()

            # Results should be unique
            results = [result1, result2, result3]
            assert len(set(results)) == 3  # All different
            assert "cmd one" in results
            assert "cmd two" in results
            assert "cmd three" in results
