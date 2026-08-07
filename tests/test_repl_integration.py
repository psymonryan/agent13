"""User experience tests for REPL mode using pexpect.

These tests spawn a REAL REPL process with a mock LLM server,
send real keystrokes, and verify real terminal output.

Unlike wiring tests (test_repl.py), these tests exercise the actual
user experience: readline, display, persistence, and the full command
processing pipeline.

Requires: pytest-httpserver (provides mock HTTP server).
"""

import os
import json
import sys
import time
import pexpect  # for pexpect.EOF only; use spawn_process() from conftest
import pytest
import pytest_httpserver

from werkzeug import Request, Response

from .mock_llm_helpers import make_chat_handler


@pytest.fixture
def repl_env(tmp_path, mock_llm_server):
    """Create a temp config directory pointing to mock server.

    Sets AGENT13_CONFIG_DIR and AGENT13_SAVES_DIR for full isolation.
    Returns dict of environment variables for pexpect.
    """
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()

    config_content = f"""[saves]
location = "central"

[[providers]]
name = "test_mock"
api_base = "http://localhost:{mock_llm_server.port}/v1"
api_key = "test-key"
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(saves_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"

    return env


def _wait_for_repl_prompt(proc, timeout):
    """Wait for the REPL '>' prompt, converting startup EOF to a retryable error.

    PopenSpawn (Windows) uses pipes instead of a pty, so the first spawn can
    occasionally hit EOF before the prompt is emitted -- a startup race that
    does not reflect a real failure. On EOF we raise `_StartupEOFError` so the
    caller can respawn; the captured `proc.before` makes final failures
    diagnosable instead of an opaque EOF.
    """
    try:
        proc.expect(r">", timeout=timeout)
    except pexpect.EOF:
        raise _StartupEOFError(proc)


class _StartupEOFError(Exception):
    """Raised when the REPL dies before emitting its first prompt.

    Carries the captured `before` output so callers can retry with a fresh
    process and, on final failure, surface a readable assertion message.
    """

    def __init__(self, proc):
        self.before = proc.before or ""
        self.exitstatus = getattr(proc, "exitstatus", None)
        super().__init__(
            f"REPL exited before emitting '>' prompt.\n"
            f"exitstatus={self.exitstatus}\n"
            f"captured output:\n{self.before!r}"
        )


def _spawn_repl_with_retry(provider, model, env, timeout=30, retries=1):
    """Spawn a REPL and wait for its first prompt, retrying on startup EOF.

    Windows PopenSpawn can non-deterministically hit EOF at startup (a pipe
    buffering race, not a real failure). On `_StartupEOFError` we close the
    dead process and respawn, up to `retries` extra attempts. The final
    attempt's error propagates as an AssertionError with captured output.
    """
    from .helpers import spawn_process
    last_err = None
    for attempt in range(retries + 1):
        proc = spawn_process(
            "uv",
            args=["run", "agent13", provider, "--repl", "--model", model],
            env=env,
            encoding="utf-8",
            timeout=timeout,
            dimensions=(50, 200),
            maxread=4096,
        )
        proc.timeout = timeout
        try:
            _wait_for_repl_prompt(proc, timeout)
            return proc
        except _StartupEOFError as err:
            last_err = err
            try:
                proc.close()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(1)
                continue
    raise AssertionError(
        f"REPL failed to start after {retries + 1} attempt(s).\n{last_err}"
    )


def spawn_repl(env, timeout=30):
    """Spawn a REPL process with the given environment.

    Returns a process handle ready for interaction.
    Uses pexpect.spawn on Unix, PopenSpawn on Windows.
    """
    return _spawn_repl_with_retry("test_mock", "mock-model", env, timeout=timeout)


# ─── Helpers ───────────────────────────────────────────────────────


def get_output_since_prompt(proc):
    """Get all output from the REPL since the last prompt match.

    This reads everything available without blocking, then returns it.
    """
    # The 'before' attribute contains everything since the last expect()
    return proc.before or ""


def wait_for_prompt(proc, timeout=10):
    """Wait for the > prompt to appear."""
    proc.expect(r">", timeout=timeout)


# ─── Test: Basic send/receive ──────────────────────────────────────


class TestBasicExperience:
    """Core user experience: send a message, get a response."""

    def test_send_receives_response(self, repl_env):
        """User sends a message and sees a response from the LLM."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            time.sleep(3)

            # Should see mock response
            proc.expect("Hello! I'm a mock assistant", timeout=10)

            # Should return to prompt
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_send_shows_processing(self, repl_env):
        """User sends a message and sees [processing] indicator."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            time.sleep(1)

            # Should see processing indicator (before response arrives)
            # Note: this may appear before the response
            proc.expect(r"\[processing\]|Hello", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /help ───────────────────────────────────────────────────


class TestHelpExperience:
    """User asks for help."""

    def test_help_shows_commands_header(self, repl_env):
        """User types /help and sees 'Commands:' header."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/help")
            time.sleep(1)

            # Should see commands header
            proc.expect("Commands:", timeout=5)

            # Should see at least one command
            proc.expect("/quit", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_help_shows_all_command_groups(self, repl_env):
        """User types /help and sees commands from each category."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/help")
            time.sleep(1)

            # Should see mode explanation at the end
            proc.expect("Streaming mode", timeout=5)
            proc.expect("Input mode", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /status ─────────────────────────────────────────────────


class TestStatusExperience:
    """User checks agent status."""

    def test_status_shows_model(self, repl_env):
        """User types /status and sees model name."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/status")
            time.sleep(1)

            # Should see model name
            proc.expect("mock-model", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_idle(self, repl_env):
        """User types /status when idle and sees idle state."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/status")
            time.sleep(1)

            # Should see idle status
            proc.expect("idle", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /save and /load round-trip ──────────────────────────────


class TestSaveLoadExperience:
    """User saves and loads conversations."""

    @pytest.fixture(autouse=True)
    def clean_saves_dir(self, repl_env):
        """Ensure saves directory is isolated in tmp_path and empty before test.

        AGENT13_SAVES_DIR is set by repl_env to tmp_path/saves, so the real
        .agent13/saves/ is never touched. Asserts emptiness as a failsafe —
        if the dir somehow has files, we fail hard instead of deleting blindly.
        tmp_path cleanup handles post-test teardown.
        """
        saves_dir = repl_env["AGENT13_SAVES_DIR"]
        assert os.path.isdir(saves_dir), f"Saves dir missing: {saves_dir}"
        contents = os.listdir(saves_dir)
        assert len(contents) == 0, (
            f"Test saves dir not empty before test: {contents}. "
            f"Refusing to proceed — check for stale files in {saves_dir}"
        )
        yield

    def test_save_creates_file_and_shows_path(self, repl_env):
        """User saves, gets overwrite warning, then overwrites with -y."""
        proc = spawn_repl(repl_env)

        try:
            # Send a message so there's something to save
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            # Save (first time — should succeed)
            proc.sendline("/save mywork")
            time.sleep(2)
            proc.expect(r"Saved.*\.ctx|\.ctx.*Saved", timeout=5)
            wait_for_prompt(proc)

            # Save again — should get overwrite warning
            proc.sendline("/save mywork")
            time.sleep(2)
            proc.expect("already exists|Use.*-y", timeout=5)
            wait_for_prompt(proc)

            # Save with -y — should succeed
            proc.sendline("/save mywork -y")
            time.sleep(2)
            proc.expect(r"Saved.*\.ctx|\.ctx.*Saved", timeout=5)
            wait_for_prompt(proc)

        finally:
            proc.close()

    def test_load_displays_conversation(self, repl_env):
        """User loads a saved conversation and sees messages displayed."""
        proc = spawn_repl(repl_env)

        try:
            # Send, save, clear, load
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            proc.sendline("/save roundtrip")
            time.sleep(2)
            proc.expect("Saved", timeout=5)
            wait_for_prompt(proc)

            proc.sendline("/clear")
            time.sleep(1)
            proc.expect("leared", timeout=5)
            wait_for_prompt(proc)

            proc.sendline("/load roundtrip")
            time.sleep(2)

            # Should see the loaded message content
            proc.expect("hello", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_load_no_name_lists_available_saves(self, repl_env):
        """User types /load with no name and sees available saves."""
        proc = spawn_repl(repl_env)

        try:
            # Create a save first
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            proc.sendline("/save mylist")
            time.sleep(2)
            proc.expect("Saved", timeout=5)
            wait_for_prompt(proc)

            # Load with no name
            proc.sendline("/load")
            time.sleep(1)

            # Should see the save name listed
            proc.expect("mylist", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_load_nonexistent_shows_error(self, repl_env):
        """User tries to load a save that doesn't exist."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/load nonexistent-xyz")
            time.sleep(1)

            # Should see "not found" or similar error
            proc.expect("(?i)not found|no save", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_load_with_ctx_extension_works(self, repl_env):
        """User types /load foo.ctx — should not double the extension."""
        proc = spawn_repl(repl_env)

        try:
            # Create a save first
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            proc.sendline("/save ctxtest")
            time.sleep(2)
            proc.expect("Saved", timeout=5)
            wait_for_prompt(proc)

            # Clear and load with explicit .ctx extension
            proc.sendline("/clear")
            time.sleep(1)
            proc.expect("leared", timeout=5)
            wait_for_prompt(proc)

            proc.sendline("/load ctxtest.ctx")
            time.sleep(2)

            # Should succeed — not show "not found"
            proc.expect("Loaded context", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ──── Test: Multi-line mode ─────────────────────────────────────────


class TestMultiLineExperience:
    """User enters multi-line mode."""

    def test_multi_line_sends_assembled_text(self, repl_env):
        """User enters /multi, types lines, sends with ."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/multi")
            time.sleep(0.5)

            # Should see continuation prompt (...)
            proc.expect(r"\.\.\.", timeout=3)

            proc.sendline("First line")
            time.sleep(0.3)
            proc.sendline("Second line")
            time.sleep(0.3)

            # Send with .
            proc.sendline(".")
            time.sleep(3)

            # Should get response with assembled content
            proc.expect("Received: First line", timeout=10)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_backslash_enters_continuation(self, repl_env):
        """User types line ending with \\ and enters continuation mode."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("First part \\")
            time.sleep(0.5)

            # Should see continuation prompt
            proc.expect(r"\.\.\.", timeout=3)

            proc.sendline("Second part")
            time.sleep(0.3)

            # Send with .
            proc.sendline(".")
            time.sleep(3)

            # Should get response
            proc.expect("Received: First part", timeout=10)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_multi_cancel_returns_to_normal(self, repl_env):
        """User enters /multi then cancels with /cancel."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/multi")
            time.sleep(0.5)
            proc.expect(r"\.\.\.", timeout=3)

            # Cancel
            proc.sendline("/cancel")
            time.sleep(1)

            # Should see cancelled message
            proc.expect("(?i)cancel", timeout=5)

            # Should be back at normal prompt
            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /clear ──────────────────────────────────────────────────


class TestClearExperience:
    """User clears conversation context."""

    def test_clear_confirms_message_count(self, repl_env):
        """User sends a message then /clear sees count of cleared messages."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            proc.sendline("/clear")
            time.sleep(1)

            # Should see "leared" (cleared/Cleared) with message count
            proc.expect("leared", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /queue ──────────────────────────────────────────────────


class TestQueueExperience:
    """User checks the message queue."""

    def test_queue_empty_shows_empty(self, repl_env):
        """User types /queue when empty and sees 'empty' message."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/queue")
            time.sleep(1)

            # Should see "empty" or "no items"
            proc.expect("(?i)empty|no items|nothing", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /history ────────────────────────────────────────────────


class TestHistoryExperience:
    """User views message history."""

    def test_history_after_send_shows_message(self, repl_env):
        """User sends message then views /history and sees it listed."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("hello")
            time.sleep(3)
            proc.expect("Hello! I'm a mock assistant", timeout=10)
            wait_for_prompt(proc)

            proc.sendline("/history")
            time.sleep(1)

            # Should see the message content in history
            proc.expect("hello", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_history_empty_shows_empty(self, repl_env):
        """User types /history with no messages and sees empty message."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/history")
            time.sleep(1)

            # Should see "no history" or "empty"
            proc.expect("(?i)no history|empty|no messages", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /delete ─────────────────────────────────────────────────


class TestDeleteExperience:
    """User deletes items."""

    def test_delete_no_args_shows_usage(self, repl_env):
        """User types /delete with no args and sees usage message."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/delete")
            time.sleep(1)

            # Should see "Usage" or "usage"
            proc.expect("[Uu]sage", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /pause and /resume ─────────────────────────────────────


class TestPauseResumeExperience:
    """User pauses and resumes the agent."""

    def test_pause_when_idle_shows_nothing(self, repl_env):
        """User types /pause when idle — sees 'nothing to pause'."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/pause")
            time.sleep(1)

            # Should see "nothing to pause" or similar
            proc.expect("(?i)nothing|idle|not processing", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: Interrupt prefix ───────────────────────────────────────


class TestInterruptExperience:
    """User sends interrupt-priority messages."""

    def test_interrupt_prefix_shows_notification(self, repl_env):
        """User sends !! message and sees interrupt notification."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("!!fix this now")
            time.sleep(3)

            # Should see interrupt notification
            proc.expect("(?i)interrupt", timeout=10)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_priority_prefix_shows_notification(self, repl_env):
        """User sends ! message and sees priority notification."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("!important task")
            time.sleep(3)

            # Should see priority notification
            proc.expect("(?i)priority", timeout=10)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /quit and Ctrl+D ───────────────────────────────────────


class TestExitExperience:
    """User exits the REPL."""

    def test_quit_exits_cleanly(self, repl_env):
        """User types /quit and process exits with Goodbye."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/quit")
            time.sleep(1)

            # Should see goodbye
            proc.expect("Goodbye", timeout=5)

            # Process should exit
            proc.expect(pexpect.EOF, timeout=5)
        finally:
            proc.close()

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Ctrl+D EOF is a Unix terminal concept; "
                               "PopenSpawn pipes can't simulate it on Windows")
    def test_eof_exits_cleanly(self, repl_env):
        """User presses Ctrl+D and process exits cleanly."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendcontrol("d")
            time.sleep(2)

            # Should see exit message (EOF or Goodbye)
            proc.expect("(?i)EOF|Goodbye|exiting", timeout=5)

            # Process should exit
            proc.expect(pexpect.EOF, timeout=5)
        finally:
            proc.close()

    def test_exit_command_works(self, repl_env):
        """User types /exit (alternative to /quit)."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/exit")
            time.sleep(1)

            proc.expect("Goodbye", timeout=5)
            proc.expect(pexpect.EOF, timeout=5)
        finally:
            proc.close()


# ─── Test: Unknown command ─────────────────────────────────────────


class TestUnknownCommandExperience:
    """User types an unknown slash command."""

    def test_unknown_command_shows_error(self, repl_env):
        """User types /foo and sees unknown command error."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/foo")
            time.sleep(1)

            # Should see "unknown" or "not found"
            proc.expect("(?i)unknown|not found", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: Banner ─────────────────────────────────────────────────


class TestBannerExperience:
    """User sees the startup banner."""

    def test_banner_shows_provider_and_model(self, repl_env):
        """User starts REPL and sees provider/model in banner."""
        proc = spawn_repl(repl_env)

        try:
            # Banner is in proc.before (matched before > prompt)
            output = proc.before or ""

            # Should show provider/model
            assert "test_mock" in output or "mock-model" in output, (
                f"Banner should show provider or model. Got: {output!r}"
            )
        finally:
            proc.close()


# ─── Test: /stop ───────────────────────────────────────────────────


class TestStopExperience:
    """User tries /stop when nothing is processing."""

    def test_stop_when_idle(self, repl_env):
        """User types /stop when idle — sees nothing to stop."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/stop")
            time.sleep(1)

            # Should see "nothing" or "not processing"
            proc.expect("(?i)nothing|idle|not processing", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /save with no name ─────────────────────────────────────


class TestSaveNoNameExperience:
    """User types /save with no filename."""

    def test_save_no_name_shows_usage(self, repl_env):
        """User types /save with no name — sees usage message."""
        proc = spawn_repl(repl_env)

        try:
            proc.sendline("/save")
            time.sleep(1)

            # Should see usage or help
            proc.expect("(?i)Usage|name", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

# -- Multi-model fixture for /model tests ----------------------------------


def make_multi_models_handler():
    """Handler for GET /v1/models returning multiple models."""

    def handler(request: Request):
        return Response(
            json.dumps(
                {
                    "data": [
                        {"id": "alpha-model", "object": "model", "owned_by": "test"},
                        {"id": "beta-model", "object": "model", "owned_by": "test"},
                        {"id": "gamma-model", "object": "model", "owned_by": "test"},
                    ]
                }
            ),
            content_type="application/json",
        )

    return handler


@pytest.fixture
def multi_model_server():
    """Start a mock LLM server that returns multiple models."""
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(
        make_multi_models_handler()
    )
    server.expect_request(
        "/v1/chat/completions", method="POST"
    ).respond_with_handler(make_chat_handler())
    server.start()
    yield server
    server.stop()


@pytest.fixture
def multi_model_repl_env(tmp_path, multi_model_server):
    """Create a temp config directory pointing to multi-model server."""
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()

    config_content = f"""[saves]
location = "central"

[[providers]]
name = "test_multi"
api_base = "http://localhost:{multi_model_server.port}/v1"
api_key = "test-key"
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_SAVES_DIR"] = str(saves_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"

    return env


def spawn_multi_model_repl(env, timeout=30):
    """Spawn a REPL with multi-model server."""
    return _spawn_repl_with_retry(
        "test_multi", "alpha-model", env, timeout=timeout
    )


# ─── Test: /model experience ───────────────────────────────────


class TestModelExperience:
    """User interacts with /model in the REPL."""

    def test_model_lists_available_models(self, multi_model_repl_env):
        """User types /model and sees all available models listed."""
        proc = spawn_multi_model_repl(multi_model_repl_env)
        try:
            proc.sendline("/model")
            time.sleep(1)

            proc.expect("alpha-model", timeout=5)
            proc.expect("beta-model", timeout=5)
            proc.expect("gamma-model", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_model_marks_current_with_asterisk(self, multi_model_repl_env):
        """User types /model and current model is marked with *."""
        proc = spawn_multi_model_repl(multi_model_repl_env)
        try:
            proc.sendline("/model")
            time.sleep(1)

            # alpha-model was selected at spawn time
            proc.expect(r"alpha-model \*", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_model_select_by_name(self, multi_model_repl_env):
        """User selects a model by name and sees confirmation."""
        proc = spawn_multi_model_repl(multi_model_repl_env)
        try:
            proc.sendline("/model beta-model")
            time.sleep(1)

            proc.expect("Model set to: beta-model", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_model_select_by_number(self, multi_model_repl_env):
        """User selects a model by number."""
        proc = spawn_multi_model_repl(multi_model_repl_env)
        try:
            proc.sendline("/model 3")
            time.sleep(1)

            proc.expect("Model set to: gamma-model", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_model_no_such_model(self, multi_model_repl_env):
        """User tries to select a non-existent model — sees error."""
        proc = spawn_multi_model_repl(multi_model_repl_env)
        try:
            proc.sendline("/model nonexistent-model")
            time.sleep(1)

            proc.expect("No model matching", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /provider experience ───────────────────────────────


class TestProviderExperience:
    """User interacts with /provider in the REPL."""

    def test_provider_no_args_shows_list(self, repl_env):
        """User types /provider with no args — sees provider list."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/provider")

            # Config has "test_mock" provider — should appear in list
            proc.expect("Available providers", timeout=10)
            proc.expect("test_mock", timeout=5)
            proc.expect("/provider <name>", timeout=5)

            wait_for_prompt(proc)
        finally:
            proc.close()


# ─── Test: /status experience ────────────────────────────────────


class TestStatusSections:
    """User interacts with /status sections in the REPL."""

    def test_status_shows_session_section(self, repl_env):
        """User types /status — sees Session section."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Session", timeout=10)
            proc.expect("status:", timeout=5)
            proc.expect("run time:", timeout=5)
            proc.expect("cwd:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_provider_section(self, repl_env):
        """/status shows Provider section with provider and model."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Provider", timeout=10)
            proc.expect("provider:", timeout=5)
            proc.expect("model:", timeout=5)
            proc.expect("prompt:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_context_section(self, repl_env):
        """/status shows Context section with token counts."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Context", timeout=10)
            proc.expect("prompt tokens:", timeout=5)
            proc.expect("completion tokens:", timeout=5)
            proc.expect("total tokens:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_connectivity(self, repl_env):
        """/status shows Connectivity section with MCP status."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Connectivity", timeout=10)
            proc.expect("mcp:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_tools_section(self, repl_env):
        """/status shows Tools section."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Tools", timeout=10)
            proc.expect("success/calls:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_settings_section(self, repl_env):
        """/status shows Settings section with all toggles."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("Settings", timeout=10)
            proc.expect("sandbox:", timeout=5)
            proc.expect("journal:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /sandbox experience ---------------------------------------------


class TestSandboxExperience:
    """User interacts with /sandbox in the REPL."""

    def test_sandbox_no_args_shows_config(self, repl_env):
        """User types /sandbox with no args — sees sandbox configuration."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/sandbox")
            proc.expect("Sandbox Configuration", timeout=10)
            proc.expect("Current mode:", timeout=5)
            proc.expect("Config default:", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_sandbox_set_mode(self, repl_env):
        """User types /sandbox <mode> — sees confirmation."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/sandbox off")
            proc.expect("Sandbox mode set to: off", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_sandbox_invalid_mode(self, repl_env):
        """User types /sandbox <invalid> — sees error."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/sandbox bogus")
            proc.expect("Error", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /devel experience -----------------------------------------------


class TestDevelExperience:
    """User interacts with /devel in the REPL."""

    def test_devel_no_args_shows_status(self, repl_env):
        """User types /devel with no args — sees current state."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/devel")
            proc.expect("Devel mode:", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_devel_toggle_on(self, repl_env):
        """User types /devel on — sees confirmation."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/devel on")
            proc.expect("Devel mode enabled", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_devel_toggle_off(self, repl_env):
        """User types /devel off — sees confirmation."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/devel on")
            proc.expect("Devel mode enabled", timeout=10)
            proc.sendline("/devel off")
            proc.expect("Devel mode disabled", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /tools experience ------------------------------------------------


class TestToolsExperience:
    """User interacts with /tools in the REPL."""

    def test_tools_no_calls(self, repl_env):
        """User types /tools with no calls — sees no tool calls message."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/tools")
            proc.expect("No tool calls yet", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /mcp experience --------------------------------------------------


class TestMcpExperience:
    """User interacts with /mcp in the REPL."""

    def test_mcp_no_servers(self, repl_env):
        """User types /mcp with no MCP configured — sees not initialized."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/mcp")
            proc.expect("Not initialized", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_mcp_connect_no_config(self, repl_env):
        """User types /mcp connect — sees no servers message."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/mcp connect")
            proc.expect("No MCP servers configured", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /bell experience ------------------------------------------------


class TestBellExperience:
    """User interacts with /bell in the REPL."""

    def test_bell_no_args_shows_status(self, repl_env):
        """User types /bell with no args — sees current state (always)."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell")
            proc.expect("Bell: on \\(always\\)", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_set_threshold(self, repl_env):
        """User types /bell 30 — sees confirmation."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell 30")
            proc.expect("Bell: on \\(30s\\)", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_off(self, repl_env):
        """User types /bell off — sees disabled confirmation."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell off")
            proc.expect("Bell: off", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_invalid_value(self, repl_env):
        """User types /bell abc — sees usage."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell abc")
            proc.expect("Usage:", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_status_shows_bell(self, repl_env):
        """/status shows bell in Settings section."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/status")
            proc.expect("bell:", timeout=10)
            proc.expect("always", timeout=5)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_toggle_off_then_on(self, repl_env):
        """User toggles bell off then back on."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell off")
            proc.expect("Bell: off", timeout=10)
            wait_for_prompt(proc)
            proc.sendline("/bell 0")
            proc.expect("Bell: on \\(always\\)", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()


# -- Test: /bell-command experience ----------------------------------------


class TestBellCommandExperience:
    """User interacts with /bell-command in the REPL."""

    def test_bell_command_no_args_shows_terminal(self, repl_env):
        """User types /bell-command with no args — sees terminal bell."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell-command")
            proc.expect("Bell command: \\(terminal bell\\)", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_command_set_valid(self, repl_env):
        """User sets a valid bell command."""
        import sys

        exe = os.path.basename(sys.executable)
        proc = spawn_repl(repl_env)
        try:
            proc.sendline(f"/bell-command {exe} -c pass")
            proc.expect("Bell command:", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_command_off(self, repl_env):
        """User clears bell command."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell-command off")
            proc.expect("cleared", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()

    def test_bell_command_invalid(self, repl_env):
        """User sets an invalid bell command — sees error."""
        proc = spawn_repl(repl_env)
        try:
            proc.sendline("/bell-command nonexistent_xyz_123")
            proc.expect("not executable", timeout=10)
            wait_for_prompt(proc)
        finally:
            proc.close()
