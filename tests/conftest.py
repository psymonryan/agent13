"""Pytest configuration and shared fixtures."""

import atexit
import pytest
import tempfile
import os
import signal

import pytest_httpserver

from .mock_llm_helpers import make_models_handler, make_chat_handler


# Create marker file at module level (before any tests run)
# This ensures subprocess-spawned processes can detect they're under test
_MARKER_PATH = os.path.join(os.path.dirname(__file__), ".testing")


def _cleanup_marker():
    """Remove the testing marker file if it exists."""
    if os.path.exists(_MARKER_PATH):
        try:
            os.unlink(_MARKER_PATH)
        except OSError:
            pass


def _signal_handler(signum, frame):
    """Handle signals by cleaning up and exiting."""
    _cleanup_marker()
    # Re-raise the signal with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register cleanup for normal exit
atexit.register(_cleanup_marker)

# Register cleanup for common interrupt signals
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# Create the marker file
with open(_MARKER_PATH, "w") as f:
    f.write("pytest marker\n")


@pytest.fixture(scope="session", autouse=True)
def testing_marker():
    """Ensure marker file exists for the test session.

    The marker file is created at module load time (above) so that it's
    available before any tests run. Cleanup is handled by atexit and
    signal handlers, so the file is removed even on interrupt/timeout.
    """
    yield
    # Cleanup is handled by atexit, but also clean up here for normal exit
    _cleanup_marker()


@pytest.fixture
def temp_file():
    """Create a temporary file and return its path."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory and return its path."""
    path = tempfile.mkdtemp()
    yield path
    import shutil

    if os.path.exists(path):
        shutil.rmtree(path)


# ── Mock LLM server fixtures ─────────────────────────────────────────────


@pytest.fixture
def mock_llm_server():
    """Start a mock LLM server on a dynamic port."""
    server = pytest_httpserver.HTTPServer()
    server.expect_request("/v1/models").respond_with_handler(make_models_handler())
    server.expect_request(
        "/v1/chat/completions", method="POST"
    ).respond_with_handler(make_chat_handler())
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_provider_env(tmp_path, mock_llm_server):
    """Create a temp config directory pointing to mock server.

    Sets AGENT13_CONFIG_DIR for full isolation.
    Returns dict of environment variables for subprocess.
    """
    config_dir = tmp_path / "agent13-config"
    config_dir.mkdir()

    config_content = f"""[[providers]]
name = "test_mock"
api_base = "http://localhost:{mock_llm_server.port}/v1"
api_key = "test-key"
"""
    (config_dir / "config.toml").write_text(config_content)

    env = os.environ.copy()
    env["AGENT13_CONFIG_DIR"] = str(config_dir)
    env["AGENT13_NO_UPDATE_CHECK"] = "1"

    return env
