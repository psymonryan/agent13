"""Pytest configuration and shared fixtures."""

import atexit
import pytest
import tempfile
import os
import signal

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
