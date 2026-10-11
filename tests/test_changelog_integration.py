"""Integration tests for the changelog whats-new block.

Only the network layer is faked (pytest-httpserver serving a fake
CHANGELOG.md); parsing/formatting run for real, exactly as the CLI
startup path will use them.
"""

import pytest

from agent13 import changelog
from agent13.changelog import fetch_whats_new

FAKE_CHANGELOG = """\
# Changelog

## [0.9.0] - 2026-10-09

### Added

- **Fancy feature** - the headline change of this release

## [0.8.1] - 2026-10-02

### Fixed

- a bug fix users on 0.8.0 care about

## [0.8.0] - 2026-09-20

### Added

- earlier feature
"""


@pytest.fixture
def changelog_server(httpserver):
    """Serve the fake changelog and point the module URL at it."""
    httpserver.expect_request("/CHANGELOG.md").respond_with_data(
        FAKE_CHANGELOG, content_type="text/markdown"
    )
    original = changelog.CHANGELOG_RAW_URL
    changelog.CHANGELOG_RAW_URL = httpserver.url_for("/CHANGELOG.md")
    yield httpserver
    changelog.CHANGELOG_RAW_URL = original


class TestFetchWhatsNew:
    def test_happy_path_block_above_offer(self, changelog_server):
        block = fetch_whats_new("0.8.0")
        assert block.startswith("  What's new since 0.8.0:")
        assert "  v0.9.0  2026-10-09" in block
        assert "  v0.8.1  2026-10-02" in block
        # Strictly newer only: the user's own section never appears
        assert "earlier feature" not in block
        assert "**" not in block

    def test_no_newer_versions_means_no_block(self, changelog_server):
        assert fetch_whats_new("0.9.0") == ""

    def test_very_old_install_gets_newest_two(self, changelog_server):
        block = fetch_whats_new("0.1.0")
        assert "  v0.9.0" in block
        assert "  v0.8.1" in block
        assert "earlier feature" not in block

    def test_http_error_yields_empty_string(self, httpserver):
        httpserver.expect_request("/CHANGELOG.md").respond_with_data(
            "server error", status=500
        )
        original = changelog.CHANGELOG_RAW_URL
        changelog.CHANGELOG_RAW_URL = httpserver.url_for("/CHANGELOG.md")
        try:
            assert fetch_whats_new("0.8.0") == ""
        finally:
            changelog.CHANGELOG_RAW_URL = original

    def test_connection_refused_yields_empty_string(self):
        """Unreachable host: port closed on localhost, no server running."""
        original = changelog.CHANGELOG_RAW_URL
        changelog.CHANGELOG_RAW_URL = "http://127.0.0.1:9/CHANGELOG.md"
        try:
            assert fetch_whats_new("0.8.0") == ""
        finally:
            changelog.CHANGELOG_RAW_URL = original

    def test_garbage_content_yields_empty_string(self, httpserver):
        httpserver.expect_request("/CHANGELOG.md").respond_with_data(
            "<html>not a changelog</html>", content_type="text/html"
        )
        original = changelog.CHANGELOG_RAW_URL
        changelog.CHANGELOG_RAW_URL = httpserver.url_for("/CHANGELOG.md")
        try:
            assert fetch_whats_new("0.8.0") == ""
        finally:
            changelog.CHANGELOG_RAW_URL = original

    def test_timeout_is_short(self):
        """The fetch must not stall startup: bounded well under the
        releases-API timeout."""
        assert changelog.FETCH_TIMEOUT_SECONDS <= 5.0
