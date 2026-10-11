"""Changelog fetch and formatting for the startup update offer.

Fetches the raw CHANGELOG.md from GitHub (same repo as releases), parses
Keep-a-Changelog format, and extracts the sections newer than the running
version for display above the interactive update offer.

Design contract (agreed 2026-10-10):
- Fetched ONLY when the daily update check has produced an offer - a user
  with no pending update never downloads a changelog.
- No cache file, no seen-version tracking: the offer is printed once and
  the process either exits (applied) or continues (declined).
- Every failure is silent: the offer displays exactly as before, minus
  notes. This module never raises into the startup path.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass

import httpx

from agent13.updater import GITHUB_OWNER, GITHUB_REPO, _parse_version

logger = logging.getLogger(__name__)

# Raw content URL - stable across releases, no API rate limit, no auth.
CHANGELOG_RAW_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/CHANGELOG.md"
)

# Generous for a static CDN file, but half the releases-API timeout so the
# worst-case added startup latency stays modest.
FETCH_TIMEOUT_SECONDS = 5.0

# Newest full sections shown above the offer.
DEFAULT_SECTION_CAP = 2

# Display width for wrapped bullets (matches the offer block's column budget).
_WRAP_WIDTH = 76

# "## [0.5.1] - 2026-09-24" -> version "0.5.1", date "2026-09-24"
# 0-3 leading spaces allowed: CommonMark still counts that as a heading
# (the live CHANGELOG.md contains one 2-space-indented heading).
_HEADING_RE = re.compile(
    r"^[ ]{0,3}##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?\s*$"
)


@dataclass
class Section:
    """One ``## [version] - date`` block from the changelog."""

    version: str
    date: str  # "" when the heading carries no date
    bullets: list[str]


def fetch_changelog(timeout: float = FETCH_TIMEOUT_SECONDS) -> str | None:
    """Fetch raw CHANGELOG.md text from GitHub.

    Returns:
        The file content, or None on any network/HTTP failure.
    """
    try:
        resp = httpx.get(CHANGELOG_RAW_URL, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        logger.debug("Changelog fetch returned status %d", resp.status_code)
    except (httpx.HTTPError, OSError) as e:
        logger.debug("Failed to fetch changelog: %s", e)
    return None


def parse_changelog(content: str) -> list[Section]:
    """Parse Keep-a-Changelog markdown into sections, newest first.

    Sections with no bullets are dropped. Non-version headings (e.g.
    ``## [Unreleased]``) are kept only if they carry bullets; their
    version strings simply fail version comparison later.

    Args:
        content: Raw CHANGELOG.md text.

    Returns:
        Sections in file order (Keep-a-Changelog is newest-first).
    """
    sections: list[Section] = []
    current: Section | None = None
    for line in content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current is not None and current.bullets:
                sections.append(current)
            current = Section(
                version=match.group(1).strip(), date=match.group(2) or "", bullets=[]
            )
        elif current is not None:
            stripped = line.strip()
            if stripped.startswith("- "):
                current.bullets.append(stripped[2:].strip())
    if current is not None and current.bullets:
        sections.append(current)
    return sections


def _strip_emphasis(text: str) -> str:
    """Remove ``**bold**`` markers - plain terminals render them as litter."""
    return text.replace("**", "")


def _wrap_bullet(text: str) -> str:
    """Wrap one bullet to the display width with hanging indentation."""
    return textwrap.fill(
        text,
        width=_WRAP_WIDTH,
        initial_indent="    - ",
        subsequent_indent="      ",
    )


def select_new_sections(
    sections: list[Section], current_version: str, cap: int = DEFAULT_SECTION_CAP
) -> list[Section]:
    """Pick the sections to display: strictly newer than current_version.

    If current_version is absent from the changelog (very old install),
    fall back to the newest ``cap`` sections regardless of comparability.

    Args:
        sections: Parsed sections, newest first.
        current_version: The running version, e.g. "0.5.0".
        cap: Maximum number of sections to return.

    Returns:
        Up to ``cap`` sections in display order (newest first).
    """
    current = _parse_version(current_version)
    if current:
        for idx, section in enumerate(sections):
            if _parse_version(section.version) == current:
                # Everything above it in a well-ordered Keep-a-Changelog
                # file is newer; the filter also drops non-version
                # sections (e.g. "## [Unreleased]") that precede it.
                return [
                    s for s in sections[:idx] if _parse_version(s.version) > current
                ][:cap]
    newer = [s for s in sections if _parse_version(s.version) > current]
    if newer:
        return newer[:cap]
    return sections[:cap]


def format_whats_new(
    sections: list[Section], current_version: str, cap: int = DEFAULT_SECTION_CAP
) -> str:
    """Format selected sections as the block printed above the update offer.

    Args:
        sections: Parsed sections, newest first.
        current_version: The running version, for the "since" header.
        cap: Maximum number of sections.

    Returns:
        The formatted block, or "" when nothing qualifies (the caller
        then shows the bare offer, as before).
    """
    chosen = select_new_sections(sections, current_version, cap)
    if not chosen:
        return ""
    lines = [f"  What's new since {current_version}:", ""]
    for section in chosen:
        heading = f"  v{section.version}"
        if section.date:
            heading += f"  {section.date}"
        lines.append(heading)
        lines.extend(_wrap_bullet(_strip_emphasis(b)) for b in section.bullets)
        lines.append("")
    return "\n".join(lines).rstrip()


def fetch_whats_new(current_version: str, cap: int = DEFAULT_SECTION_CAP) -> str:
    """One-call helper for the CLI: fetch, parse, format. Never raises.

    Args:
        current_version: The running version, e.g. "0.5.0".
        cap: Maximum number of sections to show.

    Returns:
        The formatted what's-new block, or "" on any failure.
    """
    try:
        content = fetch_changelog()
        if not content:
            return ""
        return format_whats_new(parse_changelog(content), current_version, cap)
    except Exception as e:  # noqa: BLE001 - startup path must not break
        logger.debug("Whats-new formatting failed: %s", e)
        return ""
