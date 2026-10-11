"""Unit tests for agent13.changelog - parsing, selection, formatting."""

from agent13.changelog import (
    Section,
    format_whats_new,
    parse_changelog,
    select_new_sections,
)

SAMPLE = """\
# Changelog

All notable changes to this project will be documented in this file.

## [0.5.2] - 2026-10-01

### Added

- **Remote job polling** - check detached job status without ssh wrapping
- something with a very long single-line description that must be wrapped across multiple display lines by the formatter because it exceeds the standard terminal width budget for the offer block by a wide margin

## [0.5.1] - 2026-09-24

### Changed

- fix(windows): close subprocess transports + kill children on cancel
- minor prompt change

## [0.5.0] - 2026-09-24

### Added

- **Auto-context** - keep long sessions usable
- /auto_context_action command

## [0.4.0] - 2026-08-01

### Fixed

- older fix
"""

SAMPLE_NO_DATE = """\
## [1.0.0]

- no date in heading
- second bullet
"""

SAMPLE_UNRELEASED = """\
## [Unreleased]

- unreleased work

## [0.5.0] - 2026-09-24

- released work
"""


class TestParseChangelog:
    def test_parses_sections_newest_first(self):
        sections = parse_changelog(SAMPLE)
        assert [s.version for s in sections] == ["0.5.2", "0.5.1", "0.5.0", "0.4.0"]
        assert [s.date for s in sections] == [
            "2026-10-01",
            "2026-09-24",
            "2026-09-24",
            "2026-08-01",
        ]

    def test_bullets_extracted_per_section(self):
        sections = parse_changelog(SAMPLE)
        assert sections[1].bullets == [
            "fix(windows): close subprocess transports + kill children on cancel",
            "minor prompt change",
        ]

    def test_subheaders_and_prose_ignored(self):
        sections = parse_changelog(SAMPLE)
        # "### Added" and the intro prose never appear as bullets
        assert all("###" not in b for s in sections for b in s.bullets)
        assert all("notable changes" not in b for s in sections for b in s.bullets)

    def test_heading_without_date(self):
        sections = parse_changelog(SAMPLE_NO_DATE)
        assert len(sections) == 1
        assert sections[0].version == "1.0.0"
        assert sections[0].date == ""
        assert len(sections[0].bullets) == 2

    def test_unreleased_kept_in_file_order(self):
        sections = parse_changelog(SAMPLE_UNRELEASED)
        assert [s.version for s in sections] == ["Unreleased", "0.5.0"]

    def test_empty_content(self):
        assert parse_changelog("") == []
        assert parse_changelog("just prose\nno headings\n") == []

    def test_section_without_bullets_dropped(self):
        content = "## [0.6.0] - 2026-10-10\n\n### Added\n\n## [0.5.0] - 2026-09-24\n\n- real\n"
        sections = parse_changelog(content)
        assert [s.version for s in sections] == ["0.5.0"]

    def test_indented_heading_still_parses(self):
        """CommonMark allows 0-3 spaces of indent; the live changelog has
        one 2-space-indented heading."""
        content = (
            "## [0.6.0] - 2026-10-10\n\n- new\n\n  ## [0.5.0] - 2026-09-24\n\n- old\n"
        )
        sections = parse_changelog(content)
        assert [s.version for s in sections] == ["0.6.0", "0.5.0"]
        assert sections[0].bullets == ["new"]
        assert sections[1].bullets == ["old"]


class TestSelectNewSections:
    def test_strictly_newer_than_current(self):
        sections = parse_changelog(SAMPLE)
        chosen = select_new_sections(sections, "0.5.0")
        assert [s.version for s in chosen] == ["0.5.2", "0.5.1"]

    def test_current_version_excluded(self):
        sections = parse_changelog(SAMPLE)
        chosen = select_new_sections(sections, "0.5.1")
        assert [s.version for s in chosen] == ["0.5.2"]

    def test_cap_limits_sections(self):
        sections = parse_changelog(SAMPLE)
        chosen = select_new_sections(sections, "0.4.0", cap=2)
        assert [s.version for s in chosen] == ["0.5.2", "0.5.1"]
        assert len(select_new_sections(sections, "0.1.0", cap=1)) == 1

    def test_current_absent_falls_back_to_newest(self):
        """Very old install whose version predates the changelog."""
        sections = parse_changelog(SAMPLE)
        chosen = select_new_sections(sections, "0.1.9")
        assert [s.version for s in chosen] == ["0.5.2", "0.5.1"]

    def test_unreleased_not_shown_as_newer(self):
        """'Unreleased' does not parse as a version, so never newer."""
        sections = parse_changelog(SAMPLE_UNRELEASED)
        chosen = select_new_sections(sections, "0.4.0")
        assert [s.version for s in chosen] == ["0.5.0"]

    def test_nothing_newer_returns_empty(self):
        sections = parse_changelog(SAMPLE)
        assert select_new_sections(sections, "0.5.2") == []

    def test_dev_version_suffix_tolerated(self):
        """0.5.0.dev0+git parses as (0, 5) - sections above still selected."""
        sections = parse_changelog(SAMPLE)
        chosen = select_new_sections(sections, "0.5.0.dev0")
        assert [s.version for s in chosen] == ["0.5.2", "0.5.1"]


class TestFormatWhatsNew:
    def test_block_layout(self):
        sections = parse_changelog(SAMPLE)
        block = format_whats_new(sections, "0.5.0")
        lines = block.splitlines()
        assert lines[0] == "  What's new since 0.5.0:"
        assert lines[1] == ""
        assert lines[2] == "  v0.5.2  2026-10-01"
        assert lines[3].startswith("    - Remote job polling")
        # Sections separated by a blank line
        assert "  v0.5.1  2026-09-24" in lines

    def test_emphasis_markers_stripped(self):
        block = format_whats_new(parse_changelog(SAMPLE), "0.5.0")
        assert "**" not in block

    def test_bullets_wrapped_with_hanging_indent(self):
        sections = parse_changelog(SAMPLE)
        block = format_whats_new(sections, "0.5.0", cap=1)
        lines = block.splitlines()
        body = [
            ln for ln in lines if ln.startswith("    - ") or ln.startswith("      ")
        ]
        assert all(len(ln) <= 76 for ln in body)
        # Continuation lines align under the bullet text (col 6):
        # 6-space indent, first non-space char not a dash
        cont = [
            ln
            for ln in lines
            if ln.startswith("      -") is False
            and ln.startswith("      ")
            and ln.strip()
        ]
        cont = [ln for ln in cont if not ln.startswith("    - ")]
        assert cont, "long bullet should wrap onto continuation lines"
        assert all(not ln.lstrip().startswith("- ") for ln in cont)

    def test_empty_when_nothing_newer(self):
        assert format_whats_new(parse_changelog(SAMPLE), "0.5.2") == ""

    def test_heading_without_date_omits_date_field(self):
        block = format_whats_new(parse_changelog(SAMPLE_NO_DATE), "0.9.0")
        assert "  v1.0.0\n" in block
        assert "  v1.0.0  " not in block

    def test_no_trailing_blank_line(self):
        block = format_whats_new(parse_changelog(SAMPLE), "0.5.0")
        assert not block.endswith("\n")


class TestSectionDataclass:
    def test_defaults(self):
        s = Section(version="1.2.3", date="", bullets=[])
        assert s.date == ""
        assert s.bullets == []
