# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.11] - 2026-05-05

### Added

- added extra debugging for journalling issues
- added /prompt tab completion and default prompts

### Changed

- fixed --continue to re-read saved token count
- fixed journalling to detect and preserve loaded skills correctly

## [0.1.10] - 2026-05-03

### Changed

- changed autoupdate to use correctly named wheel in a temp dir
- updated getting started guide
- fix to release script to ensure correct markdown
- fixes and improvements to update system
- modified release script to avoid putting tags on devel branch

## [0.1.9] - 2026-05-03

### Changed

- Fixs to release script
- removed unused user-invocable skill feature and doco
- removed allowed_tools experimental code from skills
- updated changelog generation to be standards compliant
- refined the researcher-deep skill
- added quickstart to user guide
- updated readme with all options
- renamed --prompt-name to --system-prompt
- added clipboard options to select between OSC-52 and system
- added auto update feature using github releases

## [0.1.8] - 2026-04-30

### Changed

- Updated release script and github actions script with pip-audit
- Updated pip in github actions before running audit

### Security

- Added SECURITY.md

## [0.1.6] - 2026-04-30

### Added

- Initial release - no git tag associated with this tag, so no assets built
- Core agent with event-driven architecture
- Textual-based TUI (studio mode)
- Batch mode for one-shot prompts
- Tool system with auto-discovery and tool groups
- Built-in tools: read_file, write_file, edit_file, command, square_number, skill
- Headless mode for debugging
- Configuration via TOML with multiple provider support
- Streaming LLM responses
- prompt_toolkit-compatible history
- Comprehensive test suite
- Debug logging infrastructure
