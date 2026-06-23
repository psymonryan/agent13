# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-23

### Changed

- windows update fix
- changed `/sandbox none` to `/dandbox off` for consistency
- changed /delete command to use python slice format
- added cwd to system prompt and fixed local save names

## [0.2.0] - 2026-06-22

### Added

- added --read parameter to allow reading files into user message from command line
- @filename.txt now inserts file into context
- REPL mode (--repl) with optional --output file for screen reader support
- allowed alias name to provider number so `/provider 6:nothink` works
- lots of new tests to improve coverage
- load and save support for filenames with full path and extensions
- rollback to write_file tool and normalised rollback paths for all edit tools
- /cwd command

### Changed

- fix for @ expansion to correctly handle tilda in path
- fix for path suffix bug when filename specified including extension .ctx
- fix for fallback save location when swapping save modes
- fixed command timeout messages
- fix for word count issues when journaling
- fix Windows drive-letter paths in @filename expansion
- fix CR detection in _is_probably_text and clean up test imports
- fix to sanitise tool call arguments before adding to message history
- general code refactor and DRY fixes for repl mode
- fix for pause and !!messages not correctly setting work_started
- fix for streaming widget early finalisation
- fix for when removing empty reasoning widgets
- fix for a snapshot bug
- tidied up /history display truncation and newline issues
- fix for tools not showing when other content present in assistant response
- changed default save location to under local directory (added config parameter)
- improved UX message sequencing for when pressing ESC
- moved tools status display into devel mode
- fix for spurious cancellations
- fix for chat window disconnecting if error inside _process_tokens
- trimmed AGENTS.md
- changed skill licenses to MIT

### Removed

- removed limit on number of allowed tab completions
- removed redundant commands table from ARCHITECTURE.md

## [0.1.13] - 2026-05-19

### Changed

- added Textual screen shot to doco
- reverted the code that pauses after error
- added previous turn count to /status for loaded sessions
- changes to dur: in status bar to remain as last: when turn ends
- added /status command
- tps calculation finetune to avoid inaccurate assessment for short messages
- updated AGENTS.md with information about ./utils/analyse-debug.py
- fixed reasoning widget collapse to only trigger on header click
- fixed journal loop issue when skills loaded
- Readme fixes and tweak to release script
- fix tab completion for names with spaces
- fix for tab completion regression

## [0.1.12] - 2026-05-13

### Changed

- updated libaries
- manual improvements to README.md and USER_GUIDE.md
- renamed researcher-deep skill to deep-researcher
- further fixes to update code
- fixed windows inplace update to fix permissions issue

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
- refined the deep-researcher skill
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
