"""Prompt management for system prompts."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from agent13.config_paths import get_prompts_file, ensure_config_dir
from agent13.yaml_store import load_yaml, save_yaml

DEFAULT_PROMPT = "You are a tool using AI assistant."

REFLECTION_PROMPT = (
    "Since you have just used tools, tersely reflect on each one, then stop.\n"
    "- what was your goal when calling the tools\n"
    "- what did you achieve with these calls\n"
    "Skip where the goal was not achieved"
)

PRIMING_PROMPT = (
    "The previous turns have been journalled to reduce context. "
    "From now on, actually call the tools rather than reflecting. Dont reflect again until asked. "
    "Acknowledge this imperative with an `ok`"
)

PRIMING_RESPONSE = "ok"

# Injected as a user message after an auto-context compact so the model resumes
# the in-progress turn (finishing the original task) instead of stopping.
AUTO_CONTEXT_CONTINUE_HINT = (
    "[Context was compacted to fit. Continue work on the outstanding items.]"
)
# Appended by core._run_auto_context_post_turn when the wrap-up captured the
# paths: the current-truth file (read-first cue; its section 6 open tickets
# are the work queue) and the session journal (append at the next wrap-up).
AUTO_CONTEXT_TRUTH_CUE = (
    " Current truth: {truth_path} - read it in full before starting new"
    " work; its section 6 tickets are the work queue (top unblocked"
    " ticket first)."
)
AUTO_CONTEXT_JOURNAL_CUE = (
    " Session journal: {journal_path} - append to it at the next wrap-up."
)

# Injected as a user message when action = "report_and_compact" and the
# auto-context threshold is reached. Instead of compacting immediately, the
# model is asked to append a dated journal entry while it still has full
# context. The agent then idles (chain=0) or continues the turn (chain>0);
# under chain>0 this may fire once per chain cycle (up to chain+1 times in
# one turn).
#
# The wrap-up burns context on tool results, and reading large doc files was
# the observed runaway (40-60k-char reads per call). So reads are bounded
# (Reconcile, <= 5 calls) and exactly one small file is rewritable
# ({project}_current_truth.md, <= 150 lines); the journal stays append-only.
# The current-truth file holds durable facts only (step 2 gives a suggested
# heading set); position/handoff content belongs to the journal and the
# compaction summary, so the file stays small and rarely needs rewriting.
# The journal name (docs_archive/{project}_journal.md) is chosen by the model
# and carried across chain restarts in the continuation nudge.
#
# The current date and time is appended to the injected message at injection
# time (core._run_report_and_compact_turn) so the model can write a dated
# heading; the constant itself stays static.
#
# Users can override by adding a "report_and_compact" entry to prompts.yaml.
# The built-in default below is what existing users get: ensure_default_prompts()
# never overwrites an existing prompts.yaml, so the YAML entry only reaches
# fresh installs.
DEFAULT_REPORT_AND_COMPACT_PROMPT = (
    "CONTEXT IS NEARLY FULL. Stop investigating. Write the handoff, in this order.\n"
    "\n"
    "SETTLE FIRST (no new work): leave in-progress work in a clean state; note\n"
    "anything left running or half-done; verify anything you claim is unchanged\n"
    "rather than assuming it.\n"
    "\n"
    "1. RECONCILE (bounded, <= 5 tool calls). Before recording anything as new:\n"
    "   - Read the current-truth file in full. You are about to rewrite it;\n"
    "     every line you keep must pass through your eyes.\n"
    "   - Skim the journal (read_file, no offset/limit): the heading map is\n"
    "     your global view of every past session's attempt and outcome. Then\n"
    "     read the LAST entry in full, starting at the line number the skim\n"
    "     gives. Not the whole journal: the last entry.\n"
    "   - If a long project doc exists, grep it for the 3-5 key terms of this\n"
    "     session's findings.\n"
    "   Anything already documented is a CONFIRMATION with a section reference,\n"
    "   not a discovery.\n"
    "\n"
    "2. CURRENT TRUTH - rewrite docs_archive/{project}_current_truth.md (<= 150\n"
    "   lines; create if absent). The ONLY file you may rewrite. Durable facts\n"
    "   only: what is true about this system regardless of which session you\n"
    "   are. Not where we are in the work - that is the journal's job. Delete\n"
    "   bullets no longer true. The next session reads this first.\n"
    '   No session-relative markers: no "NEW", no "this session", no "added\n'
    '   today". Every line must read correctly to someone arriving next week\n'
    "   who never saw this session. Dates as evidence provenance\n"
    '   ("[observed 2026-09-23]") are fine; dates as novelty labels are not.\n'
    "   Exact format - a single title, no opening paragraph, then sections. Omit\n"
    "   any section that would be empty; add sub-headings for detail:\n"
    "     # {project} - current truth (updated YYYY-MM-DD HH:MM)\n"
    "     ## 1. Environment & access   (where it runs, how to reach it, credentials,\n"
    "          standing constraints: never modify X)\n"
    "     - <statement>   [source ref, provenance]\n"
    "     ## 2. Conventions & notation (units, identifiers, encodings, how to read\n"
    "          the values you will see)\n"
    "     - <statement>   [source ref, provenance]\n"
    "     ## 3. Findings               (verified facts about the system; sub-heading\n"
    "          by topic)\n"
    "     - <statement>   [source ref, provenance]\n"
    "     ## 4. Tooling & traps        (how to drive the tooling; what wastes calls\n"
    "          or bites)\n"
    "     - <statement>   [source ref, provenance]\n"
    "     ## 5. Artifacts              (files you depend on)\n"
    "     - <statement>   [source ref, provenance]\n"
    "     ## 6. Open tickets          (the ONLY positional section here - it\n"
    "          changes every entry; everything above is durable fact)\n"
    "     - <ticket> - FIRST SEEN YYYY-MM-DD HH:MM [- BLOCKED: <what it waits\n"
    "       on>] [- NEEDS USER]\n"
    "     ### Parked                  (not ranked; one line each)\n"
    "     - <ticket> - parked YYYY-MM-DD: <reason>\n"
    "   Rewrite every ticket, every time: ranked most-blocking first, capped\n"
    "   at 10 - an 11th forces one out: CLOSE it (resolved), KILL it (ruled\n"
    "   out), or PARK it (set aside), with the reason recorded in the journal.\n"
    "   Keep each FIRST SEEN exactly as first written, to the minute - never\n"
    "   coarsen it to date-only.\n"
    "   The TOP UNBLOCKED ticket becomes this entry's \"b. NEXT SESSION, FIRST\n"
    '   ACTION"; if you cannot work it, mark it BLOCKED and name what unblocks\n'
    "   it - a block is a state, not a failure. NEEDS USER tickets wait for\n"
    "   the user; never answer them yourself.\n"
    "\n"
    "3. APPEND the journal entry to docs_archive/{project}_journal.md (append\n"
    "   only; create if absent.\n"
    "   Format: one dated heading (## YYYY-MM-DD HH:MM - {sub-task} - {terse\n"
    "   outcome}; one line, <= 15 words after the date - the skim view of\n"
    '   this journal is the project\'s map), then a bold "**<letter>. <NAME>**"\n'
    "   line per section below, in this order so truncation loses the least\n"
    "   (bold lines, NOT ### headings - ### would flood the skim view with\n"
    "   the same 8 subheadings in every entry). Every list item starts with\n"
    '   "- " - never a numbered list, never an unbulleted paragraph. Exactly\n'
    "   one blank line between sections and between entries - never two.\n"
    "   a. GOAL STATUS - one line per project end-goal: DONE / BLOCKED / NOT\n"
    "      TRIED. Then: the cheapest next step that could move a NOT-TRIED goal,\n"
    "      and whether you ran it. If not, one sentence saying why.\n"
    "   b. NEXT SESSION, FIRST ACTION - one imperative sentence that works the\n"
    "      TOP UNBLOCKED open ticket (truths file section 6), plus the check\n"
    "      that proves it worked.\n"
    "   c. STATE AT HANDOFF - anything left running/in progress; exact versions\n"
    "      or hashes of artifacts you depend on; any environment state that must\n"
    "      persist.\n"
    "   d. DERIVED VALUES - any computed/derived value you state must show the\n"
    "      arithmetic or command that produced it. No unstated mental math.\n"
    "   e. DECISIONS + RATIONALE - including what you chose NOT to do, and why.\n"
    "   f. NEW KNOWLEDGE - each item tagged with its provenance (observed /\n"
    "      inferred / assumed) and its evidence (log line, file, command output).\n"
    "      A negative result must name the method that produced it, and a\n"
    "      negative from one method does not override a positive finding from a\n"
    "      more authoritative method.\n"
    "   g. SUPERSEDES / CONTRADICTS - name the prior claim you are overriding\n"
    "      (date + section). If two existing claims disagree, resolve it or park\n"
    "      it explicitly; never leave both standing.\n"
    "   h. TICKET CHANGES - the open-ticket list lives in the truths file\n"
    "      (section 6), which you have just rewritten; do NOT restate it here.\n"
    "      Record only this entry's delta, one line each: tickets ADDED (with\n"
    "      FIRST SEEN), tickets CLOSED / KILLED / PARKED (with the reason).\n"
    "      The entry that CREATES section 6 records its seed tickets as ADDED;\n"
    "      every later PARKED move lands here too, with the reason.\n"
    '      No delta: write "- none" and nothing else.\n'
    "\n"
    "Name the journal after the PROJECT, not this session's sub-task; put the\n"
    "sub-task in the entry header. No cumulative version history. No prose\n"
    "restatement of prior entries. Keep it tight - a cold reader must be able to\n"
    "act on it.\n"
    "\n"
    "Never leave anything important only in the conversation.\n"
)


def resolve_report_and_compact_prompt(prompt_manager) -> str:
    """Resolve the report-and-compact prompt from prompts.yaml.

    Falls back to DEFAULT_REPORT_AND_COMPACT_PROMPT when the prompt manager is
    absent or the user's prompts.yaml predates this feature (which is the
    common case — ensure_default_prompts() never overwrites an existing file).

    Args:
        prompt_manager: PromptManager, or None.

    Returns:
        The prompt text to inject as a user message.
    """
    if prompt_manager is None:
        return DEFAULT_REPORT_AND_COMPACT_PROMPT
    return prompt_manager.prompts.get(
        "report_and_compact", DEFAULT_REPORT_AND_COMPACT_PROMPT
    )


JOURNAL_USER_MESSAGE_PREFIX = "[previous user message]"
JOURNAL_USER_MESSAGE = f'{JOURNAL_USER_MESSAGE_PREFIX} "{{original}}"'

# Default compaction prompt for /compact command.
# Users can override by adding a "compaction" entry to prompts.yaml
# or by passing a named prompt: /compact --prompt <name>
DEFAULT_COMPACT_PROMPT = (
    "Summarize our conversation so far into a concise but complete context summary.\n"
    "Preserve:\n"
    "- All key decisions, their rationale, and current status\n"
    "- Important code, file paths, and technical details\n"
    "- The project's durable record \u2014 the journal and current-truth files\n"
    "  under docs_archive/ (e.g. <project>_current_truth.md). Always name them,\n"
    "  even if untouched this session, with a cue on when to read them.\n"
    "  Mid-session journal appends (user-requested, or to make a hard-won\n"
    "  finding survive a crash) are fine: same dated heading with terse\n"
    '  outcome, plain "- " bullets, NO a-h sections - those belong to the\n'
    "  wrap-up entry only\n"
    "- New knowledge gained by doing: how to connect to hosts/services,\n"
    "  exact commands that work, environment quirks and gotchas\n"
    "- Open questions and unresolved issues - the full ticket ledger lives in\n"
    "  the current-truth file (section 6); do NOT copy the list into this\n"
    "  summary. Carry only the TOP unblocked ticket - it is the next chain's\n"
    "  first work item.\n"
    "- The current direction/next steps\n"
    "Skip pleasantries and hedging. Write as a direct reference document\n"
    "that lets you continue the work seamlessly.\n"
    "If you'd have to re-discover it from scratch, it belongs in this summary."
)

# Appended to the base compaction prompt when /compact is given a focus
# ("next task") string. Steers the summary sections and next steps toward
# the upcoming work and allows harder compression of unrelated detail.
COMPACT_STEERING_TEMPLATE = (
    "\nNext task: {focus}\n"
    "- Organize the summary sections around that task\n"
    '- Make "next steps" concrete actions toward it\n'
    "- Details clearly unrelated to it can be compressed harder"
)


def resolve_compact_prompt(prompt_manager, arg: str) -> tuple:
    """Resolve the /compact argument into the compaction prompt to send.

    Shared by the REPL, headless, and TUI command handlers so all
    interfaces accept the same syntax:

    - (no arg)        → the 'compaction' prompt from prompts.yaml, or
                        DEFAULT_COMPACT_PROMPT if absent
    - --prompt <name> → swap in a named prompt (existing behavior)
    - <free text>     → base prompt + steering block focusing the summary
                        on the user's next task

    Args:
        prompt_manager: PromptManager for prompt lookups.
        arg: Raw argument text after /compact (may be empty).

    Returns:
        (prompt_text, error) — exactly one is None.
    """
    arg = arg.strip()
    if arg.startswith("--prompt"):
        prompt_name = arg[len("--prompt") :].strip()
        if not prompt_name:
            return None, (
                "Usage: /compact --prompt <name>\n"
                f"Available: {', '.join(prompt_manager.prompts.keys())}"
            )
        candidate = prompt_manager.get_prompt(prompt_name)
        if (
            candidate == prompt_manager.get_prompt("default")
            and prompt_name != "default"
        ):
            return None, (
                f"Prompt '{prompt_name}' not found\n"
                f"Available: {', '.join(prompt_manager.prompts.keys())}"
            )
        return candidate, None
    base = prompt_manager.prompts.get("compaction", DEFAULT_COMPACT_PROMPT)
    if arg:
        return base + COMPACT_STEERING_TEMPLATE.format(focus=arg), None
    return base, None


# The lightweight user message that replaces the full compaction prompt
# in history after compaction. Small so it doesn't re-bloat context.
COMPACT_REPLACEMENT_MESSAGE = "Give me a summary of our previous session"

# Default prompts bundled with the package
DEFAULT_PROMPTS_FILE = Path(__file__).parent / "default_prompts.yaml"

if TYPE_CHECKING:
    from agent13.skills import SkillInfo


def ensure_default_prompts() -> None:
    """Copy default prompts to user's config directory if they don't exist.

    This provides starter prompts for new users.
    """
    prompts_file = get_prompts_file()

    # If prompts already exist, don't overwrite
    if prompts_file.exists():
        return

    # Check if we have a default prompts file to copy
    if not DEFAULT_PROMPTS_FILE.exists():
        return

    # Ensure config directory exists
    ensure_config_dir()

    # Copy default prompts
    try:
        prompts_file.write_text(DEFAULT_PROMPTS_FILE.read_text())
    except OSError as e:
        # Log warning but don't fail
        import logging

        logging.getLogger(__name__).warning("Failed to copy default prompts: %s", e)


class PromptManager:
    """Manages system prompts stored in ~/.agent13/prompts.yaml

    Prompts are stored in YAML format with prompt names as keys.
    The active prompt is used for system messages in conversations.
    """

    def __init__(self, config_path: str = None):
        """Initialize prompt manager.

        Args:
            config_path: Path to prompts YAML file (defaults to ~/.agent13/prompts.yaml).
        """
        self.config_path = Path(config_path) if config_path else get_prompts_file()
        self.prompts: dict[str, str] = {}
        self.active_prompt: str = "default"
        self.custom_additions: list[str] = []
        self.load_prompts()

    def load_prompts(self) -> None:
        """Load prompts from config file.

        Raises:
            yaml.YAMLError: If prompts file exists but is invalid YAML.
            ValueError: If prompts file exists but has wrong structure
                or contains non-string values.
        """
        self.prompts = load_yaml(self.config_path)
        # load_yaml already validates top-level is a dict (or missing → {})
        # Now strictly validate all values are strings
        for key, value in self.prompts.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"Prompt '{key}' in {self.config_path} must be a "
                    f"string, got {type(value).__name__}"
                )

        # Ensure default exists
        if "default" not in self.prompts:
            self.prompts["default"] = DEFAULT_PROMPT

    def save_prompts(self) -> None:
        """Save prompts to config file."""
        save_yaml(self.config_path, self.prompts)

    def get_prompt(self, name: str = None) -> str:
        """Get a prompt by name, or the active prompt.

        Args:
            name: Prompt name, or None for active prompt.

        Returns:
            The prompt content.
        """
        name = name or self.active_prompt
        return self.prompts.get(name, self.prompts.get("default", DEFAULT_PROMPT))

    def set_active(self, name: str) -> bool:
        """Set the active prompt.

        Args:
            name: Name of the prompt to activate.

        Returns:
            True if the prompt exists and was activated.
        """
        if name in self.prompts:
            self.active_prompt = name
            return True
        return False

    def add_prompt(self, name: str, content: str) -> None:
        """Add or update a prompt.

        Args:
            name: Prompt name.
            content: Prompt content.
        """
        self.prompts[name] = content
        self.save_prompts()

    def delete_prompt(self, name: str) -> bool:
        """Delete a prompt.

        Args:
            name: Name of the prompt to delete.

        Returns:
            True if the prompt was deleted.
        """
        if name in self.prompts and name != "default":
            del self.prompts[name]
            if self.active_prompt == name:
                self.active_prompt = "default"
            self.save_prompts()
            return True
        return False

    def append_to_active(self, addition: str) -> None:
        """Add temporary content to the active prompt.

        Args:
            addition: Text to append to the system message.
        """
        self.custom_additions.append(addition)

    def clear_additions(self) -> None:
        """Clear temporary prompt additions."""
        self.custom_additions.clear()

    def build_system_message(self) -> str:
        """Build the complete system message.

        Returns:
            The active prompt with any custom additions.
        """
        base = self.get_prompt()
        if self.custom_additions:
            additions = "\n\n".join(self.custom_additions)
            return f"{base}\n\n{additions}"
        return base

    def list_prompts(self) -> list[dict]:
        """List all available prompts.

        Returns:
            List of dicts with name, active status, and preview.
        """
        return [
            {
                "name": name,
                "active": name == self.active_prompt,
                "preview": content[:100] + "..." if len(content) > 100 else content,
            }
            for name, content in self.prompts.items()
        ]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromptManager(path={self.config_path!r}, prompts={len(self.prompts)}, active={self.active_prompt!r})"


def get_skills_section(skills: dict[str, "SkillInfo"]) -> str:
    """Generate the skills section for the system prompt.

    Args:
        skills: Dictionary of skill name to SkillInfo

    Returns:
        Formatted skills section string, or empty string if no skills
    """
    if not skills:
        return ""

    lines = [
        "# Available Skills",
        "",
        "You have access to skills for specialized workflows. When a task matches",
        "a skill's description, use the `skill` tool to load its full instructions.",
        "",
        "<available_skills>",
    ]

    for name, info in sorted(skills.items()):
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{info.description}</description>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)
