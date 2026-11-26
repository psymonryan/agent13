"""Prompt management for system prompts."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from agent13.config_paths import get_prompts_file, ensure_config_dir
from agent13.truth_format import RERANK_TOKEN, TICKET_HEADING
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
# Fallback nudge: used when the turn carries no captured user instruction
# (e.g. manual /chain while idle).
AUTO_CONTEXT_CONTINUE_HINT = (
    "[Context was compacted to fit. Continue work on the outstanding items.]"
)
# The continuation message injected after a chain restart. ONE text, because
# the restart no longer quotes the user's message: what the model needs is the
# record pointer, plus (when a goal window is armed) the goal it is working
# towards. The recency rule that chose between "carry the user's message" and
# "the record is authoritative" went with the quoting - see
# docs_archive/work_order_2026-11-03_agent13_goal_loop.md §3 for the one
# config-dependent case (action = "compact") that loses its instruction as a
# result, and why that is accepted.
AUTO_CONTEXT_RECORD_FRESH_HINT = (
    "[Context was compacted mid-turn. The durable record is current and it is "
    "authoritative: its section 6 tickets are the work queue (top unblocked "
    "ticket first), and the last journal entry's \"b. NEXT SESSION, FIRST "
    "ACTION\" is the intended next step.]"
)

# --- the goal window (/grind) ----------------------------------------------
#
# Core-internal, like the AUTO_CONTEXT_* cues: not yaml-exposed. The model
# never sees the remaining time - a visible budget invites "I'm running low,
# let me consolidate and hand back", which is the failure the window exists to
# prevent. It sees only the goal text and the step it chose itself.
#
# Three preambles, one instruction: the mechanical ticket check shapes the
# question but never ends the run, because the record it reads is only
# rewritten at a wrap-up and can be hours stale.
GOAL_QUESTION_UNBLOCKED = (
    "Can you make progress without user input? Some tickets may be blocked on "
    "the user - that is fine as long as you can continue with another."
)
GOAL_QUESTION_ALL_BLOCKED = (
    "The record lists no unblocked tickets. Is that still true? Can you make "
    "progress without user input?"
)
GOAL_QUESTION_PLAIN = "Can you make progress without user input?"
# Injected when the last handoff carried unresolved for_user items. The grind's
# own question is the gate: a model that wrote "this is a scope call for you"
# and then answers "can progress: true" is contradicting itself inside one
# turn, and no new flag is needed to force the reconciliation.
GOAL_QUESTION_FOR_USER = (
    "FOR THE USER (unresolved):\n{items}\n\n"
    "Can you make progress without an answer to this? Say no if the remaining "
    "work depends on it; say yes and continue if other work is independent of "
    "it."
)
# `next` is a BARE step because the harness reads it aloud in the continuation
# message. A live answer opened with "Continue rank 1 without user input: ...",
# which reads redundantly after the goal line.
GOAL_QUESTION_INSTRUCTION = (
    "Answer with JSON only, no tool calls:\n"
    '{"can_progress": true|false, "next": "<the step itself, no preamble>", '
    '"waiting_on_user": ["<what you need, if anything>"]}'
)
# The continuation message. No attribution ("you said ...") - the question and
# answer are deleted, so the model has no memory of having said it - and no
# hedge ("which may be that or something better"), which only invites
# meta-reasoning about why the harness is hedging.
#
# The parenthetical is PROVENANCE, and it is on the label rather than in a line
# above it because the model QUOTES this line when it writes the wrap-up's h.
# block: a separate cue is skipped when the line is copied, the parenthetical
# travels with it. Live failure it prevents - edmac 2026-11-06 06:20 recorded
# "RE-RANKED ... by user direction at 06:09" quoting this very message, while
# the history file has no user message before 18:23 that day.
GOAL_ITEM = "Next step (your own proposal, not a user instruction): {step}"
GOAL_ITEM_RECORD_FALLBACK = (
    "The durable record is authoritative: its section 6 tickets are the work "
    "queue (top unblocked first), and the last journal entry's \"b. NEXT "
    "SESSION, FIRST ACTION\" is the intended next step."
)
# Degeneration guard nudge (core-internal, NOT yaml-exposed - same family as
# the AUTO_CONTEXT_* cues). Appended as an injected user message when a
# stream ends at the max-output cap (finish_reason == "length") or on a
# whitespace wall. One universal text by design: the outcome (recovered vs
# trips the stop cap) discriminates cut-off from degeneration, not a
# classifier. docs_archive/degeneration_guard_plan.md.
DEGENERATE_NUDGE_PROMPT = (
    "Your last response was cut off before finishing. Continue exactly "
    "where you stopped, concisely; if your output had become repetitive "
    "or empty, give a concise concrete answer or a tool call instead."
)
# Pre-baked session intro injected after a mechanical chain restart
# (auto-chain post-wrap-up, or /chain). Replaces the AI-generated bridge
# summary: the durable records (truth + journal) carry all the detail,
# so no model call is needed to orient the agent.
CHAIN_RESTART_INTRO = (
    "[Session restarted from durable records after context compaction.]\n"
    "Truth: {truth_file} | Journal: {journal_file}\n"
    "Read the records below."
)
# Appended MECHANICALLY to a verbose compact summary when the durable record
# paths are known (captured from the wrap-up's own file writes). The summary
# is the only carrier into the post-compact context, and the 09-29 games
# incident showed the model-written pointer drops: the verbose summary named
# no record files, the new 28k context carried no pointer, and the agent
# re-struggled ~1 h re-discovering file-resident knowledge. A mechanical
# append is deterministic - the paths are captured facts, not model memory.
# Only the verbose DEFAULT_COMPACT_PROMPT gets the block: a user-specified
# compact prompt wins clean, and the short bridge summary prompt already
# instructs the model to name both files. Core-internal (like the nudge
# cues) - not yaml-exposed.
COMPACT_DURABLE_RECORD_HEADER = "Durable record (focus {focus}):\n"
COMPACT_DURABLE_RECORD_HEADER_BARE = "Durable record:\n"
COMPACT_DURABLE_RECORD_TRUTH_LINE = (
    "- Current truth: {truth_path} - read it in full before starting new"
    " work; its section 6 tickets are the work queue (top unblocked ticket"
    " first).\n"
)
# Index mode: the seed carried a topic index, not the facts, so "read it in
# full" is the wrong instruction - and the index's line numbers are a snapshot
# taken at the last restart, so a heading grep is the stable address.
COMPACT_DURABLE_RECORD_TRUTH_LINE_INDEXED = (
    "- Current truth: {truth_path} - the facts are NOT loaded; read the topics"
    " you need before acting on them (grep '^### <heading>' for the line"
    " number - the index's L numbers are a snapshot and drift after any edit)."
    " Its section 6 tickets are the work queue (top unblocked ticket first).\n"
)
COMPACT_DURABLE_RECORD_JOURNAL_LINE = (
    "- Session journal: {journal_path} - the LAST entry holds the state at"
    " handoff; append to it at the next wrap-up."
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
# ({focus}_current_truth.md); the journal and the truth archive stay
# append-only. The current-truth file holds durable facts only (step 2 gives a
# suggested heading set); position/handoff content belongs to the journal and
# the compaction summary, so the file stays small.
# The size regime is gone (2026-10-27 14:20). Its history, because each step
# looked necessary at the time: a 150-line cap, then a 27,000-char budget, then
# a 38-item / 700-char pair, then a post-wrap-up nudge that rewrote the file
# when it came out too big. Measured across three runs from the same starting
# file: the char budget was met by deleting six verified facts; the nudge's own
# wording ("trim to <= 150 lines and <= 27,000 chars ... delete bullets no
# longer true") triggered 14 whole-file python rewrites and destroyed another;
# the 700-char per-item cap demanded at least ten separate surgeries and the
# model turned that into 26 round-trips, +147k of context growth, and still
# missed. The cap's last justification - that an oversized file truncates the
# reconcile read - does not hold: the post-compaction rebuild injects the file
# whole via DurableRecords._record_parts() (Path.read_text, no bounder), and
# the wrap-up's own read is two calls on a cached prefix. So: one cap on item
# COUNT (navigability), a CONCISION rule for length, and a size check that only
# logs.
# The journal name (records/{focus}_journal.md) is chosen by the model
# and carried across chain restarts in the continuation nudge.
#
# The current date and time is appended to the injected message at injection
# time (core._run_report_and_compact_turn) so the model can write a dated
# heading; the constant itself stays static.
#
# The wrap-up's one cap for the current-truth file: how many TOPICS it holds.
#
# 48 is a NAVIGABILITY bound, not a size bound - a file the model can hold and
# address in one pass. Facts are uncapped: the cap sits on the unit that gets
# fetched (a "### TOPIC" heading), so adding a fact never competes with it.
# That is the fix for the 2026-10-27/28 failure, where the cap sat on the unit
# that HELD the content and the only way to add a fact was to concatenate it
# into an existing line: 2.2x growth over eleven handoffs with the count pinned
# at 38, largest item 12,511 chars, top four items 49% of the file.
#
# There is deliberately no per-topic character cap. A hard per-topic cap plus a
# hard topic cap is a deadlock: at the cap, splitting one makes cap+1 and
# merging two makes an over-size topic, so there is no legal move. A long topic
# costs a longer fetch, never a bigger load.
#
# 38 -> 48 on 2026-11-04, when the placement rule was inverted (a new topic is
# the normal answer to a fact that fits no heading, not a last resort). The cap
# had become the binding constraint: edmac's record sat at 38-39 topics with
# four topics over the 40-line split trigger, so every wrap-up was told to split
# four and merge one. The cap is a BACKSTOP, not a target - if it fires every
# wrap-up, the record has outgrown one file, which is a different problem. The
# topic list is ~11% of the injected index (the ticket section is ~89%), so ten
# more topics cost ~3.5% of the index.
#
# DEFAULT_REPORT_AND_COMPACT_PROMPT states the number literally (it is not an
# f-string - it carries a {focus} placeholder); the drift alarm is a test, not
# interpolation.
TRUTH_TOPIC_CAP = 48

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
    "1. RECONCILE (bounded, <= 6 tool calls). Before recording anything as new:\n"
    "   - Read the current-truth file in full - in as FEW calls as the read cap\n"
    "     allows (30000 chars at a time, not one topic at a time). A fact you did\n"
    "     not read is a fact you cannot correct.\n"
    "   - Name the topics THIS SESSION touched. Those are the ones you will edit.\n"
    "     A topic the session did not touch is INACTIVE, not stale: leave it\n"
    "     alone. Do not rewrite a topic to prove you read it.\n"
    "   - Grep the truth for the 3-5 key terms of this session's findings. A\n"
    "     discovery can falsify a fact filed under a topic you did not touch -\n"
    "     that is the one case where you must edit outside your set.\n"
    "   - Skim the journal (read_file, no offset/limit): the heading map is\n"
    "     your global view of every past session's attempt and outcome. Then\n"
    "     read the LAST entry in full, starting at the line number the skim\n"
    "     gives. Not the whole journal: the last entry.\n"
    "   - If a long project doc exists, grep it for the same key terms.\n"
    "   Anything already documented is a CONFIRMATION with a topic reference,\n"
    "   not a discovery.\n"
    "\n"
    "2. CURRENT TRUTH - update records/{focus}_current_truth.md in\n"
    "   place (create it if absent). This is the ONLY file you may rewrite; the\n"
    "   journal is append-only.\n"
    '   Two levels, one cap. A "### TOPIC" heading is a topic; a "- " line under\n'
    "   it is one fact. The heading is a LABEL, not the first fact - never\n"
    '   restate it in the bullet below: "### netLog" + "- netLog = `~/.../netLog*.log`"\n'
    '   is wrong; write "### netLog" + "- `~/.../netLog*.log`, newest per run."\n'
    "   Every topic heading carries its ADDRESS: \"### <section>.<item> <TOPIC>\",\n"
    '   e.g. "### 3.2 THE OBJECT STATE MACHINE". The address is assigned ONCE,\n'
    "   when the topic is created, and NEVER changes - not on insert, not on\n"
    "   delete, not on reorder. A new topic takes the next free number in its\n"
    "   section (3.16 after 3.15, even if 3.9 was deleted); gaps are expected.\n"
    "   A topic that moves to another section is a new topic with a new number.\n"
    "   Do not renumber the rest of a section to keep the sequence tidy - the\n"
    "   address is a handle the next session loads by, and a shifted number\n"
    "   silently loads the wrong topic. The address is not part of the label.\n"
    "   Cap: <= 48 TOPICS - the only hard cap. Facts are\n"
    "   uncapped: a topic holds as many as it needs, and a fact is one claim.\n"
    "   Verify before you finish:\n"
    f"     awk 'tolower($0) ~ /^## .*{TICKET_HEADING}/{{exit}} /^### /{{n++}} END{{print n+0}}' FILE\n"
    "     grep -c '^- ' FILE       (facts - informational, no cap)\n"
    "   The first command counts topics only: the ticket section's own \"### \"\n"
    "   sub-headings are not topics and must not be counted against the cap.\n"
    "   A topic is the unit that gets fetched on its own, so keep each one\n"
    "   coherent; keep each fact one claim so it can be corrected alone.\n"
    "   CONCISION: sacrifice grammar for the sake of concision. A telegraphic\n"
    "   fact that keeps every exact string, address and qualifier beats a\n"
    "   grammatical one. Never drop a qualifier (\"only when X\", \"except Y\") to\n"
    "   save words - drop the connective tissue instead.\n"
    "   write_file is allowed on this file ONLY if it does not exist; otherwise\n"
    "   edit it with edit_file. A new fact goes under the topic whose HEADING it\n"
    "   fits. When no heading fits, add a new \"### \" topic - do not force the\n"
    "   fact under a heading it does not fit. The cap is a backstop, not a\n"
    "   target: MERGE only when a new topic would take you past it. Merge two\n"
    "   topics a reader would look for TOGETHER and whose facts are the STALEST\n"
    "   - the [observed ...] stamps tell you which. Relatedness keeps the merged\n"
    "   heading honest; staleness keeps the damage small. RENAME the merged topic\n"
    "   so its heading names BOTH sets of facts - the heading is what the index\n"
    "   shows, so one that names half of them is a stub pointing at the wrong\n"
    "   subsystem (their facts stay separate bullets - a merge joins topics,\n"
    "   never content).\n"
    "   Never regenerate this file's text: no write_file (unless creating it),\n"
    "   no heredoc, no python string literal holding its content. edit_file\n"
    "   only.\n"
    "   NEVER DELETE. This file is append-and-correct, not append-only: a fact a\n"
    "   new discovery contradicts is REWRITTEN in place, and the journal's g.\n"
    "   section records what it replaced. Edit only the topics you named in\n"
    "   RECONCILE, plus any topic a grep showed is now falsified.\n"
    "   - BATCH: one edit_file per contiguous block, and issue independent\n"
    "     edits in the same turn. Do not re-read a file you have just edited;\n"
    "     verify once at the end with the count command.\n"
    "   Durable facts only: what is true about this system regardless of which\n"
    "   session you are. Not where we are in the work - that is the journal's\n"
    "   job. The next session reads this first.\n"
    '   No session-relative markers: no "NEW", no "this session", no "added\n'
    '   today". Every line must read correctly to someone arriving next week\n'
    "   who never saw this session. Dates as evidence provenance\n"
    '   ("[observed 2026-09-23]") are fine; dates as novelty labels are not.\n'
    "   Exact format - a single title, no opening paragraph, then sections. Omit\n"
    "   any section that would be empty. Under sections 1-5 the unit is a topic:\n"
    "     # {focus} - current truth (updated YYYY-MM-DD HH:MM)\n"
    "     ## 1. Environment & access   (where it runs, how to reach it, credentials,\n"
    "          standing constraints: never modify X)\n"
    "     ### <s>.<n> <TOPIC - a short noun phrase, greppable>\n"
    "     - <one claim>   [source ref, provenance]\n"
    "     - <one claim>\n"
    "     ## 2. Conventions & notation (units, identifiers, encodings, how to read\n"
    "          the values you will see)\n"
    "     ### <s>.<n> <TOPIC>\n"
    "     - <one claim>   [source ref, provenance]\n"
    "     ## 3. Findings               (verified facts about the system)\n"
    "     ### <s>.<n> <TOPIC>\n"
    "     - <one claim>   [source ref, provenance]\n"
    "     ## 4. Tooling & traps        (how to drive the tooling; what wastes calls\n"
    "          or bites)\n"
    "     ### <s>.<n> <TOPIC>\n"
    "     - <one claim>   [source ref, provenance]\n"
    "     ## 5. Artifacts              (files you depend on)\n"
    "     ### <s>.<n> <TOPIC>\n"
    "     - <one claim>   [source ref, provenance]\n"
    "     ## 6. Open tickets          (the ONLY positional section here - it\n"
    "          changes every entry; everything above is durable fact. Tickets are\n"
    "          NOT topics - this section stays a flat \"- \" list)\n"
    "     - <ticket> - FIRST SEEN YYYY-MM-DD HH:MM [- BLOCKED: <what it waits\n"
    "       on>] [- NEEDS USER]\n"
    "          Rank IS position: the first bullet is rank 1. Do not write\n"
    "          (RANK n) markers - a marker and a position can disagree.\n"
    "     ### Parked                  (not ranked; one line each)\n"
    "     - <ticket> - parked YYYY-MM-DD: <reason>\n"
    "   ONE FACT PER BULLET. A bullet states one claim. If a bullet needs a\n"
    "   semicolon to carry a second claim, it is two bullets. A topic over ~40\n"
    "   lines is a split candidate at the next wrap-up: the index prints every\n"
    "   topic's line count, so the trigger is visible at every restart. There is\n"
    "   NO hard cap on a topic - a long topic costs a longer fetch, never a\n"
    "   bigger load - but past ~40 lines the reader stops before the end, so\n"
    "   split it.\n"
    "   Rewrite every ticket, every time: ranked most-blocking first, capped\n"
    "   at 10 - an 11th forces one out: CLOSE it (resolved), KILL it (ruled\n"
    "   out), or PARK it (set aside), with the reason recorded in the journal.\n"
    "   FIRST SEEN: for every ticket that survives this rewrite, copy its\n"
    "   stamp verbatim from the truths file you read in RECONCILE - not\n"
    "   from memory, and never coarsened to date-only.\n"
    "   The TOP UNBLOCKED ticket becomes this entry's \"b. NEXT SESSION, FIRST\n"
    '   ACTION"; if you cannot work it, mark it BLOCKED and name what unblocks\n'
    "   it - a block is a state, not a failure. NEEDS USER tickets wait for\n"
    "   the user; never answer them yourself.\n"
    "\n"
    "3. APPEND the journal entry to records/{focus}_journal.md (append\n"
    "   only; create if absent) with edit_file mode=\"append\", no find - never\n"
    "   the shell (no >>, no heredoc, no tee: that path skips the snapshot,\n"
    "   the path check and the diff, so a wrong path or a truncated entry is\n"
    "   silent). If an edit_file call fails, fix the arguments and retry it.\n"
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
    "      that proves it worked. If the user agreed a plan or a priority this\n"
    "      session, this is that agreement's first step.\n"
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
    "      FIRST SEEN), tickets CLOSED / KILLED / PARKED (with the reason),\n"
    f"      tickets {RERANK_TOKEN} (old rank -> new rank, with the reason).\n"
    "      A re-rank is a change, not a rationale. If the user agreed a plan or\n"
    "      a priority this session, quote that agreement (time + the ranking it\n"
    "      fixed) and, for each rank that differs, name the message that\n"
    "      supersedes it. The goal window's \"Next step\" messages are the\n"
    "      harness echoing YOUR OWN proposal - they are not user direction, and\n"
    "      citing one as such writes a false provenance record. A rank change\n"
    "      you made on your own judgement is written \"by my own judgement:\n"
    "      <reason>\". A rank you can cite neither way is a rank you must not\n"
    "      change.\n"
    "      The entry that CREATES section 6 records its seed tickets as ADDED;\n"
    "      every later PARKED move lands here too, with the reason.\n"
    '      No delta: write "- none" and nothing else.\n'
    "\n"
    "4. THEN, as the very last thing in your final message and after the prose\n"
    "   summary, a fenced JSON block with exactly two fields:\n"
    "     ```json\n"
    '     {"load": ["3.2", "3.7"], "for_user": ["<one line>"]}\n'
    "     ```\n"
    "   load: the topic ADDRESSES the next session needs in order to work the\n"
    "   b. action, in priority order, at most 6, copied exactly from the index\n"
    "   you read. An address that does not exist is dropped, so copy, do not\n"
    "   reconstruct.\n"
    "   for_user: what the user must DO or DECIDE, one line each, at most 3 - a\n"
    "   decision needed, a warning that changes what they do, a correction of\n"
    "   an assumption they are acting on. NOT a status report, NOT your plan,\n"
    "   NOT your next step: those already have a., b. and c., and an item that\n"
    "   repeats the b. action is a copy, not a message. [] when there is\n"
    "   nothing. CARRY FORWARD: an item from the previous entry that is still\n"
    "   unanswered is repeated VERBATIM, with the date it was first raised,\n"
    "   until it is answered, withdrawn or superseded - an empty list is a\n"
    "   claim you are making deliberately, not by omission.\n"
    "   Nothing else goes in this block: facts, decisions and state stay in the\n"
    "   a-h sections above. The block is captured mechanically and appended to\n"
    "   this entry, so it is the one thing that must NOT be left only in prose.\n"
    "\n"
    "Name the journal after the FOCUS (the work target), not this session's\n"
    "sub-task; put the sub-task in the entry header. The focus name: if the\n"
    "user named the focus in any message, use that name for {focus} (both\n"
    "files and the truth title); only with no name given, fall back to the\n"
    "directory name.\n"
    "No cumulative version history. No prose\n"
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
    "- The focus's durable record \u2014 the journal and current-truth files\n"
    "  under records/ (e.g. <focus>_current_truth.md). Always name\n"
    "  them, even if untouched this session, with a cue on when to read them.\n"
    "  If the user named the focus, carry that name - it determines the\n"
    "  file names when the wrap-up creates them.\n"
    "  Mid-session journal appends (user-requested, or to make a hard-won\n"
    "  finding survive a crash) are fine: same dated heading with terse\n"
    '  outcome, plain "- " bullets, NO a-h sections - those belong to the\n'
    "  wrap-up entry only. Append with edit_file mode=\"append\", never the\n"
    "  shell - no >>, no heredoc, no tee\n"
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

# Short-form summary for the AUTO compact that follows a report-and-compact
# wrap-up turn: the wrap-up just wrote the journal entry and rewrote the
# current-truth file, so the summary only needs to bridge the next chain to
# those files. The verbose DEFAULT_COMPACT_PROMPT remains in use for
# action = "compact" and for /compact, where no wrap-up ran and the summary
# is the ONLY carrier of session state. Core-internal constant (like the
# nudge cues) - not yaml-exposed; overriding "compaction" in prompts.yaml
# still controls /compact and action = "compact".
DEFAULT_REPORT_AND_COMPACT_SUMMARY_PROMPT = (
    "The session wrap-up just ran: it appended the journal entry and\n"
    "rewrote the current-truth file. Write a SHORT bridge summary\n"
    "(<= 300 words) - the files carry the details.\n"
    "Cover ONLY:\n"
    "- The two file paths as written this turn, each with its cue:\n"
    "  current-truth file -> read in full before new work (its section 6\n"
    "  open tickets are the work queue); journal -> append at the next\n"
    "  wrap-up; the LAST entry holds the state at handoff.\n"
    "- The TOP unblocked ticket from truth section 6 - one line.\n"
    "- In-flight state the files might not show (processes left running,\n"
    "  half-done edits) - only if real, one or two lines.\n"
    "Do not restate file contents, decisions, or history - already\n"
    "recorded. Do not use tools - write from this conversation alone."
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

# The prompts the shipped yaml does NOT carry. They are written into the
# user's prompts.yaml from these constants on first run, so each has exactly
# ONE copy in the repo. A second copy in the yaml used to drift, and the drift
# was behavioural, not cosmetic: core.compact_history decides whether to
# append the durable-record pointer block by comparing the resolved compaction
# prompt to DEFAULT_COMPACT_PROMPT BY VALUE, so a drifted `compaction` silently
# dropped the block for every fresh install.
GENERATED_PROMPTS = {
    "default": DEFAULT_PROMPT,
    "report_and_compact": DEFAULT_REPORT_AND_COMPACT_PROMPT,
    "compaction": DEFAULT_COMPACT_PROMPT,
}


def _render_generated_prompts() -> str:
    """The generated keys as YAML block scalars, for the user's prompts file.

    Each value is dumped on its own with ``default_style="|"``: dumping the
    whole mapping that way also forces the STYLE onto the keys, which quotes
    them ("report_and_compact": |-). The user is meant to open this file and
    edit the text, so the prompts have to land as readable block scalars
    rather than double-quoted strings full of \\n escapes.
    """
    return "".join(
        f"{key}: "
        + yaml.safe_dump(text, default_style="|", allow_unicode=True, width=4096)
        for key, text in GENERATED_PROMPTS.items()
    )


if TYPE_CHECKING:
    from agent13.skills import SkillInfo


def ensure_default_prompts() -> None:
    """Write the starter prompts file for a new user.

    The shipped yaml carries only the hand-written prompts; GENERATED_PROMPTS
    is appended from the constants, so the repo holds one copy of each prompt
    rather than two. An existing file is never touched - the user's edits
    always win.
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

    try:
        shipped = DEFAULT_PROMPTS_FILE.read_text(encoding="utf-8").rstrip("\n")
        prompts_file.write_text(
            shipped + "\n\n" + _render_generated_prompts(), encoding="utf-8"
        )
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
        if info.disable_model_invocation:
            # Hidden from the model's list; still loadable via /name or
            # skill(name). Active-skill re-injection is unaffected.
            continue
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{info.description}</description>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)
