---
name: retro
disable-model-invocation: true
description: >-
  Conduct a retrospective on a coding session: read the session's own record
  and propose changes to the agent's environment so the next run goes better.
  User-invoked: use when the user types /retro or asks for a retrospective on
  how a session went.
  What: Reads the session record (the current context, then
  `.agent13/saves/*.ctx` and `~/.agent13/debug.log`), hunts for failures in
  navigation, automated checks, coding standards, steering files, tool
  economy, no-op instructions and information access, then presents candidates
  in order of severity. Proposes only; edits nothing.
  When: Use at the end of a session that was harder than it should have been.
  For the code itself use improve; for its shape, project-improve-architecture;
  for what happened today, eod-summary.
license: MIT
metadata:
  version: 1.0.0
  author: "Matt Pocock (original), Agent13 (adaptation)"
  created: 2026-10-08
  source: "https://github.com/mattpocock/skills (MIT, adapted)"
  source_path: skills/engineering/retro/SKILL.md
  source_ref: "a7d038f 2026-09-24"
  invocation: "user-only; agent13 has no disable-model-invocation field, so the restriction lives in the description"
  category: development
  tags: [retrospective, session-review, agent-environment, steering, guardrails]
---

The user has asked for a **retrospective**. You are suggesting improvements to
the coding agent's **environment** so future runs go better — not fixes to the
code the session produced.

Propose only. Nothing changes until the user picks a candidate: no edits, no
hooks installed, no rules written.

## Steps

1. Read the primary sources for the session the user names. If none is named,
   default to the current one — that is the best case, because the struggles are
   still in the context window.

   Agent13 session records, richest first:

   - The current context window.
   - `<project>/.agent13/saves/<date>[_N].ctx` — the auto-save. JSON: full
     `messages` (user, assistant, `tool_calls`, tool results), `token_usage`,
     `chain_count`, `session_started_at`, `focus_name`. One file per session,
     `_N` incrementing within a day. If `[saves] location = "central"` is set in
     config, the same files are at `~/.agent13/saves/<project>-<date>.ctx`.
   - `~/.agent13/debug.log` — JSON lines, one event per line. All sessions on
     the machine are interleaved, delimited by `session_start`. Events of
     interest: `user_message`, `assistant_response`, `tool_call` (name plus full
     arguments), `tool_result`, `api_request`. Rotated to `debug.log.old` at
     10 MB. Parse the JSON and match on `event` — grepping for a name counts
     prose inside tool output, not events. (not all sessions get debug enabled)
   - `~/.agent13/history/history-<project>-<date>` — user prompts only. No
     model output, no tool calls, so silence here is not evidence of non-use.

2. Look for candidates for improvement in these categories.

   - **Navigation**: how easy was it for the agent to find the right files? Are
     there hidden dependencies between files? Would a **navigation pointer**
     make it easier? _Use when_ the session took a long time to find a piece of
     information.
   - **Automated checks**: are there checks that could have caught errors the
     agent made — linting, typing, tests, filesystem linters? Read the repo's
     own check command first (its `package.json` / `pyproject.toml` / `Makefile`
     scripts, its CI workflow), so a check that already exists but sits unwired
     or silently broken is the finding, not a reinvention. A repo with no
     **guardrail** (no pre-commit hook and no CI job running its lint /
     typecheck / test command) is itself a finding: an un-linted repo is a
     standing missed opportunity, not a neutral default. _Use when_ the agent
     made a mistake an automated check could have caught, or the repo has no
     guardrail at all.
   - **Coding standards**: should the review step be given a new rule to
     enforce? Should an existing rule be removed or clarified? Classify the
     violation first: a **mechanical** one (a fixed syntactic pattern, a banned
     API, an import shape, a file-location rule) gets a deterministic check,
     full stop — a custom rule in the repo's own linter, a new pre-commit hook,
     or a new CI job, whichever the repo's language and existing guardrail make
     cheapest. Default to building the check over writing the rule. Reserve
     `CODING_STANDARDS.md` for genuine **judgement calls** (cross-file
     consistency, "matches the surrounding style", anything no guardrail could
     ever substitute for). _Use when_ a review pass failed to catch a mistake.
   - **Steering files**: are there instructions in `AGENTS.md` (read from the
     repo root into every session) or `~/.agent13/prompts.yaml` (global
     overrides) that belong in coding standards, or in an automated check,
     instead? _Use when_ a steering file is particularly large.
   - **Tool economy**: did the agent make expensive tool calls that could be
     streamlined — a whole file read where a search would do, a command re-run
     after its output was truncated, a re-read of material that was already in
     the context? Is any custom tooling (CLIs, MCP servers) token-inefficient?
     _Use when_ the agent made an expensive tool call.
   - **No-ops**: look for instructions in steering files that don't modify the
     agent's behaviour. _Use when_ the steering files are large and unwieldy.
   - **Information access**: look for opportunities to increase the agent's
     access to information. Teeing dev server logs, read-only access to
     third-party services. _Use when_ a crucial piece of information was not
     available to the agent.

3. Present these candidates to the user, in order of severity.

## Rules

- Every candidate must trace to a specific moment in the session record; cite
  it. A candidate you cannot trace is generic advice filling a category — drop
  it.
- The severity order is a first draft. A quiet, expensive failure can rank below
  a loud, cheap one.
- One session is a narrow sample. It over-weights whatever this session happened
  to be about, and it cannot see whether a check proposed last month now fires
  too often. Propose removals as candidates for the deletion test, not verdicts.
- Standards belong to review, not implementation. The implementing agent has the
  most **context pressure**: it explores, writes code, and debugs failures. The
  reviewing agent receives a diff, so it has room to apply a rule. Put the rule
  where there is room to apply it.
- `AGENTS.md` loads into every session's context whether it is relevant or not,
  and every line added takes weight from the rest. It is for **navigation
  pointers** and little else.

## Reference

### Implementation vs review

All work goes through two stages: implementation and review. The implementation
agent has the most context pressure — exploration, writing code, debugging
failures. The review agent has the least: it receives a diff, so no exploration
is needed, and it often writes no code.

This means the review stage should be responsible for imposing coding standards,
not the implementation agent. If the repo has no review step, say so: that is
itself a finding, and the first candidate is to add one (a second session, a
reviewer agent, or a CI check).

### Files

- `AGENTS.md`: pushed into the context of any agent working in this repo (agent13
  reads it from the working directory at startup). Use sparingly — usually only
  for **navigation pointers** to other files.
- `CODING_STANDARDS.md`: read during review, not implementation. No skill ships
  it; propose creating it the first time a judgement-call rule is worth keeping.
  Add navigation pointers to a docs folder once it passes ~1,000 lines. Any
  standards doc the repo already keeps (`CONTRIBUTING.md`) works the same way.
- Docs: use docs as reference files pointed to from elsewhere. Look for existing
  docs before writing new ones.
- Skills: use skills for docs (their description goes into the agent's context
  window), or for user-invoked commands. Follow `manage-skills` for the frontmatter
  contract and `README.md` in the skills collection for house conventions.
