---
name: project-folder-structure
disable-model-invocation: true
description: >-
  Audit and improve how a project or workspace is organised: which folders
  exist, what belongs in them, and what must not move. Use when the user asks
  to reorganise, declutter or tidy a repo or workspace, decide where a file
  belongs, or break up a folder that has become a dumping ground. Covers apps,
  libraries, services, data/ML, infrastructure, docs, automation, creative
  assets and monorepos. Audits before editing, then moves in small reversible
  batches. For code quality inside the files use improve; for module shape use
  project-improve-architecture.
license: MIT
metadata:
  version: 1.0.0
  author: Agent13
  created: 2026-11-17
  category: development
  tags:
    - structure
    - audit
    - refactoring
    - workspace
    - organisation
  source: "https://github.com/Gil-1/skills (no licence, ideas only - rewritten)"
  source_path: skills/engineering/project-folder-structure/SKILL.md
  source_note: "Rewritten from the method, not copied: upstream carries no licence (none added in 319 commits), so no text was carried over and no baseline diff is meaningful. The bundled Node scanners were dropped - node is not installed here, and find/rg/git cover the same ground."
---

# Project Folder Structure

Improve how a project is organised so that a maintainer can answer four
questions from the folder names alone: what lives here, who consumes it, what
changes together, and what lifecycle does it have.

This skill is self-contained. Detail lives in `references/` relative to this
skill's base directory — read each file when you reach the phase that needs it,
not up front.

**Audit before you edit. Propose before you move.** A structure change is a
reviewable batch, not a sweep.

## Core rule

Respect the project before applying a pattern. Good organisation makes the
project easier to understand, change, test, operate and archive **while
preserving** framework rules, public contracts, generated paths, persisted
state and local conventions. A tidier tree that breaks a route, a cron job or a
published import is not an improvement.

**Tool-owned directories are not part of the project's organisation.** `.agent13/`,
`.git/`, `.github/`, `.claude/`, `.vscode/`, `.devcontainer/` sit where they sit
because a tool reads them there. Never propose moving one — `.agent13/` holds
this agent's project-local `skills/` and `saves/`, and relocating it breaks
discovery silently.

## Phase 1: Read local truth

Before judging the structure, find out what shaped it. Read, in this order:

1. Agent and project rules: `AGENTS.md`, `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`.
2. Domain language: glossary or context docs, product docs, PRDs, tickets.
3. Decisions: `docs/adr/`, `docs/architecture*`, design notes.
4. Manifests and config: package/workspace files, build and framework config, CI, deploy config.
5. Tests and examples: test layout, fixtures, seed data, golden files.
6. Operational contracts: cron config, Dockerfiles, package exports, CLI entrypoints, generated artifact paths.

Where local docs are silent, prefer the official documentation for the
installed major version of the framework or tool, not a remembered convention.

## Phase 2: Quantify the shape

Read-only, no content inspection of secrets. Adapt the ignore list to the
project.

```sh
# folder tree, 3 deep, noise removed
find . -maxdepth 3 -type d \
  -not -path '*/.git*' -not -path '*/node_modules*' -not -path '*/.venv*' \
  -not -path '*/__pycache__*' -not -path '*/dist*' -not -path '*/build*' \
  | sort

# file count per top-level folder, busiest first
for d in */; do
  printf '%5s  %s\n' "$(find "$d" -type f -not -path '*/.git/*' | wc -l | tr -d ' ')" "$d"
done | sort -rn

# large or generated material sitting in source
find . -type f -not -path '*/.git/*' -size +1M -exec ls -lh {} \; | head -20

# security-sensitive names near the surface
find . -maxdepth 3 \( -name '.env*' -o -name '*secret*' -o -name '*credential*' \
  -o -name '*.pem' -o -name '*.key' \) -not -path '*/.git/*'
```

Record the numbers. "`utils/` has 41 files" is evidence; "`utils/` feels
crowded" is an opinion.

## Phase 3: Diagnose

Two passes, in this order.

**Constraints first.** List what cannot casually move — framework magic paths,
public contracts, runtime contracts, operational contracts, security
boundaries, human habits. These outrank any structural preference, and they
bound what you are allowed to propose. See
[references/axes.md](references/axes.md#constraint-first-audit).

**Then the organising axes.** Name the axis each folder is actually using —
constraint, concept, change pattern, audience, lifecycle, toolbox, layer, or
status. Most projects mix several; the problem is never mixing, it is an
*accidental* mix where one folder silently serves two axes. The full table,
with when each axis is right and how each one fails, is in
[references/axes.md](references/axes.md#organizing-axes).

Apply the cohesion tests to every folder you intend to touch — reader, change,
search, deletion, interface, lifecycle, constraint. They are in
[references/axes.md](references/axes.md#cohesion-tests). A folder that fails the
reader test is a folder whose name is doing no work.

For project-type-specific guidance (app, library, data/ML, infrastructure,
docs, creative, automation, monorepo, embedded), read
[references/project-types.md](references/project-types.md) — only the sections
that match this project.

## Phase 4: Propose

Report before editing, in this format:

```md
## Structure Audit
- Scope scanned:
- Local rules read:
- Immovable constraints:
- Current organising axes:
- Crowded or mixed-lifecycle areas:
- Ambiguous or generic folders:

## Proposed Batch
- Organising axis:
- Move:
- Keep stable:
- Wrappers/redirects:
- Docs/config to update:
- Risks:
- Verification:
```

Use a move table for anything non-trivial — current path, new path, reason,
compatibility. Worked examples for naming, workspace cleanup, generic-folder
breakup, framework-constrained apps and generated output are in
[references/examples.md](references/examples.md).

**Stop and ask** when any of these hold:

- More than one domain, package or workspace area moves at once.
- Public imports, routes, CLI commands, package exports, URLs, cron payloads or persisted paths change.
- The move touches secrets, auth config, deploy config, migrations, database state, backups or large data.
- Local conventions and tool documentation conflict.
- The justification is aesthetic rather than evidence-backed.

## Phase 5: Execute

One batch at a time, smallest useful unit: one crowded folder, one concept, one
lifecycle split, one package boundary.

1. Identify every consumer before moving: imports, package scripts, CLIs, cron, routes, docs links, CI, generated state, public URLs, persisted data.
2. Decide canonical paths and which stable paths must survive.
3. Move implementation first, then tests, docs, assets and generated references.
4. Leave a thin wrapper **only** for a real consumer that cannot move yet — enough code to delegate, one short comment if it would otherwise look accidental.
5. Update references with focused searches.
6. Add an architecture guard (a test or lint rule) when the old shape is likely to return.

**Do not delete by default.** For abandoned material propose `archive/<date>-<reason>/`,
`experiments/`, a `runtime/` path with retention rules, or a quarantine folder
for review. Deletion is a separate decision the user makes.

## Phase 6: Verify

Project-native checks first:

1. Search for old paths and names across imports, scripts, docs, config, tests, CI and generated manifests.
2. Run syntax checks for moved scripts.
3. Run focused tests for the moved area.
4. Run typecheck, lint, build, export checks, route generation or docs build as relevant.
5. Run domain checks: migration validation, CLI `--help`, data pipeline dry-run, infra plan, link check, app smoke test.
6. `git diff --check`.
7. Confirm generated, runtime, cache and temp folders did not land in source.

Report what you ran and what it said. A move without a verification result is
an unverified move.

For a document repo the checks are different: resolve every relative link, grep for
the old paths and names, and recompute every count at the moment it is written — a
number carried over from an earlier pass is usually wrong by the time it lands.

## Red flags

Audit signals, not automatic reasons to move:

- A generic folder with many direct files: `utils`, `helpers`, `common`, `shared`, `services`, `misc`.
- One folder mixing source, tests, fixtures, generated artifacts, docs and temp work.
- A normal feature change cutting across many top-level folders — the axis is probably layer, and probably wrong.
- Tests flat while source has clear module families, or the reverse.
- Docs describing a concept living far from that concept's code.
- Generated output, datasets, notebooks, reports or logs in source paths without a documented reason.
- Local scripts duplicating framework or package-manager behaviour for no project-specific reason.
- `archive`, `legacy`, `deprecated`, `tmp` or `experiments` with no dates, owners or cleanup expectation.
- External repos, forks and client projects mixed into one working directory with active source.
- Security-sensitive names near reports, examples, fixtures or committed artifacts.
