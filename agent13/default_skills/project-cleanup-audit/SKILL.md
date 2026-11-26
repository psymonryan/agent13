---
name: project-cleanup-audit
disable-model-invocation: true
description: >-
  Find what can be deleted, merged, renamed or archived in a project before
  anyone touches it. Use when the user asks to simplify a repo or workspace,
  remove dead code or stale material, hunt legacy or unused files, deduplicate
  documents or functions, or research cleanup opportunities. Audits and
  reports; edits only when asked. For where things live use
  project-folder-structure; for code quality inside the files use improve;
  when a cleanup finding turns out to be a module-shape problem, hand it to
  project-improve-architecture.
license: MIT
metadata:
  version: 1.0.0
  author: Agent13
  created: 2026-11-17
  category: development
  tags:
    - cleanup
    - dead-code
    - audit
    - duplication
    - refactoring
  source: "https://github.com/Gil-1/skills (no licence, ideas only - rewritten)"
  source_path: skills/engineering/codebase-cleanup-audit/SKILL.md
  source_note: "Rewritten from the method, not copied: upstream carries no licence (none added in 319 commits), so no text was carried over and no baseline diff is meaningful. The bundled Node scanners (~1,200 lines across 8 files) were dropped: node is not installed here, and git grep/find cover most of the same ground. Duplicate-function detection is the one signal they cannot reproduce - see references/signals.md."
---

# Project Cleanup Audit

Find what can be deleted, merged, renamed, archived or made easier to manage.
**Audit first; edit only when the user asks for implementation.**

The output is evidence, not opinion. Every finding carries a path, a number or a
search result. "Feels unused" is not a finding.

This skill is self-contained. Detail lives in `references/` relative to this
skill's base directory — read each file when you reach the phase that needs it.

## Where this skill stops

Four skills touch "the code is a mess". Keep them apart:

| Question | Skill |
| --- | --- |
| What can be removed or merged? | **this one** |
| Where should things live? | `project-folder-structure` |
| Is the code inside the files good? | `improve` |
| Are the modules the right shape? | `project-improve-architecture` |

The bridge is the **deletion test** (Phase 5). If removing something *removes*
complexity, it is a cleanup finding. If complexity merely *moves to the
callers*, it is a module-shape finding — stop auditing it and hand it to
`project-improve-architecture`.

## Phase 1: Scope

Prefer a named area over the whole repo. A scoped audit finishes and gets acted
on; a repo-wide one becomes a document nobody reads.

- Whole repo, or one package, app area, docs tree, tests, scripts, assets, generated output, workspace clutter.
- If the user named an area, take it and skip the rest.
- If the repo is large, audit the areas that change most — `git log --oneline -50 --name-only | sort | uniq -c | sort -rn | head -20` shows where the churn is.
- **Ask what is still live before classifying.** "Relevant" is the user's judgement, not an inference from age or structure. A file that looks superseded may be the only record of why a decision was made. One question at scope time prevents a batch of archive verdicts that get rejected.
- **Documents: run the architecture pass first.** If the complaint is "this is messy" and the repo is mostly documents, the cause is usually the same fact stored in several files. `project-improve-architecture` (document mode) finds that; cleanup and structure then act on a smaller, better-understood surface.

## Phase 2: Read local truth

Read `AGENTS.md`, `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, ADRs, manifests,
CI config, package exports, routes and deploy config before calling anything
"unused". A file can look dead and be load-bearing: a route, a migration, a
plugin entrypoint, a cron target, a published import, a fixture the CI reads.

**Tool-owned directories are never findings.** `.agent13/`, `.git/`, `.github/`,
`.claude/`, `.vscode/`, `.devcontainer/` and their kin sit where they sit
because a tool reads them there. Moving one breaks that tool, usually silently,
and usually weeks later. `.agent13/` in particular holds this agent's
project-local data, it is not clutter and it is not yours to
relocate. See [references/signals.md](references/signals.md#tool-owned-paths).

**"Unused" is a claim about consumers.** You have not checked consumers until
you have searched for them.

## Phase 3: Inventory

Read-only. `git grep` respects `.gitignore` and only sees tracked files; use
`find` when you also need untracked clutter. Replace the ignore list to suit the
project.

```sh
# extension mix
git ls-files | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -20

# paths that advertise their own staleness
git ls-files | grep -E '(^|/)(archive|bak|backup|copy|dead|deprecated|legacy|old|obsolete|tmp|unused|wip)(/|\.|$)'

# duplicate file names (same name, possibly different content)
git ls-files -z | xargs -0 -n1 basename | sort | uniq -d

# exact duplicate files: group by content hash
git ls-files -z | xargs -0 md5 -r | sort > /tmp/cleanup-hashes.txt
cut -d' ' -f1 /tmp/cleanup-hashes.txt | uniq -d > /tmp/cleanup-dupes.txt
grep -Ff /tmp/cleanup-dupes.txt /tmp/cleanup-hashes.txt

# TODO / FIXME / deprecation clusters
git grep -n -E '\b(TODO|FIXME|HACK|XXX|deprecated|legacy|unused|obsolete|dead code)\b'

# largest tracked files
git ls-files -z | xargs -0 ls -l | sort -k5 -rn | head -25

# untracked or ignored clutter sitting in the tree
# (tool state excluded: it is not clutter, and it is not yours to move)
find . -type f \
  -not -path './.git/*' -not -path './.agent13/saves/*' \
  -not -path './node_modules/*' -not -path './.venv/*' \
  -not -path '*/__pycache__/*' -not -path './.pytest_cache/*' \
  | wc -l
```

On Linux, `md5 -r` is `md5sum`. On a repo with filenames containing newlines,
none of this is reliable — say so rather than reporting a wrong number.

What each signal means, what it does **not** mean, and how to get duplicate
*function* detection (the one thing these recipes cannot do) is in
[references/signals.md](references/signals.md).

## Phase 4: Classify

Every finding gets exactly one verdict:

- **delete** — safe dead artifact, generated output, duplicate copy, stale scratch file.
- **merge** — duplicate docs, repeated config, one concept split across shallow modules.
- **rename / reword** — vague name, misleading description, stale terminology.
- **archive** — possibly useful history that should not sit in active source.
- **defer** — needs a product decision, an external contract, a migration plan, or more evidence.

`defer` is a real verdict, not a failure. Most findings in a healthy repo are
`defer`.

For each, record the evidence, the risk, and what would have to be true for the
verdict to be wrong. The full rubric — including the risk tiers and the
"keep because" column that stops an audit becoming a demolition order — is in
[references/classification.md](references/classification.md).

## Phase 5: Apply the deletion test

For anything that looks like a module rather than a file:

- If removing it **removes** complexity — the callers get simpler, the seams
  disappear — it is a cleanup candidate.
- If removing it **spreads** complexity to the callers — they now have to know
  what the module knew — it is a shallow module that wants deepening. Do not
  delete it. Route the finding to `project-improve-architecture`.

This is the same test as `project-improve-architecture`; read
`../project-improve-architecture/references/vocabulary.md` for the definition.
Use the same words: module, interface, implementation, depth, seam, adapter,
leverage, locality.

## Phase 6: Report

Report before editing. Concrete paths and exact search evidence, not adjectives.

```md
## Cleanup Audit
- Scope:
- Local rules read:
- Snapshot:
- Safe deletes:
- Merge/dedupe:
- Rename/reword:
- Archive:
- Architecture candidates:
- Keep because:
- Risks/blockers:
- Suggested first batch:
- Verification:
```

Then stop. If the user wants implementation, propose **one batch** — the
smallest set that is independently reviewable and revertible — and wait for
approval before touching anything.

## Rules that keep this honest

- **Do not delete by default.** Prefer a move to `archive/<date>-<reason>/`, a quarantine path, or an explicit user decision.
- **A duplicate is not automatically a merge.** Two identical files can be two independent contracts that happen to coincide today. Check whether they change together before merging them.
- **Generated output is not dead code.** It is regenerable. Deleting it is safe; deleting its *generator* is not.
- **Tests are not clutter.** A skipped test is a finding about the test, not permission to delete it.
- **Report the negative results.** "Searched for consumers of `X`, found none" is the evidence that makes a delete safe. Write it down.
- **Stop at the first ambiguity.** If you cannot tell whether something is used, the verdict is `defer`, and the report says what you could not determine.
