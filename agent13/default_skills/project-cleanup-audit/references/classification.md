# Classification and Risk

## The five verdicts

| Verdict | Means | Evidence required |
| --- | --- | --- |
| **delete** | Safe to remove entirely | No consumers found, and the search is written down |
| **merge** | Two or more things should become one | They change together, or one is a strict subset of the other |
| **rename / reword** | Content is fine, the label misleads | The name contradicts what the thing does or who consumes it |
| **archive** | Possibly useful, not needed in active source | No current consumer, but history or reference value is plausible |
| **defer** | Cannot decide yet | The open question is named, and so is who can answer it |

`defer` is the correct verdict more often than any other. An audit that deletes
nothing but correctly identifies three things that *cannot* be decided is worth
more than one that removes ten files and breaks a route.

## Risk tiers

Assign one to every finding.

**Low** — no consumer can plausibly exist.

- Scratch files, `.DS_Store`, editor backups, commented-out blocks.
- Generated output that a documented command regenerates.
- Files inside an `archive/` or `experiments/` folder that nothing imports.

**Medium** — consumers exist but are inside this repo.

- A module imported by one other module.
- A script referenced by a package manifest.
- A fixture used by one test file.
- A doc referenced from one index page.

Verify by searching, then by running the tests that cover the area.

**High** — consumers may exist outside this repo.

- Package exports, published entrypoints, CLI commands, public URLs.
- Database migrations, cron targets, queue names, webhook paths.
- Anything under `public/`, `static/`, `dist/` that a deploy serves.
- Anything a documented workflow tells a user to run.

High-risk findings are `defer` unless the user explicitly accepts the risk.
They are also the findings most worth reporting, because nobody else is looking
at them.

## The "keep because" column

Every audit must have one. It is the list of things that look removable and are
not, with the reason. Without it the report reads as a demolition order, and the
next person deletes something load-bearing.

Typical entries:

- `legacy_import.py` — one-off migration, already run, but kept as the record of how the data was transformed.
- `fixtures/v1_response.json` — the old API contract; the compatibility test reads it.
- `utils.py` in `pkg_b/` — byte-identical to `pkg_a/utils.py` today, but deliberately vendored so the packages can version independently.
- `TODO` in `auth/session.py` — a known security trade-off, documented, not neglect.
- `.agent13/` — tool-owned. Holds this agent's project-local `skills/` and `saves/`; the location is a contract, not a preference. Same for `.git/`, `.github/`, `.claude/`, `.vscode/`. Never a finding.

## Evidence standards

A finding is only as good as its search. Record the search, not the conclusion.

```md
- `src/legacy/importer.py` — **delete**, low risk.
  Searched: `git grep -n 'legacy.importer\|LegacyImporter\|legacy-importer'` → 0 hits.
  `git log --oneline -3 -- src/legacy/importer.py` → last touched 2024-03-11.
  Not referenced by any manifest, route table, or CI config.
```

Compare with the version that is not evidence:

```md
- `src/legacy/importer.py` — looks unused, probably safe to delete.
```

The second one is how audits cause incidents.

## Ordering the first batch

Sort the report so the first batch is obvious:

1. **Zero-risk removals** — scratch files, generated output, editor artifacts. Do these first; they build trust and cost nothing.
2. **Verified dead code** — no consumers, search recorded.
3. **Merges** — after the dead weight is gone, the real duplication is easier to see.
4. **Renames** — cheap, but they touch many files; do them alone so the diff is readable.
5. **Architecture candidates** — hand to `project-improve-architecture`; do not attempt inside a cleanup batch.

One batch, one kind of change. A commit that deletes files *and* renames modules
cannot be reviewed and cannot be reverted cleanly.

## Verification per verdict

| Verdict | Verify with |
| --- | --- |
| delete | Full test suite, plus a build/typecheck; grep for the old path afterwards |
| merge | Tests for both sides; confirm the merged version passes both |
| rename | Build, typecheck, grep for the old name across code, docs, config and CI |
| archive | Confirm nothing imports from the new location; check CI does not glob it |
| defer | Nothing — but the report states what would settle it |

## When to stop the audit

Stop and report when any of these is true:

- You cannot determine whether something has external consumers.
- A finding requires a product decision rather than a code decision.
- The repo has no tests, so no delete can be verified.
- You have more than ~20 findings — the report is now a project, and the user should choose a scope before you continue.

The last one matters. An unbounded audit is a way of not finishing.
