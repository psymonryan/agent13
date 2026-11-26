# Worked Examples

## Naming

| Weak name | Better name | Why |
| --- | --- | --- |
| `utils/report.js` | `reports/render/report-renderer.js` | Names the artifact and its responsibility. |
| `scripts/fetch.js` | `sources/company/company-fetcher.js` | Names the workflow input, not the file type. |
| `components/Card.tsx` | `jobs/components/JobCard.tsx` | Keeps domain UI near the feature when it is not generic. |
| `services/discord.js` | `delivery/discord/discord-client.js` | Names the workflow and the integration. |
| `helpers/date.js` | `time/format-date.js` | Names the primitive, if it is genuinely shared. |
| `old/` | `archive/2026-05-retired-importer/` | Captures status, time and reason. |
| `stuff/` | `experiments/pdf-parser-spike/` | Makes exploratory status explicit. |

## Move table

Use a table like this before editing.

| Current path | New path | Reason | Compatibility |
| --- | --- | --- | --- |
| `scripts/podcast-scan.mjs` | `workflows/podcast-discovery/scan/podcast-scan.mjs` | Domain-owned scan workflow. | Keep a thin wrapper if package scripts or cron use it. |
| `tests/unit/podcast-policy.test.mjs` | `workflows/podcast-discovery/tests/policy.test.mjs` | Test follows the behaviour. | No wrapper needed for tests. |
| `docs/podcast.md` | `workflows/podcast-discovery/README.md` | Concept docs live with the workflow. | Link from root docs if discoverability matters. |
| `latest-run.json` | `runtime/podcast-discovery/latest-run.json` | Generated runtime state leaves source. | Update consumers and `.gitignore`. |

## Documentation scan

A good audit says what shaped the plan:

```md
Local rules read:
- `AGENTS.md`: external repos belong under `workspace/projects/`.
- `CONTEXT.md`: domain vocabulary is "scan", "recommendation", "digest".
- `package.json`: `podcast-discovery-run` is a public script, so keep a wrapper.
- Framework docs: route files under `app/` are framework-owned; do not move entrypoints.
```

## Workspace cleanup

```txt
Dev/
  my-app/
  customer-a-old/
  random-test/
  react-thing/
  forked-lib/
  abandoned-cli/
```

```txt
Dev/
  active/
    my-app/
  clients/
    customer-a/
      archive/2025-retired-site/
  experiments/
    random-test/
    react-thing/
  external/
    forked-lib/
  archive/
    abandoned-cli/
```

Reason: the primary maintenance question here is status and relationship, not
language.

## Generic folder breakup

```txt
utils/
  discord.js
  report-template.js
  job-score.js
  date.js
```

```txt
delivery/discord/discord-client.js
reports/render/report-template.js
recommendation/job-score.js
time/date.js
```

Reason: each file moves toward the concept or workflow that owns it. Note that
`date.js` survives as a shared primitive — a generic folder is not wrong for
everything, it is wrong for the files that have a better home.

## Framework-constrained app

Bad move — breaks routing:

```txt
app/jobs/page.tsx  ->  features/jobs/page.tsx
```

Better — keep the entrypoint, move the weight behind it:

```txt
app/jobs/page.tsx                # thin: imports the feature
features/jobs/JobList.tsx
features/jobs/use-jobs.ts
features/jobs/job-filters.ts
```

Reason: the constraint axis wins. Improve structure inside the extension point
the framework gives you.

## Generated and runtime output

```txt
# before: generated material in source
src/reports/latest.html
src/reports/2026-10-01.html
src/cache.json

# after
src/reports/render.ts            # the generator
reports/2026-10-01.html          # generated, gitignored or retained by rule
runtime/cache.json               # runtime state, gitignored
```

Reason: lifecycle axis. The generator is source; its output is not. State the
retention rule in the project's docs or `.gitignore` so the folder does not
silently regrow.

## When not to move anything

A structure audit can legitimately conclude "leave it". Say so explicitly when:

- The only complaint is aesthetic and no consumer is confused.
- The project is small enough that the current shape is already obvious.
- The move would touch a public contract for no maintainability gain.
- The area is about to be rewritten anyway.

Recording a decision *not* to reorganise is as useful as recording a move — it
stops the next session re-proposing it.
