# Axes and Cohesion Tests

## Constraint-first audit

Discover what cannot casually move before judging whether the structure is
clear. These constraints outrank generic structure advice.

| Constraint | Examples |
| --- | --- |
| Framework magic paths | routes, pages, static assets, migrations, generated clients, native build files |
| Tool-owned paths | `.agent13/`, `.git/`, `.github/`, `.claude/`, `.vscode/`, `.devcontainer/` — fixed by the tool that reads them, not part of the project's own organisation |
| Public contracts | package exports, CLI commands, SDK imports, published URLs, documented examples, plugin entrypoints |
| Runtime contracts | cron payloads, queue names, cache keys, persisted artifacts, model paths, data directories |
| Operational contracts | CI, deploy config, Dockerfiles, infrastructure state, environment overlays, release scripts |
| Security boundaries | `.env*`, credentials, auth backups, secret stores, tokens, customer data |
| Human contracts | documented habits, onboarding docs, team ownership, client folder expectations |

Only after this pass should you judge the structure.

## Organizing axes

Most projects use several axes at once. The goal is to make the dominant axis
explicit and to stop one folder from silently serving two.

| Axis | Use when | Common homes | How it fails |
| --- | --- | --- | --- |
| Constraint | The toolchain or a public contract mandates the path | `app/`, `pages/`, `migrations/`, `cmd/`, `public/`, `.github/` | Fighting it breaks builds, routes, imports or deploys |
| Concept / capability | Changes follow product area, domain term, workflow or stable responsibility | `billing/`, `orders/`, `rendering/`, `search/` | Names drift when copied from generic architecture instead of project language |
| Change pattern | Files usually change together even though they are different kinds | `checkout/{ui,api,tests}` | Hides shared contracts if everything becomes local |
| Audience / consumer | Different people or systems consume different material | `examples/`, `public-api/`, `operator-tools/`, `internal/` | Becomes status theatre if not tied to a real consumer |
| Lifecycle | Materials have different creation, retention or cleanup rules | `src/`, `tests/`, `docs/`, `assets/`, `runtime/`, `archive/`, `experiments/` | Scatters one concept if used as the only axis |
| Toolbox | Small complementary tools are used together | `templates/`, `adapters/`, `plugins/`, `policies/`, `validators/` | Becomes a kind bucket once the items are unrelated |
| Layer / toolchain | Technical layers are required or genuinely match the workflow | `controllers/`, `repositories/`, `models/`, `infra/` | Forces one feature change across many folders |
| Status / ownership | Workspace or portfolio organisation matters | `active/`, `archived/`, `external/`, `forks/`, `clients/` | Goes stale without curation rules |

### Choosing the axis

Order of reasoning:

1. **Respect constraints.** Preserve required paths and stable contracts; improve structure inside the extension points you are given.
2. **Follow the change.** If a normal change touches many folders, the current axis is probably wrong.
3. **Name the concept.** Prefer nouns from docs, tickets, UI copy, operations and domain language over generic architecture terms.
4. **Separate lifecycles.** Keep source away from generated output, runtime state, large data, caches and temporary work.
5. **Expose consumers.** Public APIs, examples, operator tools and internal modules deserve clear boundaries.
6. **Use layers sparingly.** Layers are fine when required; they should not hide the concepts people maintain.
7. **Keep batches small.** A structure improvement should be reviewable and reversible.

Kind-only buckets (`helpers/`, `services/`, `managers/`, `misc/`, `utils/`) are
the default failure mode: they say what type a file is, never why it belongs
next to its neighbours.

### Extracting names

1. List the nouns users and docs use: `orders`, `reports`, `datasets`, `policies`, `clients`.
2. List the verbs and workflows: `scan`, `publish`, `reconcile`, `render`, `validate`, `sync`.
3. List the stable integrations: `github`, `stripe`, `postgres`, `s3`, `slack`.
4. Preserve existing casing, singular/plural style and file-naming conventions unless those are the problem being fixed.

## Cohesion tests

Run these before proposing a move.

- **Reader** — can a new maintainer guess what belongs here from the folder name alone?
- **Change** — when a feature or workflow changes, will most changed files sit near each other?
- **Search** — will searching the project term reveal the relevant files?
- **Deletion** — if the folder vanished, would a real concept disappear, or would unrelated leftovers scatter? (Same test as `project-improve-architecture`; see its `references/vocabulary.md`.)
- **Interface** — does the folder expose a clear public surface, or just a pile of private details?
- **Lifecycle** — do all files inside share generation, retention, review and cleanup rules?
- **Constraint** — does the name fit inside the required framework or tool structure instead of bypassing it?

A folder that fails the reader test is a folder whose name is doing no work. A
folder that fails the change test is a folder that will be edited in five
places for every feature.

## Compatibility

Preserve stable entrypoints when external or habitual consumers exist:

- Keep CLI commands and package exports stable while moving internals.
- Keep route files as thin delegates when framework paths are public.
- Keep docs links working through redirects or index pages when readers depend on them.
- Keep generated output paths stable until the generator and its consumers move together.
- Add wrappers only for real consumers, never for every private file.

A compatibility wrapper should contain only enough code to delegate to the
canonical path.
