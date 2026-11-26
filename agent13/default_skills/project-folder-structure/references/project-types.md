# Project Type Guidance

Use only the sections that match the target project. These are starting points,
not templates to impose.

## Workspace or home directory

Primary question: can a person quickly find active work, stale work, external
work and experiments?

- **Status**: `active/`, `archive/`, `experiments/`, `scratch/`.
- **Ownership**: `work/`, `personal/`, `clients/`, `open-source/`.
- **Relationship**: `own/`, `forks/`, `contrib/`, `external/`, `vendor/`.

Each project should own its repo, docs, assets and local state. Avoid
organising a workspace by language — polyglot projects, forks and archived work
become hard to curate that way.

## App or backend service

Primary question: where does a user-facing workflow or operational capability
live?

```txt
framework-entrypoint/   stays where the framework expects it
src/<capability>/       product or domain behaviour
src/adapters/<system>/  external systems
tests/<capability>/     tests follow behaviour
runtime/ or artifacts/  generated local state
```

Keep transport-facing files thin when routes or controllers are framework-owned;
move domain behaviour behind them.

## Frontend or UI project

Primary question: is this code route-local, feature-local, shared design system,
or app infrastructure?

- Route and page entrypoints where the framework expects them.
- Route-local components near the route when they are not reusable.
- Feature logic in named folders once several routes use it.
- Shared primitives in an established design-system area.
- Generated assets and build output outside source.

Avoid a flat `components/` catalogue where domain UI and reusable primitives are
indistinguishable.

## CLI, library, SDK or package

Primary question: what is public API, what is example material, what is internal
implementation?

```txt
src/index.* or package exports   stay stable
src/commands/<command>/          CLI workflows
src/internal/<concept>/          private implementation
examples/<scenario>/             documented usage
fixtures/                        sample and test inputs
```

Move internals behind stable public entrypoints before changing consumers.

## Data, ML, research or analytics

Primary question: what is source code, what is input data, what is derived
output, what is exploratory?

- `src/` or `pipelines/` for production code.
- `data/raw/`, `data/interim/`, `data/processed/` only when retention and sensitivity are documented.
- `notebooks/exploration/` for exploratory work.
- `models/`, `artifacts/`, `reports/`, `outputs/` for generated results.
- `evals/` or `experiments/` for repeatable evaluation.

Do not move large data, private data, trained models or generated artifacts
without explicit approval. Document retention and `.gitignore` behaviour first.

## Infrastructure or DevOps

Primary question: what is reusable module code, what is an environment
deployment, what is generated state?

- Modules: `modules/`, `roles/`, `policies/`.
- Environments: `environments/dev/`, `environments/prod/`.
- Provider boundaries only when real: `aws/`, `gcp/`, `kubernetes/`.
- Operations: `runbooks/`, `scripts/`, `checks/`.

Never casually move state files, generated plans, lock files, secrets,
inventories or environment overlays. Verify with the tool's validate or plan
command.

## Documentation, content or knowledge base

Primary question: how do readers navigate, how does publishing work, which files
are generated?

- Audience or journey: `users/`, `developers/`, `operators/`, `reference/`, `tutorials/`.
- Product or concept: `billing/`, `integrations/`, `deployment/`.
- Publishing lifecycle: `source/`, `assets/`, `snippets/`, `generated/`, `site/`.
- Locale or version only when the publishing config supports it.

Preserve static-site paths, sidebars, redirects, slugs, media paths and
generated output expectations. Run the link check or docs build.

## Creative asset, design, game or media

Primary question: what is source, what is exported, what is referenced by
runtime, what is archived?

- Source assets: `source/`, `blender/`, `psd/`, `audio/source/`.
- Runtime assets: `assets/textures/`, `assets/sprites/`, `assets/audio/`.
- Exports: `exports/`, `renders/`, `build/`.
- Concepts: `characters/`, `levels/`, `ui/`, `environment/`.
- Archive: `archive/yyyy-mm-dd-reason/`.

Preserve paths referenced by engines, manifests, scenes and asset pipelines.
Do not move binary assets without updating references.

## Automation, agent or workflow repo

Primary question: what triggers work, what executes it, what integrates
externally, what output is generated?

```txt
triggers/              schedules, webhooks, entrypoints
workflows/<flow>/      business flow
integrations/<system>/ external APIs
reports/ or outputs/   intended deliverables
runtime/               local state, logs, caches
```

Keep runtime state and generated reports distinct. Preserve cron names, webhook
paths, queue payloads and persisted output paths.

## Monorepo

Primary question: what is the workspace boundary, and what belongs inside each
package?

- `apps/` for deployable applications.
- `packages/` for reusable libraries.
- `tools/` for repo-wide developer tooling.
- `docs/` for repo-level documentation.
- `examples/`, `fixtures/` when shared across packages.

Inside each package use that package's own best axis. Do not solve
package-local clutter by adding more root-level buckets.

## Embedded, desktop or mobile

Primary question: what is platform-specific, what is shared, what is generated
by native tooling, what is shipped?

- `shared/` only for genuinely cross-platform logic with a clear interface.
- `platforms/ios/`, `platforms/android/`, `desktop/`, `firmware/` when platform boundaries are real.
- `drivers/`, `protocols/`, `ui/`, `assets/`, `tests/` when those match maintenance work.

Do not move signing keys, provisioning profiles, generated native files or
device-specific config without explicit approval.
