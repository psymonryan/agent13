---
name: project-improve-architecture
disable-model-invocation: true
description: Pass 1 of 3 (architecture → cleanup → structure). Scan a codebase or document set for shallow modules and deepening opportunities, present them as a report, then grill through whichever one the user picks. Use when the user asks to review or improve the architecture, find refactoring opportunities, or make the project more testable, navigable or AI-navigable.
license: MIT
metadata:
  version: 1.1.0
  author: Agent13
  created: 2026-08-22
  category: development
  tags:
    - architecture
    - refactoring
    - deep-modules
    - review
  source: "https://github.com/mattpocock/skills (MIT, adapted)"
  source_path: skills/engineering/improve-codebase-architecture/SKILL.md
  source_ref: "3216582 2026-08-19"
---

# Improve Project Architecture

Surface architectural friction and propose **deepening opportunities**: refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability.

This skill is self-contained. All referenced files are in `references/` relative to this skill's base directory — read them with `read_file` when you reach the relevant phase. Do not wait for the whole skill's references up front; load per phase.

**Before Phase 1, read [references/vocabulary.md](references/vocabulary.md).** Use those terms exactly in every suggestion: module, interface, implementation, depth, seam, adapter, leverage, locality. Don't drift into "component," "service," "API," or "boundary."

## Document mode

When the working area is documents rather than code, keep the theory and swap the medium. Use these substitutions throughout:

| Code | Documents |
| --- | --- |
| module | document |
| interface | what a reader must know to use it: scope, status, what it supersedes |
| implementation | the content |
| seam | the boundary between a store and a view |
| tests | link resolution, path greps, counts recomputed at write time |

- **A document that restates another's fact is shallow** — interface as wide as its content, no implementation of its own. The deletion test applies unchanged: delete it and see whether the knowledge concentrates or spreads across readers.
- **The state store is the deep module.** Small interface (a topic index), large implementation (the facts). Everything else reads from it rather than restating it.
- **Snapshot vs store.** A dated record — a visit brief, a monthly log — may duplicate by design, because it must not change after the fact. A store may not. Name the exception explicitly whenever you invoke it.
- **One owner per fact class** (dose, stock, price, state, history, evidence). Two documents disagreeing about the same dose means neither owns it — that is the drift symptom to hunt.
- **The HTML report is optional below ~100 files.** A chat summary plus one index file committed to the repo usually outlives a rendered page.

The 'records' directory can be skipped during this analysis

## Phase 1: Explore

**Scope before you scan: YAGNI.** Deepening a module pays off by making future changes to it easier, so put extra weight on the parts of the codebase that have recently changed. Decide *where* to look before you look:

- If the user named a direction (a module, a subsystem, a pain point), take it, and skip the inference below.
- Otherwise, walk back a good stretch of the commit history (`git log --oneline`) to find the codebase's hot spots — the files and areas that keep coming up — and let those paths pull your attention first. If the changes are scattered with no clear hot spot, widen the net.

**Orient with the knowledge graph.** If `.understand/knowledge-graph.json` exists, `read_file` the nodes for the files in your scan area: summary, tags, complexity, and the import/call edges reveal the cluster structure without reading every file. Use the `layers` array to understand blast radius.

**Then walk the codebase organically.** Don't follow rigid heuristics; explore and note where you experience friction:

- Where does understanding one concept require bouncing between many small modules?
- Where are modules **shallow**, with an interface nearly as complex as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called (no **locality**)?
- Where do tightly-coupled modules leak across their seams?
- Which parts of the codebase are untested, or hard to test through their current interface?

Apply the **deletion test** (see vocabulary.md) to anything you suspect is shallow: would deleting it concentrate complexity, or just move it? A "yes, concentrates" is the signal you want.

**Sub-agent note:** if a sub-agent mechanism is available, dispatch the exploration to a sub-agent (give it the paths to vocabulary.md and the scan scope; it returns a shortlist of candidates with evidence). Without one, explore in the main context — accept the context cost.

## Phase 2: Present candidates as an HTML report

Write a self-contained HTML file to `plans/architecture-review-<timestamp>.html` in the current working directory, so each run gets a fresh file. If `plans/` does not exist, create it and add `plans/` to `.gitignore` (create `.gitignore` if missing; skip the append if an entry is already present). If `plans/` already exists, write the report there but do NOT touch `.gitignore`. Open it for the user (`open <path>` on macOS, `xdg-open` on Linux, `start` on Windows) and tell them the absolute path.

**Read [references/html-report.md](references/html-report.md) before writing the file** — it has the full scaffold, diagram patterns, and styling guidance.

For each candidate, render a card with:

- **Files**: which files/modules are involved
- **Problem**: why the current architecture is causing friction
- **Solution**: plain English description of what would change
- **Benefits**: explained in terms of locality and leverage, and how tests would improve
- **Before / After diagram**: side-by-side, illustrating the shallowness and the deepening
- **Recommendation strength**: one of `Strong`, `Worth exploring`, `Speculative`, rendered as a badge

End the report with a **Top recommendation** section: which candidate you'd tackle first and why.

**ADR conflicts:** if the repo has ADRs (e.g. `docs/adr/`) and a candidate contradicts one, only surface it when the friction is real enough to warrant revisiting the ADR. Mark it clearly in the card. Don't list every theoretical refactor an ADR forbids.

Do NOT propose interfaces yet. After the file is written, ask the user: **"Which of these would you like to explore?"**

## Phase 3: Grilling loop

Once the user picks a candidate, run a grilling session to walk the decision tree with them: constraints, dependencies, the shape of the deepened module, what sits behind the seam, what tests survive.

Read and follow `../grilling/SKILL.md` (the grilling skill, one question at a time, recommended answers, summary gate). If that file doesn't exist, grill in the same spirit: one question at a time, your recommended answer under each, no acting until the user confirms.

As decisions crystallize:

- **Classify the dependencies** of the deepened module (in-process / local-substitutable / remote-but-owned / true-external) — see [references/deepening.md](references/deepening.md). The category determines how the module is tested across its seam.
- **Want to explore alternative interfaces?** Use the Design It Twice pattern in [references/design-it-twice.md](references/design-it-twice.md): design the interface several radically different ways (parallel sub-agents if available, otherwise sequentially in-context), then compare on depth, locality, and seam placement.
- **User rejects the candidate with a load-bearing reason?** Offer to record the decision so future architecture reviews don't re-suggest it (e.g. a note in AGENTS.md or an ADR if the repo uses them). Only offer when the reason would actually be needed by a future explorer; skip ephemeral reasons ("not worth it right now").

When the grilling ends with a confirmed design, hand off: the deepening is now a well-specified change — suggest breaking it into work (or implementing it directly, per the user's direction).
