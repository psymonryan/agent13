# Design It Twice

When the user wants to explore alternative interfaces for a chosen deepening candidate, use this pattern. Based on "Design It Twice" (Ousterhout): your first idea is unlikely to be the best.

Uses the vocabulary in [vocabulary.md](vocabulary.md): **module**, **interface**, **seam**, **adapter**, **leverage**.

## Process

### 1. Frame the problem space

Before designing alternatives, write a user-facing explanation of the problem space for the chosen candidate:

- The constraints any new interface would need to satisfy
- The dependencies it would rely on, and which category they fall into (see [deepening.md](deepening.md))
- A rough illustrative code sketch to ground the constraints — not a proposal, just a way to make the constraints concrete

Show this to the user, then proceed to Step 2. The user reads and thinks while the designs are produced.

### 2. Produce radically different designs

**With sub-agents available:** spawn 3+ sub-agents in parallel. Each must produce a **radically different** interface for the deepened module. Prompt each with a separate technical brief (file paths, coupling details, dependency category, what sits behind the seam) plus the paths to [vocabulary.md](vocabulary.md) and [deepening.md](deepening.md) so each agent names things consistently. Give each agent a different design constraint:

- Agent 1: "Minimize the interface: aim for 1–3 entry points max. Maximise leverage per entry point."
- Agent 2: "Maximise flexibility: support many use cases and extension."
- Agent 3: "Optimise for the most common caller: make the default case trivial."
- Agent 4 (if applicable): "Design around ports & adapters for cross-seam dependencies."

**Without sub-agents:** produce the designs sequentially in the main context, one constraint at a time. Be disciplined about not letting design 1 contaminate design 2 — for each, start from the constraints, not from the previous design.

Each design outputs:

1. Interface (types, methods, params, plus invariants, ordering, error modes)
2. Usage example showing how callers use it
3. What the implementation hides behind the seam
4. Dependency strategy and adapters (see [deepening.md](deepening.md))
5. Trade-offs: where leverage is high, where it's thin

### 3. Present and compare

Present designs sequentially so the user can absorb each one, then compare them in prose. Contrast by **depth** (leverage at the interface), **locality** (where change concentrates), and **seam placement**.

After comparing, give your own recommendation: which design you think is strongest and why. If elements from different designs would combine well, propose a hybrid. Be opinionated: the user wants a strong read, not a menu.
