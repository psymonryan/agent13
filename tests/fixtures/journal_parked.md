## 2026-10-30 12:50 - the export goal: the format list is parked, and the queue is re-ordered

Synthetic entry. A park recorded in `h.`, and a re-rank mentioned in `e.` only.
The `h.` block reports a park, not a re-rank, so `reported` must be False - the
token appears nowhere in the block.

**a. GOAL STATUS**
- Search service over a document corpus: BLOCKED - the export job still buffers the whole result set.
- Previous goal (offline ingest with a checkpoint table): DONE.

**b. NEXT SESSION, FIRST ACTION** - Measure the export's peak memory per 100k rows. CHECK: a growth curve rather than an impression.

**c. STATE AT HANDOFF** - Verified, not assumed: no worker running, working tree clean at `9d2e0f1`.

**d. DERIVED VALUES**
- Export peak RSS 3.1 GB at 2.0M rows.

**e. DECISIONS + RATIONALE**
- **Re-ranked the tickets for the NEW goal rather than keeping the old order**: the export job is now rank 2 because it is the goal's core question, and the docs build moved up because CI treats its warnings as errors.

**f. NEW KNOWLEDGE**
- (observed) The export buffers the whole result set; memory grows linearly with row count.

**g. SUPERSEDES / CONTRADICTS**
- Supersedes the 09:10 entry's ticket ranking.

**h. TICKET CHANGES**
- PARKED: widen the export format list - parked 2026-10-30: the goal changed to streaming the existing formats first. No rank changed.
