## 2026-10-30 17:55 - the correction: the re-rank belongs in h., and the markers now carry it

Synthetic entry. The fix: the `h.` block names the re-rank and the reason, so a
reader of the ticket section alone can see what changed and why.

**a. GOAL STATUS**
- Search service over a document corpus: BLOCKED - the export job still buffers the whole result set.
- Cheapest next step that could move it: stream the export.

**b. NEXT SESSION, FIRST ACTION** - Stream the export: write each batch to the response as it is produced. CHECK: a 5M-row export completes under 512 MB resident.

**c. STATE AT HANDOFF** - Written by a review session, not by the agent; no service state was changed.

**d. DERIVED VALUES** - n/a.

**e. DECISIONS + RATIONALE**
- The 16:40 entry re-ranked the tickets in `e.` while its `h.` said "none". A re-rank is a ticket change, so it belongs in `h.` with the reason attached.

**f. NEW KNOWLEDGE**
- (observed) A rank expressed only by bullet order cannot be cited, and a rank that cannot be cited cannot be checked.

**g. SUPERSEDES / CONTRADICTS**
- Supersedes the 16:40 entry's ticket ranking, which was never recorded in `h.`.

**h. TICKET CHANGES**
- RE-RANKED: first-run setup -> RANK 1; session store -> RANK 2. Recorded here because the 16:40 entry re-ranked in `e.` while its `h.` said "none".
