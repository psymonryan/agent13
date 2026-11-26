## 2026-10-30 16:40 - the cache-entry corruption: the swap experiment is refuted - settled

Synthetic entry. The failure this fixture set exists for: the re-rank is
described in `e.` and the `h.` block says "- none". A detector that reads the
whole entry calls this compliant.

**a. GOAL STATUS**
- Search service over a document corpus: BLOCKED - the export job still buffers the whole result set.
- Cheapest next step that could move it: stream the export. NOT run - the session's budget went to the swap experiment, which is now refuted.

**b. NEXT SESSION, FIRST ACTION** - Stream the export: write each batch to the response as it is produced instead of accumulating it. CHECK: a 5M-row export completes without exceeding 512 MB of resident memory.

**c. STATE AT HANDOFF** - Verified, not assumed: no worker running (`pgrep -f queue-worker` empty), working tree clean at `4b1c9ae`. The cache was flushed and the corruption did not recur in 40 minutes of traffic.

**d. DERIVED VALUES**
- Export peak RSS 3.1 GB at 2.0M rows; growth is linear at ~1.5 KB per row.
- Index rebuild: 41 minutes for 2.1M documents.

**e. DECISIONS + RATIONALE**
- **Re-ranked the tickets**: the search index to rank 1; the ingest pipeline demoted from rank 1 to rank 6 because the checkpoint table made restarts cheap, so it no longer blocks the goal.

**f. NEW KNOWLEDGE**
- (observed) Swapping the cache implementation changed nothing: the corruption recurred within 40 minutes.
- (negative, method named) The cache probe returned 0 hits for a key known to be present, so that probe is NOT validated and its negatives prove nothing.

**g. SUPERSEDES / CONTRADICTS**
- Supersedes the 12:50 entry's ticket ranking (the ingest pipeline was rank 1).
- Supersedes the truth's claim that the cache implementation is the suspect.

**h. TICKET CHANGES**
- none
