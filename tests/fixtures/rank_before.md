# example-service - current truth

Synthetic record. Ten ranked tickets, position-encoded (no `(RANK n)` markers),
in the order the wrap-up wrote them before the incident this fixture set
reproduces. See `tests/test_wrapup_rank.py` for what each fixture exercises.

## 1. Goal
### Goal
- A search service that answers queries over a document corpus. [observed 2026-10-01]

## 3. Findings
### Ingest
- The pipeline is a queue worker; a crash mid-batch loses the in-flight batch. [observed 2026-10-02]
### Index
- A full rebuild takes 40 minutes; only changed documents need reindexing. [observed 2026-10-03]

## 6. Open tickets
- **Make the ingest pipeline resume after a crash** - the queue worker drops the in-flight batch on SIGTERM; a checkpoint table would let it resume. CHECK: a killed worker resumes without duplicating rows. - FIRST SEEN 2026-10-02 09:00
- **Make the search index rebuild incrementally** - a full rebuild takes 40 minutes; only changed documents need reindexing. CHECK: a one-document change reindexes in under a minute. - FIRST SEEN 2026-10-03 11:20
- **Make the export job stream its output** - the export buffers the whole result set in memory and runs out past ~2M rows. CHECK: a 5M-row export completes. - FIRST SEEN 2026-10-04 14:05
- **Make the session store survive a restart** - sessions live in a process-local dict, so a deploy logs everyone out. CHECK: a restart keeps sessions alive. - FIRST SEEN 2026-10-05 08:40
- **Make the first-run setup reliable** - the setup script races the database container and fails on a cold start. CHECK: ten consecutive cold starts succeed. - FIRST SEEN 2026-10-06 16:15
- **Find the producer of the corrupted cache entry** - one bad write poisons the cache for every reader; the write site is not identified. CHECK: the write site is named. - FIRST SEEN 2026-10-07 10:30
- **Explain and stop the teardown hang** - shutdown waits forever on a lock the worker thread never releases. CHECK: shutdown completes. - FIRST SEEN 2026-10-08 12:00
- **Make shutdown terminate the process** - the process stays alive after the last request, parked at ~0% CPU. CHECK: the process exits. - FIRST SEEN 2026-10-09 09:45
- **Make the CLI reject unknown flags** - a mistyped flag is silently ignored. CHECK: an unknown flag exits non-zero with a message. - FIRST SEEN 2026-10-10 15:20
- **Make the docs build without warnings** - the docs build emits 40 warnings and CI treats them as errors. CHECK: a clean build. - FIRST SEEN 2026-10-11 13:10

### Parked
- Do NOT retry: a bigger cache (measured: no effect), a second worker (doubles the corruption rate).
- **Widen the export format list** - parked 2026-10-12: the goal changed to streaming the existing formats first. - FIRST SEEN 2026-10-12 10:00
