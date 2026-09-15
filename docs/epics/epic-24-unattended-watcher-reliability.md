# Epic 24: Unattended-Watcher Reliability

**Epic ID:** E24
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 6/6
**Dependencies:** Epic 10 (Centralized Corpus), Epic 14 (Watcher Embedding Integration), Epic 19 (Hybrid Retrieval — BM25 sidecar in the embed path)
**Target Completion:** TBD
**Source:** `docs/qa/assessments/load-bearing-review-20260612.md` findings W3, W4, W6, W7, W8; plus the 2026-07-18 startup-race field report (`docs/qa/bugs/BUG-002-watcher-startup-race-20260718.md`, W9)

---

## Branching Plan

`scripts/staging-watcher.sh` and the embeddings CLI ship on the public
mirror. Per-story summary:

| Story | Branch target | Private-only content? | Cross-repo coordination |
|-------|---------------|------------------------|--------------------------|
| 24.1  | public `main` (feature branch → squash PR) | No | None |
| 24.2  | public `main` (feature branch → squash PR) | No | None |
| 24.3  | public `main` (feature branch → squash PR) | No | None (reads `my-agents` YAML at runtime; no schema change) |
| 24.4  | public `main` (feature branch → squash PR) | No | None |
| 24.5  | public `main` (feature branch → squash PR) | No | None |

**Operational note:** the watcher runs as a systemd user service on the
maintainer's Ubuntu ingestion server. These fixes take effect when that
service is restarted onto the patched code (`systemctl --user restart
grounding-watcher`). None of them change the agent-YAML schema, so no
`my-agents` repo coordination is required.

**Cadence:** one squashed commit per story. 24.1 (the lock) is the
highest-value fix and ships first; the rest are independent and can land
in any order.

---

## Overview

The staging watcher is the only process keeping corpus and embeddings
coherent on the authoritative ingestion server. It runs unattended, is
stateful, and coordinates across two repos. The 2026-06-12 review found
five reliability defects, all of which degrade silently in exactly the
unattended conditions the service exists to handle.

1. **The embedding lock is the wrong primitive (W3).** The lock file's
   mtime is written once at acquisition and never refreshed
   (`scripts/staging-watcher.sh:171-186`). Three failures compound:
   - **Lock theft.** A legitimately long embedding run (full rebuild of a
     596k-chunk agent, or the O(total-corpus) BM25 re-concatenation on
     append) that exceeds `LOCK_TIMEOUT` (3600s) gets its lock declared
     stale and removed by a second invocation, which then re-acquires.
     Two `grounding embeddings --incremental` processes run against the
     same agent dir; with four independent renames (FAISS index, chunk
     map, BM25 pickle, BM25 map) interleaving can pair process A's index
     with process B's map → permanent `load_vector_index` size-mismatch
     `ValueError`. The fixed temp name `_embeddings.tmp`
     (`grounding/vector_store.py:178,740`) also collides.
   - **TOCTOU.** Acquisition is check-then-write (`[[ -f ]]` then
     `echo $$ >`), not atomic; two processes can both pass the check.
   - **PID never checked.** The documented "PID-staleness" does not
     exist — the PID is written (line 184) but never read; staleness is
     purely age-based. A watcher killed mid-update (OOM, `kill -9`,
     systemd restart; the `trap` at line 487 logs and exits but does
     **not** release the lock) strands the lock for up to `LOCK_TIMEOUT`,
     blacking out embedding updates for every agent during that window.

2. **Incremental-load corruption is unrecoverable and the CLI catches the
   wrong exception (W4).** `append_to_vector_index` renames the FAISS
   index then the chunk map (`grounding/vector_store.py:738-755`). A crash
   between them leaves `index.ntotal > index_size`, and `load_vector_index`
   raises `ValueError`. The incremental fallback in the CLI catches only
   `FileNotFoundError` (`grounding/cli.py:318-322`), so every subsequent
   `--incremental` run crashes with a traceback; the watcher logs
   "Embedding update failed" forever until someone manually runs a full
   rebuild.

3. **Hand-rolled YAML parsing silently misses agents (W6).**
   `find_affected_agents` (`scripts/staging-watcher.sh:110-153`) only
   parses block-style, unquoted lists. Flow style
   (`collections: [science, biology]`) yields zero matches; quoted entries
   (`- "science"`) survive `tr -d '[:space:]'` with quotes intact and fail
   the equality check. Any agent YAML using these perfectly-valid forms
   silently gets **no embedding updates ever**. Since agent YAMLs live in
   a separate, hand-edited repo, this is realistic drift — and the Python
   side (`agent_filter.load_agent_config`) parses real YAML, so
   `grounding agents show` looks fine while the watcher disagrees.

4. **Skipped/failed embedding updates are never retried (W7).** When the
   lock is held (`return 0`, line 221) or `grounding embeddings` exits
   non-zero (line 241), the document stays in the corpus but out of the
   index, and nothing requeues the work. FAISS self-heals opportunistically
   on the next ingestion touching the same agent's collections (the
   staleness diff picks up missed docs) — but for a quiet collection that
   may be weeks or never, and embeddings are MANDATORY for searchability.
   The BM25 half of a partial failure (Epic 23 / W1) does not self-heal
   even then.

5. **OCR backlog re-OCRs permanently failing PDFs every cycle (W8).** A
   scanned PDF for which OCR "completed but no output"
   (`scripts/staging-watcher.sh:298`) stays in `skipped/` and is
   re-symlinked and re-OCR'd on **every** subsequent event in that
   collection, indefinitely. OCR is minutes-per-document and the inotify
   loop is serial, so one poison PDF imposes an unbounded recurring tax
   that delays all other collections each cycle. No failure counter, no
   quarantine.

**Problem Statement:**
- Every one of these degrades silently under unattended operation —
  precisely when no human is watching. A held lock, a missed agent, a
  poison PDF, or a corrupted incremental index produces log lines (or
  none) and keeps running in a degraded state.
- The lock (W3) is the highest-leverage: it's the one defect that can
  actively corrupt the index (concurrent writers), not just stall.

**Solution:**
- Replace the mtime-based lock with `flock` on a held file descriptor —
  kernel-released on process death (no crash blackout, no PID heuristics),
  atomic acquisition (no TOCTOU), no timeout guesswork (24.1). Give the
  FAISS temp file a unique name.
- Catch `ValueError` (corrupted incremental state) in the CLI's
  incremental fallback and recover via full rebuild instead of looping on
  a traceback (24.2).
- Replace the hand-rolled YAML matcher with a real YAML parse via the
  Python side already present in the repo (24.3).
- Add a bounded retry / requeue for skipped or failed embedding updates
  so a quiet collection's docs don't stay unsearchable indefinitely (24.4).
- Add a failure counter / quarantine for OCR poison PDFs so one bad file
  stops taxing every cycle (24.5).

---

## Goals

1. Concurrent embedding writers against the same agent dir are impossible;
   the lock is atomic to acquire and auto-released on process death,
   without timeout heuristics or PID guesswork.
2. A crash mid-incremental-append is recoverable automatically (full
   rebuild fallback) instead of wedging every subsequent run.
3. Agent-collection matching honors all valid YAML list forms; an agent
   can never be silently skipped for embedding updates because of YAML
   style.
4. A document that misses its embedding update (lock held, or embed
   failure) is retried within a bounded window, not left to chance.
5. A PDF that permanently fails OCR is quarantined after a bounded number
   of attempts and stops taxing the ingestion loop.
6. Failure states are observable: clear log lines and, where appropriate,
   a marker/quarantine file — not silent degradation.

---

## Non-Goals

- A true OS-level sandbox or rearchitecture of the watcher into a
  long-lived daemon with a real job queue. These are targeted hardening
  fixes to the existing bash service, not a rewrite.
- Changing the agent-YAML schema or the `my-agents` repo. 24.3 changes how
  the watcher *reads* YAML, not the YAML itself.
- The FAISS↔BM25 transactional coherence work — that's Epic 23 (W1/W2/W5),
  the writer-side fix. This epic is the watcher/operational side. 24.4's
  retry interacts with Epic 23's self-heal but does not replace it.
- The W-LOW cluster (duplicate slugify in bash vs. Python, `mv` without
  `-n` clobbering originals, per-chunk embedding-failure permanence,
  inotify lowercase-extension-only matching, `--emit-embeddings` skipping
  BM25). Tracked in Future Work; fold opportunistically.

---

## Stories

### Story 24.1: Replace the embedding lock with `flock`

**Priority:** P1 (highest in epic). Source: W3.

Swap the mtime/PID-file lock for `flock` on a held FD, and give the FAISS
temp file a unique name.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. Embedding-update mutual exclusion uses `flock` on a dedicated lock FD
   (e.g. `flock -n 9` on `$EMBEDDINGS_DIR/_embeddings.lock` opened on FD
   9), replacing the `[[ -f ]]` check-then-`echo $$` pattern at
   `scripts/staging-watcher.sh:171-186`.
2. Acquisition is atomic: two concurrent watcher invocations can never
   both proceed into an embedding update for the same dir.
3. The lock is released automatically when the holding process dies (kernel
   behavior of `flock`), with no timeout heuristic and no PID parsing.
   `LOCK_TIMEOUT` and the age-based staleness logic are removed (or the
   variable retained only for backward-compat no-op with a deprecation
   note).
4. A long-running legitimate embedding job (longer than the old
   `LOCK_TIMEOUT`) is never interrupted by a second invocation stealing
   its lock; the second invocation either waits or skips-and-logs per the
   documented behavior (non-blocking `flock -n` → skip-and-log preserves
   the current "lock held → skip, not error" contract).
5. A watcher killed mid-update (`kill -9`, OOM, `systemctl restart`)
   leaves no stranded lock; the next invocation acquires immediately. Test
   by holding the lock in a subshell, killing it, and asserting immediate
   re-acquisition.
6. The FAISS index temp file uses a unique name (`mkstemp`-style) instead
   of the fixed `_embeddings.tmp` (`grounding/vector_store.py:178,740`),
   eliminating the concurrent-temp-file collision.
7. The "lock held → skip, logged not error" behavior is preserved for the
   non-blocking case; `release_embedding_lock`'s unconditional removal
   (line 192) is no longer needed (FD close releases) and is removed.
8. Documentation (CLAUDE.md lock-file section) updated to describe the
   `flock` behavior and drop the PID-staleness description, which never
   actually existed.

**Status:** Ready for Review

### Story 24.2: Incremental-load corruption recovery

**Priority:** P1. Source: W4.

Catch the corrupted-incremental-state exception and recover via full
rebuild instead of looping forever.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. The CLI's incremental fallback (`grounding/cli.py:318-322`) catches
   `ValueError` (the `load_vector_index` size-mismatch from a crash
   between the index and chunk-map renames) in addition to
   `FileNotFoundError`, and falls back to a full rebuild for that agent.
2. The fallback logs clearly that it detected a corrupted incremental
   state and is rebuilding — not a silent swallow.
3. After recovery, the rebuilt index is coherent (FAISS ntotal matches the
   chunk-map size) and a subsequent `--incremental` run proceeds normally.
4. Optionally (recommended): shrink the crash window in
   `append_to_vector_index` (`grounding/vector_store.py:738-755`) by
   ordering the renames so the chunk map is written/renamed before or
   together with the index, or by staging both and committing last. If
   done, it must preserve atomic per-file writes.
5. Test: simulate a crash state (index advanced, chunk map stale →
   `ntotal > index_size`), run `--incremental`, assert it rebuilds and
   recovers rather than raising.

**Status:** Ready for Review

### Story 24.3: Real YAML parsing in collection→agent matching

**Priority:** P1. Source: W6.

Replace the bash string-matching with a real YAML parse so no valid YAML
form silently drops an agent.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. `find_affected_agents` (`scripts/staging-watcher.sh:110-153`) determines
   collection membership via a real YAML parse, not `tr`/`grep` string
   matching. Preferred: shell out to the repo's own Python
   (`agent_filter.load_agent_config` already parses agent YAML correctly)
   or a small Python one-liner, so the watcher and `grounding agents show`
   agree by construction.
2. All valid YAML list forms match correctly: block style (`- science`),
   flow style (`collections: [science, biology]`), and quoted entries
   (`- "science"`).
3. An agent that should match a document's collection is never silently
   skipped due to YAML style. Test with fixtures in each of the three
   forms.
4. Performance: the parse runs once per ingestion batch (or is cached
   across agents within a batch), not once per agent per file, so it
   doesn't add meaningful latency to the inotify loop.
5. Behavior is identical to today for the block-style-unquoted YAML the
   watcher currently handles (no regression for the common case).

**Status:** Ready for Review

### Story 24.4: Retry skipped/failed embedding updates

**Priority:** P1. Source: W7.

Bound the time a document can stay unsearchable after a missed embedding
update.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. When an embedding update is skipped (lock held) or fails (non-zero
   exit), the watcher records the affected agent(s) for retry rather than
   dropping the work (e.g. a marker file in a `pending-embeddings/` dir, or
   a requeue at the next loop iteration).
2. Pending embedding updates are retried on a bounded schedule — at minimum
   on the next inotify batch, and ideally on a periodic tick so a quiet
   collection doesn't wait for unrelated ingestion activity.
3. Retries are bounded (a max-attempts counter) so a persistently failing
   agent doesn't retry forever; on exhaustion the failure is logged loudly
   (and optionally surfaced as a marker file) rather than silently dropped.
4. The retry path reuses the same `flock`-guarded embedding call from 24.1
   (no second locking scheme).
5. Test: simulate a lock-held skip, assert the work is recorded and
   retried on the next cycle and the doc becomes searchable; simulate a
   persistent embed failure, assert bounded retries then a loud final log.

**Status:** Ready for Review

### Story 24.5: OCR poison-pill quarantine

**Priority:** P2 (lowest in epic, but cheap). Source: W8.

Stop re-OCRing a PDF that permanently fails.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. A PDF that fails OCR ("completed but no output", or OCR error) has its
   failure counted. After a bounded number of attempts (default small,
   e.g. 2–3), it is quarantined (moved to a `quarantine/` or
   `skipped/permanent/` dir, or marked) and is **not** re-OCR'd on
   subsequent events.
2. The serial inotify loop no longer re-pays the OCR cost for a known
   poison PDF every cycle; one bad file cannot indefinitely delay other
   collections.
3. Quarantine is observable: a log line and a marker/location the
   maintainer can inspect and manually re-queue from.
4. A transient failure (e.g. OCR tool briefly unavailable) is not
   permanently quarantined on the first miss — the bounded counter gives
   it a few attempts first.
5. Test: feed a fixture that always yields empty OCR output, run the
   backlog path N+1 times, assert it's OCR'd at most N times then
   quarantined.

**Status:** Ready for Review

---

### Story 24.6: Close the startup-scan blind window

**Priority:** P1 (silent data loss). Source: `docs/qa/bugs/BUG-002-watcher-startup-race-20260718.md` (reported 2026-07-18).

Stop dropping files delivered while the startup scan is running.

**Problem:** `watch_staging()` ran `process_existing()` to completion
*before* starting `inotifywait -m`. On a large corpus that startup scan
runs 50+ minutes (OCR backlog + per-agent embedding rebuilds), during
which there is no live monitor at all, and the scan's own
`for collection_dir in "$STAGING_DIR"/*/` glob is snapshotted once at
t=0. A file Syncthing delivers into staging mid-scan is invisible to
both mechanisms — the scan already snapshotted the folder list, and
inotify (not yet running) never fires for an at-rest file once it
finally starts. The file sits in staging forever with no log line and no
error. Confirmed incident: four marketing EPUBs delivered inside a
~53-min startup window, never ingested, recovered only by a manual
mv-aside/mv-back re-trigger.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. The live `inotifywait -m` monitor is running for the entire duration
   of the startup scan, so any event that fires during the scan is
   buffered (pipe + kernel inotify queue) and processed once the scan
   returns.
2. Recursive watches are established before the scan snapshots
   `staging/*/`, so a file delivered just before the monitor is ready is
   still caught by the scan (union of scan + monitor has no hole for the
   common case).
3. Defense in depth: the existing `PENDING_RETRY_INTERVAL` idle tick also
   runs a reconciliation rescan that re-processes any collection holding
   staged-but-uningested files, guaranteeing eventual pickup (bounded by
   `PENDING_RETRY_INTERVAL`) even if an event is ever lost to an inotify
   queue overflow.
4. The reconciliation is idempotent and quiet: it reuses
   `process_collection` (which skips slugs already in the corpus) and
   skips empty collections so idle ticks over a mostly-empty staging tree
   produce no per-collection log spam.
5. The existing EOF→exit behavior is preserved: if `inotifywait` exits,
   the loop still logs and breaks so systemd restarts the service.
6. Manual validation (from the bug report): restart the watcher; while
   the startup scan is still running, drop a file into a fresh collection
   folder; confirm it is ingested without any manual re-trigger.

**Status:** Ready for Review

---

## Dependencies

### Epic Dependencies
- **Epic 14** — the watcher embedding-integration this hardens.
- **Epic 23** — W1/W2/W5 (FAISS↔BM25 writer coherence) interacts with
  24.4's retry: a retried embedding should land on the Epic-23-fixed
  coherent write path. 23 and 24 are independent but complementary; no
  hard ordering, though landing 23.3 first means 24.4 retries onto a
  self-healing index.

### Code Dependencies
- `scripts/staging-watcher.sh:110-153` — `find_affected_agents` (24.3).
- `scripts/staging-watcher.sh:171-192, 487` — lock acquire/release/trap
  (24.1).
- `scripts/staging-watcher.sh:220-243` — skip/fail paths (24.4).
- `scripts/staging-watcher.sh:253-314` — OCR backlog (24.5).
- `grounding/cli.py:318-322` — incremental fallback exception handling
  (24.2).
- `grounding/vector_store.py:178, 738-755` — FAISS temp name + append
  rename ordering (24.1 temp name, 24.2 optional window shrink).
- `grounding/agent_filter.py` — `load_agent_config`, the correct YAML
  parser to reuse (24.3).

---

## Implementation Order

```
Story 24.1 (flock lock)          ← highest value; prevents index corruption
Story 24.2 (incremental recovery) ← independent; unwedges crashed agents
Story 24.3 (real YAML matching)   ← independent; stops silent agent skips
Story 24.4 (retry skipped embeds) ← reuses 24.1's lock; after 24.1
Story 24.5 (OCR quarantine)       ← independent; cheapest, lowest priority
```

24.1 before 24.4 (shared lock). Otherwise independent; land in
value order.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `flock` not available or behaves differently on the target host | Low | `flock` is in util-linux, present on the Ubuntu ingestion server; CLAUDE.md already lists `inotify-tools`. Confirm in 24.1; fall back to `mkdir`-based atomic lock if ever needed. |
| Non-blocking `flock -n` changes the "wait vs. skip" semantics users rely on | Low | 24.1 AC 4 preserves the current skip-and-log contract via `-n`. Documented. |
| Full-rebuild fallback (24.2) on a 596k-chunk agent is slow and triggered by a transient | Medium | Only triggers on genuine `ValueError` corruption, which is rare; logs loudly so it's visible. Acceptable cost for unwedging. |
| Shelling out to Python per batch (24.3) adds latency | Low | Parse once per batch / cache across agents (24.3 AC 4). One subprocess per batch is negligible vs. ingestion + OCR. |
| Retry queue (24.4) grows unbounded or thrashes | Medium | Bounded max-attempts (24.4 AC 3); loud final log on exhaustion. Marker-file approach is inspectable. |
| Quarantine (24.5) hides a fixable PDF | Low | Quarantine is observable + manually re-queueable (24.5 AC 3); bounded counter gives transients a few tries (AC 4). |

---

## Testing Strategy

### Unit / Integration Tests
- 24.1: concurrent-acquire mutual exclusion; kill-holder → immediate
  re-acquire; unique temp name.
- 24.2: simulated corrupt incremental state → rebuild-and-recover.
- 24.3: block / flow / quoted YAML fixtures all match; common case
  unchanged.
- 24.4: lock-held-skip → recorded + retried + searchable; persistent
  failure → bounded retries + loud log.
- 24.5: always-empty-OCR fixture → OCR'd ≤ N times then quarantined.

Bash-side tests follow the repo's existing approach for
`scripts/staging-watcher.sh` (shell-level reproducers / bats-style or
Python-driven harness, matching whatever the watcher's current tests use;
confirm during 24.1).

### Manual Validation
- 24.1: on the ingestion server, start a long embedding job, attempt a
  concurrent trigger, confirm no double-write and clean skip-log.
- Restart the service mid-embed (`systemctl --user restart
  grounding-watcher`) and confirm no stranded lock.

---

## Acceptance Criteria (Epic Level)

1. Concurrent embedding writers against one agent dir are impossible;
   no lock theft, no TOCTOU, no crash-blackout.
2. A crashed incremental append self-recovers via full rebuild on the
   next run instead of looping on a traceback.
3. Every valid agent-YAML list form is matched; no agent silently skipped
   for embedding updates.
4. A missed embedding update is retried within a bounded window or fails
   loudly after bounded attempts.
5. A permanently-failing OCR PDF is quarantined after bounded attempts and
   stops taxing the loop.
6. All new failure handling is observable (logs / markers), never silent.
7. CI green; the common-case watcher behavior is unchanged.

---

## Definition of Done

- [x] All five stories closed with AC met.
- [x] Lock, recovery, YAML-match, retry, and quarantine tests committed and
  green (21/21 watcher shell tests in an ubuntu:22.04 container; 10
  `test_match_agents.py`; vector_store + CLI incremental recovery tests).
- [x] CLAUDE.md watcher/embeddings sections updated: `flock` lock behavior
  (replacing the inaccurate PID-staleness text), incremental recovery,
  retry behavior, OCR quarantine.
- [x] Maintainer has restarted the ingestion-server watcher onto the patched
  code and confirmed the lock + matching behavior manually. *(Done 2026-06-13 —
  Ubuntu watcher repo on `main`; both services restarted clean (no stale-lock
  errors, benign no-op startup sweeps); `flock` confirmed (2nd writer blocked,
  freed after `kill -9`); `match_agents.py --collection patent-law` →
  `corp-dev-hostile, ip` matching a ground-truth YAML grep. Bloat reclaim from
  the Epic-23 dedup also completed: all 26 indices `tombstone_count: 0`,
  ~2.2 GB reclaimed.)*
- [x] No regression in existing watcher / embedding tests.

---

## Open Questions / Future Work

**W-LOW cluster (fold opportunistically, from the review doc):**
- Duplicate slugify (bash `staging-watcher.sh:84` vs. `grounding/utils.py`)
  decides ingestion success/failure; divergence can move a never-ingested
  file to `originals/` (data-loss path) or a successful one to `skipped/`.
  Best fixed alongside 24.3 (both are "watcher reimplements Python logic
  in bash" problems).
- `mv` without `-n`/`--backup` in the staging→originals/skipped flow
  clobbers archived originals on same-name re-ingest (provenance loss).
- Per-chunk embedding failure is permanent (doc looks current to
  staleness; chunk never embedded) — `grounding/cli.py:436-437`.
- inotify match is lowercase-extension-only (`Report.PDF` never triggers);
  `$STAGING_DIR` interpolated unescaped into the match regex.
- `--emit-embeddings` ingestion path writes FAISS only, no BM25 sidecar,
  and swallows write failures (`grounding/controller.py:735-761`).
- Diverged agents repo → `git pull --ff-only` fails every batch forever;
  stale config with only ERROR log lines, no backoff/escalation.

**Out of scope:**
- Rearchitecting the watcher into a daemon with a real job queue.
- `get_chunk_metadata` rejecting v1.2 maps (`vector_store.py:403`) — a
  retrieval-metadata bug, belongs with the retrieval-surface cleanup, not
  the watcher.

---

## Dev Agent Record (Story 24.1)

### Agent Model Used
James (dev persona) on claude-opus-4-8[1m].

### Strategy
Replaced the mtime/PID lock with `flock -n` on a dedicated held file descriptor
(FD 9) in `scripts/staging-watcher.sh`, and gave the FAISS index temp file a
unique `tempfile.mkstemp` name in `grounding/vector_store.py`. The two fixes
together close the W3 lock cluster: `flock` removes the lock-theft / TOCTOU /
crash-blackout failure modes, and the unique temp name removes the
concurrent-temp-file collision that a stolen-lock double-writer would have hit.

`flock` was chosen over the `mkdir`-based atomic-lock fallback the epic allows:
it is in util-linux on the Ubuntu ingestion server (confirmed present in
`ubuntu:22.04`), is atomic to acquire, and — critically — is **kernel-released
on process death**, which `mkdir` locks are not (a `mkdir` lock still needs a
staleness heuristic on crash, reintroducing the very problem W3 is about).

### Implementation
- **`scripts/staging-watcher.sh` — `acquire_embedding_lock`:** opens the lock
  file on FD 9 (`exec 9>"$lock_file"`) and takes a non-blocking exclusive lock
  (`flock -n 9`). On contention it logs the existing skip message, closes the FD
  it opened (so repeated skips don't leak descriptors), and returns 1 —
  preserving the "lock held → skip, not error" contract (AC 4, AC 7). The
  `[[ -f ]]`/`stat -c %Y`/`echo $$` mtime path and the `LOCK_TIMEOUT` staleness
  branch are gone (AC 1–3).
- **`scripts/staging-watcher.sh` — `release_embedding_lock`:** drops the lock
  with `flock -u 9` and closes the FD; it no longer `rm`s the lock file (the
  file is a persistent rendezvous point; unlinking it races a concurrent
  acquirer) (AC 7).
- **`LOCK_TIMEOUT`** retained as a documented no-op so existing systemd unit env
  files referencing it don't break (AC 3).
- **`grounding/vector_store.py` — `_write_faiss_index_atomic`:** new module-level
  helper using `tempfile.mkstemp(dir=…, prefix=<stem>., suffix=.tmp)` for a
  unique temp name, then `Path.replace` (atomic same-fs rename), with cleanup of
  the temp file on failure. Both write sites (`write_vector_index` and
  `append_to_vector_index`) now call it instead of the fixed
  `index_path.with_suffix(".tmp")` (AC 6).
- **`CLAUDE.md`** lock-file section rewritten to describe the `flock` behavior
  and `LOCK_TIMEOUT` deprecation; the inaccurate "PID-staleness" text (which
  never actually existed in code) removed (AC 8).

### How this satisfies the ACs
- **AC 1–3** (flock on held FD; atomic; kernel-released, no timeout/PID): the
  acquire/release rewrite; `LOCK_TIMEOUT` is a no-op.
- **AC 4** (long job never has its lock stolen; second invocation skips-and-logs):
  `flock -n` simply fails for the second caller — there is no age check that can
  declare a live holder stale. Test `acquire_lock - blocked by holder`.
- **AC 5** (killed watcher leaves no stranded lock): kernel releases the flock on
  process death. Test `kill holder releases lock` holds the lock in a subprocess,
  `kill -9`s it, and asserts immediate re-acquisition.
- **AC 6** (unique FAISS temp name): `_write_faiss_index_atomic`; test
  `test_index_temp_name_is_unique` asserts two writes use distinct, non-fixed
  temp names and leave no `.tmp` behind.
- **AC 7** (skip-and-log preserved; rm removal gone): `release_embedding_lock`
  no longer unlinks; test `release_lock - drops flock, keeps file, reacquirable`.
- **AC 8** (docs): CLAUDE.md updated.

### Testing notes
`flock` is util-linux and is **not** present on the macOS dev box; the existing
`tests/test_watcher_embeddings.sh` already relied on GNU `stat -c`/`touch -d`, so
it was Linux-only to begin with. The flock-based lock removes the `stat`/`touch`
dependency but adds a `flock` dependency, so the shell suite is run in an
`ubuntu:22.04` Docker container (representative of the real target). The shell
test still uses the repo's inline-function-copy pattern; the copied
`acquire/release_embedding_lock` were updated in lockstep with the real script.

### File List
- `scripts/staging-watcher.sh` (modified) — flock acquire/release, dependency
  comment, `LOCK_TIMEOUT` deprecation note.
- `grounding/vector_store.py` (modified) — `os`/`tempfile` imports,
  `_write_faiss_index_atomic` helper, both write sites switched to it.
- `tests/test_watcher_embeddings.sh` (modified) — inline lock functions updated
  to flock; lock tests rewritten (blocked-by-holder, release-reacquire,
  kill-holder-releases); stale-removal test deleted.
- `tests/test_vector_store.py` (modified) — `test_index_temp_name_is_unique`.
- `CLAUDE.md` (modified) — flock lock-file behavior + `LOCK_TIMEOUT` deprecation.
- `docs/epics/epic-24-unattended-watcher-reliability.md` (modified) — status +
  this record.

### Validation (24.1)
```
docker run ubuntu:22.04 bash tests/test_watcher_embeddings.sh   → 11/11 pass (flock)
pytest tests/test_vector_store.py                               → 55 pass (incl. 1 new)
pytest tests/test_vector_store.py tests/test_cli.py             → 94 pass (no regression)
bash -n scripts/staging-watcher.sh                              → syntax OK
```

---

## Dev Agent Record (Story 24.2)

### Agent Model Used
James (dev persona) on claude-opus-4-8[1m].

### Strategy
Two changes: (1) the mandated recovery — catch `ValueError` in the CLI's
incremental staleness `try/except` and fall back to a full rebuild; (2) the
recommended window-shrink — in `append_to_vector_index`, stage both output files
to unique temp paths and commit them with two back-to-back renames so the gap
where the index is live but the chunk map is not is a single `os.replace`.

The `ValueError` originates in `load_vector_index` (size-mismatch check), which
`check_index_staleness` calls. The pre-fix `except FileNotFoundError` let a
post-crash incremental index crash every subsequent run on a traceback until a
manual rebuild — now it self-heals.

### Implementation
- **`grounding/cli.py` — incremental fallback:** added `except ValueError as exc`
  after the existing `except FileNotFoundError`. Logs a WARNING (with traceback)
  and prints a user-facing "Corrupted incremental index detected … Falling back
  to a full rebuild to recover" message, sets `incremental_mode = False`,
  `new_doc_ids = manifest_doc_ids`, and resets `deleted/updated/skipped` sets so
  the full path rewrites the whole agent-filtered index coherently.
- **`grounding/vector_store.py` — staged-commit append:** refactored the temp
  helper added in 24.1 into `_make_temp_path`, `_stage_faiss_index` (write FAISS
  to a unique temp, no rename), `_stage_text` (write text to a unique temp, no
  rename), and `_write_faiss_index_atomic` (stage + rename, used by the full
  `write_vector_index` path). `append_to_vector_index` now stages the FAISS index
  and the chunk-map JSON to temps first, then commits with two adjacent renames;
  on a chunk-map staging failure the already-staged index temp is unlinked so it
  doesn't leak and the live files are untouched.

### How this satisfies the ACs
- **AC 1** (catch ValueError too): the new `except ValueError` branch.
- **AC 2** (loud, not silent): WARNING log + stderr message; test asserts the
  message is present.
- **AC 3** (coherent after recovery; subsequent incremental works): test loads
  the rebuilt index (`ntotal == index_size`) and runs a second `--incremental`
  that reports "no changes".
- **AC 4** (optional window-shrink, atomic per-file writes preserved): the
  stage-both-then-commit pattern; the only work between the two renames is the
  second rename. Each file still appears atomically via its own rename. Unit
  tests assert coherence + no `.tmp` leftover, and the cleanup path on a
  chunk-map staging failure.
- **AC 5** (test): `test_incremental_recovers_from_corrupted_index` simulates the
  exact W4 state (FAISS advanced by one vector, chunk map stale) and asserts
  recovery rather than a raised traceback.

### Note on rename ordering
Reordering the two renames (chunk map first vs. index first) does not eliminate
the crash window — it only changes *which* side ends up ahead, and either way
`load_vector_index` raises `ValueError`, which 24.2's recovery now handles. So
the window-shrink (adjacent renames) is the meaningful mitigation; the recovery
is the actual fix and is correct regardless of ordering.

### File List
- `grounding/cli.py` (modified) — `except ValueError` recovery branch in the
  incremental staleness block.
- `grounding/vector_store.py` (modified) — `_make_temp_path`/`_stage_faiss_index`/
  `_stage_text` helpers; `append_to_vector_index` staged-then-commit rewrite.
- `tests/test_cli.py` (modified) — `test_incremental_recovers_from_corrupted_index`.
- `tests/test_vector_store.py` (modified) — append coherence/no-temp test +
  chunk-map-staging-failure cleanup test.
- `docs/epics/epic-24-unattended-watcher-reliability.md` (modified) — status +
  this record.

### Validation (24.2)
```
pytest tests/test_vector_store.py                            → 57 pass (2 new)
pytest tests/test_cli.py::TestCLIEmbeddingsIncremental       → (see commit) incl. recovery test
```

---

## Dev Agent Record (Story 24.3)

### Agent Model Used
James (dev persona) on claude-opus-4-8[1m].

### Strategy
Replaced the hand-rolled bash YAML matcher in `find_affected_agents` with a
small Python script (`scripts/match_agents.py`) that reuses the repo's real YAML
parser — `grounding.agent_filter.load_agent_config` (a `yaml.safe_load`). The
watcher and `grounding agents show` now resolve collection membership through the
same code path, so they agree by construction and every valid YAML list form is
honored.

### Why a separate script (not a `python -c` one-liner or yq)
- Reusing `load_agent_config` is the AC's preferred option and guarantees parity
  with `grounding agents show`.
- `match_agents.py` inserts the repo root onto `sys.path` itself, so it runs
  under any `python3` that has PyYAML — `grounding.agent_filter`'s only
  third-party dependency (everything else in its import chain —
  `grounding.manifest`, `grounding.utils` — is stdlib). No editable install or
  `PYTHONPATH` needed. The watcher prefers the project venv python and falls
  back to `python3`.
- One subprocess per collection per ingestion batch (AC 4) — negligible next to
  ingestion + OCR. Import is ~50 ms.

### Implementation
- **`scripts/match_agents.py` (new):** `find_matching_agents(agents_dir,
  collection)` iterates `*.yaml` in sorted (glob) order — matching the watcher's
  historical ordering — loads each via `load_agent_config`, and returns the
  **file stem** of agents whose `corpus_filter.collections` include the
  collection. Returns the stem (not the YAML `name:` field) because the watcher
  passes it to `grounding embeddings --agent <name>`, which resolves
  `<name>.yaml` by filename. A single invalid agent file is logged to stderr and
  skipped (best-effort, like the watcher before); exit code is always 0.
- **`scripts/staging-watcher.sh` — `find_affected_agents`:** now a thin wrapper
  that invokes `match_agents.py`; the ~40-line grep/`tr`/`sed` block is gone.

### How this satisfies the ACs
- **AC 1** (real YAML parse, reuse repo parser): delegates to
  `load_agent_config`.
- **AC 2/3** (block / flow / quoted all match; no silent skip): covered by
  `tests/test_match_agents.py` (block, flow `[a, b]`, double- and single-quoted)
  and bash fixtures `flow-agent.yaml` / `quoted-agent.yaml`.
- **AC 4** (parse once per batch): one subprocess per `find_affected_agents`
  call (once per collection per batch).
- **AC 5** (no regression for the common block-style case): the pre-existing
  bash `find_affected_agents` tests (block style, business-only, no-match,
  no-agents-dir) still pass, now driving the real script.

### Testing notes
The bash suite's inline `find_affected_agents` now calls the real
`scripts/match_agents.py`. It's run in `ubuntu:22.04` with `python3-yaml`
installed (the mounted macOS venv symlink is dead in Linux, so the wrapper's
`-x` check falls through to `python3` — exactly the production fallback path).
The matcher's logic is also unit-tested directly in `tests/test_match_agents.py`
on the host venv.

### File List
- `scripts/match_agents.py` (new) — `find_matching_agents` + CLI.
- `scripts/staging-watcher.sh` (modified) — `find_affected_agents` delegates to
  the script.
- `tests/test_match_agents.py` (new) — 10 unit tests (block/flow/quoted/no-match/
  no-filter/missing-dir/stem-vs-name/malformed-skip).
- `tests/test_watcher_embeddings.sh` (modified) — inline `find_affected_agents`
  calls the real script; flow-style + quoted fixtures and two W6 tests added.
- `docs/epics/epic-24-unattended-watcher-reliability.md` (modified) — status +
  this record.

### Validation (24.3)
```
pytest tests/test_match_agents.py                              → 10 pass
docker run ubuntu:22.04 (+python3-yaml) tests/test_watcher_embeddings.sh → 13/13 pass
match_agents.py manual: block/flow/quoted/no-match            → correct
```

---

## Dev Agent Record (Story 24.4)

### Agent Model Used
James (dev persona) on claude-opus-4-8[1m].

### Strategy
A marker-file retry queue under `$EMBEDDINGS_DIR/pending-embeddings/`, drained
under the same flock as the normal update (24.1), retried on every ingestion
batch and on a periodic inotify-idle tick so a quiet collection's docs don't
stay unsearchable. One marker per agent; its contents are the FAILURE count.
Lock-held skips record a marker but do **not** count as failures (lock
contention must not exhaust the retry budget). After `MAX_EMBED_ATTEMPTS`
genuine failures the agent is quarantined to `<agent>.failed` (loud log,
inspectable, manually requeueable).

### Implementation
- **`scripts/staging-watcher.sh`:**
  - `mark_pending_embedding` (lock-held skip → marker at count 0, never bumps an
    existing count, refuses to resurrect a `.failed` agent),
    `record_embed_failure` (bump count; at `MAX_EMBED_ATTEMPTS` → loud
    `log_error` + `.failed` marker), `clear_pending_embedding` (remove marker on
    success), `get_pending_dir`.
  - `embed_one_agent` — the single-agent incremental embed, extracted from the
    old per-agent loop; assumes the caller holds the lock. Used by both the
    normal path and the retry path (AC 4: one locking scheme).
  - `process_pending_embeddings` — drains all active markers (skipping
    `.failed`) under one `acquire_embedding_lock`; if the lock is held it logs
    and leaves them for the next tick (no busy-wait). Called from
    `process_existing` (startup), after every inotify batch, and on the idle
    tick.
  - The inotify loop switched from `while read` to `while true; do if read -t
    "$PENDING_RETRY_INTERVAL" …` so an event drives a normal batch and a
    timeout drives a pending-retry pass; EOF (inotifywait died) breaks the loop
    loudly instead of spinning. The periodic tick is safe precisely because the
    lock is now `flock` (24.1).
  - `trigger_embedding_update` now records pending on lock-held skip (for every
    affected agent), and on each agent clears the marker on success / records a
    failure on non-zero exit.
  - New config: `MAX_EMBED_ATTEMPTS` (default 3), `PENDING_RETRY_INTERVAL`
    (default 300s).

### How this satisfies the ACs
- **AC 1** (record, don't drop): lock-held → `mark_pending_embedding` per agent;
  embed failure → `record_embed_failure`. Test
  `pending - lock-held records pending`.
- **AC 2** (bounded schedule, periodic tick): drained on startup, after every
  batch, and every `PENDING_RETRY_INTERVAL` of inotify idle.
- **AC 3** (bounded retries, loud give-up): `MAX_EMBED_ATTEMPTS` → `log_error`
  + `.failed` marker. Test `pending - bounded retries then loud give-up`; a
  `.failed` agent is not resurrected (test `failed agent not resurrected`).
- **AC 4** (reuse 24.1 lock): both paths call `embed_one_agent` under
  `acquire_embedding_lock`; no second locking scheme.
- **AC 5** (tests): success-clears-marker and persistent-failure-bounded tests
  drive the real functions (overriding only `embed_one_agent` to avoid invoking
  the real model), in the Docker flock+yaml environment.

### Note on the periodic tick vs. a background daemon
The tick is implemented inside the existing single inotify loop via `read -t`,
not a second process. This keeps the watcher single-process (the epic's
non-goal is rearchitecting it into a daemon) while still retrying pending work
without new ingestion — the realistic W7 "quiet collection" case is covered
because any ingestion anywhere drains *all* pending markers, and the idle tick
covers a fully-idle server.

### File List
- `scripts/staging-watcher.sh` (modified) — pending-queue helpers,
  `embed_one_agent`, `process_pending_embeddings`, `trigger_embedding_update`
  rewrite, inotify-loop idle tick, `MAX_EMBED_ATTEMPTS`/`PENDING_RETRY_INTERVAL`.
- `tests/test_watcher_embeddings.sh` (modified) — inline mirrors of the new
  functions + 5 pending-retry tests; `log_error` added to the inline harness.
- `CLAUDE.md` (modified) — retry-queue behavior + new env vars.
- `docs/epics/epic-24-unattended-watcher-reliability.md` (modified) — status +
  this record.

### Validation (24.4)
```
docker run ubuntu:22.04 (+python3-yaml) tests/test_watcher_embeddings.sh → 18/18 pass
bash -n scripts/staging-watcher.sh                                       → syntax OK
```

---

## Dev Agent Record (Story 24.5)

### Agent Model Used
James (dev persona) on claude-opus-4-8[1m].

### Strategy
Per-file OCR attempt counters in a hidden `skipped/<collection>/.ocr-attempts/`
dir; after `MAX_OCR_ATTEMPTS` "completed but no output" misses, the PDF is moved
to `skipped/<collection>/quarantine/`. Both the counter dir (hidden) and the
quarantine dir (a subdir) fall outside the backlog's `"$skipped_path"/*.pdf`
glob, so a quarantined poison PDF is simply never collected for OCR again — no
extra filtering needed in the hot loop.

### Why count only the per-file "no output" case
The batch-level `grounding ... --ocr on` failure (the function's `else` branch)
is a transient, collection-wide error (OCR tool unavailable, crash) and is **not**
counted against individual files — exactly the AC 4 "transient failure, don't
quarantine on first miss" case; those files retry next cycle. Only the per-file
"OCR completed but produced no `doc.md`" miss — the actual poison-pill signature
— increments a file's counter.

### Implementation (`scripts/staging-watcher.sh`)
- `get_ocr_attempts_dir`, `ocr_attempt_count`, `record_ocr_failure` (increment +
  return), `quarantine_ocr_file` (move to `quarantine/`, drop counter, loud
  `log_error`), `clear_ocr_attempts` (on success).
- `process_ocr_backlog`'s per-file result loop: on success → `clear_ocr_attempts`
  + move to originals (unchanged); on "no output" → `record_ocr_failure`, and if
  the count reaches `MAX_OCR_ATTEMPTS` → `quarantine_ocr_file`, else log the
  attempt N/MAX and leave it in skipped.
- New config: `MAX_OCR_ATTEMPTS` (default 3).

### How this satisfies the ACs
- **AC 1** (failure counted, bounded): `record_ocr_failure` + the
  `>= MAX_OCR_ATTEMPTS` check.
- **AC 2** (no longer re-OCR'd; one bad file can't tax every cycle): quarantine
  is a subdir outside the `*.pdf` glob — test
  `ocr - quarantined excluded from backlog glob`.
- **AC 3** (observable): `log_error` "quarantined to …" + the inspectable
  `quarantine/` location.
- **AC 4** (transient not quarantined on first miss): bounded counter gives
  `MAX_OCR_ATTEMPTS` tries; batch-level failures aren't counted at all.
- **AC 5** (OCR'd ≤ N then quarantined): test
  `ocr - poison pill quarantined after max` exercises the exact per-file decision
  (`record_ocr_failure` → compare to MAX → `quarantine_ocr_file`) with
  `MAX_OCR_ATTEMPTS=2`.

### Testing notes
The OCR helper functions are mirrored into the bash test harness and tested
directly (counter increment/clear; quarantine-after-max with the loud log;
glob-exclusion of the quarantine subdir). The full `process_ocr_backlog`
end-to-end (which needs `grounding`, `slugify`, and `file`) is not re-mirrored;
its new wiring is three lines calling the tested helpers with the tested
comparison, consistent with how the watcher's other integration points (the
inotify loop) are validated.

### File List
- `scripts/staging-watcher.sh` (modified) — OCR attempt/quarantine helpers,
  `process_ocr_backlog` result-loop guard, `MAX_OCR_ATTEMPTS` config.
- `tests/test_watcher_embeddings.sh` (modified) — inline OCR helpers + 3 tests.
- `CLAUDE.md` (modified) — OCR quarantine behavior + `MAX_OCR_ATTEMPTS`.
- `docs/epics/epic-24-unattended-watcher-reliability.md` (modified) — status +
  this record.

### Validation (24.5)
```
docker run ubuntu:22.04 (+python3-yaml) tests/test_watcher_embeddings.sh → 21/21 pass
bash -n scripts/staging-watcher.sh                                       → syntax OK
```

---

## References

- `scripts/staging-watcher.sh` — the unattended service under hardening.
- `grounding/cli.py:318-322, 436-437` — incremental fallback + per-chunk
  embed failure.
- `grounding/vector_store.py:178, 738-755` — FAISS temp name + append
  renames.
- `grounding/agent_filter.py` — canonical YAML parser to reuse (24.3).
- `docs/qa/assessments/load-bearing-review-20260612.md` — findings W3,
  W4, W6, W7, W8 (and the W-LOW cluster in Future Work).
- `docs/epics/epic-14` (watcher embedding integration), `epic-23`
  (FAISS↔BM25 writer coherence — complementary).
