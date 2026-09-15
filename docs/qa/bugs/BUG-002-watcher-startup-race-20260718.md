# BUG-002: Watcher Startup Race — files delivered during the initial scan are silently dropped

**Date**: 2026-07-18
**Reporter**: Andy
**Severity**: HIGH (silent data loss — ingested documents never appear in corpus, no error)
**Status**: Resolved (Epic 24, Story 24.6)
**Category**: Ingestion / Watcher reliability
**Component**: `scripts/staging-watcher.sh` (`watch_staging()`)

## Summary

The staging watcher started its live inotify monitor ONLY AFTER its one-shot
startup scan (`process_existing`) finished. On this corpus that startup scan runs
for ~50+ minutes because it includes OCR backlog processing and per-agent
embedding rebuilds for every collection. During that entire window there was no
file monitor running, and the startup scan itself already snapshotted the
folder list at t=0. Any file that Syncthing delivered into staging during this
window was invisible to BOTH mechanisms and was never ingested. inotify only
fires on NEW events, so once the monitor finally started, the already-at-rest
files never triggered it. The files sat in staging forever until something
manually re-triggered them.

## Concrete Incident

Four marketing EPUBs were expected in the corpus and were not:

```
staging/growth,marketing,business/the-cold-start-problem-andrew-chen.epub
staging/marketing,growth,business/freemium-economics-seufert.epub
staging/marketing,growth,strategy,business/how-brands-grow-byron-sharp.epub
staging/marketing,strategy,business/positioning-ries-trout.epub
```

Timeline (all PDT):

| Time | Event |
|------|-------|
| 21:05–21:08 | file mtimes (MAC-SIDE creation times; Syncthing preserves source mtime on delivery, so mtime is NOT arrival time on the ingestion box — this is what made the bug look like "arrived before startup" at first glance) |
| 21:09:56 | watcher (re)started. `process_existing()` evaluates the glob `"$STAGING_DIR"/*/` ONCE and begins the serial scan |
| 21:09–22:03 | ~53 minutes of startup scan: OCR backlog + embedding rebuilds across all collections. `inotifywait` is NOT running yet |
| 22:03:27 | log: "Initial processing complete. Watching for new files..." — `inotifywait -m` finally starts here |

The four books were delivered by Syncthing somewhere inside the 21:09–22:03
window. Grep of the watcher log showed ZERO occurrences of any of their four
collection folders — the startup scan never saw them (folders arrived after the
glob was snapshotted) and the live monitor never saw them (not running yet / no
event for at-rest files). None of the four appeared in `corpus/`.

Manual re-trigger (`mv` aside + `mv` back to fire `moved_to`) on 2026-07-18
recovered all four. This confirms the pipeline itself is fine — the defect was
purely the startup ordering / missing reconciliation.

## Root Cause (`scripts/staging-watcher.sh`, `watch_staging()`)

```
process_existing()            # long serial scan, glob snapshotted once
log "Initial processing complete. Watching for new files..."
inotifywait -m -r -e close_write -e moved_to ... | while ...
```

Two compounding problems:

1. `inotifywait` started AFTER `process_existing()` returned, so there was a long
   blind window (as long as the startup scan takes — here ~53 min) with no
   monitor at all.
2. Even ignoring timing, the startup glob `for collection_dir in "$STAGING_DIR"/*/`
   was evaluated once; folders/files appearing mid-scan were not picked up, and
   there was no reconciliation rescan after the monitor came up.

Net effect: the union of "seen by startup scan" and "seen by live monitor" had a
hole. Files landing in the hole were dropped with no log line and no error.

## Resolution

Fixed in Epic 24, Story 24.6 — the report's recommended **A + C**:

- **A (no blind window):** `process_existing()` now runs as the first iteration
  *inside* the `inotifywait -m | while` loop, so the live monitor is already
  running throughout the startup scan. Events that fire during the scan are
  buffered in the pipe + kernel inotify queue and drained the instant the scan
  returns. A one-time `sleep 2` before the scan lets recursive watches establish
  before the folder list is snapshotted, so pre-watch arrivals are caught by the
  scan and during-scan arrivals by inotify — the union has no hole. The existing
  piped structure (and its clean EOF→exit death detection) is preserved.
- **C (defense in depth):** the existing `PENDING_RETRY_INTERVAL` idle tick now
  also runs a reconciliation rescan (`reconcile_staging`) over any collection
  holding staged-but-uningested files. Idempotent (`process_collection` skips
  slugs already in the corpus) and quiet (skips empty collections). Guarantees
  eventual pickup — bounded by `PENDING_RETRY_INTERVAL` — even if an event is
  ever lost to an inotify queue overflow.

## Verification After Fix

1. Restart watcher.
2. While the startup scan is still running (before "Watching for new files..."),
   drop/sync a new test file into a fresh collection folder.
3. Confirm it is ingested (appears in `corpus/`, log shows "Processing ... new
   files") without any manual re-trigger.
4. Confirm the idle-tick reconciliation also catches a file dropped while inotify
   is temporarily stopped.

## Note on Diagnosis

Do not trust staging file mtimes as "arrival time" on the ingestion box —
Syncthing preserves the source machine's mtime. Use the watcher log
("Checking collection" / "Processing N new files" lines) as the source of truth
for what the watcher actually observed and when.
