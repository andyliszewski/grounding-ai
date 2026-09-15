#!/usr/bin/env bash
#
# rebuild-bloated-embeddings.sh — Reclaim tombstoned vectors from agent
# FAISS indices via per-agent full rebuilds. See docs/TECH-DEBT.md TD-005.
#
# Walks $EMBEDDINGS_DIR/<agent>/_chunk_map.json to compute each agent's
# tombstone fraction and reclaimable bytes, filters by thresholds,
# sorts by reclaim descending, and for each over-threshold agent runs:
#
#   1. acquire _embeddings.lock
#   2. rm _embeddings.faiss / _chunk_map.json / _bm25.pkl / _bm25_map.json
#   3. grounding embeddings --agent <X> --corpus <Y> --agents-dir <Z>
#      --out <embeddings>/<X>     (no --incremental → full rebuild)
#   4. release lock
#
# Sequential and restartable: if a rebuild succeeds, the agent's
# tombstone count goes to 0 and the next run's filter drops it.
# If a rebuild fails midway (e.g. SIGINT after the rm but before
# grounding finishes), the agent's index files will be missing —
# re-running this script will rebuild from scratch (no .bak,
# because we have the corpus as the source of truth).
#
# Usage:
#   scripts/rebuild-bloated-embeddings.sh
#       [--min-reclaim-mb N]      default 0.5  — skip agents below this
#       [--min-tombstone-pct N]   default 0    — additional fraction filter
#       [--dry-run]               survey only, no rm/rebuild
#       [--limit N]               cap number of agents (top N by reclaim)
#       [--only AGENT]            rebuild a single named agent
#
# Environment overrides:
#   CORPUS_DIR, AGENTS_DIR, EMBEDDINGS_DIR, GROUNDING_BIN, PYTHON_BIN,
#   REBUILD_LOG (default: $REPO_DIR/rebuild-embeddings.log)
#
# **Recommended: stop the watcher first** (systemctl --user stop
# grounding-watcher) so its auto-update doesn't race on _embeddings.lock.
#

set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-~/Corpora/corpus}"
AGENTS_DIR="${AGENTS_DIR:-~/my-agents/agents}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-~/Corpora/embeddings}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GROUNDING_BIN="${GROUNDING_BIN:-$REPO_DIR/venv/bin/grounding}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"
LOG_FILE="${REBUILD_LOG:-$REPO_DIR/rebuild-embeddings.log}"

MIN_RECLAIM_MB="0.5"
MIN_TOMBSTONE_PCT="0"
DRY_RUN=0
LIMIT=0
ONLY_AGENT=""
EXCLUDE_AGENTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-reclaim-mb) MIN_RECLAIM_MB="$2"; shift 2 ;;
        --min-tombstone-pct) MIN_TOMBSTONE_PCT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --only) ONLY_AGENT="$2"; shift 2 ;;
        --exclude) EXCLUDE_AGENTS="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$GROUNDING_BIN" ]] || { echo "error: grounding not found at $GROUNDING_BIN" >&2; exit 2; }
[[ -d "$EMBEDDINGS_DIR" ]] || { echo "error: EMBEDDINGS_DIR missing: $EMBEDDINGS_DIR" >&2; exit 2; }

log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date -Iseconds)] WARN: $*" | tee -a "$LOG_FILE" >&2; }

# ---------------------------------------------------------------------------
# Survey: walk EMBEDDINGS_DIR, compute tombstone counts + reclaim bytes per
# agent, filter, sort descending, return a queue.
# Output format: agent_name<TAB>total<TAB>tombstones<TAB>pct<TAB>faiss_bytes<TAB>reclaim_bytes
# ---------------------------------------------------------------------------
survey() {
    "$PYTHON_BIN" - "$EMBEDDINGS_DIR" "$MIN_RECLAIM_MB" "$MIN_TOMBSTONE_PCT" "$ONLY_AGENT" "$EXCLUDE_AGENTS" <<'PY'
import json, pathlib, sys
emb_dir = pathlib.Path(sys.argv[1])
min_reclaim_mb = float(sys.argv[2])
min_pct = float(sys.argv[3])
only_agent = sys.argv[4] or None
exclude_set = set(s.strip() for s in (sys.argv[5] or "").split(",") if s.strip())

rows = []
for d in sorted(emb_dir.iterdir()):
    if not d.is_dir() or d.name.startswith("_"): continue
    if only_agent and d.name != only_agent: continue
    if d.name in exclude_set: continue
    cmap = d / "_chunk_map.json"
    faiss = d / "_embeddings.faiss"
    if not cmap.exists() or not faiss.exists(): continue
    try:
        m = json.loads(cmap.read_text())
    except Exception as e:
        print(f"# parse error {d.name}: {e}", file=sys.stderr)
        continue
    chunks = m.get("chunks") if isinstance(m, dict) else None
    if chunks is None and isinstance(m, list): chunks = m
    if chunks is None: continue
    total = len(chunks)
    tomb  = sum(1 for c in chunks if isinstance(c, dict) and c.get("deleted_utc"))
    faiss_bytes = faiss.stat().st_size
    pct = (tomb/total*100) if total else 0
    reclaim_bytes = int(faiss_bytes * (tomb/total)) if total else 0
    if reclaim_bytes / (1024**2) < min_reclaim_mb: continue
    if pct < min_pct: continue
    rows.append((d.name, total, tomb, pct, faiss_bytes, reclaim_bytes))

rows.sort(key=lambda r: r[5], reverse=True)
for r in rows:
    print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.2f}\t{r[4]}\t{r[5]}")
PY
}

# ---------------------------------------------------------------------------
# Lock helpers — share the watcher's _embeddings.lock convention so we don't
# race with auto-update.
# ---------------------------------------------------------------------------
LOCK_FILE="$EMBEDDINGS_DIR/_embeddings.lock"
# flock on a held FD — the same convention staging-watcher.sh uses (Epic 24,
# Story 24.1). The watcher leaves _embeddings.lock in place as a persistent
# rendezvous file (the lock is the FD, not the file), so the old
# "[[ -f LOCK_FILE ]] + mtime" check false-positived on that leftover and failed
# every agent even with nothing actually holding the lock. flock tests the real
# advisory lock and the kernel releases it on process death.
acquire_lock() {
    mkdir -p "$EMBEDDINGS_DIR"
    exec 9>"$LOCK_FILE"
    if flock -n 9; then
        return 0
    fi
    warn "embedding lock held by another process — stop the watcher or wait"
    exec 9>&- 2>/dev/null || true
    return 1
}
release_lock() {
    flock -u 9 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Signal handling
#
# Set a flag on SIGINT/SIGTERM but DO NOT exit from the trap itself. The
# main loop checks the flag between agents and exits cleanly there. This
# keeps rebuild_agent's success-logging + release_lock cleanup atomic
# with respect to grounding's return: if a signal arrives mid-rebuild,
# bash defers the trap until grounding finishes, then the trap fires
# (just sets the flag), rebuild_agent's bottom half runs to completion
# (release_lock + "rebuild OK" log), and the main loop exits before the
# next agent. Prior behavior: trap fired between grounding and the
# success log, leaking lock cleanup and missing the OK line even on
# successful rebuilds.
# ---------------------------------------------------------------------------
INTERRUPT_REQUESTED=0
on_signal() {
    INTERRUPT_REQUESTED=1
    warn "signal received — will exit cleanly after current agent finishes"
}
trap on_signal SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Per-agent rebuild
# ---------------------------------------------------------------------------
rebuild_agent() {
    local agent="$1" total="$2" tomb="$3" pct="$4" faiss_bytes="$5" reclaim_bytes="$6"
    local out_dir="$EMBEDDINGS_DIR/$agent"
    local faiss_mb reclaim_mb
    faiss_mb=$(awk "BEGIN{printf \"%.1f\", $faiss_bytes/1048576}")
    reclaim_mb=$(awk "BEGIN{printf \"%.1f\", $reclaim_bytes/1048576}")
    log "rebuild [$agent]  total=$total tomb=$tomb (${pct}%)  size=${faiss_mb}MB  reclaim=${reclaim_mb}MB"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN — would: rm indices + grounding embeddings --agent $agent"
        return 0
    fi

    if ! acquire_lock; then return 1; fi

    local start_ts=$(date +%s)
    rm -f "$out_dir/_embeddings.faiss" \
          "$out_dir/_chunk_map.json" \
          "$out_dir/_bm25.pkl" \
          "$out_dir/_bm25_map.json"
    log "  dropped prior indices for $agent"

    local rc=0
    "$GROUNDING_BIN" embeddings --agent "$agent" \
        --corpus "$CORPUS_DIR" \
        --agents-dir "$AGENTS_DIR" \
        --out "$out_dir" \
        >>"$LOG_FILE" 2>&1 || rc=$?

    local elapsed=$(($(date +%s) - start_ts))
    local new_faiss_mb=0
    if [[ -f "$out_dir/_embeddings.faiss" ]]; then
        new_faiss_mb=$(awk "BEGIN{printf \"%.1f\", $(stat -c%s "$out_dir/_embeddings.faiss")/1048576}")
    fi

    release_lock

    if [[ "$rc" -ne 0 ]]; then
        warn "  rebuild FAILED for $agent (rc=$rc) after ${elapsed}s; index files are now missing — re-run to retry"
        return 1
    fi

    log "  rebuild OK   [$agent]  new size=${new_faiss_mb}MB  reclaimed≈$(awk "BEGIN{printf \"%.1f\", $faiss_mb-$new_faiss_mb}")MB  in ${elapsed}s"
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$LOG_FILE")"
log "=== rebuild-bloated-embeddings start ==="
log "min_reclaim_mb=$MIN_RECLAIM_MB  min_tombstone_pct=$MIN_TOMBSTONE_PCT  limit=$LIMIT  only=${ONLY_AGENT:-<all>}  dry_run=$DRY_RUN"

mapfile -t queue < <(survey)
total_queue="${#queue[@]}"
if [[ "$LIMIT" -gt 0 && "$total_queue" -gt "$LIMIT" ]]; then
    queue=("${queue[@]:0:$LIMIT}")
    total_queue="$LIMIT"
fi

if [[ "$total_queue" -eq 0 ]]; then
    log "no agents over threshold — nothing to rebuild"
    exit 0
fi

# Print survey before starting
log "queue ($total_queue agents, sorted by reclaim desc):"
for row in "${queue[@]}"; do
    IFS=$'\t' read -r a t tomb pct fbytes rbytes <<<"$row"
    log "  $(printf '%-22s tomb=%5d (%5.2f%%)  faiss=%7.1fMB  reclaim=%6.1fMB' \
        "$a" "$tomb" "$pct" "$(awk "BEGIN{printf \"%.1f\",$fbytes/1048576}")" "$(awk "BEGIN{printf \"%.1f\",$rbytes/1048576}")")"
done

n=0; ok=0; fail=0; interrupted=0
start_run=$(date +%s)
for row in "${queue[@]}"; do
    n=$((n+1))
    IFS=$'\t' read -r a t tomb pct fbytes rbytes <<<"$row"
    log ""
    log "[$n/$total_queue] $a"
    if rebuild_agent "$a" "$t" "$tomb" "$pct" "$fbytes" "$rbytes"; then
        ok=$((ok+1))
    else
        fail=$((fail+1))
    fi
    # Honor SIGINT/SIGTERM cleanly between agents — never mid-rebuild.
    # By the time we reach this point, rebuild_agent has already released
    # the lock and (on success) emitted its "rebuild OK" log line.
    if [[ "$INTERRUPT_REQUESTED" -eq 1 ]]; then
        interrupted=1
        log ""
        log "interrupt acknowledged — stopping before next agent ($((total_queue - n)) remaining)"
        break
    fi
done

elapsed=$(($(date +%s) - start_run))
log ""
log "=== rebuild-bloated-embeddings complete ==="
log "rebuilt: $ok    failed: $fail    elapsed: ${elapsed}s"
[[ "$interrupted" -eq 1 ]] && exit 130
[[ "$fail" -eq 0 ]]
