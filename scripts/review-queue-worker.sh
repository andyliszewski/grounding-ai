#!/usr/bin/env bash
#
# review-queue-worker.sh - Process review-flagged docs from the watcher queue.
#
# Sidecar to staging-watcher.sh. The watcher writes a TSV line to
# $REVIEW_QUEUE whenever its filename heuristic flags an incoming file as
# "review". This worker tails that queue, derives the corpus slug from the
# original filename, and runs scripts/auto_clean_review_doc.py to extract
# a clean kebab-case name from the just-ingested chunks (via local LLM) and
# cascade the rename through originals/, meta.yaml, doc.md, chunks, and
# _index.json.
#
# Decoupling rationale: a single LLM call can take tens of seconds on
# CPU-only hosts (and 4+ minutes cold-start). Running the cleanup inline in
# the watcher would block ingestion of unrelated files. The queue gives the
# watcher a fast, non-blocking handoff while keeping the cleanup fully
# automatic.
#
# Failure handling: any entry that fails (Ollama down, LLM output invalid,
# rename collision) is appended to $REVIEW_FAILED_QUEUE for human review.
# The worker never deletes the source file from disk; the worst case is
# that a corpus doc keeps its messy original-filename-derived metadata.
#
# Usage:
#   scripts/review-queue-worker.sh
#
# Env vars (match the watcher):
#   REVIEW_QUEUE         - input queue file (default: ../filename-review-queue.tsv)
#   REVIEW_FAILED_QUEUE  - failed-entry log (default: <queue>.failed)
#   CORPUS_DIR, ORIGINALS_DIR (same defaults as the watcher)
#   AUTO_CLEAN_MODEL     - LLM model (default: llama3.2:3b)
#   AUTO_CLEAN_API_URL   - Ollama endpoint
#   LOG_FILE             - worker log (default: ../review-queue-worker.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"

REVIEW_QUEUE="${REVIEW_QUEUE:-$REPO_DIR/filename-review-queue.tsv}"
REVIEW_FAILED_QUEUE="${REVIEW_FAILED_QUEUE:-$REVIEW_QUEUE.failed}"
CORPUS_DIR="${CORPUS_DIR:-./corpus}"
ORIGINALS_DIR="${ORIGINALS_DIR:-./originals}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/review-queue-worker.log}"
AUTO_CLEAN_MODEL="${AUTO_CLEAN_MODEL:-llama3.2:3b}"
AUTO_CLEAN_API_URL="${AUTO_CLEAN_API_URL:-http://localhost:11434/v1/chat/completions}"

log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date -Iseconds)] WARN: $*" | tee -a "$LOG_FILE" >&2; }
err()  { echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2; }

# The watcher's slugify (kept in sync with grounding.utils.slugify).
slugify() {
    echo "$1" | sed -E '
        s/\.[^.]*$//                  # strip extension
        s/[_ ]/-/g                    # spaces/underscores -> hyphens
        s/[^A-Za-z0-9-]//g            # drop non-alphanumerics
        s/-+/-/g                      # collapse repeated hyphens
        s/^-+|-+$//g                  # trim leading/trailing hyphens
    ' | tr '[:upper:]' '[:lower:]'
}

process_entry() {
    local ts="$1" collection="$2" orig_name="$3" heuristic_proposal="$4"
    local slug
    slug=$(slugify "$orig_name")

    if [[ ! -d "$CORPUS_DIR/$slug" ]]; then
        warn "no corpus dir for slug=$slug (orig=$orig_name) — ingestion may not have completed yet"
        return 1
    fi

    log "processing: $collection/$orig_name  slug=$slug"
    local out
    if out=$("$PYTHON_BIN" "$SCRIPT_DIR/auto_clean_review_doc.py" \
                --slug "$slug" \
                --corpus "$CORPUS_DIR" \
                --originals "$ORIGINALS_DIR" \
                --collection "$collection" \
                --model "$AUTO_CLEAN_MODEL" \
                --api-url "$AUTO_CLEAN_API_URL" 2>&1); then
        log "  ok: $out"
        return 0
    else
        local rc=$?
        warn "  failed (rc=$rc): $out"
        printf '%s\t%s\t%s\t%s\trc=%d\t%s\n' \
            "$ts" "$collection" "$orig_name" "$heuristic_proposal" "$rc" "$(echo "$out" | tr '\n' ' ' | head -c 400)" \
            >> "$REVIEW_FAILED_QUEUE"
        return 1
    fi
}

# Process anything already sitting in the queue when the worker starts.
process_backlog() {
    [[ -f "$REVIEW_QUEUE" ]] || return 0
    local n=0
    while IFS=$'\t' read -r ts coll orig prop; do
        [[ -z "$orig" ]] && continue
        process_entry "$ts" "$coll" "$orig" "$prop" || true
        n=$((n + 1))
    done < "$REVIEW_QUEUE"
    if [[ "$n" -gt 0 ]]; then
        log "backlog processed: $n entries"
    fi
    return 0
}

main() {
    mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$REVIEW_QUEUE")"
    : >> "$REVIEW_QUEUE"   # ensure the file exists so tail -F doesn't bail

    log "=== review-queue-worker starting ==="
    log "queue:       $REVIEW_QUEUE"
    log "failed:      $REVIEW_FAILED_QUEUE"
    log "model:       $AUTO_CLEAN_MODEL"
    log "corpus:      $CORPUS_DIR"
    log "originals:   $ORIGINALS_DIR"

    process_backlog

    log "watching for new entries..."
    # tail -F follows by name (survives logrotate / rename-aside).
    # -n 0 means start at end so we don't re-process the backlog.
    tail -F -n 0 "$REVIEW_QUEUE" 2>/dev/null | while IFS=$'\t' read -r ts coll orig prop; do
        [[ -z "$orig" ]] && continue
        process_entry "$ts" "$coll" "$orig" "$prop" || true
    done
}

trap 'log "worker stopped"; exit 0' SIGINT SIGTERM
main
