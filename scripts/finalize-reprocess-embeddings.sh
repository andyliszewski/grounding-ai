#!/usr/bin/env bash
#
# finalize-reprocess-embeddings.sh - Background completion handler for the
# REPROCESS-LIST workflow.
#
# After the watcher re-ingests the 69 corpus dirs we deleted (so they pick
# up page-aware code), this script forces each re-ingested doc through
# `grounding embeddings --update-doc-id`. That tombstones the old chunk
# embeddings and adds fresh embeddings for the new (now page-aware) chunks.
#
# Polls every POLL_INTERVAL seconds until every slug in METADATA_FILE has
# a corpus/<slug>/meta.yaml again (or until MAX_WAIT_SECONDS elapses).
# Then runs --update-doc-id per (agent, doc_id).
#
# Usage:
#   scripts/finalize-reprocess-embeddings.sh
#
# Required input file:
#   /tmp/reprocess-metadata.tsv  - TSV with: slug \t old_doc_id \t agents \t sha1 \t collections

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GROUNDING_BIN="${GROUNDING_BIN:-$REPO_DIR/venv/bin/grounding}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"

METADATA_FILE="${METADATA_FILE:-/tmp/reprocess-metadata.tsv}"
CORPUS_DIR="${CORPUS_DIR:-~/Corpora/corpus}"
AGENTS_DIR="${AGENTS_DIR:-~/my-agents/agents}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-~/Corpora/embeddings}"
LOG_FILE="${FINALIZE_LOG:-$REPO_DIR/finalize-reprocess.log}"

POLL_INTERVAL="${POLL_INTERVAL:-180}"          # 3 min between polls
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"  # 24 hour ceiling

log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date -Iseconds)] WARN: $*" | tee -a "$LOG_FILE" >&2; }
err()  { echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2; }

[[ -f "$METADATA_FILE" ]] || { err "metadata file not found: $METADATA_FILE"; exit 2; }
total=$(wc -l < "$METADATA_FILE")

log "=== finalize-reprocess-embeddings starting ==="
log "metadata:  $METADATA_FILE  ($total entries)"
log "corpus:    $CORPUS_DIR"
log "embeddings: $EMBEDDINGS_DIR"
log "poll interval: ${POLL_INTERVAL}s   max wait: ${MAX_WAIT_SECONDS}s"

# Phase 1: wait until every slug has been re-ingested (meta.yaml exists).
start_ts=$(date +%s)
log ""
log "--- Phase 1: waiting for watcher to re-ingest all $total docs ---"
while true; do
    pending=0
    missing=()
    while IFS=$'\t' read -r slug old_doc_id agents sha1 colls; do
        if [[ ! -f "$CORPUS_DIR/$slug/meta.yaml" ]]; then
            pending=$((pending + 1))
            missing+=("$slug")
        fi
    done < "$METADATA_FILE"

    done_count=$((total - pending))
    elapsed=$(( $(date +%s) - start_ts ))
    log "  re-ingested: $done_count / $total   elapsed: ${elapsed}s"

    if (( pending == 0 )); then
        log "  all docs re-ingested!"
        break
    fi
    if (( elapsed >= MAX_WAIT_SECONDS )); then
        warn "  hit MAX_WAIT_SECONDS ($MAX_WAIT_SECONDS); $pending docs still pending — continuing with what we have"
        log "  still pending:"
        printf '    %s\n' "${missing[@]:0:10}" | tee -a "$LOG_FILE"
        (( ${#missing[@]} > 10 )) && log "    ... and $((${#missing[@]} - 10)) more"
        break
    fi
    sleep "$POLL_INTERVAL"
done

# Phase 2: --update-doc-id per (agent, new_doc_id). Skip docs whose meta.yaml
# is still missing (they timed out in phase 1).
log ""
log "--- Phase 2: forcing embedding refresh per agent (--update-doc-id) ---"
ok=0; fail=0; skipped=0
while IFS=$'\t' read -r slug old_doc_id agents sha1 colls; do
    meta="$CORPUS_DIR/$slug/meta.yaml"
    if [[ ! -f "$meta" ]]; then
        warn "  skip (not re-ingested): $slug"
        skipped=$((skipped + 1))
        continue
    fi
    new_doc_id=$(grep -E '^doc_id:' "$meta" | head -1 | sed -E "s/^doc_id:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/")
    if [[ -z "$new_doc_id" ]]; then
        warn "  skip (no doc_id in meta): $slug"
        skipped=$((skipped + 1))
        continue
    fi
    # The REPROCESS-LIST 'agents' field is space-separated.
    for agent in $agents; do
        if "$GROUNDING_BIN" embeddings --agent "$agent" \
                --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" \
                --out "$EMBEDDINGS_DIR/$agent" --incremental \
                --update-doc-id "$new_doc_id" \
                >>"$LOG_FILE" 2>&1; then
            ok=$((ok + 1))
            log "  ok: $slug  agent=$agent  doc_id=$new_doc_id"
        else
            rc=$?
            fail=$((fail + 1))
            err "  fail (rc=$rc): $slug  agent=$agent  doc_id=$new_doc_id"
        fi
    done
done < "$METADATA_FILE"

log ""
log "=== finalize-reprocess-embeddings complete ==="
log "ok:      $ok"
log "failed:  $fail"
log "skipped: $skipped"
[[ "$fail" -eq 0 ]]
