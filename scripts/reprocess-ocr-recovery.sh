#!/usr/bin/env bash
#
# reprocess-ocr-recovery.sh - Slow OCR pass for docs marked
#   status=skipped detail=ocr_required by reprocess.sh.
#
# Reads the same worklist as reprocess.sh, picks entries that the fast
# --ocr off path couldn't handle, and reingests them with --ocr on.
# Mirrors reprocess.sh's coordination (per-collection .reprocess-lock,
# rename-aside, embedding refresh with --update-doc-id) so it's safe to
# run while the watcher is stopped. Sequential and restartable.
#
# Usage:
#   scripts/reprocess-ocr-recovery.sh --worklist reprocess-worklist.json
#                                     [--limit N] [--dry-run]
#

set -euo pipefail

# Reuse the same env-var contract as reprocess.sh / staging-watcher.sh.
STAGING_DIR="${STAGING_DIR:-~/staging}"
CORPUS_DIR="${CORPUS_DIR:-~/Corpora/corpus}"
AGENTS_DIR="${AGENTS_DIR:-~/my-agents/agents}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-~/Corpora/embeddings}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GROUNDING_BIN="${GROUNDING_BIN:-$REPO_DIR/venv/bin/grounding}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"
LOG_FILE="${REPROCESS_OCR_LOG:-$REPO_DIR/reprocess-ocr.log}"

WORKLIST=""
LIMIT=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worklist) WORKLIST="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$WORKLIST" && -f "$WORKLIST" ]] || { echo "error: --worklist <path> required" >&2; exit 2; }

log()  { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date -Iseconds)] WARN: $*" | tee -a "$LOG_FILE" >&2; }
err()  { echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2; }

# Reuse the lock convention so the watcher (if anyone restarts it
# mid-run) defers on this collection.
acquire_lock() { mkdir -p "$STAGING_DIR/$1"; echo "$$" > "$STAGING_DIR/$1/.reprocess-lock"; }
release_lock() { rm -f "$STAGING_DIR/$1/.reprocess-lock"; }

mark_status() {
    local slug="$1" status="$2" detail="${3:-}"
    local tmp; tmp=$(mktemp "${WORKLIST}.XXXXXX")
    jq --arg slug "$slug" --arg status "$status" --arg detail "$detail" \
       --arg now "$(date -Iseconds)" '
       .entries |= map(if .slug == $slug
                         then . + {status: $status, last_attempt_utc: $now, detail: $detail}
                         else . end)' "$WORKLIST" > "$tmp"
    mv "$tmp" "$WORKLIST"
}

find_affected_agents() {
    "$PYTHON_BIN" - "$AGENTS_DIR" "$@" <<'PY'
import sys, pathlib, yaml
agents_dir = pathlib.Path(sys.argv[1])
wanted = set(sys.argv[2:])
for f in sorted(agents_dir.glob("*.yaml")):
    try: d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception: continue
    if set((d.get("corpus_filter") or {}).get("collections") or []) & wanted:
        print(f.stem)
PY
}

# --- main loop -------------------------------------------------------------
mapfile -t entries < <(
    jq -c '.entries[] | select(.status == "skipped" and .detail == "ocr_required")' "$WORKLIST"
)
total=${#entries[@]}
[[ "$LIMIT" -gt 0 && "$total" -gt "$LIMIT" ]] && { entries=("${entries[@]:0:$LIMIT}"); total=$LIMIT; }

log "=== reprocess-ocr-recovery starting ==="
log "worklist: $WORKLIST"
log "pending OCR recovery: $total docs"
[[ "$DRY_RUN" -eq 1 ]] && log "MODE: DRY-RUN"

n=0; ok=0; fail=0
for entry in "${entries[@]}"; do
    n=$((n+1))
    slug=$(jq -r .slug <<<"$entry")
    orig=$(jq -r .orig_path <<<"$entry")
    colls_csv=$(jq -r '.collections | join(",")' <<<"$entry")
    mapfile -t colls < <(jq -r '.collections[]' <<<"$entry")

    log ""
    log "[$n/$total] OCR: $slug ($colls_csv)"
    [[ -f "$orig" ]] || { err "orig missing: $orig"; mark_status "$slug" failed "orig_missing_ocr"; fail=$((fail+1)); continue; }

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN would: OCR-ingest + refresh embeddings"
        continue
    fi

    for c in "${colls[@]}"; do acquire_lock "$c"; done
    trap "for c in ${colls[*]}; do release_lock \$c; done; err 'interrupted'; exit 130" SIGINT SIGTERM

    tmp_in=$(mktemp -d -t reprocess-ocr.XXXXXX)
    cp -- "$orig" "$tmp_in/"
    bak_dir="$CORPUS_DIR/.bak.$slug.$$"
    [[ -d "$CORPUS_DIR/$slug" ]] && mv "$CORPUS_DIR/$slug" "$bak_dir"

    rc=0
    "$GROUNDING_BIN" "$tmp_in" "$CORPUS_DIR" \
        --collections "$colls_csv" --ocr on --verbose \
        >>"$LOG_FILE" 2>&1 || rc=$?
    rm -rf "$tmp_in"

    if [[ "$rc" -ne 0 || ! -f "$CORPUS_DIR/$slug/doc.md" ]]; then
        err "  OCR ingest failed (rc=$rc) for $slug"
        [[ -d "$bak_dir" ]] && { rm -rf "$CORPUS_DIR/$slug"; mv "$bak_dir" "$CORPUS_DIR/$slug"; warn "  rolled back to prior corpus dir"; }
        mark_status "$slug" failed "ocr_rc=$rc"
        fail=$((fail+1))
        for c in "${colls[@]}"; do release_lock "$c"; done
        trap - SIGINT SIGTERM
        continue
    fi
    [[ -d "$bak_dir" ]] && rm -rf "$bak_dir"

    new_doc_id=$(grep -E '^doc_id:' "$CORPUS_DIR/$slug/meta.yaml" 2>/dev/null \
                   | head -1 | sed -E "s/^doc_id:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/")
    agents=$(find_affected_agents "${colls[@]}")
    for agent in $agents; do
        "$GROUNDING_BIN" embeddings --agent "$agent" \
            --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" \
            --out "$EMBEDDINGS_DIR/$agent" --incremental \
            ${new_doc_id:+--update-doc-id "$new_doc_id"} \
            >>"$LOG_FILE" 2>&1 || warn "  embeddings failed for agent=$agent"
    done
    log "  done: $slug   embeddings refreshed for: $agents"
    mark_status "$slug" done "ocr_recovered"
    ok=$((ok+1))
    for c in "${colls[@]}"; do release_lock "$c"; done
    trap - SIGINT SIGTERM
done

log ""
log "=== reprocess-ocr-recovery complete ==="
log "processed: $n   ok: $ok   failed: $fail"
[[ "$fail" -eq 0 ]]
