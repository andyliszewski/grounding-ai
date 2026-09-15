#!/usr/bin/env bash
#
# reprocess.sh - Reprocess pre-Epic-17 corpus docs through the new ingestion
#
# Iterates a worklist produced by build_reprocess_worklist.py, one doc at a
# time. For each doc:
#   1. Touches a per-collection .reprocess-lock in staging (watcher honors it)
#   2. Removes the old slug dir from corpus
#   3. Reingests the source via `grounding` (PDF/EPUB) or `ingest_docs.py`
#      (MD/DOCX), preserving the doc's original collection tags
#   4. Runs `grounding embeddings --incremental` for every agent whose
#      corpus_filter.collections intersect the doc's collections
#   5. Marks the entry "done" in the worklist (atomic JSON write) and
#      releases the per-collection lock; nudges inotifywait if staging had
#      files queued during the lock window
#
# The worklist is the single source of truth for progress, so the script is
# fully restartable: kill it mid-run and re-invoke to pick up where it left off.
#
# Dependencies: jq, grounding (in PATH or via venv), python3.13
#
# Usage:
#   scripts/reprocess.sh --worklist reprocess-worklist.json [--dry-run]
#                        [--limit N] [--priority NAME] [--collection NAME]
#                        [--no-watcher-lock]

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment, matches staging-watcher.sh names)
# ---------------------------------------------------------------------------
STAGING_DIR="${STAGING_DIR:-~/staging}"
CORPUS_DIR="${CORPUS_DIR:-~/Corpora/corpus}"
ORIGINALS_DIR="${ORIGINALS_DIR:-~/Corpora/originals}"
AGENTS_DIR="${AGENTS_DIR:-~/my-agents/agents}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-~/Corpora/embeddings}"

# Pre-flight extractable-text gate for PDFs. Mirrors the watcher's
# MIN_TEXT_YIELD_PER_MB so scanned/image PDFs are skipped (not deleted)
# instead of being run through the fast --ocr off path that can't read them.
MIN_TEXT_YIELD_PER_MB="${MIN_TEXT_YIELD_PER_MB:-1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GROUNDING_BIN="${GROUNDING_BIN:-$REPO_DIR/venv/bin/grounding}"
INGEST_DOCS="$SCRIPT_DIR/ingest_docs.py"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/venv/bin/python}"

LOG_FILE="${REPROCESS_LOG:-$REPO_DIR/reprocess.log}"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
WORKLIST=""
DRY_RUN=0
LIMIT=0
FILTER_PRIORITY=""
FILTER_COLLECTION=""
USE_WATCHER_LOCK=1

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worklist) WORKLIST="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --priority) FILTER_PRIORITY="$2"; shift 2 ;;
        --collection) FILTER_COLLECTION="$2"; shift 2 ;;
        --no-watcher-lock) USE_WATCHER_LOCK=0; shift ;;
        -h|--help) usage ;;
        *) echo "unknown flag: $1" >&2; usage ;;
    esac
done

[[ -n "$WORKLIST" ]] || { echo "error: --worklist is required" >&2; exit 2; }
[[ -f "$WORKLIST" ]] || { echo "error: worklist not found: $WORKLIST" >&2; exit 2; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 2; }
[[ -x "$GROUNDING_BIN" ]] || { echo "error: grounding not found at $GROUNDING_BIN" >&2; exit 2; }

mkdir -p "$(dirname "$LOG_FILE")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date -Iseconds)] WARN: $*" | tee -a "$LOG_FILE" >&2; }
err() { echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2; }

# ---------------------------------------------------------------------------
# Watcher coordination (per-collection .reprocess-lock files in staging)
# ---------------------------------------------------------------------------
acquire_collection_locks() {
    [[ "$USE_WATCHER_LOCK" -eq 1 ]] || return 0
    local coll
    for coll in "$@"; do
        local dir="$STAGING_DIR/$coll"
        mkdir -p "$dir"
        echo "$$" > "$dir/.reprocess-lock"
        log "  lock acquired: $dir/.reprocess-lock"
    done
}

release_collection_locks() {
    [[ "$USE_WATCHER_LOCK" -eq 1 ]] || return 0
    local coll
    for coll in "$@"; do
        local dir="$STAGING_DIR/$coll"
        rm -f "$dir/.reprocess-lock"
        log "  lock released: $dir/.reprocess-lock"
        # Re-fire any files that landed during the lock window.
        # Watcher uses inotify close_write/moved_to; rename-rename atomically
        # fires moved_to without altering content.
        shopt -s nullglob
        for f in "$dir"/*.pdf "$dir"/*.epub "$dir"/*.md "$dir"/*.docx "$dir"/*.doc; do
            [[ -f "$f" ]] || continue
            mv -- "$f" "$f.requeued"
            mv -- "$f.requeued" "$f"
            log "    re-fired staged file: $f"
        done
        shopt -u nullglob
    done
}

# ---------------------------------------------------------------------------
# Agent matching (mirrors staging-watcher.sh:find_affected_agents but for a
# list of collections)
# ---------------------------------------------------------------------------
find_affected_agents() {
    # Args: collection names. Echoes deduplicated, space-separated agent names.
    [[ -d "$AGENTS_DIR" ]] || return 0
    "$PYTHON_BIN" - "$AGENTS_DIR" "$@" <<'PY'
import sys, pathlib, yaml
agents_dir = pathlib.Path(sys.argv[1])
wanted = set(sys.argv[2:])
hits = []
for f in sorted(agents_dir.glob("*.yaml")):
    try:
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception:
        continue
    colls = set((d.get("corpus_filter") or {}).get("collections") or [])
    if colls & wanted:
        hits.append(f.stem)
print(" ".join(hits))
PY
}

# ---------------------------------------------------------------------------
# Pre-flight: does this PDF have enough extractable text for the fast path?
# Mirrors staging-watcher.sh:has_extractable_text. Returns 0 (yes) / 1 (no).
# Non-PDF inputs always return 0 (the gate doesn't apply).
# ---------------------------------------------------------------------------
has_extractable_text() {
    local file_path="$1"
    local ext="${file_path##*.}"
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    [[ "$ext" == "pdf" ]] || return 0
    command -v pdftotext >/dev/null || return 0  # no pdftotext → can't check; let grounding decide

    local file_size_mb text text_length chars_per_mb
    file_size_mb=$(awk "BEGIN {printf \"%.2f\", $(stat -c%s "$file_path") / 1048576}")
    text=$(pdftotext -layout "$file_path" - 2>/dev/null || echo "")
    text_length=${#text}
    chars_per_mb=$(awk "BEGIN {printf \"%.0f\", $text_length / ($file_size_mb + 0.1)}")
    [[ "$chars_per_mb" -ge "$MIN_TEXT_YIELD_PER_MB" ]]
}

# ---------------------------------------------------------------------------
# Worklist mutation: atomic JSON status update
# ---------------------------------------------------------------------------
mark_status() {
    local slug="$1" status="$2" detail="${3:-}"
    local tmp
    tmp=$(mktemp "${WORKLIST}.XXXXXX")
    jq --arg slug "$slug" --arg status "$status" --arg detail "$detail" \
       --arg now "$(date -Iseconds)" \
       '.entries |= map(if .slug == $slug
                          then . + {status: $status, last_attempt_utc: $now, detail: $detail}
                          else . end)' \
       "$WORKLIST" > "$tmp"
    mv "$tmp" "$WORKLIST"
}

# ---------------------------------------------------------------------------
# Per-doc reprocessing
# ---------------------------------------------------------------------------
reprocess_one() {
    local slug="$1" orig_path="$2" colls_csv="$3"

    # Quick sanity checks
    [[ -f "$orig_path" ]] || { err "orig missing: $orig_path"; mark_status "$slug" failed "orig_missing"; return 1; }

    local ext="${orig_path##*.}"
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    # Pre-flight gate: a scanned PDF can't go through --ocr off. Skip it
    # *without touching the existing corpus dir*. A later --ocr on pass
    # can claim entries marked status=skipped detail=ocr_required.
    if ! has_extractable_text "$orig_path"; then
        log "  skip (ocr_required): $slug  ($ext)  collections=$colls_csv"
        mark_status "$slug" skipped "ocr_required"
        return 2
    fi

    # Stage into a single-file temp dir so grounding only sees this one.
    local tmp_in
    tmp_in=$(mktemp -d -t reprocess.XXXXXX)
    cp -- "$orig_path" "$tmp_in/"

    log "  reprocess: $slug  ($ext)  collections=$colls_csv"

    # Rename-aside instead of destructive rm. If the ingest fails or the
    # new dir comes up empty, we restore the .bak so the agent never sees
    # a missing slug. On success we delete the .bak. Old chunks with
    # higher chunk_id than the new doc are flushed by the rename (the
    # new ingest writes into a fresh slug dir).
    local bak_dir="$CORPUS_DIR/.bak.$slug.$$"
    if [[ -d "$CORPUS_DIR/$slug" ]]; then
        mv -- "$CORPUS_DIR/$slug" "$bak_dir"
    fi

    local rc=0
    case "$ext" in
        pdf|epub)
            "$GROUNDING_BIN" "$tmp_in" "$CORPUS_DIR" \
                --collections "$colls_csv" --ocr off --verbose \
                >>"$LOG_FILE" 2>&1 || rc=$?
            ;;
        md|docx|doc)
            "$PYTHON_BIN" "$INGEST_DOCS" "$tmp_in/$(basename "$orig_path")" \
                "$CORPUS_DIR" --collections "$colls_csv" \
                >>"$LOG_FILE" 2>&1 || rc=$?
            ;;
        *)
            warn "unsupported extension $ext for $slug; skipping"
            rm -rf "$tmp_in"
            # Restore the dir we renamed aside.
            [[ -d "$bak_dir" ]] && { rm -rf "$CORPUS_DIR/$slug"; mv "$bak_dir" "$CORPUS_DIR/$slug"; }
            mark_status "$slug" failed "unsupported_ext:$ext"
            return 1
            ;;
    esac
    rm -rf "$tmp_in"

    # Rollback helper: undo the rename-aside. Drops any partial new slug
    # dir and restores the .bak. Manifest may have a stale entry pointing
    # at the restored content's chunk count — harmless, since agents
    # resolve chunks via FAISS chunk_map, not via _index.json.
    rollback_bak() {
        [[ -d "$bak_dir" ]] || return 0
        rm -rf "$CORPUS_DIR/$slug"
        mv "$bak_dir" "$CORPUS_DIR/$slug"
        warn "  rolled back: restored prior $slug from $bak_dir"
    }

    if [[ "$rc" -ne 0 ]]; then
        err "  ingest failed (rc=$rc) for $slug"
        rollback_bak
        mark_status "$slug" failed "ingest_rc=$rc"
        return 1
    fi

    if [[ ! -f "$CORPUS_DIR/$slug/doc.md" ]]; then
        err "  ingest produced no doc.md for $slug"
        rollback_bak
        mark_status "$slug" failed "no_doc_md"
        return 1
    fi

    # Ingest succeeded; the old slug is no longer needed.
    [[ -d "$bak_dir" ]] && rm -rf "$bak_dir"

    # Sanity: does the new chunk YAML now have a page_start key? (It still
    # may be null for fallback-parsed docs; that's OK — we just want to
    # confirm the new schema is in place.)
    if ! grep -q '^page_start:' "$CORPUS_DIR/$slug/chunks/ch_0001.md" 2>/dev/null; then
        warn "  reingest succeeded but ch_0001.md has no page_start field"
    fi

    # Extract the freshly-written doc_id so we can force-update its
    # embeddings. Without this, `grounding embeddings --incremental` would
    # see the same file_sha1 as before (source bytes unchanged) and skip
    # the doc — leaving its FAISS vectors pointing at the old chunk text.
    local new_doc_id
    new_doc_id=$(grep -E '^doc_id:' "$CORPUS_DIR/$slug/meta.yaml" 2>/dev/null \
                   | head -1 | sed -E "s/^doc_id:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/")
    if [[ -z "$new_doc_id" ]]; then
        warn "  could not extract doc_id from $CORPUS_DIR/$slug/meta.yaml; embeddings may be stale"
    fi

    # Embeddings: incremental rebuild for every matched agent, with an
    # explicit --update-doc-id so the just-reprocessed doc is treated as
    # changed even though its source SHA-1 is identical to last time.
    # IFS-split colls_csv into args.
    local -a colls=(); IFS=',' read -ra colls <<< "$colls_csv"
    local agents
    agents=$(find_affected_agents "${colls[@]}")
    if [[ -z "$agents" ]]; then
        warn "  no agents match collections [$colls_csv]; skipping embeddings"
    else
        local -a force_args=()
        [[ -n "$new_doc_id" ]] && force_args=(--update-doc-id "$new_doc_id")
        for agent in $agents; do
            "$GROUNDING_BIN" embeddings --agent "$agent" \
                --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" \
                --out "$EMBEDDINGS_DIR/$agent" --incremental \
                "${force_args[@]}" \
                >>"$LOG_FILE" 2>&1 || warn "  embeddings failed for agent=$agent"
        done
        log "  embeddings updated for agents: $agents (force doc_id=$new_doc_id)"
    fi

    mark_status "$slug" done ""
    return 0
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
# Build a list of pending entries (newline-delimited JSON, one per line).
# Pass filters as jq --arg values so empty string means "no filter".
mapfile -t entries < <(
    jq -c \
        --arg prio "$FILTER_PRIORITY" \
        --arg coll "$FILTER_COLLECTION" '
        .entries[]
        | select(.status == "pending")
        | (if $prio != "" then select(.priority == $prio) else . end)
        | (if $coll != "" then select(.collections | index($coll)) else . end)
    ' "$WORKLIST"
)

total="${#entries[@]}"
if [[ "$LIMIT" -gt 0 && "$total" -gt "$LIMIT" ]]; then
    entries=("${entries[@]:0:$LIMIT}")
    total="$LIMIT"
fi

log "=== reprocess.sh starting ==="
log "worklist:    $WORKLIST"
log "pending:     $total entries to process"
[[ -n "$FILTER_PRIORITY" ]]   && log "filter:      priority=$FILTER_PRIORITY"
[[ -n "$FILTER_COLLECTION" ]] && log "filter:      collection=$FILTER_COLLECTION"
[[ "$DRY_RUN" -eq 1 ]]        && log "MODE:        DRY-RUN (no FS writes)"

count=0
ok=0
fail=0
skip=0
start_ts=$(date +%s)

for entry in "${entries[@]}"; do
    count=$((count + 1))
    slug=$(echo "$entry" | jq -r .slug)
    orig_path=$(echo "$entry" | jq -r .orig_path)
    priority=$(echo "$entry" | jq -r .priority)
    # collections as comma-separated string for grounding --collections
    colls_csv=$(echo "$entry" | jq -r '.collections | join(",")')
    # collections as bash array for lock acquire/release
    mapfile -t colls < <(echo "$entry" | jq -r '.collections[]')

    log ""
    log "[$count/$total] $slug  (priority=$priority)"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  DRY-RUN would: lock+reprocess+embed+release  for collections=${colls[*]}"
        continue
    fi

    acquire_collection_locks "${colls[@]}"

    # Ensure locks always get released, even on error/SIGINT mid-doc.
    # shellcheck disable=SC2064
    trap "release_collection_locks ${colls[*]}; err 'interrupted'; exit 130" SIGINT SIGTERM

    # reprocess_one returns: 0=done, 1=failed, 2=skipped (ocr_required etc.)
    rc=0; reprocess_one "$slug" "$orig_path" "$colls_csv" || rc=$?
    case "$rc" in
        0) ok=$((ok + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac

    release_collection_locks "${colls[@]}"
    trap - SIGINT SIGTERM
done

elapsed=$(( $(date +%s) - start_ts ))
log ""
log "=== reprocess.sh complete ==="
log "processed:   $count"
log "succeeded:   $ok"
log "skipped:     $skip   (status=skipped detail=ocr_required etc.; corpus untouched)"
log "failed:      $fail"
log "elapsed:     ${elapsed}s"

[[ "$fail" -eq 0 ]]
