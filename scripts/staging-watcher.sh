#!/usr/bin/env bash
#
# staging-watcher.sh - Monitor staging folder and process documents into corpus
#
# Deployment: Linux ingestion machine only
# Dependencies: inotifywait (inotify-tools), flock (util-linux), grounding,
#               pdftotext, python-docx, git
# Supported formats: PDF, EPUB, MD, DOCX, DOC
#

# Get script directory for finding ingest_docs.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGEST_DOCS="$SCRIPT_DIR/ingest_docs.py"

set -euo pipefail

# Configuration (override via environment)
STAGING_DIR="${STAGING_DIR:-./staging}"
CORPUS_DIR="${CORPUS_DIR:-./corpus}"
ORIGINALS_DIR="${ORIGINALS_DIR:-./originals}"
SKIPPED_DIR="${SKIPPED_DIR:-./skipped}"
LOG_FILE="${LOG_FILE:-/var/log/grounding-watcher.log}"

# Filename cleanup (rename messy filenames to clean kebab-case before
# ingestion so corpus slug + chunk source: come out clean from the start).
# Set CLEAN_FILENAMES=false to disable. Files whose status comes back as
# "review" are processed with their original name and logged to REVIEW_QUEUE
# for chunk-content-informed correction later.
CLEAN_FILENAMES="${CLEAN_FILENAMES:-true}"
REVIEW_QUEUE="${REVIEW_QUEUE:-$SCRIPT_DIR/../filename-review-queue.tsv}"

# Embedding configuration
AUTO_EMBEDDINGS="${AUTO_EMBEDDINGS:-false}"
AGENTS_DIR="${AGENTS_DIR:-}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-}"
# DEPRECATED (Epic 24, Story 24.1): the embedding lock now uses flock on a held
# file descriptor, which the kernel releases automatically when the holder dies.
# There is no longer an age-based staleness check, so LOCK_TIMEOUT is a no-op.
# Retained only so existing systemd unit env files referencing it don't break.
LOCK_TIMEOUT="${LOCK_TIMEOUT:-3600}"  # no-op; see flock-based lock below

# Retry of skipped/failed embedding updates (Epic 24, Story 24.4).
# MAX_EMBED_ATTEMPTS: how many times an agent's embedding update may FAIL before
#   it is given up on (a loud log + a .failed marker for manual inspection).
#   Lock-held skips do NOT count against this budget.
# PENDING_RETRY_INTERVAL: seconds of inotify inactivity after which pending
#   embedding updates are retried even with no new ingestion (periodic tick), so
#   a quiet collection's docs don't stay unsearchable indefinitely.
MAX_EMBED_ATTEMPTS="${MAX_EMBED_ATTEMPTS:-3}"
PENDING_RETRY_INTERVAL="${PENDING_RETRY_INTERVAL:-300}"

# OCR poison-pill quarantine (Epic 24, Story 24.5). A scanned PDF whose OCR
# "completes but produces no output" otherwise stays in skipped/ and is re-OCR'd
# (minutes each) on every cycle, taxing the serial loop. After this many OCR
# attempts the file is quarantined to skipped/<collection>/quarantine/ and is no
# longer re-OCR'd.
MAX_OCR_ATTEMPTS="${MAX_OCR_ATTEMPTS:-3}"

# Git sync configuration
# REPO_DIR: Path to git repository containing agent definitions
# If not set, derives from AGENTS_DIR (parent directory)
REPO_DIR="${REPO_DIR:-}"
GIT_PULL_ENABLED="${GIT_PULL_ENABLED:-true}"

# Minimum text yield (chars per MB) to consider pdftotext successful
MIN_TEXT_YIELD_PER_MB=1000

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Pull latest changes from git repository to get updated agent definitions
pull_latest_repo() {
    # Skip if disabled
    if [[ "${GIT_PULL_ENABLED}" != "true" ]]; then
        return 0
    fi

    # Determine repo directory
    local repo_dir="$REPO_DIR"
    if [[ -z "$repo_dir" && -n "$AGENTS_DIR" ]]; then
        # Derive from AGENTS_DIR (assume agents/ is in repo root)
        repo_dir=$(dirname "$AGENTS_DIR")
    fi

    if [[ -z "$repo_dir" || ! -d "$repo_dir/.git" ]]; then
        log "Git pull skipped: REPO_DIR not configured or not a git repository"
        return 0
    fi

    log "Pulling latest changes from git repository: $repo_dir"
    local pull_output
    if pull_output=$(cd "$repo_dir" && git pull --ff-only 2>&1); then
        if [[ "$pull_output" == "Already up to date." ]]; then
            log "Repository already up to date"
        else
            log "Git pull completed: $pull_output"
        fi
    else
        log_error "Git pull failed: $pull_output"
        # Continue anyway - use existing agent definitions
    fi
}

# Generate slug from filename (matches grounding's slugify)
slugify() {
    echo "$1" | sed 's/\.[^.]*$//' | tr '[:upper:]' '[:lower:]' | tr ' _' '-' | sed 's/[^a-z0-9-]//g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'
}

# Quick check if PDF has extractable text
has_extractable_text() {
    local file_path="$1"
    local file_size_mb
    local text_length
    local chars_per_mb

    # Get file size in MB
    file_size_mb=$(awk "BEGIN {printf \"%.2f\", $(stat -c%s "$file_path") / 1048576}")

    # Extract text with pdftotext
    local text
    text=$(pdftotext -layout "$file_path" - 2>/dev/null || echo "")
    text_length=${#text}

    # Calculate chars per MB
    chars_per_mb=$(awk "BEGIN {printf \"%.0f\", $text_length / ($file_size_mb + 0.1)}")

    [[ "$chars_per_mb" -ge "$MIN_TEXT_YIELD_PER_MB" ]]
}

# Find agents that have the given collection in their corpus_filter.collections.
# Returns newline/space-separated list of agent file stems (the value passed to
# `grounding embeddings --agent`).
#
# Story 24.3 (W6): this delegates to scripts/match_agents.py, which parses each
# agent YAML with the repo's real parser (grounding.agent_filter via
# yaml.safe_load) instead of the old hand-rolled grep/tr matcher. The old code
# only understood block-style unquoted lists; flow style
# (`collections: [a, b]`) yielded zero matches and quoted entries (`- "a"`)
# failed the equality check, silently skipping those agents' embedding updates
# forever. The Python parser handles all valid YAML list forms, so the watcher
# and `grounding agents show` agree by construction. One subprocess per batch
# (callers invoke this once per collection per ingestion batch).
# Any remaining arguments are the slugs of the documents just ingested. They let
# match_agents.py also match agents that reach a document through a
# corpus_filter.slugs pin instead of through its collection -- collections alone
# never fire for those, because a pinned document's collection typically belongs
# to a different agent or to none, so the pin silently never took effect.
find_affected_agents() {
    local collection="$1"
    shift
    local slugs=("$@")

    # Skip if AGENTS_DIR not configured
    [[ -z "$AGENTS_DIR" || ! -d "$AGENTS_DIR" ]] && return 0

    # Prefer the project venv python (where grounding is installed); fall back to
    # any python3 (match_agents.py bootstraps the repo onto sys.path itself, so
    # only PyYAML is required).
    local python_bin="$SCRIPT_DIR/../venv/bin/python"
    [[ -x "$python_bin" ]] || python_bin="python3"

    local slug_args=()
    local s
    for s in "${slugs[@]}"; do
        [[ -n "$s" ]] && slug_args+=(--slug "$s")
    done

    "$python_bin" "$SCRIPT_DIR/match_agents.py" \
        --agents-dir "$AGENTS_DIR" --collection "$collection" \
        "${slug_args[@]}" 2>>"$LOG_FILE" || true
}

# Lock file for embedding updates
get_lock_file() {
    echo "${EMBEDDINGS_DIR}/_embeddings.lock"
}

# File descriptor used to hold the embedding lock. flock() is associated with
# this open descriptor; the lock lives for exactly as long as the descriptor
# stays open in this process (and its children).
EMBEDDING_LOCK_FD=9

# Try to acquire the embedding lock via flock on a held file descriptor.
# Returns 0 if acquired, 1 if held by another process.
#
# flock replaces the old mtime/PID lock (Epic 24, Story 24.1; W3):
#   - Acquisition is atomic (no check-then-write TOCTOU).
#   - The kernel releases the lock automatically when the holding process dies
#     (crash, OOM, kill -9, systemd restart), so there is no stale-lock window,
#     no age-based timeout heuristic, and no PID parsing.
#   - The non-blocking `-n` flag preserves the existing "lock held -> skip and
#     log, not error" contract; a legitimately long embedding run can never
#     have its lock stolen by a second invocation.
acquire_embedding_lock() {
    local lock_file
    lock_file=$(get_lock_file)

    # Skip if EMBEDDINGS_DIR not configured
    [[ -z "$EMBEDDINGS_DIR" ]] && return 1

    # Ensure embeddings directory exists
    mkdir -p "$EMBEDDINGS_DIR"

    # Open the lock file on the dedicated FD. The lock file is a persistent
    # rendezvous point; opening it does not yet grant the lock.
    exec 9>"$lock_file"

    if flock -n 9; then
        return 0
    fi

    log "Embedding update skipped: lock held by another process"
    # We opened the descriptor but did not acquire the lock; close it so we
    # don't leak descriptors across repeated skips.
    exec 9>&- 2>/dev/null || true
    return 1
}

# Release the embedding lock by closing the held descriptor. The lock file
# itself is intentionally NOT removed (it is a persistent rendezvous point;
# unlinking it would race a concurrent acquirer). Closing the FD drops the
# flock immediately; the kernel would also drop it on process death.
release_embedding_lock() {
    flock -u 9 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
}

# --- Pending-embedding retry queue (Epic 24, Story 24.4 / W7) -----------------
#
# When an embedding update is skipped (lock held) or fails (non-zero exit), the
# affected agent is recorded as a marker file under
# $EMBEDDINGS_DIR/pending-embeddings/<agent> so the work is not silently dropped.
# Pending updates are retried on the next ingestion batch (any collection) and
# on a periodic inotify-idle tick, under the same flock as the normal update.
# A marker file's contents are the FAILURE count; lock-held skips do not bump it.
# After MAX_EMBED_ATTEMPTS failures the agent is given up on (loud log + a
# <agent>.failed marker for manual inspection / requeue).

get_pending_dir() {
    echo "${EMBEDDINGS_DIR}/pending-embeddings"
}

# Record an agent as pending without counting a failure (used on lock-held skip).
# Leaves an existing marker's failure count untouched.
mark_pending_embedding() {
    local agent="$1"
    [[ -z "$EMBEDDINGS_DIR" ]] && return 0
    local pending_dir
    pending_dir=$(get_pending_dir)
    mkdir -p "$pending_dir"
    local marker="$pending_dir/$agent"
    # Don't resurrect an agent we've already given up on.
    [[ -f "$marker.failed" ]] && return 0
    if [[ ! -f "$marker" ]]; then
        echo "0" > "$marker"
        log "Recorded pending embedding update for agent '$agent' (will retry)"
    fi
}

# Record an embedding FAILURE for an agent and bump its attempt counter. After
# MAX_EMBED_ATTEMPTS the agent is quarantined to <agent>.failed (loud log).
record_embed_failure() {
    local agent="$1"
    [[ -z "$EMBEDDINGS_DIR" ]] && return 0
    local pending_dir
    pending_dir=$(get_pending_dir)
    mkdir -p "$pending_dir"
    local marker="$pending_dir/$agent"

    local attempts=0
    [[ -f "$marker" ]] && attempts=$(cat "$marker" 2>/dev/null || echo 0)
    [[ "$attempts" =~ ^[0-9]+$ ]] || attempts=0
    attempts=$((attempts + 1))

    if [[ "$attempts" -ge "$MAX_EMBED_ATTEMPTS" ]]; then
        log_error "Embedding update for agent '$agent' failed $attempts times; giving up. Marker: $marker.failed (run a manual rebuild and remove it to requeue)."
        echo "$attempts" > "$marker.failed"
        rm -f "$marker"
    else
        echo "$attempts" > "$marker"
        log "Embedding update for agent '$agent' failed (attempt $attempts/$MAX_EMBED_ATTEMPTS); will retry"
    fi
}

# Clear a pending marker after a successful embedding update.
clear_pending_embedding() {
    local agent="$1"
    [[ -z "$EMBEDDINGS_DIR" ]] && return 0
    local pending_dir
    pending_dir=$(get_pending_dir)
    rm -f "$pending_dir/$agent"
}

# Run the incremental embedding update for a single agent. Assumes the caller
# already holds the embedding lock. Returns 0 on success, 1 on failure.
embed_one_agent() {
    local agent="$1"
    log "Updating embeddings for agent: $agent"
    local agent_start
    agent_start=$(date +%s)

    if grounding embeddings --agent "$agent" --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" --out "$EMBEDDINGS_DIR/$agent" --incremental 2>&1 | tee -a "$LOG_FILE"; then
        local agent_elapsed=$(($(date +%s) - agent_start))
        log "Embedding update complete for $agent in ${agent_elapsed}s"
        return 0
    fi
    log "Embedding update failed for agent: $agent"
    return 1
}

# Retry any pending embedding updates (Story 24.4). Drains all active markers
# under one lock; if the lock is held, leaves them for the next tick/batch.
process_pending_embeddings() {
    [[ "${AUTO_EMBEDDINGS}" != "true" ]] && return 0
    [[ -z "$AGENTS_DIR" || -z "$EMBEDDINGS_DIR" || -z "$CORPUS_DIR" ]] && return 0

    local pending_dir
    pending_dir=$(get_pending_dir)
    [[ -d "$pending_dir" ]] || return 0

    # Collect active markers (exclude .failed quarantine markers).
    local markers=()
    local f
    for f in "$pending_dir"/*; do
        [[ -f "$f" ]] || continue
        [[ "$f" == *.failed ]] && continue
        markers+=("$f")
    done
    [[ ${#markers[@]} -eq 0 ]] && return 0

    if ! acquire_embedding_lock; then
        log "Pending embedding retry skipped: lock held by another process"
        return 0
    fi

    log "Retrying ${#markers[@]} pending embedding update(s)"
    for f in "${markers[@]}"; do
        local agent
        agent=$(basename "$f")
        if embed_one_agent "$agent"; then
            clear_pending_embedding "$agent"
        else
            record_embed_failure "$agent"
        fi
    done

    release_embedding_lock
}

# Trigger incremental embedding updates for affected agents
# Any arguments after the collection are the slugs of the documents just
# ingested, forwarded to find_affected_agents so slug-pinned agents are matched.
trigger_embedding_update() {
    local collection="$1"
    shift
    local slugs=("$@")

    # Check if auto-embeddings enabled
    if [[ "${AUTO_EMBEDDINGS}" != "true" ]]; then
        return 0
    fi

    # Check required directories are configured
    if [[ -z "$AGENTS_DIR" || -z "$EMBEDDINGS_DIR" || -z "$CORPUS_DIR" ]]; then
        log "Embedding update skipped: AGENTS_DIR, EMBEDDINGS_DIR, or CORPUS_DIR not configured"
        return 0
    fi

    # Find affected agents
    local affected_agents
    affected_agents=$(find_affected_agents "$collection" "${slugs[@]}")

    if [[ -z "$affected_agents" ]]; then
        log "No agents matched collection '$collection' (slugs: ${slugs[*]:-none}) -- no embedding update"
        return 0
    fi

    # Try to acquire lock. If held, record the work as pending so it isn't
    # dropped -- it will be retried on the next batch / idle tick (Story 24.4).
    if ! acquire_embedding_lock; then
        local agent
        for agent in $affected_agents; do
            mark_pending_embedding "$agent"
        done
        return 0
    fi

    log "Triggering embedding update for agents: $affected_agents"
    local start_time
    start_time=$(date +%s)
    local success_count=0
    local fail_count=0

    # Update each affected agent
    for agent in $affected_agents; do
        if embed_one_agent "$agent"; then
            clear_pending_embedding "$agent"
            ((success_count++)) || true
        else
            record_embed_failure "$agent"
            ((fail_count++)) || true
        fi
    done

    release_embedding_lock

    local total_elapsed=$(($(date +%s) - start_time))
    log "Embedding updates finished: $success_count succeeded, $fail_count failed in ${total_elapsed}s"
}

# --- OCR poison-pill quarantine (Epic 24, Story 24.5 / W8) -------------------
#
# Per-file OCR attempt counters live in skipped/<collection>/.ocr-attempts/ (a
# hidden dir, so it is never matched by the *.pdf backlog glob). After
# MAX_OCR_ATTEMPTS "completed but no output" misses, the PDF is moved to
# skipped/<collection>/quarantine/ (a subdir, also outside the backlog glob) and
# is no longer re-OCR'd, so one poison PDF can't tax every cycle.

get_ocr_attempts_dir() {
    echo "$SKIPPED_DIR/$1/.ocr-attempts"
}

ocr_attempt_count() {
    local collection="$1" filename="$2"
    local counter
    counter="$(get_ocr_attempts_dir "$collection")/$filename"
    local n=0
    [[ -f "$counter" ]] && n=$(cat "$counter" 2>/dev/null || echo 0)
    [[ "$n" =~ ^[0-9]+$ ]] || n=0
    echo "$n"
}

# Increment and return the OCR failure count for a file.
record_ocr_failure() {
    local collection="$1" filename="$2"
    local attempts_dir
    attempts_dir=$(get_ocr_attempts_dir "$collection")
    mkdir -p "$attempts_dir"
    local counter="$attempts_dir/$filename"
    local n
    n=$(ocr_attempt_count "$collection" "$filename")
    n=$((n + 1))
    echo "$n" > "$counter"
    echo "$n"
}

# Move a permanently-failing PDF out of the backlog into quarantine/ and drop
# its attempt counter. Observable: a loud log + an inspectable location.
quarantine_ocr_file() {
    local collection="$1" filepath="$2" filename="$3"
    local quarantine_dir="$SKIPPED_DIR/$collection/quarantine"
    mkdir -p "$quarantine_dir"
    mv "$filepath" "$quarantine_dir/"
    rm -f "$(get_ocr_attempts_dir "$collection")/$filename"
    log_error "OCR gave up on '$filename' after $MAX_OCR_ATTEMPTS attempts; quarantined to $quarantine_dir/ (inspect and move back to retry)."
}

# Clear an OCR attempt counter after a successful OCR.
clear_ocr_attempts() {
    rm -f "$(get_ocr_attempts_dir "$1")/$2"
}

# Process scanned PDFs from skipped directory with OCR
process_ocr_backlog() {
    local collection="$1"
    local skipped_path="$SKIPPED_DIR/$collection"

    # Skip if no skipped directory for this collection
    [[ -d "$skipped_path" ]] || return 0

    # Find actual PDF files (skip HTML junk, .doc, etc.)
    local ocr_files=()
    for f in "$skipped_path"/*.pdf; do
        [[ -f "$f" ]] || continue
        # Verify it's a real PDF, not an HTML error page
        if file "$f" 2>/dev/null | grep -qi "PDF"; then
            ocr_files+=("$f")
        fi
    done

    [[ ${#ocr_files[@]} -eq 0 ]] && return 0

    # Filename cleanup pre-pass: rename messy scanned PDFs in skipped/ so the
    # subsequent OCR ingestion derives a clean slug + chunk source from the
    # start. Mirrors the hook in process_collection() for the fast path.
    if [[ "$CLEAN_FILENAMES" == "true" ]]; then
        local cleaned_files=()
        for f in "${ocr_files[@]}"; do
            local fname
            fname=$(basename "$f")
            local cleanup_result cleanup_status clean_name
            cleanup_result=$("$SCRIPT_DIR/../venv/bin/python" "$SCRIPT_DIR/clean_filename.py" "$fname" 2>>"$LOG_FILE" || echo -e "review\t$fname")
            cleanup_status=$(echo "$cleanup_result" | cut -f1)
            clean_name=$(echo "$cleanup_result" | cut -f2)

            if [[ ( "$cleanup_status" == "auto" || "$cleanup_status" == "clean" ) && "$clean_name" != "$fname" ]]; then
                local new_path="$skipped_path/$clean_name"
                if [[ -e "$new_path" ]]; then
                    log "WARN: OCR rename target exists, processing as-is: $fname"
                    cleaned_files+=("$f")
                else
                    mv -- "$f" "$new_path"
                    log "OCR-path renamed for clarity: $collection/$fname -> $clean_name"
                    cleaned_files+=("$new_path")
                fi
            else
                if [[ "$cleanup_status" == "review" ]]; then
                    log "WARN: OCR filename heuristic uncertain (queued): $collection/$fname"
                    mkdir -p "$(dirname "$REVIEW_QUEUE")"
                    printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$collection" "$fname" "$clean_name" >> "$REVIEW_QUEUE"
                fi
                cleaned_files+=("$f")
            fi
        done
        ocr_files=("${cleaned_files[@]}")
    fi

    log "OCR backlog: ${#ocr_files[@]} scanned PDF(s) in $collection"
    local ocr_start
    ocr_start=$(date +%s)

    # Create temp directory with only the PDF files (avoid processing EPUBs, etc.)
    local temp_dir
    temp_dir=$(mktemp -d)
    for f in "${ocr_files[@]}"; do
        ln -s "$f" "$temp_dir/"
    done

    # Run grounding with OCR on the temp directory (PDFs only)
    if grounding "$temp_dir" "$CORPUS_DIR" --collections "$collection" --ocr on --verbose 2>&1 | tee -a "$LOG_FILE"; then
        # Slugs that actually landed in the corpus, forwarded to the embedding
        # trigger so slug-pinned agents are matched too.
        local ingested_slugs=()

        # Move successfully processed files to originals
        for f in "${ocr_files[@]}"; do
            [[ -f "$f" ]] || continue
            local filename
            filename=$(basename "$f")
            local slug
            slug=$(slugify "$filename")

            if [[ -f "$CORPUS_DIR/$slug/doc.md" ]]; then
                log "OCR succeeded: $filename"
                ingested_slugs+=("$slug")
                clear_ocr_attempts "$collection" "$filename"
                mkdir -p "$ORIGINALS_DIR/$collection"
                mv "$f" "$ORIGINALS_DIR/$collection/"
            else
                # Poison-pill guard (Story 24.5): count the miss and quarantine
                # after MAX_OCR_ATTEMPTS so we don't re-OCR this file forever.
                local attempts
                attempts=$(record_ocr_failure "$collection" "$filename")
                if [[ "$attempts" -ge "$MAX_OCR_ATTEMPTS" ]]; then
                    quarantine_ocr_file "$collection" "$f" "$filename"
                else
                    log "OCR completed but no output for: $filename (attempt $attempts/$MAX_OCR_ATTEMPTS, leaving in skipped)"
                fi
            fi
        done

        local ocr_elapsed=$(($(date +%s) - ocr_start))
        log "OCR backlog complete for $collection in ${ocr_elapsed}s"

        # Trigger embedding updates for OCR'd documents
        trigger_embedding_update "$collection" "${ingested_slugs[@]}"
    else
        local ocr_elapsed=$(($(date +%s) - ocr_start))
        log_error "OCR backlog failed for $collection after ${ocr_elapsed}s (files remain in skipped)"
    fi

    # Clean up temp directory
    rm -rf "$temp_dir"
}

process_collection() {
    local collection_dir="$1"
    local collection
    collection=$(basename "$collection_dir")

    # Skip hidden directories like .stfolder
    [[ "$collection" == .* ]] && return 0

    # Defer if scripts/reprocess.sh holds this collection's lock.
    # The lock file is dropped by reprocess.sh into staging/<collection>/
    # before it touches any doc in that collection and removed after.
    # We leave staged files in place; reprocess.sh re-fires them via
    # rename-rename so the watcher picks them up on lock release.
    if [[ -f "$collection_dir/.reprocess-lock" ]]; then
        log "Reprocess lock held for collection $collection; deferring"
        return 0
    fi

    local needs_processing=()
    local processed_count=0
    local skipped_count=0
    local unsupported_count=0
    # Slugs of every document this pass put into (or confirmed in) the corpus.
    # Passed to trigger_embedding_update so agents that pin a slug in
    # corpus_filter.slugs are matched, not just agents declaring the collection.
    local ingested_slugs=()

    log "Checking collection: $collection"

    # First pass: handle all files
    for doc_path in "$collection_dir"/*; do
        [[ -f "$doc_path" ]] || continue

        local filename
        filename=$(basename "$doc_path")

        # Filename cleanup: confident transformations get applied in-place
        # before grounding sees the file, so the resulting slug + chunk
        # source: come out clean. The mv re-fires inotify and we'll pick up
        # the clean name on the next pass. Uncertain ("review") cases fall
        # through with the original name and get logged for hand-correction.
        if [[ "$CLEAN_FILENAMES" == "true" ]]; then
            local cleanup_result cleanup_status clean_name
            cleanup_result=$("$SCRIPT_DIR/../venv/bin/python" "$SCRIPT_DIR/clean_filename.py" "$filename" 2>>"$LOG_FILE" || echo -e "review\t$filename")
            cleanup_status=$(echo "$cleanup_result" | cut -f1)
            clean_name=$(echo "$cleanup_result" | cut -f2)

            if [[ ( "$cleanup_status" == "auto" || "$cleanup_status" == "clean" ) && "$clean_name" != "$filename" ]]; then
                local new_path="$collection_dir/$clean_name"
                if [[ -e "$new_path" ]]; then
                    log "WARN: rename target exists, processing with original name: $filename"
                else
                    mv -- "$doc_path" "$new_path"
                    log "Renamed for clarity: $collection/$filename -> $clean_name"
                    # Skip this iteration; the mv fires moved_to and the next
                    # collection scan will see the clean name and process it.
                    continue
                fi
            elif [[ "$cleanup_status" == "review" ]]; then
                log "WARN: filename heuristic uncertain (queued for review): $collection/$filename"
                mkdir -p "$(dirname "$REVIEW_QUEUE")"
                printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$collection" "$filename" "$clean_name" >> "$REVIEW_QUEUE"
                # Fall through; process with the original messy name.
            fi
        fi

        local slug
        slug=$(slugify "$filename")

        # Check file extension
        local ext="${filename##*.}"
        ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

        # Check if already in corpus
        if [[ -f "$CORPUS_DIR/$slug/doc.md" ]]; then
            log "Already processed: $filename -> moving to originals"
            # Counted as ingested even though nothing is re-chunked: an agent
            # that pins this slug may still be missing it (the pin was added
            # after ingestion, or an earlier run predates slug matching). The
            # incremental run is a no-op when the index is current, so this
            # makes re-dropping a file the recovery path for a missed pin.
            ingested_slugs+=("$slug")
            mkdir -p "$ORIGINALS_DIR/$collection"
            mv "$doc_path" "$ORIGINALS_DIR/$collection/"
            ((processed_count++)) || true
            continue
        fi

        # Handle by format
        case "$ext" in
            pdf)
                # Check if PDF needs OCR
                if ! has_extractable_text "$doc_path"; then
                    log "Needs OCR (scanned): $filename -> moving to skipped"
                    mkdir -p "$SKIPPED_DIR/$collection"
                    mv "$doc_path" "$SKIPPED_DIR/$collection/"
                    ((skipped_count++)) || true
                    continue
                fi
                # Queue for grounding processing
                needs_processing+=("$doc_path")
                ;;
            epub)
                # Queue for grounding processing
                needs_processing+=("$doc_path")
                ;;
            md|docx|doc)
                # Process with ingest_docs.py
                log "Processing document: $filename"
                if "$SCRIPT_DIR/../venv/bin/python" "$INGEST_DOCS" "$doc_path" "$CORPUS_DIR" --collections "$collection" 2>&1 | tee -a "$LOG_FILE"; then
                    log "Successfully processed: $filename"
                    ingested_slugs+=("$slug")
                    mkdir -p "$ORIGINALS_DIR/$collection"
                    mv "$doc_path" "$ORIGINALS_DIR/$collection/"
                    ((processed_count++)) || true
                else
                    log "Failed to process: $filename -> moving to skipped"
                    mkdir -p "$SKIPPED_DIR/$collection"
                    mv "$doc_path" "$SKIPPED_DIR/$collection/"
                    ((skipped_count++)) || true
                fi
                ;;
            *)
                # Unsupported format
                log "Unsupported format: $filename -> moving to skipped"
                mkdir -p "$SKIPPED_DIR/$collection"
                mv "$doc_path" "$SKIPPED_DIR/$collection/"
                ((unsupported_count++)) || true
                ;;
        esac
    done

    log "Collection $collection: $processed_count done, $skipped_count OCR, $unsupported_count unsupported, ${#needs_processing[@]} to process"

    # If there are files to process, run grounding
    if [[ ${#needs_processing[@]} -gt 0 ]]; then
        log "Processing ${#needs_processing[@]} new files in $collection"

        # Run grounding on the collection directory
        local output_file
        output_file=$(mktemp)

        grounding "$collection_dir" "$CORPUS_DIR" --collections "$collection" --ocr off --verbose 2>&1 | tee "$output_file" || true

        # Move newly processed files to originals
        for doc_path in "${needs_processing[@]}"; do
            [[ -f "$doc_path" ]] || continue

            local filename
            filename=$(basename "$doc_path")
            local slug
            slug=$(slugify "$filename")

            if [[ -f "$CORPUS_DIR/$slug/doc.md" ]]; then
                log "Successfully processed: $filename"
                ingested_slugs+=("$slug")
                mkdir -p "$ORIGINALS_DIR/$collection"
                mv "$doc_path" "$ORIGINALS_DIR/$collection/"
            else
                log "Failed to process: $filename -> moving to skipped"
                mkdir -p "$SKIPPED_DIR/$collection"
                mv "$doc_path" "$SKIPPED_DIR/$collection/"
            fi
        done

        rm -f "$output_file"
    fi

    # Trigger embedding updates for affected agents. The slugs cover agents that
    # pin this document rather than declaring its collection.
    trigger_embedding_update "$collection" "${ingested_slugs[@]}"

    log "Finished collection: $collection"
}

process_existing() {
    log "Checking for existing documents in staging..."

    # Pull latest agent definitions before processing
    pull_latest_repo

    # Process each collection directory
    for collection_dir in "$STAGING_DIR"/*/; do
        [[ -d "$collection_dir" ]] || continue
        process_collection "$collection_dir"
        # Process any scanned PDFs that were just skipped
        process_ocr_backlog "$(basename "$collection_dir")"
    done

    # Retry any embedding updates left pending from a prior run (Story 24.4).
    process_pending_embeddings
}

# --- Startup-race reconciliation (Story 24.6 / W9) ---------------------------
#
# Defense-in-depth backstop for the inotify monitor. Re-scans every collection
# that currently has staged files and processes anything not yet in the corpus.
# Idempotent: process_collection() skips slugs already ingested and moves
# already-processed sources to originals/, so a redundant pass is harmless.
#
# Runs on the inotify idle tick (see watch_staging). Guarantees eventual pickup
# of any file an inotify event was ever missed for -- a file delivered before
# recursive watches were fully established, or one lost to an inotify queue
# overflow under load. Bounded worst-case latency is PENDING_RETRY_INTERVAL.
reconcile_staging() {
    local collection_dir
    for collection_dir in "$STAGING_DIR"/*/; do
        [[ -d "$collection_dir" ]] || continue

        # Only touch collections that actually have files, so idle ticks over a
        # mostly-empty staging tree stay quiet (no per-collection log spam).
        local has_file=false f
        for f in "$collection_dir"/*; do
            [[ -f "$f" ]] && { has_file=true; break; }
        done
        [[ "$has_file" == true ]] || continue

        log "Reconciliation: staged files present in $(basename "$collection_dir"), processing"
        process_collection "$collection_dir"
        # Process any scanned PDFs that were just skipped.
        process_ocr_backlog "$(basename "$collection_dir")"
    done
}

watch_staging() {
    log "Starting watcher on: $STAGING_DIR"
    log "Corpus output: $CORPUS_DIR"
    log "Originals archive: $ORIGINALS_DIR"
    log "Skipped (OCR needed): $SKIPPED_DIR"

    # Startup-race fix (Story 24.6): start the live monitor BEFORE the startup
    # scan so files delivered during the scan -- which on a large corpus runs
    # 50+ minutes (OCR backlog + per-agent embedding rebuilds) -- are never
    # missed. Previously process_existing() ran to completion first, leaving a
    # long blind window with no monitor and a folder glob snapshotted at t=0;
    # anything Syncthing delivered mid-scan was invisible to both mechanisms and
    # silently dropped.
    #
    # Mechanism: inotifywait -m is already running when process_existing() runs
    # as the first loop iteration, so every event that fires during the scan is
    # buffered in the pipe + kernel inotify queue and drained by the loop the
    # instant the scan returns. The one-time `sleep 2` lets recursive watches
    # finish establishing before the scan snapshots the folder list, so the
    # pre-watch gap is closed too (scan catches pre-watch arrivals; inotify
    # catches during-scan arrivals -- the union has no hole). The idle-tick
    # reconciliation below is the backstop for any residual race or queue
    # overflow.
    local startup_scan_done=false

    # Watch for close_write (direct writes) and moved_to (Syncthing atomic
    # writes). `read -t` gives the loop a periodic idle tick so pending embedding
    # updates are retried even when no new files arrive (Story 24.4 AC2): an event
    # drives a normal batch; a timeout drives a pending-retry + reconciliation pass.
    inotifywait -m -r -e close_write -e moved_to --format '%w%f' "$STAGING_DIR" 2>/dev/null | while true; do
        # First iteration: run the startup scan with the monitor already live.
        if [[ "$startup_scan_done" != true ]]; then
            # Let inotifywait finish establishing recursive watches before the
            # scan snapshots staging/*/ (closes the pre-watch gap; the idle-tick
            # reconciliation covers the rare case watches take longer than this).
            sleep 2
            process_existing
            log "Initial processing complete. Watching for new files..."
            startup_scan_done=true
            continue
        fi

        if read -t "$PENDING_RETRY_INTERVAL" -r filepath; then
            # Only process supported formats in collection subfolders (depth 2)
            if [[ "$filepath" =~ \.(pdf|epub|md|docx|doc)$ ]] && [[ "$filepath" =~ ^$STAGING_DIR/[^/]+/[^/]+\.(pdf|epub|md|docx|doc)$ ]]; then
                # Wait for file to stabilize
                sleep 3

                # Pull latest agent definitions before processing
                pull_latest_repo

                local collection_dir
                collection_dir=$(dirname "$filepath")
                process_collection "$collection_dir"
                # Process any scanned PDFs that were just skipped
                process_ocr_backlog "$(basename "$collection_dir")"
                # Drain any embedding updates left pending this batch (Story 24.4).
                process_pending_embeddings
            fi
        else
            local read_rc=$?
            if [[ "$read_rc" -gt 128 ]]; then
                # read timed out (inotify idle): periodic pending-retry tick plus
                # a reconciliation rescan so any file an inotify event was ever
                # missed for is still ingested (Story 24.6 defense in depth).
                process_pending_embeddings
                reconcile_staging
            else
                # EOF: the inotifywait stream closed (it exited). Stop the loop.
                log_error "inotifywait stream closed; watcher loop exiting"
                break
            fi
        fi
    done
}

# Ensure directories exist
mkdir -p "$STAGING_DIR" "$CORPUS_DIR" "$ORIGINALS_DIR" "$SKIPPED_DIR"

# Trap for clean shutdown
trap 'log "Watcher stopped"; exit 0' SIGINT SIGTERM

log "=== grounding staging watcher starting ==="
watch_staging
