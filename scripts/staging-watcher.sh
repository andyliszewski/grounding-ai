#!/usr/bin/env bash
#
# staging-watcher.sh - Monitor staging folder and process documents into corpus
#
# Deployment: Linux ingestion machine only
# Dependencies: inotifywait (inotify-tools), grounding, pdftotext, python-docx, git
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

# Embedding configuration
AUTO_EMBEDDINGS="${AUTO_EMBEDDINGS:-false}"
AGENTS_DIR="${AGENTS_DIR:-}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-3600}"  # 1 hour default

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

# Find agents that have the given collection in their corpus_filter.collections
# Returns space-separated list of agent names
find_affected_agents() {
    local collection="$1"
    local affected=()

    # Skip if AGENTS_DIR not configured
    [[ -z "$AGENTS_DIR" || ! -d "$AGENTS_DIR" ]] && return 0

    for agent_file in "$AGENTS_DIR"/*.yaml; do
        [[ -f "$agent_file" ]] || continue

        local agent_name
        agent_name=$(basename "$agent_file" .yaml)

        # Check if agent has corpus_filter.collections containing this collection
        # Parse YAML: look for "collections:" section and check for collection name
        if grep -q "^corpus_filter:" "$agent_file" 2>/dev/null; then
            # Extract collections section (lines after "collections:" until next non-list-item)
            local in_collections=false
            while IFS= read -r line; do
                if [[ "$line" =~ ^[[:space:]]*collections: ]]; then
                    in_collections=true
                    continue
                fi
                if [[ "$in_collections" == true ]]; then
                    # Check if still in list (line starts with whitespace and -)
                    if [[ "$line" =~ ^[[:space:]]+-[[:space:]]* ]]; then
                        # Extract collection value (remove leading "- " and whitespace)
                        local coll_value
                        coll_value=$(echo "$line" | sed 's/^[[:space:]]*-[[:space:]]*//' | tr -d '[:space:]')
                        if [[ "$coll_value" == "$collection" ]]; then
                            affected+=("$agent_name")
                            break
                        fi
                    else
                        # No longer in collections list
                        break
                    fi
                fi
            done < "$agent_file"
        fi
    done

    echo "${affected[*]}"
}

# Lock file for embedding updates
get_lock_file() {
    echo "${EMBEDDINGS_DIR}/_embeddings.lock"
}

# Try to acquire embedding lock. Returns 0 if acquired, 1 if locked.
acquire_embedding_lock() {
    local lock_file
    lock_file=$(get_lock_file)

    # Skip if EMBEDDINGS_DIR not configured
    [[ -z "$EMBEDDINGS_DIR" ]] && return 1

    # Ensure embeddings directory exists
    mkdir -p "$EMBEDDINGS_DIR"

    if [[ -f "$lock_file" ]]; then
        local lock_age
        lock_age=$(($(date +%s) - $(stat -c %Y "$lock_file")))
        if [[ "$lock_age" -gt "$LOCK_TIMEOUT" ]]; then
            log "Stale embedding lock detected (${lock_age}s old), removing"
            rm -f "$lock_file"
        else
            log "Embedding update skipped: lock held by another process"
            return 1
        fi
    fi

    # Write PID to lock file
    echo "$$" > "$lock_file"
    return 0
}

# Release embedding lock
release_embedding_lock() {
    local lock_file
    lock_file=$(get_lock_file)
    rm -f "$lock_file"
}

# Trigger incremental embedding updates for affected agents
trigger_embedding_update() {
    local collection="$1"

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
    affected_agents=$(find_affected_agents "$collection")

    if [[ -z "$affected_agents" ]]; then
        log "No agents found matching collection: $collection"
        return 0
    fi

    # Try to acquire lock
    if ! acquire_embedding_lock; then
        return 0
    fi

    log "Triggering embedding update for agents: $affected_agents"
    local start_time
    start_time=$(date +%s)
    local success_count=0
    local fail_count=0

    # Update each affected agent
    for agent in $affected_agents; do
        log "Updating embeddings for agent: $agent"
        local agent_start
        agent_start=$(date +%s)

        if grounding embeddings --agent "$agent" --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" --out "$EMBEDDINGS_DIR/$agent" --incremental 2>&1 | tee -a "$LOG_FILE"; then
            local agent_elapsed=$(($(date +%s) - agent_start))
            log "Embedding update complete for $agent in ${agent_elapsed}s"
            ((success_count++)) || true
        else
            log "Embedding update failed for agent: $agent"
            ((fail_count++)) || true
        fi
    done

    release_embedding_lock

    local total_elapsed=$(($(date +%s) - start_time))
    log "Embedding updates finished: $success_count succeeded, $fail_count failed in ${total_elapsed}s"
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
        # Move successfully processed files to originals
        for f in "${ocr_files[@]}"; do
            [[ -f "$f" ]] || continue
            local filename
            filename=$(basename "$f")
            local slug
            slug=$(slugify "$filename")

            if [[ -f "$CORPUS_DIR/$slug/doc.md" ]]; then
                log "OCR succeeded: $filename"
                mkdir -p "$ORIGINALS_DIR/$collection"
                mv "$f" "$ORIGINALS_DIR/$collection/"
            else
                log "OCR completed but no output for: $filename (leaving in skipped)"
            fi
        done

        local ocr_elapsed=$(($(date +%s) - ocr_start))
        log "OCR backlog complete for $collection in ${ocr_elapsed}s"

        # Trigger embedding updates for OCR'd documents
        trigger_embedding_update "$collection"
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

    local needs_processing=()
    local processed_count=0
    local skipped_count=0
    local unsupported_count=0

    log "Checking collection: $collection"

    # First pass: handle all files
    for doc_path in "$collection_dir"/*; do
        [[ -f "$doc_path" ]] || continue

        local filename
        filename=$(basename "$doc_path")
        local slug
        slug=$(slugify "$filename")

        # Check file extension
        local ext="${filename##*.}"
        ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

        # Check if already in corpus
        if [[ -f "$CORPUS_DIR/$slug/doc.md" ]]; then
            log "Already processed: $filename -> moving to originals"
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

    # Trigger embedding updates for affected agents
    trigger_embedding_update "$collection"

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
}

watch_staging() {
    log "Starting watcher on: $STAGING_DIR"
    log "Corpus output: $CORPUS_DIR"
    log "Originals archive: $ORIGINALS_DIR"
    log "Skipped (OCR needed): $SKIPPED_DIR"

    # Process any existing files first
    process_existing

    log "Initial processing complete. Watching for new files..."

    # Monitor for new files
    # Watch for close_write (direct writes) and moved_to (Syncthing atomic writes)
    inotifywait -m -r -e close_write -e moved_to --format '%w%f' "$STAGING_DIR" 2>/dev/null | while read -r filepath; do
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
        fi
    done
}

# Ensure directories exist
mkdir -p "$STAGING_DIR" "$CORPUS_DIR" "$ORIGINALS_DIR" "$SKIPPED_DIR"

# Trap for clean shutdown
trap 'log "Watcher stopped"; exit 0' SIGINT SIGTERM

log "=== grounding staging watcher starting ==="
watch_staging
