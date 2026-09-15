#!/usr/bin/env bash
#
# test_watcher_embeddings.sh - Integration tests for watcher embedding functionality
#
# Run with: bash tests/test_watcher_embeddings.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WATCHER_SCRIPT="$PROJECT_DIR/scripts/staging-watcher.sh"

# Test fixtures
TEST_TMP_DIR=""
TEST_AGENTS_DIR=""
TEST_EMBEDDINGS_DIR=""
TEST_CORPUS_DIR=""
TEST_LOG_FILE=""

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

setup() {
    # Create temporary directories for tests
    TEST_TMP_DIR=$(mktemp -d)
    TEST_AGENTS_DIR="$TEST_TMP_DIR/agents"
    TEST_EMBEDDINGS_DIR="$TEST_TMP_DIR/embeddings"
    TEST_CORPUS_DIR="$TEST_TMP_DIR/corpus"
    TEST_LOG_FILE="$TEST_TMP_DIR/test.log"

    mkdir -p "$TEST_AGENTS_DIR" "$TEST_EMBEDDINGS_DIR" "$TEST_CORPUS_DIR"

    # Create test agent YAML files
    cat > "$TEST_AGENTS_DIR/scientist.yaml" << 'EOF'
name: scientist
description: Scientific research agent

corpus_filter:
  collections:
    - science
    - biology
    - chemistry
EOF

    cat > "$TEST_AGENTS_DIR/ceo.yaml" << 'EOF'
name: ceo
description: Executive agent

corpus_filter:
  collections:
    - business
    - strategy
EOF

    cat > "$TEST_AGENTS_DIR/data-scientist.yaml" << 'EOF'
name: data-scientist
description: Data science agent

corpus_filter:
  collections:
    - science
    - data
    - statistics
EOF

    # Flow-style list (Story 24.3 / W6): the old bash matcher returned zero
    # matches for this perfectly-valid YAML form.
    cat > "$TEST_AGENTS_DIR/flow-agent.yaml" << 'EOF'
name: flow-agent
description: Flow-style collections

corpus_filter:
  collections: [physics, astronomy]
EOF

    # Quoted block entries (Story 24.3 / W6): the old bash matcher kept the
    # quotes after tr and failed the equality check.
    cat > "$TEST_AGENTS_DIR/quoted-agent.yaml" << 'EOF'
name: quoted-agent
description: Quoted collections

corpus_filter:
  collections:
    - "geology"
    - 'meteorology'
EOF

    # Export environment for watcher functions
    export STAGING_DIR="$TEST_TMP_DIR/staging"
    export CORPUS_DIR="$TEST_CORPUS_DIR"
    export ORIGINALS_DIR="$TEST_TMP_DIR/originals"
    export SKIPPED_DIR="$TEST_TMP_DIR/skipped"
    export LOG_FILE="$TEST_LOG_FILE"
    export AGENTS_DIR="$TEST_AGENTS_DIR"
    export EMBEDDINGS_DIR="$TEST_EMBEDDINGS_DIR"
    export LOCK_TIMEOUT=5
    export AUTO_EMBEDDINGS="false"
    export MAX_EMBED_ATTEMPTS=3
    export PENDING_RETRY_INTERVAL=300
    export MAX_OCR_ATTEMPTS=3

    mkdir -p "$STAGING_DIR" "$ORIGINALS_DIR" "$SKIPPED_DIR"

    # Touch log file
    touch "$TEST_LOG_FILE"
}

teardown() {
    if [[ -n "$TEST_TMP_DIR" && -d "$TEST_TMP_DIR" ]]; then
        rm -rf "$TEST_TMP_DIR"
    fi
}

# Define the watcher functions inline (extracted from watcher script)
define_watcher_functions() {
    # Log function
    log() {
        echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
    }

    log_error() {
        echo "[$(date -Iseconds)] ERROR: $*" | tee -a "$LOG_FILE" >&2
    }

    # Find agents whose corpus_filter.collections include the given collection.
    # (Mirror of scripts/staging-watcher.sh; Epic 24, Story 24.3.) Delegates to
    # the real scripts/match_agents.py so the test exercises the actual YAML
    # parser, not a bash reimplementation.
    find_affected_agents() {
        local collection="$1"
        shift
        local slugs=("$@")

        [[ -z "$AGENTS_DIR" || ! -d "$AGENTS_DIR" ]] && return 0

        local python_bin="$PROJECT_DIR/venv/bin/python"
        [[ -x "$python_bin" ]] || python_bin="python3"

        local slug_args=()
        local s
        for s in "${slugs[@]}"; do
            [[ -n "$s" ]] && slug_args+=(--slug "$s")
        done

        "$python_bin" "$PROJECT_DIR/scripts/match_agents.py" \
            --agents-dir "$AGENTS_DIR" --collection "$collection" \
            "${slug_args[@]}" 2>>"$LOG_FILE" || true
    }

    # Lock file for embedding updates
    get_lock_file() {
        echo "${EMBEDDINGS_DIR}/_embeddings.lock"
    }

    # Try to acquire the embedding lock via flock on a held file descriptor.
    # (Mirror of scripts/staging-watcher.sh; Epic 24, Story 24.1.)
    acquire_embedding_lock() {
        local lock_file
        lock_file=$(get_lock_file)

        [[ -z "$EMBEDDINGS_DIR" ]] && return 1

        mkdir -p "$EMBEDDINGS_DIR"

        exec 9>"$lock_file"

        if flock -n 9; then
            return 0
        fi

        log "Embedding update skipped: lock held by another process"
        exec 9>&- 2>/dev/null || true
        return 1
    }

    # Release the embedding lock by closing the held descriptor (no rm).
    release_embedding_lock() {
        flock -u 9 2>/dev/null || true
        exec 9>&- 2>/dev/null || true
    }

    # --- Pending-embedding retry queue (Epic 24, Story 24.4) ---
    get_pending_dir() {
        echo "${EMBEDDINGS_DIR}/pending-embeddings"
    }

    mark_pending_embedding() {
        local agent="$1"
        [[ -z "$EMBEDDINGS_DIR" ]] && return 0
        local pending_dir
        pending_dir=$(get_pending_dir)
        mkdir -p "$pending_dir"
        local marker="$pending_dir/$agent"
        [[ -f "$marker.failed" ]] && return 0
        if [[ ! -f "$marker" ]]; then
            echo "0" > "$marker"
            log "Recorded pending embedding update for agent '$agent' (will retry)"
        fi
    }

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

    clear_pending_embedding() {
        local agent="$1"
        [[ -z "$EMBEDDINGS_DIR" ]] && return 0
        local pending_dir
        pending_dir=$(get_pending_dir)
        rm -f "$pending_dir/$agent"
    }

    # Run the incremental embedding update for one agent (caller holds the lock).
    # Tests override this to simulate success/failure without invoking grounding.
    embed_one_agent() {
        local agent="$1"
        log "Updating embeddings for agent: $agent"
        if grounding embeddings --agent "$agent" --corpus "$CORPUS_DIR" --agents-dir "$AGENTS_DIR" --out "$EMBEDDINGS_DIR/$agent" --incremental 2>&1 | tee -a "$LOG_FILE"; then
            log "Embedding update complete for $agent"
            return 0
        fi
        log "Embedding update failed for agent: $agent"
        return 1
    }

    process_pending_embeddings() {
        [[ "${AUTO_EMBEDDINGS}" != "true" ]] && return 0
        [[ -z "$AGENTS_DIR" || -z "$EMBEDDINGS_DIR" || -z "$CORPUS_DIR" ]] && return 0

        local pending_dir
        pending_dir=$(get_pending_dir)
        [[ -d "$pending_dir" ]] || return 0

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

    # --- OCR poison-pill quarantine helpers (Epic 24, Story 24.5) ---
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

    quarantine_ocr_file() {
        local collection="$1" filepath="$2" filename="$3"
        local quarantine_dir="$SKIPPED_DIR/$collection/quarantine"
        mkdir -p "$quarantine_dir"
        mv "$filepath" "$quarantine_dir/"
        rm -f "$(get_ocr_attempts_dir "$collection")/$filename"
        log_error "OCR gave up on '$filename' after $MAX_OCR_ATTEMPTS attempts; quarantined to $quarantine_dir/ (inspect and move back to retry)."
    }

    clear_ocr_attempts() {
        rm -f "$(get_ocr_attempts_dir "$1")/$2"
    }

    # Trigger incremental embedding updates for affected agents
    trigger_embedding_update() {
        local collection="$1"

        if [[ "${AUTO_EMBEDDINGS}" != "true" ]]; then
            return 0
        fi

        if [[ -z "$AGENTS_DIR" || -z "$EMBEDDINGS_DIR" || -z "$CORPUS_DIR" ]]; then
            log "Embedding update skipped: AGENTS_DIR, EMBEDDINGS_DIR, or CORPUS_DIR not configured"
            return 0
        fi

        local affected_agents
        affected_agents=$(find_affected_agents "$collection")

        if [[ -z "$affected_agents" ]]; then
            log "No agents found matching collection: $collection"
            return 0
        fi

        if ! acquire_embedding_lock; then
            local agent
            for agent in $affected_agents; do
                mark_pending_embedding "$agent"
            done
            return 0
        fi

        log "Triggering embedding update for agents: $affected_agents"
        local agent
        for agent in $affected_agents; do
            if embed_one_agent "$agent"; then
                clear_pending_embedding "$agent"
            else
                record_embed_failure "$agent"
            fi
        done
        release_embedding_lock
    }
}

assert_equals() {
    local expected="$1"
    local actual="$2"

    if [[ "$expected" == "$actual" ]]; then
        return 0
    else
        echo "  Expected: '$expected'"
        echo "  Actual:   '$actual'"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"

    if [[ "$haystack" == *"$needle"* ]]; then
        return 0
    else
        echo "  Expected to contain: '$needle'"
        echo "  Actual: '$haystack'"
        return 1
    fi
}

assert_file_exists() {
    local filepath="$1"
    if [[ -f "$filepath" ]]; then
        return 0
    else
        echo "  File does not exist: $filepath"
        return 1
    fi
}

assert_file_not_exists() {
    local filepath="$1"
    if [[ ! -f "$filepath" ]]; then
        return 0
    else
        echo "  File should not exist: $filepath"
        return 1
    fi
}

run_test() {
    local test_name="$1"
    local test_func="$2"

    ((TESTS_RUN++)) || true
    echo -n "Running: $test_name... "

    setup
    define_watcher_functions

    if $test_func 2>/dev/null; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++)) || true
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++)) || true
    fi

    teardown
}

# ============================================================================
# Test Cases
# ============================================================================

test_find_affected_agents_single_match() {
    local result
    result=$(find_affected_agents "science")

    # Should match scientist and data-scientist
    assert_contains "$result" "scientist" || return 1
    assert_contains "$result" "data-scientist" || return 1
}

test_find_affected_agents_no_match() {
    local result
    result=$(find_affected_agents "nonexistent-collection")

    assert_equals "" "$result"
}

test_find_affected_agents_business_only() {
    local result
    result=$(find_affected_agents "business")

    # Should only match ceo
    assert_equals "ceo" "$result"
}

test_find_affected_agents_no_agents_dir() {
    export AGENTS_DIR=""
    local result
    result=$(find_affected_agents "science")

    assert_equals "" "$result"
}

# Story 24.3 / W6: flow-style list must match (old bash matcher missed it).
test_find_affected_agents_flow_style() {
    local result
    result=$(find_affected_agents "physics")

    assert_contains "$result" "flow-agent"
}

# Slug-pin axis: an agent that reaches a doc through corpus_filter.slugs must be
# matched, even when no agent declares the doc's collection. Before this, the
# collection was the only axis and such a doc silently never triggered an update.
test_find_affected_agents_slug_pin() {
    cat > "$TEST_AGENTS_DIR/pinner.yaml" << 'EOF'
name: pinner
corpus_filter:
  collections:
    - strategy
  slugs:
    - surveillance-capitalism-zuboff
EOF

    # Collection axis alone: nothing declares elite-power.
    local result
    result=$(find_affected_agents "elite-power")
    assert_equals "" "$result" || return 1

    # With the ingested slug, the pinning agent is matched.
    result=$(find_affected_agents "elite-power" "surveillance-capitalism-zuboff")
    assert_equals "pinner" "$result"
}

# A batch matches on either axis, and an agent hit by both is listed once.
test_find_affected_agents_slug_and_collection_union() {
    cat > "$TEST_AGENTS_DIR/pinner.yaml" << 'EOF'
name: pinner
corpus_filter:
  collections:
    - strategy
  slugs:
    - surveillance-capitalism-zuboff
EOF

    local result
    result=$(find_affected_agents "business" "surveillance-capitalism-zuboff")
    assert_contains "$result" "ceo" || return 1
    assert_contains "$result" "pinner" || return 1

    # 'strategy' is declared by both ceo and pinner, and pinner is also hit by
    # the slug axis. It must still be listed exactly once (one embedding run).
    result=$(find_affected_agents "strategy" "surveillance-capitalism-zuboff")
    assert_contains "$result" "ceo" || return 1
    local pinner_count
    pinner_count=$(echo "$result" | grep -c '^pinner$')
    if [[ "$pinner_count" -ne 1 ]]; then
        echo "  pinner listed $pinner_count times, expected exactly 1"
        return 1
    fi
}

# Story 24.3 / W6: quoted entries must match (old bash matcher missed them).
test_find_affected_agents_quoted() {
    local result
    result=$(find_affected_agents "geology")

    assert_contains "$result" "quoted-agent" || return 1

    # Single-quoted entry too.
    result=$(find_affected_agents "meteorology")
    assert_contains "$result" "quoted-agent"
}

# Start a background process that holds the embedding flock until killed.
# Sets the global HOLDER_PID. The process replaces itself with `sleep` via exec
# so that the lock-holding PID is the one we kill (no orphaned child keeps FD 9).
# stdout/stderr go to /dev/null so the lock-holder doesn't keep a capture pipe
# open (which would deadlock a command substitution around this call).
HOLDER_PID=""
_start_lock_holder() {
    local lock_file="$TEST_EMBEDDINGS_DIR/_embeddings.lock"
    mkdir -p "$TEST_EMBEDDINGS_DIR"
    bash -c "exec 9>'$lock_file'; flock -n 9 || exit 1; exec sleep 30" >/dev/null 2>&1 &
    HOLDER_PID=$!
    sleep 0.5  # let the holder take the lock
}

_stop_lock_holder() {
    [[ -n "$HOLDER_PID" ]] || return 0
    kill -9 "$HOLDER_PID" 2>/dev/null
    wait "$HOLDER_PID" 2>/dev/null
    HOLDER_PID=""
}

test_acquire_lock_success() {
    acquire_embedding_lock
    local result=$?

    assert_equals "0" "$result" || { release_embedding_lock; return 1; }
    # flock opens (creates) the lock file as a rendezvous point.
    assert_file_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock" || { release_embedding_lock; return 1; }
    release_embedding_lock
}

# AC 2 / AC 4: a lock held by another process blocks acquisition (skip-and-log,
# never a steal). flock is the real mutual-exclusion primitive.
test_acquire_lock_blocked() {
    _start_lock_holder

    acquire_embedding_lock
    local result=$?

    _stop_lock_holder

    assert_equals "1" "$result"
}

# AC 7: release drops the flock but does NOT remove the rendezvous file, and a
# subsequent acquire succeeds.
test_release_lock() {
    acquire_embedding_lock || return 1
    release_embedding_lock

    # File persists (flock no longer unlinks it)...
    assert_file_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock" || return 1

    # ...and the lock is re-acquirable after release.
    acquire_embedding_lock
    local result=$?
    release_embedding_lock
    assert_equals "0" "$result"
}

# AC 5: a holder killed mid-update (kill -9 / OOM / systemd restart) leaves no
# stranded lock; the kernel releases the flock and the next acquire succeeds
# immediately, with no timeout heuristic.
test_kill_holder_releases_lock() {
    _start_lock_holder

    # Sanity: lock is currently held, so our acquire is denied.
    acquire_embedding_lock
    local blocked=$?
    if [[ "$blocked" -ne 1 ]]; then
        _stop_lock_holder
        echo "  Expected acquire to be blocked while holder alive"
        return 1
    fi

    # Kill the holder; kernel drops the flock.
    _stop_lock_holder
    sleep 0.2

    acquire_embedding_lock
    local result=$?
    release_embedding_lock
    assert_equals "0" "$result"
}

test_trigger_embedding_disabled() {
    export AUTO_EMBEDDINGS="false"

    trigger_embedding_update "science"

    # No lock file should be created
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock"
}

test_trigger_embedding_no_agents_dir() {
    export AUTO_EMBEDDINGS="true"
    export AGENTS_DIR=""

    trigger_embedding_update "science"

    # Should skip and log
    assert_contains "$(cat "$TEST_LOG_FILE")" "not configured" || return 1
}

test_trigger_embedding_no_matching_agents() {
    export AUTO_EMBEDDINGS="true"

    trigger_embedding_update "nonexistent-collection"

    # Should log no agents found
    assert_contains "$(cat "$TEST_LOG_FILE")" "No agents found matching collection"
}

# --- Story 24.4: retry of skipped/failed embedding updates -------------------

# AC 1: a lock-held skip records the affected agents as pending (work not
# dropped) and does NOT count as a failure.
test_lock_held_records_pending() {
    export AUTO_EMBEDDINGS="true"
    _start_lock_holder  # another process holds the embedding lock

    trigger_embedding_update "science"

    _stop_lock_holder

    # scientist is one of the agents matching "science"; it must be recorded.
    assert_file_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist" || return 1
    # Lock-held skip is attempt count 0 (not a failure).
    assert_equals "0" "$(cat "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist")"
}

# AC 1/2/5: a pending update is retried and, on success, becomes searchable
# (marker cleared).
test_pending_retry_success() {
    export AUTO_EMBEDDINGS="true"
    mkdir -p "$TEST_EMBEDDINGS_DIR/pending-embeddings"
    echo "0" > "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist"

    # Simulate a successful embedding without invoking grounding.
    embed_one_agent() { return 0; }

    process_pending_embeddings

    # Marker cleared -> agent embedded.
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist"
}

# AC 3/5: a persistently failing agent is retried a bounded number of times,
# then given up on loudly with a .failed marker.
test_pending_retry_bounded_then_loud() {
    export AUTO_EMBEDDINGS="true"
    export MAX_EMBED_ATTEMPTS=2
    mkdir -p "$TEST_EMBEDDINGS_DIR/pending-embeddings"
    echo "0" > "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist"

    # Simulate a permanently failing embedding.
    embed_one_agent() { return 1; }

    # First retry: attempt 1 (< MAX) -> still pending, no .failed yet.
    process_pending_embeddings
    assert_file_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist" || return 1
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist.failed" || return 1

    # Second retry: attempt 2 (>= MAX) -> quarantined.
    process_pending_embeddings
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist" || return 1
    assert_file_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist.failed" || return 1
    # Loud final log.
    assert_contains "$(cat "$TEST_LOG_FILE")" "giving up"
}

# AC 3: a quarantined (.failed) agent is not resurrected by a later lock-held
# skip.
test_failed_agent_not_resurrected() {
    export AUTO_EMBEDDINGS="true"
    mkdir -p "$TEST_EMBEDDINGS_DIR/pending-embeddings"
    echo "3" > "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist.failed"

    mark_pending_embedding "scientist"

    # No active marker should be (re)created for a given-up agent.
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist"
}

# process_pending_embeddings is a no-op when nothing is pending.
test_pending_noop_when_empty() {
    export AUTO_EMBEDDINGS="true"
    embed_one_agent() { echo "embed_one_agent should not be called"; return 1; }

    process_pending_embeddings

    # No lock taken, no error.
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/pending-embeddings/scientist"
}

# --- Story 24.5: OCR poison-pill quarantine ---------------------------------

# AC 1/4: OCR misses are counted; a transient miss is not quarantined on the
# first attempt.
test_ocr_attempts_counted_and_cleared() {
    local coll="science" fn="x.pdf"
    assert_equals "0" "$(ocr_attempt_count "$coll" "$fn")" || return 1

    record_ocr_failure "$coll" "$fn" >/dev/null
    assert_equals "1" "$(ocr_attempt_count "$coll" "$fn")" || return 1
    record_ocr_failure "$coll" "$fn" >/dev/null
    assert_equals "2" "$(ocr_attempt_count "$coll" "$fn")" || return 1

    # A success clears the counter.
    clear_ocr_attempts "$coll" "$fn"
    assert_equals "0" "$(ocr_attempt_count "$coll" "$fn")"
}

# AC 1/3/4/5: a PDF that always yields empty OCR output is counted up to
# MAX_OCR_ATTEMPTS, then quarantined with a loud log. This mirrors the exact
# decision process_ocr_backlog makes on each "completed but no output" miss.
test_ocr_poison_pill_quarantined_after_max() {
    export MAX_OCR_ATTEMPTS=2
    local coll="science" fn="poison.pdf"
    mkdir -p "$SKIPPED_DIR/$coll"
    : > "$SKIPPED_DIR/$coll/$fn"

    # Attempt 1: below MAX -> stays in skipped.
    local n
    n=$(record_ocr_failure "$coll" "$fn")
    [[ "$n" -ge "$MAX_OCR_ATTEMPTS" ]] && { echo "  quarantined too early"; return 1; }
    assert_file_exists "$SKIPPED_DIR/$coll/$fn" || return 1

    # Attempt 2: reaches MAX -> quarantine.
    n=$(record_ocr_failure "$coll" "$fn")
    if [[ "$n" -ge "$MAX_OCR_ATTEMPTS" ]]; then
        quarantine_ocr_file "$coll" "$SKIPPED_DIR/$coll/$fn" "$fn"
    fi

    assert_file_not_exists "$SKIPPED_DIR/$coll/$fn" || return 1
    assert_file_exists "$SKIPPED_DIR/$coll/quarantine/$fn" || return 1
    assert_contains "$(cat "$TEST_LOG_FILE")" "quarantined"
}

# AC 2: a quarantined PDF (in the quarantine/ subdir) is NOT picked up by the
# backlog's *.pdf glob, so it is never re-OCR'd.
test_quarantined_excluded_from_backlog_glob() {
    local coll="science"
    mkdir -p "$SKIPPED_DIR/$coll/quarantine"
    : > "$SKIPPED_DIR/$coll/quarantine/poison.pdf"
    mkdir -p "$(get_ocr_attempts_dir "$coll")"  # hidden dir, also excluded

    # Mirror the backlog's collection step.
    local found=()
    local f
    for f in "$SKIPPED_DIR/$coll"/*.pdf; do
        [[ -f "$f" ]] && found+=("$f")
    done

    assert_equals "0" "${#found[@]}"
}

# ============================================================================
# Main
# ============================================================================

echo "=================================="
echo "Watcher Embedding Integration Tests"
echo "=================================="
echo ""

run_test "find_affected_agents - single match" test_find_affected_agents_single_match
run_test "find_affected_agents - no match" test_find_affected_agents_no_match
run_test "find_affected_agents - business only" test_find_affected_agents_business_only
run_test "find_affected_agents - no agents dir" test_find_affected_agents_no_agents_dir
run_test "find_affected_agents - flow style (W6)" test_find_affected_agents_flow_style
run_test "find_affected_agents - quoted entries (W6)" test_find_affected_agents_quoted
run_test "find_affected_agents - slug pin matches" test_find_affected_agents_slug_pin
run_test "find_affected_agents - slug + collection union" test_find_affected_agents_slug_and_collection_union
run_test "acquire_lock - success" test_acquire_lock_success
run_test "acquire_lock - blocked by holder (flock)" test_acquire_lock_blocked
run_test "release_lock - drops flock, keeps file, reacquirable" test_release_lock
run_test "kill holder releases lock (no stranded lock)" test_kill_holder_releases_lock
run_test "trigger_embedding - disabled" test_trigger_embedding_disabled
run_test "trigger_embedding - no agents dir" test_trigger_embedding_no_agents_dir
run_test "trigger_embedding - no matching agents" test_trigger_embedding_no_matching_agents
run_test "pending - lock-held records pending (W7)" test_lock_held_records_pending
run_test "pending - retry success clears marker (W7)" test_pending_retry_success
run_test "pending - bounded retries then loud give-up (W7)" test_pending_retry_bounded_then_loud
run_test "pending - failed agent not resurrected (W7)" test_failed_agent_not_resurrected
run_test "pending - no-op when empty (W7)" test_pending_noop_when_empty
run_test "ocr - attempts counted and cleared (W8)" test_ocr_attempts_counted_and_cleared
run_test "ocr - poison pill quarantined after max (W8)" test_ocr_poison_pill_quarantined_after_max
run_test "ocr - quarantined excluded from backlog glob (W8)" test_quarantined_excluded_from_backlog_glob

echo ""
echo "=================================="
echo "Results: $TESTS_PASSED/$TESTS_RUN passed"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "${RED}$TESTS_FAILED test(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
