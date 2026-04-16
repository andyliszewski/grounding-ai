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

    # Find agents that have the given collection in their corpus_filter.collections
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
            if grep -q "^corpus_filter:" "$agent_file" 2>/dev/null; then
                local in_collections=false
                while IFS= read -r line; do
                    if [[ "$line" =~ ^[[:space:]]*collections: ]]; then
                        in_collections=true
                        continue
                    fi
                    if [[ "$in_collections" == true ]]; then
                        if [[ "$line" =~ ^[[:space:]]+-[[:space:]]* ]]; then
                            local coll_value
                            coll_value=$(echo "$line" | sed 's/^[[:space:]]*-[[:space:]]*//' | tr -d '[:space:]')
                            if [[ "$coll_value" == "$collection" ]]; then
                                affected+=("$agent_name")
                                break
                            fi
                        else
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

    # Try to acquire embedding lock
    acquire_embedding_lock() {
        local lock_file
        lock_file=$(get_lock_file)

        [[ -z "$EMBEDDINGS_DIR" ]] && return 1

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
            return 0
        fi

        log "Triggering embedding update for agents: $affected_agents"
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

test_acquire_lock_success() {
    acquire_embedding_lock
    local result=$?

    assert_equals "0" "$result" || return 1
    assert_file_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock"
}

test_acquire_lock_blocked() {
    # Create existing lock
    echo "12345" > "$TEST_EMBEDDINGS_DIR/_embeddings.lock"

    acquire_embedding_lock
    local result=$?

    assert_equals "1" "$result"
}

test_acquire_lock_stale_removed() {
    # Create old lock file
    echo "12345" > "$TEST_EMBEDDINGS_DIR/_embeddings.lock"
    touch -d "2 hours ago" "$TEST_EMBEDDINGS_DIR/_embeddings.lock"

    acquire_embedding_lock
    local result=$?

    assert_equals "0" "$result"
}

test_release_lock() {
    acquire_embedding_lock
    assert_file_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock" || return 1

    release_embedding_lock
    assert_file_not_exists "$TEST_EMBEDDINGS_DIR/_embeddings.lock"
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
run_test "acquire_lock - success" test_acquire_lock_success
run_test "acquire_lock - blocked" test_acquire_lock_blocked
run_test "acquire_lock - stale removed" test_acquire_lock_stale_removed
run_test "release_lock" test_release_lock
run_test "trigger_embedding - disabled" test_trigger_embedding_disabled
run_test "trigger_embedding - no agents dir" test_trigger_embedding_no_agents_dir
run_test "trigger_embedding - no matching agents" test_trigger_embedding_no_matching_agents

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
