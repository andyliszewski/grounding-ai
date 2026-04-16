# Technical Debt

**Last Updated**: 2026-04-15 (TD-004 added)

This document tracks implementation-level technical debt and optimization opportunities that don't warrant full epic planning. For larger initiatives, see `docs/ROADMAP.md` and `docs/epics/`.

---

## TD-001: Incremental Embedding Updates

**Status**: Resolved (2026-04-13)
**Priority**: P2 (Medium)
**Added**: 2026-01-09
**Resolved**: 2026-04-13
**Related**: Stories 14.1, 14.2, 14.3

### Resolution

Shipped as part of the v0.3 retrieval layer:

- `--incremental` flag on `grounding embeddings` appends new vectors to an existing FAISS index and tombstones deleted documents.
- The staging watcher (`scripts/staging-watcher.sh`) auto-triggers per-agent incremental updates when `AUTO_EMBEDDINGS=true`, using a `_embeddings.lock` file to prevent concurrent runs.
- `_chunk_map.json` v1.1 carries `doc_id` for filtering and tombstone handling.

See `CLAUDE.md` ("Embedding Generation" and "Integration with Watcher") for the user-facing contract.

---

## TD-002: nDCG@10 Ideal-DCG Normalization

**Status**: Open
**Priority**: P3 (Low)
**Added**: 2026-04-15
**Related**: Story 17.4 dev notes; Story 18.4 rerank comparison

### Current Limitation

The `ndcg_at_10` metric emitted by `grounding eval` can exceed `1.0`. For
example, the mini corpus baseline reports `ndcg_at_10 ≈ 1.421`. nDCG is
mathematically bounded by `[0, 1]`, so any value above 1 indicates the
normalization is wrong, not that the retrieval is "better than perfect."

### Technical Analysis

nDCG = DCG / IDCG. The current implementation computes DCG correctly but
constructs the ideal DCG (IDCG) without accounting for the case where
multiple relevant chunks in the top-k belong to the same expected `doc_id`.
When several chunks from one hit all count as relevant, the numerator
sums contributions that the denominator does not cap, producing values
> 1.

This does not affect the CI gate's load-bearing metrics (`recall@k`,
`MRR`, `citation_accuracy`) and does not affect the Story 18.4 flip
decision — the flip rule keys on `recall@5`, which is correctly bounded.

### Proposed Solution

Normalize IDCG against the set of expected `doc_ids` with de-duplication,
or switch the metric to a chunk-level evaluation that treats each
expected `(doc_id, chunk_id)` tuple as atomic. Either approach keeps
nDCG@10 in `[0, 1]`.

### Complexity Estimate

Small: isolated to the eval metric computation; fixture set is already in
place; regression test is straightforward.

### Workaround

Current docs note the quirk and instruct readers to interpret nDCG@10
directionally (bigger = better) rather than as an absolute quality
number. Flip decisions and refreshes use `recall@k` and
`citation_accuracy` instead.

---

## TD-003: Rerank Default Flip Decision (Deferred)

**Status**: Open
**Priority**: P2 (Medium)
**Added**: 2026-04-15
**Related**: Epic 18 (Story 18.4 measured outcome)

### Current Limitation

Epic 18 shipped cross-encoder reranking end-to-end but left the default at
`retrieval.rerank.enabled: false`. The flip decision rule (Story 18.4 AC#4)
requires at least one corpus tier to show `recall@5` lift ≥ 0.03 AND
`citation_accuracy` non-decreasing. The mini corpus is saturation-limited
(all metrics already at 1.000), and no private-corpus fixture set with
≥ 10 items existed at epic close to measure lift on realistic query
diversity.

### Technical Analysis

The plumbing is shipped and proven. The only missing input is a measurement
against realistic queries. Published reranker results suggest a 10–20%
relative `recall@5` lift is typical on bi-encoder baselines, which would
clear the 0.03 absolute threshold at any plausible baseline value. The
measurement has to happen in `grounding-ai-private` / `my-agents` since
real corpus content cannot live in the public repo.

### Proposed Solution

Scoped refresh PR when a private fixture set lands:

1. Build a ≥ 10-item fixture set against one real agent (e.g.,
   `data-scientist`) in `grounding-ai-private`.
2. Run `grounding eval` with rerank off and on; capture aggregate numbers.
3. Apply the AC#4 rule. If lift clears 0.03 and `citation_accuracy`
   non-decreasing: PR flips `config.example.yaml` to
   `retrieval.rerank.enabled: true`, regenerates the mini baseline, and
   updates `docs/eval/README.md#reranking-comparison-story-184` with the
   private-corpus row.
4. PR title follows Epic 16 refresh discipline (`eval: flip rerank default
   — <rationale>`), description includes before/after aggregates.

### Complexity Estimate

Low: no new code. ~1 hour to build the fixture set if one doesn't exist,
~15 minutes to run the comparison and prepare the PR.

### Workaround

Users who want reranking today can turn it on explicitly via `--rerank` or
`retrieval.rerank.enabled: true` in their local `config.yaml`. The default
only affects the out-of-box experience.

---

## TD-004: Hybrid Default Flip Decision (Deferred)

**Status**: Open
**Priority**: P2 (Medium)
**Added**: 2026-04-15
**Related**: Epic 19 (Story 19.4 measured outcome)

### Current Limitation

Epic 19 shipped hybrid retrieval (BM25 + dense via RRF) end-to-end but
left the default at `retrieval.hybrid.enabled: false`. The flip decision
rule (Story 19.4 AC#4) requires at least one corpus tier to show
`recall@5` lift ≥ 0.03 (hybrid-on vs hybrid-off, rerank held at its
current default) AND `citation_accuracy` non-decreasing. The mini corpus
is saturation-limited (same constraint that blocked TD-003's rerank
flip — all metrics already at 1.000), and no ≥ 10-item private-corpus
fixture set existed at epic close to measure lift on realistic query
diversity.

### Technical Analysis

The plumbing is shipped and proven. The only missing input is a
measurement against realistic queries. Published BM25+dense hybrid
results suggest a 5–15% relative `recall@5` lift is typical on
dense-only baselines, which would clear the 0.03 absolute threshold at
any plausible baseline value. The measurement has to happen in
`grounding-ai-private` / `my-agents` since real corpus content cannot
live in the public repo.

### Proposed Solution

Scoped refresh PR when a private fixture set lands (reuses TD-003's
fixture set if TD-003 is resolved in the interim; both follow-ups
unblock each other):

1. Use a ≥ 10-item fixture set against one real agent (e.g.,
   `data-scientist`) in `grounding-ai-private`.
2. Run `grounding eval` in the four-cell matrix; capture aggregate
   numbers.
3. Apply the AC#4 rule on `{hh}` vs `{Hh}` (rerank held at current
   default). If lift clears 0.03 and `citation_accuracy` non-decreasing:
   PR flips `config.example.yaml` to `retrieval.hybrid.enabled: true`,
   regenerates the mini baseline, and updates the hybrid comparison
   section in `docs/eval/README.md` with the private-corpus row.
4. PR title follows Epic 16 refresh discipline (`eval: flip hybrid
   default — <rationale>`), description includes before/after
   aggregates.

### Re-measurement Command Set

Pinned to commit `f38eb44` at the time TD-004 was opened. Update the
SHA when the measurement runs.

```bash
# {hybrid off, rerank off} — baseline
grounding eval --agent <private-agent> \
    --agents-dir <private-agents-dir> \
    --fixtures <private-fixture>.yaml \
    --corpus <private-corpus> \
    --embeddings <private-embeddings-dir> \
    --out /tmp/eval-hybrid-refresh-hh/

# {hybrid on, rerank off}
grounding eval --agent <private-agent> \
    --agents-dir <private-agents-dir> \
    --fixtures <private-fixture>.yaml \
    --corpus <private-corpus> \
    --embeddings <private-embeddings-dir> \
    --hybrid --hybrid-pool-size 50 \
    --out /tmp/eval-hybrid-refresh-Hh/

# {hybrid off, rerank on}
grounding eval --agent <private-agent> \
    --agents-dir <private-agents-dir> \
    --fixtures <private-fixture>.yaml \
    --corpus <private-corpus> \
    --embeddings <private-embeddings-dir> \
    --rerank --rerank-pool-size 50 \
    --out /tmp/eval-hybrid-refresh-hH/

# {hybrid on, rerank on}
grounding eval --agent <private-agent> \
    --agents-dir <private-agents-dir> \
    --fixtures <private-fixture>.yaml \
    --corpus <private-corpus> \
    --embeddings <private-embeddings-dir> \
    --hybrid --hybrid-pool-size 50 \
    --rerank --rerank-pool-size 50 \
    --out /tmp/eval-hybrid-refresh-HH/
```

### Complexity Estimate

Low: no new code. ~1 hour to build the fixture set if one doesn't
exist (free-ride if TD-003's fixture set lands first), ~15 minutes to
run the comparison and prepare the PR.

### Workaround

Users who want hybrid retrieval today can turn it on explicitly via
`--hybrid` or `retrieval.hybrid.enabled: true` in their local
`config.yaml`. The default only affects the out-of-box experience.

---

## Template for New Items

```markdown
## TD-XXX: Title

**Status**: Open | In Progress | Resolved
**Priority**: P1 (High) | P2 (Medium) | P3 (Low)
**Added**: YYYY-MM-DD
**Related**: Epic/Story reference if applicable

### Current Limitation
Description of the problem.

### Technical Analysis
Technical details and root cause.

### Proposed Solution
Implementation approach.

### Complexity Estimate
Effort assessment.

### Workaround
Current mitigation if any.
```
