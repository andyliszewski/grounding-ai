# Epic 18: Cross-Encoder Reranking

**Epic ID:** E18
**Owner:** Andy
**Status:** Shipped
**Priority:** P1
**Completed Stories:** 4/4
**Dependencies:** Epic 6 (Vector Embeddings), Epic 16 (Evaluation Harness)
**Target Completion:** Shipped 2026-04-15

---

## Measured Outcome (Story 18.4)

Mini-corpus eval run on public-repo commit `f38eb44`, 2026-04-15 UTC:

- **Rerank off (baseline):** `recall@1 = recall@5 = recall@10 = MRR = 1.000`, `citation_accuracy = 1.000`, `n_items = 3`.
- **Rerank on (bge-reranker-base, pool=50):** identical — all metrics unchanged.

The mini corpus is a 3-document, 5-chunk smoke test; recall is already
saturated at `1.000` and reranking cannot lift a maxed score. No private
fixture set with ≥ 10 items against the maintainer's `grounding-ai-private`
/ `my-agents` corpus was committed at the time of this epic close, so the
"realistic query diversity" measurement is deferred.

**Flip decision:** the AC#4 rule requires at least one corpus tier to show
`recall@5` lift ≥ 0.03 AND `citation_accuracy` non-decreasing. Mini shows
zero lift (saturation); private is N/A. Default stays
`retrieval.rerank.enabled: false`. Plumbing is shipped end-to-end and
opt-in via `--rerank` on both CLIs plus `rerank_enabled` on the MCP tool;
the default will flip in a follow-up story once a private fixture set
lands and shows the expected 10–20% lift on realistic queries.

Latency on the measurement host (Apple Silicon, 16 cores, 48 GB RAM,
torch CPU backend): no rerank ≈ 28 ms, `bge-reranker-base` pool=50 ≈ 1.8 s,
pool=100 ≈ 3.4 s. Full comparison table and interpretation in
`docs/eval/README.md#reranking-comparison-story-184`.

---

## Overview

Add a cross-encoder reranking stage between FAISS retrieval and result formatting. The existing bi-encoder (`all-MiniLM-L6-v2`) is fast but approximate; a small cross-encoder re-scoring the top-N FAISS results typically yields a 10–20% retrieval quality lift at modest CPU cost. Epic 16's harness will quantify the lift precisely, Epic 17's citation metric will verify that reranking doesn't degrade page/section accuracy.

**Problem Statement:**
- Bi-encoders embed query and document independently, so they score surface similarity rather than true relevance.
- A query like "what does the paper say about transformer scaling *below* 1B params" can rank a chunk about 1B+ params higher because the words overlap more.
- Cross-encoders read query + candidate together and assign a true relevance score; they're accurate but expensive, so they only work as a second-stage re-rank over a small pool.
- Without reranking, the project is leaving a well-documented quality improvement on the table.

**Solution:**
- Add a `grounding/reranker.py` module wrapping a small cross-encoder (default `BAAI/bge-reranker-base`).
- Change retrieval to a two-stage flow: FAISS returns top-N candidates (default N=50), cross-encoder rescores, top-k are returned.
- Make reranking opt-in via config and CLI flag in the first release, then flip the default once Epic 16 numbers show the lift is real.
- Extend the eval harness to compare rerank-on vs rerank-off runs and commit the measured lift as part of the epic-close PR.

---

## Goals

1. Cross-encoder reranking is available end-to-end through `SearchCorpusTool`, the MCP server, and `local_rag.py`.
2. Default configuration can be flipped with a single setting; opt-in is the initial default so the feature ships without forcing a baseline reset.
3. Reranker model is lazy-loaded, CPU-friendly, and cached across queries within a process.
4. Retrieval pool size and final top-k are independently configurable.
5. Epic 16 eval harness measures the lift; baseline refresh captures new numbers only after the feature is flipped to default-on.
6. Epic 17 `citation_accuracy` stays intact under reranking (reranking changes order, not which chunks are returned first-page-wise).

---

## Non-Goals

- GPU acceleration for the reranker (Tier-3 roadmap item).
- Training or fine-tuning rerankers on this project's corpora.
- Multi-model ensemble reranking.
- Query rewriting before retrieval (separate Tier-2 item).
- Reranking at index-build time (only at query time).

---

## Architecture

```
User Query
   │
   ▼
┌────────────────────────────┐
│ Bi-encoder query embedding │  existing: grounding/embedder.py
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ FAISS top-N (default N=50) │  existing: grounding/vector_store.py
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Cross-encoder rerank       │  NEW: grounding/reranker.py
│ (query, chunk_text) pairs  │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Take top-k (default k=10)  │
└─────────────┬──────────────┘
              ▼
    ┌──────────────────┐
    │ Citation prefix  │  existing: grounding/citations.py
    └─────────┬────────┘
              ▼
         Retrieval output
```

### Data Flow

Reranking operates on `List[RetrievedChunk]` (the type established in Story 16.2 and extended in 17.4). The reranker receives the query string plus the chunk bodies, produces a new score per chunk, and returns the list re-sorted. The `RetrievedChunk.score` field is replaced with the cross-encoder score (original FAISS distance is preserved in a new `faiss_distance` field so eval reports can inspect both).

### Configuration Shape

```yaml
# config.yaml (optional; falls back to CLI flags / defaults)
retrieval:
  rerank:
    enabled: false                        # default off in 18.1-18.3; flipped to true in 18.4
    model: "BAAI/bge-reranker-base"       # sentence-transformers-compatible
    pool_size: 50                         # N candidates from FAISS before rerank
    batch_size: 16                        # cross-encoder batch size
```

---

## Stories Breakdown

### Story 18.1: Reranker Module

- Create `grounding/reranker.py` with a `CrossEncoderReranker` class wrapping a sentence-transformers `CrossEncoder`.
- Lazy-load the model on first call; cache in a module-level singleton keyed on model name.
- Expose a pure `rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]` method that preserves `RetrievedChunk` identity but returns a re-ordered list with updated `score` and new `faiss_distance`.
- Add `RerankConfig` dataclass (`model`, `pool_size`, `batch_size`, `enabled`).
- Unit tests with a stub `CrossEncoder` that returns deterministic scores; no real model download in tests.

**AC:**
- `rerank()` is deterministic given identical inputs and model.
- `rerank()` on an empty list returns an empty list without loading the model.
- Model is loaded at most once per process per model name.
- CPU-only; no CUDA assumptions.
- Tests pass in under 2s without network access.

**Status:** Done

### Story 18.2: Wire into Retrieval Paths

- Extend `SearchCorpusTool` (`scripts/search_corpus_tool.py`) to optionally apply the reranker after FAISS.
- Extend MCP server (`mcp_servers/corpus_search/server.py`) with the same capability.
- Extend `local_rag.py` for parity.
- Two-stage flow: FAISS returns `pool_size` results, reranker rescores, tool returns top `top_k`.
- Config resolution order: explicit CLI flag → `config.yaml` → default `enabled=False`.

**AC:**
- `SearchCorpusTool.execute(...)` accepts `rerank_config: RerankConfig | None` kwarg; when present and enabled, applies reranking.
- MCP server honors a `rerank_enabled` input parameter on the tool schema.
- `local_rag.py --rerank` CLI flag turns it on for that session.
- When reranking is disabled, behavior is bit-for-bit identical to pre-epic output (no regressions in existing tests).
- Citation prefix (Epic 17) continues to reflect the re-ranked top chunk.

**Status:** Done

### Story 18.3: CLI, MCP Schema, and Config Plumbing

- Add `--rerank`, `--rerank-model`, `--rerank-pool-size`, `--rerank-top-k` flags to `grounding eval` (16.3) and `scripts/local_rag.py`.
- Extend the MCP corpus-search tool schema with `rerank_enabled` and document in `mcp_servers/corpus_search/README.md` (or equivalent).
- Extend `config.yaml.example` with the `retrieval.rerank` block and document defaults.
- Document the two-stage flow and per-query latency expectations in `CLAUDE.md`.

**AC:**
- All retrieval-adjacent CLIs expose consistent rerank flags; flag names match across surfaces.
- `grounding eval --rerank` produces an eval run with reranking applied; reports note `rerank.enabled=True` in the JSON header.
- MCP tool schema validates; clients that don't send `rerank_enabled` get default behavior.
- Help text and docs describe the latency trade-off (typical: 200–500ms per query on CPU for pool_size=50).

**Status:** Done

### Story 18.4: Measure the Lift, Flip the Default, Refresh Baseline

- Run eval with rerank on and rerank off against the mini corpus; record numbers.
- If rerank produces a statistically meaningful lift without regressing `citation_accuracy`, flip the default to `enabled=true`.
- Refresh `docs/eval/baselines/mini-corpus.json` to reflect the new defaults.
- Extend `docs/eval/README.md` with a "Reranking" section showing the measured lift.
- Update ROADMAP: Tier 1 #1 moves to Shipped; Tier 1 #2 (hybrid retrieval) called out as unblocked.
- Open a tracking issue for the known Tier-3 GPU-acceleration follow-up.

**AC:**
- Eval run comparison table committed to `docs/eval/README.md` showing `recall@5`, `MRR`, `nDCG@10`, `citation_accuracy` with and without reranking on the mini corpus.
- If the lift is real and `citation_accuracy` non-decreasing: default flipped, baseline refreshed in the same PR with rationale in PR description (per Epic 16 refresh discipline).
- If the lift is **not** real on the mini corpus: default stays off, epic still closes with the plumbing shipped; README documents the result and notes that real-corpus eval (private) may tell a different story.
- CI gate green on the merge commit.
- ROADMAP updated.

**Status:** Done

---

## Technical Details

### Model Choice

- **Default: `BAAI/bge-reranker-base`** (~278MB, ~300ms/query on 8-core CPU for pool_size=50). Good balance of quality and latency.
- Alternative: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB, ~100ms/query, slightly lower quality).
- Alternative: `BAAI/bge-reranker-large` (~1.3GB, ~800ms/query, best quality). Document as opt-in for users willing to pay latency.
- Users swap via `--rerank-model` or `config.yaml`.

### Why `sentence-transformers` CrossEncoder

Already in the project's dependency tree for bi-encoder embeddings. No new runtime dependency. `CrossEncoder(model_name_or_path)` handles download, caching (same `~/.cache/huggingface` path Epic 16's CI already warms), and `.predict([(query, doc), ...])` returns scores.

### Latency Budget

| Configuration | Typical CPU latency per query |
|---------------|-------------------------------|
| No rerank (baseline) | ~20ms (FAISS only) |
| bge-reranker-base, pool=50 | ~300ms |
| bge-reranker-base, pool=100 | ~600ms |
| bge-reranker-large, pool=50 | ~800ms |

Document these in `CLAUDE.md` so users making interactive tools can size pool_size appropriately.

### Determinism and Caching

- Cross-encoder forward pass is deterministic on CPU.
- Module-level singleton cache keyed on model name keeps the model in memory across queries within a process (MCP server, agentic loop).
- The watcher / CLI one-shot runs re-pay the load cost per invocation; acceptable.

### Invariants

- Reranking operates purely on chunk **body text**, never on front matter. Citations, page numbers, section headings are unchanged by reranking.
- Reranking does **not** filter: output length equals `min(top_k, len(input))`. Pool size only affects what's considered, not what's returned.
- FAISS distance is preserved (as `faiss_distance`) on each `RetrievedChunk` so eval reports can show both scores.

---

## Dependencies

### Epic Dependencies
- **Epic 6** — FAISS retrieval is the first stage of the two-stage flow.
- **Epic 16** — eval harness measures the lift; `citation_accuracy` from 17.4 guards against unintended citation regressions.
- **Epic 17** — citation prefix layer sits *after* reranking and is unchanged.

### External Dependencies
- `sentence-transformers` (already present) — `CrossEncoder` class.
- Cross-encoder model weights (downloaded on first use; cached in `~/.cache/huggingface`).

### Code Dependencies
- `grounding/vector_store.py` — first-stage retrieval.
- `grounding/embedder.py` — query embedding.
- `scripts/search_corpus_tool.py`, `mcp_servers/corpus_search/server.py`, `scripts/local_rag.py` — integration points.
- `grounding/eval/runner.py`, `grounding/eval/report.py` — rerank-aware eval reporting (18.3).

---

## Implementation Order

```
Story 18.1 (Reranker Module)
    └── Pure module + tests; fully testable in isolation.

Story 18.2 (Wire into Retrieval Paths)
    └── Integrates 18.1 into three retrieval surfaces; opt-in.

Story 18.3 (CLI, MCP Schema, Config Plumbing)
    └── Makes the feature usable end-to-end from every surface.

Story 18.4 (Measure + Flip + Refresh Baseline)
    └── The measurement story; Epic 16's harness earns its keep here.
```

Stories are sequential. 18.1 and 18.2 could be collapsed if the reranker module is trivial, but recommend keeping them separate so the pure module has its own focused test suite.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Model download fails in CI | Medium | Cache same as Epic 16 (`~/.cache/huggingface`). Ship a stub model for tests so unit tests never hit the network. |
| Reranker slows MCP responses past interactive threshold | Medium | Document per-pool-size latency; default pool_size=50 keeps typical queries under 500ms on CPU. |
| Reranker changes retrieval order enough to move wrong chunks to top, dropping `citation_accuracy` | Medium | 18.4 measures this explicitly; `citation_accuracy` gate is a pre-flip requirement. |
| Cross-encoder model size bloats CI runtime or disk | Low | Default `base` model is ~278MB; cache survives between runs. `large` variant is opt-in, not CI. |
| Non-determinism across machines (CPU vs GPU, PyTorch versions) | Low | Project is CPU-only; document PyTorch version pin already in `pyproject.toml`. |
| Flip-to-default changes every downstream user's retrieval behavior silently | Medium | 18.4's flip PR needs reviewer sign-off with before/after eval numbers; Epic 16 refresh discipline handles this. |

---

## Testing Strategy

### Unit Tests
- Reranker module (18.1): stub CrossEncoder, determinism, empty-input guard, singleton caching.
- Config plumbing (18.3): flag parsing, config.yaml resolution order.

### Integration Tests
- `SearchCorpusTool.execute(...)` with rerank enabled and disabled, asserting output ordering differs (or is identical when all chunks are equally relevant).
- MCP server `call_tool` with `rerank_enabled=True`.
- Eval run with `--rerank` against the mini corpus; verify report shows `rerank.enabled=True`.

### Manual Validation
- Run rerank-on eval against a real private-repo corpus; spot-check 5 queries where pre-rerank top-1 differed from post-rerank top-1.
- Measure per-query latency on a real CPU-only workstation; confirm within documented budget.

---

## Acceptance Criteria (Epic Level)

1. Cross-encoder reranking available end-to-end through all three retrieval surfaces.
2. Opt-in via config / CLI flag in 18.1–18.3; default decision made in 18.4 based on measured lift.
3. Mini-corpus eval comparison (on vs off) committed to `docs/eval/README.md`.
4. `citation_accuracy` from Epic 17 non-decreasing with reranking enabled (gate).
5. Epic 16 CI gate stays green on the merge commit.
6. ROADMAP updated; Tier 1 #2 (hybrid retrieval) explicitly called out as unblocked.

---

## Definition of Done

- All four stories closed with AC met.
- If flipped: baseline refreshed with reviewed PR per Epic 16 discipline.
- Comparison table + "Reranking" section committed to `docs/eval/README.md`.
- `CLAUDE.md` documents latency trade-offs and config surface.
- ROADMAP Tier 1 #1 moved to Shipped.

---

## Future Enhancements (Out of Scope)

- GPU acceleration (Tier 3 deferred).
- Query rewriting before retrieval (Tier 2).
- Multi-model ensemble reranking.
- Learning-to-rerank using the eval harness's fixture set as training data.
- Per-agent reranker selection (different agents using different reranker models).

---

## References

- `docs/ROADMAP.md` — Tier 1 #1 (this epic).
- `docs/epics/epic-16-evaluation-harness.md` — measurement dependency.
- `docs/epics/epic-17-page-and-section-citations.md` — citation_accuracy gate for rerank flips.
- `grounding/embedder.py`, `grounding/vector_store.py` — first-stage retrieval.
- `scripts/search_corpus_tool.py`, `mcp_servers/corpus_search/server.py`, `scripts/local_rag.py` — integration points.
- Sentence-Transformers CrossEncoder docs: https://www.sbert.net/docs/cross_encoder/usage/usage.html
- BGE reranker model cards: https://huggingface.co/BAAI/bge-reranker-base
