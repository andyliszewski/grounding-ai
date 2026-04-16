# Epic 19: Hybrid Retrieval (BM25 + Dense)

**Epic ID:** E19
**Owner:** Andy
**Status:** Shipped
**Priority:** P1
**Completed Stories:** 4/4
**Dependencies:** Epic 6 (Vector Embeddings), Epic 16 (Evaluation Harness), Epic 18 (Cross-Encoder Reranking)
**Target Completion:** 2026-04-15

---

## Measured Outcome (Story 19.4)

Four-cell `{hybrid off/on} × {rerank off/on}` eval matrix at public-repo
commit `f38eb44` (2026-04-15):

- **Mini corpus (public, n=3):** all four cells return identical numbers
  — `recall@{1,5,10} = MRR = citation_accuracy = 1.000`. Saturation-
  limited; the mini corpus is a pipeline smoke test, not a retrieval
  discriminator. Hybrid cannot lift a score already at 1.000.
- **Private corpus (maintainer):** N/A — no ≥ 10-item fixture set
  committed to `grounding-ai-private` / `my-agents` at epic close.
  Mirrors Story 18.4's outcome; accepted per 19.4 Dev Notes.
- **Flip decision: NO FLIP.** Neither tier meets the `recall@5 lift
  ≥ 0.03 AND citation_accuracy non-decreasing` rule.
  `retrieval.hybrid.enabled: false` stays in `config.example.yaml`;
  the mini-corpus baseline is untouched. TD-004 tracks the follow-up
  measurement once a private fixture set lands.

**Four-cell latency (Apple Silicon, 16k-chunk private-agent index, warm):**

| Cell | median per-query | notes |
|------|-----------------:|-------|
| `{hh}` dense only                   | ~7 ms    | FAISS dominates |
| `{Hh}` hybrid (BM25+dense+RRF)      | ~240 ms  | BM25 tokenize+rank adds the gap |
| `{hH}` dense+rerank                 | ~1300 ms | cross-encoder on 50 pairs dominates |
| `{HH}` hybrid+rerank                | ~1580 ms | rerank + ~240 ms BM25 overhead |

See `docs/eval/README.md#hybrid-retrieval-comparison-story-194` for the
full comparison matrix, decision justification, and HybridProvenance
shape verification.

---

## Overview

Add a BM25 lexical retrieval channel alongside the existing FAISS dense channel, merge candidates from both, then (optionally) pass the merged pool through the Epic 18 cross-encoder reranker. Dense embeddings excel at paraphrase and concept matching but miss rare names, code identifiers, API names, acronyms, and exact-phrase queries — the places where users most often know exactly what they're looking for. BM25 catches those. Together, the two channels produce a retrieval floor neither reaches alone.

**Problem Statement:**
- Dense embeddings score semantic similarity; they systematically underperform on queries that hinge on exact tokens (e.g., `FooBarV2Client`, `GPG-signed`, `§3.2.1`, author surnames, product SKUs).
- FAISS has no concept of token presence — two chunks with the same meaning but different rare-term content can score identically, even when the user typed the rare term verbatim.
- Reranking (Epic 18) can reorder a dense-only pool but cannot recover a chunk that FAISS never surfaced in the first place. If the right answer is at FAISS rank 200, it's invisible.
- Published results consistently show BM25 + dense fusion adding 5–15% recall on top of dense alone, with the biggest lift on rare-term queries.

**Solution:**
- Add a BM25 index built alongside each per-agent FAISS index during `grounding embeddings`.
- At query time, fetch top-N candidates from each channel, merge via Reciprocal Rank Fusion (RRF), and return the merged pool.
- Wire the hybrid path into the same three retrieval surfaces Epic 18 touched (`SearchCorpusTool`, MCP server, `local_rag.py`), behind a `retrieval.hybrid.enabled` flag that composes cleanly with `retrieval.rerank.enabled`.
- Use the Epic 16 harness to measure the lift and the Epic 17 `citation_accuracy` metric to guard against citation regression. Commit the measured numbers; flip default only if the rule passes — same discipline that governed Epic 18.

---

## Goals

1. Per-agent BM25 indexes exist alongside FAISS indexes, built incrementally in lockstep with embeddings.
2. Hybrid retrieval is available end-to-end through `SearchCorpusTool`, the MCP server, and `scripts/local_rag.py`, composable with Epic 18 reranking.
3. Opt-in via config and CLI flags; default is off in the first release, flipped in the final story only if measured lift clears the threshold.
4. Epic 16 eval harness measures the lift; `citation_accuracy` stays non-decreasing under hybrid.
5. Index rebuild for an existing agent does not require re-parsing documents — chunks on disk already carry the tokens BM25 needs.
6. BM25 index storage format is versioned and migrateable (same schema discipline as `_chunk_map.json`).

---

## Non-Goals

- Query-time term expansion or stemming beyond what `rank_bm25` / the chosen tokenizer does out of the box. HyDE and query rewriting are Tier 2 #5, tracked separately.
- Learned fusion weights. Start with RRF (parameter-free) and document alternatives as future work.
- Full-text search UI features (highlighting, snippets). Out of scope; retrieval-only.
- Multi-language tokenization. English-first; multilingual is deferred per ROADMAP.
- Replacing FAISS. Hybrid composes with the existing dense channel.

---

## Architecture

```
User Query
   │
   ├─────────────────────────────┐
   ▼                             ▼
┌─────────────────┐    ┌──────────────────────┐
│ Dense (FAISS)   │    │ BM25                 │
│ top-N_dense     │    │ top-N_lex            │
└────────┬────────┘    └──────────┬───────────┘
         │                        │
         └───────┬────────────────┘
                 ▼
       ┌────────────────────┐
       │ RRF merge          │   NEW: grounding/hybrid.py
       │ (reciprocal rank   │
       │  fusion, k=60)     │
       └──────────┬─────────┘
                  ▼
       ┌────────────────────┐
       │ Optional: rerank   │   existing: grounding/reranker.py
       │ (Epic 18)          │
       └──────────┬─────────┘
                  ▼
            Top-k results
```

### RRF Formula

For a document `d` appearing at rank `r_dense` in the dense channel and `r_lex` in the lexical channel:

```
RRF(d) = Σ  1 / (k + r_channel(d))
```

`k=60` is the standard choice in the literature. Documents missing from a channel contribute 0 for that channel. Tunable via config later, but start with k=60.

### Storage Shape

Per-agent directory structure extends from:
```
embeddings/<agent>/
├── _embeddings.faiss
├── _chunk_map.json
```
to:
```
embeddings/<agent>/
├── _embeddings.faiss
├── _chunk_map.json
├── _bm25.pkl              # NEW: rank_bm25 serialized state
├── _bm25_map.json         # NEW: maps BM25 integer doc ids → chunk_id
```

BM25 map mirrors the chunk_map contract and carries the same `format_version` discipline established in Story 14.1. Tombstones applied to FAISS (Story 14.1) are applied to BM25 the same way.

### Library Choice: `rank_bm25`

- Pure Python, no compilation, MIT-licensed, ~400 lines.
- `BM25Okapi` is the canonical variant; also `BM25Plus` and `BM25L` available if needed.
- Works on pre-tokenized document lists, so tokenization is under our control (simple whitespace + lowercase to start; stemmer optional).
- Serializable via `pickle`; we version the file and the tokenizer so future tokenizer changes invalidate cleanly.
- No new heavy dependencies (alternative would be `whoosh` or `meilisearch`, both of which pull in substantially more).

---

## Stories Breakdown

### Story 19.1: BM25 Index Builder

- Create `grounding/bm25.py` with tokenization, index build, save, load, and search helpers.
- Integrate into `grounding embeddings` so a BM25 index is built alongside FAISS for each agent; support the existing `--incremental` flag (append new chunks, tombstone deleted ones).
- Version the BM25 serialization format (`format_version: 1`) with the tokenizer name embedded so future tokenizer changes can migrate cleanly.
- Unit tests with small in-repo corpora; no network, no model downloads.

**AC:**
- Running `grounding embeddings --agent <name>` produces `_bm25.pkl` and `_bm25_map.json` alongside existing FAISS artifacts.
- `--incremental` updates BM25 in place without full rebuild.
- Tokenizer identity is recorded in the map file; loader rejects files built with a different tokenizer.
- Empty corpus and single-chunk corpus are handled without exceptions.
- Tests pass in < 3s without network.

**Status:** Done

### Story 19.2: Hybrid Retrieval + RRF

- Create `grounding/hybrid.py` with `search_hybrid(query, agent_dir, *, top_k, pool_size, k_rrf=60) -> list[dict]`.
- Internally: run dense search (existing code), run BM25 search (19.1), merge via RRF, return top-k or pool_size.
- Pure function over the index directory; no retrieval-surface coupling in this story.
- Unit tests cover: dense-only candidates, lex-only candidates, overlapping candidates, RRF math correctness, empty result handling, tombstone filtering from both channels.

**AC:**
- `search_hybrid(...)` returns dicts matching the existing retrieval-result shape (so 19.3 can drop it into place).
- Each returned dict carries `faiss_rank`, `bm25_rank` (either may be `None` for channel-exclusive hits), and `rrf_score`.
- RRF math matches hand-computed values on a documented worked example.
- When BM25 index is absent, function falls back to dense-only with a WARNING log and a `hybrid_degraded=True` marker on each result dict.
- Tests pass in < 2s.

**Status:** Done

### Story 19.3: Wire into Retrieval Paths

- Extend `SearchCorpusTool`, the MCP server's `search_corpus`, and `scripts/local_rag.py` to optionally use the hybrid path before (not instead of) reranking.
- Add a `HybridConfig` dataclass mirroring `RerankConfig`'s shape: `enabled: bool`, `pool_size: int`, `k_rrf: int`.
- Config resolution order follows 18.3's established pattern (flag → `config.yaml` → default).
- When both hybrid and rerank are enabled: hybrid produces the pool, reranker re-orders, citation prefix formats the final top-k.

**AC:**
- `SearchCorpusTool.execute(...)` accepts `hybrid_config: HybridConfig | None`; when enabled, calls `search_hybrid` instead of `_search`.
- MCP tool schema gains `hybrid_enabled`, `hybrid_pool_size`, `hybrid_k_rrf` optional inputs.
- `local_rag.py` gains `--hybrid`, `--hybrid-pool-size`, `--hybrid-k-rrf` flags; semantics match between surfaces.
- `config.example.yaml` gains `retrieval.hybrid.{enabled, pool_size, k_rrf}` block.
- When hybrid is disabled, behavior is bit-for-bit identical to pre-epic output.
- Rerank + hybrid interact correctly: rerank receives the hybrid pool, not the dense pool.

**Status:** Done

### Story 19.4: Measure, Decide, Close

- Run the eval harness in four configurations against the mini corpus and (if available) a private fixture set: `{hybrid off, hybrid on} × {rerank off, rerank on}`.
- Commit a comparison matrix to `docs/eval/README.md` with the same discipline as Story 18.4's table.
- Apply a flip rule identical to 18.4's (`recall@5` lift ≥ 0.03 AND `citation_accuracy` non-decreasing) to decide whether to flip `retrieval.hybrid.enabled` to true.
- If no private fixture set exists, null result is accepted — ship plumbing + opt-in, document the measurement gap, open a TD entry for the flip follow-up (same pattern as TD-003 for rerank).
- Update ROADMAP: Tier 1 #1 (hybrid) moves to Shipped; Tier 1/2 #2 (larger embedding model) becomes next up.

**AC:**
- Four-cell comparison matrix committed to the eval README.
- Flip decision justified with the measured numbers, per 18.4's precedent.
- ROADMAP updated; Epic 19 header set to Shipped.
- CI gate green on the merge commit.
- If default not flipped: TD entry tracks the follow-up measurement.

**Status:** Done

---

## Technical Details

### Tokenization (starting point)

```python
def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]
```

Simple and portable. `\w+` handles alphanumerics + underscores, which is right for code identifiers. Captures all the rare-term cases that motivate this epic.

Stemming / lemmatization is deferred. It helps on some query styles and hurts on others (code identifiers); without measurement it would be a guess. Add only if the eval harness shows stemmed variants outperform.

### Fusion Choice: RRF over Weighted Linear

Weighted linear fusion (`α * dense_score + (1-α) * bm25_score`) requires score normalization across channels with very different score distributions. RRF operates on *ranks*, which are channel-agnostic and parameter-free (beyond `k=60`). Standard in the IR literature since Cormack et al. 2009. Sanity-check alternative in 19.4 if time permits; default is RRF.

### Incremental Maintenance

BM25 indexes are rebuilt in memory on every load, so "incremental" for BM25 means: append new chunks to the tokenized document list, tombstone deleted chunks in the map (same mechanism as FAISS tombstones from 14.1), re-pickle. Faster than re-tokenizing the full corpus; slower than FAISS `add()`. Acceptable — BM25 build on 100k chunks is seconds.

### Tombstone Handling

Both channels respect the same tombstones. If a doc is tombstoned in `_chunk_map.json` (FAISS), the corresponding BM25 entries are filtered at search time too. Lives in 19.2's merge step: filter candidates from *either* channel if tombstoned in *either* map.

### Channel Asymmetry

A tombstone may exist in one map but not the other if a partial rebuild fails. Story 19.1 documents the graceful-degradation contract: either channel can surface a chunk the other marks tombstoned; merge step drops such candidates and logs a DEBUG-level inconsistency.

### Memory Footprint

Rough estimate: tokenized `all_chunks` list + BM25 in-memory state is ~1.5× the raw text size. For a 100k-chunk corpus at ~1KB per chunk body = ~150MB resident. Acceptable on any dev machine.

### Interaction with Reranker

Rerank (Epic 18) always operates on the last stage's pool. When hybrid is enabled, the pool is the RRF-merged list (capped at `hybrid.pool_size`). The reranker doesn't know or care where the candidates came from — it just re-scores the `(query, chunk_body)` pairs. Epic 18's contract (preserve input keys, add `rerank_score` / `faiss_distance`) is preserved; we add `bm25_rank` / `rrf_score` to the dict without clashing.

---

## Dependencies

### Epic Dependencies
- **Epic 6** — dense channel to fuse with.
- **Epic 16** — eval harness to measure the lift.
- **Epic 17** — `citation_accuracy` guard.
- **Epic 18** — rerank composes after hybrid; shared config plumbing pattern.

### External Dependencies
- `rank_bm25>=0.2.2` — new dependency, pure Python, MIT. Add to `pyproject.toml`.

### Code Dependencies
- `grounding/embedder.py`, `grounding/vector_store.py` — dense channel.
- `grounding/cli.py:embeddings_command` — integration point for `--incremental` BM25 build.
- `grounding/eval/runner.py` — pool-building path extended to support hybrid.
- `grounding/reranker.py` — unchanged; rerank just receives a different pool when hybrid is on.
- `scripts/search_corpus_tool.py`, `mcp_servers/corpus_search/server.py`, `scripts/local_rag.py` — integration surfaces.
- `grounding/config.py` — `resolve_hybrid_config` helper analogous to `resolve_rerank_config`.

---

## Implementation Order

```
Story 19.1 (BM25 Index Builder)
    └── Foundation; isolated module + embeddings CLI integration.

Story 19.2 (Hybrid Retrieval + RRF)
    └── Pure retrieval function over the built indexes.

Story 19.3 (Wire into Retrieval Paths)
    └── Integration into three surfaces; opt-in flags and config.

Story 19.4 (Measure, Decide, Close)
    └── Numbers-driven close, same discipline as Epic 18.
```

Strictly sequential. 19.1 and 19.2 don't share state but 19.2's tests need 19.1's artifacts.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| BM25 adds indexing latency | Low | Pure-Python BM25 builds ~100k chunks in seconds; documented. `--incremental` keeps the per-document cost constant. |
| Disk footprint doubles for hybrid-enabled agents | Low | BM25 pickle is small (~10MB per 100k chunks); clearly documented so users see the trade. |
| RRF `k=60` is suboptimal on our corpora | Medium | Tunable via config; 19.4 measures and documents the effect. Default of 60 is the literature standard and a reasonable starting point. |
| Tokenizer choice favors English | Medium | Documented in Dev Notes; multilingual support deferred to a separate epic. |
| Asymmetric tombstones produce phantom candidates | Low | 19.2 merge step filters on both maps; logged at DEBUG. |
| `rank_bm25` is ~5-year-old unmaintained library | Medium | Small (~400 LOC), MIT, stable API. Vendor-in fork is the escape hatch if it goes stale. |
| Hybrid + rerank compound latency | Medium | Document additive cost in `CLAUDE.md` alongside the rerank latency table. Both are opt-in; users choose their pain. |
| Measurement may show no lift on private corpus | Medium | 18.4's precedent accepts null results; 19.4 ships plumbing regardless and opens a TD entry. |

---

## Testing Strategy

### Unit Tests
- BM25 build, save, load, search, tombstone handling (19.1).
- RRF math on documented worked examples (19.2).
- Merge edge cases: dense-only, lex-only, overlapping, all-tombstoned, empty (19.2).
- Flag parsing and config resolution (19.3).

### Integration Tests
- Extend `tests/eval_fixtures/mini_corpus/` to verify BM25 artifacts are generated during `grounding embeddings`.
- End-to-end hybrid retrieval test against the mini corpus: asserts both channels contribute to at least one fixture's hit.
- Hybrid + rerank composed: result ordering differs from dense-only + rerank, and `citation_accuracy` holds.

### Manual Validation
- Run against the maintainer's private corpus with a rare-term query (e.g., a specific author surname, a code identifier). Spot-check that the hybrid path surfaces the right chunk while dense-only does not.

---

## Acceptance Criteria (Epic Level)

1. BM25 indexes exist alongside FAISS for every agent; build is incremental-compatible.
2. Hybrid retrieval available end-to-end through all three retrieval surfaces and `grounding eval`.
3. Opt-in via CLI flags + config; default decision made in 19.4 based on measurement.
4. Four-cell comparison matrix (`{hybrid off, on} × {rerank off, on}`) committed to `docs/eval/README.md`.
5. `citation_accuracy` non-decreasing under hybrid (hard gate for the flip).
6. Epic 16 CI gate stays green.
7. ROADMAP updated; next roadmap candidate explicitly called out.

---

## Definition of Done

- All four stories closed with AC met.
- BM25 artifacts committed to the mini corpus index directory and loaded by the eval harness.
- Comparison matrix and flip decision text committed to `docs/eval/README.md`.
- `config.example.yaml` has `retrieval.hybrid` block.
- `CLAUDE.md` documents the two-stage-with-hybrid flow, including the interaction with reranking.
- ROADMAP Tier 1 #1 moved to Shipped.

---

## Future Enhancements (Out of Scope)

- Learned fusion weights (e.g., per-agent or per-query-type tuning).
- Multilingual tokenization.
- Query rewriting (HyDE) — Tier 2 #5.
- Snippet highlighting in retrieval output.
- SPLADE or other learned sparse retrievers as a third channel.
- Stemmer / lemmatizer — adopt only if measured to help.

---

## References

- `docs/ROADMAP.md` — Tier 1 #1 (this epic).
- `docs/epics/epic-16-evaluation-harness.md` — measurement dependency.
- `docs/epics/epic-17-page-and-section-citations.md` — `citation_accuracy` guard.
- `docs/epics/epic-18-cross-encoder-reranking.md` — rerank composes after hybrid; shared config pattern.
- `docs/TECH-DEBT.md#td-003-rerank-default-flip-decision-deferred` — same pattern will likely apply to hybrid.
- `grounding/vector_store.py`, `grounding/embedder.py` — dense channel.
- Cormack, Clarke, Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009).
- `rank_bm25`: https://github.com/dorianbrown/rank_bm25
