# Epic 20: Larger Embedding Model Option

**Epic ID:** E20
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 0/4
**Dependencies:** Epic 6 (Vector Embeddings), Epic 16 (Evaluation Harness), Epic 17 (Citation Accuracy Guard), Epic 18 (Cross-Encoder Reranking), Epic 19 (Hybrid Retrieval)
**Target Completion:** TBD

---

## Overview

Parameterize the embedding model so users can choose between the current default `all-MiniLM-L6-v2` (384-dim, ~80MB) and a higher-recall alternative, initially `BAAI/bge-large-en-v1.5` (1024-dim, ~1.3GB), with a migration path that doesn't force a one-way door. Ship with a multi-index-per-agent directory layout so users can A/B models against the same corpus without committing, then measure whether the bigger model's recall lift clears the Epic 16 flip-rule threshold against the composed hybrid+rerank pipeline. Default stays MiniLM unless measurement justifies the flip.

**Problem Statement:**
- `all-MiniLM-L6-v2` was chosen for small size and fast CPU inference, not for top-tier recall. On MTEB, larger models typically outperform it by 8–15 percentage points on retrieval benchmarks.
- Epic 18 (rerank) and Epic 19 (hybrid BM25+dense) layered quality on top of the fixed-MiniLM baseline. Neither could touch the underlying vector quality: if MiniLM didn't encode a passage's semantics well, rerank and hybrid can only work with what FAISS surfaced.
- Users currently cannot swap models without re-embedding their entire corpus, because the embedder is hardcoded and no model identity is stamped into the index. Stale indexes load silently and produce nonsense distances.
- Switching embedding models is a one-way door today: there's one index per agent, so migrating means losing the old index before the new one is validated.

**Solution:**
- Parameterize the embedder to accept a model name (config or flag). Stamp the model identity into `_chunk_map.json` so search-time can refuse to query an index with the wrong embedder.
- Restructure per-agent index storage from `embeddings/<agent>/_embeddings.faiss` to `embeddings/<agent>/<model-slug>/_embeddings.faiss`, with a `current` pointer (symlink or marker file) indicating which model is active for query time. Old layouts are detected and migrated to the new shape on first touch.
- Add `--model` to `grounding embeddings` and `retrieval.embedding.model` to `config.yaml`, following the 18.3/19.3 plumbing pattern.
- Measure the lift via the Epic 16 harness against the full composition (model × hybrid × rerank), apply the flip rule (`recall@5` lift ≥ 0.03 AND `citation_accuracy` non-decreasing), and flip the default only if evidence holds.

---

## Goals

1. Embedder module reads the model name from config/flag; no more hardcoded `all-MiniLM-L6-v2` calls.
2. Per-agent storage supports multiple co-resident indexes keyed by model-slug; query-time selects which one to load.
3. Backward compatibility: an existing `embeddings/<agent>/_embeddings.faiss` built pre-Epic-20 continues to work without user intervention (auto-detected and wrapped as an implicit `minilm-l6` sub-index, or a one-time migration on next touch).
4. Model-mismatch between query embedder and index embedder is detected and refused at search time, with an actionable error.
5. Users can rebuild an agent's index under a different model (`grounding embeddings --agent X --model bge-large-en-v1.5`) without disturbing the existing index, and switch between them at query time.
6. Epic 16 harness measures the lift across the `{MiniLM, bge-large} × {hybrid off/on} × {rerank off/on}` matrix (eight cells total against a real corpus); flip decision applied per 18.4/19.4 discipline.
7. `citation_accuracy` stays non-decreasing under any new default.
8. `nomic-embed-text` is supported by the parameterization (same flag shape, same storage layout) but not measured in this epic. Measuring it is a scoped refresh PR after 20.4 closes.

---

## Non-Goals

- Fine-tuning any embedding model on the project's own data. Pick an off-the-shelf model.
- GPU-only inference paths. CPU stays the reference target. GPU is a separate future epic.
- Dimensionality-reduction tricks (PCA, Matryoshka truncation). Store the model's native dimension.
- Automatic model selection per query. Users pick a model at build time; it applies to the whole agent index.
- Cross-model fusion (ensemble of MiniLM + bge-large vectors). Out of scope; hybrid already provides channel fusion.
- Swapping the sentence-transformers dependency. Stay on `sentence-transformers>=2.2.0`.
- Multilingual embedding models (`bge-multilingual`, etc.). English-first, per ROADMAP.

---

## Architecture

### Directory Layout Evolution

**Pre-Epic-20 (current):**
```
embeddings/<agent>/
├── _embeddings.faiss
├── _chunk_map.json
├── _bm25.pkl              # Epic 19
├── _bm25_map.json         # Epic 19
```

**Post-Epic-20:**
```
embeddings/<agent>/
├── current                          # marker file: "bge-large-en-v1.5" (model-slug)
├── minilm-l6/
│   ├── _embeddings.faiss
│   ├── _chunk_map.json              # stamps "model": "all-MiniLM-L6-v2", "dim": 384
│   ├── _bm25.pkl                    # BM25 is model-independent; see below
│   └── _bm25_map.json
└── bge-large-en-v1.5/
    ├── _embeddings.faiss
    ├── _chunk_map.json              # stamps "model": "BAAI/bge-large-en-v1.5", "dim": 1024
    ├── _bm25.pkl                    # BM25 is model-independent; see below
    └── _bm25_map.json
```

### Query-Time Model Resolution

1. Caller specifies `model=None` (default) → read `embeddings/<agent>/current` marker → load that sub-index.
2. Caller specifies `model="bge-large-en-v1.5"` → load that sub-index directly; ignore the marker.
3. `_chunk_map.json` carries `"model"` and `"dim"` fields; search-time sanity-checks that the query embedder produces a vector of that dimension. Mismatch → `EmbeddingModelMismatchError` with the expected and actual model names.
4. On load, the FAISS index's `.d` attribute is cross-checked against the chunk_map's `"dim"` value. A legacy index without `"dim"` is tolerated (treated as MiniLM 384 per back-compat rule; logged once).

### BM25 Sidecar is Model-Independent

BM25 tokens don't depend on the embedding model. The natural tension: do we duplicate `_bm25.pkl` across every sub-index (wasted disk) or keep it at the agent level (violates the "each sub-index is self-contained" invariant)?

**Decision:** duplicate. Each sub-index is self-contained, which simplifies the multi-index contract (no cross-directory reads, no ordering dependencies). Per 16k-chunk agent, BM25 pickle is ~5MB; duplicating it across two models is 10MB versus the FAISS indexes being ~20MB (MiniLM) and ~65MB (bge-large). BM25 duplication is in the noise.

Incremental embeddings rebuilds the BM25 sidecar for whichever sub-index is being updated. Cross-sub-index BM25 drift is impossible because the tokenized corpus is identical (tokens are pure function of chunk bodies, which are shared).

### Storage Cost

| Model | Dim | Disk per 100k chunks | Build time (CPU, ~16 cores) |
|-------|-----|----------------------|------------------------------|
| `all-MiniLM-L6-v2` | 384 | ~150 MB | ~8 min |
| `BAAI/bge-large-en-v1.5` | 1024 | ~400 MB | ~35 min |
| `nomic-embed-text` | 768 | ~300 MB | ~20 min (est., untested) |

Numbers are rough estimates; exact figures land in 20.4 after measurement against `private-agent` (16k chunks, same host as 19.4's latency measurement).

---

## Stories Breakdown

### Story 20.1: Multi-Index Directory Contract + Embedder Parameterization

- Extend `grounding/embedder.py` to accept a model name; lazy-load via `SentenceTransformer(model_name)`.
- Introduce the `embeddings/<agent>/<model-slug>/` layout with a `current` marker file.
- Stamp `"model"` and `"dim"` into `_chunk_map.json` v1.2.
- Detect legacy `embeddings/<agent>/_embeddings.faiss` at load time; auto-wrap as `minilm-l6/` on first touch (one-time in-place migration, atomic).
- Search-time validation: `EmbeddingModelMismatchError` when query-embedder dim ≠ index dim or model-slug mismatch (where known).
- Unit tests cover: parameterization, legacy-layout migration, mismatch rejection, BM25 duplication invariant.

**AC:**
- `grounding/embedder.py` exposes `generate_embedding(text, model="all-MiniLM-L6-v2")`.
- Fresh `grounding embeddings --agent X --model bge-large-en-v1.5` writes to `embeddings/X/bge-large-en-v1.5/` and updates `embeddings/X/current`.
- Pre-20 indexes load without error; one INFO log records the migration.
- `_chunk_map.json` v1.2 carries `{format_version: "1.2", model, dim}`; v1.1 maps load under the legacy-MiniLM assumption.
- `EmbeddingModelMismatchError` fires when query model ≠ index model; message names both and suggests the rebuild command.
- Tests pass in < 5s without network or model downloads (`SentenceTransformer` is mocked).

**Status:** Draft

### Story 20.2: Wire Model Selection Through CLI, Config, and MCP

- `grounding embeddings`: add `--model <name>` flag. Validates the name against a small allowlist (`all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`, `nomic-embed-text` initially; additional names pass through with a WARN).
- `grounding eval`: add `--embedding-model <name>` to let the eval harness measure a non-default model without touching global config.
- `scripts/local_rag.py`: add `--embedding-model <name>` (query-time override; most users will leave it on the `current` marker).
- `config.example.yaml`: new `retrieval.embedding` block with `model` key. Resolution order (CLI > config.yaml > `current` marker > MiniLM default), documented in `CLAUDE.md`.
- `grounding/config.py`: add `resolve_embedding_config(...)` mirroring `resolve_rerank_config` / `resolve_hybrid_config`. The 19.3-QA MNT-001 debt (three near-duplicate resolvers) grows to four; collapse deferred to a standalone refactor story.
- MCP schema: add `embedding_model` optional arg (mostly for debugging; most callers won't set it).

**AC:**
- `grounding embeddings --agent X --model bge-large-en-v1.5` builds the sub-index and updates `current`.
- `grounding eval --agent X --embedding-model bge-large-en-v1.5` runs the harness against the named sub-index without touching `current`.
- `local_rag.py --embedding-model bge-large-en-v1.5` queries the named sub-index.
- `config.yaml` `retrieval.embedding.model` is honored by all three CLIs.
- MCP tool schema gains a documented `embedding_model` input.
- Backward compatibility: no flag, no config, no `current` marker → loads MiniLM 384-dim index as before (bit-for-bit identical eval gate).

**Status:** Draft

### Story 20.3: Migration Command, Stale-Index Guards, and Docs

- `grounding embeddings --agent X --model bge-large-en-v1.5 --set-current` atomically updates the `current` marker after a successful build. Without `--set-current`, the build lands but `current` stays on whatever it was.
- `grounding embeddings --agent X --list-models` prints the available sub-indexes and which one is `current`.
- `grounding embeddings --agent X --drop-model <slug>` deletes a sub-index (with an interactive confirmation unless `--yes`). Forbids dropping the `current` model without an explicit reassignment.
- Search-time: if `current` points at a missing sub-index, fall back to any sub-index that exists with a WARN log. If the agent has no sub-indexes at all, error clearly.
- Document the migration flow in `CLAUDE.md` ("Embedding Generation" section gains a "Multi-model support" subsection).
- Update `docs/multi-machine.md` if it references the old single-index layout (scope check during implementation; likely small).

**AC:**
- `--set-current`, `--list-models`, `--drop-model` work as described.
- Missing `current` target produces a recoverable WARN, not an error, if any sub-index exists.
- CLAUDE.md documents: legacy layout is auto-migrated; how to add a model; how to A/B; how to switch defaults.
- `docs/multi-machine.md` unchanged or updated as needed.
- Unit tests cover each new CLI verb and the missing-current fallback.

**Status:** Draft

### Story 20.4: Measure, Decide, Close

- Run the Epic 16 harness in an eight-cell matrix against the mini corpus and (if available) a private fixture set: `{MiniLM, bge-large-en-v1.5} × {hybrid off/on} × {rerank off/on}`.
- Commit a comparison matrix to `docs/eval/README.md` with the same discipline as 18.4's and 19.4's tables. Report `recall@5`, `citation_accuracy`, median per-query latency for each cell.
- Apply the flip rule: flip `retrieval.embedding.model` default from `all-MiniLM-L6-v2` to `BAAI/bge-large-en-v1.5` only if at least one corpus tier shows `recall@5` lift ≥ 0.03 (bge-large vs MiniLM, hybrid and rerank held at their current defaults) AND `citation_accuracy` non-decreasing.
- If measurement is blocked by fixture-set unavailability (same constraint that produced TD-003 and TD-004): accept null result, ship plumbing default-MiniLM, open TD-005 mirroring TD-003's shape.
- Update ROADMAP: Tier 1 #1 (larger embedding model) moves to Shipped; next Tier item becomes Tier 1 #1.
- Mark Epic 20 Shipped.
- Scoped follow-up: measure `nomic-embed-text` when a ≥ 10-item private fixture set lands. Capture as TD or as a scoped refresh PR.

**AC:**
- Eight-cell comparison matrix committed to `docs/eval/README.md`.
- Flip decision justified with measured numbers.
- ROADMAP updated; Epic 20 header Shipped.
- CI gate green on the merge commit.
- If default not flipped: TD-005 entry tracks the follow-up.

**Status:** Draft

---

## Technical Details

### Model Comparison (candidate models for Epic 20)

| Property | `all-MiniLM-L6-v2` (current default) | `BAAI/bge-large-en-v1.5` (primary new) | `nomic-embed-text` (secondary, deferred) |
|----------|--------------------------------------|----------------------------------------|------------------------------------------|
| Dimension | 384 | 1024 | 768 |
| Download size | ~80 MB | ~1.3 GB | ~550 MB |
| Max input tokens | 512 | 512 | 8192 |
| License | Apache 2.0 | MIT | Apache 2.0 |
| MTEB retrieval avg | ~52 | ~65 | ~62 |
| CPU inference speed (relative) | 1.0× | ~3× slower | ~2× slower |
| Architectural note | 6-layer MiniLM | Same lineage as `bge-reranker-base` (Epic 18) | Native long-context support |

bge-large is the primary measurement target because (a) it shares training lineage with the Epic-18 reranker, so the composed pipeline is architecturally coherent, and (b) it's the standard "bigger MiniLM" comparison in the IR literature. Nomic is the better candidate if long-context or permissive licensing dominate a user's needs; measured separately.

### Model-Slug Naming

The model-slug is the sub-directory name. It needs to be filesystem-safe and deterministic.

| Model name (as passed to `sentence-transformers`) | Slug |
|---------------------------------------------------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | `minilm-l6` |
| `all-MiniLM-L6-v2` | `minilm-l6` (normalize to canonical form) |
| `BAAI/bge-large-en-v1.5` | `bge-large-en-v1.5` |
| `nomic-ai/nomic-embed-text-v1` | `nomic-embed-text-v1` |

A tiny `grounding/embeddings/slugs.py` module owns the canonical mapping. Unknown model names get a conservative slug via `slugify(model_name.split("/")[-1])`.

### Legacy-Layout Migration

On first load of an agent that has the pre-20 layout:

1. Detect: `embeddings/<agent>/_embeddings.faiss` exists at the agent-level (not in a sub-directory).
2. Migrate atomically: `mkdir embeddings/<agent>/minilm-l6/`, move the four artifacts into it, write `embeddings/<agent>/current` containing the text `minilm-l6`.
3. Log one INFO line: `migrated legacy embedding layout for agent=<X> to minilm-l6/`.
4. If the migration fails (disk full, permission denied), abort with a clear error pointing the user at a manual rebuild command. Never leave the directory half-migrated.

The migration is idempotent: if it's already happened, step 1 finds no agent-level artifact and the function returns immediately.

### `_chunk_map.json` Schema v1.2

```json
{
  "format_version": "1.2",
  "model": "BAAI/bge-large-en-v1.5",
  "dim": 1024,
  "rank_bm25_version": "0.2.2",
  "tokenizer": "whitespace_lowercase_v1",
  "total": 16251,
  "tombstone_count": 3,
  "created_utc": "...",
  "updated_utc": "...",
  "chunks": [
    {"faiss_index": 0, "chunk_id": "...", "doc_id": "...", "deleted_utc": null},
    ...
  ]
}
```

v1.1 maps (Epic 19's BM25 work) load cleanly: `model` defaults to `all-MiniLM-L6-v2`, `dim` defaults to 384. No schema-migration pass is needed on read; the upgrade happens on next write.

### Interaction with Hybrid + Rerank

- **BM25:** no interaction. BM25 tokenizes chunk bodies (not embeddings), so it's model-agnostic. The sidecar duplicates across sub-indexes for self-containment (see Architecture).
- **Dense (FAISS):** fully model-dependent. Each sub-index is an independent FAISS index.
- **Rerank:** no interaction. The cross-encoder re-scores `(query, chunk_body)` pairs; it doesn't see the bi-encoder's vectors. Any embedding model produces the same downstream rerank behavior.

The composition `{model × hybrid × rerank}` therefore has 2 × 2 × 2 = 8 distinct configurations for 20.4's measurement. Expected finding: bge-large's lift is largest on dense-only retrieval and partially absorbed by rerank (which already promotes the right chunks given a reasonable pool).

### Cross-Encoder Tokenizer Note

`BAAI/bge-reranker-base` (Epic 18) accepts any sentence pair regardless of which bi-encoder produced the candidates. There's no hidden coupling between `bge-large-en-v1.5` and `bge-reranker-base` beyond shared training philosophy. Don't assume architectural compatibility buys accuracy; measure.

### One-Way-Door Cost (User-Facing)

The multi-index layout explicitly breaks the "switching models is a one-way door" framing. Concretely:

- A user running MiniLM today can build bge-large alongside (`grounding embeddings --agent X --model bge-large-en-v1.5`), test both via `--embedding-model` at query time, and flip `current` only after they're satisfied.
- Disk cost is additive (2× for two models co-resident), not multiplicative.
- Reversion is instant (rewrite `current`). No re-embed required.

This is the key UX win over the single-index approach considered earlier in the epic's design discussion.

---

## Dependencies

### Epic Dependencies
- **Epic 6** — embedder module to parameterize.
- **Epic 16** — eval harness to measure the lift.
- **Epic 17** — `citation_accuracy` guard.
- **Epic 18** — rerank composes over whichever embedder produced the candidates; eight-cell matrix.
- **Epic 19** — hybrid composes over whichever embedder; BM25 sidecar duplicated across sub-indexes.

### External Dependencies
- No new runtime deps. `sentence-transformers` already supports arbitrary HuggingFace model names.
- bge-large model weights downloaded on first use (~1.3 GB, HF hub). Users need disk space; document in `CLAUDE.md`.

### Code Dependencies
- `grounding/embedder.py` — parameterize.
- `grounding/vector_store.py` — path resolution (agent dir → agent/model-slug/ dir).
- `grounding/bm25.py` — same path-resolution change.
- `grounding/cli.py:embeddings_command` — `--model`, `--set-current`, `--list-models`, `--drop-model`.
- `grounding/eval/cli.py`, `scripts/local_rag.py`, `mcp_servers/corpus_search/server.py` — `--embedding-model` plumbing.
- `grounding/config.py` — `resolve_embedding_config` helper.

---

## Implementation Order

```
Story 20.1 (Multi-Index Contract + Parameterization)
    └── Foundation; directory layout, embedder param, chunk_map schema, legacy migration.

Story 20.2 (Wire Through CLI, Config, MCP)
    └── Plumbing; mirrors 18.3/19.3 pattern.

Story 20.3 (Migration Command + Guards + Docs)
    └── User-facing ergonomics; --set-current, --list-models, --drop-model, doc update.

Story 20.4 (Measure, Decide, Close)
    └── Eight-cell matrix; flip decision; ROADMAP close.
```

Strictly sequential. 20.2 depends on 20.1's storage contract; 20.3 depends on 20.2's CLI surface; 20.4 depends on all of the above.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| bge-large download (~1.3 GB) surprises users | Low | Documented in CLAUDE.md; first-time download is one-shot and cached in `~/.cache/huggingface`. |
| bge-large CPU inference is ~3× slower than MiniLM | Medium | Build-time cost, not query-time (dense query is a single forward pass). Documented with build-time numbers in 20.4. |
| Legacy-layout auto-migration corrupts an existing index | High | Atomic move (same-device rename); idempotent; log every migration; tested against a real pre-20 agent as part of 20.3's test suite. Rollback: rename the directory back. |
| `current` marker gets out of sync with filesystem | Medium | 20.3's guard handles this: missing target falls back with WARN. Always write the marker last in any build flow. |
| Users mix model-slugs across machines via Syncthing | Low | Each machine's `current` marker is independent; sub-indexes sync cleanly as regular files. Document briefly. |
| Measurement shows no lift on private corpus | Medium | 18.4/19.4 precedent accepts null results; 20.4 ships plumbing regardless and opens TD-005. |
| 4-resolver duplication in `grounding/config.py` grows to 5 | Medium | Acknowledged; consider a standalone refactor story after Epic 20 closes. Tracks with 19.3-QA MNT-001. |
| bge-large index size becomes a disk pressure issue on large corpora | Medium | Document the per-100k-chunk disk cost; users choose. `--drop-model` gives them an exit. |
| Cross-encoder rerank (Epic 18) shows no meaningful additional lift on bge-large | Low | Informational; doesn't affect Epic 20's flip rule (which compares model-only with hybrid/rerank held fixed at current defaults). The 8-cell matrix makes this observation automatic. |

---

## Testing Strategy

### Unit Tests
- Embedder parameterization: `generate_embedding(text, model=...)` produces vectors of the advertised dimension (20.1).
- Legacy-layout migration: fake pre-20 directory structure; run load path; assert migration + one INFO log + correct final layout (20.1).
- `_chunk_map.json` v1.1 → v1.2 read-compatibility (20.1).
- Model-mismatch rejection: query-embedder with dim 384 vs index with dim 1024 → `EmbeddingModelMismatchError` (20.1).
- CLI flag parsing for `--model`, `--set-current`, `--list-models`, `--drop-model` (20.2, 20.3).
- Config-yaml resolution precedence for `retrieval.embedding.model` (20.2).
- Missing `current` fallback behavior (20.3).

### Integration Tests
- Build a MiniLM sub-index and a bge-large sub-index against the mini corpus; query each via `--embedding-model`; assert neither leaks into the other's results.
- Eval harness against both models end-to-end; assert aggregate blocks are distinct and `citation_accuracy` holds in both.

### Manual Validation
- Full rebuild against the maintainer's `private-agent` agent (16k chunks) with bge-large. Measure build time, disk size, and warm-query latency. Feed numbers into 20.4.

---

## Acceptance Criteria (Epic Level)

1. `grounding embeddings` accepts `--model <name>`; produces a `embeddings/<agent>/<model-slug>/` sub-index.
2. Legacy `embeddings/<agent>/_embeddings.faiss` layouts are auto-migrated to `minilm-l6/` on first touch.
3. `_chunk_map.json` v1.2 stamps `model` and `dim`; search-time refuses mismatched embedders.
4. All three retrieval surfaces (`grounding eval`, `local_rag.py`, MCP server) accept `--embedding-model` / `embedding_model` arguments.
5. `config.yaml` `retrieval.embedding.model` is honored; resolution order matches 18.3/19.3.
6. `--set-current`, `--list-models`, `--drop-model` verbs land on `grounding embeddings`.
7. Eight-cell comparison matrix committed to `docs/eval/README.md`.
8. Flip decision applied per 18.4/19.4 discipline; default flipped or TD entry opened.
9. `citation_accuracy` non-decreasing under the new default (hard gate).
10. Epic 16 CI gate stays green.
11. ROADMAP updated; next roadmap candidate called out.

---

## Definition of Done

- All four stories closed with AC met.
- Multi-index layout works end-to-end against the mini corpus and at least one real agent.
- Comparison matrix and flip decision text committed to `docs/eval/README.md`.
- `config.example.yaml` has `retrieval.embedding` block.
- `CLAUDE.md` documents the multi-model flow, including legacy-layout migration, `--set-current` ergonomics, and disk-cost trade-offs.
- ROADMAP Tier 1 #1 moved to Shipped.
- TD-005 opened if the default did not flip.

---

## Future Enhancements (Out of Scope)

- Automatic per-query model selection (query classifier → model router).
- Matryoshka / PCA dimension reduction on bge-large to reclaim disk.
- GPU inference paths for bge-large build-time speedup.
- Cross-model ensemble (MiniLM + bge-large vectors concatenated or averaged).
- Domain-adapted fine-tuning on the project's own corpus.
- Multilingual embedding models.
- `nomic-embed-text` measurement (deferred to scoped refresh PR post-20.4).

---

## References

- `docs/ROADMAP.md` — Tier 1 #1 (this epic), unblocked by Epic 19.
- `docs/epics/epic-16-evaluation-harness.md` — measurement dependency.
- `docs/epics/epic-17-page-and-section-citations.md` — `citation_accuracy` guard.
- `docs/epics/epic-18-cross-encoder-reranking.md` — rerank composes over any embedder; plumbing precedent.
- `docs/epics/epic-19-hybrid-retrieval-bm25-dense.md` — hybrid composes over any embedder; BM25 sidecar handling.
- `docs/TECH-DEBT.md#td-003-rerank-default-flip-decision-deferred` — TD pattern for the possible null result.
- `docs/TECH-DEBT.md#td-004-hybrid-default-flip-decision-deferred` — same.
- `grounding/embedder.py`, `grounding/vector_store.py` — parameterization targets.
- MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard (retrieval sub-track).
- `BAAI/bge-large-en-v1.5`: https://huggingface.co/BAAI/bge-large-en-v1.5
- `nomic-embed-text`: https://huggingface.co/nomic-ai/nomic-embed-text-v1
