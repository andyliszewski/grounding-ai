# Roadmap

**Status:** Directional, non-binding. No dates. Items may be reordered, deferred, or dropped as the project learns. Contributions welcome on any item.

## Shipped

- **v0.1 — v0.2 core pipeline:** PDF/EPUB/DOCX/MD parsing (Unstructured, Marker), deterministic chunking, content-hash provenance (SHA-1, SHA-256, BLAKE3), corpus manifest.
- **v0.3 retrieval layer:** Per-agent FAISS vector indexes, agent YAML + collection filtering, incremental embedding updates, MCP corpus-search server, agentic tool-calling RAG via Ollama.
- **v0.3 specialized modalities:** Optical Music Recognition (Audiveris), LaTeX formula extraction (pix2tex).
- **v0.3 workflow:** Staging watcher with auto-embedding updates, optional multi-machine setup via Syncthing.
- **Evaluation harness (Epic 16).** Fixture schema, runner, metrics (recall@k, MRR, nDCG), `grounding eval` CLI, committed mini-corpus baseline, and a GitHub Actions gate on PRs touching retrieval code. Every retrieval change downstream of this is now measurable against a published number. See `docs/eval/README.md`.
- **Page and section citations (Epic 17).** Parser/formatter surface an element→page/section map; chunker derives `page_start`, `page_end`, and `section_heading` per chunk; retrieval surfaces (`search_corpus` tool + MCP server) prefix each result with a compact `[slug, p.247, §3.2 Bootstrap Methods]` line. Eval harness gained `citation_accuracy` (CI-gated) so future changes that drop page/section metadata fail before merge. See `docs/epics/epic-17-page-and-section-citations.md`.
- **Cross-encoder reranking (Epic 18).** Two-stage retrieval: FAISS returns a candidate pool, `BAAI/bge-reranker-base` rescores the pool with joint `(query, chunk)` attention, top-k are returned. Wired end-to-end through `SearchCorpusTool`, the MCP server, `scripts/local_rag.py`, and `grounding eval`; opt-in via `--rerank` and `retrieval.rerank.enabled` in `config.yaml`. Shipped default-off because the public mini corpus is saturation-limited and no private-corpus fixture set is yet committed to measure the lift on realistic query diversity; the flip-to-default comparison, decision rule, and measured numbers are recorded in `docs/eval/README.md#reranking-comparison-story-184`. See `docs/epics/epic-18-cross-encoder-reranking.md`.
- **Hybrid retrieval — BM25 + dense (Epic 19).** Lexical BM25 sidecar (`_bm25.pkl` / `_bm25_map.json`) written alongside the FAISS index on every full or incremental embeddings run; `search_hybrid` fuses FAISS and BM25 via Reciprocal Rank Fusion (k_rrf=60). Wired end-to-end through `search_corpus`, the MCP server, `scripts/local_rag.py`, and `grounding eval`; opt-in via `--hybrid` and `retrieval.hybrid.enabled` in `config.yaml`, composes with `--rerank`. Shipped default-off because the public mini corpus is saturation-limited (same constraint as 18.4) and no ≥ 10-item private-corpus fixture set was available at epic close; the four-cell `{hybrid off/on} × {rerank off/on}` comparison, decision rule, and measured latency are recorded in `docs/eval/README.md#hybrid-retrieval-comparison-story-194`. Follow-up measurement tracked as TD-004. See `docs/epics/epic-19-hybrid-retrieval-bm25-dense.md`.

## Tier 1 — retrieval quality (next up)

Sequenced; each builds on the previous.

1. **Larger embedding model option.** *Unblocked by Epic 19.* Offer `bge-large-en-v1.5` or `nomic-embed-text` alongside `all-MiniLM-L6-v2`. Trade index size for recall.

## Tier 2 — retrieval depth

2. **Content-aware deduplication.** Detect near-duplicate documents (same paper, different PDFs) via chunk-hash overlap; merge or skip.
3. **Table extraction quality.** Verify tables survive chunking intact; add tests. Critical for technical and financial corpora.
4. **Query rewriting (HyDE or expansion).** Modest latency cost, meaningful recall improvement on short queries.

## Tier 3 — polish and scale

5. **Chunk re-embedding without re-ingest.** Swap embedding models without reparsing the corpus.
6. **Entity/concept graph layer.** Cross-document "what does author X say elsewhere about Y?" queries.
7. **Web UI.** Lightweight FastAPI + htmx page for browsing agent corpora; currently CLI + MCP only.

## Deferred / under consideration

- **Symbolic math integration** (sympy validation, LaTeX → Python function generation, dimensional analysis via Pint). Pending real-world feedback on formula extraction quality.
- **GPU acceleration** for OMR, formula extraction, and embeddings.
- **Additional modalities:** chemical structures (SMILES/InChI), diagram/figure understanding.
- **Multilingual document support.**

## How items move forward

1. A roadmap line gets picked up → `po` agent drafts an epic or story with acceptance criteria in `docs/stories/`.
2. A matching GitHub issue is opened so contributors can see and engage.
3. Work lands on a feature branch → PR → review → merge. The eval harness (shipped) catches retrieval regressions automatically.
