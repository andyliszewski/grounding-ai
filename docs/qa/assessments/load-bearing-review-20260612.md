# Load-Bearing Surface Review — 2026-06-12

**Scope:** publish boundary (`scripts/publish.sh`), retrieval surfaces (MCP
server, agentic search tool, local_rag), embedding/index contracts
(FAISS + BM25), staging watcher, determinism core (slugify / hashing /
atomic writes / manifest / chunker).

**Method:** one reviewer pass on `publish.sh` + the uncommitted working
tree, plus three parallel review agents (retrieval, unattended ingestion,
determinism core). The headline retrieval finding (R1) was reproduced
in-memory by the reviewing agent, not just read from code.

**Verdict:** publish boundary is sound (HEAD passes the sanitizer
end-to-end in simulation). The serious defects are: hybrid retrieval
functionally broken on all production surfaces (R1), corpus corruption on
the ordinary re-ingest workflow (D1/D2), and permanent silent FAISS↔BM25
desync in unattended operation (W1/W2).

**Proposed work packages:**

| Package | Findings | Vehicle |
|---------|----------|---------|
| Hotfix, no story | R1 (+ note TD-004 re-measure) | direct dev pass |
| Corpus integrity story/epic | D1, D2, D3, D4, W5 | new story (PM to draft) |
| Unattended hardening | W3, W4, R3, R2, W6, W7 | fold into Epic 21 |
| Publish boundary nits | P1, P2 | small standalone PR |
| Backlog / track only | remaining MED/LOW | PM judgment |

---

## R — Retrieval surfaces

### R1 — HIGH: hybrid dense channel returns zero results on all three surfaces

`mcp_servers/corpus_search/server.py:96-97,173`, `scripts/search_corpus_tool.py:267-272`,
`scripts/local_rag.py:179-180,248-252`

All three surfaces strip the chunk-map wrapper dict on load
(`chunk_map = chunk_map_data["chunks"]`), then the hybrid path re-wraps it
as `{"chunks": chunk_map}` and injects it into `search_hybrid` via
`load_index_fn`. The adapted dict has no `format_version` key, so
`search_similar_chunks` (`grounding/vector_store.py:332-336`) defaults to
"1.0", reads the absent `chunk_ids` list (empty), and skips every FAISS
hit. Reproduced in-memory: adapted map → dense channel returns `[]`;
with `format_version: "1.2"` present → correct results.

Consequences:
- `hybrid_enabled=true` with BM25 present → results are **BM25-only**
  masquerading as fused RRF (`faiss_rank=None` on every hit). Semantic
  recall silently collapses to lexical-only — the inverse of the
  feature's purpose.
- `hybrid_enabled=true` with BM25 missing (pre-19.1 agents) → degraded
  fallback iterates an empty dense list → **zero results**, rendered as
  "No relevant documents found".
- ~`pool_size` spurious "Invalid index N returned by FAISS" warnings
  per query.

**Fix:** keep the full on-disk dict (or set `format_version` when
adapting). One-liner at three call sites + a regression test.

**Follow-on:** Epic 19.4's four-cell measurement may have run through
this bug — TD-004's re-measurement should be re-run after the fix before
any default-flip decision.

### R2 — MEDIUM (security): path traversal via MCP `agent` argument

`mcp_servers/corpus_search/server.py:78,109,437-442`

`embeddings_dir / agent_name` with caller-controlled `agent`: `..`
traverses, an absolute value replaces the base entirely. A crafted
chunk map with absolute `file_path` entries then turns `read_chunk`
(`corpus_dir / chunk_path` — absolute RHS wins in pathlib) into an
arbitrary-file read returned in tool output. The `FileNotFoundError`
echo (lines 470-475) also enables path probing.

**Fix:** validate `agent` against `^[a-z0-9][a-z0-9-]*$` or membership in
`list_available_agents()`; reject absolute / `..`-containing `file_path`
entries in `read_chunk`.

### R3 — MEDIUM (latent): default dense-only paths bypass tombstone filtering

`server.py:230-259`, `search_corpus_tool.py:219-249`, `local_rag.py:296-314`

The non-hybrid (default) path calls `index.search()` directly, never
checks `deleted_utc`, and lacks the tombstone-aware fetch multiplier that
`vector_store.search_similar_chunks` and `search_bm25` implement. Latent
today (all 26 chunk maps have `tombstone_count: 0`) but the first
document the watcher tombstones will resurface in search results. Also:
these paths fetch exactly `top_k` and `continue` on skips → silently
fewer than `top_k` results with rank gaps.

### R4 — MEDIUM: `hybrid_degraded` flag dies at the text formatter

`server.py:288-317`, `search_corpus_tool.py:386-412`

The flag is correctly set and propagated on result dicts, but neither
`format_results_for_context` nor `_format_results` reads it, and MCP /
agentic consumers only see formatted text. A degraded response is
indistinguishable from healthy fused retrieval; the "rebuild your BM25
index" hint never reaches the one consumer who could act on it.

### R5 — LOW: wrong fallback citation slug; surfaces diverge

`search_corpus_tool.py:226-231,354-358,376-378` vs `server.py:210,252`

On chunks without `source` front matter, the agentic tool falls back to
the chunk filename/path → citation renders as `[ch-0001]`. The MCP
server's fallback (`Path(chunk_path).parts[0]`) correctly yields the doc
slug. Same input, two different citations, one wrong.

### R6 — LOW: MCP numeric-argument validation gaps

`server.py:437,444-458`

No lower bound / type check on `top_k`; `int(...)` coercions run before
the `try` at line 460 so malformed values raise out of `call_tool`
instead of returning graceful `TextContent`; `RerankConfig.validate()` /
`HybridConfig.validate()` never invoked (`rerank_pool_size=-5` silently
degrades pool to `top_k`).

### R7 — LOW: rerank call sits outside error envelope in agentic tool

`search_corpus_tool.py:153-157` — `try/except` covers the search only;
reranker model-load failure (e.g. offline first run) propagates raw out
of `execute()`. The agentic loop happens to catch it; other callers
won't.

### R8 — LOW: result-shape inconsistencies

- Rank gaps from `enumerate` + `continue` in both hybrid enrichment loops.
- `score` semantics differ per surface: MCP dense = `exp(-L2)` (higher
  better), agentic/local_rag dense = raw L2 (lower better), hybrid = RRF.
- `format_citation_prefix` renders inverted page ranges as-is.

### Retrieval contracts verified HOLDING

Citation prefix format (en-dash, degradation variants, never naked
commas; `grounding/citations.py`); rerank pool ordering
`max(hybrid.pool, rerank.pool, top_k)` with truncate-last at all three
call sites (functionally inverted by R1 until fixed); MCP tool-arg
defaults mirror CLI, read from invocation not config.yaml;
parallel-array contract structurally intact in the writers;
`hybrid.py` degraded fallback correct at library level (one WARNING,
flag set, key absent on happy path).

---

## D — Determinism core / corpus integrity

### D1 — HIGH: stale chunk files survive re-ingest and get embedded

`grounding/writer.py:48-58`, `grounding/cli.py:390,410`

`write_document` never removes pre-existing `ch_*.md`; embedding
generation globs `ch_*.md` from disk rather than trusting the manifest's
`chunk_count`. Re-ingesting a revised document that chunks 80 where the
old version chunked 120 leaves old chunks 81–120 on disk (old doc_id in
front matter), embedded, and retrievable as live content with stale page
citations. **This corrupts the corpus on the ordinary update workflow.**

**Fix:** clear `chunks/` (or delete `ch_*.md`) before writing; have the
embedder trust `chunk_count` or reconcile against the manifest.

### D2 — HIGH: slug collisions undetected → silent directory merge

`grounding/pipeline.py:205-211`, `grounding/utils.py:8-47`

`slugify` is non-injective by design (`Report 2024.pdf` /
`report_2024.PDF` / `Report-2024.pdf` → `report-2024`). The pipeline
checks doc_id collisions but never slug collisions, and slug is the
directory name: the second document silently overwrites the first's
`doc.md`/`meta.yaml` while the manifest keeps **both** entries pointing
at the same `doc_path`. Compounds with D1 (orphaned chunks from the
loser remain and get embedded).

**Fix:** detect slug collision at registration (manifest lookup by slug
with differing file_sha1) and fail the file or de-dupe the slug.

### D3 — MEDIUM: doc_id collision handling only warns; manifest silently evicts the loser; cross-run collisions undetected

`grounding/pipeline.py:303-322`, `grounding/manifest.py:113-119`

On collision: WARNING + stats, both files keep the same doc_id;
`register_document` keys by doc_id so the second registration overwrites
the first → orphan directory invisible to agent filtering.
`seen_doc_ids` is per-run only. 32-bit id: P(collision) ≈ 1% at 10k
docs, 50% at ~77k.

### D4 — MEDIUM: doc_id is derived from formatted markdown, not file SHA-1 (contradicts CLAUDE.md)

`grounding/pipeline.py:273-282,296-297`, `grounding/cli.py:928-934`

`doc_id = short_doc_id(sha1(markdown))` where the markdown front matter
embeds the filename and chunking params. Renaming a byte-identical PDF
or changing `--chunk-size` yields a different doc_id → duplicate
manifest entries for the same content under one slug. Decide which
identity is canonical and align code + docs.

### D5 — MEDIUM: `atomic_write` has no fsync

`grounding/utils.py:85-94` — atomic vs. process crash, not vs. power
loss: ext4 delayed allocation can commit the rename with unwritten data
blocks → torn `_index.json` blocks all subsequent ingestion until manual
repair. fsync the temp fd before rename (and ideally the dir after).

### D6 — MEDIUM: manifest has no concurrent-writer protection

`grounding/manifest.py:93-126`, `grounding/pipeline.py:187-192,385-391`

Load-at-start → mutate → write-at-end. Watcher + manual CLI run
interleaved = classic lost update; B's write silently drops every doc A
registered. Embeddings have a lock; the manifest has nothing.

### D7 — MEDIUM: empty-slug fallback diverges from retrieval-time slug re-derivation

`grounding/pipeline.py:206` — `slug = slugify(path.name) or path.stem`:
a filename with no `[a-z0-9]` content lives at `corpus/<raw-stem>/` while
the citation layer re-derives `slugify(source)` = `""`. Citation can't be
joined back to the directory.

### D8 — LOW: `langchain-text-splitters` unpinned upward and unrecorded

Splitter semantics changed across releases; only a lower bound declared;
`meta.yaml` records parser version but not splitter version. Cross-
machine chunk drift is possible and undiagnosable from provenance.

### D9 — LOW: chunk offset-mapping silent-mismatch window

`grounding/chunker.py:167-182` — `text.find(chunk, cursor)` locks onto a
later duplicate if a chunk's true position precedes the cursor; wrong
`char_start` → wrong page/section, silently.

### D10 — LOW: blake3 fallback mislabels SHA-256 digests as blake3

`grounding/hashing.py:75-82` — broken installs produce metadata that
mismatches verification later with no on-disk indication. Label the
fallback or hard-fail.

### D11 — LOW: manifest forward-compat raises raw `TypeError`; `_validate_raw_manifest` requires `updated_utc` which the documented schema omits

`grounding/manifest.py:105,129-143`

### Determinism invariants verified HOLDING

`slugify` pure/stable (no locale or unicodedata dependence);
`atomic_write` temp co-located + cleanup on failure; per-file error
handling never aborts batch; manifest write deterministic
(`sort_keys=True`, docs sorted by `(slug, doc_id)`); corrupt-JSON load
fails loudly as `ManifestError`; chunker deterministic within a pinned
environment; doc_id reproducible for fixed inputs (no timestamps in
front matter).

---

## W — Staging watcher / embeddings lifecycle

### W1 — HIGH: FAISS→BM25 append is non-transactional and never reconciled

`grounding/cli.py:460-465`

FAISS append succeeds, BM25 append fails (`BM25FormatError`, tokenizer
mismatch, I/O) → CLI exits non-zero with FAISS already advanced. Next
`--incremental` computes staleness from the FAISS map only
(`check_index_staleness`, `cli.py:302`) → docs never re-appended to
BM25. Permanent, silent, only a full rebuild heals it.

**Fix:** BM25-vs-FAISS chunk-count reconciliation on load (cheap), or
write BM25 before/with FAISS, or derive staleness from both maps.

### W2 — HIGH: BM25 fresh-write inside append shadows the documented degraded mode

`grounding/bm25.py:290-297`

FAISS exists, BM25 absent (every pre-19.1 agent — exactly the population
that should get the `hybrid_degraded` WARNING): first `--incremental`
writes a fresh BM25 index containing **only the newly appended chunks**.
`load_bm25_index` then succeeds forever, the degraded warning never
fires, and lexical coverage spans ~30 chunks of a 16k-chunk corpus with
no signal anywhere.

### W3 — HIGH/MED (lock cluster): replace the embedding lock with `flock`

`scripts/staging-watcher.sh:171-192,487`

Three defects, one fix:
- Staleness is mtime-only and never refreshed → a legitimately long run
  (>`LOCK_TIMEOUT`=3600s; plausible: full rebuild of a 596k-chunk agent,
  O(corpus) BM25 re-concat) gets its lock stolen → two incremental
  writers interleave four independent renames → mismatched
  index/map pairs (then permanent `ValueError`, see W4). Fixed temp
  name `_embeddings.tmp` (`vector_store.py:178,740`) also collides.
- Acquisition is check-then-write (TOCTOU); release is unconditional
  (deletes a thief's lock).
- PID is written but never read — documented PID-staleness check does
  not exist; `kill -9` mid-update strands a 1-hour embedding blackout
  (the `trap` at line 487 does not release the lock).

`flock` on a held FD is kernel-released on crash and needs no timeout
heuristics.

### W4 — MEDIUM: crash between FAISS index and chunk-map renames bricks incremental mode

`grounding/vector_store.py:738-755`, `grounding/cli.py:318-322`

Crash between the two renames → `index.ntotal > index_size` →
`load_vector_index` raises `ValueError`; the CLI's incremental fallback
catches only `FileNotFoundError` → every subsequent run crashes until a
manual full rebuild. **Fix:** catch `ValueError` too and fall back to
full rebuild.

### W5 — MEDIUM: tombstone pair non-atomic; BM25 side never re-detected

`grounding/cli.py:365-367,456-457` — crash between
`tombstone_documents` (FAISS) and `tombstone_bm25_documents` →
deleted docs live forever in BM25; next run's `get_indexed_doc_ids`
excludes FAISS-tombstoned chunks so the doc never reappears in
`deleted_docs`.

### W6 — MEDIUM: hand-rolled YAML parsing in `find_affected_agents` silently misses agents

`scripts/staging-watcher.sh:110-153` — block-style unquoted lists only.
Flow style (`collections: [a, b]`) → zero matches; quoted entries fail
the equality check. Agent silently never gets embedding updates; the
Python side parses real YAML so `grounding agents show` looks fine while
the watcher disagrees. **Fix:** shell out to Python/yq for the match.

### W7 — MEDIUM: skipped/failed embedding updates never retried

`scripts/staging-watcher.sh:220-222,240-243` — lock-held skip or
non-zero exit leaves docs unsearchable with no requeue. FAISS self-heals
opportunistically on the next ingestion touching the same agent (the
staleness diff picks up missed docs) — which may be weeks or never for a
quiet collection. The BM25 half of a W1 partial failure does not
self-heal even then.

### W8 — MEDIUM: OCR backlog re-OCRs permanently failing PDFs every cycle

`scripts/staging-watcher.sh:253-314` — "completed but no output" PDFs
stay in `skipped/` and are re-OCR'd on every event in that collection;
serial inotify loop → one poison PDF taxes every cycle. Add a failure
counter / quarantine.

### W-LOW (tracked, fix opportunistically)

- Duplicate slugify (bash `staging-watcher.sh:84` vs Python) decides
  success/failure; divergence can move a never-ingested file to
  `originals/` (silent data loss path) or a successful one to `skipped/`.
- `mv` without `-n` clobbers archived originals on same-name re-ingest
  (provenance loss).
- Per-chunk embedding failure is permanent (doc looks current to
  staleness; chunk never embedded) — `cli.py:436-437`.
- `get_chunk_metadata` rejects v1.2 maps (`vector_store.py:403` checks
  `!= FORMAT_VERSION_WITH_METADATA` only) → music metadata vanishes
  after first incremental append.
- inotify match is lowercase-extension only (`Report.PDF` never
  triggers); `$STAGING_DIR` interpolated unescaped into the regex.
- `--emit-embeddings` ingestion path writes FAISS only (no BM25
  sidecar) and swallows write failures (`controller.py:735-761`).
- Diverged agents repo → `--ff-only` fails every batch forever; stale
  config with only ERROR log lines, no backoff/escalation.

### Watcher contracts verified HOLDING

Git pull `--ff-only` at startup and before every batch, proceed-on-
failure as coded intent; scanned-PDF threshold 1000 chars/MB (denominator
is `size_mb + 0.1`, slightly damping tiny files; missing pdftotext fails
safe to skipped); AUTO_EMBEDDINGS collection→agent matching, per-agent
incremental, lock-held-skip-logged (modulo W3/W6); incremental core
mechanics (manifest−index diff, soft-delete + query-time filter with
fetch multiplier, tokenizer identity check with `BM25FormatError` +
rebuild hint); per-file atomic writes throughout; shell quoting on
user-controlled paths generally clean.

---

## P — publish.sh

Verified end-to-end: a simulation of the full sanitize pipeline
(excludes → scrubs → line deletions → leak scan) on current HEAD passes
clean. Design is scrub-narrow / scan-wide and every scrub target is
backstopped by a leak pattern, so failures are loud, not silent.
`context/`, `embeddings/`, `dist/`, `venv/` are untracked. Replay mode
publishes commit subjects only (bodies never leave private).

### P1 — MEDIUM: squash-mode commit subject not leak-scanned

`scripts/publish.sh:152` — the subject pre-scan runs only in replay
mode, but squash mode puts HEAD's subject verbatim into the public
commit message. Move the scan outside the `if [ "$MODE" = "replay" ]`.

### P2 — LOW: binaries exempt from both scrub and scan

`grep -rI` skips binaries (and UTF-16 text). Today's tracked binaries
are benign (test_pdfs fixtures incl. the public FDA guidance PDF, all
already mirrored), but any future tracked binary ships unscanned.
Options: scan `pdftotext` output of tracked PDFs, or maintain a tracked-
binary allowlist and fail on new ones.

### P3 — INFO (by design, documented here)

- Manual commits made directly on the public repo are silently clobbered
  on next publish (tree-wipe + replace) and break `LAST_PUB_SHA`
  recovery into a quiet squash fallback.
- `|| true` on the scrub xargs swallows perl failures; backstopped by
  the leak scan for known patterns only.
- Files outside `TEXT_EXTENSIONS` (e.g. `grounding-watcher.service.example`)
  are scanned but not scrubbed — a leak there blocks publish rather than
  auto-scrubbing. Acceptable.

---

## Working tree at review time (for the record)

Uncommitted changes reviewed and benign: CLAUDE.md cross-repo
propagation section; epic-16 Story 16.5 addition; timeout 120s→600s in
`scripts/agentic.py:84` and `scripts/local_rag.py:408` (note: duplicated
constant, drifted in lockstep manually — candidate for a shared config
value when Epic 21 touches these files).
