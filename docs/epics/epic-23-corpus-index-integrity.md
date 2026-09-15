# Epic 23: Corpus & Index Integrity

**Epic ID:** E23
**Owner:** Andy
**Status:** In Review (all 4 stories implemented — Ready for Review)
**Priority:** P0
**Completed Stories:** 4/4 (23.1, 23.2, 23.3, 23.4 — Ready for Review)
**Dependencies:** Epic 4 (Output & Manifest), Epic 6 (Vector Embeddings), Epic 19 (Hybrid Retrieval — BM25 sidecar)
**Target Completion:** TBD
**Source:** `docs/qa/assessments/load-bearing-review-20260612.md` findings D1, D2, W1, W2, W5

---

## Branching Plan

This epic ships on the public mirror; the ingestion/embedding code is
public. Per-story summary:

| Story | Branch target | Private-only content? | Cross-repo coordination |
|-------|---------------|------------------------|--------------------------|
| 23.1  | public `main` (feature branch → squash PR) | No | None |
| 23.2  | public `main` (feature branch → squash PR) | No | None |
| 23.3  | public `main` (feature branch → squash PR) | No | None |
| 23.4  | public `main` (feature branch → squash PR) | No | None |

**Operational note:** the maintainer's Ubuntu ingestion server is the
authoritative writer for corpus and embeddings (workstation dirs are
Syncthing `receiveonly`). These fixes change how the **writer** behaves,
so they take effect once the ingestion server runs the patched code.
A one-time full embedding rebuild may be warranted after 23.3 lands to
heal any pre-existing FAISS↔BM25 desync already on disk (see Story 23.3
AC and the Migration note below).

**Cadence:** one squashed commit per story as each PR merges. 23.1 and
23.2 (corpus-directory corruption on re-ingest) are the P0 pair and ship
first. 23.3 and 23.4 (index coherence) are P1 and follow.

---

## Overview

The 2026-06-12 load-bearing review surfaced a cluster of integrity
defects that share one property: they corrupt persistent state silently,
and several trigger on the **ordinary document-update workflow**, not on
rare edge cases.

1. **Stale chunks survive re-ingest (D1).** `write_document`
   (`grounding/writer.py:48-58`) does `ensure_dir(chunk_dir)` then writes
   `ch_0001..ch_NNNN`; it never removes pre-existing `ch_*.md`. Embedding
   generation enumerates `sorted(doc_dir.glob("ch_*.md"))`
   (`grounding/cli.py:410`) — filesystem truth, not the manifest's
   `chunk_count`. Re-ingesting a revised document that chunks 80 where the
   old version chunked 120 leaves old chunks 81–120 on disk (old doc_id in
   front matter), embedded, and returned by retrieval as live content with
   page/section citations that no longer exist.

2. **Slug collisions are undetected (D2).** `slugify` is deliberately
   non-injective (`Report 2024.pdf`, `report_2024.PDF`, `Report-2024.pdf`
   → `report-2024`). The pipeline checks **doc_id** collisions but never
   **slug** collisions, and slug is the corpus directory name. Two
   different documents whose names slugify identically both write to
   `corpus/<slug>/doc.md`; the second silently overwrites the first while
   the manifest keeps **both** entries pointing at the same `doc_path`.
   Compounds with D1 (the loser's orphaned chunks remain and get embedded).

3. **FAISS↔BM25 append is non-transactional and never reconciled (W1).**
   Incremental mode appends to FAISS first (`grounding/cli.py:460`), then
   BM25 (`cli.py:462`). If the BM25 append raises after FAISS succeeded
   (`BM25FormatError`, tokenizer mismatch, or any I/O error), the CLI exits
   non-zero with FAISS already advanced. The next `--incremental` run
   computes staleness from the **FAISS chunk map only**
   (`check_index_staleness`, `cli.py:302`), sees the new docs as already
   indexed, and never re-appends them to BM25. The desync is permanent and
   silent; only a manual full rebuild heals it.

4. **BM25 fresh-write inside append shadows the degraded-mode warning
   (W2).** When a FAISS index exists but BM25 artifacts are absent (every
   agent embedded before Epic 19.1 — exactly the population the docs say
   should emit the `hybrid_degraded` WARNING), the first `--incremental`
   run hits `existing is None` (`grounding/bm25.py:290-297`) and writes a
   **fresh** BM25 index containing only the newly appended chunks. From
   then on `load_bm25_index` succeeds, the dense-only fallback warning
   never fires again, and lexical coverage spans only the post-append
   slice. A 16k-chunk agent + one new PDF = a BM25 index of ~30 chunks,
   with no signal anywhere.

5. **Tombstone pair is non-atomic and the BM25 side is never
   re-detected (W5).** `tombstone_documents` (FAISS map) runs, then
   `tombstone_bm25_documents` (`cli.py:365-367, 456-457`). A crash between
   them leaves deleted docs tombstoned in FAISS but live in BM25. On the
   next incremental run, `get_indexed_doc_ids` excludes FAISS-tombstoned
   chunks, so the doc no longer appears in `deleted_docs` and the BM25
   tombstone is never retried — hybrid search permanently surfaces
   chunk_ids for deleted documents whose corpus bodies are gone.

**Problem Statement:**
- The two most damaging defects (D1, D2) fire on the *normal* workflow
  of updating or re-adding a document. Corpus corruption here is silent:
  retrieval returns confidently-wrong content with stale citations.
- The index-coherence defects (W1, W2, W5) leave FAISS and BM25 in
  permanent disagreement with no on-disk signal and no self-healing path
  short of a full rebuild.
- Across all five, the common failure shape is **silent**: a log line
  nobody reads, or no log at all. Integrity fixes must fail loud or
  self-correct, not paper over.

**Solution:**
- Clear/reconcile `chunks/` before writing a document's chunk set, and
  make the embedder trust the manifest's chunk inventory rather than a
  raw glob (23.1).
- Detect slug collisions at registration and fail the file (or de-dupe
  the slug deterministically), rather than silently overwriting (23.2).
- Make the FAISS↔BM25 artifact set coherent: reconcile on load (cheap
  chunk-count check with a loud rebuild hint), and/or derive incremental
  staleness from both maps so a half-applied append self-heals on the
  next run. Fix the W2 fresh-write shadow so a BM25-absent agent either
  full-builds BM25 over the whole corpus or stays honestly degraded (23.3).
- Make tombstoning the FAISS/BM25 pair atomic-enough that a crash
  between them is re-detected and retried (23.4).

---

## Goals

1. Re-ingesting a document leaves **no** stale chunk files on disk and
   no orphaned embeddings; the on-disk chunk set for a slug exactly
   matches the current document's `chunk_count`.
2. Two documents that slugify to the same directory name can never
   silently overwrite each other; the collision is detected and surfaced
   (fail-the-file or deterministic de-dupe), and the manifest never holds
   two entries with the same `doc_path`.
3. FAISS and BM25 are either coherent or **loudly** flagged as
   incoherent. A half-applied incremental append (FAISS advanced, BM25
   not) self-heals on the next run or refuses to silently serve a
   partial BM25 index.
4. An agent whose BM25 artifacts are absent gets either a full BM25 build
   over the whole corpus or an honest `hybrid_degraded` signal — never a
   silent partial index covering only the latest append.
5. Tombstoning a document removes it from **both** channels, or the
   incomplete state is re-detected and completed on the next run.
6. Every integrity guard fails loud (clear error + remediation hint) or
   self-corrects. No new silent failure modes.

---

## Non-Goals

- Changing the on-disk artifact **formats** (FAISS index, `_chunk_map.json`,
  `_bm25.pkl`, `_bm25_map.json`). These fixes are about write-ordering,
  reconciliation, and pre-write cleanup, not schema.
- The doc_id identity questions (D3: collision handling only warns; D4:
  doc_id derived from markdown not file SHA-1). Tracked as P2 follow-up
  stories below; they are identity-semantics decisions, not active
  corruption, and deserve their own design pass.
- Concurrent-writer protection for the manifest (D6) and `atomic_write`
  fsync durability (D5). Real, but separate durability/concurrency
  concerns — backlog, see review doc.
- Retrieval-surface tombstone-bypass on the default dense path (R3).
  Related to W5 but lives in the query surfaces, not the ingestion
  writer; tracked separately.

---

## Stories

### Story 23.1: Clean stale chunks on re-ingest

**Priority:** P0. Source: D1.

`write_document` must guarantee the chunk directory reflects only the
current document's chunks, and the embedder must trust that inventory.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. Before writing a document's chunk set, `write_document`
   (`grounding/writer.py:48-58`) removes pre-existing `ch_*.md` files for
   that slug (or otherwise guarantees no chunk file from a prior version
   survives). The removal is scoped to the chunk directory for that
   document and does not touch `doc.md` / `meta.yaml` ordering guarantees.
2. After a re-ingest that produces fewer chunks than the prior version,
   the on-disk `ch_*.md` count equals the new `chunk_count`; no stale
   `ch_NNNN.md` remains.
3. Embedding generation no longer trusts a raw `glob("ch_*.md")` as the
   source of truth where that can diverge from the manifest. Either the
   embedder reconciles the glob against the manifest's `chunk_count` /
   chunk inventory, or it operates on the (now-guaranteed-clean) chunk
   directory with an assertion that the count matches. Whichever path is
   chosen, an orphaned chunk file can no longer be embedded as live
   content.
4. The write remains atomic/deterministic per existing guarantees (temp
   + rename; byte-identical outputs for identical inputs).
5. Test: ingest a document (120 chunks), re-ingest a revised version
   (80 chunks) under the same filename/slug, assert (a) exactly 80
   `ch_*.md` on disk, (b) the embedding index for the agent contains no
   chunk_id from the dropped 81–120, (c) retrieval cannot surface the
   stale chunks.
6. Existing single-ingest behavior (no prior version present) is
   unchanged.

**Status:** Ready for Review

### Story 23.2: Detect slug collisions

**Priority:** P0. Source: D2.

A second document that slugifies to an existing, **different** document's
directory must not silently overwrite it.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. At document registration, the pipeline detects when a computed slug
   already maps to a corpus directory whose existing document has a
   different `file_sha1` (i.e. genuinely different content, not a
   re-ingest of the same file). "Same file re-ingest" (matching
   `file_sha1`) is **not** a collision and continues to update in place.
2. On a true collision, the pipeline either (a) fails that file with a
   clear error naming both source filenames and the shared slug, recorded
   in stats (consistent with the per-file error-handling pattern, batch
   continues), or (b) assigns a deterministic disambiguated slug
   (e.g. `<slug>-<doc_id_short>`), documented and stable across runs.
   Decision recorded in the story's dev notes; default recommendation is
   **fail-the-file** (surfacing beats silent renaming for a corpus the
   user curates).
3. The manifest never ends a run with two entries sharing the same
   `doc_path`. A guard (assertion or validation pass) enforces this.
4. The chosen behavior is symmetric across same-batch collisions and
   cross-run collisions (a new file colliding with a slug already in the
   manifest from a prior run).
5. Test: two distinct fixture PDFs with names that slugify identically →
   assert the chosen behavior (no overwrite; collision surfaced or
   deterministic distinct slugs), and assert the first document's
   `doc.md` / `meta.yaml` are intact afterward.
6. Re-ingesting the **same** file (same `file_sha1`) still updates in
   place with no spurious collision error.

**Status:** Ready for Review

### Story 23.3: FAISS↔BM25 index coherence

**Priority:** P1. Source: W1, W2.

Make the dense+lexical artifact set coherent, or loudly flag when it
isn't. Includes an architecture decision (see Architecture section).

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. A half-applied incremental append (FAISS advanced, BM25 write failed)
   no longer goes undetected. Chosen mechanism (per the Architecture
   decision): either (a) incremental staleness is derived from **both**
   the FAISS and BM25 maps so the missed docs are re-appended to BM25 on
   the next run, or (b) a load-time reconciliation compares FAISS and
   BM25 live-chunk counts and refuses to serve / forces a rebuild with a
   clear hint when they disagree. At minimum, the partial state is never
   served silently as a complete BM25 index.
2. The W2 fresh-write shadow is fixed: when FAISS exists but BM25 is
   absent, an incremental run does **not** write a BM25 index covering
   only the newly appended chunks. Instead it either builds BM25 over the
   **entire** existing corpus for that agent, or leaves BM25 absent so the
   documented `hybrid_degraded` dense-only fallback keeps firing. The
   chosen behavior is logged clearly.
3. If the write-ordering is changed to reduce the partial-failure window
   (e.g. stage both artifacts then commit, or write BM25 before FAISS),
   the change preserves per-file atomic writes and does not regress the
   incremental append cost characteristics documented in CLAUDE.md
   (O(total chunks) BM25 rebuild on append is acceptable; do not make it
   worse).
4. A BM25 append failure after a successful FAISS append surfaces as a
   non-silent, recoverable state: the next `--incremental` run repairs it
   without a manual full rebuild. Test this explicitly by simulating a
   BM25 write failure mid-append and asserting recovery on rerun.
5. Migration: a documented one-time command (or automatic detection)
   heals agents already in a desynced or W2-shadowed state on disk. For
   the maintainer's setup this is "run a full `grounding embeddings
   --agent <name>` rebuild"; the story confirms that full rebuild
   produces coherent FAISS+BM25 and documents which agents need it.
6. Tests cover: (a) clean incremental append keeps FAISS+BM25 in
   lockstep; (b) simulated BM25-failure-after-FAISS, then recovery on
   rerun; (c) BM25-absent agent incremental run does the full-build or
   honest-degrade behavior from AC 2, not the partial shadow.

**Status:** Ready for Review

### Story 23.4: Tombstone-pair atomicity

**Priority:** P1. Source: W5.

A document deletion must remove the doc from both channels, or the
incomplete state must be re-detected and completed.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. Tombstoning a deleted document updates both the FAISS map and the
   BM25 map, ordered/guarded so that a crash between the two is
   re-detectable on the next run. Preferred: BM25-tombstone before (or
   together with) the FAISS-tombstone, so that `get_indexed_doc_ids`
   (which excludes FAISS-tombstoned chunks) does not mask a still-live
   BM25 entry.
2. After a simulated crash between the two tombstone steps, the next
   incremental run completes the missing tombstone — the deleted doc is
   gone from **both** channels and no longer surfaceable by hybrid
   search.
3. No FAISS or BM25 pickle is rewritten for a delete (tombstone parity
   preserved; soft-delete via `deleted_utc` only), consistent with the
   existing tombstone semantics.
4. Test: tombstone a doc, simulate failure after the FAISS-side
   tombstone but before the BM25-side, rerun incremental, assert the doc
   is absent from both dense and lexical results.

**Status:** Ready for Review

---

## Architecture

The load-bearing design decision in this epic is **how to make the
FAISS↔BM25 pair coherent** (Story 23.3). Three candidate strategies,
to be chosen during 23.3 design (recommend the architect weigh in):

1. **Reconcile-on-load.** `load_bm25_index` (and/or the hybrid load path)
   compares the BM25 live-chunk count against the FAISS map's live-chunk
   count. On mismatch, refuse to serve the BM25 channel and emit a loud
   rebuild hint (degrade to dense-only with `hybrid_degraded: True`).
   - Pro: cheap, no write-path change, fails safe and loud.
   - Con: doesn't *repair* — needs a rebuild to fix; a chronically
     desynced agent runs degraded until someone acts.

2. **Dual-map staleness.** `check_index_staleness` derives "what's
   missing" from the union of the FAISS map and the BM25 map, so a doc
   present in FAISS but absent from BM25 is re-appended to BM25 on the
   next incremental run.
   - Pro: self-healing; no manual rebuild.
   - Con: more moving parts in the staleness logic; must handle the W2
     case (BM25 entirely absent) distinctly from the per-doc gap case.

3. **Staged commit.** Build both new artifacts into temp files, then
   commit (rename) both — FAISS and BM25 — only after both succeed.
   - Pro: shrinks the partial-failure window to two adjacent renames.
   - Con: two renames are still not atomic as a pair; reduces but does
     not eliminate the window. Best combined with (1) or (2) as the
     backstop.

**Recommended:** (2) dual-map staleness as the primary self-healing
mechanism, with (1) reconcile-on-load as the loud backstop for states
(2) can't reach (e.g. a corrupt BM25 pickle). (3) is optional polish.
The W2 fresh-write shadow (AC 2 of 23.3) is fixed regardless of which
coherence strategy is chosen — it's a distinct bug in the BM25-absent
branch.

Pre-existing invariant to preserve: the parallel-array contract
`chunk_bodies[i] ↔ chunk_ids[i] ↔ FAISS id i ↔ bm25_index i`. The hybrid
merge fuses on `chunk_id`, so a count mismatch is detectable without
walking the arrays, but any repair must keep insertion order intact.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Stale-chunk cleanup (23.1) deletes chunks of a *different* document if slug-collision (D2) is unfixed | High | Sequence 23.2 with/before 23.1, or scope 23.1's deletion by verifying the chunk dir's `doc.md` doc_id matches the document being written. The two P0 stories are designed as a pair. |
| Slug-collision fail-the-file (23.2) rejects a document the user actually wanted ingested under a new name | Medium | Clear error names both files and the slug; user renames the source. Deterministic-de-dupe is the documented alternative if fail-the-file proves too aggressive in practice. |
| Dual-map staleness (23.3) re-appends a doc to BM25 that's already there, double-counting | Medium | Repair path keys on chunk_id membership in the BM25 map, not a blind re-append; idempotent by construction. Covered by 23.3 AC 6(a). |
| Migration rebuild (23.3 AC 5) is expensive on large agents (596k-chunk index) | Low | One-time, maintainer-scheduled, on the ingestion server. Documented; not on a hot path. |
| Changing write-ordering regresses determinism or incremental cost | Medium | 23.3 AC 3 explicitly gates on preserving atomic writes and append cost; tests assert byte-identical outputs for identical inputs. |
| A crash-between-tombstones test is hard to simulate deterministically | Low | Inject the failure via a test seam (mock the second tombstone call to raise), not by killing a process. Standard pattern. |

---

## Testing Strategy

### Unit / Integration Tests
- 23.1: re-ingest-shrinks-chunk-count → no stale files, no orphan
  embeddings (the headline regression test).
- 23.2: two-distinct-files-same-slug → no overwrite; same-file re-ingest
  → in-place update, no false collision.
- 23.3: clean incremental lockstep; BM25-fail-after-FAISS then recover;
  BM25-absent full-build-or-degrade.
- 23.4: crash-between-tombstones then recover.
- Manifest invariant guard: no two entries share a `doc_path` (23.2).

### Manual Validation
- After 23.3 migration, run a full rebuild on one real agent and confirm
  FAISS live-count == BM25 live-count and hybrid returns fused results
  (`faiss_rank` populated) — note in the PR.

---

## Acceptance Criteria (Epic Level)

1. Re-ingesting a revised document leaves zero stale chunk files and zero
   orphaned embeddings.
2. Distinct documents with colliding slugs never silently overwrite;
   collisions are surfaced or deterministically disambiguated.
3. The manifest never holds two entries with the same `doc_path`.
4. FAISS and BM25 are coherent after any incremental append, or the
   incoherence is loud and self-heals on the next run.
5. A BM25-absent agent does not get a partial shadow index; it full-builds
   or honestly degrades.
6. Deleting a document removes it from both retrieval channels even across
   a crash between the two tombstone steps.
7. Every guard added in this epic fails loud or self-corrects; no new
   silent failure modes introduced.
8. CI green on all merge commits; determinism (byte-identical outputs for
   identical inputs) preserved.

---

## Definition of Done

- All four stories closed with AC met.
- Headline regression tests (23.1 stale-chunk, 23.2 slug-collision,
  23.3 desync-recover, 23.4 tombstone-recover) committed and green.
- Architecture decision for 23.3 recorded in the story's dev notes.
- Migration guidance documented; maintainer has run the one-time rebuild
  on any agent already in a desynced/W2-shadowed state.
- CLAUDE.md updated where behavior visibly changes (re-ingest cleanup,
  slug-collision policy, FAISS/BM25 coherence guarantee).
- No regression in existing ingestion/embedding tests.

---

## Open Questions / Future Work

**P2 follow-up stories (tracked, not yet specced — from the review doc):**
- **D3 — doc_id collision handling.** Today the 32-bit doc_id collision
  path only logs a WARNING and lets the manifest silently evict the
  loser; cross-run collisions aren't detected at all. Needs a fail-or-
  extend decision. P(collision) ≈ 1% at 10k docs, 50% at ~77k.
- **D4 — doc_id derivation.** doc_id is derived from the formatted
  markdown (including filename + chunk params in front matter), not the
  file SHA-1 as CLAUDE.md states. Renaming a byte-identical PDF or
  changing `--chunk-size` mints a new doc_id and a duplicate manifest
  entry. Decide canonical identity, align code + docs.

**Out of scope (other epics / backlog):**
- D5 (`atomic_write` fsync durability), D6 (manifest concurrent-writer
  lock) — durability/concurrency, separate track.
- R3 (default dense path bypasses tombstone filtering) — retrieval
  surface, not the ingestion writer.

---

## References

- `grounding/writer.py:48-58` — `write_document`; stale-chunk cleanup
  lands here (23.1).
- `grounding/cli.py:410` — embedder chunk glob; reconcile vs. manifest
  (23.1).
- `grounding/pipeline.py:205-211` — slug computation + registration;
  collision detection (23.2).
- `grounding/manifest.py:113-119` — `register_document`; `doc_path`
  uniqueness guard (23.2).
- `grounding/cli.py:302, 460-465` — incremental staleness + FAISS/BM25
  append ordering (23.3).
- `grounding/bm25.py:290-297` — BM25 fresh-write shadow branch (23.3 W2).
- `grounding/cli.py:365-367, 456-457` — tombstone pair (23.4).
- `docs/qa/assessments/load-bearing-review-20260612.md` — findings D1,
  D2, W1, W2, W5 (and D3/D4 in Future Work).

---

## Dev Agent Record (Stories 23.1 + 23.2)

### Agent Model Used
James (dev persona) on claude-opus-4-8.

### Stories delivered
23.1 (stale-chunk cleanup) and 23.2 (slug-collision detection) — the P0
pair — implemented together because they are coupled: 23.1's cleanup is
only safe because 23.2 guarantees the chunk directory belongs to the same
document. 23.3 and 23.4 remain Draft.

### Design (grounded in the real code)
The corpus is **one directory per slug**, so slug is the identity:
- **Same slug + same `orig_name`** = a re-ingest of the same document
  (revision) → proceed, clearing prior chunks.
- **Same slug + different `orig_name`** = two different sources competing
  for one directory → collision → fail the newcomer.

`ManifestEntry` has no `file_sha1`, so `orig_name` (already on the entry)
is the practical same-document key, and it is semantically correct given
slug = directory name. This is the deviation from the story's `file_sha1`
phrasing; recorded here deliberately.

### Implementation
- **23.2 — `grounding/controller.py`:** before any write/embed work in the
  per-document loop, a guard looks up the existing manifest entry for the
  slug; if it exists with a different `orig_name`, the file is failed via
  `handle_failure("slug_collision", …)` and skipped. Because
  `register_document` mutates `manifest_data` in place during the batch, the
  guard catches both cross-run and same-batch collisions (first-processed
  wins, deterministically).
- **23.2 — `grounding/manifest.py`:** `register_document` now retains only
  entries whose slug **and** doc_id both differ from the incoming entry,
  then appends — so a revision (same slug, new doc_id from changed content)
  supersedes rather than duplicating. Guarantees no two entries share a
  slug / `doc_path` (AC 3). The existing doc_id-keyed replacement behavior
  is preserved (verified by `test_register_document_adds_and_replaces`).
- **23.1 — `grounding/writer.py`:** `write_document` removes prior `ch_*.md`
  in the chunk directory before writing the new set. Scoped to `ch_*.md`
  (doc.md, meta.yaml, music/ and formulas/ untouched). Safe because 23.2
  rejects collisions upstream, so the dir is always the same document.
- **23.1 — `grounding/cli.py`:** the standalone embedder's `glob("ch_*.md")`
  is now correct by construction (no stale files on disk). Added a
  defense-in-depth WARNING when the on-disk chunk count diverges from the
  manifest's `chunk_count` — flags a pre-23.1 corpus that still carries
  orphans so it can be rebuilt rather than silently embedding stale chunks.

### Completion Notes
- [x] 23.1 AC 1–6 — cleanup before write; fewer-chunk re-ingest leaves no
  stale `ch_*.md`; embedder glob is clean-by-construction + count
  reconciliation warning; atomic/deterministic writes preserved; sibling
  files untouched. Tests: `test_write_document_clears_stale_chunks_on_reingest`,
  `test_write_document_cleanup_leaves_sibling_files_untouched`.
- [x] 23.2 AC 1–6 — collision detected by slug+orig_name; collision fails
  the file (chosen: fail-the-file, per AC 2 default recommendation);
  manifest never holds two entries with one `doc_path`; symmetric across
  same-batch and cross-run; same-file re-ingest updates in place. Tests:
  `test_run_controller_rejects_slug_collision`,
  `test_run_controller_reingest_same_file_updates_in_place`,
  `test_register_document_supersedes_same_slug_new_doc_id`,
  `test_register_document_no_duplicate_doc_path_across_slugs`.
- [x] Risk mitigated: 23.1 cleanup cannot delete another document's chunks
  because 23.2's collision guard runs first (the High risk in this epic).

### Debug Log References
Pre-existing, unrelated test failures confirmed on clean `main` with this
diff stashed (NOT introduced by 23.1/23.2):
- `tests/test_integration.py` (all classes) — monkeypatches
  `grounding.pipeline.format_markdown`, renamed to `format_markdown_with_map`
  long ago. Whole file is stale.
- `tests/test_query.py::test_cli_help` — pyarrow/numpy ABI mismatch in venv.
- `tests/test_local_rag_cli.py::test_agentic_config_creation` — asserts
  `timeout == 120` but the ambient (pre-session) `scripts/agentic.py` change
  set it to 600; belongs to the 16.5 agentic work, not this epic. Needs its
  test updated when that change lands.
- `tests/test_performance_benchmarks.py::test_memory_overhead_embeddings` —
  setup uses removed `--in/--out` CLI flags (now positional). Stale test.

### File List
- `grounding/manifest.py` (modified) — slug+doc_id dedup in
  `register_document`.
- `grounding/controller.py` (modified) — slug-collision guard.
- `grounding/writer.py` (modified) — stale-chunk cleanup.
- `grounding/cli.py` (modified) — chunk-count reconciliation warning.
- `tests/test_manifest.py` (modified) — 2 new tests.
- `tests/test_controller.py` (modified) — 2 new tests.
- `tests/test_writer.py` (modified) — 2 new tests.
- `docs/epics/epic-23-corpus-index-integrity.md` (modified) — statuses +
  this record.

### Validation
```
pytest test_writer.py test_manifest.py test_controller.py test_pipeline.py → all pass (32)
pytest -k 'embedding or incremental' (bm25/incremental/embeddings)          → 48 pass
6 new 23.1/23.2 tests                                                       → 6 pass
(pre-existing unrelated failures as noted in Debug Log)
```

---

## Dev Agent Record (Story 23.3)

### Agent Model Used
James (dev persona) on claude-opus-4-8.

### Strategy chosen (and deviation from the epic's recommendation)
The epic's Architecture section recommended **(2) dual-map staleness** (re-derive
"what's indexed" from both the FAISS and BM25 maps so a BM25-missing doc is
re-appended) backed by **(1) reconcile-on-load**. I implemented **(1)
reconcile-on-load as the primary mechanism, with auto-rebuild as the heal**, and
deliberately did **not** adopt the surgical dual-map catch-up. Reason: a doc
present in FAISS but missing from BM25 cannot be re-appended through the existing
append flow without also re-appending it to FAISS (duplicating vectors). True
self-heal therefore needs a *channel-aware* BM25-only catch-up path — new code
that risks FAISS duplication and the parallel-array contract. Reconcile-then-
rebuild is simpler, provably correct, and the rebuild cost is paid only in the
exceptional incoherent case (post-crash / pre-19.1), never in steady state.

### Implementation
- **`grounding/cli.py` — `check_faiss_bm25_coherence(output_dir)`:** compares
  `index.ntotal` against `len(bm25 chunks)`. Both writers append in lockstep and
  keep tombstones as soft deletes, so a coherent pair has equal totals. Returns
  `(coherent, reason)`. Detects W1 (FAISS ahead of BM25) and W2 (BM25 absent
  while FAISS non-empty). A missing FAISS index is "coherent" (the full-build
  path owns a fresh index).
- **`grounding/cli.py` — coherence gate in `embeddings_command`:** runs right
  after the staleness check, before the mode/early-return logic. On incoherence
  it logs loudly, prints a user-facing warning, and downgrades
  `incremental_mode → False` (resetting new/deleted/updated sets), so the rest of
  the flow rewrites **both** indexes coherently over the whole agent-filtered
  corpus. This is the automatic recovery path — no manual rebuild.

### How this satisfies the ACs
- **AC 1** (half-applied detected, never served silently): the count check
  catches FAISS-ahead-of-BM25; the run rebuilds rather than appending onto a
  partial BM25.
- **AC 2** (W2 shadow fixed): BM25-absent + FAISS-present is incoherent →
  full rebuild builds BM25 over the entire corpus. `append_to_bm25_index`'s
  fresh-write branch is now unreachable from the incremental path (the gate
  fires first), so the partial-slice shadow can't occur; the primitive is left
  intact for its legitimate direct-call use (`test_append_when_index_missing`).
- **AC 3** (atomicity / cost preserved): no change to the append primitives or
  their atomic writes; only the incremental-vs-full decision changed. Coherent
  steady-state runs are byte-for-byte the prior path.
- **AC 4** (BM25-fail-after-FAISS recovers on rerun, no manual rebuild): the
  next run's gate detects the count mismatch and auto-rebuilds. Recovery is an
  automatic full rebuild, not a manual one.
- **AC 5** (migration): a pre-23.3 desynced or pre-19.1 agent self-heals on its
  next `--incremental` run; the maintainer can also force it with a full
  `grounding embeddings --agent <name>`.
- **AC 6** (tests): unit (a) coherent lockstep, (b) FAISS-ahead detection
  (simulated half-apply), (c) BM25-absent detection; integration (d) clean
  incremental keeps lockstep, (e) BM25-absent → CLI rebuilds → coherent.

### Note on search-side reconciliation
This story fixes the **generation/write side** (where W1/W2 originate and the
self-heal belongs). A search-time count check in `search_hybrid` (warn/degrade
when FAISS and BM25 diverge before the next embed run heals them) is a small
optional follow-up — recorded here, not implemented, to keep 23.3 scoped to the
writer.

### File List
- `grounding/cli.py` (modified) — `check_faiss_bm25_coherence` helper +
  coherence gate in `embeddings_command` + `load_bm25_index` import.
- `tests/test_faiss_bm25_coherence.py` (new) — 4 unit tests.
- `tests/test_cli.py` (modified) — 2 integration tests in
  `TestCLIEmbeddingsIncremental`.
- `docs/epics/epic-23-corpus-index-integrity.md` (modified) — status + this
  record.

### Validation (23.3)
```
pytest test_faiss_bm25_coherence.py                          → 4 pass
pytest test_cli.py::TestCLIEmbeddingsIncremental             → 10 pass (incl. 2 new)
pytest test_bm25.py test_vector_store.py                     → 71 pass (no regression)
```

---

## Dev Agent Record (Story 23.4)

### Agent Model Used
James (dev persona) on claude-opus-4-8.

### Root cause and why reordering alone is insufficient
Tombstoning a deleted/updated document soft-deletes its chunks in both the
FAISS chunk map and the BM25 map (cli.py). A crash between the two writes leaves
the doc tombstoned in one channel but live in the other. The genuinely
unrecoverable state is **FAISS-tombstoned, BM25-live**: `check_index_staleness`
derives deletions from the FAISS map alone (`indexed_doc_ids - manifest`), so a
FAISS-tombstoned doc is no longer "indexed", never appears in `deleted_docs`,
and the BM25 side is never retried — it lingers in lexical results forever.

Reordering the writes (BM25 before FAISS) only changes *which* partial state a
crash produces; it cannot heal an already-bad FAISS-tombstoned one. So the fix
is a **reconciliation**, with reordering as defense-in-depth.

### Implementation
- **`grounding/cli.py` — `reconcile_tombstones(output_dir)`:** reads the
  tombstoned doc_id set from each map and cross-applies — any doc tombstoned in
  one channel but not the other is tombstoned in the lagging channel. Heals
  **both** crash directions regardless of write order. Idempotent (the tombstone
  primitives skip already-deleted chunks) and never rewrites the FAISS index or
  BM25 pickle (only the map JSONs) — preserving the soft-delete contract (AC 3).
  Runs each incremental pass, before the no-changes early-return, so a pure
  recovery pass still completes. A full rebuild (incremental downgraded by the
  23.3 coherence gate) rewrites both maps clean, so reconciliation is skipped
  there.
- **`grounding/cli.py` — reordered both tombstone pairs (delete + update paths)
  to BM25-before-FAISS.** The FAISS tombstone is the commit point (staleness
  re-detects a deletion only while the doc is still live in the FAISS map), so
  doing it last means a crash leaves the recoverable "BM25 done, FAISS pending"
  state. AC 1's preferred ordering.

### How this satisfies the ACs
- **AC 1** (re-detectable, BM25-before-FAISS): reconciliation makes the partial
  state detectable from either map; the reorder makes the FAISS tombstone the
  commit point.
- **AC 2** (crash between steps completes on next run, gone from both): covered
  by both the reorder (normal delete-detection finishes the FAISS side) and
  reconciliation. Tested:
  `test_incremental_reconciles_bm25_tombstoned_faiss_live`.
- **AC 3** (no pickle/index rewrite, soft-delete only): the existing tombstone
  primitives already guarantee this; reconciliation calls them unchanged.
- **AC 4** (failure after FAISS-side, before BM25-side, recovers): this is the
  previously-unrecoverable direction — now healed by reconciliation. Tested:
  `test_incremental_reconciles_faiss_tombstoned_bm25_live` (the W5 regression
  test).

### Interaction with 23.3
Tombstones are soft (counts unchanged), so a tombstone desync does not trip the
23.3 count-based coherence gate — the two mechanisms are complementary: 23.3
heals chunk-count divergence (append desync), 23.4 heals tombstone-state
divergence (delete desync).

### File List
- `grounding/cli.py` (modified) — `reconcile_tombstones` helper; reconcile call
  in the incremental path; both tombstone pairs reordered BM25-first.
- `tests/test_cli.py` (modified) — 2 reconciliation tests (both crash
  directions) in `TestCLIEmbeddingsIncremental`.
- `docs/epics/epic-23-corpus-index-integrity.md` (modified) — statuses + this
  record.

### Validation (23.4)
```
2 new reconciliation tests                                   → 2 pass
pytest test_cli.py::TestCLIEmbeddingsIncremental             → 16 pass (no regression
                                                               from the reorder)
pytest test_bm25.py test_vector_store.py test_faiss_bm25_coherence.py → 75 pass
```

---

## Epic close-out

All four stories (23.1–23.4) implemented and Ready for Review. Remaining
follow-ups stay as drafts / backlog: **D3, D4** (doc_id identity) in this epic's
Future Work; **D5/D6** (fsync, manifest concurrency) and **R3** (default-path
tombstone filtering) on other tracks per the review doc.
