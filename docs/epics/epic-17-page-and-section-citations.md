# Epic 17: Page and Section Citations

**Epic ID:** E17
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 0/4
**Dependencies:** Epic 6 (Vector Embeddings), Epic 16 (Evaluation Harness)
**Target Completion:** TBD

---

## Overview

Populate the `page_start`, `page_end`, and `section_heading` fields that already exist in chunk YAML front matter, and surface them in retrieval results so agents can cite "p. 247, §3.2 Bootstrap Methods" instead of referring vaguely to a document.

**Problem Statement:**
- `grounding/chunk_metadata.py` declares `page_start`, `page_end`, and `section_heading` as optional fields on every chunk; they are currently **always `None`**.
- Parsers (Unstructured, Marker) produce page numbers and heading structure in their element metadata, but the pipeline drops this information before the chunker sees it.
- Agents answering from retrieved chunks cannot produce precise citations, which undermines the core "grounded" promise of the project.
- Without page/section metadata, users must scroll through a full document to verify a retrieved snippet.

**Solution:**
- Preserve parser element metadata (page number, heading hierarchy) through the Markdown formatter.
- Teach the chunker to track which parser elements fall inside each chunk so it can compute `page_start`/`page_end` and the nearest preceding heading.
- Wire the populated metadata into the chunk front matter and into the strings that the retriever returns to the LLM.
- Extend the Epic 16 eval harness with a citation-accuracy metric so this change is measured, not assumed.

---

## Goals

1. Every chunk produced from a PDF carries correct `page_start` and `page_end`.
2. Every chunk carries a `section_heading` string whenever the source document has heading structure.
3. Retrieved chunks shown to the LLM include a compact, LLM-friendly citation prefix (e.g., `[alpha-paper, p.247, §3.2]`).
4. The eval harness reports a `citation_accuracy` metric alongside existing retrieval metrics; committed baseline reflects the new numbers.
5. No regression on existing retrieval scores (Epic 16 gate stays green).
6. EPUB and Markdown inputs degrade gracefully: EPUBs get section headings but `None` pages; raw Markdown gets headings only.

---

## Non-Goals

- Full bounding-box / coordinate-level citations (future work if needed).
- Figure/table number citations (handled by separate roadmap items).
- Cross-document citation formatting in LLM output (that's prompt engineering, not pipeline).
- Backfilling page numbers for scanned PDFs where OCR didn't produce reliable page boundaries.

---

## Architecture

```
Parser (unstructured/marker)
    │  emits elements with metadata.page_number + heading-text
    ▼
Formatter (formatter.py)
    │  currently flattens elements → markdown text
    │  CHANGE: emit a parallel "element map" (char_offset → page, heading_stack)
    ▼
Chunker (chunker.py)
    │  LangChain RecursiveCharacterTextSplitter
    │  CHANGE: given the element map, compute per-chunk page_start/page_end and
    │          nearest preceding heading for section_heading
    ▼
Controller (controller.py:533)
    │  build_chunk_metadata(..., page_start=..., page_end=..., section_heading=...)
    ▼
Chunk file (chunks/ch_NNNN.md) — front matter now populated

Retrieval path (Epic 6, search_corpus tool, MCP)
    │  CHANGE: format each retrieved chunk with `[slug, p.PAGE, §SECTION]` prefix
    ▼
LLM sees citation-ready context
```

### Data Flow — Element Map

```python
# New intermediate structure produced by the formatter
@dataclass(frozen=True)
class FormattedElement:
    char_start: int       # byte offset into formatted markdown
    char_end: int
    page_number: Optional[int]
    heading_stack: tuple[str, ...]   # e.g. ("3", "3.2", "3.2.1")
```

The chunker receives `(markdown_text, List[FormattedElement])` and, for each chunk at `[chunk_char_start, chunk_char_end]`, scans the element map to derive:
- `page_start = min(page_number)` over elements overlapping the chunk (ignoring None).
- `page_end = max(page_number)` over same.
- `section_heading = last heading_stack[-1]` whose element starts at or before `chunk_char_start`.

---

## Stories Breakdown

### Story 17.1: Parser & Formatter Element Map

- Capture `page_number` from Unstructured element `metadata.page_number` and from Marker's per-block page info.
- Capture heading text from heading-typed elements (`Title`, `Header`, markdown `#`/`##` blocks).
- Emit `List[FormattedElement]` alongside the formatted markdown.
- Unit tests cover: PDF with multi-page chapter, PDF with no headings, EPUB (no pages, yes headings), raw Markdown (no pages, yes headings).

**AC:**
- `formatter.py` returns a tuple `(markdown_text, element_map)` from its public entry point.
- Element map is ordered by `char_start`; spans are non-overlapping and cover the formatted markdown.
- Heading stack correctly nests `h1 > h2 > h3`; pops back up on new higher-level heading.
- Existing callers that only want text receive the text via a backward-compatible wrapper or updated call sites.

**Status:** Draft

### Story 17.2: Chunker Page/Section Derivation

- Consume the element map in the chunker; attach per-chunk `page_start`, `page_end`, `section_heading`.
- Preserve determinism: same input produces identical metadata.
- Handle edge cases: chunk with no mapped elements (e.g., pure whitespace), chunk spanning a heading boundary, document with one page, document with no headings.

**AC:**
- `build_chunk_metadata` receives populated `page_start`, `page_end`, `section_heading` for PDF input when parser provides page data.
- A chunk spanning pages 3–5 gets `page_start: 3`, `page_end: 5`.
- A chunk beginning inside "§3.2 Bootstrap Methods" and ending inside the same section gets `section_heading: "3.2 Bootstrap Methods"`.
- A chunk straddling the boundary between §3.2 and §3.3 gets the **earlier** section heading; document this convention.
- Markdown-only input (no page info) produces `page_start: null, page_end: null, section_heading: "..."`.
- EPUB input produces `section_heading` when available; pages are null.
- New unit tests in `tests/test_chunker.py` cover all six scenarios above.

**Status:** Draft

### Story 17.3: Retrieval Output Formatting

- Update the text returned by `search_corpus` tool (`scripts/search_corpus_tool.py`) and the MCP corpus-search server to prefix each retrieved chunk with a citation tag.
- Format: `[<slug>, p.<page_start>[–<page_end>], §<section_heading>]` with missing fields omitted gracefully.
- Keep existing fields (chunk text, score) unchanged; add the prefix only.
- Document the format in `CLAUDE.md` so LLM agents know how to parse/echo it.

**AC:**
- A retrieved chunk with `page_start=247, page_end=247, section_heading="3.2 Bootstrap Methods"` prefixes as `[alpha-paper, p.247, §3.2 Bootstrap Methods]`.
- A chunk with `page_start=247, page_end=249` prefixes as `[alpha-paper, p.247–249, §...]`.
- A Markdown-only chunk with no page info prefixes as `[slug, §heading]`.
- A chunk with neither page nor heading prefixes as `[slug]` (no naked bracket or empty fields).
- MCP server response includes the prefix in the returned text blocks.
- Existing callers that parse raw chunk text are either updated or the prefix is separable (e.g., leading line).
- `CLAUDE.md` updated with the citation format.

**Status:** Draft

### Story 17.4: Eval Harness Citation Metric & Baseline Refresh

- Extend the Epic 16 fixture schema with optional `expected.page` and `expected.section` fields.
- Add a `citation_accuracy` metric to `grounding/eval/metrics.py`: fraction of hits where the retrieved chunk's page/section matches the fixture's expectation.
- Update mini corpus fixtures to include page and section expectations.
- Regenerate the mini baseline and commit.
- Document the metric in `docs/eval/README.md`.

**AC:**
- `FixtureItem.expected` gains optional `page` (int or `[int, int]`) and `section` (str) fields; schema docs updated.
- `metrics.py` adds `citation_accuracy(eval_items) -> float`, pure function, unit-tested.
- `EvalAggregate` gains `citation_accuracy`; `EvalRun` JSON bumps to `format_version: 2` with migration note.
- Mini-corpus fixtures include page/section expectations where applicable.
- `docs/eval/baselines/mini-corpus.json` refreshed with the new schema; refresh PR title follows the Epic 16 convention.
- README documents the new metric.
- CI gate still green; deliberate citation regression (e.g., chunker drops pages) demonstrably fails the gate.

**Status:** Draft

---

## Technical Details

### Backward-Compatible Front Matter

Chunks already declare `page_start`, `page_end`, and `section_heading` fields (`grounding/chunk_metadata.py:25-29`). Populating them does not break readers; existing consumers that check for `None` continue to work. Chunks ingested before this epic remain on disk with `None` values until the source document is re-ingested — this is acceptable; re-ingestion is cheap now that Epic 14 landed incremental embeddings.

### Parser Capability Matrix

| Parser | Page numbers | Heading structure | Notes |
|--------|--------------|-------------------|-------|
| Unstructured | Yes (`metadata.page_number`) | Yes (`Title` element type) | Primary PDF path |
| Marker | Yes (per-block `page_id`) | Yes (Markdown heading output) | Alternative PDF path |
| Markdown (raw) | No | Yes (`#`/`##` syntax) | No pages |
| EPUB | No (pages are synthetic) | Yes (chapter/section structure) | Pages intentionally null |
| DOCX (via ingest_docs.py) | Sometimes | Yes | Best-effort; test and document |

### Deterministic Boundary Convention

For a chunk that begins mid-section: use the nearest preceding heading (the one the reader is actually "inside"). This matches how a human would cite the passage. Document this in Story 17.2.

---

## Dependencies

### Epic Dependencies
- **Epic 6** — retrieval path that surfaces chunks to the LLM.
- **Epic 16** — eval harness used by Story 17.4 to measure the lift and guard against regression.

### External Dependencies
- None new. `unstructured` and `marker` already produce the required metadata.

### Code Dependencies
- `grounding/parser.py` — read element metadata.
- `grounding/formatter.py` — emit element map.
- `grounding/chunker.py` — consume element map.
- `grounding/controller.py:533` — pass new fields into `build_chunk_metadata`.
- `grounding/chunk_metadata.py` — already wired for these fields.
- `scripts/search_corpus_tool.py`, MCP server — citation prefix formatting.
- `grounding/eval/metrics.py`, `grounding/eval/fixtures.py` — Story 17.4.

---

## Implementation Order

```
Story 17.1 (Parser & Formatter Element Map)
    └── Foundation; produces the data the chunker needs.

Story 17.2 (Chunker Page/Section Derivation)
    └── Populates chunk front matter; can ship without 17.3/17.4 — chunks have better metadata immediately.

Story 17.3 (Retrieval Output Formatting)
    └── Surfaces metadata to the LLM; depends on 17.2 landing so real data exists to format.

Story 17.4 (Eval Harness Citation Metric & Baseline Refresh)
    └── Measures the lift, locks in the gain, catches regressions going forward.
```

17.1 and 17.2 must ship in sequence. 17.3 can proceed in parallel with 17.4 once 17.2 is merged.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Parser page numbers are off-by-one or unreliable on some PDFs | High | Pick a representative corpus subset, spot-check, document known failure modes; ship with `page_number or None` semantics so bad data surfaces as `null` rather than wrong numbers. |
| Chunker refactor introduces retrieval regression | High | Epic 16 gate; run on full mini corpus before merging 17.2. |
| Marker and Unstructured produce different page numbering (cover page offset) | Medium | Document the difference; choose one as canonical per-document (already selected via `--parser`); eval spot-checks catch mismatches. |
| Element map memory cost on large documents | Low | Element entries are small (~64 bytes); a 1,000-page PDF at ~50 elements/page is ~3 MB — well within budgets. |
| Citation prefix breaks existing retrieval consumers that parse raw chunk text | Medium | Make prefix a separable leading line; document in CLAUDE.md; search for existing consumers before shipping 17.3. |
| Fixture schema v2 breaks tools reading v1 fixtures | Low | `format_version` field was specified in Epic 16; bump and document migration. |

---

## Testing Strategy

### Unit Tests
- Formatter element map correctness across parser types (17.1).
- Chunker page/section derivation across the six edge cases listed in 17.2 AC (17.2).
- Citation prefix formatter with all combinations of present/absent fields (17.3).
- `citation_accuracy` metric on synthetic inputs (17.4).

### Integration Tests
- Full pipeline against a small committed PDF with known page/section structure; assert chunk front matter matches hand-verified expectations.
- Mini corpus gains a chunked document with page/section metadata for the eval harness.

### Manual Validation
- Run on a real book-length PDF from the user's private corpus; spot-check 10 random chunks.
- Query the MCP server; verify retrieved chunks carry citations.

---

## Acceptance Criteria (Epic Level)

1. All PDF chunks in the mini corpus carry correct `page_start` / `page_end` / `section_heading`.
2. EPUB chunks carry `section_heading`; pages remain null.
3. Markdown chunks carry `section_heading`; pages null.
4. Retrieved chunks shown via `search_corpus` and MCP prefix citations in the documented format.
5. `citation_accuracy` metric lives in the eval harness with mini-corpus baseline committed.
6. Epic 16 CI gate remains green; deliberate citation regression fails the gate as a sanity check.
7. `CLAUDE.md` and `docs/eval/README.md` updated; `docs/ROADMAP.md` moves Tier 1 #2 to Shipped.

---

## Definition of Done

- All four stories closed with AC met.
- Mini-corpus baseline refreshed and committed via a reviewed PR (per Epic 16's refresh discipline).
- Real-corpus spot-check recorded in epic notes or a manual-validation log.
- ROADMAP updated; Tier 1 #3 (cross-encoder reranking) explicitly called out as the next candidate — or reprioritized based on what citation_accuracy reveals.

---

## Future Enhancements (Out of Scope)

- Bounding-box / coordinate citations (e.g., "p.247, col.2, lines 15-22").
- Figure/table number citations.
- Footnote and endnote tracking.
- Cross-reference resolution (e.g., "see §4.1" → link).
- Citation-aware reranking (prefer chunks whose citations appear in user query).

---

## References

- `docs/ROADMAP.md` — Tier 1 #2 (this epic).
- `docs/epics/epic-16-evaluation-harness.md` — measurement dependency.
- `grounding/chunk_metadata.py:25-29` — fields already declared, awaiting data.
- `grounding/controller.py:533` — chunk metadata construction site.
- `CLAUDE.md` — "Metadata Contracts / Chunk Front Matter" section will be updated in 17.2 / 17.3.
