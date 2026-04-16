# Epics Overview — PDF → LLM Converter v0.2 (Fast MVP)

**Project:** grounding-ai
**Architecture Version:** 0.2 (Fast MVP using Unstructured + Marker + LangChain)
**Product Owner:** Sarah (📝)
**Created:** 2025-10-14

---

## Epic Summary

This project is divided into **5 streamlined epics** that build the Fast MVP using existing OSS libraries. Total estimated stories: **16-19** (vs. 32 in v0.1).

**Key Difference from v0.1:** We're integrating proven libraries instead of building custom parsing/normalization logic. This reduces code by ~80% and timeline by 50%.

---

## Epic Breakdown

### [Epic 1: Project Setup & CLI Foundation](./epic-1-project-setup-v02.md)
**Status:** Draft | **Priority:** P0 | **Stories:** 3-4 | **LOC:** <100

Establish project structure with v0.2 dependencies (Unstructured, Marker, LangChain).

**Key Deliverables:**
- Python project with v0.2 dependencies
- CLI argument parsing (Typer)
- Minimal utilities (slugify, atomic write)
- Logging infrastructure

**Dependencies:** None (foundational)

---

### [Epic 2: Unstructured/Marker Integration](./epic-2-parser-integration-v02.md)
**Status:** Draft | **Priority:** P0 | **Stories:** 3-4 | **LOC:** <80

Integrate Unstructured for parsing and Marker for Markdown conversion.

**Key Deliverables:**
- PDF scanner
- Unstructured parser integration
- Marker Markdown formatter
- OCR strategy (auto/on/off)
- Error handling

**Dependencies:** Epic 1

**Replaces:** v0.1 Epics 2 & 3 (custom parsing + normalization)

---

### [Epic 3: Chunking & Metadata](./epic-3-chunking-metadata-v02.md)
**Status:** Draft | **Priority:** P0 | **Stories:** 3-4 | **LOC:** <70

Chunk Markdown documents and generate metadata using LangChain splitter.

**Key Deliverables:**
- LangChain text splitter integration
- Document ID generation (SHA-1)
- YAML front matter generation
- Content hashing (BLAKE3/SHA-256)

**Dependencies:** Epic 2

---

### [Epic 4: Output & Manifest Management](./epic-4-output-manifest-v02.md)
**Status:** Draft | **Priority:** P0 | **Stories:** 3-4 | **LOC:** <80

Write structured output files and maintain corpus manifest.

**Key Deliverables:**
- Output file writer (doc.md, chunks/)
- Per-document meta.yaml
- Corpus manifest (_index.json)
- Atomic writes
- Directory management

**Dependencies:** Epic 3

---

### [Epic 5: Integration & Testing](./epic-5-integration-testing-v02.md)
**Status:** Draft | **Priority:** P0 | **Stories:** 4-5 | **LOC:** <70

Wire components together, add progress reporting, test comprehensively.

**Key Deliverables:**
- Complete pipeline integration
- Progress reporting (tqdm)
- Error handling & recovery
- Processing summary
- Comprehensive test suite
- README documentation
- MVP validation

**Dependencies:** All previous epics (1-4)

---

## Development Sequence

```
Epic 1: Foundation (Day 1)
   ↓
Epic 2: Parser Integration (Day 2)
   ↓
Epic 3: Chunking & Metadata (Day 3)
   ↓
Epic 4: Output & Manifest (Day 4)
   ↓
Epic 5: Integration & Testing (Days 5-7)
   ↓
MVP Complete ✅
```

---

## Total Effort Estimate

- **Epic 1:** 3-4 stories → ~1 day
- **Epic 2:** 3-4 stories → ~1 day
- **Epic 3:** 3-4 stories → ~1 day
- **Epic 4:** 3-4 stories → ~1 day
- **Epic 5:** 4-5 stories → ~3 days

**Total:** ~16-19 stories, **~7 days** (1 week sprint)

**Compare to v0.1:** 32 stories, 9-14 days (40% fewer stories, 50% faster)

---

## Target Code Size

| Epic | Target LOC |
|------|-----------|
| Epic 1 | <100 |
| Epic 2 | <80 |
| Epic 3 | <70 |
| Epic 4 | <80 |
| Epic 5 | <70 |
| **Total** | **<400 LOC** |

[Source: docs/prd.md#13-definition-of-done]

---

## MVP Success Criteria

From PRD Section 13:
- ✅ CLI works end-to-end on test folder
- ✅ Markdown corpus + manifest generated
- ✅ Deterministic outputs verified
- ✅ README and usage examples written
- ✅ Codebase under 400 LOC

---

## Key Technical Decisions (v0.2)

1. **Parser:** Unstructured (primary) or Marker
2. **Formatter:** Marker (Markdown conversion)
3. **Chunker:** LangChain RecursiveCharacterTextSplitter
4. **CLI:** Typer
5. **Hashing:** BLAKE3 (preferred) or SHA-256
6. **Strategy:** Simple OSS integration (no custom parsers)

---

## What's Different from v0.1?

### Removed
- ❌ Custom Docling integration (Epic 2)
- ❌ Custom Markdown normalizer (Epic 3)
- ❌ Custom table extraction logic
- ❌ Router strategies (deferred)

### Added
- ✅ Unstructured parser (proven OSS)
- ✅ Marker formatter (proven OSS)
- ✅ Simplified integration approach

### Kept
- ✅ LangChain chunking
- ✅ Metadata & manifest generation
- ✅ Output structure
- ✅ Testing strategy

---

## Next Steps

1. **Review epics** - Examine each epic for completeness
2. **Switch to SM agent** (`/sm`) to create stories from epics
3. **Begin implementation** - Start with Epic 1
4. **Target:** Complete MVP in 1 week

---

## Notes

- Epic structure is dramatically simpler than v0.1
- Focus on integration, not custom implementation
- Each epic builds on previous work sequentially
- Code size target is realistic for OSS integration approach
- Stories will be shorter with clear library usage examples
- Testing is built into Epic 5, not deferred

---

## Post-MVP Epics

### [Epic 6: Vector Embeddings & Semantic Search](./epic-6-vector-embeddings-v02.md)
**Status:** Complete | **Priority:** P1 | **Stories:** 5

Add vector embedding generation and semantic search capabilities.

### [Epic 7: OMR Support](./epic-7-omr-support-v02.md)
**Status:** Complete | **Priority:** P2 | **Stories:** 6

Add Optical Music Recognition for music notation PDFs.

### [Epic 8: Mathematical Formula Extraction](./epic-8-mathematical-formula-extraction-v02.md)
**Status:** Complete | **Priority:** P2 | **Stories:** 5

Extract mathematical formulas from PDFs as LaTeX.

### [Epic 9: Symbolic Math Validation](./epic-9-symbolic-math-validation-v02.md)
**Status:** Draft | **Priority:** P3 | **Stories:** TBD

Validate extracted formulas using symbolic math.

### [Epic 10: Centralized Corpus Architecture](./epic-10-centralized-corpus-v02.md)
**Status:** Draft | **Priority:** P1 | **Stories:** 5

Replace per-agent corpus silos with centralized corpus and agent filtering.

**Key Deliverables:**
- Schema extension for `collections` metadata
- Agent filter module for selective document access
- Migration script for existing corpora
- CLI `--collections` flag
- Agent-filtered embedding generation

**Dependencies:** Epic 4, Epic 6

---

**Status:** ✅ Ready for SM agent to create stories from these v0.2 epics
