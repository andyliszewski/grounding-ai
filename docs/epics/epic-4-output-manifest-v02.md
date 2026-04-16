# Epic 4: Output & Manifest Management (v0.2)

**Epic ID:** E4-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P0
**Estimated Stories:** 3-4
**Dependencies:** Epic 3 (Chunking & Metadata)
**Architecture Version:** 0.2 (Fast MVP)

---

## Overview

Write structured output files (doc.md, chunks/, meta.yaml) and maintain corpus manifest (_index.json). Implement atomic writes and proper file organization.

---

## Goals

1. Write full document Markdown to `doc.md`
2. Write chunked files to `chunks/ch_NNNN.md` with front matter
3. Generate per-document `meta.yaml`
4. Maintain corpus-level `_index.json` manifest
5. Ensure atomic writes prevent corruption
6. Target <80 LOC for this epic

---

## Stories Breakdown

### Story 4.1: Implement Output File Writer
- Create slug-based output directory
- Write `doc.md` with full Markdown
- Create `chunks/` subdirectory
- Write chunk files with zero-padded names
- Use atomic writes for safety

**AC:**
- Directory structure matches spec
- Chunk files: ch_0001.md, ch_0002.md, etc.
- All files written atomically
- UTF-8 encoding throughout

### Story 4.2: Generate Per-Document meta.yaml
- Create metadata structure
- Include: doc_id, slug, orig_name, strategy, tooling, params, hashes
- Detect tool versions automatically
- Serialize to YAML file

**AC:**
- All fields present and correct
- Tool versions detected
- Parameters reflect CLI inputs

### Story 4.3: Implement Manifest Manager
- Load existing `_index.json` if present
- Add document entry
- Include: doc_id, slug, orig_name, pages, strategy, chunk_count
- Write manifest atomically

**AC:**
- Manifest loads existing entries
- New entries appended
- Valid JSON schema
- Atomic write

### Story 4.4: Handle Output Directory Setup
- Implement `--clean` flag (remove output before processing)
- Create output directory if missing
- Validate write permissions
- Initialize empty manifest

**AC:**
- `--clean` safely removes directory
- Directory created with permissions
- Dry-run mode shows operations

---

## Definition of Done

- ✅ All ACs met
- ✅ Output structure matches architecture
- ✅ Atomic writes implemented
- ✅ Code <80 LOC
- ✅ Target: Day 4 of sprint
