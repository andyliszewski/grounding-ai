# Epic 3: Chunking & Metadata (v0.2)

**Epic ID:** E3-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P0
**Estimated Stories:** 3-4
**Dependencies:** Epic 2 (Parser Integration)
**Architecture Version:** 0.2 (Fast MVP)

---

## Overview

Split Markdown documents into LLM-optimized chunks and generate metadata (YAML front matter, doc IDs, hashes). Leverage LangChain's proven chunking library and implement minimal custom metadata logic.

---

## Goals

1. Integrate LangChain `RecursiveCharacterTextSplitter` for chunking
2. Generate unique document IDs (SHA-1 first 8 chars)
3. Create YAML front matter for each chunk
4. Compute content hashes (BLAKE3 or SHA-256)
5. Track page numbers and section context
6. Target <70 LOC for this epic

---

## Stories Breakdown

### Story 3.1: Integrate LangChain Text Splitter
- Install langchain-text-splitters
- Configure `RecursiveCharacterTextSplitter`
- Set chunk_size and chunk_overlap from CLI
- Split Markdown into chunks
- Return list of chunk strings

**AC:**
- Chunks average ~1200 characters (configurable)
- Overlap ~150 characters (configurable)
- Splitter respects Markdown structure

### Story 3.2: Implement Document ID Generation
- Generate doc_id from file SHA-1 (first 8 hex chars)
- Implement file hashing function
- Create slugify integration for filenames
- Store doc_id with file context

**AC:**
- doc_id is 8-character hex string
- Same PDF always produces same doc_id
- Collision detection logs warnings

### Story 3.3: Generate Chunk YAML Front Matter
- Create YAML structure for chunk metadata
- Include: doc_id, source, chunk_id, page_start, page_end, hash, created_utc
- Serialize to YAML string
- Combine front matter + chunk content

**AC:**
- Front matter includes all required fields
- YAML is valid and parseable
- Timestamps in ISO8601 format

### Story 3.4: Implement Content Hashing
- Install BLAKE3 (or use SHA-256 fallback)
- Hash chunk text content
- Hash full document Markdown
- Return hex-encoded hash strings

**AC:**
- Same content produces identical hash
- Hashes exclude YAML front matter
- Deterministic across runs

---

## Definition of Done

- ✅ All ACs met
- ✅ LangChain splitter integrated
- ✅ Metadata contracts match architecture
- ✅ Code <70 LOC
- ✅ Target: Day 3 of sprint
