
# PRD — PDF → LLM Artifact Converter (Fast MVP via Existing Libraries)

**Owner:** Andy  
**Agent:** Sarah (Product Owner, 📝)**  
**Version:** 0.2 (Fast MVP using Open Source Integration)**  
**Last Updated:** 2025-10-14  

---

## 1. Objective
Deliver a working Python CLI MVP that converts a folder of PDFs into **LLM-ready Markdown artifacts** (structured, chunked, and metadata-rich) by leveraging existing open-source libraries such as **Unstructured**, **Marker**, and **LangChain text splitters**.  
The goal is to minimize original implementation effort while achieving full PDF ingestion, normalization, and chunking within days.

---

## 2. Strategy Overview

Instead of building a new PDF parser or normalizer, reuse the **strongest existing open-source components**:

| Stage | Library | Purpose |
|-------|----------|----------|
| Parsing / OCR / Layout | **Unstructured** (primary) or **Marker** | Extract text, tables, headings, and preserve document hierarchy. |
| Markdown Formatting | **Marker** | Convert parsed content into structured Markdown while preserving layout and tables. |
| Chunking | **LangChain text splitters** | Produce consistent LLM-sized chunks with overlap control. |
| Manifest + Metadata | Custom lightweight module | Create `_index.json` and `meta.yaml` for traceability and deterministic re-use. |

This drastically reduces engineering time — roughly 70–80% less custom code than writing a pipeline around Docling directly.

---

## 3. Goals and Non-Goals

### 🎯 Goals
1. Ingest PDFs from an input directory and emit Markdown chunks with metadata.  
2. Use Unstructured or Marker for parsing and conversion to Markdown.  
3. Use LangChain’s `RecursiveCharacterTextSplitter` for chunking.  
4. Generate `_index.json` and `meta.yaml` with file-level hashes and provenance.  
5. Achieve CLI usability with minimal parameters.  

### 🚫 Non-Goals
- No custom parser or OCR engine development.  
- No embeddings or retrieval router (future milestone).  
- No web UI or hosted service.  

---

## 4. System Architecture

```
CLI
 └── Controller
     ├── Scanner (discovers PDFs)
     ├── Parser (Unstructured or Marker)
     ├── Normalizer (Markdown formatter)
     ├── Chunker (LangChain splitter)
     ├── Metadata (meta.yaml + chunk front matter)
     ├── Manifest writer (_index.json)
     └── Reporter (progress + summary)
```

Each stage is modular for later replacement (e.g., Docling or hybrid parser).

---

## 5. Libraries & Justification

| Library | Role | Benefit |
|----------|------|----------|
| **Unstructured** | Parsing and layout extraction | Stable, production-grade, handles OCR, multiple document types. |
| **Marker** | Markdown output formatter | Preserves layout and supports tables/images; works with Unstructured outputs. |
| **LangChain text splitters** | Chunking | Industry-standard, supports multiple strategies. |
| **PyYAML + ujson** | Metadata & manifest writing | Lightweight and fast. |
| **Typer** | CLI framework | Clean API for command-line UX. |
| **tqdm** | Progress display | Immediate user feedback. |

---

## 6. MVP Workflow

1. **User command**
   ```bash
   pdf2llm --in ./pdfs --out ./corpus --chunk-size 1200 --chunk-overlap 150
   ```
2. **Parse** each PDF → Markdown via Unstructured or Marker.
3. **Chunk** Markdown using LangChain splitter.
4. **Annotate** each chunk with YAML front matter.
5. **Generate** `meta.yaml` and `_index.json`.
6. **Summarize** progress, skipped, or failed docs.

---

## 7. Directory Layout

```
out/
  _index.json
  <slug>/
    meta.yaml
    doc.md
    chunks/
      ch_0001.md
      ch_0002.md
```

---

## 8. Metadata Contract

### Chunk Front Matter
```yaml
doc_id: "<sha1-8>"
source: "<original_filename.pdf>"
chunk_id: 1
page_start: 3
page_end: 4
hash: "<blake3>"
created_utc: "2025-10-14T18:20:10Z"
```

### `meta.yaml`
```yaml
doc_id: "<sha1-8>"
slug: "<slug>"
orig_name: "<original_filename.pdf>"
strategy: "marker"
tooling:
  unstructured: "<version>"
  marker: "<version>"
  langchain_splitter: "<version>"
params:
  chunk_size: 1200
  chunk_overlap: 150
hashes:
  file_sha1: "<sha1>"
```

---

## 9. CLI Design

```
pdf2llm   --in ./pdfs   --out ./corpus   --chunk-size 1200   --chunk-overlap 150   [--parser unstructured|marker]   [--ocr auto|on|off]   [--dry-run]
```

---

## 10. Acceptance Criteria

| # | Criteria | Verification |
|---|-----------|---------------|
| 1 | CLI runs and converts all PDFs to Markdown | Smoke test |
| 2 | `_index.json` and `meta.yaml` created | File existence |
| 3 | Default chunk size ~1200 chars | Sample validation |
| 4 | Deterministic re-run output | Hash comparison |
| 5 | Failures logged, not fatal | Log review |

---

## 11. Future Enhancements

| Next Step | Description |
|------------|-------------|
| **Retrieval Router** | Add vector embeddings + semantic search layer (FAISS/Chroma). |
| **Hybrid Parser** | Add table‑aware mode or Docling fallback for layout fidelity. |
| **Embeddings Cache** | Persist chunk vectors for reuse. |
| **Web Interface** | Visualize corpus and metadata. |

---

## 12. Development Plan (1-Week Sprint)

| Day | Task |
|-----|------|
| 1 | Set up repo, Typer CLI, Unstructured + Marker parsing test |
| 2 | Implement Markdown normalization and chunking pipeline |
| 3 | Write manifest + metadata writer |
| 4 | Integrate CLI flags and logging |
| 5 | Run end-to-end tests on 5 sample PDFs |
| 6 | Package and publish initial release |
| 7 | Documentation and README.md completion |

---

## 13. Definition of Done

✅ CLI works end-to-end on test folder  
✅ Markdown corpus + manifest generated  
✅ Deterministic outputs verified  
✅ README.md and usage examples written  
✅ Codebase under 400 LOC  

---

**End of PRD v0.2 (Fast MVP using existing open-source components)**  
