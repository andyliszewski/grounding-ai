
# Architecture — PDF → LLM Artifact Converter (Fast MVP via Existing Libraries)

**Owner:** Andy
**Agent:** Winston (Architect, 🏗️)**
**Version:** 0.3 (Adds Centralized Corpus Architecture - Epic 10)**
**Last Updated:** 2025-12-30  

---

## 1. Overview

This architecture defines a **fast-path MVP** for converting a folder of PDFs into **LLM-ready Markdown artifacts** using *existing open-source components*.  
The focus is minimal original code — a single Python CLI orchestrating prebuilt parsing, formatting, and chunking modules.

---

## 2. System Composition

### Key Components
| Stage | Implementation | Library |
|--------|----------------|----------|
| Input Scanning | Python stdlib (`pathlib`, `os`) | — |
| Parsing | `Unstructured` or `Marker` | `unstructured`, `marker` |
| Normalization | Markdown export | `marker` |
| Chunking | Recursive text splitter | `langchain-text-splitters` |
| Metadata | YAML + JSON manifest | `pyyaml`, `ujson` |
| CLI Interface | Command-line UX | `typer` |
| Logging | Minimal structured logging | `logging`, `tqdm` |

---

## 3. Architecture Diagram

```
CLI (Typer)
 └── Controller
      ├── Scanner (input_dir → PDF list)
      ├── Parser (Unstructured / Marker)
      ├── Formatter (Markdown)
      ├── Chunker (LangChain Splitter)
      ├── Metadata Manager
      │    ├── meta.yaml
      │    └── _index.json
      ├── Writer (output tree builder)
      └── Reporter (progress, summary, errors)
```

---

## 4. Design Philosophy

- **Reuse First**: Avoid reimplementing parsing or layout logic; leverage mature OSS projects.  
- **Modular Extensibility**: Keep each stage replaceable (e.g., swap Marker with Docling).  
- **Deterministic Outputs**: Stable hashes, naming, and chunk parameters for reproducibility.  
- **Minimal Dependencies**: Limit to battle-tested Python libs only.  
- **No Network Calls**: Entirely offline operation for security and portability.  

---

## 5. Data Contracts

### Directory Structure
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

### Chunk File Example (`chunks/ch_0001.md`)
```yaml
doc_id: "a1b2c3d4"
source: "report_2024.pdf"
chunk_id: 1
page_start: 1
page_end: 2
hash: "e5f6..."
created_utc: "2025-10-14T18:24:33Z"
---
# Executive Summary

This report describes the performance of...
```

### `meta.yaml`
```yaml
doc_id: "a1b2c3d4"
slug: "report-2024"
orig_name: "report_2024.pdf"
tooling:
  parser: "marker"
  langchain_splitter: "0.2.x"
params:
  chunk_size: 1200
  chunk_overlap: 150
hashes:
  file_sha1: "abc123..."
```

### `_index.json`
```json
{
  "created_utc": "2025-10-14T18:24:33Z",
  "docs": [
    {
      "doc_id": "a1b2c3d4",
      "slug": "report-2024",
      "orig_name": "report_2024.pdf",
      "pages": 22,
      "strategy": "marker",
      "chunk_count": 28
    }
  ]
}
```

---

## 6. Processing Pipeline

1. **Scan Input Directory**  
   Collect all `.pdf` files recursively from input path.

2. **Parse & Normalize**  
   Use `unstructured.partition_pdf()` or `marker.parse_pdf()` to extract text and tables → Markdown.

3. **Chunk Markdown**  
   Use LangChain’s `RecursiveCharacterTextSplitter` with size/overlap params (default 1200 / 150).

4. **Generate Metadata**  
   Hashes, timestamps, parameters, and tool versions stored in `meta.yaml`.

5. **Write Output**  
   Save Markdown + chunked files under `out/<slug>/`, plus `_index.json` manifest.

6. **Report Summary**  
   Count successes, skips, failures; print elapsed time.

---

## 7. CLI Interface

### Command
```
pdf2llm   --in ./pdfs   --out ./corpus   --parser marker   --chunk-size 1200   --chunk-overlap 150   [--ocr auto|on|off]   [--dry-run]
```

### Example Run
```
[✔] Parsed: 12 PDFs
[✔] Generated: 368 chunks
[⚠] Skipped: 1 (invalid encoding)
Output written to ./corpus/
```

---

## 8. Error Handling

| Failure | Behavior | Mitigation |
|----------|-----------|-------------|
| PDF parsing failure | Log + skip file | Continue batch |
| Markdown write failure | Retry once, then skip | Atomic write |
| Empty text extraction | Mark as skipped | Record in `_index.json` |
| Invalid chunk size | Use fallback default | Warn user |

---

## 9. Testing & Validation

| Test Type | Scope |
|------------|--------|
| **Unit** | CLI args, hashing, metadata writer |
| **Integration** | End-to-end run with mixed PDFs |
| **Regression** | Compare hashes across reruns |
| **Golden Files** | Verify deterministic chunk outputs |
| **Performance** | Process 100 PDFs within target runtime |

---

## 10. Dependencies

```
unstructured>=0.15.0
marker>=0.2.0
langchain-text-splitters>=0.2.0
typer>=0.9.0
tqdm>=4.66.0
pyyaml>=6.0
ujson>=5.10
blake3>=0.4.1
```

---

## 11. Packaging & Distribution

- Package name: `pdf2llm`
- Structure:
  ```
  pdf2llm/
    __init__.py
    cli.py
    pipeline.py
    writer.py
    manifest.py
    utils.py
  ```
- Entry point: `pdf2llm = pdf2llm.cli:app`
- Install via: `pip install -e .`
- License: MIT-compatible (verify Unstructured and Marker license compatibility)

---

## 12. Extensibility Roadmap

| Milestone | Addition | Status |
|-----------|----------|--------|
| v0.2.1 | Vector embeddings export (`--emit-embeddings`) | ✅ Complete (§13) |
| v0.2 | OMR Support - Optical Music Recognition | ✅ Complete (§14, Epic 7) |
| v0.2 | Formula Extraction - LaTeX OCR | ✅ Complete (§15, Epic 8) |
| v0.3 | Centralized Corpus Architecture | ✅ Documented (§17, Epic 10) |
| v0.4 | Add Router for table-heavy documents | Planned |
| v0.5 | Hybrid search (semantic + keyword) | Planned |
| v0.6 | Web dashboard for corpus inspection | Planned |

---

## 13. Vector Embeddings & Semantic Search (v0.2.1)

### Overview

Optional semantic search capability using sentence-transformers embeddings and FAISS vector store.

### Architecture

```
Enhanced Pipeline:
Parse → Format → Chunk → Metadata → [Embed] → Write → [Vector Store]
                                      ^^^^^^      ^^^^^^
                                      NEW STEPS

Query Flow:
User Query → Generate Embedding → FAISS Search → Load Chunks → Return Results
```

### Components

**embedder.py**: Embedding generation using sentence-transformers
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Performance: ~1000 chunks/second on CPU
- Deterministic outputs

**vector_store.py**: FAISS index management
- Index type: Flat L2 (exact search)
- Persistence: _embeddings.faiss, _chunk_map.json
- Scalability: 100k+ chunks

**query.py**: Semantic search interface
- CLI: python -m pdf2llm.query
- API: query_corpus(corpus, query, top_k)
- Output: ranked chunks with scores

### Dependencies

```python
sentence-transformers>=2.2.0  # Embedding model
faiss-cpu>=1.7.0              # Vector similarity search
torch>=2.0.0                  # Required by sentence-transformers
```

### Usage

```bash
# Ingest with embeddings
pdf2llm --in ./pdfs --out ./corpus --emit-embeddings

# Query corpus
python -m pdf2llm.query --corpus ./corpus --query "..." --top-k 5
```

### Performance Characteristics

- Embedding generation: >100 chunks/sec
- Query latency: <100ms for top-10
- Memory overhead: ~1.5KB per chunk
- Disk overhead: ~1.5KB per chunk

### Future Enhancements

- GPU acceleration (faiss-gpu)
- Alternative vector stores (Chroma, Pinecone)
- Approximate nearest neighbor (ANN) indices
- Incremental index updates
- Hybrid search (semantic + keyword)

---

## 14. Optical Music Recognition (OMR) Support (v0.2 - Epic 7)

### Overview

pdf2llm extends beyond text to support music notation extraction from PDF sheet music using Audiveris OMR engine.

### Architecture

```
OMR Pipeline:
CLI (--parser omr)
 └── Controller
      ├── OMR Parser (Audiveris via subprocess)
      ├── Music Formatter (MusicXML/ABC/MIDI)
      ├── Writer (music files + chunks)
      └── Manifest (with music metadata)

Hybrid Pipeline:
CLI (--parser hybrid)
 └── Controller
      ├── Hybrid Processor
      │    ├── Region Detector (text vs music pages)
      │    ├── Text Parser (Unstructured/Marker)
      │    └── Music Parser (OMR)
      ├── Combined Chunks (preserving document order)
      └── Manifest (hybrid content_type)
```

### Components

**omr_parser.py**: Music notation extraction via Audiveris
- Integration: Subprocess calls to Audiveris CLI
- Input: PDF with music notation
- Output: List of MusicElement objects (notes, rests, clefs, etc.)
- Prerequisites: Java >=11, Audiveris >=5.3
- Error handling: Graceful failure with detailed error messages

**music_formatter.py**: Multi-format music output
- MusicXML: Industry standard, semantically rich (primary format)
- ABC: Text-based notation, human-readable
- MIDI: Audio playback format
- Markdown: Metadata summary (key, time signature, statistics)

**hybrid_processor.py**: Combined text + music processing
- Auto-detection: Heuristic-based identification of music pages
- Region routing: Text pages → text parser, music pages → OMR parser
- Phrase-based chunking: Chunks aligned with musical phrases (4 measures)
- Sequential output: Preserves original document page order

### Data Contracts Extensions

#### Music Manifest Entry
```json
{
  "doc_id": "def456",
  "slug": "beethoven-sonata",
  "orig_name": "beethoven_op13.pdf",
  "content_type": "music",
  "chunk_count": 12,
  "strategy": "omr",
  "music_format": "all",
  "music_metadata": {
    "key": "C minor",
    "time_signature": "4/4",
    "phrase_count": 12,
    "measure_count": 48
  },
  "music_files": [
    "beethoven-sonata/music.musicxml",
    "beethoven-sonata/music.abc",
    "beethoven-sonata/music.mid"
  ]
}
```

#### Music Output Structure
```
out/
  beethoven-sonata/
    doc.md                   # Markdown metadata summary
    music.musicxml           # MusicXML notation
    music.abc                # ABC notation
    music.mid                # MIDI audio file
    meta.yaml                # Document metadata
    chunks/
      ch_0001.md            # Phrase-based chunks
      ch_0002.md
```

### Processing Flow

**OMR Mode (--parser omr)**:
1. CLI validates Audiveris/JRE prerequisites
2. Controller routes to `_process_music_pdf()`
3. OMR Parser: PDF → Audiveris CLI → MusicXML → MusicElement list
4. Music Formatter: Generate requested formats (musicxml/abc/midi/all)
5. Writer: Save music files + metadata summary
6. Manifest: Register with music_metadata fields

**Hybrid Mode (--parser hybrid)**:
1. Controller routes to hybrid_processor
2. Hybrid Processor detects text vs music pages (heuristic analysis)
3. Text pages → Text parser (Unstructured/Marker)
4. Music pages → OMR parser (Audiveris)
5. Combined chunks: Preserves document sequence
6. Manifest: Registers as content_type="hybrid"

### Dependencies

```python
# Required for OMR support
music21>=9.1.0              # MusicXML parsing, ABC/MIDI export
pillow>=10.0.0              # Image processing for detection

# External dependencies (system)
# Java Runtime Environment (JRE) >=11
# Audiveris >=5.3
```

### CLI Extensions

```bash
# OMR-only processing
pdf2llm ./sheet-music ./corpus --parser omr --music-format musicxml

# Hybrid processing (auto-detect text + music)
pdf2llm ./music-textbooks ./corpus --parser hybrid --music-format all

# Options
--parser omr|hybrid         # Enable OMR processing
--music-format musicxml|abc|midi|all  # Output format selection
```

### Performance Characteristics

- **Processing speed**: 5-10 pages/minute (slower than text)
- **Accuracy**: 85-95% for printed scores, 50-70% for handwritten
- **Overhead**: Audiveris subprocess adds ~2-3s startup per file
- **Memory**: Similar to text processing (~4GB recommended)

### Error Handling

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| Audiveris not found | Early validation failure with install instructions | Check prerequisites at CLI startup |
| JRE missing | Early validation failure | Check java -version at CLI startup |
| OMR parsing failure | Log error, skip file, continue batch | Graceful degradation |
| Poor accuracy | Log warning, output partial results | Best-effort processing |

### Limitations

- Requires external dependencies (Java, Audiveris)
- Slower than text processing
- Accuracy varies with PDF quality
- Limited to Western music notation

### Use Cases

1. **Music Education**: Build searchable corpus of pedagogical materials
2. **Performance Library**: Create digital sheet music repository
3. **Musicological Research**: Analyze corpus for compositional patterns

### Testing

- Unit tests: OMR parser, music formatter, hybrid processor
- Integration tests: End-to-end OMR and hybrid processing
- Test PDFs: Simple melody, piano score, complex score, handwritten, hybrid document

---

## 15. Mathematical Formula Extraction Support (v0.2 - Epic 8)

### Overview

pdf2llm can extract mathematical formulas from scientific papers and textbooks using pix2tex (LaTeX OCR).

### Architecture

```
Formula Extraction Pipeline:
CLI (--extract-formulas)
 └── Controller
      ├── Text Parser (Unstructured/Marker)
      ├── Formula Extractor (pix2tex)
      ├── Formula Formatter (LaTeX/MathML)
      ├── Hybrid Processor (merge text + formulas)
      ├── Writer (formula files + chunks)
      └── Manifest (with formula metadata)

Hybrid Pipeline:
PDF → Parse Text → Extract Formulas
                ↓
        Merge by Position
                ↓
      Combined Markdown (text with $E=mc^2$ embedded)
```

### Components

**formula_extractor.py**: Mathematical formula extraction using pix2tex
- Integration: pix2tex LaTeX OCR model
- Input: PDF with mathematical formulas
- Output: List of FormulaElement objects (latex_str, formula_type, bbox)
- Prerequisites: pix2tex>=0.1.2, torch>=2.0.0, pypdfium2>=4.0.0, scipy>=1.10.0
- Error handling: Lazy loading pattern, graceful failure on missing dependencies

**formula_formatter.py**: Multi-format formula output
- LaTeX: Primary output format ($...$, $$...$$)
- MathML: Semantic XML representation (accessibility)
- Plaintext: ASCII fallback for accessibility
- Validation: LaTeX syntax and MathML structure validation

**hybrid_processor.py**: Text + formula merging
- Position-based merging: Formulas embedded at correct document positions
- Formula detection: Heuristic-based region detection (>75% recall)
- Inline/display classification: Auto-detection of formula type
- Markdown integration: Formulas embedded as LaTeX in markdown

### Data Contracts Extensions

#### Formula Manifest Entry
```json
{
  "doc_id": "def456",
  "slug": "scientific-paper-2024",
  "orig_name": "paper.pdf",
  "content_type": "scientific",
  "chunk_count": 42,
  "strategy": "marker",
  "formula_metadata": {
    "formula_count": 18,
    "inline_count": 12,
    "display_count": 6,
    "complexity": "moderate"
  },
  "formula_files": [
    "scientific-paper-2024/formulas/formula_0001_0001.tex",
    "scientific-paper-2024/formulas/formula_0001_0001.mathml"
  ]
}
```

#### Formula Output Structure
```
out/
  scientific-paper-2024/
    doc.md                          # Text with embedded LaTeX
    meta.yaml                       # Document metadata
    formulas/
      formula_0001_0001.tex         # Individual formula LaTeX
      formula_0001_0001.mathml      # Individual formula MathML
      formula_0002_0001.tex
      ...
    chunks/
      ch_0001.md                    # Chunks with formula metadata
      ch_0002.md
```

### Processing Flow

**Formula Extraction Mode (--extract-formulas)**:
1. CLI validates pix2tex/PyTorch prerequisites
2. Controller routes through standard text pipeline
3. Formula Extractor: PDF → pix2tex OCR → FormulaElement list
4. Formula Formatter: Generate requested formats (latex/mathml/both)
5. Hybrid Processor: Merge formulas into markdown by position
6. Writer: Save formula files + chunks with embedded formulas
7. Manifest: Register with formula_metadata fields

### Dependencies

```python
# Required for formula extraction
pix2tex>=0.1.2                 # LaTeX OCR for formula extraction
torch>=2.0.0                   # PyTorch backend for pix2tex
pypdfium2>=4.0.0               # PDF to image conversion
scipy>=1.10.0                  # Image processing

# Optional for MathML output
latex2mathml>=3.0.0            # LaTeX to MathML conversion
```

### CLI Extensions

```bash
# Extract formulas with LaTeX output
pdf2llm ./papers ./corpus --extract-formulas

# Extract formulas with both LaTeX and MathML
pdf2llm ./textbooks ./corpus --extract-formulas --formula-format both

# Combined with embeddings
pdf2llm ./research ./corpus --extract-formulas --emit-embeddings

# Options
--extract-formulas              # Enable formula extraction
--formula-format latex|mathml|both  # Output format selection (default: latex)
```

### Performance Characteristics

- **Processing speed**: >30 formulas/minute on CPU
- **First run**: ~100-200MB model download (one-time)
- **Accuracy**: >75% recall on printed formulas, lower for handwritten
- **Overhead**: ~10-15% slower than text-only processing
- **Memory**: Additional ~500MB for pix2tex model

### Error Handling

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| pix2tex not installed | Early validation failure with install instructions | Check prerequisites at CLI startup |
| Model download failure | Clear error message with retry instructions | Require internet connection for first run |
| Formula extraction failure | Log warning, continue with text-only | Graceful degradation |
| LaTeX formatting error | Skip invalid formula, log warning | Best-effort processing |

### Limitations

- Requires internet connection for first-run model download
- Works best with printed (not handwritten) formulas
- Heuristic-based detection may miss some formulas
- Model size (~100-200MB) increases package footprint
- Processing overhead for formula extraction

### Use Cases

1. **Research Literature**: Build searchable corpus of scientific papers
2. **Textbook Processing**: Extract formulas from educational materials
3. **Scientific Documentation**: Process technical documentation with equations

### Testing

- Unit tests: Formula extractor, formula formatter, hybrid processor
- Integration tests: End-to-end formula extraction with various PDF types
- Test PDFs: Simple equations, complex equations, research paper, textbook page, edge cases

---

## 17. Centralized Corpus Architecture (v0.3 - Epic 10)

### Overview

Replace per-agent corpus silos with a single centralized corpus that agents selectively filter. Synchronization between machines handled via Syncthing.

### Problem Statement

Current per-agent corpus structure creates:
- Duplicate ingestion if same book used by multiple agents
- No shared resources across agents
- Multiple sync targets needed
- "Where is X?" requires searching multiple locations

### Target Architecture

```
corpus_root/                        # Single Syncthing sync target
├── corpus/                         # Processed output
│   ├── _index.json                # Master manifest (all docs)
│   ├── ap-biology-for-dummies/
│   │   ├── doc.md
│   │   ├── meta.yaml              # Includes 'collections' field
│   │   └── chunks/
│   ├── feynman-lectures-vol-1/
│   └── ...
├── originals/                      # Source PDFs (named by slug)
│   ├── ap-biology-for-dummies.pdf
│   └── ...
└── agents/                         # Agent definitions
    ├── scientist.yaml
    ├── ceo.yaml
    └── ...

staging/                            # LOCAL ONLY - not synced
└── *.pdf                          # PDFs awaiting ingestion
```

> **Note:** The `staging/` directory must be **outside** `corpus_root/` to prevent accidental sync.

### Machine Roles

| Machine | Role | Responsibilities |
|---------|------|------------------|
| Ingestion Master | Primary | Ingest PDFs, manage corpus, push via Syncthing |
| Consumer(s) | Secondary | Query corpus via agents, receive sync |

### Agent Definition Schema

Agents define which corpus documents they can access:

```yaml
# agents/scientist.yaml
name: scientist
description: "Science and research specialist"

corpus_filter:
  # Option 1: Explicit slug list
  slugs:
    - ap-biology-for-dummies
    - feynman-lectures-vol-1
    - biochemistry-for-dummies

  # Option 2: Collection-based (for larger corpora)
  # collections:
  #   - science
  #   - research-methods
```

### Corpus Metadata Extension

Add `collections` field to `meta.yaml` for tag-based filtering:

```yaml
# corpus/ap-biology-for-dummies/meta.yaml
doc_id: "b7f2f069"
slug: "ap-biology-for-dummies"
orig_name: "AP Biology for DUMmIES.pdf"
strategy: "unstructured"
collections:                        # NEW FIELD
  - science
  - biology
  - reference
source_agent: "scientist"           # Migration provenance tracking
# ... existing fields ...
```

### Manifest Structure Extension

The `_index.json` aggregates all documents with collection metadata:

```json
{
  "created_utc": "2025-12-23T00:00:00+00:00",
  "updated_utc": "2025-12-23T00:00:00+00:00",
  "docs": [
    {
      "doc_id": "b7f2f069",
      "slug": "ap-biology-for-dummies",
      "orig_name": "AP Biology for DUMmIES.pdf",
      "chunk_count": 964,
      "strategy": "unstructured",
      "collections": ["science", "biology"],
      "doc_path": "ap-biology-for-dummies/doc.md",
      "meta_path": "ap-biology-for-dummies/meta.yaml"
    }
  ]
}
```

### Embeddings Strategy

**Do not sync embeddings.** Generate per-agent on consuming machines.

```
consuming_machine/
└── embeddings/                     # Generated locally, not synced
    ├── scientist/
    │   ├── index.faiss
    │   └── chunk_map.json
    └── ceo/
        └── ...
```

Rationale:
- Embeddings are large (~10MB per 1k chunks)
- Machine-specific (different torch versions)
- Agent filtering means each agent only embeds its subset
- Regeneration is fast compared to initial PDF ingestion

### Syncthing Configuration

| Folder | Sync Direction | Notes |
|--------|----------------|-------|
| `corpus_root/` | Master → Consumers | Send-only from master |
| `staging/` | Not synced | Local to ingestion machine |
| `embeddings/` | Not synced | Generated locally per-agent |

### Ingestion Workflow

```
1. Drop PDF in staging/
            ↓
2. Run: pdf2llm --in staging/ --out corpus_root/corpus/ --collections science,biology
            ↓
3. Move original: staging/*.pdf → corpus_root/originals/<slug>.pdf
            ↓
4. Update agent: Add slug to relevant agent definitions
            ↓
5. Verify: Check formulas/images look correct
            ↓
6. Syncthing auto-syncs corpus_root/ to consuming machines
            ↓
7. On consuming machine: Regenerate embeddings for affected agents
```

### CLI Extensions

```bash
# Add collections during ingestion
pdf2llm --in staging/ --out corpus/ --collections science,biology

# Agent-filtered embedding generation
pdf2llm embeddings --agent scientist --corpus corpus/ --out embeddings/scientist/

# Check embedding freshness
pdf2llm embeddings --agent scientist --check
```

### Components

**agent_filter.py** (New Module):
- `load_agent_config(agent_name) -> AgentConfig`
- `filter_manifest(manifest, agent_config) -> filtered_manifest`
- Integration with `vector_store.py` for filtered embedding generation

**Files to Modify**:
| File | Change |
|------|--------|
| `pdf2llm/manifest.py` | Add `collections` to `ManifestEntry` |
| `pdf2llm/meta.py` | Add `collections` param to `build_meta_yaml()` |
| `pdf2llm/cli.py` | Add `--collections` flag |
| `pdf2llm/agent_filter.py` | **New file** - agent config and filtering |
| `pdf2llm/vector_store.py` | Integrate agent filtering |

### Implementation Phases

1. **Phase 0: Schema Extension** - Add `collections` field to manifest and meta
2. **Phase 1: Migration Script** - Merge existing agent corpora
3. **Phase 2: Agent Filter Module** - New filtering capabilities
4. **Phase 3: Syncthing Setup** - Configure multi-machine sync

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Slug collision handling | Append `<slug>-<doc_id[:8]>/` | Preserves readability + guarantees uniqueness |
| Collection naming | Lowercase kebab-case | Prevents `Science` vs `science` collisions |
| Embedding regeneration | Manual with `--check` helper | Simple first, can automate later |
| Agent definitions | Inside `corpus_root/agents/` | Single sync target reduces complexity |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Migration corrupts data | High | Backup before migration; dry-run mode |
| Slug collision during merge | Medium | Pre-scan for collisions before merge |
| Syncthing conflicts | Low | Master is send-only; no write conflicts |
| Embeddings out of sync | Low | `--check` command detects staleness |

---

## 18. Definition of Done

✅ CLI converts all PDFs to Markdown and chunks  
✅ `_index.json` and metadata files generated  
✅ Deterministic re-runs verified  
✅ No unhandled exceptions on invalid PDFs  
✅ End-to-end tested with ≥5 diverse documents  

---

**End of Architecture v0.3 — Includes Centralized Corpus Architecture (Epic 10)**
