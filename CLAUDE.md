# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL REQUIREMENTS

### Python Version
**REQUIRED: Python 3.13.x** - Do NOT use Python 3.14+

Python 3.14 is blocked by `unstructured` (`requires-python: <3.14`). Always verify:
```bash
python3.13 --version  # Must show 3.13.x
./venv/bin/python --version  # Verify venv uses 3.13
```

### Embedding Generation is MANDATORY

**IMPORTANT:** Any script or process that ingests documents into the corpus MUST include embedding generation as a final step. Documents without embeddings are not searchable by agents.

When writing ingestion scripts:
1. Process documents into corpus
2. Identify affected collections
3. Update embeddings for all agents that use those collections

Example pattern:
```bash
# After ingestion, update embeddings for affected agents
for agent in mathematician data-scientist scientist; do
  ./venv/bin/grounding embeddings --agent "$agent" \
    --corpus /path/to/corpus \
    --agents-dir ./agents \
    --out /path/to/embeddings/"$agent" \
    --incremental
done
```

The staging watcher handles this automatically when `AUTO_EMBEDDINGS=true`, but manual/batch ingestion scripts must explicitly include this step.

---

## Project Overview

**Grounding** (PyPI: `grounding-ai`) is a local-first document corpus pipeline for grounded AI agents. It converts documents into LLM-ready Markdown artifacts with structured metadata, deterministic chunking, and corpus-level manifest generation, using open-source libraries (Unstructured, Marker, LangChain).

**Supported formats:** PDF, EPUB, Markdown (.md), Word (.docx, .doc)

### Key Design Principles
- **Determinism**: Same inputs produce byte-identical outputs (excluding timestamps)
- **Error resilience**: Per-file error handling; batch processing continues on failures
- **Local-only**: No network calls; privacy-focused
- **Provenance**: Content hashing and tool version tracking in metadata

## Virtual Environment

This project uses a Python virtual environment at `./venv/`. Always use the venv when running commands:

```bash
# Activate venv (for interactive use)
source venv/bin/activate

# Or use venv python directly
./venv/bin/python -m grounding.cli --help
./venv/bin/pip install <package>
./venv/bin/pytest
```

**Important paths:**
- Python: `./venv/bin/python`
- pip: `./venv/bin/pip`
- grounding CLI: `./venv/bin/grounding`
- pytest: `./venv/bin/pytest`

The staging watcher systemd service has the venv in its PATH.

## Development Commands

### Installation
```bash
# Create venv if needed
python3 -m venv venv
source venv/bin/activate

# Install package in editable mode
pip install -e .
```

### Running the CLI
```bash
# Basic usage
grounding --in ./pdfs --out ./corpus

# With all options
grounding --in ./pdfs --out ./corpus --chunk-size 1200 --chunk-overlap 150 --parser marker --ocr auto --verbose

# Dry run (no file writes)
grounding --in ./pdfs --out ./corpus --dry-run

# Clean output directory before processing
grounding --in ./pdfs --out ./corpus --clean

# With collection tags for agent filtering
grounding --in ./pdfs --out ./corpus --collections science,biology
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_parser.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/test_integration.py

# Run with coverage
pytest --cov=grounding
```

### Code Quality
```bash
# The project uses standard Python tooling (exact commands depend on setup)
python -m grounding.cli --help  # Verify CLI works
```

## CI and Branch Protection

**`main` is protected. You cannot push to it directly, and neither can the owner.**
Every change goes through a PR. Attempting `git push origin main` fails, including
with `--force`.

Protection settings on `andyliszewski/grounding-ai-private`:

| Setting | Value | Consequence |
|---------|-------|-------------|
| `required_status_checks.contexts` | `["test"]` | The `test` job in `ci.yml` must pass |
| `required_status_checks.strict` | `true` | A branch must be up to date with `main` before it can merge; if `main` moves, update the branch |
| `enforce_admins` | `true` | No direct pushes by anyone, owner included |
| `allow_force_pushes` / `allow_deletions` | `false` | `main` cannot be force-pushed or deleted |

### Workflows

| Workflow | Triggers | Runner | Required check? |
|----------|----------|--------|-----------------|
| `ci.yml` (job `test`) | `pull_request`, `workflow_dispatch` | `ubuntu-latest` | **Yes** (`test`) |
| `eval.yml` (job `eval`) | `pull_request` filtered to retrieval paths | `ubuntu-latest` | **No**, deliberately |

**Why `ci.yml` has no `push: branches: [main]` trigger.** It used to, which meant
every merged PR ran `ci` twice on an identical tree: once on the PR, once on the merge
commit. 21 of the 55 jobs run between 2026-04-14 and 2026-07-30 (~38%) were that
duplicate. `strict: true` + `enforce_admins: true` mean the only tree that can reach
`main` is one `test` already passed on, so the push run is redundant rather than merely
wasteful. `workflow_dispatch` is the manual escape hatch for running `ci` against
`main` on demand.

**Why `eval` is NOT a required check.** `eval.yml` is path-filtered. A required check
whose workflow gets skipped by a path filter stays "Pending" forever and the PR can
never merge; GitHub's own guidance is not to use path or branch filtering on workflows
that are required to pass before merging. If you ever add a required check, gate it
with a job-level `if:` (a skipped job reports Success) rather than a workflow-level
`paths:` filter.

### Runners stay GitHub-hosted, on Linux

Do not move these jobs to a self-hosted macOS runner. Both jobs are Linux-only and
cheap (~163 billable min/month across both workflows, measured from the Actions jobs
API over 2026-04-14 to 2026-07-30), and self-hosting would break `sudo apt-get install
-y poppler-utils`, churn the `runner.os`-keyed caches, and stop validating the platform
the repo's `Dockerfile` and `deploy/` actually target.

## Architecture

### High-Level Design
```
CLI (cli.py)
 └── Controller (controller.py)
     ├── Scanner (scanner.py) - Discover PDFs/EPUBs
     ├── Pipeline (pipeline.py)
     │   ├── Parser (parser.py) - Unstructured/Marker parsing
     │   ├── Formatter (formatter.py) - Markdown normalization
     │   └── Hashing (hashing.py) - SHA-1, SHA-256, BLAKE3
     ├── Chunker (chunker.py) - LangChain text splitting
     ├── Metadata (chunk_metadata.py, meta.py) - YAML front matter
     ├── Writer (writer.py) - Atomic file writes
     ├── Manifest (manifest.py) - Corpus _index.json
     └── Stats (stats.py) - Progress tracking and summary
```

### Processing Flow
The system processes each document (PDF or EPUB) through a pipeline:

1. **Scanner** discovers documents (PDF/EPUB) in input directory
2. **Pipeline** (pipeline.py:96-384):
   - Computes file SHA-1 hash
   - Parses PDF via Unstructured or Marker (parser.py)
   - Formats to Markdown (formatter.py)
   - Computes document hashes (BLAKE3, SHA-256)
   - Generates 8-character doc_id from SHA-1
3. **Controller** (controller.py:21-147) adds:
   - Chunks Markdown via LangChain (chunker.py)
   - Adds YAML front matter to each chunk (chunk_metadata.py)
   - Writes output files atomically (writer.py)
   - Generates meta.yaml (meta.py)
   - Updates manifest (manifest.py)

### Key Modules

**pipeline.py**: Core processing orchestration
- `run_pipeline()`: Main entry point for batch PDF processing
- `FileContext`: Tracks per-file state (sha1, doc_id, status, timing)
- `PipelineConfig`: Configuration dataclass
- Error handling: Catches exceptions per file, logs and continues

**controller.py**: Adds chunking and output generation
- `run_controller()`: Wraps pipeline and adds chunking/metadata
- Calls `run_pipeline()` with `generate_outputs=False`
- Handles chunking failures gracefully without aborting batch

**parser.py**: PDF parsing abstraction
- Supports Unstructured and Marker parsers
- OCR modes: auto (detect text yield), on (always), off (never)
- Returns list of elements with text/metadata

**formatter.py**: Markdown normalization
- Converts parsed elements to clean Markdown
- Preserves structure (headings, paragraphs, lists)
- Optional plaintext fallback mode

**chunker.py**: Text splitting
- Uses LangChain's RecursiveCharacterTextSplitter
- Default: 1200 chars with 150 char overlap
- Preserves Markdown structure where possible

**manifest.py**: Corpus-level index
- `ManifestManager.load()`: Read existing _index.json
- `ManifestManager.register_document()`: Add/update entries
- `ManifestManager.write()`: Atomic write with temp file

**agent_filter.py**: Agent-based corpus filtering (Epic 10)
- `AgentConfig`: Dataclass for agent filter configuration (name, description, slugs, collections, exclude_slugs)
- `load_agent_config(agent_name, agents_dir)`: Load agent YAML config
- `filter_manifest(manifest, config)`: Filter manifest entries based on agent config
- Agent YAML format: `name`, `description`, `corpus_filter.collections`, `corpus_filter.slugs`, `corpus_filter.exclude_slugs`

**Agent YAML Schema (Epic 11)**: Extended schema with persona blocks
```yaml
name: <agent-name>              # Required: kebab-case identifier
description: <description>      # Required: Brief agent description

persona:                        # Optional: Persona configuration for slash commands
  icon: "<emoji>"               # Display icon (e.g., "🎯", "⚖️")
  style: |                      # Communication style instructions
    Multi-line description of how the agent communicates.
  expertise:                    # List of expertise areas
    - Area 1
    - Area 2
  greeting: |                   # Activation greeting message
    Message displayed when agent is activated.

corpus_filter:                  # Required: Corpus filtering configuration
  collections:                  # List of collection tags to include
    - collection-1
    - collection-2
  slugs:                        # Optional: Specific document slugs to include
    - specific-doc
  exclude_slugs:                # Optional: Document slugs to exclude
    - excluded-doc
```

**utils.py**: Core utilities
- `slugify()`: Filename → kebab-case slug
- `atomic_write()`: Write via temp file + rename
- `ensure_dir()`: Create directories safely

**omr_parser.py**: Optical Music Recognition (OMR) parsing (v0.2)
- `parse_music_pdf()`: Extract music notation from PDFs using Audiveris
- `detect_music_content()`: Heuristic detection of music notation PDFs (>80% accuracy)
- `MusicElement`: Dataclass representing musical elements (notes, rests, clefs, etc.)
- Integration approach: Subprocess calls to Audiveris CLI → MusicXML → music21 parsing
- Prerequisites: Java Runtime Environment (JRE) >=11, Audiveris >=5.3
- Error handling: `AudiverisOMRError` for JRE/Audiveris missing or processing failures

**music_formatter.py**: Music notation output formatting (v0.2)
- `format_to_musicxml()`: Convert MusicElement list to MusicXML format (primary output)
- `format_to_abc()`: Convert to ABC notation (lightweight, human-readable)
- `format_to_midi()`: Convert to MIDI bytes (playback format)
- `format_to_markdown()`: Generate metadata summary (key, time signature, statistics)
- Pipeline position: OMR Parser → Music Formatter → Writer
- Output formats: MusicXML (semantic-rich), ABC (text-based), MIDI (audio playback)
- Error handling: `FormattingError` for conversion failures

**formula_extractor.py**: Mathematical formula extraction using pix2tex (v0.2)
- `extract_formulas()`: Extract mathematical formulas from PDFs and convert to LaTeX
- `detect_formula_regions()`: Heuristic detection of formula regions in page images (>75% recall)
- `FormulaElement`: Dataclass representing extracted formulas (latex_str, formula_type, page_num, bbox)
- Integration approach: pypdfium2 page rendering → image processing → pix2tex LaTeX OCR
- Prerequisites: pix2tex>=0.1.2, torch>=2.0.0, pypdfium2>=4.0.0, scipy>=1.10.0
- Error handling: `FormulaExtractionError` for pix2tex missing or extraction failures
- Lazy loading pattern: Model initialized on first use (~100-200MB download)
- Formula types: "inline" (in-line with text) or "display" (centered equations)

### Agentic Tool Calling (Epic 12)

The agentic system enables local LLMs to autonomously use tools for corpus search:

**Module Structure:**
- `scripts/agentic.py` - Agent loop and tool calling infrastructure
- `scripts/search_corpus_tool.py` - search_corpus tool implementation
- `scripts/local_rag.py` - CLI with `--agentic` flag support

**Key Components in agentic.py:**
- `AgenticConfig`: Configuration dataclass (max_iterations, verbose, timeout)
- `ToolCall`: Parsed tool call representation (name, arguments)
- `AgentLoopResult`: Result with content, tool_calls_made, iterations, stopped_reason
- `ToolRegistry`: Registers tools with schemas and executors
- `run_agentic_loop()`: Main loop handling tool call cycles
- `query_ollama_with_tools()`: API call with tools parameter
- `parse_tool_calls()`: Extract tool calls from LLM response
- `format_tool_call_trace()`: Verbose output formatting

**Key Components in search_corpus_tool.py:**
- `get_search_corpus_schema()`: OpenAI-format tool schema
- `SearchCorpusTool`: Executes FAISS search, formats results
- `create_search_corpus_tool()`: Factory returning (schema, executor) tuple

**Agentic Loop Flow:**
1. User query sent to LLM with tool schemas
2. LLM responds with optional `tool_calls` array
3. If tool_calls present, execute each and append results to messages
4. Loop until no tool_calls or max_iterations reached
5. Return final response with AgentLoopResult

**API Compatibility:**
Uses Ollama's OpenAI-compatible `/v1/chat/completions` endpoint with `tools` parameter.
Works with LM Studio and any OpenAI-compatible local LLM API.

**CLI Flags (local_rag.py):**
- `--agentic`, `-A`: Enable agentic mode (LLM decides when to search)
- `--max-iterations`: Maximum tool-calling iterations (default: 5)
- `--verbose`, `-v`: Show tool call traces

### Output Structure
```
corpus_root/
├── corpus/                      # Processed output
│   ├── _index.json              # Corpus manifest (all documents)
│   └── <slug>/
│       ├── doc.md               # Full normalized document
│       ├── meta.yaml            # Per-document metadata
│       └── chunks/
│           ├── ch_0001.md       # Chunks with YAML front matter
│           └── ch_0002.md
├── agents/                      # Agent filter definitions
│   ├── ceo.yaml
│   ├── ip.yaml
│   └── marketing-director.yaml
├── embeddings/                  # Vector embeddings (per-agent)
│   ├── ceo/
│   │   └── _embeddings.faiss
│   └── ip/
│       └── _embeddings.faiss
├── staging/                     # Incoming documents (PDF/EPUB) for watcher
│   └── <collection>/
└── originals/                   # Archived source documents
    └── <collection>/
```

## CLI Parameters

- `--in`: Input directory (required)
- `--out`: Output directory (required)
- `--chunk-size`: Characters per chunk (default: 1200)
- `--chunk-overlap`: Overlap between chunks (default: 150)
- `--parser`: `unstructured` or `marker` (default: marker)
- `--ocr`: `auto`, `on`, or `off` (default: auto)
- `--collections`: Comma-separated collection tags (e.g., 'science,biology'). Must be lowercase kebab-case.
- `--dry-run`: Print operations without writing
- `--clean`: Remove output directory before processing
- `--verbose` / `-v`: Enable DEBUG logging

## CLI Subcommands

### Agent Management
```bash
# List all available agents
grounding agents list --agents-dir ./agents

# Show agent details
grounding agents show ceo --agents-dir ./agents

# Show agent with matching documents from corpus
grounding agents show ceo --agents-dir ./agents --corpus ./corpus
```

### Embeddings Generation
```bash
# Generate embeddings for specific agent
grounding embeddings --agent scientist --corpus ./corpus

# With explicit output directory
grounding embeddings --agent scientist --corpus ./corpus --out ./embeddings/scientist

# Check if embeddings are stale
grounding embeddings --agent scientist --corpus ./corpus --check

# Full corpus embeddings (no agent filter)
grounding embeddings --corpus ./corpus --out ./embeddings
```

## Metadata Contracts

### Chunk Front Matter (YAML)
```yaml
doc_id: "<8-char-sha1>"
source: "<original_filename.pdf>"
chunk_id: 1
page_start: 247          # null for Markdown/EPUB/pdftotext fallback (no page info)
page_end: 249            # null in the same cases; equals page_start for single-page chunks
hash: "<blake3-of-chunk>"
created_utc: "<ISO8601>"
section_heading: "3.2 Bootstrap Methods"  # omitted from YAML when null
```

**Section heading boundary convention (Story 17.2):** When a chunk straddles a heading boundary (starts inside §3.2 and ends inside §3.3), the **earlier** section wins — `section_heading` reflects the section the chunk's start is inside. Rationale: a reader arriving at the start of the chunk is still inside that section; that's the context in which the passage should be cited. A chunk whose start falls before any heading in the document (e.g., inside doc-level front matter or opening body text) gets `section_heading: null`.

### Retrieval Output Format (Story 17.3)

Retrieval surfaces (`scripts/search_corpus_tool.py`, `mcp_servers/corpus_search/server.py`) prefix each returned chunk body with a compact citation string on its own leading line. The prefix is produced by `grounding.citations.format_citation_prefix(source, page_start, page_end, section_heading)`.

**Format variants:**

| Inputs | Rendered prefix |
|--------|-----------------|
| slug + page + section, single page | `[alpha-paper, p.247, §3.2 Bootstrap Methods]` |
| slug + page range + section | `[alpha-paper, p.247–249, §3.2 Bootstrap Methods]` |
| slug + page only (section null) | `[alpha-paper, p.247]` |
| slug + section only (pages null) | `[beta-study, §4. Methods]` |
| slug only (all null) | `[gamma-notes]` |

**Semantics:**
- The prefix is the **first line** of each result block, separable by downstream consumers.
- Page ranges use U+2013 EN DASH (`–`), not ASCII hyphen.
- `<slug>` is derived from the chunk's `source` filename via `grounding.utils.slugify` (extension stripped); strings already kebab-case pass through.
- Missing fields (pre-17.2 chunks, pdftotext fallback) degrade silently to the smallest variant that fits; never `[slug, , ]` or naked commas.
- The prefix is additive. Existing result headers, `*doc_id: X, chunk: Y*` annotations, and body text are preserved.

### meta.yaml (Per-Document)
```yaml
doc_id: "<8-char-sha1>"
slug: "<kebab-case-slug>"
orig_name: "<original_filename.pdf>"
collections: ["science", "biology"]  # Optional, from --collections flag
strategy: "marker"
tooling:
  parser: "<parser-version>"
params:
  chunk_size: 1200
  chunk_overlap: 150
  parser: "marker"
  ocr_mode: "auto"
hashes:
  file_sha1: "<sha1-hex>"
  blake3: "<blake3-hex>"
  sha256: "<sha256-hex>"
```

### _index.json (Corpus Manifest)
```json
{
  "created_utc": "<ISO8601>",
  "docs": [
    {
      "doc_id": "7a9b2c1",
      "slug": "report-2024-q3",
      "orig_name": "report_2024_Q3.pdf",
      "chunk_count": 28,
      "strategy": "marker",
      "doc_path": "report-2024-q3/doc.md",
      "meta_path": "report-2024-q3/meta.yaml",
      "collections": ["science", "biology"]
    }
  ]
}
```

## Important Implementation Details

### Determinism Requirements
- Slugification is stable (kebab-case from filename via utils.py:slugify())
- Hashing excludes timestamps
- File writes are atomic (temp file → rename)
- Tool versions stored in meta.yaml for reproducibility

### Error Handling Strategy
- Pipeline catches exceptions per file (pipeline.py:227-249)
- Controller catches post-processing exceptions (controller.py:79-143)
- Errors logged with context but don't abort batch
- Exit code 1 if any files failed (cli.py:178)
- Stats track: total, processed, succeeded, failed, skipped

### Hashing Strategy (hashing.py)
- **File SHA-1**: Computed from PDF bytes for file identity
- **Doc ID**: First 8 chars of SHA-1 (collision detection in pipeline.py:282-299)
- **Document hashes**: BLAKE3 and SHA-256 of Markdown content
- **Chunk hash**: BLAKE3 of chunk text (without front matter by default)

### Testing Approach
- Unit tests: Per-module in tests/test_*.py
- Integration tests: tests/test_integration.py with sample PDFs
- Test utilities: tests/integration_utils.py
- Fixtures expected in test_pdfs/ directory
- Run tests with: `pytest` or `pytest -v`

## BMad-Method Integration

This repository follows BMad-Method framework:
- Epics and stories tracked in docs/epics/ and docs/stories/
- QA documentation in docs/qa/
When working on stories, refer to:
- docs/prd.md for product requirements
- docs/architecture.md for system design (if exists)
- Individual story files in docs/stories/ for acceptance criteria

## Ingestion Workflow

This system uses a staging watcher service that automatically processes documents dropped into the staging folder.

### Staging Watcher Service

**Location:** `scripts/staging-watcher.sh`
**Service:** `~/.config/systemd/user/grounding-watcher.service`

```bash
# Service management
systemctl --user status grounding-watcher
systemctl --user restart grounding-watcher
journalctl --user -u grounding-watcher -f  # Follow logs
```

### Processing Flow

1. **Drop files** into staging folder organized by collection:
   ```
   ~/staging/<collection>/<document.pdf>
   ```

2. **Watcher detects** new files via inotifywait and processes by format:

   | Format | Processing | Result |
   |--------|-----------|--------|
   | PDF (text-based) | `grounding --ocr off` | corpus/ |
   | PDF (scanned) | Detected via pdftotext yield | skipped/ |
   | EPUB | `grounding` | corpus/ |
   | MD, DOCX | `ingest_docs.py` | corpus/ |
   | DOC | Requires antiword | skipped/ |
   | Other | Unsupported | skipped/ |

3. **After processing:**
   - Successful → source moved to `originals/<collection>/`
   - Failed/scanned → source moved to `skipped/<collection>/`
   - Output in `corpus/<slug>/` with doc.md, meta.yaml, chunks/

**Startup ordering and reconciliation (Epic 24, Story 24.6):** the live
`inotifywait -m` monitor starts **before** the one-shot startup scan
(`process_existing`), not after it. On a large corpus the startup scan runs
50+ minutes (OCR backlog + per-agent embedding rebuilds); running it inside the
first loop iteration with the monitor already live means events that fire during
the scan are buffered in the pipe + kernel inotify queue and drained the instant
the scan returns. A one-time `sleep 2` lets recursive watches establish before
the scan snapshots `staging/*/`, so pre-watch arrivals are caught by the scan and
during-scan arrivals by inotify — the union has no hole. Previously the monitor
started only after the scan finished, leaving a ~50-min blind window in which
Syncthing-delivered files were seen by neither mechanism and silently dropped
(no log line, no error) until manually re-triggered. As defense in depth, the
`PENDING_RETRY_INTERVAL` idle tick also runs a **reconciliation rescan**
(`reconcile_staging`) that re-processes any collection with staged-but-uningested
files, guaranteeing eventual pickup (bounded by `PENDING_RETRY_INTERVAL`) even if
an event is ever lost to an inotify queue overflow.

### Environment Variables

Set in systemd service:
```ini
Environment=STAGING_DIR=/path/to/staging
Environment=CORPUS_DIR=/path/to/data/corpus
Environment=ORIGINALS_DIR=/path/to/data/originals
Environment=SKIPPED_DIR=/path/to/data/skipped
Environment=LOG_FILE=/path/to/watcher.log
Environment=AUTO_EMBEDDINGS=true
Environment=AGENTS_DIR=/path/to/grounding/agents
Environment=EMBEDDINGS_DIR=/path/to/data/embeddings
```

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_EMBEDDINGS` | `false` | Enable automatic embedding updates after ingestion |
| `AGENTS_DIR` | (none) | Path to agent YAML definitions for collection matching |
| `EMBEDDINGS_DIR` | (none) | Path to embeddings output directory |
| `LOCK_TIMEOUT` | `3600` | **Deprecated/no-op** since Epic 24.1 (flock lock is kernel-released on death); retained for backward-compat |
| `REPO_DIR` | (derived) | Git repo path; if unset, derives from AGENTS_DIR parent |
| `GIT_PULL_ENABLED` | `true` | Pull latest git changes before processing documents |
| `MAX_EMBED_ATTEMPTS` | `3` | Epic 24.4: failures before a pending agent's embedding update is given up on (loud log + `.failed` marker) |
| `PENDING_RETRY_INTERVAL` | `300` | Epic 24.4/24.6: seconds of inotify inactivity after which pending embedding updates are retried **and** staging is reconciled for any un-ingested files (periodic tick) |
| `MAX_OCR_ATTEMPTS` | `3` | Epic 24.5: OCR "no output" misses before a scanned PDF is quarantined to `skipped/<collection>/quarantine/` |

### Agent-Config Propagation (Cross-Repo)

Agent YAML files (`qms.yaml`, `ceo.yaml`, etc.) live in a **separate git
repository** from this one — on the maintainer's setup,
`github.com/andyliszewski/my-agents`, cloned to `AGENTS_DIR`'s parent. The
watcher treats that repo as the authoritative source of agent
configuration: `pull_latest_repo()` runs `git pull --ff-only` against
`REPO_DIR` (derived from `AGENTS_DIR`) both at `process_existing()`
startup and inside the inotifywait loop **before every ingestion batch**.

Practical consequence: an agent YAML change (e.g., adding a collection
to `corpus_filter.collections`) takes effect on Ubuntu only after the
change is committed and pushed to the agents repo. Editing the agent
YAML on a workstation without pushing leaves Ubuntu using the old
config, and the next ingestion — along with any `AUTO_EMBEDDINGS`
regeneration it triggers — will honor the pre-push state. The
workstation's corpus / embeddings directories are typically
`receiveonly` under Syncthing (Ubuntu authoritative), so the
workstation cannot make this effective by writing locally either.

To propagate an agent YAML change end-to-end: commit + push in the
agents repo, then either drop a new doc in `staging/` to trigger the
watcher (which will pull first), or restart the watcher service.

### Scanned PDF Detection

The watcher uses a quick pdftotext check to detect scanned PDFs:
- Threshold: 1000 chars per MB (`MIN_TEXT_YIELD_PER_MB`)
- Below threshold → treated as scanned, moved to skipped/
- Above threshold → text-extractable, processed with `--ocr off`

**OCR poison-pill quarantine (Epic 24, Story 24.5):** scanned PDFs in
`skipped/<collection>/` are re-OCR'd each cycle until they succeed. A PDF whose
OCR repeatedly "completes but produces no output" would otherwise be re-OCR'd
(minutes each) forever, taxing the serial loop. Each such miss is counted in a
hidden `skipped/<collection>/.ocr-attempts/` dir; after `MAX_OCR_ATTEMPTS` misses
the file is moved to `skipped/<collection>/quarantine/` (a loud `ERROR` log) and
is no longer re-OCR'd — both the counter dir and the quarantine subdir sit
outside the backlog's `*.pdf` glob. Inspect and move a file back out of
`quarantine/` to retry it. Batch-level OCR failures (the OCR tool erroring on the
whole collection) are treated as transient and are **not** counted, so they keep
retrying.

## Embedding Generation

### CLI Usage
```bash
# Generate embeddings during ingestion
grounding ./staging ./corpus --emit-embeddings

# Generate embeddings for entire corpus
grounding embeddings --corpus ./corpus --out ./embeddings

# Generate embeddings for specific agent
grounding embeddings --agent scientist --corpus ./corpus

# Check if embeddings are stale
grounding embeddings --agent scientist --corpus ./corpus --check
```

### Output Files
- `_embeddings.faiss` - FAISS vector index (384-dim L2)
- `_chunk_map.json` - Maps index positions to chunk IDs

### Integration with Watcher (Automatic Embedding Updates)

When `AUTO_EMBEDDINGS=true`, the watcher automatically updates agent embeddings after successful document ingestion:

1. After document processing, watcher identifies affected agents by matching document collection to agent `corpus_filter.collections`
2. For each affected agent, runs `grounding embeddings --agent <name> --incremental`
3. Lock file (`_embeddings.lock`) prevents concurrent updates
4. Embedding failures are logged but don't block document ingestion

**How agent detection works:**
- Watcher parses each agent YAML file in `AGENTS_DIR`
- Matches document collection against agent's `corpus_filter.collections` list
- Only agents with matching collections receive embedding updates

**Lock file behavior (`flock`-based since Epic 24, Story 24.1):**
- Mutual exclusion uses `flock` (util-linux) on a held file descriptor opened
  against `$EMBEDDINGS_DIR/_embeddings.lock`. Acquisition is atomic (no
  check-then-write TOCTOU).
- The lock is held for exactly as long as the holding process keeps that
  descriptor open. The **kernel releases it automatically when the process
  dies** (crash, OOM, `kill -9`, `systemctl restart`), so a killed watcher
  never strands the lock — the next invocation acquires immediately. There is
  no age-based staleness check and no PID parsing.
- A legitimately long embedding run can never have its lock stolen by a second
  invocation, regardless of how long it runs.
- Acquisition is non-blocking (`flock -n`): if the lock is held, the embedding
  update is skipped and logged (not an error). The lock file itself is a
  persistent rendezvous point and is intentionally never removed.
- `LOCK_TIMEOUT` is **deprecated and a no-op** — retained only so existing
  systemd unit env files referencing it don't break.

**Pending-embedding retry queue (Epic 24, Story 24.4):**
- When an embedding update is skipped (lock held) or fails (non-zero exit), the
  affected agent is recorded as a marker file under
  `$EMBEDDINGS_DIR/pending-embeddings/<agent>` so the work is never silently
  dropped. The marker's contents are the failure count.
- Pending updates are retried under the same `flock` on: watcher startup, after
  every ingestion batch (any collection drains all markers), and on a periodic
  `PENDING_RETRY_INTERVAL`-second inotify-idle tick — so a quiet collection's
  docs don't stay unsearchable.
- Lock-held skips record a marker but do **not** count as failures (lock
  contention can't exhaust the retry budget). After `MAX_EMBED_ATTEMPTS` genuine
  failures the agent is given up on: a loud `ERROR` log and a
  `<agent>.failed` marker you can inspect and remove to requeue (after a manual
  rebuild).
- On a successful update the marker is removed.

**Incremental vs full rebuild:**
- `--incremental` appends new embeddings to existing FAISS index
- Detects new documents (in corpus but not in index)
- Tombstones deleted documents (soft-delete, filtered from search)
- No need to regenerate entire index for each new document

Manual embedding generation:
```bash
# Incremental update for specific agent
./venv/bin/grounding embeddings --agent scientist --corpus /path/to/data/corpus --incremental

# Full rebuild for specific agent (without --incremental)
./venv/bin/grounding embeddings --agent scientist --corpus /path/to/data/corpus

# Check if embeddings are stale
./venv/bin/grounding embeddings --agent scientist --corpus /path/to/data/corpus --check
```

### BM25 Sidecar (Epic 19.1)

Every full-build and `--incremental` run of `grounding embeddings` writes a
BM25 lexical index alongside the FAISS artifacts. This is the lexical half
of the hybrid-retrieval pipeline; Story 19.2 fuses it with the dense channel
via RRF, and Story 19.3 wires the hybrid function into the retrieval
surfaces. Story 19.1 is purely additive — the FAISS format and every
existing call path are unchanged.

**Files** (written next to `_embeddings.faiss` / `_chunk_map.json`):

- `_bm25.pkl` — pickled `rank_bm25.BM25Okapi` state (tokenized corpus kept
  under a namespaced attribute so incremental appends can rebuild IDF).
  Empty-corpus sentinel is a pickled `None`.
- `_bm25_map.json` — schema v1: `format_version`, `tokenizer`,
  `rank_bm25_version`, `total`, `tombstone_count`, timestamps, and a
  `chunks[]` list (`bm25_index`, `chunk_id`, `doc_id`, `deleted_utc`).

**Parallel-array contract:** `chunk_bodies[i] ↔ chunk_ids[i] ↔ FAISS
integer id i ↔ bm25_index i`. The CLI preserves insertion order across
both writers so the hybrid merge in 19.2 is a pure dict lookup on
`chunk_id` — no remapping.

**Tokenizer identity:** `whitespace_lowercase_v1` — simple `re.findall(r"\w+",
text.lower())`. If this ever changes (stemming, language-specific tokenizer)
the identity string gets bumped and `load_bm25_index` rejects mismatched
on-disk artifacts with `BM25FormatError` + a rebuild hint rather than
silently returning wrong scores.

**Tombstone semantics:** parity with FAISS — soft-delete via `deleted_utc`
in the map, pickle is never rewritten for a delete, `search_bm25` filters
at query time with a tombstone-aware fetch multiplier.

**Incremental append:** `rank_bm25.BM25Okapi` computes IDF in `__init__`
and exposes no mutating API, so append re-concatenates the old tokenized
corpus with the new tokens and rebuilds. Cost is O(total chunks) per
append; acceptable for the realistic `--incremental` pattern (tens of
chunks at a time).

## Reranking

Two-stage retrieval (FAISS bi-encoder + cross-encoder reranker). Off by default — opt in per query.

### When to enable

Turn rerank on when query/chunk wording drifts from the document's phrasing (e.g., paraphrased questions, domain synonyms, long conversational queries). The cross-encoder scores `(query, chunk)` jointly, which usually pulls genuinely-relevant chunks up that FAISS cosine left mid-pack. Keep it off for exact-term lookups or when latency matters more than precision.

### Latency (CPU)

| Configuration | Median per-query latency |
|---------------|--------------------------|
| No rerank (baseline) | ~28ms |
| bge-reranker-base, pool=50 | ~1.8s |
| bge-reranker-base, pool=100 | ~3.4s |
| bge-reranker-large, pool=50 | ~4s (extrapolated, not re-measured) |

Measured in Story 18.4 on the maintainer's `data-scientist` index (~596k
chunks) against a 5-query sample. Numbers on other hardware vary: x86 Linux
boxes with AVX-512 / optimized BLAS can run the reranker 3–6× faster than
Apple Silicon's default torch CPU path.

> **Measurement host:** Apple Silicon arm64, 16 physical cores, 48 GB RAM,
> Python 3.13.5, `sentence-transformers` / `torch` running on CPU (no MPS
> backend). Chunk bodies ~1.2 KB. First-call model download excluded; warm
> model only. The relative ordering (no-rerank < base < large, and pool=100
> ≈ 2× pool=50) is stable across hardware.

The measured cost of reranking on Apple Silicon is higher than earlier
ballpark estimates (~300ms/query). Users building interactive tools on
similar hardware should size `pool_size` accordingly — pool=50 adds
~1.8s of latency to each query. For sub-second interactive retrieval on
Apple Silicon, keep rerank off or drop to a lighter model like
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Default is off

Epic 18 shipped with `retrieval.rerank.enabled: false` as the default.
Story 18.4 applied the flip decision rule (`recall@5` lift ≥ 0.03 AND
`citation_accuracy` non-decreasing) against the public mini corpus and
found a null result — mini recall is saturated at 1.000 so the reranker
cannot lift it. No private-corpus fixture set was available to measure
realistic query diversity. Reranking is fully functional and opt-in; the
default will flip in a follow-up once a ≥ 10-item private fixture lands.
See `docs/eval/README.md#reranking-comparison-story-184` for the full
comparison and decision.

### CLI flags

Same names and semantics across `grounding eval` and `scripts/local_rag.py`:

| Flag | Type | Default | Behavior |
|------|------|---------|----------|
| `--rerank` | bool | off | Enable cross-encoder reranking. |
| `--rerank-model` | str | `BAAI/bge-reranker-base` | Cross-encoder model name. |
| `--rerank-pool-size` | int | 50 | FAISS candidate count fed to the reranker. |
| `--rerank-top-k` | int | (uses `--top-k`) | Post-rerank truncation. Lets you fetch a large pool and return only the best N. Silently ignored when `--rerank` is off — `--top-k` controls the final output in that case. |

### Configuration resolution order

1. Explicit CLI flag value.
2. `config.yaml` → `retrieval.rerank.*` when present (see `config.example.yaml`).
3. `RerankConfig` default (off, `BAAI/bge-reranker-base`, pool=50, batch=16).

Invoking either CLI without any `--rerank*` flag and without a `config.yaml` keeps behavior bit-for-bit identical to the pre-epic flow.

### MCP tool arguments

The MCP `search_corpus` tool accepts `rerank_enabled`, `rerank_pool_size`, and `rerank_model` as optional arguments (defaults mirror the CLI). See the tool schema in `mcp_servers/corpus_search/server.py`.

## Hybrid Retrieval

BM25 + dense fusion via Reciprocal Rank Fusion (RRF). Off by default — opt in per query. Story 19.4 measured the four-cell `{hybrid off/on} × {rerank off/on}` matrix against the mini corpus (saturated) and found a null result; no ≥ 10-item private-corpus fixture set was available at epic close, so the default stayed off pending a follow-up measurement (TD-004). The plumbing is shipped and proven; only the out-of-box default is unchanged.

### When to enable

Turn hybrid on when the query depends on exact tokens the dense channel
routinely paraphrases away: rare identifiers, function/class names, author
surnames, statute numbers, product SKUs. FAISS + MiniLM is strong on
semantic paraphrase but weak on out-of-vocabulary exact matches; BM25 is
the other way around. RRF (k_rrf=60) fuses both ranked lists and surfaces
candidates that either channel ranked highly. Leave it off for
conversational paraphrased questions where dense alone already wins.

### Interaction with rerank

When both `--hybrid` and `--rerank` are on, the pipeline is:

```
query → search_hybrid(pool=max(hybrid.pool, rerank.pool, top_k))
       → enrich(body + front matter)
       → rerank(query, pool)
       → truncate to top_k
```

The reranker never sees a dense-only pool when hybrid is on: it always
receives the fused pool. When hybrid is on and rerank is off, the fused
pool is truncated to `top_k` after RRF.

### Dense-only fallback

When the BM25 artifacts (`_bm25.pkl`, `_bm25_map.json`) are missing at
`embeddings_dir` — common for agents embedded before Epic 19.1 shipped
— `search_hybrid` logs one WARNING, runs the dense channel only, and
marks each result dict with `hybrid_degraded: True`. Downstream code
must use `result.get("hybrid_degraded", False)` since the key is absent
on the happy path. Rebuild the index with `grounding embeddings --agent
<name> --corpus <path>` to restore lexical coverage.

### Latency (CPU, per-query, warm)

Measured in Story 19.4 on the maintainer's `private-agent` agent index
(16,251 chunks), 5 queries × several iterations, warm model. The mini
corpus (3 chunks) cannot exercise `pool_size=50` so latency is measured
on a realistic private index.

| Cell | hybrid | rerank | Median per-query latency | What dominates |
|------|:------:|:------:|-------------------------:|----------------|
| `{hh}` | off | off | **~7 ms** | FAISS search |
| `{Hh}` | on  | off | **~240 ms** | BM25 tokenize + rank over 16k tokens |
| `{hH}` | off | on  | **~1300 ms** | Cross-encoder scores 50 (q,c) pairs |
| `{HH}` | on  | on  | **~1580 ms** | Rerank + ~240 ms BM25 overhead |

> **Measurement host:** Apple Silicon arm64, 16 physical cores, 48 GB
> RAM, Python 3.13.5, `sentence-transformers` / `torch` running on CPU
> (no MPS backend). x86 Linux hosts with AVX-512 / optimized BLAS
> typically run the rerank cells 3–6× faster.

**Compound-cost callout:** the `{HH}` cell is ~1.58 s/query — roughly
the `{hH}` rerank cost plus the `{Hh}` BM25 overhead. Turning on *both*
flags on Apple Silicon is a real budget item; the rerank cost dominates
but the BM25 overhead is not free. For sub-second interactive retrieval
on Apple Silicon, keep both flags off or drop the reranker to a lighter
model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The BM25 overhead is
linear in corpus size (token-count rebuild on append, per Epic 19.1);
expect ~1.5 s on a ~100 k-chunk index, ~2–3 s on ~500 k.

### CLI flags

Same names and semantics across `grounding eval` and `scripts/local_rag.py`:

| Flag | Type | Default | Behavior |
|------|------|---------|----------|
| `--hybrid` | bool | off | Enable BM25 + dense fusion via RRF. |
| `--hybrid-pool-size` | int | 50 | Candidates fetched from each channel before fusion. |
| `--hybrid-k-rrf` | int | 60 | RRF damping constant (literature standard: 60). |

### Configuration resolution order

1. Explicit CLI flag value.
2. `config.yaml` → `retrieval.hybrid.*` when present (see `config.example.yaml`).
3. `HybridConfig` default (off, `pool_size=50`, `k_rrf=60`).

Invoking either CLI without any `--hybrid*` flag and without a `config.yaml` keeps behavior bit-for-bit identical to pre-19.3.

### MCP tool arguments

The MCP `search_corpus` tool accepts `hybrid_enabled`, `hybrid_pool_size`, and `hybrid_k_rrf` as optional arguments (defaults mirror the CLI). The MCP server reads these from the tool invocation, not from `config.yaml` — same pattern as the rerank arguments.

## Multi-Machine Architecture

This project can optionally run across multiple machines with Syncthing synchronization.
See `docs/multi-machine.md` for the full setup guide.

**Default (single machine):** Everything runs locally. All paths in `config.yaml` point to directories within or near the repo. No sync needed.

**Multi-machine:** A dedicated ingestion server runs the staging watcher and generates embeddings. Workstations sync corpus and embeddings via Syncthing, and upload new documents by dropping files into a shared staging folder. Configure paths in `config.yaml` to point at Syncthing-shared directories.

## Dependencies

### Python Version (CRITICAL)
**REQUIRED: Python 3.13.x** - See [Critical Requirements](#critical-requirements) above.

### Core Libraries
- `unstructured>=0.18.0` - PDF/EPUB parsing with structure preservation
- `ebooklib>=0.20` - EPUB fallback parser (used when unstructured/pypandoc fails)
- `langchain-text-splitters>=0.2.0` - Text chunking
- `typer>=0.9.0` - CLI framework
- `tqdm>=4.66.0` - Progress bars
- `pyyaml>=6.0` - YAML metadata
- `ujson>=5.10` - Fast JSON serialization
- `blake3>=0.4.1` - Content hashing
- `beautifulsoup4>=4.12.0` - HTML parsing for EPUB extraction

### Embeddings (Required for agent search)
- `sentence-transformers>=2.2.0` - Embedding model (all-MiniLM-L6-v2)
- `faiss-cpu>=1.7.0` - Vector index storage
- `torch>=2.0.0` - PyTorch backend
- `numpy>=1.24.0` - Numerical operations

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils  # pdftotext for text extraction
sudo apt-get install inotify-tools  # inotifywait for staging watcher

# macOS
brew install poppler
brew install fswatch  # Alternative to inotifywait
```

### OMR Support (v0.2)
- `music21>=9.1.0` - MusicXML parsing, ABC/MIDI export, music theory analysis
- `pillow>=10.0.0` - Image processing for staff line detection
- `mido>=1.2.0` - MIDI file reading and validation
- Java Runtime Environment (JRE) >=11 - Required for Audiveris (external dependency)
- Audiveris >=5.3 - OMR engine (external dependency, see docs/epics/epic-7-installation-guide.md)

### Formula Extraction Support (v0.2)
- `pix2tex>=0.1.2` - LaTeX OCR for mathematical formula extraction
- `pypdfium2>=4.0.0` - PDF to image conversion (PDFium)
- `scipy>=1.10.0` - Image processing for formula region detection

### Known Compatibility Issues
| Package | Issue | Workaround |
|---------|-------|------------|
| `pypandoc` + `unstructured` | EPUB parsing fails with PosixPath error | Uses ebooklib fallback automatically |
| Python 3.14 | `unstructured` requires `<3.14` | Use Python 3.13.x |
| `marker` | Module import errors in some versions | Falls back to plaintext mode |

## Common Patterns

### Adding New Pipeline Stages
1. Create module in grounding/
2. Add corresponding test in tests/test_*.py
3. Wire into pipeline.py or controller.py
4. Update PipelineConfig if new parameters needed
5. Update CLI arguments in cli.py if exposed to user

### Atomic File Writes
Always use `utils.atomic_write()` for deterministic output:
```python
from grounding.utils import atomic_write
atomic_write(path, content)  # Writes to temp, then renames
```

### Error Handling in Pipeline
```python
try:
    # Processing logic
    context.status = "success"
except SpecificError as exc:
    context.status = "failed"
    context.error = f"stage_name: {exc}"
    stats.record_failure(path.name, context.error)
    logger.warning("Error message", exc_info=True)
    continue  # Don't abort batch
```

### Logging Conventions
- Module-level logger: `logger = logging.getLogger("grounding.module_name")`
- Use structured logging with context: `logger.info("message slug=%s", slug)`
- Log levels: DEBUG for detailed trace, INFO for progress, WARNING for recoverable errors, ERROR for unexpected failures
