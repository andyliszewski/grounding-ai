# Epic 1: Project Setup & CLI Foundation (v0.2)

**Epic ID:** E1-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P0
**Estimated Stories:** 3-4
**Architecture Version:** 0.2 (Fast MVP - OSS Integration)

---

## Overview

Establish the foundational project structure and CLI interface for the Fast MVP using existing open-source libraries (Unstructured, Marker, LangChain). This epic creates a minimal scaffolding focused on integration rather than custom implementation.

---

## Goals

1. Initialize Python project with v0.2 dependencies (Unstructured, Marker, LangChain)
2. Configure CLI argument parsing with essential flags
3. Implement minimal cross-platform utilities (slugify, atomic write)
4. Set up basic logging and progress reporting with tqdm
5. Target <100 LOC for this epic (foundation only)

---

## User Stories

**From PRD Section 1:** As a researcher, I need to install and run the tool easily so I can start converting my PDFs immediately.

**From PRD Section 12:** As a developer, I want to complete the MVP in 1 week using proven libraries.

---

## Epic Acceptance Criteria

1. ✅ Project installs with `pip install -e .`
2. ✅ CLI accepts required flags: `--in`, `--out`, `--chunk-size`, `--chunk-overlap`
3. ✅ CLI accepts optional flags: `--parser`, `--ocr`, `--dry-run`
4. ✅ Dependencies installed: unstructured, marker, langchain-text-splitters, typer, tqdm, pyyaml
5. ✅ Basic utilities available: slugify(), atomic_write(), ensure_dir()
6. ✅ Logging configured with structured output
7. ✅ README with installation and basic usage

---

## Technical Scope

### Components to Build

- **Project Structure (Minimal)**
  - `pyproject.toml` with v0.2 dependencies
  - `pdf2llm/` package directory
  - `pdf2llm/__init__.py` with version
  - `pdf2llm/cli.py` for CLI
  - `pdf2llm/utils.py` for helpers
  - `tests/` directory structure

- **CLI Module (`cli.py`)**
  - Argument parsing using Typer
  - Validation of input/output paths
  - Flag handling for all parameters
  - Version display
  - Configuration object creation

- **Utility Module (`utils.py`)**
  - `slugify(filename)` → kebab-case conversion
  - `atomic_write(path, content)` → safe file writes
  - `ensure_dir(path)` → directory creation
  - Path normalization helpers

- **Logging Setup**
  - Configure logging module
  - Set up tqdm for progress bars
  - Structured log format (timestamp, level, message)

---

## Dependencies (v0.2 Architecture)

**Python Libraries:**
```toml
[project.dependencies]
python = ">=3.10"
unstructured = ">=0.15.0"
marker = ">=0.2.0"
langchain-text-splitters = ">=0.2.0"
typer = ">=0.9.0"
tqdm = ">=4.66.0"
pyyaml = ">=6.0"
ujson = ">=5.10"
blake3 = ">=0.4.1"
```

**Key Changes from v0.1:**
- ❌ Removed: docling (custom parser)
- ✅ Added: unstructured (OSS parser)
- ✅ Added: marker (Markdown formatter)
- ✅ Kept: langchain-text-splitters, typer, tqdm, pyyaml

---

## Stories Breakdown

### Story 1.1: Initialize Project with v0.2 Dependencies
- Create project structure (`pdf2llm/` package)
- Set up `pyproject.toml` with v0.2 dependencies
- Configure entry point: `pdf2llm = pdf2llm.cli:main`
- Create initial README
- Initialize git repository

**AC:**
- Project installs with `pip install -e .`
- All v0.2 dependencies install correctly
- `pdf2llm --version` works

---

### Story 1.2: Implement CLI Argument Parsing
- Define all CLI arguments with Typer
- Required: `--in`, `--out`
- Optional: `--chunk-size` (default 1200), `--chunk-overlap` (default 150)
- Optional: `--parser` (unstructured|marker), `--ocr` (auto|on|off)
- Optional: `--dry-run`, `--clean`
- Validation: paths exist, sizes > 0
- Help text with examples

**AC:**
- All flags parse correctly
- Validation shows helpful errors
- `--help` displays complete usage
- Defaults work as specified

---

### Story 1.3: Build Minimal Utilities
- Implement `slugify()` for filename normalization
- Implement `atomic_write()` for safe file operations
- Implement `ensure_dir()` for directory creation
- Write unit tests for all utilities

**AC:**
- `slugify("Report_2024.pdf")` → `"report-2024"`
- Atomic writes prevent corruption
- All utilities have passing tests

---

### Story 1.4: Set Up Logging and Progress Reporting
- Configure Python logging module
- Integrate tqdm for progress bars
- Implement statistics tracker (files, chunks, errors)
- Format: `[timestamp] [level] message`
- Summary output at completion

**AC:**
- Logs include timestamp, level, message
- Progress bars show current file
- Summary displays: files processed, succeeded, failed, chunks
- Logs are human-readable

---

## CLI Contract (v0.2)

[Source: docs/architecture.md#7-cli-interface]
```bash
pdf2llm --in ./pdfs --out ./corpus --parser marker --chunk-size 1200 --chunk-overlap 150 [--ocr auto] [--dry-run]
```

**New in v0.2:**
- `--parser` flag to choose unstructured or marker
- Simplified flags (removed --strategy, will be added in later version)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dependency conflicts | High | Pin versions in pyproject.toml |
| Unstructured installation issues | Medium | Document installation, provide troubleshooting |
| Cross-platform path issues | Medium | Use pathlib throughout, test on Windows |

---

## Definition of Done

- ✅ All story acceptance criteria met
- ✅ Unit tests pass for utilities
- ✅ CLI can be invoked with all flags
- ✅ All v0.2 dependencies install cleanly
- ✅ README documentation complete
- ✅ Code <100 LOC for this epic

---

## Notes

- This epic establishes foundation ONLY - no actual PDF processing yet
- Focus on integration setup, not custom implementation
- V0.2 is dramatically simpler than v0.1 (80% less code)
- Next epic (E2) will integrate Unstructured/Marker for actual parsing
- Target completion: Day 1 of 1-week sprint
