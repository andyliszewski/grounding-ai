# Epic 2: Unstructured/Marker Integration (v0.2)

**Epic ID:** E2-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P0
**Estimated Stories:** 3-4
**Dependencies:** Epic 1 (Project Setup)
**Architecture Version:** 0.2 (Fast MVP - OSS Integration)

---

## Overview

Integrate Unstructured and Marker libraries for PDF parsing and Markdown conversion. This epic leverages existing OSS tools instead of building custom parsing logic, dramatically reducing implementation complexity.

---

## Goals

1. Integrate Unstructured for PDF parsing (text extraction, layout preservation)
2. Integrate Marker for Markdown formatting (structure preservation)
3. Handle OCR strategy (auto/on/off) using Unstructured's built-in OCR
4. Implement error handling for parsing failures
5. Scanner for PDF discovery in input directory
6. Target <80 LOC for this epic (mostly library integration)

---

## User Stories

**From PRD Section 2:** As a developer, I want to use proven OSS libraries so I can avoid building custom parsing logic.

**From PRD Section 6:** As a user, I want PDFs parsed into Markdown automatically so I get structured output.

---

## Epic Acceptance Criteria

1. ✅ Unstructured successfully parses text-based PDFs
2. ✅ Marker converts parsed content to clean Markdown
3. ✅ Scanner discovers all `.pdf` files in input directory
4. ✅ OCR triggers automatically for scanned PDFs (when `--ocr auto`)
5. ✅ Tables and headings preserved in Markdown output
6. ✅ Individual PDF failures logged but don't stop batch
7. ✅ Parser returns full document Markdown for each PDF

---

## Technical Scope

### Components to Build

- **Scanner Module (`scanner.py`)**
  - Discover PDFs in input directory
  - Sort files deterministically (alphabetical)
  - Return list of Path objects

- **Parser Module (`parser.py`)**
  - Wrapper for Unstructured's `partition_pdf()`
  - OCR strategy handling (auto/on/off)
  - Error handling for parsing failures
  - Return parsed document object

- **Formatter Module (`formatter.py`)**
  - Wrapper for Marker's Markdown conversion
  - Preserve headings, paragraphs, tables
  - Handle special formatting (code blocks, lists)
  - Return formatted Markdown string

- **File Context**
  - Data structure to hold per-file state
  - Contains: file path, output dir, parameters, file SHA-1

---

## Dependencies

**Python Libraries (from Epic 1):**
- `unstructured>=0.15.0` - PDF parsing
- `marker>=0.2.0` - Markdown conversion
- `pathlib` (stdlib) - file operations

[Source: docs/architecture.md#10-dependencies]

---

## Stories Breakdown

### Story 2.1: Implement PDF Scanner
- Create function to discover PDFs in input directory
- Sort files alphabetically for determinism
- Filter by `.pdf` extension (case-insensitive)
- Handle empty directories gracefully
- Return sorted list of Path objects

**AC:**
- Discovers all `.pdf` and `.PDF` files
- Sorts files alphabetically
- Empty directory returns empty list
- Invalid paths raise clear errors

---

### Story 2.2: Integrate Unstructured Parser
- Install and test Unstructured library
- Wrap `unstructured.partition_pdf()` function
- Configure parser settings (OCR, layout detection)
- Extract text, tables, and structure
- Return parsed document object
- Handle parser exceptions gracefully

**AC:**
- Successfully parses standard text PDFs
- Extracts headings, paragraphs, tables
- Parser exceptions caught and logged
- Returns structured document object

---

### Story 2.3: Integrate Marker Formatter
- Install and test Marker library
- Wrap Marker's Markdown conversion functions
- Preserve document structure in Markdown
- Handle tables → Markdown table format
- Handle headings → proper Markdown levels
- Return complete Markdown string

**AC:**
- Converts parsed docs to clean Markdown
- Preserves heading hierarchy
- Tables render as Markdown tables
- Output is valid Markdown syntax

---

### Story 2.4: Implement OCR Strategy
- Add `--ocr` flag handling (auto/on/off)
- Configure Unstructured OCR settings
- Implement auto-detection logic (text yield check)
- Trigger OCR when text yield < threshold
- Log OCR trigger decisions

**AC:**
- `--ocr off` disables OCR completely
- `--ocr on` always enables OCR
- `--ocr auto` checks text yield and enables if low
- OCR decisions logged with file name
- Scanned PDFs successfully processed

---

## Data Contracts

### Parsed Document Structure (Unstructured Output)
```python
{
    "elements": [
        {
            "type": "Title" | "NarrativeText" | "Table",
            "text": "Content...",
            "metadata": {
                "page_number": 1,
                "coordinates": {...}
            }
        }
    ]
}
```

### Formatted Markdown Output (Marker)
```markdown
# Document Title

## Section 1

Paragraph content...

### Subsection 1.1

More content...

| Column 1 | Column 2 |
|----------|----------|
| Data     | Data     |
```

---

## Integration Notes

[Source: docs/architecture.md#2-system-composition]

**Unstructured Integration:**
```python
from unstructured.partition.pdf import partition_pdf

def parse_pdf(file_path: Path, ocr_mode: str = "auto"):
    """Parse PDF using Unstructured."""
    elements = partition_pdf(
        filename=str(file_path),
        strategy="auto",  # Unstructured chooses best method
        infer_table_structure=True,  # Extract tables
        extract_images_in_pdf=False,  # Skip images for MVP
        ocr_languages="eng"  # English OCR
    )
    return elements
```

**Marker Integration:**
```python
from marker.convert import convert_single_pdf

def format_markdown(parsed_doc) -> str:
    """Convert parsed PDF to Markdown using Marker."""
    # Marker converts directly from PDF
    # Or accepts Unstructured output
    markdown = marker.convert(parsed_doc)
    return markdown
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Unstructured parsing failures | High | Graceful error handling, log details, continue batch |
| Marker conversion issues | Medium | Fallback to simple text extraction if needed |
| OCR performance overhead | Medium | Make OCR optional, use "auto" mode by default |
| Memory issues with large PDFs | Medium | Stream processing if possible, document limitations |
| Table extraction quality | Medium | Accept imperfect tables for MVP, improve later |

---

## Testing Strategy

- **Unit Tests:**
  - Scanner with mock directory
  - Parser with sample PDF
  - Formatter with sample parsed doc

- **Integration Tests:**
  - Parse 3 sample PDFs:
    1. Standard text PDF
    2. Table-heavy PDF
    3. Scanned PDF (OCR test)

- **Error Cases:**
  - Corrupted PDF
  - Zero-text PDF
  - Password-protected PDF

---

## Definition of Done

- ✅ All story acceptance criteria met
- ✅ Integration tests pass with 3 sample PDFs
- ✅ Error handling tested and working
- ✅ OCR strategy verified
- ✅ Code <80 LOC (mostly library wrappers)
- ✅ Code review completed

---

## Notes

- Unstructured and Marker do the heavy lifting - we just integrate
- This is **dramatically simpler** than custom Docling implementation (v0.1 Epic 2 + 3)
- Focus on correct library usage, not custom algorithms
- Marker may handle Markdown conversion directly from PDF (check docs)
- If Unstructured + Marker combo has issues, can use either library standalone
- Target completion: Day 2 of 1-week sprint
