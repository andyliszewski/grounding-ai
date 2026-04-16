# Working Configuration - Quick Reference

**Last Verified**: 2025-10-16
**Status**: ✅ Production Ready

---

## Environment Requirements

### Python Version
- **Required**: Python 3.12.x
- **Not Supported**: Python 3.13 (aiohttp/cgi module incompatibility)
- **Check**: `python3.12 --version` should show `3.12.x`

### System Dependencies
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Verify
which pdfinfo  # Should return path to poppler tools
```

---

## Installation

```bash
# 1. Create virtual environment with Python 3.12
python3.12 -m venv pdf2llmenv
source pdf2llmenv/bin/activate  # Windows: pdf2llmenv\Scripts\activate

# 2. Install package
pip install -e .

# 3. Install additional dependencies
pip install --upgrade typer "unstructured[pdf]" pdfminer.six

# 4. Verify installation
pdf2llm --version
```

---

## Working Commands

### Basic Usage (Recommended)
```bash
pdf2llm --in ./pdfs --out ./corpus --parser unstructured
```

### With Options
```bash
# Clean output directory first
pdf2llm --in ./pdfs --out ./corpus --parser unstructured --clean

# With OCR for scanned documents
pdf2llm --in ./pdfs --out ./corpus --parser unstructured --ocr on

# Custom chunk sizes
pdf2llm --in ./pdfs --out ./corpus --parser unstructured \
  --chunk-size 800 --chunk-overlap 100

# Verbose logging
pdf2llm --in ./pdfs --out ./corpus --parser unstructured --verbose
```

### Dry Run (Test Without Processing)
```bash
pdf2llm --in ./pdfs --out ./corpus --parser unstructured --dry-run
```

---

## Key Dependencies

### Python Packages
```
typer==0.19.2                    # CLI framework
unstructured==0.18.15            # PDF parsing
marker-pdf==1.10.1               # PDF converter (currently not used)
pdfminer.six==20250506           # PDF utilities
langchain-text-splitters==0.3.11 # Text chunking
blake3==1.0.8                    # Hashing
pyyaml==6.0.3                    # Metadata
tqdm==4.67.1                     # Progress bars
```

### System Dependencies
```
poppler==25.10.0  # PDF utilities (pdfinfo, pdftotext)
```

---

## Current Limitations

### Parser Configuration
- ✅ **Use**: `--parser unstructured` (recommended)
- ❌ **Avoid**: `--parser marker` (API incompatibility)
- ℹ️ **Note**: Unstructured parser uses plaintext fallback mode

### Output Metadata
All chunks will include:
```yaml
fallback: true  # Indicates plaintext fallback was used
parser: unstructured
```

---

## Verified Test Results

**Command Used**:
```bash
pdf2llm --in test_pdfs --out /tmp/pdf2llm_test --parser unstructured --clean
```

**Results**:
```
Files processed: 4
Succeeded: 3 (75%)
Failed: 1 (malformed.pdf - expected)
Total chunks: 34
Processing time: 29.3s
Parse time: 28889.05ms
Format time: 2.19ms
```

**Output Structure**:
```
/tmp/pdf2llm_test/
├── _index.json                 # ✅ Created
├── 000-fda-plates/
│   ├── doc.md                  # ✅ 28.6KB
│   ├── meta.yaml               # ✅ Metadata
│   └── chunks/                 # ✅ 32 chunks
├── plain/
│   ├── doc.md                  # ✅ 168 bytes
│   ├── meta.yaml               # ✅ Metadata
│   └── chunks/ch_0001.md       # ✅ 1 chunk
└── table/
    ├── doc.md                  # ✅ 166 bytes
    ├── meta.yaml               # ✅ Metadata
    └── chunks/ch_0001.md       # ✅ 1 chunk
```

---

## Troubleshooting Quick Fixes

### "Got unexpected extra arguments"
```bash
# Upgrade Typer
pip install --upgrade typer
```

### "No module named 'cgi'"
```bash
# Use Python 3.12 instead of 3.13
python3.12 -m venv pdf2llmenv
source pdf2llmenv/bin/activate
pip install -e .
```

### "Unable to get page count"
```bash
# Install poppler
brew install poppler  # macOS
sudo apt-get install poppler-utils  # Ubuntu
```

### "marker.convert module not found"
```bash
# Use unstructured parser instead
pdf2llm --in ./pdfs --out ./corpus --parser unstructured
```

---

## For LLM Agent Consumption

### Reading the Manifest
```python
import json
from pathlib import Path

# Load corpus manifest
manifest_path = Path("/tmp/pdf2llm_test/_index.json")
manifest = json.loads(manifest_path.read_text())

# Iterate through documents
for doc in manifest["docs"]:
    print(f"Document: {doc['orig_name']}")
    print(f"  Slug: {doc['slug']}")
    print(f"  Chunks: {doc['chunk_count']}")
    print(f"  Path: {doc['doc_path']}")
```

### Reading Individual Chunks
```python
# Read a specific chunk
chunk_path = Path("/tmp/pdf2llm_test/plain/chunks/ch_0001.md")
chunk_content = chunk_path.read_text()

# Chunk includes YAML front matter with metadata
print(chunk_content)
```

### Chunk Format
```markdown
---
doc_id: 141dd45b
source: plain.pdf
chunk_id: 141dd45b-0001
page_start: null
page_end: null
hash: 43a8a5c7040c95a67f3bcaf5927ca49c48a4a1d57aaafbb2519d5fea29d66338
created_utc: '2025-10-16T16:23:30+00:00'
---

[Markdown content here]
```

---

## Support

**Issues Found?**
- See: `docs/qa/TROUBLESHOOTING-LOG-20251016.md`
- See: `README.md` Troubleshooting section
- Report: [GitHub Issues](https://github.com/anthropics/claude-code/issues)

**Documentation**:
- Architecture: `docs/architecture.md`
- Requirements: `docs/prd.md`
- Epics: `docs/epics/`
- Stories: `docs/stories/`

---

**Last Updated**: 2025-10-16
**Verified By**: Claude Code
**Platform**: macOS Darwin 25.1.0
