# Migration Guide: v0.1 (Docling) → v0.2 (Fast MVP)

**Date:** 2025-10-14
**Status:** Ready for epic/story recreation

---

## What Changed

### Architecture Pivot
Shifted from custom implementation to OSS-first approach:

| Component | v0.1 (OLD) | v0.2 (NEW) |
|-----------|-----------|-----------|
| **Parsing** | Docling | Unstructured or Marker |
| **Markdown** | Custom normalizer | Marker (built-in) |
| **Chunking** | LangChain | LangChain (same) |
| **Target LOC** | ~500 lines | <400 lines |
| **Timeline** | 9-14 days | 1 week |

### Why This Change?
✅ **80% less custom code** - Leverage battle-tested libraries
✅ **Faster MVP** - 1 week vs 2 weeks
✅ **Proven reliability** - Unstructured/Marker are production-grade
✅ **Easier maintenance** - Less code to debug and maintain

---

## Impact on Existing Work

### Epic/Story Status

| Epic | v0.1 Status | v0.2 Status | Action |
|------|------------|------------|---------|
| **Epic 1** (Project Setup) | 5 stories created | ~80% reusable | Update dependencies |
| **Epic 2** (Docling Parsing) | 6 stories defined | ❌ Obsolete | Replace with Unstructured/Marker |
| **Epic 3** (Markdown Normalization) | 6 stories defined | ❌ Mostly obsolete | Marker handles this |
| **Epic 4** (Metadata & Output) | 7 stories defined | ✅ ~90% reusable | Minor adjustments |
| **Epic 5** (Integration & Testing) | 8 stories defined | ✅ ~85% reusable | Adjust for new libs |

### What Was Preserved?
✅ CLI framework design
✅ Output directory structure
✅ Metadata contracts (meta.yaml, _index.json)
✅ Chunk front matter format
✅ Testing strategy
✅ Error handling approach

### What Was Simplified?
- ❌ Custom PDF parser integration → Use Unstructured/Marker directly
- ❌ Custom Markdown normalizer → Marker handles conversion
- ❌ Custom table extraction → Unstructured handles tables
- ❌ Custom OCR strategy → Unstructured has built-in OCR

---

## New Epic Structure (Estimated)

Based on v0.2 architecture, expected epics:

### **Epic 1: Project Setup & CLI** (~3-4 stories)
- Initialize project with new dependencies
- CLI argument parsing
- Basic utilities (slugify, atomic write, logging)

### **Epic 2: Unstructured/Marker Integration** (~3-4 stories)
- Integrate Unstructured parser
- Integrate Marker Markdown formatter
- Handle parsing errors and edge cases

### **Epic 3: Chunking & Metadata** (~3-4 stories)
- LangChain text splitter integration
- Chunk metadata (YAML front matter)
- Document metadata (meta.yaml)

### **Epic 4: Output & Manifest** (~2-3 stories)
- File writer and directory management
- Manifest generation (_index.json)
- Deterministic output verification

### **Epic 5: Integration & Testing** (~3-4 stories)
- End-to-end pipeline
- Progress reporting
- Comprehensive test suite
- README and documentation

**Total: ~14-19 stories** (vs. 32 stories in v0.1)

---

## Configuration Updates

### ✅ Updated `.bmad-core/core-config.yaml`
- Changed `architectureVersion` to `"0.2"`
- Added `parsingStrategy: "unstructured-marker"`
- Added `targetLOC: 400`
- Updated dependency list
- Added architecture notes

### ✅ Updated Documentation
- `docs/prd.md` → v0.2
- `docs/architecture.md` → v0.2

### ✅ Cleaned Up
- Old epics and stories archived (if they existed)
- Ready for fresh epic/story creation

---

## Next Steps (CONFIRMED)

### ✅ Your Proposed Workflow is CORRECT:

1. **Switch to PO Agent** (`/po`)
   - Create new epics based on v0.2 PRD and Architecture
   - Expected: 5 epics with 14-19 total stories
   - Much simpler structure than v0.1

2. **Switch to SM Agent** (`/sm`)
   - Create stories from the new epics
   - Use `draft` command for each story
   - Stories will reference v0.2 architecture

3. **Begin Implementation**
   - Start with Epic 1 (Project Setup)
   - Use `/dev` agent or implement manually
   - Target completion: 1 week

---

## Key Differences in Story Content

### v0.1 Stories Had:
- Complex Docling configuration
- Custom parsing logic
- Custom Markdown normalization code
- Table extraction strategies

### v0.2 Stories Will Have:
- Simple Unstructured/Marker API calls
- Minimal configuration
- Focus on integration, not implementation
- Leverage library defaults

### Example Comparison:

**v0.1 Story 2.3:** "Integrate Docling Parser (Basic)"
- Install Docling
- Configure parser settings
- Handle Docling output format
- Extract blocks manually
- ~50 lines of custom code

**v0.2 Story 2.1:** "Integrate Unstructured Parser"
- Install Unstructured
- Call `partition_pdf()`
- Use Marker for Markdown conversion
- ~15 lines of code

---

## Migration Checklist

- [x] Update PRD to v0.2
- [x] Update Architecture to v0.2
- [x] Update core-config.yaml
- [x] Archive/clean old epics and stories
- [x] Document migration rationale
- [ ] Create new v0.2 epics (use PO agent)
- [ ] Create stories from new epics (use SM agent)
- [ ] Begin implementation

---

## Questions or Concerns?

If you have questions about:
- **Epic structure:** Use PO agent to validate
- **Story details:** Use SM agent for clarification
- **Architecture decisions:** Review `docs/architecture.md`
- **Timeline:** 1-week sprint plan in PRD Section 12

---

**Status:** ✅ Ready for PO agent to create new epics based on v0.2 architecture
