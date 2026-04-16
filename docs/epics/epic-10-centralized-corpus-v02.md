# Epic 10: Centralized Corpus Architecture (v0.2)

**Epic ID:** E10-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 0/5
**Dependencies:** Epic 4 (Output & Manifest), Epic 6 (Vector Embeddings)
**Architecture Version:** 0.2 (Enhanced)
**Target Completion:** TBD
**Architecture Document:** [DRAFT-centralized-corpus-architecture.md](../DRAFT-centralized-corpus-architecture.md)

---

## Overview

Replace per-agent corpus silos with a single centralized corpus that agents selectively filter. This epic implements the infrastructure for multi-machine corpus management with agent-based document filtering and Syncthing synchronization.

**Problem Statement:**
- Duplicate ingestion when same book used by multiple agents
- No shared resources across agents
- Multiple sync targets needed
- "Where is X?" requires searching multiple locations

**Solution:**
- Single centralized corpus with `collections` metadata
- Agent definitions specify which documents they can access (by slug or collection)
- Embeddings generated per-agent on consuming machines
- Syncthing handles multi-machine sync

---

## Goals

1. Extend manifest and metadata schemas with `collections` field
2. Create agent filter module for selective document access
3. Build migration script to merge existing agent corpora
4. Add CLI support for tagging documents with collections
5. Enable agent-filtered embedding generation
6. Maintain backward compatibility with existing corpus structure

---

## Stories Breakdown

### Story 10.1: Schema Extension (collections field)
- Add `collections` field to `ManifestEntry` in `manifest.py`
- Add `collections` parameter to `build_meta_yaml()` in `meta.py`
- Update manifest validation to accept optional `collections`
- Add `source_agent` field for migration provenance tracking
- Unit tests for schema changes

**AC:**
- `ManifestEntry` includes optional `collections: List[str]`
- `meta.yaml` generation supports collections parameter
- Existing manifests without collections load correctly (backward compatible)
- Collections use lowercase kebab-case validation
- Unit tests verify schema serialization/deserialization

---

### Story 10.2: Agent Filter Module
- Create new `pdf2llm/agent_filter.py` module
- Implement `AgentConfig` dataclass for agent definitions
- Implement `load_agent_config(agent_name, agents_dir) -> AgentConfig`
- Implement `filter_manifest(manifest, agent_config) -> filtered_manifest`
- Support filtering by explicit slugs and by collections
- Unit tests for all filter operations

**AC:**
- Agent YAML files parsed correctly
- Slug-based filtering works (explicit list)
- Collection-based filtering works (tag matching)
- Exclude patterns work (exclude_slugs)
- Filter returns subset of manifest entries
- Edge cases handled (empty filter, no matches)

---

### Story 10.3: Migration Script
- Create `scripts/migrate_corpus.py` one-time migration tool
- Merge existing agent corpora into unified structure
- Detect and handle slug collisions (append doc_id suffix)
- Populate `source_agent` field for provenance
- Generate initial `agents/*.yaml` definitions
- Merge multiple `_index.json` files
- Dry-run mode for safety

**AC:**
- Script discovers all existing agent corpora
- Documents copied to unified corpus without duplication
- Slug collisions detected and resolved
- `source_agent` populated in meta.yaml
- Agent definition files generated
- Merged manifest validates correctly
- Dry-run shows what would happen without writing

---

### Story 10.4: CLI Collections Flag
- Add `--collections` flag to `pdf2llm` CLI
- Accept comma-separated collection names
- Pass collections through pipeline to meta.yaml generation
- Validate collection names (lowercase kebab-case)
- Update help text and documentation

**AC:**
- `pdf2llm --in ./pdfs --out ./corpus --collections science,biology` works
- Collections appear in generated meta.yaml
- Collections appear in _index.json manifest
- Invalid collection names rejected with clear error
- Help text documents the flag

---

### Story 10.5: Agent-Filtered Embeddings
- Extend `pdf2llm embeddings` subcommand with `--agent` flag
- Load agent config and filter manifest before embedding
- Generate embeddings only for agent's filtered document set
- Add `--check` flag to detect stale embeddings
- Store embeddings in agent-specific directory

**AC:**
- `pdf2llm embeddings --agent scientist --corpus ./corpus --out ./embeddings/scientist/` works
- Only documents matching agent filter are embedded
- `--check` flag compares timestamps and reports staleness
- Exit code 1 if embeddings are stale
- Embeddings stored in correct agent subdirectory

---

## Technical Architecture

### New Module: agent_filter.py

```python
@dataclass
class AgentConfig:
    name: str
    description: str
    slugs: List[str] | None = None
    collections: List[str] | None = None
    exclude_slugs: List[str] | None = None

def load_agent_config(agent_name: str, agents_dir: Path) -> AgentConfig:
    """Load agent definition from YAML file."""
    pass

def filter_manifest(manifest: ManifestData, config: AgentConfig) -> ManifestData:
    """Return manifest entries matching agent's filter criteria."""
    pass
```

### Schema Extensions

**ManifestEntry (manifest.py):**
```python
@dataclass
class ManifestEntry:
    # ... existing fields ...
    collections: List[str] | None = None  # NEW
    source_agent: str | None = None       # NEW (migration provenance)
```

**meta.yaml additions:**
```yaml
collections:
  - science
  - biology
source_agent: "scientist"  # Optional, for migration tracking
```

### Directory Structure (Target State)

```
corpus_root/
├── corpus/
│   ├── _index.json
│   └── <slug>/
│       ├── doc.md
│       ├── meta.yaml  # Now includes collections
│       └── chunks/
├── originals/
│   └── <slug>.pdf
└── agents/
    ├── scientist.yaml
    ├── ceo.yaml
    └── musician.yaml

embeddings/              # On consuming machine only
├── scientist/
│   ├── index.faiss
│   └── chunk_map.json
└── ceo/
    └── ...
```

---

## Dependencies

### Python Packages (No New Dependencies)
- Uses existing PyYAML for agent config parsing
- Uses existing manifest/meta modules

### Epic Dependencies
- **Epic 4**: Requires manifest and meta.yaml infrastructure
- **Epic 6**: Requires vector store for filtered embeddings

### External Dependencies
- **Syncthing** (ops setup, not code): Multi-machine sync
- **Agent corpora** (data): Existing agent corpus directories for migration

---

## Implementation Order

1. **Story 10.1** (Schema) - Foundation for all other stories
2. **Story 10.4** (CLI) - Can be done in parallel with 10.2
3. **Story 10.2** (Agent Filter) - Core filtering logic
4. **Story 10.3** (Migration) - Requires 10.1 and 10.2
5. **Story 10.5** (Embeddings) - Requires 10.2

```
Story 10.1 (Schema)
   ├── Story 10.4 (CLI) ─────────────────────┐
   └── Story 10.2 (Agent Filter) ────────────┼──→ Story 10.5 (Embeddings)
              └── Story 10.3 (Migration) ────┘
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Migration corrupts existing data | High | Backup before migration; dry-run mode first |
| Slug collision during merge | Medium | Pre-scan for collisions; append doc_id suffix |
| Breaking existing workflows | Medium | All new fields optional; backward compatible |
| Embeddings out of sync | Low | `--check` command detects staleness |

---

## Testing Strategy

### Unit Tests
- Schema serialization/deserialization with collections
- Agent config loading from YAML
- Manifest filtering (by slug, by collection, excludes)
- Collection name validation (kebab-case)

### Integration Tests
- Full pipeline with `--collections` flag
- Migration script on test corpora
- Agent-filtered embedding generation
- Staleness detection with `--check`

### Manual Testing
- Migrate real agent corpora (with backup)
- Verify Syncthing sync works correctly
- Test query against agent-filtered embeddings

---

## Acceptance Criteria (Epic Level)

1. Schema supports `collections` field in manifest and meta.yaml
2. Agent filter module correctly filters by slugs and collections
3. Migration script successfully merges test corpora
4. CLI `--collections` flag tags documents during ingestion
5. Agent-filtered embeddings generated correctly
6. All existing tests continue to pass (no regressions)
7. Documentation updated (CLAUDE.md, README)

---

## Definition of Done

- All 5 stories completed and tested
- Unit and integration tests passing
- Backward compatibility verified
- Migration script tested on real data (with backup)
- CLAUDE.md updated with new CLI flags and modules
- No regressions in existing functionality

---

## Notes

- This epic enables multi-machine corpus management
- Syncthing configuration is an ops task, not covered in code
- Migration is one-time; script can be removed after use
- Collections are free-form but must be kebab-case
- Agent definitions synced with corpus (single sync target)
- Embeddings generated locally, not synced

---

## References

- [DRAFT-centralized-corpus-architecture.md](../DRAFT-centralized-corpus-architecture.md) - Full architecture document
- [Epic 4](./epic-4-output-manifest-v02.md) - Manifest infrastructure
- [Epic 6](./epic-6-vector-embeddings-v02.md) - Embedding infrastructure
