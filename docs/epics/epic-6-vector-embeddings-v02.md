# Epic 6: Vector Embeddings & Semantic Search (v0.2)

**Epic ID:** E6-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 6/6
**Dependencies:** Epic 3 (Chunking & Metadata), Epic 4 (Output & Manifest)
**Architecture Version:** 0.2 (Enhanced)
**Target Completion:** TBD

---

## Overview

Extend the pdf2llm ingestion pipeline to generate vector embeddings for semantic search capabilities. Enable LLM agents and applications to query the corpus using natural language instead of keyword-based search.

---

## Goals

1. Integrate sentence-transformers embedding model into pipeline
2. Generate embeddings for all chunks during ingestion
3. Implement FAISS vector store for efficient similarity search
4. Create query interface for semantic retrieval
5. Add CLI flags for embedding generation and configuration
6. Validate with integration tests
7. Target ~1,000 LOC for embedding infrastructure
8. Expose FAISS search as MCP tool for agent integration

---

## Stories Breakdown

### Story 6.1: Integrate Embedding Model (sentence-transformers)
- Add sentence-transformers dependency
- Create `pdf2llm/embedder.py` module
- Implement `generate_embedding(text: str) -> np.ndarray`
- Select and load embedding model (all-MiniLM-L6-v2)
- Handle model download and caching
- Unit tests for embedding generation

**AC:**
- Embedding model loads successfully
- Text inputs generate 384-dimensional vectors
- Model cached locally after first download
- Embeddings are deterministic for same input
- Unit tests validate embedding properties

### Story 6.2: Generate Embeddings During Ingestion Pipeline
- Add embedding generation step to pipeline
- Generate embeddings for each chunk after chunking
- Store embeddings alongside chunk metadata
- Add `--emit-embeddings` CLI flag
- Update pipeline config to support embeddings
- Tests validate embeddings generated per chunk

**AC:**
- Embeddings generated for all chunks when flag enabled
- Pipeline performance acceptable (<5s per chunk)
- Embeddings accessible during write phase
- CLI flag properly toggles feature
- Integration tests verify embedding generation

### Story 6.3: Implement Vector Store (FAISS)
- Add faiss-cpu dependency
- Create `pdf2llm/vector_store.py` module
- Implement `write_vector_index(embeddings, chunk_ids, output_dir)`
- Create `_embeddings.faiss` index file
- Create `_chunk_map.json` mapping chunk_ids to embeddings
- Unit tests for vector store operations

**AC:**
- FAISS index created in corpus output directory
- Chunk map correctly associates IDs with embeddings
- Index supports similarity search
- Index file format is portable
- Tests validate write/read operations

### Story 6.4: Create Query Interface for Semantic Search
- Create `pdf2llm/query.py` module
- Implement `query_corpus(corpus_path, query, top_k) -> List[ChunkResult]`
- Return chunk_ids, similarity scores, and content
- CLI interface: `python -m pdf2llm.query --corpus ./corpus --query "..." --top-k 5`
- Display results with scores and source info
- Integration tests with sample queries

**AC:**
- Query returns top-k most relevant chunks
- Similarity scores are meaningful (0-1 range)
- Results include chunk content and metadata
- CLI provides user-friendly output
- Tests validate retrieval accuracy

### Story 6.5: CLI Integration & Testing
- Update README with embedding examples
- Add `--vector-db` option for future extensibility (faiss, chroma)
- End-to-end integration test: ingest → query → verify
- Performance benchmark (chunks/second for embedding)
- Update architecture.md with embedding architecture
- Validate with multiple document corpus

**AC:**
- README examples show full workflow
- End-to-end test passes with sample corpus
- Performance meets targets (>100 chunks/sec embedding)
- Documentation explains use cases and limitations
- All CLI flags documented

### Story 6.6: MCP Corpus Search Server
- Create MCP server package structure
- Implement `search_corpus` tool with query, agent, top_k parameters
- Implement `list_corpus_agents` tool for agent discovery
- Lazy-load embedding model and cache FAISS indexes
- Register server in .mcp.json for Claude Code integration
- Graceful error handling for missing agents/indexes

**AC:**
- MCP server starts without errors when configured
- search_corpus returns relevant chunks from agent's filtered index
- Results include source attribution (doc_id, chunk_id, source)
- Index caching prevents redundant disk reads
- Graceful error handling with helpful hints

---

## Technical Architecture

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Size**: ~80MB
- **Performance**: ~1000 sentences/sec on CPU

### Vector Store
- **Primary**: FAISS (Facebook AI Similarity Search)
- **Index Type**: Flat L2 (exact search)
- **Storage**: Local files in corpus directory
- **Scalability**: Handles 100k+ chunks efficiently

### Pipeline Integration
```
Existing Pipeline:
Parse → Format → Chunk → Metadata → Write

Enhanced Pipeline:
Parse → Format → Chunk → Metadata → [Embed] → Write → [Vector Store]
```

### Output Structure
```
corpus/
├── _index.json                 # Document manifest (existing)
├── _embeddings.faiss           # NEW: FAISS vector index
├── _chunk_map.json             # NEW: chunk_id → embedding mapping
├── doc-slug/
│   ├── doc.md
│   ├── meta.yaml
│   └── chunks/
│       ├── ch_0001.md          # Chunk with YAML front matter
│       └── ...
```

---

## Dependencies

### Python Packages (New)
- `sentence-transformers>=2.2.0` - Embedding model
- `faiss-cpu>=1.7.0` - Vector similarity search
- `numpy>=1.24.0` - Array operations (likely already present)

### System Dependencies
- None (all Python-based)

### Epic Dependencies
- **Epic 3**: Requires chunking infrastructure and chunk metadata
- **Epic 4**: Requires output structure and manifest generation

---

## Use Cases

### Use Case 1: Question Answering
```bash
# Ingest textbooks with embeddings
pdf2llm --in ./textbooks --out ./corpus --emit-embeddings --parser unstructured

# Query the corpus
python -m pdf2llm.query --corpus ./corpus --query "What is the stress-strain relationship?" --top-k 5
```

### Use Case 2: Agent Integration
```python
# Agent instruction references corpus
"""
When answering questions, use semantic search:
1. Run: python -m pdf2llm.query --corpus ./corpus --query "{question}" --top-k 5
2. Read the returned chunks
3. Synthesize answer from relevant content
"""
```

### Use Case 3: Research Assistant
- Ingest multiple research papers
- Query for concepts across papers
- Find similar content without exact keyword matches

---

## Performance Targets

- **Embedding Generation**: >100 chunks/second on CPU
- **Index Build**: <10 seconds for 10k chunks
- **Query Latency**: <100ms for top-10 search
- **Memory Overhead**: <2GB for 100k chunk corpus
- **Disk Overhead**: ~1.5KB per chunk (embedding + metadata)

---

## Known Limitations

### Current Scope (v0.2)
- CPU-only embedding generation (GPU support future enhancement)
- Flat index (exact search) - approximate search for larger corpora in future
- English-optimized model - multilingual support future enhancement
- No incremental updates - full reindex required for corpus changes

### Future Enhancements (Post-Epic 6)
- GPU acceleration for faster embedding
- Approximate nearest neighbor (ANN) indices for scale
- Alternative vector stores (Chroma, Pinecone)
- Incremental index updates
- Multilingual embedding models
- Hybrid search (semantic + keyword)

---

## Testing Strategy

### Unit Tests
- Embedding generation determinism
- Vector store write/read operations
- Query result ranking correctness
- Edge cases (empty query, no results)

### Integration Tests
- Full pipeline with embeddings enabled
- Query against test corpus
- Deterministic embedding verification
- Performance benchmarks

### Golden Tests
- Known queries → expected results
- Similarity score validation
- Result ranking correctness

---

## Acceptance Criteria (Epic Level)

1. ✅ Embedding model integrates cleanly into pipeline
2. ✅ `--emit-embeddings` flag generates embeddings during ingestion
3. ✅ FAISS index created in corpus output directory
4. ✅ Query interface returns relevant chunks with scores
5. ✅ Performance targets met for 1k chunk corpus
6. ✅ End-to-end test: ingest → query → verify
7. ✅ Documentation complete (README, architecture.md)
8. ✅ No breaking changes to existing functionality

---

## Definition of Done

- All 6 stories completed and tested
- Integration tests passing
- Performance benchmarks met
- README and architecture.md updated
- No regressions in existing functionality
- Code reviewed and production-ready
- CLI flags documented
- Query examples working

---

## Risk Assessment

### Technical Risks
- **Model download failures**: Mitigated by retry logic and clear error messages
- **Memory constraints**: Mitigated by batch processing and configurable batch sizes
- **Index corruption**: Mitigated by atomic writes and validation checks
- **Performance degradation**: Mitigated by benchmarking and optimization

### Integration Risks
- **Breaking existing pipeline**: Mitigated by feature flag and comprehensive tests
- **Dependency conflicts**: Mitigated by version pinning and testing
- **File format changes**: Mitigated by backward compatibility checks

### Mitigation Strategy
- Feature is opt-in via `--emit-embeddings` flag
- Existing workflows unaffected when flag not used
- Comprehensive testing before merge
- Performance monitoring during development

---

## Success Metrics

- Query retrieval accuracy >80% for test queries
- Embedding generation <50ms per chunk average
- Query latency <100ms for top-10 results
- Zero regressions in existing test suite
- Documentation clarity validated by test user

---

## Notes

- This epic extends v0.2 MVP with semantic search capabilities
- Maintains backward compatibility with existing corpus structure
- Optional feature - doesn't affect users who don't need embeddings
- Foundation for future RAG (Retrieval-Augmented Generation) integration
- Enables more intelligent agent interactions with corpus
- FAISS chosen for simplicity and local-first approach
- Future epics may add cloud vector stores or advanced features
