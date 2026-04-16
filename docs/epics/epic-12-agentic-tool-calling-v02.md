# Epic 12: Agentic Tool Calling for Local LLMs

**Epic ID:** E12-v0.2
**Owner:** Andy
**Status:** Draft
**Priority:** P2
**Completed Stories:** 0/4
**Dependencies:** Epic 11 (Interactive Corpus Agents), Epic 6 (Vector Embeddings)
**Target Completion:** TBD

---

## Overview

Enable local LLMs (via Ollama) to use tool calling for agentic corpus search, allowing the model to autonomously decide when to search rather than always retrieving context upfront.

**Problem Statement:**
- Current `local_rag.py` implements simple RAG: every query triggers corpus search
- The LLM cannot decide whether it needs to search or has sufficient knowledge
- No multi-turn reasoning capability - one-shot retrieval only
- The MCP server works with Claude but cannot be called by Ollama (different protocol)

**Solution:**
- Implement Ollama-compatible tool calling with `search_corpus` tool
- Create agentic loop that processes tool calls until LLM returns final response
- Reuse existing FAISS search infrastructure from `local_rag.py`
- Maintain backward compatibility with existing simple RAG mode

---

## Goals

1. Enable local LLMs to autonomously decide when to search the corpus
2. Implement multi-turn agentic loop with tool calling support
3. Reuse existing FAISS search code - no new dependencies required
4. Maintain backward compatibility with simple RAG mode
5. Support Qwen 2.5 models (14B and 32B) via Ollama
6. Provide clear documentation and examples

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Qwen 2.5 + tools   │ ◄── Tool schema: search_corpus(query, top_k)
│  (via Ollama API)   │
└─────────┬───────────┘
          │
          ▼
    ┌───────────┐
    │Tool call? │──No──► Return final response
    └─────┬─────┘
          │ Yes
          ▼
┌─────────────────────┐
│ Execute locally     │ ◄── Reuse existing FAISS search
│ (search_corpus)     │
└─────────┬───────────┘
          │
          └──► Add results to messages, loop back to LLM
```

### Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "search_corpus",
    "description": "Search the agent's corpus for relevant documents using semantic similarity.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural language search query"
        },
        "top_k": {
          "type": "integer",
          "description": "Number of results to return (default: 5)",
          "default": 5
        }
      },
      "required": ["query"]
    }
  }
}
```

---

## Stories Breakdown

### Story 12.1: Agentic Loop Infrastructure
- Implement tool calling request/response handling for Ollama API
- Create agent loop that processes tool calls until LLM returns final response
- Add max_iterations safeguard to prevent infinite loops
- Handle tool execution errors gracefully

**AC:**
- Ollama API called with `tools` parameter in request
- Tool calls in response are detected and processed
- Agent loop continues until no more tool calls or max iterations
- Errors during tool execution returned as tool result (not exceptions)
- Max iterations configurable (default: 5)

**Status:** Draft

### Story 12.2: search_corpus Tool Implementation
- Define JSON schema for search_corpus tool
- Adapt existing FAISS search to tool executor interface
- Format search results for tool response
- Support configurable top_k parameter

**AC:**
- Tool schema matches Ollama/OpenAI function calling format
- Existing `search_corpus()` function wrapped for tool interface
- Results formatted as readable text with source citations
- Invalid arguments handled gracefully
- Works with existing FAISS indexes from Epic 6

**Status:** Draft

### Story 12.3: CLI Integration
- Add `--agentic` flag to enable tool-calling mode
- Update REPL to support agentic conversation flow
- Show tool calls in output (optional verbose mode)
- Maintain backward compatibility with simple RAG mode

**AC:**
- `--agentic` flag enables tool-calling mode
- Default behavior unchanged (simple RAG)
- Tool calls shown in verbose mode (`-v`)
- REPL supports multi-turn agentic conversations
- Help text updated with new options

**Status:** Draft

### Story 12.4: Documentation Updates
- Update `how-to-local-agent.md` with agentic mode instructions
- Document tool calling behavior and limitations
- Add examples of multi-turn agentic conversations
- Update CLAUDE.md with architecture details

**AC:**
- `how-to-local-agent.md` has Agentic Mode section
- Examples show single-turn and multi-turn tool calling
- Limitations documented (latency, token usage, max iterations)
- CLAUDE.md updated with tool calling architecture
- Troubleshooting section covers common issues

**Status:** Draft

---

## Technical Details

### Ollama Tool Calling API

**Request format:**
```json
{
  "model": "qwen2.5:14b",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [{ "type": "function", "function": {...} }],
  "stream": false
}
```

**Response with tool call:**
```json
{
  "message": {
    "role": "assistant",
    "content": "I'll search for...",
    "tool_calls": [{
      "type": "function",
      "function": {
        "name": "search_corpus",
        "arguments": {"query": "...", "top_k": 5}
      }
    }]
  }
}
```

**Tool result format:**
```json
{
  "role": "tool",
  "tool_name": "search_corpus",
  "content": "...search results..."
}
```

### Why Not Bridge to MCP Server?

- MCP uses stdio protocol, not HTTP
- Would require subprocess spawning and JSON marshaling
- Adds latency and complexity
- Direct local execution is simpler and faster

---

## Dependencies

### Epic Dependencies
- **Epic 11**: Agent personas for greeting and style
- **Epic 6**: FAISS embeddings infrastructure

### External Dependencies
- Ollama with OpenAI-compatible API (already used)
- Qwen 2.5 models (already downloaded)

### Code Dependencies
- `scripts/local_rag.py` - Base implementation to extend
- `pdf2llm/embedder.py` - Embedding generation
- `pdf2llm/vector_store.py` - FAISS search

---

## Implementation Order

```
Story 12.1 (Agentic Loop Infrastructure)
    └── Core loop mechanism, tool call detection

Story 12.2 (search_corpus Tool)
    └── Tool schema, executor, result formatting

Story 12.3 (CLI Integration)
    └── --agentic flag, REPL support, verbose mode

Story 12.4 (Documentation)
    └── User docs, examples, CLAUDE.md updates
```

Stories 12.1 and 12.2 can be developed in parallel as they address different concerns. Story 12.3 depends on both. Story 12.4 should be completed last.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Infinite tool loops | High | Hard limit on iterations (default: 5) |
| Model ignores tools | Medium | Test with Qwen 2.5; fallback to simple RAG |
| Slow multi-turn | Medium | Document latency expectations; optimize prompts |
| Context window exhaustion | Medium | Track tokens; truncate old turns if needed |
| Invalid tool arguments | Low | Validate before execution; return error as result |

---

## Testing Strategy

### Unit Tests
- Tool schema generation
- Tool executor wrapper
- Agent loop iteration logic
- Error handling paths

### Integration Tests
- Full agentic conversation with mock Ollama
- Tool call → execution → response cycle
- Max iterations enforcement
- Backward compatibility with simple RAG

### Manual Testing
- Test with Qwen 2.5 14B and 32B
- Verify tool calls appear in verbose mode
- Test REPL multi-turn conversations
- Verify simple RAG still works without `--agentic`

---

## Acceptance Criteria (Epic Level)

1. Local LLM can decide whether to search corpus or answer directly
2. Multi-turn tool calling works (LLM can search multiple times)
3. Existing simple RAG mode still works (backward compatible)
4. Works with Qwen 2.5 14B and 32B models
5. Max iterations prevents runaway loops
6. Tool call trace visible in verbose mode
7. Documentation complete with examples

---

## Definition of Done

- Story 12.1 completed: Agentic loop infrastructure functional
- Story 12.2 completed: search_corpus tool working
- Story 12.3 completed: CLI integration with --agentic flag
- Story 12.4 completed: Documentation updated
- All acceptance criteria verified
- Backward compatibility confirmed
- Manual testing with Qwen 2.5 successful

---

## Future Enhancements (Out of Scope)

- Additional tools: `list_agents`, `get_document`, `summarize_chunk`
- Streaming tool calls
- Tool call caching
- Integration with MCP protocol (if Ollama adds support)
- Tool call history persistence

---

## References

- [Feature Request](../feature-requests/agentic-local-llm-tool-calling.md) - Original feature request
- [Story 12.1](../stories/12.1-agentic-loop-infrastructure-v02.md) - Agentic loop infrastructure
- [Story 12.2](../stories/12.2-search-corpus-tool-v02.md) - Tool implementation
- [Story 12.3](../stories/12.3-cli-integration-v02.md) - CLI integration
- [Story 12.4](../stories/12.4-documentation-updates-v02.md) - Documentation
- [Epic 11](./epic-11-interactive-corpus-agents-v02.md) - Interactive corpus agents
- [Epic 6](./epic-6-vector-embeddings-v02.md) - Vector embeddings
- `scripts/local_rag.py` - Current simple RAG implementation
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
