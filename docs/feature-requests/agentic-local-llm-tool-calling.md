# Feature Request: Agentic Tool Calling for Local LLMs

**Requestor:** Andy
**Date:** 2026-02-22
**Priority:** P2
**Related Epics:** Epic 11 (Interactive Corpus Agents), Epic 6 (Vector Embeddings)

---

## Summary

Enable local LLMs (via Ollama) to use tool calling for agentic corpus search, allowing the model to decide when to search rather than always retrieving context upfront.

---

## Problem Statement

**Current State:**
- `scripts/local_rag.py` implements simple RAG: every query triggers corpus search, then context is injected
- The LLM cannot decide whether it needs to search or has enough knowledge
- No multi-turn reasoning - one-shot retrieval only
- The MCP server (`mcp_servers/corpus_search/server.py`) works with Claude but cannot be called by Ollama (different protocol)

**Desired State:**
- Local LLMs can autonomously decide when to search the corpus
- Multi-turn agentic loop: LLM reasons → calls tool if needed → receives results → continues reasoning
- Same corpus search capabilities as Claude Code's MCP integration, but for local models

---

## Technical Research Summary

### Ollama Tool Calling Support
- Ollama exposes OpenAI-compatible `/api/chat` endpoint with `tools` parameter
- Request format: `{"model": "...", "messages": [...], "tools": [...]}`
- Response includes `tool_calls` array when model wants to invoke a tool
- Tool results sent back as `{"role": "tool", "tool_name": "...", "content": "..."}`

### Qwen 2.5 Capabilities
- Fully supports function calling (specifically trained for it)
- All quantizations (Q4_K_M, Q6_K, Q8_0) maintain tool calling ability
- Excellent accuracy - one of best-in-class for tool use
- Both 14B and 32B variants work equally well

### Recommended Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Qwen 2.5 + tools   │ ◄── Tool schema: search_corpus(query, agent, top_k)
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
│ Execute locally     │ ◄── Reuse existing FAISS search from local_rag.py
│ (search_corpus)     │
└─────────┬───────────┘
          │
          └──► Add results to messages, loop back to LLM
```

### Why Not Bridge to MCP Server?
- MCP uses stdio protocol, not HTTP
- Would require subprocess spawning and JSON marshaling
- Adds latency and complexity
- Direct local execution is simpler and faster

### Estimated Effort
- ~150-200 lines of new code
- Mostly refactoring existing functions
- No new dependencies (requests library already used)

---

## Proposed Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "search_corpus",
    "description": "Search the agent's corpus for relevant documents using semantic similarity. Returns chunks from ingested PDFs and documents that match the query.",
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

Note: `agent` parameter may be implicit from the session context.

---

## User Stories Needed

### Story: Agentic Loop Infrastructure
- Implement tool calling request/response handling for Ollama API
- Create agent loop that processes tool calls until LLM returns final response
- Add max_iterations safeguard to prevent infinite loops
- Handle tool execution errors gracefully

### Story: search_corpus Tool Implementation
- Define JSON schema for search_corpus tool
- Adapt existing FAISS search to tool executor interface
- Format search results for tool response
- Support configurable top_k parameter

### Story: CLI Integration
- Add `--agentic` flag to enable tool-calling mode
- Update REPL to support agentic conversation flow
- Show tool calls in output (optional verbose mode)
- Maintain backward compatibility with simple RAG mode

### Story: Documentation
- Update `how-to-local-agent.md` with agentic mode instructions
- Document tool calling behavior and limitations
- Add examples of multi-turn agentic conversations
- Update CLAUDE.md with architecture details

---

## Acceptance Criteria (Feature Level)

1. Local LLM can decide whether to search corpus or answer directly
2. Multi-turn tool calling works (LLM can search multiple times)
3. Existing simple RAG mode still works (backward compatible)
4. Works with Qwen 2.5 14B and 32B models
5. Max iterations prevents runaway loops
6. Tool call trace visible in verbose mode
7. Documentation updated

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Infinite tool loops | High | Hard limit on iterations (default: 5) |
| Model ignores tools | Medium | Test with Qwen 2.5; fallback to simple RAG |
| Slow multi-turn | Medium | Document latency expectations; optimize prompts |
| Context window exhaustion | Medium | Track tokens; truncate old turns if needed |
| Invalid tool arguments | Low | Validate before execution; return error as result |

---

## Future Enhancements (Out of Scope)

- Additional tools: `list_agents`, `get_document`, `summarize_chunk`
- Streaming tool calls
- Tool call caching
- Integration with MCP protocol (if Ollama adds support)

---

## References

- `scripts/local_rag.py` - Current simple RAG implementation
- `mcp_servers/corpus_search/server.py` - MCP server for Claude
- `how-to-local-agent.md` - User documentation
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
- Qwen 2.5 tool calling: https://qwen.readthedocs.io/
