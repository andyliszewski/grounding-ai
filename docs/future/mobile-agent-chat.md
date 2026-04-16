# Future Feature: Mobile Agent Chat

**Status:** Idea / Not Started
**Date:** 2026-01-31

## Concept

Access any agent (defined in `agents/`) from phone via Telegram bot. Processing happens on your server. Inspired by clawdbot/mawdbot.

## Architecture Options

### Option A: Local LLM (Preferred for Privacy/Cost)

```
Phone (Telegram)
    ↓
Telegram Bot (your server)
    ↓
Load agent YAML → system prompt
    ↓
sentence-transformers (local) → query FAISS
    ↓
Ollama (local LLM) with RAG context
    ↓
Response → Telegram
```

**Pros:** Free, private, offline-capable, self-contained
**Cons:** Lower quality than Claude for complex reasoning

### Option B: Claude API

```
Phone (Telegram)
    ↓
Telegram Bot (your server)
    ↓
Load agent YAML → system prompt
    ↓
Query FAISS embeddings for RAG
    ↓
Claude API call
    ↓
Response → Telegram
```

**Pros:** Higher quality responses
**Cons:** API costs, requires internet

## Key Components

### Already Have
- Agent YAMLs with personas (`persona.style`, `persona.expertise`, `persona.greeting`)
- FAISS indexes per agent (embeddings already built)
- `all-MiniLM-L6-v2` via sentence-transformers (for semantic search)
- your server server infrastructure

### Need to Build
- Telegram bot daemon (python-telegram-bot)
- Agent YAML → system prompt converter
- RAG query integration (load agent's FAISS, search, inject chunks)
- Ollama setup (if going local LLM route)

## Local LLM Model Recommendations

| RAM Available | Model | Quality |
|---------------|-------|---------|
| 8GB | Phi-3 (3.8B), Llama 3.2 (3B) | Decent |
| 16GB | Mistral 7B, Llama 3.2 (8B) | Good |
| 32GB+ | Qwen 2.5 14B, Llama 3.1 (8B Q8) | Very good |

## Before Starting

1. **Check your server specs** (RAM, CPU/GPU) to determine viable models
2. **Test Ollama** manually first: `ollama pull mistral` and play with it
3. **Decide on Telegram vs Signal vs Matrix** (Telegram easiest, Signal most private)

## Bot Features (Future)

- `/agent <name>` - switch active agent
- `/list` - show available agents
- Conversation history per chat
- Automatic RAG from agent's corpus filter
- Maybe: voice messages via Whisper

## References

- Ollama: https://ollama.ai
- python-telegram-bot: https://python-telegram-bot.org
- Existing agent YAMLs: `agents/*.yaml`
- Existing embeddings: `embeddings/<agent>/`
