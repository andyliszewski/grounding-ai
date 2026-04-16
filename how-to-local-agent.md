# How to Use Local Agents

Query your corpus using a local LLM via Ollama and the `local_rag.py` script.

## Quick Start

**Assumes:** You are in the project root directory and Ollama is running.

```bash
# Basic agentic session (recommended)
python scripts/local_rag.py --agent scientist -A

# With verbose output (see tool calls)
python scripts/local_rag.py --agent scientist -A -v

# Longer, more detailed responses
python scripts/local_rag.py --agent scientist -A --max-tokens 4096

# Use larger model for complex reasoning
python scripts/local_rag.py --agent scientist -A -m qwen2.5:32b

# Full options example
python scripts/local_rag.py --agent scientist -A -v -m qwen2.5:14b --max-tokens 4096
```

Once loaded, type questions at the prompt. Type `quit` to exit, `clear` to reset conversation.

## Shell Aliases (Run from Anywhere)

Add aliases to your shell profile to run agents from any directory:

```bash
# Local RAG agent aliases (adjust path to your install location)
GROUNDING_DIR="$HOME/grounding-ai"
alias scientist="python $GROUNDING_DIR/scripts/local_rag.py --agent scientist -A"
alias coder="python $GROUNDING_DIR/scripts/local_rag.py --agent coder -A"

# Variant with larger model / more tokens
alias scientist-deep="python $GROUNDING_DIR/scripts/local_rag.py --agent scientist -A -m qwen2.5:32b --max-tokens 4096"
```

Then reload your shell: `source ~/.zshrc` (or `~/.bashrc`).

## Prerequisites

1. **Ollama** installed and running (`ollama list` to check, `ollama serve` to start)
2. **Embeddings** generated for at least one agent (check `./embeddings/`)

## Recommended Models

| Hardware | Purpose | Model | Size | Command |
|----------|---------|-------|------|---------|
| Apple Silicon (32GB+) | Everyday | qwen2.5:14b | ~9GB | `--model qwen2.5:14b` (default) |
| Apple Silicon (32GB+) | Complex reasoning | qwen2.5:32b | ~19GB | `--model qwen2.5:32b` |
| 16GB RAM / CPU-only | Everyday | qwen2.5:7b | ~4.5GB | `--model qwen2.5:7b` |
| 8GB RAM / CPU-only | Lightweight | llama3.2:3b | ~2GB | `--model llama3.2:3b` |

## Setup

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download a Model

```bash
# Pick one based on your hardware
ollama pull qwen2.5:14b    # Recommended for 32GB+ machines
ollama pull qwen2.5:7b     # Good for 16GB machines
ollama pull llama3.2:3b    # Lightweight option
```

### 3. Verify Ollama is Running

```bash
ollama list
```

If not running, start it:
```bash
ollama serve
```

## Usage Examples

### Interactive Session (REPL)

```bash
# Start interactive session with an agent
python scripts/local_rag.py --agent scientist --agentic

# With verbose output (shows tool calls)
python scripts/local_rag.py --agent scientist --agentic --verbose

# Use larger model
python scripts/local_rag.py --agent scientist --agentic --model qwen2.5:32b
```

**REPL Commands:**
- Type questions normally
- `sources` - toggle source citations on/off
- `clear` - reset conversation (start fresh context)
- `quit` or `q` - exit

**Conversation Memory (Agentic Mode):**
In agentic mode, the agent remembers your conversation. You can ask follow-up questions like "what do you think of that?" and it will understand the context. Use `clear` to start fresh.

### Single Query

```bash
python scripts/local_rag.py --agent scientist --agentic \
  --query "What are the thermodynamic limits of a Carnot engine?"
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--agent`, `-a` | Agent name (required) | - |
| `--agentic`, `-A` | Enable agentic mode (LLM decides when to search) | false |
| `--model`, `-m` | Ollama model name | qwen2.5:14b |
| `--query`, `-q` | Single query (omit for REPL) | - |
| `--top-k`, `-k` | Number of chunks to retrieve | 5 |
| `--verbose`, `-v` | Show tool calls | false |
| `--max-iterations` | Max tool-calling rounds | 5 |
| `--max-tokens` | Max tokens in response (for longer answers) | 2048 |
| `--no-sources` | Hide source citations | false |
| `--corpus`, `-c` | Path to corpus | ./corpus |
| `--embeddings`, `-e` | Path to embeddings | ./embeddings |

## Simple RAG vs Agentic Mode

| Aspect | Simple RAG | Agentic Mode (`-A`) |
|--------|------------|---------------------|
| Search behavior | Always searches first | LLM decides when to search |
| Multi-turn | No | Yes (can search multiple times) |
| Best for | Fact-heavy queries | Complex reasoning, conversation |

**Recommendation:** Use agentic mode (`-A`) by default. It's more natural and the LLM will search when needed.

## Available Tools (Agentic Mode)

In agentic mode, the LLM can call these tools:

| Tool | Description | Example Use |
|------|-------------|-------------|
| `search_corpus` | Search your curated knowledge base | "What does my corpus say about X?" |
| `web_search` | Search the web via DuckDuckGo | "Find recent news about topic" |
| `web_fetch` | Fetch content from a URL | "Summarize this article: https://..." |
| `read_file` | Read local files | "Read /path/to/document.md" |
| `glob` | Find files by pattern | "Find all .py files in this project" |
| `grep` | Search text in files | "Find all TODO comments" |
| `bash` | Execute shell commands | "Run `git status`" |
| `write_file` | Create/overwrite files | "Create a summary file" |
| `edit_file` | Make targeted edits | "Replace X with Y in config" |
| `notebook_edit` | Edit Jupyter notebooks | "Update cell 3" |

**Note:** When asking the agent to "search the web", "look something up online", or "find current news", it should use the `web_search` tool. If it doesn't, try being more explicit: "Use the web_search tool to find..."

### Example: Agentic Mode

```
$ python scripts/local_rag.py --agent scientist --agentic -v \
  --query "Compare Carnot and Stirling engine efficiency"

[Agentic] Iterations: 3
[Agentic] Tool calls made: 2
  - search_corpus({"query": "Carnot engine efficiency thermodynamics"})
  - search_corpus({"query": "Stirling engine efficiency comparison"})

Based on the physics texts in the corpus:

**Carnot Engine:**
The Carnot cycle represents the theoretical maximum efficiency...
```

### Example: LLM Answers Without Searching

```
$ python scripts/local_rag.py --agent scientist --agentic -v \
  --query "What is 2 + 2?"

[Agentic] Iterations: 1
[Agentic] Tool calls made: 0

The answer is 4.
```

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
ollama list

# Start if needed
ollama serve
```

### "FAISS index not found"

The agent doesn't have embeddings. Generate them:

```bash
grounding embeddings --agent <agent-name> --corpus ./corpus
```

### Slow first query

The first query loads the embedding model (~100MB). Subsequent queries are faster.

### Slow generation on CPU-only machines

CPU inference is inherently slower. Tips:
- Use smaller models (3B-7B)
- Reduce `--top-k` to send less context
- Close other applications to free RAM

### LLM never calls tools in agentic mode

Some models don't handle tool calling well via Ollama's API.

**Symptoms:**
- Model says "Let me search..." but doesn't actually search
- Model says "I don't have access to tools" despite having them
- Response describes what it would do instead of doing it

**Solutions:**

1. **Use verbose mode** to debug: `--verbose` or `-v`
   ```bash
   python scripts/local_rag.py --agent scientist -A -v --query "Search the web for recent news"
   ```
   This shows what tools are being sent and whether the LLM is calling them.

2. **Try a different model.** Best tool-calling support on Ollama:
   - Qwen 2.5 (14B, 32B) - generally good
   - Llama 3.1+ variants with function calling
   - Mistral models with tool support

3. **Be explicit in your request:**
   - Instead of: "Look up recent events"
   - Try: "Use the web_search tool to search for recent events"

4. **Fall back to simple RAG mode** (omit `-A`) - always searches your corpus first

### Agentic mode is slow

Each tool call requires a round-trip to the LLM:
- Simple RAG: 1 LLM call (~2-5 seconds)
- Agentic (1 tool call): 2 LLM calls (~4-10 seconds)

Consider simple RAG mode (omit `-A`) for time-sensitive queries.

### Responses are too short or shallow

Increase the max tokens setting:
```bash
python scripts/local_rag.py --agent scientist -A --max-tokens 4096
```

The default is 2048 tokens. Increase to 4096 or 8192 for more detailed responses.

### Agent doesn't remember previous questions

The agent maintains conversation memory in agentic mode. If it seems to forget:
1. Make sure you're in agentic mode (`-A`)
2. Use `clear` to reset if the context gets confused
3. Be explicit about what you're referring to ("Based on what you just found...")
