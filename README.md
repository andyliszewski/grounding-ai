# Grounding

[![PyPI version](https://img.shields.io/pypi/v/grounding-ai.svg)](https://pypi.org/project/grounding-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/grounding-ai.svg)](https://pypi.org/project/grounding-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Local-first document corpus pipeline for grounded AI agents.**

Grounding converts PDF, EPUB, DOCX, and Markdown documents into a structured, searchable corpus with per-agent embedding indexes. Drop documents into staging, get chunked Markdown with provenance hashing, FAISS vector indexes, and agent-filtered search -- all running locally, no cloud APIs required.

## What It Does

```
Documents (PDF/EPUB/DOCX/MD)
    |
    v
[ Parse ] ── Unstructured / Marker
    |
    v
[ Chunk ] ── LangChain text splitters + YAML front matter
    |
    v
[ Hash ]  ── SHA-1 + SHA-256 + BLAKE3 provenance
    |
    v
[ Index ] ── FAISS embeddings, filtered per agent
    |
    v
[ Query ] ── Local RAG via Ollama with agentic tool calling
```

## Key Features

- **Deterministic pipeline** -- same inputs produce byte-identical outputs
- **Content provenance** -- SHA-1, SHA-256, and BLAKE3 hashing on every document and chunk
- **Agent-based corpus partitioning** -- YAML-defined agents filter the corpus by collection tags, each with their own FAISS embedding index
- **Persona system** -- agents have configurable communication styles, expertise areas, and greeting messages
- **Staging watcher** -- drop files into a folder, auto-ingest with embedding updates
- **Multi-machine ready** -- optional Syncthing-based architecture for dedicated ingestion servers
- **Fully local** -- no cloud APIs, no telemetry, your documents stay on your machine
- **Agentic RAG** -- local LLMs autonomously decide when to search the corpus via tool calling

## Why Grounding? (vs. LlamaIndex / Haystack / txtai)

Grounding is an opinionated pipeline, not a framework:

- **Per-agent FAISS partitioning as a first-class primitive.** Each agent YAML defines a corpus slice and gets its own index -- no single monolithic index where unrelated domains compete for top-k. Other frameworks treat this as a filter you bolt on at query time; here it's the core data model.
- **Determinism and provenance for citable output.** Every chunk carries SHA-1 / SHA-256 / BLAKE3 hashes, page ranges, and section headings; same inputs produce byte-identical chunks. Built for agents that must cite sources.
- **Local-first by construction.** No cloud APIs, no telemetry. Runs against Ollama or any OpenAI-compatible local server. Cloud isn't a default path -- it's not a path at all.
- **Opinionated fixed pipeline** (parse → chunk → hash → index → query). Less surface area than a framework, less to configure, less to break.

Use LlamaIndex / Haystack if you want a framework to assemble custom retrieval flows. Use grounding-ai if you want a citation-grade local RAG pipeline with per-agent separation, working today.

## Quick Start

### Install from PyPI

```bash
python3 -m venv venv   # Python 3.10-3.13 supported
source venv/bin/activate
pip install grounding-ai
```

Then grab the example configs and agents from the repo:

```bash
curl -O https://raw.githubusercontent.com/andyliszewski/grounding-ai/main/config.example.yaml
curl -O https://raw.githubusercontent.com/andyliszewski/grounding-ai/main/.mcp.example.json
mkdir -p agents && cd agents && \
  curl -O https://raw.githubusercontent.com/andyliszewski/grounding-ai/main/agents/examples/scientist.yaml && \
  cd ..
cp config.example.yaml config.yaml
cp .mcp.example.json .mcp.json
```

### Install from source (for development)

```bash
git clone https://github.com/andyliszewski/grounding-ai.git
cd grounding-ai
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
cp .mcp.example.json .mcp.json
cp agents/examples/*.yaml agents/
```

### First run (end-to-end)

```bash
# 1. Ingest some documents
grounding ./my-documents ./corpus --collections science

# 2. Generate embeddings for the scientist agent
grounding embeddings --agent scientist --corpus ./corpus

# 3. Query with a local LLM (requires Ollama running)
python scripts/local_rag.py --agent scientist -A
```

A typical session looks like this:

```
$ python scripts/local_rag.py --agent scientist -A
🔬 Scientist agent ready (3,142 chunks indexed across 8 collections)

> What does Kuhn mean by a paradigm shift?

[searching corpus... 5 chunks retrieved]

A paradigm shift, in Kuhn's framing, is a discontinuous change in the
fundamental assumptions of a scientific community [Source: Kuhn, The
Structure of Scientific Revolutions, corpus]. It happens when accumulated
anomalies can no longer be explained within the existing paradigm and a
new framework displaces the old one — not through gradual refinement but
through a gestalt-like reorientation.

[Derived] The process is social as much as epistemic: Kuhn emphasizes that
competing paradigms are often incommensurable, meaning proponents of each
literally see the world differently.
```

## Agent System

Agents are YAML files that define a persona and a corpus filter:

```yaml
name: scientist
description: Scientific research agent

persona:
  icon: "🔬"
  style: |
    You communicate like a rigorous scientist: analytical,
    evidence-based, and methodical.
  expertise:
    - Scientific method and experimental design
    - Biology and biochemistry
    - Physics fundamentals
  greeting: |
    I'm your scientific advisor. What would you like to investigate?

corpus_filter:
  collections:
    - science
    - biology
    - chemistry
    - physics
```

Each agent gets its own FAISS embedding index containing only documents matching its collections. See `agents/examples/` for starter templates.

### Creating Your Own Agents

1. **Define the agent.** Create a YAML file in `agents/`:

   ```yaml
   # agents/my-agent.yaml
   name: my-agent
   description: What this agent knows about

   persona:
     icon: "🎯"
     style: |
       How you want the agent to communicate.
     expertise:
       - Domain area 1
       - Domain area 2
     greeting: |
       Message shown when the agent activates.

   corpus_filter:
     collections:
       - collection-tag-1
       - collection-tag-2
   ```

2. **Ingest documents with matching collection tags.** Collections are kebab-case labels you assign when ingesting:

   ```bash
   grounding ./physics-textbooks ./corpus --collections physics
   grounding ./biology-papers ./corpus --collections biology,science
   ```

   A document can belong to multiple collections (comma-separated). An agent sees all documents whose collection tags overlap with its `corpus_filter.collections` list.

3. **Generate embeddings** for the agent:

   ```bash
   grounding embeddings --agent my-agent --corpus ./corpus
   ```

   This builds a FAISS index at `embeddings/my-agent/` containing only chunks from documents matching the agent's collection filter.

4. **Query** the agent:

   ```bash
   python scripts/local_rag.py --agent my-agent -A
   ```

### Where Do My Agent Files Live?

By default, `agents/*.yaml` is **gitignored** (only `agents/examples/` is tracked). This means YAMLs you create in `agents/` won't show up in `git status` and won't be committed to your fork. There are three common workflows depending on how you want to manage your agents:

#### Workflow A: Local-only (simplest, no version control)

Just create YAMLs in `agents/` and use them. Nothing extra to manage.

```bash
cp agents/examples/scientist.yaml agents/my-physicist.yaml
# Edit, then use immediately
grounding embeddings --agent my-physicist --corpus ./corpus
```

**Good for:** Trying things out, single machine, agents you don't need to back up.

#### Workflow B: Separate private agents repo (recommended for serious use)

Create your own private repo for agent definitions and point `AGENTS_DIR` at it. This is how the maintainer runs grounding -- agents are version-controlled and sync between machines via git.

```bash
# Create a private repo with this structure:
#   my-agents/
#   ├── agents/
#   │   ├── physicist.yaml
#   │   └── biologist.yaml
#   └── commands/         # Optional: Claude Code slash commands

# Clone it alongside grounding-ai
git clone git@github.com:youruser/my-agents.git ~/my-agents

# Point grounding at it
grounding embeddings --agent physicist --corpus ./corpus --agents-dir ~/my-agents/agents

# Or set the environment variable for the staging watcher
export AGENTS_DIR=~/my-agents/agents
```

**Good for:** Multi-machine setups, version-controlled agent definitions, keeping personal agents private while contributing back to grounding-ai.

#### Workflow C: Fork grounding-ai

Fork the repo and remove `agents/*.yaml` from `.gitignore`. Your agents become part of your fork.

```bash
# After forking
sed -i '' '/agents\/\*\.yaml/d' .gitignore   # remove the gitignore line
git add agents/ .gitignore
git commit -m "track personal agent definitions"
```

**Good for:** Single-repo workflow, public agent libraries, contributing agent templates back upstream.

### Organizing Collections

Collections are free-form tags -- there's no predefined list. Choose whatever makes sense for your domain:

```
staging/
├── physics/           # Collection: physics
├── biology/           # Collection: biology
├── game-theory/       # Collection: game-theory
└── machine-learning/  # Collection: machine-learning
```

One agent can span many collections (a "scientist" agent might include physics, biology, and chemistry). Multiple agents can share the same collections. The agent YAML is the only thing that defines which slices of the corpus each agent can search.

## Project Structure

```
grounding-ai/
├── grounding/              # Python package (the pipeline)
├── scripts/                # Watcher, local RAG, utilities
├── mcp_servers/            # MCP corpus search server
├── agents/
│   └── examples/           # Starter agent definitions
├── tests/                  # Test suite
├── config.example.yaml     # Configuration template
├── .mcp.example.json       # MCP server config template
└── staging/                # Drop documents here for ingestion
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust paths:

```yaml
paths:
  corpus: ./corpus
  embeddings: ./embeddings
  staging: ./staging
  agents: ./agents
  originals: ./originals
```

**Single machine (default):** All paths are relative, everything runs locally.

**Multi-machine:** Point paths at Syncthing-shared directories. A dedicated server runs the staging watcher and generates embeddings; workstations sync the corpus and query it. See `docs/multi-machine.md`.

## CLI Reference

```bash
# Ingest documents
grounding ./input-dir ./output-dir [options]
  --chunk-size 1200        # Characters per chunk (default: 1200)
  --chunk-overlap 150      # Overlap between chunks (default: 150)
  --parser marker          # Parser: unstructured or marker
  --ocr auto               # OCR: auto, on, or off
  --collections sci,math   # Collection tags (comma-separated)
  --dry-run                # Preview without writing
  --verbose                # Debug logging

# Agent management
grounding agents list --agents-dir ./agents
grounding agents show scientist --agents-dir ./agents

# Embedding generation
grounding embeddings --agent scientist --corpus ./corpus
grounding embeddings --agent scientist --corpus ./corpus --incremental
grounding embeddings --agent scientist --corpus ./corpus --check
```

## Staging Watcher (Auto-Ingest)

For continuous ingestion, run the staging watcher: drop a document into your staging folder and it's automatically parsed, chunked, hashed, moved to `originals/`, and (optionally) added to affected agents' embedding indexes.

### Single-machine setup

**Requirements:**
- Linux: `inotify-tools` (`sudo apt install inotify-tools`)
- macOS: `fswatch` (`brew install fswatch`) — the shipped script uses `inotifywait`; macOS users typically wrap it with `fswatch` or run the watcher inside a Linux VM/container

**Run manually:**

```bash
export STAGING_DIR=./staging
export CORPUS_DIR=./corpus
export ORIGINALS_DIR=./originals
export SKIPPED_DIR=./skipped
export AGENTS_DIR=./agents
export EMBEDDINGS_DIR=./embeddings
export AUTO_EMBEDDINGS=true
export LOG_FILE=./watcher.log

./scripts/staging-watcher.sh
```

Then drop documents into a collection subfolder:

```bash
mkdir -p staging/science
cp ~/Downloads/paper.pdf staging/science/
# Watcher logs show: parsing → chunking → hashing → embeddings update
```

**Processing rules:**

| Source location | Collection tag | Destination after processing |
|---|---|---|
| `staging/science/paper.pdf` | `science` | `corpus/<slug>/`, original → `originals/science/` |
| `staging/biology/book.epub` | `biology` | `corpus/<slug>/`, original → `originals/biology/` |
| Scanned PDF (no text yield) | — | moved to `skipped/<collection>/` |

When `AUTO_EMBEDDINGS=true`, every ingested document triggers incremental embedding updates for each agent whose `corpus_filter.collections` matches the document's collection.

### Run as a systemd service (Linux)

```bash
# Copy the sample unit file and edit the paths
cp scripts/grounding-watcher.service.example ~/.config/systemd/user/grounding-watcher.service
# Edit Environment= lines to point at your directories

systemctl --user daemon-reload
systemctl --user enable --now grounding-watcher
journalctl --user -u grounding-watcher -f    # follow logs
```

### Multi-machine setup

Point the watcher's paths at Syncthing-shared directories and run it on a dedicated ingestion server. Workstations sync corpus and embeddings and never run the watcher themselves. See [`docs/multi-machine.md`](docs/multi-machine.md).

## Querying Your Corpus

Once you have an agent with embeddings, there are two ways to query its corpus. Both run entirely locally -- no cloud APIs, no telemetry.

### Option 1: Local LLM via Ollama (recommended)

Ground a local LLM in your corpus using `scripts/local_rag.py`. The script loads your agent's persona, retrieves relevant chunks from its FAISS index, and feeds them to the LLM as context.

```bash
# Install Ollama and pull a model
brew install ollama          # macOS (or curl ... | sh on Linux)
ollama pull qwen2.5:14b      # Recommended for 32GB+ RAM
ollama serve

# Query your agent
python scripts/local_rag.py --agent scientist -A
```

The `-A` flag enables **agentic mode**, where the LLM autonomously decides when to search the corpus via tool calling. It can also perform multi-step reasoning, web searches, and file operations.

```bash
# Interactive REPL session
python scripts/local_rag.py --agent scientist -A

# Single query
python scripts/local_rag.py --agent scientist -A \
  --query "What are the thermodynamic limits of a Carnot engine?"

# With verbose output (see tool calls)
python scripts/local_rag.py --agent scientist -A -v
```

See [`how-to-local-agent.md`](how-to-local-agent.md) for the full guide -- model recommendations, REPL commands, troubleshooting, and tool reference.

### Option 2: MCP Server (for Claude Code, Cursor, etc.)

Grounding ships with an MCP (Model Context Protocol) server that exposes corpus search to any MCP-compatible client.

```bash
# 1. Install the MCP runtime into your venv (not a default dependency)
./venv/bin/pip install mcp

# 2. Copy the example and edit paths to match your setup
cp .mcp.example.json .mcp.json
```

Edit the three env vars in `.mcp.json`:

| Variable | Points at |
|----------|-----------|
| `CORPUS_DIR` | Root containing `_index.json` and `<slug>/chunks/` (e.g. `./corpus` or `~/Documents/Corpora/corpus`) |
| `EMBEDDINGS_DIR` | Root containing `<agent>/_embeddings.faiss` and `_chunk_map.json` |
| `AGENTS_DIR` | Directory of agent YAML files. Can point outside the repo if you keep agents in a separate repo. |

`CORPUS_DIR` must be the same root used when embeddings were generated -- chunk paths in the FAISS chunk map are stored relative to it.

`.mcp.json` is gitignored by default so machine-specific paths don't leak into commits. Restart your MCP client (e.g., Claude Code) to load the server. Once configured, your client can call `search_corpus` to query any agent's corpus directly from inside the chat interface.

### How Grounding Works

The "grounding" comes from three layers working together:

1. **Persona** -- the agent YAML's `persona.style`, `expertise`, and `greeting` shape how the LLM communicates
2. **Corpus filter** -- the agent's `corpus_filter.collections` restricts what documents the LLM can see
3. **Retrieval** -- relevant chunks from those documents are pulled in via FAISS similarity search and injected as context

The LLM never has access to your entire corpus at once. It sees the agent's persona prompt plus only the chunks most semantically relevant to the current question. This keeps responses focused, lets you scale to large corpora, and gives different agents different "knowledge" from the same underlying document set.

## Quality

Retrieval changes are gated by an evaluation harness that scores each agent's
FAISS index against a hand-curated fixture of query → expected-document pairs.
A GitHub Actions workflow runs the harness on every PR that touches retrieval
code and fails the check if any aggregate metric drops more than the configured
threshold relative to a committed baseline.

**Public CI gate = regression trip-wire, not a quality benchmark.** The mini
corpus is three synthetic documents (`alpha-paper`, `beta-study`, `gamma-notes`).
Metrics on it are saturated at `recall@5 = 1.000` / `citation_accuracy = 1.000`
by design — the gate exists to trip when a retrieval-code change drops them,
not to rank retrieval configurations. Sample runs are checked in:

- [Latest mini-corpus eval run](docs/eval/reports/mini-20260415-231110.md) — full per-item breakdown
- [All mini-corpus runs](docs/eval/reports/)

**Real quality measurement** runs against private-corpus fixtures that can't
be publicly distributed. That's where the rerank / hybrid flip decisions
(Stories 18.4 and 19.4) are made. See the
[three-repo layering](docs/eval/README.md#three-repo-layering) section for the
division of concerns: public repo guards retrieval-code regressions; private
runs guard corpus-quality regressions.

See [`docs/eval/README.md`](docs/eval/README.md) for the fixture schema, CLI
usage, CI gate details, and the baseline-refresh procedure.

## Requirements

- **Python 3.10 - 3.13** (3.14+ not yet supported due to `unstructured` compatibility)
- `poppler-utils` (system package for PDF text extraction)
- [Ollama](https://ollama.ai) (optional, for local RAG queries)

## Troubleshooting

**`grounding: command not found`** — activate the venv (`source venv/bin/activate`) or invoke directly: `./venv/bin/grounding`.

**`unstructured` install fails on Python 3.14** — `unstructured` pins `python<3.14`. Use Python 3.13: `python3.13 -m venv venv`.

**`pdftotext: command not found`** — install poppler: `sudo apt install poppler-utils` (Linux) or `brew install poppler` (macOS).

**PDFs are moved to `skipped/` instead of `corpus/`** — they're likely scanned (no extractable text). The watcher's yield threshold is 1000 chars per MB. To force OCR, run `grounding` directly with `--ocr on`.

**Agent search returns zero results** — check embeddings exist: `grounding embeddings --agent <name> --corpus ./corpus --check`. If stale, regenerate: `grounding embeddings --agent <name> --corpus ./corpus --incremental`.

**Watcher doesn't pick up files on macOS** — the shipped `staging-watcher.sh` uses `inotifywait` (Linux only). On macOS, run the watcher inside a Linux VM/container, or port it to `fswatch`.

**Ollama queries hang or time out** — confirm the model is pulled and `ollama serve` is running: `ollama list` and `curl http://localhost:11434/api/tags`.

## License

MIT
