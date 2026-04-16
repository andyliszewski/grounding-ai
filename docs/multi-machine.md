# Multi-Machine Setup

Grounding supports an optional two-machine architecture where a dedicated ingestion server processes documents and generates embeddings, while workstations sync the results and query the corpus.

## Architecture

```
  Workstation (macOS/Linux)              Ingestion Server (Linux)
  ┌─────────────────────────┐            ┌─────────────────────────┐
  │                         │            │                         │
  │  Drop PDFs into         │  Syncthing │  staging-watcher.sh     │
  │  staging/ folder   ────────────────► │  detects new files      │
  │                         │            │       │                 │
  │                         │            │       ▼                 │
  │                         │            │  grounding CLI          │
  │                         │  Syncthing │  processes → corpus/    │
  │  corpus/ + embeddings/ ◄──────────── │  generates embeddings/  │
  │                         │            │                         │
  │  Query with local LLM   │            │  originals/ (archive)   │
  │  or MCP server          │            │  skipped/ (failures)    │
  └─────────────────────────┘            └─────────────────────────┘
```

## When to Use This

**Single machine (default):** Everything runs locally. No sync needed. Use this if you have one machine with enough resources.

**Two machines:** Use this when:
- You want a dedicated always-on server for document ingestion
- Your workstation doesn't have the resources for parsing + embedding generation
- You want ingestion to happen in the background without impacting your workflow

## Setup

### 1. Install Grounding on Both Machines

On each machine:

```bash
git clone https://github.com/andyliszewski/grounding-ai.git
cd grounding-ai
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Install Syncthing

**macOS:**
```bash
brew install syncthing
brew services start syncthing
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install syncthing
systemctl --user enable syncthing
systemctl --user start syncthing
```

Access the Syncthing web UI at `http://localhost:8384` on each machine. Pair the two devices using their device IDs.

### 3. Create Shared Directories

On the **ingestion server**, create the data directories:

```bash
mkdir -p ~/Corpora/{staging,corpus,embeddings,originals,skipped}
```

On the **workstation**, create matching directories:

```bash
mkdir -p ~/Documents/{staging,Corpora/corpus,Corpora/embeddings}
```

### 4. Configure Syncthing Shares

Set up these shared folders in the Syncthing web UI:

| Folder | Workstation | Server | Direction |
|--------|-------------|--------|-----------|
| `staging/` | Send Only | Receive Only | Workstation -> Server |
| `corpus/` | Receive Only | Send Only | Server -> Workstation |
| `embeddings/` | Receive Only | Send Only | Server -> Workstation |

**staging/** flows from workstation to server (you drop files on your workstation, the server picks them up).

**corpus/** and **embeddings/** flow from server to workstation (the server generates them, your workstation receives them for querying).

### 5. Configure the Ingestion Server

Update `config.yaml` on the server to point at the shared directories:

```yaml
paths:
  corpus: ~/Corpora/corpus
  embeddings: ~/Corpora/embeddings
  staging: ~/Corpora/staging
  agents: ~/path/to/agents       # Your agent YAML definitions
  originals: ~/Corpora/originals
  skipped: ~/Corpora/skipped
```

### 6. Set Up the Staging Watcher Service

Create a systemd user service on the ingestion server:

```bash
mkdir -p ~/.config/systemd/user
```

Create `~/.config/systemd/user/grounding-watcher.service`:

```ini
[Unit]
Description=Grounding Staging Watcher
After=network.target

[Service]
Type=simple
ExecStart=/path/to/grounding-ai/scripts/staging-watcher.sh
Restart=always
RestartSec=10

# Core paths
Environment=STAGING_DIR=/home/user/Corpora/staging
Environment=CORPUS_DIR=/home/user/Corpora/corpus
Environment=ORIGINALS_DIR=/home/user/Corpora/originals
Environment=SKIPPED_DIR=/home/user/Corpora/skipped
Environment=LOG_FILE=/home/user/Corpora/watcher.log

# Embedding auto-generation
Environment=AUTO_EMBEDDINGS=true
Environment=AGENTS_DIR=/home/user/path/to/agents
Environment=EMBEDDINGS_DIR=/home/user/Corpora/embeddings

# Git sync (pulls agent definitions before processing)
Environment=GIT_PULL_ENABLED=true

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable grounding-watcher
systemctl --user start grounding-watcher

# Verify
systemctl --user status grounding-watcher
journalctl --user -u grounding-watcher -f
```

### 7. Configure the Workstation

Update `config.yaml` on the workstation to point at the synced directories:

```yaml
paths:
  corpus: ~/Documents/Corpora/corpus
  embeddings: ~/Documents/Corpora/embeddings
  staging: ~/Documents/staging
  agents: ./agents
```

The workstation only needs `corpus/` and `embeddings/` for querying. It doesn't run the watcher.

## Workflow

1. **Drop documents** into `~/Documents/staging/<collection>/` on your workstation
   ```
   staging/
   ├── science/
   │   └── thermodynamics-textbook.pdf
   ├── economics/
   │   └── game-theory-intro.pdf
   └── math/
       └── linear-algebra-cheat-sheet.pdf
   ```

2. **Syncthing syncs** the file to the server's staging directory

3. **Watcher detects** the new file and:
   - Determines format (PDF, EPUB, DOCX, MD)
   - Checks if PDF is text-extractable (skips scanned PDFs)
   - Runs `grounding` to parse, chunk, and index
   - Moves the original to `originals/<collection>/`
   - If `AUTO_EMBEDDINGS=true`, updates embeddings for matching agents

4. **Syncthing syncs** the new corpus entries and embeddings back to your workstation

5. **Query** on your workstation using local RAG or MCP:
   ```bash
   python scripts/local_rag.py --agent scientist -A
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STAGING_DIR` | (required) | Root staging folder |
| `CORPUS_DIR` | (required) | Output corpus folder |
| `ORIGINALS_DIR` | (required) | Archive for processed source files |
| `SKIPPED_DIR` | (required) | Unsupported/failed files |
| `LOG_FILE` | `./watcher.log` | Log file path |
| `AUTO_EMBEDDINGS` | `false` | Auto-update embeddings after ingestion |
| `AGENTS_DIR` | (none) | Path to agent YAML definitions |
| `EMBEDDINGS_DIR` | (none) | Path to embeddings output |
| `LOCK_TIMEOUT` | `3600` | Seconds before stale embedding lock is auto-removed |
| `GIT_PULL_ENABLED` | `true` | Pull latest git changes before processing |

## Supported Formats

| Format | Processing | Result |
|--------|-----------|--------|
| PDF (text-based) | `grounding --ocr off` | corpus/ |
| PDF (scanned) | Detected via pdftotext yield | skipped/ |
| EPUB | `grounding` | corpus/ |
| Markdown (.md) | `ingest_docs.py` | corpus/ |
| Word (.docx) | `ingest_docs.py` | corpus/ |

Scanned PDF detection: if `pdftotext` extracts fewer than 1000 characters per MB, the PDF is treated as scanned and moved to `skipped/`.

## Troubleshooting

### Files not being picked up

Check that the watcher is running:
```bash
systemctl --user status grounding-watcher
```

Check the logs:
```bash
journalctl --user -u grounding-watcher -f
```

### Embeddings not updating

Verify `AUTO_EMBEDDINGS=true` and that `AGENTS_DIR` points to a directory containing agent YAML files with `corpus_filter.collections` matching the document's collection tag.

### Syncthing conflicts

Syncthing is configured as one-directional per folder (Send Only / Receive Only), so conflicts should not occur. If you see `.sync-conflict` files, check that the folder directions are set correctly.

### Lock file stuck

If embedding generation was interrupted, a stale lock file may remain. It auto-expires after `LOCK_TIMEOUT` seconds (default: 1 hour). To clear manually:
```bash
rm $EMBEDDINGS_DIR/_embeddings.lock
```
