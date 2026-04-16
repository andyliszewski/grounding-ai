# Scripts

Operational scripts for Grounding deployment and automation.

## staging-watcher.sh

**Purpose:** Monitor a staging folder for incoming documents (PDF, EPUB, DOCX, MD) and automatically process them into the corpus.

**Deployment Target:** Your ingestion machine (can be the same machine or a dedicated server).

### Architecture Context

This script supports both single-machine and two-machine workflows:

**Single machine:** Watcher monitors `./staging/` and writes to `./corpus/`.

**Two machines (with Syncthing):**

| Folder     | Workstation   | Ingestion Server |
|------------|---------------|------------------|
| `staging/` | Send Only     | Receive Only     |
| `corpus/`  | Receive Only  | Send Only        |

### Configuration

| Variable        | Description                          | Default            |
|-----------------|--------------------------------------|--------------------|
| `STAGING_DIR`   | Root staging folder                  | `./staging`        |
| `CORPUS_DIR`    | Output corpus folder                 | `./corpus`         |
| `ORIGINALS_DIR` | Archive for processed source files   | `./originals`      |
| `SKIPPED_DIR`   | Unsupported/failed files             | `./skipped`        |
| `LOG_FILE`      | Log file path                        | `./watcher.log`    |
| `AUTO_EMBEDDINGS` | Auto-update embeddings after ingest | `false`            |
| `AGENTS_DIR`    | Path to agent YAML definitions       | `./agents`         |
| `EMBEDDINGS_DIR`| Path to embeddings output            | `./embeddings`     |

### Dependencies

- `inotifywait` (from `inotify-tools` on Linux) or `fswatch` (macOS) for filesystem monitoring
- `grounding` CLI installed and on PATH

### Usage

```bash
# Run in foreground (for testing)
./scripts/staging-watcher.sh

# Run with custom paths
STAGING_DIR=/path/to/staging CORPUS_DIR=/path/to/corpus ./scripts/staging-watcher.sh
```

### Systemd Service (Linux, recommended for servers)

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
Environment=STAGING_DIR=/path/to/staging
Environment=CORPUS_DIR=/path/to/corpus
Environment=ORIGINALS_DIR=/path/to/originals
Environment=AUTO_EMBEDDINGS=true
Environment=AGENTS_DIR=/path/to/agents
Environment=EMBEDDINGS_DIR=/path/to/embeddings

[Install]
WantedBy=default.target
```

Then enable:

```bash
systemctl --user daemon-reload
systemctl --user enable grounding-watcher
systemctl --user start grounding-watcher
```
