# Reprocessing Pre-Epic-17 Corpus Documents

A guided, restartable workflow for upgrading docs ingested before
[Epic 17](epics/epic-17-page-and-section-citations.md) (page/section
citations) to the current ingestion pipeline. Each doc is swapped one at a
time so the corpus never has a hole: by the end every chunk that *can*
carry `page_start` / `page_end` / `section_heading` will.

## What it does

For every doc whose first chunk was created before **2026-04-14T00:00Z**
*and* whose source file is still findable under `originals/`, the
workflow:

1. Removes the doc's slug dir from `corpus/`.
2. Reingests the source via `grounding` (PDF/EPUB) or `ingest_docs.py`
   (MD/DOCX), preserving the original `collections` tags.
3. Runs `grounding embeddings --incremental` for every agent whose
   `corpus_filter.collections` intersect the doc's collections —
   tombstoning the old FAISS rows and appending the new ones (BM25 sidecar
   rebuilds the same way; see CLAUDE.md → "BM25 Sidecar (Epic 19.1)").
4. Updates the worklist JSON atomically so the run is fully restartable.

Docs whose original file is missing get logged once to
`reprocess-missing-originals.txt` and are otherwise left alone — their
old chunks stay in the corpus and remain searchable.

## How it coordinates with the staging watcher

Before processing a doc, `reprocess.sh` writes a `.reprocess-lock` file
into every staging collection dir the doc touches. The watcher honors
that lock at the top of `process_collection()` (see
`scripts/staging-watcher.sh`) and simply *defers* any files in that
collection until the lock is released. Files in *other* collections keep
flowing.

When the lock is released, `reprocess.sh` rename-renames any queued
files in the staging collection (`mv f f.requeued && mv f.requeued f`),
which fires an inotify `moved_to` event so the watcher picks them up.

If you'd rather keep things simple, just stop the watcher for the
duration:

```bash
systemctl --user stop grounding-watcher
# ... run reprocess.sh ...
systemctl --user start grounding-watcher
```

The watcher's on-startup `process_existing` will drain whatever piled up.

## Running it

### 1. Build the worklist (idempotent — re-run any time)

```bash
./venv/bin/python scripts/build_reprocess_worklist.py \
  --corpus    ~/Corpora/corpus \
  --originals ~/Corpora/originals \
  --agents-dir ~/my-agents/agents \
  --out reprocess-worklist.json \
  --missing-report reprocess-missing-originals.txt
```

The script prints bucket counts grouped by priority agent
(`mathematician`, `data-scientist`, `mechanical-engineer`, `ceo`,
`the-rest`). Entries are emitted in priority + slug order, so the driver
just iterates the list.

### 2. Dry-run to verify scoping

```bash
./scripts/reprocess.sh --worklist reprocess-worklist.json \
  --dry-run --limit 5 --priority mathematician
```

No filesystem writes; just logs what each step would do.

### 3. Real run — pilot first, then everything else

Pilot with a small sample to confirm one full round-trip:

```bash
./scripts/reprocess.sh --worklist reprocess-worklist.json \
  --limit 5 --priority mathematician
```

After it finishes, spot-check a reprocessed doc:

```bash
# Page numbers should appear in front-matter (or stay null if the parser
# can't surface pages — EPUB, fallback, etc.).
head -10 /path/to/corpus/<slug>/chunks/ch_0001.md

# The MCP corpus-search tool should now include a citation prefix like
#   [<slug>, p.42, §3.2 Bootstrap Methods]
# in retrieval results.
```

Then drain the priority buckets:

```bash
./scripts/reprocess.sh --worklist reprocess-worklist.json --priority mathematician
./scripts/reprocess.sh --worklist reprocess-worklist.json --priority data-scientist
./scripts/reprocess.sh --worklist reprocess-worklist.json --priority mechanical-engineer
./scripts/reprocess.sh --worklist reprocess-worklist.json --priority ceo
./scripts/reprocess.sh --worklist reprocess-worklist.json --priority the-rest
```

Or just let it run through everything in priority order:

```bash
./scripts/reprocess.sh --worklist reprocess-worklist.json
```

A typical Marker run is 30s–2min per PDF. At ~1,000 docs, plan for
several hours of wall time. Run inside `tmux` or `nohup` so a closed
terminal doesn't kill it.

### 4. Resuming after a kill / crash / reboot

Just re-invoke with the same `--worklist` path. Entries marked `done`
get skipped; entries marked `failed` also get skipped (rerun with
`--priority`/`--collection` to target retries explicitly, or hand-edit
their status back to `pending` in the JSON).

## CLI flags

| Flag | Default | Behavior |
|------|---------|----------|
| `--worklist FILE` | (required) | Worklist JSON produced by `build_reprocess_worklist.py` |
| `--dry-run` | off | Print what would happen; touch no files |
| `--limit N` | unlimited | Process at most N pending entries from the front |
| `--priority NAME` | (any) | Only entries in the given priority bucket |
| `--collection NAME` | (any) | Only entries whose collections list contains NAME |
| `--no-watcher-lock` | off | Skip writing `.reprocess-lock` (testing only) |

## Rolling back a single doc

The reingest is destructive (old slug dir is removed before the new one
is written). If a single doc needs to be rolled back, the easiest path is
to drop the original back into staging:

```bash
# Force watcher to skip its "already processed" check by removing the slug
rm -rf /path/to/corpus/<slug>
cp /path/to/originals/<collection>/<orig_name> /path/to/staging/<collection>/
```

The watcher picks it up and processes it the same way as a fresh
ingestion.

## Environment variables

Override any of these to match your install:

| Var | Default |
|-----|---------|
| `STAGING_DIR` | `~/staging` |
| `CORPUS_DIR` | `~/Corpora/corpus` |
| `ORIGINALS_DIR` | `~/Corpora/originals` |
| `AGENTS_DIR` | `~/my-agents/agents` |
| `EMBEDDINGS_DIR` | `~/Corpora/embeddings` |
| `GROUNDING_BIN` | `<repo>/venv/bin/grounding` |
| `PYTHON_BIN` | `<repo>/venv/bin/python` |
| `REPROCESS_LOG` | `<repo>/reprocess.log` |

## Known limitations

- **Lost originals stay old-format.** The 165-ish docs in
  `reprocess-missing-originals.txt` will keep their pre-Epic-17 chunks.
  Re-acquire the source PDF and drop it into staging to fix individually.
- **Some parsers still emit null pages.** EPUBs, Word docs, scanned
  PDFs, and PDFs that fall back from Marker to Unstructured will reingest
  successfully but their chunks will still have `page_start: null`. That
  isn't a bug in this workflow — it's the underlying parser. The MCP
  citation prefix degrades to `[<slug>]` for those, which is the
  intended behavior (see CLAUDE.md → "Retrieval Output Format").
- **Manifest churn.** `corpus/_index.json` is rewritten once per doc.
  Acceptable for one-off use; if you ever do this multiple times a
  week, batching is a future optimization.
