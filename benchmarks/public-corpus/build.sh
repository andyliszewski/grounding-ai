#!/usr/bin/env bash
# Rebuild the public-domain benchmark corpus end to end:
#
#   1. fetch.py      download every manifest PDF into $ROOT/originals and verify SHA-256
#   2. grounding     ingest each collection group into $ROOT/corpus (Markdown + chunks)
#   3. grounding embeddings   FAISS index + BM25 sidecar for the benchmark agent
#
# Usage:
#   benchmarks/public-corpus/build.sh [--record] [--clean] [--skip-fetch] [--skip-embed]
#
#   --record      first run only: let fetch.py write hashes, sizes and page counts into
#                 manifest.yaml (later runs verify against them and fail on a mismatch)
#   --clean       remove $ROOT/corpus, $ROOT/embeddings/<agent> and $ROOT/work first
#   --skip-fetch  do not touch the network (originals must already be present and verified)
#   --skip-embed  stop after ingestion
#
# Environment:
#   CORPORA_PUBLIC   root for originals/, corpus/, embeddings/ (default ~/Documents/Corpora-public)
#   GROUNDING, PY    the grounding CLI and python to use (default: this repo's ./venv)
#
# Ingestion parameters are pinned below to the repo defaults so that a rebuild on another
# machine is byte-comparable (chunk boundaries depend on chunk_size, chunk_overlap and
# min_chunk_size; doc_ids depend only on the PDF bytes, which the manifest hashes pin).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$REPO/venv/bin/python}"
GROUNDING="${GROUNDING:-$REPO/venv/bin/grounding}"
ROOT="${CORPORA_PUBLIC:-$HOME/Documents/Corpora-public}"

AGENT="implant-eng-public"
MANIFEST="$HERE/manifest.yaml"
AGENTS_DIR="$HERE/agents"
ORIGINALS="$ROOT/originals"
CORPUS="$ROOT/corpus"
EMBED="$ROOT/embeddings/$AGENT"
WORK="$ROOT/work"

# Pinned ingestion parameters (repo defaults, stated explicitly).
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
MIN_CHUNK_SIZE=200
PARSER=unstructured
OCR=auto

RECORD=0; CLEAN=0; SKIP_FETCH=0; SKIP_EMBED=0
for arg in "$@"; do
  case "$arg" in
    --record) RECORD=1 ;;
    --clean) CLEAN=1 ;;
    --skip-fetch) SKIP_FETCH=1 ;;
    --skip-embed) SKIP_EMBED=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

for tool in "$PY" "$GROUNDING"; do
  [ -x "$tool" ] || { echo "ERROR: not executable: $tool (create the venv first, see CLAUDE.md)" >&2; exit 2; }
done
command -v pdftotext >/dev/null || echo "WARNING: pdftotext (poppler) not on PATH; every PDF will take the slow unstructured path" >&2

t0=$(date +%s)
echo "== public-corpus build  root=$ROOT  agent=$AGENT"
echo "   chunk_size=$CHUNK_SIZE chunk_overlap=$CHUNK_OVERLAP min_chunk_size=$MIN_CHUNK_SIZE parser=$PARSER ocr=$OCR"

if [ "$CLEAN" = 1 ]; then
  echo "== clean: removing $CORPUS $EMBED $WORK"
  rm -rf "$CORPUS" "$EMBED" "$WORK"
fi
mkdir -p "$ORIGINALS" "$CORPUS" "$WORK" "$(dirname "$EMBED")"

# ---------------------------------------------------------------- 1. fetch
if [ "$SKIP_FETCH" = 1 ]; then
  echo "== fetch: skipped (--skip-fetch); verifying files on disk"
  "$PY" "$HERE/fetch.py" --manifest "$MANIFEST" --dest "$ORIGINALS" --verify-only || { echo "ERROR: verification failed" >&2; exit 1; }
else
  echo "== fetch"
  fetch_args=()
  [ "$RECORD" = 1 ] && fetch_args+=(--record)
  "$PY" "$HERE/fetch.py" --manifest "$MANIFEST" --dest "$ORIGINALS" "${fetch_args[@]}" || { echo "ERROR: fetch failed" >&2; exit 1; }
fi

# ---------------------------------------------------------------- 2. ingest
# One grounding run per distinct collection set. The input directory for a group holds
# symlinks named <id>.pdf, so the corpus slug equals the manifest id.
echo "== ingest"
rm -rf "$WORK"; mkdir -p "$WORK"
groups=$("$PY" - "$MANIFEST" "$ORIGINALS" "$WORK" <<'PYEOF'
import os, sys, yaml
manifest, originals, work = sys.argv[1:4]
docs = yaml.safe_load(open(manifest))["documents"]
groups = {}
for d in docs:
    cols = sorted(d["collections"])
    key = "+".join(cols)
    src = os.path.join(originals, d["id"] + ".pdf")
    if not os.path.exists(src):
        sys.exit(f"missing original: {src}")
    gdir = os.path.join(work, key)
    os.makedirs(gdir, exist_ok=True)
    dst = os.path.join(gdir, d["id"] + ".pdf")
    if not os.path.lexists(dst):
        os.symlink(src, dst)
    groups[key] = ",".join(cols)
for key, cols in sorted(groups.items()):
    print(f"{key}\t{cols}")
PYEOF
) || { echo "ERROR: could not build ingestion groups" >&2; exit 1; }

ingest_failures=0
while IFS=$'\t' read -r key cols; do
  [ -n "$key" ] || continue
  n=$(ls "$WORK/$key"/*.pdf | wc -l | tr -d ' ')
  echo "-- group $key ($n files) --collections $cols"
  "$GROUNDING" "$WORK/$key" "$CORPUS" \
      --collections "$cols" \
      --chunk-size "$CHUNK_SIZE" --chunk-overlap "$CHUNK_OVERLAP" --min-chunk-size "$MIN_CHUNK_SIZE" \
      --parser "$PARSER" --ocr "$OCR"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "WARNING: grounding exited $rc for group $key (see log above); continuing" >&2
    ingest_failures=$((ingest_failures + 1))
  fi
done <<< "$groups"

# ---------------------------------------------------------------- 3. embeddings
if [ "$SKIP_EMBED" = 1 ]; then
  echo "== embeddings: skipped (--skip-embed)"
else
  echo "== embeddings -> $EMBED"
  "$GROUNDING" embeddings --agent "$AGENT" --agents-dir "$AGENTS_DIR" --corpus "$CORPUS" --out "$EMBED" \
    || { echo "ERROR: embeddings failed" >&2; exit 1; }
  for f in _embeddings.faiss _chunk_map.json _bm25.pkl _bm25_map.json; do
    if [ -s "$EMBED/$f" ]; then
      echo "   ok  $f ($(du -h "$EMBED/$f" | cut -f1))"
    else
      echo "ERROR: expected $EMBED/$f (the BM25 sidecar is required for the hybrid conditions)" >&2
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------- summary
"$PY" - "$CORPUS/_index.json" "$MANIFEST" <<'PYEOF'
import json, sys, yaml
index = json.load(open(sys.argv[1]))
want = {d["id"] for d in yaml.safe_load(open(sys.argv[2]))["documents"]}
docs = index.get("docs", [])
have = {d["slug"] for d in docs}
print(f"== corpus: {len(docs)} documents, {sum(d.get('chunk_count', 0) for d in docs)} chunks")
missing = sorted(want - have)
if missing:
    print("   MISSING from corpus:", ", ".join(missing))
extra = sorted(have - want)
if extra:
    print("   not in manifest:", ", ".join(extra))
PYEOF

echo "== done in $(( $(date +%s) - t0 ))s; ingest groups with failures: $ingest_failures"
[ "$ingest_failures" -eq 0 ]
