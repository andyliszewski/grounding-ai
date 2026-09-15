#!/usr/bin/env python3
"""Apply rename-preview.csv to originals/ and cascade through corpus metadata.

Reads rename-preview.csv. For each row with status=auto where current != proposed:
  1. mv  <originals>/<collection>/<current> -> <originals>/<collection>/<proposed>
  2. For every corpus dir whose meta.yaml orig_name matches <current>:
       - rewrite orig_name in meta.yaml
       - rewrite source in doc.md front matter
       - rewrite source in every chunks/ch_*.md front matter
  3. Rewrite orig_name entries in _index.json
  4. Regenerate reprocess-worklist.json

Rows with status=review or collision are skipped (left for human review).
Rows with status=clean are no-ops by definition (current == proposed).

Idempotent: if <current> is missing but <proposed> exists, the rename is
treated as already-applied and metadata is still cascaded.

Writes applied-rename-log.csv as an audit trail.

Usage:
  python3 scripts/apply_rename.py \\
      --preview rename-preview.csv \\
      --originals ~/Corpora/originals \\
      --corpus    ~/Corpora/corpus \\
      --log       applied-rename-log.csv \\
      [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Match `source: <value>` (or `orig_name: <value>`) plus YAML continuation lines.
# Continuation lines start with whitespace; block ends at the next non-indented
# line (real key) or end-of-stream.
def _build_field_regex(field: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(field)}:[ \t][^\n]*(?:\n[ \t]+[^\n]*)*\n",
        re.MULTILINE,
    )

SOURCE_BLOCK_RE = _build_field_regex("source")
ORIG_NAME_BLOCK_RE = _build_field_regex("orig_name")


def _extract_value(block: str, field: str) -> str:
    """From a matched block 'field: line1\\n  line2\\n', return joined value."""
    body = block[len(field) + 1:]  # drop "field:"
    body = body.rstrip("\n")
    # Join continuation lines with a single space.
    lines = [ln.strip() for ln in body.split("\n")]
    return " ".join(p for p in lines if p)


def rewrite_field(text: str, field: str, old_to_new: dict[str, str]) -> tuple[str, int]:
    """Replace `field: <old>` with `field: <new>` if old is in map. Returns (new_text, n_changes)."""
    regex = SOURCE_BLOCK_RE if field == "source" else ORIG_NAME_BLOCK_RE
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        value = _extract_value(m.group(0), field)
        if value in old_to_new:
            n += 1
            return f"{field}: {old_to_new[value]}\n"
        return m.group(0)

    new_text = regex.sub(repl, text)
    return new_text, n


def atomic_write(path: Path, content: str) -> None:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try:
            os.unlink(tmp.name)
        finally:
            raise


def log_row(writer: csv.writer, *cols: str) -> None:
    writer.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), *cols])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preview", required=True, type=Path)
    p.add_argument("--originals", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--index", type=Path, help="Path to _index.json (defaults to <corpus>/_index.json)")
    p.add_argument("--log", type=Path, default=Path("applied-rename-log.csv"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.preview.is_file():
        sys.exit(f"error: preview CSV not found: {args.preview}")
    if not args.originals.is_dir():
        sys.exit(f"error: originals dir not found: {args.originals}")
    if not args.corpus.is_dir():
        sys.exit(f"error: corpus dir not found: {args.corpus}")
    index_path = args.index or (args.corpus / "_index.json")

    # ---- Load + validate the preview CSV --------------------------------
    rename_rows: list[dict] = []
    skipped_blocked = 0
    skipped_clean = 0
    with args.preview.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("review", "collision", "skip"):
                skipped_blocked += 1
                continue
            if r["current_name"] == r["proposed_name"]:
                skipped_clean += 1
                continue
            rename_rows.append(r)

    # Detect inconsistent cross-collection mappings (same current_name → different proposed_names).
    by_current: dict[str, set[str]] = {}
    for r in rename_rows:
        by_current.setdefault(r["current_name"], set()).add(r["proposed_name"])
    inconsistent = [k for k, v in by_current.items() if len(v) > 1]
    if inconsistent:
        sys.exit(
            f"error: same current_name maps to >1 proposed_name (would corrupt metadata):\n  "
            + "\n  ".join(inconsistent)
        )

    # Build the metadata cascade map (orig_name old → new). One entry per
    # unique current_name regardless of collection.
    old_to_new: dict[str, str] = {r["current_name"]: r["proposed_name"] for r in rename_rows}

    print(f"loaded preview: {args.preview}")
    print(f"  renames to apply: {len(rename_rows)}")
    print(f"  no-op (clean):    {skipped_clean}")
    print(f"  skipped (blocked):{skipped_blocked}")
    print(f"  unique old->new entries: {len(old_to_new)}")
    print(f"  dry-run: {args.dry_run}")
    print()

    # ---- Audit log --------------------------------------------------------
    log_f = open(args.log, "w", newline="", encoding="utf-8") if not args.dry_run else None
    log_w = csv.writer(log_f) if log_f else None
    if log_w:
        log_w.writerow(["timestamp", "stage", "collection", "old_name", "new_name", "result"])

    # ---- Step 1: rename files in originals/ ------------------------------
    print(f"[1/5] renaming files in {args.originals}")
    n_renamed = 0
    n_already = 0
    n_missing = 0
    for r in rename_rows:
        coll_dir = args.originals / r["collection"]
        old_path = coll_dir / r["current_name"]
        new_path = coll_dir / r["proposed_name"]
        if new_path.exists() and not old_path.exists():
            n_already += 1
            if log_w: log_row(log_w, "rename", r["collection"], r["current_name"], r["proposed_name"], "already-renamed")
            continue
        if not old_path.exists():
            n_missing += 1
            if log_w: log_row(log_w, "rename", r["collection"], r["current_name"], r["proposed_name"], "missing-source")
            continue
        if new_path.exists():
            # Both exist and not idempotent — bail loudly.
            sys.exit(f"error: both old and new exist for {r['collection']}/{r['current_name']} -> {r['proposed_name']}")
        if not args.dry_run:
            old_path.rename(new_path)
        n_renamed += 1
        if log_w: log_row(log_w, "rename", r["collection"], r["current_name"], r["proposed_name"], "ok")
    print(f"      renamed: {n_renamed}  already-renamed: {n_already}  missing: {n_missing}")

    # ---- Step 2: cascade through corpus metadata ------------------------
    # For each corpus slug dir: if its meta.yaml orig_name is in old_to_new,
    # rewrite meta.yaml + doc.md + every chunk file.
    print(f"[2/5] cascading metadata through {args.corpus}")
    n_dirs_touched = 0
    n_meta = 0
    n_doc = 0
    n_chunks = 0
    n_chunk_dirs_scanned = 0

    for slug_dir in sorted(args.corpus.iterdir()):
        if not slug_dir.is_dir():
            continue
        meta_path = slug_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        meta_text = meta_path.read_text(encoding="utf-8")
        new_meta, mhits = rewrite_field(meta_text, "orig_name", old_to_new)
        if mhits == 0:
            # Also try source field in case meta uses both
            new_meta2, src_hits = rewrite_field(new_meta, "source", old_to_new)
            if src_hits == 0:
                continue
            new_meta = new_meta2
            mhits = src_hits

        n_dirs_touched += 1
        if not args.dry_run:
            atomic_write(meta_path, new_meta)
        n_meta += 1
        if log_w: log_row(log_w, "meta", slug_dir.name, "", "", "updated")

        # doc.md
        doc_path = slug_dir / "doc.md"
        if doc_path.exists():
            doc_text = doc_path.read_text(encoding="utf-8")
            new_doc, dhits = rewrite_field(doc_text, "source", old_to_new)
            if dhits > 0:
                if not args.dry_run:
                    atomic_write(doc_path, new_doc)
                n_doc += 1
                if log_w: log_row(log_w, "doc", slug_dir.name, "", "", f"updated:{dhits}")

        # chunks/ch_*.md
        chunks_dir = slug_dir / "chunks"
        if not chunks_dir.is_dir():
            continue
        n_chunk_dirs_scanned += 1
        chunk_files = sorted(chunks_dir.glob("ch_*.md"))
        n_in_dir = 0
        for cf in chunk_files:
            ctext = cf.read_text(encoding="utf-8")
            new_ctext, chits = rewrite_field(ctext, "source", old_to_new)
            if chits > 0:
                if not args.dry_run:
                    atomic_write(cf, new_ctext)
                n_chunks += 1
                n_in_dir += 1
        if log_w and n_in_dir > 0:
            log_row(log_w, "chunks", slug_dir.name, "", "", f"updated:{n_in_dir}")

    print(f"      affected slug dirs: {n_dirs_touched}")
    print(f"      meta.yaml updates:  {n_meta}")
    print(f"      doc.md updates:     {n_doc}")
    print(f"      chunk updates:      {n_chunks}  (across {n_chunk_dirs_scanned} dirs scanned)")

    # ---- Step 3: _index.json --------------------------------------------
    print(f"[3/5] updating {index_path}")
    if not index_path.exists():
        print("      no _index.json found; skipping")
    else:
        idx_text = index_path.read_text(encoding="utf-8")
        idx = json.loads(idx_text)
        n_idx = 0
        for d in idx.get("docs", []):
            on = d.get("orig_name")
            if on in old_to_new:
                d["orig_name"] = old_to_new[on]
                n_idx += 1
        if n_idx and not args.dry_run:
            atomic_write(index_path, json.dumps(idx, indent=2) + "\n")
        print(f"      _index.json entries updated: {n_idx}")
        if log_w: log_row(log_w, "index", "", "", "", f"updated:{n_idx}")

    # ---- Step 4: regenerate reprocess-worklist.json ----------------------
    print("[4/5] regenerating reprocess-worklist.json")
    repo_dir = Path(__file__).resolve().parent.parent
    worklist_path = repo_dir / "reprocess-worklist.json"
    missing_report = repo_dir / "reprocess-missing-originals.txt"
    agents_dir = Path("~/my-agents/agents")
    if not args.dry_run and agents_dir.is_dir():
        rc = subprocess.run(
            [
                str(repo_dir / "venv" / "bin" / "python"),
                str(repo_dir / "scripts" / "build_reprocess_worklist.py"),
                "--corpus", str(args.corpus),
                "--originals", str(args.originals),
                "--agents-dir", str(agents_dir),
                "--out", str(worklist_path),
                "--missing-report", str(missing_report),
            ],
            check=False,
        ).returncode
        if rc != 0:
            print(f"      WARNING: worklist regeneration returned rc={rc}")
        else:
            print(f"      regenerated: {worklist_path}")
        if log_w: log_row(log_w, "worklist", "", "", "", f"rc={rc}")
    else:
        print("      skipped (dry-run or no agents dir)")

    # ---- Step 5: done ----------------------------------------------------
    print("[5/5] complete")
    if log_f:
        log_f.close()
        print(f"      audit log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
