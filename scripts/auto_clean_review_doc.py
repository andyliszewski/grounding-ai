#!/usr/bin/env python3
"""Post-ingestion chunk-content-informed cleanup for review-flagged docs.

Called by the watcher right after a review-flagged file has been ingested.
Reads the freshly-written first chunks, asks a local LLM to extract a clean
kebab-case filename from the title page text, validates the response, and
performs a single-doc rename cascade (originals/ + meta.yaml + doc.md +
chunks/*.md + _index.json).

The corpus directory name is intentionally left unchanged — citations are
derived from each chunk's `source:` field at query time, so updating chunks
makes citations clean without touching the on-disk slug. Renaming the corpus
dir would also require remapping FAISS chunk paths and _index.json paths,
which is a larger operation reserved for the bulk workflow.

Failure modes are non-fatal: if Ollama is down, the LLM returns garbage, or
the proposed name collides with something in originals/, the script logs
and exits non-zero. The watcher should treat that as "leave for human"
without aborting ingestion.

Usage:
  python3 scripts/auto_clean_review_doc.py \\
      --slug <corpus-slug> \\
      --corpus /path/to/corpus \\
      --originals /path/to/originals \\
      --collection <collection-name> \\
      [--model qwen2.5:7b] [--api-url http://localhost:11434/v1/chat/completions]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import requests
import yaml

DEFAULT_API_URL = "http://localhost:11434/v1/chat/completions"
# llama3.2:3b is small enough to stay responsive on CPU-only hosts; bigger
# models (qwen2.5:7b) cold-start at 4+ minutes on Apple Silicon CPU. The
# extraction is structurally simple (title page → kebab filename), so the
# smaller model is sufficient.
DEFAULT_MODEL = "llama3.2:3b"
LLM_TIMEOUT_S = 180

# How many leading chunks to feed the LLM. Title pages usually fit in 1,
# but some books push the title past chunk 1 (long copyright pages,
# series-volume listings, etc.). Two is a safe default.
N_CHUNKS_FOR_TITLE = 2

# Strip yaml front matter when feeding chunk text to the model.
FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.S)

# Validation: the proposed filename must look like a kebab-case slug with
# one extension. We accept lowercase ASCII letters, digits, and hyphens.
PROPOSAL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+$")


def read_chunk_body(p: Path) -> str:
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = FRONT_MATTER_RE.sub("", text, count=1)
    return text.strip()


def build_prompt(orig_name: str, chunk_text: str, ext: str) -> list[dict]:
    system = (
        "You extract a clean filename from a book title page. "
        "Output ONLY the filename — no explanation, no quotes, no markdown."
    )
    user = (
        f"Original (messy) filename: {orig_name}\n\n"
        f"Title-page text from the first chunks of the ingested document:\n"
        f"---\n{chunk_text[:4000]}\n---\n\n"
        f"Produce a clean kebab-case filename in this exact format:\n"
        f"  <main-title>-<author-surname>[-<edition-tag>]{ext}\n\n"
        f"Rules:\n"
        f"- main-title: 2 to 8 most distinctive words from the title, lowercase, words joined by hyphens\n"
        f"- author-surname: primary author's last name (or 1–2 last names joined by hyphen if co-authors)\n"
        f"- edition-tag (optional): use '2e','3e','4e' for numbered editions, or 'Nth-anniversary' for anniversary editions, or 'vol-N' for volume markers\n"
        f"- Extension MUST be {ext} (lowercase)\n"
        f"- Only lowercase letters, digits, hyphens. No commas, parens, spaces, dots (except the one before extension)\n"
        f"- No leading or trailing hyphens. No double hyphens.\n"
        f"- Total length 10–100 characters\n"
        f"- If you can't confidently identify a title and author, output the literal token UNKNOWN (with no extension)\n\n"
        f"Filename:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_ollama(messages: list[dict], model: str, api_url: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 80,
    }
    r = requests.post(api_url, json=payload, timeout=LLM_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def validate_proposal(raw: str, expected_ext: str) -> str | None:
    """Return the validated proposal or None if invalid."""
    if not raw:
        return None
    # Strip wrapping noise.
    s = raw.strip().strip("`").strip('"').strip("'").strip()
    # Some models prefix with "Filename:" — drop it.
    s = re.sub(r"^filename:\s*", "", s, flags=re.I)
    # Take the first whitespace-separated token (in case the model added explanation).
    s = s.split()[0] if s.split() else ""
    if s.upper() == "UNKNOWN":
        return None
    s = s.lower()
    # Re-attach extension if model dropped or mismatched it.
    p = Path(s)
    if p.suffix.lower() != expected_ext.lower():
        s = p.stem + expected_ext.lower()
    # Final shape check.
    if not PROPOSAL_RE.match(s):
        return None
    if len(s) < 10 or len(s) > 100:
        return None
    # Reject if it's effectively unchanged garbage (no real letters in the stem).
    stem = Path(s).stem
    if len(stem) < 3 or not re.search(r"[a-z]", stem):
        return None
    return s


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


def patch_yaml_field_file(path: Path, field: str, new_value: str) -> bool:
    """Read YAML, set field, write back. Used for meta.yaml."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if data.get(field) == new_value:
        return False
    data[field] = new_value
    new_text = yaml.safe_dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    atomic_write(path, new_text)
    return True


def patch_frontmatter_field_file(path: Path, field: str, new_value: str) -> bool:
    """Patch the YAML front matter of a markdown file (doc.md or chunks/*.md)."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return False
    fm = yaml.safe_load(m.group(1)) or {}
    if field not in fm:
        return False
    if fm[field] == new_value:
        return False
    fm[field] = new_value
    new_fm = yaml.safe_dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    atomic_write(path, f"---\n{new_fm}---\n{text[m.end():]}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slug", required=True)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--originals", required=True, type=Path)
    p.add_argument("--collection", required=True)
    p.add_argument("--model", default=os.environ.get("AUTO_CLEAN_MODEL", DEFAULT_MODEL))
    p.add_argument("--api-url", default=os.environ.get("AUTO_CLEAN_API_URL", DEFAULT_API_URL))
    p.add_argument("--index", type=Path, help="Path to _index.json (defaults to <corpus>/_index.json)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    slug_dir = args.corpus / args.slug
    if not slug_dir.is_dir():
        print(f"error: slug dir not found: {slug_dir}", file=sys.stderr)
        return 2

    meta_path = slug_dir / "meta.yaml"
    if not meta_path.exists():
        print(f"error: meta.yaml not found in {slug_dir}", file=sys.stderr)
        return 2
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    orig_name = meta.get("orig_name")
    if not orig_name:
        print(f"error: meta.yaml has no orig_name: {meta_path}", file=sys.stderr)
        return 2

    ext = Path(orig_name).suffix.lower()
    if not ext:
        print(f"error: original name has no extension: {orig_name}", file=sys.stderr)
        return 2

    # Concatenate the first N chunks for richer title-page coverage.
    chunks_dir = slug_dir / "chunks"
    chunk_text_parts = []
    for i in range(1, N_CHUNKS_FOR_TITLE + 1):
        body = read_chunk_body(chunks_dir / f"ch_{i:04d}.md")
        if body:
            chunk_text_parts.append(body)
    chunk_text = "\n\n".join(chunk_text_parts)
    if not chunk_text.strip():
        print(f"error: no chunk content to feed LLM for {args.slug}", file=sys.stderr)
        return 3

    # Ask the LLM.
    messages = build_prompt(orig_name, chunk_text, ext)
    try:
        raw = call_ollama(messages, args.model, args.api_url)
    except Exception as exc:
        print(f"error: LLM call failed: {exc}", file=sys.stderr)
        return 4

    proposed = validate_proposal(raw, ext)
    if proposed is None:
        print(f"error: LLM proposal invalid or UNKNOWN: {raw!r}", file=sys.stderr)
        return 5
    if proposed == orig_name:
        print(f"no change needed: {orig_name}")
        return 0

    # Pre-flight on the rename target.
    coll_dir = args.originals / args.collection
    old_path = coll_dir / orig_name
    new_path = coll_dir / proposed
    if not old_path.exists():
        print(f"error: original file not found in expected location: {old_path}", file=sys.stderr)
        return 6
    if new_path.exists():
        print(f"error: rename target already exists: {new_path}", file=sys.stderr)
        return 7

    if args.dry_run:
        print(f"DRY-RUN: would rename {orig_name} -> {proposed} and cascade metadata")
        return 0

    # 1) Rename in originals/
    old_path.rename(new_path)

    # 2) Cascade through corpus dir
    n_meta = patch_yaml_field_file(meta_path, "orig_name", proposed)
    doc_path = slug_dir / "doc.md"
    n_doc = patch_frontmatter_field_file(doc_path, "source", proposed) if doc_path.exists() else False
    n_chunks = 0
    if chunks_dir.is_dir():
        for cf in sorted(chunks_dir.glob("ch_*.md")):
            if patch_frontmatter_field_file(cf, "source", proposed):
                n_chunks += 1

    # 3) Update _index.json — every entry whose orig_name matches the old one
    #    (there can be more than one if the doc was re-ingested with the same slug).
    index_path = args.index or (args.corpus / "_index.json")
    n_index = 0
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        for d in idx.get("docs", []):
            if d.get("orig_name") == orig_name:
                d["orig_name"] = proposed
                n_index += 1
        if n_index:
            atomic_write(index_path, json.dumps(idx, indent=2) + "\n")

    print(
        f"renamed: {orig_name} -> {proposed}  "
        f"(meta={int(n_meta)} doc={int(n_doc)} chunks={n_chunks} index={n_index})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
