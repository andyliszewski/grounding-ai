#!/usr/bin/env python3
"""Build a worklist of pre-Epic-17 corpus docs that need reprocessing.

A doc is included in the worklist iff:
  - Its first chunk's `created_utc` predates the Epic-17 cutover, AND
  - Its `orig_name` from meta.yaml is still findable on disk under
    `<originals>/<collection>/<orig_name>`.

Docs that match the first condition but not the second are reported in
`--missing-report` and skipped.

Worklist entries are bucketed by the first matching agent in the priority
list (anchored to `corpus_filter.collections`); docs that match no
priority agent land in the `the-rest` bucket. The output JSON preserves
priority order, then alphabetical slug order within each bucket — so the
driver can iterate it sequentially without sorting.

Usage:
  python3 scripts/build_reprocess_worklist.py \\
      --corpus /path/to/corpus \\
      --originals /path/to/originals \\
      --agents-dir /path/to/agents \\
      --out reprocess-worklist.json \\
      --missing-report reprocess-missing-originals.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import yaml

# Epic 17 ("page and section citations") shipped 2026-04-14.
# Chunks written before this date are pre-cutover and need reprocessing
# to gain page_start / page_end / section_heading.
DEFAULT_CUTOVER = "2026-04-14T00:00:00+00:00"

DEFAULT_PRIORITY = [
    "mathematician",
    "data-scientist",
    "mechanical-engineer",
    "ceo",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--originals", required=True, type=Path)
    p.add_argument("--agents-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("reprocess-worklist.json"))
    p.add_argument(
        "--missing-report",
        type=Path,
        default=Path("reprocess-missing-originals.txt"),
        help="Text file listing docs that are pre-cutover but whose source is gone.",
    )
    p.add_argument(
        "--cutover",
        default=DEFAULT_CUTOVER,
        help=f"ISO8601 timestamp marking the new-ingestion start (default: {DEFAULT_CUTOVER}).",
    )
    p.add_argument(
        "--priority",
        default=",".join(DEFAULT_PRIORITY),
        help="Comma-separated agent priority order (default: %(default)s).",
    )
    return p.parse_args()


def load_agent_collections(agents_dir: Path, priority: list[str]) -> dict[str, set[str]]:
    """Return {agent_name: set_of_collections} for each priority agent."""
    out: dict[str, set[str]] = {}
    for agent in priority:
        path = agents_dir / f"{agent}.yaml"
        if not path.exists():
            sys.exit(f"error: agent file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        colls = (data.get("corpus_filter") or {}).get("collections") or []
        out[agent] = set(colls)
    return out


def first_chunk_created_utc(slug_dir: Path) -> dt.datetime | None:
    """Read created_utc from `chunks/ch_0001.md` front-matter."""
    first = slug_dir / "chunks" / "ch_0001.md"
    if not first.exists():
        return None
    text = first.read_text(encoding="utf-8", errors="ignore")
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    if not m:
        return None
    dm = re.search(r"created_utc:\s*'?([0-9T:+\-]+)'?", m.group(1))
    if not dm:
        return None
    try:
        ts = dt.datetime.fromisoformat(dm.group(1))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts


def locate_original(originals_root: Path, orig_name: str, collections: list[str]) -> Path | None:
    """Find orig_name under originals_root, preferring its own collection dirs."""
    if not orig_name:
        return None
    for c in collections:
        cand = originals_root / c / orig_name
        if cand.exists():
            return cand
    # Fallback: any subdir (slow on huge trees but tractable here).
    for cand in originals_root.rglob(orig_name):
        return cand
    return None


def bucket_for(doc_collections: set[str], priority: list[str], agent_colls: dict[str, set[str]]) -> str:
    for agent in priority:
        if doc_collections & agent_colls[agent]:
            return agent
    return "the-rest"


def main() -> int:
    args = parse_args()

    cutover = dt.datetime.fromisoformat(args.cutover)
    if cutover.tzinfo is None:
        cutover = cutover.replace(tzinfo=dt.timezone.utc)

    priority = [p.strip() for p in args.priority.split(",") if p.strip()]
    agent_colls = load_agent_collections(args.agents_dir, priority)

    if not args.corpus.is_dir():
        sys.exit(f"error: --corpus is not a directory: {args.corpus}")
    if not args.originals.is_dir():
        sys.exit(f"error: --originals is not a directory: {args.originals}")

    entries: list[dict] = []
    missing: list[str] = []
    counts = {b: 0 for b in priority + ["the-rest"]}
    skipped_post_cutover = 0
    skipped_no_meta = 0

    for slug_dir in sorted(args.corpus.iterdir()):
        if not slug_dir.is_dir():
            continue
        meta_path = slug_dir / "meta.yaml"
        if not meta_path.exists():
            skipped_no_meta += 1
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            skipped_no_meta += 1
            continue

        ts = first_chunk_created_utc(slug_dir)
        if ts is None:
            skipped_no_meta += 1
            continue
        if ts >= cutover:
            skipped_post_cutover += 1
            continue

        orig_name = meta.get("orig_name") or ""
        collections = meta.get("collections") or []
        orig_path = locate_original(args.originals, orig_name, collections)
        if orig_path is None:
            missing.append(f"{slug_dir.name}\t{orig_name}\t{','.join(collections)}")
            continue

        bucket = bucket_for(set(collections), priority, agent_colls)
        counts[bucket] += 1
        entries.append({
            "slug": slug_dir.name,
            "orig_name": orig_name,
            "orig_path": str(orig_path),
            "collections": list(collections),
            "priority": bucket,
            "status": "pending",
            "created_utc": ts.isoformat(),
        })

    # Sort: priority bucket first (in given order), then slug alphabetically.
    bucket_index = {b: i for i, b in enumerate(priority + ["the-rest"])}
    entries.sort(key=lambda e: (bucket_index[e["priority"]], e["slug"]))

    # Write outputs.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "cutover": cutover.isoformat(),
        "priority": priority,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "counts": counts,
        "skipped_post_cutover": skipped_post_cutover,
        "skipped_no_meta": skipped_no_meta,
        "missing_originals": len(missing),
        "entries": entries,
    }, indent=2) + "\n", encoding="utf-8")

    args.missing_report.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Pre-cutover docs whose original source could not be found.\n"
        f"# Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        f"# Format: <slug>\\t<orig_name>\\t<collections>\n"
    )
    args.missing_report.write_text(header + "\n".join(missing) + "\n", encoding="utf-8")

    # Console summary.
    print(f"worklist written: {args.out}")
    print(f"missing-originals report: {args.missing_report}  ({len(missing)} docs)")
    print()
    print("bucket counts:")
    total = 0
    for b in priority + ["the-rest"]:
        print(f"  {b:<22} {counts[b]:>5}")
        total += counts[b]
    print(f"  {'TOTAL':<22} {total:>5}")
    print()
    print(f"skipped (already post-cutover):  {skipped_post_cutover}")
    print(f"skipped (no usable meta/chunks): {skipped_no_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
