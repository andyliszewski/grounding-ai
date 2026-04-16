"""Eval report serialization (Story 16.3, extended in Story 17.4).

Renders an :class:`~grounding.eval.runner.EvalRun` to two artifacts:

- **JSON** (machine-readable, stable schema with ``format_version: 2``) for
  CI consumption and baseline capture.
- **Markdown** (human-readable, GitHub-flavored tables) for PR review.

``format_version`` migration (Story 17.4)
-----------------------------------------
The writer always emits ``format_version: 2``. The reader
(``grounding.eval.baseline.load_baseline``) accepts both ``1`` and ``2``; a v1
baseline is coerced in-memory as if it carried ``citation_accuracy=None`` and
``n_citation_items=0``. Tests assert ``format_version == 2`` where they exercise
the writer, and accept either where they only check round-trip readability.

Canonical JSON schema (``format_version: 2``)::

    {
      "format_version": 2,
      "run_id": "<agent>-<UTC-yyyymmdd-hhmmss>",
      "agent": "<agent-name>",
      "fixture_path": "<str>",
      "corpus_dir": "<str>",
      "embeddings_dir": "<str>",
      "top_k": <int>,
      "started_utc": "<ISO8601>",
      "finished_utc": "<ISO8601>",
      "rerank": {                               # optional, Story 18.4
        "enabled": <bool>,
        "model": "<str>",
        "pool_size": <int>,
        "batch_size": <int>
      },
      "aggregate": {
        "n_items": <int>,
        "n_skipped": <int>,
        "recall_at_1": <float>,
        "recall_at_3": <float>,
        "recall_at_5": <float>,
        "recall_at_10": <float>,
        "mrr": <float>,
        "ndcg_at_10": <float>,
        "citation_accuracy": <float|null>,        # Story 17.4
        "n_citation_items": <int>,                # Story 17.4
        "per_tag": {
          "<tag>": {
            "recall_at_5": <float>,
            "mrr": <float>,
            "n_items": <int>,
            "low_sample": <bool>
          }
        }
      },
      "items": [
        {
          "item_id": "<str>",
          "query": "<str>",
          "expected_doc_ids": ["<str>", ...],
          "first_hit_rank": <int|null>,
          "strict_first_hit_rank": <int|null>,
          "tags": ["<str>", ...],
          "retrieved": [
            {
              "doc_id": "<str>", "chunk_id": "<str>", "score": <float>, "rank": <int>,
              "page_start": <int|null>, "page_end": <int|null>,
              "section_heading": <str|null>
            }
          ]
        }
      ],
      "skipped": ["<item_id>", ...]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from grounding.eval.runner import EvalAggregate, EvalRun, TagMetrics
from grounding.utils import atomic_write

FORMAT_VERSION = 2
_QUERY_TRUNCATE = 80


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _aggregate_to_dict(agg: EvalAggregate, *, n_items: int, n_skipped: int) -> dict[str, Any]:
    return {
        "n_items": n_items,
        "n_skipped": n_skipped,
        "recall_at_1": agg.recall_at_1,
        "recall_at_3": agg.recall_at_3,
        "recall_at_5": agg.recall_at_5,
        "recall_at_10": agg.recall_at_10,
        "mrr": agg.mrr,
        "ndcg_at_10": agg.ndcg_at_10,
        "citation_accuracy": agg.citation_accuracy,
        "n_citation_items": agg.n_citation_items,
        "per_tag": {
            tag: {
                "recall_at_5": tm.recall_at_5,
                "mrr": tm.mrr,
                "n_items": tm.n_items,
                "low_sample": tm.low_sample,
            }
            for tag, tm in agg.per_tag.items()
        },
    }


def eval_run_to_dict(eval_run: EvalRun, *, run_id: str, corpus_dir: Path, embeddings_dir: Path) -> dict[str, Any]:
    """Serialize an EvalRun to the canonical JSON-compatible dict (format_version 2)."""
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "agent": eval_run.agent,
        "fixture_path": str(eval_run.fixture_path),
        "corpus_dir": str(corpus_dir),
        "embeddings_dir": str(embeddings_dir),
        "top_k": eval_run.top_k,
        "started_utc": eval_run.started_utc,
        "finished_utc": eval_run.finished_utc,
    }
    if eval_run.rerank is not None:
        payload["rerank"] = {
            "enabled": eval_run.rerank.enabled,
            "model": eval_run.rerank.model,
            "pool_size": eval_run.rerank.pool_size,
            "batch_size": eval_run.rerank.batch_size,
        }
    if eval_run.hybrid is not None:
        payload["hybrid"] = {
            "enabled": eval_run.hybrid.enabled,
            "pool_size": eval_run.hybrid.pool_size,
            "k_rrf": eval_run.hybrid.k_rrf,
        }
    payload.update({
        "aggregate": _aggregate_to_dict(
            eval_run.aggregate,
            n_items=len(eval_run.items),
            n_skipped=len(eval_run.skipped),
        ),
        "items": [
            {
                "item_id": it.item_id,
                "query": it.query,
                "expected_doc_ids": list(it.expected_doc_ids),
                "first_hit_rank": it.first_hit_rank,
                "strict_first_hit_rank": it.strict_first_hit_rank,
                "tags": list(it.tags),
                "retrieved": [
                    {
                        "doc_id": r.doc_id,
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "rank": r.rank,
                        "page_start": r.page_start,
                        "page_end": r.page_end,
                        "section_heading": r.section_heading,
                    }
                    for r in it.retrieved
                ],
            }
            for it in eval_run.items
        ],
        "skipped": list(eval_run.skipped),
    })
    return payload


def to_json(eval_run: EvalRun, *, run_id: str, corpus_dir: Path, embeddings_dir: Path) -> str:
    """Render an EvalRun as a pretty-printed JSON string."""
    payload = eval_run_to_dict(
        eval_run, run_id=run_id, corpus_dir=corpus_dir, embeddings_dir=embeddings_dir
    )
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MetricRow:
    name: str
    value: float
    delta: float | None  # None when no baseline


def _format_retrieved_citation(item: Any) -> str:
    """Render the first retrieved chunk's page/section for the per-item table."""
    if not item.retrieved:
        return "-"
    r = item.retrieved[0]
    parts: list[str] = []
    if r.page_start is not None and r.page_end is not None:
        if r.page_start == r.page_end:
            parts.append(f"p.{r.page_start}")
        else:
            parts.append(f"p.{r.page_start}\u2013{r.page_end}")
    if r.section_heading:
        section = r.section_heading.replace("|", "\\|")
        parts.append(f"\u00a7{section}")
    return ", ".join(parts) if parts else "-"


def _truncate_query(query: str, limit: int = _QUERY_TRUNCATE) -> str:
    q = query.strip().replace("\n", " ").replace("|", "\\|")
    if len(q) <= limit:
        return q
    return q[: limit - 1].rstrip() + "\u2026"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return ""
    if abs(delta) < 1e-6:
        return "="
    arrow = "\u2191" if delta > 0 else "\u2193"
    return f"{arrow}{abs(delta):.3f}"


def _aggregate_rows(
    aggregate: EvalAggregate, baseline_aggregate: Mapping[str, Any] | None
) -> list[_MetricRow]:
    metrics: list[tuple[str, float, str]] = [
        ("recall@1", aggregate.recall_at_1, "recall_at_1"),
        ("recall@3", aggregate.recall_at_3, "recall_at_3"),
        ("recall@5", aggregate.recall_at_5, "recall_at_5"),
        ("recall@10", aggregate.recall_at_10, "recall_at_10"),
        ("MRR", aggregate.mrr, "mrr"),
        ("nDCG@10", aggregate.ndcg_at_10, "ndcg_at_10"),
    ]
    if aggregate.citation_accuracy is not None:
        metrics.append(
            ("citation_accuracy", aggregate.citation_accuracy, "citation_accuracy")
        )
    rows: list[_MetricRow] = []
    for label, value, key in metrics:
        delta: float | None = None
        if (
            baseline_aggregate is not None
            and key in baseline_aggregate
            and baseline_aggregate[key] is not None
        ):
            delta = value - float(baseline_aggregate[key])
        rows.append(_MetricRow(label, value, delta))
    return rows


def to_markdown(
    eval_run: EvalRun,
    *,
    run_id: str,
    corpus_dir: Path,
    embeddings_dir: Path,
    baseline: Mapping[str, Any] | None = None,
) -> str:
    """Render an EvalRun as a GitHub-flavored Markdown report.

    ``baseline`` is the parsed baseline mapping (full file contents); if
    provided, the aggregate-metrics table includes a vs-Baseline column.
    """
    baseline_aggregate = (baseline or {}).get("aggregate") if baseline else None

    lines: list[str] = []
    lines.append(f"# Eval Run: {run_id}")
    lines.append("")
    lines.append(f"**Agent:** {eval_run.agent}")
    lines.append(
        f"**Fixture:** {eval_run.fixture_path} "
        f"({len(eval_run.items)} items, {len(eval_run.skipped)} skipped)"
    )
    lines.append(f"**Corpus:** {corpus_dir}")
    lines.append(f"**Embeddings:** {embeddings_dir}")
    lines.append(f"**top_k:** {eval_run.top_k}")
    if eval_run.rerank is not None:
        rr = eval_run.rerank
        lines.append(
            f"**Rerank:** enabled={rr.enabled} model=`{rr.model}` "
            f"pool_size={rr.pool_size} batch_size={rr.batch_size}"
        )
    if eval_run.hybrid is not None:
        hy = eval_run.hybrid
        lines.append(
            f"**Hybrid:** enabled={hy.enabled} pool_size={hy.pool_size} "
            f"k_rrf={hy.k_rrf}"
        )
    lines.append(f"**Started:** {eval_run.started_utc}")
    lines.append(f"**Finished:** {eval_run.finished_utc}")
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    if baseline_aggregate is not None:
        lines.append("| Metric | Value | vs Baseline |")
        lines.append("|--------|-------|-------------|")
        for row in _aggregate_rows(eval_run.aggregate, baseline_aggregate):
            lines.append(
                f"| {row.name} | {_fmt_float(row.value)} | {_fmt_delta(row.delta)} |"
            )
    else:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for row in _aggregate_rows(eval_run.aggregate, None):
            lines.append(f"| {row.name} | {_fmt_float(row.value)} |")
    lines.append("")

    # Per-tag
    lines.append("## Per-Tag")
    lines.append("")
    if eval_run.aggregate.per_tag:
        lines.append("| Tag | n | recall@5 | MRR | low_sample |")
        lines.append("|-----|---|----------|-----|------------|")
        for tag in sorted(eval_run.aggregate.per_tag):
            tm: TagMetrics = eval_run.aggregate.per_tag[tag]
            lines.append(
                f"| {tag} | {tm.n_items} | {_fmt_float(tm.recall_at_5)} | "
                f"{_fmt_float(tm.mrr)} | {'yes' if tm.low_sample else 'no'} |"
            )
    else:
        lines.append("_No tagged items._")
    lines.append("")

    # Per-item
    lines.append("## Per-Item")
    lines.append("")
    lines.append(
        "| ID | Query | First Hit | Top-3 doc_ids | Citation (retrieved p./\u00a7) |"
    )
    lines.append("|----|-------|-----------|---------------|------------------------------|")
    for item in eval_run.items:
        first = "-" if item.first_hit_rank is None else str(item.first_hit_rank)
        top3 = ", ".join(r.doc_id for r in item.retrieved[:3]) or "-"
        citation = _format_retrieved_citation(item)
        lines.append(
            f"| {item.item_id} | {_truncate_query(item.query)} | {first} | {top3} | {citation} |"
        )
    lines.append("")

    # Skipped
    lines.append(f"## Skipped ({len(eval_run.skipped)})")
    lines.append("")
    if eval_run.skipped:
        for sid in eval_run.skipped:
            lines.append(f"- `{sid}`")
    else:
        lines.append("_None._")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disk artifacts
# ---------------------------------------------------------------------------

def write_artifacts(
    eval_run: EvalRun,
    *,
    run_id: str,
    out_dir: Path,
    corpus_dir: Path,
    embeddings_dir: Path,
    baseline: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically write ``<run_id>.json`` and ``<run_id>.md`` to ``out_dir``.

    Returns the (json_path, md_path) tuple.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"

    json_text = to_json(
        eval_run, run_id=run_id, corpus_dir=corpus_dir, embeddings_dir=embeddings_dir
    )
    md_text = to_markdown(
        eval_run,
        run_id=run_id,
        corpus_dir=corpus_dir,
        embeddings_dir=embeddings_dir,
        baseline=baseline,
    )

    atomic_write(json_path, json_text)
    atomic_write(md_path, md_text)
    return json_path, md_path
