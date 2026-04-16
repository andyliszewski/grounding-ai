"""Tests for grounding.eval.report (Story 16.3 — report serialization)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounding.eval import to_json, to_markdown, write_artifacts
from grounding.eval.runner import (
    EvalAggregate,
    EvalItemResult,
    EvalRun,
    RetrievedChunk,
    TagMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(*, items: list[EvalItemResult] | None = None, skipped: tuple[str, ...] = ()) -> EvalRun:
    items = items if items is not None else [
        EvalItemResult(
            item_id="ds-001",
            query="What does ablation tell us about attention heads?",
            expected_doc_ids=("doc-alpha",),
            retrieved=(
                RetrievedChunk(doc_id="doc-alpha", chunk_id="doc-alpha-0001", score=0.9, rank=1),
                RetrievedChunk(doc_id="doc-other", chunk_id="doc-other-0001", score=0.5, rank=2),
                RetrievedChunk(doc_id="doc-third", chunk_id="doc-third-0001", score=0.4, rank=3),
            ),
            first_hit_rank=1,
            strict_first_hit_rank=None,
            tags=("methodology",),
        )
    ]
    aggregate = EvalAggregate(
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=0.9,
        per_tag={
            "methodology": TagMetrics(
                recall_at_5=1.0, mrr=1.0, n_items=len(items), low_sample=len(items) < 2,
            )
        },
    )
    return EvalRun(
        agent="data-scientist",
        fixture_path=Path("docs/eval/fixtures/data-scientist.yaml"),
        top_k=10,
        items=tuple(items),
        aggregate=aggregate,
        skipped=skipped,
        started_utc="2026-04-13T15:30:00+00:00",
        finished_utc="2026-04-13T15:30:42+00:00",
    )


_RUN_KW = dict(
    run_id="data-scientist-20260413-153000",
    corpus_dir=Path("/tmp/corpus"),
    embeddings_dir=Path("/tmp/embeddings/data-scientist"),
)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def test_to_json_round_trip():
    run = _make_run()
    payload_str = to_json(run, **_RUN_KW)
    payload = json.loads(payload_str)

    assert payload["format_version"] == 2
    assert payload["run_id"] == "data-scientist-20260413-153000"
    assert payload["agent"] == "data-scientist"
    assert payload["top_k"] == 10
    assert payload["aggregate"]["n_items"] == 1
    assert payload["aggregate"]["n_skipped"] == 0
    assert payload["aggregate"]["recall_at_5"] == 1.0
    assert payload["aggregate"]["per_tag"]["methodology"]["n_items"] == 1
    assert payload["items"][0]["item_id"] == "ds-001"
    assert payload["items"][0]["retrieved"][0]["doc_id"] == "doc-alpha"
    assert payload["skipped"] == []


def test_to_json_includes_skipped_items():
    run = _make_run(skipped=("ds-007",))
    payload = json.loads(to_json(run, **_RUN_KW))
    assert payload["skipped"] == ["ds-007"]
    assert payload["aggregate"]["n_skipped"] == 1


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def test_to_markdown_contains_all_sections():
    md = to_markdown(_make_run(), **_RUN_KW)
    assert "# Eval Run: data-scientist-20260413-153000" in md
    assert "## Aggregate Metrics" in md
    assert "## Per-Tag" in md
    assert "## Per-Item" in md
    assert "## Skipped (0)" in md
    # Header table columns
    assert "| Metric | Value |" in md
    assert "| Tag | n | recall@5 | MRR | low_sample |" in md
    assert "| ID | Query | First Hit | Top-3 doc_ids | Citation" in md


def test_to_markdown_truncates_long_queries():
    long_query = "alpha " * 40  # well over 80 chars
    items = [
        EvalItemResult(
            item_id="ds-002",
            query=long_query,
            expected_doc_ids=("doc-alpha",),
            retrieved=(
                RetrievedChunk(doc_id="doc-alpha", chunk_id="doc-alpha-0001", score=0.9, rank=1),
            ),
            first_hit_rank=1,
            strict_first_hit_rank=None,
            tags=(),
        )
    ]
    md = to_markdown(_make_run(items=items), **_RUN_KW)
    # Find the per-item row for ds-002 and confirm truncation.
    row = next(line for line in md.splitlines() if line.startswith("| ds-002"))
    # Truncation marker is the unicode horizontal ellipsis.
    assert "\u2026" in row
    # Query cell is bounded by pipes; extract and assert length cap.
    query_cell = row.split(" | ")[1]
    assert len(query_cell) <= 80


def test_to_markdown_with_baseline_shows_delta_column():
    baseline = {
        "aggregate": {
            "recall_at_1": 0.8,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 0.9,
            "ndcg_at_10": 0.9,
        }
    }
    md = to_markdown(_make_run(), baseline=baseline, **_RUN_KW)
    assert "vs Baseline" in md
    # Current recall@1=1.0, baseline 0.8 → delta +0.2 (up arrow)
    assert "\u2191" in md


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_write_artifacts_writes_both_files(tmp_path: Path):
    run = _make_run()
    out_dir = tmp_path / "reports"
    json_path, md_path = write_artifacts(
        run,
        run_id="data-scientist-20260413-153000",
        out_dir=out_dir,
        corpus_dir=Path("/tmp/corpus"),
        embeddings_dir=Path("/tmp/embeddings/data-scientist"),
    )
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.name.endswith(".json")
    assert md_path.name.endswith(".md")
    payload = json.loads(json_path.read_text())
    assert payload["format_version"] == 2


def _make_citation_run() -> EvalRun:
    items = [
        EvalItemResult(
            item_id="ds-001",
            query="bootstrap confidence interval",
            expected_doc_ids=("doc-beta",),
            retrieved=(
                RetrievedChunk(
                    doc_id="doc-beta",
                    chunk_id="doc-beta-0001",
                    score=0.9,
                    rank=1,
                    page_start=247,
                    page_end=247,
                    section_heading="3.2 Bootstrap Methods",
                ),
            ),
            first_hit_rank=1,
            strict_first_hit_rank=None,
            tags=("statistics",),
            expected_page=247,
            expected_section="3.2 Bootstrap Methods",
        ),
        EvalItemResult(
            item_id="ds-002",
            query="plain query",
            expected_doc_ids=("doc-alpha",),
            retrieved=(
                RetrievedChunk(
                    doc_id="doc-alpha",
                    chunk_id="doc-alpha-0001",
                    score=0.9,
                    rank=1,
                ),
            ),
            first_hit_rank=1,
            strict_first_hit_rank=None,
            tags=(),
        ),
    ]
    aggregate = EvalAggregate(
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        per_tag={},
        citation_accuracy=1.0,
        n_citation_items=1,
    )
    return EvalRun(
        agent="data-scientist",
        fixture_path=Path("docs/eval/fixtures/data-scientist.yaml"),
        top_k=10,
        items=tuple(items),
        aggregate=aggregate,
        skipped=(),
        started_utc="2026-04-14T00:00:00+00:00",
        finished_utc="2026-04-14T00:00:01+00:00",
    )


def test_aggregate_includes_citation_accuracy_when_items_have_expectations():
    run = _make_citation_run()
    payload = json.loads(to_json(run, **_RUN_KW))
    assert payload["aggregate"]["citation_accuracy"] == 1.0
    assert payload["aggregate"]["n_citation_items"] == 1


def test_aggregate_citation_accuracy_none_when_no_expectations():
    # _make_run() creates items without citation expectations.
    run = _make_run()
    payload = json.loads(to_json(run, **_RUN_KW))
    assert payload["aggregate"]["citation_accuracy"] is None
    assert payload["aggregate"]["n_citation_items"] == 0


def test_json_format_version_is_2():
    payload = json.loads(to_json(_make_citation_run(), **_RUN_KW))
    assert payload["format_version"] == 2


def test_markdown_report_surfaces_citation_accuracy_row():
    md = to_markdown(_make_citation_run(), **_RUN_KW)
    # Metric appears in aggregate table when non-null.
    assert "citation_accuracy" in md


def test_markdown_report_omits_citation_accuracy_when_none():
    md = to_markdown(_make_run(), **_RUN_KW)
    assert "citation_accuracy" not in md


def test_markdown_report_per_item_citation_column():
    md = to_markdown(_make_citation_run(), **_RUN_KW)
    assert "Citation (retrieved p." in md
    # Row with citation metadata renders page + section.
    row = next(line for line in md.splitlines() if line.startswith("| ds-001"))
    assert "p.247" in row
    assert "\u00a73.2 Bootstrap Methods" in row


def test_retrieved_entries_carry_page_and_section_in_json():
    payload = json.loads(to_json(_make_citation_run(), **_RUN_KW))
    first = payload["items"][0]["retrieved"][0]
    assert first["page_start"] == 247
    assert first["page_end"] == 247
    assert first["section_heading"] == "3.2 Bootstrap Methods"
    # Items without metadata serialize null.
    second = payload["items"][1]["retrieved"][0]
    assert second["page_start"] is None
    assert second["section_heading"] is None


def test_write_artifacts_atomic_does_not_leave_tempfiles_on_success(tmp_path: Path):
    run = _make_run()
    out_dir = tmp_path / "reports"
    write_artifacts(
        run,
        run_id="agent-20260413-153000",
        out_dir=out_dir,
        corpus_dir=Path("/tmp/corpus"),
        embeddings_dir=Path("/tmp/embeddings/agent"),
    )
    # No leftover .tmp files from the atomic_write helper.
    leftovers = list(out_dir.glob(".*.tmp"))
    assert leftovers == []
