"""Tests for grounding.eval.runner (Story 16.2).

All tests inject ``embed_fn``, ``search_fn``, ``load_index_fn``, and
``load_manifest_doc_ids_fn`` so they exercise no FAISS, no embeddings
model, and no filesystem beyond temp dirs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from grounding.eval import (
    EvalRun,
    Expected,
    FixtureItem,
    FixtureSet,
    RetrievedChunk,
    run_eval,
)


# ---------------------------------------------------------------------------
# Helpers: build a FixtureSet in memory and a fake search function.
# ---------------------------------------------------------------------------

def make_fixture_set(
    items: list[FixtureItem],
    *,
    agent: str = "scientist",
    source: Path | None = None,
) -> FixtureSet:
    return FixtureSet(
        agent=agent,
        version=1,
        items=tuple(items),
        source_path=source or Path("/tmp/fake-fixture.yaml"),
    )


def item(
    id: str,
    query: str,
    doc_ids: list[str],
    *,
    chunk_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> FixtureItem:
    return FixtureItem(
        id=id,
        query=query,
        expected=Expected(
            doc_ids=tuple(doc_ids),
            chunk_ids=tuple(chunk_ids or []),
        ),
        tags=tuple(tags or []),
        notes="",
    )


def fake_chunk_map(entries: list[tuple[str, str]]) -> dict:
    """entries: list of (chunk_id, doc_id)."""
    return {
        "format_version": "1.1",
        "chunks": [
            {"chunk_id": cid, "doc_id": did, "embedding_index": i}
            for i, (cid, did) in enumerate(entries)
        ],
    }


class SearchRecorder:
    """Records calls so tests can assert invocation count."""

    def __init__(self, results_by_query: dict[str, list[tuple[str, float]]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[object, int]] = []
        self._last_query = ""

    def embed(self, text: str):
        self._last_query = text
        return text  # pass-through; search_fn uses it to dispatch

    def search(self, index, chunk_map, query_embedding, top_k):
        self.calls.append((query_embedding, top_k))
        return self.results_by_query.get(str(query_embedding), [])


@pytest.fixture
def tmp_paths(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    emb = tmp_path / "embeddings" / "scientist"
    corpus.mkdir()
    emb.mkdir(parents=True)
    return corpus, emb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_runner_calls_search_per_item(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set(
        [
            item("a", "query-a", ["doc1"]),
            item("b", "query-b", ["doc2"]),
            item("c", "query-c", ["doc3"]),
        ]
    )
    chunk_map = fake_chunk_map([("doc1-0001", "doc1"), ("doc2-0001", "doc2"), ("doc3-0001", "doc3")])
    rec = SearchRecorder(
        {
            "query-a": [("doc1-0001", 0.1)],
            "query-b": [("doc2-0001", 0.2)],
            "query-c": [("doc3-0001", 0.3)],
        }
    )

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"doc1", "doc2", "doc3"},
    )

    assert isinstance(result, EvalRun)
    assert len(rec.calls) == 3
    assert len(result.items) == 3
    assert result.top_k == 10


def test_runner_records_first_hit_rank_doc_level(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set([item("a", "q", ["doc-target"])])
    chunk_map = fake_chunk_map(
        [("n1-0001", "n1"), ("t-0001", "doc-target"), ("n2-0001", "n2")]
    )

    rec = SearchRecorder(
        {"q": [("n1-0001", 0.1), ("t-0001", 0.2), ("n2-0001", 0.3)]}
    )
    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"doc-target", "n1", "n2"},
    )

    assert result.items[0].first_hit_rank == 2
    assert result.items[0].strict_first_hit_rank is None


def test_runner_records_strict_first_hit_rank_when_chunk_ids_provided(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set(
        [
            item(
                "a",
                "q",
                ["doc-target"],
                chunk_ids=["doc-target/t-0002"],
            )
        ]
    )
    chunk_map = fake_chunk_map(
        [
            ("t-0001", "doc-target"),
            ("t-0002", "doc-target"),  # the strict match
            ("t-0003", "doc-target"),
        ]
    )
    rec = SearchRecorder(
        {
            "q": [
                ("t-0001", 0.1),
                ("t-0002", 0.2),
                ("t-0003", 0.3),
            ]
        }
    )

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"doc-target"},
    )

    assert result.items[0].first_hit_rank == 1  # doc-level
    assert result.items[0].strict_first_hit_rank == 2  # specific chunk


def test_runner_skips_unknown_doc_id_with_warning(
    tmp_paths, caplog: pytest.LogCaptureFixture
) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set(
        [
            item("keep", "q1", ["doc1"]),
            item("drop", "q2", ["doc-missing"]),
        ]
    )
    chunk_map = fake_chunk_map([("doc1-0001", "doc1")])
    rec = SearchRecorder({"q1": [("doc1-0001", 0.1)]})

    with caplog.at_level(logging.WARNING, logger="grounding.eval.runner"):
        result = run_eval(
            fs,
            "scientist",
            corpus_dir=corpus,
            embeddings_dir=emb,
            embed_fn=rec.embed,
            search_fn=rec.search,
            load_index_fn=lambda _: (object(), chunk_map),
            load_manifest_doc_ids_fn=lambda _: {"doc1"},
        )

    assert result.skipped == ("drop",)
    assert [i.item_id for i in result.items] == ["keep"]
    assert any("doc-missing" in rec.message for rec in caplog.records)


def test_runner_aggregate_matches_metrics_module_output(tmp_paths) -> None:
    """Aggregate recall@1 over 2 items where 1 hits at rank 1 -> 0.5."""
    corpus, emb = tmp_paths
    fs = make_fixture_set(
        [
            item("hit", "q1", ["d1"]),
            item("miss", "q2", ["d2"]),
        ]
    )
    chunk_map = fake_chunk_map(
        [("d1-0001", "d1"), ("d2-0001", "d2"), ("other-0001", "other")]
    )
    rec = SearchRecorder(
        {
            "q1": [("d1-0001", 0.1)],  # hit at rank 1
            "q2": [("other-0001", 0.1)],  # miss
        }
    )

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"d1", "d2", "other"},
    )

    assert result.aggregate.recall_at_1 == 0.5
    assert result.aggregate.recall_at_5 == 0.5
    assert result.aggregate.mrr == pytest.approx(0.5)


def test_runner_per_tag_breakdown_correct(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set(
        [
            item("a", "q1", ["d1"], tags=["methodology", "shared"]),
            item("b", "q2", ["d2"], tags=["methodology"]),
            item("c", "q3", ["d3"], tags=["shared"]),
        ]
    )
    chunk_map = fake_chunk_map(
        [("d1-0001", "d1"), ("d2-0001", "d2"), ("d3-0001", "d3")]
    )
    # a and c hit at rank 1; b misses
    rec = SearchRecorder(
        {
            "q1": [("d1-0001", 0.1)],
            "q2": [("d1-0001", 0.1)],  # wrong doc
            "q3": [("d3-0001", 0.1)],
        }
    )

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"d1", "d2", "d3"},
    )

    per_tag = result.aggregate.per_tag
    assert set(per_tag.keys()) == {"methodology", "shared"}
    assert per_tag["methodology"].n_items == 2
    assert per_tag["methodology"].recall_at_5 == 0.5  # a hits, b misses
    assert per_tag["methodology"].low_sample is False
    assert per_tag["shared"].n_items == 2
    assert per_tag["shared"].recall_at_5 == 1.0  # a and c both hit


def test_runner_per_tag_low_sample_flag(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set([item("a", "q", ["d1"], tags=["rare"])])
    chunk_map = fake_chunk_map([("d1-0001", "d1")])
    rec = SearchRecorder({"q": [("d1-0001", 0.1)]})

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"d1"},
    )

    assert result.aggregate.per_tag["rare"].low_sample is True
    assert result.aggregate.per_tag["rare"].n_items == 1


def test_runner_rejects_mismatched_agent(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set([item("a", "q", ["d1"])], agent="scientist")

    with pytest.raises(ValueError, match="agent mismatch"):
        run_eval(
            fs,
            "other-agent",
            corpus_dir=corpus,
            embeddings_dir=emb,
            embed_fn=lambda t: t,
            search_fn=lambda *a, **kw: [],
            load_index_fn=lambda _: (object(), fake_chunk_map([])),
            load_manifest_doc_ids_fn=lambda _: set(),
        )


def test_retrieved_chunk_is_frozen() -> None:
    chunk = RetrievedChunk(doc_id="d1", chunk_id="c1", score=0.1, rank=1)
    with pytest.raises(Exception):
        chunk.rank = 2  # type: ignore[misc]


def test_runner_records_timestamps(tmp_paths) -> None:
    corpus, emb = tmp_paths
    fs = make_fixture_set([item("a", "q", ["d1"])])
    chunk_map = fake_chunk_map([("d1-0001", "d1")])
    rec = SearchRecorder({"q": [("d1-0001", 0.1)]})

    result = run_eval(
        fs,
        "scientist",
        corpus_dir=corpus,
        embeddings_dir=emb,
        embed_fn=rec.embed,
        search_fn=rec.search,
        load_index_fn=lambda _: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _: {"d1"},
    )

    assert result.started_utc.endswith("+00:00")
    assert result.finished_utc.endswith("+00:00")
    assert result.finished_utc >= result.started_utc
