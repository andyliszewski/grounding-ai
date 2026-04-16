"""Hybrid-retrieval tests for ``SearchCorpusTool`` (Story 19.3).

Exercises the hybrid branch in ``SearchCorpusTool.execute`` and the
shared rerank + hybrid composition. All FAISS / BM25 / embedder / rerank
calls are stubbed — no model loads.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import search_corpus_tool as sct  # noqa: E402
from search_corpus_tool import SearchCorpusTool  # noqa: E402
from grounding.hybrid import HybridConfig  # noqa: E402
from grounding.reranker import RerankConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    for slug, body, page in [
        ("doc-a", "alpha body content", 1),
        ("doc-b", "beta body content", 2),
        ("doc-c", "gamma body content", 3),
    ]:
        d = corpus_dir / slug / "chunks"
        d.mkdir(parents=True)
        (d / "ch_0001.md").write_text(
            f"---\ndoc_id: {slug}\nsource: {slug}.pdf\nchunk_id: {slug}-1\n"
            f"page_start: {page}\npage_end: {page}\n---\n{body}"
        )
    return corpus_dir


@pytest.fixture
def chunk_map():
    return [
        {"file_path": "doc-a/chunks/ch_0001.md", "chunk_id": "doc-a-1", "doc_id": "doc-a"},
        {"file_path": "doc-b/chunks/ch_0001.md", "chunk_id": "doc-b-1", "doc_id": "doc-b"},
        {"file_path": "doc-c/chunks/ch_0001.md", "chunk_id": "doc-c-1", "doc_id": "doc-c"},
    ]


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return embedder


@pytest.fixture
def mock_faiss_index():
    idx = MagicMock()
    idx.search.return_value = (
        np.array([[0.1, 0.2, 0.3]]),
        np.array([[0, 1, 2]]),
    )
    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hybrid_config_none_uses_dense_path(
    mock_embedder, mock_faiss_index, chunk_map, temp_corpus, monkeypatch
):
    """BC: hybrid_config=None must never call search_hybrid."""
    def boom(*a, **kw):
        raise AssertionError("search_hybrid must not be called when disabled")

    monkeypatch.setattr("grounding.hybrid.search_hybrid", boom)

    tool = SearchCorpusTool(
        index=mock_faiss_index,
        chunk_map=chunk_map,
        embedder=mock_embedder,
        corpus_dir=temp_corpus,
    )
    out = tool.execute({"query": "test", "top_k": 3})
    assert "Result 1" in out
    mock_faiss_index.search.assert_called_once()


def test_hybrid_config_enabled_calls_search_hybrid(
    mock_embedder, mock_faiss_index, chunk_map, temp_corpus, monkeypatch
):
    """When hybrid_config is enabled, search_hybrid is invoked."""
    called = {}

    def fake_search_hybrid(query, embeddings_dir, *, top_k, pool_size, k_rrf,
                          load_index_fn=None, load_bm25_fn=None, embed_fn=None):
        called["query"] = query
        called["top_k"] = top_k
        called["pool_size"] = pool_size
        called["k_rrf"] = k_rrf
        return [
            {"chunk_id": "doc-a-1", "doc_id": "doc-a", "faiss_rank": 1,
             "bm25_rank": 1, "rrf_score": 0.1, "faiss_distance": 0.1, "rank": 1},
            {"chunk_id": "doc-b-1", "doc_id": "doc-b", "faiss_rank": 2,
             "bm25_rank": 2, "rrf_score": 0.05, "faiss_distance": 0.2, "rank": 2},
        ]

    monkeypatch.setattr("grounding.hybrid.search_hybrid", fake_search_hybrid)

    tool = SearchCorpusTool(
        index=mock_faiss_index,
        chunk_map=chunk_map,
        embedder=mock_embedder,
        corpus_dir=temp_corpus,
        hybrid_config=HybridConfig(enabled=True, pool_size=20, k_rrf=40),
        embeddings_dir=temp_corpus.parent / "emb",
    )
    out = tool.execute({"query": "bootstrap", "top_k": 2})
    assert called["query"] == "bootstrap"
    assert called["pool_size"] == 20
    assert called["k_rrf"] == 40
    # pool_k = max(20, 0, 2) = 20
    assert called["top_k"] == 20
    # FAISS search must not have been called (hybrid path instead)
    mock_faiss_index.search.assert_not_called()
    assert "Result 1" in out
    assert "doc-a" in out


def test_hybrid_plus_rerank_rerank_receives_hybrid_pool(
    mock_embedder, mock_faiss_index, chunk_map, temp_corpus, monkeypatch
):
    """Rerank must be called with the hybrid-produced pool, not dense."""
    hybrid_ids = ["doc-c-1", "doc-b-1", "doc-a-1"]  # deliberately reverse order

    def fake_search_hybrid(query, embeddings_dir, *, top_k, pool_size, k_rrf,
                          load_index_fn=None, load_bm25_fn=None, embed_fn=None):
        return [
            {"chunk_id": cid, "doc_id": cid.rsplit("-", 1)[0],
             "faiss_rank": i + 1, "bm25_rank": i + 1,
             "rrf_score": 1.0 / (i + 1), "faiss_distance": 0.1, "rank": i + 1}
            for i, cid in enumerate(hybrid_ids)
        ]

    recorded = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        recorded["chunk_ids"] = [c.get("chunk_id") for c in chunks]
        recorded["n"] = len(chunks)
        return [
            {**dict(c), "rerank_score": float(i), "score": float(i)}
            for i, c in enumerate(chunks)
        ]

    monkeypatch.setattr("grounding.hybrid.search_hybrid", fake_search_hybrid)
    monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

    tool = SearchCorpusTool(
        index=mock_faiss_index,
        chunk_map=chunk_map,
        embedder=mock_embedder,
        corpus_dir=temp_corpus,
        hybrid_config=HybridConfig(enabled=True, pool_size=3, k_rrf=60),
        rerank_config=RerankConfig(enabled=True, pool_size=10),
        embeddings_dir=temp_corpus.parent / "emb",
    )
    tool.execute({"query": "q", "top_k": 2})

    # Rerank got the hybrid pool (3 items, in hybrid order), not dense pool.
    assert recorded["chunk_ids"] == hybrid_ids
    assert recorded["n"] == 3
    mock_faiss_index.search.assert_not_called()


def test_rerank_only_when_hybrid_disabled_uses_dense_pool(
    mock_embedder, mock_faiss_index, chunk_map, temp_corpus, monkeypatch
):
    """BC twin: rerank with hybrid off receives the dense _search output."""

    def boom_hybrid(*a, **kw):
        raise AssertionError("search_hybrid must not be called when hybrid is off")

    recorded = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        recorded["n"] = len(chunks)
        recorded["chunk_ids"] = [c.get("chunk_id") for c in chunks]
        return [
            {**dict(c), "rerank_score": float(i), "score": float(i)}
            for i, c in enumerate(chunks)
        ]

    monkeypatch.setattr("grounding.hybrid.search_hybrid", boom_hybrid)
    monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

    tool = SearchCorpusTool(
        index=mock_faiss_index,
        chunk_map=chunk_map,
        embedder=mock_embedder,
        corpus_dir=temp_corpus,
        rerank_config=RerankConfig(enabled=True, pool_size=3),
    )
    tool.execute({"query": "q", "top_k": 2})

    # Dense-sourced chunk_ids come from the chunk YAML, in FAISS-returned order.
    assert recorded["n"] == 3
    assert recorded["chunk_ids"] == ["doc-a-1", "doc-b-1", "doc-c-1"]
    mock_faiss_index.search.assert_called_once()


def test_hybrid_degraded_flag_passes_through_to_result_dict(
    mock_embedder, mock_faiss_index, chunk_map, temp_corpus, monkeypatch
):
    """hybrid_degraded=True on the hit dict propagates to the enriched result."""
    def fake_search_hybrid(query, embeddings_dir, *, top_k, pool_size, k_rrf,
                          load_index_fn=None, load_bm25_fn=None, embed_fn=None):
        return [
            {"chunk_id": "doc-a-1", "doc_id": "doc-a", "faiss_rank": 1,
             "bm25_rank": None, "rrf_score": 0.1, "faiss_distance": 0.1,
             "rank": 1, "hybrid_degraded": True},
        ]

    monkeypatch.setattr("grounding.hybrid.search_hybrid", fake_search_hybrid)

    tool = SearchCorpusTool(
        index=mock_faiss_index,
        chunk_map=chunk_map,
        embedder=mock_embedder,
        corpus_dir=temp_corpus,
        hybrid_config=HybridConfig(enabled=True, pool_size=1, k_rrf=60),
        embeddings_dir=temp_corpus.parent / "emb",
    )
    # Inspect enriched dicts directly rather than the formatted string.
    enriched = tool._search_hybrid("q", pool_k=1)
    assert len(enriched) == 1
    assert enriched[0].get("hybrid_degraded") is True
    assert enriched[0]["bm25_rank"] is None
    assert enriched[0]["faiss_rank"] == 1
