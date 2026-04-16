"""Tests for Story 18.2 rerank_config wiring in scripts/local_rag.py.

The retrieval helper (`search_corpus` in local_rag) gains an optional
`rerank_config` argument; when enabled it runs the two-stage flow before
returning results. CLI flag wiring is deferred to Story 18.3, so these
tests exercise the helper directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import local_rag  # noqa: E402
from grounding.reranker import RerankConfig  # noqa: E402


@pytest.fixture
def rerank_corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    for slug, body in [
        ("doc-w", "tiny"),
        ("doc-x", "middling body"),
        ("doc-y", "considerably longer body content"),
        ("doc-z", "the longest body of them all by a fair margin"),
    ]:
        d = corpus_dir / slug / "chunks"
        d.mkdir(parents=True)
        (d / "ch_0001.md").write_text(
            f"---\ndoc_id: {slug}\nsource: {slug}.pdf\nchunk_id: 1\n---\n{body}"
        )
    return corpus_dir


@pytest.fixture
def chunk_map():
    return [
        {"file_path": "doc-w/chunks/ch_0001.md"},
        {"file_path": "doc-x/chunks/ch_0001.md"},
        {"file_path": "doc-y/chunks/ch_0001.md"},
        {"file_path": "doc-z/chunks/ch_0001.md"},
    ]


def _fake_index(n):
    idx = MagicMock()

    def _search(_vec, k):
        k = min(k, n)
        return (
            np.array([[0.1 * (i + 1) for i in range(k)]]),
            np.array([[i for i in range(k)]]),
        )

    idx.search.side_effect = _search
    return idx


@pytest.fixture
def fake_embedder():
    e = MagicMock()
    e.encode.return_value = np.array([[0.0]])
    return e


def test_retrieval_helper_skips_rerank_when_disabled(
    rerank_corpus, chunk_map, fake_embedder, monkeypatch
):
    """rerank_config=None leaves the FAISS-order output unchanged and
    never invokes the reranker."""

    def boom(*a, **kw):
        raise AssertionError("reranker should not be called when disabled")

    monkeypatch.setattr(local_rag._reranker_module, "rerank", boom)

    results_none = local_rag.search_corpus(
        "q",
        _fake_index(4),
        chunk_map,
        fake_embedder,
        rerank_corpus,
        top_k=3,
    )
    results_disabled = local_rag.search_corpus(
        "q",
        _fake_index(4),
        chunk_map,
        fake_embedder,
        rerank_corpus,
        top_k=3,
        rerank_config=RerankConfig(enabled=False, pool_size=10),
    )

    # Both paths produce identical ordering (doc-w first — FAISS returned
    # indices in ascending order per the fake index).
    assert [r["metadata"]["source"] for r in results_none] == [
        "doc-w.pdf",
        "doc-x.pdf",
        "doc-y.pdf",
    ]
    assert [r["metadata"]["source"] for r in results_disabled] == [
        "doc-w.pdf",
        "doc-x.pdf",
        "doc-y.pdf",
    ]


def test_retrieval_helper_honors_rerank_config_when_enabled(
    rerank_corpus, chunk_map, fake_embedder, monkeypatch
):
    """With rerank_config.enabled=True, the helper fetches pool_size
    candidates, runs them through the reranker, and truncates to top_k."""

    captured = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        captured["pool_len"] = len(chunks)
        captured["model"] = config.model
        rev = list(reversed(list(chunks)))
        return [
            {**dict(c), "rerank_score": float(i), "faiss_distance": c.get("score"), "score": float(i)}
            for i, c in enumerate(rev)
        ]

    monkeypatch.setattr(local_rag._reranker_module, "rerank", fake_rerank)

    idx = _fake_index(4)
    results = local_rag.search_corpus(
        "q",
        idx,
        chunk_map,
        fake_embedder,
        rerank_corpus,
        top_k=2,
        rerank_config=RerankConfig(enabled=True, pool_size=4),
    )

    # FAISS was asked for the pool size, not top_k.
    assert idx.search.call_args_list[0][0][1] == 4
    # Reranker saw all four candidates.
    assert captured["pool_len"] == 4
    # Output is truncated to top_k=2 and renumbered 1..2 in reranked order.
    assert len(results) == 2
    assert [r["rank"] for r in results] == [1, 2]
    # Reversal flips order: doc-z first, then doc-y.
    assert results[0]["metadata"]["source"] == "doc-z.pdf"
    assert results[1]["metadata"]["source"] == "doc-y.pdf"
