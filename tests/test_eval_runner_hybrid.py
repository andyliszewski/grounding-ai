"""Hybrid-retrieval tests for ``grounding.eval.runner.run_eval`` (Story 19.3).

Uses the existing mini corpus / fixture / stub embedder from
``test_eval_runner_integration`` and injects a stub ``hybrid_fn`` so no
real FAISS / BM25 / reranker loads.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from grounding.eval.fixtures import load_fixtures
from grounding.eval.runner import run_eval
from grounding.hybrid import HybridConfig
from grounding.reranker import RerankConfig
from grounding.vector_store import write_vector_index

from tests.test_eval_runner_integration import (
    AGENTS_DIR,
    MINI_CHUNKS,
    MINI_CORPUS,
    MINI_FIXTURE_YAML,
    stub_embed,
)


@pytest.fixture(scope="module")
def mini_embeddings_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("hybrid_mini_embeddings")
    embeddings = {c["chunk_id"]: stub_embed(c["text"]) for c in MINI_CHUNKS}
    chunk_meta = {
        c["chunk_id"]: {
            "doc_id": c["doc_id"],
            "file_path": f"{c['doc_id']}/chunks/{c['chunk_id']}.md",
            "is_music": False,
        }
        for c in MINI_CHUNKS
    }
    write_vector_index(embeddings, out, chunk_metadata=chunk_meta)
    return out


def _stub_hybrid_fn(hit_limit: int | None = None):
    """Return a stub hybrid_fn that returns MINI_CHUNKS (or a prefix) as hits."""
    chunks = MINI_CHUNKS if hit_limit is None else MINI_CHUNKS[:hit_limit]

    def stub(query: str, embeddings_dir: Path, *, top_k: int, pool_size: int, k_rrf: int) -> List[dict]:
        return [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "faiss_rank": i + 1,
                "bm25_rank": i + 1,
                "rrf_score": 1.0 / (k_rrf + i + 1) + 1.0 / (k_rrf + i + 1),
                "faiss_distance": 0.1 * (i + 1),
                "rank": i + 1,
            }
            for i, c in enumerate(chunks[:top_k])
        ]

    return stub


def test_run_eval_hybrid_config_none_calls_default_search(mini_embeddings_dir: Path):
    """BC: hybrid_config=None must use the dense search path."""
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)
    search_calls = []

    def tracking_search(index, chunk_map, query_embedding, top_k):
        search_calls.append(top_k)
        # Return empty; we don't care about correctness, only that it's called.
        return []

    def boom_hybrid(*a, **kw):
        raise AssertionError("hybrid_fn must not be called when hybrid_config=None")

    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=5,
        embed_fn=stub_embed,
        search_fn=tracking_search,
        hybrid_fn=boom_hybrid,
    )
    assert result.hybrid is None
    assert len(search_calls) > 0


def test_run_eval_hybrid_config_enabled_calls_hybrid_fn(mini_embeddings_dir: Path):
    """When hybrid is enabled, hybrid_fn is invoked and dense search is skipped."""
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)
    hybrid_calls = []

    def tracking_hybrid(query, embeddings_dir, *, top_k, pool_size, k_rrf):
        hybrid_calls.append({"top_k": top_k, "pool_size": pool_size, "k_rrf": k_rrf})
        return _stub_hybrid_fn()(
            query, embeddings_dir, top_k=top_k, pool_size=pool_size, k_rrf=k_rrf
        )

    def boom_search(*a, **kw):
        raise AssertionError("dense search must not be called when hybrid is on")

    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=5,
        embed_fn=stub_embed,
        search_fn=boom_search,
        hybrid_config=HybridConfig(enabled=True, pool_size=10, k_rrf=50),
        hybrid_fn=tracking_hybrid,
    )
    assert len(hybrid_calls) == len(fixture_set.items)
    assert hybrid_calls[0]["pool_size"] == 10
    assert hybrid_calls[0]["k_rrf"] == 50
    # pool_k = max(pool_size=10, 0, top_k=5) = 10
    assert hybrid_calls[0]["top_k"] == 10
    assert result.hybrid is not None


def test_run_eval_hybrid_plus_rerank_composition(mini_embeddings_dir: Path, monkeypatch):
    """When both are on, rerank receives the hybrid-produced pool."""
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)
    recorded = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        recorded.setdefault("pool_sizes", []).append(len(chunks))
        recorded.setdefault("chunk_ids", []).append([c["chunk_id"] for c in chunks])
        return [dict(c, rerank_score=1.0, score=1.0) for c in chunks]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    def boom_search(*a, **kw):
        raise AssertionError("dense search must not run when hybrid is on")

    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=3,
        embed_fn=stub_embed,
        search_fn=boom_search,
        hybrid_config=HybridConfig(enabled=True, pool_size=5, k_rrf=60),
        hybrid_fn=_stub_hybrid_fn(),
        rerank_config=RerankConfig(enabled=True, pool_size=4),
    )
    # pool_k = max(5, 4, 3) = 5; each fixture item sent 5 chunks to rerank
    assert all(n == 5 for n in recorded["pool_sizes"])
    # Hybrid order: MINI_CHUNKS[0..4] by chunk_id order.
    expected_order = [c["chunk_id"] for c in MINI_CHUNKS[:5]]
    assert recorded["chunk_ids"][0] == expected_order
    assert result.hybrid is not None


def test_run_eval_hybrid_config_populates_provenance(mini_embeddings_dir: Path):
    """Enabled hybrid run produces EvalRun.hybrid with pool_size + k_rrf."""
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)
    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=3,
        embed_fn=stub_embed,
        hybrid_config=HybridConfig(enabled=True, pool_size=7, k_rrf=42),
        hybrid_fn=_stub_hybrid_fn(),
    )
    assert result.hybrid is not None
    assert result.hybrid.enabled is True
    assert result.hybrid.pool_size == 7
    assert result.hybrid.k_rrf == 42

    # Disabled (None) run produces result.hybrid is None.
    result_off = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=3,
        embed_fn=stub_embed,
    )
    assert result_off.hybrid is None
