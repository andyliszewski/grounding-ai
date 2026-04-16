"""Tests for grounding.hybrid (Story 19.2).

All tests inject ``load_index_fn``, ``load_bm25_fn``, and ``embed_fn`` so
they never touch real FAISS, BM25, embedder, or disk. The dense channel
reaches into ``vector_store.search_similar_chunks``; we satisfy its minimal
interface with a tiny ``FakeFaissIndex``.

The fusion math test (RRF worked example) directly verifies AC #3 against
the inputs documented in Story 19.2 Dev Notes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from grounding.hybrid import HybridConfig, search_hybrid


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeFaissIndex:
    """Minimal FAISS-compatible stub.

    ``vector_store.search_similar_chunks`` reads ``ntotal``, ``d``, and
    calls ``search(query, k)`` returning ``(distances, indices)`` numpy
    arrays of shape (1, k). We pre-bake a ranked id list and synthesize
    distances 0.1, 0.2, ... so the relative order is stable.
    """

    def __init__(self, ranked_indices: List[int], dim: int = 4):
        self._ranked = ranked_indices
        self.ntotal = max(ranked_indices) + 1 if ranked_indices else 0
        self.d = dim

    def search(self, query: np.ndarray, k: int):
        ids = self._ranked[:k]
        distances = np.array([[0.1 * (i + 1) for i in range(len(ids))]], dtype=np.float32)
        # Pad with -1 if asked for more than we have, matching FAISS's behavior.
        if len(ids) < k:
            ids = ids + [-1] * (k - len(ids))
            pad = np.full((1, k - distances.shape[1]), 0.0, dtype=np.float32)
            distances = np.concatenate([distances, pad], axis=1)
        return distances, np.array([ids], dtype=np.int64)


def _make_v11_chunk_map(chunk_ids: List[str], tombstoned: Optional[set] = None) -> Dict:
    """v1.1+ FAISS chunk_map shape with optional tombstones."""
    tombstoned = tombstoned or set()
    chunks = []
    for i, cid in enumerate(chunk_ids):
        chunks.append(
            {
                "chunk_id": cid,
                "doc_id": f"doc_{cid}",
                "deleted_utc": "2026-01-01T00:00:00+00:00" if cid in tombstoned else None,
            }
        )
    return {
        "format_version": "1.2",
        "index_size": len(chunk_ids),
        "dimension": 4,
        "chunks": chunks,
    }


def _make_bm25_chunk_map(chunk_ids: List[str], tombstoned: Optional[set] = None) -> Dict:
    """BM25 map shape (Story 19.1)."""
    tombstoned = tombstoned or set()
    chunks = []
    for i, cid in enumerate(chunk_ids):
        chunks.append(
            {
                "bm25_index": i,
                "chunk_id": cid,
                "doc_id": f"doc_{cid}",
                "deleted_utc": "2026-01-01T00:00:00+00:00" if cid in tombstoned else None,
            }
        )
    return {
        "format_version": 1,
        "tokenizer": "whitespace_lowercase_v1",
        "rank_bm25_version": "0.2.2",
        "total": len(chunk_ids),
        "tombstone_count": len(tombstoned),
        "chunks": chunks,
    }


class FakeBM25Index:
    """Stand-in for ``grounding.bm25.BM25Index``.

    Only attributes the hybrid module touches: ``chunk_map`` and ``bm25``
    (presence-only — search is intercepted via the bm25 channel monkeypatch).
    """

    def __init__(self, chunk_map: Dict):
        self.chunk_map = chunk_map
        self.bm25 = object()  # truthy sentinel; not actually used in tests


def _make_load_index_fn(chunk_ids: List[str], ranked: List[int], tombstoned: Optional[set] = None):
    """Build a load_index_fn that returns (FakeFaissIndex, chunk_map).

    ``ranked`` is the order in which FakeFaissIndex returns indices for any
    query; ``chunk_ids[i]`` is the chunk id at FAISS position i.
    """
    chunk_map = _make_v11_chunk_map(chunk_ids, tombstoned=tombstoned)

    def loader(_index_dir: Path):
        return FakeFaissIndex(ranked, dim=4), chunk_map

    return loader


def _make_load_bm25_fn(
    chunk_ids: List[str],
    tombstoned: Optional[set] = None,
    *,
    return_none: bool = False,
):
    if return_none:
        def loader(_index_dir: Path):
            return None
        return loader

    chunk_map = _make_bm25_chunk_map(chunk_ids, tombstoned=tombstoned)
    fake = FakeBM25Index(chunk_map)

    def loader(_index_dir: Path):
        return fake

    return loader


def _make_bm25_search_stub(monkeypatch, ranked_chunk_ids: List[str]):
    """Patch grounding.hybrid.search_bm25-like behavior.

    Hybrid's _bm25_channel imports search_bm25 from grounding.bm25 inside
    the function body. We monkeypatch the canonical reference so the stub
    is picked up. Returns a list[(chunk_id, rank, score)] limited to top_k.
    """
    def fake_search_bm25(index, tokens, top_k):
        # Filter tombstones based on the index's chunk_map so the channel
        # mirrors real search_bm25's contract.
        tombstones = {
            c["chunk_id"]
            for c in index.chunk_map.get("chunks", [])
            if c.get("deleted_utc") is not None
        }
        out = []
        for cid in ranked_chunk_ids:
            if cid in tombstones:
                continue
            out.append((cid, len(out) + 1, 1.0 / (len(out) + 1)))
            if len(out) >= top_k:
                break
        return out

    monkeypatch.setattr("grounding.bm25.search_bm25", fake_search_bm25)


def _embed_zeros(_query: str) -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# AC #3 — RRF math worked example
# ---------------------------------------------------------------------------


def test_rrf_math_worked_example(monkeypatch):
    """B > A > E > C > D on the documented toy inputs."""
    chunk_ids = ["A", "B", "C", "D", "E"]
    # FAISS returns A, B, C, D in that order (positions 0,1,2,3 in chunk_ids).
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2, 3])
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["B", "E", "A"])

    out = search_hybrid(
        "anything",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        k_rrf=60,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )

    assert [r["chunk_id"] for r in out] == ["B", "A", "E", "C", "D"]
    assert [r["rank"] for r in out] == [1, 2, 3, 4, 5]

    expected = {
        "A": 1 / 61 + 1 / 63,
        "B": 1 / 62 + 1 / 61,
        "C": 1 / 63,
        "D": 1 / 64,
        "E": 1 / 62,
    }
    for r in out:
        assert r["rrf_score"] == pytest.approx(expected[r["chunk_id"]], rel=1e-9)
        assert "hybrid_degraded" not in r


# ---------------------------------------------------------------------------
# AC #5 — dense-only fallback
# ---------------------------------------------------------------------------


def test_dense_only_fallback_marks_results_degraded(caplog):
    chunk_ids = ["A", "B", "C"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2])
    load_bm25 = _make_load_bm25_fn([], return_none=True)

    with caplog.at_level(logging.WARNING, logger="grounding.hybrid"):
        out = search_hybrid(
            "q",
            Path("/tmp/unused"),
            top_k=5,
            pool_size=10,
            k_rrf=60,
            load_index_fn=load_index,
            load_bm25_fn=load_bm25,
            embed_fn=_embed_zeros,
        )

    assert [r["chunk_id"] for r in out] == ["A", "B", "C"]
    for i, r in enumerate(out):
        assert r["bm25_rank"] is None
        assert r["faiss_rank"] == i + 1
        assert r["rrf_score"] == pytest.approx(1 / (60 + (i + 1)))
        assert r["hybrid_degraded"] is True
        assert r["doc_id"] == f"doc_{r['chunk_id']}"

    matching = [rec for rec in caplog.records if "BM25 artifacts missing" in rec.message]
    assert len(matching) == 1


def test_dense_only_fallback_filters_faiss_tombstones():
    """Dense-only path must still drop FAISS-tombstoned chunks (QA TEST-001).

    When BM25 artifacts are absent we skip the cross-channel union, but the
    FAISS chunk_map's own tombstones are still authoritative — a chunk
    flagged ``deleted_utc`` there should not appear in dense-only results.
    """
    chunk_ids = ["A", "B", "C"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2], tombstoned={"B"})
    load_bm25 = _make_load_bm25_fn([], return_none=True)

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    chunk_ids_out = [r["chunk_id"] for r in out]
    assert "B" not in chunk_ids_out
    assert chunk_ids_out == ["A", "C"]
    for r in out:
        assert r["hybrid_degraded"] is True


# ---------------------------------------------------------------------------
# AC #8 — both channels empty
# ---------------------------------------------------------------------------


def test_both_channels_empty_returns_empty_list(monkeypatch):
    load_index = _make_load_index_fn([], ranked=[])
    load_bm25 = _make_load_bm25_fn([])
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=[])

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    assert out == []


# ---------------------------------------------------------------------------
# AC #4 — pool/top_k truncation
# ---------------------------------------------------------------------------


def test_pool_size_truncates_to_top_k(monkeypatch):
    chunk_ids = ["A", "B", "C", "D", "E"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2, 3, 4])
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["A", "B", "C", "D", "E"])

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=2,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    assert len(out) == 2
    assert out[0]["rank"] == 1
    assert out[1]["rank"] == 2


# ---------------------------------------------------------------------------
# AC #6 — tombstone cross-check
# ---------------------------------------------------------------------------


def test_cross_channel_tombstone_filters_candidate(monkeypatch):
    """A chunk tombstoned in either map is dropped from merge."""
    chunk_ids = ["A", "B", "C"]
    # FAISS has B tombstoned; BM25 ranks B #1.
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2], tombstoned={"B"})
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["B", "A", "C"])

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    chunk_ids_out = [r["chunk_id"] for r in out]
    assert "B" not in chunk_ids_out
    assert set(chunk_ids_out) == {"A", "C"}


def test_asymmetric_tombstone_logs_once_and_drops(monkeypatch, caplog):
    """When BM25 tombstones a chunk that FAISS did not, drop and log DEBUG once."""
    chunk_ids = ["A", "B", "C"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1, 2])
    # BM25 tombstones B; the BM25 search stub already filters it out, but
    # FAISS will surface B at rank 2 → cross-channel drop.
    load_bm25 = _make_load_bm25_fn(chunk_ids, tombstoned={"B"})
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["A", "C"])

    with caplog.at_level(logging.DEBUG, logger="grounding.hybrid"):
        out = search_hybrid(
            "q",
            Path("/tmp/unused"),
            top_k=5,
            pool_size=10,
            load_index_fn=load_index,
            load_bm25_fn=load_bm25,
            embed_fn=_embed_zeros,
        )

    assert "B" not in [r["chunk_id"] for r in out]
    cross_logs = [
        rec for rec in caplog.records if "cross-channel tombstone" in rec.message
    ]
    assert len(cross_logs) == 1


# ---------------------------------------------------------------------------
# AC #7 — empty-query tokens
# ---------------------------------------------------------------------------


def test_empty_query_tokens_runs_dense_only_without_degraded_flag(monkeypatch):
    chunk_ids = ["A", "B"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1])
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=[])

    # Empty/punctuation-only query → tokenize() yields []; bm25 channel
    # contributes nothing; dense channel still runs.
    out = search_hybrid(
        "   ",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    assert [r["chunk_id"] for r in out] == ["A", "B"]
    for r in out:
        assert "hybrid_degraded" not in r
        assert r["bm25_rank"] is None
        assert r["faiss_rank"] is not None


# ---------------------------------------------------------------------------
# AC #3 (cont.) — tie-break is lexicographic chunk_id
# ---------------------------------------------------------------------------


def test_tie_break_is_lexicographic_chunk_id(monkeypatch):
    """Two chunks with identical RRF scores → smaller chunk_id wins."""
    # Both Z and A appear only in FAISS at rank 1 → impossible (one rank
    # per channel per chunk). Instead: Z at FAISS rank 1, A at BM25 rank 1.
    # Both have identical rrf_score = 1/61. Tie-break: A < Z.
    chunk_ids = ["A", "Z"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[1])  # FAISS returns Z first
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["A"])

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=5,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    assert [r["chunk_id"] for r in out] == ["A", "Z"]
    assert out[0]["rrf_score"] == pytest.approx(out[1]["rrf_score"])


# ---------------------------------------------------------------------------
# AC #9 — no real FAISS / BM25 / embedder access
# ---------------------------------------------------------------------------


def test_pure_no_real_faiss_or_bm25_access_with_injected_fns(monkeypatch):
    """Sanity: with all three fns injected, none of the real production
    callables (which would hit disk / import torch) are invoked."""
    real_calls = {"load_vector_index": 0, "generate_embedding": 0, "load_bm25_index": 0}

    def boom_load_vector(*a, **kw):
        real_calls["load_vector_index"] += 1
        raise AssertionError("real load_vector_index should not be called")

    def boom_generate_embedding(*a, **kw):
        real_calls["generate_embedding"] += 1
        raise AssertionError("real generate_embedding should not be called")

    def boom_load_bm25(*a, **kw):
        real_calls["load_bm25_index"] += 1
        raise AssertionError("real load_bm25_index should not be called")

    monkeypatch.setattr("grounding.vector_store.load_vector_index", boom_load_vector)
    monkeypatch.setattr("grounding.embedder.generate_embedding", boom_generate_embedding)
    monkeypatch.setattr("grounding.bm25.load_bm25_index", boom_load_bm25)

    chunk_ids = ["A", "B"]
    load_index = _make_load_index_fn(chunk_ids, ranked=[0, 1])
    load_bm25 = _make_load_bm25_fn(chunk_ids)
    _make_bm25_search_stub(monkeypatch, ranked_chunk_ids=["A", "B"])

    out = search_hybrid(
        "q",
        Path("/tmp/unused"),
        top_k=2,
        pool_size=10,
        load_index_fn=load_index,
        load_bm25_fn=load_bm25,
        embed_fn=_embed_zeros,
    )
    assert len(out) == 2
    assert all(v == 0 for v in real_calls.values())


# ---------------------------------------------------------------------------
# HybridConfig validation
# ---------------------------------------------------------------------------


def test_hybrid_config_defaults_and_validation():
    cfg = HybridConfig()
    assert cfg.enabled is False
    assert cfg.pool_size == 50
    assert cfg.k_rrf == 60
    cfg.validate()  # no-op on defaults

    with pytest.raises(ValueError):
        HybridConfig(pool_size=0).validate()
    with pytest.raises(ValueError):
        HybridConfig(k_rrf=0).validate()
