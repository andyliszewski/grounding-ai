"""Unit tests for the FAISS/BM25 coherence check (Story 23.3).

Fast: builds tiny FAISS + BM25 indexes directly (no embedding model) and
exercises ``check_faiss_bm25_coherence`` against coherent and half-applied
states (W1 FAISS-ahead, W2 BM25-absent).
"""
from __future__ import annotations

import numpy as np

from grounding.cli import check_faiss_bm25_coherence
from grounding.vector_store import write_vector_index, append_to_vector_index
from grounding.bm25 import (
    write_bm25_index,
    BM25_PICKLE_FILENAME,
    BM25_MAP_FILENAME,
)


def _build_pair(index_dir, n):
    embs = {f"c{i}": np.random.rand(384).astype("float32") for i in range(n)}
    meta = {
        cid: {"doc_id": "d", "file_path": f"d/chunks/ch_{i:04d}.md"}
        for i, cid in enumerate(embs)
    }
    write_vector_index(embs, index_dir, meta)
    ids = list(embs.keys())
    write_bm25_index(
        [f"body number {i}" for i in range(n)], ids, index_dir, chunk_doc_ids=["d"] * n
    )
    return ids


def test_coherent_pair(tmp_path):
    _build_pair(tmp_path, 3)
    coherent, reason = check_faiss_bm25_coherence(tmp_path)
    assert coherent
    assert reason == ""


def test_no_faiss_index_is_coherent(tmp_path):
    # Nothing to reconcile; the full-build path owns a fresh index.
    coherent, _ = check_faiss_bm25_coherence(tmp_path)
    assert coherent


def test_bm25_absent_is_incoherent(tmp_path):
    """W2: FAISS present, BM25 sidecar missing."""
    _build_pair(tmp_path, 3)
    (tmp_path / BM25_PICKLE_FILENAME).unlink()
    (tmp_path / BM25_MAP_FILENAME).unlink()

    coherent, reason = check_faiss_bm25_coherence(tmp_path)
    assert not coherent
    assert "missing" in reason.lower()


def test_faiss_ahead_of_bm25_is_incoherent(tmp_path):
    """W1: an append advanced FAISS but BM25 did not."""
    _build_pair(tmp_path, 3)
    append_to_vector_index(
        {"extra": np.random.rand(384).astype("float32")},
        tmp_path,
        {"extra": {"doc_id": "d", "file_path": "d/chunks/ch_0099.md"}},
    )

    coherent, reason = check_faiss_bm25_coherence(tmp_path)
    assert not coherent
    assert "4" in reason and "3" in reason
