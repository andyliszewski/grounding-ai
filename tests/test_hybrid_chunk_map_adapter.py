"""Regression tests for the hybrid chunk-map adapter (Story 19.5).

The bug: the three retrieval surfaces cache the chunk_map as a bare list
(discarding ``format_version``) and re-wrapped it as ``{"chunks": list}``
before handing it to ``search_hybrid``. ``search_similar_chunks`` then
defaulted ``format_version`` to ``"1.0"``, read the absent ``chunk_ids``
list, and skipped every FAISS hit -> the dense channel returned nothing.

These tests deliberately drive the REAL ``search_hybrid`` /
``search_similar_chunks`` path (the existing hybrid suites stub
``search_hybrid`` entirely, which is why the regression slipped through).
Each of the integration tests fails against pre-19.5 code (empty dense
channel) and passes once ``adapt_chunk_map_for_search`` stamps the
metadata ``format_version``.
"""
from __future__ import annotations

import numpy as np
import pytest

from grounding.vector_store import (
    FORMAT_VERSION_INCREMENTAL,
    adapt_chunk_map_for_search,
    search_similar_chunks,
)
from grounding.hybrid import search_hybrid


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

BARE_LIST = [
    {"file_path": "doc-a/chunks/ch_0001.md", "chunk_id": "doc-a-1", "doc_id": "doc-a"},
    {"file_path": "doc-b/chunks/ch_0001.md", "chunk_id": "doc-b-1", "doc_id": "doc-b"},
    {"file_path": "doc-c/chunks/ch_0001.md", "chunk_id": "doc-c-1", "doc_id": "doc-c"},
]


class FakeIndex:
    """Minimal FAISS stand-in for ``search_similar_chunks``."""

    def __init__(self, indices, distances, dim=4):
        self.ntotal = len(indices)
        self.d = dim
        self._indices = np.array([indices])
        self._distances = np.array([distances])

    def search(self, query, k):
        return self._distances[:, :k], self._indices[:, :k]


def _embed(_query):
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


def _fake_index():
    return FakeIndex(indices=[0, 1, 2], distances=[0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Unit tests for the helper
# ---------------------------------------------------------------------------


def test_bare_list_of_dicts_gets_metadata_format_version():
    adapted = adapt_chunk_map_for_search(BARE_LIST)
    assert adapted["format_version"] == FORMAT_VERSION_INCREMENTAL
    assert adapted["chunks"] is BARE_LIST


def test_dict_without_format_version_is_stamped():
    adapted = adapt_chunk_map_for_search({"chunks": BARE_LIST})
    assert adapted["format_version"] == FORMAT_VERSION_INCREMENTAL


def test_dict_with_format_version_passes_through_unchanged():
    src = {"chunks": BARE_LIST, "format_version": "1.1", "index_size": 3}
    assert adapt_chunk_map_for_search(src) is src


def test_bare_string_list_is_not_stamped_with_metadata_version():
    # Oldest path-only format: no chunk_id, cannot fuse; must not crash and
    # must not pretend to be a metadata map.
    legacy = ["doc-a/chunks/ch_0001.md", "doc-b/chunks/ch_0001.md"]
    adapted = adapt_chunk_map_for_search(legacy)
    assert adapted == {"chunks": legacy}
    assert "format_version" not in adapted


def test_empty_list_does_not_index_error():
    assert adapt_chunk_map_for_search([]) == {"chunks": []}


# ---------------------------------------------------------------------------
# The tight regression: adapter output must drive a non-empty dense result
# ---------------------------------------------------------------------------


def test_search_similar_chunks_on_adapted_map_returns_hits():
    """Pre-19.5 this returned [] (v1.0 branch, empty chunk_ids)."""
    adapted = adapt_chunk_map_for_search(BARE_LIST)
    results = search_similar_chunks(_fake_index(), adapted, _embed("q"), top_k=3)
    assert [cid for cid, _dist in results] == ["doc-a-1", "doc-b-1", "doc-c-1"]


def test_bare_list_without_adapter_would_return_nothing():
    """Documents the bug: the naive re-wrap yields zero dense hits."""
    naive = {"chunks": BARE_LIST}  # what the old inline adapter produced
    naive_no_fmt = dict(naive)
    naive_no_fmt.pop("format_version", None)
    results = search_similar_chunks(_fake_index(), naive_no_fmt, _embed("q"), top_k=3)
    assert results == []


# ---------------------------------------------------------------------------
# AC 2 / AC 3: real search_hybrid through the adapter
# ---------------------------------------------------------------------------


def test_hybrid_with_bm25_present_populates_faiss_rank(monkeypatch):
    """AC 2: fused path returns dense candidates with non-null faiss_rank."""

    class FakeBM25:
        chunk_map = {"chunks": []}

    monkeypatch.setattr("grounding.bm25.search_bm25", lambda *a, **k: [])

    hits = search_hybrid(
        "bootstrap methods",
        "/unused",
        top_k=3,
        pool_size=3,
        load_index_fn=lambda _d: (_fake_index(), adapt_chunk_map_for_search(BARE_LIST)),
        load_bm25_fn=lambda _d: FakeBM25(),
        embed_fn=_embed,
    )
    assert hits, "fused hybrid path returned no results (dense channel empty?)"
    assert any(h["faiss_rank"] is not None for h in hits)
    assert all("hybrid_degraded" not in h for h in hits)


def test_hybrid_with_bm25_absent_degrades_not_empties(monkeypatch):
    """AC 3: BM25 missing -> dense results, each marked hybrid_degraded."""
    hits = search_hybrid(
        "bootstrap methods",
        "/unused",
        top_k=3,
        pool_size=3,
        load_index_fn=lambda _d: (_fake_index(), adapt_chunk_map_for_search(BARE_LIST)),
        load_bm25_fn=lambda _d: None,
        embed_fn=_embed,
    )
    assert hits, "dense-only fallback returned no results (dense channel empty?)"
    assert all(h.get("hybrid_degraded") is True for h in hits)
    assert all(h["faiss_rank"] is not None for h in hits)
