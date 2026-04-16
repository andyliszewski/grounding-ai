"""Hybrid retrieval: FAISS dense + BM25 lexical, fused via Reciprocal Rank Fusion.

Two-channel candidate pipeline:

    dense    = FAISS top-N (bi-encoder cosine over MiniLM embeddings)
    lexical  = BM25 top-N (rank_bm25.BM25Okapi over whitespace_lowercase tokens)
    merged   = RRF(dense, lexical, k_rrf=60) → sort desc → truncate to top_k

The fusion math (Cormack et al. 2009)::

    rrf_score(d) = Σ over channels c where d appears
                       1 / (k_rrf + rank_c(d))

Documents missing from a channel contribute 0 for that channel; ranks are
1-indexed. ``k_rrf`` (default 60) damps the contribution of high-rank
candidates so that a chunk only ranked by one channel can still surface if
that channel ranked it #1 or #2.

Worked example (k_rrf=60)::

    FAISS:  [A, B, C, D]   ranks 1, 2, 3, 4
    BM25:   [B, E, A]      ranks 1, 2, 3

    RRF(A) = 1/61 + 1/63   = 0.032266
    RRF(B) = 1/62 + 1/61   = 0.032522
    RRF(C) = 1/63          = 0.015873
    RRF(D) = 1/64          = 0.015625
    RRF(E) = 1/62          = 0.016129

    Final order: B, A, E, C, D

Dense-only fallback: if BM25 artifacts are missing at ``index_dir`` (e.g.
agent embedded before Story 19.1 shipped, or a partial rebuild), the function
runs the dense channel only, sets ``bm25_rank=None`` on every result,
computes ``rrf_score = 1/(k_rrf + faiss_rank)``, marks each result with
``hybrid_degraded: True``, and logs one WARNING. Empty query tokens trigger
a different path: BM25 contributes nothing but no degradation is logged
(this is normal behavior for stopword-only queries).

Return shape (per dict):

    {
      "chunk_id":       str,
      "doc_id":         str | None,
      "faiss_rank":     int | None,
      "bm25_rank":      int | None,
      "rrf_score":      float,
      "faiss_distance": float | None,
      "rank":           int,           # 1-indexed post-fusion
      "hybrid_degraded": True,         # only present in dense-only fallback
    }

No body text, no front matter — enrichment is the caller's job (matches the
``vector_store.search_similar_chunks`` contract). Story 19.3 layers the
existing chunk-reading helpers on top.

Pure function: all expensive resources (FAISS index, BM25 index, embedding
model) are pulled in via injected ``*_fn`` kwargs that default to the real
production callables. Tests pass stubs and never touch disk or models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("grounding.hybrid")

DEFAULT_K_RRF = 60


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid retrieval.

    Declared in 19.2 so Story 19.3 can import a stable surface when wiring
    CLI flags / config.yaml / MCP schema. Enablement stays off by default;
    19.3 owns the flip and the surface plumbing.
    """

    enabled: bool = False
    pool_size: int = 50
    k_rrf: int = DEFAULT_K_RRF

    def validate(self) -> None:
        if self.pool_size < 1:
            raise ValueError(
                f"HybridConfig.pool_size must be >= 1, got {self.pool_size}"
            )
        if self.k_rrf < 1:
            raise ValueError(
                f"HybridConfig.k_rrf must be >= 1, got {self.k_rrf}"
            )


def _default_load_index_fn(index_dir: Path):
    from grounding.vector_store import load_vector_index

    return load_vector_index(index_dir)


def _default_load_bm25_fn(index_dir: Path):
    from grounding.bm25 import load_bm25_index

    return load_bm25_index(index_dir)


def _default_embed_fn(query: str):
    from grounding.embedder import generate_embedding

    return generate_embedding(query)


def _extract_tombstones_and_docids(chunk_map: Dict) -> Tuple[set, Dict[str, Optional[str]]]:
    """Extract (tombstoned chunk_ids, chunk_id → doc_id map) from a chunk_map.

    Works against both the FAISS v1.1+ chunk_map and the BM25 19.1 map: both
    keep ``{chunk_id, doc_id, deleted_utc}`` per entry under ``chunks[]``.
    The v1.0 FAISS format (no per-chunk metadata) yields an empty tombstone
    set and an empty doc_id map — same as if no tombstones existed.
    """
    tombstones: set = set()
    doc_ids: Dict[str, Optional[str]] = {}
    for chunk in chunk_map.get("chunks", []) or []:
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        if chunk.get("deleted_utc") is not None:
            tombstones.add(cid)
        doc_ids[cid] = chunk.get("doc_id")
    return tombstones, doc_ids


def _dense_channel(
    query: str,
    index_dir: Path,
    pool_size: int,
    embed_fn: Callable[[str], Any],
    load_index_fn: Callable[[Path], Tuple[Any, Dict]],
) -> Tuple[List[Tuple[str, int, float]], Dict]:
    """Run the dense channel; return ((chunk_id, rank, distance) list, chunk_map).

    Wraps ``vector_store.load_vector_index`` + ``search_similar_chunks``.
    Both injected functions default to the real implementations; tests pass
    stubs that synthesize ranked results without touching FAISS or torch.
    """
    from grounding.vector_store import search_similar_chunks

    index, chunk_map = load_index_fn(index_dir)
    query_vec = embed_fn(query)
    pairs = search_similar_chunks(index, chunk_map, query_vec, top_k=pool_size)
    results = [(cid, i + 1, float(dist)) for i, (cid, dist) in enumerate(pairs)]
    return results, chunk_map


def _bm25_channel(
    query: str,
    bm25_index: Any,
    pool_size: int,
) -> List[Tuple[str, int, float]]:
    """Run the BM25 channel; return (chunk_id, rank, score) list.

    Empty query-token lists produce an empty result (NOT a degraded marker);
    that is per-AC normal behavior for stopword-only queries.
    """
    from grounding.bm25 import search_bm25, tokenize

    tokens = tokenize(query)
    if not tokens:
        return []
    return search_bm25(bm25_index, tokens, top_k=pool_size)


def search_hybrid(
    query: str,
    index_dir: Path,
    *,
    top_k: int,
    pool_size: int,
    k_rrf: int = DEFAULT_K_RRF,
    load_index_fn: Optional[Callable[[Path], Tuple[Any, Dict]]] = None,
    load_bm25_fn: Optional[Callable[[Path], Any]] = None,
    embed_fn: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, Any]]:
    """Hybrid retrieval over a single agent index directory.

    See module docstring for fusion math, return shape, and the dense-only
    fallback semantics. Three injected callables let tests replace FAISS,
    BM25, and the embedder with stubs.

    Args:
        query: User query string.
        index_dir: Directory containing FAISS + BM25 artifacts (typically
            ``embeddings/<agent>/``).
        top_k: Number of fused results to return.
        pool_size: Per-channel candidate count (so up to 2*pool_size unique
            chunks enter the merge).
        k_rrf: RRF damping constant (default 60).
        load_index_fn: Override for ``vector_store.load_vector_index``.
        load_bm25_fn: Override for ``grounding.bm25.load_bm25_index``.
        embed_fn: Override for ``grounding.embedder.generate_embedding``.

    Returns:
        List of result dicts (see module docstring for shape), 1-indexed
        ``rank`` field, sorted by descending ``rrf_score`` with lexicographic
        ``chunk_id`` tie-break. Empty list if both channels return nothing.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if k_rrf < 1:
        raise ValueError(f"k_rrf must be >= 1, got {k_rrf}")

    load_index_fn = load_index_fn or _default_load_index_fn
    load_bm25_fn = load_bm25_fn or _default_load_bm25_fn
    embed_fn = embed_fn or _default_embed_fn

    index_dir = Path(index_dir)

    dense_results, faiss_chunk_map = _dense_channel(
        query, index_dir, pool_size, embed_fn, load_index_fn
    )
    faiss_tombstones, faiss_doc_ids = _extract_tombstones_and_docids(faiss_chunk_map)

    bm25_index = load_bm25_fn(index_dir)

    if bm25_index is None:
        logger.warning(
            "BM25 artifacts missing at %s; running dense-only", index_dir
        )
        return _dense_only_results(
            dense_results, faiss_doc_ids, k_rrf, top_k, faiss_tombstones
        )

    bm25_results = _bm25_channel(query, bm25_index, pool_size)
    bm25_tombstones, bm25_doc_ids = _extract_tombstones_and_docids(bm25_index.chunk_map)

    forbidden = faiss_tombstones | bm25_tombstones
    cross_dropped = _count_cross_channel_drops(
        dense_results, bm25_results, faiss_tombstones, bm25_tombstones
    )
    if cross_dropped:
        logger.debug(
            "hybrid: dropped %d candidate(s) due to cross-channel tombstone "
            "inconsistency at %s",
            cross_dropped,
            index_dir,
        )

    merged: Dict[str, Dict[str, Any]] = {}
    for chunk_id, rank, distance in dense_results:
        if chunk_id in forbidden:
            continue
        entry = merged.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "doc_id": faiss_doc_ids.get(chunk_id) or bm25_doc_ids.get(chunk_id),
                "faiss_rank": None,
                "bm25_rank": None,
                "faiss_distance": None,
                "rrf_score": 0.0,
            },
        )
        entry["faiss_rank"] = rank
        entry["faiss_distance"] = distance
        entry["rrf_score"] += 1.0 / (k_rrf + rank)

    for chunk_id, rank, _score in bm25_results:
        if chunk_id in forbidden:
            continue
        entry = merged.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "doc_id": faiss_doc_ids.get(chunk_id) or bm25_doc_ids.get(chunk_id),
                "faiss_rank": None,
                "bm25_rank": None,
                "faiss_distance": None,
                "rrf_score": 0.0,
            },
        )
        entry["bm25_rank"] = rank
        entry["rrf_score"] += 1.0 / (k_rrf + rank)

    if not merged:
        return []

    ordered = sorted(
        merged.values(), key=lambda e: (-e["rrf_score"], e["chunk_id"])
    )
    truncated = ordered[:top_k]
    for i, entry in enumerate(truncated, start=1):
        entry["rank"] = i
    return truncated


def _dense_only_results(
    dense_results: List[Tuple[str, int, float]],
    doc_ids: Dict[str, Optional[str]],
    k_rrf: int,
    top_k: int,
    tombstones: set,
) -> List[Dict[str, Any]]:
    """Build dense-only result dicts when BM25 artifacts are missing.

    Same dict shape as the fused path, but ``bm25_rank=None`` everywhere and
    ``hybrid_degraded=True`` flagged on each result so callers can surface
    the degradation upstream (e.g. for telemetry).
    """
    out: List[Dict[str, Any]] = []
    for chunk_id, rank, distance in dense_results:
        if chunk_id in tombstones:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_ids.get(chunk_id),
                "faiss_rank": rank,
                "bm25_rank": None,
                "faiss_distance": distance,
                "rrf_score": 1.0 / (k_rrf + rank),
                "hybrid_degraded": True,
            }
        )
    out.sort(key=lambda e: (-e["rrf_score"], e["chunk_id"]))
    truncated = out[:top_k]
    for i, entry in enumerate(truncated, start=1):
        entry["rank"] = i
    return truncated


def _count_cross_channel_drops(
    dense_results: List[Tuple[str, int, float]],
    bm25_results: List[Tuple[str, int, float]],
    faiss_tombstones: set,
    bm25_tombstones: set,
) -> int:
    """Count candidates dropped due to cross-channel tombstone inconsistency.

    A "cross-channel drop" is a candidate one channel surfaced that the
    other channel has tombstoned. Counted once per (chunk_id, channel) pair.
    Used only for the DEBUG log; does not affect the final ordering.
    """
    count = 0
    for chunk_id, _rank, _dist in dense_results:
        if chunk_id in bm25_tombstones and chunk_id not in faiss_tombstones:
            count += 1
    for chunk_id, _rank, _score in bm25_results:
        if chunk_id in faiss_tombstones and chunk_id not in bm25_tombstones:
            count += 1
    return count
