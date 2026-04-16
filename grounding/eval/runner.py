"""Eval runner (Story 16.2).

Executes fixture queries against an agent's FAISS index, computes per-item
and aggregate retrieval metrics, and returns a structured ``EvalRun``.

The runner reuses the existing retrieval path:
- ``grounding.embedder.generate_embedding`` for query embedding
- ``grounding.vector_store.load_vector_index`` + ``search_similar_chunks``
- ``grounding.manifest.ManifestManager`` for corpus validation

For testability, the embed and search functions can be injected; the
default implementations delegate to the real modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml

from grounding.eval.fixtures import FixtureSet
from grounding.eval.metrics import CitationCase, citation_accuracy, mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger("grounding.eval.runner")

# Type aliases for injected callables (kept simple to avoid numpy import cost here).
EmbedFn = Callable[[str], "object"]  # returns np.ndarray at runtime
SearchFn = Callable[[object, Dict, object, int], List[Tuple[str, float]]]


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    rank: int  # 1-indexed
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None


@dataclass(frozen=True)
class EvalItemResult:
    item_id: str
    query: str
    expected_doc_ids: Tuple[str, ...]
    retrieved: Tuple[RetrievedChunk, ...]
    first_hit_rank: int | None
    strict_first_hit_rank: int | None
    tags: Tuple[str, ...]
    expected_page: int | Tuple[int, int] | None = None
    expected_section: str | None = None


@dataclass(frozen=True)
class TagMetrics:
    recall_at_5: float
    mrr: float
    n_items: int
    low_sample: bool


@dataclass(frozen=True)
class EvalAggregate:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    per_tag: Dict[str, TagMetrics] = field(default_factory=dict)
    citation_accuracy: float | None = None
    n_citation_items: int = 0


@dataclass(frozen=True)
class RerankProvenance:
    """Snapshot of the RerankConfig that was in effect for an EvalRun.

    Serialized into JSON reports so baselines carry unambiguous provenance
    of which retrieval mode produced the numbers. Populated by ``run_eval``
    whenever a ``rerank_config`` is supplied.
    """
    enabled: bool
    model: str
    pool_size: int
    batch_size: int


@dataclass(frozen=True)
class HybridProvenance:
    """Snapshot of the HybridConfig that was in effect for an EvalRun.

    Mirror of :class:`RerankProvenance`. Populated by ``run_eval`` whenever
    a ``hybrid_config`` is supplied; ``None`` otherwise. Story 19.4 reads
    this back out of the JSON report to label the `{hybrid × rerank}`
    matrix rows unambiguously.
    """
    enabled: bool
    pool_size: int
    k_rrf: int


@dataclass(frozen=True)
class EvalRun:
    agent: str
    fixture_path: Path
    top_k: int
    items: Tuple[EvalItemResult, ...]
    aggregate: EvalAggregate
    skipped: Tuple[str, ...]
    started_utc: str
    finished_utc: str
    rerank: RerankProvenance | None = None
    hybrid: HybridProvenance | None = None


# ---------------------------------------------------------------------------
# Default embed/search wiring
# ---------------------------------------------------------------------------

def _default_embed(text: str):
    from grounding.embedder import generate_embedding

    return generate_embedding(text)


def _default_search(index, chunk_map: Dict, query_embedding, top_k: int):
    from grounding.vector_store import search_similar_chunks

    return search_similar_chunks(index, chunk_map, query_embedding, top_k)


def _default_load_index(embeddings_dir: Path):
    from grounding.vector_store import load_vector_index

    return load_vector_index(embeddings_dir)


def _default_load_manifest_doc_ids(corpus_dir: Path) -> set[str]:
    from grounding.manifest import ManifestManager

    manifest = ManifestManager.load(corpus_dir / "_index.json")
    return {doc.doc_id for doc in manifest.docs}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _default_hybrid_search(
    query: str,
    embeddings_dir: Path,
    *,
    top_k: int,
    pool_size: int,
    k_rrf: int,
) -> List[Dict]:
    """Thin wrapper around ``grounding.hybrid.search_hybrid`` for injection.

    Mirrors the ``_default_search`` / ``_default_embed`` pattern so tests can
    substitute a stub without touching FAISS, BM25, or the embedder.
    """
    from grounding.hybrid import search_hybrid

    return search_hybrid(
        query,
        embeddings_dir,
        top_k=top_k,
        pool_size=pool_size,
        k_rrf=k_rrf,
    )


def run_eval(
    fixture_set: FixtureSet,
    agent_name: str,
    *,
    corpus_dir: Path,
    embeddings_dir: Path,
    top_k: int = 10,
    embed_fn: EmbedFn | None = None,
    search_fn: SearchFn | None = None,
    load_index_fn: Callable[[Path], Tuple[object, Dict]] | None = None,
    load_manifest_doc_ids_fn: Callable[[Path], set[str]] | None = None,
    rerank_config: "RerankConfig | None" = None,
    hybrid_config: "HybridConfig | None" = None,
    hybrid_fn: Callable[..., List[Dict]] | None = None,
) -> EvalRun:
    """Run fixture queries against an agent's FAISS index.

    Args:
        fixture_set: Parsed fixture (from ``grounding.eval.load_fixtures``).
        agent_name: Agent to evaluate. Must match ``fixture_set.agent``.
        corpus_dir: Directory containing ``_index.json`` (corpus manifest).
        embeddings_dir: Directory containing the agent's FAISS index
            (e.g. ``embeddings/<agent_name>/``).
        top_k: Retrieval cutoff (default 10).
        embed_fn, search_fn, load_index_fn, load_manifest_doc_ids_fn:
            Injection points for testing. If None, the real modules are used.

    Returns:
        EvalRun with per-item results and aggregate metrics.

    Raises:
        ValueError: If ``agent_name`` does not match ``fixture_set.agent``.
    """
    if agent_name != fixture_set.agent:
        raise ValueError(
            f"agent mismatch: fixture targets '{fixture_set.agent}' "
            f"but run_eval called with '{agent_name}'"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    embed = embed_fn or _default_embed
    search = search_fn or _default_search
    load_index = load_index_fn or _default_load_index
    load_manifest_ids = load_manifest_doc_ids_fn or _default_load_manifest_doc_ids
    hybrid_search = hybrid_fn or _default_hybrid_search

    rerank_on = bool(rerank_config and rerank_config.enabled)
    hybrid_on = bool(hybrid_config and hybrid_config.enabled)
    if hybrid_on:
        # Pool sized to the widest of the three so rerank has the full pool.
        fetch_k_for_hybrid = max(
            hybrid_config.pool_size,
            rerank_config.pool_size if rerank_on else 0,
            top_k,
        )
    else:
        fetch_k_for_hybrid = top_k
    fetch_k = max(rerank_config.pool_size, top_k) if rerank_on else top_k

    started = _utc_now_iso()

    index, chunk_map = load_index(embeddings_dir)
    chunk_to_doc = _build_chunk_to_doc(chunk_map)
    chunk_to_path = _build_chunk_to_path(chunk_map)
    manifest_doc_ids = load_manifest_ids(corpus_dir)

    items: list[EvalItemResult] = []
    skipped: list[str] = []

    for fixture_item in fixture_set.items:
        expected_set = set(fixture_item.expected.doc_ids)
        unknown = expected_set - manifest_doc_ids
        if unknown:
            logger.warning(
                "fixture item %s references unknown doc_id(s) %s; skipping",
                fixture_item.id,
                sorted(unknown),
            )
            skipped.append(fixture_item.id)
            continue

        if hybrid_on:
            hybrid_hits = hybrid_search(
                fixture_item.query,
                embeddings_dir,
                top_k=fetch_k_for_hybrid,
                pool_size=hybrid_config.pool_size,
                k_rrf=hybrid_config.k_rrf,
            )
            raw_hits = [
                (h["chunk_id"], float(h.get("rrf_score", 0.0)))
                for h in hybrid_hits
            ]
        else:
            query_embedding = embed(fixture_item.query)
            raw_hits = search(index, chunk_map, query_embedding, fetch_k)

        if rerank_on and raw_hits:
            raw_hits = _apply_rerank(
                query=fixture_item.query,
                raw_hits=raw_hits,
                chunk_to_path=chunk_to_path,
                corpus_dir=corpus_dir,
                rerank_config=rerank_config,
                top_k=top_k,
            )
        elif hybrid_on:
            # Hybrid fetched a wider pool than top_k (so a future rerank
            # would see the full pool); when rerank is off, truncate here.
            raw_hits = raw_hits[:top_k]

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(raw_hits, start=1):
            doc_id = chunk_to_doc.get(chunk_id, "")
            rel_path = chunk_to_path.get(chunk_id)
            if rel_path:
                ps, pe, sh = _read_chunk_citation_metadata(corpus_dir, rel_path)
            else:
                ps, pe, sh = (None, None, None)
            retrieved.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=float(score),
                    rank=rank,
                    page_start=ps,
                    page_end=pe,
                    section_heading=sh,
                )
            )

        first_hit = _first_hit_rank(retrieved, expected_set)
        strict = _strict_first_hit_rank(
            retrieved, set(fixture_item.expected.chunk_ids)
        )

        items.append(
            EvalItemResult(
                item_id=fixture_item.id,
                query=fixture_item.query,
                expected_doc_ids=fixture_item.expected.doc_ids,
                retrieved=tuple(retrieved),
                first_hit_rank=first_hit,
                strict_first_hit_rank=strict,
                tags=fixture_item.tags,
                expected_page=fixture_item.expected.page,
                expected_section=fixture_item.expected.section,
            )
        )

    aggregate = compute_aggregate(items, top_k=top_k)
    finished = _utc_now_iso()

    rerank_prov = (
        RerankProvenance(
            enabled=rerank_config.enabled,
            model=rerank_config.model,
            pool_size=rerank_config.pool_size,
            batch_size=rerank_config.batch_size,
        )
        if rerank_config is not None
        else None
    )

    hybrid_prov = (
        HybridProvenance(
            enabled=hybrid_config.enabled,
            pool_size=hybrid_config.pool_size,
            k_rrf=hybrid_config.k_rrf,
        )
        if hybrid_config is not None
        else None
    )

    return EvalRun(
        agent=agent_name,
        fixture_path=fixture_set.source_path,
        top_k=top_k,
        items=tuple(items),
        aggregate=aggregate,
        skipped=tuple(skipped),
        started_utc=started,
        finished_utc=finished,
        rerank=rerank_prov,
        hybrid=hybrid_prov,
    )


# ---------------------------------------------------------------------------
# Aggregate + per-tag computation
# ---------------------------------------------------------------------------

def compute_aggregate(items: List[EvalItemResult], *, top_k: int = 10) -> EvalAggregate:
    """Compute aggregate metrics (+ per-tag breakdown) from item results.

    Note on ``recall_at_10`` / ``ndcg_at_10`` when ``top_k < 10``: the
    retrieved list is already truncated to ``top_k`` by the runner, so
    no item can have rank > ``top_k``. In that case ``recall_at_10``
    trivially equals ``recall_at_{top_k}`` and ``ndcg_at_10`` equals
    ``ndcg_at_{top_k}``. Callers displaying these numbers (e.g. the
    16.3 Markdown report) should surface ``top_k`` alongside the
    metrics so readers know the effective cutoff.
    """
    ranks = [item.first_hit_rank for item in items]
    rel_lists, expected_counts = _nDCG_inputs(items, top_k=top_k)

    per_tag: dict[str, TagMetrics] = {}
    tag_buckets: dict[str, list[EvalItemResult]] = {}
    for item in items:
        for tag in item.tags:
            tag_buckets.setdefault(tag, []).append(item)

    for tag, bucket in sorted(tag_buckets.items()):
        tag_ranks = [i.first_hit_rank for i in bucket]
        per_tag[tag] = TagMetrics(
            recall_at_5=recall_at_k(tag_ranks, 5),
            mrr=mrr(tag_ranks),
            n_items=len(bucket),
            low_sample=len(bucket) < 2,
        )

    citation_cases = [
        CitationCase(
            expected_page=item.expected_page,
            expected_section=item.expected_section,
            retrieved_page_start=item.retrieved[0].page_start if item.retrieved else None,
            retrieved_page_end=item.retrieved[0].page_end if item.retrieved else None,
            retrieved_section=item.retrieved[0].section_heading if item.retrieved else None,
        )
        for item in items
    ]
    cit_acc = citation_accuracy(citation_cases)
    n_cit = sum(
        1
        for c in citation_cases
        if c.expected_page is not None or c.expected_section is not None
    )

    return EvalAggregate(
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_3=recall_at_k(ranks, 3),
        recall_at_5=recall_at_k(ranks, 5),
        recall_at_10=recall_at_k(ranks, 10),
        mrr=mrr(ranks),
        ndcg_at_10=ndcg_at_k(rel_lists, expected_counts, k=10),
        per_tag=per_tag,
        citation_accuracy=cit_acc,
        n_citation_items=n_cit,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chunk_to_doc(chunk_map: Dict) -> Dict[str, str]:
    """chunk_id -> doc_id lookup. v1.1+ chunk maps only; v1.0 returns empty."""
    chunks = chunk_map.get("chunks") or []
    lookup: Dict[str, str] = {}
    for entry in chunks:
        cid = entry.get("chunk_id")
        did = entry.get("doc_id")
        if cid and did:
            lookup[cid] = did
    return lookup


def _build_chunk_to_path(chunk_map: Dict) -> Dict[str, str]:
    """chunk_id -> relative file_path lookup (v1.1+ chunk maps)."""
    lookup: Dict[str, str] = {}
    for entry in chunk_map.get("chunks") or []:
        cid = entry.get("chunk_id")
        fp = entry.get("file_path")
        if cid and fp:
            lookup[cid] = fp
    return lookup


def _read_chunk_citation_metadata(
    corpus_dir: Path, rel_path: str
) -> tuple[int | None, int | None, str | None]:
    """Read ``page_start``, ``page_end``, ``section_heading`` from a chunk's YAML front matter.

    Chunk files live at ``<corpus_dir>/<rel_path>`` and open with ``---\\n``-delimited
    YAML front matter (see ``grounding/chunk_metadata.py``). Missing files, missing
    front matter, or missing fields degrade silently to ``(None, None, None)``; the
    citation metric treats that as a miss when the fixture expects either field.
    A per-query cost of ~10 small-file reads is negligible; if profiling ever shows
    otherwise, stash the fields in ``chunk_map`` at index build time.
    """
    path = corpus_dir / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return (None, None, None)

    if not text.startswith("---\n"):
        return (None, None, None)
    end = text.find("\n---", 4)
    if end == -1:
        return (None, None, None)
    front_matter = text[4:end]
    try:
        data = yaml.safe_load(front_matter) or {}
    except yaml.YAMLError:
        return (None, None, None)
    if not isinstance(data, dict):
        return (None, None, None)

    ps = data.get("page_start")
    pe = data.get("page_end")
    sh = data.get("section_heading")
    if not isinstance(ps, int) or isinstance(ps, bool):
        ps = None
    if not isinstance(pe, int) or isinstance(pe, bool):
        pe = None
    if not isinstance(sh, str) or not sh:
        sh = None
    return (ps, pe, sh)


def _read_chunk_body(corpus_dir: Path, rel_path: str) -> str:
    """Return the chunk body text (front matter stripped).

    Used only by the rerank path. Missing file or unparseable front matter
    degrades to the raw text so the cross-encoder still sees *something*.
    """
    path = corpus_dir / rel_path
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def _apply_rerank(
    *,
    query: str,
    raw_hits: List[Tuple[str, float]],
    chunk_to_path: Dict[str, str],
    corpus_dir: Path,
    rerank_config,
    top_k: int,
) -> List[Tuple[str, float]]:
    """Run the cross-encoder on ``raw_hits`` and return the top-``top_k``.

    Kept as a pure helper so tests can monkeypatch
    ``grounding.reranker.rerank`` without touching the runner's main loop.
    """
    from grounding import reranker as _reranker_module

    pool = []
    for chunk_id, score in raw_hits:
        rel_path = chunk_to_path.get(chunk_id, "")
        content = _read_chunk_body(corpus_dir, rel_path) if rel_path else ""
        pool.append(
            {
                "chunk_id": chunk_id,
                "score": float(score),
                "content": content,
            }
        )
    reranked = _reranker_module.rerank(
        query, pool, config=rerank_config, text_key="content"
    )
    return [(item["chunk_id"], float(item["score"])) for item in reranked[:top_k]]


def _first_hit_rank(
    retrieved: List[RetrievedChunk], expected_doc_ids: set[str]
) -> int | None:
    for r in retrieved:
        if r.doc_id in expected_doc_ids:
            return r.rank
    return None


def _strict_first_hit_rank(
    retrieved: List[RetrievedChunk], expected_chunk_ids: set[str]
) -> int | None:
    if not expected_chunk_ids:
        return None
    for r in retrieved:
        candidate = f"{r.doc_id}/{r.chunk_id}"
        if candidate in expected_chunk_ids or r.chunk_id in expected_chunk_ids:
            return r.rank
    return None


def _nDCG_inputs(
    items: List[EvalItemResult], *, top_k: int
) -> Tuple[List[List[int]], List[int]]:
    rel_lists: List[List[int]] = []
    expected_counts: List[int] = []
    for item in items:
        expected_set = set(item.expected_doc_ids)
        rel = [1 if r.doc_id in expected_set else 0 for r in item.retrieved[:top_k]]
        rel_lists.append(rel)
        expected_counts.append(len(expected_set))
    return rel_lists, expected_counts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
