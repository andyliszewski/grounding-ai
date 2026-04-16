"""
BM25 sidecar index for hybrid retrieval.

This module builds a per-agent BM25 lexical index that lives alongside the
existing FAISS dense index. It is the lexical half of the two-stage hybrid
retrieval flow introduced in Epic 19: Story 19.1 writes the artifacts, Story
19.2 fuses them with the dense channel (RRF), Story 19.3 wires the hybrid
function into the CLI / MCP / local_rag surfaces.

Artifacts (written side-by-side with `_embeddings.faiss` and `_chunk_map.json`):

- ``_bm25.pkl``       — pickled ``rank_bm25.BM25Okapi`` state (tokenized corpus
                        + internal arrays), or ``None`` for the empty-corpus
                        sentinel shape.
- ``_bm25_map.json``  — metadata and chunk map (see schema below).

Map schema v1::

    {
      "format_version": 1,
      "tokenizer": "whitespace_lowercase_v1",
      "rank_bm25_version": "<pkg_version>",
      "total": 5,
      "tombstone_count": 0,
      "created_utc": "...",
      "updated_utc": "...",
      "chunks": [
        {"bm25_index": 0, "chunk_id": "...", "doc_id": "...", "deleted_utc": null}
      ]
    }

Parallel-array contract with FAISS: ``chunk_bodies[i] ↔ chunk_ids[i] ↔ bm25
internal id i`` and — when the caller preserves insertion order — the same
integer ``i`` is the FAISS internal id. 19.2 exploits that by doing RRF on
chunk_id handles directly, no remapping.

Tombstone semantics mirror FAISS: the pickle is never rewritten for a delete;
``deleted_utc`` lands in the map, and ``search_bm25`` filters at query time
with a tombstone-aware fetch multiplier, same shape as
``vector_store.search_similar_chunks``.

Tokenizer identity (``TOKENIZER_IDENTITY``) is the migration guard. If anyone
ever swaps in stemming or a language-specific tokenizer, they bump the
identity string; ``load_bm25_index`` rejects mismatched pickles with
``BM25FormatError`` and an actionable rebuild hint instead of silently
producing wrong scores.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger("grounding.bm25")

BM25_PICKLE_FILENAME = "_bm25.pkl"
BM25_MAP_FILENAME = "_bm25_map.json"
TOKENIZER_IDENTITY = "whitespace_lowercase_v1"
FORMAT_VERSION = 1
TOMBSTONE_WARNING_THRESHOLD = 0.20

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class BM25FormatError(Exception):
    """Raised when a BM25 artifact on disk is incompatible with this build."""


@dataclass
class BM25Index:
    """In-memory handle for a loaded BM25 index.

    ``bm25`` is ``None`` for an empty-corpus sentinel. ``chunk_map`` is the
    parsed JSON map (v1 schema). ``chunk_map["chunks"][i]`` is the metadata
    entry for the i-th tokenized document; its ``deleted_utc`` field is the
    tombstone marker.
    """

    bm25: Optional[BM25Okapi]
    chunk_map: dict


def tokenize(text: str) -> List[str]:
    """Whitespace + lowercase tokenizer. Strips punctuation via ``\\w+``."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _rank_bm25_version() -> str:
    try:
        return importlib_metadata.version("rank_bm25")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes atomically via temp-then-rename, same pattern as
    ``grounding.utils.atomic_write`` but without UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(fd, "wb") as f:
            f.write(payload)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, obj: dict) -> None:
    payload = json.dumps(obj, indent=2).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_map(
    chunk_entries: List[dict],
    *,
    created_utc: Optional[str] = None,
    tombstone_count: int = 0,
) -> dict:
    now = _now_utc()
    return {
        "format_version": FORMAT_VERSION,
        "tokenizer": TOKENIZER_IDENTITY,
        "rank_bm25_version": _rank_bm25_version(),
        "total": len(chunk_entries),
        "tombstone_count": tombstone_count,
        "created_utc": created_utc or now,
        "updated_utc": now,
        "chunks": chunk_entries,
    }


def _chunk_entry(bm25_index: int, chunk_id: str, doc_id: Optional[str]) -> dict:
    return {
        "bm25_index": bm25_index,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "deleted_utc": None,
    }


def _attach_tokenized_corpus(bm25: BM25Okapi, tokenized_corpus: List[List[str]]) -> None:
    """Attach the tokenized corpus to the BM25 object as a private attr.

    ``rank_bm25.BM25Okapi`` does not retain the original tokenized corpus on
    the instance (it consumes it to build ``doc_freqs`` / ``doc_len``).
    Reindexing on append requires the original tokens, so we stash them
    under a namespaced attribute that round-trips via pickle.
    """
    bm25._grounding_tokenized_corpus = tokenized_corpus  # type: ignore[attr-defined]


def write_bm25_index(
    chunk_bodies: List[str],
    chunk_ids: List[str],
    output_dir: Path,
    chunk_doc_ids: Optional[List[Optional[str]]] = None,
) -> None:
    """Build a BM25 index from chunk bodies and persist both artifacts.

    Parallel arrays: ``chunk_bodies[i]`` ↔ ``chunk_ids[i]`` ↔ BM25 internal
    id ``i``. Optional ``chunk_doc_ids`` is the per-chunk doc_id plumbed
    through for tombstone-by-doc_id semantics.

    Empty input writes a sentinel pickle (``None``) plus a map with
    ``total: 0`` so downstream code can distinguish "no index" (files absent)
    from "index exists but is empty" (files present, total 0).
    """
    if len(chunk_bodies) != len(chunk_ids):
        raise ValueError(
            f"chunk_bodies ({len(chunk_bodies)}) and chunk_ids ({len(chunk_ids)}) length mismatch"
        )
    if chunk_doc_ids is not None and len(chunk_doc_ids) != len(chunk_ids):
        raise ValueError(
            f"chunk_doc_ids ({len(chunk_doc_ids)}) and chunk_ids ({len(chunk_ids)}) length mismatch"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = output_dir / BM25_PICKLE_FILENAME
    map_path = output_dir / BM25_MAP_FILENAME

    if not chunk_bodies:
        _atomic_write_bytes(pickle_path, pickle.dumps(None))
        _atomic_write_json(map_path, _build_map([]))
        logger.info("Wrote empty BM25 index (0 chunks) to %s", output_dir)
        return

    tokenized_corpus = [tokenize(body) for body in chunk_bodies]
    bm25 = BM25Okapi(tokenized_corpus)
    _attach_tokenized_corpus(bm25, tokenized_corpus)

    entries: List[dict] = []
    for i, cid in enumerate(chunk_ids):
        doc_id = chunk_doc_ids[i] if chunk_doc_ids is not None else None
        entries.append(_chunk_entry(i, cid, doc_id))

    _atomic_write_bytes(pickle_path, pickle.dumps(bm25))
    _atomic_write_json(map_path, _build_map(entries))
    logger.info("Wrote BM25 index (%d chunks) to %s", len(entries), output_dir)


def load_bm25_index(output_dir: Path) -> Optional[BM25Index]:
    """Load BM25 artifacts from disk.

    Returns ``None`` if either file is missing. Raises ``BM25FormatError``
    if the map's tokenizer identity or format version does not match this
    build — with an actionable rebuild hint.
    """
    output_dir = Path(output_dir)
    pickle_path = output_dir / BM25_PICKLE_FILENAME
    map_path = output_dir / BM25_MAP_FILENAME

    if not pickle_path.exists() or not map_path.exists():
        return None

    with open(map_path, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    fmt = chunk_map.get("format_version")
    if fmt != FORMAT_VERSION:
        raise BM25FormatError(
            f"BM25 map format_version={fmt!r} (expected {FORMAT_VERSION}). "
            f"This version of grounding supports BM25 map v{FORMAT_VERSION}; "
            f"upgrade grounding or rebuild with `grounding embeddings --agent <name>`."
        )

    tokenizer = chunk_map.get("tokenizer")
    if tokenizer != TOKENIZER_IDENTITY:
        raise BM25FormatError(
            f"BM25 tokenizer identity mismatch: on-disk={tokenizer!r}, "
            f"current={TOKENIZER_IDENTITY!r}. Rebuild with "
            f"`grounding embeddings --agent <name>`."
        )

    with open(pickle_path, "rb") as f:
        bm25 = pickle.load(f)

    return BM25Index(bm25=bm25, chunk_map=chunk_map)


def append_to_bm25_index(
    new_bodies: List[str],
    new_ids: List[str],
    output_dir: Path,
    new_doc_ids: Optional[List[Optional[str]]] = None,
) -> int:
    """Append new chunks to an existing BM25 index (or create one).

    ``rank_bm25.BM25Okapi`` computes IDF at ``__init__`` from the full
    tokenized corpus; it has no public mutating API. We concatenate the old
    tokenized corpus (re-tokenized from stored bodies is not available, so
    we keep the tokenized corpus on the in-memory ``BM25Okapi`` via its own
    ``corpus`` attribute) with the new tokens and rebuild. See Story 19.1
    dev notes for why this is acceptable for Day 1.

    Returns the count of newly-appended chunks.
    """
    if len(new_bodies) != len(new_ids):
        raise ValueError(
            f"new_bodies ({len(new_bodies)}) and new_ids ({len(new_ids)}) length mismatch"
        )
    if new_doc_ids is not None and len(new_doc_ids) != len(new_ids):
        raise ValueError(
            f"new_doc_ids ({len(new_doc_ids)}) and new_ids ({len(new_ids)}) length mismatch"
        )

    output_dir = Path(output_dir)

    if not new_bodies:
        return 0

    existing = load_bm25_index(output_dir)
    if existing is None or existing.bm25 is None:
        # No usable existing index (either missing files or empty-corpus
        # sentinel). Fall through to a full build — same pattern as
        # append_to_vector_index with FileNotFoundError.
        write_bm25_index(new_bodies, new_ids, output_dir, chunk_doc_ids=new_doc_ids)
        logger.info("append_to_bm25_index: no existing index, wrote fresh (%d chunks)", len(new_ids))
        return len(new_ids)

    # Reconstruct the tokenized corpus from the loaded BM25Okapi. rank_bm25
    # stores each doc's tokenized form in `self.corpus_size`/internal arrays,
    # but the simplest public access is via `bm25.corpus` when preserved.
    # rank_bm25's BM25Okapi stores individual docs via `doc_freqs` (list of
    # Counter) and per-doc lengths; the tokenized corpus itself is NOT kept
    # on the object. We fix this by storing it alongside at build time.
    prior_tokenized = getattr(existing.bm25, "_grounding_tokenized_corpus", None)
    if prior_tokenized is None:
        # Old pickle without the attribute — rebuild from scratch using
        # what we can recover. Since bodies aren't on disk, the only safe
        # thing is to raise so callers know to do a full rebuild.
        raise BM25FormatError(
            "Existing BM25 pickle lacks tokenized-corpus attachment; rebuild with "
            "`grounding embeddings --agent <name>` (no --incremental)."
        )

    new_tokenized = [tokenize(b) for b in new_bodies]
    merged_tokenized = prior_tokenized + new_tokenized

    bm25 = BM25Okapi(merged_tokenized)
    _attach_tokenized_corpus(bm25, merged_tokenized)

    chunks = list(existing.chunk_map.get("chunks", []))
    start = len(chunks)
    for offset, cid in enumerate(new_ids):
        doc_id = new_doc_ids[offset] if new_doc_ids is not None else None
        chunks.append(_chunk_entry(start + offset, cid, doc_id))

    new_map = _build_map(
        chunks,
        created_utc=existing.chunk_map.get("created_utc"),
        tombstone_count=existing.chunk_map.get("tombstone_count", 0),
    )

    pickle_path = output_dir / BM25_PICKLE_FILENAME
    map_path = output_dir / BM25_MAP_FILENAME
    _atomic_write_bytes(pickle_path, pickle.dumps(bm25))
    _atomic_write_json(map_path, new_map)

    logger.info("BM25 append: +%d chunks (total %d)", len(new_ids), len(chunks))
    return len(new_ids)


def tombstone_bm25_documents(doc_ids: List[str], output_dir: Path) -> int:
    """Soft-delete chunks by doc_id. Returns count of chunks tombstoned.

    Mirrors ``grounding.vector_store.tombstone_documents``: the BM25 pickle
    is never modified; only ``deleted_utc`` on matching map entries is set.
    ``search_bm25`` filters at query time.
    """
    if not doc_ids:
        return 0

    output_dir = Path(output_dir)
    map_path = output_dir / BM25_MAP_FILENAME
    if not map_path.exists():
        return 0

    with open(map_path, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    doc_ids_set = set(doc_ids)
    now = _now_utc()
    tombstoned = 0
    for chunk in chunk_map.get("chunks", []):
        if chunk.get("doc_id") in doc_ids_set and chunk.get("deleted_utc") is None:
            chunk["deleted_utc"] = now
            tombstoned += 1

    if tombstoned:
        chunk_map["updated_utc"] = now
        chunk_map["tombstone_count"] = chunk_map.get("tombstone_count", 0) + tombstoned
        _atomic_write_json(map_path, chunk_map)
        tombstone_count = chunk_map["tombstone_count"]
        total = chunk_map.get("total", 0)
        ratio = tombstone_count / total if total else 0.0
        if ratio > TOMBSTONE_WARNING_THRESHOLD:
            logger.warning(
                "BM25 tombstone ratio %.1f%% (%d / %d) — consider rebuilding.",
                ratio * 100, tombstone_count, total,
            )
        logger.info(
            "BM25 tombstoned %d chunks across %d doc_ids (total tombstones: %d)",
            tombstoned, len(doc_ids_set), tombstone_count,
        )

    return tombstoned


def search_bm25(
    index: BM25Index,
    query_tokens: List[str],
    top_k: int,
) -> List[Tuple[str, int, float]]:
    """Top-k BM25 search with tombstone filtering.

    Returns ``list[tuple[chunk_id, rank, score]]`` with rank 1-indexed.
    Requests extra candidates proportional to the tombstone ratio to avoid
    truncating below ``top_k`` usable results (mirrors
    ``vector_store.search_similar_chunks``).
    """
    if top_k <= 0:
        return []
    if index.bm25 is None:
        return []
    if not query_tokens:
        return []

    chunks = index.chunk_map.get("chunks", [])
    total = len(chunks)
    if total == 0:
        return []

    tombstone_count = index.chunk_map.get("tombstone_count", 0)
    tombstone_ratio = tombstone_count / total if total else 0.0
    fetch_multiplier = 1.0 + tombstone_ratio + 0.1
    fetch_k = min(int(top_k * fetch_multiplier) + 1, total)

    scores = index.bm25.get_scores(query_tokens)
    # Argsort descending by score; only consider `fetch_k` best.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]

    results: List[Tuple[str, int, float]] = []
    for idx in order:
        if idx >= total:
            continue
        chunk = chunks[idx]
        if chunk.get("deleted_utc") is not None:
            continue
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        results.append((chunk_id, len(results) + 1, float(scores[idx])))
        if len(results) >= top_k:
            break
    return results
