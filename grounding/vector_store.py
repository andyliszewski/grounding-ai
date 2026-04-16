"""
Vector store module for FAISS-based semantic search.

This module provides functionality to:
- Create and persist FAISS vector indexes from embeddings
- Load existing indexes from disk
- Perform similarity search on embeddings
- Incrementally append embeddings to existing indexes
- Tombstone (soft-delete) documents without index rebuild

The vector store uses FAISS Flat L2 index for exact nearest neighbor search,
suitable for corpora up to 100k chunks.

Chunk Map Schema Versions:
- v1.0: Basic format with chunk_ids list
- v1.1: Extended format with chunks list containing metadata (doc_id, is_music, etc.)
- v1.2: Incremental format with tombstone support:
    - Added `deleted_utc` field to chunk entries (null if active, ISO8601 if deleted)
    - Added `tombstone_count` to metadata (count of soft-deleted chunks)
    - Added `updated_utc` to metadata (last modification timestamp)
    - Backward compatible: can read v1.0 and v1.1 formats
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np

from grounding.utils import atomic_write

logger = logging.getLogger("grounding.vector_store")

# Constants
DEFAULT_EMBEDDING_DIM = 384
FAISS_INDEX_FILENAME = "_embeddings.faiss"
CHUNK_MAP_FILENAME = "_chunk_map.json"
FORMAT_VERSION = "1.0"
FORMAT_VERSION_WITH_METADATA = "1.1"
FORMAT_VERSION_INCREMENTAL = "1.2"
TOMBSTONE_REBUILD_THRESHOLD = 0.30  # Recommend rebuild when >30% tombstoned
TOMBSTONE_WARNING_THRESHOLD = 0.20  # Log warning when >20% tombstoned


@dataclass
class StalenessReport:
    """Report on index staleness relative to corpus manifest.

    Attributes:
        new_docs: doc_ids present in manifest but not in index
        deleted_docs: doc_ids present in index but not in manifest
        updated_docs: doc_ids with changed file hashes
        is_stale: True if any of the above lists are non-empty
        tombstone_ratio: Current percentage of tombstoned chunks (0.0-1.0)
        should_rebuild: True if tombstone_ratio exceeds threshold
    """
    new_docs: List[str]
    deleted_docs: List[str]
    updated_docs: List[str]
    is_stale: bool
    tombstone_ratio: float
    should_rebuild: bool


def write_vector_index(
    embeddings: Dict[str, np.ndarray],
    output_dir: Path,
    chunk_metadata: Optional[Dict[str, dict]] = None
) -> None:
    """
    Create and persist a FAISS vector index from embeddings.

    Args:
        embeddings: Dictionary mapping chunk_id to embedding vector (384-dim numpy array)
        output_dir: Directory where index files will be written
        chunk_metadata: Optional dictionary mapping chunk_id to metadata dict.
            For music chunks, metadata should include:
            - is_music: bool
            - music_metadata: dict with key, time_signature, harmony, rhythm
            - description: str (the description that was embedded)
            - file_path: str (relative path to chunk file)

    Creates two files in output_dir:
        - _embeddings.faiss: Binary FAISS index file
        - _chunk_map.json: JSON mapping index positions to chunk IDs (v1.0)
          or detailed chunk metadata (v1.1 with chunk_metadata)

    Raises:
        ValueError: If embeddings have inconsistent dimensions or wrong dimension
    """
    if not embeddings:
        logger.info("No embeddings provided, skipping vector store creation")
        return

    logger.info(f"Creating FAISS index for {len(embeddings)} embeddings")

    # Extract chunk IDs and embeddings in consistent order
    chunk_ids = list(embeddings.keys())
    embedding_arrays = [embeddings[cid] for cid in chunk_ids]

    # Validate dimensions
    dimensions = [emb.shape[0] for emb in embedding_arrays]
    if len(set(dimensions)) > 1:
        raise ValueError(f"Inconsistent embedding dimensions: {set(dimensions)}")

    embedding_dim = dimensions[0]
    if embedding_dim != DEFAULT_EMBEDDING_DIM:
        logger.warning(
            f"Embedding dimension {embedding_dim} differs from expected {DEFAULT_EMBEDDING_DIM}"
        )

    # Convert to float32 matrix (FAISS requirement)
    embeddings_matrix = np.array(embedding_arrays, dtype=np.float32)

    # Create FAISS Flat L2 index
    logger.debug(f"Creating IndexFlatL2 with dimension {embedding_dim}")
    index = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to index
    index.add(embeddings_matrix)
    logger.debug(f"Added {index.ntotal} vectors to index")

    # Prepare chunk map metadata
    # Use extended format (v1.1) if chunk_metadata provided, otherwise simple format (v1.0)
    if chunk_metadata:
        logger.debug("Using extended chunk map format (v1.1) with metadata")
        chunks_list = []
        for idx, chunk_id in enumerate(chunk_ids):
            chunk_entry = {
                "chunk_id": chunk_id,
                "embedding_index": idx
            }
            # Add optional metadata if provided for this chunk
            if chunk_id in chunk_metadata:
                meta = chunk_metadata[chunk_id]
                chunk_entry["is_music"] = meta.get("is_music", False)
                if meta.get("music_metadata"):
                    chunk_entry["music_metadata"] = meta["music_metadata"]
                if meta.get("description"):
                    chunk_entry["description"] = meta["description"]
                if meta.get("file_path"):
                    chunk_entry["file_path"] = meta["file_path"]
                if meta.get("doc_id"):
                    chunk_entry["doc_id"] = meta["doc_id"]
            else:
                # Default: not music, minimal metadata
                chunk_entry["is_music"] = False

            chunks_list.append(chunk_entry)

        chunk_map = {
            "format_version": FORMAT_VERSION_WITH_METADATA,
            "faiss_version": faiss.__version__,
            "dimension": embedding_dim,
            "index_size": index.ntotal,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks_list
        }
    else:
        # Simple format (v1.0) - backward compatible
        logger.debug("Using simple chunk map format (v1.0)")
        chunk_map = {
            "format_version": FORMAT_VERSION,
            "faiss_version": faiss.__version__,
            "dimension": embedding_dim,
            "index_size": index.ntotal,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "chunk_ids": chunk_ids,
        }

    # Write FAISS index to disk (atomic)
    index_path = output_dir / FAISS_INDEX_FILENAME
    try:
        temp_index_path = index_path.with_suffix(".tmp")
        faiss.write_index(index, str(temp_index_path))
        temp_index_path.replace(index_path)
        logger.info(f"Wrote FAISS index to {index_path}")
    except Exception as e:
        logger.error(f"Failed to write FAISS index: {e}", exc_info=True)
        raise

    # Write chunk map to disk (atomic)
    chunk_map_path = output_dir / CHUNK_MAP_FILENAME
    try:
        chunk_map_json = json.dumps(chunk_map, indent=2)
        atomic_write(chunk_map_path, chunk_map_json)
        logger.info(f"Wrote chunk map to {chunk_map_path}")
    except Exception as e:
        logger.error(f"Failed to write chunk map: {e}", exc_info=True)
        raise


def load_vector_index(corpus_dir: Path) -> Tuple[faiss.Index, Dict]:
    """
    Load a FAISS vector index and chunk map from disk.

    Args:
        corpus_dir: Directory containing the vector store files

    Returns:
        Tuple of (FAISS index, chunk_map dict)

    Raises:
        FileNotFoundError: If index or chunk map files are missing
        ValueError: If index dimensions don't match chunk map metadata
    """
    index_path = corpus_dir / FAISS_INDEX_FILENAME
    chunk_map_path = corpus_dir / CHUNK_MAP_FILENAME

    # Check files exist
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found: {chunk_map_path}")

    logger.info(f"Loading FAISS index from {index_path}")

    # Load FAISS index
    try:
        index = faiss.read_index(str(index_path))
        logger.debug(f"Loaded index with {index.ntotal} vectors, dimension {index.d}")
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
        raise

    # Load chunk map
    try:
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        # Detect format version and count chunks
        format_version = chunk_map.get("format_version", "1.0")
        if format_version in (FORMAT_VERSION_WITH_METADATA, FORMAT_VERSION_INCREMENTAL):
            # v1.1 or v1.2 format with chunks list
            chunk_count = len(chunk_map.get("chunks", []))
            logger.debug(f"Loaded chunk map {format_version} with {chunk_count} chunks (extended metadata)")
        else:
            # v1.0 format with chunk_ids list
            chunk_count = len(chunk_map.get("chunk_ids", []))
            logger.debug(f"Loaded chunk map v1.0 with {chunk_count} chunk IDs")

    except Exception as e:
        logger.error(f"Failed to load chunk map: {e}", exc_info=True)
        raise

    # Validate consistency
    expected_size = chunk_map.get("index_size", 0)
    if index.ntotal != expected_size:
        raise ValueError(
            f"Index size mismatch: index has {index.ntotal} vectors, "
            f"chunk map expects {expected_size}"
        )

    expected_dim = chunk_map.get("dimension", DEFAULT_EMBEDDING_DIM)
    if index.d != expected_dim:
        raise ValueError(
            f"Dimension mismatch: index has {index.d} dimensions, "
            f"chunk map expects {expected_dim}"
        )

    if chunk_count != index.ntotal:
        raise ValueError(
            f"Chunk map size mismatch: {chunk_count} chunks for {index.ntotal} vectors"
        )

    logger.info(f"Successfully loaded vector store with {index.ntotal} embeddings")
    return index, chunk_map


def search_similar_chunks(
    index: faiss.Index,
    chunk_map: Dict,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Search for similar chunks using a query embedding.

    Automatically filters out tombstoned chunks (those with deleted_utc set).
    Requests extra results from FAISS to compensate for filtered tombstones.

    Args:
        index: FAISS index
        chunk_map: Chunk map dictionary (from load_vector_index)
        query_embedding: Query vector (384-dim numpy array)
        top_k: Number of results to return

    Returns:
        List of (chunk_id, distance) tuples, sorted by distance (ascending)
        Lower distance means higher similarity

    Raises:
        ValueError: If query embedding dimension doesn't match index
    """
    if index.ntotal == 0:
        logger.warning("Search attempted on empty index")
        return []

    # Validate query dimension
    if query_embedding.shape[0] != index.d:
        raise ValueError(
            f"Query dimension {query_embedding.shape[0]} doesn't match "
            f"index dimension {index.d}"
        )

    # Check tombstone ratio and log warning if significant
    tombstone_ratio = _compute_tombstone_ratio(chunk_map)
    if tombstone_ratio > TOMBSTONE_WARNING_THRESHOLD:
        logger.warning(
            f"High tombstone ratio ({tombstone_ratio:.1%}) detected. "
            f"Consider rebuilding the index for better performance."
        )

    # Calculate fetch_k: request extra results to account for tombstones
    # Add 10% buffer on top of tombstone ratio
    fetch_multiplier = 1.0 + tombstone_ratio + 0.1
    fetch_k = min(int(top_k * fetch_multiplier) + 1, index.ntotal)

    logger.debug(f"Searching for top {fetch_k} chunks (requested {top_k}, tombstone_ratio={tombstone_ratio:.1%})")

    # Prepare query for FAISS (must be float32, 2D)
    query = query_embedding.astype(np.float32).reshape(1, -1)

    # Perform search
    distances, indices = index.search(query, fetch_k)

    # Map indices to chunk entries (handle v1.0, v1.1, and v1.2 formats)
    format_version = chunk_map.get("format_version", "1.0")

    if format_version in (FORMAT_VERSION_WITH_METADATA, FORMAT_VERSION_INCREMENTAL):
        # v1.1/v1.2 format: chunks list with metadata
        chunks = chunk_map.get("chunks", [])

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(chunks):
                logger.warning(f"Invalid index {idx} returned by FAISS, skipping")
                continue

            chunk = chunks[idx]

            # Skip tombstoned chunks (v1.2 format)
            if chunk.get("deleted_utc") is not None:
                continue

            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                results.append((chunk_id, float(distance)))

            # Stop once we have enough results
            if len(results) >= top_k:
                break
    else:
        # v1.0 format: simple chunk_ids list (no tombstone support)
        chunk_ids = chunk_map.get("chunk_ids", [])

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(chunk_ids):
                logger.warning(f"Invalid index {idx} returned by FAISS, skipping")
                continue
            chunk_id = chunk_ids[idx]
            results.append((chunk_id, float(distance)))

            if len(results) >= top_k:
                break

    logger.debug(f"Found {len(results)} similar chunks (after filtering tombstones)")
    return results


def get_chunk_metadata(chunk_map: Dict, chunk_id: str) -> Optional[dict]:
    """
    Extract metadata for a specific chunk from the chunk map.

    Args:
        chunk_map: Chunk map dictionary (from load_vector_index)
        chunk_id: Chunk identifier to look up

    Returns:
        Metadata dictionary if found and format is v1.1, None otherwise

    Example return for music chunk:
        {
            "is_music": True,
            "music_metadata": {
                "key": "C major",
                "time_signature": "4/4",
                "harmony": ["I", "IV", "V"],
                "rhythm": "quarter notes"
            },
            "description": "Music phrase in C major...",
            "file_path": "doc/chunks/ch_0001.md",
            "doc_id": "abc123"
        }
    """
    format_version = chunk_map.get("format_version", "1.0")

    if format_version != FORMAT_VERSION_WITH_METADATA:
        # v1.0 format doesn't have metadata
        return None

    # v1.1 format: search chunks list
    chunks = chunk_map.get("chunks", [])
    for chunk in chunks:
        if chunk.get("chunk_id") == chunk_id:
            # Return all metadata except chunk_id and embedding_index
            metadata = {
                "is_music": chunk.get("is_music", False)
            }
            if chunk.get("music_metadata"):
                metadata["music_metadata"] = chunk["music_metadata"]
            if chunk.get("description"):
                metadata["description"] = chunk["description"]
            if chunk.get("file_path"):
                metadata["file_path"] = chunk["file_path"]
            if chunk.get("doc_id"):
                metadata["doc_id"] = chunk["doc_id"]

            return metadata

    # Chunk ID not found
    logger.debug(f"Chunk {chunk_id} not found in chunk map")
    return None


def get_indexed_doc_ids(chunk_map: Dict, include_tombstoned: bool = False) -> set:
    """
    Extract unique doc_ids from chunk map.

    Args:
        chunk_map: Chunk map dictionary (from load_vector_index)
        include_tombstoned: If True, include tombstoned docs; if False, filter them out

    Returns:
        Set of doc_ids currently in the index (optionally excluding tombstoned)

    Note:
        For v1.0 format (no doc_id per chunk), returns empty set.
        For v1.1+ format, extracts doc_id from each chunk entry.
    """
    format_version = chunk_map.get("format_version", "1.0")

    if format_version == FORMAT_VERSION:
        # v1.0 format doesn't have doc_id per chunk
        logger.debug("Chunk map v1.0 does not contain doc_id metadata")
        return set()

    chunks = chunk_map.get("chunks", [])
    doc_ids = set()

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue

        # Check tombstone status
        if not include_tombstoned and chunk.get("deleted_utc") is not None:
            continue

        doc_ids.add(doc_id)

    logger.debug(f"Found {len(doc_ids)} unique doc_ids in chunk map (include_tombstoned={include_tombstoned})")
    return doc_ids


def _get_doc_hashes_from_chunk_map(chunk_map: Dict) -> Dict[str, str]:
    """
    Extract doc_id -> file_sha1 mapping from chunk map.

    Only v1.2+ format stores file_sha1 in chunk entries.
    Returns empty dict for older formats.
    """
    format_version = chunk_map.get("format_version", "1.0")
    if format_version not in (FORMAT_VERSION_INCREMENTAL, FORMAT_VERSION_WITH_METADATA):
        return {}

    chunks = chunk_map.get("chunks", [])
    doc_hashes = {}

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        file_sha1 = chunk.get("file_sha1")
        if doc_id and file_sha1 and doc_id not in doc_hashes:
            # Only store first occurrence (all chunks for same doc have same hash)
            if chunk.get("deleted_utc") is None:
                doc_hashes[doc_id] = file_sha1

    return doc_hashes


def _load_manifest_doc_info(corpus_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load manifest and extract doc_id -> slug and doc_id -> file_sha1 mappings.

    Args:
        corpus_dir: Directory containing _index.json

    Returns:
        Tuple of (doc_id_to_slug, doc_id_to_hash) dicts

    Raises:
        FileNotFoundError: If manifest not found
    """
    manifest_path = corpus_dir / "_index.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    doc_id_to_slug = {}
    doc_id_to_hash = {}

    for doc in manifest.get("docs", []):
        doc_id = doc.get("doc_id")
        slug = doc.get("slug")
        if doc_id and slug:
            doc_id_to_slug[doc_id] = slug

            # Try to read file_sha1 from meta.yaml
            meta_path = corpus_dir / slug / "meta.yaml"
            if meta_path.exists():
                try:
                    import yaml
                    with open(meta_path, "r") as mf:
                        meta = yaml.safe_load(mf)
                        if meta and "hashes" in meta:
                            file_sha1 = meta["hashes"].get("file_sha1")
                            if file_sha1:
                                doc_id_to_hash[doc_id] = file_sha1
                except Exception:
                    pass  # Skip if meta.yaml is invalid

    return doc_id_to_slug, doc_id_to_hash


def _compute_tombstone_ratio(chunk_map: Dict) -> float:
    """Compute ratio of tombstoned chunks to total chunks."""
    tombstone_count = chunk_map.get("tombstone_count", 0)
    index_size = chunk_map.get("index_size", 0)
    if index_size == 0:
        return 0.0
    return tombstone_count / index_size


def check_index_staleness(
    corpus_dir: Path,
    index_dir: Path,
) -> StalenessReport:
    """
    Check if embedding index is stale relative to corpus manifest.

    Args:
        corpus_dir: Directory containing _index.json manifest
        index_dir: Directory containing FAISS index and chunk map

    Returns:
        StalenessReport with lists of new/deleted/updated doc_ids

    Raises:
        FileNotFoundError: If manifest or index files not found
    """
    logger.info(f"Checking index staleness: corpus={corpus_dir}, index={index_dir}")

    # Load manifest info
    manifest_doc_ids_to_slug, manifest_doc_hashes = _load_manifest_doc_info(corpus_dir)
    manifest_doc_ids = set(manifest_doc_ids_to_slug.keys())

    # Load index chunk map
    _, chunk_map = load_vector_index(index_dir)

    # Get indexed doc_ids (excluding tombstoned)
    indexed_doc_ids = get_indexed_doc_ids(chunk_map, include_tombstoned=False)

    # Get doc hashes from chunk map (for update detection)
    indexed_doc_hashes = _get_doc_hashes_from_chunk_map(chunk_map)

    # Detect new docs (in manifest, not in index)
    new_docs = sorted(manifest_doc_ids - indexed_doc_ids)

    # Detect deleted docs (in index, not in manifest)
    deleted_docs = sorted(indexed_doc_ids - manifest_doc_ids)

    # Detect updated docs (in both, but hash changed)
    updated_docs = []
    common_docs = indexed_doc_ids & manifest_doc_ids
    for doc_id in common_docs:
        manifest_hash = manifest_doc_hashes.get(doc_id)
        indexed_hash = indexed_doc_hashes.get(doc_id)
        # Only detect update if we have both hashes to compare
        if manifest_hash and indexed_hash and manifest_hash != indexed_hash:
            updated_docs.append(doc_id)
    updated_docs.sort()

    # Compute staleness
    is_stale = bool(new_docs or deleted_docs or updated_docs)
    tombstone_ratio = _compute_tombstone_ratio(chunk_map)
    should_rebuild = tombstone_ratio > TOMBSTONE_REBUILD_THRESHOLD

    report = StalenessReport(
        new_docs=new_docs,
        deleted_docs=deleted_docs,
        updated_docs=updated_docs,
        is_stale=is_stale,
        tombstone_ratio=tombstone_ratio,
        should_rebuild=should_rebuild,
    )

    logger.info(
        f"Staleness check: new={len(new_docs)}, deleted={len(deleted_docs)}, "
        f"updated={len(updated_docs)}, tombstone_ratio={tombstone_ratio:.1%}, "
        f"should_rebuild={should_rebuild}"
    )

    return report


def append_to_vector_index(
    new_embeddings: Dict[str, np.ndarray],
    index_dir: Path,
    chunk_metadata: Optional[Dict[str, dict]] = None,
) -> int:
    """
    Append new embeddings to an existing FAISS index.

    Args:
        new_embeddings: Dictionary mapping chunk_id to embedding vector (384-dim numpy array)
        index_dir: Directory containing existing FAISS index and chunk map
        chunk_metadata: Optional dictionary mapping chunk_id to metadata dict.
            Should include: doc_id, file_path, is_music, file_sha1 (for update detection)

    Returns:
        Number of vectors appended

    Raises:
        FileNotFoundError: If existing index not found (use write_vector_index instead)
        ValueError: If embedding dimensions don't match existing index
    """
    if not new_embeddings:
        logger.info("No new embeddings to append")
        return 0

    index_path = index_dir / FAISS_INDEX_FILENAME
    chunk_map_path = index_dir / CHUNK_MAP_FILENAME

    # Check index exists
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Use write_vector_index() to create a new index."
        )

    # Load existing index and chunk map
    logger.info(f"Loading existing index from {index_dir}")
    index, chunk_map = load_vector_index(index_dir)

    # Validate dimensions
    chunk_ids = list(new_embeddings.keys())
    embedding_arrays = [new_embeddings[cid] for cid in chunk_ids]
    new_dim = embedding_arrays[0].shape[0]

    if new_dim != index.d:
        raise ValueError(
            f"New embedding dimension {new_dim} doesn't match "
            f"existing index dimension {index.d}"
        )

    # Convert to float32 matrix
    new_matrix = np.array(embedding_arrays, dtype=np.float32)

    # Record starting position for new vectors
    start_idx = index.ntotal
    vectors_added = len(new_embeddings)

    logger.info(f"Appending {vectors_added} vectors to index (starting at position {start_idx})")

    # Append to FAISS index
    index.add(new_matrix)

    # Update chunk map
    format_version = chunk_map.get("format_version", "1.0")

    if format_version == FORMAT_VERSION:
        # v1.0 format: simple chunk_ids list - upgrade to v1.2
        logger.info("Upgrading chunk map from v1.0 to v1.2 format")
        old_chunk_ids = chunk_map.get("chunk_ids", [])
        chunks_list = [
            {"chunk_id": cid, "embedding_index": idx, "is_music": False}
            for idx, cid in enumerate(old_chunk_ids)
        ]
        chunk_map = {
            "format_version": FORMAT_VERSION_INCREMENTAL,
            "faiss_version": faiss.__version__,
            "dimension": index.d,
            "index_size": index.ntotal,
            "tombstone_count": 0,
            "created_utc": chunk_map.get("created_utc", datetime.now(timezone.utc).isoformat()),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks_list,
        }
    else:
        # v1.1 or v1.2 format: chunks list
        chunk_map["format_version"] = FORMAT_VERSION_INCREMENTAL
        chunk_map["index_size"] = index.ntotal
        chunk_map["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if "tombstone_count" not in chunk_map:
            chunk_map["tombstone_count"] = 0

    # Append new chunk entries
    chunks = chunk_map.get("chunks", [])
    for idx, chunk_id in enumerate(chunk_ids):
        entry = {
            "chunk_id": chunk_id,
            "embedding_index": start_idx + idx,
            "deleted_utc": None,
        }
        # Add metadata if provided
        if chunk_metadata and chunk_id in chunk_metadata:
            meta = chunk_metadata[chunk_id]
            entry["doc_id"] = meta.get("doc_id")
            entry["is_music"] = meta.get("is_music", False)
            entry["file_path"] = meta.get("file_path")
            entry["file_sha1"] = meta.get("file_sha1")
            if meta.get("music_metadata"):
                entry["music_metadata"] = meta["music_metadata"]
        else:
            entry["is_music"] = False

        chunks.append(entry)

    chunk_map["chunks"] = chunks

    # Write updated FAISS index atomically
    try:
        temp_index_path = index_path.with_suffix(".tmp")
        faiss.write_index(index, str(temp_index_path))
        temp_index_path.replace(index_path)
        logger.debug(f"Updated FAISS index at {index_path}")
    except Exception as e:
        logger.error(f"Failed to write FAISS index: {e}", exc_info=True)
        raise

    # Write updated chunk map atomically
    try:
        chunk_map_json = json.dumps(chunk_map, indent=2)
        atomic_write(chunk_map_path, chunk_map_json)
        logger.debug(f"Updated chunk map at {chunk_map_path}")
    except Exception as e:
        logger.error(f"Failed to write chunk map: {e}", exc_info=True)
        raise

    logger.info(
        f"Appended {vectors_added} vectors. Index now has {index.ntotal} total vectors."
    )

    return vectors_added


def tombstone_documents(
    doc_ids: List[str],
    index_dir: Path,
) -> int:
    """
    Mark documents as deleted (tombstone) in the chunk map.

    This soft-deletes documents by setting deleted_utc on their chunks.
    The FAISS index is NOT modified - vectors remain in place but are
    filtered out during search operations.

    Args:
        doc_ids: List of doc_ids to tombstone
        index_dir: Directory containing FAISS index and chunk map

    Returns:
        Number of chunks tombstoned

    Raises:
        FileNotFoundError: If chunk map not found
        ValueError: If chunk map is v1.0 format (no doc_id support)
    """
    if not doc_ids:
        logger.info("No doc_ids provided for tombstoning")
        return 0

    chunk_map_path = index_dir / CHUNK_MAP_FILENAME

    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found: {chunk_map_path}")

    # Load chunk map
    with open(chunk_map_path, "r") as f:
        chunk_map = json.load(f)

    format_version = chunk_map.get("format_version", "1.0")
    if format_version == FORMAT_VERSION:
        raise ValueError(
            "Cannot tombstone documents in v1.0 chunk map format. "
            "Rebuild the index with chunk metadata to enable tombstoning."
        )

    chunks = chunk_map.get("chunks", [])
    doc_ids_set = set(doc_ids)
    tombstoned_count = 0
    now = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        # Only tombstone if doc_id matches and not already tombstoned
        if doc_id in doc_ids_set and chunk.get("deleted_utc") is None:
            chunk["deleted_utc"] = now
            tombstoned_count += 1

    if tombstoned_count > 0:
        # Update metadata
        chunk_map["format_version"] = FORMAT_VERSION_INCREMENTAL
        chunk_map["updated_utc"] = now
        current_tombstone_count = chunk_map.get("tombstone_count", 0)
        chunk_map["tombstone_count"] = current_tombstone_count + tombstoned_count

        # Write atomically
        chunk_map_json = json.dumps(chunk_map, indent=2)
        atomic_write(chunk_map_path, chunk_map_json)

        logger.info(
            f"Tombstoned {tombstoned_count} chunks for {len(doc_ids)} documents. "
            f"Total tombstones: {chunk_map['tombstone_count']}"
        )
    else:
        logger.info(f"No chunks found for doc_ids: {doc_ids}")

    return tombstoned_count


def should_rebuild_index(index_dir: Path) -> Tuple[bool, str]:
    """
    Determine if the index should be rebuilt based on tombstone ratio.

    Args:
        index_dir: Directory containing FAISS index and chunk map

    Returns:
        Tuple of (should_rebuild: bool, reason: str)

    Raises:
        FileNotFoundError: If chunk map not found
    """
    chunk_map_path = index_dir / CHUNK_MAP_FILENAME

    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found: {chunk_map_path}")

    with open(chunk_map_path, "r") as f:
        chunk_map = json.load(f)

    tombstone_ratio = _compute_tombstone_ratio(chunk_map)
    tombstone_count = chunk_map.get("tombstone_count", 0)
    index_size = chunk_map.get("index_size", 0)

    if tombstone_ratio > TOMBSTONE_REBUILD_THRESHOLD:
        reason = (
            f"Tombstone ratio {tombstone_ratio:.1%} exceeds threshold "
            f"({TOMBSTONE_REBUILD_THRESHOLD:.0%}). "
            f"{tombstone_count} of {index_size} chunks are tombstoned. "
            f"Rebuilding will reclaim space and improve search performance."
        )
        return True, reason
    elif tombstone_ratio > TOMBSTONE_WARNING_THRESHOLD:
        reason = (
            f"Tombstone ratio {tombstone_ratio:.1%} is elevated but below rebuild threshold. "
            f"{tombstone_count} of {index_size} chunks are tombstoned. "
            f"Consider rebuilding if search performance degrades."
        )
        return False, reason
    else:
        reason = (
            f"Tombstone ratio {tombstone_ratio:.1%} is healthy. "
            f"{tombstone_count} of {index_size} chunks are tombstoned. "
            f"No rebuild needed."
        )
        return False, reason
