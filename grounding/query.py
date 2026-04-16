"""
Query interface for semantic search over PDF corpus.

This module provides both a Python API and CLI for querying a processed
PDF corpus using natural language queries. Results are ranked by semantic
similarity using FAISS vector search.

Implements Epic 6 Story 6.4.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import typer
import yaml

from grounding import embedder, vector_store

logger = logging.getLogger("grounding.query")

# Performance targets (milliseconds)
TARGET_EMBEDDING_MS = 100
TARGET_SEARCH_MS = 100
TARGET_CHUNK_LOAD_MS = 50


@dataclass
class ChunkResult:
    """Result from a semantic query containing chunk content and metadata."""

    chunk_id: str
    score: float  # 0-1, higher is better
    content: str  # Markdown content
    metadata: dict  # From YAML front matter
    source_document: str  # Original PDF name
    is_music: bool = False  # True if this is a music chunk
    music_metadata: Optional[dict] = None  # Music-specific metadata (key, harmony, rhythm)

    def __str__(self) -> str:
        """Return human-readable string representation."""
        # Format music chunks differently
        if self.is_music and self.music_metadata:
            music_info = []
            if self.music_metadata.get("key"):
                music_info.append(f"Key: {self.music_metadata['key']}")
            if self.music_metadata.get("time_signature"):
                music_info.append(f"Time: {self.music_metadata['time_signature']}")
            if self.music_metadata.get("harmony"):
                harmony_str = " - ".join(self.music_metadata['harmony'][:6])
                music_info.append(f"Harmony: {harmony_str}")
            if self.music_metadata.get("rhythm"):
                music_info.append(f"Rhythm: {self.music_metadata['rhythm']}")

            music_summary = " | ".join(music_info)
            preview = self.content[:200] + "..." if len(self.content) > 200 else self.content

            return (
                f"[{self.chunk_id}] Score: {self.score:.3f} | Source: {self.source_document} | MUSIC\n"
                f"{music_summary}\n"
                f"{preview}"
            )
        else:
            # Standard text chunk
            preview = self.content[:200] + "..." if len(self.content) > 200 else self.content
            return (
                f"[{self.chunk_id}] Score: {self.score:.3f} | Source: {self.source_document}\n"
                f"{preview}"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def normalize_scores(distances: np.ndarray) -> np.ndarray:
    """
    Convert L2 distances to similarity scores in [0, 1] range.

    FAISS returns L2 distances where lower is better. We convert to
    similarity scores where higher is better using exponential decay.

    Args:
        distances: Array of L2 distances from FAISS

    Returns:
        Similarity scores in range [0, 1], higher is better
    """
    # Exponential decay: distance 0 → score 1.0, large distance → score ~0
    scores = np.exp(-distances)
    return scores


def load_chunk_content(
    corpus_path: Path,
    chunk_id: str,
    index_data: Dict
) -> Tuple[str, dict]:
    """
    Load chunk content and metadata from disk.

    Args:
        corpus_path: Root corpus directory
        chunk_id: Chunk identifier (format: doc_id-chunk_num)
        index_data: Loaded _index.json manifest

    Returns:
        Tuple of (content, metadata_dict)

    Raises:
        FileNotFoundError: If chunk file not found
        ValueError: If chunk file has invalid format
    """
    # Parse chunk_id to extract doc_id and chunk number
    parts = chunk_id.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")

    doc_id = parts[0]
    chunk_num = parts[1]

    # Find slug for this doc_id in manifest
    slug = None
    for doc in index_data.get("docs", []):
        if doc.get("doc_id") == doc_id:
            slug = doc.get("slug")
            break

    if not slug:
        raise ValueError(f"doc_id {doc_id} not found in corpus manifest")

    # Construct chunk file path
    chunk_file = corpus_path / slug / "chunks" / f"ch_{chunk_num}.md"

    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    # Read and parse chunk file
    content_str = chunk_file.read_text(encoding="utf-8")

    # Split YAML front matter and content
    if not content_str.startswith("---\n"):
        raise ValueError(f"Chunk file missing YAML front matter: {chunk_file}")

    # Find second "---" delimiter
    parts = content_str.split("---\n", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid YAML front matter format: {chunk_file}")

    yaml_str = parts[1]
    markdown_content = parts[2].strip()

    # Parse YAML metadata
    try:
        metadata = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML in {chunk_file}: {e}")

    return markdown_content, metadata


def query_corpus(
    corpus_path: Path,
    query: str,
    top_k: int = 5
) -> List[ChunkResult]:
    """
    Query corpus for semantically similar chunks.

    Args:
        corpus_path: Path to corpus directory
        query: Natural language query
        top_k: Number of results to return

    Returns:
        List of ChunkResult objects, sorted by relevance (highest first)

    Raises:
        FileNotFoundError: If corpus or index files missing
        ValueError: If query empty or top_k invalid
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_path}\n"
            f"Run 'grounding --in ./pdfs --out {corpus_path} --emit-embeddings' first."
        )

    # Check for vector index files
    index_file = corpus_path / "_embeddings.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector index not found in {corpus_path}\n"
            f"Re-run ingestion with --emit-embeddings flag to generate embeddings."
        )

    logger.info(f"Querying corpus at {corpus_path} with: {query[:50]}...")

    # Load manifest for chunk-to-file mapping
    manifest_path = corpus_path / "_index.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        index_data = json.load(f)

    # Load vector index
    logger.debug("Loading vector index...")
    index, chunk_map = vector_store.load_vector_index(corpus_path)

    # Generate query embedding (with timing)
    logger.debug("Generating query embedding...")
    t0_embed = time.perf_counter()
    query_embedding = embedder.generate_embedding(query)
    t1_embed = time.perf_counter()
    embed_time_ms = (t1_embed - t0_embed) * 1000
    logger.debug(f"Embedding generation took {embed_time_ms:.1f}ms")
    if embed_time_ms > TARGET_EMBEDDING_MS:
        logger.warning(
            f"Embedding generation exceeded target: {embed_time_ms:.1f}ms > {TARGET_EMBEDDING_MS}ms"
        )

    # Search for similar chunks (with timing)
    logger.debug(f"Searching for top {top_k} similar chunks...")
    t0_search = time.perf_counter()
    raw_results = vector_store.search_similar_chunks(
        index,
        chunk_map,
        query_embedding,
        top_k
    )
    t1_search = time.perf_counter()
    search_time_ms = (t1_search - t0_search) * 1000
    logger.debug(f"Vector search took {search_time_ms:.1f}ms")
    if search_time_ms > TARGET_SEARCH_MS:
        logger.warning(
            f"Vector search exceeded target: {search_time_ms:.1f}ms > {TARGET_SEARCH_MS}ms"
        )

    # Normalize scores
    chunk_ids = [r[0] for r in raw_results]
    distances = np.array([r[1] for r in raw_results])
    scores = normalize_scores(distances)

    # Load chunk content and build results (with timing)
    t0_load = time.perf_counter()
    results = []
    for chunk_id, score in zip(chunk_ids, scores):
        try:
            content, metadata = load_chunk_content(corpus_path, chunk_id, index_data)
            source_doc = metadata.get("source", "unknown")

            # Check for music metadata in vector store
            chunk_meta = vector_store.get_chunk_metadata(chunk_map, chunk_id)
            is_music = chunk_meta.get("is_music", False) if chunk_meta else False
            music_meta = chunk_meta.get("music_metadata") if chunk_meta else None

            result = ChunkResult(
                chunk_id=chunk_id,
                score=float(score),
                content=content,
                metadata=metadata,
                source_document=source_doc,
                is_music=is_music,
                music_metadata=music_meta
            )
            results.append(result)

        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load chunk {chunk_id}: {e}")
            continue
    t1_load = time.perf_counter()
    load_time_ms = (t1_load - t0_load) * 1000
    avg_chunk_load_ms = load_time_ms / len(results) if results else 0
    logger.debug(f"Chunk loading took {load_time_ms:.1f}ms total ({avg_chunk_load_ms:.1f}ms avg per chunk)")
    if avg_chunk_load_ms > TARGET_CHUNK_LOAD_MS:
        logger.warning(
            f"Average chunk load time exceeded target: {avg_chunk_load_ms:.1f}ms > {TARGET_CHUNK_LOAD_MS}ms"
        )

    # Total query time
    total_time_ms = embed_time_ms + search_time_ms + load_time_ms
    logger.info(f"Found {len(results)} results in {total_time_ms:.1f}ms total")

    return results


def format_results_text(query: str, results: List[ChunkResult]) -> str:
    """
    Format results as human-readable text.

    Args:
        query: The original query string
        results: List of ChunkResult objects

    Returns:
        Formatted text string
    """
    if not results:
        return f"Query: {query}\nNo results found."

    lines = [f"Query: {query}", f"Found {len(results)} results:", ""]

    for rank, result in enumerate(results, 1):
        preview = result.content[:200]
        if len(result.content) > 200:
            preview += "..."

        # Header with score, ID, and source
        header = f"[{rank}] Score: {result.score:.2f} | ID: {result.chunk_id} | Source: {result.source_document}"
        if result.is_music:
            header += " | MUSIC"

        lines.append(header)
        lines.append("─" * 60)

        # Show music metadata if available
        if result.is_music and result.music_metadata:
            music_info = []
            if result.music_metadata.get("key"):
                music_info.append(f"Key: {result.music_metadata['key']}")
            if result.music_metadata.get("time_signature"):
                music_info.append(f"Time: {result.music_metadata['time_signature']}")
            if result.music_metadata.get("harmony"):
                harmony_str = " - ".join(result.music_metadata['harmony'][:8])
                music_info.append(f"Harmony: {harmony_str}")
            if result.music_metadata.get("rhythm"):
                music_info.append(f"Rhythm: {result.music_metadata['rhythm']}")

            if music_info:
                lines.append("Music Info: " + " | ".join(music_info))
                lines.append("")

        lines.append(preview)
        lines.append("─" * 60)
        lines.append("")

    return "\n".join(lines)


def format_results_json(query: str, results: List[ChunkResult]) -> str:
    """
    Format results as JSON.

    Args:
        query: The original query string
        results: List of ChunkResult objects

    Returns:
        JSON string
    """
    output = {
        "query": query,
        "result_count": len(results),
        "results": []
    }

    for rank, result in enumerate(results, 1):
        result_dict = result.to_dict()
        result_dict["rank"] = rank
        output["results"].append(result_dict)

    return json.dumps(output, indent=2)


# CLI interface
app = typer.Typer(help="Query PDF corpus using semantic search")


@app.command()
def main(
    corpus: Path = typer.Option(..., "--corpus", help="Path to corpus directory"),
    query: str = typer.Option(..., "--query", help="Natural language query"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return"),
    format: str = typer.Option("text", "--format", help="Output format: text or json")
):
    """
    Query a PDF corpus using semantic search.

    Returns the top-k most semantically similar chunks to the query.
    """
    try:
        # Run query
        results = query_corpus(corpus, query, top_k)

        # Format and print results
        if format == "json":
            output = format_results_json(query, results)
        elif format == "text":
            output = format_results_text(query, results)
        else:
            typer.echo(f"Error: Invalid format '{format}'. Use 'text' or 'json'.", err=True)
            raise typer.Exit(1)

        typer.echo(output)

    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
