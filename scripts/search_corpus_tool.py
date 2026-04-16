#!/usr/bin/env python3
"""
search_corpus tool for agentic LLM tool calling.

This module implements the search_corpus tool that allows local LLMs
to search the agent's document corpus using semantic similarity.

Usage:
    from search_corpus_tool import create_search_corpus_tool
    from agentic import ToolRegistry

    registry = ToolRegistry()
    schema, executor = create_search_corpus_tool(
        index=faiss_index,
        chunk_map=chunk_map,
        embedder=embedder,
        corpus_dir=Path("~/Documents/Corpora/corpus")
    )
    registry.register("search_corpus", schema, executor)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import logging
import sys

logger = logging.getLogger("search_corpus_tool")

# Allow running as a script (from scripts/ directory) without installing
# the package first — ensures ``grounding.citations`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from grounding.citations import format_citation_prefix  # noqa: E402
from grounding.hybrid import HybridConfig  # noqa: E402
from grounding.reranker import RerankConfig, reassign_ranks  # noqa: E402
from grounding import reranker as _reranker_module  # noqa: E402


# =============================================================================
# Tool Schema
# =============================================================================

def get_search_corpus_schema() -> dict:
    """
    Return the JSON schema for the search_corpus tool.

    This schema follows the OpenAI function calling format
    which Ollama also supports.

    Returns:
        Tool schema dict in OpenAI function calling format
    """
    return {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the agent's document corpus for relevant information "
                "using semantic similarity. Returns text chunks from ingested "
                "PDFs and documents that match the query. Use this tool when "
                "you need specific information from the knowledge base to "
                "answer a question accurately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search query describing what "
                            "information you need. Be specific for better results."
                        )
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of results to return. Default is 5. "
                            "Use more for broad topics, fewer for specific facts."
                        ),
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    }


# =============================================================================
# Tool Executor
# =============================================================================

@dataclass
class SearchCorpusTool:
    """
    Tool executor for corpus search.

    Wraps the existing FAISS search functionality for use
    in the agentic tool calling loop.
    """
    index: Any  # faiss.IndexFlatL2
    chunk_map: list  # List of chunk entries (strings or dicts)
    embedder: Any  # SentenceTransformer
    corpus_dir: Path
    rerank_config: RerankConfig | None = None
    hybrid_config: HybridConfig | None = None
    embeddings_dir: Path | None = None

    def execute(self, arguments: dict) -> str:
        """
        Execute a corpus search and return formatted results.

        Args:
            arguments: Dict with 'query' (required) and 'top_k' (optional)

        Returns:
            Formatted string with search results or error message
        """
        # Validate arguments
        validation_error = self._validate_arguments(arguments)
        if validation_error:
            return validation_error

        query = arguments["query"].strip()
        top_k = self._clamp_top_k(arguments.get("top_k", 5))

        rerank_cfg = self.rerank_config
        rerank_on = bool(rerank_cfg and rerank_cfg.enabled)
        hybrid_cfg = self.hybrid_config
        hybrid_on = bool(hybrid_cfg and hybrid_cfg.enabled)

        # Perform search
        try:
            if hybrid_on:
                pool_k = max(
                    hybrid_cfg.pool_size,
                    rerank_cfg.pool_size if rerank_on else 0,
                    top_k,
                )
                results = self._search_hybrid(query, pool_k)
            else:
                fetch_k = max(rerank_cfg.pool_size, top_k) if rerank_on else top_k
                results = self._search(query, fetch_k)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {e}"

        if rerank_on and results:
            reranked = _reranker_module.rerank(
                query, results, config=rerank_cfg, text_key="content"
            )
            results = reassign_ranks(reranked[:top_k])
        elif hybrid_on and len(results) > top_k:
            # Hybrid fetched a wider pool; truncate when no rerank follows.
            results = results[:top_k]

        # Format results
        if not results:
            return self._format_no_results(query)

        return self._format_results(results, query)

    def _validate_arguments(self, arguments: dict) -> str | None:
        """
        Validate arguments, return error message or None if valid.

        Args:
            arguments: Arguments dict to validate

        Returns:
            Error message string if invalid, None if valid
        """
        if "query" not in arguments:
            return "Error: 'query' parameter is required"

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' must be a non-empty string"

        top_k = arguments.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int):
                return "Error: 'top_k' must be an integer"

        return None

    def _clamp_top_k(self, top_k: int) -> int:
        """Clamp top_k to valid range (1-20)."""
        if not isinstance(top_k, int):
            return 5
        return min(max(top_k, 1), 20)

    def _search(self, query: str, top_k: int) -> list[dict]:
        """
        Perform semantic search using FAISS.

        Reuses the existing search logic from local_rag.py.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of result dicts with rank, source, content, score
        """
        # Embed the query
        query_vector = self.embedder.encode([query], normalize_embeddings=True)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        # Build results list
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunk_map):
                continue

            chunk_entry = self.chunk_map[idx]

            # Handle both old format (string path) and new format (dict with file_path)
            if isinstance(chunk_entry, str):
                chunk_path = chunk_entry
                source = Path(chunk_entry).name
            else:
                chunk_path = chunk_entry.get("file_path", "")
                source = chunk_entry.get("source", Path(chunk_path).name if chunk_path else "unknown")

            if not chunk_path:
                continue

            # Read chunk content
            chunk_data = self._read_chunk(Path(chunk_path))

            results.append({
                "rank": i + 1,
                "source": chunk_data.get("source", source),
                "content": chunk_data["content"],
                "score": float(distances[0][i]),
                "chunk_id": chunk_data.get("chunk_id", "?"),
                "doc_id": chunk_data.get("doc_id", "?"),
                "page_start": chunk_data.get("page_start"),
                "page_end": chunk_data.get("page_end"),
                "section_heading": chunk_data.get("section_heading"),
            })

        return results

    def _search_hybrid(self, query: str, pool_k: int) -> list[dict]:
        """Run hybrid (BM25 + dense) retrieval and enrich results.

        Wraps ``grounding.hybrid.search_hybrid`` using the tool's already-
        loaded FAISS index and embedder to avoid reloading from disk. BM25
        loader uses its default (loads from ``embeddings_dir``).
        """
        from grounding.hybrid import search_hybrid

        if self.embeddings_dir is None:
            raise ValueError(
                "hybrid_config is enabled but embeddings_dir was not provided"
            )

        # Adapt chunk_map: search_hybrid expects a dict with "chunks" key.
        adapted_map = (
            self.chunk_map
            if isinstance(self.chunk_map, dict)
            else {"chunks": self.chunk_map}
        )

        def _load_index_fn(_dir):
            return self.index, adapted_map

        def _embed_fn(q: str):
            return self.embedder.encode([q], normalize_embeddings=True)[0]

        hits = search_hybrid(
            query,
            self.embeddings_dir,
            top_k=pool_k,
            pool_size=self.hybrid_config.pool_size,
            k_rrf=self.hybrid_config.k_rrf,
            load_index_fn=_load_index_fn,
            embed_fn=_embed_fn,
        )

        return self._enrich_from_chunk_ids(hits)

    def _enrich_from_chunk_ids(self, hits: list[dict]) -> list[dict]:
        """Enrich hybrid hits with body / source / page / section metadata.

        Reuses ``_read_chunk`` so the enrichment shape matches the dense
        ``_search`` output. Hybrid-only keys (``faiss_rank``, ``bm25_rank``,
        ``rrf_score``, optional ``hybrid_degraded``) pass through untouched.
        """
        # Build chunk_id -> file_path lookup from the chunk_map.
        entries = (
            self.chunk_map
            if isinstance(self.chunk_map, list)
            else self.chunk_map.get("chunks", [])
        )
        chunk_id_to_path: dict = {}
        for entry in entries:
            if isinstance(entry, str):
                continue
            cid = entry.get("chunk_id")
            fp = entry.get("file_path", "")
            if cid and fp:
                chunk_id_to_path[cid] = (fp, entry.get("source", Path(fp).name))

        enriched: list[dict] = []
        for rank_i, hit in enumerate(hits, start=1):
            fp_and_source = chunk_id_to_path.get(hit["chunk_id"])
            if not fp_and_source:
                continue
            chunk_path, fallback_source = fp_and_source
            chunk_data = self._read_chunk(Path(chunk_path))

            result = {
                "rank": rank_i,
                "source": chunk_data.get("source", fallback_source),
                "content": chunk_data["content"],
                "score": float(hit.get("rrf_score", 0.0)),
                "chunk_id": chunk_data.get("chunk_id", hit.get("chunk_id", "?")),
                "doc_id": chunk_data.get("doc_id", hit.get("doc_id") or "?"),
                "page_start": chunk_data.get("page_start"),
                "page_end": chunk_data.get("page_end"),
                "section_heading": chunk_data.get("section_heading"),
                "faiss_rank": hit.get("faiss_rank"),
                "bm25_rank": hit.get("bm25_rank"),
                "rrf_score": hit.get("rrf_score"),
            }
            if "hybrid_degraded" in hit:
                result["hybrid_degraded"] = hit["hybrid_degraded"]
            enriched.append(result)

        return enriched

    def _read_chunk(self, chunk_path: Path) -> dict:
        """
        Read a chunk file and parse its content and metadata.

        Args:
            chunk_path: Relative path to chunk file

        Returns:
            Dict with content, source, and metadata
        """
        full_path = self.corpus_dir / chunk_path

        if not full_path.exists():
            return {
                "content": "[Chunk file not found]",
                "source": str(chunk_path),
            }

        content = full_path.read_text()

        # Parse YAML front matter if present
        metadata = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    metadata = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except Exception:
                    body = content

        return {
            "content": body,
            "source": metadata.get("source", str(chunk_path)),
            "chunk_id": metadata.get("chunk_id", "?"),
            "doc_id": metadata.get("doc_id", "?"),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "section_heading": metadata.get("section_heading"),
        }

    def _format_results(self, results: list[dict], query: str) -> str:
        """
        Format search results as readable text for the LLM.

        Args:
            results: List of result dicts from search
            query: Original search query

        Returns:
            Formatted string with results
        """
        lines = [f"Found {len(results)} result(s) for: \"{query}\"\n"]

        for r in results:
            lines.append(f"--- Result {r['rank']} (source: {r['source']}) ---")
            lines.append(
                format_citation_prefix(
                    r.get("source", ""),
                    r.get("page_start"),
                    r.get("page_end"),
                    r.get("section_heading"),
                )
            )
            lines.append(r["content"].strip())
            lines.append("")

        return "\n".join(lines)

    def _format_no_results(self, query: str) -> str:
        """
        Format message when no results found.

        Args:
            query: Original search query

        Returns:
            Formatted no-results message
        """
        return (
            f"No results found for: \"{query}\"\n\n"
            "Try:\n"
            "- Using different keywords\n"
            "- Broadening your search terms\n"
            "- Checking if this topic is in the corpus"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_search_corpus_tool(
    index: Any,
    chunk_map: list,
    embedder: Any,
    corpus_dir: Path,
    rerank_config: RerankConfig | None = None,
    hybrid_config: HybridConfig | None = None,
    embeddings_dir: Path | None = None,
) -> tuple[dict, Callable[[dict], str]]:
    """
    Create a search_corpus tool with its schema and executor.

    This factory function creates a tool that can be registered
    with a ToolRegistry for use in the agentic loop.

    Args:
        index: FAISS index (faiss.IndexFlatL2 or similar)
        chunk_map: List of chunk entries (strings or dicts)
        embedder: SentenceTransformer model for query embedding
        corpus_dir: Path to corpus directory

    Returns:
        Tuple of (schema dict, executor callable)

    Example:
        schema, executor = create_search_corpus_tool(
            index=faiss_index,
            chunk_map=chunk_map,
            embedder=embedder,
            corpus_dir=Path("~/Documents/Corpora/corpus")
        )

        registry = ToolRegistry()
        registry.register("search_corpus", schema, executor)
    """
    tool = SearchCorpusTool(
        index=index,
        chunk_map=chunk_map,
        embedder=embedder,
        corpus_dir=Path(corpus_dir).expanduser(),
        rerank_config=rerank_config,
        hybrid_config=hybrid_config,
        embeddings_dir=Path(embeddings_dir).expanduser() if embeddings_dir else None,
    )

    schema = get_search_corpus_schema()

    def executor(arguments: dict) -> str:
        return tool.execute(arguments)

    return schema, executor
