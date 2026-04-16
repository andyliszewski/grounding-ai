#!/usr/bin/env python3
"""
MCP Server for Corpus Search

Exposes FAISS semantic search as a tool for Claude Code.
Agents can search their filtered corpora during conversations.

Usage:
    python -m mcp_servers.corpus_search.server

Configuration (via environment or .mcp.json):
    CORPUS_DIR: Path to corpus directory
    EMBEDDINGS_DIR: Path to embeddings directory
    AGENTS_DIR: Path to agent YAML definitions
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import faiss
import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer

from grounding.citations import format_citation_prefix
from grounding.hybrid import HybridConfig, search_hybrid
from grounding.reranker import RerankConfig, reassign_ranks
from grounding import reranker as _reranker_module

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("corpus_search")

# Defaults (can be overridden via environment)
DEFAULT_CORPUS = Path.home() / "Documents/Corpora/corpus"
DEFAULT_EMBEDDINGS = Path.home() / "Documents/Corpora/embeddings"
DEFAULT_AGENTS = Path(__file__).parent.parent.parent / "agents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Lazy-loaded resources
_embedder: Optional[SentenceTransformer] = None
_index_cache: dict = {}  # agent_name -> (index, chunk_map)


def get_config() -> dict:
    """Get configuration from environment variables."""
    return {
        "corpus_dir": Path(os.environ.get("CORPUS_DIR", DEFAULT_CORPUS)),
        "embeddings_dir": Path(os.environ.get("EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS)),
        "agents_dir": Path(os.environ.get("AGENTS_DIR", DEFAULT_AGENTS)),
    }


def get_embedder() -> SentenceTransformer:
    """Get or initialize the embedding model (lazy loading)."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _embedder


def load_faiss_index(embeddings_dir: Path, agent_name: str):
    """Load FAISS index and chunk map for an agent (with caching)."""
    global _index_cache

    if agent_name in _index_cache:
        return _index_cache[agent_name]

    agent_embeddings = embeddings_dir / agent_name
    index_path = agent_embeddings / "_embeddings.faiss"
    chunk_map_path = agent_embeddings / "_chunk_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found: {chunk_map_path}")

    logger.info(f"Loading FAISS index for agent: {agent_name}")
    index = faiss.read_index(str(index_path))

    with open(chunk_map_path) as f:
        chunk_map_data = json.load(f)

    # Handle both old format (list) and new format (dict with "chunks" key)
    if isinstance(chunk_map_data, list):
        chunk_map = chunk_map_data
    elif isinstance(chunk_map_data, dict) and "chunks" in chunk_map_data:
        chunk_map = chunk_map_data["chunks"]
    else:
        raise ValueError(f"Unknown chunk map format in {chunk_map_path}")

    _index_cache[agent_name] = (index, chunk_map)
    logger.info(f"Loaded {index.ntotal} vectors for {agent_name}")

    return index, chunk_map


def read_chunk(chunk_path: Path, corpus_dir: Path) -> dict:
    """Read a chunk file and parse its content and metadata."""
    full_path = corpus_dir / chunk_path

    if not full_path.exists():
        return {"path": str(chunk_path), "content": "[Chunk file not found]", "metadata": {}}

    content = full_path.read_text()
    metadata = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except yaml.YAMLError:
                body = content

    return {
        "path": str(chunk_path),
        "content": body,
        "metadata": metadata,
    }


def search_corpus(
    query: str,
    agent_name: str,
    top_k: int = DEFAULT_TOP_K,
    rerank_config: Optional[RerankConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
) -> list[dict]:
    """
    Search the corpus for chunks similar to the query.

    Args:
        query: Natural language query
        agent_name: Agent name (determines which filtered index to use)
        top_k: Number of results to return
        rerank_config: Optional cross-encoder rerank config. When enabled,
            FAISS fetches max(pool_size, top_k) candidates and the
            cross-encoder produces the final top-k ordering.

    Returns:
        List of chunk results with content, metadata, and scores
    """
    config = get_config()

    # Load index and embedder
    index, chunk_map = load_faiss_index(config["embeddings_dir"], agent_name)
    embedder = get_embedder()

    rerank_on = bool(rerank_config and rerank_config.enabled)
    hybrid_on = bool(hybrid_config and hybrid_config.enabled)
    agent_embeddings_dir = config["embeddings_dir"] / agent_name

    results = []
    if hybrid_on:
        pool_k = max(
            hybrid_config.pool_size,
            rerank_config.pool_size if rerank_on else 0,
            top_k,
        )

        adapted_map = chunk_map if isinstance(chunk_map, dict) else {"chunks": chunk_map}

        def _load_index_fn(_dir):
            return index, adapted_map

        def _embed_fn(q: str):
            return embedder.encode([q], normalize_embeddings=True)[0]

        hits = search_hybrid(
            query,
            agent_embeddings_dir,
            top_k=pool_k,
            pool_size=hybrid_config.pool_size,
            k_rrf=hybrid_config.k_rrf,
            load_index_fn=_load_index_fn,
            embed_fn=_embed_fn,
        )

        chunk_id_to_entry: dict = {}
        entries = chunk_map if isinstance(chunk_map, list) else chunk_map.get("chunks", [])
        for entry in entries:
            if isinstance(entry, dict):
                cid = entry.get("chunk_id")
                if cid:
                    chunk_id_to_entry[cid] = entry

        for rank_i, hit in enumerate(hits, start=1):
            entry = chunk_id_to_entry.get(hit["chunk_id"])
            if not entry:
                continue
            chunk_path = entry.get("file_path", "")
            if not chunk_path:
                continue
            chunk_data = read_chunk(Path(chunk_path), config["corpus_dir"])
            result = {
                "rank": rank_i,
                "score": round(float(hit.get("rrf_score", 0.0)), 4),
                "source": chunk_data["metadata"].get("source", Path(chunk_path).parts[0]),
                "doc_id": chunk_data["metadata"].get("doc_id", hit.get("doc_id") or ""),
                "chunk_id": chunk_data["metadata"].get("chunk_id", hit.get("chunk_id", "")),
                "page_start": chunk_data["metadata"].get("page_start"),
                "page_end": chunk_data["metadata"].get("page_end"),
                "section_heading": chunk_data["metadata"].get("section_heading"),
                "content": chunk_data["content"],
                "faiss_rank": hit.get("faiss_rank"),
                "bm25_rank": hit.get("bm25_rank"),
                "rrf_score": hit.get("rrf_score"),
            }
            if "hybrid_degraded" in hit:
                result["hybrid_degraded"] = hit["hybrid_degraded"]
            results.append(result)
    else:
        # Dense-only path
        query_vector = embedder.encode([query], normalize_embeddings=True)
        fetch_k = max(rerank_config.pool_size, top_k) if rerank_on else top_k
        distances, indices = index.search(query_vector, fetch_k)

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunk_map):
                continue

            chunk_entry = chunk_map[idx]
            if isinstance(chunk_entry, str):
                chunk_path = chunk_entry
            else:
                chunk_path = chunk_entry.get("file_path", "")

            if not chunk_path:
                continue

            chunk_data = read_chunk(Path(chunk_path), config["corpus_dir"])

            # Convert L2 distance to similarity score (0-1, higher is better)
            import math
            similarity = math.exp(-distances[0][i])

            results.append({
                "rank": i + 1,
                "score": round(similarity, 4),
                "source": chunk_data["metadata"].get("source", Path(chunk_path).parts[0]),
                "doc_id": chunk_data["metadata"].get("doc_id", ""),
                "chunk_id": chunk_data["metadata"].get("chunk_id", ""),
                "page_start": chunk_data["metadata"].get("page_start"),
                "page_end": chunk_data["metadata"].get("page_end"),
                "section_heading": chunk_data["metadata"].get("section_heading"),
                "content": chunk_data["content"],
            })

    if rerank_on and results:
        reranked = _reranker_module.rerank(
            query, results, config=rerank_config, text_key="content"
        )
        results = reassign_ranks(reranked[:top_k])
    elif hybrid_on and len(results) > top_k:
        results = results[:top_k]

    return results


def list_available_agents() -> list[str]:
    """List agents that have embeddings available."""
    config = get_config()
    embeddings_dir = config["embeddings_dir"]

    if not embeddings_dir.exists():
        return []

    agents = []
    for path in embeddings_dir.iterdir():
        if path.is_dir() and (path / "_embeddings.faiss").exists():
            agents.append(path.name)

    return sorted(agents)


def format_results_for_context(results: list[dict], query: str) -> str:
    """Format search results as context for the LLM."""
    if not results:
        return f"No relevant documents found for query: {query}"

    lines = [
        f"## Corpus Search Results",
        f"Query: {query}",
        f"Found {len(results)} relevant chunks:",
        ""
    ]

    for r in results:
        lines.append(f"### [{r['rank']}] {r['source']} (score: {r['score']})")
        lines.append(
            format_citation_prefix(
                r.get("source", ""),
                r.get("page_start"),
                r.get("page_end"),
                r.get("section_heading"),
            )
        )
        lines.append(f"*doc_id: {r['doc_id']}, chunk: {r['chunk_id']}*")
        lines.append("")
        lines.append(r["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# Create MCP server
server = Server("corpus-search")


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="search_corpus",
            description=(
                "Search the agent's corpus for relevant documents using semantic similarity. "
                "Returns chunks from ingested PDFs, EPUBs, and documents that match the query. "
                "Use this to find information in your knowledge base before answering questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent name (e.g., 'scientist', 'ceo'). Determines which filtered corpus to search."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "rerank_enabled": {
                        "type": "boolean",
                        "description": (
                            "Whether to apply cross-encoder reranking after "
                            "FAISS. When true, FAISS fetches a candidate pool "
                            "(rerank_pool_size) and a cross-encoder produces "
                            "the final top-k ordering. Off by default; see "
                            "CLAUDE.md for the latency/quality trade-off."
                        ),
                        "default": False
                    },
                    "rerank_pool_size": {
                        "type": "integer",
                        "description": (
                            "FAISS candidate count fed to the reranker; "
                            "higher pool = better quality, slower. Default 50. "
                            "Ignored unless rerank_enabled is true."
                        ),
                        "default": 50,
                        "minimum": 1
                    },
                    "rerank_model": {
                        "type": "string",
                        "description": (
                            "Override the default reranker model name "
                            "(default: 'BAAI/bge-reranker-base'). Ignored "
                            "unless rerank_enabled is true."
                        ),
                        "default": "BAAI/bge-reranker-base"
                    },
                    "hybrid_enabled": {
                        "type": "boolean",
                        "description": (
                            "Whether to fuse FAISS + BM25 via RRF before "
                            "(optional) rerank. Off by default. See "
                            "CLAUDE.md 'Hybrid Retrieval' for when to enable "
                            "and the latency/quality trade-off."
                        ),
                        "default": False
                    },
                    "hybrid_pool_size": {
                        "type": "integer",
                        "description": (
                            "Candidates fetched from each channel (FAISS and "
                            "BM25) before fusion; higher = better recall, "
                            "slightly slower. Default 50. Ignored unless "
                            "hybrid_enabled is true."
                        ),
                        "default": 50,
                        "minimum": 1
                    },
                    "hybrid_k_rrf": {
                        "type": "integer",
                        "description": (
                            "RRF damping constant (default 60; literature "
                            "standard). Raising it flattens rank "
                            "contributions. Ignored unless hybrid_enabled "
                            "is true."
                        ),
                        "default": 60,
                        "minimum": 1
                    }
                },
                "required": ["query", "agent"]
            }
        ),
        Tool(
            name="list_corpus_agents",
            description="List all agents that have corpus embeddings available for search.",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "search_corpus":
        query = arguments.get("query", "")
        agent = arguments.get("agent", "")
        top_k = min(arguments.get("top_k", DEFAULT_TOP_K), 20)

        if not query:
            return [TextContent(type="text", text="Error: query is required")]
        if not agent:
            return [TextContent(type="text", text="Error: agent is required")]

        rerank_config: Optional[RerankConfig] = None
        if arguments.get("rerank_enabled", False):
            rerank_config = RerankConfig(
                enabled=True,
                model=arguments.get("rerank_model", "BAAI/bge-reranker-base"),
                pool_size=int(arguments.get("rerank_pool_size", 50)),
            )

        hybrid_config: Optional[HybridConfig] = None
        if arguments.get("hybrid_enabled", False):
            hybrid_config = HybridConfig(
                enabled=True,
                pool_size=int(arguments.get("hybrid_pool_size", 50)),
                k_rrf=int(arguments.get("hybrid_k_rrf", 60)),
            )

        try:
            results = search_corpus(
                query,
                agent,
                top_k,
                rerank_config=rerank_config,
                hybrid_config=hybrid_config,
            )
            formatted = format_results_for_context(results, query)
            return [TextContent(type="text", text=formatted)]
        except FileNotFoundError as e:
            available = list_available_agents()
            return [TextContent(
                type="text",
                text=f"Error: {e}\n\nAvailable agents: {', '.join(available)}"
            )]
        except Exception as e:
            logger.exception("Search failed")
            return [TextContent(type="text", text=f"Error: {e}")]

    elif name == "list_corpus_agents":
        agents = list_available_agents()
        if agents:
            return [TextContent(
                type="text",
                text=f"Agents with corpus embeddings:\n" + "\n".join(f"- {a}" for a in agents)
            )]
        else:
            return [TextContent(type="text", text="No agent embeddings found.")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting corpus search MCP server")

    config = get_config()
    logger.info(f"Corpus: {config['corpus_dir']}")
    logger.info(f"Embeddings: {config['embeddings_dir']}")
    logger.info(f"Agents: {config['agents_dir']}")

    # Verify paths exist
    if not config['embeddings_dir'].exists():
        logger.warning(f"Embeddings directory not found: {config['embeddings_dir']}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
