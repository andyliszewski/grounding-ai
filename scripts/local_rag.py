#!/usr/bin/env python3
"""
local_rag.py - Local RAG query tool using existing FAISS indexes and Ollama

Uses your pre-built FAISS embeddings to retrieve relevant chunks from the corpus,
then sends them as context to a local LLM via Ollama's OpenAI-compatible API.

Supports two modes:
- Simple RAG: Always searches corpus before responding (default)
- Agentic Mode: LLM decides when to use tools (search_corpus, read_file, bash, glob, grep, write_file)

Requirements:
    pip install sentence-transformers faiss-cpu requests pyyaml

Usage:
    # Interactive REPL with doctor agent
    python scripts/local_rag.py --agent doctor --agentic

    # Single query
    python scripts/local_rag.py --agent doctor --query "What are TCM patterns for insomnia?"

    # Agentic mode (LLM decides when to search)
    python scripts/local_rag.py --agent scientist --agentic --query "What is 2+2?"

    # With verbose output (shows tool calls)
    python scripts/local_rag.py --agent scientist -A -v --query "Explain entropy"

    # Use larger model
    python scripts/local_rag.py --agent scientist --model qwen2.5:32b

    # Adjust retrieval
    python scripts/local_rag.py --agent survivor --top-k 10
"""

import argparse
import json
import logging
import sys
import textwrap
import threading
import itertools
import shutil
from pathlib import Path
from typing import Optional

import faiss
import requests
import yaml
from sentence_transformers import SentenceTransformer

# Agentic mode imports
from agentic import run_agentic_loop, AgenticConfig, ToolRegistry, format_tool_call_trace
from search_corpus_tool import create_search_corpus_tool
from filesystem_tools import create_all_filesystem_tools
from grounding.citations import format_citation_prefix
from grounding.hybrid import HybridConfig
from grounding.reranker import RerankConfig, reassign_ranks
from grounding import reranker as _reranker_module

logger = logging.getLogger("local_rag")


# =============================================================================
# UI Helpers
# =============================================================================

class Spinner:
    """Simple terminal spinner for indicating work in progress."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

    def _spin(self):
        while self.running:
            char = next(self.spinner_chars)
            sys.stdout.write(f'\r{char} {self.message}...')
            sys.stdout.flush()
            threading.Event().wait(0.1)
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)


def wrap_text(text: str, width: int = None) -> str:
    """Wrap text to terminal width for cleaner output."""
    if width is None:
        # Get terminal width, default to 80 if not available
        width = shutil.get_terminal_size().columns - 4  # Leave margin

    paragraphs = text.split('\n\n')
    wrapped_paragraphs = []

    for para in paragraphs:
        # Preserve code blocks and lists
        lines = para.split('\n')
        wrapped_lines = []

        for line in lines:
            # Don't wrap code blocks, lists, or already short lines
            if (line.startswith('```') or
                line.startswith('    ') or
                line.startswith('- ') or
                line.startswith('* ') or
                line.startswith('  ') or
                len(line) <= width):
                wrapped_lines.append(line)
            else:
                # Wrap long lines
                wrapped = textwrap.fill(line, width=width)
                wrapped_lines.append(wrapped)

        wrapped_paragraphs.append('\n'.join(wrapped_lines))

    return '\n\n'.join(wrapped_paragraphs)


# Defaults
DEFAULT_CORPUS = Path.home() / "Documents/Corpora/corpus"
DEFAULT_EMBEDDINGS = Path.home() / "Documents/Corpora/embeddings"
DEFAULT_AGENTS = Path(__file__).parent.parent / "agents"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_agent_persona(agent_name: str, agents_dir: Path) -> dict:
    """Load agent YAML and extract persona information."""
    agent_file = agents_dir / f"{agent_name}.yaml"
    if not agent_file.exists():
        return {"name": agent_name, "style": "", "expertise": [], "greeting": ""}

    with open(agent_file) as f:
        config = yaml.safe_load(f)

    persona = config.get("persona", {})
    return {
        "name": agent_name,
        "description": config.get("description", ""),
        "style": persona.get("style", ""),
        "expertise": persona.get("expertise", []),
        "greeting": persona.get("greeting", ""),
        "icon": persona.get("icon", ""),
    }


def load_faiss_index(embeddings_dir: Path, agent_name: str):
    """Load FAISS index and chunk map for an agent."""
    agent_embeddings = embeddings_dir / agent_name

    index_path = agent_embeddings / "_embeddings.faiss"
    chunk_map_path = agent_embeddings / "_chunk_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found: {chunk_map_path}")

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

    return index, chunk_map


def read_chunk(chunk_path: Path, corpus_dir: Path) -> dict:
    """Read a chunk file and parse its content and metadata."""
    full_path = corpus_dir / chunk_path

    if not full_path.exists():
        return {"path": str(chunk_path), "content": "[Chunk file not found]", "metadata": {}}

    content = full_path.read_text()

    # Parse YAML front matter if present
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
    index: faiss.Index,
    chunk_map: list,
    embedder: SentenceTransformer,
    corpus_dir: Path,
    top_k: int = 5,
    rerank_config: Optional[RerankConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    embeddings_dir: Optional[Path] = None,
) -> list[dict]:
    """Search the corpus and return top-k relevant chunks.

    When ``hybrid_config.enabled`` is true, BM25 + dense are fused via RRF
    before (optional) rerank; otherwise the pre-19.3 dense-only path runs.
    When ``rerank_config.enabled``, a cross-encoder scores the pool and
    produces the final top-k. With both off, behavior is unchanged.
    """
    rerank_on = bool(rerank_config and rerank_config.enabled)
    hybrid_on = bool(hybrid_config and hybrid_config.enabled)

    results: list[dict] = []

    if hybrid_on and embeddings_dir is not None:
        from grounding.hybrid import search_hybrid

        pool_k = max(
            hybrid_config.pool_size,
            rerank_config.pool_size if rerank_on else 0,
            top_k,
        )

        def _load_index_fn(_dir):
            # Reuse already-loaded index/chunk_map rather than reloading from disk.
            # search_hybrid expects a chunk_map dict (with "chunks" list); adapt.
            adapted_map = chunk_map if isinstance(chunk_map, dict) else {"chunks": chunk_map}
            return index, adapted_map

        def _embed_fn(q: str):
            return embedder.encode([q], normalize_embeddings=True)[0]

        hybrid_hits = search_hybrid(
            query,
            embeddings_dir,
            top_k=pool_k,
            pool_size=hybrid_config.pool_size,
            k_rrf=hybrid_config.k_rrf,
            load_index_fn=_load_index_fn,
            embed_fn=_embed_fn,
        )

        # Enrich with body / metadata from each chunk's file.
        chunk_id_to_path = {}
        entries = chunk_map if isinstance(chunk_map, list) else chunk_map.get("chunks", [])
        for entry in entries:
            if isinstance(entry, str):
                continue
            cid = entry.get("chunk_id")
            fp = entry.get("file_path", "")
            if cid and fp:
                chunk_id_to_path[cid] = fp

        for rank_i, hit in enumerate(hybrid_hits, start=1):
            fp = chunk_id_to_path.get(hit["chunk_id"], "")
            if not fp:
                continue
            chunk_data = read_chunk(Path(fp), corpus_dir)
            chunk_data["score"] = float(hit.get("rrf_score", 0.0))
            chunk_data["rank"] = rank_i
            chunk_data["faiss_rank"] = hit.get("faiss_rank")
            chunk_data["bm25_rank"] = hit.get("bm25_rank")
            chunk_data["rrf_score"] = hit.get("rrf_score")
            if "hybrid_degraded" in hit:
                chunk_data["hybrid_degraded"] = hit["hybrid_degraded"]
            results.append(chunk_data)
    else:
        # Dense-only (pre-19.3 path)
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

            chunk_data = read_chunk(Path(chunk_path), corpus_dir)
            chunk_data["score"] = float(distances[0][i])
            chunk_data["rank"] = i + 1
            results.append(chunk_data)

    if rerank_on and results:
        reranked = _reranker_module.rerank(
            query, results, config=rerank_config, text_key="content"
        )
        results = reassign_ranks(reranked[:top_k])
    elif hybrid_on and len(results) > top_k:
        # Hybrid fetched a wider pool; truncate when no rerank follows.
        results = results[:top_k]

    return results


def format_context(chunks: list[dict], include_sources: bool = True) -> str:
    """Format retrieved chunks as context for the LLM."""
    context_parts = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        source = metadata.get("source", chunk["path"])
        content = chunk["content"]

        if include_sources:
            citation = format_citation_prefix(
                source,
                metadata.get("page_start"),
                metadata.get("page_end"),
                metadata.get("section_heading"),
            )
            context_parts.append(f"[Source: {source}]\n{citation}\n{content}")
        else:
            context_parts.append(content)

    return "\n\n---\n\n".join(context_parts)


def build_system_prompt(persona: dict) -> str:
    """Build system prompt from agent persona."""
    parts = []

    if persona.get("icon") and persona.get("name"):
        parts.append(f"{persona['icon']} {persona['name'].title()}")

    if persona.get("style"):
        parts.append(persona["style"].strip())

    if persona.get("expertise"):
        expertise_list = "\n".join(f"- {e}" for e in persona["expertise"])
        parts.append(f"Your areas of expertise:\n{expertise_list}")

    parts.append(
        "Answer questions based on the provided context. "
        "If the context doesn't contain relevant information, say so clearly. "
        "Cite sources when possible."
    )

    return "\n\n".join(parts)


def query_llm(
    query: str,
    context: str,
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.7,
    show_spinner: bool = True,
) -> str:
    """Send query with context to Ollama and return response."""
    user_message = f"""Context from reference materials:

{context}

---

Question: {query}"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "stream": False,
    }

    spinner = None
    if show_spinner:
        spinner = Spinner("Thinking")
        spinner.start()

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "[Error: Cannot connect to Ollama. "
            "Make sure Ollama is running (ollama serve or check 'ollama list').]"
        )
    except requests.exceptions.Timeout:
        return "[Error: LLM request timed out]"
    except Exception as e:
        return f"[Error querying LLM: {e}]"
    finally:
        if spinner:
            spinner.stop()


def print_sources(chunks: list[dict]):
    """Print source information for retrieved chunks."""
    print("\n--- Sources ---")
    for chunk in chunks:
        source = chunk["metadata"].get("source", Path(chunk["path"]).name)
        doc_id = chunk["metadata"].get("doc_id", "?")
        chunk_id = chunk["metadata"].get("chunk_id", "?")
        score = chunk.get("score", 0)
        print(f"  [{chunk['rank']}] {source} (doc:{doc_id} chunk:{chunk_id}) score:{score:.4f}")


def interactive_repl(
    index: faiss.Index,
    chunk_map: list,
    embedder: SentenceTransformer,
    corpus_dir: Path,
    persona: dict,
    top_k: int,
    model: str,
    api_url: str,
    show_sources: bool = True,
    agentic: bool = False,
    max_iterations: int = 5,
    verbose: bool = False,
    max_tokens: int = 2048,
    rerank_config: Optional[RerankConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    embeddings_dir: Optional[Path] = None,
):
    """Run interactive query REPL supporting both simple RAG and agentic modes."""
    # Setup for agentic mode if enabled
    registry = None
    config = None
    if agentic:
        registry = ToolRegistry()
        schema, executor = create_search_corpus_tool(
            index=index,
            chunk_map=chunk_map,
            embedder=embedder,
            corpus_dir=corpus_dir,
            rerank_config=rerank_config,
            hybrid_config=hybrid_config,
            embeddings_dir=embeddings_dir,
        )
        registry.register("search_corpus", schema, executor)

        # Register filesystem tools
        for name, schema, executor in create_all_filesystem_tools():
            registry.register(name, schema, executor)

        config = AgenticConfig(
            max_iterations=max_iterations,
            verbose=verbose,
            num_predict=max_tokens
        )
        system_prompt = build_agentic_system_prompt(persona)
    else:
        system_prompt = build_system_prompt(persona)

    # Print greeting
    mode_indicator = " [Agentic Mode]" if agentic else ""
    if persona.get("greeting"):
        print(f"\n{persona.get('icon', '')} {persona['greeting'].strip()}{mode_indicator}\n")
    else:
        print(f"\n{persona.get('icon', '')} {persona['name']} agent ready.{mode_indicator}\n")

    print("Commands: 'quit' to exit, 'sources' to toggle source display, 'clear' to reset conversation")
    print("-" * 60)

    # Prompt indicator
    prompt = "\n[Agentic] You: " if agentic else "\nYou: "

    # Conversation history for multi-turn (agentic mode only)
    conversation_history = None
    if agentic:
        # Initialize with system prompt
        conversation_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            query = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if query.lower() == "sources":
            show_sources = not show_sources
            print(f"Source display: {'on' if show_sources else 'off'}")
            continue

        if query.lower() == "clear":
            if agentic:
                conversation_history = [{"role": "system", "content": system_prompt}]
                print("Conversation cleared. Starting fresh.")
            else:
                print("(Clear only applies to agentic mode)")
            continue

        if agentic:
            # Agentic mode: LLM decides when to search
            result = run_agentic_loop(
                user_query=query,
                tools=registry.schemas,
                tool_executor=registry.execute,
                model=model,
                api_url=api_url,
                config=config,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )

            # Update conversation history for next turn
            conversation_history = result.messages

            if verbose:
                print(format_tool_call_trace(result))
                print()

            # Wrap text for cleaner output
            wrapped = wrap_text(result.content)
            print(f"\n{persona.get('icon', '>')} {wrapped}")

        else:
            # Simple RAG mode: always search first
            chunks = search_corpus(
                query,
                index,
                chunk_map,
                embedder,
                corpus_dir,
                top_k,
                rerank_config=rerank_config,
                hybrid_config=hybrid_config,
                embeddings_dir=embeddings_dir,
            )

            if not chunks:
                print("\nNo relevant documents found in corpus.")
                continue

            # Format context and query LLM
            context = format_context(chunks)
            response = query_llm(query, context, system_prompt, model, api_url)

            # Wrap text for cleaner output
            wrapped = wrap_text(response)
            print(f"\n{persona.get('icon', '>')} {wrapped}")

            if show_sources:
                print_sources(chunks)


def single_query(
    query: str,
    index: faiss.Index,
    chunk_map: list,
    embedder: SentenceTransformer,
    corpus_dir: Path,
    persona: dict,
    top_k: int,
    model: str,
    api_url: str,
    show_sources: bool = True,
    rerank_config: Optional[RerankConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    embeddings_dir: Optional[Path] = None,
):
    """Execute a single query in simple RAG mode and print results."""
    system_prompt = build_system_prompt(persona)

    # Search corpus
    chunks = search_corpus(
        query,
        index,
        chunk_map,
        embedder,
        corpus_dir,
        top_k,
        rerank_config=rerank_config,
        hybrid_config=hybrid_config,
        embeddings_dir=embeddings_dir,
    )

    if not chunks:
        print("No relevant documents found in corpus.")
        return

    # Format context and query LLM
    context = format_context(chunks)
    response = query_llm(query, context, system_prompt, model, api_url)

    # Wrap text for cleaner output
    print(wrap_text(response))

    if show_sources:
        print_sources(chunks)


def agentic_query(
    query: str,
    index: faiss.Index,
    chunk_map: list,
    embedder: SentenceTransformer,
    corpus_dir: Path,
    persona: dict,
    model: str,
    api_url: str,
    max_iterations: int = 5,
    verbose: bool = False,
    max_tokens: int = 2048,
    rerank_config: Optional[RerankConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    embeddings_dir: Optional[Path] = None,
):
    """Execute a single query in agentic mode and print results."""
    # Create tool registry
    registry = ToolRegistry()
    schema, executor = create_search_corpus_tool(
        index=index,
        chunk_map=chunk_map,
        embedder=embedder,
        corpus_dir=corpus_dir,
        rerank_config=rerank_config,
        hybrid_config=hybrid_config,
        embeddings_dir=embeddings_dir,
    )
    registry.register("search_corpus", schema, executor)

    # Register filesystem tools
    for name, schema, executor in create_all_filesystem_tools():
        registry.register(name, schema, executor)

    # Build system prompt
    system_prompt = build_agentic_system_prompt(persona)

    # Configure agentic loop
    config = AgenticConfig(
        max_iterations=max_iterations,
        verbose=verbose,
        num_predict=max_tokens
    )

    # Run the agentic loop
    result = run_agentic_loop(
        user_query=query,
        tools=registry.schemas,
        tool_executor=registry.execute,
        model=model,
        api_url=api_url,
        config=config,
        system_prompt=system_prompt
    )

    # Print verbose output if enabled
    if verbose:
        print(format_tool_call_trace(result))
        print()

    # Print the response with wrapping
    print(wrap_text(result.content))


def build_agentic_system_prompt(persona: dict) -> str:
    """Build system prompt for agentic mode with tool usage guidance."""
    parts = []

    if persona.get("icon") and persona.get("name"):
        parts.append(f"{persona['icon']} {persona['name'].title()}")

    if persona.get("style"):
        parts.append(persona["style"].strip())

    if persona.get("expertise"):
        expertise_list = "\n".join(f"- {e}" for e in persona["expertise"])
        parts.append(f"Your areas of expertise:\n{expertise_list}")

    # Agentic-specific instructions - emphasize tool calling and depth
    parts.append(
        "You have access to these tools. ACTIVELY USE THEM to provide thorough, well-researched answers:\n\n"
        "KNOWLEDGE BASE (your curated expert sources):\n"
        "- search_corpus(query: str): Search your knowledge base. Use this FIRST for substantive questions. "
        "Your corpus contains carefully curated expert materials - leverage them.\n\n"
        "WEB SEARCH & FETCH:\n"
        "- web_search(query: str): Search the web for current events, news, recent developments\n"
        "- web_fetch(url: str): Fetch full content from a URL\n\n"
        "FILE OPERATIONS:\n"
        "- read_file(path: str): Read local files referenced by the user\n"
        "- glob(pattern: str): Find files matching a pattern (e.g., '**/*.md')\n"
        "- grep(pattern: str): Search for text patterns in files\n"
        "- write_file, edit_file: Create or modify files\n\n"
        "SHELL: bash(command: str): Execute shell commands\n\n"
        "GUIDELINES FOR QUALITY RESPONSES:\n"
        "1. For complex questions, search MULTIPLE times with different queries to gather comprehensive context\n"
        "2. Synthesize information from your corpus with current events when relevant\n"
        "3. Provide thorough analysis, not just summaries - draw connections, identify patterns, offer insights\n"
        "4. When the user references files, READ them and integrate their content into your analysis\n"
        "5. Cite your sources and explain your reasoning\n\n"
        "REMEMBER: You are an expert advisor with access to a deep knowledge base. "
        "Use your tools proactively to provide substantive, well-researched responses. "
        "Don't just describe what you would do - actually DO it by calling the tools."
    )

    return "\n\n".join(parts)


def _resolve_rerank_config_from_args(args):
    """Build a RerankConfig from CLI flags + optional config.yaml, or None.

    Returns ``None`` when no flags are set and no config file was provided,
    preserving bit-for-bit backward compatibility with pre-18.3 behavior.
    On invalid values, prints a clear error and exits 2 (argument error).
    """
    from grounding.config import load_retrieval_config, resolve_rerank_config

    no_flags = (
        not args.rerank
        and args.rerank_model is None
        and args.rerank_pool_size is None
        and args.rerank_top_k is None
    )
    retrieval_cfg = load_retrieval_config(args.config) if args.config else {}
    if no_flags and not retrieval_cfg:
        return None
    try:
        return resolve_rerank_config(
            retrieval_config=retrieval_cfg,
            cli_enabled=args.rerank,
            cli_model=args.rerank_model,
            cli_pool_size=args.rerank_pool_size,
            cli_batch_size=None,
        )
    except ValueError as exc:
        print(f"Error: invalid rerank configuration: {exc}", file=sys.stderr)
        sys.exit(2)


def _resolve_hybrid_config_from_args(args):
    """Build a HybridConfig from CLI flags + optional config.yaml, or None.

    Returns ``None`` when no hybrid flags are set and no config file was
    provided, preserving the pre-19.3 zero-change path bit-for-bit.
    On invalid values, prints a clear error and exits 2.
    """
    from grounding.config import load_retrieval_config, resolve_hybrid_config

    no_flags = (
        not args.hybrid
        and args.hybrid_pool_size is None
        and args.hybrid_k_rrf is None
    )
    retrieval_cfg = load_retrieval_config(args.config) if args.config else {}
    if no_flags and not retrieval_cfg:
        return None
    try:
        return resolve_hybrid_config(
            retrieval_config=retrieval_cfg,
            cli_enabled=args.hybrid,
            cli_pool_size=args.hybrid_pool_size,
            cli_k_rrf=args.hybrid_k_rrf,
        )
    except ValueError as exc:
        print(f"Error: invalid hybrid configuration: {exc}", file=sys.stderr)
        sys.exit(2)


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``local_rag`` argparse parser.

    Extracted so tests can assert against the *real* parser (and catch
    flag-removal drift) rather than mirroring it locally.
    """
    parser = argparse.ArgumentParser(
        description="Local RAG query tool using FAISS indexes and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Agent name (e.g., survivor, scientist, mechanical-engineer)",
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to execute (omit for interactive REPL)",
    )
    parser.add_argument(
        "--corpus", "-c",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Path to corpus directory (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Path to embeddings directory (default: {DEFAULT_EMBEDDINGS})",
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=DEFAULT_AGENTS,
        help=f"Path to agents directory (default: {DEFAULT_AGENTS})",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model name for Ollama (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"LLM API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citations",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (show tool calls in agentic mode)",
    )

    # Agentic mode arguments
    parser.add_argument(
        "--agentic", "-A",
        action="store_true",
        help="Enable agentic mode with tool calling (LLM decides when to search)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum tool-calling iterations in agentic mode (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens in LLM response (default: 2048, increase for longer answers)",
    )

    # Reranking (Story 18.3)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml for retrieval.rerank defaults (CLI flags override config).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking of FAISS results.",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="Cross-encoder model name (default: BAAI/bge-reranker-base).",
    )
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        default=None,
        help="FAISS candidate count fed to the reranker (default: 50).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=(
            "Post-rerank truncation. When omitted, --top-k applies to the final output. "
            "Lets you fetch a large pool (--rerank-pool-size) and return only the best N."
        ),
    )

    # Hybrid retrieval (Story 19.3)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help="Enable BM25 + dense fusion via RRF before (optional) rerank.",
    )
    parser.add_argument(
        "--hybrid-pool-size",
        type=int,
        default=None,
        help=(
            "Candidates fetched from each channel (FAISS + BM25) before fusion "
            "(default: 50). Higher = better recall, slightly slower."
        ),
    )
    parser.add_argument(
        "--hybrid-k-rrf",
        type=int,
        default=None,
        help=(
            "RRF damping constant (default: 60; literature standard). "
            "Raising it flattens rank contributions."
        ),
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve rerank + hybrid configs: CLI > config.yaml > defaults.
    rerank_config = _resolve_rerank_config_from_args(args)
    hybrid_config = _resolve_hybrid_config_from_args(args)
    effective_top_k = args.top_k
    if rerank_config is not None and rerank_config.enabled and args.rerank_top_k:
        effective_top_k = args.rerank_top_k

    # Configure logging for verbose mode
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(name)s: %(message)s'
        )

    # Validate paths
    if not args.corpus.exists():
        print(f"Error: Corpus directory not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    if not args.embeddings.exists():
        print(f"Error: Embeddings directory not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)

    # Load agent persona
    persona = load_agent_persona(args.agent, args.agents_dir)

    # Load FAISS index
    print(f"Loading embeddings for agent: {args.agent}...")
    try:
        index, chunk_map = load_faiss_index(args.embeddings, args.agent)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Available agents: {[d.name for d in args.embeddings.iterdir() if d.is_dir()]}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {index.ntotal} vectors, {len(chunk_map)} chunks")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Log mode
    print(f"Using model: {args.model}")
    if args.agentic:
        print("Running in AGENTIC mode (tool calling enabled)")
    else:
        print("Running in SIMPLE RAG mode")

    # Run query
    show_sources = not args.no_sources

    agent_embeddings_dir = args.embeddings / args.agent

    if args.query:
        # Single query mode
        if args.agentic:
            agentic_query(
                args.query,
                index,
                chunk_map,
                embedder,
                args.corpus,
                persona,
                args.model,
                args.api_url,
                args.max_iterations,
                args.verbose,
                args.max_tokens,
                rerank_config=rerank_config,
                hybrid_config=hybrid_config,
                embeddings_dir=agent_embeddings_dir,
            )
        else:
            single_query(
                args.query,
                index,
                chunk_map,
                embedder,
                args.corpus,
                persona,
                effective_top_k,
                args.model,
                args.api_url,
                show_sources,
                rerank_config=rerank_config,
                hybrid_config=hybrid_config,
                embeddings_dir=agent_embeddings_dir,
            )
    else:
        # Interactive REPL mode
        interactive_repl(
            index,
            chunk_map,
            embedder,
            args.corpus,
            persona,
            effective_top_k,
            args.model,
            args.api_url,
            show_sources,
            agentic=args.agentic,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            max_tokens=args.max_tokens,
            rerank_config=rerank_config,
            hybrid_config=hybrid_config,
            embeddings_dir=agent_embeddings_dir,
        )


if __name__ == "__main__":
    main()
