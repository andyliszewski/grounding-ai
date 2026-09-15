"""In-process ``search_corpus`` tool for the answer benchmark (Epic 25, D2).

The benchmark calls ``mcp_servers.corpus_search.server.search_corpus`` directly
and formats results with that module's ``format_results_for_context``, so the
model sees byte-for-byte what the MCP tool returns (citation prefix, doc/chunk
annotation, body), and the benchmark exercises the exact retrieval code the MCP
server serves, without transport noise.

Deliberate difference from the MCP schema: the tool offered to the model keeps
the MCP tool's name, description, ``query`` and ``top_k``, but drops ``agent``
and the ``rerank_*`` / ``hybrid_*`` arguments. The harness pins those per
condition; letting the model set them would let it switch conditions
mid-answer. ``top_k`` defaults to 5 and is clamped to [1, 20], as in the MCP
server's ``call_tool``.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List

from grounding.citations import _derive_slug, format_citation_prefix
from grounding.eval.answers.conditions import ConditionSpec

logger = logging.getLogger("grounding.eval.answers.tool")

TOOL_NAME = "search_corpus"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Copied from the MCP server's list_tools(); tests/test_eval_answers_runner.py
# asserts this stays identical to the server's definition. It names no
# retrieval method ("semantic similarity" was wrong for the hybrid conditions,
# which add BM25), so one description is accurate in every grounded condition
# and the conditions still differ only in the pinned retrieval settings.
TOOL_DESCRIPTION = (
    "Search the agent's corpus for relevant documents. "
    "Returns chunks from ingested PDFs, EPUBs, and documents that match the query. "
    "Use this to find information in your knowledge base before answering questions."
)

TOOL_DEFINITION: Dict[str, Any] = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 20)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}


class HarnessToolError(RuntimeError):
    """The retrieval stack itself failed (not a bad argument from the model).

    The runner aborts the answer and logs it to errors.jsonl instead of letting
    the model continue on a broken tool, which would score plumbing as a model
    failure.
    """


@dataclass
class ToolExecution:
    query: str
    top_k: int
    output_text: str
    is_error: bool
    latency_s: float
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def citation_prefixes(self) -> List[str]:
        return [r["prefix"] for r in self.results]


def load_mcp_server_module():
    """Import ``mcp_servers.corpus_search.server``.

    ``mcp_servers`` is not part of the installed ``grounding`` package, so the
    ``grounding`` console script cannot import it from ``sys.path`` alone. Fall
    back to the repository checkout that holds this package.
    """
    name = "mcp_servers.corpus_search.server"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == "mcp" or (exc.name or "").startswith("mcp."):
            raise RuntimeError(
                "the answer benchmark calls the corpus-search MCP server in-process "
                "and needs the 'mcp' package: pip install -e '.[bench]'"
            ) from exc
        if not (exc.name or "").startswith("mcp_servers"):
            raise
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "mcp_servers" / "corpus_search" / "server.py").exists():
        raise RuntimeError(
            f"cannot find mcp_servers/corpus_search/server.py next to {repo_root}; "
            "run the benchmark from a grounding-ai checkout"
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module(name)


@contextmanager
def _server_env(corpus_dir: Path, embeddings_root: Path) -> Iterator[None]:
    """Point the MCP server's env-var configuration at this run's corpus."""
    keys = {"CORPUS_DIR": str(corpus_dir), "EMBEDDINGS_DIR": str(embeddings_root)}
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clamp_top_k(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K
    return max(1, min(value, MAX_TOP_K))


class SearchCorpusTool:
    """Executes ``search_corpus`` for one grounded condition.

    ``embeddings_dir`` is the agent's index directory (the same path
    ``grounding eval --embeddings`` takes). The MCP server expects
    ``EMBEDDINGS_DIR/<agent>``, so the tool sets ``EMBEDDINGS_DIR`` to the
    parent directory and passes the directory name as the agent.
    """

    def __init__(
        self,
        condition: ConditionSpec,
        *,
        corpus_dir: Path,
        embeddings_dir: Path,
        server: Any = None,
    ) -> None:
        if not condition.grounded:
            raise ValueError(f"condition '{condition.name}' has no tool")
        self.condition = condition
        self.corpus_dir = Path(corpus_dir).resolve()
        self.embeddings_dir = Path(embeddings_dir).resolve()
        self.index_name = self.embeddings_dir.name
        self._server = server
        self._cache_cleared = False

    @property
    def server(self):
        if self._server is None:
            self._server = load_mcp_server_module()
        if not self._cache_cleared:
            # The server caches indexes by agent name; drop any entry from an
            # earlier run in this process so this run's --embeddings is used.
            cache = getattr(self._server, "_index_cache", None)
            if isinstance(cache, dict):
                cache.pop(self.index_name, None)
            self._cache_cleared = True
        return self._server

    def execute(self, name: str, tool_input: Dict[str, Any] | None) -> ToolExecution:
        tool_input = tool_input or {}
        query = tool_input.get("query", "")
        top_k = _clamp_top_k(tool_input.get("top_k", DEFAULT_TOP_K))
        if name != TOOL_NAME:
            return ToolExecution(str(query), top_k, f"Unknown tool: {name}", True, 0.0)
        if not isinstance(query, str) or not query.strip():
            return ToolExecution("", top_k, "Error: query is required", True, 0.0)

        server = self.server
        with _server_env(self.corpus_dir, self.embeddings_dir.parent):
            start = time.perf_counter()
            try:
                results = server.search_corpus(
                    query,
                    self.index_name,
                    top_k,
                    rerank_config=self.condition.rerank,
                    hybrid_config=self.condition.hybrid,
                )
                text = server.format_results_for_context(results, query)
            except Exception as exc:  # retrieval stack failure, not model error
                raise HarnessToolError(f"search_corpus failed: {exc}") from exc
            latency = time.perf_counter() - start

        records: List[Dict[str, Any]] = []
        for result in results:
            record = _jsonable(dict(result))
            source = record.get("source") or ""
            record["slug"] = _derive_slug(source)
            record["prefix"] = format_citation_prefix(
                source,
                record.get("page_start"),
                record.get("page_end"),
                record.get("section_heading"),
            )
            records.append(record)
        return ToolExecution(query, top_k, text, False, latency, records)
