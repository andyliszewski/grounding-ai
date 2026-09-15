"""Benchmark conditions for the grounded-answer benchmark (Epic 25).

Four conditions share the same answer model and the same base system prompt.
They differ only in whether the ``search_corpus`` tool is offered and, when it
is, which retrieval pipeline serves it:

==============  ======  =================================================
condition       tool    retrieval
==============  ======  =================================================
ungrounded      no      none
dense           yes     FAISS dense (today's default)
hybrid          yes     dense plus BM25, reciprocal rank fusion
hybrid-rerank   yes     hybrid plus the bge-reranker-base cross-encoder
==============  ======  =================================================

The retrieval settings are pinned by the harness, never chosen by the model.
The values mirror the MCP tool's defaults in
``mcp_servers/corpus_search/server.py`` (pool 50, k_rrf 60, reranker
``BAAI/bge-reranker-base`` with pool 50).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from grounding.hybrid import HybridConfig
from grounding.reranker import RerankConfig

CONDITION_ORDER: Tuple[str, ...] = ("ungrounded", "dense", "hybrid", "hybrid-rerank")

_HYBRID = HybridConfig(enabled=True, pool_size=50, k_rrf=60)
_RERANK = RerankConfig(enabled=True, model="BAAI/bge-reranker-base", pool_size=50)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    grounded: bool
    hybrid: HybridConfig | None = None
    rerank: RerankConfig | None = None

    def retrieval_config(self) -> Dict[str, Any] | None:
        """JSON-ready snapshot of the pinned retrieval settings (D8)."""
        if not self.grounded:
            return None
        return {
            "dense": True,
            "hybrid": (
                {"pool_size": self.hybrid.pool_size, "k_rrf": self.hybrid.k_rrf}
                if self.hybrid
                else None
            ),
            "rerank": (
                {
                    "model": self.rerank.model,
                    "pool_size": self.rerank.pool_size,
                    "batch_size": self.rerank.batch_size,
                }
                if self.rerank
                else None
            ),
        }


CONDITIONS: Dict[str, ConditionSpec] = {
    "ungrounded": ConditionSpec("ungrounded", grounded=False),
    "dense": ConditionSpec("dense", grounded=True),
    "hybrid": ConditionSpec("hybrid", grounded=True, hybrid=_HYBRID),
    "hybrid-rerank": ConditionSpec(
        "hybrid-rerank", grounded=True, hybrid=_HYBRID, rerank=_RERANK
    ),
}


def parse_conditions(raw: str) -> Tuple[ConditionSpec, ...]:
    """Parse a comma-separated condition list, keeping the canonical order.

    Raises:
        ValueError: unknown or empty condition names.
    """
    names = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not names:
        raise ValueError("no conditions given")
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}; choose from {list(CONDITION_ORDER)}"
        )
    wanted = set(names)
    return tuple(CONDITIONS[n] for n in CONDITION_ORDER if n in wanted)
