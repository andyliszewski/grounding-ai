"""Cross-encoder reranker for two-stage retrieval (Epic 18 Story 18.1).

Two-stage flow:
    1. FAISS bi-encoder retrieves a pool of candidate chunks (fast, approximate).
    2. Cross-encoder reranks that pool jointly encoding (query, chunk) pairs
       (slower, more accurate) and produces the final ordering.

This module is pure and generic: it operates on plain dicts with a configurable
`text_key`, so retrieval code (dict-shaped results) and eval code (dataclass
results, converted at the call site) can share it.

Latency expectations (CPU-only, BAAI/bge-reranker-base):
    ~50–150 ms per pool of 50 pairs on modern laptops; budgeted per-query
    overhead for Epic 18 is sub-second on the CPU path.

Downstream integration (SearchCorpusTool, MCP server, local_rag.py, eval harness)
lands in Stories 18.2+; this module has no knowledge of those surfaces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger("grounding.reranker")

_model_cache: dict[str, Any] = {}
_reranker_singletons: dict[str, "CrossEncoderReranker"] = {}


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for cross-encoder reranking.

    Keyed on `model` for caching: the loaded model is the only expensive
    resource; `pool_size`, `batch_size`, `enabled` are per-call choices.
    """

    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    pool_size: int = 50
    batch_size: int = 16

    def validate(self) -> None:
        if not self.model:
            raise ValueError("RerankConfig.model must be a non-empty string")
        if self.pool_size < 1:
            raise ValueError(
                f"RerankConfig.pool_size must be >= 1, got {self.pool_size}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"RerankConfig.batch_size must be >= 1, got {self.batch_size}"
            )


def _get_cross_encoder(model: str) -> Any:
    """Load and cache a CrossEncoder by model name (singleton).

    Forces `device="cpu"` for determinism and to match Epic 18's latency budget.
    Retries up to 3 times on failure, then raises RuntimeError.
    """
    cached = _model_cache.get(model)
    if cached is not None:
        logger.debug("Using cached cross-encoder '%s'", model)
        return cached

    from sentence_transformers import CrossEncoder

    logger.info(
        "Loading cross-encoder '%s' (first run may download model weights)", model
    )

    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            encoder = CrossEncoder(model, device="cpu")
            _model_cache[model] = encoder
            logger.info("Successfully loaded cross-encoder '%s'", model)
            return encoder
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed to load cross-encoder (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
                exc_info=True,
            )
            if attempt == max_retries:
                break

    error_msg = (
        f"Failed to load cross-encoder '{model}' after {max_retries} attempts: "
        f"{last_error}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error


class CrossEncoderReranker:
    """Rerank a list of candidate chunks with a cross-encoder.

    The model is loaded lazily on first `.rerank(...)` call and reused via a
    module-level cache keyed on `config.model`.
    """

    def __init__(self, config: RerankConfig) -> None:
        config.validate()
        self.config = config
        self._encoder: Any | None = None

    def rerank(
        self,
        query: str,
        chunks: Sequence[Mapping[str, Any]],
        *,
        text_key: str = "content",
    ) -> list[dict]:
        """Return a new list of dicts re-ordered by cross-encoder score.

        For each input dict, the returned dict:
          - copies every original key,
          - adds `rerank_score: float` (raw cross-encoder score),
          - adds `faiss_distance` set to the original `score` (or None if absent),
          - overwrites `score` with `rerank_score`.

        Input dicts are not mutated. Empty `chunks` short-circuits without
        loading the model. Missing `text_key` raises KeyError before any
        model call.
        """
        if not chunks:
            return []

        for idx, chunk in enumerate(chunks):
            if text_key not in chunk:
                raise KeyError(
                    f"chunk at index {idx} is missing required text_key '{text_key}'"
                )

        if self._encoder is None:
            self._encoder = _get_cross_encoder(self.config.model)
        encoder = self._encoder

        pairs = [(query, chunk[text_key]) for chunk in chunks]
        raw_scores = encoder.predict(pairs, batch_size=self.config.batch_size)

        scored: list[dict] = []
        for chunk, raw in zip(chunks, raw_scores):
            new = dict(chunk)
            rerank_score = float(raw)
            new["rerank_score"] = rerank_score
            new["faiss_distance"] = chunk.get("score")
            new["score"] = rerank_score
            scored.append(new)

        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored


def reassign_ranks(results: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Return a new list of dicts with `rank` set to 1..N in order.

    Input dicts are copied; caller's dicts are not mutated. Used by the
    retrieval surfaces (Story 18.2) to renumber after reranking so
    downstream formatters see a 1-indexed sequence that matches the
    reranked order.
    """
    return [{**dict(r), "rank": i + 1} for i, r in enumerate(results)]


def rerank(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    config: RerankConfig,
    text_key: str = "content",
) -> list[dict]:
    """Module-level convenience: rerank using a per-model singleton.

    The singleton is keyed on `config.model`, matching the model cache in
    `_get_cross_encoder`. Distinct model names yield distinct reranker
    instances; a different `pool_size` / `batch_size` on the same model
    reuses the singleton but takes effect on the next call via the new
    config's `batch_size` (applied by the latest-config semantics of the
    singleton registry).
    """
    config.validate()
    existing = _reranker_singletons.get(config.model)
    if existing is None or existing.config != config:
        existing = CrossEncoderReranker(config)
        _reranker_singletons[config.model] = existing
    return existing.rerank(query, chunks, text_key=text_key)
