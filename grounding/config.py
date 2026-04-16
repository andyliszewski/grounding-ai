"""Shared config-file helpers for the CLI surfaces (Story 18.3).

Two CLIs now consume ``config.yaml`` for retrieval settings: ``grounding
eval`` and ``scripts/local_rag.py``. This module exposes a tiny helper
that both can share so they don't diverge on key handling.

Resolution order (per Story 18.3 AC4) is enforced at the call site:

    1. Explicit CLI flag value.
    2. ``config.yaml`` -> ``retrieval.rerank.*`` when present.
    3. ``RerankConfig`` default.

Scope is intentionally tiny: YAML load + safe dict access. No Pydantic,
no schema validation beyond what ``RerankConfig.validate()`` already does.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("grounding.config")


def load_retrieval_config(path: Path | None) -> dict:
    """Return the ``retrieval`` block from ``config.yaml`` as a plain dict.

    Missing file, unreadable file, empty file, or file without a
    ``retrieval`` key all return ``{}``. Malformed YAML is logged at
    WARNING and also returns ``{}``. Callers treat the empty dict as
    "no config" and fall through to ``RerankConfig`` defaults.
    """
    if path is None:
        return {}

    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Could not parse %s as YAML: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        return {}

    retrieval = data.get("retrieval")
    if not isinstance(retrieval, dict):
        return {}
    return retrieval


def resolve_rerank_config(
    *,
    retrieval_config: dict,
    cli_enabled: bool,
    cli_model: str | None,
    cli_pool_size: int | None,
    cli_batch_size: int | None = None,
):
    """Merge CLI overrides on top of ``retrieval.rerank`` config values.

    Returns a validated :class:`grounding.reranker.RerankConfig`. The
    CLI ``--rerank`` flag is a store_true, so ``cli_enabled=True`` always
    wins; when ``cli_enabled`` is False, the config-file ``enabled`` flag
    applies (so users can enable rerank purely from ``config.yaml``).
    Other CLI args override only when the caller passed a non-None value.
    """
    from grounding.reranker import RerankConfig

    block = retrieval_config.get("rerank") if isinstance(retrieval_config, dict) else None
    if not isinstance(block, dict):
        block = {}

    enabled = bool(cli_enabled) or bool(block.get("enabled", False))
    model = cli_model if cli_model is not None else block.get("model", RerankConfig.model)
    pool_size = (
        cli_pool_size
        if cli_pool_size is not None
        else block.get("pool_size", RerankConfig.pool_size)
    )
    batch_size = (
        cli_batch_size
        if cli_batch_size is not None
        else block.get("batch_size", RerankConfig.batch_size)
    )

    cfg = RerankConfig(
        enabled=enabled,
        model=str(model),
        pool_size=int(pool_size),
        batch_size=int(batch_size),
    )
    cfg.validate()
    return cfg


def resolve_hybrid_config(
    *,
    retrieval_config: dict,
    cli_enabled: bool,
    cli_pool_size: int | None,
    cli_k_rrf: int | None,
):
    """Merge CLI overrides on top of ``retrieval.hybrid`` config values.

    Modeled on :func:`resolve_rerank_config`: ``--hybrid`` is a store_true
    so ``cli_enabled=True`` always wins, config-file ``enabled`` applies
    when the CLI flag is absent, and the numeric CLI overrides only apply
    when non-None. Returns a validated :class:`grounding.hybrid.HybridConfig`
    and raises ``ValueError`` on invalid values.

    Duplication with ``resolve_rerank_config`` is intentional for now; 18.3
    flagged the collapse as a low-priority cleanup and 19.3 inherits the
    scope fence rather than paying it off here.
    """
    from grounding.hybrid import HybridConfig

    block = retrieval_config.get("hybrid") if isinstance(retrieval_config, dict) else None
    if not isinstance(block, dict):
        block = {}

    enabled = bool(cli_enabled) or bool(block.get("enabled", False))
    pool_size = (
        cli_pool_size
        if cli_pool_size is not None
        else block.get("pool_size", HybridConfig.pool_size)
    )
    k_rrf = (
        cli_k_rrf
        if cli_k_rrf is not None
        else block.get("k_rrf", HybridConfig.k_rrf)
    )

    cfg = HybridConfig(
        enabled=enabled,
        pool_size=int(pool_size),
        k_rrf=int(k_rrf),
    )
    cfg.validate()
    return cfg
