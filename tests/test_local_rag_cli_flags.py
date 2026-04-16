"""CLI flag-parsing tests for ``scripts/local_rag.py`` (Story 18.3).

Exercises the ``_resolve_rerank_config_from_args`` helper and the argparse
block directly. Does NOT spin up the full LLM path — model loading, agentic
loop, and REPL are out of scope for flag-parsing tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import local_rag  # noqa: E402
from grounding.hybrid import HybridConfig  # noqa: E402
from grounding.reranker import RerankConfig  # noqa: E402


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        config=None,
        rerank=False,
        rerank_model=None,
        rerank_pool_size=None,
        rerank_top_k=None,
        hybrid=False,
        hybrid_pool_size=None,
        hybrid_k_rrf=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_local_rag_accepts_rerank_flag():
    cfg = local_rag._resolve_rerank_config_from_args(_ns(rerank=True))
    assert isinstance(cfg, RerankConfig)
    assert cfg.enabled is True
    assert cfg.pool_size == 50
    assert cfg.model == "BAAI/bge-reranker-base"


def test_local_rag_no_flags_no_config_returns_none():
    """Pre-18.3 backward compat: no flags & no config => legacy path."""
    assert local_rag._resolve_rerank_config_from_args(_ns()) is None


def test_local_rag_invalid_pool_size_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        local_rag._resolve_rerank_config_from_args(
            _ns(rerank=True, rerank_pool_size=0)
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "rerank" in err.lower() or "pool_size" in err.lower()


def test_local_rag_rerank_model_override():
    cfg = local_rag._resolve_rerank_config_from_args(
        _ns(rerank=True, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    assert cfg.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_local_rag_cli_overrides_config_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "retrieval:\n  rerank:\n    enabled: true\n    pool_size: 100\n",
        encoding="utf-8",
    )
    cfg = local_rag._resolve_rerank_config_from_args(
        _ns(config=cfg_file, rerank_pool_size=3)
    )
    assert cfg.pool_size == 3
    assert cfg.enabled is True  # config-file value carried through


def test_local_rag_config_yaml_alone_enables_rerank(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "retrieval:\n  rerank:\n    enabled: true\n", encoding="utf-8"
    )
    cfg = local_rag._resolve_rerank_config_from_args(_ns(config=cfg_file))
    assert cfg is not None
    assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Hybrid retrieval flags (Story 19.3)
# ---------------------------------------------------------------------------


def test_local_rag_accepts_hybrid_flag():
    cfg = local_rag._resolve_hybrid_config_from_args(_ns(hybrid=True))
    assert isinstance(cfg, HybridConfig)
    assert cfg.enabled is True
    assert cfg.pool_size == 50
    assert cfg.k_rrf == 60


def test_local_rag_no_hybrid_flags_no_config_returns_none():
    """Pre-19.3 backward compat: no flags & no config => legacy path."""
    assert local_rag._resolve_hybrid_config_from_args(_ns()) is None


def test_local_rag_invalid_hybrid_pool_size_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        local_rag._resolve_hybrid_config_from_args(
            _ns(hybrid=True, hybrid_pool_size=0)
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "hybrid" in err.lower() or "pool_size" in err.lower()


def test_local_rag_argparse_accepts_three_hybrid_flags():
    """The real local_rag parser accepts the three hybrid flags."""
    parser = local_rag._build_parser()
    ns = parser.parse_args([
        "--agent", "mini",
        "--hybrid",
        "--hybrid-pool-size", "25",
        "--hybrid-k-rrf", "42",
    ])
    assert ns.hybrid is True
    assert ns.hybrid_pool_size == 25
    assert ns.hybrid_k_rrf == 42


def test_local_rag_argparse_accepts_four_rerank_flags():
    """The *real* local_rag parser accepts the four rerank flags.

    Uses ``local_rag._build_parser()`` so that removing a flag from the
    shipping CLI would fail this test — the earlier local-mirror version
    was a false-green waiting to happen.
    """
    parser = local_rag._build_parser()
    ns = parser.parse_args([
        "--agent", "mini",
        "--rerank",
        "--rerank-model", "custom/model",
        "--rerank-pool-size", "25",
        "--rerank-top-k", "3",
    ])
    assert ns.rerank is True
    assert ns.rerank_model == "custom/model"
    assert ns.rerank_pool_size == 25
    assert ns.rerank_top_k == 3
    # --config flag is part of the same story; cover it while we're here.
    assert hasattr(ns, "config")
