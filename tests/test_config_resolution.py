"""Tests for ``grounding.config`` helpers (Story 18.3 Task 4).

Covers:
- ``load_retrieval_config`` robustness to missing / malformed YAML.
- ``resolve_rerank_config`` precedence: CLI > config.yaml > RerankConfig defaults.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from grounding.config import (
    load_retrieval_config,
    resolve_hybrid_config,
    resolve_rerank_config,
)
from grounding.hybrid import HybridConfig
from grounding.reranker import RerankConfig


# ---------------------------------------------------------------------------
# load_retrieval_config
# ---------------------------------------------------------------------------


def test_load_retrieval_config_missing_file_returns_empty_dict(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    assert load_retrieval_config(missing) == {}


def test_load_retrieval_config_none_returns_empty_dict():
    assert load_retrieval_config(None) == {}


def test_load_retrieval_config_parses_rerank_block(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "retrieval:\n"
        "  rerank:\n"
        "    enabled: true\n"
        "    model: cross-encoder/ms-marco-MiniLM-L-6-v2\n"
        "    pool_size: 80\n"
        "    batch_size: 32\n",
        encoding="utf-8",
    )
    result = load_retrieval_config(cfg)
    assert result["rerank"]["enabled"] is True
    assert result["rerank"]["pool_size"] == 80
    assert result["rerank"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_load_retrieval_config_ignores_unrelated_sections(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "paths:\n  corpus: ./corpus\nwatcher:\n  git_pull_enabled: false\n",
        encoding="utf-8",
    )
    assert load_retrieval_config(cfg) == {}


def test_load_retrieval_config_malformed_yaml_returns_empty(tmp_path: Path):
    cfg = tmp_path / "broken.yaml"
    cfg.write_text(": : :\n  not: valid: yaml\n", encoding="utf-8")
    assert load_retrieval_config(cfg) == {}


# ---------------------------------------------------------------------------
# resolve_rerank_config
# ---------------------------------------------------------------------------


def test_resolve_rerank_config_empty_returns_defaults():
    cfg = resolve_rerank_config(
        retrieval_config={},
        cli_enabled=False,
        cli_model=None,
        cli_pool_size=None,
    )
    assert cfg == RerankConfig()
    assert cfg.enabled is False


def test_resolve_rerank_config_cli_enabled_overrides_config_disabled():
    cfg = resolve_rerank_config(
        retrieval_config={"rerank": {"enabled": False, "pool_size": 100}},
        cli_enabled=True,
        cli_model=None,
        cli_pool_size=None,
    )
    assert cfg.enabled is True
    assert cfg.pool_size == 100  # from yaml since no CLI override


def test_resolve_rerank_config_cli_pool_size_overrides_yaml():
    cfg = resolve_rerank_config(
        retrieval_config={"rerank": {"pool_size": 100, "enabled": True}},
        cli_enabled=False,
        cli_model=None,
        cli_pool_size=25,
    )
    assert cfg.pool_size == 25
    assert cfg.enabled is True


def test_resolve_rerank_config_yaml_enabled_without_cli_flag():
    cfg = resolve_rerank_config(
        retrieval_config={"rerank": {"enabled": True}},
        cli_enabled=False,
        cli_model=None,
        cli_pool_size=None,
    )
    assert cfg.enabled is True


def test_resolve_rerank_config_invalid_pool_size_raises():
    with pytest.raises(ValueError):
        resolve_rerank_config(
            retrieval_config={},
            cli_enabled=True,
            cli_model=None,
            cli_pool_size=0,
        )


def test_resolve_rerank_config_missing_rerank_subblock_uses_defaults():
    cfg = resolve_rerank_config(
        retrieval_config={"something_else": 1},
        cli_enabled=False,
        cli_model=None,
        cli_pool_size=None,
    )
    assert cfg == RerankConfig()


# ---------------------------------------------------------------------------
# resolve_hybrid_config (Story 19.3)
# ---------------------------------------------------------------------------


def test_resolve_hybrid_config_defaults_when_all_none():
    cfg = resolve_hybrid_config(
        retrieval_config={},
        cli_enabled=False,
        cli_pool_size=None,
        cli_k_rrf=None,
    )
    assert cfg == HybridConfig()
    assert cfg.enabled is False
    assert cfg.pool_size == 50
    assert cfg.k_rrf == 60


def test_resolve_hybrid_config_cli_enabled_wins_over_disabled_yaml():
    cfg = resolve_hybrid_config(
        retrieval_config={"hybrid": {"enabled": False, "pool_size": 75}},
        cli_enabled=True,
        cli_pool_size=None,
        cli_k_rrf=None,
    )
    assert cfg.enabled is True
    assert cfg.pool_size == 75  # yaml value carried through


def test_resolve_hybrid_config_yaml_enabled_when_cli_flag_absent():
    cfg = resolve_hybrid_config(
        retrieval_config={"hybrid": {"enabled": True, "k_rrf": 40}},
        cli_enabled=False,
        cli_pool_size=None,
        cli_k_rrf=None,
    )
    assert cfg.enabled is True
    assert cfg.k_rrf == 40


def test_resolve_hybrid_config_cli_pool_size_overrides_yaml():
    cfg = resolve_hybrid_config(
        retrieval_config={"hybrid": {"enabled": True, "pool_size": 100}},
        cli_enabled=False,
        cli_pool_size=25,
        cli_k_rrf=None,
    )
    assert cfg.pool_size == 25
    assert cfg.enabled is True


def test_resolve_hybrid_config_cli_k_rrf_overrides_yaml():
    cfg = resolve_hybrid_config(
        retrieval_config={"hybrid": {"enabled": True, "k_rrf": 100}},
        cli_enabled=False,
        cli_pool_size=None,
        cli_k_rrf=30,
    )
    assert cfg.k_rrf == 30


def test_resolve_hybrid_config_invalid_k_rrf_zero_raises():
    with pytest.raises(ValueError):
        resolve_hybrid_config(
            retrieval_config={},
            cli_enabled=True,
            cli_pool_size=None,
            cli_k_rrf=0,
        )


def test_resolve_hybrid_config_missing_subblock_uses_defaults():
    cfg = resolve_hybrid_config(
        retrieval_config={"rerank": {"enabled": True}},
        cli_enabled=False,
        cli_pool_size=None,
        cli_k_rrf=None,
    )
    assert cfg == HybridConfig()
