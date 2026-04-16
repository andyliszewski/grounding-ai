"""Tests for the ``grounding eval`` CLI (Story 16.3).

These tests reuse the mini corpus + stub embedder from
``test_eval_runner_integration.py`` so they don't download a real
embedding model. We monkeypatch ``_default_embed`` in the runner so
the CLI handler — which doesn't expose an ``embed_fn`` injection —
still uses the deterministic stub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from grounding.eval import cli as eval_cli
from grounding.eval.cli import (
    EXIT_BAD_ARGS,
    EXIT_BASELINE_REGRESSION,
    EXIT_FIXTURE_OR_AGENT_NOT_FOUND,
    EXIT_OK,
)
from grounding.vector_store import write_vector_index

# Reuse the stub embedder + mini chunks from the integration test module.
from tests.test_eval_runner_integration import (
    AGENTS_DIR,
    MINI_CHUNKS,
    MINI_CORPUS,
    MINI_FIXTURE_YAML,
    stub_embed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mini_embeddings_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli_mini_embeddings")
    embeddings = {c["chunk_id"]: stub_embed(c["text"]) for c in MINI_CHUNKS}
    chunk_meta = {
        c["chunk_id"]: {
            "doc_id": c["doc_id"],
            "file_path": f"{c['doc_id']}/chunks/{c['chunk_id']}.md",
            "is_music": False,
        }
        for c in MINI_CHUNKS
    }
    write_vector_index(embeddings, out, chunk_metadata=chunk_meta)
    return out


@pytest.fixture(autouse=True)
def _stub_default_embed(monkeypatch):
    """Force the runner's default embedder to the deterministic stub."""
    monkeypatch.setattr("grounding.eval.runner._default_embed", stub_embed)


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        agent="mini",
        agents_dir=AGENTS_DIR,
        fixtures=MINI_FIXTURE_YAML,
        corpus=MINI_CORPUS,
        embeddings=None,
        top_k=5,
        out=None,
        baseline=None,
        fail_under=0.0,
        # Story 18.3 rerank flags — defaults match argparse defaults.
        config=None,
        rerank=False,
        rerank_model=None,
        rerank_pool_size=None,
        rerank_top_k=None,
        # Story 19.3 hybrid flags — defaults match argparse defaults.
        hybrid=False,
        hybrid_pool_size=None,
        hybrid_k_rrf=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# --help (smoke)
# ---------------------------------------------------------------------------

def test_cli_help_lists_eval_subcommand(capsys):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cli._create_eval_parser(sub)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["eval", "--help"])
    # argparse exits 0 for --help.
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--agent" in out
    assert "--fixtures" in out
    assert "--baseline" in out
    assert "--fail-under" in out


def test_eval_reachable_via_python_m_grounding_cli():
    """Guard against subcommand-registration drift in grounding/cli.py:main().

    Story 16.3 added the eval parser but left the ``subcommands`` set in
    ``grounding/cli.py:main()`` unmodified; direct ``eval_command(args)``
    tests passed while ``python -m grounding.cli eval`` fell through to
    the PDF-converter argparse. The false-green surfaced only during
    16.4's live demo PR. This subprocess smoke asserts the shell path
    actually reaches the eval parser.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "grounding.cli", "eval", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`python -m grounding.cli eval --help` exited {result.returncode}. "
        f"stderr: {result.stderr}"
    )
    # Eval parser specifics (not present in the PDF-converter argparse).
    assert "--agent" in result.stdout
    assert "--baseline" in result.stdout
    assert "--fail-under" in result.stdout
    # Negative assertion: must NOT be the PDF converter's help.
    assert "input_dir output_dir" not in result.stdout
    assert "--chunk-size" not in result.stdout


# ---------------------------------------------------------------------------
# End-to-end against the mini corpus
# ---------------------------------------------------------------------------

def test_cli_runs_against_mini_corpus_exit_zero(
    tmp_path: Path, mini_embeddings_dir: Path, capsys
):
    args = _make_args(embeddings=mini_embeddings_dir, out=tmp_path / "reports")
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "eval: mini" in out
    assert "items=3" in out


def test_cli_writes_md_and_json_artifacts(
    tmp_path: Path, mini_embeddings_dir: Path
):
    out_dir = tmp_path / "reports"
    args = _make_args(embeddings=mini_embeddings_dir, out=out_dir)
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    md_files = list(out_dir.glob("mini-*.md"))
    json_files = list(out_dir.glob("mini-*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1

    payload = json.loads(json_files[0].read_text())
    assert payload["format_version"] == 2
    assert payload["agent"] == "mini"
    assert payload["aggregate"]["n_items"] == 3
    assert payload["aggregate"]["recall_at_5"] >= 2.0 / 3.0


# ---------------------------------------------------------------------------
# Baseline scenarios
# ---------------------------------------------------------------------------

def _write_baseline(path: Path, *, tighter_by: float = 0.0) -> Path:
    """Write a baseline file. tighter_by raises baseline metrics by that delta
    (so the current run will appear to drop)."""
    base = {
        "format_version": 1,
        "agent": "mini",
        "captured_utc": "2026-04-13T15:00:00+00:00",
        "fixture_version": 1,
        "aggregate": {
            "recall_at_1": 1.0 + tighter_by,
            "recall_at_3": 1.0 + tighter_by,
            "recall_at_5": 1.0 + tighter_by,
            "recall_at_10": 1.0 + tighter_by,
            "mrr": 1.0 + tighter_by,
            "ndcg_at_10": 0.5 + tighter_by,
        },
    }
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


def test_cli_with_baseline_within_tolerance_exit_zero(
    tmp_path: Path, mini_embeddings_dir: Path
):
    baseline = _write_baseline(tmp_path / "baseline.json", tighter_by=0.0)
    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        baseline=baseline,
        fail_under=0.05,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK


def test_cli_with_baseline_drop_exceeds_fail_under_exit_one(
    tmp_path: Path, mini_embeddings_dir: Path
):
    # Inflate baseline so the current run shows a 0.5 drop on every metric.
    baseline = _write_baseline(tmp_path / "baseline.json")
    # Overwrite to set a drop scenario explicitly
    payload = {
        "format_version": 1,
        "agent": "mini",
        "captured_utc": "2026-04-13T15:00:00+00:00",
        "fixture_version": 1,
        "aggregate": {
            # Current achieves 1.0 across the board; baseline at 1.5 forces a 0.5 drop.
            "recall_at_1": 1.5, "recall_at_3": 1.5, "recall_at_5": 1.5,
            "recall_at_10": 1.5, "mrr": 1.5, "ndcg_at_10": 1.5,
        },
    }
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        baseline=baseline,
        fail_under=0.05,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_BASELINE_REGRESSION


# ---------------------------------------------------------------------------
# Validation / not-found cases
# ---------------------------------------------------------------------------

def test_cli_missing_fixture_exit_two(tmp_path: Path, mini_embeddings_dir: Path):
    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        fixtures=tmp_path / "does-not-exist.yaml",
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_FIXTURE_OR_AGENT_NOT_FOUND


def test_cli_warns_when_fail_under_without_baseline(
    tmp_path: Path, mini_embeddings_dir: Path, capsys
):
    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        fail_under=0.05,  # no baseline
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    err = capsys.readouterr().err
    assert "fail-under" in err.lower() or "baseline" in err.lower()


# ---------------------------------------------------------------------------
# Reranking flags (Story 18.3)
# ---------------------------------------------------------------------------


def test_eval_accepts_rerank_flag(tmp_path: Path, mini_embeddings_dir: Path, monkeypatch):
    """--rerank flag propagates through to a RerankConfig with enabled=True.

    We stub the reranker so no model loads; stub must be called to confirm
    the two-stage flow actually ran.
    """
    called = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        called["config"] = config
        called["n_chunks"] = len(chunks)
        return [dict(c, rerank_score=1.0, score=1.0) for c in chunks]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        rerank=True,
        rerank_pool_size=3,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    assert called.get("config") is not None
    assert called["config"].enabled is True
    assert called["config"].pool_size == 3

    # JSON report stamps the rerank provenance so baselines are unambiguous
    # about which retrieval mode produced the numbers (Story 18.4 gotcha fix).
    import json
    json_files = list((tmp_path / "reports").glob("*.json"))
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert "rerank" in payload, "eval JSON must carry rerank provenance"
    assert payload["rerank"]["enabled"] is True
    assert payload["rerank"]["pool_size"] == 3
    assert payload["rerank"]["model"]  # non-empty


def test_eval_rerank_model_flag_overrides_default(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    seen_models = []

    def fake_rerank(query, chunks, *, config, text_key="content"):
        seen_models.append(config.model)
        return [dict(c, rerank_score=1.0, score=1.0) for c in chunks]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    assert "ms-marco-MiniLM-L-6-v2" in seen_models[0]


def test_eval_invalid_rerank_pool_size_zero_exits_2(
    tmp_path: Path, mini_embeddings_dir: Path
):
    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        rerank=True,
        rerank_pool_size=0,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_BAD_ARGS


def test_eval_rerank_top_k_overrides_top_k(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """--rerank-top-k truncates the final output independently of --top-k.

    Here we set --top-k=5 but --rerank-top-k=2; eval run should expose
    at most 2 retrieved chunks per item.
    """

    def fake_rerank(query, chunks, *, config, text_key="content"):
        return [dict(c, rerank_score=float(i), score=float(i))
                for i, c in enumerate(reversed(chunks))]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        top_k=5,
        rerank=True,
        rerank_pool_size=10,
        rerank_top_k=2,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    json_path = next((tmp_path / "reports").glob("mini-*.json"))
    payload = json.loads(json_path.read_text())
    for item in payload["items"]:
        assert len(item["retrieved"]) <= 2


def test_eval_without_rerank_flag_rerank_config_disabled(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """No --rerank flag, no --config => reranker is never invoked."""

    def fake_rerank(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("reranker invoked when disabled")

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(embeddings=mini_embeddings_dir, out=tmp_path / "reports")
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK


def test_eval_config_yaml_rerank_enabled_is_picked_up(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """config.yaml retrieval.rerank.enabled: true => reranker runs w/o --rerank."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "retrieval:\n"
        "  rerank:\n"
        "    enabled: true\n"
        "    pool_size: 7\n",
        encoding="utf-8",
    )
    seen = []

    def fake_rerank(query, chunks, *, config, text_key="content"):
        seen.append(config)
        return [dict(c, rerank_score=1.0, score=1.0) for c in chunks]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        config=cfg,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    assert seen and seen[0].enabled is True
    assert seen[0].pool_size == 7


def test_eval_cli_flag_overrides_config_yaml(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """CLI --rerank-pool-size wins over the config.yaml value."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "retrieval:\n  rerank:\n    enabled: true\n    pool_size: 100\n",
        encoding="utf-8",
    )
    seen = []

    def fake_rerank(query, chunks, *, config, text_key="content"):
        seen.append(config)
        return [dict(c, rerank_score=1.0, score=1.0) for c in chunks]

    monkeypatch.setattr("grounding.reranker.rerank", fake_rerank)

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        config=cfg,
        rerank_pool_size=5,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK
    assert seen and seen[0].pool_size == 5


# ---------------------------------------------------------------------------
# Hybrid retrieval flags (Story 19.3)
# ---------------------------------------------------------------------------


def _stub_hybrid_fn(hits_by_query=None):
    """Produce a stub hybrid_fn that mirrors search_hybrid's dict output.

    We bypass the real FAISS+BM25 path by patching runner._default_hybrid_search.
    The stub returns one synthetic chunk per known chunk_id in the mini corpus.
    """
    from tests.test_eval_runner_integration import MINI_CHUNKS

    def stub(query, embeddings_dir, *, top_k, pool_size, k_rrf):
        return [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "faiss_rank": i + 1,
                "bm25_rank": i + 1,
                "rrf_score": 1.0 / (k_rrf + i + 1) + 1.0 / (k_rrf + i + 1),
                "faiss_distance": 0.1 * (i + 1),
                "rank": i + 1,
            }
            for i, c in enumerate(MINI_CHUNKS[:top_k])
        ]

    return stub


def test_eval_accepts_hybrid_flag(tmp_path: Path, mini_embeddings_dir: Path, monkeypatch):
    """--hybrid flag propagates through to HybridConfig with enabled=True."""
    monkeypatch.setattr(
        "grounding.eval.runner._default_hybrid_search", _stub_hybrid_fn()
    )

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        hybrid=True,
        hybrid_pool_size=7,
        hybrid_k_rrf=50,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    json_path = next((tmp_path / "reports").glob("mini-*.json"))
    payload = json.loads(json_path.read_text())
    assert "hybrid" in payload
    assert payload["hybrid"]["enabled"] is True
    assert payload["hybrid"]["pool_size"] == 7
    assert payload["hybrid"]["k_rrf"] == 50


def test_eval_invalid_hybrid_pool_size_zero_exits_2(
    tmp_path: Path, mini_embeddings_dir: Path
):
    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        hybrid=True,
        hybrid_pool_size=0,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_BAD_ARGS


def test_eval_without_hybrid_flag_hybrid_config_is_none(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """No --hybrid flag, no --config => runner receives hybrid_config=None.

    Checked via the JSON report: the 'hybrid' provenance block is absent.
    """
    args = _make_args(embeddings=mini_embeddings_dir, out=tmp_path / "reports")
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    json_path = next((tmp_path / "reports").glob("mini-*.json"))
    payload = json.loads(json_path.read_text())
    assert "hybrid" not in payload


def test_eval_hybrid_config_yaml_is_picked_up(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    """config.yaml retrieval.hybrid.enabled: true => hybrid runs w/o --hybrid."""
    monkeypatch.setattr(
        "grounding.eval.runner._default_hybrid_search", _stub_hybrid_fn()
    )

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "retrieval:\n  hybrid:\n    enabled: true\n    pool_size: 11\n",
        encoding="utf-8",
    )

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        config=cfg,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    json_path = next((tmp_path / "reports").glob("mini-*.json"))
    payload = json.loads(json_path.read_text())
    assert payload["hybrid"]["enabled"] is True
    assert payload["hybrid"]["pool_size"] == 11


def test_eval_cli_hybrid_flag_overrides_config_yaml(
    tmp_path: Path, mini_embeddings_dir: Path, monkeypatch
):
    monkeypatch.setattr(
        "grounding.eval.runner._default_hybrid_search", _stub_hybrid_fn()
    )

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "retrieval:\n  hybrid:\n    enabled: true\n    pool_size: 100\n",
        encoding="utf-8",
    )

    args = _make_args(
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
        config=cfg,
        hybrid_pool_size=5,
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_OK

    json_path = next((tmp_path / "reports").glob("mini-*.json"))
    payload = json.loads(json_path.read_text())
    assert payload["hybrid"]["pool_size"] == 5


def test_cli_agent_mismatch_with_fixture_exits_two(
    tmp_path: Path, mini_embeddings_dir: Path
):
    # Put a fake agent yaml in place so the agents-dir check passes and we hit
    # the explicit agent-mismatch branch.
    fake_agents = tmp_path / "agents"
    fake_agents.mkdir()
    (fake_agents / "mini.yaml").write_text("name: mini\n", encoding="utf-8")
    (fake_agents / "ceo.yaml").write_text("name: ceo\n", encoding="utf-8")

    args = _make_args(
        agent="ceo",  # fixture targets 'mini'
        agents_dir=fake_agents,
        embeddings=mini_embeddings_dir,
        out=tmp_path / "reports",
    )
    rc = eval_cli.eval_command(args)
    assert rc == EXIT_FIXTURE_OR_AGENT_NOT_FOUND
