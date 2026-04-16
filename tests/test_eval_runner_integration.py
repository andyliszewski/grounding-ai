"""End-to-end runner test against a tiny in-repo corpus (Story 16.2, AC9/10).

This test exercises the real FAISS + chunk_map + manifest path in
``grounding.vector_store`` and ``grounding.manifest``. To keep the run
fast and deterministic (no model download), we use a **stub embedder**
that produces a 384-d vector whose only non-zero component is selected
by a cheap hash of bag-of-words. The same function is used to embed
both the indexed chunks and the queries, so semantically related
text (same dominant token) collides in vector space.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from grounding.eval import load_fixtures, run_eval
from grounding.vector_store import write_vector_index

FIXTURES_ROOT = Path(__file__).resolve().parent / "eval_fixtures"
MINI_CORPUS = FIXTURES_ROOT / "mini_corpus"
AGENTS_DIR = FIXTURES_ROOT / "agents"
MINI_FIXTURE_YAML = FIXTURES_ROOT / "mini_fixtures.yaml"

EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Stub embedder: deterministic, no model download.
# ---------------------------------------------------------------------------

_TOKEN_WEIGHTS = {
    # strong tokens tied to each doc
    "quantum": ("alpha", 1.0),
    "mechanics": ("alpha", 0.8),
    "transformer": ("alpha", 0.6),
    "bootstrap": ("beta", 1.0),
    "confidence": ("beta", 0.8),
    "interval": ("beta", 0.6),
    "falsifiability": ("gamma", 1.0),
    "popper": ("gamma", 0.8),
    "philosophical": ("gamma", 0.6),
    # topic labels present in queries
    "alpha": ("alpha", 0.3),
    "beta": ("beta", 0.3),
    "gamma": ("gamma", 0.3),
}

_TOPICS = {"alpha": 0, "beta": 1, "gamma": 2}


def stub_embed(text: str) -> np.ndarray:
    """Deterministic 384-d embedding keyed by topic dominance.

    Positions 0-2 carry topic weights (alpha/beta/gamma). The remaining
    positions carry a stable per-text noise signature so unrelated strings
    are not perfectly colinear, which lets FAISS return a clean ordering.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = text.lower().split()
    for token in tokens:
        if token in _TOKEN_WEIGHTS:
            topic, weight = _TOKEN_WEIGHTS[token]
            vec[_TOPICS[topic]] += weight

    # Per-text noise signature, low amplitude.
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    noise = np.frombuffer(digest * 12, dtype=np.uint8)[:EMBEDDING_DIM - 3]
    vec[3:] = (noise.astype(np.float32) / 255.0) * 0.05

    # L2 normalize so distances are meaningful.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Chunks that populate the mini FAISS index.
# ---------------------------------------------------------------------------

MINI_CHUNKS = [
    {
        "chunk_id": "doc-alpha-0001",
        "doc_id": "doc-alpha",
        "text": "quantum mechanics transformer attention heads alpha",
    },
    {
        "chunk_id": "doc-alpha-0002",
        "doc_id": "doc-alpha",
        "text": "quantum field theory renormalization alpha",
    },
    {
        "chunk_id": "doc-beta-0001",
        "doc_id": "doc-beta",
        "text": "bootstrap confidence interval small sample beta",
    },
    {
        "chunk_id": "doc-beta-0002",
        "doc_id": "doc-beta",
        "text": "confidence interval percentile bootstrap beta",
    },
    {
        "chunk_id": "doc-gamma-0001",
        "doc_id": "doc-gamma",
        "text": "falsifiability popper philosophical demarcation gamma",
    },
]


@pytest.fixture(scope="session")
def mini_embeddings_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real FAISS index + chunk_map for the mini corpus, once per session."""
    out = tmp_path_factory.mktemp("mini_embeddings")
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


def test_runner_against_mini_corpus_recall_at_5_above_threshold(
    mini_embeddings_dir: Path,
) -> None:
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)

    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=5,
        embed_fn=stub_embed,
    )

    assert len(result.items) == 3
    assert result.skipped == ()
    # AC: recall@5 >= 2/3
    assert result.aggregate.recall_at_5 >= 2.0 / 3.0
    # All 3 queries should find their doc at rank 1 with this stub.
    ranks = [i.first_hit_rank for i in result.items]
    assert all(r == 1 for r in ranks), f"expected all rank-1 hits, got {ranks}"


def test_runner_mini_corpus_populates_per_tag_metrics(
    mini_embeddings_dir: Path,
) -> None:
    fixture_set = load_fixtures(MINI_FIXTURE_YAML, agents_dir=AGENTS_DIR)

    result = run_eval(
        fixture_set,
        "mini",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=mini_embeddings_dir,
        top_k=5,
        embed_fn=stub_embed,
    )

    per_tag = result.aggregate.per_tag
    assert set(per_tag.keys()) == {"physics", "statistics", "philosophy"}
    for metrics in per_tag.values():
        assert metrics.n_items == 1
        assert metrics.low_sample is True
