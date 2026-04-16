"""Tests for grounding.reranker (Epic 18 Story 18.1).

All tests stub `_get_cross_encoder` to avoid network access and model downloads.
"""
from __future__ import annotations

import pytest

from grounding import reranker as reranker_mod
from grounding.reranker import (
    CrossEncoderReranker,
    RerankConfig,
    reassign_ranks,
    rerank,
)


class _StubCrossEncoder:
    """Deterministic stub: score = len(doc_text). Records call counts."""

    def __init__(self) -> None:
        self.predict_calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs, batch_size):
        self.predict_calls.append((list(pairs), batch_size))
        return [float(len(doc)) for _query, doc in pairs]


@pytest.fixture(autouse=True)
def _clear_caches():
    reranker_mod._model_cache.clear()
    reranker_mod._reranker_singletons.clear()
    yield
    reranker_mod._model_cache.clear()
    reranker_mod._reranker_singletons.clear()


@pytest.fixture
def stub_factory(monkeypatch):
    """Patch _get_cross_encoder with a counting stub factory."""
    calls: dict[str, int] = {}
    instances: dict[str, _StubCrossEncoder] = {}

    def factory(model: str):
        calls[model] = calls.get(model, 0) + 1
        if model not in instances:
            instances[model] = _StubCrossEncoder()
        return instances[model]

    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", factory)
    return calls, instances


# ---------------------------------------------------------------------------
# RerankConfig
# ---------------------------------------------------------------------------


def test_config_defaults_match_spec():
    cfg = RerankConfig()
    assert cfg.enabled is False
    assert cfg.model == "BAAI/bge-reranker-base"
    assert cfg.pool_size == 50
    assert cfg.batch_size == 16
    cfg.validate()


def test_config_rejects_pool_size_zero():
    with pytest.raises(ValueError, match="pool_size"):
        RerankConfig(pool_size=0).validate()


def test_config_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        RerankConfig(model="").validate()


def test_config_rejects_negative_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        RerankConfig(batch_size=-1).validate()


# ---------------------------------------------------------------------------
# Reranker behavior
# ---------------------------------------------------------------------------


def test_rerank_empty_list_returns_empty_without_loading_model(monkeypatch):
    called = {"n": 0}

    def boom(_model):
        called["n"] += 1
        raise AssertionError("should not be called on empty input")

    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", boom)

    r = CrossEncoderReranker(RerankConfig())
    assert r.rerank("q", []) == []
    assert called["n"] == 0


def test_rerank_single_chunk_returns_single_chunk(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    out = r.rerank("q", [{"content": "hello", "score": 0.1}])
    assert len(out) == 1
    assert out[0]["content"] == "hello"
    assert out[0]["rerank_score"] == 5.0


def test_rerank_reorders_by_stub_scores(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    chunks = [
        {"content": "aa", "score": 0.9},       # stub score 2
        {"content": "aaaaa", "score": 0.1},    # stub score 5
        {"content": "aaa", "score": 0.5},      # stub score 3
    ]
    out = r.rerank("q", chunks)
    assert [c["content"] for c in out] == ["aaaaa", "aaa", "aa"]


def test_rerank_preserves_all_input_keys(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    chunks = [{"content": "abc", "score": 0.5, "doc_id": "d1", "rank": 1, "extra": {"k": 1}}]
    out = r.rerank("q", chunks)
    assert out[0]["doc_id"] == "d1"
    assert out[0]["rank"] == 1
    assert out[0]["extra"] == {"k": 1}


def test_rerank_promotes_rerank_score_into_score_field(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    out = r.rerank("q", [{"content": "abcd", "score": 0.9}])
    assert out[0]["score"] == out[0]["rerank_score"] == 4.0


def test_rerank_preserves_original_score_as_faiss_distance(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    out = r.rerank("q", [{"content": "abcd", "score": 0.9}])
    assert out[0]["faiss_distance"] == 0.9


def test_rerank_null_original_score_gives_null_faiss_distance(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    out = r.rerank("q", [{"content": "abcd"}])  # no "score" key
    assert out[0]["faiss_distance"] is None
    assert out[0]["score"] == out[0]["rerank_score"] == 4.0


def test_rerank_custom_text_key(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    chunks = [
        {"body": "short", "score": 0.5},
        {"body": "longer body text", "score": 0.5},
    ]
    out = r.rerank("q", chunks, text_key="body")
    assert out[0]["body"] == "longer body text"
    assert out[1]["body"] == "short"


def test_rerank_missing_text_key_raises_keyerror_before_model_call(monkeypatch):
    called = {"n": 0}

    def boom(_model):
        called["n"] += 1
        raise AssertionError("should not load model when chunks are malformed")

    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", boom)

    r = CrossEncoderReranker(RerankConfig())
    with pytest.raises(KeyError, match="content"):
        r.rerank("q", [{"score": 0.1}])
    assert called["n"] == 0


def test_rerank_deterministic_across_repeated_calls(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    chunks = [
        {"content": "aa", "score": 0.9},
        {"content": "aaaaa", "score": 0.1},
        {"content": "aaa", "score": 0.5},
    ]
    out1 = r.rerank("q", chunks)
    out2 = r.rerank("q", chunks)
    assert out1 == out2


def test_rerank_does_not_mutate_input_dicts(stub_factory):
    r = CrossEncoderReranker(RerankConfig())
    original = {"content": "abcd", "score": 0.9}
    snapshot = dict(original)
    r.rerank("q", [original])
    assert original == snapshot


# ---------------------------------------------------------------------------
# Singleton + model factory
# ---------------------------------------------------------------------------


def test_model_factory_called_once_per_model_name(stub_factory):
    calls, _ = stub_factory
    r = CrossEncoderReranker(RerankConfig())
    r.rerank("q", [{"content": "abc", "score": 0.1}])
    r.rerank("q", [{"content": "def", "score": 0.1}])
    r.rerank("q", [{"content": "ghi", "score": 0.1}])
    assert calls["BAAI/bge-reranker-base"] == 1


# ---------------------------------------------------------------------------
# reassign_ranks (Story 18.2)
# ---------------------------------------------------------------------------


def test_reassign_ranks_sets_one_indexed_sequence():
    out = reassign_ranks([
        {"source": "a", "rank": 999},
        {"source": "b", "rank": 7},
        {"source": "c"},
    ])
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert [r["source"] for r in out] == ["a", "b", "c"]


def test_reassign_ranks_does_not_mutate_input():
    original = [{"source": "a", "rank": 42}, {"source": "b", "rank": 43}]
    snapshot = [dict(r) for r in original]
    reassign_ranks(original)
    assert original == snapshot


def test_reassign_ranks_empty_list_returns_empty():
    assert reassign_ranks([]) == []


def test_module_level_rerank_convenience_uses_singleton(stub_factory):
    calls, _ = stub_factory
    cfg = RerankConfig()
    out1 = rerank("q", [{"content": "aa", "score": 0.1}], config=cfg)
    out2 = rerank("q", [{"content": "bbbb", "score": 0.1}], config=cfg)
    assert out1[0]["rerank_score"] == 2.0
    assert out2[0]["rerank_score"] == 4.0
    assert calls["BAAI/bge-reranker-base"] == 1
    assert len(reranker_mod._reranker_singletons) == 1
