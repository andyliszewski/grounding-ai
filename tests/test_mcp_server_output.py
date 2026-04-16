"""MCP server output formatting tests (Story 17.3).

These tests call ``format_results_for_context`` directly — no MCP
runtime, no network, no model loads.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def mcp_module():
    pytest.importorskip("mcp")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from mcp_servers.corpus_search import server  # noqa: WPS433
    return server


def _result(
    *,
    rank: int = 1,
    source: str = "alpha-paper.pdf",
    page_start=None,
    page_end=None,
    section=None,
    doc_id: str = "alpha123",
    chunk_id: str = "alpha123-0001",
    content: str = "body text",
    score: float = 0.95,
) -> dict:
    return {
        "rank": rank,
        "score": score,
        "source": source,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "page_start": page_start,
        "page_end": page_end,
        "section_heading": section,
        "content": content,
    }


def test_format_results_for_context_includes_citation_prefix(mcp_module):
    results = [
        _result(
            page_start=247, page_end=249,
            section="3.2 Bootstrap Methods",
        )
    ]
    out = mcp_module.format_results_for_context(results, "bootstrap")
    assert "[alpha-paper, p.247\u2013249, §3.2 Bootstrap Methods]" in out


def test_mcp_output_preserves_doc_id_annotation(mcp_module):
    results = [
        _result(
            doc_id="alpha123",
            chunk_id="alpha123-0001",
            page_start=5, page_end=5,
        )
    ]
    out = mcp_module.format_results_for_context(results, "q")
    assert "*doc_id: alpha123, chunk: alpha123-0001*" in out
    # Citation prefix appears before the annotation line.
    assert out.index("[alpha-paper, p.5]") < out.index("*doc_id:")


def test_mcp_output_handles_missing_metadata_gracefully(mcp_module):
    results = [
        _result(
            source="legacy-paper.pdf",
            page_start=None, page_end=None, section=None,
        )
    ]
    out = mcp_module.format_results_for_context(results, "q")
    assert "[legacy-paper]" in out
    assert "[legacy-paper, " not in out


# ---------------------------------------------------------------------------
# Rerank integration (Story 18.2)
# ---------------------------------------------------------------------------


def test_search_corpus_with_rerank_config_applies_reranker(mcp_module, monkeypatch):
    """search_corpus() threads rerank_config through to the reranker."""
    import numpy as np
    from grounding.reranker import RerankConfig

    # Stub out index / chunk_map / embedder / read_chunk / config.
    fake_index = type("FI", (), {})()

    def fake_search(_vec, k):
        return (
            np.array([[0.1 * (i + 1) for i in range(k)]]),
            np.array([[i for i in range(k)]]),
        )

    fake_index.search = fake_search
    fake_chunk_map = [
        {"file_path": f"doc-{s}/chunks/ch_0001.md"} for s in ("a", "b", "c", "d")
    ]

    monkeypatch.setattr(
        mcp_module,
        "load_faiss_index",
        lambda _e, _a: (fake_index, fake_chunk_map),
    )

    fake_embedder = type("E", (), {})()
    fake_embedder.encode = lambda qs, normalize_embeddings: np.array([[0.0]])
    monkeypatch.setattr(mcp_module, "get_embedder", lambda: fake_embedder)

    def fake_read_chunk(chunk_path, _corpus_dir):
        slug = str(chunk_path).split("/")[0]
        return {
            "path": str(chunk_path),
            "content": f"content-{slug}",
            "metadata": {"source": f"{slug}.pdf", "doc_id": slug, "chunk_id": "1"},
        }

    monkeypatch.setattr(mcp_module, "read_chunk", fake_read_chunk)

    captured = {}

    def fake_rerank(query, chunks, *, config, text_key="content"):
        captured["n"] = len(chunks)
        captured["cfg"] = config
        # Reverse order to prove reranker affected output.
        rev = list(reversed(list(chunks)))
        return [
            {**dict(c), "rerank_score": float(i), "faiss_distance": c.get("score"), "score": float(i)}
            for i, c in enumerate(rev)
        ]

    monkeypatch.setattr(mcp_module._reranker_module, "rerank", fake_rerank)

    results = mcp_module.search_corpus(
        "q",
        "agent-x",
        top_k=2,
        rerank_config=RerankConfig(enabled=True, pool_size=4),
    )

    assert captured["n"] == 4  # FAISS fetched pool_size
    assert captured["cfg"].enabled is True
    assert len(results) == 2
    # After reversal, doc-d is the first result.
    assert results[0]["source"] == "doc-d.pdf"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2


def test_search_corpus_without_rerank_does_not_call_reranker(mcp_module, monkeypatch):
    """When rerank_config is None, the reranker is never invoked."""
    import numpy as np

    fake_index = type("FI", (), {})()
    fake_index.search = lambda _v, k: (
        np.array([[0.1, 0.2]]),
        np.array([[0, 1]]),
    )
    monkeypatch.setattr(
        mcp_module,
        "load_faiss_index",
        lambda _e, _a: (fake_index, [{"file_path": "a/chunks/ch_0001.md"}, {"file_path": "b/chunks/ch_0001.md"}]),
    )
    fake_embedder = type("E", (), {})()
    fake_embedder.encode = lambda qs, normalize_embeddings: np.array([[0.0]])
    monkeypatch.setattr(mcp_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(
        mcp_module,
        "read_chunk",
        lambda p, _c: {"path": str(p), "content": "x", "metadata": {}},
    )

    def boom(*a, **kw):
        raise AssertionError("reranker should not be called when rerank_config is None")

    monkeypatch.setattr(mcp_module._reranker_module, "rerank", boom)

    results = mcp_module.search_corpus("q", "agent-x", top_k=2)
    assert len(results) == 2


def test_call_tool_builds_rerank_config_from_arguments(mcp_module, monkeypatch):
    """call_tool constructs a RerankConfig when rerank_enabled=True."""
    import asyncio

    captured = {}

    def fake_search_corpus(query, agent, top_k, rerank_config=None, hybrid_config=None):
        captured["rerank_config"] = rerank_config
        captured["hybrid_config"] = hybrid_config
        return []

    monkeypatch.setattr(mcp_module, "search_corpus", fake_search_corpus)

    asyncio.run(
        mcp_module.call_tool(
            "search_corpus",
            {
                "query": "q",
                "agent": "scientist",
                "top_k": 5,
                "rerank_enabled": True,
                "rerank_pool_size": 32,
                "rerank_model": "BAAI/bge-reranker-base",
            },
        )
    )

    cfg = captured["rerank_config"]
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.pool_size == 32
    assert cfg.model == "BAAI/bge-reranker-base"


def test_call_tool_no_rerank_when_flag_absent(mcp_module, monkeypatch):
    """rerank_config stays None when rerank_enabled is not provided."""
    import asyncio

    captured = {}

    def fake_search_corpus(query, agent, top_k, rerank_config=None, hybrid_config=None):
        captured["rerank_config"] = rerank_config
        captured["hybrid_config"] = hybrid_config
        return []

    monkeypatch.setattr(mcp_module, "search_corpus", fake_search_corpus)

    asyncio.run(
        mcp_module.call_tool(
            "search_corpus",
            {"query": "q", "agent": "scientist", "top_k": 5},
        )
    )

    assert captured["rerank_config"] is None


def test_tool_schema_declares_rerank_fields(mcp_module):
    """search_corpus tool schema exposes the three rerank fields."""
    import asyncio

    tools = asyncio.run(mcp_module.list_tools())
    search_tool = next(t for t in tools if t.name == "search_corpus")
    props = search_tool.inputSchema["properties"]

    assert "rerank_enabled" in props
    assert props["rerank_enabled"]["type"] == "boolean"
    assert props["rerank_enabled"]["default"] is False

    assert "rerank_pool_size" in props
    assert props["rerank_pool_size"]["type"] == "integer"
    assert props["rerank_pool_size"]["default"] == 50

    assert "rerank_model" in props
    assert props["rerank_model"]["type"] == "string"
    assert props["rerank_model"]["default"] == "BAAI/bge-reranker-base"
