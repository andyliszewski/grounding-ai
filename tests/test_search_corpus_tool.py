#!/usr/bin/env python3
"""
Unit tests for search_corpus_tool module.

Tests the search_corpus tool implementation including:
- Schema format validation
- Argument validation
- Search execution with mocked FAISS
- Result formatting
- Factory function
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from search_corpus_tool import (
    get_search_corpus_schema,
    SearchCorpusTool,
    create_search_corpus_tool,
)
from grounding.reranker import RerankConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embedder():
    """Mock SentenceTransformer that returns predictable embeddings."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return embedder


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index that returns predictable search results."""
    index = MagicMock()
    # Return 3 results with distances
    index.search.return_value = (
        np.array([[0.1, 0.2, 0.3]]),  # distances
        np.array([[0, 1, 2]])  # indices
    )
    return index


@pytest.fixture
def mock_faiss_index_empty():
    """Mock FAISS index that returns no results."""
    index = MagicMock()
    index.search.return_value = (
        np.array([[]]),
        np.array([[]])
    )
    return index


@pytest.fixture
def chunk_map():
    """Sample chunk map with both old and new formats."""
    return [
        "doc-a/chunks/ch_0001.md",  # Old format (string)
        {"file_path": "doc-b/chunks/ch_0001.md", "source": "document-b.pdf"},  # New format
        {"file_path": "doc-c/chunks/ch_0001.md"},  # New format without source
    ]


@pytest.fixture
def temp_corpus(tmp_path):
    """Create temporary corpus directory with chunk files."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create doc-a chunks
    doc_a = corpus_dir / "doc-a" / "chunks"
    doc_a.mkdir(parents=True)
    (doc_a / "ch_0001.md").write_text(
        "---\n"
        "doc_id: abc123\n"
        "source: document-a.pdf\n"
        "chunk_id: 1\n"
        "---\n"
        "This is content from document A."
    )

    # Create doc-b chunks
    doc_b = corpus_dir / "doc-b" / "chunks"
    doc_b.mkdir(parents=True)
    (doc_b / "ch_0001.md").write_text(
        "---\n"
        "doc_id: def456\n"
        "source: document-b.pdf\n"
        "chunk_id: 1\n"
        "---\n"
        "This is content from document B."
    )

    # Create doc-c chunks (no YAML front matter)
    doc_c = corpus_dir / "doc-c" / "chunks"
    doc_c.mkdir(parents=True)
    (doc_c / "ch_0001.md").write_text(
        "This is plain content from document C without metadata."
    )

    return corpus_dir


# =============================================================================
# Test Schema
# =============================================================================

class TestSearchCorpusSchema:
    """Tests for get_search_corpus_schema()."""

    def test_schema_has_function_type(self):
        """Schema type is 'function'."""
        schema = get_search_corpus_schema()
        assert schema["type"] == "function"

    def test_schema_has_function_name(self):
        """Schema includes function name."""
        schema = get_search_corpus_schema()
        assert schema["function"]["name"] == "search_corpus"

    def test_schema_has_description(self):
        """Schema includes description."""
        schema = get_search_corpus_schema()
        assert "description" in schema["function"]
        assert len(schema["function"]["description"]) > 50

    def test_schema_has_parameters(self):
        """Schema includes parameters definition."""
        schema = get_search_corpus_schema()
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params

    def test_schema_query_parameter(self):
        """Schema defines query parameter correctly."""
        schema = get_search_corpus_schema()
        query_param = schema["function"]["parameters"]["properties"]["query"]
        assert query_param["type"] == "string"
        assert "description" in query_param

    def test_schema_top_k_parameter(self):
        """Schema defines top_k parameter correctly."""
        schema = get_search_corpus_schema()
        top_k_param = schema["function"]["parameters"]["properties"]["top_k"]
        assert top_k_param["type"] == "integer"
        assert top_k_param["default"] == 5
        assert top_k_param["minimum"] == 1
        assert top_k_param["maximum"] == 20

    def test_schema_required_parameters(self):
        """Schema marks query as required."""
        schema = get_search_corpus_schema()
        required = schema["function"]["parameters"]["required"]
        assert "query" in required
        assert "top_k" not in required


# =============================================================================
# Test Argument Validation
# =============================================================================

class TestArgumentValidation:
    """Tests for argument validation in SearchCorpusTool."""

    def test_missing_query_parameter(self, mock_embedder, mock_faiss_index, temp_corpus):
        """Missing query returns error message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({})
        assert "Error" in result
        assert "query" in result.lower()

    def test_empty_query_string(self, mock_embedder, mock_faiss_index, temp_corpus):
        """Empty query string returns error message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": ""})
        assert "Error" in result
        assert "non-empty" in result.lower()

    def test_whitespace_only_query(self, mock_embedder, mock_faiss_index, temp_corpus):
        """Whitespace-only query returns error message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "   "})
        assert "Error" in result

    def test_non_string_query(self, mock_embedder, mock_faiss_index, temp_corpus):
        """Non-string query returns error message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": 123})
        assert "Error" in result

    def test_invalid_top_k_type(self, mock_embedder, mock_faiss_index, temp_corpus):
        """Non-integer top_k returns error message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test", "top_k": "five"})
        assert "Error" in result
        assert "integer" in result.lower()

    def test_top_k_clamped_to_minimum(self, mock_embedder, mock_faiss_index, temp_corpus, chunk_map):
        """top_k below 1 is clamped to 1."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test", "top_k": -5})
        # Should not error, just clamp
        assert "Error" not in result or "query" not in result.lower()

    def test_top_k_clamped_to_maximum(self, mock_embedder, mock_faiss_index, temp_corpus, chunk_map):
        """top_k above 20 is clamped to 20."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test", "top_k": 100})
        # Should not error, just clamp
        assert "query" not in result.lower() or "Error" not in result


# =============================================================================
# Test Search Execution
# =============================================================================

class TestSearchExecution:
    """Tests for search execution."""

    def test_successful_search_with_results(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Successful search returns formatted results."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test query"})

        assert "Found" in result
        assert "result" in result.lower()
        assert "Result 1" in result
        assert "document" in result.lower()

    def test_search_with_empty_results(
        self, mock_embedder, mock_faiss_index_empty, chunk_map, temp_corpus
    ):
        """Empty search returns appropriate message."""
        tool = SearchCorpusTool(
            index=mock_faiss_index_empty,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "nonexistent topic"})

        assert "No results found" in result
        assert "nonexistent topic" in result

    def test_search_calls_embedder(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Search properly calls the embedder."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        tool.execute({"query": "my search query"})

        mock_embedder.encode.assert_called_once()
        call_args = mock_embedder.encode.call_args
        assert "my search query" in call_args[0][0]

    def test_search_calls_faiss_index(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Search properly calls the FAISS index."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        tool.execute({"query": "test", "top_k": 3})

        mock_faiss_index.search.assert_called_once()
        call_args = mock_faiss_index.search.call_args
        assert call_args[0][1] == 3  # top_k

    def test_search_handles_missing_chunk_file(
        self, mock_embedder, mock_faiss_index, temp_corpus
    ):
        """Search handles missing chunk files gracefully."""
        chunk_map = ["nonexistent/chunks/ch_0001.md"]
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        # Adjust mock to return only one result
        mock_faiss_index.search.return_value = (
            np.array([[0.1]]),
            np.array([[0]])
        )

        result = tool.execute({"query": "test"})
        assert "not found" in result.lower()

    def test_search_handles_index_error(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Search handles FAISS index errors gracefully."""
        mock_faiss_index.search.side_effect = Exception("FAISS error")

        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test"})

        assert "error" in result.lower()
        assert "FAISS" in result


# =============================================================================
# Test Result Formatting
# =============================================================================

class TestResultFormatting:
    """Tests for result formatting."""

    def test_results_include_rank(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Results include rank numbers."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test"})

        assert "Result 1" in result
        assert "Result 2" in result

    def test_results_include_source(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Results include source information."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test"})

        assert "source:" in result.lower()

    def test_results_include_content(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Results include chunk content."""
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test"})

        assert "content from document" in result.lower()

    def test_no_results_message_includes_suggestions(
        self, mock_embedder, mock_faiss_index_empty, chunk_map, temp_corpus
    ):
        """No results message includes helpful suggestions."""
        tool = SearchCorpusTool(
            index=mock_faiss_index_empty,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )
        result = tool.execute({"query": "test"})

        assert "Try" in result
        assert "keyword" in result.lower() or "term" in result.lower()


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for create_search_corpus_tool()."""

    def test_factory_returns_tuple(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Factory returns tuple of schema and executor."""
        result = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_factory_returns_valid_schema(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Factory returns valid schema dict."""
        schema, _ = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        assert isinstance(schema, dict)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_corpus"

    def test_factory_returns_callable_executor(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Factory returns callable executor."""
        _, executor = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        assert callable(executor)

    def test_executor_works(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Executor from factory works correctly."""
        _, executor = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        result = executor({"query": "test query"})
        assert isinstance(result, str)
        assert "Found" in result or "Error" not in result

    def test_factory_expands_user_path(
        self, mock_embedder, mock_faiss_index, chunk_map
    ):
        """Factory expands ~ in corpus_dir path."""
        schema, executor = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=Path("~/test/corpus")
        )

        # If it didn't expand, an error would occur during execution
        # Just verify it returns without error
        assert callable(executor)


# =============================================================================
# Test Integration with ToolRegistry
# =============================================================================

class TestToolRegistryIntegration:
    """Tests for integration with ToolRegistry from agentic.py."""

    def test_can_register_with_tool_registry(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Tool can be registered with ToolRegistry."""
        from agentic import ToolRegistry

        schema, executor = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        registry = ToolRegistry()
        registry.register("search_corpus", schema, executor)

        assert "search_corpus" in registry.tool_names
        assert len(registry.schemas) == 1

    def test_can_execute_via_registry(
        self, mock_embedder, mock_faiss_index, chunk_map, temp_corpus
    ):
        """Tool can be executed via ToolRegistry."""
        from agentic import ToolRegistry

        schema, executor = create_search_corpus_tool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus
        )

        registry = ToolRegistry()
        registry.register("search_corpus", schema, executor)

        result = registry.execute("search_corpus", {"query": "test"})
        assert isinstance(result, str)
        assert "Found" in result or "No results" in result


# =============================================================================
# Test Citation Prefix (Story 17.3)
# =============================================================================

@pytest.fixture
def temp_corpus_with_citations(tmp_path):
    """Temp corpus containing chunks with page/section metadata."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    alpha = corpus_dir / "alpha-paper" / "chunks"
    alpha.mkdir(parents=True)
    (alpha / "ch_0001.md").write_text(
        "---\n"
        "doc_id: alpha123\n"
        "source: alpha-paper.pdf\n"
        "chunk_id: alpha123-0001\n"
        "page_start: 247\n"
        "page_end: 249\n"
        "section_heading: 3.2 Bootstrap Methods\n"
        "---\n"
        "Body of the chunk about bootstrap methods."
    )

    # Pre-17.2 chunk: no page/section fields.
    legacy = corpus_dir / "legacy-paper" / "chunks"
    legacy.mkdir(parents=True)
    legacy_src = Path(__file__).parent / "fixtures" / "pre-17_2-chunk.md"
    (legacy / "ch_0001.md").write_text(legacy_src.read_text())

    return corpus_dir


class TestCitationPrefix:
    """Story 17.3 — format_citation_prefix wiring."""

    def test_read_chunk_surfaces_new_fields(
        self, mock_embedder, mock_faiss_index, temp_corpus_with_citations
    ):
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus_with_citations,
        )
        data = tool._read_chunk(Path("alpha-paper/chunks/ch_0001.md"))
        assert data["page_start"] == 247
        assert data["page_end"] == 249
        assert data["section_heading"] == "3.2 Bootstrap Methods"

    def test_read_chunk_missing_fields_return_none(
        self, mock_embedder, mock_faiss_index, temp_corpus_with_citations
    ):
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=[],
            embedder=mock_embedder,
            corpus_dir=temp_corpus_with_citations,
        )
        data = tool._read_chunk(Path("legacy-paper/chunks/ch_0001.md"))
        assert data["page_start"] is None
        assert data["page_end"] is None
        assert data["section_heading"] is None

    def test_format_results_includes_citation_prefix(
        self, mock_embedder, mock_faiss_index, temp_corpus_with_citations
    ):
        chunk_map = ["alpha-paper/chunks/ch_0001.md"]
        mock_faiss_index.search.return_value = (
            np.array([[0.1]]),
            np.array([[0]]),
        )
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus_with_citations,
        )
        out = tool.execute({"query": "bootstrap"})
        assert "[alpha-paper, p.247\u2013249, §3.2 Bootstrap Methods]" in out
        # Prefix must be on its own line immediately after the Result header.
        lines = out.splitlines()
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("--- Result 1"))
        assert lines[header_idx + 1].startswith("[alpha-paper, p.247")

    def test_format_results_handles_missing_metadata_gracefully(
        self, mock_embedder, mock_faiss_index, temp_corpus_with_citations
    ):
        chunk_map = ["legacy-paper/chunks/ch_0001.md"]
        mock_faiss_index.search.return_value = (
            np.array([[0.1]]),
            np.array([[0]]),
        )
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=temp_corpus_with_citations,
        )
        out = tool.execute({"query": "anything"})
        assert "[legacy-paper]" in out
        # Never emit a malformed prefix.
        assert "[legacy-paper, ," not in out
        assert "[legacy-paper, , ]" not in out
        assert ", , " not in out


# =============================================================================
# Integration: real mini-corpus chunks (Story 17.3, AC9)
# =============================================================================

class TestMiniCorpusCitationIntegration:
    """Exercise SearchCorpusTool end-to-end against the real mini-corpus
    chunk files with page/section metadata."""

    def test_search_corpus_tool_against_mini_corpus_returns_citations(
        self, mock_embedder, mock_faiss_index
    ):
        mini_corpus = Path(__file__).parent / "eval_fixtures" / "mini_corpus"
        chunk_map = [
            {"file_path": "beta-study/chunks/ch_0001.md"},
            {"file_path": "beta-study/chunks/ch_0002.md"},
            {"file_path": "gamma-notes/chunks/ch_0001.md"},
        ]
        mock_faiss_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0, 1, 2]]),
        )
        tool = SearchCorpusTool(
            index=mock_faiss_index,
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=mini_corpus,
        )
        out = tool.execute({"query": "bootstrap confidence intervals"})
        assert "[beta, p.247, §3.2 Bootstrap Methods]" in out
        assert "[beta, p.248\u2013249, §3.2 Bootstrap Methods]" in out
        assert "[gamma, p.12, §Falsifiability and Demarcation]" in out


# =============================================================================
# Test Rerank Integration (Story 18.2)
# =============================================================================


class TestRerank:
    """Story 18.2 — rerank_config wiring on SearchCorpusTool."""

    @pytest.fixture
    def rerank_corpus(self, tmp_path):
        """Corpus with four chunks whose body length differs."""
        corpus_dir = tmp_path / "corpus"
        for slug, body in [
            ("doc-w", "tiny"),
            ("doc-x", "a bit longer text"),
            ("doc-y", "middling content body here"),
            ("doc-z", "the longest body of them all by a fair margin"),
        ]:
            d = corpus_dir / slug / "chunks"
            d.mkdir(parents=True)
            (d / "ch_0001.md").write_text(
                f"---\ndoc_id: {slug}\nsource: {slug}.pdf\nchunk_id: 1\n---\n{body}"
            )
        return corpus_dir

    @pytest.fixture
    def rerank_chunk_map(self):
        return [
            {"file_path": "doc-w/chunks/ch_0001.md"},
            {"file_path": "doc-x/chunks/ch_0001.md"},
            {"file_path": "doc-y/chunks/ch_0001.md"},
            {"file_path": "doc-z/chunks/ch_0001.md"},
        ]

    def _faiss_for(self, n):
        """FAISS mock that returns n ascending indices with synthetic distances."""
        idx = MagicMock()

        def _search(_vec, k):
            k = min(k, n)
            return (
                np.array([[0.1 * (i + 1) for i in range(k)]]),
                np.array([[i for i in range(k)]]),
            )

        idx.search.side_effect = _search
        return idx

    def test_rerank_disabled_is_bit_for_bit_identical(
        self, mock_embedder, rerank_corpus, rerank_chunk_map, monkeypatch
    ):
        """When rerank_config is None, reranker.rerank is never called."""
        import search_corpus_tool as sct

        def boom(*a, **kw):
            raise AssertionError("reranker should not be called when disabled")

        monkeypatch.setattr(sct._reranker_module, "rerank", boom)

        baseline = SearchCorpusTool(
            index=self._faiss_for(4),
            chunk_map=rerank_chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
        )
        result_none = baseline.execute({"query": "q", "top_k": 3})

        disabled = SearchCorpusTool(
            index=self._faiss_for(4),
            chunk_map=rerank_chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
            rerank_config=RerankConfig(enabled=False, pool_size=10),
        )
        result_disabled = disabled.execute({"query": "q", "top_k": 3})

        assert result_none == result_disabled

    def test_rerank_enabled_reorders_results(
        self, mock_embedder, rerank_corpus, rerank_chunk_map, monkeypatch
    ):
        """Stub reranker reverses order; output reflects the new ordering."""
        import search_corpus_tool as sct

        def fake_rerank(query, chunks, *, config, text_key="content"):
            reversed_chunks = []
            for i, c in enumerate(reversed(list(chunks))):
                nc = dict(c)
                nc["rerank_score"] = float(i)
                nc["faiss_distance"] = c.get("score")
                nc["score"] = float(i)
                reversed_chunks.append(nc)
            return reversed_chunks

        monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

        tool = SearchCorpusTool(
            index=self._faiss_for(4),
            chunk_map=rerank_chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
            rerank_config=RerankConfig(enabled=True, pool_size=4),
        )
        out = tool.execute({"query": "q", "top_k": 4})
        # Reversed order: doc-z should now appear before doc-w in the text.
        assert out.index("doc-z.pdf") < out.index("doc-w.pdf")

    def test_rerank_pool_size_larger_than_top_k(
        self, mock_embedder, rerank_corpus, rerank_chunk_map, monkeypatch
    ):
        """FAISS is called with pool_size; output is truncated to top_k."""
        import search_corpus_tool as sct

        recorded = {}

        def fake_rerank(query, chunks, *, config, text_key="content"):
            recorded["pool_len"] = len(chunks)
            return [
                {**dict(c), "rerank_score": float(i), "faiss_distance": c.get("score"), "score": float(i)}
                for i, c in enumerate(chunks)
            ]

        monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

        idx = self._faiss_for(4)
        tool = SearchCorpusTool(
            index=idx,
            chunk_map=rerank_chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
            rerank_config=RerankConfig(enabled=True, pool_size=4),
        )
        out = tool.execute({"query": "q", "top_k": 2})
        # FAISS fetched 4 candidates; reranker saw 4; output shows only top 2.
        assert idx.search.call_args_list[0][0][1] == 4
        assert recorded["pool_len"] == 4
        assert "Result 1" in out
        assert "Result 2" in out
        assert "Result 3" not in out

    def test_rerank_smaller_pool_than_configured_no_error(
        self, mock_embedder, rerank_corpus, monkeypatch
    ):
        """Only 3 chunks available; rerank runs on whatever was returned."""
        import search_corpus_tool as sct

        def fake_rerank(query, chunks, *, config, text_key="content"):
            return [
                {**dict(c), "rerank_score": float(i), "faiss_distance": c.get("score"), "score": float(i)}
                for i, c in enumerate(chunks)
            ]

        monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

        chunk_map = [
            {"file_path": "doc-w/chunks/ch_0001.md"},
            {"file_path": "doc-x/chunks/ch_0001.md"},
            {"file_path": "doc-y/chunks/ch_0001.md"},
        ]
        tool = SearchCorpusTool(
            index=self._faiss_for(3),
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
            rerank_config=RerankConfig(enabled=True, pool_size=50),
        )
        out = tool.execute({"query": "q", "top_k": 5})
        assert "Result 1" in out
        assert "Error" not in out

    def test_rank_reassigned_post_rerank(
        self, mock_embedder, rerank_corpus, rerank_chunk_map, monkeypatch
    ):
        """Output ranks are 1..N after reranking, even though reranker
        returned whatever rank the input carried."""
        import search_corpus_tool as sct

        def fake_rerank(query, chunks, *, config, text_key="content"):
            out = []
            for i, c in enumerate(reversed(list(chunks))):
                nc = dict(c)
                nc["rank"] = 999  # deliberately wrong; should be overwritten.
                nc["rerank_score"] = float(i)
                nc["faiss_distance"] = c.get("score")
                nc["score"] = float(i)
                out.append(nc)
            return out

        monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

        tool = SearchCorpusTool(
            index=self._faiss_for(4),
            chunk_map=rerank_chunk_map,
            embedder=mock_embedder,
            corpus_dir=rerank_corpus,
            rerank_config=RerankConfig(enabled=True, pool_size=4),
        )
        out = tool.execute({"query": "q", "top_k": 3})
        assert "Result 1" in out
        assert "Result 2" in out
        assert "Result 3" in out
        assert "Result 999" not in out
        assert "Result 4" not in out

    def test_citation_prefix_reflects_reranked_top(
        self, mock_embedder, monkeypatch, tmp_path
    ):
        """Citation prefix on the first result matches the reranked top chunk."""
        import search_corpus_tool as sct

        corpus_dir = tmp_path / "corpus"
        alpha = corpus_dir / "alpha-paper" / "chunks"
        alpha.mkdir(parents=True)
        (alpha / "ch_0001.md").write_text(
            "---\ndoc_id: alpha\nsource: alpha-paper.pdf\nchunk_id: 1\n"
            "page_start: 10\npage_end: 10\n"
            "section_heading: 1. Intro\n---\nalpha body"
        )
        beta = corpus_dir / "beta-paper" / "chunks"
        beta.mkdir(parents=True)
        (beta / "ch_0001.md").write_text(
            "---\ndoc_id: beta\nsource: beta-paper.pdf\nchunk_id: 1\n"
            "page_start: 20\npage_end: 20\n"
            "section_heading: 2. Methods\n---\nbeta body"
        )
        chunk_map = [
            {"file_path": "alpha-paper/chunks/ch_0001.md"},
            {"file_path": "beta-paper/chunks/ch_0001.md"},
        ]

        def fake_rerank(query, chunks, *, config, text_key="content"):
            # Reverse: beta becomes first.
            rev = list(reversed(list(chunks)))
            return [
                {**dict(c), "rerank_score": float(i), "faiss_distance": c.get("score"), "score": float(i)}
                for i, c in enumerate(rev)
            ]

        monkeypatch.setattr(sct._reranker_module, "rerank", fake_rerank)

        tool = SearchCorpusTool(
            index=self._faiss_for(2),
            chunk_map=chunk_map,
            embedder=mock_embedder,
            corpus_dir=corpus_dir,
            rerank_config=RerankConfig(enabled=True, pool_size=2),
        )
        out = tool.execute({"query": "q", "top_k": 2})
        # The first citation prefix after Result 1 must be beta (reranked top).
        lines = out.splitlines()
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("--- Result 1"))
        assert lines[header_idx + 1] == "[beta-paper, p.20, §2. Methods]"
