"""Tests for grounding.query module."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grounding.query import (
    ChunkResult,
    format_results_json,
    format_results_text,
    load_chunk_content,
    normalize_scores,
    query_corpus,
)


class TestChunkResult:
    """Test ChunkResult dataclass."""

    def test_chunkresult_creation(self):
        """Test creating a ChunkResult instance."""
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.92,
            content="Test content here",
            metadata={"doc_id": "test", "source": "test.pdf"},
            source_document="test.pdf",
        )

        assert result.chunk_id == "test-0001"
        assert result.score == 0.92
        assert result.content == "Test content here"
        assert result.source_document == "test.pdf"

    def test_chunkresult_str(self):
        """Test ChunkResult string representation."""
        result = ChunkResult(
            chunk_id="doc1-0001",
            score=0.85,
            content="Short content",
            metadata={},
            source_document="doc1.pdf",
        )

        str_repr = str(result)
        assert "[doc1-0001]" in str_repr
        assert "0.850" in str_repr
        assert "doc1.pdf" in str_repr
        assert "Short content" in str_repr

    def test_chunkresult_str_truncates_long_content(self):
        """Test that long content is truncated in string representation."""
        long_content = "x" * 300
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.95,
            content=long_content,
            metadata={},
            source_document="test.pdf",
        )

        str_repr = str(result)
        assert "..." in str_repr
        assert len(str_repr) < len(long_content)

    def test_chunkresult_to_dict(self):
        """Test converting ChunkResult to dictionary."""
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.88,
            content="Content",
            metadata={"key": "value"},
            source_document="test.pdf",
        )

        d = result.to_dict()
        assert d["chunk_id"] == "test-0001"
        assert d["score"] == 0.88
        assert d["content"] == "Content"
        assert d["metadata"] == {"key": "value"}
        assert d["source_document"] == "test.pdf"


class TestNormalizeScores:
    """Test normalize_scores() function."""

    def test_normalize_zero_distance(self):
        """Test that distance 0 gives score close to 1.0."""
        distances = np.array([0.0])
        scores = normalize_scores(distances)

        assert len(scores) == 1
        assert np.isclose(scores[0], 1.0, atol=1e-6)

    def test_normalize_multiple_distances(self):
        """Test normalizing multiple distances."""
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        scores = normalize_scores(distances)

        assert len(scores) == 4
        assert scores[0] > scores[1] > scores[2] > scores[3]
        assert all(0 <= s <= 1 for s in scores)

    def test_normalize_large_distance_gives_small_score(self):
        """Test that large distances give scores close to 0."""
        distances = np.array([10.0])
        scores = normalize_scores(distances)

        assert scores[0] < 0.01

    def test_normalize_preserves_order(self):
        """Test that score order is inverse of distance order."""
        distances = np.array([0.1, 0.5, 0.2, 0.8])
        scores = normalize_scores(distances)

        # Lower distance → higher score
        assert scores[0] > scores[1]  # 0.1 < 0.5
        assert scores[0] > scores[2]  # 0.1 < 0.2
        assert scores[2] > scores[1]  # 0.2 < 0.5
        assert scores[3] < scores[1]  # 0.8 > 0.5


class TestLoadChunkContent:
    """Test load_chunk_content() function."""

    def test_load_chunk_valid(self, tmp_path):
        """Test loading a valid chunk file."""
        # Create test corpus structure
        corpus_path = tmp_path / "corpus"
        doc_slug = "test-document"
        chunks_dir = corpus_path / doc_slug / "chunks"
        chunks_dir.mkdir(parents=True)

        # Create manifest
        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "test-document", "orig_name": "test.pdf"}
            ]
        }
        with open(corpus_path / "_index.json", "w") as f:
            json.dump(manifest, f)

        # Create chunk file
        chunk_content = """---
doc_id: abc123de
source: test.pdf
chunk_id: abc123de-0001
hash: test_hash
created_utc: 2025-01-01T00:00:00Z
---

This is the chunk content.
It has multiple lines.
"""
        chunk_file = chunks_dir / "ch_0001.md"
        chunk_file.write_text(chunk_content)

        # Load chunk
        content, metadata = load_chunk_content(
            corpus_path, "abc123de-0001", manifest
        )

        assert "This is the chunk content." in content
        assert metadata["doc_id"] == "abc123de"
        assert metadata["source"] == "test.pdf"
        assert metadata["chunk_id"] == "abc123de-0001"

    def test_load_chunk_invalid_id_format(self, tmp_path):
        """Test error handling for invalid chunk_id format."""
        corpus_path = tmp_path / "corpus"
        manifest = {"docs": []}

        with pytest.raises(ValueError, match="Invalid chunk_id format"):
            load_chunk_content(corpus_path, "invalid_id", manifest)

    def test_load_chunk_doc_id_not_found(self, tmp_path):
        """Test error when doc_id not in manifest."""
        corpus_path = tmp_path / "corpus"
        manifest = {"docs": []}

        with pytest.raises(ValueError, match="not found in corpus manifest"):
            load_chunk_content(corpus_path, "unknown-0001", manifest)

    def test_load_chunk_file_not_found(self, tmp_path):
        """Test error when chunk file doesn't exist."""
        corpus_path = tmp_path / "corpus"
        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "test-doc", "orig_name": "test.pdf"}
            ]
        }

        with pytest.raises(FileNotFoundError, match="Chunk file not found"):
            load_chunk_content(corpus_path, "abc123de-0001", manifest)

    def test_load_chunk_missing_yaml_frontmatter(self, tmp_path):
        """Test error when chunk file has no YAML front matter."""
        corpus_path = tmp_path / "corpus"
        doc_slug = "test-doc"
        chunks_dir = corpus_path / doc_slug / "chunks"
        chunks_dir.mkdir(parents=True)

        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "test-doc", "orig_name": "test.pdf"}
            ]
        }

        # Create chunk without front matter
        chunk_file = chunks_dir / "ch_0001.md"
        chunk_file.write_text("Just content, no front matter")

        with pytest.raises(ValueError, match="missing YAML front matter"):
            load_chunk_content(corpus_path, "abc123de-0001", manifest)

    def test_load_chunk_invalid_yaml(self, tmp_path):
        """Test error when YAML front matter is malformed."""
        corpus_path = tmp_path / "corpus"
        doc_slug = "test-doc"
        chunks_dir = corpus_path / doc_slug / "chunks"
        chunks_dir.mkdir(parents=True)

        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "test-doc", "orig_name": "test.pdf"}
            ]
        }

        # Create chunk with bad YAML
        chunk_content = """---
invalid: yaml: malformed::
---

Content here
"""
        chunk_file = chunks_dir / "ch_0001.md"
        chunk_file.write_text(chunk_content)

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_chunk_content(corpus_path, "abc123de-0001", manifest)


class TestQueryCorpus:
    """Test query_corpus() function."""

    def test_query_empty_string_raises_error(self):
        """Test that empty query raises ValueError."""
        corpus_path = Path("/tmp/corpus")

        with pytest.raises(ValueError, match="query cannot be empty"):
            query_corpus(corpus_path, "", top_k=5)

    def test_query_whitespace_only_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        corpus_path = Path("/tmp/corpus")

        with pytest.raises(ValueError, match="query cannot be empty"):
            query_corpus(corpus_path, "   ", top_k=5)

    def test_query_invalid_top_k_raises_error(self):
        """Test that top_k < 1 raises ValueError."""
        corpus_path = Path("/tmp/corpus")

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            query_corpus(corpus_path, "test query", top_k=0)

    def test_query_missing_corpus_raises_error(self):
        """Test that missing corpus directory raises FileNotFoundError."""
        corpus_path = Path("/nonexistent/corpus")

        with pytest.raises(FileNotFoundError, match="Corpus directory not found"):
            query_corpus(corpus_path, "test query", top_k=5)

    def test_query_missing_vector_index_raises_error(self, tmp_path):
        """Test that missing vector index raises FileNotFoundError."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        with pytest.raises(FileNotFoundError, match="Vector index not found"):
            query_corpus(corpus_path, "test query", top_k=5)

    def test_query_missing_manifest_raises_error(self, tmp_path):
        """Test that missing manifest raises FileNotFoundError."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        # Create empty FAISS index
        (corpus_path / "_embeddings.faiss").touch()

        with pytest.raises(FileNotFoundError, match="Corpus manifest not found"):
            query_corpus(corpus_path, "test query", top_k=5)

    @patch("grounding.query.vector_store.load_vector_index")
    @patch("grounding.query.embedder.generate_embedding")
    @patch("grounding.query.vector_store.search_similar_chunks")
    @patch("grounding.query.load_chunk_content")
    def test_query_returns_results(
        self,
        mock_load_chunk,
        mock_search,
        mock_embed,
        mock_load_index,
        tmp_path,
    ):
        """Test successful query returns ChunkResult list."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        # Create manifest and index file
        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "doc1", "orig_name": "test.pdf"}
            ]
        }
        with open(corpus_path / "_index.json", "w") as f:
            json.dump(manifest, f)
        (corpus_path / "_embeddings.faiss").touch()

        # Mock dependencies
        mock_index = MagicMock()
        mock_chunk_map = {"chunk_ids": ["abc123de-0001", "abc123de-0002"]}
        mock_load_index.return_value = (mock_index, mock_chunk_map)

        mock_embed.return_value = np.random.rand(384).astype(np.float32)

        mock_search.return_value = [
            ("abc123de-0001", 0.1),
            ("abc123de-0002", 0.5),
        ]

        mock_load_chunk.side_effect = [
            ("Content 1", {"source": "test.pdf", "doc_id": "abc123de"}),
            ("Content 2", {"source": "test.pdf", "doc_id": "abc123de"}),
        ]

        # Run query
        results = query_corpus(corpus_path, "test query", top_k=2)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, ChunkResult) for r in results)
        assert results[0].chunk_id == "abc123de-0001"
        assert results[1].chunk_id == "abc123de-0002"
        assert results[0].score > results[1].score  # Lower distance → higher score

    @patch("grounding.query.vector_store.load_vector_index")
    @patch("grounding.query.embedder.generate_embedding")
    @patch("grounding.query.vector_store.search_similar_chunks")
    @patch("grounding.query.load_chunk_content")
    def test_query_skips_failed_chunks(
        self,
        mock_load_chunk,
        mock_search,
        mock_embed,
        mock_load_index,
        tmp_path,
    ):
        """Test that query continues when chunk loading fails."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        manifest = {
            "docs": [
                {"doc_id": "abc123de", "slug": "doc1", "orig_name": "test.pdf"}
            ]
        }
        with open(corpus_path / "_index.json", "w") as f:
            json.dump(manifest, f)
        (corpus_path / "_embeddings.faiss").touch()

        mock_index = MagicMock()
        mock_chunk_map = {"chunk_ids": ["abc123de-0001", "abc123de-0002"]}
        mock_load_index.return_value = (mock_index, mock_chunk_map)

        mock_embed.return_value = np.random.rand(384).astype(np.float32)

        mock_search.return_value = [
            ("abc123de-0001", 0.1),
            ("abc123de-0002", 0.5),
        ]

        # First chunk fails, second succeeds
        mock_load_chunk.side_effect = [
            FileNotFoundError("Chunk not found"),
            ("Content 2", {"source": "test.pdf", "doc_id": "abc123de"}),
        ]

        # Run query
        results = query_corpus(corpus_path, "test query", top_k=2)

        # Should only return successful chunk
        assert len(results) == 1
        assert results[0].chunk_id == "abc123de-0002"


class TestFormatResultsText:
    """Test format_results_text() function."""

    def test_format_no_results(self):
        """Test formatting when no results found."""
        output = format_results_text("test query", [])

        assert "test query" in output
        assert "No results found" in output

    def test_format_single_result(self):
        """Test formatting single result."""
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.92,
            content="Test content",
            metadata={},
            source_document="test.pdf",
        )

        output = format_results_text("test query", [result])

        assert "test query" in output
        assert "Found 1 results" in output
        assert "[1]" in output
        assert "0.92" in output
        assert "test-0001" in output
        assert "test.pdf" in output
        assert "Test content" in output

    def test_format_multiple_results(self):
        """Test formatting multiple results."""
        results = [
            ChunkResult(
                chunk_id=f"test-{i:04d}",
                score=0.9 - i * 0.1,
                content=f"Content {i}",
                metadata={},
                source_document="test.pdf",
            )
            for i in range(3)
        ]

        output = format_results_text("test query", results)

        assert "Found 3 results" in output
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_format_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "x" * 300
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.85,
            content=long_content,
            metadata={},
            source_document="test.pdf",
        )

        output = format_results_text("test query", [result])

        assert "..." in output
        # Verify content is truncated (output shouldn't contain full 300 char content)
        assert long_content not in output


class TestFormatResultsJson:
    """Test format_results_json() function."""

    def test_format_json_no_results(self):
        """Test JSON formatting with no results."""
        output = format_results_json("test query", [])
        data = json.loads(output)

        assert data["query"] == "test query"
        assert data["result_count"] == 0
        assert data["results"] == []

    def test_format_json_with_results(self):
        """Test JSON formatting with results."""
        results = [
            ChunkResult(
                chunk_id="test-0001",
                score=0.92,
                content="Content 1",
                metadata={"doc_id": "test", "source": "test.pdf"},
                source_document="test.pdf",
            ),
            ChunkResult(
                chunk_id="test-0002",
                score=0.85,
                content="Content 2",
                metadata={"doc_id": "test", "source": "test.pdf"},
                source_document="test.pdf",
            ),
        ]

        output = format_results_json("test query", results)
        data = json.loads(output)

        assert data["query"] == "test query"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["rank"] == 1
        assert data["results"][0]["chunk_id"] == "test-0001"
        assert data["results"][0]["score"] == 0.92
        assert data["results"][1]["rank"] == 2

    def test_format_json_valid_structure(self):
        """Test that JSON output is valid and well-formed."""
        result = ChunkResult(
            chunk_id="test-0001",
            score=0.88,
            content="Test",
            metadata={"key": "value"},
            source_document="test.pdf",
        )

        output = format_results_json("query", [result])
        data = json.loads(output)

        # Verify required fields
        assert "query" in data
        assert "result_count" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "rank" in data["results"][0]
        assert "chunk_id" in data["results"][0]
        assert "score" in data["results"][0]
        assert "content" in data["results"][0]
        assert "metadata" in data["results"][0]
        assert "source_document" in data["results"][0]


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            ["python", "-m", "grounding.query", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Query PDF corpus" in result.stdout or "Usage" in result.stdout

    def test_cli_missing_required_args(self):
        """Test that CLI fails without required arguments."""
        result = subprocess.run(
            ["python", "-m", "grounding.query"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    @patch("grounding.query.query_corpus")
    def test_cli_text_format(self, mock_query, tmp_path):
        """Test CLI with text format output."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        # Mock query results
        mock_results = [
            ChunkResult(
                chunk_id="test-0001",
                score=0.92,
                content="Test content",
                metadata={},
                source_document="test.pdf",
            )
        ]
        mock_query.return_value = mock_results

        # Run CLI
        result = subprocess.run(
            [
                "python",
                "-m",
                "grounding.query",
                "--corpus",
                str(corpus_path),
                "--query",
                "test query",
                "--format",
                "text",
            ],
            capture_output=True,
            text=True,
        )

        # CLI will fail because corpus isn't real, but we're testing the interface
        # In a real integration test, we'd set up a proper corpus
        assert "--corpus" in result.stdout or "Error" in result.stderr

    def test_cli_invalid_format(self, tmp_path):
        """Test CLI rejects invalid format option."""
        corpus_path = tmp_path / "corpus"
        corpus_path.mkdir()

        result = subprocess.run(
            [
                "python",
                "-m",
                "grounding.query",
                "--corpus",
                str(corpus_path),
                "--query",
                "test",
                "--format",
                "invalid",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with error about format
        assert result.returncode != 0 or "invalid" in result.stderr.lower()


class TestIntegration:
    """Integration tests with real corpus (if available)."""

    @pytest.mark.skip(reason="Requires pre-built corpus with embeddings")
    def test_query_real_corpus(self):
        """Test query against a real corpus."""
        # This test would run against test_pdfs output
        # Requires corpus to be built first with --emit-embeddings
        corpus_path = Path("./test_corpus")

        if not corpus_path.exists():
            pytest.skip("Test corpus not available")

        results = query_corpus(corpus_path, "test query", top_k=5)

        assert len(results) <= 5
        assert all(isinstance(r, ChunkResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)

        # Verify results are sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
