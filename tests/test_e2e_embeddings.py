"""End-to-end integration tests for embeddings workflow."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def temp_corpus(tmp_path):
    """Create temporary corpus for testing."""
    corpus_path = tmp_path / "test_corpus"
    return corpus_path


def test_end_to_end_embeddings_workflow(temp_corpus):
    """Test complete workflow: ingest → query → verify."""
    # Step 1: Ingest with embeddings
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.cli",
            "--in", "test_pdfs",
            "--out", str(temp_corpus),
            "--parser", "unstructured",
            "--emit-embeddings",
            "--clean"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Ingestion failed: {result.stderr}"

    # Step 2: Verify vector store files created
    embeddings_file = temp_corpus / "_embeddings.faiss"
    chunk_map_file = temp_corpus / "_chunk_map.json"

    assert embeddings_file.exists(), "Vector index file not created"
    assert chunk_map_file.exists(), "Chunk map file not created"

    # Step 3: Load chunk map and verify
    chunk_map = json.loads(chunk_map_file.read_text())
    assert "chunk_ids" in chunk_map, "Chunk map missing chunk_ids"
    assert len(chunk_map["chunk_ids"]) > 0, "Chunk map has no chunks"

    # Step 4: Query corpus
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.query",
            "--corpus", str(temp_corpus),
            "--query", "plain text",
            "--top-k", "3",
            "--format", "json"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

    assert result.returncode == 0, f"Query failed: {result.stderr}"

    # Step 5: Verify query results
    query_results = json.loads(result.stdout)
    assert "results" in query_results, "Query results missing 'results' key"
    assert len(query_results["results"]) <= 3, "Too many results returned"
    assert len(query_results["results"]) > 0, "No results returned"

    # Verify each result has required fields
    for r in query_results["results"]:
        assert "score" in r, "Result missing score"
        assert r["score"] > 0, "Result has invalid score"
        assert "chunk_id" in r, "Result missing chunk_id"
        assert "content" in r, "Result missing content"


def test_ingestion_without_embeddings_flag(temp_corpus):
    """Test that ingestion works without --emit-embeddings (backward compatibility)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.cli",
            "--in", "test_pdfs",
            "--out", str(temp_corpus),
            "--parser", "unstructured",
            "--clean"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Ingestion without embeddings failed: {result.stderr}"

    # Verify vector store files NOT created
    embeddings_file = temp_corpus / "_embeddings.faiss"
    chunk_map_file = temp_corpus / "_chunk_map.json"

    assert not embeddings_file.exists(), "Vector index should not exist without --emit-embeddings"
    assert not chunk_map_file.exists(), "Chunk map should not exist without --emit-embeddings"


def test_vector_db_validation():
    """Test that --vector-db flag validates correctly."""
    # Test with unsupported vector DB
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.cli",
            "--in", "test_pdfs",
            "--out", "/tmp/test_corpus",
            "--parser", "unstructured",
            "--emit-embeddings",
            "--vector-db", "chroma"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 1, "Should fail with unsupported vector DB"
    assert "not yet implemented" in result.stderr, "Error message missing"


def test_query_without_embeddings_fails(tmp_path):
    """Test that querying without embeddings produces helpful error."""
    corpus_without_embeddings = tmp_path / "no_embeddings"
    corpus_without_embeddings.mkdir()

    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.query",
            "--corpus", str(corpus_without_embeddings),
            "--query", "test query",
            "--top-k", "5"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode != 0, "Should fail when embeddings not found"
    assert "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower(), \
        "Error message should mention missing files"
