"""Performance benchmark tests for embeddings."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def sample_corpus_with_embeddings(tmp_path):
    """Create a sample corpus with embeddings for query testing."""
    corpus_path = tmp_path / "benchmark_corpus"

    # Ingest test PDFs with embeddings
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.cli",
            "--in", "test_pdfs",
            "--out", str(corpus_path),
            "--parser", "unstructured",
            "--emit-embeddings",
            "--clean"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Corpus creation failed: {result.stderr}"
    assert (corpus_path / "_embeddings.faiss").exists()

    return corpus_path


def test_embedding_generation_performance():
    """Benchmark embedding generation speed."""
    try:
        from grounding.embedder import generate_embedding
    except ImportError:
        pytest.skip("Embedder module not available")

    # Generate sample texts
    sample_texts = [f"Sample text content for testing purposes {i}" for i in range(100)]

    # Warm up (first call may include model loading time)
    _ = generate_embedding(sample_texts[0])

    # Benchmark
    start = time.perf_counter()
    for text in sample_texts:
        embedding = generate_embedding(text)
        assert embedding is not None
        assert len(embedding) > 0
    elapsed = time.perf_counter() - start

    chunks_per_sec = len(sample_texts) / elapsed

    # Assert meets target: >100 chunks/sec
    assert chunks_per_sec > 100, \
        f"Embedding generation too slow: {chunks_per_sec:.1f} chunks/sec (target: >100)"

    print(f"Embedding performance: {chunks_per_sec:.1f} chunks/sec")


def test_query_latency_performance(sample_corpus_with_embeddings):
    """Benchmark query latency."""
    corpus = sample_corpus_with_embeddings
    query_text = "test query for performance"

    # Warm up query (first query may include model loading)
    warmup_result = subprocess.run(
        [
            sys.executable, "-m", "grounding.query",
            "--corpus", str(corpus),
            "--query", query_text,
            "--top-k", "10",
            "--format", "json"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert warmup_result.returncode == 0, "Warm-up query failed"

    # Benchmark query latency
    start = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable, "-m", "grounding.query",
            "--corpus", str(corpus),
            "--query", query_text,
            "--top-k", "10",
            "--format", "json"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    elapsed = time.perf_counter() - start

    assert result.returncode == 0, f"Query failed: {result.stderr}"

    elapsed_ms = elapsed * 1000

    # Assert meets target: <100ms
    # Note: subprocess overhead included, so we allow some margin
    assert elapsed_ms < 500, \
        f"Query too slow: {elapsed_ms:.1f}ms (target: <100ms for pure query, <500ms with subprocess)"

    # Verify results are valid
    query_results = json.loads(result.stdout)
    assert "results" in query_results
    assert len(query_results["results"]) > 0

    print(f"Query latency: {elapsed_ms:.1f}ms")


def test_batch_embedding_performance():
    """Test performance characteristics of batch embedding generation."""
    try:
        from grounding.embedder import generate_embedding
    except ImportError:
        pytest.skip("Embedder module not available")

    # Test with different batch sizes
    batch_sizes = [10, 50, 100]

    for batch_size in batch_sizes:
        texts = [f"Text {i}" for i in range(batch_size)]

        # Warm up
        _ = generate_embedding(texts[0])

        # Measure
        start = time.perf_counter()
        for text in texts:
            _ = generate_embedding(text)
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed

        # Ensure reasonable performance scales
        assert throughput > 50, \
            f"Batch size {batch_size}: {throughput:.1f} chunks/sec (too slow)"

        print(f"Batch {batch_size}: {throughput:.1f} chunks/sec")


def test_memory_overhead_embeddings(sample_corpus_with_embeddings):
    """Verify memory overhead is reasonable."""
    corpus = sample_corpus_with_embeddings

    # Get embedding file size
    embeddings_file = corpus / "_embeddings.faiss"
    chunk_map_file = corpus / "_chunk_map.json"

    assert embeddings_file.exists()
    assert chunk_map_file.exists()

    # Load chunk map to count chunks
    chunk_map = json.loads(chunk_map_file.read_text())
    num_chunks = len(chunk_map["chunk_ids"])

    # Get file sizes
    embeddings_size = embeddings_file.stat().st_size
    chunk_map_size = chunk_map_file.stat().st_size
    total_size = embeddings_size + chunk_map_size

    # Calculate per-chunk overhead
    bytes_per_chunk = total_size / num_chunks
    kb_per_chunk = bytes_per_chunk / 1024

    # Verify overhead is reasonable (~1.5KB per chunk)
    # Allow up to 5KB per chunk as upper bound
    assert kb_per_chunk < 5.0, \
        f"Memory overhead too high: {kb_per_chunk:.2f}KB per chunk (expected ~1.5KB)"

    print(f"Memory overhead: {kb_per_chunk:.2f}KB per chunk ({num_chunks} chunks)")
