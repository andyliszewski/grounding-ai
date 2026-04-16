"""Tests for grounding.vector_store module."""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from grounding.vector_store import (
    CHUNK_MAP_FILENAME,
    DEFAULT_EMBEDDING_DIM,
    FAISS_INDEX_FILENAME,
    FORMAT_VERSION_INCREMENTAL,
    FORMAT_VERSION_WITH_METADATA,
    TOMBSTONE_REBUILD_THRESHOLD,
    TOMBSTONE_WARNING_THRESHOLD,
    StalenessReport,
    append_to_vector_index,
    get_indexed_doc_ids,
    load_vector_index,
    search_similar_chunks,
    should_rebuild_index,
    tombstone_documents,
    write_vector_index,
)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)  # For reproducibility
    return {
        "doc1-0001": np.random.rand(384).astype(np.float32),
        "doc1-0002": np.random.rand(384).astype(np.float32),
        "doc2-0001": np.random.rand(384).astype(np.float32),
        "doc2-0002": np.random.rand(384).astype(np.float32),
        "doc2-0003": np.random.rand(384).astype(np.float32),
    }


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "corpus"


class TestWriteVectorIndex:
    """Test write_vector_index() function."""

    def test_write_creates_files(self, sample_embeddings, output_dir):
        """Test that index and chunk map files are created."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        index_path = output_dir / FAISS_INDEX_FILENAME
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME

        assert index_path.exists(), "FAISS index file should be created"
        assert chunk_map_path.exists(), "Chunk map file should be created"

    def test_index_contains_all_embeddings(self, sample_embeddings, output_dir):
        """Test that index contains all provided embeddings."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        index_path = output_dir / FAISS_INDEX_FILENAME
        index = faiss.read_index(str(index_path))

        assert index.ntotal == len(sample_embeddings), "Index should contain all embeddings"
        assert index.d == DEFAULT_EMBEDDING_DIM, "Index dimension should be 384"

    def test_chunk_map_structure(self, sample_embeddings, output_dir):
        """Test chunk map JSON structure and content."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        # Validate structure
        assert "format_version" in chunk_map
        assert "faiss_version" in chunk_map
        assert "dimension" in chunk_map
        assert "index_size" in chunk_map
        assert "created_utc" in chunk_map
        assert "chunk_ids" in chunk_map

        # Validate values
        assert chunk_map["dimension"] == DEFAULT_EMBEDDING_DIM
        assert chunk_map["index_size"] == len(sample_embeddings)
        assert len(chunk_map["chunk_ids"]) == len(sample_embeddings)
        assert set(chunk_map["chunk_ids"]) == set(sample_embeddings.keys())

    def test_chunk_id_order_preserved(self, sample_embeddings, output_dir):
        """Test that chunk ID order is preserved in chunk map."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        # The order should match the input dict order
        assert chunk_map["chunk_ids"] == list(sample_embeddings.keys())

    def test_empty_embeddings_skips_creation(self, output_dir, caplog):
        """Test graceful handling of empty embeddings dict."""
        import logging

        output_dir.mkdir(parents=True)

        # Capture logs from the vector_store module
        with caplog.at_level(logging.INFO, logger="grounding.vector_store"):
            write_vector_index({}, output_dir)

        index_path = output_dir / FAISS_INDEX_FILENAME
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME

        assert not index_path.exists(), "Index should not be created for empty embeddings"
        assert not chunk_map_path.exists(), "Chunk map should not be created for empty embeddings"
        assert "No embeddings provided" in caplog.text

    def test_inconsistent_dimensions_raises_error(self, output_dir):
        """Test validation of inconsistent embedding dimensions."""
        output_dir.mkdir(parents=True)
        bad_embeddings = {
            "chunk1": np.random.rand(384).astype(np.float32),
            "chunk2": np.random.rand(512).astype(np.float32),  # Wrong dimension
        }

        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            write_vector_index(bad_embeddings, output_dir)

    def test_wrong_dimension_logs_warning(self, output_dir, caplog):
        """Test that non-384 dimensions log a warning but still work."""
        output_dir.mkdir(parents=True)
        embeddings_512 = {
            "chunk1": np.random.rand(512).astype(np.float32),
            "chunk2": np.random.rand(512).astype(np.float32),
        }

        write_vector_index(embeddings_512, output_dir)
        assert "differs from expected" in caplog.text

    def test_atomic_write_pattern(self, sample_embeddings, output_dir):
        """Test that files are written atomically (no .tmp files left)."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        # Check no temporary files remain
        tmp_files = list(output_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, "No temporary files should remain after write"


class TestLoadVectorIndex:
    """Test load_vector_index() function."""

    def test_load_after_write(self, sample_embeddings, output_dir):
        """Test loading an index that was just written."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        index, chunk_map = load_vector_index(output_dir)

        assert index.ntotal == len(sample_embeddings)
        assert index.d == DEFAULT_EMBEDDING_DIM
        assert len(chunk_map["chunk_ids"]) == len(sample_embeddings)

    def test_missing_index_file_raises_error(self, output_dir):
        """Test error when index file is missing."""
        output_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            load_vector_index(output_dir)

    def test_missing_chunk_map_raises_error(self, sample_embeddings, output_dir):
        """Test error when chunk map is missing."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        # Remove chunk map
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        chunk_map_path.unlink()

        with pytest.raises(FileNotFoundError, match="Chunk map not found"):
            load_vector_index(output_dir)

    def test_index_chunk_map_consistency_validation(self, sample_embeddings, output_dir):
        """Test validation of consistency between index and chunk map."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        # Corrupt chunk map by changing index_size
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        chunk_map["index_size"] = 999  # Wrong size
        with open(chunk_map_path, "w") as f:
            json.dump(chunk_map, f)

        with pytest.raises(ValueError, match="Index size mismatch"):
            load_vector_index(output_dir)

    def test_dimension_mismatch_raises_error(self, sample_embeddings, output_dir):
        """Test validation of dimension mismatch."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        # Corrupt chunk map by changing dimension
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        chunk_map["dimension"] = 512  # Wrong dimension
        with open(chunk_map_path, "w") as f:
            json.dump(chunk_map, f)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            load_vector_index(output_dir)

    def test_chunk_id_count_mismatch_raises_error(self, sample_embeddings, output_dir):
        """Test validation of chunk ID count."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        # Corrupt chunk map by removing a chunk ID
        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        chunk_map["chunk_ids"].pop()  # Remove one ID
        with open(chunk_map_path, "w") as f:
            json.dump(chunk_map, f)

        with pytest.raises(ValueError, match="Chunk map size mismatch"):
            load_vector_index(output_dir)


class TestSearchSimilarChunks:
    """Test search_similar_chunks() function."""

    def test_search_returns_results(self, sample_embeddings, output_dir):
        """Test that search returns expected number of results."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        # Use one of the original embeddings as query
        query = sample_embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=3)

        assert len(results) == 3, "Should return top 3 results"
        assert all(isinstance(r, tuple) for r in results), "Results should be tuples"
        assert all(len(r) == 2 for r in results), "Each result should be (chunk_id, distance)"

    def test_search_finds_exact_match(self, sample_embeddings, output_dir):
        """Test that search finds exact match with near-zero distance."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        # Use one of the original embeddings as query
        query = sample_embeddings["doc2-0002"]
        results = search_similar_chunks(index, chunk_map, query, top_k=1)

        assert len(results) == 1
        chunk_id, distance = results[0]
        assert chunk_id == "doc2-0002", "Should find the exact match"
        assert distance < 1e-5, f"Distance should be near zero, got {distance}"

    def test_search_result_order(self, sample_embeddings, output_dir):
        """Test that results are ordered by distance (ascending)."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        query = sample_embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=5)

        # Distances should be in ascending order
        distances = [r[1] for r in results]
        assert distances == sorted(distances), "Results should be ordered by distance"

    def test_search_with_top_k_larger_than_index(self, sample_embeddings, output_dir):
        """Test search when top_k exceeds index size."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        query = sample_embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=100)

        # Should return only available embeddings
        assert len(results) == len(sample_embeddings), "Should return all available embeddings"

    def test_search_empty_index_returns_empty(self, output_dir):
        """Test search on empty index returns empty list."""
        output_dir.mkdir(parents=True)

        # Create empty index manually
        index = faiss.IndexFlatL2(384)
        chunk_map = {"chunk_ids": [], "dimension": 384, "index_size": 0}

        query = np.random.rand(384).astype(np.float32)
        results = search_similar_chunks(index, chunk_map, query, top_k=5)

        assert results == [], "Empty index should return empty results"

    def test_search_dimension_mismatch_raises_error(self, sample_embeddings, output_dir):
        """Test that query with wrong dimension raises error."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        # Wrong dimension query
        wrong_query = np.random.rand(512).astype(np.float32)

        with pytest.raises(ValueError, match="Query dimension"):
            search_similar_chunks(index, chunk_map, wrong_query, top_k=5)

    def test_search_chunk_id_mapping_accuracy(self, sample_embeddings, output_dir):
        """Test that index positions correctly map to chunk IDs."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        # For each embedding, search and verify it finds itself first
        for chunk_id, embedding in sample_embeddings.items():
            results = search_similar_chunks(index, chunk_map, embedding, top_k=1)
            found_id, distance = results[0]

            assert found_id == chunk_id, f"Search for {chunk_id} should find itself first"
            assert distance < 1e-5, f"Self-match distance should be near zero"


class TestPortability:
    """Test file format portability."""

    def test_write_read_cycle_preserves_data(self, sample_embeddings, output_dir):
        """Test that write/read cycle preserves all data."""
        output_dir.mkdir(parents=True)

        # Write
        write_vector_index(sample_embeddings, output_dir)

        # Read
        index, chunk_map = load_vector_index(output_dir)

        # Verify all chunk IDs present
        assert set(chunk_map["chunk_ids"]) == set(sample_embeddings.keys())

        # Verify we can search with each original embedding
        for chunk_id, embedding in sample_embeddings.items():
            results = search_similar_chunks(index, chunk_map, embedding, top_k=1)
            assert results[0][0] == chunk_id

    def test_faiss_version_recorded(self, sample_embeddings, output_dir):
        """Test that FAISS version is recorded in chunk map."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        assert "faiss_version" in chunk_map
        assert chunk_map["faiss_version"] == faiss.__version__

    def test_format_version_recorded(self, sample_embeddings, output_dir):
        """Test that format version is recorded."""
        output_dir.mkdir(parents=True)
        write_vector_index(sample_embeddings, output_dir)

        chunk_map_path = output_dir / CHUNK_MAP_FILENAME
        with open(chunk_map_path, "r") as f:
            chunk_map = json.load(f)

        assert chunk_map["format_version"] == "1.0"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_embedding(self, output_dir):
        """Test with just one embedding."""
        output_dir.mkdir(parents=True)
        embeddings = {"single-chunk": np.random.rand(384).astype(np.float32)}

        write_vector_index(embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        assert index.ntotal == 1
        assert len(chunk_map["chunk_ids"]) == 1

    def test_large_chunk_ids(self, output_dir):
        """Test with long chunk ID strings."""
        output_dir.mkdir(parents=True)
        embeddings = {
            "very-long-document-id-with-many-characters-0001": np.random.rand(384).astype(
                np.float32
            ),
            "very-long-document-id-with-many-characters-0002": np.random.rand(384).astype(
                np.float32
            ),
        }

        write_vector_index(embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        assert len(chunk_map["chunk_ids"]) == 2

    def test_special_characters_in_chunk_ids(self, output_dir):
        """Test chunk IDs with special characters."""
        output_dir.mkdir(parents=True)
        embeddings = {
            "doc-with-dash_and_underscore-0001": np.random.rand(384).astype(np.float32),
            "doc.with.dots-0002": np.random.rand(384).astype(np.float32),
        }

        write_vector_index(embeddings, output_dir)
        index, chunk_map = load_vector_index(output_dir)

        assert set(chunk_map["chunk_ids"]) == set(embeddings.keys())


# ============================================================================
# Story 14.1: Incremental Vector Store Append Tests
# ============================================================================


@pytest.fixture
def sample_embeddings_with_metadata():
    """Generate sample embeddings with chunk metadata for v1.1+ format."""
    np.random.seed(42)
    embeddings = {
        "doc1-0001": np.random.rand(384).astype(np.float32),
        "doc1-0002": np.random.rand(384).astype(np.float32),
        "doc2-0001": np.random.rand(384).astype(np.float32),
        "doc2-0002": np.random.rand(384).astype(np.float32),
    }
    metadata = {
        "doc1-0001": {"doc_id": "doc1", "file_path": "doc1/chunks/ch_0001.md", "is_music": False},
        "doc1-0002": {"doc_id": "doc1", "file_path": "doc1/chunks/ch_0002.md", "is_music": False},
        "doc2-0001": {"doc_id": "doc2", "file_path": "doc2/chunks/ch_0001.md", "is_music": False},
        "doc2-0002": {"doc_id": "doc2", "file_path": "doc2/chunks/ch_0002.md", "is_music": False},
    }
    return embeddings, metadata


@pytest.fixture
def index_dir_v11(tmp_path, sample_embeddings_with_metadata):
    """Create an index with v1.1 format (with metadata)."""
    embeddings, metadata = sample_embeddings_with_metadata
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    write_vector_index(embeddings, index_dir, chunk_metadata=metadata)
    return index_dir


class TestAppendToVectorIndex:
    """Test append_to_vector_index() function."""

    def test_append_to_existing_index(self, index_dir_v11):
        """Test that vectors are correctly appended to existing index."""
        # Get initial state
        index_before, chunk_map_before = load_vector_index(index_dir_v11)
        initial_count = index_before.ntotal

        # Create new embeddings to append
        np.random.seed(100)
        new_embeddings = {
            "doc3-0001": np.random.rand(384).astype(np.float32),
            "doc3-0002": np.random.rand(384).astype(np.float32),
        }
        new_metadata = {
            "doc3-0001": {"doc_id": "doc3", "file_path": "doc3/chunks/ch_0001.md"},
            "doc3-0002": {"doc_id": "doc3", "file_path": "doc3/chunks/ch_0002.md"},
        }

        # Append
        count = append_to_vector_index(new_embeddings, index_dir_v11, new_metadata)

        # Verify
        assert count == 2, "Should report 2 vectors appended"

        index_after, chunk_map_after = load_vector_index(index_dir_v11)
        assert index_after.ntotal == initial_count + 2, "Index should have 2 more vectors"

    def test_append_chunk_map_extended(self, index_dir_v11):
        """Test that chunk map is extended with new entries while preserving existing."""
        index_before, chunk_map_before = load_vector_index(index_dir_v11)
        original_chunks = chunk_map_before.get("chunks", [])
        original_chunk_ids = {c["chunk_id"] for c in original_chunks}

        np.random.seed(101)
        new_embeddings = {
            "doc4-0001": np.random.rand(384).astype(np.float32),
        }
        new_metadata = {
            "doc4-0001": {"doc_id": "doc4", "file_path": "doc4/chunks/ch_0001.md"},
        }

        append_to_vector_index(new_embeddings, index_dir_v11, new_metadata)

        index_after, chunk_map_after = load_vector_index(index_dir_v11)
        new_chunks = chunk_map_after.get("chunks", [])
        new_chunk_ids = {c["chunk_id"] for c in new_chunks}

        # Original chunks should still be present
        assert original_chunk_ids.issubset(new_chunk_ids), "Original chunks should be preserved"
        # New chunk should be added
        assert "doc4-0001" in new_chunk_ids, "New chunk should be in chunk map"

    def test_append_updates_metadata(self, index_dir_v11):
        """Test that index_size and updated_utc are updated correctly."""
        _, chunk_map_before = load_vector_index(index_dir_v11)
        size_before = chunk_map_before.get("index_size", 0)

        np.random.seed(102)
        new_embeddings = {
            "doc5-0001": np.random.rand(384).astype(np.float32),
        }

        append_to_vector_index(new_embeddings, index_dir_v11)

        _, chunk_map_after = load_vector_index(index_dir_v11)

        assert chunk_map_after["index_size"] == size_before + 1
        assert chunk_map_after["format_version"] == FORMAT_VERSION_INCREMENTAL
        assert "updated_utc" in chunk_map_after

    def test_append_to_missing_index_raises_error(self, tmp_path):
        """Test that append to non-existent index raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        np.random.seed(103)
        new_embeddings = {
            "chunk1": np.random.rand(384).astype(np.float32),
        }

        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            append_to_vector_index(new_embeddings, empty_dir)

    def test_append_dimension_mismatch(self, index_dir_v11):
        """Test that dimension mismatch raises ValueError."""
        # Create embeddings with wrong dimension
        new_embeddings = {
            "chunk1": np.random.rand(512).astype(np.float32),  # Wrong dim
        }

        with pytest.raises(ValueError, match="doesn't match"):
            append_to_vector_index(new_embeddings, index_dir_v11)

    def test_append_empty_embeddings(self, index_dir_v11):
        """Test that appending empty dict is a no-op."""
        _, chunk_map_before = load_vector_index(index_dir_v11)
        size_before = chunk_map_before.get("index_size", 0)

        count = append_to_vector_index({}, index_dir_v11)

        assert count == 0
        _, chunk_map_after = load_vector_index(index_dir_v11)
        assert chunk_map_after.get("index_size", 0) == size_before

    def test_append_upgrades_v10_to_v12(self, tmp_path, sample_embeddings):
        """Test that appending to v1.0 chunk map upgrades to v1.2."""
        index_dir = tmp_path / "v10_index"
        index_dir.mkdir()

        # Create v1.0 index (no metadata)
        write_vector_index(sample_embeddings, index_dir)

        _, chunk_map_before = load_vector_index(index_dir)
        assert chunk_map_before["format_version"] == "1.0"

        # Append new embeddings
        np.random.seed(104)
        new_embeddings = {
            "new-chunk": np.random.rand(384).astype(np.float32),
        }

        append_to_vector_index(new_embeddings, index_dir)

        _, chunk_map_after = load_vector_index(index_dir)
        assert chunk_map_after["format_version"] == FORMAT_VERSION_INCREMENTAL
        assert "chunks" in chunk_map_after


class TestTombstoneDocuments:
    """Test tombstone_documents() function."""

    def test_tombstone_marks_deleted_utc(self, index_dir_v11):
        """Test that tombstoning sets deleted_utc on chunks."""
        count = tombstone_documents(["doc1"], index_dir_v11)

        assert count == 2, "Should tombstone 2 chunks for doc1"

        _, chunk_map = load_vector_index(index_dir_v11)
        chunks = chunk_map.get("chunks", [])

        for chunk in chunks:
            if chunk.get("doc_id") == "doc1":
                assert chunk.get("deleted_utc") is not None
            else:
                assert chunk.get("deleted_utc") is None

    def test_tombstone_updates_count(self, index_dir_v11):
        """Test that tombstone_count metadata is updated."""
        _, chunk_map_before = load_vector_index(index_dir_v11)
        assert chunk_map_before.get("tombstone_count", 0) == 0

        tombstone_documents(["doc1"], index_dir_v11)

        _, chunk_map_after = load_vector_index(index_dir_v11)
        assert chunk_map_after.get("tombstone_count", 0) == 2

    def test_tombstone_preserves_vectors(self, index_dir_v11):
        """Test that FAISS index is NOT modified by tombstoning."""
        index_before, _ = load_vector_index(index_dir_v11)
        count_before = index_before.ntotal

        tombstone_documents(["doc1"], index_dir_v11)

        index_after, _ = load_vector_index(index_dir_v11)
        assert index_after.ntotal == count_before, "FAISS index should not change"

    def test_tombstone_v10_raises_error(self, tmp_path, sample_embeddings):
        """Test that tombstoning v1.0 format raises ValueError."""
        index_dir = tmp_path / "v10_index"
        index_dir.mkdir()
        write_vector_index(sample_embeddings, index_dir)  # v1.0 format

        with pytest.raises(ValueError, match="Cannot tombstone"):
            tombstone_documents(["doc1"], index_dir)

    def test_tombstone_nonexistent_doc(self, index_dir_v11):
        """Test tombstoning non-existent doc_id."""
        count = tombstone_documents(["nonexistent"], index_dir_v11)
        assert count == 0

    def test_tombstone_multiple_docs(self, index_dir_v11):
        """Test tombstoning multiple documents at once."""
        count = tombstone_documents(["doc1", "doc2"], index_dir_v11)
        assert count == 4  # 2 chunks each

        _, chunk_map = load_vector_index(index_dir_v11)
        assert chunk_map.get("tombstone_count", 0) == 4


class TestSearchWithTombstones:
    """Test search_similar_chunks() with tombstone filtering."""

    def test_search_excludes_tombstoned(self, index_dir_v11, sample_embeddings_with_metadata):
        """Test that tombstoned chunks are excluded from search results."""
        embeddings, _ = sample_embeddings_with_metadata

        # Tombstone doc1
        tombstone_documents(["doc1"], index_dir_v11)

        index, chunk_map = load_vector_index(index_dir_v11)

        # Search with doc1's embedding - should NOT find doc1 chunks
        query = embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=10)

        result_chunk_ids = [r[0] for r in results]
        assert "doc1-0001" not in result_chunk_ids
        assert "doc1-0002" not in result_chunk_ids
        # Should find doc2 chunks
        assert len(results) == 2

    def test_search_adjusts_for_tombstones(self, index_dir_v11, sample_embeddings_with_metadata):
        """Test that search fetches extra to compensate for tombstones."""
        embeddings, _ = sample_embeddings_with_metadata

        # Tombstone doc1
        tombstone_documents(["doc1"], index_dir_v11)

        index, chunk_map = load_vector_index(index_dir_v11)

        # Request top 2 - should get 2 results from doc2
        query = embeddings["doc2-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=2)

        assert len(results) == 2
        # First result should be exact match
        assert results[0][0] == "doc2-0001"

    def test_search_warns_high_tombstone_ratio(self, index_dir_v11, caplog):
        """Test that search logs warning when tombstone ratio is high."""
        import logging

        # Tombstone enough to exceed warning threshold
        # We have 4 chunks, tombstoning 1 doc (2 chunks) gives 50% ratio
        tombstone_documents(["doc1"], index_dir_v11)

        index, chunk_map = load_vector_index(index_dir_v11)

        query = np.random.rand(384).astype(np.float32)

        with caplog.at_level(logging.WARNING, logger="grounding.vector_store"):
            search_similar_chunks(index, chunk_map, query, top_k=5)

        assert "tombstone ratio" in caplog.text.lower()


class TestGetIndexedDocIds:
    """Test get_indexed_doc_ids() function."""

    def test_get_doc_ids_from_v11(self, index_dir_v11):
        """Test extracting doc_ids from v1.1 chunk map."""
        _, chunk_map = load_vector_index(index_dir_v11)
        doc_ids = get_indexed_doc_ids(chunk_map)

        assert doc_ids == {"doc1", "doc2"}

    def test_get_doc_ids_excludes_tombstoned(self, index_dir_v11):
        """Test that tombstoned doc_ids are excluded by default."""
        tombstone_documents(["doc1"], index_dir_v11)

        _, chunk_map = load_vector_index(index_dir_v11)
        doc_ids = get_indexed_doc_ids(chunk_map, include_tombstoned=False)

        assert doc_ids == {"doc2"}

    def test_get_doc_ids_includes_tombstoned(self, index_dir_v11):
        """Test that tombstoned doc_ids can be included."""
        tombstone_documents(["doc1"], index_dir_v11)

        _, chunk_map = load_vector_index(index_dir_v11)
        doc_ids = get_indexed_doc_ids(chunk_map, include_tombstoned=True)

        assert doc_ids == {"doc1", "doc2"}

    def test_get_doc_ids_v10_returns_empty(self, tmp_path, sample_embeddings):
        """Test that v1.0 format returns empty set (no doc_id info)."""
        index_dir = tmp_path / "v10_index"
        index_dir.mkdir()
        write_vector_index(sample_embeddings, index_dir)

        _, chunk_map = load_vector_index(index_dir)
        doc_ids = get_indexed_doc_ids(chunk_map)

        assert doc_ids == set()


class TestShouldRebuildIndex:
    """Test should_rebuild_index() function."""

    def test_should_rebuild_below_threshold(self, index_dir_v11):
        """Test that rebuild not recommended when below threshold."""
        should_rebuild, reason = should_rebuild_index(index_dir_v11)

        assert should_rebuild is False
        assert "healthy" in reason.lower() or "no rebuild" in reason.lower()

    def test_should_rebuild_above_threshold(self, index_dir_v11):
        """Test that rebuild recommended when above threshold."""
        # Tombstone enough to exceed 30% threshold
        # 4 chunks total, tombstone all of doc1 (2 chunks) = 50%
        tombstone_documents(["doc1"], index_dir_v11)

        should_rebuild, reason = should_rebuild_index(index_dir_v11)

        assert should_rebuild is True
        assert "exceeds threshold" in reason.lower() or "rebuild" in reason.lower()

    def test_should_rebuild_missing_chunk_map(self, tmp_path):
        """Test error when chunk map is missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            should_rebuild_index(empty_dir)


class TestStalenessReport:
    """Test StalenessReport dataclass."""

    def test_staleness_report_fields(self):
        """Test that StalenessReport has all expected fields."""
        report = StalenessReport(
            new_docs=["doc1"],
            deleted_docs=["doc2"],
            updated_docs=["doc3"],
            is_stale=True,
            tombstone_ratio=0.15,
            should_rebuild=False,
        )

        assert report.new_docs == ["doc1"]
        assert report.deleted_docs == ["doc2"]
        assert report.updated_docs == ["doc3"]
        assert report.is_stale is True
        assert report.tombstone_ratio == 0.15
        assert report.should_rebuild is False


class TestBackwardCompatibility:
    """Test backward compatibility with older chunk map formats."""

    def test_backward_compat_v10_chunk_map(self, tmp_path, sample_embeddings):
        """Test that v1.0 format still works for basic operations."""
        index_dir = tmp_path / "v10"
        index_dir.mkdir()

        # Create v1.0 index
        write_vector_index(sample_embeddings, index_dir)

        # Load and search should work
        index, chunk_map = load_vector_index(index_dir)
        assert chunk_map["format_version"] == "1.0"

        query = sample_embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=3)
        assert len(results) == 3

    def test_backward_compat_v11_chunk_map(self, index_dir_v11, sample_embeddings_with_metadata):
        """Test that v1.1 format works with new functions."""
        embeddings, _ = sample_embeddings_with_metadata

        index, chunk_map = load_vector_index(index_dir_v11)
        assert chunk_map["format_version"] == FORMAT_VERSION_WITH_METADATA

        # Search should work
        query = embeddings["doc1-0001"]
        results = search_similar_chunks(index, chunk_map, query, top_k=3)
        assert len(results) == 3

        # get_indexed_doc_ids should work
        doc_ids = get_indexed_doc_ids(chunk_map)
        assert doc_ids == {"doc1", "doc2"}


class TestDocumentUpdateFlow:
    """Test the document update workflow (tombstone old + append new)."""

    def test_update_document_flow(self, index_dir_v11, sample_embeddings_with_metadata):
        """Test full document update: tombstone old, append new."""
        embeddings, _ = sample_embeddings_with_metadata

        # Step 1: Tombstone the old document
        tombstone_count = tombstone_documents(["doc1"], index_dir_v11)
        assert tombstone_count == 2

        # Step 2: Append new version of the document
        np.random.seed(200)
        new_embeddings = {
            "doc1-0001-v2": np.random.rand(384).astype(np.float32),
            "doc1-0002-v2": np.random.rand(384).astype(np.float32),
        }
        new_metadata = {
            "doc1-0001-v2": {"doc_id": "doc1_v2", "file_path": "doc1/chunks/ch_0001.md"},
            "doc1-0002-v2": {"doc_id": "doc1_v2", "file_path": "doc1/chunks/ch_0002.md"},
        }

        append_count = append_to_vector_index(new_embeddings, index_dir_v11, new_metadata)
        assert append_count == 2

        # Step 3: Verify search finds new version, not old
        index, chunk_map = load_vector_index(index_dir_v11)

        query = new_embeddings["doc1-0001-v2"]
        results = search_similar_chunks(index, chunk_map, query, top_k=1)

        assert results[0][0] == "doc1-0001-v2"

        # Old doc1 chunks should not appear in search
        old_query = embeddings["doc1-0001"]
        old_results = search_similar_chunks(index, chunk_map, old_query, top_k=10)
        result_ids = [r[0] for r in old_results]

        assert "doc1-0001" not in result_ids
        assert "doc1-0002" not in result_ids
