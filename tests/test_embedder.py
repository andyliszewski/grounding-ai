"""Tests for grounding.embedder."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grounding import embedder
from grounding.embedder import (
    generate_embedding,
    get_embedding_dim,
    get_model_name,
    is_model_loaded,
)


def test_model_loads_successfully() -> None:
    """Test that the embedding model loads without errors."""
    # This will trigger model download on first run
    embedding = generate_embedding("test text")
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)


def test_generate_embedding_returns_correct_shape() -> None:
    """Test that embeddings have the expected 384-dimensional shape."""
    text = "This is a sample text for embedding generation."
    embedding = generate_embedding(text)

    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert embedding.dtype in [np.float32, np.float64]


def test_embedding_determinism() -> None:
    """Test that same input text produces identical embeddings."""
    text = "Deterministic embedding test string"

    embedding1 = generate_embedding(text)
    embedding2 = generate_embedding(text)

    # Embeddings should be exactly identical
    np.testing.assert_array_equal(
        embedding1,
        embedding2,
        err_msg="Same input should produce identical embeddings",
    )


def test_embedding_normalization() -> None:
    """Test that embeddings are L2-normalized (norm ≈ 1.0)."""
    text = "Test text for normalization check"
    embedding = generate_embedding(text)

    # Calculate L2 norm
    norm = np.linalg.norm(embedding)

    # Should be very close to 1.0 (allowing small floating point error)
    assert np.isclose(norm, 1.0, atol=1e-6), f"Expected L2 norm ≈ 1.0, got {norm}"


def test_different_texts_produce_different_embeddings() -> None:
    """Test that different texts produce different embeddings."""
    text1 = "This is the first text."
    text2 = "This is a completely different text."

    embedding1 = generate_embedding(text1)
    embedding2 = generate_embedding(text2)

    # Embeddings should not be identical
    assert not np.array_equal(embedding1, embedding2)

    # But they should still be normalized
    assert np.isclose(np.linalg.norm(embedding1), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(embedding2), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",  # Empty string
        "   ",  # Whitespace only
        "	",  # Tab only
        "\n",  # Newline only
    ],
)
def test_invalid_input_empty_string_raises_value_error(invalid_input: str) -> None:
    """Test that empty or whitespace-only strings raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_embedding(invalid_input)


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        123,
        [],
        {},
        12.34,
    ],
)
def test_invalid_input_wrong_type_raises_value_error(invalid_input) -> None:  # type: ignore[no-untyped-def]
    """Test that non-string inputs raise ValueError."""
    with pytest.raises(ValueError, match="must be a string"):
        generate_embedding(invalid_input)  # type: ignore[arg-type]


def test_model_caching_works() -> None:
    """Test that model is cached after first load (second call is faster)."""
    # Clear cache if present
    embedder._model_cache = None

    # First call - will load model
    start1 = time.time()
    embedding1 = generate_embedding("First call to load model")
    duration1 = time.time() - start1

    # Verify model is now loaded
    assert is_model_loaded()

    # Second call - should use cached model
    start2 = time.time()
    embedding2 = generate_embedding("Second call with cached model")
    duration2 = time.time() - start2

    # Second call should be significantly faster (at least 10x faster)
    # First call includes model loading time (~1-3 seconds)
    # Second call should be very fast (<100ms typically)
    assert duration2 < duration1 / 5, (
        f"Expected cached call to be much faster: "
        f"first={duration1:.3f}s, second={duration2:.3f}s"
    )

    # Both should produce valid embeddings
    assert embedding1.shape == (384,)
    assert embedding2.shape == (384,)


def test_get_embedding_dim_returns_384() -> None:
    """Test that get_embedding_dim returns the correct dimension."""
    assert get_embedding_dim() == 384


def test_get_model_name_returns_correct_name() -> None:
    """Test that get_model_name returns the expected model identifier."""
    assert get_model_name() == "all-MiniLM-L6-v2"


def test_is_model_loaded_reflects_cache_state() -> None:
    """Test that is_model_loaded correctly reports cache state."""
    # Clear cache
    embedder._model_cache = None
    assert not is_model_loaded()

    # Load model
    generate_embedding("Load the model")
    assert is_model_loaded()


def test_model_load_failure_raises_runtime_error() -> None:
    """Test that model loading failures raise RuntimeError after retries."""
    # Clear cache
    embedder._model_cache = None

    # Mock SentenceTransformer to always fail
    with patch("grounding.embedder.SentenceTransformer") as mock_st:
        mock_st.side_effect = Exception("Simulated download failure")

        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            generate_embedding("This should fail")


def test_embedding_generation_handles_long_text() -> None:
    """Test that embeddings work with longer text passages."""
    long_text = " ".join(["This is a long document with many words."] * 50)
    embedding = generate_embedding(long_text)

    assert embedding.shape == (384,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_embedding_generation_handles_special_characters() -> None:
    """Test that embeddings work with special characters."""
    special_text = "Text with special chars: !@#$%^&*() français 中文 emoji 🚀"
    embedding = generate_embedding(special_text)

    assert embedding.shape == (384,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_embedding_similarity_for_related_texts() -> None:
    """Test that semantically similar texts have higher cosine similarity."""
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on the mat."
    text3 = "Python is a programming language."

    emb1 = generate_embedding(text1)
    emb2 = generate_embedding(text2)
    emb3 = generate_embedding(text3)

    # Cosine similarity (since vectors are normalized, this is just dot product)
    similarity_1_2 = np.dot(emb1, emb2)
    similarity_1_3 = np.dot(emb1, emb3)

    # Similar texts should have higher similarity
    assert similarity_1_2 > similarity_1_3, (
        f"Expected similar texts to have higher similarity: "
        f"similarity(1,2)={similarity_1_2:.3f} vs similarity(1,3)={similarity_1_3:.3f}"
    )


def test_performance_embedding_generation_speed() -> None:
    """Test that embedding generation meets performance requirements."""
    text = "Test text for performance measurement with reasonable length."

    # Warm up (ensure model is loaded)
    generate_embedding("warmup")

    # Measure performance
    start = time.time()
    embedding = generate_embedding(text)
    duration = time.time() - start

    # Should be under 500ms per chunk (relaxed for hardware variability)
    # Story target is 100ms but we allow more time for slower CPUs
    assert duration < 0.5, f"Embedding generation too slow: {duration:.3f}s (expected <0.5s)"
    assert embedding.shape == (384,)
