"""Embedding generation for grounding using sentence-transformers.

Implements Epic 6 Story 6.1 by integrating all-MiniLM-L6-v2 embedding model
for semantic vector generation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("grounding.embedder")

# Module-level model cache for singleton pattern
_model_cache: Optional[SentenceTransformer] = None

# Model configuration constants
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model() -> SentenceTransformer:
    """
    Load and cache the embedding model (singleton pattern).

    Downloads model on first call (~80MB) and caches for subsequent calls.
    Uses sentence-transformers default cache location:
    - Linux/Mac: ~/.cache/torch/sentence_transformers/
    - Windows: %USERPROFILE%\\.cache\\torch\\sentence_transformers\\

    Returns:
        SentenceTransformer: Loaded and ready-to-use model instance.

    Raises:
        RuntimeError: If model fails to load after retries.
    """
    global _model_cache

    if _model_cache is not None:
        logger.debug("Using cached embedding model")
        return _model_cache

    logger.info("Loading embedding model '%s' (first run may download ~80MB)", MODEL_NAME)

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            _model_cache = SentenceTransformer(MODEL_NAME)
            logger.info("Successfully loaded embedding model '%s'", MODEL_NAME)
            return _model_cache
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed to load model (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
                exc_info=True,
            )
            if attempt == max_retries:
                break

    error_msg = f"Failed to load embedding model after {max_retries} attempts: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate 384-dimensional embedding for input text.

    Produces deterministic, L2-normalized embeddings using all-MiniLM-L6-v2.
    Same input text always produces identical output vectors.

    Args:
        text: Input text to embed (non-empty string).

    Returns:
        np.ndarray: 384-dimensional L2-normalized embedding vector.

    Raises:
        ValueError: If text is empty or invalid type.
        RuntimeError: If model fails to load.

    Example:
        >>> embedding = generate_embedding("Example text for embedding")
        >>> embedding.shape
        (384,)
        >>> np.linalg.norm(embedding)  # Should be ~1.0
        1.0000001
    """
    # Validate input
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")

    if not text or not text.strip():
        raise ValueError("text cannot be empty or whitespace-only")

    # Load model (cached after first call)
    model = _get_model()

    # Generate embedding with normalization for determinism
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    logger.debug(
        "Generated embedding for text length=%d, embedding_shape=%s",
        len(text),
        embedding.shape,
    )

    return embedding


def get_embedding_dim() -> int:
    """
    Return embedding dimensionality for all-MiniLM-L6-v2.

    Returns:
        int: Embedding dimension (always 384 for this model).
    """
    return EMBEDDING_DIM


def get_model_name() -> str:
    """
    Return the embedding model identifier.

    Returns:
        str: HuggingFace model name.
    """
    return MODEL_NAME


def is_model_loaded() -> bool:
    """
    Check if embedding model is currently loaded in memory.

    Returns:
        bool: True if model is cached, False otherwise.
    """
    return _model_cache is not None
