"""Hashing utilities for grounding.

Provides helpers for generating deterministic document and chunk identifiers.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

logger = logging.getLogger("grounding.hashing")


def compute_sha1(data: Union[str, bytes]) -> str:
    """
    Compute SHA-1 digest for the given data.

    Args:
        data: Text (str) or raw bytes to hash.

    Returns:
        40-character lowercase hexadecimal SHA-1 string.

    Raises:
        TypeError: If data is not str or bytes.
    """
    data_bytes = _normalize_input(data)
    digest = hashlib.sha1()
    digest.update(data_bytes)
    return digest.hexdigest()


def short_doc_id(sha1_hex: str) -> str:
    """
    Compute 8 character doc ID from a full SHA-1 hex string.

    Args:
        sha1_hex: 40-character SHA-1 hex digest.

    Returns:
        First 8 characters of the digest.

    Raises:
        ValueError: If sha1_hex is not 40 characters.
    """
    if len(sha1_hex) != 40:
        raise ValueError("sha1_hex must be a 40-character SHA-1 digest")
    return sha1_hex[:8]


def hash_content(
    text: str,
    algorithm: Literal["blake3", "sha256"] = "blake3",
) -> str:
    """
    Hash arbitrary text content using the requested algorithm.

    Args:
        text: Content to hash.
        algorithm: 'blake3' (default) or 'sha256'.

    Returns:
        Hexadecimal digest string.

    Raises:
        ValueError: If algorithm is unsupported.
    """
    normalized = _normalize_input(text)

    if algorithm == "sha256":
        digest = hashlib.sha256(normalized)
        return digest.hexdigest()

    if algorithm == "blake3":
        blake3_hash = _get_blake3_hash()
        if blake3_hash is None:
            _log_blake3_warning_once()
            digest = hashlib.sha256(normalized)
            return digest.hexdigest()

        return blake3_hash(normalized).hexdigest()

    raise ValueError(f"Unsupported hash algorithm '{algorithm}'")


def hash_document(markdown: str) -> Dict[str, str]:
    """
    Compute both BLAKE3 (or SHA-256 fallback) and SHA-256 digests for Markdown documents.

    Args:
        markdown: Document Markdown string.

    Returns:
        Mapping with keys 'blake3' and 'sha256'.
    """
    content = _normalize_input(markdown)
    blake3_digest = hash_content(content.decode("utf-8"), algorithm="blake3")
    sha256_digest = hash_content(content.decode("utf-8"), algorithm="sha256")
    return {"blake3": blake3_digest, "sha256": sha256_digest}


def hash_chunk(chunk_text: str, *, skip_front_matter: bool = True) -> str:
    """
    Hash chunk content, excluding YAML front matter if present.

    Args:
        chunk_text: Chunk content that may include front matter.
        skip_front_matter: Strip YAML front matter before hashing (default: True).

    Returns:
        Hex digest string using preferred algorithm.
    """
    body = _strip_front_matter(chunk_text) if skip_front_matter else chunk_text
    return hash_content(body, algorithm="blake3")


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text

    lines = text.splitlines()
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        return text

    stripped = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return stripped


def _normalize_input(data: Union[str, bytes]) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, bytes):
        return data
    raise TypeError("input must be str or bytes")


@lru_cache(maxsize=1)
def _get_blake3_hash():
    try:
        from blake3 import blake3  # type: ignore

        return blake3
    except ModuleNotFoundError:
        return None


_BLK3_WARNED = False


def _log_blake3_warning_once() -> None:
    global _BLK3_WARNED
    if not _BLK3_WARNED:
        logger.warning("blake3 module not available; falling back to sha256 hashing")
        _BLK3_WARNED = True
