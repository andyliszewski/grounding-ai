"""Tests for grounding.hashing."""
from __future__ import annotations

from typing import List

import pytest

import grounding.hashing as hashing


def test_compute_sha1_string_matches_expected() -> None:
    digest = hashing.compute_sha1("hello world")
    assert digest == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_compute_sha1_bytes_matches_expected() -> None:
    digest = hashing.compute_sha1(b"hello world")
    assert digest == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_compute_sha1_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        hashing.compute_sha1(123)  # type: ignore[arg-type]


def test_short_doc_id_returns_first_eight_characters() -> None:
    sha1_hex = "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"
    doc_id = hashing.short_doc_id(sha1_hex)
    assert doc_id == "2aae6c35"


def test_short_doc_id_preserves_leading_zeros() -> None:
    sha1_hex = "00abcdef1234567890abcdef1234567890abcdef"
    doc_id = hashing.short_doc_id(sha1_hex)
    assert doc_id == "00abcdef"


def test_short_doc_id_rejects_invalid_length() -> None:
    with pytest.raises(ValueError):
        hashing.short_doc_id("short")


def test_hash_content_blake3_and_sha256(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = "sample text"

    blake3_digest = hashing.hash_content(sample, algorithm="blake3")
    sha256_digest = hashing.hash_content(sample, algorithm="sha256")

    assert len(blake3_digest) == 64
    assert len(sha256_digest) == 64
    assert blake3_digest != sha256_digest


def test_hash_content_blake3_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="grounding.hashing")

    monkeypatch.setattr("grounding.hashing._get_blake3_hash", lambda: None)
    global_state = getattr(hashing, "_BLK3_WARNED", False)
    hashing._BLK3_WARNED = False  # reset

    digest = hashing.hash_content("fallback", algorithm="blake3")
    assert len(digest) == 64
    assert any("falling back" in message for message in caplog.messages)

    # Subsequent call shouldn't log again.
    caplog.clear()
    hashing.hash_content("fallback", algorithm="blake3")
    assert not caplog.messages

    hashing._BLK3_WARNED = global_state


def test_hash_chunk_strips_front_matter() -> None:
    chunk = "---\ndoc_id: abc123\n---\n\nBody text here."
    digest_with_skip = hashing.hash_chunk(chunk)
    digest_without_skip = hashing.hash_chunk(chunk, skip_front_matter=False)

    assert digest_with_skip != digest_without_skip
    assert digest_with_skip == hashing.hash_chunk("Body text here.", skip_front_matter=False)


def test_hash_chunk_handles_missing_front_matter() -> None:
    chunk = "Body text only."
    digest = hashing.hash_chunk(chunk)
    assert digest == hashing.hash_chunk(chunk)


def test_strip_front_matter_handles_malformed_blocks() -> None:
    malformed = "---\nno closing marker"
    assert hashing._strip_front_matter(malformed) == malformed
