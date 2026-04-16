"""Tests for grounding.chunk_metadata."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yaml

from grounding.chunk_metadata import build_chunk_metadata, render_chunk
from grounding.hashing import hash_chunk


def test_build_chunk_metadata_creates_expected_chunk_id_and_timestamp(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG", logger="grounding.chunk_metadata")
    created = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    content = "Body content for hashing."
    chunk_hash = hash_chunk(content, skip_front_matter=False)

    metadata = build_chunk_metadata(
        doc_id="abc12345",
        source="sample.pdf",
        chunk_index=1,
        chunk_hash=chunk_hash,
        page_start=None,
        page_end=None,
        created_utc=created,
    )

    assert metadata.chunk_id == "abc12345-0001"
    assert metadata.created_utc == "2025-01-01T12:00:00+00:00"
    assert metadata.page_start is None
    assert metadata.page_end is None
    assert metadata.content_hash == chunk_hash
    assert any("missing page metadata" in message for message in caplog.messages)


def test_build_chunk_metadata_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        build_chunk_metadata(
            doc_id="abc12345",
            source="sample.pdf",
            chunk_index=0,
            chunk_hash="hash",
        )


def test_render_chunk_outputs_valid_yaml_front_matter() -> None:
    created = datetime(2025, 1, 2, 9, 30, tzinfo=timezone.utc)
    content = "# Heading\n\nBody text."
    chunk_hash = hash_chunk(content, skip_front_matter=False)

    metadata = build_chunk_metadata(
        doc_id="abc12345",
        source="sample.pdf",
        chunk_index=12,
        chunk_hash=chunk_hash,
        page_start=3,
        page_end=4,
        section_heading="Results",
        created_utc=created,
    )

    rendered = render_chunk(metadata, content)

    assert rendered.startswith("---\n")
    assert "\n---\n\n" in rendered
    front_matter, body = rendered.split("\n---\n\n", maxsplit=1)
    fm_lines = front_matter.splitlines()
    assert fm_lines[0] == "---"
    yaml_payload = yaml.safe_load("---\n" + "\n".join(fm_lines[1:]))
    assert yaml_payload == {
        "doc_id": "abc12345",
        "source": "sample.pdf",
        "chunk_id": "abc12345-0012",
        "page_start": 3,
        "page_end": 4,
        "hash": chunk_hash,
        "created_utc": "2025-01-02T09:30:00+00:00",
        "section_heading": "Results",
    }

    assert body == "# Heading\n\nBody text.\n"


def test_render_chunk_is_deterministic() -> None:
    created = datetime(2025, 1, 2, 9, 30, tzinfo=timezone.utc)
    metadata = build_chunk_metadata(
        doc_id="abc12345",
        source="sample.pdf",
        chunk_index=2,
        chunk_hash=hash_chunk("Example", skip_front_matter=False),
        page_start=1,
        page_end=2,
        created_utc=created,
    )

    content = "Example"
    first = render_chunk(metadata, content)
    second = render_chunk(metadata, content)

    assert first == second
