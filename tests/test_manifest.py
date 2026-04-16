"""Tests for grounding.manifest."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from grounding.manifest import ManifestData, ManifestEntry, ManifestError, ManifestManager


def test_load_missing_manifest_initializes_new(tmp_path: Path) -> None:
    manifest_path = tmp_path / "_index.json"
    manifest = ManifestManager.load(manifest_path)

    assert manifest.created_utc is not None
    assert manifest.updated_utc is not None
    assert manifest.docs == []


def test_load_corrupt_manifest_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "_index.json"
    manifest_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ManifestError):
        ManifestManager.load(manifest_path)


def test_register_document_adds_and_replaces(tmp_path: Path) -> None:
    manifest = ManifestData(
        created_utc="2025-01-01T00:00:00+00:00",
        updated_utc="2025-01-01T00:00:00+00:00",
        docs=[
            ManifestEntry(doc_id="a", slug="alpha", orig_name="alpha.pdf"),
        ],
    )

    ManifestManager.register_document(
        manifest,
        ManifestEntry(doc_id="b", slug="beta", orig_name="beta.pdf"),
    )

    ManifestManager.register_document(
        manifest,
        ManifestEntry(doc_id="a", slug="alpha", orig_name="alpha_updated.pdf"),
    )

    assert [doc.doc_id for doc in manifest.docs] == ["a", "b"]
    assert manifest.docs[0].orig_name == "alpha_updated.pdf"


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "_index.json"
    manifest = ManifestData(
        created_utc="2025-01-01T00:00:00+00:00",
        updated_utc="2025-01-01T00:00:00+00:00",
        docs=[
            ManifestEntry(doc_id="b", slug="beta", orig_name="beta.pdf"),
            ManifestEntry(doc_id="a", slug="alpha", orig_name="alpha.pdf"),
        ],
    )

    ManifestManager.register_document(manifest, ManifestEntry(doc_id="c", slug="gamma", orig_name="gamma.pdf"))
    ManifestManager.write(manifest, manifest_path)

    reloaded = ManifestManager.load(manifest_path)
    assert [doc.doc_id for doc in reloaded.docs] == ["a", "b", "c"]
    assert reloaded.created_utc == manifest.created_utc
    assert reloaded.updated_utc >= manifest.updated_utc


def test_write_overwrites_atomically(tmp_path: Path) -> None:
    manifest_path = tmp_path / "_index.json"
    manifest_path.write_text(json.dumps({"created_utc": "x", "updated_utc": "y", "docs": []}), encoding="utf-8")

    manifest = ManifestManager.load(manifest_path)
    ManifestManager.register_document(manifest, ManifestEntry(doc_id="z", slug="zed", orig_name="zed.pdf"))
    ManifestManager.write(manifest, manifest_path)

    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert reloaded["docs"][0]["doc_id"] == "z"


def test_manifest_entry_with_collections() -> None:
    """Entry serializes collections correctly."""
    entry = ManifestEntry(
        doc_id="abc123",
        slug="test-doc",
        orig_name="test.pdf",
        collections=["science", "biology"],
        source_agent="scientist",
    )
    data = entry.to_dict()
    assert data["collections"] == ["science", "biology"]
    assert data["source_agent"] == "scientist"


def test_manifest_entry_without_collections() -> None:
    """Entry works without collections (backward compat)."""
    entry = ManifestEntry(
        doc_id="abc123",
        slug="test-doc",
        orig_name="test.pdf",
    )
    data = entry.to_dict()
    assert "collections" not in data
    assert "source_agent" not in data


def test_manifest_load_legacy_without_collections(tmp_path: Path) -> None:
    """Legacy manifest without collections loads correctly."""
    manifest_path = tmp_path / "_index.json"
    legacy_data = {
        "created_utc": "2025-01-01T00:00:00+00:00",
        "updated_utc": "2025-01-01T00:00:00+00:00",
        "docs": [
            {"doc_id": "a", "slug": "alpha", "orig_name": "alpha.pdf"},
            {"doc_id": "b", "slug": "beta", "orig_name": "beta.pdf", "pages": 10},
        ],
    }
    manifest_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    manifest = ManifestManager.load(manifest_path)
    assert len(manifest.docs) == 2
    assert manifest.docs[0].collections is None
    assert manifest.docs[0].source_agent is None
    assert manifest.docs[1].collections is None


def test_manifest_roundtrip_with_collections(tmp_path: Path) -> None:
    """Write/read preserves collections."""
    manifest_path = tmp_path / "_index.json"
    manifest = ManifestData(
        created_utc="2025-01-01T00:00:00+00:00",
        updated_utc="2025-01-01T00:00:00+00:00",
        docs=[],
    )

    entry = ManifestEntry(
        doc_id="c",
        slug="charlie",
        orig_name="charlie.pdf",
        collections=["reference", "science"],
        source_agent="researcher",
    )
    ManifestManager.register_document(manifest, entry)
    ManifestManager.write(manifest, manifest_path)

    reloaded = ManifestManager.load(manifest_path)
    assert len(reloaded.docs) == 1
    assert reloaded.docs[0].collections == ["reference", "science"]
    assert reloaded.docs[0].source_agent == "researcher"
