"""Tests for grounding.meta."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
import logging

from grounding.meta import build_meta_yaml
from grounding.pipeline import FileContext


def make_context(tmp_path: Path) -> FileContext:
    return FileContext(
        source_path=tmp_path / "report.pdf",
        slug="report",
        output_path=tmp_path / "out" / "report" / "doc.md",
        sha1="filesha1",
        doc_sha1="docsha1",
        doc_id="deadbeef",
        doc_hashes={"blake3": "hashb3", "sha256": "hashsha"},
    )


def test_build_meta_yaml_includes_expected_fields(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    params: Dict[str, Any] = {"ocr_mode": "auto", "chunk_size": 1200}
    tooling_overrides = {"parser": "test-parser"}
    created = datetime(2025, 1, 2, 9, 30, tzinfo=timezone.utc)

    yaml_str = build_meta_yaml(
        context,
        params=params,
        tooling=tooling_overrides,
        generated_utc=created,
    )

    payload = yaml.safe_load(yaml_str)
    assert payload["doc_id"] == "deadbeef"
    assert payload["slug"] == "report"
    assert payload["orig_name"] == "report.pdf"
    assert payload["created_utc"] == "2025-01-02T09:30:00+00:00"
    assert payload["params"] == params
    assert payload["hashes"] == {
        "file_sha1": "filesha1",
        "doc_sha1": "docsha1",
        "blake3": "hashb3",
        "sha256": "hashsha",
    }
    assert payload["tooling"]["parser"] == "test-parser"


def test_build_meta_yaml_warns_when_hashes_missing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    context = FileContext(
        source_path=tmp_path / "nohash.pdf",
        slug="nohash",
        output_path=tmp_path / "out" / "nohash" / "doc.md",
        sha1=None,
        doc_sha1=None,
        doc_id="feedface",
        doc_hashes=None,
    )

    class _CaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    logger = logging.getLogger("grounding.meta")
    handler = _CaptureHandler()
    logger.addHandler(handler)
    try:
        yaml_str = build_meta_yaml(context, params={})
    finally:
        logger.removeHandler(handler)
    payload = yaml.safe_load(yaml_str)

    assert payload["hashes"]["file_sha1"] is None
    assert payload["hashes"]["doc_sha1"] is None
    assert any("Missing doc_hashes" in rec.getMessage() for rec in handler.records)


def test_build_meta_yaml_is_deterministic(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    params = {"ocr_mode": "auto"}

    first = build_meta_yaml(context, params=params)
    second = build_meta_yaml(context, params=params)

    assert first == second


def test_build_meta_yaml_with_collections(tmp_path: Path) -> None:
    """Collections appear in output."""
    context = make_context(tmp_path)
    params: Dict[str, Any] = {"ocr_mode": "auto"}
    collections = ["science", "biology", "reference"]

    yaml_str = build_meta_yaml(context, params=params, collections=collections)
    payload = yaml.safe_load(yaml_str)

    assert payload["collections"] == ["science", "biology", "reference"]


def test_build_meta_yaml_without_collections(tmp_path: Path) -> None:
    """Works without collections."""
    context = make_context(tmp_path)
    params: Dict[str, Any] = {"ocr_mode": "auto"}

    yaml_str = build_meta_yaml(context, params=params)
    payload = yaml.safe_load(yaml_str)

    assert "collections" not in payload


def test_build_meta_yaml_with_source_agent(tmp_path: Path) -> None:
    """Source agent appears in output."""
    context = make_context(tmp_path)
    params: Dict[str, Any] = {"ocr_mode": "auto"}

    yaml_str = build_meta_yaml(context, params=params, source_agent="scientist")
    payload = yaml.safe_load(yaml_str)

    assert payload["source_agent"] == "scientist"


def test_build_meta_yaml_filters_invalid_collections(tmp_path: Path) -> None:
    """Invalid collection names are filtered out."""
    context = make_context(tmp_path)
    params: Dict[str, Any] = {"ocr_mode": "auto"}
    collections = ["science", "Bad_Name", "biology", "Invalid Name"]

    yaml_str = build_meta_yaml(context, params=params, collections=collections)
    payload = yaml.safe_load(yaml_str)

    assert payload["collections"] == ["science", "biology"]
