"""Tests for grounding.writer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytest

from grounding.pipeline import FileContext
from grounding.writer import write_document


def make_context(tmp_path: Path, slug: str = "sample") -> FileContext:
    output_dir = tmp_path / slug
    output_path = output_dir / "doc.md"
    return FileContext(
        source_path=tmp_path / f"{slug}.pdf",
        slug=slug,
        output_path=output_path,
    )


def test_write_document_writes_doc_and_chunks(tmp_path: Path) -> None:
    context = make_context(tmp_path, slug="report")
    markdown = "# Document\n\nBody.\n"
    chunks = ["---\nchunk: 1\n---\n\nChunk body 1\n", "---\nchunk: 2\n---\n\nChunk body 2\n"]

    write_document(context, markdown, chunks)

    doc_path = context.output_path
    assert doc_path is not None
    assert doc_path.exists()
    assert doc_path.read_text(encoding="utf-8") == markdown

    chunk_dir = doc_path.parent / "chunks"
    files = sorted(chunk_dir.glob("*.md"))
    assert [f.name for f in files] == ["ch_0001.md", "ch_0002.md"]
    assert files[0].read_text(encoding="utf-8") == chunks[0]


def test_write_document_handles_empty_chunks(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    markdown = "Content"

    write_document(context, markdown, [])

    doc_path = context.output_path
    assert doc_path is not None
    assert doc_path.exists()
    assert (doc_path.parent / "chunks").exists()
    assert not any((doc_path.parent / "chunks").iterdir())


def test_write_document_dry_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="grounding.writer")
    context = make_context(tmp_path)

    write_document(context, "Content", ["Chunk"], dry_run=True)

    doc_path = context.output_path
    assert doc_path is not None
    assert not doc_path.exists()
    assert "Dry-run" in "".join(caplog.messages)


def test_write_document_missing_output_path_raises(tmp_path: Path) -> None:
    context = FileContext(
        source_path=tmp_path / "input.pdf",
        slug="missing",
        output_path=None,
    )

    with pytest.raises(ValueError):
        write_document(context, "Content", [])


def test_write_document_overwrites_existing_file(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    doc_path = context.output_path
    assert doc_path is not None
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("Old", encoding="utf-8")

    write_document(context, "New", [])

    assert doc_path.read_text(encoding="utf-8") == "New"
