"""Tests for grounding.scanner."""
from pathlib import Path

import logging
import pytest

from grounding.scanner import scan_pdfs, SUPPORTED_EXTENSIONS


def test_scan_pdfs_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_pdfs(missing_dir)


def test_scan_pdfs_raises_for_file_input(tmp_path: Path) -> None:
    file_path = tmp_path / "document.pdf"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_pdfs(file_path)


def test_scan_pdfs_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_text("b", encoding="utf-8")
    (tmp_path / "A.PDF").write_text("a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")

    result = scan_pdfs(tmp_path)

    assert [path.name for path in result] == ["A.PDF", "b.pdf"]


def test_scan_pdfs_empty_directory_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.INFO, logger="grounding.scanner")

    result = scan_pdfs(tmp_path)

    assert result == []
    assert any("No documents found" in message for message in caplog.messages)


def test_scan_pdfs_rejects_non_path() -> None:
    with pytest.raises(TypeError):
        scan_pdfs("not-a-path")  # type: ignore[arg-type]


def test_supported_extensions_includes_epub() -> None:
    """Verify EPUB is in supported extensions."""
    assert ".epub" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_scan_pdfs_detects_epub_files(tmp_path: Path) -> None:
    """Verify scanner detects both PDF and EPUB files."""
    (tmp_path / "book.epub").write_text("epub", encoding="utf-8")
    (tmp_path / "doc.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")

    result = scan_pdfs(tmp_path)

    assert len(result) == 2
    names = [path.name for path in result]
    assert "book.epub" in names
    assert "doc.pdf" in names
    assert "notes.txt" not in names
