"""Document discovery utilities for grounding.

Implements the deterministic scanner described in Epic 2 Story 2.1.
Supports PDF and EPUB formats.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger("grounding.scanner")

# Supported document formats
SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".epub"}


def scan_pdfs(input_dir: Path) -> List[Path]:
    """
    Discover documents (PDF, EPUB) in the provided directory with deterministic ordering.

    Args:
        input_dir: Directory containing PDF or EPUB documents.

    Returns:
        Alphabetically sorted list of document file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        TypeError: If input_dir is not a Path instance.
    """
    if not isinstance(input_dir, Path):
        raise TypeError("input_dir must be a pathlib.Path instance")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    doc_files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    ]

    doc_files.sort(key=lambda path: path.name.casefold())

    if doc_files:
        pdf_count = sum(1 for f in doc_files if f.suffix.casefold() == ".pdf")
        epub_count = sum(1 for f in doc_files if f.suffix.casefold() == ".epub")
        logger.info(
            "Discovered %d document(s) in %s (%d PDF, %d EPUB)",
            len(doc_files), input_dir, pdf_count, epub_count
        )
    else:
        logger.info("No documents found in %s", input_dir)

    return doc_files
