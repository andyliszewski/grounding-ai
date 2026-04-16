"""PDF parser adapter for grounding.

Implements Epic 2 Story 2.2 with fast path for text-based PDFs.

Strategy:
- For OCR off/auto: Try pdftotext first (fast, works for text-based PDFs)
- Fall back to unstructured only when OCR is needed or pdftotext fails
"""
from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("grounding.parser")

OCR_MODE_MAP = {
    "auto": "auto",
    "on": "always",
    "off": "never",
}

# Minimum text yield (chars per MB) to consider pdftotext successful
MIN_TEXT_YIELD_PER_MB = 1000


class ParseError(Exception):
    """Raised when a PDF fails to parse."""

    def __init__(self, file_path: Path, message: str):
        self.file_path = file_path
        super().__init__(message)


@dataclass
class TextElement:
    """Simple text element for pdftotext output."""
    text: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


_partition_pdf = None
_partition_epub = None


def _get_partition_pdf():
    global _partition_pdf
    if _partition_pdf is None:
        try:
            module = importlib.import_module("unstructured.partition.pdf")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Unstructured dependency not installed. Install unstructured>=0.15.0."
            ) from exc
        _partition_pdf = module.partition_pdf
    return _partition_pdf


def _get_partition_epub():
    global _partition_epub
    if _partition_epub is None:
        try:
            module = importlib.import_module("unstructured.partition.epub")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Unstructured EPUB support not installed. Install unstructured>=0.15.0."
            ) from exc
        _partition_epub = module.partition_epub
    return _partition_epub


def _extract_epub_with_ebooklib(file_path: Path) -> Optional[List[TextElement]]:
    """
    Extract text from EPUB using ebooklib as fallback.

    Returns list of TextElements or None if ebooklib unavailable/fails.
    """
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError:
        logger.debug("ebooklib or beautifulsoup4 not installed")
        return None

    try:
        book = epub.read_epub(str(file_path), options={'ignore_ncx': True})
        elements = []

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n\n', strip=True)
                if text.strip():
                    elements.append(TextElement(text=text))

        return elements if elements else None
    except Exception as exc:
        logger.debug("ebooklib failed for %s: %s", file_path.name, exc)
        return None


def _extract_with_pdftotext(file_path: Path) -> Optional[str]:
    """
    Extract text using pdftotext (poppler-utils).

    Returns extracted text or None if pdftotext unavailable/fails.
    """
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        logger.debug("pdftotext not found in PATH")
        return None

    try:
        result = subprocess.run(
            [pdftotext, "-layout", str(file_path), "-"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        logger.debug("pdftotext returned no output for %s", file_path.name)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdftotext timed out for %s", file_path.name)
        return None
    except Exception as exc:
        logger.debug("pdftotext failed for %s: %s", file_path.name, exc)
        return None


def _has_sufficient_text(text: str, file_size_mb: float) -> bool:
    """Check if extracted text has sufficient yield for the file size."""
    if not text:
        return False
    chars_per_mb = len(text) / max(file_size_mb, 0.1)
    return chars_per_mb >= MIN_TEXT_YIELD_PER_MB


def parse_pdf(file_path: Path, ocr_mode: str = "auto") -> List[Any]:
    """
    Parse a PDF into structured elements.

    Uses fast pdftotext extraction for text-based PDFs, falls back to
    unstructured for OCR or when pdftotext yields insufficient text.

    Args:
        file_path: Path to the PDF file.
        ocr_mode: One of {"auto", "on", "off"} controlling OCR strategy.

    Returns:
        List of elements (TextElement for pdftotext, unstructured elements otherwise).

    Raises:
        TypeError: If file_path is not a Path.
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If ocr_mode is invalid.
        ParseError: If parsing fails.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path instance")

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Expected file but received directory: {file_path}")

    normalized_mode = ocr_mode.lower()
    if normalized_mode not in OCR_MODE_MAP:
        raise ValueError(
            f"Invalid ocr_mode '{ocr_mode}'. Expected one of {sorted(OCR_MODE_MAP)}."
        )

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    start = time.perf_counter()

    # Fast path: Try pdftotext first (unless OCR is forced on)
    if normalized_mode != "on":
        logger.info("Trying fast extraction (pdftotext) for %s (%.1f MB)", file_path.name, file_size_mb)
        text = _extract_with_pdftotext(file_path)

        if _has_sufficient_text(text, file_size_mb):
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Split into paragraphs for better chunking
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            elements = [TextElement(text=p) for p in paragraphs]
            logger.info(
                "Fast extraction succeeded: %s (%d elements, %d chars) elapsed_ms=%.2f",
                file_path.name,
                len(elements),
                len(text),
                elapsed_ms,
            )
            return elements
        else:
            # If OCR is off, skip scanned PDFs - leave in staging for review
            if normalized_mode == "off":
                logger.warning(
                    "Fast extraction insufficient for %s (likely scanned/image-based PDF). "
                    "Skipping - file left in staging for review. Use --ocr auto to process.",
                    file_path.name
                )
                raise ParseError(
                    file_path,
                    f"Scanned/image-based PDF requires OCR: {file_path.name}. "
                    "Left in staging for manual review."
                )
            logger.info(
                "Fast extraction insufficient for %s, falling back to unstructured",
                file_path.name
            )

    # Slow path: Use unstructured (for OCR or when pdftotext fails, only when ocr != off)
    ocr_strategy = OCR_MODE_MAP[normalized_mode]

    if file_size_mb > 10:
        estimated_minutes = int(file_size_mb * 2)
        logger.info(
            "Processing large PDF with unstructured: %s (%.1f MB). Estimated time: %d-%d minutes.",
            file_path.name,
            file_size_mb,
            estimated_minutes,
            estimated_minutes * 2
        )

    partition_pdf = _get_partition_pdf()
    logger.info("Starting unstructured extraction: %s (strategy=auto)", file_path.name)

    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy="auto",
            infer_table_structure=True,
            extract_images_in_pdf=False,
            languages=["eng"],
            ocr_strategy=ocr_strategy,
        )
    except Exception as exc:
        logger.error(
            "Failed to parse %s with ocr_mode=%s", file_path, ocr_mode, exc_info=True
        )
        raise ParseError(file_path, f"Failed to parse {file_path}: {exc}") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    elements_list = list(elements)
    logger.info(
        "Unstructured extraction complete: %s (%d elements) elapsed_ms=%.2f",
        file_path.name,
        len(elements_list),
        elapsed_ms,
    )
    return elements_list


def parse_epub(file_path: Path) -> List[Any]:
    """
    Parse an EPUB file into structured elements.

    Uses unstructured's partition_epub for text extraction.

    Args:
        file_path: Path to the EPUB file.

    Returns:
        List of elements from unstructured.

    Raises:
        TypeError: If file_path is not a Path.
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
        ParseError: If parsing fails.
    """
    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path instance")

    if not file_path.exists():
        raise FileNotFoundError(f"EPUB not found: {file_path}")

    if file_path.is_dir():
        raise IsADirectoryError(f"Expected file but received directory: {file_path}")

    start = time.perf_counter()
    partition_epub = _get_partition_epub()
    logger.info("Starting EPUB extraction: %s", file_path.name)

    elements_list = None
    try:
        elements = partition_epub(filename=str(file_path))
        elements_list = list(elements)
    except Exception as exc:
        logger.warning(
            "Unstructured EPUB parsing failed for %s: %s. Trying ebooklib fallback.",
            file_path.name, exc
        )
        # Try ebooklib fallback
        fallback_elements = _extract_epub_with_ebooklib(file_path)
        if fallback_elements:
            logger.info("ebooklib fallback succeeded for %s", file_path.name)
            elements_list = fallback_elements
        else:
            logger.error("Failed to parse EPUB %s (both unstructured and ebooklib failed)", file_path)
            raise ParseError(file_path, f"Failed to parse EPUB {file_path}: {exc}") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "EPUB extraction complete: %s (%d elements) elapsed_ms=%.2f",
        file_path.name,
        len(elements_list),
        elapsed_ms,
    )
    return elements_list
