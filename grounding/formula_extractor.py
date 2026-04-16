"""Formula extraction module for grounding using pix2tex.

Implements Epic 8 Story 8.2 by integrating pix2tex for LaTeX formula extraction.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pypdfium2 as pdfium
import numpy as np
from PIL import Image

logger = logging.getLogger("grounding.formula_extractor")


class FormulaExtractionError(Exception):
    """Raised when formula extraction fails."""

    def __init__(self, message: str, file_path: Optional[Path] = None):
        self.file_path = file_path
        super().__init__(message)


@dataclass
class FormulaElement:
    """Represents a single mathematical formula extracted from a PDF.

    Attributes:
        formula_type: Either "inline" or "display" equation.
        latex_str: The extracted LaTeX string.
        page_num: Zero-indexed page number.
        bbox: Bounding box coordinates (x1, y1, x2, y2) in points.
        confidence: Optional confidence score from extractor (0.0-1.0).
        metadata: Additional metadata (complexity, domain hints, etc.).
    """

    formula_type: Literal["inline", "display"]
    latex_str: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"FormulaElement(type={self.formula_type}, "
            f"latex={self.latex_str[:50]}..., "
            f"page={self.page_num}, "
            f"bbox={self.bbox})"
        )


# Lazy loading pattern for pix2tex model
_pix2tex_model = None


def _init_pix2tex_model():
    """Initialize pix2tex LatexOCR model with lazy loading.

    Returns:
        LatexOCR: Initialized pix2tex model instance.

    Raises:
        FormulaExtractionError: If pix2tex is not installed.
    """
    global _pix2tex_model
    if _pix2tex_model is None:
        try:
            from pix2tex.cli import LatexOCR
        except ModuleNotFoundError as exc:
            raise FormulaExtractionError(
                "pix2tex not installed. Install with: pip install 'pix2tex[full]'"
            ) from exc

        logger.info("Initializing pix2tex model (first use may download ~100-200MB)...")
        try:
            _pix2tex_model = LatexOCR()
            logger.info("pix2tex model initialized successfully")
        except Exception as exc:
            raise FormulaExtractionError(
                f"Failed to initialize pix2tex model: {exc}. "
                "Check network connection or model files."
            ) from exc

    return _pix2tex_model


def detect_formula_regions(page_image: Image.Image, page_num: int) -> List[Tuple[float, float, float, float]]:
    """Detect regions likely to contain formulas using heuristics.

    Args:
        page_image: PIL Image of the PDF page.
        page_num: Page number for logging.

    Returns:
        List of bounding boxes (x1, y1, x2, y2) in pixel coordinates.
    """
    # Convert to grayscale and numpy array
    gray = page_image.convert("L")
    img_array = np.array(gray)

    # Simple threshold to get binary image
    threshold = 240  # White background assumption
    binary = img_array < threshold

    # Find connected components (contours)
    from scipy import ndimage

    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        logger.debug(f"No formula regions detected on page {page_num}")
        return []

    # Extract bounding boxes for each component
    bboxes = []
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)

        if not rows.any() or not cols.any():
            continue

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Filter heuristics:
        # 1. Minimum size (avoid noise)
        if area < 200:  # Skip tiny components
            continue

        # 2. Aspect ratio (formulas are often wider than tall)
        aspect_ratio = width / max(height, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 20:  # Skip very tall or very wide
            continue

        # 3. Position heuristics (centered formulas are display equations)
        page_width = page_image.width
        center_x = (x1 + x2) / 2
        is_centered = abs(center_x - page_width / 2) < page_width * 0.2

        formula_type = "display" if is_centered and width > page_width * 0.3 else "inline"

        bboxes.append((float(x1), float(y1), float(x2), float(y2)))

    logger.debug(f"Detected {len(bboxes)} potential formula regions on page {page_num}")
    return bboxes


def extract_formulas(pdf_path: Path) -> List[FormulaElement]:
    """Extract mathematical formulas from a PDF using pix2tex.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of FormulaElement instances, one per detected formula.

    Raises:
        TypeError: If pdf_path is not a Path.
        FileNotFoundError: If the PDF does not exist.
        FormulaExtractionError: If extraction fails critically.
    """
    if not isinstance(pdf_path, Path):
        raise TypeError("pdf_path must be a pathlib.Path instance")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.is_dir():
        raise IsADirectoryError(f"Expected file but received directory: {pdf_path}")

    # Initialize pix2tex model
    try:
        model = _init_pix2tex_model()
    except FormulaExtractionError:
        logger.error(f"Cannot extract formulas from {pdf_path}: pix2tex not available")
        raise

    start = time.perf_counter()
    formulas: List[FormulaElement] = []

    logger.info(f"Starting formula extraction: {pdf_path.name}")

    try:
        # Open PDF with pypdfium2
        pdf_doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise FormulaExtractionError(
            f"Failed to open PDF {pdf_path}: {exc}", file_path=pdf_path
        ) from exc

    try:
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # Convert page to image
            try:
                bitmap = page.render(scale=2)  # 2x scale for better quality
                img = bitmap.to_pil()
            except Exception as exc:
                logger.warning(
                    f"Failed to convert page {page_num} to image in {pdf_path.name}: {exc}"
                )
                continue

            # Detect formula regions
            try:
                bboxes = detect_formula_regions(img, page_num)
            except Exception as exc:
                logger.warning(
                    f"Formula detection failed on page {page_num} of {pdf_path.name}: {exc}"
                )
                continue

            # Extract each formula
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox

                # Crop formula region from image
                try:
                    formula_img = img.crop((int(x1), int(y1), int(x2), int(y2)))
                except Exception as exc:
                    logger.warning(
                        f"Failed to crop formula at {bbox} on page {page_num}: {exc}"
                    )
                    continue

                # Extract LaTeX with pix2tex
                try:
                    latex_str = model(formula_img)
                except Exception as exc:
                    logger.warning(
                        f"pix2tex extraction failed for formula at page {page_num}, "
                        f"bbox {bbox}: {exc}"
                    )
                    continue

                # Validate LaTeX output
                if not latex_str or len(latex_str.strip()) == 0:
                    logger.debug(f"Empty LaTeX output for formula at page {page_num}, bbox {bbox}")
                    continue

                # Determine formula type (inline vs display)
                page_width = img.width
                center_x = (x1 + x2) / 2
                width = x2 - x1
                is_centered = abs(center_x - page_width / 2) < page_width * 0.2
                formula_type: Literal["inline", "display"] = (
                    "display" if is_centered and width > page_width * 0.3 else "inline"
                )

                # Create FormulaElement
                formula = FormulaElement(
                    formula_type=formula_type,
                    latex_str=latex_str.strip(),
                    page_num=page_num,
                    bbox=bbox,
                    confidence=None,  # pix2tex doesn't provide confidence
                    metadata={"width": width, "height": y2 - y1},
                )
                formulas.append(formula)

    finally:
        pdf_doc.close()

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"Extracted {len(formulas)} formulas from {pdf_path.name} "
        f"(elapsed_ms={elapsed_ms:.2f})"
    )

    return formulas
