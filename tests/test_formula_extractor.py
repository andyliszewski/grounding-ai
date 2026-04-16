"""Tests for grounding.formula_extractor."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure pix2tex dependency is stubbed if not installed
if "pix2tex.cli" not in sys.modules:
    pix2tex_module = ModuleType("pix2tex")
    cli_module = ModuleType("pix2tex.cli")

    class _LatexOCRStub:
        """Stub for LatexOCR."""

        def __call__(self, image: Any) -> str:
            return "E=mc^2"

    cli_module.LatexOCR = _LatexOCRStub  # type: ignore[attr-defined]
    pix2tex_module.cli = cli_module  # type: ignore[attr-defined]

    sys.modules["pix2tex"] = pix2tex_module
    sys.modules["pix2tex.cli"] = cli_module

# Now import our module
from grounding.formula_extractor import (
    FormulaElement,
    FormulaExtractionError,
    _init_pix2tex_model,
    detect_formula_regions,
    extract_formulas,
)


@pytest.fixture(autouse=True)
def reset_pix2tex_model():
    """Reset pix2tex model singleton before each test."""
    import grounding.formula_extractor as fe

    fe._pix2tex_model = None
    yield
    fe._pix2tex_model = None


def test_formula_element_creation():
    """Test FormulaElement dataclass creation."""
    formula = FormulaElement(
        formula_type="display",
        latex_str=r"\frac{1}{2}",
        page_num=0,
        bbox=(10.0, 20.0, 100.0, 50.0),
        confidence=0.95,
        metadata={"complexity": "simple"},
    )

    assert formula.formula_type == "display"
    assert formula.latex_str == r"\frac{1}{2}"
    assert formula.page_num == 0
    assert formula.bbox == (10.0, 20.0, 100.0, 50.0)
    assert formula.confidence == 0.95
    assert formula.metadata["complexity"] == "simple"


def test_formula_element_str():
    """Test FormulaElement __str__ method."""
    formula = FormulaElement(
        formula_type="inline",
        latex_str="E=mc^2",
        page_num=5,
        bbox=(100.0, 200.0, 150.0, 220.0),
    )

    str_repr = str(formula)
    assert "FormulaElement" in str_repr
    assert "type=inline" in str_repr
    assert "page=5" in str_repr


def test_formula_element_default_metadata():
    """Test FormulaElement with default metadata."""
    formula = FormulaElement(
        formula_type="display",
        latex_str=r"\int_0^1 x dx",
        page_num=0,
        bbox=(0.0, 0.0, 100.0, 50.0),
    )

    assert formula.metadata == {}
    assert formula.confidence is None


def test_init_pix2tex_model_success():
    """Test successful pix2tex model initialization."""
    with patch("pix2tex.cli.LatexOCR") as mock_ocr:
        mock_instance = MagicMock()
        mock_ocr.return_value = mock_instance

        model = _init_pix2tex_model()

        assert model is mock_instance
        mock_ocr.assert_called_once()


def test_init_pix2tex_model_not_installed():
    """Test pix2tex model initialization when pix2tex is not installed."""
    import grounding.formula_extractor as fe

    fe._pix2tex_model = None

    # Simulate ModuleNotFoundError when importing pix2tex
    with patch("pix2tex.cli.LatexOCR") as mock_ocr:
        mock_ocr.side_effect = ModuleNotFoundError("No module named 'pix2tex'")

        with pytest.raises(FormulaExtractionError) as exc_info:
            _init_pix2tex_model()

        # The error is caught in the try-except and wrapped
        assert "Failed to initialize pix2tex model" in str(exc_info.value)


def test_init_pix2tex_model_singleton():
    """Test that pix2tex model is initialized only once (singleton pattern)."""
    with patch("pix2tex.cli.LatexOCR") as mock_ocr:
        mock_instance = MagicMock()
        mock_ocr.return_value = mock_instance

        model1 = _init_pix2tex_model()
        model2 = _init_pix2tex_model()

        assert model1 is model2
        mock_ocr.assert_called_once()  # Only initialized once


def test_detect_formula_regions_success():
    """Test formula region detection on a synthetic page image."""
    # Create a synthetic image with a dark rectangle (simulating formula)
    img = Image.new("L", (800, 1000), color=255)  # White background
    pixels = img.load()

    # Draw a dark rectangle (formula-like region)
    for x in range(300, 500):
        for y in range(400, 450):
            pixels[x, y] = 0  # Black

    bboxes = detect_formula_regions(img, page_num=0)

    # Should detect at least one region
    assert len(bboxes) >= 1

    # Check that bbox is in reasonable range
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 <= 800
        assert 0 <= y1 < y2 <= 1000


def test_detect_formula_regions_empty_page():
    """Test formula detection on empty white page."""
    img = Image.new("L", (800, 1000), color=255)  # White background only

    bboxes = detect_formula_regions(img, page_num=0)

    # Should not detect any formulas
    assert len(bboxes) == 0


def test_extract_formulas_invalid_path_type():
    """Test extract_formulas with invalid path type."""
    with pytest.raises(TypeError) as exc_info:
        extract_formulas("not_a_path")  # type: ignore[arg-type]

    assert "must be a pathlib.Path" in str(exc_info.value)


def test_extract_formulas_file_not_found():
    """Test extract_formulas with non-existent file."""
    fake_path = Path("/nonexistent/file.pdf")

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_formulas(fake_path)

    assert "PDF not found" in str(exc_info.value)


def test_extract_formulas_directory_error(tmp_path: Path):
    """Test extract_formulas with directory instead of file."""
    with pytest.raises(IsADirectoryError) as exc_info:
        extract_formulas(tmp_path)

    assert "Expected file but received directory" in str(exc_info.value)


def test_extract_formulas_pix2tex_not_available(tmp_path: Path):
    """Test extract_formulas when pix2tex is not available."""
    # Create a dummy PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

    with patch(
        "grounding.formula_extractor._init_pix2tex_model",
        side_effect=FormulaExtractionError("pix2tex not installed"),
    ):
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(pdf_file)

        assert "pix2tex not installed" in str(exc_info.value)


def test_extract_formulas_success_with_mocks(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Test successful formula extraction with mocked pix2tex and PDF."""
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.INFO, logger="grounding.formula_extractor")

    # Create a dummy PDF file
    pdf_file = tmp_path / "equations.pdf"

    # Mock pypdfium2
    mock_page = MagicMock()
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000), color=(255, 255, 255))

    mock_page.render.return_value = mock_bitmap

    mock_pdf_doc = MagicMock()
    mock_pdf_doc.__len__.return_value = 1
    mock_pdf_doc.__getitem__.return_value = mock_page

    # Mock pix2tex model
    mock_model = MagicMock()
    mock_model.return_value = r"E=mc^2"

    with patch("grounding.formula_extractor.pdfium.PdfDocument", return_value=mock_pdf_doc):
        with patch("grounding.formula_extractor._init_pix2tex_model", return_value=mock_model):
            with patch("grounding.formula_extractor.detect_formula_regions") as mock_detect:
                # Simulate detecting one formula region
                mock_detect.return_value = [(100.0, 200.0, 300.0, 250.0)]

                # Create fake file
                pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

                formulas = extract_formulas(pdf_file)

                # Should extract one formula
                assert len(formulas) == 1
                assert formulas[0].latex_str == r"E=mc^2"
                assert formulas[0].page_num == 0
                assert formulas[0].bbox == (100.0, 200.0, 300.0, 250.0)

                # Check logging
                assert "Starting formula extraction" in caplog.text
                assert "Extracted 1 formulas" in caplog.text


def test_extract_formulas_no_formulas_found(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Test extraction when no formulas are detected."""
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.INFO, logger="grounding.formula_extractor")

    pdf_file = tmp_path / "text_only.pdf"

    # Mock pypdfium2
    mock_page = MagicMock()
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000), color=(255, 255, 255))

    mock_page.render.return_value = mock_bitmap

    mock_pdf_doc = MagicMock()
    mock_pdf_doc.__len__.return_value = 1
    mock_pdf_doc.__getitem__.return_value = mock_page

    mock_model = MagicMock()

    with patch("grounding.formula_extractor.pdfium.PdfDocument", return_value=mock_pdf_doc):
        with patch("grounding.formula_extractor._init_pix2tex_model", return_value=mock_model):
            with patch("grounding.formula_extractor.detect_formula_regions", return_value=[]):
                pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

                formulas = extract_formulas(pdf_file)

                # Should return empty list
                assert formulas == []
                assert "Extracted 0 formulas" in caplog.text


def test_extract_formulas_handles_extraction_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Test that extraction continues when individual formulas fail."""
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.WARNING, logger="grounding.formula_extractor")

    pdf_file = tmp_path / "mixed.pdf"

    # Mock pypdfium2
    mock_page = MagicMock()
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000), color=(255, 255, 255))

    mock_page.render.return_value = mock_bitmap

    mock_pdf_doc = MagicMock()
    mock_pdf_doc.__len__.return_value = 1
    mock_pdf_doc.__getitem__.return_value = mock_page

    # Mock pix2tex to fail on first call, succeed on second
    mock_model = MagicMock()
    mock_model.side_effect = [
        Exception("Model inference failed"),  # First formula fails
        r"\alpha + \beta",  # Second formula succeeds
    ]

    with patch("grounding.formula_extractor.pdfium.PdfDocument", return_value=mock_pdf_doc):
        with patch("grounding.formula_extractor._init_pix2tex_model", return_value=mock_model):
            with patch("grounding.formula_extractor.detect_formula_regions") as mock_detect:
                # Two formula regions
                mock_detect.return_value = [
                    (100.0, 200.0, 200.0, 250.0),
                    (300.0, 400.0, 500.0, 450.0),
                ]

                pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

                formulas = extract_formulas(pdf_file)

                # Should extract one formula (second one succeeded)
                assert len(formulas) == 1
                assert formulas[0].latex_str == r"\alpha + \beta"

                # Check that error was logged
                assert "pix2tex extraction failed" in caplog.text


def test_extract_formulas_formula_type_detection(tmp_path: Path):
    """Test that formula type (inline vs display) is correctly determined."""
    pdf_file = tmp_path / "formulas.pdf"

    mock_page = MagicMock()
    mock_bitmap = MagicMock()
    mock_bitmap.to_pil.return_value = Image.new("RGB", (800, 1000), color=(255, 255, 255))

    mock_page.render.return_value = mock_bitmap

    mock_pdf_doc = MagicMock()
    mock_pdf_doc.__len__.return_value = 1
    mock_pdf_doc.__getitem__.return_value = mock_page

    mock_model = MagicMock()
    mock_model.side_effect = [r"E=mc^2", r"\int_0^1 f(x) dx"]

    with patch("grounding.formula_extractor.pdfium.PdfDocument", return_value=mock_pdf_doc):
        with patch("grounding.formula_extractor._init_pix2tex_model", return_value=mock_model):
            with patch("grounding.formula_extractor.detect_formula_regions") as mock_detect:
                # One centered wide formula (display), one narrow formula (inline)
                # Page width is 800
                # Formula 1: x1=250, x2=550, center=400 (centered), width=300 (37.5% of page)
                # Formula 2: x1=50, x2=100, center=75 (left-aligned), width=50 (6.25% of page)
                mock_detect.return_value = [
                    (250.0, 200.0, 550.0, 250.0),  # Centered, wide -> display
                    (50.0, 400.0, 100.0, 420.0),  # Left-aligned, narrow -> inline
                ]

                pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

                formulas = extract_formulas(pdf_file)

                assert len(formulas) == 2
                # First formula should be display (centered and wide)
                assert formulas[0].formula_type == "display"
                # Second formula should be inline (not centered)
                assert formulas[1].formula_type == "inline"
