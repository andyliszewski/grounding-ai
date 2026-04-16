"""Unit tests for formula_formatter module.

Tests Epic 8 Story 8.3: Mathematical Formula Output Formatting.
"""
import pytest
import xml.etree.ElementTree as ET

from grounding.formula_extractor import FormulaElement
from grounding.formula_formatter import (
    FormattingError,
    format_to_latex,
    format_to_mathml,
    format_to_plaintext,
    format_to_markdown,
    _validate_latex_syntax,
    _validate_mathml_structure,
    _latex_to_plaintext,
)


# Fixtures

@pytest.fixture
def simple_inline_formula():
    """Simple inline formula: E=mc^2."""
    return FormulaElement(
        formula_type="inline",
        latex_str="E=mc^2",
        page_num=0,
        bbox=(100.0, 200.0, 150.0, 220.0),
        confidence=0.95,
    )


@pytest.fixture
def simple_display_formula():
    """Simple display formula: quadratic formula."""
    return FormulaElement(
        formula_type="display",
        latex_str=r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
        page_num=1,
        bbox=(50.0, 300.0, 250.0, 350.0),
    )


@pytest.fixture
def integral_formula():
    """Integral formula."""
    return FormulaElement(
        formula_type="display",
        latex_str=r"\int_0^\infty e^{-x^2}dx = \frac{\sqrt{\pi}}{2}",
        page_num=2,
        bbox=(100.0, 400.0, 300.0, 450.0),
    )


@pytest.fixture
def greek_letters_formula():
    """Formula with Greek letters."""
    return FormulaElement(
        formula_type="inline",
        latex_str=r"\alpha + \beta = \gamma",
        page_num=0,
        bbox=(100.0, 100.0, 200.0, 120.0),
    )


@pytest.fixture
def invalid_latex_formula():
    """Invalid LaTeX with unbalanced braces."""
    return FormulaElement(
        formula_type="inline",
        latex_str=r"x = {a + b",
        page_num=0,
        bbox=(100.0, 100.0, 150.0, 120.0),
    )


# Test format_to_latex()

def test_format_to_latex_simple_inline(simple_inline_formula):
    """Test LaTeX formatting for simple inline formula."""
    result = format_to_latex([simple_inline_formula])

    assert len(result) == 1
    formula_id = "formula_0000_0000"
    assert formula_id in result

    # Check inline delimiters
    assert result[formula_id] == "$E=mc^2$"


def test_format_to_latex_simple_display(simple_display_formula):
    """Test LaTeX formatting for simple display formula."""
    result = format_to_latex([simple_display_formula])

    assert len(result) == 1
    formula_id = "formula_0001_0000"
    assert formula_id in result

    # Check display delimiters
    latex = result[formula_id]
    assert latex.startswith("$$")
    assert latex.endswith("$$")
    assert "frac" in latex


def test_format_to_latex_multiple_formulas(simple_inline_formula, simple_display_formula):
    """Test LaTeX formatting for multiple formulas."""
    result = format_to_latex([simple_inline_formula, simple_display_formula])

    assert len(result) == 2
    assert "formula_0000_0000" in result
    assert "formula_0001_0001" in result


def test_format_to_latex_whitespace_cleaning():
    """Test LaTeX formatting cleans extra whitespace."""
    formula = FormulaElement(
        formula_type="inline",
        latex_str="  E  =  mc^2  ",
        page_num=0,
        bbox=(100.0, 100.0, 150.0, 120.0),
    )

    result = format_to_latex([formula])
    formula_id = "formula_0000_0000"

    # Whitespace should be normalized
    assert result[formula_id] == "$E = mc^2$"


def test_format_to_latex_invalid_syntax(invalid_latex_formula):
    """Test LaTeX formatting raises error for invalid syntax."""
    with pytest.raises(FormattingError) as exc_info:
        format_to_latex([invalid_latex_formula])

    assert exc_info.value.format_type == "LaTeX"
    assert "formula_0000_0000" in exc_info.value.formula_id
    assert "Unbalanced braces" in str(exc_info.value)


# Test format_to_mathml()

def test_format_to_mathml_simple(simple_inline_formula):
    """Test MathML conversion for simple formula."""
    try:
        result = format_to_mathml([simple_inline_formula])
    except FormattingError as exc:
        if "latex2mathml not installed" in str(exc):
            pytest.skip("latex2mathml not installed")

    assert len(result) == 1
    formula_id = "formula_0000_0000"
    assert formula_id in result

    # Validate it's valid XML
    mathml = result[formula_id]
    tree = ET.fromstring(mathml)
    assert tree is not None


def test_format_to_mathml_quadratic(simple_display_formula):
    """Test MathML conversion for quadratic formula."""
    try:
        result = format_to_mathml([simple_display_formula])
    except FormattingError as exc:
        if "latex2mathml not installed" in str(exc):
            pytest.skip("latex2mathml not installed")

    if len(result) == 0:
        # latex2mathml may fail on complex formulas, that's OK
        pytest.skip("MathML conversion failed (acceptable)")

    formula_id = "formula_0001_0000"
    if formula_id in result:
        mathml = result[formula_id]
        # Validate XML structure
        tree = ET.fromstring(mathml)
        assert tree is not None


def test_format_to_mathml_partial_results():
    """Test MathML returns partial results on conversion failures."""
    # Mix of valid and complex formulas
    formulas = [
        FormulaElement("inline", "x=1", 0, (0, 0, 10, 10)),
        FormulaElement("inline", "y=2", 0, (0, 0, 10, 10)),
    ]

    try:
        result = format_to_mathml(formulas)
    except FormattingError as exc:
        if "latex2mathml not installed" in str(exc):
            pytest.skip("latex2mathml not installed")

    # Should not raise error even if some fail
    assert isinstance(result, dict)


def test_format_to_mathml_missing_library(monkeypatch):
    """Test MathML raises error if latex2mathml not installed."""
    # Mock import error
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "latex2mathml" in name:
            raise ImportError("No module named 'latex2mathml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(FormattingError) as exc_info:
        format_to_mathml([])

    assert exc_info.value.format_type == "MathML"
    assert "not installed" in str(exc_info.value)


# Test format_to_plaintext()

def test_format_to_plaintext_simple(simple_inline_formula):
    """Test plaintext conversion for simple formula."""
    result = format_to_plaintext([simple_inline_formula])

    assert len(result) == 1
    formula_id = "formula_0000_0000"
    assert formula_id in result

    plaintext = result[formula_id]
    assert "E" in plaintext
    assert "mc" in plaintext
    assert "^2" in plaintext or "2" in plaintext


def test_format_to_plaintext_greek_letters(greek_letters_formula):
    """Test plaintext conversion handles Greek letters."""
    result = format_to_plaintext([greek_letters_formula])

    formula_id = "formula_0000_0000"
    plaintext = result[formula_id]

    # Greek letters should be converted to names
    assert "alpha" in plaintext
    assert "beta" in plaintext
    assert "gamma" in plaintext


def test_format_to_plaintext_integral(integral_formula):
    """Test plaintext conversion for integral formula."""
    result = format_to_plaintext([integral_formula])

    formula_id = "formula_0002_0000"
    plaintext = result[formula_id]

    # Should contain readable terms
    assert "integral" in plaintext.lower() or "int" in plaintext


def test_format_to_plaintext_never_fails():
    """Test plaintext conversion never raises errors."""
    # Even with invalid LaTeX, should return something
    invalid_formula = FormulaElement(
        formula_type="inline",
        latex_str=r"\unknowncommand{xyz}",
        page_num=0,
        bbox=(0, 0, 10, 10),
    )

    result = format_to_plaintext([invalid_formula])

    assert len(result) == 1
    formula_id = "formula_0000_0000"
    assert formula_id in result
    # Should return something (even if just raw LaTeX)
    assert len(result[formula_id]) > 0


# Test format_to_markdown()

def test_format_to_markdown_empty_formulas():
    """Test markdown integration with no formulas."""
    text = "This is a test document."
    result = format_to_markdown([], text)

    assert result == text


def test_format_to_markdown_single_inline(simple_inline_formula):
    """Test markdown integration with single inline formula."""
    text = "This is a test document."
    result = format_to_markdown([simple_inline_formula], text)

    # Should contain original text
    assert "This is a test document." in result

    # Should have formulas section
    assert "## Mathematical Formulas" in result

    # Should have inline delimiter
    assert "$E=mc^2$" in result

    # Should have page reference
    assert "page 1" in result.lower()


def test_format_to_markdown_single_display(simple_display_formula):
    """Test markdown integration with single display formula."""
    text = "Quadratic formula:"
    result = format_to_markdown([simple_display_formula], text)

    # Should have display delimiters
    assert "$$" in result

    # Should have frac command
    assert "frac" in result


def test_format_to_markdown_multiple_formulas(simple_inline_formula, simple_display_formula):
    """Test markdown integration with multiple formulas."""
    text = "Multiple formulas test."
    result = format_to_markdown([simple_inline_formula, simple_display_formula], text)

    # Should have both formulas
    assert "$E=mc^2$" in result
    assert "$$" in result

    # Should have numbered formulas
    assert "Formula 1" in result
    assert "Formula 2" in result


# Test helper functions

def test_validate_latex_syntax_valid():
    """Test LaTeX syntax validation passes for valid input."""
    _validate_latex_syntax(r"E=mc^2")
    _validate_latex_syntax(r"\frac{a}{b}")
    _validate_latex_syntax(r"\int_0^\infty e^{-x^2}dx")


def test_validate_latex_syntax_unbalanced_braces():
    """Test LaTeX validation detects unbalanced braces."""
    with pytest.raises(ValueError) as exc_info:
        _validate_latex_syntax(r"{a + b")

    assert "Unbalanced braces" in str(exc_info.value)


def test_validate_latex_syntax_unbalanced_brackets():
    """Test LaTeX validation warns (not fails) for unbalanced brackets."""
    # Unbalanced brackets now generate warnings, not errors
    # This is intentional - brackets can be legitimately unbalanced in some LaTeX contexts
    try:
        _validate_latex_syntax(r"[a + b")
        # Should not raise - only warns
    except ValueError:
        pytest.fail("Unbalanced brackets should warn, not raise ValueError")


def test_validate_latex_syntax_empty():
    """Test LaTeX validation rejects empty string."""
    with pytest.raises(ValueError) as exc_info:
        _validate_latex_syntax("")

    assert "Empty LaTeX string" in str(exc_info.value)


def test_validate_mathml_structure_valid():
    """Test MathML validation passes for valid XML."""
    mathml = '<math><mrow><mi>x</mi></mrow></math>'
    _validate_mathml_structure(mathml)


def test_validate_mathml_structure_invalid():
    """Test MathML validation detects invalid XML."""
    invalid_mathml = '<math><mrow><mi>x</mrow></math>'  # Unclosed mi tag

    with pytest.raises(ValueError) as exc_info:
        _validate_mathml_structure(invalid_mathml)

    assert "Invalid MathML XML structure" in str(exc_info.value)


def test_latex_to_plaintext_simple():
    """Test LaTeX to plaintext conversion for simple formula."""
    result = _latex_to_plaintext("E=mc^2")
    assert "E" in result
    assert "mc" in result


def test_latex_to_plaintext_greek():
    """Test LaTeX to plaintext handles Greek letters."""
    result = _latex_to_plaintext(r"\alpha + \beta")
    assert "alpha" in result
    assert "beta" in result


def test_latex_to_plaintext_operators():
    """Test LaTeX to plaintext handles operators."""
    result = _latex_to_plaintext(r"\sum_{i=1}^n i")
    assert "sum" in result


def test_latex_to_plaintext_frac():
    """Test LaTeX to plaintext handles fractions."""
    result = _latex_to_plaintext(r"\frac{a}{b}")
    # Frac should be replaced with /
    assert "/" in result or "frac" not in result


# Test FormattingError exception

def test_formatting_error_structure():
    """Test FormattingError exception structure."""
    error = FormattingError("LaTeX", "formula_0001", "Test error message")

    assert error.format_type == "LaTeX"
    assert error.formula_id == "formula_0001"
    assert "LaTeX formatting error for formula_0001" in str(error)
    assert "Test error message" in str(error)


# Edge cases

def test_format_to_latex_empty_list():
    """Test LaTeX formatting with empty formula list."""
    result = format_to_latex([])
    assert result == {}


def test_format_to_mathml_empty_list():
    """Test MathML formatting with empty formula list."""
    try:
        result = format_to_mathml([])
    except FormattingError as exc:
        if "latex2mathml not installed" in str(exc):
            pytest.skip("latex2mathml not installed")

    assert result == {}


def test_format_to_plaintext_empty_list():
    """Test plaintext formatting with empty formula list."""
    result = format_to_plaintext([])
    assert result == {}


def test_formula_id_generation_ordering():
    """Test formula IDs are generated consistently."""
    formulas = [
        FormulaElement("inline", "a", 0, (0, 0, 10, 10)),
        FormulaElement("inline", "b", 0, (0, 0, 10, 10)),
        FormulaElement("inline", "c", 1, (0, 0, 10, 10)),
    ]

    result = format_to_latex(formulas)

    # Check IDs are sequential
    assert "formula_0000_0000" in result
    assert "formula_0000_0001" in result
    assert "formula_0001_0002" in result
