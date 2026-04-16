"""Formula formatting module for grounding.

Converts FormulaElement instances to multiple output formats:
- LaTeX (primary output)
- MathML (semantic representation)
- Plain text (accessibility fallback)
- Markdown integration

Implements Epic 8 Story 8.3.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from grounding.formula_extractor import FormulaElement

logger = logging.getLogger("grounding.formula_formatter")


class FormattingError(Exception):
    """Raised when formula formatting fails."""

    def __init__(self, format_type: str, formula_id: str, message: str):
        """Initialize FormattingError.

        Args:
            format_type: The output format that failed (e.g., "LaTeX", "MathML").
            formula_id: Identifier for the formula that failed.
            message: Error message with context.
        """
        self.format_type = format_type
        self.formula_id = formula_id
        super().__init__(f"{format_type} formatting error for {formula_id}: {message}")


def format_to_latex(formula_elements: List[FormulaElement]) -> Dict[str, str]:
    """Convert formula elements to LaTeX format.

    This is the primary output format. LaTeX strings are cleaned and validated.

    Args:
        formula_elements: List of FormulaElement instances.

    Returns:
        Dictionary mapping formula_id → LaTeX string.
        formula_id format: "formula_{page:04d}_{index:04d}"

    Raises:
        FormattingError: If LaTeX validation fails for a formula.
    """
    result = {}

    for idx, formula in enumerate(formula_elements):
        formula_id = f"formula_{formula.page_num:04d}_{idx:04d}"

        # Clean LaTeX string
        latex_str = formula.latex_str.strip()

        # Remove extra whitespace
        latex_str = re.sub(r'\s+', ' ', latex_str)

        # Validate LaTeX syntax
        try:
            _validate_latex_syntax(latex_str)
        except ValueError as exc:
            error_msg = f"Invalid LaTeX syntax: {exc}"
            logger.warning(f"{error_msg} for {formula_id}")
            raise FormattingError("LaTeX", formula_id, error_msg) from exc

        # Add delimiters based on formula type
        if formula.formula_type == "inline":
            formatted = f"${latex_str}$"
        else:  # display
            formatted = f"$${latex_str}$$"

        result[formula_id] = formatted
        logger.debug(f"Formatted LaTeX for {formula_id}: {formatted[:50]}...")

    logger.info(f"Formatted {len(result)} formulas to LaTeX")
    return result


def format_to_mathml(formula_elements: List[FormulaElement]) -> Dict[str, str]:
    """Convert formula elements to MathML format.

    Uses latex2mathml library for conversion. Provides semantic structure
    for accessibility and computational processing.

    Args:
        formula_elements: List of FormulaElement instances.

    Returns:
        Dictionary mapping formula_id → MathML string.
        Returns partial results if some conversions fail.

    Raises:
        FormattingError: If latex2mathml library is not available.
    """
    # Import latex2mathml
    try:
        from latex2mathml.converter import convert
    except ImportError as exc:
        raise FormattingError(
            "MathML",
            "N/A",
            "latex2mathml not installed. Install with: pip install latex2mathml"
        ) from exc

    result = {}

    for idx, formula in enumerate(formula_elements):
        formula_id = f"formula_{formula.page_num:04d}_{idx:04d}"
        latex_str = formula.latex_str.strip()

        try:
            # Convert LaTeX to MathML
            mathml_str = convert(latex_str)

            # Validate MathML structure
            _validate_mathml_structure(mathml_str)

            result[formula_id] = mathml_str
            logger.debug(f"Converted to MathML for {formula_id}")

        except Exception as exc:
            # Log error but continue with partial results
            logger.warning(
                f"MathML conversion failed for {formula_id}: {exc}. "
                "Skipping this formula."
            )
            continue

    logger.info(f"Formatted {len(result)}/{len(formula_elements)} formulas to MathML")
    return result


def format_to_plaintext(formula_elements: List[FormulaElement]) -> Dict[str, str]:
    """Convert formula elements to plain text format.

    Provides best-effort readable ASCII representation for accessibility.
    Complex notation loses structure; this is a fallback only.

    Args:
        formula_elements: List of FormulaElement instances.

    Returns:
        Dictionary mapping formula_id → plain text string.
        Never raises errors; returns best-effort conversion.
    """
    result = {}

    for idx, formula in enumerate(formula_elements):
        formula_id = f"formula_{formula.page_num:04d}_{idx:04d}"
        latex_str = formula.latex_str.strip()

        # Convert LaTeX to readable plain text
        try:
            plaintext = _latex_to_plaintext(latex_str)
            result[formula_id] = plaintext
            logger.debug(f"Converted to plaintext for {formula_id}: {plaintext[:50]}...")
        except Exception as exc:
            # Best-effort: if conversion fails, use raw LaTeX as fallback
            logger.warning(f"Plaintext conversion warning for {formula_id}: {exc}")
            result[formula_id] = latex_str

    logger.info(f"Formatted {len(result)} formulas to plaintext")
    return result


def format_to_markdown(
    formula_elements: List[FormulaElement],
    text: str
) -> str:
    """Integrate formulas into markdown text with proper delimiters.

    Embeds formulas at their original positions in the text using
    $ for inline and $$ for display equations.

    Args:
        formula_elements: List of FormulaElement instances.
        text: Original markdown text to embed formulas into.

    Returns:
        Markdown string with embedded formulas.

    Note:
        This function assumes formulas should be inserted at specific positions.
        For now, we append formulas at the end since position mapping is complex.
        Full integration requires position tracking from extraction.
    """
    if not formula_elements:
        return text

    # Start with original text
    result = text

    # Add formulas section at the end
    result += "\n\n## Mathematical Formulas\n\n"

    for idx, formula in enumerate(formula_elements):
        latex_str = formula.latex_str.strip()

        # Add formula with delimiters
        if formula.formula_type == "inline":
            result += f"Formula {idx + 1} (page {formula.page_num + 1}): ${latex_str}$\n\n"
        else:  # display
            result += f"Formula {idx + 1} (page {formula.page_num + 1}):\n\n$${latex_str}$$\n\n"

    logger.info(f"Integrated {len(formula_elements)} formulas into markdown")
    return result


# Helper functions

def _validate_latex_syntax(latex_str: str) -> None:
    """Validate LaTeX syntax for common errors.

    Args:
        latex_str: LaTeX string to validate.

    Raises:
        ValueError: If syntax errors are detected.
    """
    # Check balanced braces (critical for LaTeX)
    if latex_str.count('{') != latex_str.count('}'):
        raise ValueError("Unbalanced braces in LaTeX string")

    # Check balanced brackets (warning only - not critical)
    # Brackets can be legitimately unbalanced in some LaTeX contexts
    if latex_str.count('[') != latex_str.count(']'):
        logger.warning(
            f"Unbalanced brackets in LaTeX (not critical): "
            f"[{latex_str.count('[')}] vs ]{latex_str.count(']')}"
        )

    # Check for invalid commands (basic check)
    # Valid LaTeX commands start with backslash
    invalid_patterns = [
        r'\\[^a-zA-Z]',  # Backslash not followed by letter
    ]

    for pattern in invalid_patterns:
        if re.search(pattern, latex_str):
            # Allow some exceptions like \\ (line break)
            if not re.match(r'\\\\', latex_str):
                logger.debug(f"Potential invalid LaTeX pattern found: {pattern}")

    # Empty check
    if not latex_str or len(latex_str.strip()) == 0:
        raise ValueError("Empty LaTeX string")

    logger.debug(f"LaTeX syntax validation passed for: {latex_str[:50]}...")


def _validate_mathml_structure(mathml_str: str) -> None:
    """Validate MathML structure using basic XML parsing.

    Args:
        mathml_str: MathML XML string to validate.

    Raises:
        ValueError: If MathML structure is invalid.
    """
    import xml.etree.ElementTree as ET

    try:
        # Parse XML
        ET.fromstring(mathml_str)
        logger.debug("MathML structure validation passed")
    except ET.ParseError as exc:
        raise ValueError(f"Invalid MathML XML structure: {exc}") from exc


def _latex_to_plaintext(latex_str: str) -> str:
    """Convert LaTeX to readable plain text.

    Best-effort conversion with limited accuracy.

    Args:
        latex_str: LaTeX string to convert.

    Returns:
        Plain text representation.
    """
    text = latex_str

    # Common LaTeX symbol replacements
    replacements = {
        # Greek letters
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        r'\\epsilon': 'epsilon',
        r'\\theta': 'theta',
        r'\\lambda': 'lambda',
        r'\\mu': 'mu',
        r'\\pi': 'pi',
        r'\\sigma': 'sigma',
        r'\\omega': 'omega',
        r'\\Gamma': 'Gamma',
        r'\\Delta': 'Delta',
        r'\\Theta': 'Theta',
        r'\\Lambda': 'Lambda',
        r'\\Sigma': 'Sigma',
        r'\\Omega': 'Omega',

        # Operators
        r'\\sum': 'sum',
        r'\\int': 'integral',
        r'\\frac': '/',
        r'\\sqrt': 'sqrt',
        r'\\times': '*',
        r'\\div': '/',
        r'\\pm': '+/-',
        r'\\leq': '<=',
        r'\\geq': '>=',
        r'\\neq': '!=',
        r'\\approx': '≈',
        r'\\infty': 'infinity',
        r'\\partial': 'partial',
        r'\\nabla': 'nabla',

        # Special
        r'\\left': '',
        r'\\right': '',
        r'\\,': ' ',
        r'\\ ': ' ',
        r'\\;': ' ',
        r'\\quad': ' ',
        r'\\qquad': '  ',
    }

    for latex_cmd, replacement in replacements.items():
        text = re.sub(latex_cmd, replacement, text)

    # Handle superscripts: x^{2} → x^2 or x²
    text = re.sub(r'\^{([^}]+)}', r'^\1', text)
    text = re.sub(r'\^(\w)', r'^\1', text)

    # Handle subscripts: x_{i} → x_i
    text = re.sub(r'_{([^}]+)}', r'_\1', text)
    text = re.sub(r'_(\w)', r'_\1', text)

    # Remove remaining braces
    text = text.replace('{', '').replace('}', '')

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
