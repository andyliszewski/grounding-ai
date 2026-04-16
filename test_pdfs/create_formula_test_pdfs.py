#!/usr/bin/env python3
"""
Create test PDFs with mathematical formulas for Epic 8 integration testing.

This script generates 5 diverse test PDFs:
1. simple_equations.pdf - Basic algebra and calculus formulas
2. complex_equations.pdf - Advanced formulas (integrals, matrices, summations)
3. mixed_content.pdf - Text paragraphs with embedded formulas
4. display_equations.pdf - Centered display-style equations
5. edge_cases.pdf - Challenging formulas (fractions, nested expressions)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
from pathlib import Path

# Configure matplotlib to render LaTeX
plt.rcParams['text.usetex'] = False  # Use mathtext, not full LaTeX (more portable)
plt.rcParams['font.size'] = 14


def create_formula_image(latex_formula: str, output_path: Path, display_mode: bool = False):
    """
    Create an image of a mathematical formula using matplotlib.

    Args:
        latex_formula: LaTeX math string (without $ delimiters)
        output_path: Path to save the PNG image
        display_mode: If True, center the formula (display style)
    """
    fig, ax = plt.subplots(figsize=(6, 1.5) if display_mode else (4, 1))
    ax.axis('off')

    # Render formula using mathtext
    # Note: matplotlib's mathtext automatically handles $ delimiters
    formula_text = f"${latex_formula}$"
    ax.text(
        0.5 if display_mode else 0.1,
        0.5,
        formula_text,
        fontsize=20 if display_mode else 16,
        ha='center' if display_mode else 'left',
        va='center'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def create_simple_equations_pdf():
    """Create PDF with 5 simple, well-known formulas."""
    output_path = Path(__file__).parent / "simple_equations.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Simple Mathematical Equations", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Simple formulas with descriptions
    formulas = [
        ("E = mc^2", "Einstein's mass-energy equivalence", True),
        ("a^2 + b^2 = c^2", "Pythagorean theorem", True),
        (r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", "Quadratic formula", True),
        (r"F = ma", "Newton's second law", True),
        (r"\pi r^2", "Area of a circle", True),
    ]

    for i, (formula, description, display) in enumerate(formulas, 1):
        # Description
        story.append(Paragraph(f"<b>Equation {i}:</b> {description}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        # Formula image
        img_path = Path(__file__).parent / f"temp_formula_{i}.png"
        create_formula_image(formula, img_path, display_mode=display)
        story.append(Image(str(img_path), width=3*inch, height=0.75*inch))
        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)

    # Cleanup temporary images
    for i in range(1, 6):
        img_path = Path(__file__).parent / f"temp_formula_{i}.png"
        if img_path.exists():
            img_path.unlink()

    print(f"✓ Created: {output_path} (5 simple formulas)")
    return output_path


def create_complex_equations_pdf():
    """Create PDF with 10+ complex mathematical formulas."""
    output_path = Path(__file__).parent / "complex_equations.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Complex Mathematical Equations", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Complex formulas
    formulas = [
        (r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}", "Gaussian integral"),
        (r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}", "Basel problem"),
        (r"\frac{d}{dx}\left(\frac{u}{v}\right) = \frac{v\frac{du}{dx} - u\frac{dv}{dx}}{v^2}", "Quotient rule"),
        (r"e^{i\pi} + 1 = 0", "Euler's identity"),
        (r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}", "Faraday's law"),
        (r"\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}", "Stokes' theorem"),
        (r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e", "Limit definition of e"),
        (r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u", "Wave equation"),
        (r"H = -\sum_{i} p_i \log p_i", "Shannon entropy"),
        (r"\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}", "Determinant formula"),
    ]

    for i, (formula, description) in enumerate(formulas, 1):
        story.append(Paragraph(f"<b>Equation {i}:</b> {description}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        img_path = Path(__file__).parent / f"temp_complex_{i}.png"
        create_formula_image(formula, img_path, display_mode=True)
        story.append(Image(str(img_path), width=4*inch, height=1*inch))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)

    # Cleanup
    for i in range(1, 11):
        img_path = Path(__file__).parent / f"temp_complex_{i}.png"
        if img_path.exists():
            img_path.unlink()

    print(f"✓ Created: {output_path} (10 complex formulas)")
    return output_path


def create_mixed_content_pdf():
    """Create PDF with text paragraphs and embedded formulas (hybrid document)."""
    output_path = Path(__file__).parent / "mixed_content.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Introduction to Calculus", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Section 1: Derivatives
    story.append(Paragraph("<b>1. Derivatives</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "The derivative of a function measures how the function value changes as its input changes. "
        "For a function f(x), the derivative is defined as the following limit:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Display equation: derivative definition
    img_path = Path(__file__).parent / "temp_mixed_1.png"
    create_formula_image(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", img_path, display_mode=True)
    story.append(Image(str(img_path), width=3.5*inch, height=1*inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "For power functions, we have a simple rule. If f(x) = x^n, then the derivative is:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Display equation: power rule
    img_path2 = Path(__file__).parent / "temp_mixed_2.png"
    create_formula_image(r"f'(x) = nx^{n-1}", img_path2, display_mode=True)
    story.append(Image(str(img_path2), width=2.5*inch, height=0.75*inch))
    story.append(Spacer(1, 0.3 * inch))

    # Section 2: Integrals
    story.append(Paragraph("<b>2. Integrals</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "Integration is the reverse process of differentiation. The definite integral of a function "
        "f(x) from a to b represents the area under the curve:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Display equation: integral
    img_path3 = Path(__file__).parent / "temp_mixed_3.png"
    create_formula_image(r"\int_a^b f(x) dx", img_path3, display_mode=True)
    story.append(Image(str(img_path3), width=2.5*inch, height=1*inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "The Fundamental Theorem of Calculus connects derivatives and integrals. It states that if F is an antiderivative of f, then:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Display equation: FTC
    img_path4 = Path(__file__).parent / "temp_mixed_4.png"
    create_formula_image(r"\int_a^b f(x) dx = F(b) - F(a)", img_path4, display_mode=True)
    story.append(Image(str(img_path4), width=3*inch, height=0.75*inch))
    story.append(Spacer(1, 0.3 * inch))

    doc.build(story)

    # Cleanup
    for i in range(1, 5):
        img_path = Path(__file__).parent / f"temp_mixed_{i}.png"
        if img_path.exists():
            img_path.unlink()

    print(f"✓ Created: {output_path} (text with 4 embedded formulas)")
    return output_path


def create_display_equations_pdf():
    """Create PDF with centered display-style equations."""
    output_path = Path(__file__).parent / "display_equations.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Famous Mathematical Formulas", title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Display equations (all centered, large)
    equations = [
        (r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}", "Sum of first n natural numbers"),
        (r"\prod_{k=1}^{n} k = n!", "Factorial definition"),
        (r"(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k", "Binomial theorem"),
        (r"\sin^2\theta + \cos^2\theta = 1", "Pythagorean trigonometric identity"),
        (r"\mathbb{E}[X] = \sum_{i} x_i P(X = x_i)", "Expected value (discrete)"),
    ]

    for i, (formula, description) in enumerate(equations, 1):
        # Centered formula
        img_path = Path(__file__).parent / f"temp_display_{i}.png"
        create_formula_image(formula, img_path, display_mode=True)

        center_style = ParagraphStyle(
            'Center',
            parent=styles['Normal'],
            alignment=TA_CENTER
        )

        story.append(Spacer(1, 0.2 * inch))
        story.append(Image(str(img_path), width=4*inch, height=1*inch))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"<i>{description}</i>", center_style))
        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)

    # Cleanup
    for i in range(1, 6):
        img_path = Path(__file__).parent / f"temp_display_{i}.png"
        if img_path.exists():
            img_path.unlink()

    print(f"✓ Created: {output_path} (5 display equations)")
    return output_path


def create_edge_cases_pdf():
    """Create PDF with challenging formulas for edge case testing."""
    output_path = Path(__file__).parent / "edge_cases.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='black',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Edge Cases: Complex Notation", title_style))
    story.append(Spacer(1, 0.3 * inch))

    # Challenging formulas
    formulas = [
        (r"\frac{\frac{a}{b}}{\frac{c}{d}} = \frac{ad}{bc}", "Nested fractions"),
        (r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}", "Newton's method (subscripts)"),
        (r"\binom{n}{k} = \frac{n!}{k!(n-k)!}", "Binomial coefficient (factorials)"),
        (r"\sqrt[3]{x^2 + y^2}", "Cube root with expression"),
        (r"\lim_{x \to \infty} \left(1 + \frac{k}{x}\right)^x = e^k", "Nested parentheses"),
    ]

    for i, (formula, description) in enumerate(formulas, 1):
        story.append(Paragraph(f"<b>Case {i}:</b> {description}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

        img_path = Path(__file__).parent / f"temp_edge_{i}.png"
        create_formula_image(formula, img_path, display_mode=True)
        story.append(Image(str(img_path), width=3.5*inch, height=1*inch))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)

    # Cleanup
    for i in range(1, 6):
        img_path = Path(__file__).parent / f"temp_edge_{i}.png"
        if img_path.exists():
            img_path.unlink()

    print(f"✓ Created: {output_path} (5 edge case formulas)")
    return output_path


def main():
    """Generate all test PDFs."""
    print("Generating test PDFs with mathematical formulas...")
    print("-" * 60)

    try:
        create_simple_equations_pdf()
        create_complex_equations_pdf()
        create_mixed_content_pdf()
        create_display_equations_pdf()
        create_edge_cases_pdf()

        print("-" * 60)
        print("✓ All test PDFs created successfully!")
        print(f"\nTest PDFs location: {Path(__file__).parent}")
        print("\nCreated files:")
        print("  1. simple_equations.pdf (5 basic formulas)")
        print("  2. complex_equations.pdf (10 advanced formulas)")
        print("  3. mixed_content.pdf (text with 4 embedded formulas)")
        print("  4. display_equations.pdf (5 centered equations)")
        print("  5. edge_cases.pdf (5 challenging formulas)")
        print("\nTotal formulas: 29")

    except Exception as e:
        print(f"\n✗ Error creating PDFs: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
