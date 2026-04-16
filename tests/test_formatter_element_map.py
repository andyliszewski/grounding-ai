"""Tests for format_markdown_with_map (Story 17.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from grounding.formatter import (
    FormatResult,
    FormattedElement,
    _coverage_check,
    _compute_heading_stacks,
    format_markdown_with_map,
)
from grounding.parser import TextElement


# ---------------------------------------------------------------------------
# Lightweight stand-ins for Unstructured elements
# ---------------------------------------------------------------------------


@dataclass
class StubMetadata:
    page_number: Optional[int] = None
    category_depth: Optional[int] = None


@dataclass
class StubElement:
    text: str
    category: str = "NarrativeText"
    metadata: Any = field(default_factory=StubMetadata)


def _title(text: str, *, page: Optional[int] = None, level: int = 1) -> StubElement:
    return StubElement(
        text=text,
        category="Title",
        metadata=StubMetadata(page_number=page, category_depth=level - 1),
    )


def _para(text: str, *, page: Optional[int] = None) -> StubElement:
    return StubElement(
        text=text,
        category="NarrativeText",
        metadata=StubMetadata(page_number=page),
    )


# ---------------------------------------------------------------------------
# Unstructured-path tests (AC 4, 7, 8)
# ---------------------------------------------------------------------------


def test_multi_page_pdf_elements_populate_page_numbers():
    elements = [
        _title("Chapter 1", page=1, level=1),
        _para("Paragraph on page one.", page=1),
        _para("Paragraph that spans onto page two.", page=2),
        _title("Chapter 2", page=3, level=1),
        _para("Start of chapter two body.", page=3),
    ]
    result = format_markdown_with_map(elements)
    assert isinstance(result, FormatResult)
    assert [e.page_number for e in result.elements] == [1, 1, 2, 3, 3]
    assert _coverage_check(result.markdown, result.elements)


def test_heading_stack_nesting_h1_h2_h3():
    elements = [
        _title("Alpha", level=1),
        _title("Beta", level=2),
        _para("under beta"),
        _title("Gamma", level=3),
        _para("under gamma"),
        _title("Delta", level=1),  # pops Beta and Gamma
        _para("under delta"),
    ]
    result = format_markdown_with_map(elements)
    stacks = [e.heading_stack for e in result.elements]
    assert stacks[0] == ("Alpha",)
    assert stacks[1] == ("Alpha", "Beta")
    assert stacks[2] == ("Alpha", "Beta")
    assert stacks[3] == ("Alpha", "Beta", "Gamma")
    assert stacks[4] == ("Alpha", "Beta", "Gamma")
    assert stacks[5] == ("Delta",)
    assert stacks[6] == ("Delta",)


def test_no_heading_elements_gives_empty_stack():
    elements = [_para("first", page=1), _para("second", page=1)]
    result = format_markdown_with_map(elements)
    assert all(e.heading_stack == () for e in result.elements)
    assert _coverage_check(result.markdown, result.elements)


# ---------------------------------------------------------------------------
# TextElement path (AC 5)
# ---------------------------------------------------------------------------


def test_pdftotext_fallback_page_numbers_are_none():
    elements = [
        TextElement(text="Plain text paragraph without markup."),
        TextElement(text="Another paragraph, also plain."),
    ]
    result = format_markdown_with_map(elements)
    assert all(e.page_number is None for e in result.elements)
    assert all(e.heading_stack == () for e in result.elements)
    assert _coverage_check(result.markdown, result.elements)


# ---------------------------------------------------------------------------
# Raw-Markdown / EPUB path (AC 6, 9)
# ---------------------------------------------------------------------------


def test_raw_markdown_headings_populate_stack():
    elements = [
        TextElement(text="# Intro"),
        TextElement(text="Intro body paragraph."),
        TextElement(text="## Details"),
        TextElement(text="Detail body paragraph."),
        TextElement(text="# Epilogue"),
        TextElement(text="Final words."),
    ]
    result = format_markdown_with_map(elements)
    stacks = [e.heading_stack for e in result.elements]
    assert stacks == [
        ("Intro",),
        ("Intro",),
        ("Intro", "Details"),
        ("Intro", "Details"),
        ("Epilogue",),
        ("Epilogue",),
    ]
    assert all(e.page_number is None for e in result.elements)
    assert _coverage_check(result.markdown, result.elements)


def test_epub_style_input_no_pages_has_headings():
    elements = [
        _title("Chapter One", page=None, level=1),
        _para("Chapter content.", page=None),
        _title("Section A", page=None, level=2),
        _para("Section body.", page=None),
    ]
    result = format_markdown_with_map(elements)
    assert all(e.page_number is None for e in result.elements)
    stacks = [e.heading_stack for e in result.elements]
    assert stacks == [
        ("Chapter One",),
        ("Chapter One",),
        ("Chapter One", "Section A"),
        ("Chapter One", "Section A"),
    ]
    assert _coverage_check(result.markdown, result.elements)


# ---------------------------------------------------------------------------
# Invariants (AC 7)
# ---------------------------------------------------------------------------


def test_elements_are_ordered_by_char_start():
    elements = [_para("alpha"), _para("beta"), _para("gamma")]
    result = format_markdown_with_map(elements)
    starts = [e.char_start for e in result.elements]
    assert starts == sorted(starts)


def test_element_spans_are_non_overlapping():
    elements = [_para("aaaa"), _para("bbbb"), _para("cccc")]
    result = format_markdown_with_map(elements)
    for prev, curr in zip(result.elements, result.elements[1:]):
        assert prev.char_end <= curr.char_start


def test_coverage_check_invariant_on_mixed_inputs():
    unstructured = [
        _title("Title", page=1, level=1),
        _para("Body one.", page=1),
        _para("Body two.", page=2),
    ]
    text_elements = [
        TextElement(text="# Heading"),
        TextElement(text="Body paragraph."),
    ]
    fallback = [
        TextElement(text="Line one."),
        TextElement(text="Line two."),
    ]
    for scenario in (unstructured, text_elements, fallback):
        result = format_markdown_with_map(scenario)
        assert _coverage_check(result.markdown, result.elements), scenario


# ---------------------------------------------------------------------------
# Additional sanity coverage for helpers + edge cases
# ---------------------------------------------------------------------------


def test_compute_heading_stacks_pops_equal_and_higher_levels():
    stacks = _compute_heading_stacks(
        [
            (1, "A"),
            None,
            (2, "B"),
            (2, "C"),  # sibling: pops B, pushes C
            None,
            (1, "D"),  # new top-level: pops all
        ]
    )
    assert stacks == [
        ("A",),
        ("A",),
        ("A", "B"),
        ("A", "C"),
        ("A", "C"),
        ("D",),
    ]


def test_empty_input_returns_empty_result():
    result = format_markdown_with_map([])
    assert result.markdown == ""
    assert result.elements == ()
    assert _coverage_check(result.markdown, result.elements)


def test_front_matter_offsets_element_positions():
    elements = [_para("hello")]
    result = format_markdown_with_map(elements, metadata={"slug": "x"})
    assert result.markdown.startswith("---\n")
    first = result.elements[0]
    # Body text "hello" should appear at char_start within the markdown
    assert result.markdown[first.char_start : first.char_end] == "hello"


def test_whitespace_only_elements_are_skipped():
    elements = [_para("real"), _para("   "), _para("also real")]
    result = format_markdown_with_map(elements)
    assert len(result.elements) == 2
    assert _coverage_check(result.markdown, result.elements)


def test_text_element_non_heading_line_is_not_treated_as_heading():
    # Text starting with '#' but not a heading shape
    elements = [TextElement(text="#notaheading is part of sentence")]
    result = format_markdown_with_map(elements)
    assert result.elements[0].heading_stack == ()


def test_rejects_string_input():
    import pytest

    with pytest.raises(TypeError):
        format_markdown_with_map("not a sequence")  # type: ignore[arg-type]
