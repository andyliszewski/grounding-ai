"""Tests for chunker provenance API (Story 17.2).

Covers:
- AC #5 page range derivation (incl. mixed-None handling).
- AC #6 section heading — earlier section wins on boundary.
- AC #9 six required scenarios plus monotonic offsets and backward compat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from grounding.chunker import (
    ChunkConfig,
    ChunkProvenance,
    ChunkWithProvenance,
    derive_chunk_metadata,
    split_markdown,
    split_markdown_with_map,
)
from grounding.formatter import FormattedElement, format_markdown_with_map
from grounding.parser import TextElement


# ---------------------------------------------------------------------------
# Helpers: reuse 17.1 stand-in shapes locally.
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


def _fe(char_start: int, char_end: int, page, stack=()) -> FormattedElement:
    return FormattedElement(
        char_start=char_start,
        char_end=char_end,
        page_number=page,
        heading_stack=tuple(stack),
    )


# ---------------------------------------------------------------------------
# AC #5 — page range derivation.
# ---------------------------------------------------------------------------


def test_single_page_chunk_gets_single_page():
    elements = [
        _fe(0, 50, 7, ("Intro",)),
        _fe(52, 100, 7, ("Intro",)),
    ]
    chunk = ChunkWithProvenance(text="...", char_start=10, char_end=80)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.page_start == 7
    assert prov.page_end == 7
    assert prov.section_heading == "Intro"


def test_chunk_spanning_pages_3_to_5_gets_range():
    elements = [
        _fe(0, 30, 3),
        _fe(32, 70, 4),
        _fe(72, 110, 5),
        _fe(112, 150, 6),
    ]
    chunk = ChunkWithProvenance(text="...", char_start=10, char_end=100)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.page_start == 3
    assert prov.page_end == 5


def test_chunk_with_all_null_pages_gets_null_pages():
    elements = [
        _fe(0, 30, None),
        _fe(32, 70, None),
    ]
    chunk = ChunkWithProvenance(text="...", char_start=0, char_end=70)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.page_start is None
    assert prov.page_end is None


def test_chunk_with_mixed_null_and_page_excludes_null():
    elements = [
        _fe(0, 30, None),
        _fe(32, 70, 4),
        _fe(72, 110, None),
        _fe(112, 150, 6),
    ]
    chunk = ChunkWithProvenance(text="...", char_start=0, char_end=150)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.page_start == 4
    assert prov.page_end == 6


# ---------------------------------------------------------------------------
# AC #6 — section heading boundary convention.
# ---------------------------------------------------------------------------


def test_chunk_straddling_heading_gets_earlier_section():
    elements = [
        _fe(0, 10, None, ("3.2 Bootstrap",)),      # heading
        _fe(12, 60, None, ("3.2 Bootstrap",)),     # body under 3.2
        _fe(62, 72, None, ("3.3 Kernels",)),       # next heading
        _fe(74, 120, None, ("3.3 Kernels",)),      # body under 3.3
    ]
    # chunk begins mid-§3.2 and ends mid-§3.3 — earlier section wins.
    chunk = ChunkWithProvenance(text="...", char_start=30, char_end=100)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.section_heading == "3.2 Bootstrap"


def test_chunk_before_any_heading_has_null_section():
    elements = [
        _fe(0, 40, 1, ()),                         # body, no heading yet
        _fe(42, 60, 1, ("Intro",)),                # first heading later
    ]
    chunk = ChunkWithProvenance(text="...", char_start=0, char_end=30)
    prov = derive_chunk_metadata(chunk, elements)
    assert prov.section_heading is None


# ---------------------------------------------------------------------------
# AC #9 — integration-ish scenarios using format_markdown_with_map.
# ---------------------------------------------------------------------------


def test_markdown_only_input_null_pages_populated_sections():
    # Raw-markdown path: TextElement inputs → no pages, headings from `#` lines.
    elements = [
        TextElement(text="# Introduction"),
        TextElement(text="Opening paragraph under intro."),
        TextElement(text="## Background"),
        TextElement(text="Background paragraph one."),
        TextElement(text="Background paragraph two."),
    ]
    result = format_markdown_with_map(elements)
    chunk_records = split_markdown_with_map(
        result.markdown, result.elements, ChunkConfig(chunk_size=120, chunk_overlap=20)
    )
    assert chunk_records, "expected at least one chunk"
    # Every chunk: page_start/end None; section_heading populated.
    any_section = False
    for rec in chunk_records:
        prov = derive_chunk_metadata(rec, result.elements)
        assert prov.page_start is None
        assert prov.page_end is None
        if prov.section_heading is not None:
            any_section = True
            assert prov.section_heading in ("Introduction", "Background")
    assert any_section, "expected at least one chunk to carry a section heading"


def test_pdftotext_fallback_all_null():
    # TextElement with no `#` headings ≈ pdftotext fallback.
    elements = [
        TextElement(text="Line of pdftotext output."),
        TextElement(text="Another paragraph with no markdown heading syntax."),
        TextElement(text="Third paragraph likewise."),
    ]
    result = format_markdown_with_map(elements)
    chunk_records = split_markdown_with_map(
        result.markdown, result.elements, ChunkConfig(chunk_size=80, chunk_overlap=10)
    )
    assert chunk_records
    for rec in chunk_records:
        prov = derive_chunk_metadata(rec, result.elements)
        assert prov.page_start is None
        assert prov.page_end is None
        assert prov.section_heading is None


def test_empty_element_map_returns_all_null():
    chunk = ChunkWithProvenance(text="anything", char_start=0, char_end=8)
    prov = derive_chunk_metadata(chunk, ())
    assert prov == ChunkProvenance(None, None, None)


# ---------------------------------------------------------------------------
# Invariants.
# ---------------------------------------------------------------------------


def test_char_offsets_monotonic():
    elements = [_title("Chapter 1", page=1), _para("Body body body " * 20, page=1)]
    result = format_markdown_with_map(elements)
    chunk_records = split_markdown_with_map(
        result.markdown, result.elements, ChunkConfig(chunk_size=80, chunk_overlap=20)
    )
    prev_start = -1
    for rec in chunk_records:
        assert rec.char_start >= prev_start
        assert rec.char_end >= rec.char_start
        assert result.markdown[rec.char_start:rec.char_end] == rec.text
        prev_start = rec.char_start


def test_split_markdown_backward_compatible():
    # AC #2: old API unchanged.
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    cfg = ChunkConfig(chunk_size=30, chunk_overlap=5)
    strings = split_markdown(text, cfg)
    records = split_markdown_with_map(text, (), cfg)
    assert [r.text for r in records] == strings


def test_unstructured_elements_full_integration():
    # End-to-end using format_markdown_with_map + provenance derivation
    # against unstructured-style stub elements across multiple pages.
    elements = [
        _title("Chapter 1", page=1, level=1),
        _para("Body of chapter one on page one. " * 3, page=1),
        _para("Continuation on page two. " * 3, page=2),
        _title("Section 1.1", page=2, level=2),
        _para("Subsection content on page two. " * 3, page=2),
        _para("Subsection continues to page three. " * 3, page=3),
    ]
    result = format_markdown_with_map(elements)
    records = split_markdown_with_map(
        result.markdown, result.elements, ChunkConfig(chunk_size=200, chunk_overlap=30)
    )
    # First chunk should be under Chapter 1, starting on page 1.
    first = derive_chunk_metadata(records[0], result.elements)
    assert first.page_start == 1
    assert first.section_heading == "Chapter 1"
    # At least one chunk should touch page 3.
    pages_seen = set()
    for rec in records:
        prov = derive_chunk_metadata(rec, result.elements)
        if prov.page_start is not None:
            pages_seen.update(range(prov.page_start, prov.page_end + 1))
    assert 3 in pages_seen
