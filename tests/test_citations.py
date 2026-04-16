"""Tests for grounding.citations (Story 17.3)."""
from __future__ import annotations

from grounding.citations import EN_DASH, format_citation_prefix


def test_full_prefix_all_fields_present():
    got = format_citation_prefix(
        "alpha-paper.pdf", 247, 247, "3.2 Bootstrap Methods"
    )
    assert got == "[alpha-paper, p.247, §3.2 Bootstrap Methods]"


def test_page_range_renders_en_dash():
    got = format_citation_prefix(
        "alpha-paper.pdf", 247, 249, "3.2 Bootstrap Methods"
    )
    assert got == f"[alpha-paper, p.247{EN_DASH}249, §3.2 Bootstrap Methods]"
    assert "-" not in got.split("p.247")[1].split(",")[0]


def test_page_only_when_section_null():
    got = format_citation_prefix("alpha-paper.pdf", 247, 247, None)
    assert got == "[alpha-paper, p.247]"


def test_page_range_only_when_section_null():
    got = format_citation_prefix("alpha-paper.pdf", 247, 249, None)
    assert got == f"[alpha-paper, p.247{EN_DASH}249]"


def test_section_only_when_pages_null():
    got = format_citation_prefix("beta-study.epub", None, None, "4. Methods")
    assert got == "[beta-study, §4. Methods]"


def test_slug_only_when_all_null():
    got = format_citation_prefix("gamma-notes.md", None, None, None)
    assert got == "[gamma-notes]"


def test_slug_derived_from_filename_stem():
    got = format_citation_prefix("Report_2024 Q3.pdf", None, None, None)
    assert got == "[report-2024-q3]"


def test_slug_passed_through_when_already_slug_shaped():
    got = format_citation_prefix("alpha-paper", None, None, None)
    assert got == "[alpha-paper]"


def test_section_with_special_characters_preserved():
    got = format_citation_prefix(
        "alpha-paper.pdf", 1, 1, "§A.1 — 'Weird': chars/OK?"
    )
    assert got == "[alpha-paper, p.1, §§A.1 — 'Weird': chars/OK?]"


def test_empty_section_string_treated_as_null():
    got = format_citation_prefix("alpha-paper.pdf", 1, 1, "   ")
    assert got == "[alpha-paper, p.1]"


def test_page_end_none_treated_as_single_page():
    got = format_citation_prefix("alpha-paper.pdf", 5, None, None)
    assert got == "[alpha-paper, p.5]"
