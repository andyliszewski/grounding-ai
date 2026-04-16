"""Unit tests for the citation_accuracy metric (Story 17.4)."""
from __future__ import annotations

from grounding.eval.metrics import CitationCase, citation_accuracy


def _case(
    *,
    expected_page=None,
    expected_section=None,
    rps=None,
    rpe=None,
    rsection=None,
) -> CitationCase:
    return CitationCase(
        expected_page=expected_page,
        expected_section=expected_section,
        retrieved_page_start=rps,
        retrieved_page_end=rpe,
        retrieved_section=rsection,
    )


def test_page_int_match_within_range() -> None:
    assert citation_accuracy([_case(expected_page=248, rps=247, rpe=249)]) == 1.0


def test_page_int_miss_outside_range() -> None:
    assert citation_accuracy([_case(expected_page=250, rps=247, rpe=249)]) == 0.0


def test_page_list_overlap_hit() -> None:
    assert citation_accuracy([_case(expected_page=(15, 18), rps=15, rpe=16)]) == 1.0


def test_page_list_partial_overlap_hit() -> None:
    assert citation_accuracy([_case(expected_page=(15, 18), rps=17, rpe=25)]) == 1.0


def test_page_list_no_overlap_miss() -> None:
    assert citation_accuracy([_case(expected_page=(15, 18), rps=20, rpe=21)]) == 0.0


def test_section_exact_match() -> None:
    assert (
        citation_accuracy(
            [_case(expected_section="3.2 Bootstrap Methods", rsection="3.2 Bootstrap Methods")]
        )
        == 1.0
    )


def test_section_case_mismatch_miss() -> None:
    assert (
        citation_accuracy(
            [
                _case(
                    expected_section="3.2 Bootstrap Methods",
                    rsection="3.2 bootstrap methods",
                )
            ]
        )
        == 0.0
    )


def test_both_fields_both_must_match() -> None:
    # page matches but section doesn't => miss
    case_miss = _case(
        expected_page=247,
        expected_section="A",
        rps=247,
        rpe=247,
        rsection="B",
    )
    # both match => hit
    case_hit = _case(
        expected_page=247,
        expected_section="A",
        rps=247,
        rpe=247,
        rsection="A",
    )
    assert citation_accuracy([case_miss]) == 0.0
    assert citation_accuracy([case_hit]) == 1.0


def test_item_without_citation_expectations_excluded() -> None:
    # One excluded, one hit => 1/1 = 1.0
    result = citation_accuracy(
        [
            _case(),  # excluded
            _case(expected_page=10, rps=10, rpe=10),
        ]
    )
    assert result == 1.0


def test_returns_none_when_no_citation_items() -> None:
    assert citation_accuracy([_case(), _case()]) is None


def test_returns_none_for_empty_sequence() -> None:
    assert citation_accuracy([]) is None


def test_missing_retrieved_metadata_counts_as_miss() -> None:
    assert citation_accuracy([_case(expected_page=10)]) == 0.0
    assert citation_accuracy([_case(expected_section="X")]) == 0.0


def test_mixed_hits_and_misses_compute_fraction() -> None:
    cases = [
        _case(expected_page=1, rps=1, rpe=1),  # hit
        _case(expected_page=2, rps=5, rpe=5),  # miss
        _case(expected_section="S", rsection="S"),  # hit
        _case(),  # excluded
    ]
    # 2 hits / 3 checked
    assert citation_accuracy(cases) == 2 / 3
