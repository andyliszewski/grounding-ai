"""Tests for grounding.eval.metrics (Story 16.2)."""
from __future__ import annotations

import math

import pytest

from grounding.eval.metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_perfect() -> None:
    assert recall_at_k([1, 1, 1, 1], k=5) == 1.0


def test_recall_at_k_no_hits() -> None:
    assert recall_at_k([None, None, None], k=10) == 0.0


def test_recall_at_k_partial() -> None:
    # ranks [1, 3, None, 7] for k=5 -> hits at 1 and 3 -> 2/4 = 0.5
    assert recall_at_k([1, 3, None, 7], k=5) == 0.5


def test_recall_at_k_boundary() -> None:
    # rank exactly k counts as hit
    assert recall_at_k([5], k=5) == 1.0
    # rank one past k does not
    assert recall_at_k([6], k=5) == 0.0


def test_recall_at_k_empty_input() -> None:
    assert recall_at_k([], k=10) == 0.0


def test_recall_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1], k=0)


def test_mrr_basic() -> None:
    # ranks [1, 2, None, 4] -> (1 + 0.5 + 0 + 0.25) / 4
    expected = (1.0 + 0.5 + 0.0 + 0.25) / 4
    assert mrr([1, 2, None, 4]) == pytest.approx(expected)


def test_mrr_all_miss() -> None:
    assert mrr([None, None]) == 0.0


def test_mrr_empty_input() -> None:
    assert mrr([]) == 0.0


def test_ndcg_at_k_perfect_single_relevant() -> None:
    # retrieved [1, 0, 0], expected_count = 1, k=3 -> DCG = 1/log2(2) = 1, IDCG = 1 -> 1.0
    assert ndcg_at_k([[1, 0, 0]], [1], k=3) == pytest.approx(1.0)


def test_ndcg_at_k_hit_at_rank_three() -> None:
    # relevance [0,0,1], expected_count=1, k=3
    # DCG = 1 / log2(4) = 0.5; IDCG = 1 / log2(2) = 1.0; nDCG = 0.5
    assert ndcg_at_k([[0, 0, 1]], [1], k=3) == pytest.approx(0.5)


def test_ndcg_at_k_multiple_relevant_hand_computed() -> None:
    # Two relevant docs expected, retrieved [1,1,0,0] over k=4
    # DCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297...
    # IDCG (R=2) = 1/log2(2) + 1/log2(3) = same -> nDCG = 1.0
    val = ndcg_at_k([[1, 1, 0, 0]], [2], k=4)
    assert val == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_retrieved() -> None:
    assert ndcg_at_k([[0, 0, 0]], [1], k=3) == 0.0


def test_ndcg_at_k_no_expected_zero_idcg() -> None:
    # expected_count = 0 -> IDCG = 0 -> contribute 0 (not NaN)
    assert ndcg_at_k([[0, 0, 0]], [0], k=3) == 0.0


def test_ndcg_at_k_averages_across_items() -> None:
    # First item perfect (1.0), second a miss (0.0) -> mean 0.5
    val = ndcg_at_k([[1, 0], [0, 0]], [1, 1], k=2)
    assert val == pytest.approx(0.5)


def test_ndcg_at_k_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([[1, 0]], [1, 1], k=2)


def test_ndcg_at_k_empty_input() -> None:
    assert ndcg_at_k([], [], k=10) == 0.0


def test_ndcg_discount_matches_formula() -> None:
    # Single relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k([[0, 1, 0]], [1], k=3) == pytest.approx(expected)
