"""Retrieval metrics for the evaluation harness (Story 16.2).

Pure functions with no I/O. All metrics operate on already-computed rank
information; the runner handles retrieval and converts results into the
inputs these functions expect.

Metric definitions
------------------
- ``recall@k``: fraction of items whose first hit falls within top-k.
- ``mrr``: mean reciprocal rank; items with no hit contribute 0.
- ``ndcg@k``: binary-relevance nDCG using a list of 0/1 relevance flags
  over the retrieved top-k. Formula::

      DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
      IDCG@k = sum_{i=1..min(k, R)} 1 / log2(i + 1)   where R = |expected|
      nDCG@k = DCG@k / IDCG@k   (0 if IDCG@k == 0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def recall_at_k(first_hit_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of items whose first hit is at rank <= k.

    Ranks are 1-indexed. ``None`` means no hit in the retrieved list.
    Returns 0.0 for an empty input (no items) so the metric stays defined
    in degenerate cases.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not first_hit_ranks:
        return 0.0
    hits = sum(1 for r in first_hit_ranks if r is not None and r <= k)
    return hits / len(first_hit_ranks)


def mrr(first_hit_ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank. Items with no hit contribute 0."""
    if not first_hit_ranks:
        return 0.0
    total = sum(1.0 / r for r in first_hit_ranks if r is not None and r > 0)
    return total / len(first_hit_ranks)


def ndcg_at_k(
    relevance_lists: Iterable[Sequence[int]],
    expected_counts: Iterable[int],
    k: int,
) -> float:
    """Binary-relevance nDCG@k averaged over items.

    Args:
        relevance_lists: For each item, a sequence of 0/1 flags over the
            retrieved top-k (already ordered by rank). Lists shorter than
            k are padded with zeros implicitly.
        expected_counts: For each item, ``|expected.doc_ids|`` (used to
            compute IDCG@k = sum of 1/log2(i+1) for i in 1..min(k, R)).
        k: Cutoff.

    Returns:
        Mean nDCG@k across items. 0.0 for empty input or when every item
        has IDCG@k == 0 (i.e. no expected relevant docs).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rel_lists = list(relevance_lists)
    counts = list(expected_counts)
    if len(rel_lists) != len(counts):
        raise ValueError(
            f"relevance_lists and expected_counts length mismatch: "
            f"{len(rel_lists)} vs {len(counts)}"
        )
    if not rel_lists:
        return 0.0

    per_item: list[float] = []
    for rel, r in zip(rel_lists, counts):
        dcg = _dcg_at_k(rel, k)
        idcg = _idcg_at_k(r, k)
        per_item.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(per_item) / len(per_item)


@dataclass(frozen=True)
class CitationCase:
    """Inputs for a single item's citation-accuracy check.

    Retrieved fields describe the *first* retrieved chunk (rank 1).
    Expected fields come from the fixture's ``expected.page`` / ``expected.section``.
    """

    expected_page: int | tuple[int, int] | None
    expected_section: str | None
    retrieved_page_start: int | None
    retrieved_page_end: int | None
    retrieved_section: str | None


def citation_accuracy(cases: Sequence[CitationCase]) -> float | None:
    """Fraction of citation-checked items whose first-hit chunk matches expectations.

    Hit semantics (Story 17.4):

    - ``expected_page`` as int ``N``: match when
      ``retrieved_page_start <= N <= retrieved_page_end``.
    - ``expected_page`` as ``(start, end)``: match when the retrieved page
      range ``[retrieved_page_start, retrieved_page_end]`` overlaps
      ``[start, end]``.
    - ``expected_section`` as str: match when ``retrieved_section`` equals
      it exactly (case-sensitive, by design).
    - When both expected fields are set, *both* must match.
    - Items with neither ``expected_page`` nor ``expected_section`` are
      excluded from the metric. If no item carries citation expectations,
      returns ``None``.
    - An item with citation expectations but missing retrieved metadata
      (``retrieved_page_start is None`` where page is expected, etc.) counts
      as a miss.
    """
    n_checked = 0
    n_hits = 0
    for case in cases:
        if case.expected_page is None and case.expected_section is None:
            continue
        n_checked += 1
        if _citation_case_matches(case):
            n_hits += 1
    if n_checked == 0:
        return None
    return n_hits / n_checked


def _citation_case_matches(case: CitationCase) -> bool:
    if case.expected_page is not None:
        if not _page_matches(
            case.expected_page,
            case.retrieved_page_start,
            case.retrieved_page_end,
        ):
            return False
    if case.expected_section is not None:
        if case.retrieved_section != case.expected_section:
            return False
    return True


def _page_matches(
    expected: int | tuple[int, int],
    retrieved_start: int | None,
    retrieved_end: int | None,
) -> bool:
    if retrieved_start is None or retrieved_end is None:
        return False
    if isinstance(expected, int):
        return retrieved_start <= expected <= retrieved_end
    exp_start, exp_end = expected
    return retrieved_start <= exp_end and exp_start <= retrieved_end


def _dcg_at_k(relevance: Sequence[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        if rel:
            total += 1.0 / math.log2(i + 1)
    return total


def _idcg_at_k(num_relevant: int, k: int) -> float:
    total = 0.0
    for i in range(1, min(k, num_relevant) + 1):
        total += 1.0 / math.log2(i + 1)
    return total
