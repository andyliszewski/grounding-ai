"""Bootstrap statistics for the answer benchmark (Epic 25).

One home for the resampling the report, the replicate check and the power
simulation all use, so every interval in the benchmark comes from the same
code. ``report.py`` re-exports these under its own names.

Every interval is a 95 percent percentile bootstrap with a fixed seed, so a
report is reproducible. **The resampling is over items only**: it says how far
the estimate would move on another sample of questions, not on another run of
the model. Model sampling is measured separately by the replicate check.
"""
from __future__ import annotations

import math
import random
from typing import Sequence, Tuple

N_BOOT = 2000
SEED = 0


def percentile(sorted_values: Sequence[float], q: float) -> float:
    index = q * (len(sorted_values) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def bootstrap_mean_ci(values: Sequence[float], *, seed: int = SEED, n_boot: int = N_BOOT):
    """95 percent interval for the mean of per-item values (correctness grades)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def bootstrap_ratio_ci(pairs: Sequence[Tuple[int, int]], *, seed: int = SEED, n_boot: int = N_BOOT):
    """CI of sum(num)/sum(den), resampling the (num, den) clusters.

    One cluster is one answer, because citations inside an answer are not
    independent: a verified-citation rate is a ratio over clustered counts.
    """
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    ratios = []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            num += a
            den += b
        if den:
            ratios.append(num / den)
    if not ratios:
        return None
    ratios.sort()
    return [percentile(ratios, 0.025), percentile(ratios, 0.975)]


def paired_ratio_diff(
    pairs_a: Sequence[Tuple[int, int]], pairs_b: Sequence[Tuple[int, int]],
    *, seed: int = SEED, n_boot: int = N_BOOT,
):
    """sum(a_num)/sum(a_den) minus the same for b, with a paired item bootstrap.

    ``pairs_a[i]`` and ``pairs_b[i]`` are the two conditions' (numerator,
    denominator) for the same item; a resample draws items, so both
    conditions' citations of an item stay together. Returns (estimate, ci95).
    """
    n = len(pairs_a)

    def diff(index: Sequence[int]):
        na = sum(pairs_a[i][0] for i in index)
        da = sum(pairs_a[i][1] for i in index)
        nb = sum(pairs_b[i][0] for i in index)
        db = sum(pairs_b[i][1] for i in index)
        return None if not da or not db else na / da - nb / db

    estimate = diff(range(n))
    if estimate is None or n < 2:
        return estimate, None
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        value = diff([rng.randrange(n) for _ in range(n)])
        if value is not None:
            boots.append(value)
    boots.sort()
    return estimate, ([percentile(boots, 0.025), percentile(boots, 0.975)] if boots else None)
