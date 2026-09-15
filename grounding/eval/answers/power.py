"""Power simulation for the answer benchmark (Epic 25, pre-registered analysis).

The epic states how wide the intervals will be at the planned number of items,
and how far a replicate of the same condition moves when nothing has changed.
This script produces those numbers, with the report's own bootstrap functions
(``grounding.eval.answers.stats``, re-exported by ``report.py``), so the
pre-registered figures and the run's figures come from one implementation.

    ./venv/bin/python -m grounding.eval.answers.power

What it simulates, per simulated benchmark:

* ``items`` answerable items. Each gets a citation count drawn from
  ``CITATION_COUNTS`` (mean about 3.4), and, per condition, a per-item
  propensity to be verified drawn from a Beta with that condition's mean and
  ``CONCENTRATION``. Citations within an answer are correlated, which is why
  the report's citation intervals resample answers rather than citations; the
  simulation reproduces that clustering rather than assuming independence.
* Correctness per item from ``CORRECTNESS_MIX``, a distribution over 1, 0.5
  and 0 with the stated mean.
* A replicate: the same items and the same per-item propensities, drawn
  again. Only the model's sampling differs, so the spread of the difference
  in verified rate is the run-to-run noise floor.

Everything is seeded, so the table is reproducible. Nothing here touches the
API, a corpus or a run directory.
"""
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from grounding.eval.answers.report import (
    N_BOOT,
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    paired_ratio_diff,
)

#: Citations per answer: the shape the scorer sees when a model is told to
#: state one claim per sentence and cite it.
CITATION_COUNTS: Tuple[Tuple[int, float], ...] = ((1, 0.1), (2, 0.2), (3, 0.25),
                                                  (4, 0.25), (5, 0.15), (6, 0.05))
#: Beta concentration for the per-item verified propensity: lower is lumpier.
#: At 4, an item's citations are strongly correlated, the conservative case.
CONCENTRATION = 4.0
#: Plausible true verified rates: the grounded condition roughly double the
#: ungrounded one. The half-widths barely move with these; the difference does.
VERIFIED_RATES = {"hybrid-rerank": 0.65, "ungrounded": 0.35}
#: Correctness grades and their probabilities (mean 0.70).
CORRECTNESS_MIX: Tuple[Tuple[float, float], ...] = ((1.0, 0.55), (0.5, 0.30), (0.0, 0.15))
DEFAULT_ITEMS = 35
DEFAULT_SIMS = 200
#: The old replicate rule re-ran 12 items and called the result noise when the
#: verified rate moved by more than half the primary difference.
OLD_REPLICATE_ITEMS = 12
OLD_REPLICATE_THRESHOLD = 0.12


def _pick(rng: random.Random, options: Sequence[Tuple[float, float]]) -> float:
    roll = rng.random()
    total = 0.0
    for value, weight in options:
        total += weight
        if roll <= total:
            return value
    return options[-1][0]


def _draw_condition(rng: random.Random, counts: Sequence[int], rate: float) -> List[Tuple[int, int]]:
    """Per item, (verified, citations), with citations inside an answer correlated."""
    alpha, beta = rate * CONCENTRATION, (1 - rate) * CONCENTRATION
    pairs = []
    for n in counts:
        propensity = rng.betavariate(alpha, beta)
        pairs.append((sum(1 for _ in range(n) if rng.random() < propensity), n))
    return pairs


def _half_width(ci) -> float | None:
    return None if ci is None else (ci[1] - ci[0]) / 2


@dataclass
class PowerResult:
    items: int
    sims: int
    n_boot: int
    seed: int
    half_widths: Dict[str, float]
    replicate: Dict[str, float]

    def render(self) -> str:
        lines = [
            f"Power simulation: {self.items} answerable items, {self.sims} simulated "
            f"benchmarks, {self.n_boot} bootstrap resamples, seed {self.seed}.",
            f"Citations per answer mean "
            f"{sum(n * w for n, w in CITATION_COUNTS):.1f}; true verified rates "
            + ", ".join(f"{c} {r:.0%}" for c, r in VERIFIED_RATES.items())
            + f"; Beta concentration {CONCENTRATION:g}; correctness mean "
            f"{sum(v * w for v, w in CORRECTNESS_MIX):.2f}.",
            "",
            f"{'quantity':<52}{'expected 95% half-width':>24}",
        ]
        for label, value in self.half_widths.items():
            lines.append(f"{label:<52}{value * 100:>22.0f} pts")
        lines += [
            "",
            "Replicate of one condition (same items, same true rates, the model sampled again):",
        ]
        for label, value in self.replicate.items():
            unit = "%" if label.startswith("share") else " pts"
            lines.append(f"{label:<52}{value * 100:>22.0f}{unit}")
        return "\n".join(lines)


def simulate(
    *, items: int = DEFAULT_ITEMS, sims: int = DEFAULT_SIMS, seed: int = 0,
    n_boot: int = N_BOOT,
) -> PowerResult:
    """Expected interval half-widths, and the replicate noise floor."""
    rng = random.Random(seed)
    one_condition, difference, correctness = [], [], []
    replicate_full, replicate_old = [], []
    for _ in range(sims):
        counts = [int(_pick(rng, CITATION_COUNTS)) for _ in range(items)]
        grounded = _draw_condition(rng, counts, VERIFIED_RATES["hybrid-rerank"])
        ungrounded = _draw_condition(rng, counts, VERIFIED_RATES["ungrounded"])
        grades = [_pick(rng, CORRECTNESS_MIX) for _ in range(items)]

        width = _half_width(bootstrap_ratio_ci(grounded, n_boot=n_boot))
        if width is not None:
            one_condition.append(width)
        _, ci = paired_ratio_diff(grounded, ungrounded, n_boot=n_boot)
        if ci is not None:
            difference.append(_half_width(ci))
        grade_ci = bootstrap_mean_ci(grades, n_boot=n_boot)
        if grade_ci is not None:
            correctness.append(_half_width(grade_ci))

        # A replicate: the same items, the same true rates, sampled again.
        for size, out in ((items, replicate_full), (OLD_REPLICATE_ITEMS, replicate_old)):
            sub = counts[:size]
            first = _draw_condition(rng, sub, VERIFIED_RATES["hybrid-rerank"])
            second = _draw_condition(rng, sub, VERIFIED_RATES["hybrid-rerank"])
            rate = lambda pairs: sum(v for v, _ in pairs) / max(1, sum(n for _, n in pairs))  # noqa: E731
            out.append(abs(rate(first) - rate(second)))

    def mean(values):
        return statistics.mean(values) if values else float("nan")

    return PowerResult(
        items=items, sims=sims, n_boot=n_boot, seed=seed,
        half_widths={
            "verified rate of one condition": mean(one_condition),
            "primary difference (hybrid-rerank minus ungrounded)": mean(difference),
            "correctness of one condition": mean(correctness),
        },
        replicate={
            f"mean |difference|, full re-run of {items} items": mean(replicate_full),
            f"mean |difference|, {OLD_REPLICATE_ITEMS}-item replicate": mean(replicate_old),
            f"share above {OLD_REPLICATE_THRESHOLD:.0%}, {OLD_REPLICATE_ITEMS}-item replicate":
                sum(1 for d in replicate_old if d > OLD_REPLICATE_THRESHOLD) / max(1, sims),
            f"share above {OLD_REPLICATE_THRESHOLD:.0%}, full re-run":
                sum(1 for d in replicate_full if d > OLD_REPLICATE_THRESHOLD) / max(1, sims),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interval half-widths and replicate noise for the answer benchmark"
    )
    parser.add_argument("--items", type=int, default=DEFAULT_ITEMS)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    print(simulate(items=args.items, sims=args.sims, seed=args.seed, n_boot=args.n_boot).render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
