"""Baseline loading and diff computation (Story 16.3, extended in Story 17.4).

A *baseline file* is a JSON document with the following shape (v2)::

    {
      "format_version": 2,
      "agent": "<agent-name>",
      "captured_utc": "<ISO8601>",
      "fixture_version": 1,
      "aggregate": { ... }   # same shape as EvalRun.aggregate
    }

``format_version: 1`` baselines are still accepted; the loader treats missing
``citation_accuracy`` / ``n_citation_items`` as absent, and ``diff()`` skips
asymmetric metrics from ``worst_drop`` (with an INFO log).

The diff computes per-metric and per-tag deltas (current minus baseline) and
a ``worst_drop`` value used for the ``--fail-under`` exit policy. Improvements
(positive deltas) are clamped to 0 when computing ``worst_drop``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger("grounding.eval.baseline")

FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = (1, 2)

# Aggregate keys that participate in worst_drop (excludes counts).
_AGG_METRIC_KEYS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "citation_accuracy",
)

# Per-tag keys that participate in worst_drop (excludes n_items, low_sample).
_TAG_METRIC_KEYS = ("recall_at_5", "mrr")


class BaselineError(Exception):
    """Raised when a baseline file is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class Baseline:
    """Parsed baseline file."""

    agent: str
    captured_utc: str
    fixture_version: int
    aggregate: dict[str, Any]
    source_path: Path


@dataclass(frozen=True)
class AggregateDiff:
    """Result of comparing a current EvalAggregate against a baseline aggregate.

    ``aggregate_deltas`` and ``per_tag_deltas`` are signed (current - baseline);
    ``worst_drop`` is the maximum of (baseline - current) across all metrics,
    clamped to 0 (improvements never count as drops).
    """

    aggregate_deltas: dict[str, float] = field(default_factory=dict)
    per_tag_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    worst_drop: float = 0.0
    worst_drop_metric: str = ""

    def passes(self, fail_under: float) -> bool:
        """Return True iff worst_drop does not exceed ``fail_under``."""
        return self.worst_drop <= fail_under


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> Baseline:
    """Load and validate a baseline JSON file.

    Raises ``BaselineError`` for missing files, malformed JSON, missing fields,
    or ``format_version`` mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise BaselineError(f"baseline file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"malformed baseline JSON {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise BaselineError(f"baseline {path}: top-level must be a JSON object")

    fv = raw.get("format_version")
    if fv not in _SUPPORTED_FORMAT_VERSIONS:
        raise BaselineError(
            f"baseline {path}: unsupported format_version {fv!r}; "
            f"expected one of {_SUPPORTED_FORMAT_VERSIONS}"
        )

    for key in ("agent", "aggregate"):
        if key not in raw:
            raise BaselineError(f"baseline {path}: missing required field '{key}'")

    aggregate = raw["aggregate"]
    if not isinstance(aggregate, dict):
        raise BaselineError(f"baseline {path}: 'aggregate' must be an object")

    # v1 → v2 in-memory coercion: absent citation fields read as (None, 0).
    if fv == 1:
        aggregate = dict(aggregate)
        aggregate.setdefault("citation_accuracy", None)
        aggregate.setdefault("n_citation_items", 0)

    return Baseline(
        agent=str(raw["agent"]),
        captured_utc=str(raw.get("captured_utc", "")),
        fixture_version=int(raw.get("fixture_version", 0)),
        aggregate=aggregate,
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff(current_aggregate: Mapping[str, Any], baseline_aggregate: Mapping[str, Any]) -> AggregateDiff:
    """Compute per-metric / per-tag deltas and worst_drop.

    Both inputs are aggregate-shaped mappings (typically the ``aggregate``
    subtree of an EvalRun JSON dict and of a baseline file).
    """
    aggregate_deltas: dict[str, float] = {}
    worst_drop = 0.0
    worst_drop_metric = ""

    for key in _AGG_METRIC_KEYS:
        cur_val = current_aggregate.get(key)
        base_val = baseline_aggregate.get(key)
        if cur_val is None or base_val is None:
            if (cur_val is None) != (base_val is None):
                logger.info(
                    "diff: skipping metric '%s' from worst_drop (present on only one side)",
                    key,
                )
            continue
        cur = float(cur_val)
        base = float(base_val)
        delta = cur - base
        aggregate_deltas[key] = delta
        drop = max(0.0, -delta)
        if drop > worst_drop:
            worst_drop = drop
            worst_drop_metric = key

    per_tag_deltas: dict[str, dict[str, float]] = {}
    cur_tags = current_aggregate.get("per_tag") or {}
    base_tags = baseline_aggregate.get("per_tag") or {}
    for tag in sorted(set(cur_tags) & set(base_tags)):
        cur_tag = cur_tags[tag]
        base_tag = base_tags[tag]
        tag_deltas: dict[str, float] = {}
        for tk in _TAG_METRIC_KEYS:
            if tk not in cur_tag or tk not in base_tag:
                continue
            delta = float(cur_tag[tk]) - float(base_tag[tk])
            tag_deltas[tk] = delta
            drop = max(0.0, -delta)
            if drop > worst_drop:
                worst_drop = drop
                worst_drop_metric = f"per_tag.{tag}.{tk}"
        if tag_deltas:
            per_tag_deltas[tag] = tag_deltas

    return AggregateDiff(
        aggregate_deltas=aggregate_deltas,
        per_tag_deltas=per_tag_deltas,
        worst_drop=worst_drop,
        worst_drop_metric=worst_drop_metric,
    )
