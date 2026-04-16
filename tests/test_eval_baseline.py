"""Tests for grounding.eval.baseline (Story 16.3 — baseline diff)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounding.eval.baseline import BaselineError, diff, load_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline_payload(**overrides) -> dict:
    base = {
        "format_version": 1,
        "agent": "data-scientist",
        "captured_utc": "2026-04-13T15:00:00+00:00",
        "fixture_version": 1,
        "aggregate": {
            "recall_at_1": 0.50,
            "recall_at_3": 0.70,
            "recall_at_5": 0.80,
            "recall_at_10": 0.90,
            "mrr": 0.60,
            "ndcg_at_10": 0.70,
            "per_tag": {
                "methodology": {
                    "recall_at_5": 0.80, "mrr": 0.62, "n_items": 5, "low_sample": False,
                }
            },
        },
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, payload: dict, name: str = "baseline.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------

def test_load_baseline_valid(tmp_path: Path):
    path = _write(tmp_path, _baseline_payload())
    bl = load_baseline(path)
    assert bl.agent == "data-scientist"
    assert bl.fixture_version == 1
    assert bl.aggregate["recall_at_5"] == 0.80


def test_load_baseline_missing_file_raises(tmp_path: Path):
    with pytest.raises(BaselineError, match="not found"):
        load_baseline(tmp_path / "nope.json")


def test_load_baseline_format_version_mismatch_raises(tmp_path: Path):
    path = _write(tmp_path, _baseline_payload(format_version=999))
    with pytest.raises(BaselineError, match="format_version"):
        load_baseline(path)


def test_load_baseline_malformed_json_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(BaselineError, match="malformed"):
        load_baseline(path)


def test_load_baseline_missing_aggregate_raises(tmp_path: Path):
    payload = _baseline_payload()
    del payload["aggregate"]
    path = _write(tmp_path, payload)
    with pytest.raises(BaselineError, match="aggregate"):
        load_baseline(path)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic_signed_deltas():
    current = {
        "recall_at_1": 0.55, "recall_at_3": 0.70, "recall_at_5": 0.78,
        "recall_at_10": 0.92, "mrr": 0.62, "ndcg_at_10": 0.74,
    }
    baseline = {
        "recall_at_1": 0.50, "recall_at_3": 0.70, "recall_at_5": 0.80,
        "recall_at_10": 0.90, "mrr": 0.60, "ndcg_at_10": 0.70,
    }
    d = diff(current, baseline)
    assert d.aggregate_deltas["recall_at_1"] == pytest.approx(0.05)
    assert d.aggregate_deltas["recall_at_3"] == pytest.approx(0.0)
    assert d.aggregate_deltas["recall_at_5"] == pytest.approx(-0.02)
    assert d.aggregate_deltas["recall_at_10"] == pytest.approx(0.02)
    # worst_drop is the largest baseline-current drop, clamped at 0.
    assert d.worst_drop == pytest.approx(0.02)
    assert d.worst_drop_metric == "recall_at_5"


def test_worst_drop_clamps_improvements_to_zero():
    current = {
        "recall_at_1": 0.99, "recall_at_3": 0.99, "recall_at_5": 0.99,
        "recall_at_10": 0.99, "mrr": 0.99, "ndcg_at_10": 0.99,
    }
    baseline = {
        "recall_at_1": 0.50, "recall_at_3": 0.50, "recall_at_5": 0.50,
        "recall_at_10": 0.50, "mrr": 0.50, "ndcg_at_10": 0.50,
    }
    d = diff(current, baseline)
    assert d.worst_drop == 0.0
    assert d.worst_drop_metric == ""


def test_diff_per_tag_contributes_to_worst_drop():
    current = {
        "recall_at_5": 0.80, "mrr": 0.60,
        "per_tag": {
            "methodology": {"recall_at_5": 0.50, "mrr": 0.60},
        },
    }
    baseline = {
        "recall_at_5": 0.80, "mrr": 0.60,
        "per_tag": {
            "methodology": {"recall_at_5": 0.80, "mrr": 0.60},
        },
    }
    d = diff(current, baseline)
    assert d.per_tag_deltas["methodology"]["recall_at_5"] == pytest.approx(-0.30)
    assert d.worst_drop == pytest.approx(0.30)
    assert d.worst_drop_metric == "per_tag.methodology.recall_at_5"


def test_diff_ignores_missing_keys():
    current = {"recall_at_1": 0.5}
    baseline = {"mrr": 0.5}
    d = diff(current, baseline)
    assert d.aggregate_deltas == {}
    assert d.worst_drop == 0.0


# ---------------------------------------------------------------------------
# fail_under policy
# ---------------------------------------------------------------------------

def test_fail_under_policy_pass():
    current = {"recall_at_5": 0.79}
    baseline = {"recall_at_5": 0.80}
    d = diff(current, baseline)
    # 0.01 drop, fail_under=0.02 → PASS
    assert d.passes(0.02) is True


def test_fail_under_policy_fail():
    current = {"recall_at_5": 0.70}
    baseline = {"recall_at_5": 0.80}
    d = diff(current, baseline)
    # 0.10 drop, fail_under=0.05 → FAIL
    assert d.passes(0.05) is False


def test_load_baseline_accepts_v1_and_coerces_citation_fields(tmp_path: Path):
    # v1 file has no citation_accuracy / n_citation_items.
    path = _write(tmp_path, _baseline_payload())
    bl = load_baseline(path)
    # Coerced defaults so callers can read them uniformly.
    assert bl.aggregate["citation_accuracy"] is None
    assert bl.aggregate["n_citation_items"] == 0


def test_load_baseline_accepts_v2(tmp_path: Path):
    payload = _baseline_payload(format_version=2)
    payload["aggregate"]["citation_accuracy"] = 0.8
    payload["aggregate"]["n_citation_items"] = 5
    path = _write(tmp_path, payload)
    bl = load_baseline(path)
    assert bl.aggregate["citation_accuracy"] == 0.8
    assert bl.aggregate["n_citation_items"] == 5


def test_worst_drop_includes_citation_accuracy():
    current = {"citation_accuracy": 0.70}
    baseline = {"citation_accuracy": 0.90}
    d = diff(current, baseline)
    assert d.aggregate_deltas["citation_accuracy"] == pytest.approx(-0.20)
    assert d.worst_drop == pytest.approx(0.20)
    assert d.worst_drop_metric == "citation_accuracy"


def test_worst_drop_skips_citation_when_one_side_missing(caplog):
    import logging

    current = {"citation_accuracy": 0.9, "recall_at_5": 0.5}
    baseline = {"citation_accuracy": None, "recall_at_5": 0.5}
    with caplog.at_level(logging.INFO, logger="grounding.eval.baseline"):
        d = diff(current, baseline)
    # citation_accuracy is excluded; worst_drop stays 0 for recall_at_5 tie.
    assert "citation_accuracy" not in d.aggregate_deltas
    assert d.worst_drop == 0.0
    assert any(
        "citation_accuracy" in record.message for record in caplog.records
    )


def test_fail_under_zero_strict_policy():
    current = {"recall_at_5": 0.799}
    baseline = {"recall_at_5": 0.800}
    d = diff(current, baseline)
    # Any drop > 0 fails when fail_under=0.0.
    assert d.passes(0.0) is False
