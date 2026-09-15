"""Replicate check for the answer benchmark (Epic 25, pre-registered analysis).

A replicate is a second run of the same condition on **every item of the first
run** (``--conditions hybrid-rerank`` into a second run directory), not a
subset: over a handful of items the run-to-run difference is mostly item
sampling rather than the model sampling it is meant to measure, so a subset
replicate is flagged as partial.

``compare_runs`` measures how far scores move between two runs that differ
only in the model's sampling (adaptive thinking is not deterministic), which
is the noise floor a between-condition difference has to clear. It compares
judge scores, so it measures model plus judge noise, and reports the
difference in verified rate with a paired bootstrap interval over the shared
items, the same resampling the primary comparison uses.

The result is written to ``replicate-<other run id>.json`` in the first run
directory, where the report picks it up. It holds numbers and run ids only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from grounding.eval.answers.blind import cohens_kappa
from grounding.eval.answers.citations import BUCKETS
from grounding.eval.answers.runner import MANIFEST_FILE, read_jsonl, write_json
from grounding.eval.answers.stats import paired_ratio_diff
from grounding.eval.answers.scoring import SCORES_FILE

REPLICATE_SCHEMA = "grounding-answer-replicate/1"
REPLICATE_PREFIX = "replicate-"

# Provenance that must match for two runs to count as replicates.
_SAME = (
    ("answer_model", lambda m: m.get("answer_model")),
    ("judge_model", lambda m: (m.get("scoring") or {}).get("judge_model") or m.get("judge_model")),
    ("answer_prompt", lambda m: (m.get("prompts") or {}).get("answer_system_sha256")),
    ("judge_prompts", lambda m: (m.get("scoring") or {}).get("prompts")),
    ("fixture_sha256", lambda m: (m.get("fixture") or {}).get("sha256")),
    ("index_fingerprint", lambda m: (m.get("index_fingerprint") or {}).get("combined_sha256")),
    ("max_iterations", lambda m: m.get("max_iterations")),
)


def _load(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest = json.loads((Path(run_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    scores = read_jsonl(Path(run_dir) / SCORES_FILE)
    if not scores:
        raise ValueError(f"no {SCORES_FILE} in {run_dir}; score both runs first")
    return manifest, scores


def _by_item(scores, condition: str) -> Dict[str, Mapping[str, Any]]:
    return {s["item_id"]: s for s in scores if s["condition"] == condition and s.get("scored")}


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _condition(rows_a: Mapping[str, Any], rows_b: Mapping[str, Any]) -> Dict[str, Any]:
    common = sorted(set(rows_a) & set(rows_b))
    corr: List[Tuple[float, float]] = []
    declined: List[Tuple[bool, bool]] = []
    cites = {"a": [0, 0, 0], "b": [0, 0, 0]}  # verified, total, unresolvable
    # Per item, both runs' (verified, citations), so the difference in verified
    # rate carries a paired bootstrap interval like the primary comparison's.
    per_item_a: List[Tuple[int, int]] = []
    per_item_b: List[Tuple[int, int]] = []
    for item_id in common:
        a, b = rows_a[item_id], rows_b[item_id]
        ca, cb = (a.get("correctness") or {}).get("score"), (b.get("correctness") or {}).get("score")
        if ca is not None and cb is not None:
            corr.append((float(ca), float(cb)))
        da, db = (a.get("abstention") or {}).get("declined"), (b.get("abstention") or {}).get("declined")
        if da is not None and db is not None:
            declined.append((bool(da), bool(db)))
        for tag, row in (("a", a), ("b", b)):
            counts = row.get("bucket_counts") or {}
            cites[tag][0] += int(counts.get("verified", 0))
            cites[tag][1] += sum(int(counts.get(k, 0)) for k in BUCKETS)
            cites[tag][2] += int(counts.get("unresolvable", 0))
        for pairs, row in ((per_item_a, a), (per_item_b, b)):
            counts = row.get("bucket_counts") or {}
            pairs.append((int(counts.get("verified", 0)),
                          sum(int(counts.get(k, 0)) for k in BUCKETS)))
    va, ta, ua = cites["a"]
    vb, tb, ub = cites["b"]
    rate_a, rate_b = _ratio(va, ta), _ratio(vb, tb)
    check_a, check_b = _ratio(va, ta - ua), _ratio(vb, tb - ub)
    diff, diff_ci = paired_ratio_diff(per_item_a, per_item_b)
    n = len(corr)
    return {
        "n_items": len(common),
        "n_items_a": len(rows_a),
        "n_items_b": len(rows_b),
        # A replicate re-runs the condition on every item of the first run; a
        # subset measures the same noise far less precisely.
        "covers_all_items": len(common) == len(rows_a) and len(rows_a) > 0,
        "correctness": {
            "n": n,
            "exact_agreement": (sum(1 for x, y in corr if x == y) / n) if n else None,
            "cohens_kappa": cohens_kappa(corr),
            "mean_abs_diff": (sum(abs(x - y) for x, y in corr) / n) if n else None,
            "mean_a": (sum(x for x, _ in corr) / n) if n else None,
            "mean_b": (sum(y for _, y in corr) / n) if n else None,
        },
        "abstention": {
            "n": len(declined),
            "agreement": (sum(1 for x, y in declined if x == y) / len(declined)) if declined else None,
        },
        "verified_of_all": {
            "a": rate_a, "b": rate_b,
            "diff": None if rate_a is None or rate_b is None else rate_a - rate_b,
            # Paired over the shared items, the same resampling the primary
            # comparison uses, so the two numbers are read on one scale.
            "diff_ci95": diff_ci,
            "paired_diff": diff,
            "citations_a": ta, "citations_b": tb,
        },
        "verified_of_checkable": {
            "a": check_a, "b": check_b,
            "diff": None if check_a is None or check_b is None else check_a - check_b,
        },
    }


def compare_runs(run_dir: Path, other_dir: Path) -> Dict[str, Any]:
    """Run-to-run agreement for the conditions two scored runs share."""
    run_dir, other_dir = Path(run_dir), Path(other_dir)
    manifest_a, scores_a = _load(run_dir)
    manifest_b, scores_b = _load(other_dir)
    differences = [name for name, get in _SAME if get(manifest_a) != get(manifest_b)]
    names_b = {c["name"] for c in manifest_b.get("conditions", [])}
    shared = [c["name"] for c in manifest_a.get("conditions", []) if c["name"] in names_b]
    if not shared:
        raise ValueError("the two runs share no condition")
    result = {
        "schema": REPLICATE_SCHEMA,
        "run_a": manifest_a.get("run_id"),
        "run_b": manifest_b.get("run_id"),
        "comparable": not differences,
        "differences": differences,
        "conditions": {
            c: _condition(_by_item(scores_a, c), _by_item(scores_b, c)) for c in shared
        },
    }
    write_json(run_dir / f"{REPLICATE_PREFIX}{manifest_b.get('run_id')}.json", result)
    return result


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_replicate_summary(result: Mapping[str, Any]) -> str:
    lines = [f"replicate check: {result['run_a']} vs {result['run_b']}"
             + ("" if result["comparable"] else
                f" (NOT comparable, differs in: {', '.join(result['differences'])})")]
    for condition, st in result["conditions"].items():
        corr, ver = st["correctness"], st["verified_of_all"]
        diff = "n/a" if ver["diff"] is None else f"{ver['diff'] * 100:+.0f} pts"
        mad = "n/a" if corr["mean_abs_diff"] is None else f"{corr['mean_abs_diff']:.2f}"
        interval = ("" if not ver.get("diff_ci95") else
                    f" [95% CI {ver['diff_ci95'][0] * 100:+.0f} to {ver['diff_ci95'][1] * 100:+.0f}]")
        scope = "" if st.get("covers_all_items") else (
            f" (PARTIAL replicate: {st['n_items']} of {st['n_items_a']} items)")
        lines.append(
            f"  {condition}: {st['n_items']} shared items{scope}; correctness exact agreement "
            f"{_pct(corr['exact_agreement'])} (n={corr['n']}), mean |diff| {mad}; "
            f"verified {_pct(ver['a'])} vs {_pct(ver['b'])} ({diff}{interval})"
        )
    return "\n".join(lines)
