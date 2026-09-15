"""Reports for the grounded-answer benchmark (Epic 25, Story 25.4).

``build_report`` aggregates a scored run directory into one JSON-ready dict:
the pre-registered primary comparison; per-condition correctness (human and
judge), citation buckets with both denominators, abstention per unanswerable
kind, retrieval recall, retrieval-versus-generation attribution, tokens, cost
and latency; a per-category breakdown; the worst failures by item id; judge
agreement once blind grades are imported; replicate agreement; and run
health. The same dict renders to ``report.md`` and ``report.json``, plus
``chart.png``.

Correctness: when a human has graded every scored answerable answer (the
default blind export is every answer), the human grades are the primary
correctness metric and the judge's are secondary; otherwise the judge's are
primary and human coverage is shown.

Citations: ``verified`` of all citations is the headline rate, and verified
of checkable citations (all minus ``unresolvable``) sits beside it, because
``unresolvable`` means the scorer could not check the citation against the
library, not that it is wrong. Unresolvable citations are broken down by
reason, and reasons that come from missing fixture metadata (no page offset,
unknown edition, section-paged, no page index, no text at the page, an
ambiguous title) are counted as bookkeeping.

Uncertainty: every rate carries a 95 percent percentile bootstrap interval
over items (fixed seed, so reports are reproducible). Citation rates resample
answers, not citations, because citations within one answer are not
independent. Paired differences resample items and keep both conditions'
answers to an item together.

Publishable mode (D7) writes ``publishable/`` with aggregate scores only: no
answer text, claims, passages, citation strings, judge reasons, gold answers
or local paths, and question texts only with ``include_questions`` (after
review). ``tests/test_eval_answers_report.py`` proves no chunk text reaches
those files.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from grounding.eval.answers.blind import AGREEMENT_FILE, HUMAN_GRADES_FILE, PUBLISH_GATE
from grounding.eval.answers.citations import BUCKETS
from grounding.eval.fixtures import PUBLIC_DOMAIN
from grounding.eval.answers.pricing import usage_cost
from grounding.eval.answers.replicate import REPLICATE_PREFIX
from grounding.eval.answers.runner import (
    ERRORS_FILE,
    MANIFEST_FILE,
    RETRIEVAL_FILE,
    TRANSCRIPTS_FILE,
    read_jsonl,
    write_json,
)
from grounding.eval.answers.scoring import (
    SCORES_FILE,
    abstained_as_desired,
    desired_behavior,
)
# One implementation of the resampling, shared with the replicate check and the
# power simulation (grounding/eval/answers/power.py).
from grounding.eval.answers.stats import (  # noqa: F401  (re-exported)
    N_BOOT,
    SEED,
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    paired_ratio_diff,
    percentile as _percentile,
)

logger = logging.getLogger("grounding.eval.answers.report")

REPORT_SCHEMA = "grounding-answer-report/2"
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
CHART_PNG = "chart.png"
PUBLISH_DIR = "publishable"
WORST_LIMIT = 10
SMALL_N = 10  # below this, rates render as counts ("3/4"), never bare percentages

# The pre-registered primary comparison (docs/epics/epic-25-grounded-answer-benchmark.md).
PRIMARY_CONDITIONS = ("hybrid-rerank", "ungrounded")
# Unresolvable reasons that come from missing fixture metadata, not from the model.
# Unresolvable reasons a fixture edit can fix: declare the offset, the edition
# or revision, the identifier, or re-ingest the document with pages. NOT
# section_paged: a work paged "5-20" has no printed page to map, so no fixture
# entry makes those citations checkable, and counting it here would let a
# corpus of section-paged handbooks fail the gate with nothing to fix.
BOOKKEEPING_DETAILS = (
    "no_page_offset", "edition_unknown", "identifier_unknown",
    "work_has_no_page_index", "no_text_at_page", "ambiguous_title",
)
BOOKKEEPING_MAX_SHARE = 0.10

# Reference categorical palette slot 1 and the chart chrome tokens. Every
# panel of the chart is one series, named by its title, so one hue suffices.
_SERIES = ("#2a78d6",)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _paired_delta(mine: Mapping[str, float], base: Mapping[str, float]) -> Dict[str, Any] | None:
    common = sorted(set(mine) & set(base))
    diffs = [mine[i] - base[i] for i in common]
    if not diffs:
        return None
    return {"mean": _mean(diffs), "ci95": bootstrap_mean_ci(diffs), "n_paired": len(diffs)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _load(run_dir: Path) -> Dict[str, Any]:
    def _json(name: str):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    replicates = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted(run_dir.glob(f"{REPLICATE_PREFIX}*.json"))]
    return {
        "manifest": json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8")),
        "transcripts": read_jsonl(run_dir / TRANSCRIPTS_FILE),
        "scores": read_jsonl(run_dir / SCORES_FILE),
        "errors": read_jsonl(run_dir / ERRORS_FILE),
        "retrieval": _json(RETRIEVAL_FILE) or {},
        "agreement": _json(AGREEMENT_FILE),
        "human": _json(HUMAN_GRADES_FILE),
        "replicates": replicates,
    }


def _human_map(human: Mapping[str, Any] | None) -> Dict[Tuple[str, str], float]:
    if not human:
        return {}
    return {(h["item_id"], h["condition"]): float(h["human_grade"]) for h in human.get("answers", [])}


def _correctness_by_item(scores: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        s["item_id"]: s["correctness"]["score"]
        for s in scores
        if s.get("answerable") and s.get("correctness") and s["correctness"].get("score") is not None
    }


def _human_by_item(rows: Sequence[Mapping[str, Any]], human: Mapping[Tuple[str, str], float]):
    return {s["item_id"]: human[(s["item_id"], s["condition"])] for s in rows
            if s.get("answerable") and s.get("scored") and (s["item_id"], s["condition"]) in human}


def _claim_words(citation: Mapping[str, Any]) -> int:
    if citation.get("claim_words") is not None:
        return int(citation["claim_words"])
    return len(re.findall(r"\w+", citation.get("claim") or ""))


def _citation_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {b: 0 for b in BUCKETS}
    per_answer: List[Tuple[int, int]] = []
    per_answer_checkable: List[Tuple[int, int]] = []
    unjudged = 0
    matches: Dict[str, int] = {}
    notes: Dict[str, int] = {}
    unresolvable_by_detail: Dict[str, int] = {}
    claim_words: List[int] = []
    scored_rows = [s for s in rows if s.get("scored")]
    for s in scored_rows:
        answer_counts = s.get("bucket_counts") or {}
        for b in BUCKETS:
            counts[b] += int(answer_counts.get(b, 0))
        total = sum(int(answer_counts.get(b, 0)) for b in BUCKETS)
        verified = int(answer_counts.get("verified", 0))
        per_answer.append((verified, total))
        per_answer_checkable.append((verified, total - int(answer_counts.get("unresolvable", 0))))
        unjudged += int(s.get("n_unjudged_citations", 0))
        for cit in s.get("citations") or []:
            res = cit.get("resolution") or {}
            if res.get("match"):
                matches[res["match"]] = matches.get(res["match"], 0) + 1
            for note in res.get("notes") or []:
                key = note.split(":", 1)[0]
                notes[key] = notes.get(key, 0) + 1
            if cit.get("bucket") == "unresolvable":
                detail = res.get("detail") or "unknown"
                unresolvable_by_detail[detail] = unresolvable_by_detail.get(detail, 0) + 1
            claim_words.append(_claim_words(cit))
    n_cit = sum(counts.values())
    checkable = n_cit - counts["unresolvable"]
    bookkeeping = sum(n for d, n in unresolvable_by_detail.items() if d in BOOKKEEPING_DETAILS)
    return {
        **counts,
        "total": n_cit,
        "unjudged": unjudged,
        "partial_matches": matches.get("partial", 0),
        "label_matches": matches.get("label", 0),
        "match_types": dict(sorted(matches.items())),
        "notes": dict(sorted(notes.items())),
        "verified_rate": counts["verified"] / n_cit if n_cit else None,
        "verified_rate_ci95": bootstrap_ratio_ci([p for p in per_answer if p[1]]),
        "checkable": checkable,
        "verified_of_checkable": counts["verified"] / checkable if checkable else None,
        "verified_of_checkable_ci95": bootstrap_ratio_ci([p for p in per_answer_checkable if p[1]]),
        "unresolvable_by_detail": dict(sorted(unresolvable_by_detail.items())),
        "bookkeeping_unresolvable": bookkeeping,
        "bookkeeping_share": bookkeeping / n_cit if n_cit else None,
        "median_claim_words": statistics.median(claim_words) if claim_words else None,
        "per_answer": n_cit / len(scored_rows) if scored_rows else None,
        "answers_with_citations": sum(1 for _, t in per_answer if t),
    }


def _abstention_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Did unanswerable answers do what their kind asks for (scoring.DESIRED_ABSTENTION)?

    ``no_source``: declining. ``not_in_corpus``: acknowledging the gap, which
    the grounded prompt asks for, whether the model then answered from labeled
    general knowledge or not. ``declined`` is reported beside it.
    """
    abstain_rows = [s for s in rows if not s.get("answerable") and s.get("abstention")]
    values = [1.0 if abstained_as_desired(s["abstention"], s.get("unanswerable_kind")) else 0.0
              for s in abstain_rows
              if abstained_as_desired(s["abstention"], s.get("unanswerable_kind")) is not None]
    by_kind: Dict[str, Any] = {}
    for kind in sorted({s.get("unanswerable_kind") or "unknown" for s in abstain_rows}):
        judged = [s["abstention"] for s in abstain_rows
                  if (s.get("unanswerable_kind") or "unknown") == kind
                  and abstained_as_desired(s["abstention"], kind) is not None]
        kind_values = [1.0 if abstained_as_desired(a, kind) else 0.0 for a in judged]
        stats = {
            "desired_behavior": desired_behavior(kind),
            "met": int(sum(kind_values)),
            "n": len(kind_values),
            "rate": _mean(kind_values),
            "ci95": bootstrap_mean_ci(kind_values),
            "declined": sum(1 for a in judged if a.get("declined")),
        }
        if kind == "not_in_corpus":
            stats["acknowledged_gap"] = int(sum(kind_values))
            stats["answered_with_labeled_knowledge"] = sum(
                1 for a in judged if a.get("gave_answer") and a.get("acknowledged_gap")
            )
            stats["answered_without_flagging"] = sum(
                1 for a in judged if not a.get("acknowledged_gap")
            )
        by_kind[kind] = stats
    return {
        "rate": _mean(values),
        "ci95": bootstrap_mean_ci(values),
        "n": len(values),
        "met": int(sum(values)),
        "by_kind": by_kind,
    }


def _attribution(rows: Sequence[Mapping[str, Any]], primary: Mapping[str, float]) -> Dict[str, Any]:
    """Split not-fully-correct grounded answers by whether the gold page reached the model."""
    graded = [s for s in rows if s.get("answerable") and s.get("scored") and s["item_id"] in primary]
    out: Dict[str, Any] = {
        "n_graded": len(graded),
        "n_gold_page_in_context": sum(1 for s in graded if s.get("gold_page_in_context")),
    }
    for label, value in (("wrong", 0.0), ("partial", 0.5)):
        bucket = {"gold_not_in_context": 0, "gold_in_context": 0, "unknown": 0}
        for s in graded:
            if primary[s["item_id"]] != value:
                continue
            flag = s.get("gold_page_in_context")
            key = "unknown" if flag is None else ("gold_in_context" if flag else "gold_not_in_context")
            bucket[key] += 1
        out[label] = bucket
    return out


def _condition_stats(
    condition: str,
    transcripts: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    retrieval: Mapping[str, Any],
    *,
    grounded: bool,
    human: Mapping[Tuple[str, str], float],
    primary_source: str,
    baseline_judge: Mapping[str, float] | None,
    baseline_human: Mapping[str, float] | None,
) -> Dict[str, Any]:
    trans = [t for t in transcripts if t["condition"] == condition]
    rows = [s for s in scores if s["condition"] == condition]
    errs = [e for e in errors if e.get("condition") == condition]
    statuses = [t["status"] for t in trans]
    scored_rows = [s for s in rows if s.get("scored")]

    graded = [s for s in rows if s.get("answerable") and s.get("correctness")]
    values = [s["correctness"]["score"] for s in graded if s["correctness"].get("score") is not None]
    methods = [s["correctness"].get("method") for s in graded]
    declined_known = [s for s in graded if s["correctness"].get("declined") is not None
                      and s["correctness"].get("score") is not None]
    n_declined = sum(1 for s in declined_known if s["correctness"]["declined"])
    # Confidently wrong: a zero that did not decline (an empty answer is not confident).
    confidently_wrong = [s for s in declined_known
                         if s["correctness"]["score"] == 0 and not s["correctness"]["declined"]
                         and s["correctness"].get("method") != "empty_answer"]
    judge_by_item = _correctness_by_item(rows)

    human_by_item = _human_by_item(rows, human)
    eligible = [s for s in rows if s.get("answerable") and s.get("scored")]
    human_values = list(human_by_item.values())
    human_block = None
    if human_values:
        human_block = {
            "mean": _mean(human_values),
            "ci95": bootstrap_mean_ci(human_values),
            "n": len(human_values),
            "eligible": len(eligible),
            "delta_vs_ungrounded": _paired_delta(human_by_item, baseline_human)
            if baseline_human is not None else None,
        }
    judge_delta = _paired_delta(judge_by_item, baseline_judge) if baseline_judge is not None else None
    if primary_source == "human" and human_block is not None:
        primary = {"source": "human", **{k: human_block[k] for k in
                                         ("mean", "ci95", "n", "delta_vs_ungrounded")}}
        primary_by_item = human_by_item
    else:
        primary = {"source": "judge", "mean": _mean(values), "ci95": bootstrap_mean_ci(values),
                   "n": len(values), "delta_vs_ungrounded": judge_delta}
        primary_by_item = judge_by_item

    human_abstention = None
    unans = [s for s in rows if not s.get("answerable") and s.get("scored")
             and (s["item_id"], condition) in human]
    if unans:
        met = [human[(s["item_id"], condition)] == 1.0 for s in unans]
        human_abstention = {"met": sum(met), "n": len(met), "rate": sum(met) / len(met)}

    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    answer_cost = 0.0
    for t in trans:
        for k in usage:
            usage[k] += int((t.get("usage") or {}).get(k, 0))
        answer_cost += t.get("cost_usd") or 0.0
    for e in errs:
        if e.get("stage") == "answer":
            answer_cost += e.get("cost_usd") or 0.0
    judge_usage = {"input_tokens": 0, "output_tokens": 0}
    judge_cost = 0.0
    for s in rows:
        for call in s.get("judge_calls") or []:
            for k in judge_usage:
                judge_usage[k] += int((call.get("usage") or {}).get(k, 0))
            judge_cost += call.get("cost_usd") or 0.0
    for e in errs:
        if str(e.get("stage", "")).startswith("judge") and e.get("usage"):
            judge_cost += usage_cost(e.get("model") or "", e["usage"]) or 0.0

    latencies = sorted(t["latency_s"] for t in trans if t.get("latency_s") is not None)
    recall = (retrieval.get("conditions") or {}).get(condition) or {}

    return {
        "n_answers": len(trans),
        "n_scored": len(scored_rows),
        "n_excluded": len([s for s in rows if s.get("excluded")]),
        "n_errors": len([e for e in errs if e.get("stage") == "answer"]),
        "n_judge_errors": len([e for e in errs if str(e.get("stage", "")).startswith("judge")]),
        "n_refusal": statuses.count("refusal"),
        "n_truncated": statuses.count("truncated"),
        "n_max_iterations": statuses.count("max_iterations"),
        "primary_correctness": primary,
        "correctness": {
            "source": "judge",
            "mean": _mean(values),
            "ci95": bootstrap_mean_ci(values),
            "n": len(values),
            "n_numeric": methods.count("numeric"),
            "n_judge": methods.count("judge"),
            "n_empty": methods.count("empty_answer"),
            "n_unjudged": methods.count("judge_error"),
            "n_declined": n_declined,
            "declined_rate": n_declined / len(declined_known) if declined_known else None,
            "n_confidently_wrong": len(confidently_wrong),
            "confidently_wrong_rate": len(confidently_wrong) / len(declined_known)
            if declined_known else None,
            "n_with_declined_flag": len(declined_known),
            "delta_vs_ungrounded": judge_delta,
        },
        "human_correctness": human_block,
        "citations": _citation_stats(rows),
        "abstention": _abstention_stats(rows),
        "human_abstention": human_abstention,
        "attribution": _attribution(rows, primary_by_item) if grounded else None,
        "retrieval": {
            "recall_at_5_gold_page": recall.get("recall_at_5_gold_page"),
            "recall_at_5_doc": recall.get("recall_at_5_doc"),
            "n_items": recall.get("n_items"),
        } if recall else None,
        "tokens": {"answer": usage, "judge": judge_usage},
        "cost_usd": {
            "answer": round(answer_cost, 6),
            "judge": round(judge_cost, 6),
            "per_answer": round(answer_cost / len(trans), 6) if trans else None,
        },
        "latency_s": {
            "median": _percentile(latencies, 0.5) if latencies else None,
            "p90": _percentile(latencies, 0.9) if latencies else None,
        },
        "tool_calls_per_answer": _mean([len(t.get("tool_calls") or []) for t in trans]),
    }


def _answerable_rows(scores: Sequence[Mapping[str, Any]], condition: str) -> List[Mapping[str, Any]]:
    """Scored answers to answerable items in one condition: what the primary comparison reads."""
    return [s for s in scores
            if s["condition"] == condition and s.get("scored") and s.get("answerable")]


def _item_citations(scores: Sequence[Mapping[str, Any]], condition: str) -> Dict[str, Tuple[int, int, int]]:
    out = {}
    for s in _answerable_rows(scores, condition):
        counts = s.get("bucket_counts") or {}
        out[s["item_id"]] = (int(counts.get("verified", 0)),
                             sum(int(counts.get(b, 0)) for b in BUCKETS),
                             int(counts.get("unresolvable", 0)))
    return out


def _primary_comparison(scores, conditions) -> Dict[str, Any] | None:
    """Pre-registered: hybrid-rerank minus ungrounded, verified rate of all citations.

    **Answerable items only.** An unanswerable question asks the model not to
    answer, so the citations in an answer to one measure something else; they
    are reported in the per-condition table and in abstention instead.
    """
    a, b = PRIMARY_CONDITIONS
    if a not in conditions or b not in conditions:
        return None
    ca, cb = _item_citations(scores, a), _item_citations(scores, b)
    common = sorted(set(ca) & set(cb))
    all_a = [(ca[i][0], ca[i][1]) for i in common]
    all_b = [(cb[i][0], cb[i][1]) for i in common]
    chk_a = [(ca[i][0], ca[i][1] - ca[i][2]) for i in common]
    chk_b = [(cb[i][0], cb[i][1] - cb[i][2]) for i in common]
    estimate, ci = paired_ratio_diff(all_a, all_b)
    estimate_chk, ci_chk = paired_ratio_diff(chk_a, chk_b)
    # The metadata gate reads the same citations as the comparison it protects.
    baseline = _citation_stats(_answerable_rows(scores, b))
    share = baseline.get("bookkeeping_share")
    by_detail = baseline.get("unresolvable_by_detail") or {}

    def side(pairs):
        num, den = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        return {"verified": num, "citations": den, "rate": num / den if den else None}

    return {
        "conditions": [a, b],
        "metric": "verified_of_all",
        "items": "answerable only",
        "estimate": estimate,
        "ci95": ci,
        "n_items": len(common),
        a: side(all_a),
        b: side(all_b),
        "secondary_verified_of_checkable": {"estimate": estimate_chk, "ci95": ci_chk,
                                            a: side(chk_a), b: side(chk_b)},
        "bookkeeping_share_ungrounded": share,
        "bookkeeping_ok": share is None or share <= BOOKKEEPING_MAX_SHARE,
        "bookkeeping_max_share": BOOKKEEPING_MAX_SHARE,
        "bookkeeping_by_detail_ungrounded": {d: n for d, n in sorted(by_detail.items())
                                             if d in BOOKKEEPING_DETAILS},
        # Not a metadata gap: the model cited a work the library does not hold.
        "work_not_in_corpus_ungrounded": int(by_detail.get("work_not_in_corpus", 0)),
        "unresolvable_ungrounded": int(baseline.get("unresolvable", 0)),
    }


def _category_stats(scores: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for category in sorted({s.get("category") or "uncategorized" for s in scores}):
        out[category] = {}
        for condition in conditions:
            rows = [s for s in scores if s["condition"] == condition
                    and (s.get("category") or "uncategorized") == category and s.get("scored")]
            if not rows:
                continue
            corr = [s["correctness"]["score"] for s in rows
                    if s.get("correctness") and s["correctness"].get("score") is not None]
            abst = [1.0 if abstained_as_desired(s["abstention"], s.get("unanswerable_kind"))
                    else 0.0 for s in rows if s.get("abstention")
                    and abstained_as_desired(s["abstention"], s.get("unanswerable_kind"))
                    is not None]
            verified = sum(int(s["bucket_counts"].get("verified", 0)) for s in rows)
            total = sum(sum(int(v) for v in s["bucket_counts"].values()) for s in rows)
            out[category][condition] = {
                "n": len(rows),
                "correctness": _mean(corr),
                "n_correctness": len(corr),
                "abstention": _mean(abst),
                "n_abstention": len(abst),
                "declined": int(sum(abst)),
                "verified_rate": verified / total if total else None,
                "verified": verified,
                "citations": total,
            }
    return out


def _worst_failures(scores: Sequence[Mapping[str, Any]], order: Sequence[str]) -> List[Dict[str, Any]]:
    ranked = []
    for s in scores:
        if not s.get("scored"):
            continue
        counts = s.get("bucket_counts") or {}
        why, severity = [], 0.0
        corr = s.get("correctness") or {}
        if s.get("answerable"):
            score = corr.get("score")
            if score is not None and score < 1:
                severity += 2 * (1 - score)
                label = "declined" if corr.get("declined") else f"scored {score:g}"
                why.append(f"correctness {label} ({corr.get('method')})")
        elif abstained_as_desired(s.get("abstention"), s.get("unanswerable_kind")) is False:
            severity += 2
            kind = s.get("unanswerable_kind")
            why.append("answered an unanswerable question"
                       if desired_behavior(kind) == "declined"
                       else "answered without flagging the gap in the sources"
                       + f" ({kind})")
        if counts.get("invented"):
            severity += counts["invented"]
            why.append(f"{counts['invented']} invented citation(s)")
        if counts.get("unsupported"):
            severity += 0.5 * counts["unsupported"]
            why.append(f"{counts['unsupported']} unsupported citation(s)")
        if severity > 0:
            ranked.append((severity, s, why))
    rank = {c: i for i, c in enumerate(order)}
    ranked.sort(key=lambda r: (-r[0], r[1]["item_id"], rank.get(r[1]["condition"], 99)))
    return [
        {
            "item_id": s["item_id"],
            "condition": s["condition"],
            "category": s.get("category"),
            "severity": round(sev, 3),
            "correctness": (s.get("correctness") or {}).get("score"),
            "declined": (s.get("correctness") or s.get("abstention") or {}).get("declined"),
            "buckets": {b: int((s.get("bucket_counts") or {}).get(b, 0)) for b in BUCKETS},
            "why": "; ".join(why),
            "judge_reason": (s.get("correctness") or s.get("abstention") or {}).get("reason"),
        }
        for sev, s, why in ranked[:WORST_LIMIT]
    ]


def _human_coverage(scores, human) -> Dict[str, Any]:
    out = {}
    for label, answerable in (("answerable", True), ("unanswerable", False)):
        eligible = [s for s in scores if s.get("scored") and bool(s.get("answerable")) == answerable]
        graded = [s for s in eligible if (s["item_id"], s["condition"]) in human]
        out[label] = {"graded": len(graded), "eligible": len(eligible)}
    return out


def build_report(run_dir: Path) -> Dict[str, Any]:
    """Aggregate a scored run directory into the report dict."""
    run_dir = Path(run_dir)
    data = _load(run_dir)
    manifest = data["manifest"]
    order = [c["name"] for c in manifest.get("conditions", [])]
    grounded = {c["name"]: bool(c.get("grounded")) for c in manifest.get("conditions", [])}
    human = _human_map(data["human"])
    coverage = _human_coverage(data["scores"], human)
    answerable_cov = coverage["answerable"]
    primary_source = ("human" if answerable_cov["eligible"]
                      and answerable_cov["graded"] == answerable_cov["eligible"] else "judge")
    baseline_judge = baseline_human = None
    if "ungrounded" in order:
        ungrounded_rows = [s for s in data["scores"] if s["condition"] == "ungrounded"]
        baseline_judge = _correctness_by_item(ungrounded_rows)
        baseline_human = _human_by_item(ungrounded_rows, human)
    per_condition = {
        c: _condition_stats(
            c, data["transcripts"], data["scores"], data["errors"], data["retrieval"],
            grounded=grounded.get(c, c != "ungrounded"),
            human=human,
            primary_source=primary_source,
            baseline_judge=baseline_judge if c != "ungrounded" else None,
            baseline_human=baseline_human if c != "ungrounded" else None,
        )
        for c in order
    }
    scoring = manifest.get("scoring") or {}
    answer_model = manifest.get("answer_model")
    judge_model = scoring.get("judge_model") or manifest.get("judge_model")
    items = manifest.get("fixture", {}).get("selected_item_ids", [])
    unanswerable = {s["item_id"] for s in data["scores"] if not s.get("answerable")}
    questions = {}
    for t in data["transcripts"]:
        questions.setdefault(t["item_id"], t.get("question"))

    warnings = []
    if primary_source == "judge" and answerable_cov["graded"]:
        # Human and judge grades are never mixed: until every answerable answer
        # is graded, the judge's are primary and the human's are shown beside them.
        warnings.append(
            f"Human grades cover {answerable_cov['graded']} of {answerable_cov['eligible']} "
            "scored answerable answers, so correctness is reported judge-primary. Grade the "
            "rest (blind/answers.csv) to make the human grades primary; the two are never "
            "mixed."
        )
    if judge_model and judge_model == answer_model:
        warnings.append(
            f"The judge ({judge_model}) is the answer model. Self-preference is possible; "
            "treat judge scores as provisional until blind-grade agreement is imported."
        )
    if manifest.get("status") != "complete":
        warnings.append(f"Run status is '{manifest.get('status')}'; some answers were not run.")
    if not data["scores"]:
        warnings.append("No scores yet; run --run-dir <dir> --score.")
    if data["retrieval"].get("error"):
        warnings.append(f"Retrieval recall failed: {data['retrieval']['error']}")
    embedding = (manifest.get("environment") or {}).get("embedding") or {}
    if embedding.get("query_model") and embedding.get("index_model") \
            and embedding["query_model"] != embedding["index_model"]:
        warnings.append(
            f"Queries are embedded with {embedding['query_model']} but the index was built "
            f"with {embedding['index_model']}; dense retrieval scores are not comparable."
        )
    degraded = sorted({
        t["condition"] for t in data["transcripts"]
        for call in t.get("tool_calls") or [] for r in call.get("results") or []
        if r.get("hybrid_degraded")
    })
    if degraded:
        warnings.append(f"Hybrid retrieval fell back to dense-only in: {', '.join(degraded)}.")
    primary = _primary_comparison(data["scores"], order)
    if primary is not None and not primary["bookkeeping_ok"]:
        warnings.append(
            f"{primary['bookkeeping_share_ungrounded']:.0%} of ungrounded citations are "
            "unresolvable for fixture-metadata reasons (above the pre-registered "
            f"{BOOKKEEPING_MAX_SHARE:.0%}); add the missing page offsets or editions and "
            "re-score before reading the primary comparison."
        )
    for rep in data["replicates"]:
        if not rep.get("comparable"):
            warnings.append(
                f"Replicate {rep.get('run_b')} differs in {', '.join(rep.get('differences') or [])}; "
                "it is not a pure replicate."
            )

    return {
        "schema": REPORT_SCHEMA,
        "run": {
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "started_utc": manifest.get("started_utc"),
            "finished_utc": manifest.get("finished_utc"),
            "agent": manifest.get("agent"),
            "answer_model": answer_model,
            "judge_model": judge_model,
            "n_items": len(items),
            "n_unanswerable": len(unanswerable),
            "conditions": order,
            "git": manifest.get("git"),
            "fixture": {k: manifest.get("fixture", {}).get(k)
                        for k in ("path", "sha256", "source_license")},
            "corpus_dir": manifest.get("corpus_dir"),
            "embeddings_dir": manifest.get("embeddings_dir"),
            "index_fingerprint": (manifest.get("index_fingerprint") or {}).get("combined_sha256"),
            "retrieval_config": {c["name"]: c.get("retrieval") for c in manifest.get("conditions", [])},
            "generation": {"answer": manifest.get("generation", {}).get("answer"),
                           "judge": scoring.get("generation")},
            "max_iterations": manifest.get("max_iterations"),
            "prompts": {"answer": manifest.get("prompts"), "judge": scoring.get("prompts")},
            "pricing_source": (manifest.get("pricing") or {}).get("source"),
            "estimate_usd": (manifest.get("estimate") or {}).get("total_usd"),
            "environment": manifest.get("environment"),
        },
        "statistics": {
            "ci": "95 percent percentile bootstrap over items (answers for citation rates)",
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "primary_correctness": primary_source,
        "human_coverage": coverage,
        "primary_comparison": primary,
        "conditions": per_condition,
        "per_category": _category_stats(data["scores"], order),
        "worst_failures": _worst_failures(data["scores"], order),
        "judge_validation": data["agreement"],
        "replicates": data["replicates"],
        "questions": questions,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Publishable view (D7)
# ---------------------------------------------------------------------------

def publishable_view(
    report: Mapping[str, Any], *, include_questions: bool = False,
    include_source_text: bool = False,
) -> Dict[str, Any]:
    """A copy that keeps aggregate scores and drops every text field.

    ``include_source_text`` keeps the judge reasons and the questions, for a
    public-domain corpus that may be published (see ``write_report``).
    """
    run = dict(report["run"])
    for key in ("corpus_dir", "embeddings_dir"):
        run.pop(key, None)
    fixture = report["run"].get("fixture") or {}
    run["fixture"] = {k: fixture.get(k) for k in ("sha256", "source_license") if fixture.get(k)}
    worst = [
        {k: v for k, v in row.items() if include_source_text or k not in ("judge_reason",)}
        for row in report["worst_failures"]
    ]
    include_questions = include_questions or include_source_text
    # Warnings are generated text, but an exception message could carry a path.
    warnings = [
        "Retrieval recall failed (details in the full report)."
        if w.startswith("Retrieval recall failed") else w
        for w in report.get("warnings") or []
    ]
    out = {
        **{k: v for k, v in report.items()
           if k not in ("questions", "worst_failures", "run", "warnings")},
        "run": run,
        "worst_failures": worst,
        "warnings": warnings,
        "publishable": True,
        "source_text_published": include_source_text,
    }
    if include_questions:
        out["questions"] = dict(report.get("questions") or {})
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _ci(ci) -> str:
    return "" if not ci else f" ({ci[0] * 100:.0f} to {ci[1] * 100:.0f})"


def _count_rate(k: float | None, n: int | None, ci=None) -> str:
    """'k/n' when n is small, else 'rate (CI) n=N'; 'n/a' when nothing was counted."""
    if not n:
        return "n/a"
    if n < SMALL_N:
        return f"{k:g}/{n}"
    return f"{_pct(k / n)}{_ci(ci)} n={n}"


def _abstention_cell(ab: Mapping[str, Any]) -> str:
    by_kind = ab.get("by_kind")
    if not by_kind:
        return f"{_pct(ab.get('rate'))} n={ab.get('n', 0)}"
    parts = []
    for kind, st in by_kind.items():
        behavior = ("acknowledged gap" if st.get("desired_behavior") == "acknowledged_gap"
                    else "declined")
        cell = (f"{kind.replace('_', ' ')} {behavior} "
                f"{_count_rate(st.get('met', 0), st['n'], st.get('ci95'))}")
        if st.get("answered_with_labeled_knowledge"):
            cell += (f" ({st['answered_with_labeled_knowledge']} answered from labeled "
                     "general knowledge)")
        parts.append(cell)
    return "; ".join(parts)


def _render_not_in_corpus(report: Mapping[str, Any], lines: List[str]) -> None:
    """Headline how the grounded conditions handled questions the library cannot answer."""
    conds = [c for c in report["run"]["conditions"]
             if ((report["conditions"][c]["abstention"].get("by_kind") or {})
                 .get("not_in_corpus"))]
    if not conds:
        return
    lines += ["## Questions the library cannot answer (not_in_corpus)", ""]
    lines.append(
        "The grounded prompt asks the model to say plainly when the sources do not contain "
        "the answer, and to label any general knowledge it adds as not from the sources. "
        "**Acknowledging the gap is the desired behavior**, whether the model then answers "
        "from labeled general knowledge or declines outright."
    )
    lines.append("")
    lines.append("| Condition | Acknowledged the gap (desired) | Declined outright | "
                 "Answered from labeled general knowledge | Answered without flagging the gap |")
    lines.append("|" + "---|" * 5)
    for c in conds:
        st = report["conditions"][c]["abstention"]["by_kind"]["not_in_corpus"]
        lines.append(
            f"| {c} | {_count_rate(st.get('met', 0), st['n'], st.get('ci95'))} | "
            f"{st.get('declined', 0)} | {st.get('answered_with_labeled_knowledge', 0)} | "
            f"{st.get('answered_without_flagging', 0)} |"
        )
    lines.append("")
    lines.append(
        "The ungrounded condition is not scored on these items: without the library, a model "
        "may simply know the answer, so they never count against it."
    )
    lines.append("")


def _signed_pts(delta: Mapping[str, Any] | None) -> str:
    if not delta or delta.get("mean") is None:
        return "n/a"
    ci = delta.get("ci95")
    tail = f" ({ci[0] * 100:+.0f} to {ci[1] * 100:+.0f})" if ci else ""
    return f"{delta['mean'] * 100:+.0f} pts{tail}"


def _num(value: float | None, fmt: str = "{:,.0f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _primary_correctness(st: Mapping[str, Any]) -> Mapping[str, Any]:
    return st.get("primary_correctness") or {"source": "judge", **st["correctness"]}


def _secondary_correctness_cell(st: Mapping[str, Any]) -> str:
    source = _primary_correctness(st).get("source")
    if source == "human":
        other = st["correctness"]
        return f"{_pct(other.get('mean'))} n={other.get('n', 0)} (judge)"
    other = st.get("human_correctness")
    if not other:
        return "not graded"
    return f"{_pct(other['mean'])} n={other['n']} of {other['eligible']} (human)"


def _render_primary(report: Mapping[str, Any], lines: List[str]) -> None:
    pc = report.get("primary_comparison")
    if not pc:
        return
    a, b = pc["conditions"]
    lines += ["## Primary comparison (pre-registered)", ""]
    est = pc["estimate"]
    ci = pc.get("ci95")
    diff = "n/a" if est is None else f"{est * 100:+.0f} pts"
    tail = f" (95% CI {ci[0] * 100:+.0f} to {ci[1] * 100:+.0f})" if ci else ""
    lines.append(
        f"Verified rate of all citations, **{a} minus {b}**: **{diff}**{tail}, paired over "
        f"{pc['n_items']} answerable items ({a} {pc[a]['verified']}/{pc[a]['citations']} "
        f"verified, {b} {pc[b]['verified']}/{pc[b]['citations']}). Answerable items only: an "
        "unanswerable question asks the model not to answer, so the per-condition table below, "
        "which counts every scored answer, gives slightly different denominators. The interval "
        "resamples items, not model runs; the replicate check measures run-to-run noise."
    )
    sec = pc["secondary_verified_of_checkable"]
    sec_diff = "n/a" if sec["estimate"] is None else f"{sec['estimate'] * 100:+.0f} pts"
    sec_ci = sec.get("ci95")
    sec_tail = f" (95% CI {sec_ci[0] * 100:+.0f} to {sec_ci[1] * 100:+.0f})" if sec_ci else ""
    lines.append(
        f"Secondary, verified of checkable citations (unresolvable left out): {sec_diff}{sec_tail}."
    )
    share = pc.get("bookkeeping_share_ungrounded")
    if share is not None:
        verdict = "within" if pc["bookkeeping_ok"] else "ABOVE"
        by_detail = pc.get("bookkeeping_by_detail_ungrounded") or {}
        detail = (" (" + ", ".join(f"{d} {n}" for d, n in by_detail.items()) + ")"
                  if by_detail else "")
        lines.append(
            f"Ungrounded citations unresolvable for fixable fixture-metadata reasons: "
            f"{share:.0%}{detail} ({verdict} the pre-registered "
            f"{pc['bookkeeping_max_share']:.0%})."
        )
        lines.append(
            f"Beside it, not a metadata gap: {pc.get('work_not_in_corpus_ungrounded', 0)} "
            f"ungrounded citation(s) name a work that is not in the library, of "
            f"{pc.get('unresolvable_ungrounded', 0)} unresolvable in all. Nothing in the "
            "fixture can make those checkable; they are the model citing outside the library."
        )
    lines.append("")


def render_markdown(report: Mapping[str, Any], *, chart_name: str | None = CHART_PNG) -> str:
    run = report["run"]
    publishable = bool(report.get("publishable"))
    conds = run["conditions"]
    source = report.get("primary_correctness", "judge")
    lines = [f"# Grounded-answer benchmark: {run['run_id']}", ""]
    if publishable:
        lines += ["_Publishable summary: aggregate scores only, no source text._"
                  if not report.get("source_text_published") else
                  "_Publishable copy of a public-domain corpus run: passages, transcripts and "
                  "blind grades are included._", ""]
    git = run.get("git") or {}
    lines.append(
        f"**Agent:** {run['agent']} · **Items:** {run['n_items']} "
        f"({run['n_items'] - run['n_unanswerable']} answerable, {run['n_unanswerable']} unanswerable) · "
        f"**Answer model:** `{run['answer_model']}` · **Judge model:** `{run['judge_model']}`"
    )
    lines.append(
        f"**Git:** `{(git.get('sha') or 'unknown')[:12]}`{' (dirty tree)' if git.get('dirty') else ''} · "
        f"**Index fingerprint:** `{(run.get('index_fingerprint') or '')[:12]}` · "
        f"**Fixture sha256:** `{((run.get('fixture') or {}).get('sha256') or '')[:12]}` · "
        f"**Status:** {run['status']}"
    )
    lines.append(f"**Started:** {run.get('started_utc')} · **Finished:** {run.get('finished_utc')}")
    lines.append("")
    for warning in report.get("warnings") or []:
        lines.append(f"> **Note:** {warning}")
    if report.get("warnings"):
        lines.append("")

    _render_primary(report, lines)

    lines += ["## Results by condition", ""]
    lines.append(
        f"| Condition | Correctness, {source} (95% CI) | vs ungrounded (paired) | "
        "Verified of all citations (95% CI) | Verified of checkable citations (95% CI) | "
        "Correctness, other grader | Declined (judge) | Confidently wrong (judge) | Abstention |"
    )
    lines.append("|" + "---|" * 9)
    for c in conds:
        st = report["conditions"][c]
        corr, cit, ab = _primary_correctness(st), st["citations"], st["abstention"]
        judge = st["correctness"]
        n_flag = judge.get("n_with_declined_flag", judge.get("n", 0))
        lines.append(
            f"| {c} | {_pct(corr['mean'])}{_ci(corr['ci95'])} n={corr['n']} | "
            f"{_signed_pts(corr.get('delta_vs_ungrounded')) if c != 'ungrounded' else 'baseline'} | "
            f"{_pct(cit['verified_rate'])}{_ci(cit['verified_rate_ci95'])} of {cit['total']} | "
            f"{_pct(cit.get('verified_of_checkable'))}{_ci(cit.get('verified_of_checkable_ci95'))} "
            f"of {cit.get('checkable', cit['total'])} | "
            f"{_secondary_correctness_cell(st)} | "
            f"{_count_rate(judge.get('n_declined', 0), n_flag)} | "
            f"{_count_rate(judge.get('n_confidently_wrong', 0), n_flag)} | "
            f"{_abstention_cell(ab)} |"
        )
    lines.append("")
    lines.append(
        f"Correctness is graded by the {'human grader (blind to condition) for every answer; the judge is secondary' if source == 'human' else 'rubric judge; human grades are secondary until every answer is graded'}. "
        "Declined and confidently wrong (scored 0 without declining) come from the judge's "
        "declined flag. Verified of checkable citations leaves out unresolvable ones: "
        "unresolvable means the citation could not be checked against the library, not "
        "that it is wrong."
    )
    lines.append("")
    if chart_name:
        lines += [f"![Correctness and verified-citation rate by condition]({chart_name})", ""]

    _render_not_in_corpus(report, lines)

    lines += ["## Retrieval, cost and latency", ""]
    lines.append("| Condition | Citations per answer | Median claim words per citation | "
                 "Recall@5 gold page | Recall@5 doc | Tokens in / out | Cost (answer + judge) | "
                 "Latency p50 / p90 |")
    lines.append("|" + "---|" * 8)
    for c in conds:
        st = report["conditions"][c]
        rec = st.get("retrieval") or {}
        tok = st["tokens"]["answer"]
        cit = st["citations"]
        lines.append(
            f"| {c} | {_num(cit['per_answer'], '{:.1f}')} | "
            f"{_num(cit.get('median_claim_words'), '{:.0f}')} | "
            f"{_pct(rec.get('recall_at_5_gold_page')) if c != 'ungrounded' else 'n/a'} | "
            f"{_pct(rec.get('recall_at_5_doc')) if c != 'ungrounded' else 'n/a'} | "
            f"{_num(tok['input_tokens'])} / {_num(tok['output_tokens'])} | "
            f"${st['cost_usd']['answer']:.2f} + ${st['cost_usd']['judge']:.2f} | "
            f"{_num(st['latency_s']['median'], '{:.1f}')}s / {_num(st['latency_s']['p90'], '{:.1f}')}s |"
        )
    lines.append("")

    lines += ["## Citation buckets", ""]
    lines.append("| Condition | Verified | Partial | Unsupported | Invented | Unresolvable | Total | "
                 "Unjudged | Answers with citations | Checkable | Label matches |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        cit = report["conditions"][c]["citations"]
        lines.append(
            f"| {c} | {cit['verified']} | {cit.get('partial', 0)} | {cit['unsupported']} | {cit['invented']} | "
            f"{cit['unresolvable']} | {cit['total']} | {cit['unjudged']} | "
            f"{cit['answers_with_citations']} of {report['conditions'][c]['n_scored']} | "
            f"{cit.get('checkable', cit['total'] - cit['unresolvable'])} | {cit.get('label_matches', 0)} |"
        )
    partial = {c: report["conditions"][c]["citations"].get("partial_matches", 0) for c in conds}
    if any(partial.values()):
        lines.append("")
        lines.append(
            "Partial matches (a grounded citation that named a returned document but gave no "
            "page or section; the supported ones are in the Partial column, never counted as "
            "verified): "
            + ", ".join(f"{c} {n}" for c, n in partial.items()) + "."
        )
    lines.append("")
    lines.append(
        "**Unresolvable means \"could not be checked\", not \"wrong\".** It is left out of "
        "verified of checkable citations and counted in verified of all citations."
    )
    details = sorted({d for c in conds for d in (report["conditions"][c]["citations"]
                                                  .get("unresolvable_by_detail") or {})})
    if details:
        lines += ["", "| Condition | " + " | ".join(details) + " | Of which fixture metadata |",
                  "|" + "---|" * (len(details) + 2)]
        for c in conds:
            cit = report["conditions"][c]["citations"]
            by = cit.get("unresolvable_by_detail") or {}
            lines.append(f"| {c} | " + " | ".join(str(by.get(d, 0)) for d in details)
                         + f" | {cit.get('bookkeeping_unresolvable', 0)} |")
    notes = {c: report["conditions"][c]["citations"].get("notes") or {} for c in conds}
    flagged = [(c, k, n) for c, by in notes.items() for k, n in by.items()
               if k in ("slug_fuzzy_match", "section_mismatch", "edition_unchecked")]
    if flagged:
        lines.append("")
        lines.append(
            "Resolution notes (grounded citations located despite a slug typo, a wrong "
            "section string, or an edition the metadata cannot confirm): "
            + ", ".join(f"{c} {k} {n}" for c, k, n in flagged) + "."
        )
    lines.append("")

    grounded_conds = [c for c in conds if report["conditions"][c].get("attribution")]
    if grounded_conds:
        lines += ["## Retrieval versus generation (grounded conditions)", ""]
        lines.append(
            f"Grounded answers that were not fully correct ({source} grades), split by whether "
            "any tool call surfaced the gold page: not surfaced is a retrieval miss, surfaced "
            "and still wrong is a generation miss."
        )
        lines.append("")
        lines.append("| Condition | Answers graded | Gold page surfaced | Wrong (0): gold not surfaced | "
                     "Wrong (0): gold surfaced | Partial (0.5): gold not surfaced | "
                     "Partial (0.5): gold surfaced |")
        lines.append("|" + "---|" * 7)
        for c in grounded_conds:
            at = report["conditions"][c]["attribution"]
            lines.append(
                f"| {c} | {at['n_graded']} | {_count_rate(at['n_gold_page_in_context'], at['n_graded'])} | "
                f"{at['wrong']['gold_not_in_context']} | {at['wrong']['gold_in_context']} | "
                f"{at['partial']['gold_not_in_context']} | {at['partial']['gold_in_context']} |"
            )
        lines.append("")

    lines += ["## By category", ""]
    lines.append("| Category | Condition | n | Correctness (judge) | Verified citations | Abstention |")
    lines.append("|---|---|---|---|---|---|")
    for category, by_cond in report["per_category"].items():
        for c in conds:
            if c not in by_cond:
                continue
            row = by_cond[c]
            n_corr = row.get("n_correctness")
            corr_cell = _pct(row["correctness"]) if n_corr is None else (
                "n/a" if not n_corr else f"{_pct(row['correctness'])} (n={n_corr})")
            ver_cell = (_pct(row["verified_rate"]) if row.get("verified") is None
                        else _count_rate(row["verified"], row["citations"]))
            ab_cell = (_pct(row["abstention"]) if row.get("n_abstention") is None
                       else _count_rate(row.get("declined", 0), row["n_abstention"]))
            lines.append(
                f"| {category} | {c} | **n={row['n']}** | {corr_cell} | {ver_cell} | {ab_cell} |"
            )
    lines.append("")

    lines += ["## Worst failures", ""]
    if report["worst_failures"]:
        header = "| Item | Condition | Category | Correctness | Verified / Unsupported / Invented / Unresolvable | Why |"
        if not publishable:
            header += " Judge reason |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|" + ("---|" if not publishable else ""))
        for row in report["worst_failures"]:
            b = row["buckets"]
            corr = "n/a" if row["correctness"] is None else f"{row['correctness']:g}"
            line = (
                f"| {row['item_id']} | {row['condition']} | {row.get('category') or ''} | {corr} | "
                f"{b['verified']} / {b['unsupported']} / {b['invented']} / {b['unresolvable']} | "
                f"{_md_escape(row['why'])} |"
            )
            if not publishable:
                line += f" {_md_escape(row.get('judge_reason') or '')} |"
            lines.append(line)
    else:
        lines.append("_None._")
    lines.append("")

    lines += ["## Judge validation", ""]
    coverage = report.get("human_coverage")
    if coverage:
        ans, unans = coverage["answerable"], coverage["unanswerable"]
        lines.append(
            f"Human grades: {ans['graded']} of {ans['eligible']} answerable and "
            f"{unans['graded']} of {unans['eligible']} unanswerable scored answers. "
            f"Primary correctness source: **{source}**."
        )
        lines.append("")
    agreement = report.get("judge_validation")
    if agreement:
        gate = agreement["publish_gate"]
        lines.append("| Metric | Double-graded | Agreement | Cohen's kappa |")
        lines.append("|---|---|---|---|")
        rows_out = [("correctness, all rows", agreement["correctness"])]
        judge_only = (agreement["correctness"].get("by_method") or {}).get("judge")
        if judge_only is not None:
            rows_out.append(("correctness, rubric-judge rows (gate)", judge_only))
        rows_out.append(("abstention", agreement["abstention"]))
        for kind, a in (agreement["abstention"].get("by_kind") or {}).items():
            rows_out.append((f"abstention, {kind}", a))
        rows_out.append(("support", agreement["support"]))
        for label, a in rows_out:
            kappa = "n/a" if a.get("cohens_kappa") is None else f"{a['cohens_kappa']:.2f}"
            lines.append(f"| {label} | {a['n']} | {_pct(a['agreement'])} | {kappa} |")
        audit = agreement.get("resolution_audit")
        if audit and audit.get("n"):
            lines.append(f"| resolution audit (human agrees with invented / unresolvable) | "
                         f"{audit['n']} | {_pct(audit['agreement'])} | n/a |")
        lines.append("")
        lines.append(
            f"Publish gate (correctness agreement at least {PUBLISH_GATE:.0%} on rubric-judge "
            f"rows, n at least {gate.get('min_n', 1)}; numeric auto-scored rows agree almost "
            f"by construction and are excluded): **{gate['result']}** "
            f"(n={gate.get('n', agreement['correctness']['n'])})."
        )
        support_gate = agreement.get("support_gate")
        if support_gate:
            kappa = support_gate.get("cohens_kappa")
            min_kappa = support_gate.get("min_kappa")
            lines.append(
                f"Support gate (the support judge decides the primary citation metric; "
                f"agreement at least {PUBLISH_GATE:.0%} **and** Cohen's kappa at least "
                f"{min_kappa if min_kappa is not None else 'n/a'} on human-graded citations, "
                f"n at least {support_gate.get('min_n', 1)}): **{support_gate['result']}** "
                f"(n={support_gate['n']}, agreement {_pct(support_gate.get('agreement'))}, "
                f"kappa {'n/a' if kappa is None else f'{kappa:.2f}'}). Raw agreement alone "
                "passes when nearly every citation is supported; kappa is what shows the "
                "judge discriminates."
            )
    else:
        lines.append(
            "Not imported yet. Fill `blind/answers.csv`, `blind/citations.csv` and "
            "`blind/resolution_audit.csv`, then run "
            "`grounding eval-answers --run-dir <run> --import-grades`. Do not publish before "
            f"correctness agreement on rubric-judge rows reaches {PUBLISH_GATE:.0%}."
        )
    lines.append("")

    replicates = report.get("replicates") or []
    if replicates:
        lines += ["## Replicate agreement", ""]
        lines.append("Run-to-run agreement with a second run of the same condition on the same "
                     "items (judge scores): the noise floor a between-condition difference has to clear.")
        lines.append("")
        lines.append("| Replicate run | Condition | Shared items | Correctness exact agreement | "
                     "Mean abs. difference | Verified of all, this run / replicate | "
                     "Difference (95% CI) |")
        lines.append("|---|---|---|---|---|---|---|")
        for rep in replicates:
            for c, st in rep["conditions"].items():
                corr, ver = st["correctness"], st["verified_of_all"]
                scope = "" if st.get("covers_all_items", True) else f" of {st.get('n_items_a')}"
                diff = "n/a" if ver.get("diff") is None else f"{ver['diff'] * 100:+.0f} pts"
                ci = ver.get("diff_ci95")
                if ci:
                    diff += f" ({ci[0] * 100:+.0f} to {ci[1] * 100:+.0f})"
                lines.append(
                    f"| {rep['run_b']}{'' if rep.get('comparable') else ' (not comparable)'} | {c} | "
                    f"{st['n_items']}{scope} | {_count_rate(round((corr['exact_agreement'] or 0) * corr['n']), corr['n'])} | "
                    f"{_num(corr['mean_abs_diff'], '{:.2f}')} | {_pct(ver['a'])} / {_pct(ver['b'])} | "
                    f"{diff} |"
                )
        lines.append("")
        if any(not st.get("covers_all_items", True)
               for rep in replicates for st in rep["conditions"].values()):
            lines.append(
                "A replicate that covers only some of the run's items measures run-to-run "
                "noise far less precisely than a full re-run of the condition; read it as "
                "indicative only."
            )
            lines.append("")

    if report.get("questions"):
        lines += ["## Questions", ""]
        lines.append("| Item | Question |")
        lines.append("|---|---|")
        for item_id, question in sorted(report["questions"].items()):
            lines.append(f"| {item_id} | {_md_escape(question)} |")
        lines.append("")

    lines += ["## Run health", ""]
    lines.append("| Condition | Answers | Scored | Excluded | Answer errors | Judge errors | Refusals | Truncated | Hit tool cap | Tool calls per answer |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        st = report["conditions"][c]
        lines.append(
            f"| {c} | {st['n_answers']} | {st['n_scored']} | {st.get('n_excluded', 0)} | "
            f"{st['n_errors']} | {st['n_judge_errors']} | "
            f"{st['n_refusal']} | {st['n_truncated']} | {st['n_max_iterations']} | "
            f"{_num(st['tool_calls_per_answer'], '{:.1f}')} |"
        )
    if any(report["conditions"][c].get("n_excluded") for c in conds):
        lines.append("")
        lines.append(
            "Excluded: not_in_corpus items in the ungrounded condition. That condition had no "
            "library, so declining is not the expected behavior there; they are never scored "
            "and never count against it."
        )
    lines.append("")

    prompts = run.get("prompts") or {}
    judge_versions = ", ".join(
        v["version"] for _, v in sorted((prompts.get("judge") or {}).items())
    )
    lines += ["## Method", ""]
    lines.append(
        f"Conditions share the answer model and base prompt (`{(prompts.get('answer') or {}).get('answer_version')}`); "
        f"grounded conditions add the in-process `search_corpus` tool, at most {run.get('max_iterations')} "
        f"tool rounds per answer. Judge prompts: {judge_versions or 'n/a'}. Correctness is graded against the "
        "gold answer with citations removed; the judge never sees retrieved text. Numeric items are "
        "auto-scored first when the answer states the gold unit unambiguously. A support verdict "
        "is reused when the same claim meets the same passage again, so its judge cost is charged "
        "to the first answer that needed it. Intervals are "
        f"{report['statistics']['ci']}, {report['statistics']['n_boot']} resamples, seed "
        f"{report['statistics']['seed']}; paired differences resample items. Costs use "
        f"{run.get('pricing_source')}."
    )
    embedding = (run.get("environment") or {}).get("embedding") or {}
    if embedding:
        lines.append("")
        lines.append(
            f"Embedding model: `{embedding.get('query_model')}` for queries, "
            f"`{embedding.get('index_model')}` for the index (sentence-transformers "
            f"{embedding.get('sentence_transformers')}, torch {embedding.get('torch')}, "
            f"faiss {embedding.get('faiss')})."
        )
    lines.append("")
    return "\n".join(lines)


def _chart_panels(report: Mapping[str, Any]):
    conds = report["run"]["conditions"]
    source = report.get("primary_correctness", "judge")

    def pull(getter):
        return [getter(report["conditions"][c]) for c in conds]

    return [
        (f"Correctness ({source} grades)",
         pull(lambda st: _primary_correctness(st).get("mean")),
         pull(lambda st: _primary_correctness(st).get("ci95"))),
        ("Verified, of all citations",
         pull(lambda st: st["citations"].get("verified_rate")),
         pull(lambda st: st["citations"].get("verified_rate_ci95"))),
        ("Verified, of checkable citations",
         pull(lambda st: st["citations"].get("verified_of_checkable")),
         pull(lambda st: st["citations"].get("verified_of_checkable_ci95"))),
    ]


def render_chart(report: Mapping[str, Any], path: Path) -> bool:
    """Three small-multiple panels of horizontal bars, one metric each, per condition.

    Each panel is a single series named by its title (one hue, no legend);
    values sit at the bar tips and whiskers are 95 percent bootstrap
    intervals. Returns False (and writes nothing) when matplotlib is
    unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError:  # pragma: no cover - matplotlib ships with music21
        logger.warning("matplotlib not installed; skipping %s", path)
        return False

    conds = report["run"]["conditions"]
    run = report["run"]
    panels = _chart_panels(report)
    x_max = 1.22
    height = 1.45 + 0.42 * len(conds)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, height), dpi=160, sharey=True)
    fig.patch.set_facecolor(_SURFACE)
    fig.suptitle("Correctness and verified-citation rate by condition",
                 x=0.02, y=0.975, ha="left", fontsize=11, color=_INK)
    fig.text(0.02, 1 - 0.4 / height,
             f"{run['n_items']} items; answer model {run['answer_model']}; judge {run['judge_model']}. "
             "Whiskers: 95% bootstrap interval. Unresolvable citations count against "
             "\"of all\" and are left out of \"of checkable\".",
             ha="left", va="top", fontsize=7, color=_INK_2, wrap=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=1 - 0.95 / height, bottom=0.42 / height,
                        wspace=0.12)
    fig.canvas.draw()
    for ax, (title, values, cis) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        ax.set_xlim(0, x_max)
        ax.set_ylim(len(conds) - 0.5, -0.5)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"], color=_MUTED, fontsize=7.5)
        ax.xaxis.grid(True, color=_GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(_BASELINE)
        ax.tick_params(axis="both", length=0)
        ax.set_title(title, loc="left", fontsize=8.5, color=_INK, pad=4)
        box = ax.get_window_extent()
        x_per_px = x_max / box.width
        y_per_px = len(conds) / box.height
        bar_h = min(0.6, 22 * y_per_px)
        for y_index, (value, ci) in enumerate(zip(values, cis)):
            y = y_index - bar_h / 2
            if value is None:
                ax.text(0.02, y_index, "n/a", ha="left", va="center", fontsize=7, color=_MUTED)
                continue
            if value > 0:
                radius = min(4 * y_per_px, bar_h / 2)
                ax.add_patch(FancyBboxPatch(
                    (0, y), value, bar_h, boxstyle=f"round,pad=0,rounding_size={radius}",
                    mutation_aspect=x_per_px / y_per_px, linewidth=0, facecolor=_SERIES[0],
                ))
                # Square the baseline corners.
                ax.add_patch(Rectangle((0, y), min(value, 5 * x_per_px), bar_h,
                                       linewidth=0, facecolor=_SERIES[0]))
            tip = value
            if ci:
                half = bar_h * 0.3
                for xs, ys in ((ci, [y_index, y_index]), ([ci[0]] * 2, [y_index - half, y_index + half]),
                               ([ci[1]] * 2, [y_index - half, y_index + half])):
                    ax.plot(xs, ys, color=_INK_2, linewidth=0.8, solid_capstyle="butt")
                tip = max(value, ci[1])
            ax.text(tip + 5 * x_per_px, y_index, f"{value * 100:.0f}%", ha="left", va="center",
                    fontsize=7.5, color=_INK_2)
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels(conds, color=_INK, fontsize=8.5)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE, metadata={"Software": None})
    plt.close(fig)
    return True


#: Run files a publishable copy may carry when the corpus is public domain.
SOURCE_TEXT_FILES = (
    TRANSCRIPTS_FILE, SCORES_FILE, RETRIEVAL_FILE, AGREEMENT_FILE, HUMAN_GRADES_FILE,
    "blind/answers.csv", "blind/citations.csv", "blind/resolution_audit.csv", "blind/key.json",
)


def _copy_source_text(run_dir: Path, pub_dir: Path) -> List[Path]:
    """Copy the run files that carry source text into the publishable directory.

    Only for a ``source_license: public_domain`` fixture. ``errors.jsonl`` is
    left out: an exception message can carry a local path.
    """
    written = []
    for name in SOURCE_TEXT_FILES:
        source = run_dir / name
        if not source.exists():
            continue
        target = pub_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        written.append(target)
    return written


def write_report(
    run_dir: Path, *, publishable: bool = False, include_questions: bool = False,
    include_source_text: bool = False,
) -> Dict[str, Path]:
    """Write report.md, report.json and chart.png (and the publishable copies).

    ``include_source_text`` also copies the transcripts, scores and blind CSVs
    into ``publishable/``. It is refused unless the fixture declared
    ``source_license: public_domain`` (D7).
    """
    run_dir = Path(run_dir)
    report = build_report(run_dir)
    license_ = (report["run"].get("fixture") or {}).get("source_license")
    if include_source_text and license_ != PUBLIC_DOMAIN:
        raise ValueError(
            "source text can only be published for a public-domain corpus: this run's fixture "
            f"declares source_license {license_!r}. Add `source_license: public_domain` to the "
            "fixture and re-run, or publish without --include-source-text."
        )
    written: Dict[str, Path] = {}
    chart_ok = render_chart(report, run_dir / CHART_PNG)
    if chart_ok:
        written["chart"] = run_dir / CHART_PNG
    (run_dir / REPORT_MD).write_text(
        render_markdown(report, chart_name=CHART_PNG if chart_ok else None), encoding="utf-8"
    )
    write_json(run_dir / REPORT_JSON, report)
    written.update(markdown=run_dir / REPORT_MD, json=run_dir / REPORT_JSON)

    if publishable:
        pub_dir = run_dir / PUBLISH_DIR
        pub_dir.mkdir(exist_ok=True)
        view = publishable_view(report, include_questions=include_questions,
                                include_source_text=include_source_text)
        pub_chart = render_chart(view, pub_dir / CHART_PNG)
        (pub_dir / REPORT_MD).write_text(
            render_markdown(view, chart_name=CHART_PNG if pub_chart else None), encoding="utf-8"
        )
        write_json(pub_dir / REPORT_JSON, view)
        written.update(publishable_markdown=pub_dir / REPORT_MD,
                       publishable_json=pub_dir / REPORT_JSON)
        if pub_chart:
            written["publishable_chart"] = pub_dir / CHART_PNG
        if include_source_text:
            for index, path in enumerate(_copy_source_text(run_dir, pub_dir)):
                written[f"publishable_source_{index}"] = path
    return written
