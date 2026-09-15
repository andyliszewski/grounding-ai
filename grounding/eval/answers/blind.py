"""Blind-grade export and import for judge validation (Story 25.3).

Export writes, with the condition hidden and rows shuffled with a fixed seed:

* ``blind/answers.csv``: one row per sampled answer. The answer text has its
  citations removed, which is also what the correctness judge read, so the
  citation style (slug brackets or title and page) does not reveal whether
  the answer was grounded. The grader fills ``human_grade``: 1, 0.5 or 0 for
  answerable items; for unanswerable ones, 1 when the answer did what the
  question's kind asks for (a ``no_source`` question: declined; a
  ``not_in_corpus`` question: said plainly that the sources do not contain the
  answer and labeled any general knowledge it added), 0 otherwise. The scale
  is framed per ``unanswerable_kind``, as the judge's was.
* ``blind/citations.csv``: every citation of the sampled answers that went to
  the support judge, with its claim and the passage it resolved to. The
  grader fills ``human_supported`` with y or n.
* ``blind/resolution_audit.csv``: a sample of the citations the scorer put
  in ``invented`` or ``unresolvable`` without a judge (stratified by
  condition and bucket), with the scorer's reason and what it looked at. The
  grader fills ``human_agrees`` (y or n) and, when n, ``human_bucket``. This
  audits the scorer's bookkeeping, not the model: the citation string is
  shown, so these rows reveal grounded versus ungrounded (never which
  retrieval variant).

The default ``fraction`` is 1.0: the maintainer grades every answer, and the
report then uses the human grades as the primary correctness metric with the
judge as secondary. A smaller fraction draws a stratified random sample: per
(condition, category, unanswerable kind) stratum, ``round(fraction * n)``
answers and at least one, so the realized fraction can exceed the requested
one on small strata. Answers whose judge call failed are still exported, so
human coverage does not depend on the judge.

``blind/key.json`` maps sample ids back to item, condition and the grader's
scores. Re-exporting over CSVs that already hold grades first moves them to
``blind/archive-<UTC>/``, so filled grades are never overwritten.

Import reads the filled CSVs (the pre-rename ``andy_grade``, ``andy_supported``
and ``andy_notes`` columns are still accepted) and writes ``agreement.json``
(per metric: double-graded rows, raw agreement, Cohen's kappa) and
``human_grades.json`` (one human grade per answer, no text). The publish gate
is correctness agreement of at least 80 percent on the rows the rubric judge
graded (``by_method["judge"]``), with at least ``PUBLISH_GATE_MIN_N`` of them.
Numeric rows are auto-scored by the unit-and-tolerance check and agree with a
careful human almost by construction, so counting them would let the gate
pass on bookkeeping rather than on the judge; agreement over all rows is
reported next to it. The support gate holds the support judge, which decides
the primary citation metric, to the same agreement **and** to Cohen's kappa of
at least ``SUPPORT_GATE_MIN_KAPPA`` over the citations the human graded: raw
agreement alone passes when nearly every citation is supported, which says
nothing about the judge's discrimination. Grade the citation rows from the top
of the file: they are in the answers' shuffled order, so the first rows are a
random sample of answers.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from grounding.eval.answers.scoring import abstained_as_desired, desired_behavior

BLIND_DIR = "blind"
ANSWERS_CSV = "answers.csv"
CITATIONS_CSV = "citations.csv"
AUDIT_CSV = "resolution_audit.csv"
KEY_FILE = "key.json"
AGREEMENT_FILE = "agreement.json"
HUMAN_GRADES_FILE = "human_grades.json"
DEFAULT_FRACTION = 1.0
DEFAULT_SEED = 0
PUBLISH_GATE = 0.80
# Fewer judge-graded rows than this and a gate reports "insufficient": a rate
# over a handful of rows says little about the rubric.
PUBLISH_GATE_MIN_N = 10
# The support judge decides the headline citation metric, and most citations
# in a good answer are supported: raw agreement alone can pass on the majority
# label. Cohen's kappa has to clear this too (0.6: "substantial" agreement).
SUPPORT_GATE_MIN_KAPPA = 0.6

ANSWER_COLUMNS = [
    "sample_id", "item_id", "category", "question", "gold", "must_include",
    "answer_without_citations", "grade_scale", "human_grade", "human_notes",
]
CITATION_COLUMNS = [
    "sample_id", "cite_id", "question", "claim", "passage", "human_supported", "human_notes",
]
AUDIT_COLUMNS = [
    "audit_id", "item_id", "question", "citation", "claim", "auto_bucket", "auto_detail",
    "scorer_looked_at", "human_agrees", "human_bucket", "human_notes",
]
# Columns a human fills; any non-empty one means the file holds grading work.
_HUMAN_COLUMNS = ("human_grade", "human_supported", "human_agrees", "human_bucket",
                  "andy_grade", "andy_supported")
_AUDIT_BUCKETS = ("invented", "unresolvable")


def _grade_scale(answerable: bool, kind: str | None) -> str:
    """What the grader writes, framed the way the judge was (per unanswerable kind)."""
    if answerable:
        return "1, 0.5 or 0"
    if kind == "not_in_corpus":
        return ("1 = said plainly that the sources do not contain the answer, and "
                "labeled any general knowledge it added as not from the sources; "
                "0 = otherwise (answered as if the sources supported it, or never "
                "said they lack the answer)")
    return ("1 = declined, pointed out the false premise, or asked for the missing "
            "information; 0 = answered")


def _eligible(score: Mapping[str, Any]) -> bool:
    """Every scored answer, including ones whose judge call failed."""
    return bool(score.get("scored")) and not score.get("excluded")


def _stratum(score: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (score["condition"], score.get("category") or "uncategorized",
            score.get("unanswerable_kind") or "")


def _take(pool_size: int, fraction: float) -> int:
    # Round half up (deterministic, unlike banker's rounding), at least one.
    return min(pool_size, max(1, math.floor(fraction * pool_size + 0.5)))


def draw_sample(
    scores: Sequence[Mapping[str, Any]], *, fraction: float = DEFAULT_FRACTION, seed: int = DEFAULT_SEED
) -> List[Mapping[str, Any]]:
    """Stratified random sample of scored answers (deterministic for a seed).

    Strata are (condition, category, unanswerable kind); category also
    separates answerable from unanswerable items.
    """
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for score in scores:
        if _eligible(score):
            strata.setdefault(_stratum(score), []).append(score)
    picked: List[Mapping[str, Any]] = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda s: s["item_id"])
        picked.extend(rng.sample(pool, _take(len(pool), fraction)))
    rng.shuffle(picked)
    return picked


def draw_audit_sample(
    scores: Sequence[Mapping[str, Any]], *, fraction: float = DEFAULT_FRACTION, seed: int = DEFAULT_SEED
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """(score, citation) pairs for the resolution audit, stratified by condition and bucket."""
    rng = random.Random(seed + 1)
    strata: Dict[Tuple[str, str], List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for score in scores:
        if not _eligible(score):
            continue
        for cit in score.get("citations") or []:
            if cit.get("bucket") in _AUDIT_BUCKETS:
                strata.setdefault((score["condition"], cit["bucket"]), []).append((score, cit))
    picked = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda pair: (pair[0]["item_id"], pair[1]["cite_id"]))
        picked.extend(rng.sample(pool, _take(len(pool), fraction)))
    rng.shuffle(picked)
    return picked


def _looked_at(score: Mapping[str, Any], cit: Mapping[str, Any]) -> str:
    """What the scorer compared the citation against, for a human to check."""
    res = cit.get("resolution") or {}
    parts = []
    if score.get("returned_prefixes") is not None:
        shown = list(dict.fromkeys(score["returned_prefixes"]))
        parts.append("returned in this transcript: " + (" | ".join(shown) if shown else "nothing"))
    for key in ("doc_id", "candidate_doc_id", "pdf_pages", "title_score"):
        if res.get(key) is not None:
            parts.append(f"{key}: {res[key]}")
    if res.get("notes"):
        parts.append("notes: " + ", ".join(res["notes"]))
    return "; ".join(parts)[:4000]


def _has_human_work(path: Path) -> bool:
    if not path.exists():
        return False
    return any((row.get(col) or "").strip() for row in _read_csv(path) for col in _HUMAN_COLUMNS)


def _archive_filled(out_dir: Path) -> str | None:
    """Move CSVs that hold human grades (and their key) aside before re-exporting."""
    files = [out_dir / name for name in (ANSWERS_CSV, CITATIONS_CSV, AUDIT_CSV)]
    if not any(_has_human_work(f) for f in files):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive = out_dir / f"archive-{stamp}"
    n = 1
    while archive.exists():
        n += 1
        archive = out_dir / f"archive-{stamp}-{n}"
    archive.mkdir()
    for path in files + [out_dir / KEY_FILE]:
        if path.exists():
            shutil.move(str(path), str(archive / path.name))
    return str(archive)


def export_blind(
    run_dir: Path,
    scores: Sequence[Mapping[str, Any]],
    items: Mapping[str, Any],
    *,
    fraction: float = DEFAULT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Write the blind-grade CSVs and key. Returns a summary for the manifest."""
    out_dir = Path(run_dir) / BLIND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    archived = _archive_filled(out_dir)
    sample = draw_sample(scores, fraction=fraction, seed=seed)
    key: Dict[str, Any] = {"fraction": fraction, "seed": seed, "samples": {}, "audit": {}}
    answer_rows, citation_rows, audit_rows = [], [], []
    for index, score in enumerate(sample, start=1):
        sample_id = f"S{index:03d}"
        item = items[score["item_id"]]
        answerable = score["answerable"]
        answer_rows.append(
            {
                "sample_id": sample_id,
                "item_id": score["item_id"],
                "category": score.get("category") or "",
                "question": item.query,
                "gold": item.answer.gold if item.answer else "",
                "must_include": "; ".join(item.answer.must_include) if item.answer else "",
                "answer_without_citations": score.get("answer_without_citations", ""),
                "grade_scale": _grade_scale(answerable, score.get("unanswerable_kind")),
                "human_grade": "",
                "human_notes": "",
            }
        )
        judged_citations = {}
        for cit in score.get("citations", []):
            if cit.get("judge") is None:
                continue  # invented, unresolvable, or no judge verdict (see the audit)
            judged_citations[cit["cite_id"]] = cit["judge"]["supported"]
            citation_rows.append(
                {
                    "sample_id": sample_id,
                    "cite_id": cit["cite_id"],
                    "question": item.query,
                    "claim": cit.get("claim", ""),
                    "passage": cit.get("passage", ""),
                    "human_supported": "",
                    "human_notes": "",
                }
            )
        key["samples"][sample_id] = {
            "item_id": score["item_id"],
            "condition": score["condition"],
            "answerable": answerable,
            "unanswerable_kind": score.get("unanswerable_kind"),
            "grader_correctness": (score.get("correctness") or {}).get("score"),
            "grader_method": (score.get("correctness") or score.get("abstention") or {}).get("method"),
            "grader_declined": (score.get("abstention") or {}).get("declined"),
            # What this unanswerable kind asks for: declining, or acknowledging
            # the gap. The human grades the same thing (see _grade_scale).
            "grader_abstained": abstained_as_desired(
                score.get("abstention"), score.get("unanswerable_kind")
            ),
            "grader_desired_behavior": desired_behavior(score.get("unanswerable_kind")),
            "grader_supported": judged_citations,
        }
    for index, (score, cit) in enumerate(draw_audit_sample(scores, fraction=fraction, seed=seed), 1):
        audit_id = f"A{index:03d}"
        item = items[score["item_id"]]
        audit_rows.append(
            {
                "audit_id": audit_id,
                "item_id": score["item_id"],
                "question": item.query,
                "citation": cit.get("text", ""),
                "claim": cit.get("claim", ""),
                "auto_bucket": cit["bucket"],
                "auto_detail": (cit.get("resolution") or {}).get("detail", ""),
                "scorer_looked_at": _looked_at(score, cit),
                "human_agrees": "",
                "human_bucket": "",
                "human_notes": "",
            }
        )
        key["audit"][audit_id] = {
            "item_id": score["item_id"],
            "condition": score["condition"],
            "cite_id": cit["cite_id"],
            "bucket": cit["bucket"],
            "detail": (cit.get("resolution") or {}).get("detail"),
        }
    _write_csv(out_dir / ANSWERS_CSV, ANSWER_COLUMNS, answer_rows)
    _write_csv(out_dir / CITATIONS_CSV, CITATION_COLUMNS, citation_rows)
    _write_csv(out_dir / AUDIT_CSV, AUDIT_COLUMNS, audit_rows)
    (out_dir / KEY_FILE).write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    counts = Counter(s["condition"] for s in sample)
    eligible = sum(1 for s in scores if _eligible(s))
    return {
        "fraction": fraction,
        "seed": seed,
        "n_answers": len(answer_rows),
        "n_eligible_answers": eligible,
        "realized_fraction": round(len(answer_rows) / eligible, 4) if eligible else None,
        "n_citations": len(citation_rows),
        "n_audit_citations": len(audit_rows),
        "per_condition": dict(sorted(counts.items())),
        "archived_previous": archived,
        "dir": str(out_dir),
    }


def _write_csv(path: Path, columns: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(row: Mapping[str, str], name: str) -> str:
    """A human column, accepting the pre-rename andy_ spelling."""
    value = row.get(f"human_{name}")
    if value is None or not value.strip():
        value = row.get(f"andy_{name}") or value
    return value or ""


def _parse_grade(raw: str) -> float | None:
    text = (raw or "").strip().lower()
    if not text:
        return None
    aliases = {"y": 1.0, "yes": 1.0, "n": 0.0, "no": 0.0, "half": 0.5}
    if text in aliases:
        return aliases[text]
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"cannot read grade {raw!r}; use 1, 0.5 or 0") from exc
    if value not in (0.0, 0.5, 1.0):
        raise ValueError(f"grade {raw!r} is not one of 1, 0.5, 0")
    return value


def _parse_yes_no(raw: str) -> bool | None:
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text in ("y", "yes", "1", "true", "supported", "agree"):
        return True
    if text in ("n", "no", "0", "false", "unsupported", "disagree"):
        return False
    raise ValueError(f"cannot read yes/no grade {raw!r}; use y or n")


def cohens_kappa(pairs: Sequence[Tuple[Any, Any]]) -> float | None:
    """Unweighted Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum(first[k] * second.get(k, 0) for k in first) / (n * n)
    if expected >= 1.0:
        return None  # both raters used one label only; kappa is undefined
    return (observed - expected) / (1.0 - expected)


def _agreement(pairs: Sequence[Tuple[Any, Any]]) -> Dict[str, Any]:
    n = len(pairs)
    return {
        "n": n,
        "agreement": (sum(1 for a, b in pairs if a == b) / n) if n else None,
        "cohens_kappa": cohens_kappa(pairs),
    }


def publish_gate(
    judge_rows: Mapping[str, Any], *, overall: Mapping[str, Any],
    metric: str = "correctness agreement on rubric-judge rows",
    min_kappa: float | None = None,
) -> Dict[str, Any]:
    """A gate: agreement on judge-graded rows, at least PUBLISH_GATE_MIN_N of them.

    ``judge_rows`` is the agreement over rows a judge graded; ``overall``
    (every double-graded row, numeric auto-scores included) is carried along
    for the report but does not decide the gate. With ``min_kappa``, Cohen's
    kappa must clear it as well, and a kappa that is undefined (both graders
    used one label only, so there is nothing to discriminate) reports
    "insufficient" rather than passing on raw agreement.
    """
    n = judge_rows.get("n") or 0
    kappa = judge_rows.get("cohens_kappa")
    if n < PUBLISH_GATE_MIN_N:
        result = "insufficient"
    elif judge_rows["agreement"] < PUBLISH_GATE:
        result = "fail"
    elif min_kappa is None:
        result = "pass"
    elif kappa is None:
        result = "insufficient"
    else:
        result = "pass" if kappa >= min_kappa else "fail"
    return {
        "threshold": PUBLISH_GATE,
        "min_n": PUBLISH_GATE_MIN_N,
        "min_kappa": min_kappa,
        "metric": metric,
        "n": n,
        "agreement": judge_rows.get("agreement"),
        "cohens_kappa": kappa,
        "overall_n": overall.get("n"),
        "overall_agreement": overall.get("agreement"),
        "result": result,
    }


def import_blind(run_dir: Path, source: Path | None = None) -> Dict[str, Any]:
    """Compute judge-versus-human agreement from the filled CSVs.

    ``source`` may be the blind directory, or one of the CSV files; by default
    the run's own ``blind/`` directory is read. Writes ``agreement.json`` and
    ``human_grades.json``.
    """
    run_dir = Path(run_dir)
    blind_dir = run_dir / BLIND_DIR
    key_data = json.loads((blind_dir / KEY_FILE).read_text(encoding="utf-8"))
    key, audit_key = key_data["samples"], key_data.get("audit", {})
    answers_path = blind_dir / ANSWERS_CSV
    citations_path = blind_dir / CITATIONS_CSV
    audit_path = blind_dir / AUDIT_CSV
    if source is not None:
        source = Path(source)
        if source.is_dir():
            answers_path = source / ANSWERS_CSV
            citations_path = source / CITATIONS_CSV
            audit_path = source / AUDIT_CSV
        elif source.name == CITATIONS_CSV:
            citations_path = source
        elif source.name == AUDIT_CSV:
            audit_path = source
        else:
            answers_path = source

    correctness: List[Tuple[float, float]] = []
    by_method: Dict[str, List[Tuple[float, float]]] = {}
    abstention: List[Tuple[bool, bool]] = []
    abstention_by_kind: Dict[str, List[Tuple[bool, bool]]] = {}
    human_rows: List[Dict[str, Any]] = []
    if answers_path.exists():
        for row in _read_csv(answers_path):
            entry = key.get(row.get("sample_id", ""))
            human = _parse_grade(_cell(row, "grade"))
            if entry is None or human is None:
                continue
            human_rows.append({
                "item_id": entry["item_id"],
                "condition": entry["condition"],
                "answerable": entry["answerable"],
                "unanswerable_kind": entry.get("unanswerable_kind"),
                "human_grade": human,
            })
            if entry["answerable"]:
                if entry.get("grader_correctness") is None:
                    continue  # the judge failed; the human grade still counts for coverage
                pair = (human, float(entry["grader_correctness"]))
                correctness.append(pair)
                by_method.setdefault(entry.get("grader_method") or "unknown", []).append(pair)
            else:
                # The grader's verdict on what this kind asks for; older runs
                # recorded only "declined".
                grader = entry.get("grader_abstained", entry.get("grader_declined"))
                if grader is None:
                    continue
                pair_b = (human == 1.0, bool(grader))
                abstention.append(pair_b)
                kind = entry.get("unanswerable_kind") or "unknown"
                abstention_by_kind.setdefault(kind, []).append(pair_b)

    support: List[Tuple[bool, bool]] = []
    if citations_path.exists():
        for row in _read_csv(citations_path):
            entry = key.get(row.get("sample_id", ""))
            human = _parse_yes_no(_cell(row, "supported"))
            if entry is None or human is None:
                continue
            grader = entry["grader_supported"].get(row.get("cite_id", ""))
            if grader is not None:
                support.append((human, bool(grader)))

    audit: Dict[str, List[bool]] = {}
    if audit_path.exists():
        for row in _read_csv(audit_path):
            entry = audit_key.get(row.get("audit_id", ""))
            agrees = _parse_yes_no(row.get("human_agrees", ""))
            if entry is None or agrees is None:
                continue
            audit.setdefault(entry["bucket"], []).append(agrees)
    all_audit = [a for rows in audit.values() for a in rows]

    corr = _agreement(correctness)
    judge_rows = _agreement(by_method.get("judge", []))
    support_agreement = _agreement(support)
    result = {
        "correctness": {**corr, "by_method": {m: _agreement(p) for m, p in sorted(by_method.items())}},
        "abstention": {
            **_agreement(abstention),
            "by_kind": {k: _agreement(p) for k, p in sorted(abstention_by_kind.items())},
        },
        "support": support_agreement,
        "resolution_audit": {
            "n": len(all_audit),
            "agreement": (sum(all_audit) / len(all_audit)) if all_audit else None,
            "by_bucket": {
                b: {"n": len(v), "agreement": sum(v) / len(v)} for b, v in sorted(audit.items())
            },
        },
        "n_human_graded_answers": len(human_rows),
        "publish_gate": publish_gate(judge_rows, overall=corr),
        "support_gate": publish_gate(
            support_agreement, overall=support_agreement,
            metric="support agreement and kappa on human-graded citations",
            min_kappa=SUPPORT_GATE_MIN_KAPPA,
        ),
    }
    (run_dir / AGREEMENT_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (run_dir / HUMAN_GRADES_FILE).write_text(
        json.dumps({"n": len(human_rows), "answers": human_rows}, indent=2) + "\n", encoding="utf-8"
    )
    return result
