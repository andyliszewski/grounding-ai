"""Scoring for the grounded-answer benchmark (Epic 25, Story 25.3).

Reads ``transcripts.jsonl`` and writes one row per answer to
``scores.jsonl``:

* **citations**: every extracted citation with its bucket (verified,
  unsupported, invented, unresolvable), how it resolved, and the support
  judge's verdict and one-line reason where a judge was needed;
* **correctness** (answerable items): numeric auto-score first, else the
  rubric judge against the gold answer and ``must_include``. The judge sees
  the answer with citations removed and never sees retrieved text (D5);
* **abstention** (unanswerable items): did the model do what the question's
  kind asks for? The judge's framing depends on
  ``answer.unanswerable_kind``. A ``no_source`` question has no answer from
  any source, so the desired behavior is declining, and it is scored in every
  condition. A ``not_in_corpus`` question is answered elsewhere but not by
  the library, and the grounded prompt asks the model to say plainly that the
  sources do not contain the answer and to label any general knowledge it
  adds, so the desired behavior is ``acknowledged_gap`` (flagged, and any
  answer labeled), with ``declined`` reported beside it; it is scored for
  grounded conditions only. In the ungrounded condition a ``not_in_corpus``
  answer is not scored at all (``excluded``), because a model without the
  library may know the answer from elsewhere, so it never counts against that
  condition.

Judge calls share the run's ``--max-cost`` budget. A judge call that fails
leaves that grade empty (counted as unjudged in the report) and is logged to
``errors.jsonl``; it is never scored as a model failure.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from grounding.eval.answers.citations import (
    BUCKETS,
    CorpusIndex,
    extract_citations,
    resolve_grounded,
    resolve_ungrounded,
    returned_chunks,
    strip_citations,
)
from grounding.eval.answers.conditions import CONDITIONS
from grounding.eval.answers.model_client import (
    JUDGE_MAX_TOKENS,
    HarnessError,
    call_model,
    generation_options,
    response_text,
)
from grounding.eval.answers.numeric import numeric_check
from grounding.eval.answers.prompts import (
    JUDGE_PROMPTS,
    abstention_prompt_key,
    abstention_user_message,
    correctness_user_message,
    support_user_message,
)
from grounding.eval.answers.runner import (
    ERRORS_FILE,
    TRANSCRIPTS_FILE,
    Budget,
    BudgetExceeded,
    append_jsonl,
    read_jsonl,
    utc_now,
)
from grounding.eval.fixtures import FixtureItem
from grounding.eval.metrics import _page_matches

logger = logging.getLogger("grounding.eval.answers.scoring")

SCORES_FILE = "scores.jsonl"
SCORE_SCHEMA = "grounding-answer-score/3"
UNSCORED_STATUSES = ("refusal", "truncated")
# The behavior each unanswerable kind asks for. A ``no_source`` question has no
# answer anywhere, so declining is the whole of it. A ``not_in_corpus`` question
# is answered elsewhere: the grounded prompt asks the model to say plainly that
# the sources do not contain it and to label any general knowledge it adds, so
# acknowledging the gap is the desired behavior, whether or not it then answers.
DESIRED_ABSTENTION = {"no_source": "declined", "not_in_corpus": "acknowledged_gap"}
# A not_in_corpus item in the ungrounded condition: the model had no library,
# so declining is not the expected behavior there. Never scored.
EXCLUDED_NOT_IN_CORPUS = "not_in_corpus_ungrounded"


def abstained_as_desired(abstention: Mapping[str, Any] | None, kind: str | None) -> bool | None:
    """Did this answer do what its unanswerable kind asks for? None if unjudged."""
    value = (abstention or {}).get(DESIRED_ABSTENTION.get(kind or "", "declined"))
    return None if value is None else bool(value)


def desired_behavior(kind: str | None) -> str:
    """The name of the behavior scored for an unanswerable kind."""
    return DESIRED_ABSTENTION.get(kind or "", "declined")


def judge_generation(model: str) -> Dict[str, Any]:
    """Request options for judge calls (recorded in the manifest)."""
    return {"max_tokens": JUDGE_MAX_TOKENS, **generation_options(model)}


@dataclass
class Verdict:
    data: Dict[str, Any] | None
    error: str | None = None
    cached: bool = False


class Judge:
    """Structured-output judge calls against versioned prompts."""

    def __init__(
        self,
        client: Any,
        model: str,
        budget: Budget,
        *,
        errors_path: Path | None = None,
        run_id: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget
        self.errors_path = errors_path
        self.run_id = run_id
        self.sleep = sleep
        self._support_cache: Dict[str, Verdict] = {}

    def _ask(
        self, kind: str, user: str, *, item_id: str, condition: str, calls: List[Dict[str, Any]]
    ) -> Verdict:
        version, system, schema = JUDGE_PROMPTS[kind]
        options = generation_options(self.model)
        output_config = dict(options.pop("output_config", {}))
        output_config["format"] = {"type": "json_schema", "schema": schema}
        params = {
            "model": self.model,
            "max_tokens": JUDGE_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
            **options,
        }
        self.budget.check(self.model, len(system) + len(user), JUDGE_MAX_TOKENS, tools=False)
        try:
            response, record = call_model(
                self.client, params, purpose=f"judge_{kind}", sleep=self.sleep
            )
        except HarnessError as exc:
            if exc.usage:
                self.budget.charge(exc.model or self.model, exc.usage)
            self._log_error(kind, item_id, condition, exc.failure_class, str(exc), exc.usage)
            return Verdict(None, f"{exc.failure_class}: {exc}")
        cost = self.budget.charge(record.model, record.usage)
        calls.append(
            {
                "kind": kind,
                "prompt_version": version,
                "model": record.model,
                "stop_reason": record.stop_reason,
                "usage": record.usage,
                "cost_usd": cost,
                "latency_s": round(record.latency_s, 4),
                "attempts": record.attempts,
            }
        )
        if record.stop_reason in ("refusal", "max_tokens"):
            self._log_error(kind, item_id, condition, f"judge_{record.stop_reason}",
                            "judge produced no verdict", record.usage)
            return Verdict(None, f"judge stop_reason {record.stop_reason}")
        try:
            data = json.loads(response_text(response))
            missing = [k for k in schema["required"] if k not in data]
            if missing:
                raise ValueError(f"missing keys {missing}")
        except (ValueError, TypeError) as exc:
            self._log_error(kind, item_id, condition, "judge_unparseable", str(exc), record.usage)
            return Verdict(None, f"unparseable verdict: {exc}")
        return Verdict(data)

    def _log_error(self, kind, item_id, condition, failure, message, usage) -> None:
        if self.errors_path is None:
            return
        append_jsonl(
            self.errors_path,
            {
                "run_id": self.run_id,
                "stage": f"judge_{kind}",
                "item_id": item_id,
                "condition": condition,
                "failure_class": failure,
                "message": message,
                "model": self.model,
                "usage": usage or {},
                "utc": utc_now(),
            },
        )

    def correctness(self, item: FixtureItem, candidate: str, *, condition: str, calls) -> Verdict:
        user = correctness_user_message(item.query, item.answer, candidate)
        return self._ask("correctness", user, item_id=item.id, condition=condition, calls=calls)

    def abstention(self, item: FixtureItem, candidate: str, *, condition: str, calls) -> Verdict:
        user = abstention_user_message(item.query, candidate)
        kind = abstention_prompt_key(item.answer.unanswerable_kind)
        return self._ask(kind, user, item_id=item.id, condition=condition, calls=calls)

    def support(self, item: FixtureItem, claim: str, passage: str, *, condition: str, calls) -> Verdict:
        key = hashlib.sha256(f"{item.query}\x00{claim}\x00{passage}".encode("utf-8")).hexdigest()
        if key in self._support_cache:
            cached = self._support_cache[key]
            return Verdict(cached.data, cached.error, cached=True)
        user = support_user_message(item.query, claim, passage)
        verdict = self._ask("support", user, item_id=item.id, condition=condition, calls=calls)
        if verdict.data is not None:
            self._support_cache[key] = verdict
        return verdict


def gold_in_context(item: FixtureItem, returned) -> Dict[str, Any]:
    """Whether any tool call in the transcript surfaced the gold document and page.

    A wrong grounded answer with the gold page in context is a generation
    failure; without it, a retrieval failure. Uses the same page-overlap rule
    as gold-page recall.
    """
    expected = set(item.expected.doc_ids)
    docs = [r for r in returned if r.get("doc_id") in expected]
    page = item.expected.page
    page_hit = None
    if page is not None:
        page_hit = any(_page_matches(page, r.get("page_start"), r.get("page_end")) for r in docs)
    return {"gold_doc_in_context": bool(docs), "gold_page_in_context": page_hit}


def score_answer(
    row: Mapping[str, Any],
    item: FixtureItem,
    *,
    corpus: CorpusIndex,
    page_offsets: Mapping[str, int],
    judge: Judge,
) -> Dict[str, Any]:
    """Score one transcript row. Raises BudgetExceeded if the cap is hit."""
    condition = row["condition"]
    grounded = CONDITIONS[condition].grounded
    answer = item.answer
    answerable = answer is None or answer.answerable
    kind = answer.unanswerable_kind if answer is not None else None
    calls: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "run_id": row.get("run_id"),
        "item_id": item.id,
        "condition": condition,
        "status": row["status"],
        "category": answer.category if answer else None,
        "answerable": answerable,
        "unanswerable_kind": kind,
        "tags": list(item.tags),
        "scored": True,
        "excluded": None,
        "gold_doc_in_context": None,
        "gold_page_in_context": None,
        "correctness": None,
        "abstention": None,
        "citations": [],
        "bucket_counts": {b: 0 for b in BUCKETS},
        "n_citations": 0,
        "n_unjudged_citations": 0,
        "judge_calls": calls,
    }
    if answer is None:
        # Item selection refuses these before any spend; a fixture edited
        # between answering and re-scoring must not crash scoring.
        out["scored"] = False
        out["note"] = "not scored: the fixture item has no answer: block"
        return out
    if row["status"] in UNSCORED_STATUSES:
        out["scored"] = False
        out["note"] = f"not scored: answer status {row['status']}"
        return out
    if not answerable and kind == "not_in_corpus" and not grounded:
        out["scored"] = False
        out["excluded"] = EXCLUDED_NOT_IN_CORPUS
        out["note"] = ("not scored: not_in_corpus items are scored for grounded "
                       "conditions only")
        return out

    text = row.get("final_text") or ""
    returned = returned_chunks(row.get("tool_calls") or [])
    if grounded:
        # For the resolution audit: what the model was shown.
        out["returned_prefixes"] = [r.get("prefix") for r in returned if r.get("prefix")]
        if answerable:
            # Retrieval versus generation: was the gold page ever in the model's context?
            out.update(gold_in_context(item, returned))
    known_slugs = {r.get("slug") for r in returned if r.get("slug")} | corpus.prefix_slugs
    citations = extract_citations(text, known_slugs)
    stripped = strip_citations(text, citations)
    out["answer_without_citations"] = stripped

    for citation in citations:
        if grounded:
            resolution = resolve_grounded(citation, returned, corpus)
        else:
            resolution = resolve_ungrounded(citation, corpus, page_offsets)
        entry = citation.as_dict()
        entry["resolution"] = resolution.as_dict()
        entry["passage"] = resolution.passage
        entry["bucket"] = resolution.bucket
        entry["judge"] = None
        if resolution.bucket is None:
            claim = citation.claim or stripped[:600]
            verdict = judge.support(item, claim, resolution.passage, condition=condition, calls=calls)
            if verdict.data is None:
                entry["judge_error"] = verdict.error
                out["n_unjudged_citations"] += 1
            else:
                if not verdict.data["supported"]:
                    entry["bucket"] = "unsupported"
                elif resolution.match == "partial":
                    entry["bucket"] = "partial"
                else:
                    entry["bucket"] = "verified"
                entry["judge"] = {
                    "supported": bool(verdict.data["supported"]),
                    "reason": verdict.data.get("reason", ""),
                    "cached": verdict.cached,
                }
        if entry["bucket"] is not None:
            out["bucket_counts"][entry["bucket"]] += 1
        out["citations"].append(entry)
    out["n_citations"] = len(citations)

    if answerable:
        out["correctness"] = _grade_correctness(item, stripped, judge, condition, calls)
    else:
        out["abstention"] = _grade_abstention(item, stripped, judge, condition, calls)

    costs = [c["cost_usd"] for c in calls]
    out["judge_cost_usd"] = None if any(c is None for c in costs) else round(sum(costs), 6)
    return out


def _grade_correctness(item, stripped, judge, condition, calls) -> Dict[str, Any]:
    if not stripped.strip():
        return {"score": 0.0, "method": "empty_answer", "declined": False,
                "reason": "the answer has no text", "numeric_check": None}
    numeric = None
    if item.answer.numeric is not None:
        numeric = numeric_check(
            stripped, item.answer.numeric, has_must_include=bool(item.answer.must_include)
        )
        if numeric["verdict"] is not None:
            return {"score": numeric["verdict"], "method": "numeric", "declined": False,
                    "reason": numeric["reason"], "numeric_check": numeric}
    verdict = judge.correctness(item, stripped, condition=condition, calls=calls)
    if verdict.data is None:
        return {"score": None, "method": "judge_error", "declined": None,
                "reason": verdict.error, "numeric_check": numeric}
    return {
        "score": float(verdict.data["score"]),
        "method": "judge",
        "declined": bool(verdict.data["declined"]),
        "missing_facts": list(verdict.data.get("missing_facts") or []),
        "reason": verdict.data.get("reason", ""),
        "numeric_check": numeric,
    }


def _grade_abstention(item, stripped, judge, condition, calls) -> Dict[str, Any]:
    kind = item.answer.unanswerable_kind
    if not stripped.strip():
        out = {"declined": False, "method": "empty_answer", "reason": "the answer has no text"}
        if kind == "not_in_corpus":
            out["acknowledged_gap"] = False
        return out
    verdict = judge.abstention(item, stripped, condition=condition, calls=calls)
    if verdict.data is None:
        out = {"declined": None, "method": "judge_error", "reason": verdict.error}
        if kind == "not_in_corpus":
            out["acknowledged_gap"] = None
        return out
    data = verdict.data
    out = {"method": "judge", "reason": data.get("reason", "")}
    if kind == "not_in_corpus":
        # The grounded prompt tells the model to say plainly when the sources do
        # not contain the answer and to label any general knowledge it adds, so
        # the desired behavior is acknowledging the gap, not silence.
        flagged, answered = bool(data["gap_flagged"]), bool(data["gave_answer"])
        labeled = bool(data["answer_labeled"])
        out.update(
            gap_flagged=flagged,
            gave_answer=answered,
            answer_labeled=labeled,
            acknowledged_gap=flagged and (not answered or labeled),
            declined=flagged and not answered,
        )
    else:
        out["declined"] = bool(data["declined"])
    return out


@dataclass
class ScoringSummary:
    scored: int = 0
    skipped: int = 0
    judge_calls: int = 0
    judge_cost_usd: float = 0.0
    aborted: str | None = None


def score_run(
    run_dir: Path,
    *,
    items: Mapping[str, FixtureItem],
    page_offsets: Mapping[str, int],
    corpus: CorpusIndex,
    client: Any,
    judge_model: str,
    budget: Budget,
    run_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ScoringSummary:
    """Score every transcript in ``run_dir`` into a fresh ``scores.jsonl``."""
    transcripts = read_jsonl(run_dir / TRANSCRIPTS_FILE)
    scores_path = run_dir / SCORES_FILE
    scores_path.write_text("", encoding="utf-8")
    judge = Judge(client, judge_model, budget, errors_path=run_dir / ERRORS_FILE,
                  run_id=run_id, sleep=sleep)
    summary = ScoringSummary()
    spent_before = budget.spent
    for row in transcripts:
        item = items.get(row["item_id"])
        if item is None:
            logger.warning("transcript item %s is not in the fixture; skipping", row["item_id"])
            summary.skipped += 1
            continue
        if item.answer is None:
            logger.warning("fixture item %s has no answer: block; skipping", row["item_id"])
            summary.skipped += 1
            continue
        try:
            scored = score_answer(row, item, corpus=corpus, page_offsets=page_offsets, judge=judge)
        except BudgetExceeded as exc:
            summary.aborted = str(exc)
            break
        append_jsonl(scores_path, scored)
        summary.scored += 1
        summary.judge_calls += len(scored["judge_calls"])
    summary.judge_cost_usd = budget.spent - spent_before
    return summary
