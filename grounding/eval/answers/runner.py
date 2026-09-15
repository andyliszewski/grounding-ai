"""Answer runner for the grounded-answer benchmark (Epic 25, Story 25.2).

For each selected fixture item and each condition, ask the answer model the
question, run a ``search_corpus`` tool loop in grounded conditions, and append
one JSON line per answer to ``transcripts.jsonl`` as soon as it finishes.

A manual loop over ``client.messages.create`` is used rather than the SDK's
beta tool runner because the benchmark needs an injectable client, per-call
usage and latency, a hard cap on tool rounds with a forced final answer, and
no beta dependency.

Loop semantics: at most ``max_iterations`` tool-use rounds. If the model still
asks for the tool after that, one final call with ``tool_choice: none`` forces
a text answer and the transcript is marked ``max_iterations``, so a scaffold
limit is visible rather than silently scored as a model failure.

Outcome classes (per eval hygiene): ``ok``, ``max_iterations``, ``truncated``
(hit ``max_tokens``) and ``refusal`` rows go to ``transcripts.jsonl``; attempts
that produced nothing scorable (API errors after retries, a served-model
mismatch, a retrieval-stack failure, a budget stop mid-answer) go to
``errors.jsonl`` with their usage, so billed spend is never lost.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from grounding.eval.answers.citations import (
    CorpusIndex,
    parse_cited_version,
    parse_identifiers,
)
from grounding.eval.answers.conditions import ConditionSpec
from grounding.eval.answers.estimate import CHARS_PER_TOKEN, TOOL_USE_OVERHEAD_TOKENS
from grounding.eval.answers.model_client import (
    ANSWER_MAX_TOKENS,
    HarnessError,
    call_model,
    generation_options,
    get_field,
    response_text,
    sdk_version,
    tool_use_blocks,
)
from grounding.eval.answers.pricing import pricing_snapshot, tokens_cost, usage_cost
from grounding.eval.answers.prompts import (
    ANSWER_PROMPT_VERSION,
    DEFAULT_PERSONA,
    answer_system_prompt,
    sha256_json,
    sha256_text,
)
from grounding.eval.answers.provenance import (
    embedding_provenance,
    git_state,
    index_fingerprint,
    sha256_file,
)
from grounding.eval.answers.tool import TOOL_DEFINITION, HarnessToolError, SearchCorpusTool
from grounding.eval.fixtures import FixtureItem, FixtureSet
from grounding.eval.metrics import _page_matches

logger = logging.getLogger("grounding.eval.answers.runner")

TRANSCRIPT_SCHEMA = "grounding-answer-transcript/1"
RUN_SCHEMA = "grounding-answer-run/1"
TRANSCRIPTS_FILE = "transcripts.jsonl"
ERRORS_FILE = "errors.jsonl"
MANIFEST_FILE = "run.json"
RETRIEVAL_FILE = "retrieval.json"
RECALL_K = 5
_MAX_PAUSE_CONTINUATIONS = 3


# ---------------------------------------------------------------------------
# Budget (D6)
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    """The next call could push spend past ``--max-cost``."""


class Budget:
    """Tracks measured spend and refuses calls that could exceed the cap.

    Before each call the projected worst case (estimated input at the model's
    input rate plus ``max_tokens`` at the output rate) is added to the spend so
    far; if that would pass ``max_cost`` the call is not made.
    """

    def __init__(self, max_cost: float | None) -> None:
        self.max_cost = max_cost
        self.spent = 0.0
        self.unpriced_calls = 0

    def check(self, model: str, context_chars: int, max_tokens: int, *, tools: bool) -> None:
        if self.max_cost is None:
            return
        in_tokens = context_chars / CHARS_PER_TOKEN + (TOOL_USE_OVERHEAD_TOKENS if tools else 0)
        projected = tokens_cost(model, in_tokens, max_tokens)
        if projected is None:
            raise BudgetExceeded(f"no price on record for {model}; cannot enforce --max-cost")
        if self.spent + projected > self.max_cost:
            raise BudgetExceeded(
                f"stopping before a call that could exceed --max-cost "
                f"${self.max_cost:.2f} (spent ${self.spent:.4f}, next call up to ${projected:.4f})"
            )

    def charge(self, model: str, usage: Dict[str, Any]) -> float | None:
        cost = usage_cost(model, usage)
        if cost is None:
            self.unpriced_calls += 1
            return None
        self.spent += cost
        return cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(agent: str, *, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return f"answers-{agent}-{moment.strftime('%Y%m%d-%H%M%S')}"


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sum_usage(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for usage in records:
        for key, value in usage.items():
            total[key] = total.get(key, 0) + int(value or 0)
    return total


def select_items(
    fixture_set: FixtureSet,
    *,
    limit: int | None = None,
    item_ids: Sequence[str] | None = None,
) -> Tuple[FixtureItem, ...]:
    """Items to run, in fixture order, filtered by ``--items`` then ``--limit``.

    Every selected item needs an ``answer:`` block: without one there is no
    gold to grade against, and scoring would fail only after the answers had
    been paid for.

    Raises:
        ValueError: an ``--items`` id is not in the fixture, ``limit < 1``, or
            a selected item has no ``answer:`` block.
    """
    items = list(fixture_set.items)
    if item_ids:
        known = {it.id for it in items}
        missing = [i for i in item_ids if i not in known]
        if missing:
            raise ValueError(f"--items not found in fixture: {missing}")
        wanted = set(item_ids)
        items = [it for it in items if it.id in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be >= 1, got {limit}")
        items = items[:limit]
    no_answer = [it.id for it in items if it.answer is None]
    if no_answer:
        raise ValueError(
            f"{len(no_answer)} selected item(s) have no answer: block, so there is no gold "
            f"to grade against: {', '.join(no_answer)}. Add an answer: block to each, or "
            "leave them out with --items"
        )
    return tuple(items)


def check_page_index(items: Sequence[FixtureItem], corpus: CorpusIndex) -> None:
    """Refuse answerable items whose gold document has no page index.

    The gold page is a PDF page index (chunk ``page_start``/``page_end``). A
    document ingested without pages (Markdown, EPUB, a pdftotext fallback)
    can never show the gold page, so gold-page recall, retrieval attribution
    and page-based citation checks would all silently read as misses. The
    same holds for a gold document missing from the agent's index. Checked
    before the estimate, so nothing is spent on an item that cannot be scored.

    Raises:
        ValueError: listing every offending item and document.
    """
    problems: List[str] = []
    for item in items:
        if item.answer is None or not item.answer.answerable:
            continue
        for doc_id in item.expected.doc_ids:
            if doc_id not in corpus.docs:
                problems.append(f"{item.id} (doc {doc_id}: not in the agent's index)")
            elif corpus.max_page(doc_id) is None:
                problems.append(f"{item.id} (doc {doc_id}: no page_start in any chunk)")
    if problems:
        raise ValueError(
            "these answerable items name a gold document without a page index, so the "
            f"gold page cannot be checked: {'; '.join(problems)}. Leave them out with "
            "--items, or point them at a document ingested with pages"
        )


def edition_warnings(fixture_set: FixtureSet, corpus: CorpusIndex) -> List[str]:
    """Documents a citation is likely to version, whose version the fixture cannot check.

    A citation that states an edition, a revision or a year is only checked
    against a document whose version is known (from its name, or the fixture's
    ``editions:`` / ``revisions:`` maps); otherwise it scores ``unresolvable``
    (edition_unknown). Warning about every document with an unknown version
    would fire for most government documents, which are cited without one, so
    this warns only where a version is likely to be stated or to matter:

    * another indexed document has the same title words, so only the version
      tells the two apart;
    * the document's name carries a designation such as NASA-STD-5001 or
      ISO 13485, and standards are revised;
    * a fixture item that names the document states an edition, a revision or
      a year, so the ungrounded answers probably will too.
    """
    offsets = fixture_set.page_offsets
    unknown = {doc_id for doc_id in offsets
               if doc_id in corpus.docs and not corpus.docs[doc_id].version.stated}
    reasons: Dict[str, str] = {}
    titles: Dict[Tuple[str, ...], set] = {}
    for doc in corpus.docs.values():
        for words in doc.title_token_lists:
            titles.setdefault(tuple(sorted(set(words))), set()).add(doc.doc_id)
    for doc_ids in titles.values():
        if len(doc_ids) > 1:
            for doc_id in doc_ids & unknown:
                reasons[doc_id] = "another indexed document has the same title words"
    for doc_id in sorted(unknown):
        if doc_id not in reasons and corpus.docs[doc_id].identifiers:
            reasons[doc_id] = (
                f"its name carries {corpus.docs[doc_id].identifiers[0].text}, and standards "
                "and government documents are revised"
            )
    for item in fixture_set.items:
        gold = item.answer.gold if item.answer is not None else ""
        if not parse_cited_version(f"{item.query} {gold}").stated:
            continue
        for doc_id in item.expected.doc_ids:
            if doc_id in unknown and doc_id not in reasons:
                reasons[doc_id] = f"fixture item {item.id} states an edition, revision or year"
    warnings = [
        f"document {doc_id} has a page_offsets entry but no known edition, revision or year "
        f"({reason}), so a citation that states one cannot be checked against it. Declare it "
        "under the fixture's editions: or revisions: map."
        for doc_id, reason in sorted(reasons.items())
    ]
    bad = sorted(doc_id for doc_id, value in fixture_set.identifiers.items()
                 if not parse_identifiers(value))
    if bad:
        warnings.append(
            f"{len(bad)} declared identifier(s) are not in a recognized series, so they cannot "
            f"be matched: {', '.join(bad)}. See _ID_SERIES in "
            "grounding/eval/answers/citations.py."
        )
    return warnings


# ---------------------------------------------------------------------------
# One answer
# ---------------------------------------------------------------------------

def _content_chars(content: Any) -> int:
    try:
        return len(json.dumps(content, default=str))
    except (TypeError, ValueError):
        return len(str(content))


def answer_one(
    client: Any,
    *,
    item: FixtureItem,
    condition: ConditionSpec,
    model: str,
    tool: SearchCorpusTool | None,
    max_iterations: int,
    budget: Budget,
    run_id: str,
    persona: str = DEFAULT_PERSONA,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Answer one item in one condition and return its transcript row.

    Raises:
        HarnessError, HarnessToolError, BudgetExceeded: with ``partial_calls``
            attached (the call records completed before the failure).
    """
    system = answer_system_prompt(condition.grounded, persona)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": item.query}]
    context_chars = len(system) + len(item.query)
    if condition.grounded:
        context_chars += len(json.dumps(TOOL_DEFINITION))
    calls: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_rounds = 0
    pauses = 0
    forced_final = False
    wall_start = time.perf_counter()
    response = None

    try:
        while True:
            params: Dict[str, Any] = {
                "model": model,
                "max_tokens": ANSWER_MAX_TOKENS,
                "system": system,
                "messages": messages,
                **generation_options(model),
            }
            if condition.grounded:
                params["tools"] = [TOOL_DEFINITION]
                if tool_rounds >= max_iterations:
                    params["tool_choice"] = {"type": "none"}
                    forced_final = True
            budget.check(model, context_chars, ANSWER_MAX_TOKENS, tools=condition.grounded)
            response, record = call_model(client, params, purpose="answer", sleep=sleep)
            row = record.as_dict()
            row["cost_usd"] = budget.charge(record.model, record.usage)
            calls.append(row)

            stop = record.stop_reason
            uses = tool_use_blocks(response)
            if stop == "tool_use" and uses and tool is not None and not forced_final:
                messages.append({"role": "assistant", "content": get_field(response, "content")})
                context_chars += _content_chars(record.content)
                results = []
                for block in uses:
                    block_id = get_field(block, "id")
                    name = get_field(block, "name")
                    tool_input = get_field(block, "input")
                    execution = tool.execute(name, tool_input)
                    tool_calls.append(
                        {
                            "id": block_id,
                            "round": tool_rounds,
                            "name": name,
                            "input": tool_input,
                            "query": execution.query,
                            "top_k": execution.top_k,
                            "is_error": execution.is_error,
                            "latency_s": round(execution.latency_s, 4),
                            "citation_prefixes": execution.citation_prefixes,
                            "results": execution.results,
                            "output_text": execution.output_text,
                        }
                    )
                    result_block: Dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block_id,
                        "content": execution.output_text,
                    }
                    if execution.is_error:
                        result_block["is_error"] = True
                    results.append(result_block)
                    context_chars += len(execution.output_text)
                # All results for one assistant turn go back in one user message.
                messages.append({"role": "user", "content": results})
                tool_rounds += 1
                continue
            if stop == "pause_turn" and pauses < _MAX_PAUSE_CONTINUATIONS:
                messages.append({"role": "assistant", "content": get_field(response, "content")})
                context_chars += _content_chars(record.content)
                pauses += 1
                continue
            break
    except (HarnessError, HarnessToolError, BudgetExceeded) as exc:
        exc.partial_calls = calls  # type: ignore[attr-defined]
        exc.partial_tool_calls = tool_calls  # type: ignore[attr-defined]
        raise

    final_stop = calls[-1]["stop_reason"] if calls else None
    if final_stop == "refusal":
        status = "refusal"
    elif final_stop == "max_tokens":
        status = "truncated"
    elif forced_final:
        status = "max_iterations"
    else:
        status = "ok"

    usage = _sum_usage([c["usage"] for c in calls])
    costs = [c["cost_usd"] for c in calls]
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "run_id": run_id,
        "item_id": item.id,
        "condition": condition.name,
        "status": status,
        "question": item.query,
        "final_text": response_text(response) if response is not None else "",
        "stop_reason": final_stop,
        "model": calls[-1]["model"] if calls else model,
        "n_model_calls": len(calls),
        "n_tool_rounds": tool_rounds,
        "forced_final": forced_final,
        "model_calls": calls,
        "tool_calls": tool_calls,
        "usage": usage,
        "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 6),
        "latency_s": round(
            sum(c["latency_s"] for c in calls) + sum(t["latency_s"] for t in tool_calls), 4
        ),
        "wall_s": round(time.perf_counter() - wall_start, 4),
    }


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------

@dataclass
class AnswerRunSummary:
    run_dir: Path
    written: int = 0
    errors: int = 0
    aborted: str | None = None
    not_run: List[Tuple[str, str]] = field(default_factory=list)
    answer_cost_usd: float = 0.0


def build_manifest(
    *,
    run_id: str,
    agent: str,
    fixture_set: FixtureSet,
    items: Sequence[FixtureItem],
    corpus_dir: Path,
    embeddings_dir: Path,
    conditions: Sequence[ConditionSpec],
    answer_model: str,
    judge_model: str,
    max_iterations: int,
    max_cost: float | None,
    estimate: Dict[str, Any] | None,
    persona: str = DEFAULT_PERSONA,
    source_license: str | None = None,
) -> Dict[str, Any]:
    """Run manifest with the provenance D8 asks for."""
    return {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "agent": agent,
        # Absolute paths, so --run-dir --score works from any directory.
        "fixture": {
            "path": str(Path(fixture_set.source_path).resolve()),
            "sha256": sha256_file(Path(fixture_set.source_path)),
            "n_items_total": len(fixture_set.items),
            "selected_item_ids": [it.id for it in items],
            # Whether the corpus behind this fixture may be published (F6).
            "source_license": source_license,
        },
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "embeddings_dir": str(Path(embeddings_dir).resolve()),
        "conditions": [
            {"name": c.name, "grounded": c.grounded, "retrieval": c.retrieval_config()}
            for c in conditions
        ],
        "answer_model": answer_model,
        "judge_model": judge_model,
        "generation": {
            "answer": {"max_tokens": ANSWER_MAX_TOKENS, **generation_options(answer_model)},
        },
        "max_iterations": max_iterations,
        "max_cost_usd": max_cost,
        "prompts": {
            "answer_version": ANSWER_PROMPT_VERSION,
            "persona": persona,
            "answer_system_sha256": {
                "grounded": sha256_text(answer_system_prompt(True, persona)),
                "ungrounded": sha256_text(answer_system_prompt(False, persona)),
            },
            "tool_definition_sha256": sha256_json(TOOL_DEFINITION),
        },
        "git": git_state(),
        "index_fingerprint": index_fingerprint(embeddings_dir, corpus_dir),
        "pricing": pricing_snapshot([answer_model, judge_model]),
        "environment": {
            "python": platform.python_version(),
            "anthropic_sdk": sdk_version(),
            "embedding": embedding_provenance(),
        },
        "estimate": estimate,
        "answers": None,
    }


def run_answers(
    client: Any,
    *,
    items: Sequence[FixtureItem],
    conditions: Sequence[ConditionSpec],
    answer_model: str,
    corpus_dir: Path,
    embeddings_dir: Path,
    run_dir: Path,
    run_id: str,
    max_iterations: int = 5,
    persona: str = DEFAULT_PERSONA,
    budget: Budget | None = None,
    tool_factory: Callable[[ConditionSpec], SearchCorpusTool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AnswerRunSummary:
    """Answer every (item, condition) pair, item-major.

    Item-major order interleaves conditions in time, so API latency drift or
    a budget stop affects every condition alike instead of starving the last.
    """
    budget = budget or Budget(None)
    run_dir.mkdir(parents=True, exist_ok=True)
    transcripts = run_dir / TRANSCRIPTS_FILE
    errors = run_dir / ERRORS_FILE

    def _default_factory(cond: ConditionSpec) -> SearchCorpusTool:
        return SearchCorpusTool(cond, corpus_dir=corpus_dir, embeddings_dir=embeddings_dir)

    factory = tool_factory or _default_factory
    tools = {c.name: factory(c) for c in conditions if c.grounded}
    summary = AnswerRunSummary(run_dir=run_dir)
    pairs = [(item, cond) for item in items for cond in conditions]

    for index, (item, cond) in enumerate(pairs):
        try:
            row = answer_one(
                client,
                item=item,
                condition=cond,
                model=answer_model,
                tool=tools.get(cond.name),
                max_iterations=max_iterations,
                budget=budget,
                run_id=run_id,
                persona=persona,
                sleep=sleep,
            )
        except (HarnessError, HarnessToolError, BudgetExceeded) as exc:
            partial = getattr(exc, "partial_calls", [])
            usage_rows = [c["usage"] for c in partial]
            extra_usage = getattr(exc, "usage", None)
            served = getattr(exc, "model", None) or answer_model
            if extra_usage:
                budget.charge(served, extra_usage)
                usage_rows.append(extra_usage)
            failure = (
                "budget_exceeded"
                if isinstance(exc, BudgetExceeded)
                else "tool_error"
                if isinstance(exc, HarnessToolError)
                else exc.failure_class
            )
            usage = _sum_usage(usage_rows)
            append_jsonl(
                errors,
                {
                    "run_id": run_id,
                    "stage": "answer",
                    "item_id": item.id,
                    "condition": cond.name,
                    "failure_class": failure,
                    "message": str(exc),
                    "attempts": getattr(exc, "attempts", None),
                    "model": served,
                    "usage": usage,
                    "cost_usd": usage_cost(served, usage) if usage else 0.0,
                    "partial_model_calls": len(partial),
                    "utc": utc_now(),
                },
            )
            summary.errors += 1
            logger.warning("answer %s/%s failed: %s", item.id, cond.name, exc)
            if isinstance(exc, BudgetExceeded):
                summary.aborted = str(exc)
                summary.not_run = [(i.id, c.name) for i, c in pairs[index:]]
                break
            continue

        append_jsonl(transcripts, row)
        summary.written += 1
        logger.info(
            "answered %s/%s status=%s tool_rounds=%d cost=%s",
            item.id,
            cond.name,
            row["status"],
            row["n_tool_rounds"],
            row["cost_usd"],
        )

    summary.answer_cost_usd = budget.spent
    return summary


# ---------------------------------------------------------------------------
# Retrieval recall (metric 4, from the Epic 16 runner)
# ---------------------------------------------------------------------------

def compute_retrieval_recall(
    fixture_set: FixtureSet,
    items: Sequence[FixtureItem],
    conditions: Sequence[ConditionSpec],
    *,
    corpus_dir: Path,
    embeddings_dir: Path,
    run_eval_fn: Callable[..., Any] | None = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """recall@5 of the gold document and gold page, per grounded condition.

    Runs the Epic 16 ``run_eval`` on the same fixture items with each
    condition's retrieval settings. The runner's own recall is doc-level; the
    gold-page figure counts an item as a hit when a top-5 chunk from an
    expected document overlaps ``expected.page``.
    """
    if run_eval_fn is None:
        from grounding.eval.runner import run_eval as run_eval_fn  # noqa: N811

    selected = replace(fixture_set, items=tuple(items))
    out: Dict[str, Any] = {"k": RECALL_K, "conditions": {}}
    for cond in conditions:
        if not cond.grounded:
            continue
        result = run_eval_fn(
            selected,
            fixture_set.agent,
            corpus_dir=corpus_dir,
            embeddings_dir=embeddings_dir,
            top_k=top_k,
            rerank_config=cond.rerank,
            hybrid_config=cond.hybrid,
        )
        per_item: Dict[str, Any] = {}
        doc_hits = page_hits = page_items = 0
        for res in result.items:
            expected = set(res.expected_doc_ids)
            page_rank = None
            if res.expected_page is not None:
                page_items += 1
                for r in res.retrieved:
                    if r.rank > RECALL_K:
                        break
                    if r.doc_id in expected and _page_matches(
                        res.expected_page, r.page_start, r.page_end
                    ):
                        page_rank = r.rank
                        break
                if page_rank is not None:
                    page_hits += 1
            doc_rank = res.first_hit_rank
            if doc_rank is not None and doc_rank <= RECALL_K:
                doc_hits += 1
            per_item[res.item_id] = {"doc_hit_rank": doc_rank, "page_hit_rank": page_rank}
        n = len(result.items)
        out["conditions"][cond.name] = {
            "n_items": n,
            "n_page_items": page_items,
            "recall_at_5_doc": doc_hits / n if n else None,
            "recall_at_5_gold_page": page_hits / page_items if page_items else None,
            "skipped": list(result.skipped),
            "per_item": per_item,
        }
    return out
