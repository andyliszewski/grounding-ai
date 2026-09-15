"""Token and dollar estimates for ``grounding eval-answers --dry-run`` (D6).

A dry run needs no API key, so it cannot use the token-counting endpoint.
It estimates from character counts with a fixed chars-per-token ratio and a
short list of behavioral assumptions (searches per answer, output lengths,
citations per answer). Every assumption is printed with the estimate. The
estimate is deliberately conservative: it assumes every answer goes to the
correctness judge and every citation to the support judge.

Treat it as an order-of-magnitude guide. For a measured figure, run a small
pilot (``--limit 2``) and read the recorded usage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from grounding.eval.answers.conditions import ConditionSpec
from grounding.eval.answers.pricing import price_for, tokens_cost
from grounding.eval.answers.prompts import DEFAULT_PERSONA, answer_system_prompt
from grounding.eval.answers.tool import DEFAULT_TOP_K, TOOL_DEFINITION
from grounding.eval.fixtures import FixtureItem

CHARS_PER_TOKEN = 3.5
TOOL_USE_OVERHEAD_TOKENS = 350  # API-side tool-use system prompt
MESSAGE_OVERHEAD_TOKENS = 10
EST_SEARCHES_PER_ANSWER = 2
EST_TOOL_TURN_OUTPUT_TOKENS = 400  # thinking plus one tool_use block
EST_FINAL_OUTPUT_TOKENS = 1500  # thinking plus the final answer
EST_ANSWER_TEXT_CHARS = 1600  # final answer text a judge reads
EST_CITATIONS_PER_ANSWER = 4
EST_CLAIM_CHARS = 250
EST_JUDGE_OUTPUT_TOKENS = 600  # thinking plus a short JSON verdict
EST_UNGROUNDED_PASSAGE_CHUNKS = 2  # chunks covering one resolved printed page
RESULT_HEADER_CHARS = 120  # "### [n] source (score)", prefix, doc/chunk line
FALLBACK_CHUNK_CHARS = 1200  # grounding's default chunk size
MAX_CHUNK_SAMPLE = 200

# Judge prompt sizes, in characters. Story 25.3 replaces these with the
# lengths of the real versioned prompts via ``judge_prompt_chars``.
DEFAULT_JUDGE_PROMPT_CHARS = {
    "correctness": 2500,
    "abstention_no_source": 1500,
    "abstention_not_in_corpus": 1800,
    "support": 2000,
}


def tokens(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


@dataclass
class ConditionEstimate:
    condition: str
    n_answers: int
    answer_input_tokens: float = 0.0
    answer_output_tokens: float = 0.0
    judge_input_tokens: float = 0.0
    judge_output_tokens: float = 0.0
    answer_usd: float | None = 0.0
    judge_usd: float | None = 0.0

    @property
    def total_usd(self) -> float | None:
        if self.answer_usd is None or self.judge_usd is None:
            return None
        return self.answer_usd + self.judge_usd


@dataclass
class RunEstimate:
    answer_model: str
    judge_model: str
    n_items: int
    n_answerable: int
    n_unanswerable: int
    include_judging: bool
    per_condition: List[ConditionEstimate] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_usd(self) -> float | None:
        totals = [c.total_usd for c in self.per_condition]
        if any(t is None for t in totals):
            return None
        return float(sum(totals))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "n_items": self.n_items,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "include_judging": self.include_judging,
            "total_usd": self.total_usd,
            "per_condition": [
                {
                    "condition": c.condition,
                    "n_answers": c.n_answers,
                    "answer_input_tokens": round(c.answer_input_tokens),
                    "answer_output_tokens": round(c.answer_output_tokens),
                    "judge_input_tokens": round(c.judge_input_tokens),
                    "judge_output_tokens": round(c.judge_output_tokens),
                    "answer_usd": c.answer_usd,
                    "judge_usd": c.judge_usd,
                }
                for c in self.per_condition
            ],
            "assumptions": self.assumptions,
        }

    def render(self) -> str:
        lines = []
        lines.append(
            f"items={self.n_items} ({self.n_answerable} answerable, "
            f"{self.n_unanswerable} unanswerable)  "
            f"conditions={','.join(c.condition for c in self.per_condition)}"
        )
        lines.append(
            f"answer model: {self.answer_model} {_price_label(self.answer_model)}"
        )
        if self.include_judging:
            lines.append(f"judge model:  {self.judge_model} {_price_label(self.judge_model)}")
        header = (
            f"{'condition':<15}{'answers':>8}{'answer in':>12}{'answer out':>12}"
            f"{'judge in':>11}{'judge out':>11}{'est. USD':>11}"
        )
        lines.append(header)
        for c in self.per_condition:
            lines.append(
                f"{c.condition:<15}{c.n_answers:>8}{c.answer_input_tokens:>12,.0f}"
                f"{c.answer_output_tokens:>12,.0f}{c.judge_input_tokens:>11,.0f}"
                f"{c.judge_output_tokens:>11,.0f}{_usd(c.total_usd):>11}"
            )
        lines.append(f"{'total':<15}{sum(c.n_answers for c in self.per_condition):>8}"
                     f"{'':>56}{_usd(self.total_usd):>11}")
        lines.append("assumptions: " + "; ".join(f"{k}={v}" for k, v in self.assumptions.items()))
        return "\n".join(lines)


def _usd(value: float | None) -> str:
    return "unpriced" if value is None else f"${value:,.2f}"


def _price_label(model: str) -> str:
    price = price_for(model)
    if price is None:
        return "(no price on record)"
    return f"(${price.input_per_mtok:.2f} in / ${price.output_per_mtok:.2f} out per MTok)"


def average_chunk_chars(corpus_dir: Path, embeddings_dir: Path) -> tuple[float, int]:
    """Mean chunk body length over an evenly spaced sample of indexed chunks.

    Returns ``(mean_chars, n_sampled)``; falls back to the default chunk size
    when no chunk file can be read.
    """
    chunk_map_path = Path(embeddings_dir) / "_chunk_map.json"
    try:
        data = json.loads(chunk_map_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return float(FALLBACK_CHUNK_CHARS), 0
    entries = data if isinstance(data, list) else data.get("chunks", [])
    paths = [
        (e if isinstance(e, str) else e.get("file_path"))
        for e in entries
        if (isinstance(e, str) or not e.get("deleted_utc"))
    ]
    paths = [p for p in paths if p]
    if not paths:
        return float(FALLBACK_CHUNK_CHARS), 0
    step = max(1, len(paths) // MAX_CHUNK_SAMPLE)
    lengths = []
    for rel in paths[::step][:MAX_CHUNK_SAMPLE]:
        try:
            text = (Path(corpus_dir) / rel).read_text(encoding="utf-8")
        except OSError:
            continue
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2]
        lengths.append(len(text.strip()))
    if not lengths:
        return float(FALLBACK_CHUNK_CHARS), 0
    return sum(lengths) / len(lengths), len(lengths)


def estimate_run(
    items: Sequence[FixtureItem],
    conditions: Sequence[ConditionSpec],
    *,
    answer_model: str,
    judge_model: str,
    avg_chunk_chars: float,
    n_chunks_sampled: int,
    max_iterations: int,
    include_judging: bool = True,
    judge_prompt_chars: Dict[str, int] | None = None,
    persona: str = DEFAULT_PERSONA,
) -> RunEstimate:
    judge_chars = dict(DEFAULT_JUDGE_PROMPT_CHARS)
    judge_chars.update(judge_prompt_chars or {})
    searches = min(EST_SEARCHES_PER_ANSWER, max_iterations)
    tool_def_tokens = TOOL_USE_OVERHEAD_TOKENS + tokens(len(json.dumps(TOOL_DEFINITION)))
    result_tokens = tokens(
        DEFAULT_TOP_K * (avg_chunk_chars + RESULT_HEADER_CHARS) + 80
    )

    answerable = [it for it in items if it.answer is None or it.answer.answerable]
    unanswerable = [it for it in items if it.answer is not None and not it.answer.answerable]

    est = RunEstimate(
        answer_model=answer_model,
        judge_model=judge_model,
        n_items=len(items),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        include_judging=include_judging,
        assumptions={
            "chars_per_token": CHARS_PER_TOKEN,
            "avg_chunk_chars": round(avg_chunk_chars),
            "chunks_sampled": n_chunks_sampled,
            "searches_per_grounded_answer": searches,
            "top_k": DEFAULT_TOP_K,
            "output_tokens_per_search_turn": EST_TOOL_TURN_OUTPUT_TOKENS,
            "output_tokens_final_answer": EST_FINAL_OUTPUT_TOKENS,
            "citations_per_answer_all_judged": EST_CITATIONS_PER_ANSWER,
            "judge_output_tokens_per_call": EST_JUDGE_OUTPUT_TOKENS,
            "prompt_caching": "off",
        },
    )

    for cond in conditions:
        ce = ConditionEstimate(condition=cond.name, n_answers=len(items))
        system_chars = len(answer_system_prompt(cond.grounded, persona))
        for item in items:
            base_in = tokens(system_chars + len(item.query)) + MESSAGE_OVERHEAD_TOKENS
            if cond.grounded:
                for k in range(searches + 1):
                    ce.answer_input_tokens += (
                        base_in
                        + tool_def_tokens
                        + k * (EST_TOOL_TURN_OUTPUT_TOKENS + result_tokens)
                    )
                ce.answer_output_tokens += (
                    searches * EST_TOOL_TURN_OUTPUT_TOKENS + EST_FINAL_OUTPUT_TOKENS
                )
            else:
                ce.answer_input_tokens += base_in
                ce.answer_output_tokens += EST_FINAL_OUTPUT_TOKENS

            if not include_judging:
                continue
            is_unanswerable = item.answer is not None and not item.answer.answerable
            kind = item.answer.unanswerable_kind if is_unanswerable else None
            if kind == "not_in_corpus" and not cond.grounded:
                continue  # never scored in the ungrounded condition
            gold_chars = 0
            if item.answer is not None:
                gold_chars = len(item.answer.gold) + sum(len(f) for f in item.answer.must_include)
            grade_prompt = f"abstention_{kind}" if is_unanswerable else "correctness"
            ce.judge_input_tokens += tokens(
                judge_chars[grade_prompt] + len(item.query) + gold_chars + EST_ANSWER_TEXT_CHARS
            )
            ce.judge_output_tokens += EST_JUDGE_OUTPUT_TOKENS
            passage_chars = (avg_chunk_chars + RESULT_HEADER_CHARS) * (
                1 if cond.grounded else EST_UNGROUNDED_PASSAGE_CHUNKS
            )
            per_citation_in = tokens(
                judge_chars["support"] + len(item.query) + EST_CLAIM_CHARS + passage_chars
            )
            ce.judge_input_tokens += EST_CITATIONS_PER_ANSWER * per_citation_in
            ce.judge_output_tokens += EST_CITATIONS_PER_ANSWER * EST_JUDGE_OUTPUT_TOKENS

        ce.answer_usd = tokens_cost(answer_model, ce.answer_input_tokens, ce.answer_output_tokens)
        ce.judge_usd = (
            tokens_cost(judge_model, ce.judge_input_tokens, ce.judge_output_tokens)
            if include_judging
            else 0.0
        )
        est.per_condition.append(ce)
    return est
