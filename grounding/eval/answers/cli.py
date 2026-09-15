"""``grounding eval-answers`` subcommand (Epic 25).

A new run answers each fixture question in each condition with Claude,
records transcripts, computes retrieval recall with the Epic 16 runner,
scores correctness and citations with the judge, and exports a blind-grade
sample. Later steps work on an existing run directory:

* ``--run-dir DIR --score``: re-score its transcripts (for example with a
  different ``--judge-model``) without re-answering;
* ``--run-dir DIR --import-grades [PATH]``: read the human blind grades and
  compute judge agreement (with every answer graded, the report makes the
  human grades the primary correctness metric);
* ``--run-dir DIR --export-blind``: re-export the blind-grade files from the
  existing scores (for example at another ``--blind-fraction``), with no judge
  calls; CSVs that already hold grades are archived first;
* ``--run-dir DIR --compare-run OTHER``: run-to-run agreement with a second run
  of the same condition(s) on the same items (the replicate check);
* ``--run-dir DIR --report [--publishable]``: re-render the report. Scoring
  and grade import re-render it too. ``--publishable`` also writes a
  ``publishable/`` copy with aggregate scores only and no source text (D7).

Exit codes:

====  ==========================================================
0     success (or a dry run)
2     bad arguments, fixture or agent problems
3     embeddings index (or BM25 sidecar for hybrid) missing
4     unexpected failure
5     no ANTHROPIC_API_KEY (or SDK missing) for a real run
6     --max-cost would be or was exceeded (partial results kept)
====  ==========================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

from grounding.eval.answers.blind import (
    DEFAULT_FRACTION,
    DEFAULT_SEED,
    export_blind,
    import_blind,
)
from grounding.eval.answers.citations import CorpusIndex
from grounding.eval.answers.conditions import CONDITION_ORDER, parse_conditions
from grounding.eval.answers.estimate import average_chunk_chars, estimate_run
from grounding.eval.answers.model_client import make_anthropic_client
from grounding.eval.answers.pricing import price_for
from grounding.eval.answers.replicate import compare_runs, render_replicate_summary
from grounding.eval.answers.report import write_report
from grounding.eval.answers.prompts import (
    DEFAULT_PERSONA,
    JUDGE_PROMPTS,
    judge_prompt_fingerprints,
)
from grounding.eval.answers.provenance import sha256_file
from grounding.eval.answers.runner import (
    MANIFEST_FILE,
    RETRIEVAL_FILE,
    Budget,
    build_manifest,
    check_page_index,
    compute_retrieval_recall,
    edition_warnings,
    make_run_id,
    read_jsonl,
    run_answers,
    select_items,
    utc_now,
    write_json,
)
from grounding.eval.answers.scoring import SCORES_FILE, judge_generation, score_run
from grounding.eval.fixtures import (
    FixtureValidationError,
    UnknownAgentError,
    load_fixtures,
)

logger = logging.getLogger("grounding.eval.answers.cli")

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_INDEX_MISSING = 3
EXIT_UNEXPECTED = 4
EXIT_NO_API_KEY = 5
EXIT_MAX_COST = 6

DEFAULT_MODEL = "claude-opus-5"
# A different model grades than answers, so the judge is not grading its own work.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


def _create_eval_answers_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``eval-answers`` subcommand on the given subparsers."""
    p = subparsers.add_parser(
        "eval-answers",
        help="Grounded-answer benchmark: score Claude's answers and citations",
        description=(
            "Answer each fixture question with Claude in up to four conditions "
            "(ungrounded, dense, hybrid, hybrid-rerank), record transcripts, and "
            "score correctness and citation verifiability. Use --dry-run first: "
            "it estimates tokens and dollars and needs no API key."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate cost, no key needed
  grounding eval-answers --agent mini --agents-dir tests/eval_fixtures/agents \\
      --fixtures tests/eval_fixtures/mini_answers.yaml \\
      --corpus tests/eval_fixtures/mini_corpus \\
      --embeddings tests/eval_fixtures/mini_index --dry-run

  # Real run with a spending cap (reads ANTHROPIC_API_KEY)
  grounding eval-answers --agent mechanical-engineer \\
      --fixtures docs/eval/fixtures/private/mechanical-engineer-answers.yaml \\
      --corpus /path/to/corpus --embeddings /path/to/embeddings/mechanical-engineer \\
      --max-cost 40

  # Re-score an existing run with another judge; import blind grades
  grounding eval-answers --run-dir docs/eval/reports/<run-id> --score \\
      --judge-model claude-sonnet-5 --max-cost 10
  grounding eval-answers --run-dir docs/eval/reports/<run-id> --import-grades

  # Re-render the report plus a publishable copy with no source text
  grounding eval-answers --run-dir docs/eval/reports/<run-id> --report --publishable

  # Replicate check: one condition again on a subset, then compare
  grounding eval-answers ... --conditions hybrid-rerank --items me-001,me-004,me-009
  grounding eval-answers --run-dir docs/eval/reports/<run-id> \\
      --compare-run docs/eval/reports/<replicate-run-id>
        """,
    )
    p.add_argument("--fixtures", type=Path, default=None, help="Answer fixture YAML")
    p.add_argument("--agent", default=None, help="Agent name (matches agents/<name>.yaml)")
    p.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent YAML definitions (default: ./agents)",
    )
    p.add_argument("--corpus", type=Path, default=None, help="Corpus directory (with _index.json)")
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="The agent's index directory (default: embeddings/<agent>/)",
    )
    p.add_argument(
        "--conditions",
        default=",".join(CONDITION_ORDER),
        help=f"Comma-separated conditions (default: {','.join(CONDITION_ORDER)})",
    )
    p.add_argument(
        "--answer-model",
        default=DEFAULT_MODEL,
        help=f"Model that answers the questions (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help=(
            f"Model that grades correctness and citation support (default: {DEFAULT_JUDGE_MODEL}; "
            "with --run-dir --score, the run's recorded judge model)"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/eval/reports"),
        help="Parent directory for the run directory (default: docs/eval/reports, gitignored)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated tokens and cost, make no API calls, need no key",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort before the run could exceed this many US dollars (answers plus judging)",
    )
    p.add_argument("--limit", type=int, default=None, help="Run only the first N selected items")
    p.add_argument(
        "--items",
        default=None,
        help="Comma-separated item ids to run (applied before --limit)",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum search_corpus tool rounds per answer (default: 5)",
    )
    p.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Collect transcripts only; score later with --run-dir DIR --score",
    )
    p.add_argument(
        "--blind-fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help=(
            "Fraction of answers in the blind-grade export, stratified by condition and "
            "category (default: 1.0, grade every answer; the report then uses the human "
            "grades as primary correctness)"
        ),
    )
    p.add_argument(
        "--blind-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the blind-grade sample (default: 0)",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Work on an existing run directory instead of starting a new run",
    )
    p.add_argument(
        "--score",
        action="store_true",
        help="With --run-dir: (re)score the run's transcripts",
    )
    p.add_argument(
        "--import-grades",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "With --run-dir: read blind grades (default: the run's blind/ directory; "
            "or a directory or CSV path) and compute judge agreement"
        ),
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="With --run-dir: re-render report.md, report.json and chart.png",
    )
    p.add_argument(
        "--export-blind",
        action="store_true",
        help=(
            "With --run-dir: re-export the blind-grade files from the existing scores "
            "(no judge calls); filled CSVs are archived under blind/archive-<UTC>/ first"
        ),
    )
    p.add_argument(
        "--compare-run",
        type=Path,
        default=None,
        metavar="OTHER_RUN_DIR",
        help=(
            "With --run-dir: agreement between this run and a replicate run of the same "
            "condition(s) on the same items; writes replicate-<other-run-id>.json"
        ),
    )
    p.add_argument(
        "--publishable",
        action="store_true",
        help=(
            "Also write publishable/: aggregate scores only, with no answer text, "
            "citations, passages, judge reasons or local paths"
        ),
    )
    p.add_argument(
        "--include-questions",
        action="store_true",
        help="With --publishable: include the question texts (only after reviewing them)",
    )
    p.add_argument(
        "--include-source-text",
        action="store_true",
        help=(
            "With --publishable: also publish the passages, transcripts and blind CSVs. "
            "Only for a fixture that declares source_license: public_domain (a US "
            "government corpus); refused otherwise"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Log each answer")
    p.set_defaults(func=eval_answers_command)
    return p


def _err(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)


def _validate_index(embeddings_dir: Path, conditions) -> str | None:
    for name in ("_embeddings.faiss", "_chunk_map.json"):
        if not (embeddings_dir / name).exists():
            return (
                f"embeddings index not found: {embeddings_dir / name}; run "
                "`grounding embeddings` or pass --embeddings"
            )
    if any(c.hybrid for c in conditions):
        for name in ("_bm25.pkl", "_bm25_map.json"):
            if not (embeddings_dir / name).exists():
                return (
                    f"hybrid conditions need the BM25 sidecar, missing {embeddings_dir / name}; "
                    "rebuild the index with `grounding embeddings` (a silent dense-only "
                    "fallback would mislabel the hybrid conditions)"
                )
    return None


def _judge_prompt_chars() -> Dict[str, int]:
    return {
        name: len(system) + len(json.dumps(schema))
        for name, (_, system, schema) in JUDGE_PROMPTS.items()
    }


def _check_prices(models, max_cost) -> str | None:
    if max_cost is None:
        return None
    for model in sorted(set(models)):
        if price_for(model) is None:
            return f"no price on record for {model}; --max-cost cannot be enforced"
    return None


def _client_or_exit(client: Any, env: Dict[str, str] | None) -> Any:
    if client is not None:
        return client
    try:
        return make_anthropic_client(env)
    except RuntimeError as exc:
        _err(str(exc))
        return None


def eval_answers_command(
    args: argparse.Namespace,
    *,
    client: Any = None,
    run_eval_fn: Callable[..., Any] | None = None,
    tool_factory: Callable[..., Any] | None = None,
    env: Dict[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Execute ``grounding eval-answers``. Injection points are for tests."""
    if getattr(args, "verbose", False):
        logging.getLogger("grounding.eval.answers").setLevel(logging.INFO)
    if args.max_cost is not None and args.max_cost <= 0:
        _err("--max-cost must be a positive number of dollars")
        return EXIT_BAD_INPUT
    if not 0 < args.blind_fraction <= 1:
        _err("--blind-fraction must be in (0, 1]")
        return EXIT_BAD_INPUT
    if getattr(args, "include_source_text", False) and not args.publishable:
        _err("--include-source-text only applies with --publishable")
        return EXIT_BAD_INPUT
    if args.run_dir is not None:
        return _existing_run(args, client=client, env=env, sleep=sleep)
    if _run_dir_actions(args):
        _err("--score, --import-grades, --report, --export-blind and --compare-run need --run-dir")
        return EXIT_BAD_INPUT
    return _new_run(
        args, client=client, run_eval_fn=run_eval_fn, tool_factory=tool_factory,
        env=env, sleep=sleep,
    )


# ---------------------------------------------------------------------------
# New run
# ---------------------------------------------------------------------------

def _new_run(args, *, client, run_eval_fn, tool_factory, env, sleep) -> int:
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    if args.fixtures is None or args.agent is None or args.corpus is None:
        _err("--fixtures, --agent and --corpus are required")
        return EXIT_BAD_INPUT
    embeddings_dir = args.embeddings or Path("embeddings") / args.agent
    if not args.fixtures.exists():
        _err(f"fixture not found at {args.fixtures}")
        return EXIT_BAD_INPUT
    if not (args.corpus / "_index.json").exists():
        _err(f"corpus manifest not found at {args.corpus}/_index.json; pass --corpus")
        return EXIT_BAD_INPUT
    if args.max_iterations < 0:
        _err("--max-iterations must be >= 0")
        return EXIT_BAD_INPUT

    try:
        conditions = parse_conditions(args.conditions)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT

    try:
        fixture_set = load_fixtures(args.fixtures, agents_dir=args.agents_dir)
    except (UnknownAgentError, FixtureValidationError) as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT
    if fixture_set.agent != args.agent:
        _err(f"fixture targets agent '{fixture_set.agent}' but --agent is '{args.agent}'")
        return EXIT_BAD_INPUT

    item_ids = [s.strip() for s in args.items.split(",") if s.strip()] if args.items else None
    try:
        items = select_items(fixture_set, limit=args.limit, item_ids=item_ids)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT

    missing = _validate_index(embeddings_dir, conditions)
    if missing:
        _err(missing)
        return EXIT_INDEX_MISSING

    # Before the estimate: an answerable item whose gold document has no page
    # index could never be scored against its gold page.
    corpus = CorpusIndex(args.corpus, embeddings_dir, editions=fixture_set.editions,
                         revisions=fixture_set.revisions,
                         identifiers=fixture_set.identifiers)
    try:
        check_page_index(items, corpus)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT
    for warning in edition_warnings(fixture_set, corpus):
        print(f"Warning: {warning}", file=sys.stderr)

    persona = fixture_set.persona or DEFAULT_PERSONA
    avg_chars, n_sampled = average_chunk_chars(args.corpus, embeddings_dir)
    estimate = estimate_run(
        items,
        conditions,
        persona=persona,
        answer_model=args.answer_model,
        judge_model=judge_model,
        avg_chunk_chars=avg_chars,
        n_chunks_sampled=n_sampled,
        max_iterations=args.max_iterations,
        include_judging=not args.skip_scoring,
        judge_prompt_chars=_judge_prompt_chars(),
    )

    if args.dry_run:
        print("eval-answers dry run: no API calls made, no key needed")
        print(estimate.render())
        print(
            "This is a heuristic estimate. For measured usage, run a small pilot "
            "(--limit 2) and read cost_usd in the transcripts and scores."
        )
        return EXIT_OK

    price_problem = _check_prices([args.answer_model, judge_model], args.max_cost)
    if price_problem:
        _err(price_problem)
        return EXIT_BAD_INPUT
    if args.max_cost is not None and (estimate.total_usd or 0.0) > args.max_cost:
        _err(
            f"estimated cost ${estimate.total_usd:.2f} exceeds --max-cost ${args.max_cost:.2f}; "
            "lower --limit or raise --max-cost (see --dry-run for the breakdown)"
        )
        return EXIT_MAX_COST

    client = _client_or_exit(client, env)
    if client is None:
        return EXIT_NO_API_KEY

    run_id = make_run_id(args.agent)
    run_dir = args.out / run_id
    suffix = 1
    while run_dir.exists():
        # Two runs started in the same second (a replicate launched right after
        # the main run) must never share, and append to, one directory.
        suffix += 1
        run_id = f"{make_run_id(args.agent)}-{suffix}"
        run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    manifest = build_manifest(
        run_id=run_id,
        agent=args.agent,
        fixture_set=fixture_set,
        items=items,
        corpus_dir=args.corpus,
        embeddings_dir=embeddings_dir,
        conditions=conditions,
        answer_model=args.answer_model,
        judge_model=judge_model,
        max_iterations=args.max_iterations,
        max_cost=args.max_cost,
        estimate=estimate.as_dict(),
        persona=persona,
        source_license=fixture_set.source_license,
    )
    write_json(run_dir / MANIFEST_FILE, manifest)
    print(estimate.render())
    print(f"run: {run_dir}")

    budget = Budget(args.max_cost)
    try:
        summary = run_answers(
            client,
            items=items,
            conditions=conditions,
            answer_model=args.answer_model,
            corpus_dir=args.corpus,
            embeddings_dir=embeddings_dir,
            run_dir=run_dir,
            run_id=run_id,
            max_iterations=args.max_iterations,
            persona=persona,
            budget=budget,
            tool_factory=tool_factory,
            sleep=sleep,
        )
    except Exception as exc:  # pragma: no cover - last-resort safety net
        logger.exception("answer run failed")
        manifest.update(status="failed", finished_utc=utc_now())
        write_json(run_dir / MANIFEST_FILE, manifest)
        _err(f"unexpected failure: {exc}")
        return EXIT_UNEXPECTED

    manifest["answers"] = {
        "written": summary.written,
        "errors": summary.errors,
        "cost_usd": round(summary.answer_cost_usd, 6),
        "aborted": summary.aborted,
        "not_run": [list(p) for p in summary.not_run],
    }
    print(
        f"answers written={summary.written} errors={summary.errors} "
        f"spent=${summary.answer_cost_usd:.4f}"
    )

    try:
        retrieval = compute_retrieval_recall(
            fixture_set,
            items,
            conditions,
            corpus_dir=args.corpus,
            embeddings_dir=embeddings_dir,
            run_eval_fn=run_eval_fn,
        )
    except Exception as exc:
        logger.exception("retrieval recall failed")
        retrieval = {"error": f"{type(exc).__name__}: {exc}"}
    write_json(run_dir / RETRIEVAL_FILE, retrieval)

    aborted = summary.aborted
    scored = False
    if not aborted and not args.skip_scoring:
        scoring = _score(
            run_dir, manifest, fixture_set, args.corpus, embeddings_dir,
            client=client, judge_model=judge_model, budget=budget,
            blind_fraction=args.blind_fraction, blind_seed=args.blind_seed, sleep=sleep,
        )
        aborted = scoring.get("aborted")
        scored = True

    manifest["status"] = "aborted_max_cost" if aborted else "complete"
    manifest["finished_utc"] = utc_now()
    write_json(run_dir / MANIFEST_FILE, manifest)
    if scored and _write_report(run_dir, args) != EXIT_OK:
        return EXIT_BAD_INPUT
    print(f"total spent=${budget.spent:.4f}")
    if aborted:
        print(f"stopped early: {aborted}", file=sys.stderr)
        return EXIT_MAX_COST
    return EXIT_OK


def _score(
    run_dir: Path,
    manifest: Dict[str, Any],
    fixture_set,
    corpus_dir: Path,
    embeddings_dir: Path,
    *,
    client: Any,
    judge_model: str,
    budget: Budget,
    blind_fraction: float,
    blind_seed: int,
    sleep: Callable[[float], None],
) -> Dict[str, Any]:
    """Score a run, export the blind sample, and record both in the manifest."""
    items = {it.id: it for it in fixture_set.items}
    corpus = CorpusIndex(corpus_dir, embeddings_dir, editions=fixture_set.editions,
                         revisions=fixture_set.revisions,
                         identifiers=fixture_set.identifiers)
    summary = score_run(
        run_dir,
        items=items,
        page_offsets=fixture_set.page_offsets,
        corpus=corpus,
        client=client,
        judge_model=judge_model,
        budget=budget,
        run_id=manifest.get("run_id"),
        sleep=sleep,
    )
    scores = read_jsonl(run_dir / SCORES_FILE)
    blind = export_blind(run_dir, scores, items, fraction=blind_fraction, seed=blind_seed)
    manifest["scoring"] = {
        "judge_model": judge_model,
        "generation": judge_generation(judge_model),
        "prompts": judge_prompt_fingerprints(),
        "fixture_sha256": sha256_file(Path(fixture_set.source_path)),
        "scored_utc": utc_now(),
        "scored": summary.scored,
        "skipped": summary.skipped,
        "judge_calls": summary.judge_calls,
        "judge_cost_usd": round(summary.judge_cost_usd, 6),
        "aborted": summary.aborted,
        "blind": blind,
    }
    print(
        f"scored={summary.scored} judge_calls={summary.judge_calls} "
        f"judge_spent=${summary.judge_cost_usd:.4f} blind_export={blind['n_answers']} "
        f"of {blind['n_eligible_answers']} answers"
    )
    return manifest["scoring"]


# ---------------------------------------------------------------------------
# Existing run
# ---------------------------------------------------------------------------

def _existing_run(args, *, client, env, sleep) -> int:
    run_dir: Path = args.run_dir
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        _err(f"no {MANIFEST_FILE} in {run_dir}; is this an eval-answers run directory?")
        return EXIT_BAD_INPUT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _run_dir_actions(args):
        _err("with --run-dir pass --score, --import-grades, --report, --export-blind "
             "or --compare-run")
        return EXIT_BAD_INPUT

    if args.score:
        code = _rescore(args, run_dir, manifest, client=client, env=env, sleep=sleep)
        if code != EXIT_OK:
            return code

    if args.export_blind:
        code = _reexport_blind(args, run_dir, manifest)
        if code != EXIT_OK:
            return code

    if args.compare_run is not None:
        try:
            replicate = compare_runs(run_dir, args.compare_run)
        except (OSError, ValueError, KeyError) as exc:
            _err(f"could not compare runs: {exc}")
            return EXIT_BAD_INPUT
        print(render_replicate_summary(replicate))

    if args.import_grades is not None:
        source = Path(args.import_grades) if args.import_grades else None
        try:
            agreement = import_blind(run_dir, source)
        except (OSError, ValueError, KeyError) as exc:
            _err(f"could not import blind grades: {exc}")
            return EXIT_BAD_INPUT
        corr = agreement["correctness"]
        gate = agreement["publish_gate"]
        rate = "n/a" if corr["agreement"] is None else f"{corr['agreement']:.0%}"
        judge_rate = "n/a" if gate["agreement"] is None else f"{gate['agreement']:.0%}"
        print(
            f"blind grades: {agreement['n_human_graded_answers']} answers graded; correctness "
            f"agreement all rows n={corr['n']} {rate}, rubric-judge rows n={gate['n']} "
            f"{judge_rate}; publish gate={gate['result']} "
            f"support gate={agreement['support_gate']['result']}"
        )

    if not (run_dir / SCORES_FILE).exists():
        _err(f"no {SCORES_FILE} in {run_dir}; score the run before rendering a report")
        return EXIT_BAD_INPUT
    return _write_report(run_dir, args)


def _run_dir_actions(args) -> bool:
    return bool(args.score or args.import_grades is not None or args.report
                or args.export_blind or args.compare_run is not None)


def _reexport_blind(args, run_dir: Path, manifest: Dict[str, Any]) -> int:
    """Re-export the blind-grade files from scores.jsonl; no judge calls, no spend."""
    if not (run_dir / SCORES_FILE).exists():
        _err(f"no {SCORES_FILE} in {run_dir}; score the run before exporting blind grades")
        return EXIT_BAD_INPUT
    fixture_path = args.fixtures or Path(manifest["fixture"]["path"])
    try:
        fixture_set = load_fixtures(fixture_path, agents_dir=args.agents_dir)
    except (UnknownAgentError, FixtureValidationError) as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT
    items = {it.id: it for it in fixture_set.items}
    scores = read_jsonl(run_dir / SCORES_FILE)
    blind = export_blind(run_dir, scores, items, fraction=args.blind_fraction, seed=args.blind_seed)
    manifest.setdefault("scoring", {})["blind"] = blind
    write_json(run_dir / MANIFEST_FILE, manifest)
    print(
        f"blind export: {blind['n_answers']} of {blind['n_eligible_answers']} answers, "
        f"{blind['n_citations']} judged citations, {blind['n_audit_citations']} for the "
        f"resolution audit"
        + (f"; previous grades archived to {blind['archived_previous']}"
           if blind["archived_previous"] else "")
    )
    return EXIT_OK


def _write_report(run_dir: Path, args) -> int:
    """Render the report; returns an exit code (source text is refused for a
    fixture that has not declared a public-domain corpus)."""
    try:
        written = write_report(
            run_dir,
            publishable=args.publishable,
            include_questions=args.include_questions,
            include_source_text=getattr(args, "include_source_text", False),
        )
    except ValueError as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT
    for key in ("markdown", "publishable_markdown"):
        if key in written:
            print(f"wrote: {written[key]}")
    if any(k.startswith("publishable_source") for k in written):
        print("published source text: transcripts, scores and blind grades "
              "(source_license: public_domain)")
    return EXIT_OK


def _rescore(args, run_dir: Path, manifest: Dict[str, Any], *, client, env, sleep) -> int:
    fixture_path = args.fixtures or Path(manifest["fixture"]["path"])
    corpus_dir = args.corpus or Path(manifest["corpus_dir"])
    embeddings_dir = args.embeddings or Path(manifest["embeddings_dir"])
    judge_model = args.judge_model or (manifest.get("scoring") or {}).get("judge_model") or manifest["judge_model"]
    try:
        fixture_set = load_fixtures(fixture_path, agents_dir=args.agents_dir)
    except (UnknownAgentError, FixtureValidationError) as exc:
        _err(str(exc))
        return EXIT_BAD_INPUT
    if sha256_file(fixture_path) != manifest["fixture"]["sha256"]:
        print(
            "Warning: the fixture changed since the answers were collected; "
            "scoring against the current file (its hash is recorded).",
            file=sys.stderr,
        )
    price_problem = _check_prices([judge_model], args.max_cost)
    if price_problem:
        _err(price_problem)
        return EXIT_BAD_INPUT
    client = _client_or_exit(client, env)
    if client is None:
        return EXIT_NO_API_KEY
    budget = Budget(args.max_cost)
    scoring = _score(
        run_dir, manifest, fixture_set, corpus_dir, embeddings_dir,
        client=client, judge_model=judge_model, budget=budget,
        blind_fraction=args.blind_fraction, blind_seed=args.blind_seed, sleep=sleep,
    )
    write_json(run_dir / MANIFEST_FILE, manifest)
    if scoring.get("aborted"):
        print(f"stopped early: {scoring['aborted']}", file=sys.stderr)
        return EXIT_MAX_COST
    return EXIT_OK
