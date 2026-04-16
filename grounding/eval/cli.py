"""``grounding eval`` subcommand (Story 16.3).

Wires the runner, report serialization, and baseline diff into a CLI
handler. Kept in its own module so tests can drive the handler without
touching the top-level argparse setup.

Exit codes (see story 16.3 / Dev Notes):

============================================  ====
scenario                                      code
============================================  ====
successful run, no ``--baseline``             0
run with baseline, all metrics within delta   0
baseline drop > ``--fail-under``              1
fixture file not found / agent unknown        2
embeddings index missing                      3
unexpected exception                          4
============================================  ====
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from grounding.config import (
    load_retrieval_config,
    resolve_hybrid_config,
    resolve_rerank_config,
)
from grounding.eval.baseline import AggregateDiff, BaselineError, diff, load_baseline
from grounding.eval.fixtures import (
    FixtureValidationError,
    UnknownAgentError,
    load_fixtures,
)
from grounding.eval.report import _aggregate_rows, write_artifacts
from grounding.eval.runner import EvalRun, run_eval

logger = logging.getLogger("grounding.eval.cli")

EXIT_OK = 0
EXIT_BASELINE_REGRESSION = 1
EXIT_FIXTURE_OR_AGENT_NOT_FOUND = 2
EXIT_EMBEDDINGS_MISSING = 3
EXIT_UNEXPECTED = 4

# Alias: argparse-style "bad arguments" exit code. Shares the value with
# EXIT_FIXTURE_OR_AGENT_NOT_FOUND (both are user-input errors → 2); exposed
# as a separate name so invalid-rerank-value exits read clearly at the
# call site instead of masquerading as a fixture-not-found error.
EXIT_BAD_ARGS = 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _create_eval_parser(subparsers) -> argparse.ArgumentParser:
    """Register the ``eval`` subcommand on the given subparsers."""
    p = subparsers.add_parser(
        "eval",
        help="Run the retrieval evaluation harness against an agent",
        description=(
            "Run fixture queries against an agent's FAISS index, write "
            "Markdown + JSON reports, and (optionally) compare against a "
            "committed baseline for regression gating."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run against the default fixture/embeddings paths for an agent
  grounding eval --agent data-scientist --corpus ./corpus

  # Custom fixture and explicit embeddings dir
  grounding eval --agent ceo --corpus ./corpus \\
      --fixtures ./docs/eval/fixtures/ceo.yaml \\
      --embeddings ./embeddings/ceo

  # Compare against a committed baseline; fail if any metric drops > 0.02
  grounding eval --agent data-scientist --corpus ./corpus \\
      --baseline docs/eval/baselines/data-scientist.json --fail-under 0.02
        """,
    )
    p.add_argument("--agent", required=True, help="Agent name (matches agents/<name>.yaml)")
    p.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent YAML definitions (default: ./agents)",
    )
    p.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Fixture YAML path (default: docs/eval/fixtures/<agent>.yaml)",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Corpus directory (containing _index.json)",
    )
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Embeddings directory (default: embeddings/<agent>/)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval cutoff (default: 10)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs/eval/reports"),
        help="Directory to write <run-id>.md and <run-id>.json (default: docs/eval/reports)",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to compare against",
    )
    p.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Fail (exit 1) if any aggregate metric drops more than this absolute delta vs baseline (default: 0.0)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to config.yaml for retrieval.rerank defaults "
            "(CLI flags override config; config overrides built-in defaults)."
        ),
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking of FAISS results.",
    )
    p.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="Cross-encoder model name (default: BAAI/bge-reranker-base).",
    )
    p.add_argument(
        "--rerank-pool-size",
        type=int,
        default=None,
        help="FAISS candidate count fed to the reranker (default: 50).",
    )
    p.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=(
            "Post-rerank truncation. When omitted, --top-k applies to the "
            "final output. Lets you ask the reranker to score a large pool "
            "(--rerank-pool-size) and return only the best N."
        ),
    )
    # Hybrid retrieval (Story 19.3)
    p.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help="Enable BM25 + dense fusion via RRF before (optional) rerank.",
    )
    p.add_argument(
        "--hybrid-pool-size",
        type=int,
        default=None,
        help=(
            "Candidates fetched from each channel (FAISS + BM25) before fusion "
            "(default: 50). Higher = better recall, slightly slower."
        ),
    )
    p.add_argument(
        "--hybrid-k-rrf",
        type=int,
        default=None,
        help=(
            "RRF damping constant (default: 60; literature standard). "
            "Raising it flattens rank contributions."
        ),
    )
    p.set_defaults(func=eval_command)
    return p


# ---------------------------------------------------------------------------
# Defaults resolution
# ---------------------------------------------------------------------------

def _resolve_fixture_path(args: argparse.Namespace) -> Path:
    if args.fixtures is not None:
        return args.fixtures
    return Path("docs/eval/fixtures") / f"{args.agent}.yaml"


def _resolve_embeddings_dir(args: argparse.Namespace) -> Path:
    if args.embeddings is not None:
        return args.embeddings
    return Path("embeddings") / args.agent


def _resolve_rerank_config_from_args(args: argparse.Namespace):
    """Build a :class:`RerankConfig` from CLI flags + optional config.yaml.

    Returns ``None`` when no CLI flag was passed and no config file was
    provided (so callers can keep the pre-18.3 disabled-default path
    bit-for-bit identical). Otherwise returns a validated ``RerankConfig``
    whose ``enabled`` field reflects the merged resolution order.
    """
    retrieval_cfg = load_retrieval_config(args.config) if args.config else {}
    no_flags_passed = (
        not args.rerank
        and args.rerank_model is None
        and args.rerank_pool_size is None
        and args.rerank_top_k is None
    )
    # Opt-in-preserved: no flags and no config file => legacy path.
    if no_flags_passed and not retrieval_cfg:
        return None
    return resolve_rerank_config(
        retrieval_config=retrieval_cfg,
        cli_enabled=args.rerank,
        cli_model=args.rerank_model,
        cli_pool_size=args.rerank_pool_size,
        cli_batch_size=None,
    )


def _resolve_hybrid_config_from_args(args: argparse.Namespace):
    """Build a :class:`HybridConfig` from CLI flags + optional config.yaml.

    Returns ``None`` when no hybrid CLI flag was passed and no config file
    was provided, preserving the pre-19.3 zero-change path bit-for-bit.
    Otherwise returns a validated ``HybridConfig``.
    """
    retrieval_cfg = load_retrieval_config(args.config) if args.config else {}
    no_flags_passed = (
        not args.hybrid
        and args.hybrid_pool_size is None
        and args.hybrid_k_rrf is None
    )
    if no_flags_passed and not retrieval_cfg:
        return None
    return resolve_hybrid_config(
        retrieval_config=retrieval_cfg,
        cli_enabled=args.hybrid,
        cli_pool_size=args.hybrid_pool_size,
        cli_k_rrf=args.hybrid_k_rrf,
    )


def _make_run_id(agent: str, *, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return f"{agent}-{moment.strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _arrow(delta: float | None) -> str:
    if delta is None:
        return ""
    if abs(delta) < 1e-6:
        return " ="
    return f" \u2191{delta:.3f}" if delta > 0 else f" \u2193{abs(delta):.3f}"


def render_stdout_summary(
    eval_run: EvalRun,
    *,
    diff_result: AggregateDiff | None,
    fail_under: float | None,
    baseline_present: bool,
    baseline_aggregate: dict | None = None,
) -> str:
    """Build the compact one-screen stdout summary."""
    n = len(eval_run.items)
    lines: list[str] = []
    lines.append(f"eval: {eval_run.agent}  items={n}  skipped={len(eval_run.skipped)}")

    rows = _aggregate_rows(eval_run.aggregate, baseline_aggregate)
    # Layout: 3 metrics per row.
    chunks = [rows[i : i + 3] for i in range(0, len(rows), 3)]
    for chunk in chunks:
        parts = []
        for row in chunk:
            arrow = _arrow(row.delta) if baseline_present else ""
            parts.append(f"{row.name} {row.value:.3f}{arrow}")
        lines.append("  " + "   ".join(parts))

    if diff_result is not None and fail_under is not None:
        verdict = "PASS" if diff_result.passes(fail_under) else "FAIL"
        suffix = ""
        if diff_result.worst_drop_metric:
            suffix = f"  ({diff_result.worst_drop_metric})"
        lines.append(
            f"  worst_drop {diff_result.worst_drop:.3f}  fail_under {fail_under:.3f}  -> {verdict}{suffix}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def eval_command(args: argparse.Namespace) -> int:  # noqa: C901 - linear flow
    """Execute the eval subcommand and return an exit code."""
    fixture_path = _resolve_fixture_path(args)
    embeddings_dir = _resolve_embeddings_dir(args)
    corpus_dir = args.corpus
    agents_dir = args.agents_dir

    # Validate paths.
    if not fixture_path.exists():
        print(
            f"Error: fixture not found at {fixture_path}; pass --fixtures",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND
    if not agents_dir.exists():
        print(
            f"Error: agents directory not found at {agents_dir}; pass --agents-dir",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND
    if not corpus_dir.exists() or not (corpus_dir / "_index.json").exists():
        print(
            f"Error: corpus manifest not found at {corpus_dir}/_index.json; pass --corpus",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND
    if not embeddings_dir.exists():
        print(
            f"Error: embeddings index not found at {embeddings_dir}; "
            f"run `grounding embeddings --agent {args.agent}` or pass --embeddings",
            file=sys.stderr,
        )
        return EXIT_EMBEDDINGS_MISSING

    # Load fixtures.
    try:
        fixture_set = load_fixtures(fixture_path, agents_dir=agents_dir)
    except UnknownAgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND
    except FixtureValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND

    if fixture_set.agent != args.agent:
        print(
            f"Error: fixture targets agent '{fixture_set.agent}' "
            f"but --agent is '{args.agent}'",
            file=sys.stderr,
        )
        return EXIT_FIXTURE_OR_AGENT_NOT_FOUND

    # Optional baseline.
    baseline = None
    if args.baseline is not None:
        try:
            baseline = load_baseline(args.baseline)
        except BaselineError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_FIXTURE_OR_AGENT_NOT_FOUND

    if args.fail_under and args.baseline is None:
        print(
            "Warning: --fail-under has no effect without --baseline; ignoring.",
            file=sys.stderr,
        )

    # Resolve hybrid config first: CLI flags > config.yaml > HybridConfig defaults.
    try:
        hybrid_config = _resolve_hybrid_config_from_args(args)
    except ValueError as exc:
        print(f"Error: invalid hybrid configuration: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS

    # Resolve rerank config: CLI flags > config.yaml > RerankConfig defaults.
    try:
        rerank_config = _resolve_rerank_config_from_args(args)
    except ValueError as exc:
        print(f"Error: invalid rerank configuration: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS

    # Effective final top-k: --rerank-top-k overrides --top-k when rerank is on.
    effective_top_k = args.top_k
    if rerank_config is not None and rerank_config.enabled and args.rerank_top_k:
        effective_top_k = args.rerank_top_k

    # Run.
    try:
        eval_run = run_eval(
            fixture_set,
            args.agent,
            corpus_dir=corpus_dir,
            embeddings_dir=embeddings_dir,
            top_k=effective_top_k,
            rerank_config=rerank_config,
            hybrid_config=hybrid_config,
        )
    except FileNotFoundError as exc:
        print(f"Error: embeddings load failed: {exc}", file=sys.stderr)
        return EXIT_EMBEDDINGS_MISSING
    except Exception as exc:  # pragma: no cover - last-resort safety net
        logger.exception("eval run failed")
        print(f"Error: unexpected failure: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED

    # Diff (if baseline).
    diff_result: AggregateDiff | None = None
    if baseline is not None:
        from grounding.eval.report import eval_run_to_dict

        run_id_for_diff = _make_run_id(args.agent)
        cur_dict = eval_run_to_dict(
            eval_run,
            run_id=run_id_for_diff,
            corpus_dir=corpus_dir,
            embeddings_dir=embeddings_dir,
        )
        diff_result = diff(cur_dict["aggregate"], baseline.aggregate)

    # Write artifacts.
    run_id = _make_run_id(args.agent)
    baseline_payload = None
    if baseline is not None:
        baseline_payload = {
            "agent": baseline.agent,
            "captured_utc": baseline.captured_utc,
            "aggregate": baseline.aggregate,
        }
    json_path, md_path = write_artifacts(
        eval_run,
        run_id=run_id,
        out_dir=args.out,
        corpus_dir=corpus_dir,
        embeddings_dir=embeddings_dir,
        baseline=baseline_payload,
    )

    # Stdout summary.
    sys.stdout.write(
        render_stdout_summary(
            eval_run,
            diff_result=diff_result,
            fail_under=args.fail_under if baseline is not None else None,
            baseline_present=baseline is not None,
            baseline_aggregate=baseline.aggregate if baseline is not None else None,
        )
    )
    sys.stdout.write(f"wrote: {md_path}\n")
    sys.stdout.write(f"wrote: {json_path}\n")

    if diff_result is not None and args.baseline is not None:
        if not diff_result.passes(args.fail_under):
            return EXIT_BASELINE_REGRESSION
    return EXIT_OK


# ---------------------------------------------------------------------------
# Standalone main (test / script invocation convenience)
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grounding-eval")
    sub = parser.add_subparsers(dest="command")
    _create_eval_parser(sub)
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return EXIT_OK
    return args.func(args)
