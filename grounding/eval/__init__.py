"""Retrieval evaluation harness (Epic 16).

Exposes the fixture schema, loader, runner, metrics, and reporting.
"""
from grounding.eval.baseline import (
    AggregateDiff,
    Baseline,
    BaselineError,
    diff,
    load_baseline,
)
from grounding.eval.fixtures import (
    Expected,
    FixtureItem,
    FixtureSet,
    FixtureValidationError,
    SCHEMA_VERSION,
    UnknownAgentError,
    load_fixtures,
)
from grounding.eval.metrics import CitationCase, citation_accuracy, mrr, ndcg_at_k, recall_at_k
from grounding.eval.report import (
    eval_run_to_dict,
    to_json,
    to_markdown,
    write_artifacts,
)
from grounding.eval.runner import (
    EvalAggregate,
    EvalItemResult,
    EvalRun,
    RetrievedChunk,
    TagMetrics,
    compute_aggregate,
    run_eval,
)

__all__ = [
    "Expected",
    "FixtureItem",
    "FixtureSet",
    "FixtureValidationError",
    "SCHEMA_VERSION",
    "UnknownAgentError",
    "load_fixtures",
    # metrics
    "CitationCase",
    "citation_accuracy",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    # runner
    "EvalAggregate",
    "EvalItemResult",
    "EvalRun",
    "RetrievedChunk",
    "TagMetrics",
    "compute_aggregate",
    "run_eval",
    # report
    "eval_run_to_dict",
    "to_json",
    "to_markdown",
    "write_artifacts",
    # baseline
    "AggregateDiff",
    "Baseline",
    "BaselineError",
    "diff",
    "load_baseline",
]
