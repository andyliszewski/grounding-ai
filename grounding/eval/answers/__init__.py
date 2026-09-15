"""Grounded-answer benchmark (Epic 25).

Scores whether Claude's final answers, and the citations inside them, hold up
with and without corpus grounding. See
``docs/epics/epic-25-grounded-answer-benchmark.md`` and the "Answer benchmark
(Epic 25)" section of ``docs/eval/README.md``.

Nothing in this package imports the ``anthropic`` SDK or the MCP server at
import time, so dry runs and tests work without either installed.
"""
from grounding.eval.answers.conditions import CONDITION_ORDER, CONDITIONS, ConditionSpec, parse_conditions

__all__ = ["CONDITION_ORDER", "CONDITIONS", "ConditionSpec", "parse_conditions"]
