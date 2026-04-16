# Epic 16: Retrieval Evaluation Harness

**Epic ID:** E16
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 0/4
**Dependencies:** Epic 6 (Vector Embeddings), Epic 10 (Centralized Corpus)
**Target Completion:** TBD

---

## Overview

Establish a measurable, reproducible retrieval evaluation harness for grounding-ai. Without quantified retrieval quality, every downstream improvement (page citations, reranking, hybrid retrieval, larger embedding models) ships blind. This epic delivers the baseline that gates Tier 1 roadmap items #2–#4.

**Problem Statement:**
- Retrieval quality is currently judged anecdotally.
- There is no regression signal when embeddings, chunking, or parsers change.
- Contributors cannot demonstrate the value of a PR that touches retrieval.

**Solution:**
- Per-agent fixture sets of `query → expected source(s)` (chunk-level and doc-level).
- A CLI runner that scores recall@k, MRR, and nDCG@k against an agent's FAISS index.
- A baseline report committed to the repo and refreshed on demand.
- A GitHub Actions job that runs the harness on PRs touching retrieval surfaces, failing if scores drop beyond a configurable tolerance.

---

## Goals

1. Make retrieval quality observable and comparable across changes.
2. Provide a low-friction format for contributors to add eval items.
3. Run fast enough to live in CI (< 2 minutes for the default suite).
4. Produce human-readable reports and machine-readable JSON for tracking over time.
5. Establish a published baseline so Tier 1 roadmap improvements can be measured against it.

---

## Non-Goals

- Generation quality evaluation (LLM answer grading) — out of scope; this epic measures retrieval only.
- Cross-agent leaderboards or public benchmarks.
- Auto-generation of eval items from documents (manual curation only in v1).

---

## Architecture

```
docs/eval/
├── fixtures/
│   ├── ceo.yaml                 # Per-agent eval set
│   ├── data-scientist.yaml
│   └── ...
├── baselines/
│   └── 2026-04-XX.json          # Committed baseline scores
└── reports/                     # Generated, gitignored
    └── <run-id>.json

grounding/eval/
├── __init__.py
├── fixtures.py                  # Load + validate fixture YAML
├── runner.py                    # Execute queries against FAISS
├── metrics.py                   # recall@k, MRR, nDCG
└── report.py                    # Markdown + JSON output

CLI: grounding eval --agent <name> [--baseline <path>] [--fail-under <delta>]
```

### Fixture Format

```yaml
agent: ceo
version: 1
items:
  - id: ceo-001
    query: "What did the board decide about the Q3 fundraise?"
    expected:
      doc_ids: ["7a9b2c1f"]              # any-of match
      chunk_ids: ["7a9b2c1f/ch_0023"]    # optional, stricter
    tags: ["finance", "board"]
    notes: "Pulled from 2026-Q3 board minutes."
```

### Metrics (initial set)

- **recall@k** for k in {1, 3, 5, 10} — did the expected source land in the top-k?
- **MRR** — mean reciprocal rank of the first correct hit.
- **nDCG@10** — when multiple expected sources are listed.
- **per-tag breakdown** — surface weak query categories.

---

## Stories Breakdown

### Story 16.1: Fixture Schema and Loader
- Define YAML fixture schema (above) and validate on load.
- Implement `grounding/eval/fixtures.py` with strict parsing and helpful error messages.
- Add 5–10 seed fixtures for one pilot agent (recommend `data-scientist` — large, diverse corpus).

**AC:**
- Fixture YAML validates against documented schema; bad files produce actionable errors.
- Loader returns typed objects (dataclasses) usable by the runner.
- At least one agent has ≥ 5 committed fixture items.
- Unit tests cover schema validation, missing-field handling, and unknown-agent handling.

**Status:** Draft

### Story 16.2: Runner and Metrics
- Implement `runner.py` that executes each fixture query against the agent's FAISS index using existing search code.
- Implement `metrics.py`: recall@k, MRR, nDCG@k, per-tag aggregation.
- Return both per-item results (for debugging) and aggregate scores.

**AC:**
- Runner reuses `grounding`'s existing embedding + FAISS search path (no duplicated retrieval logic).
- Metrics module is pure and unit-tested with synthetic ranked lists.
- Per-item output records: query, expected, retrieved top-10, hit position, score.
- Aggregate output records: per-metric scalar + per-tag breakdown.

**Status:** Draft

### Story 16.3: CLI and Reporting
- Add `grounding eval` subcommand: `--agent`, `--fixtures`, `--out`, `--baseline`, `--fail-under`.
- Generate a Markdown report and a JSON artifact per run.
- `--baseline` compares current run against a committed baseline; `--fail-under N.NN` sets exit-code policy on metric regression.

**AC:**
- `grounding eval --agent <name>` runs default fixtures and prints a Markdown summary to stdout.
- `--out <dir>` writes `<run-id>.md` and `<run-id>.json`.
- `--baseline <path> --fail-under 0.02` exits non-zero if recall@5 drops by more than 2 absolute points vs baseline.
- Help text and `--help` output are clear.

**Status:** Draft

### Story 16.4: CI Integration and Baseline Publication
- Add a GitHub Actions workflow that runs the eval harness on PRs touching retrieval surfaces (`grounding/embedder.py`, `grounding/vector_store.py`, `grounding/chunker.py`, `grounding/eval/**`).
- Cache embeddings to keep CI runtime under 2 minutes.
- Commit an initial baseline JSON to `docs/eval/baselines/`.
- Document the workflow in `docs/eval/README.md`.

**AC:**
- Workflow runs on relevant PRs and posts a comment with score deltas vs baseline.
- Initial baseline committed and dated.
- README explains: how to add fixtures, how to run locally, how to refresh the baseline, how the CI gate works.
- A deliberate regression PR (e.g., chunk size 100) is shown to fail the gate as a sanity check.

**Status:** Draft

---

## Dependencies

### Epic Dependencies
- **Epic 6** — FAISS embeddings infrastructure (search path reuse).
- **Epic 10** — Agent YAML + collection filtering (per-agent fixture targeting).

### External Dependencies
- None. Uses existing `sentence-transformers` and `faiss-cpu`.

### Code Dependencies
- `grounding/embedder.py`, `grounding/vector_store.py` — search reuse.
- `grounding/agent_filter.py` — agent resolution.

---

## Implementation Order

```
Story 16.1 (Fixture Schema and Loader)
    └── Foundation; unblocks 16.2

Story 16.2 (Runner and Metrics)
    └── Core measurement; unblocks 16.3

Story 16.3 (CLI and Reporting)
    └── User surface; unblocks 16.4

Story 16.4 (CI Integration and Baseline)
    └── Closes the loop; produces the published baseline
```

Stories are sequential. 16.1 and 16.2 could overlap if metrics work uses synthetic data, but recommend strict order to keep scope tight.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fixture maintenance burden | Medium | Keep schema minimal; document the "5 items per agent" floor; treat fixtures as living docs. |
| CI runtime exceeds budget | Medium | Cache embeddings; pin small agent for CI; allow opt-in full-suite run via label. |
| Metric overfitting (gaming the score) | Medium | Pair recall@k with MRR + tag breakdown; require new fixtures alongside large retrieval changes. |
| Baseline drift on benign changes | Low | `--fail-under` tolerance; require human review for baseline refresh PRs. |
| Stale fixtures after corpus changes | Medium | `expected.doc_ids` is content-hash-stable; runner warns when expected doc_id no longer in manifest. |

---

## Testing Strategy

### Unit Tests
- Fixture schema validation (good + bad inputs).
- Metrics module against synthetic ranked lists with known answers.
- Runner with a mocked FAISS search returning controlled results.

### Integration Tests
- End-to-end run against a tiny test corpus committed to `tests/eval_fixtures/`.
- CLI smoke test for `grounding eval --agent <name> --out <tmp>`.

### Manual Validation
- Run baseline against `data-scientist` and `ceo` agents; sanity-check scores.
- Introduce a deliberate regression (smaller chunk size) and confirm the CI gate fires.

---

## Acceptance Criteria (Epic Level)

1. `grounding eval` subcommand exists, documented, and unit-tested.
2. Fixture format documented in `docs/eval/README.md` with a worked example.
3. At least one agent has a committed fixture set of ≥ 10 items.
4. Initial baseline JSON committed to `docs/eval/baselines/`.
5. CI workflow gates retrieval-touching PRs against the baseline with a configurable tolerance.
6. A deliberate regression PR demonstrably fails the gate (recorded in epic notes or PR link).

---

## Definition of Done

- All four stories closed with their AC met.
- Baseline committed and reproducible from a clean checkout.
- CI workflow green on `main`, fires on a regression PR.
- `docs/ROADMAP.md` Tier 1 #1 marked shipped; Tier 1 #2 unblocked.
- `docs/eval/README.md` exists and is linked from the top-level README.

---

## Future Enhancements (Out of Scope)

- LLM-judged answer quality (generation eval).
- Auto-generation of eval items from corpus content.
- Public benchmark comparisons (BEIR, MTEB subsets).
- Per-PR retrieval diff comments showing which queries changed rank.
- Eval-driven hyperparameter search (chunk size, top-k, reranker thresholds).

---

## References

- `docs/ROADMAP.md` — Tier 1 #1 (this epic) gates #2–#4.
- `docs/epics/epic-6-vector-embeddings-v02.md` — search path being measured.
- `docs/epics/epic-10-centralized-corpus-v02.md` — agent filtering used to scope eval per agent.
- `CLAUDE.md` — "Embeddings Generation" section for current retrieval contract.
