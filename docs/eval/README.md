# Retrieval Evaluation Harness

This directory holds fixture files and documentation for the grounding-ai retrieval
evaluation harness (Epic 16).

A fixture is a small, hand-curated set of queries with known-good answers. The harness
runs the queries through an agent's FAISS index and scores the top-k retrievals against
the expected documents or chunks. Fixtures let us catch regressions in retrieval quality
as embeddings, chunking, or ranking logic changes.

This file covers the fixture schema (Story 16.1), the runner and metrics (Story 16.2),
the `grounding eval` CLI (Story 16.3), and the CI gate and baseline lifecycle (Story 16.4).

## Three-repo layering

`grounding-ai` coordinates with two adjacent repos to keep the public CI gate useful
without leaking private content.

| Repo | Visibility | What lives there | Eval implication |
|------|------------|------------------|------------------|
| `grounding-ai` | Public | Pipeline code, schema, mini test corpus, schema-example fixture, mini baseline, CI workflow | CI gate runs here against the mini corpus only |
| `grounding-ai-private` | Private | Real corpus content, baselines for real agents | Real-agent fixtures and baselines live here; eval is run locally or in private CI |
| `my-agents` | Private | The maintainer's actual agent YAMLs (e.g., `data-scientist.yaml`) | Real-agent fixtures reference these; never copied into public repo |

Public-repo contributors ship retrieval *code*. Real-agent eval is the maintainer's
private concern. The public CI gate guards against retrieval-code regressions;
private eval runs guard against corpus-quality regressions.

## Running the harness locally

Against a real agent you maintain:

```bash
grounding eval \
    --agent data-scientist \
    --corpus /path/to/corpus \
    --baseline docs/eval/baselines/data-scientist.json \
    --fail-under 0.02
```

Against the in-repo mini corpus (same invocation CI uses):

```bash
grounding embeddings \
    --corpus tests/eval_fixtures/mini_corpus \
    --out tests/eval_fixtures/mini_index

grounding eval \
    --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --baseline docs/eval/baselines/mini-corpus.json \
    --fail-under 0.05 \
    --out eval-output/
```

The command writes `<run-id>.md` and `<run-id>.json` under `--out`, prints a one-screen
stdout summary, and exits non-zero if any metric drops more than `--fail-under`.

### Reranking (opt-in, Story 18.3)

`grounding eval` accepts the same reranker flags as `scripts/local_rag.py`:
`--rerank`, `--rerank-model`, `--rerank-pool-size`, `--rerank-top-k`. With
reranking off (the default), behavior is bit-for-bit identical to the
pre-epic flow — the existing baselines stay valid.

Compare rerank-on vs rerank-off locally:

```bash
# Baseline (rerank off)
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --out eval-output/off/

# With bge-reranker-base
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --rerank --rerank-pool-size 50 \
    --out eval-output/on/
```

See the "Reranking comparison" section below for the measured off-vs-on
numbers and the flip decision.

### Hybrid retrieval (opt-in, Story 19.3)

`grounding eval` accepts `--hybrid`, `--hybrid-pool-size`, `--hybrid-k-rrf`
with the same names on `scripts/local_rag.py`. With the flags absent (the
default), behavior is bit-for-bit identical to pre-19.3 — existing
baselines stay valid.

Run the `{hybrid off/on} × {rerank off/on}` matrix against the mini corpus:

```bash
# Dense only (baseline)
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --out eval-output/dense-only/

# Hybrid on, rerank off
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --hybrid --hybrid-pool-size 50 \
    --out eval-output/hybrid-only/

# Rerank on, hybrid off
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --rerank --rerank-pool-size 50 \
    --out eval-output/rerank-only/

# Hybrid + rerank (reranker scores the fused pool)
grounding eval --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_fixtures.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --hybrid --hybrid-pool-size 50 \
    --rerank --rerank-pool-size 50 \
    --out eval-output/hybrid-plus-rerank/
```

The measured comparison matrix (mini vs private corpora) and the flip
decision land in Story 19.4. Story 19.3 ships the plumbing so those
measurements go through the same code paths users will hit.

## Reranking comparison (Story 18.4)

Comparison run of `grounding eval` with rerank off vs on. Both tiers use
identical fixtures and the same public-repo commit. The decision rule for
flipping the default to `enabled: true` is: **at least one corpus tier shows
`recall@5` lift ≥ 0.03 AND `citation_accuracy` non-decreasing**.

| Corpus | Mode | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 † | citation_accuracy | n_items |
|--------|------|---------:|---------:|----------:|----:|----------:|------------------:|--------:|
| mini (public) | off | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| mini (public) | on  | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| private corpus (maintainer) | off | N/A | N/A | N/A | N/A | N/A | N/A | — |
| private corpus (maintainer) | on  | N/A | N/A | N/A | N/A | N/A | N/A | — |

**Configuration:** `BAAI/bge-reranker-base`, `pool_size=50`, `top_k=10`,
public-repo commit `f38eb44`, run 2026-04-15 UTC.

**Interpretation:**

- *mini (public):* the mini corpus is a 3-document, 5-chunk smoke test;
  `recall@5` is already saturated at `1.000` so reranking cannot lift it.
  Delta is `0.000` across every metric. This is the expected null result
  and does not say anything about rerank quality on real corpora.
- *private corpus:* no committed fixture set with ≥ 10 items against the
  maintainer's private `grounding-ai-private` / `my-agents` corpus is
  available at the time of this epic close, so the "realistic query
  diversity" row is N/A. Shipping the null result honestly rather than
  fabricating a number.

† nDCG@10 > 1.0 is a known quirk: the current implementation does not
normalize by the ideal DCG when multiple relevant chunks map to the same
doc. Tracked in `docs/TECH-DEBT.md` as a separate follow-up; does not
affect this flip decision because `recall@k` and `citation_accuracy` do
the load-bearing work here.

### Flip decision: **no flip**

Neither tier meets the `recall@5 lift ≥ 0.03` threshold (mini: zero lift by
saturation; private: no measurement). Per AC#4, the default stays
`retrieval.rerank.enabled: false` in `config.example.yaml`. The mini-corpus
baseline at `docs/eval/baselines/mini-corpus.json` is untouched; rerank-off
remains the CI-gated floor.

Reranking is fully wired (Stories 18.1–18.3) and opt-in via `--rerank` on
both `grounding eval` and `scripts/local_rag.py`, plus `rerank_enabled` on
the MCP `search_corpus` tool. Users with realistic query diversity can turn
it on per-query or per-config. A future story will re-run this comparison
once a ≥ 10-item private fixture set is committed; if the lift materializes
there, the default will flip then via a scoped refresh PR per the Epic 16
refresh discipline.

Because the default did **not** flip, the demonstration-regression PR
described in AC#10 is N/A: CI continues to gate rerank-off numbers, which
is what the existing `eval.yml` workflow already does.

## Hybrid retrieval comparison (Story 19.4)

Comparison run of `grounding eval` in the full `{hybrid off/on} × {rerank
off/on}` matrix. All four cells use identical fixtures and the same
public-repo commit. The flip decision rule for
`retrieval.hybrid.enabled` is identical to 18.4's rerank rule: **at least
one corpus tier shows `recall@5` lift ≥ 0.03 (hybrid-on vs hybrid-off,
with rerank held at its current default) AND `citation_accuracy`
non-decreasing**. Hybrid and rerank flip decisions are independent.

### Four-cell metric matrix

Shorthand: lowercase = off, uppercase = on, order is `{hybrid, rerank}`.

| Corpus | Cell | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 † | citation_accuracy | n_items |
|--------|------|---------:|---------:|----------:|----:|----------:|------------------:|--------:|
| mini (public) | `{hh}` off, off | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| mini (public) | `{Hh}` on,  off | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| mini (public) | `{hH}` off, on  | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| mini (public) | `{HH}` on,  on  | 1.000 | 1.000 | 1.000 | 1.000 | 1.421 | 1.000 | 3 |
| private corpus (maintainer) | `{hh}` off, off | N/A | N/A | N/A | N/A | N/A | N/A | — |
| private corpus (maintainer) | `{Hh}` on,  off | N/A | N/A | N/A | N/A | N/A | N/A | — |
| private corpus (maintainer) | `{hH}` off, on  | N/A | N/A | N/A | N/A | N/A | N/A | — |
| private corpus (maintainer) | `{HH}` on,  on  | N/A | N/A | N/A | N/A | N/A | N/A | — |

**Deltas vs `{hh}` baseline (mini):** `0.000` across every metric and
every cell. Saturation-limited (same constraint as 18.4's mini rerank
row) — no cell can beat `1.000` once the baseline already scores it.

**Configuration:** `hybrid.pool_size=50`, `hybrid.k_rrf=60`,
`rerank.model=BAAI/bge-reranker-base`, `rerank.pool_size=50`,
`top_k=10`, public-repo commit `f38eb44`, runs 2026-04-15 UTC.

† Same nDCG@10 quirk footnoted above (TD-002): the current
implementation does not normalize by the ideal DCG when multiple
relevant chunks map to the same doc. `recall@5` and `citation_accuracy`
do the load-bearing work for the flip decision; the nDCG quirk is
decision-irrelevant.

### Four-cell latency (per-query, warm, Apple Silicon)

Median per-query latency over 5 queries × several iterations on the
maintainer's `private-agent` agent index (16,251 chunks). The mini
corpus (3 chunks) cannot exercise a `pool_size=50` configuration so
latency is measured on a realistic private index. Exact numbers vary
with hardware; the ordering and order-of-magnitude are the user-facing
takeaway.

| Cell | hybrid | rerank | Median per-query latency | What dominates |
|------|:------:|:------:|-------------------------:|----------------|
| `{hh}` | off | off | **~7 ms** | FAISS search |
| `{Hh}` | on  | off | **~240 ms** | BM25 tokenize + rank over 16k tokens |
| `{hH}` | off | on  | **~1310 ms** | Cross-encoder scores 50 (q,c) pairs |
| `{HH}` | on  | on  | **~1580 ms** | Same rerank + ~240 ms BM25 overhead |

Measurement host: Apple Silicon arm64, 16 physical cores, 48 GB RAM,
Python 3.13.5, `sentence-transformers` / `torch` on CPU (no MPS).
BM25 dominates the non-rerank gap; the rerank cost dominates the
compound cell. Story 18.4 measured `bge-reranker-base` pool=50 at ~1.8
s on a 596k-chunk index; the 1.3 s figure here reflects the shorter
average chunk bodies in the 16k private-agent index. Both are
within the same order of magnitude.

### Interpretation

- *mini, `{hh}` vs `{Hh}`:* saturated at `1.000`; hybrid cannot lift
  what dense already maxes. Expected null result.
- *mini, `{hH}`, `{HH}`:* same saturation. Rerank + hybrid adds
  nothing because the 3-chunk pool has no wrong answer to demote.
- *private corpus:* unmeasured. No committed ≥ 10-item fixture set
  against `grounding-ai-private` / `my-agents` exists at the time of
  this epic close. Null row is accepted per Dev Notes, mirroring
  18.4's Task 2 outcome.

**Matrix-level take:** on the only tier that was measurable, every
cell is identical. The matrix cannot answer *"does hybrid add on top of
rerank?"* or *"does rerank absorb hybrid's gains?"* from public-repo
data alone. Both questions defer to a follow-up private measurement,
tracked in `docs/TECH-DEBT.md` under TD-004.

### Flip decision: **no flip**

Neither tier meets the `recall@5 lift ≥ 0.03` threshold (mini: zero
lift by saturation; private: no measurement). Per AC#4, the default
stays `retrieval.hybrid.enabled: false` in `config.example.yaml`.
Mini-corpus baseline at `docs/eval/baselines/mini-corpus.json` is
untouched; the dense-only `{hh}` numbers remain the CI-gated floor.

The rerank flip decision was independent and also NO FLIP (18.4). The
effective default remains `{hybrid=off, rerank=off}`. Users with
realistic query diversity can turn either or both on per-query via
`--hybrid` / `--rerank`, per-agent via `config.yaml`, or per-request
via the MCP `search_corpus` tool's `hybrid_enabled` / `rerank_enabled`
arguments.

TD-004 (hybrid flip decision deferred) in `docs/TECH-DEBT.md` tracks
the follow-up measurement once a private fixture set lands. TD-004
re-uses TD-003's fixture-set unblock — whichever follow-up story builds
the fixture set first unblocks the other.

Because the hybrid default did **not** flip, the demonstration-
regression PR described in AC#11 is N/A: the CI gate continues to gate
the `{hh}` baseline, which is what the existing `eval.yml` workflow
already does. Rationale documented here per Story 19.4 Task 10.

### HybridProvenance shape (AC#10 verification)

Spot-check of the four run JSONs confirms the expected provenance
shape: `{hh}` has no `hybrid` or `rerank` keys (legacy shape
preserved); `{Hh}` has `hybrid: {enabled: true, pool_size: 50,
k_rrf: 60}`; `{hH}` has `rerank: {enabled: true, model: "BAAI/bge-
reranker-base", pool_size: 50, batch_size: 16}`; `{HH}` has both.
Confirms Story 19.3's `HybridProvenance` serialization (runner.py:94,
report.py:137–141) lands correctly through the real eval stack.

## CI gate

`.github/workflows/eval.yml` runs on pull requests that touch retrieval-affecting
paths (the embedder, vector store, chunker, CLI, manifest, agent filter, the eval
package itself, mini fixtures, or `docs/eval/`). The workflow:

1. Installs the project against Python 3.13 with pip + sentence-transformers caches.
2. Builds the mini FAISS index (cached by content hash of the mini corpus).
3. Runs `grounding eval` against the mini baseline with `--fail-under 0.05`.
4. On regression, posts the Markdown report as a PR comment using a stable marker
   (`<!-- grounding-eval-comment -->`) so subsequent pushes update the same comment.
5. Uploads `eval-output/` as a workflow artifact.

The `paths:` filter in `eval.yml` is load-bearing. When introducing a new module
that affects retrieval, add its path to the filter in the same PR.

## Refreshing the baseline

**Baseline refreshes are intentional, human-reviewed, and PR-gated.** Do not
rubber-stamp a refresh to make the gate green on a regression.

1. Run the eval locally with the new code (see above).
2. Inspect the per-metric deltas in the stdout summary and the generated Markdown.
   Confirm each change is *intended* (e.g., a reranker shipped, embedding model
   upgraded).
3. Regenerate the baseline JSON:

   ```bash
   grounding eval --agent mini \
       --agents-dir tests/eval_fixtures/agents \
       --fixtures tests/eval_fixtures/mini_fixtures.yaml \
       --corpus tests/eval_fixtures/mini_corpus \
       --embeddings tests/eval_fixtures/mini_index \
       --out tmp/

   # Wrap the aggregate block with the baseline envelope
   # (format_version, agent, captured_utc, fixture_version, notes)
   # and overwrite docs/eval/baselines/mini-corpus.json.
   ```

4. Open a PR titled `eval: refresh mini-corpus baseline (<reason>)`. Describe
   what changed in the retrieval path and why the new numbers are acceptable.
5. Reviewer confirms the deltas match the change rationale before approving.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success; all metrics within `--fail-under` of baseline (or no baseline given) |
| `1` | Baseline regression; at least one metric dropped more than `--fail-under` |
| `2` | Fixture file or agent YAML not found, or fixture agent name mismatched `--agent` |
| `3` | Embeddings index not found at `--embeddings` |
| `4` | Unexpected runtime exception |

CI fails the PR on exit code 1 (regression) and surfaces the error on codes 2-4.

## FAQ

**Why does the CI gate run only against the mini corpus?**
CI cannot ship real corpora (size, licensing, privacy). The mini corpus is a
synthetic 3-document, 5-chunk fixture that exercises the full retrieval path.
Real-agent retrieval quality is the maintainer's concern, run privately.

**What happens if I rename a retrieval module?**
Update `.github/workflows/eval.yml` `paths:` filter in the same PR so the gate
still triggers on changes to the renamed file.

**How do I add a new fixture item?**
Edit `tests/eval_fixtures/mini_fixtures.yaml` (or your private fixture file) and
add an entry per the schema below. If the expected `doc_id` is not yet in the
corpus, the runner will skip that item and note it in the report rather than fail.

**Why is the CI fail-under 0.05 so loose?**
The mini corpus is tiny; noise floors are correspondingly high. Real-agent
fail-under thresholds (in private CI) should be tighter (e.g., 0.02).

## Where fixtures live

The public `grounding-ai` repo ships only two kinds of fixtures:

- **Schema example**: `docs/eval/fixtures/example.yaml`. Demonstrates the YAML format.
  Uses placeholder `doc_ids` and is **not** scored against a real corpus.
- **Mini test corpus fixtures** (Story 16.2): tiny synthetic corpus + fixtures used by
  CI to exercise the runner end-to-end.

Real-agent fixtures (targeting a contributor's actual agents and corpus, like
`data-scientist` or `ceo`) live in **private repos** (`grounding-ai-private`, `my-agents`)
because they reference private document IDs. Story 16.4 documents this layering.

## Fixture YAML schema

### Worked example

```yaml
agent: scientist
version: 1
items:
  - id: sci-001
    query: "What does Popper argue distinguishes science from pseudoscience?"
    expected:
      doc_ids: ["7a9b2c1f"]
      chunk_ids: ["7a9b2c1f/ch_0023"]   # optional, stricter form
    tags: ["methodology", "philosophy-of-science"]
    notes: "Source: Popper, The Logic of Scientific Discovery, Ch. 1."

  - id: sci-002
    query: "Bootstrap confidence interval procedure for a small sample mean"
    expected:
      doc_ids: ["3c4d5e6f", "9b1a2c3d"]   # any-of match
    tags: ["statistics"]
```

### Field reference

| Field | Required | Type | Rule |
|-------|----------|------|------|
| `agent` | yes | str | Must match `agents/<name>.yaml` filename stem. |
| `version` | yes | int | Currently must equal `1`. |
| `items` | yes | list | Must be non-empty. |
| `items[].id` | yes | str | Unique within file; recommend `<agent-prefix>-NNN`. |
| `items[].query` | yes | str | Non-empty after strip. |
| `items[].expected.doc_ids` | yes | list[str] | Non-empty. |
| `items[].expected.chunk_ids` | no | list[str] | Each must match `<doc_id>/ch_NNNN`. |
| `items[].expected.page` | no | int \| [start, end] | Positive int or ordered pair. See Citation Accuracy. |
| `items[].expected.section` | no | str | Non-empty. Matched case-sensitive against chunk `section_heading`. |
| `items[].tags` | no | list[str] | Lowercase kebab-case recommended. |
| `items[].notes` | no | str | Free text. |

### Matching semantics

- **`expected.doc_ids` is any-of**: a retrieval counts as a hit if *any* listed doc
  appears in the top-k results. Use this when multiple documents in the corpus cover
  the same ground equally well.
- **`expected.chunk_ids` is the stricter form**: the specific chunk must appear in
  top-k. Use this when you want to pin retrieval precision on a particular passage.
  The runner reports both metrics separately.

### Validation errors

The loader raises `FixtureValidationError` with structured context:

```
docs/eval/fixtures/example.yaml [item=sci-003 field=expected.doc_ids]: must be non-empty
```

Each error carries `.path`, `.item_id`, `.field`, and `.reason` attributes so CI logs
and IDE integrations can surface the problem without parsing strings.

### Citation Accuracy (Story 17.4)

The `citation_accuracy` metric measures how often the **first retrieved chunk**
matches a fixture's expected page and/or section. It is the CI-gated guardrail
against regressions that drop `page_start` / `section_heading` from the chunking
pipeline.

**Worked example:**

```yaml
agent: mini
version: 1
items:
  - id: mini-001
    query: "What does the paper conclude about bootstrap CI coverage?"
    expected:
      doc_ids: ["doc-beta"]
      page: 247                          # single-page expectation
      section: "3.2 Bootstrap Methods"   # exact string match
    tags: ["methodology"]

  - id: mini-002
    query: "How does falsifiability relate to demarcation?"
    expected:
      doc_ids: ["doc-gamma"]
      page: [15, 18]                     # range expectation
    tags: ["epistemology"]
```

**Hit rules:**

- `expected.page` as int `N`: the first retrieved chunk hits when
  `page_start <= N <= page_end`.
- `expected.page` as `[start, end]`: the first retrieved chunk's page range
  `[page_start, page_end]` must overlap `[start, end]`.
- `expected.section` as string: the first retrieved chunk's `section_heading`
  must equal it **exactly**, case-sensitive (intentional; fuzzy matching may be
  added later).
- When both fields are set, **both** must match.
- Items with neither `expected.page` nor `expected.section` are excluded from
  the metric. The report surfaces `n_citation_items` separately from `n_items`
  so small-N citation sets aren't over-read.

**Aggregate reporting:**

```json
"aggregate": {
  ...
  "citation_accuracy": 1.0,
  "n_citation_items": 2
}
```

The metric is `null` when no fixture item carries citation expectations.
When non-null, it participates in the CI `--fail-under` gate via the same
`worst_drop` computation as the other aggregate metrics. The Markdown report
surfaces it in the aggregate metrics table and adds a per-item
"Citation (retrieved p./§)" column showing the first retrieved chunk's
page/section.

**Report `format_version`:**

Eval JSON reports and baselines use `format_version: 2` from Story 17.4
onward. The loader accepts `format_version: 1` baselines and coerces missing
citation fields to `null` / `0` in-memory, so older baselines keep working
until the next intentional refresh.

### Using the loader

```python
from pathlib import Path
from grounding.eval import load_fixtures

fixtures = load_fixtures(
    Path("docs/eval/fixtures/example.yaml"),
    agents_dir=Path("agents/examples"),
)

for item in fixtures.items:
    print(item.id, item.query)
```

The loader performs no corpus or FAISS I/O. It only reads the fixture YAML and verifies
that `agents/<agent>.yaml` exists.
