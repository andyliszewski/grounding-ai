# Retrieval Evaluation Harness

This directory holds fixture files and documentation for the grounding-ai retrieval
evaluation harness (Epic 16).

A fixture is a small, hand-curated set of queries with known-good answers. The harness
runs the queries through an agent's FAISS index and scores the top-k retrievals against
the expected documents or chunks. Fixtures let us catch regressions in retrieval quality
as embeddings, chunking, or ranking logic changes.

This file covers the fixture schema (Story 16.1), the runner and metrics (Story 16.2),
the `grounding eval` CLI (Story 16.3), and the CI gate and baseline lifecycle (Story 16.4).
The grounded-answer benchmark (`grounding eval-answers`, Epic 25) has its own section,
"Answer benchmark (Epic 25)".

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

## Answer benchmark (Epic 25)

`grounding eval-answers` scores whether Claude's final answers, and the
citations inside them, hold up. The retrieval harness above checks that search
returns the right page; this benchmark checks what the model does with it. The
spec is `docs/epics/epic-25-grounded-answer-benchmark.md`.

Each fixture question (see "Answer block and page offsets") is answered in up
to four conditions, with the same model and the same base system prompt:

| Condition | Tool | Retrieval |
|-----------|------|-----------|
| `ungrounded` | none | none; cites title, edition or revision, and a clause, section, table or printed-page location |
| `dense` | `search_corpus` | FAISS dense |
| `hybrid` | `search_corpus` | dense plus BM25, reciprocal rank fusion |
| `hybrid-rerank` | `search_corpus` | hybrid plus `BAAI/bge-reranker-base` |

The tool runs in-process: it calls the corpus-search MCP server's
`search_corpus` and `format_results_for_context` directly, so the model sees
exactly what the MCP tool returns. The model may set `query` and `top_k`; the
harness pins the agent and the hybrid and rerank settings per condition.

### Install

```bash
pip install -e '.[bench]'   # anthropic SDK and mcp; a dry run needs neither
```

### Dry run (no key, no spend)

```bash
grounding eval-answers --agent mini \
    --agents-dir tests/eval_fixtures/agents \
    --fixtures tests/eval_fixtures/mini_answers.yaml \
    --corpus tests/eval_fixtures/mini_corpus \
    --embeddings tests/eval_fixtures/mini_index \
    --dry-run
```

This validates the fixture and index and prints estimated tokens and dollars
per condition, with every assumption it made (chars per token, searches per
answer, output lengths). It cannot call the token-counting endpoint without a
key, so it is a heuristic. For a measured figure, run a pilot first.

### Real run

Keep the key in a private file and pass it per command rather than exporting it
in a shell profile: an exported `ANTHROPIC_API_KEY` also reaches every other
program that reads it, Claude Code included. The runner reads the key from the
environment only and never logs it.

```bash
mkdir -p ~/.config/grounding-ai && chmod 700 ~/.config/grounding-ai
# paste the key into ~/.config/grounding-ai/anthropic-api.key, then:
chmod 600 ~/.config/grounding-ai/anthropic-api.key

ANTHROPIC_API_KEY="$(cat ~/.config/grounding-ai/anthropic-api.key)" \
grounding eval-answers --agent mechanical-engineer \
    --fixtures docs/eval/fixtures/private/mechanical-engineer-answers.yaml \
    --corpus /path/to/corpus \
    --embeddings /path/to/embeddings/mechanical-engineer \
    --limit 2 --max-cost 5            # pilot: measure real usage first

ANTHROPIC_API_KEY="$(cat ~/.config/grounding-ai/anthropic-api.key)" \
grounding eval-answers ... --max-cost 40   # full run; exports every answer for blind grading
```

`--answer-model` defaults to `claude-opus-5` and `--judge-model` to
`claude-sonnet-5`, so a different model grades than answers; pass either flag
to override. `--max-cost` refuses to start when the estimate exceeds it and
stops before any call that could push measured spend past it, keeping every
answer already written. `--items id1,id2` and `--limit N` select items;
`--max-iterations N` (default 5) caps tool rounds per answer, after which one
final call with `tool_choice: none` forces a text answer and the transcript is
marked `max_iterations`. `--blind-fraction` (default 1.0) sets how much of the
run goes into the blind-grade export; the default is the real-run path: every
answer is exported for human grading (see "Blind grading").

### Item selection (refused before the estimate)

Everything below fails a dry run too, with exit code 2 and the offending items
listed, so nothing is spent on an item that cannot be scored:

- **No `answer:` block.** A selected item without one has no gold to grade
  against. Add the block, or leave the item out with `--items`.
- **Gold document without a page index.** An answerable item whose
  `expected.doc_ids` names a document with no `page_start` in any chunk
  (Markdown, EPUB, a pdftotext fallback), or a document missing from the
  agent's index, can never show its gold page, so gold-page recall, the
  retrieval-versus-generation split and page-based citation checks would all
  read as misses. Point the item at a paged document or leave it out.

A warning (not an error) lists documents whose version the fixture cannot
check, but only where a version is likely to be stated or to matter: another
indexed document has the same title words, the document's name carries a
designation such as NASA-STD-5001 (standards are revised), or a fixture item
states an edition, revision or year of it. Declare the version under
`editions:` or `revisions:`. Government documents that are cited without a
version no longer produce a warning each. A second warning lists declared
`identifiers:` values whose series the scorer does not recognize, so a typo is
visible rather than silently ignored.

### What a run writes

Each run gets its own directory, `<--out>/answers-<agent>-<UTC timestamp>/`
(with a `-2`, `-3` suffix if two runs start in the same second). The default
`--out` is `docs/eval/reports/`, which is gitignored.

| File | Contents |
|------|----------|
| `run.json` | Provenance: git SHA and dirty flag, answer and judge model IDs, pinned generation settings, prompt versions and SHA-256 hashes, retrieval settings per condition, an index fingerprint (SHA-256 of the FAISS, chunk map and BM25 files and the corpus manifest), the embedding models (query and index side) and the sentence-transformers, torch and faiss versions, the pricing table used, and the pre-run estimate. |
| `transcripts.jsonl` | One line per answer: status (`ok`, `max_iterations`, `truncated`, `refusal`), final text, every model call (stop reason, usage, latency, attempts), every tool call with the citation prefixes it returned, total tokens, cost and latency. |
| `errors.jsonl` | Attempts that produced nothing scorable (API error after retries, served-model mismatch, retrieval failure, budget stop), with their usage so billed spend is never lost. |
| `retrieval.json` | recall@5 of the gold document and of the gold page per grounded condition, from the Epic 16 runner on the same fixture items. |
| `scores.jsonl` | One line per answer: every citation with its bucket, match type, resolution notes, claim length and judge verdict; correctness or abstention; whether the gold page reached the model (grounded); judge calls and cost. |
| `blind/` | The blind-grade export and resolution audit (see below). |
| `agreement.json`, `human_grades.json` | Written by `--import-grades`: judge-versus-human agreement, and one human grade per answer (no text). |
| `replicate-<run-id>.json` | Written by `--compare-run`: run-to-run agreement with a replicate run. |
| `report.md`, `report.json`, `chart.png` | The run report (see below). |
| `publishable/` | With `--publishable`: the report with aggregate scores only; with `--include-source-text` on a `source_license: public_domain` fixture, the transcripts, scores and blind grades as well. |

**Raw transcripts contain copyrighted source text** (the tool results), and
so do `scores.jsonl`, `blind/citations.csv` and the full report. Keep them in
`docs/eval/reports/` and never commit them. Only `publishable/` is meant to
leave the machine.

### Scoring (Story 25.3)

A new run scores itself after answering (skip with `--skip-scoring`). To
re-score existing transcripts, for example with a cheaper judge, run
`grounding eval-answers --run-dir <run> --score [--judge-model M] [--max-cost N]`.
Scores land in `scores.jsonl`, one row per answer.

**Citations.** Bracketed prefixes (`[slug, p.N, §Section]` and its smaller
variants) and free-text citations that name a work and a location (`(Title,
10th ed., p. 250)`, including one level of nested parentheses such as
`Eq. (5-19)`) are extracted, each with the sentence it supports and the length
of that claim. Every citation lands in exactly one bucket:

| Bucket | Grounded conditions | Ungrounded condition |
|--------|---------------------|----------------------|
| verified | located at a chunk a tool call in the same transcript returned (match `exact`, `located` or `label`, below), and the support judge finds the claim in that text | the title matches a corpus document, the text is located by a label or by the printed page mapped through `page_offsets`, and the judge finds the claim there |
| partial | names a returned document but gives no page, section or label; that document's returned chunks support the claim. Reported on its own, never counted as verified. | not used |
| unsupported | located, but the text does not support the claim | located, but the text does not support the claim |
| invented | no tool call in the transcript returned that location, the cited title is not a returned document, the cited edition differs from the returned one, or the cited label appears in the document only in chunks no tool call returned (`label_outside_returned_chunks`) | the mapped page does not exist in the cited work |
| unresolvable | not used | could not be checked: the work is not in the corpus (`work_not_in_corpus`), the title matches two documents equally (`ambiguous_title`), the cited edition differs from the document's (`edition_mismatch`) or the document's edition is unknown (`edition_unknown`), the work is section-paged (`section_paged`), no page is cited (`no_page_cited`), the cited label appears in the work only in passing and no page can be mapped (`label_mention_only`), the document has no `page_offsets` entry (`no_page_offset`) or no page index (`work_has_no_page_index`), or the mapped page has no extracted text (`no_text_at_page`) |

**Unresolvable means "could not be checked", not "wrong".** The report gives
verified of all citations and verified of checkable citations (unresolvable
left out) side by side, and splits unresolvable citations by reason. Reasons a
fixture edit can fix (`no_page_offset`, `edition_unknown`, `identifier_unknown`,
`work_has_no_page_index`, `no_text_at_page`, `ambiguous_title`) are counted
separately as fixable metadata, and the pre-registered gate reads that share.

Three reasons are **not** fixable metadata and are never counted toward the
gate: `section_paged` (a work paged "5-20" has no printed page to map, so no
fixture entry makes those citations checkable), `label_mention_only` (the
cited work names the label only in passing), and `work_not_in_corpus` (the
model cited a work the library does not hold, which is a finding about the
answer, not bookkeeping; the report prints its count beside the gate).

Match types, for citations that resolve:

| Match | Meaning |
|-------|---------|
| `exact` | grounded: the citation is a returned prefix, character for character |
| `located` | grounded: a page or section of a returned document matches returned chunks. Notes record leniency: `slug_fuzzy_match` (the slug named no returned document but fuzzy-matches one, for example with the sort prefix or edition token dropped; only documents this transcript returned are candidates, a slug that names another corpus document is never remapped, and a stated edition must agree), `section_mismatch` (the page matches but the section string does not), `edition_unchecked` (the title names a returned document whose edition the metadata cannot confirm) |
| `label` | both conditions: the citation names a table, figure, equation or clause label ("Table A-20", "Eq. (5-19)", "Clause 7.1", "§7.1", "para. 3.2") and its document resolves, so the label is searched for independent of page mapping. A **caption** locates it: the label opening a line or a chunk's section heading (the table itself, the clause heading). A single clause number needs its keyword to be a caption ("Section 5", "§ 5"), because lines such as "5 mm bolts" or a table row starting "5 " are not section 5; a dotted number followed by a capitalized word is one on its own ("7.1 Planning", but not "4.2 mm"). "Table A-20" does not match "Table A-201". The passage the judge reads is the caption chunks only. **Grounded conditions search only the chunks a tool call returned in that transcript**: a label found only in the rest of the document is `invented` (`label_outside_returned_chunks`), because the model was never shown that text, and a label a returned chunk mentions only in passing ("see Table A-20") resolves to that chunk with a `label_mention_only` note. **The ungrounded condition searches the whole cited work**, and a label found there only in passing does not locate text: the citation falls back to its page, and is `unresolvable` (`label_mention_only`) when it has none to map. A label that is not found at all falls back to the page (grounded: to the page, or to `partial` without one). |
| `partial` | grounded: the document only (see the bucket) |

An ungrounded citation located by its printed page has no match type; its
resolution detail is `mapped_printed_page`.

**Document identifiers are identity, not title words.** A citation that names
a designation such as `DOE-HDBK-1018-93`, `NASA-STD-5001B`, `MIL-HDBK-5J`,
`NASA RP-1228`, `ASME Y14.5-2018`, `ISO 13485:2016` or `21 CFR 820.30` only
matches a document carrying that same identifier: a different number is a
different document (DOE-HDBK-1019 is not DOE-HDBK-1018), and the title words
around it cannot override that. The recognized series codes are listed in
`_ID_SERIES` in `grounding/eval/answers/citations.py`; a number outside a
recognized series is not an identity token. A document whose name carries no
identifier cannot confirm a cited one: that citation is `identifier_unknown`
(unresolvable, counted as fixture metadata), and the fixture's `identifiers:`
map declares the identifier so it can be checked.

**Editions, revisions and years are one version check.** A version is an
edition number ("10th ed.", `...-10e`), a revision letter or number ("Rev. C",
the B of NASA-STD-5001B, "Rev. 1") or a year ("ISO 13485:2016", the -93 of
DOE-HDBK-1018-93, a year in the document's name). Only the kinds the citation
states and the document knows are compared: a citation giving an edition and a
year matches a document that knows only its edition; one giving only a year
against a document that knows only its edition is `edition_unknown`, never
assumed. Any comparable kind that differs is `edition_mismatch`. Declare what
a name does not carry with `editions:` (a numbered edition) or `revisions:` (a
revision letter or a year).

**Title matching** (ungrounded titles, grounded free-text titles and fuzzy
slugs) scores the whole cited title and each of its comma, semicolon or colon
segments against every document's `slug` and `orig_name`, so "Budynas &
Nisbett, Shigley's Mechanical Engineering Design, 10th ed." matches
`000-a-a-shigleys-mechanical-engineering-design-10e`. A candidate matches when
at least 75 percent of its words are in the document's names and it covers at
least 60 percent of the document's title words; or it covers at least 75
percent (and at least two) of the title words, in order, with at least 60
percent of its own words explained (an author prefix without a comma); or, for the whole title
only, its first word is the document's first title word ("Roark",
"Shigley's"). Document title words drop numbering noise (`000`, `10e`, years).
The best-scoring document wins; ties are `ambiguous_title` unless the cited
edition separates them. Document names give editions as "10th Edition" or the
`10e` / `27e` / `2e` convention (never `1e-3`), or the fixture declares them.

**Correctness** (answerable items): numeric items are auto-scored first. The
check only decides when the answer states the gold unit unambiguously (score 1
when every such value is within tolerance, 0 when all are outside); ranges,
mixed values, other units and `must_include` items go to the rubric judge,
which scores 1, 0.5 or 0 against the gold answer and `must_include`. The judge
reads the answer with citations removed and never sees retrieved text. The
report also counts **confidently wrong** answers: scored 0 without declining
(empty answers are not counted as confident).

**Abstention** (unanswerable items) depends on `answer.unanswerable_kind`,
and each kind scores the behavior its question asks for:

| Kind | Scored in | Desired behavior | The judge is told |
|------|-----------|------------------|-------------------|
| `no_source` | every condition | `declined` | no reliable source answers the question as asked (false premise, missing information, unpublished data); declining, pointing out the premise or asking for the missing information counts as declining |
| `not_in_corpus` | grounded conditions only | `acknowledged_gap` | the library does not contain the answer, though it may be published elsewhere, and the model was told to say so plainly and to label any general knowledge it adds. The judge reports three facts: `gap_flagged`, `gave_answer` and `answer_labeled`. `acknowledged_gap` = flagged, and any answer labeled; `declined` = flagged and no answer. |

**`acknowledged_gap` is the headline for `not_in_corpus` items**, and the
report gives it its own section: the grounded prompt (`answer-v3`) asks the
model to say plainly when the sources do not contain the answer and to label
any general knowledge it adds, so a model that flags the gap and then answers
from labeled general knowledge followed its instructions. `declined` (flagged
and gave no answer) is reported beside it, along with how many answered from
labeled general knowledge and how many answered without flagging the gap.

A `not_in_corpus` answer in the ungrounded condition is not scored at all
(`excluded`): a model without the library may know the answer, so it never
counts against that condition. Abstention is reported per kind, as counts
("2/3") whenever fewer than 10 answers are behind a rate.

**Retrieval versus generation.** Each grounded score row records whether any
tool call surfaced the gold document and the gold page (the same page-overlap
rule as gold-page recall). The report splits grounded answers that were not
fully correct into "gold page not surfaced" (a retrieval miss) and "surfaced,
still wrong" (a generation miss).

Prompts are versioned constants in `grounding/eval/answers/prompts.py`: the
answer prompt `answer-v3` (all conditions are told "State one claim per
sentence and cite it."; grounded conditions are told to say plainly when the
sources do not contain the answer and to label any general knowledge as not
from the sources; the ungrounded condition is asked for the clause, section or
paragraph number where a source numbers them, the table, figure or equation
number, and the printed page when known), and the judge prompts
`correctness-v2`, `abstention-no-source-v2`, `abstention-not-in-corpus-v2` and
`support-v2`. The answer prompt names the persona it answers for, which the
fixture's `persona:` sets (default: "a practicing mechanical engineer"), and
the judges speak of "the question writer", so one set of prompts serves a
mechanical-engineering corpus and a public-domain government one.
`run.json` records each version and a SHA-256 of the prompt and schema. Judge
calls use structured JSON output. A judge call that fails leaves that grade
empty (reported as unjudged) and is logged in `errors.jsonl`; it is never
scored as a model failure.

### Blind grading and judge agreement

Scoring also writes the blind-grade export, with the condition hidden and rows
shuffled. **The default, and the path for the real run, is every scored
answer** (`--blind-fraction 1.0`): the maintainer grades all of them, and the
report then uses the human grades as the primary correctness metric, with the
judge secondary. A smaller fraction draws a stratified random sample (per
condition, category and unanswerable kind: that share of each stratum and at
least one; fixed `--blind-seed`). Answers whose judge call failed are still
exported.

| File | Fill in |
|------|---------|
| `blind/answers.csv` | `human_grade`: 1, 0.5 or 0 for answerable items; for unanswerable items, 1 when the answer did what its kind asks for (`no_source`: declined; `not_in_corpus`: said plainly that the sources do not contain the answer and labeled any general knowledge it added), else 0. The scale is worded per unanswerable kind, as the judge's was. The answer text has its citations removed, as the judge saw it. |
| `blind/citations.csv` | `human_supported`: y or n, for each judged citation of the exported answers, with its claim and passage. The rows are in the answers' shuffled order, so **grade them from the top and stop where you like**: the first rows are a random sample of answers. The support gate needs at least 10, and a sample with some unsupported citations in it (see the gate below). |
| `blind/resolution_audit.csv` | `human_agrees`: y or n, for citations the scorer put in `invented` or `unresolvable` without a judge (stratified by condition and bucket), with its reason and what it compared against (the prefixes returned in that transcript, or the matched document and mapped pages). This audits the scorer's bookkeeping, and shows the citation string, so these rows reveal grounded versus ungrounded (never which retrieval variant). |
| `blind/key.json` | Nothing. Maps sample ids to conditions and grader scores; do not open it while grading. |

The pre-rename columns `andy_grade`, `andy_supported` and `andy_notes` are
still accepted on import.

Then run `grounding eval-answers --run-dir <run> --import-grades` (or pass the
path of edited copies). It writes `agreement.json` (per metric: double-graded
rows, raw agreement and Cohen's kappa; correctness overall and by method;
abstention per kind; the resolution audit) and `human_grades.json`.

- **Publish gate**: correctness agreement of at least 80 percent on the rows
  the rubric judge graded (`by_method["judge"]`), with at least 10 of them,
  else `insufficient`. Numeric auto-scored rows agree with a careful human
  almost by construction, so they are shown in the all-rows figure but do not
  decide the gate.
- **Support gate**: the support judge decides the headline citation metric and
  grades every citation, while the human grades a sample, so it is held to
  agreement of at least 80 percent **and** Cohen's kappa of at least 0.6 over
  the citations the human graded (at least 10 of them, else `insufficient`).
  Kappa is what the raw rate cannot show: when nearly every citation is
  supported, a judge that says "supported" to everything still agrees 90
  percent of the time and has told you nothing. A kappa that is undefined
  (both graders used one label only) reports `insufficient` rather than
  passing. Both numbers are reported.

To export again without spending anything (for example at another fraction):
`grounding eval-answers --run-dir <run> --export-blind [--blind-fraction F]`.
Any CSV that already holds grades is moved to `blind/archive-<UTC>/` first, so
filled grades are never overwritten.

### Report (Story 25.4)

Scoring, re-scoring and grade import all (re)write `report.md`, `report.json`
and `chart.png`; `grounding eval-answers --run-dir <run> --report` re-renders
them on demand. The report has:

- the **pre-registered primary comparison** (see the epic doc): verified rate
  of all citations, `hybrid-rerank` minus `ungrounded`, **over answerable
  items only**, as a paired difference with a 95 percent bootstrap interval
  that resamples items (not model runs); the same for verified of checkable
  citations as the secondary; the share of those ungrounded citations
  unresolvable for reasons a fixture edit can fix, with a warning above 10
  percent; and, beside it, how many named a work the library does not hold,
  which no fixture entry can fix;
- a per-condition table: correctness from the primary source (human grades
  once every answerable answer is graded, else the judge) with a 95 percent
  interval and the paired difference from `ungrounded`, the other grader's
  correctness, verified of all and verified of checkable citations (intervals
  resample answers, not citations), declined and confidently wrong, and
  abstention per kind;
- retrieval, cost and latency: citations per answer, median claim words per
  citation, recall@5 of the gold page and document, tokens, answer and judge
  cost, latency;
- how the grounded conditions handled questions the library cannot answer
  (`not_in_corpus`): acknowledged the gap (the desired behavior), declined
  outright, answered from labeled general knowledge, answered without flagging;
- the citation buckets, unresolvable citations by reason, and counts of label
  matches and resolution notes;
- retrieval versus generation for the grounded conditions;
- a per-category breakdown (n first; counts when n is under 10), the worst
  failures by item id, judge validation (human coverage, both gates, the
  resolution audit), replicate agreement, and run health (errors, refusals,
  truncations, answers that hit the tool-round cap, excluded answers);
- one PNG with three small-multiple panels (correctness, verified of all,
  verified of checkable) by condition, with 95 percent whiskers.

`--publishable` also writes `publishable/report.md`, `report.json` and
`chart.png` with aggregate scores only: no answer text, claims, citation
strings, passages, judge reasons, gold answers or local paths. Question texts
are added only with `--include-questions`, after reviewing them (D7).
`tests/test_eval_answers_report.py` checks that no chunk text, answer text or
question reaches those files.

**Public-domain corpora may publish everything.** When the fixture declares
`source_license: public_domain` (a corpus of US government works, for example
`benchmarks/public-corpus/`), `--publishable --include-source-text` also
copies `transcripts.jsonl`, `scores.jsonl`, `retrieval.json`, the blind CSVs
and their key, and `agreement.json` into `publishable/`, and keeps the judge
reasons and question texts in the report. Without that declaration the flag is
refused (exit code 2), so a copyrighted corpus cannot be published by passing
a flag. `errors.jsonl` is never copied: an exception message can carry a local
path.

```bash
grounding eval-answers --run-dir docs/eval/reports/<run-id> --report --publishable

# A public-domain corpus: publish the passages, transcripts and blind grades too
grounding eval-answers --run-dir docs/eval/reports/<run-id> --report --publishable \
    --include-source-text
```

### Replicate check

A replicate is a second run of the same condition on the same items: it
measures how far scores move from the model's sampling alone, the noise floor
a between-condition difference has to clear. **Re-run the whole condition, not
a subset**: the pre-registered replicate is every item again. A subset moves
the verified rate much further with nothing changed (the power simulation puts
a 12-item replicate at about 10 points of movement against about 6 for a full
re-run), and the report flags it as partial.

```bash
grounding eval-answers ... --conditions hybrid-rerank      # second run dir, every item
grounding eval-answers --run-dir docs/eval/reports/<run-id> \
    --compare-run docs/eval/reports/<replicate-run-id>
```

`--compare-run` writes `replicate-<replicate-run-id>.json` (per shared
condition: shared items and whether they cover the whole run, correctness
exact agreement, kappa and mean absolute difference, abstention agreement, and
both runs' verified rates with the paired 95 percent interval of their
difference) and the report shows it. It compares judge scores, so it measures
model plus judge noise. Runs that differ in models, prompts, fixture, index or
tool-round cap are flagged as not comparable.

### Power simulation

`python -m grounding.eval.answers.power` prints the expected interval
half-widths at a given number of items, and the run-to-run movement of a
replicate, using the report's own bootstrap functions
(`grounding/eval/answers/stats.py`). The epic's pre-registered power paragraph
quotes its output at the default settings (35 items, 200 simulations, seed 0);
re-run it with `--items` to size a different question set.

### Exit codes (`eval-answers`)

| Code | Meaning |
|------|---------|
| `0` | Success, or a dry run |
| `2` | Bad arguments, fixture or agent problems |
| `3` | Embeddings index missing, or the BM25 sidecar missing for a hybrid condition |
| `4` | Unexpected failure |
| `5` | No `ANTHROPIC_API_KEY` (or SDK not installed) for a real run |
| `6` | `--max-cost` would be or was exceeded; results so far are kept |

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
| `items[].expected.doc_ids` | yes | list[str] | Non-empty, except on items with `answer.answerable: false` (Story 25.1). |
| `items[].answer` | no | mapping | Grounded-answer benchmark block (Story 25.1). See "Answer block and page offsets". |
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

### Answer block and page offsets (Epic 25, Story 25.1)

Fixture items may carry an optional `answer:` block that the grounded-answer
benchmark (`grounding eval-answers`, see "Answer benchmark (Epic 25)" above)
scores against. The retrieval harness ignores it, except that it skips items
marked `answerable: false`, because they have no gold document to retrieve.

```yaml
agent: mechanical-engineer
version: 1
page_offsets:            # optional: doc_id -> offset, pdf_page = printed_page + offset
  3f2a9c1d: 22
  8b7e6a5f: section      # a section-paged handbook ("5-20"): printed pages are never mapped
editions:                # optional: doc_id -> edition, when the document name has none
  3f2a9c1d: 10
revisions:               # optional: doc_id -> revision letter or year, same check as editions
  9c0d1e2f: "Rev. C"
  4a5b6c7d: 2023
identifiers:             # optional: doc_id -> designation, when the name does not carry it
  9c0d1e2f: NASA-STD-5005
items:
  - id: me-001
    query: "What is the modulus of elasticity of carbon steel?"
    expected:
      doc_ids: ["3f2a9c1d"]
      page: 1017         # PDF page index of the gold passage (required for answerable items)
    answer:
      category: table    # table | formula | standard | guidance | judgment | unanswerable
      gold: "About 200 GPa (29,000 ksi)."
      numeric: {value: 200, unit: GPa, rel_tol: 0.05}   # optional
      must_include: ["200 GPa"]                        # optional
  - id: me-040
    query: "A question the corpus cannot answer"
    answer:
      category: unanswerable
      answerable: false  # expected.doc_ids may be empty or absent; no page needed
      unanswerable_kind: not_in_corpus   # or no_source
```

| Field | Required | Rule |
|-------|----------|------|
| `answer.category` | yes | One of `table`, `formula`, `standard`, `guidance`, `judgment`, `unanswerable`. |
| `answer.answerable` | no | Default `true`. `false` if and only if `category: unanswerable`. |
| `answer.unanswerable_kind` | unanswerable items | `no_source`: no reliable source answers the question as asked (a false premise, missing information, unpublished data); scored in every condition. `not_in_corpus`: answered elsewhere but not by the corpus; scored for grounded conditions only and never counted against the ungrounded one. Not allowed on answerable items. |
| `answer.gold` | answerable items | Short gold answer text. Optional for unanswerable items. |
| `answer.numeric` | no | `value` (number), optional `unit`, and exactly one of `rel_tol` (fraction of value) or `abs_tol` (same unit). Not allowed on unanswerable items. |
| `answer.must_include` | no | List of key facts a fully correct answer must state. Not allowed on unanswerable items. |
| `expected.page` | answerable items with an answer block | The gold page as a **PDF page index** (chunk `page_start`/`page_end`), not a printed page number. The gold document must have a page index: `eval-answers` refuses answerable items whose gold document has none. |
| `revisions` | no | Top-level map from `doc_id` to a revision letter ("B", "Rev. C"), a numbered revision ("Rev. 1") or a four-digit year (2016), for documents whose names carry none ("NASA-STD-5001B", "ISO 13485:2016", an FDA guidance year). Checked exactly like `editions`: a cited revision or year that differs is `edition_mismatch`, one that cannot be compared is `edition_unknown`. |
| `identifiers` | no | Top-level map from `doc_id` to a document identifier ("DOE-HDBK-1018-93", "NASA RP-1228", "21 CFR Part 820"), for documents whose names do not carry it. Without it, a citation that names an identifier scores `identifier_unknown` against that document. |
| `page_offsets` | no | Top-level map from `doc_id` to an integer (maps printed pages cited by an ungrounded answer to PDF pages) or to `section` (a section-paged work such as a handbook paged "5-20", whose printed pages cannot be mapped; page citations to it are unresolvable, though table, figure, equation and clause labels still locate text). Without an entry, page citations are scored unresolvable rather than guessed. |
| `persona` | no | Top-level string naming who the answers are for ("a NASA structures engineer"). It is rendered into the answer prompt in every condition; the default is "a practicing mechanical engineer". The judges never see it and speak of "the question writer". |
| `editions` | no | Top-level map from `doc_id` to a positive integer, for documents whose names carry no edition ("10th Edition" or "10e"). A citation that states an edition is only checked against a document whose edition is known; a different edition, or an unknown one, is unresolvable, so a page is never mapped into the wrong edition. |

Unknown keys inside `answer:` or `answer.numeric:` are rejected, so a typo such
as `must_inlcude` fails loudly instead of silently dropping a fact.

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
