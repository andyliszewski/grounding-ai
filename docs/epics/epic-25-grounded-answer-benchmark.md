# Epic 25: Grounded-Answer Benchmark

**Epic ID:** E25
**Owner:** Andy
**Status:** In progress (25.1 to 25.4 built on branch, methodology-review fixes applied; 25.5 waits on Andy's questions)
**Priority:** P1
**Completed Stories:** 4/6
**Dependencies:** Epic 16 (Retrieval Evaluation Harness), Epic 17 (page and section citations), Epic 18 (reranking), Epic 19 (hybrid retrieval), Epic 22 (MCP tool bridge)
**Target Completion:** 2026-09-28 (proposed)
**Source:** Andy, 2026-09-14. The Epic 16 harness scores whether search returns the right page. Nothing yet scores whether the final answer, and the citations inside it, hold up.

---

## Branching Plan

| Story | Branch target | Private-only content? | Cross-repo coordination |
|-------|---------------|------------------------|--------------------------|
| 25.1  | public `main` (feature branch, squash PR) | No | None |
| 25.2  | public `main` (feature branch, squash PR) | No | None |
| 25.3  | public `main` (feature branch, squash PR) | No | None |
| 25.4  | public `main` (feature branch, squash PR) | No | None |
| 25.5  | private only | Yes: `docs/eval/fixtures/private/`, excluded by `scripts/publish.sh` | Reads `my-agents/agents/mechanical-engineer.yaml` |
| 25.6  | public `main` for the summary only | Raw transcripts stay in `docs/eval/reports/` (gitignored) | None |

**Spend:** nothing runs against the API until Andy starts a real run with his own key.

---

## Overview

The question: when Claude answers an engineering question, can a reader verify the answer?
More precisely, does grounding Claude in a cited corpus change (a) whether the answer is right
and (b) whether its citations lead to text that supports it? Summarizing is easy. Producing
citations that check out is the hard part, so **citation verifiability is the headline metric**.

### Conditions

| Condition | Tools | Retrieval |
|-----------|-------|-----------|
| `ungrounded` | none | none |
| `dense` | `search_corpus` | FAISS dense (today's default) |
| `hybrid` | `search_corpus` | dense plus BM25, reciprocal rank fusion |
| `hybrid-rerank` | `search_corpus` | hybrid plus `bge-reranker-base` |

Same model and the same system prompt in every condition, except that grounded conditions get
the tool and are told to cite tool results by their `[slug, p.N, §section]` prefix, to say
plainly when the sources do not contain the answer, and to label any general knowledge they add
as not from the sources. The ungrounded condition is told to cite title, edition or revision,
and a location for each factual claim: the clause, section or paragraph number where the source
numbers them, the table, figure or equation number, and the printed page when it knows it, so
section-paged works can still be checked. Every condition is told to state one claim per
sentence and cite it (`answer-v3`), so each citation is paired with one claim the support judge
can check, and the base prompt names the persona the fixture sets (`persona:`, default "a
practicing mechanical engineer"). The tool description names no retrieval method, so it is
accurate in every grounded condition.

### Metrics, per condition

1. **Correctness:** 1, 0.5 or 0 against Andy's gold. **Primary: Andy's blind grades of every
   answer** (condition hidden); the judge's grades are secondary. For the judge, numeric items
   are auto-scored against `value` and tolerance first; the rest go to an LLM judge with a fixed
   rubric. The judge never sees retrieved text when grading correctness. Also reported:
   declined, and **confidently wrong** (scored 0 without declining).
2. **Citation verifiability (headline):** every citation lands in exactly one bucket.
   - *verified:* resolves to a corpus location whose text supports the claim;
   - *partial:* grounded only, names a returned document with no page, section or label, and
     that document's returned text supports the claim; never counted as verified;
   - *unsupported:* resolves, but the text does not support the claim;
   - *invented:* grounded conditions, a location no tool call returned; ungrounded, a page that
     does not exist in the cited work;
   - *unresolvable:* ungrounded only, the citation **could not be checked**: the work is not in
     the corpus, its edition differs or is unknown, it is section-paged, or its printed page
     cannot be mapped. Unresolvable is not "wrong".

   A citation resolves by exact prefix, by page or section of a returned document, or by a
   table, figure, equation or clause label ("Table A-20", "Clause 7.1"), independent of page
   mapping. A label is located by its caption (the table itself, the clause heading), never by
   a passing mention, and a single clause number needs its keyword ("Section 5"). In grounded
   conditions only the chunks a tool call returned are searched: a label that sits only in the
   rest of the document is *invented*, since the model never saw that text. Report the
   buckets, verified of all citations, and verified of checkable citations (unresolvable left
   out), with unresolvable split by reason.
3. **Abstention**, per `unanswerable_kind`, scored on the behavior each kind asks for.
   `no_source` items (no correct answer from any source) are scored in every condition, and
   the desired behavior is declining. `not_in_corpus` items (answered elsewhere, not by the
   library) are scored in grounded conditions only, never counted against the ungrounded
   condition, and the desired behavior is **acknowledging the gap**: saying plainly that the
   sources do not contain the answer, with any general knowledge labeled as not from them. A
   model that flags the gap and then answers from labeled general knowledge did what the prompt
   asked; declining outright is reported beside it. Counts, not bare rates, when n is under 10.
4. **Retrieval recall@5** of the gold page, grounded conditions only, from the Epic 16 runner
   on the same fixture.
5. **Retrieval versus generation:** for wrong grounded answers, whether any tool call surfaced
   the gold page (a generation miss) or not (a retrieval miss).
6. **Cost and latency:** input and output tokens, dollars, wall time.

### Judge validation

Andy grades **every** answer blind (the default export, `--blind-fraction 1.0`; a smaller
fraction draws a sample stratified by condition, category and unanswerable kind), from a CSV
with the condition hidden, and audits a sample of the citations the scorer called invented or
unresolvable. The report gives judge-versus-Andy agreement per metric, overall and on the rows
the rubric judge graded. Publish gate: correctness agreement on rubric-judge rows of at least
80% (n at least 10); numeric auto-scored rows agree almost by construction and do not count.
Support gate: the support judge stays the primary citation grader (Andy hand-grades every
answer's correctness, not every citation), so it is held to agreement of at least 80% **and**
Cohen's kappa of at least 0.6 on the citations Andy graded (n at least 10). Kappa is what raw
agreement cannot show when nearly every citation is supported. If a gate fails, fix that rubric
and re-score before publishing anything that rests on it.

### Design decisions

- **D1. One fixture drives both harnesses.** Answer fields sit in a new optional `answer:`
  block. The Epic 16 loader reads only the keys it knows, so existing runs and baselines are
  unchanged.
- **D2. The tool runs in-process.** It calls `mcp_servers.corpus_search.server.search_corpus`
  directly, so the benchmark exercises the exact retrieval code the MCP server serves, without
  transport noise. Tool name, description and input schema mirror the MCP tool.
- **D3. Messages API with a tool-use loop** (the `anthropic` SDK), not `claude -p`. That gives
  pinned model IDs, fixed system prompts and full transcripts. Answer and judge models are
  flags, and both are recorded per run.
- **D4. Pages are PDF page indices** (chunk `page_start`/`page_end`), not printed page numbers.
  Ungrounded answers cite printed pages. Map them with a per-document `page_offsets:` entry when
  Andy provides one (`pdf_page = printed_page + offset`); otherwise mark the location
  unresolvable rather than guessing with a wide window.
- **D5. Correctness is graded against Andy's gold**, never against retrieved text, so grounding
  is not graded by itself.
- **D6. No spend without Andy.** `--dry-run` prints estimated tokens and cost; `--max-cost`
  aborts a run that would exceed it. The key comes only from `ANTHROPIC_API_KEY`.
- **D7. Raw transcripts contain copyrighted source text.** They go to `docs/eval/reports/`
  (gitignored). The publishable summary carries aggregate scores and, after Andy's review, the
  question texts. Never source text.
- **D8. Every run is reproducible.** Each run records git SHA, model IDs, prompt hashes,
  retrieval config, an index fingerprint, and the embedding models and library versions.
- **D9. The scorer never grades its own bookkeeping as the model.** A citation it cannot
  check is `unresolvable`, reported apart from wrong ones, with its reason. Items it could
  never score (no `answer:` block, a gold document with no page index) are refused before any
  spend. A cited edition, revision or year is checked, never assumed, so a page is never mapped
  into the wrong version; and a cited document identifier (DOE-HDBK-1018-93, NASA-STD-5001B,
  21 CFR 820) must match the document's exactly, because a different number is a different
  document.
- **D10. Human grades are primary for correctness.** Andy grades every answer; the judge is
  secondary and validated against him. The judge still decides citation support, validated by
  the support gate.

---

## Stories

### Story 25.1: Answer-fixture schema

Add an optional `answer:` block to fixture items:

- `category`: one of `table`, `formula`, `standard`, `guidance`, `judgment`, `unanswerable`;
- `gold`: short gold answer text;
- `numeric` (optional): `value`, `unit`, and one of `rel_tol` or `abs_tol`;
- `must_include` (optional): list of key facts a correct answer must state;
- `answerable`: default `true`.

- `unanswerable_kind`: required when `answerable: false`; `no_source` or `not_in_corpus`.

Add an optional top-level `persona:` (who the answers are for; it is rendered into the answer
prompt) and `source_license: public_domain` (the opt-in that lets a publishable report carry
source text).

Add an optional top-level `page_offsets:` map from `doc_id` to integer, or to `section` for a
section-paged work, an optional `editions:` map from `doc_id` to edition for documents whose
names carry none, an optional `revisions:` map for revision letters and years (NASA-STD-5001B,
ISO 13485:2016, an FDA guidance year), and an optional `identifiers:` map for documents whose
names do not carry their designation (DOE-HDBK-1018-93, 21 CFR Part 820). Items with `answer.answerable: false` may leave `expected.doc_ids` empty; the
Epic 16 runner skips them.

**Acceptance:** the Epic 16 runner passes on the mini fixture unchanged; each validation path
has a test (numeric needs a value and exactly one tolerance, unknown category rejected,
unanswerable items need no page).

### Story 25.2: Answer runner

`grounding eval-answers --fixtures F --agent A --corpus C --embeddings E
--conditions ungrounded,dense,hybrid,hybrid-rerank --answer-model M --judge-model J --out DIR
[--dry-run] [--max-cost USD] [--limit N] [--items id1,id2]`

- Tool loop with a maximum number of iterations (default 5). Tool results are formatted exactly
  as the MCP server formats them: citation prefix, then body.
- Records per answer: final text, every tool call with the citation prefixes it returned,
  tokens, latency and stop reason. Transcripts are JSONL under `--out`.
- The client is injectable. Tests use a fake client with scripted responses and no network.

**Acceptance:** `--dry-run` works without a key; a fake-client run over the mini corpus writes
transcripts for all four conditions.

### Story 25.3: Scoring

- **Citation extraction:** bracketed prefixes in the `grounding/citations.py` formats, plus
  free-text citations (title, edition, page) from ungrounded answers.
- **Resolution:** a grounded citation must match a chunk returned by a tool call in the same
  transcript, or it is *invented*; before that verdict, an unrecognized slug is fuzzy-matched
  against the documents that transcript returned, and a correct page with a wrong section is
  *located* with a note; a label is searched for in the returned chunks only, and a label found
  only in chunks no tool call returned is *invented* (`label_outside_returned_chunks`). An
  ungrounded citation resolves title to document by fuzzy match on `slug` and `orig_name`
  (whole title and each comma-separated segment, with a document-side recall term), checks the
  edition, then locates text by a label's caption or by printed page through `page_offsets`; a
  label the work names only in passing locates nothing (`label_mention_only`).
- **Support check:** the judge sees the claim and the text at the resolved location, and
  returns supported or not with a one-line reason.
- **Correctness:** numeric auto-score first; otherwise the judge with a rubric against `gold`
  and `must_include`.
- **Abstention** scoring for `answerable: false`.
- **Blind-grade export and import:** a CSV of every answer (or a sample stratified by condition,
  category and unanswerable kind) with the condition hidden, plus a resolution audit of
  invented and unresolvable citations; import computes agreement and the gates.

**Acceptance:** unit tests cover every citation bucket with scripted answers; judge prompts are
versioned constants.

### Story 25.4: Report

Markdown and JSON per run:

- the pre-registered primary comparison (below);
- a per-condition table (correctness from Andy's grades once complete, else the judge;
  verified of all and of checkable citations; declined and confidently wrong; abstention per
  kind; recall@5, tokens, cost, latency);
- citation buckets with unresolvable reasons, retrieval versus generation, and replicate
  agreement;
- a per-category breakdown;
- the worst failures by item id;
- one PNG chart of correctness and verified-citation rates by condition.

`--publishable` drops all source text and transcript excerpts.

**Acceptance:** a report builds from the fake-client run, and a test proves publishable output
contains no chunk text.

### Story 25.5: Question set (Andy, with Claude drafts; amended below)

30 to 50 questions in `docs/eval/fixtures/private/mechanical-engineer-answers.yaml`, written
from Andy's own knowledge and then located in the PDF. Rough mix:

| Category | Count |
|---|---|
| tables | 8 to 10 |
| formulas | 8 to 10 |
| standards | 6 to 8 |
| FDA guidance | 4 to 6 |
| engineering judgment | 4 to 6 |
| unanswerable | 4 to 6 |

Rules:
- No confidential client details.
- Write each question before searching, so the set is not biased toward what retrieval already
  finds.
- The gold page is the PDF page index, in a document ingested with pages. The runner refuses an
  answerable item whose gold document has no page index.
- Every unanswerable item states its `unanswerable_kind`.
- Give `page_offsets` (and `editions` where the file name has none) for every work the
  ungrounded condition is likely to cite, and mark section-paged handbooks `section`; the
  primary comparison is read only when metadata gaps leave at most 10% of ungrounded citations
  unresolvable.

**Amended 2026-09-15, before the first run (Andy's decision).** The benchmark's question set is
`benchmarks/public-corpus/questions.yaml`: 30 items (26 answerable, 4 unanswerable) against the
public corpus. The private mechanical-engineer file above remains a separate, private set.
Andy wrote four items. Claude (Opus 5) drafted the other 26 from the documents and checked
each gold against its PDF page; Andy then vetted all 30, kept every one, and asked for two
changes (a wider gold page on pub-002, a clearer question on pub-403). The rule "write each
question before searching" therefore does not hold for the drafted items. Every item's tags
record its author (`author-andy`, `author-claude-draft`, `edited-claude`), and the three
textbook controls are tagged `likely-known`.

Two biases follow, and the write-up names both:
- A question drafted from the page that answers it leans toward what retrieval finds. The
  drafts use work language rather than the source's wording, and no question repeats more than
  seven consecutive words of its gold page; that reduces the bias without removing it.
- A question drafted by the model under test may favour facts that model already knows, which
  would narrow the gap between grounded and ungrounded correctness.

Results by author tag are reported as exploratory, beside the primary comparison and never in
place of it. Mix as drafted: table 7, standard 5, guidance 5, formula 5, judgment 4,
unanswerable 4. At 26 answerable items the power simulation (`power.py --items 26`) gives
expected 95% half-widths of ±12 points for one condition's verified rate, ±17 for the primary
difference and ±14 for correctness, wider than the ±11, ±15 and ±12 computed at 35 below.

### Story 25.6: Run and write-up

Run all four conditions, then Andy blind-grades every answer and audits the resolution sample.
Run the replicate check. Publish the summary in the README eval section, plus a one-page
write-up, following the pre-registered analysis below and only for the metrics whose gates
pass.

---

## Pre-registered analysis (approved by Andy 2026-09-14, before the first run)

Frozen before the first answer: the fixture (SHA-256 in `run.json`), the prompts (`answer-v3`
and the judge versions), the models (`claude-opus-5` answers, `claude-sonnet-5` judges) and
this section. No item is added or edited after the first run; a dropped item is reported with
its reason.

**Which answers enter the primary comparison.** Every **answerable** item, scored in both
`hybrid-rerank` and `ungrounded` (`report._item_citations`). Unanswerable items are left out:
those questions ask the model not to answer, so the citations in an answer to one measure
something else. They are reported under abstention, and their citations appear in the
per-condition table, which counts every scored answer and therefore has slightly larger
denominators than the primary comparison. An answer whose status is `refusal` or `truncated`
is not scored at all and appears only in run health.

**Primary comparison.** Verified rate of all citations, `hybrid-rerank` minus `ungrounded`: a
paired difference over items with a 95% percentile bootstrap (2000 resamples, seed 0).
Secondary: verified of checkable citations (unresolvable left out). **The bootstrap resamples
items only**: it says how far the estimate would move on another set of questions, not on
another run of the model. Model sampling is measured separately by the replicate below, and
the two are reported side by side, never combined into one interval.

**Correctness.** Primary: Andy's blind grades, and only when **every** scored answerable
answer has one. If any are missing, the report stays judge-primary, says so in a warning, and
shows the human grades beside it as the secondary grader. Human and judge grades are never
mixed into one number.

**Judge failures.** A judge call that returns nothing leaves that grade empty. An unjudged
correctness row is excluded from the judge's mean and counted (`n_unjudged`); the human grade
for that answer still stands, so human-primary correctness is unaffected. An unjudged citation
is counted (`unjudged`) and left out of both the numerator and the denominator of the verified
rate, because the scorer cannot say what it was. If more than 5% of citations in any condition
are unjudged, re-score before reading the comparison.

**Gates.** Judge correctness is published only if rubric-judge agreement is at least 80%
(n ≥ 10). The citation metric is published only if the support judge agrees with Andy on at
least 80% of the citations he graded **and** Cohen's kappa is at least 0.6 (n ≥ 10). Kappa is
the load-bearing half: most citations in a decent answer are supported, so a judge that says
"supported" to everything still agrees about 90% of the time. A kappa that is undefined (both
graders used one label) reads as `insufficient`, not as a pass.

**Metadata gate.** The comparison is read only if at most 10% of the ungrounded citations that
enter it are unresolvable for reasons a fixture edit can fix (`no_page_offset`,
`edition_unknown`, `identifier_unknown`, `work_has_no_page_index`, `no_text_at_page`,
`ambiguous_title`); otherwise fix the metadata and re-score, with no new answers. A
section-paged work is **not** in that set: "p. 5-20" has no printed page to map, so no fixture
entry makes those citations checkable, and counting them would fail the gate with nothing to
fix. Reported beside the gate, and also not fixable: how many ungrounded citations named a
work the library does not hold (`work_not_in_corpus`). That number is a finding about the
ungrounded condition, not bookkeeping.

**Exploratory:** everything else (dense vs hybrid vs hybrid-rerank, categories, abstention and
the `not_in_corpus` acknowledged-gap rate, confidently wrong, retrieval vs generation), with
intervals and no significance claims.

**Power.** `grounding/eval/answers/power.py` simulates the benchmark with the report's own
bootstrap functions (`./venv/bin/python -m grounding.eval.answers.power`). At 35 answerable
items, about 3.3 citations per answer, true verified rates of 65% (hybrid-rerank) and 35%
(ungrounded), and citations within an answer correlated (Beta concentration 4), 200 simulated
benchmarks at seed 0 give expected 95% half-widths of **±11 points** for one condition's
verified rate, **±15 points** for the primary difference, and **±12 points** for correctness.
A difference under about 15 points will be inconclusive at this sample size, and no
post-hoc item count is added to chase one.

**Replicate.** Before reporting, re-run **the whole `hybrid-rerank` condition on every item**
into a second run directory and compare with `--compare-run`. Report per-item correctness
agreement (exact agreement, kappa and mean absolute difference) and the run-to-run difference
in verified rate with its paired 95% interval. The same simulation says that a full re-run
moves the verified rate by about 6 points on average with nothing changed, while the 12-item
replicate this replaces moves it by about 10 points and exceeds 12 points a third of the time,
which is why a small replicate cannot tell noise from a real difference. The replicate is
descriptive: it is reported beside the primary result and never revises it. The write-up says
plainly when the run-to-run move is at least half the primary difference, which is the case
where the difference cannot be separated from the model's own sampling.

### Amendment 2026-09-17, after grading (Andy's decision)

Written after the first run was graded and its gates computed. It changes nothing above; it
records what was done when the support gate failed.

- **Round 1.** Andy's blind grades, plus two changes to citations he had flagged as uncertain in
  his own notes (made before any comparison with the judge), were frozen and both gates computed
  on them: correctness 89% agreement, kappa 0.67, n = 99, **pass**; support 90%, kappa 0.57,
  n = 50, **fail**.
- **Reviewed set.** The 16 rows where Andy and a judge disagreed were then reviewed with the
  judge's verdict shown, and 12 grades changed. Only disagreements were reviewed, so that set can
  only move toward the judge; it is reported as a sensitivity check and never gates.
- **Retest.** In 4 of the 5 round-1 support disagreements the claim had two parts and the grading
  screen lacked the judge's rule that support for one part is not support. The screen was
  aligned; the judge (`support-v2`) was not changed. One batch of 30 citations Andy had not seen
  was graded blind under terms written before grading: the same thresholds, one batch, no
  extension. Result: 80% agreement, kappa 0.44, **fail**.
- **Consequence.** Under **Gates**, the verified-citation rate and the primary comparison are
  not published. Correctness, whose gate passed, is the reported result, human-primary. The
  replicate was not run: its purpose was the run-to-run noise of the verified rate.
