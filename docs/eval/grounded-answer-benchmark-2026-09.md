# Grounded-answer benchmark: first run, September 2026

With a public-domain engineering library to search, Claude answered **96%** of 26 answerable
regulatory and engineering questions correctly, against **46%** without it: **+50 points**
(95% CI +31 to +69), graded by hand with the condition hidden. The headline metric that was
pre-registered, the share of citations a model judge verifies against the cited passage, is
**not reported**: the citation judge failed its validation gate twice. What follows is what the
run shows, what it does not, and why.

## Setup

- **Questions:** 30 (26 answerable, 4 that the library cannot or should not answer), in
  `benchmarks/public-corpus/questions.yaml`. Andy wrote 4; Claude drafted 26 from the
  documents and Andy vetted all 30 before the run (Epic 25, Story 25.5 amendment).
- **Library:** 25 public-domain US government documents (FDA, NASA, DOE, DoD, FAA), rebuildable
  from the manifest with SHA-256 checks ([`benchmarks/public-corpus/`](../../benchmarks/public-corpus/)).
- **Conditions:** the same questions answered with no library (`ungrounded`) and with the search
  tool over three retrieval configurations (`dense`, `hybrid`, `hybrid-rerank`).
- **Models:** `claude-opus-5` answers, `claude-sonnet-5` judges. 120 answers, 0 errors, $19.27.
- **Analysis:** written down and approved before the first answer
  ([pre-registered analysis](../epics/epic-25-grounded-answer-benchmark.md#pre-registered-analysis-approved-by-andy-2026-09-14-before-the-first-run)),
  with one dated amendment after grading (at the end of the same document).

## Result: correctness, graded blind by hand

| Condition | Correct (95% CI) | vs ungrounded (paired) | Judge's score |
|---|---|---|---|
| ungrounded | 46% (27 to 65) | baseline | 52% |
| dense | 92% (81 to 100) | +46 (+27 to +65) | 90% |
| hybrid | 92% (81 to 100) | +46 (+27 to +65) | 94% |
| hybrid-rerank | 96% (88 to 100) | **+50 (+31 to +69)** | 96% |

Andy graded every answer against a written gold answer, blind to the condition. The judge's
correctness scores agreed with his on 89% of the 99 rows it graded with its rubric (Cohen's
kappa 0.67), which passes the pre-registered gate, so they are shown beside his.

- **Where the grounded misses come from:** in every grounded condition the page holding the
  answer was retrieved for all 26 questions, so the few wrong answers (2, 2 and 1) are
  generation misses, not retrieval misses.
- **Questions the library cannot answer:** all three grounded conditions said so on 3 of 3, and
  declined the question with no source at all (1 of 1).
- **Without the library,** the judge flags 31% of answers as declined and 8% as confidently
  wrong (scored 0 without declining).
- **Citations without the library:** 91 of the ungrounded condition's 107 citations named a
  work the library does not hold, so they could not be checked at all (a count pre-registered as
  its own finding, not part of the citation metric). Andy checked that call by hand on 20 of them
  and agreed with all 20. It says nothing about whether those works are real or the claims right.

## Citation check: failed validation, not reported

The pre-registered headline was the verified-citation rate: the share of citations whose cited
passage supports the claim, as decided by a support judge. It may be published only if that
judge agrees with blind human grades on at least 80% of citations **and** reaches Cohen's kappa
of 0.6, since a judge that calls everything "supported" still agrees about 90% of the time.

1. **Round 1:** 50 citations, 90% agreement, kappa 0.57. **Fail.** In 4 of the 5 disagreements
   the claim had two parts and the passage backed one; the judge's rubric counts that as
   unsupported, and the human grading screen did not say so. The screen was aligned.
2. **Retest, pre-registered before grading:** 30 citations not seen before, one batch only,
   same thresholds. 80% agreement, kappa 0.44. **Fail.** This time 4 of the 6 disagreements were
   the judge misapplying its own rubric: it rejected a claim that quotes its passage word for
   word (twice, in two answers) and passed two claims whose passage backs only one part.

The verified-citation rate is therefore not reported, and neither is anything computed from it.

**Disclosure.** After round 1, the 16 rows where Andy and a judge disagreed were reviewed with
the judge's verdict shown, and he changed 12. Because only disagreements were reviewed, that
set can only move toward the judge, so it never enters a gate; it is reported as a sensitivity
check only. On it, the correctness gap is +46 points (+31 to +63) rather than +50.

## Limits

- **Small:** 26 answerable questions give wide intervals (about ±14 points expected for
  correctness, from the pre-registered power simulation at 26 items).
- **Question authorship,** both biases named before the run. Exploratory, by author:

  | Questions | n | ungrounded | hybrid-rerank |
  |---|---|---|---|
  | written by Andy | 4 | 0% | 100% |
  | drafted by Claude, vetted by Andy | 22 | 55% | 95% |
  | textbook controls (`likely-known`) | 3 | 100% | 100% |

  Questions drafted by the model under test were easier to answer without the library, as
  predicted. With 4 items this is a direction, not a measurement.
- **No replicate:** its pre-registered purpose was the run-to-run noise of the verified rate,
  which is not reported. The intervals above resample questions only; how far correctness moves
  between two runs of the same model is unmeasured.
- **One grader,** who also wrote or vetted the gold answers.

## Reproduce

Rebuild the library with [`benchmarks/public-corpus/`](../../benchmarks/public-corpus/), then run
and score with `grounding eval-answers` ([`docs/eval/README.md`](README.md)). The local grading
screen is `benchmarks/public-corpus/grade.py`. Run `answers-implant-eng-public-20260915-220711`,
git `0831f21`, fixture sha256 `4c128bcf6807`.
