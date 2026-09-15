"""Report tests for the grounded-answer benchmark (Epic 25, Story 25.4).

The report is built from a fake-client run over the mini corpus (answers,
recall, scoring and blind export all real code). Values asserted below are
the ones the scripted scenario in ``test_eval_answers_pipeline`` produces.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from grounding.eval.answers import cli as answers_cli
from grounding.eval.answers.report import (
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    build_report,
    publishable_view,
    render_markdown,
)
from grounding.eval.fixtures import load_fixtures
from tests.answers_fakes import (
    MINI_AGENTS_DIR,
    MINI_ANSWERS_YAML,
    MINI_CORPUS,
    FakeClient,
    FakeJudge,
    FakeSearchTool,
    ScriptedModel,
    build_stub_mini_index,
    lexical_rerank,
    mini_chunks,
    read_jsonl,
    stub_run_eval,
)
from tests.test_eval_answers_pipeline import UNGROUNDED, _parse, _run_dir, grounded_answer


@pytest.fixture
def scored_run(tmp_path, monkeypatch):
    """A scored run over the mini corpus that needs neither mcp nor a model.

    The same scripted scenario as the pipeline tests, but the search tool is
    FakeSearchTool, so this runs in CI where the optional mcp extra is absent.
    """
    pytest.importorskip("faiss")
    monkeypatch.setattr("grounding.reranker.rerank", lexical_rerank)
    index = build_stub_mini_index(tmp_path / "embeddings")
    judge = FakeJudge(
        supported=lambda claim, passage: "falsifiab" in claim.lower() and "falsifiab" in passage.lower(),
        correctness=lambda candidate: ("1", False) if any(
            key in candidate.lower() for key in ("fals", "discretize", "read off the percentiles")
        ) else ("0.5", False),
        declined=lambda candidate: "does not contain" in candidate,
    )
    client = FakeClient(
        ScriptedModel(ungrounded_answers=UNGROUNDED, grounded_answer=grounded_answer, judge=judge)
    )
    argv = [
        "--agent", "mini", "--agents-dir", str(MINI_AGENTS_DIR),
        "--fixtures", str(MINI_ANSWERS_YAML), "--corpus", str(MINI_CORPUS),
        "--embeddings", str(index), "--out", str(tmp_path / "out"), "--publishable",
    ]
    code = answers_cli.eval_answers_command(
        _parse(argv), client=client, run_eval_fn=stub_run_eval,
        tool_factory=FakeSearchTool, env={},
    )
    assert code == answers_cli.EXIT_OK
    return _run_dir(tmp_path / "out")


# ---------------------------------------------------------------------------
# The report builds from the fake-client run
# ---------------------------------------------------------------------------

def test_report_builds_from_the_fake_client_run(scored_run):
    for name in ("report.md", "report.json", "chart.png"):
        assert (scored_run / name).stat().st_size > 0
    assert (scored_run / "chart.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    report = json.loads((scored_run / "report.json").read_text())
    conds = report["conditions"]
    assert report["run"]["conditions"] == ["ungrounded", "dense", "hybrid", "hybrid-rerank"]

    # Scripted scenario: ungrounded scores 1, 0.5, 1; grounded answers score 1.
    assert conds["ungrounded"]["correctness"]["mean"] == pytest.approx(2.5 / 3)
    assert conds["dense"]["correctness"]["mean"] == 1.0
    delta = conds["dense"]["correctness"]["delta_vs_ungrounded"]
    assert delta["n_paired"] == 3 and delta["mean"] == pytest.approx(0.5 / 3)
    assert conds["ungrounded"]["correctness"]["delta_vs_ungrounded"] is None

    assert {b: conds["ungrounded"]["citations"][b] for b in
            ("verified", "unsupported", "invented", "unresolvable")} == {
        "verified": 1, "unsupported": 0, "invented": 0, "unresolvable": 1,
    }
    assert conds["hybrid"]["citations"]["verified"] == 1
    assert conds["hybrid"]["citations"]["unsupported"] == 2
    assert conds["hybrid"]["citations"]["verified_rate"] == pytest.approx(1 / 3)

    # ans-004 is not_in_corpus: never scored for the ungrounded condition.
    assert conds["ungrounded"]["abstention"]["n"] == 0
    assert conds["ungrounded"]["n_excluded"] == 1
    assert conds["dense"]["abstention"]["by_kind"]["not_in_corpus"] == {
        "desired_behavior": "acknowledged_gap", "met": 1, "n": 1, "rate": 1.0, "ci95": None,
        "declined": 1, "acknowledged_gap": 1, "answered_with_labeled_knowledge": 0,
        "answered_without_flagging": 0,
    }
    assert conds["dense"]["abstention"]["rate"] == 1.0
    assert conds["dense"]["retrieval"]["recall_at_5_gold_page"] == 1.0
    assert conds["ungrounded"]["retrieval"] is None
    assert conds["dense"]["cost_usd"]["answer"] > 0
    assert conds["dense"]["tool_calls_per_answer"] == 1.0

    # Per-category breakdown and worst failures by item id.
    assert "ungrounded" not in report["per_category"]["unanswerable"]  # excluded, not scored
    assert report["per_category"]["unanswerable"]["dense"]["abstention"] == 1.0
    assert report["per_category"]["formula"]["ungrounded"]["correctness"] == 0.5
    worst = report["worst_failures"]
    # The ungrounded answer to ans-004 (not_in_corpus) is excluded, so it is not a failure.
    assert not any(w["item_id"] == "ans-004" for w in worst)
    assert (worst[0]["item_id"], worst[0]["condition"]) == ("ans-002", "ungrounded")
    assert worst[0]["why"] == "correctness scored 0.5 (judge)"
    assert (worst[1]["item_id"], worst[1]["condition"]) == ("ans-002", "dense")

    md = (scored_run / "report.md").read_text()
    assert "## Results by condition" in md and "## Worst failures" in md
    assert "![Correctness and verified-citation rate by condition](chart.png)" in md
    assert "| hybrid | 1 | 0 | 2 | 0 | 0 | 3 | 0 | 3 of 4 |" in md
    assert "Self-preference is possible" not in md  # the judge defaults to a different model


def test_self_preference_warning_only_when_the_judge_is_the_answer_model(scored_run):
    assert not any("Self-preference" in w for w in build_report(scored_run)["warnings"])
    manifest_path = scored_run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["judge_model"] = manifest["answer_model"]
    manifest.setdefault("scoring", {})["judge_model"] = manifest["answer_model"]
    manifest_path.write_text(json.dumps(manifest))
    assert any("Self-preference is possible" in w for w in build_report(scored_run)["warnings"])


def test_report_shows_judge_agreement_after_import(scored_run):
    (scored_run / "agreement.json").write_text(json.dumps({
        "correctness": {"n": 15, "agreement": 0.8, "cohens_kappa": 0.5, "by_method": {
            "judge": {"n": 10, "agreement": 0.7, "cohens_kappa": 0.2},
            "numeric": {"n": 5, "agreement": 1.0, "cohens_kappa": None},
        }},
        "abstention": {"n": 1, "agreement": 1.0, "cohens_kappa": None,
                       "by_kind": {"not_in_corpus": {"n": 1, "agreement": 1.0, "cohens_kappa": None}}},
        "support": {"n": 14, "agreement": 0.93, "cohens_kappa": 0.45},
        "publish_gate": {"threshold": 0.8, "min_n": 10, "n": 10, "agreement": 0.7,
                         "metric": "correctness agreement on rubric-judge rows", "result": "fail"},
        "support_gate": {"threshold": 0.8, "min_n": 10, "min_kappa": 0.6, "n": 14,
                         "agreement": 0.93, "cohens_kappa": 0.45,
                         "metric": "support agreement and kappa on human-graded citations",
                         "result": "fail"},
    }))
    code = answers_cli.eval_answers_command(_parse(["--run-dir", str(scored_run), "--report"]), env={})
    assert code == answers_cli.EXIT_OK
    md = (scored_run / "report.md").read_text()
    # Both figures are shown; only the rubric-judge rows decide the gate.
    assert "| correctness, all rows | 15 | 80% | 0.50 |" in md
    assert "| correctness, rubric-judge rows (gate) | 10 | 70% | 0.20 |" in md
    assert "| abstention, not_in_corpus | 1 | 100% | n/a |" in md
    assert "**fail** (n=10)" in md
    # The citation metric is gated on kappa as well as raw agreement.
    assert "Cohen's kappa at least 0.6" in md
    assert "**fail** (n=14, agreement 93%, kappa 0.45)" in md


# ---------------------------------------------------------------------------
# Publishable mode carries no source text
# ---------------------------------------------------------------------------

def _fragments(text: str, size: int = 24) -> list[str]:
    """Distinctive pieces of a text: every sentence clause of at least ``size`` chars."""
    pieces = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    return [p.strip()[:size] for p in pieces if len(p.strip()) >= size]


def test_publishable_output_contains_no_chunk_text(scored_run):
    pub = scored_run / "publishable"
    published = {p.name: p.read_text() for p in pub.iterdir() if p.suffix in (".md", ".json")}
    assert set(published) == {"report.md", "report.json"}
    assert (pub / "chart.png").exists()

    transcripts = read_jsonl(scored_run / "transcripts.jsonl")
    scores = read_jsonl(scored_run / "scores.jsonl")
    secrets: set[str] = set()
    for chunk in mini_chunks():  # every chunk body in the corpus
        secrets.update(_fragments(chunk["body"]))
    for row in transcripts:  # everything the tool returned and the model wrote
        secrets.update(_fragments(row["final_text"]))
        for call in row["tool_calls"]:
            secrets.update(_fragments(call["output_text"]))
            for result in call["results"]:
                secrets.update(_fragments(result["content"]))
                secrets.add(result["prefix"])
    for score in scores:
        for cit in score["citations"]:
            secrets.update(_fragments(cit.get("passage", "")))
            secrets.add(cit["text"])
            if len(cit.get("claim") or "") >= 12:
                secrets.add(cit["claim"])
    fixture = load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR)
    for item in fixture.items:  # gold answers and (by default) question texts
        secrets.update(_fragments(item.answer.gold))
        secrets.add(item.query)
    assert len(secrets) > 30

    # Positive control: the raw run files do contain these strings.
    raw = (scored_run / "transcripts.jsonl").read_text()
    assert sum(1 for s in secrets if s in raw) > 20

    for name, text in published.items():
        leaked = sorted(s for s in secrets if s and s in text)
        assert leaked == [], f"{name} leaks source or answer text: {leaked[:3]}"
        assert str(scored_run.parent.parent) not in text  # no local paths
        assert "judge_reason" not in text and "scripted" not in text


def test_source_text_is_published_only_for_a_public_domain_corpus(scored_run):
    argv = ["--run-dir", str(scored_run), "--report", "--publishable", "--include-source-text"]
    pub = scored_run / "publishable"
    # The mini fixture declares no source_license: refused, and nothing is written.
    assert answers_cli.eval_answers_command(_parse(argv), env={}) == answers_cli.EXIT_BAD_INPUT
    assert not (pub / "transcripts.jsonl").exists()

    manifest_path = scored_run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fixture"]["source_license"] = "public_domain"
    manifest_path.write_text(json.dumps(manifest))
    assert answers_cli.eval_answers_command(_parse(argv), env={}) == answers_cli.EXIT_OK

    for name in ("transcripts.jsonl", "scores.jsonl", "blind/answers.csv", "blind/citations.csv"):
        assert (pub / name).exists(), name
    transcripts = (pub / "transcripts.jsonl").read_text()
    assert mini_chunks()[0]["body"][:40] in transcripts  # the passages are published
    assert str(scored_run.parent.parent) not in transcripts  # but never a local path
    # An exception message can carry a path, so the error log is never copied.
    assert not (pub / "errors.jsonl").exists()
    md = (pub / "report.md").read_text()
    assert "public-domain corpus run" in md
    questions = [it.query for it in load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR).items]
    assert all(q in md for q in questions)


def test_include_source_text_is_refused_without_publishable(scored_run):
    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(scored_run), "--report", "--include-source-text"]), env={}
    )
    assert code == answers_cli.EXIT_BAD_INPUT


def test_publishable_questions_only_on_request(scored_run):
    questions = [it.query for it in load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR).items]
    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(scored_run), "--report", "--publishable", "--include-questions"]),
        env={},
    )
    assert code == answers_cli.EXIT_OK
    md = (scored_run / "publishable" / "report.md").read_text()
    assert all(q in md for q in questions)
    chunk_text = mini_chunks()[0]["body"][:40]
    assert chunk_text not in md


# ---------------------------------------------------------------------------
# Statistics and rendering edge cases
# ---------------------------------------------------------------------------

def test_the_power_simulation_runs_on_the_reports_own_bootstrap():
    from grounding.eval.answers import power

    result = power.simulate(items=8, sims=3, seed=0, n_boot=200)
    assert set(result.half_widths) == {
        "verified rate of one condition",
        "primary difference (hybrid-rerank minus ungrounded)",
        "correctness of one condition",
    }
    assert all(0 < width < 1 for width in result.half_widths.values())
    # A 12-item replicate moves more than a full re-run of the same items.
    assert (result.replicate["mean |difference|, 12-item replicate"]
            >= result.replicate["mean |difference|, full re-run of 8 items"] * 0.5)
    rendered = result.render()
    assert "Power simulation: 8 answerable items" in rendered
    assert "expected 95% half-width" in rendered
    # Reproducible for a seed, which is what lets the epic quote the numbers.
    assert power.simulate(items=8, sims=3, seed=0, n_boot=200).half_widths == result.half_widths
    assert power.main(["--items", "6", "--sims", "2", "--n-boot", "100"]) == 0


def test_bootstrap_intervals_are_reproducible_and_bracket_the_estimate():
    values = [1, 1, 0.5, 0, 1, 1, 0, 1, 0.5, 1]
    ci = bootstrap_mean_ci(values)
    assert ci == bootstrap_mean_ci(values)
    assert ci[0] <= sum(values) / len(values) <= ci[1]
    assert bootstrap_mean_ci([1.0]) is None
    ratio = bootstrap_ratio_ci([(1, 2), (3, 3), (0, 1), (2, 4)])
    assert ratio[0] <= 6 / 10 <= ratio[1]


def test_markdown_renders_conditions_without_citations_or_scores(tmp_path):
    report = {
        "run": {"run_id": "r", "status": "complete", "agent": "a", "answer_model": "m",
                "judge_model": "j", "n_items": 1, "n_unanswerable": 0,
                "conditions": ["ungrounded"], "git": {}, "fixture": {}, "prompts": {}},
        "statistics": {"ci": "ci", "n_boot": 1, "seed": 0},
        "conditions": {"ungrounded": {
            "n_answers": 1, "n_scored": 1, "n_errors": 0, "n_judge_errors": 0, "n_refusal": 0,
            "n_truncated": 0, "n_max_iterations": 0,
            "correctness": {"mean": None, "ci95": None, "n": 0, "delta_vs_ungrounded": None},
            "citations": {"verified": 0, "unsupported": 0, "invented": 0, "unresolvable": 0,
                          "total": 0, "unjudged": 0, "verified_rate": None, "partial_matches": 2,
                          "verified_rate_ci95": None, "per_answer": 0.0, "answers_with_citations": 0},
            "abstention": {"rate": None, "ci95": None, "n": 0},
            "retrieval": None,
            "tokens": {"answer": {"input_tokens": 0, "output_tokens": 0}, "judge": {}},
            "cost_usd": {"answer": 0.0, "judge": 0.0, "per_answer": 0.0},
            "latency_s": {"median": None, "p90": None},
            "tool_calls_per_answer": 0.0,
        }},
        "per_category": {}, "worst_failures": [], "judge_validation": None,
        "questions": {},
        "warnings": ["Retrieval recall failed: FileNotFoundError: /Users/someone/corpus/_x.json"],
    }
    md = render_markdown(report, chart_name=None)
    assert "| ungrounded | n/a n=0 | baseline | n/a of 0 |" in md
    assert "never counted as verified): ungrounded 2." in md
    assert "/Users/someone" in md  # the local report keeps the detail
    pub = render_markdown(publishable_view(report), chart_name=None)
    assert "Publishable summary" in pub
    assert "/Users/someone" not in pub
    assert "Retrieval recall failed (details in the full report)." in pub
    from grounding.eval.answers.report import render_chart

    assert render_chart(report, tmp_path / "c.png") is True
    assert (tmp_path / "c.png").stat().st_size > 1000


def test_small_samples_render_as_counts_not_bare_rates(scored_run):
    from grounding.eval.answers.report import _count_rate

    md = (scored_run / "report.md").read_text()
    # One not_in_corpus answer per grounded condition: "1/1", never "100%".
    assert "not in corpus acknowledged gap 1/1" in md
    # The desired behavior on those questions gets its own section.
    assert "## Questions the library cannot answer (not_in_corpus)" in md
    assert "| dense | 1/1 | 1 | 0 | 0 |" in md
    assert _count_rate(2, 3) == "2/3"
    assert _count_rate(9, 12, [0.5, 0.9]) == "75% (50 to 90) n=12"
    assert _count_rate(0, 0) == "n/a"


def test_build_report_needs_only_the_run_directory(scored_run):
    report = build_report(scored_run)
    assert report["schema"] == "grounding-answer-report/2"
    assert report["run"]["index_fingerprint"]
    assert report["statistics"]["seed"] == 0


# ---------------------------------------------------------------------------
# Synthetic runs: primary comparison, human-primary correctness, confidently
# wrong, attribution, and both citation denominators
# ---------------------------------------------------------------------------

BUCKET_NAMES = ("verified", "partial", "unsupported", "invented", "unresolvable")


def _citations(spec: dict) -> list[dict]:
    """Citation entries matching bucket counts; unresolvable ones carry a detail."""
    out = []
    for bucket, detail, n in spec:
        out += [{"cite_id": f"c{len(out) + k + 1}", "bucket": bucket, "claim": "one two three",
                 "resolution": {"detail": detail, "match": None, "notes": []}} for k in range(n)]
    return out


def _row(item, cond, *, score=1.0, declined=False, method="judge", cites=(), gold=None,
         answerable=True):
    counts = {b: 0 for b in BUCKET_NAMES}
    for bucket, _, n in cites:
        counts[bucket] += n
    return {
        "item_id": item, "condition": cond, "scored": True, "answerable": answerable,
        "category": "table" if answerable else "unanswerable", "excluded": None,
        "unanswerable_kind": None if answerable else "no_source",
        "correctness": {"score": score, "declined": declined, "method": method} if answerable else None,
        "abstention": None if answerable else {"declined": declined, "method": "judge"},
        "bucket_counts": counts, "citations": _citations(cites), "n_unjudged_citations": 0,
        "gold_page_in_context": gold, "judge_calls": [],
    }


def _synthetic_run(tmp_path: Path, rows: list[dict], human: dict | None = None) -> Path:
    run_dir = tmp_path / "synthetic"
    run_dir.mkdir(parents=True)
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": "synthetic-1", "status": "complete", "agent": "mini",
        "answer_model": "claude-opus-5", "judge_model": "claude-sonnet-5", "max_iterations": 5,
        "conditions": [{"name": c, "grounded": c != "ungrounded"} for c in conditions],
        "fixture": {"selected_item_ids": sorted({r["item_id"] for r in rows}), "sha256": "abc"},
    }))
    (run_dir / "transcripts.jsonl").write_text("".join(
        json.dumps({"item_id": r["item_id"], "condition": r["condition"], "status": "ok",
                    "question": "q", "tool_calls": [], "latency_s": 1.0}) + "\n" for r in rows))
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if human is not None:
        (run_dir / "human_grades.json").write_text(json.dumps({"answers": [
            {"item_id": i, "condition": c, "answerable": True, "human_grade": g}
            for (i, c), g in human.items()]}))
    return run_dir


def _primary_rows() -> list[dict]:
    rows = []
    for i in range(12):
        item = f"q{i:02d}"
        # hybrid-rerank: 2 verified + 1 unsupported per answer, one item with 3 verified.
        rows.append(_row(item, "hybrid-rerank", gold=i % 3 != 0, score=1.0 if i % 4 else 0.0,
                         cites=[("verified", None, 3 if i == 0 else 2), ("unsupported", None, 1)]))
        # ungrounded: 1 verified, 1 invented, and two unresolvable citations: one
        # the fixture could fix (a missing page offset) and one it cannot (a
        # section-paged work). The first item also cites a work the library lacks.
        cites = [("verified", None, 1), ("invented", "page_not_in_work", 1),
                 ("unresolvable", "no_page_offset", 1), ("unresolvable", "section_paged", 1)]
        if i == 0:
            cites.append(("unresolvable", "work_not_in_corpus", 1))
        rows.append(_row(item, "ungrounded", score=0.5, cites=cites))
    # An unanswerable item, answered with citations in both conditions: it is
    # scored for abstention, and never enters the primary comparison.
    for condition in ("hybrid-rerank", "ungrounded"):
        rows.append(_row("u-01", condition, answerable=False, declined=True,
                         cites=[("verified", None, 5)]))
    return rows


def test_primary_comparison_is_the_paired_verified_rate_difference(tmp_path):
    report = build_report(_synthetic_run(tmp_path, _primary_rows()))
    pc = report["primary_comparison"]
    # 12 answerable items; the unanswerable one is not in the comparison, and
    # neither are the 5 citations in each condition's answer to it.
    assert pc["conditions"] == ["hybrid-rerank", "ungrounded"] and pc["n_items"] == 12
    assert pc["items"] == "answerable only"
    assert pc["hybrid-rerank"] == {"verified": 25, "citations": 37, "rate": 25 / 37}
    assert pc["ungrounded"] == {"verified": 12, "citations": 49, "rate": 12 / 49}
    assert pc["estimate"] == pytest.approx(25 / 37 - 12 / 49)
    assert pc["ci95"][0] <= pc["estimate"] <= pc["ci95"][1]
    # The per-condition table counts every scored answer, so its denominators differ.
    assert report["conditions"]["hybrid-rerank"]["citations"]["total"] == 42
    # Secondary: unresolvable left out of the denominator.
    assert pc["secondary_verified_of_checkable"]["estimate"] == pytest.approx(25 / 37 - 12 / 24)
    # Only the fixable reason counts toward the metadata gate: a section-paged
    # work cannot be fixed by any fixture entry, and neither can a work the
    # library does not hold, which is reported beside the gate instead.
    assert pc["bookkeeping_share_ungrounded"] == pytest.approx(12 / 49)
    assert pc["bookkeeping_by_detail_ungrounded"] == {"no_page_offset": 12}
    assert pc["work_not_in_corpus_ungrounded"] == 1
    assert pc["unresolvable_ungrounded"] == 25
    assert pc["bookkeeping_ok"] is False
    assert any("fixture-metadata reasons" in w for w in report["warnings"])
    cit = report["conditions"]["ungrounded"]["citations"]
    assert cit["unresolvable_by_detail"] == {"no_page_offset": 12, "section_paged": 12,
                                             "work_not_in_corpus": 1}
    md = render_markdown(report, chart_name=None)
    assert "## Primary comparison (pre-registered)" in md
    assert "**hybrid-rerank minus ungrounded**: **+43 pts**" in md
    assert "paired over 12 answerable items" in md
    assert "1 ungrounded citation(s) name a work that is not in the library" in md
    assert "Unresolvable means \"could not be checked\", not \"wrong\"." in md


def test_confidently_wrong_counts_zeros_that_did_not_decline(tmp_path):
    rows = [
        _row("a", "ungrounded", score=0.0, declined=False),  # confidently wrong
        _row("b", "ungrounded", score=0.0, declined=True),  # declined: not confident
        _row("c", "ungrounded", score=0.0, declined=False, method="numeric"),  # wrong number
        _row("d", "ungrounded", score=0.0, declined=False, method="empty_answer"),  # nothing said
        _row("e", "ungrounded", score=0.5, declined=False),
        _row("f", "ungrounded", score=1.0, declined=False),
    ]
    corr = build_report(_synthetic_run(tmp_path, rows))["conditions"]["ungrounded"]["correctness"]
    assert (corr["n_confidently_wrong"], corr["n_declined"], corr["n_with_declined_flag"]) == (2, 1, 6)
    assert corr["confidently_wrong_rate"] == pytest.approx(2 / 6)
    assert corr["declined_rate"] == pytest.approx(1 / 6)


def test_attribution_splits_wrong_grounded_answers_by_gold_page_in_context(tmp_path):
    rows = [
        _row("a", "hybrid-rerank", score=0.0, gold=False),  # retrieval miss
        _row("b", "hybrid-rerank", score=0.0, gold=True),  # generation miss
        _row("c", "hybrid-rerank", score=0.0, gold=True),
        _row("d", "hybrid-rerank", score=0.5, gold=False),
        _row("e", "hybrid-rerank", score=1.0, gold=True),
        _row("a", "ungrounded", score=0.0),
    ]
    report = build_report(_synthetic_run(tmp_path, rows))
    at = report["conditions"]["hybrid-rerank"]["attribution"]
    assert at["n_graded"] == 5 and at["n_gold_page_in_context"] == 3
    assert at["wrong"] == {"gold_not_in_context": 1, "gold_in_context": 2, "unknown": 0}
    assert at["partial"] == {"gold_not_in_context": 1, "gold_in_context": 0, "unknown": 0}
    assert report["conditions"]["ungrounded"]["attribution"] is None
    assert "## Retrieval versus generation (grounded conditions)" in render_markdown(report, chart_name=None)


def test_human_grades_become_primary_correctness_only_with_full_coverage(tmp_path):
    rows = [_row(f"q{i}", c, score=1.0) for i in range(4) for c in ("ungrounded", "hybrid-rerank")]
    everything = {(r["item_id"], r["condition"]): 0.5 for r in rows}
    full = build_report(_synthetic_run(tmp_path / "full", rows, human=everything))
    assert full["primary_correctness"] == "human"
    st = full["conditions"]["hybrid-rerank"]
    assert st["primary_correctness"]["source"] == "human"
    assert st["primary_correctness"]["mean"] == 0.5 and st["correctness"]["mean"] == 1.0
    assert full["human_coverage"]["answerable"] == {"graded": 8, "eligible": 8}
    md = render_markdown(full, chart_name=None)
    assert "| Condition | Correctness, human (95% CI) |" in md
    assert "100% n=4 (judge)" in md  # the judge as the secondary grader

    partial = dict(list(everything.items())[:5])
    some = build_report(_synthetic_run(tmp_path / "some", rows, human=partial))
    assert some["primary_correctness"] == "judge"
    assert some["conditions"]["hybrid-rerank"]["primary_correctness"]["mean"] == 1.0
    assert "n=" in render_markdown(some, chart_name=None).split("| hybrid-rerank |")[1].split("(human)")[0]


# ---------------------------------------------------------------------------
# Human grading round trip (every answer graded) and the replicate check
# ---------------------------------------------------------------------------

def _rewrite_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    import csv

    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _grade_everything(run_dir: Path, *, column: str = "human_grade", disagree: int = 1) -> str:
    """Fill every answer row like the judge did, except the first `disagree` answerable ones."""
    blind = run_dir / "blind"
    key = json.loads((blind / "key.json").read_text())["samples"]
    rows = _read_csv(blind / "answers.csv")
    changed = None
    for row in rows:
        entry = key[row["sample_id"]]
        if entry["answerable"]:
            grade = entry["grader_correctness"]
            if disagree and grade == 1.0:
                grade, disagree, changed = 0.0, disagree - 1, entry["condition"]
            row[column] = f"{grade:g}"
        else:
            row[column] = "1" if entry["grader_declined"] else "0"
    if column != "human_grade":
        for row in rows:
            row.pop("human_grade", None)
    _rewrite_csv(blind / "answers.csv", rows)
    return changed


def test_grading_every_answer_makes_human_correctness_primary(scored_run, capsys):
    blind = scored_run / "blind"
    assert len(_read_csv(blind / "answers.csv")) == 15  # default export: every scored answer
    changed = _grade_everything(scored_run)
    audit = _read_csv(blind / "resolution_audit.csv")
    # The ungrounded Efron citation is unresolvable (not in the corpus): audited.
    assert [(r["auto_bucket"], r["auto_detail"]) for r in audit] == [("unresolvable", "work_not_in_corpus")]
    for row in audit:
        row["human_agrees"] = "y"
    _rewrite_csv(blind / "resolution_audit.csv", audit)

    code = answers_cli.eval_answers_command(_parse(["--run-dir", str(scored_run), "--import-grades"]), env={})
    assert code == answers_cli.EXIT_OK
    assert "15 answers graded" in capsys.readouterr().out
    agreement = json.loads((scored_run / "agreement.json").read_text())
    assert agreement["correctness"]["n"] == 12
    assert agreement["publish_gate"]["n"] == 12  # every mini answer went to the rubric judge
    assert agreement["correctness"]["agreement"] == pytest.approx(11 / 12)
    assert agreement["resolution_audit"] == {"n": 1, "agreement": 1.0,
                                            "by_bucket": {"unresolvable": {"n": 1, "agreement": 1.0}}}
    assert json.loads((scored_run / "human_grades.json").read_text())["n"] == 15

    report = json.loads((scored_run / "report.json").read_text())
    assert report["primary_correctness"] == "human"
    for condition, st in report["conditions"].items():
        primary, judge = st["primary_correctness"], st["correctness"]
        assert primary["source"] == "human"
        if condition == changed:
            assert primary["mean"] < judge["mean"]
        else:
            assert primary["mean"] == judge["mean"]
    md = (scored_run / "report.md").read_text()
    assert "| Condition | Correctness, human (95% CI) |" in md
    assert "| correctness, rubric-judge rows (gate) | 12 | 92% |" in md
    assert "Primary correctness source: **human**." in md


def test_old_andy_columns_still_import(scored_run):
    _grade_everything(scored_run, column="andy_grade", disagree=0)
    header = _read_csv(scored_run / "blind" / "answers.csv")[0]
    assert "andy_grade" in header and "human_grade" not in header
    code = answers_cli.eval_answers_command(_parse(["--run-dir", str(scored_run), "--import-grades"]), env={})
    assert code == answers_cli.EXIT_OK
    assert json.loads((scored_run / "agreement.json").read_text())["correctness"]["agreement"] == 1.0


def test_reexporting_archives_filled_grades_and_costs_nothing(scored_run, capsys):
    _grade_everything(scored_run)
    filled = (scored_run / "blind" / "answers.csv").read_text()
    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(scored_run), "--export-blind", "--blind-fraction", "0.5",
                "--agents-dir", str(MINI_AGENTS_DIR)]),
        env={},  # no client and no key: re-exporting makes no judge calls
    )
    assert code == answers_cli.EXIT_OK
    (archive,) = (scored_run / "blind").glob("archive-*")
    assert (archive / "answers.csv").read_text() == filled
    assert (archive / "key.json").exists()
    fresh = _read_csv(scored_run / "blind" / "answers.csv")
    assert fresh and all(not r["human_grade"] for r in fresh)
    assert "previous grades archived" in capsys.readouterr().out


def test_blind_actions_need_a_run_dir_and_a_valid_fraction(tmp_path):
    assert answers_cli.eval_answers_command(_parse(["--export-blind"]), env={}) == answers_cli.EXIT_BAD_INPUT
    assert answers_cli.eval_answers_command(
        _parse(["--run-dir", str(tmp_path), "--report", "--blind-fraction", "0"]), env={}
    ) == answers_cli.EXIT_BAD_INPUT


def _replicate_run(scored_run: Path, argv_extra: list[str], judge_model=None) -> Path:
    manifest = json.loads((scored_run / "run.json").read_text())
    judge = FakeJudge(
        supported=lambda claim, passage: "falsifiab" in claim.lower() and "falsifiab" in passage.lower(),
        correctness=lambda candidate: ("1", False) if any(
            key in candidate.lower() for key in ("fals", "discretize", "read off the percentiles")
        ) else ("0.5", False),
        declined=lambda candidate: "does not contain" in candidate,
    )
    client = FakeClient(ScriptedModel(ungrounded_answers=UNGROUNDED, grounded_answer=grounded_answer,
                                      judge=judge))
    before = set(scored_run.parent.iterdir())
    argv = ["--agent", "mini", "--agents-dir", str(MINI_AGENTS_DIR),
            "--fixtures", str(MINI_ANSWERS_YAML), "--corpus", manifest["corpus_dir"],
            "--embeddings", manifest["embeddings_dir"], "--out", str(scored_run.parent), *argv_extra]
    code = answers_cli.eval_answers_command(_parse(argv), client=client, run_eval_fn=stub_run_eval,
                                            tool_factory=FakeSearchTool, env={})
    assert code == answers_cli.EXIT_OK
    (new_dir,) = set(scored_run.parent.iterdir()) - before
    return new_dir


def test_replicate_check_reports_run_to_run_agreement(scored_run, capsys):
    replicate = _replicate_run(
        scored_run, ["--conditions", "hybrid-rerank", "--items", "ans-001,ans-002,ans-004"]
    )
    assert replicate != scored_run  # never appended to the first run's directory
    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(scored_run), "--compare-run", str(replicate)]), env={}
    )
    assert code == answers_cli.EXIT_OK
    assert "replicate check:" in capsys.readouterr().out
    rep_id = json.loads((replicate / "run.json").read_text())["run_id"]
    result = json.loads((scored_run / f"replicate-{rep_id}.json").read_text())
    assert result["comparable"] is True and list(result["conditions"]) == ["hybrid-rerank"]
    st = result["conditions"]["hybrid-rerank"]
    assert st["n_items"] == 3
    # The scripted model is deterministic, so the replicate agrees exactly.
    assert st["correctness"] == {"n": 2, "exact_agreement": 1.0, "cohens_kappa": None,
                                 "mean_abs_diff": 0.0, "mean_a": 1.0, "mean_b": 1.0}
    assert st["verified_of_all"]["diff"] == 0.0
    # The difference in verified rate carries a paired interval over the shared items.
    assert st["verified_of_all"]["diff_ci95"] == [0.0, 0.0]
    # This replicate re-ran 3 of the run's 4 items, so it is flagged as partial.
    assert (st["n_items_a"], st["covers_all_items"]) == (4, False)
    md = (scored_run / "report.md").read_text()
    assert "## Replicate agreement" in md
    assert f"| {rep_id} | hybrid-rerank | 3 of 4 | 2/2 |" in md
    assert "read it as indicative only" in md


def test_a_replicate_with_different_provenance_is_flagged(scored_run):
    replicate = _replicate_run(scored_run, ["--conditions", "dense", "--items", "ans-001"])
    manifest = json.loads((replicate / "run.json").read_text())
    manifest["answer_model"] = "claude-sonnet-5"
    (replicate / "run.json").write_text(json.dumps(manifest))
    answers_cli.eval_answers_command(
        _parse(["--run-dir", str(scored_run), "--compare-run", str(replicate)]), env={}
    )
    report = build_report(scored_run)
    (rep,) = report["replicates"]
    assert rep["comparable"] is False and rep["differences"] == ["answer_model"]
    assert any("not a pure replicate" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Provenance, per-category n, tool description
# ---------------------------------------------------------------------------

def test_run_records_the_embedding_models_and_library_versions(scored_run):
    manifest = json.loads((scored_run / "run.json").read_text())
    embedding = manifest["environment"]["embedding"]
    assert embedding["query_model"] == embedding["index_model"] == "all-MiniLM-L6-v2"
    assert embedding["sentence_transformers"]  # the installed version string
    md = (scored_run / "report.md").read_text()
    assert "Embedding model: `all-MiniLM-L6-v2` for queries" in md
    # A mismatch between the query and index models is flagged.
    embedding["query_model"] = "bge-small-en"
    (scored_run / "run.json").write_text(json.dumps(manifest))
    assert any("not comparable" in w for w in build_report(scored_run)["warnings"])


def test_module_constants_are_read_without_importing(tmp_path):
    from grounding.eval.answers.provenance import _module_constant

    source = tmp_path / "m.py"
    source.write_text('import torch  # never executed\nEMBEDDING_MODEL = "some-model"\nOTHER = 3\n')
    assert _module_constant(source, "EMBEDDING_MODEL") == "some-model"
    assert _module_constant(source, "OTHER") is None
    assert _module_constant(tmp_path / "missing.py", "EMBEDDING_MODEL") is None


def test_per_category_rows_lead_with_n_and_use_counts_when_small(scored_run):
    md = (scored_run / "report.md").read_text()
    assert "| unanswerable | dense | **n=1** | n/a | n/a | 1/1 |" in md
    assert "| formula | ungrounded | **n=1** | 50% (n=1) | 0/1 | n/a |" in md


def test_tool_description_names_no_retrieval_method():
    from grounding.eval.answers.tool import TOOL_DESCRIPTION

    assert "semantic" not in TOOL_DESCRIPTION.lower()
    assert TOOL_DESCRIPTION.startswith("Search the agent's corpus for relevant documents.")
