"""End-to-end ``grounding eval-answers`` runs with a fake client (Epic 25).

Answers, retrieval recall, scoring and the blind-grade round trip all run
through the CLI handler over the mini corpus, with the real in-process
``search_corpus`` (stub embedder and reranker) and a scripted judge.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from grounding.eval.answers import cli as answers_cli
from grounding.eval.answers.tool import load_mcp_server_module
from tests.answers_fakes import (
    MINI_AGENTS_DIR,
    MINI_ANSWERS_YAML,
    MINI_CORPUS,
    FakeClient,
    FakeJudge,
    ScriptedModel,
    StubEncoder,
    build_stub_mini_index,
    lexical_rerank,
    read_jsonl,
    stub_run_eval,
)

UNGROUNDED = {
    # Verified: printed p. 10 of the gamma notes maps to PDF p. 12.
    "According to Popper, what criterion separates a scientific theory from pseudoscience?":
        "Falsifiability separates science from pseudoscience (Gamma Notes, 1st ed., p. 10).",
    # Unresolvable: the book is not in the corpus.
    "How is a percentile bootstrap confidence interval for a sample mean constructed?":
        "Resample with replacement and take percentiles (Efron, An Introduction to the Bootstrap, 1st ed., p. 170).",
    # No citation at all.
    "Why are lattice formulations useful in quantum field theory?":
        "They discretize spacetime so simulations can probe non-perturbative regimes.",
    # Answers an unanswerable question.
    "What is the minimum yield strength of ASTM A36 structural steel?":
        "It is 250 MPa (36 ksi).",
}


GROUNDED = {
    "According to Popper": "Falsifiability is what separates science from pseudoscience {p}.",
    "How is a percentile": "Resample with replacement and read off the percentiles {p}.",
    "Why are lattice": "Lattices discretize spacetime for non-perturbative simulations {p}.",
    "What is the minimum": "The library does not contain the yield strength of A36 steel.",
}


def grounded_answer(question: str, prefixes: list[str]) -> str:
    template = next(v for k, v in GROUNDED.items() if question.startswith(k))
    return template.format(p=prefixes[0] if prefixes else "")


def _parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    answers_cli._create_eval_answers_parser(sub)
    return parser.parse_args(["eval-answers", *argv])


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("sentence_transformers")
    server = load_mcp_server_module()
    monkeypatch.setattr(server, "get_embedder", lambda: StubEncoder())
    monkeypatch.setattr("grounding.reranker.rerank", lexical_rerank)
    index = build_stub_mini_index(tmp_path / "embeddings")
    judge = FakeJudge(
        supported=lambda claim, passage: "falsifiab" in claim.lower() and "falsifiab" in passage.lower(),
        # Full credit for the scripted grounded answers; half for the terser
        # ungrounded bootstrap answer.
        correctness=lambda candidate: ("1", False) if any(
            key in candidate.lower() for key in ("fals", "discretize", "read off the percentiles")
        ) else ("0.5", False),
        declined=lambda candidate: "does not contain" in candidate,
    )
    model = ScriptedModel(
        ungrounded_answers=UNGROUNDED, grounded_answer=grounded_answer, judge=judge
    )
    client = FakeClient(model)
    argv = [
        "--agent", "mini",
        "--agents-dir", str(MINI_AGENTS_DIR),
        "--fixtures", str(MINI_ANSWERS_YAML),
        "--corpus", str(MINI_CORPUS),
        "--embeddings", str(index),
        "--out", str(tmp_path / "out"),
    ]
    return {"argv": argv, "client": client, "judge": judge, "out": tmp_path / "out"}


def _run_dir(out: Path) -> Path:
    (run_dir,) = list(out.iterdir())
    return run_dir


def test_full_run_answers_scores_and_exports_blind_sample(pipeline):
    code = answers_cli.eval_answers_command(
        _parse(pipeline["argv"]), client=pipeline["client"], run_eval_fn=stub_run_eval, env={}
    )
    assert code == answers_cli.EXIT_OK
    run_dir = _run_dir(pipeline["out"])
    scores = {(s["item_id"], s["condition"]): s for s in read_jsonl(run_dir / "scores.jsonl")}
    assert len(scores) == 16

    # Ungrounded buckets come from the scripted answers above.
    assert [c["bucket"] for c in scores[("ans-001", "ungrounded")]["citations"]] == ["verified"]
    assert [c["bucket"] for c in scores[("ans-002", "ungrounded")]["citations"]] == ["unresolvable"]
    assert scores[("ans-003", "ungrounded")]["n_citations"] == 0
    # ans-004 is not_in_corpus: the ungrounded condition had no library, so its
    # answer is never scored and never counts against it.
    assert scores[("ans-004", "ungrounded")]["scored"] is False
    assert scores[("ans-004", "ungrounded")]["excluded"] == "not_in_corpus_ungrounded"

    # Grounded answers cite the first returned prefix, so they always resolve;
    # on the unanswerable item they decline without citing.
    for (item_id, condition), score in scores.items():
        if condition == "ungrounded":
            continue
        if item_id == "ans-004":
            assert score["n_citations"] == 0
            assert score["abstention"] == {
                "method": "judge", "reason": "scripted", "gap_flagged": True,
                "gave_answer": False, "answer_labeled": True, "acknowledged_gap": True,
                "declined": True,
            }
            continue
        assert score["bucket_counts"]["invented"] == 0
        assert score["n_citations"] == 1
        assert score["citations"][0]["resolution"]["match"] == "exact"
        assert score["correctness"] == {
            "score": 1.0, "method": "judge", "declined": False, "missing_facts": [],
            "reason": "scripted", "numeric_check": None,
        }
    assert [c["bucket"] for c in scores[("ans-001", "hybrid")]["citations"]] == ["verified"]
    assert [c["bucket"] for c in scores[("ans-002", "hybrid")]["citations"]] == ["unsupported"]
    assert scores[("ans-002", "ungrounded")]["correctness"]["score"] == 0.5

    manifest = json.loads((run_dir / "run.json").read_text())
    scoring = manifest["scoring"]
    assert scoring["judge_model"] == "claude-sonnet-5"
    assert scoring["prompts"]["support"]["version"] == "support-v2"
    assert scoring["judge_calls"] > 0 and scoring["judge_cost_usd"] > 0
    # The blind export defaults to every scored answer (fraction 1.0).
    assert scoring["blind"]["per_condition"] == {
        "dense": 4, "hybrid": 4, "hybrid-rerank": 4, "ungrounded": 3,
    }
    assert (run_dir / "blind" / "answers.csv").exists()
    assert manifest["status"] == "complete"


def test_rescore_with_another_judge_and_import_blind_grades(pipeline, capsys):
    answers_cli.eval_answers_command(
        _parse(pipeline["argv"] + ["--skip-scoring"]),
        client=pipeline["client"], run_eval_fn=stub_run_eval, env={},
    )
    run_dir = _run_dir(pipeline["out"])
    assert not (run_dir / "scores.jsonl").exists()
    judge_calls_before = len(pipeline["judge"].requests)

    pipeline["judge"].model = "claude-sonnet-5"
    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(run_dir), "--score", "--judge-model", "claude-sonnet-5",
                "--agents-dir", str(MINI_AGENTS_DIR), "--blind-fraction", "1.0"]),
        client=pipeline["client"], env={},
    )
    assert code == answers_cli.EXIT_OK
    assert len(pipeline["judge"].requests) > judge_calls_before
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["scoring"]["judge_model"] == "claude-sonnet-5"
    assert len(read_jsonl(run_dir / "scores.jsonl")) == 16

    answers_csv = run_dir / "blind" / "answers.csv"
    key = json.loads((run_dir / "blind" / "key.json").read_text())["samples"]
    rows = list(csv.DictReader(open(answers_csv)))
    for row in rows:
        entry = key[row["sample_id"]]
        if entry["answerable"]:
            row["andy_grade"] = str(entry["grader_correctness"])
        else:
            row["andy_grade"] = "1" if entry["grader_declined"] else "0"
    with open(answers_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    code = answers_cli.eval_answers_command(
        _parse(["--run-dir", str(run_dir), "--import-grades"]), env={}
    )
    assert code == answers_cli.EXIT_OK
    agreement = json.loads((run_dir / "agreement.json").read_text())
    assert agreement["correctness"]["n"] == 12
    assert agreement["correctness"]["agreement"] == 1.0
    assert agreement["publish_gate"]["result"] == "pass"
    assert "publish gate=pass" in capsys.readouterr().out


def test_run_dir_needs_an_action(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{}")
    assert answers_cli.eval_answers_command(_parse(["--run-dir", str(run_dir)]), env={}) == \
        answers_cli.EXIT_BAD_INPUT
    assert answers_cli.eval_answers_command(_parse(["--score"]), env={}) == answers_cli.EXIT_BAD_INPUT
