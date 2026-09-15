"""Answer runner tests (Epic 25, Story 25.2).

Every test uses a fake client with scripted responses; nothing touches the
network. Tests that exercise the real in-process ``search_corpus`` need the
optional ``mcp`` package and skip without it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from grounding.eval.answers import cli as answers_cli
from grounding.eval.answers.conditions import CONDITIONS, parse_conditions
from grounding.eval.answers.model_client import (
    HarnessError,
    call_model,
    generation_options,
)
from grounding.eval.answers.prompts import (
    BASE_SYSTEM_PROMPT,
    GROUNDED_CITATION_INSTRUCTIONS,
    UNGROUNDED_CITATION_INSTRUCTIONS,
)
from grounding.eval.answers.citations import CorpusIndex
from grounding.eval.answers.runner import (
    Budget,
    check_page_index,
    compute_retrieval_recall,
    run_answers,
    select_items,
)
from grounding.eval.answers.tool import (
    TOOL_DEFINITION,
    HarnessToolError,
    SearchCorpusTool,
    _server_env,
    load_mcp_server_module,
)
from grounding.eval.fixtures import load_fixtures
from grounding.eval.runner import EvalItemResult, EvalRun, RetrievedChunk
from tests.answers_fakes import (
    MINI_AGENTS_DIR,
    MINI_ANSWERS_YAML,
    MINI_CORPUS,
    FakeAPIError,
    FakeClient,
    ScriptedModel,
    StubEncoder,
    build_stub_mini_index,
    lexical_rerank,
    make_response,
    read_jsonl,
    stub_run_eval,
    text_block,
    tool_use_block,
)

ALL = "ungrounded,dense,hybrid,hybrid-rerank"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    answers_cli._create_eval_answers_parser(sub)
    return parser.parse_args(["eval-answers", *argv])


def _base_argv(embeddings: Path, out: Path) -> list[str]:
    return [
        "--agent", "mini",
        "--agents-dir", str(MINI_AGENTS_DIR),
        "--fixtures", str(MINI_ANSWERS_YAML),
        "--corpus", str(MINI_CORPUS),
        "--embeddings", str(embeddings),
        "--out", str(out),
    ]


@pytest.fixture
def mini_fixtures():
    return load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR)


@pytest.fixture
def stub_index(tmp_path: Path) -> Path:
    pytest.importorskip("faiss")
    return build_stub_mini_index(tmp_path / "embeddings")


@pytest.fixture
def mcp_server(monkeypatch):
    """The real MCP server module with the embedder and reranker stubbed."""
    pytest.importorskip("mcp")
    pytest.importorskip("sentence_transformers")
    server = load_mcp_server_module()
    monkeypatch.setattr(server, "get_embedder", lambda: StubEncoder())
    calls: list[str] = []

    def rerank(query, chunks, *, config, text_key="content"):
        calls.append(query)
        return lexical_rerank(query, chunks, config=config, text_key=text_key)

    monkeypatch.setattr("grounding.reranker.rerank", rerank)
    return SimpleNamespace(module=server, rerank_calls=calls)


# ---------------------------------------------------------------------------
# Acceptance: a fake-client run over the mini corpus, all four conditions
# ---------------------------------------------------------------------------

def test_fake_client_run_writes_transcripts_for_all_four_conditions(
    tmp_path, stub_index, mcp_server, mini_fixtures
):
    client = FakeClient(ScriptedModel())
    args = _parse(
        _base_argv(stub_index, tmp_path / "out") + ["--conditions", ALL, "--skip-scoring"]
    )

    code = answers_cli.eval_answers_command(
        args, client=client, run_eval_fn=stub_run_eval, env={}
    )

    assert code == answers_cli.EXIT_OK
    (run_dir,) = list((tmp_path / "out").iterdir())
    rows = read_jsonl(run_dir / "transcripts.jsonl")
    pairs = {(r["item_id"], r["condition"]) for r in rows}
    assert pairs == {
        (it.id, c) for it in mini_fixtures.items for c in ALL.split(",")
    }
    assert not (run_dir / "errors.jsonl").exists()
    for row in rows:
        assert row["status"] == "ok"
        assert row["final_text"]
        assert row["usage"]["input_tokens"] > 0 and row["cost_usd"] > 0
        assert all(c["model"] == "claude-opus-5" for c in row["model_calls"])
        if row["condition"] == "ungrounded":
            assert row["tool_calls"] == [] and row["n_model_calls"] == 1
            continue
        assert row["n_tool_rounds"] == 1
        (call,) = row["tool_calls"]
        # Each recorded prefix is a line of the exact text the model received.
        assert call["citation_prefixes"]
        for prefix in call["citation_prefixes"]:
            assert f"\n{prefix}\n" in call["output_text"]
        # The final answer cites the first returned prefix (scripted).
        assert call["citation_prefixes"][0] in row["final_text"]

    # Only the hybrid-rerank condition ran the reranker (4 answers + recall).
    assert len(mcp_server.rerank_calls) == len(mini_fixtures.items) + 3

    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["answer_model"] == "claude-opus-5"
    assert manifest["judge_model"] == "claude-sonnet-5"  # different from the answer model
    assert manifest["git"]["sha"]
    assert set(manifest["index_fingerprint"]["files"]) >= {
        "_embeddings.faiss", "_chunk_map.json", "_bm25.pkl", "_bm25_map.json",
        "corpus/_index.json",
    }
    assert manifest["prompts"]["answer_version"] == "answer-v3"
    assert manifest["prompts"]["persona"] == "a practicing mechanical engineer"
    for key in ("corpus_dir", "embeddings_dir"):
        assert Path(manifest[key]).is_absolute()
    assert Path(manifest["fixture"]["path"]).is_absolute()
    assert len(manifest["prompts"]["answer_system_sha256"]["grounded"]) == 64
    assert manifest["answers"]["written"] == 16
    retrieval = json.loads((run_dir / "retrieval.json").read_text())
    assert set(retrieval["conditions"]) == {"dense", "hybrid", "hybrid-rerank"}
    for stats in retrieval["conditions"].values():
        assert stats["n_items"] == 3 and stats["skipped"] == ["ans-004"]


def test_conditions_share_one_system_prompt_except_citation_instructions(
    tmp_path, stub_index, mcp_server, mini_fixtures
):
    client = FakeClient(ScriptedModel())
    items = mini_fixtures.items[:1]
    run_answers(
        client,
        items=items,
        conditions=parse_conditions(ALL),
        answer_model="claude-opus-5",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=stub_index,
        run_dir=tmp_path / "run",
        run_id="r1",
    )
    first_calls = [c for c in client.messages.calls if len(c["messages"]) == 1]
    systems = [c["system"] for c in first_calls]
    # Claim pairing: every condition is told to state one claim per sentence and cite it.
    assert all("State one claim per sentence and cite it." in s for s in systems)
    assert systems[0] == f"{BASE_SYSTEM_PROMPT}\n\n{UNGROUNDED_CITATION_INSTRUCTIONS}"
    assert set(systems[1:]) == {f"{BASE_SYSTEM_PROMPT}\n\n{GROUNDED_CITATION_INSTRUCTIONS}"}
    assert "tools" not in first_calls[0]
    assert all(c["tools"] == [TOOL_DEFINITION] for c in first_calls[1:])
    # Same model and pinned generation options everywhere.
    for call in client.messages.calls:
        assert call["model"] == "claude-opus-5"
        assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# Dry run, keys, and cost caps
# ---------------------------------------------------------------------------

def test_the_persona_comes_from_the_fixture_and_judges_name_the_question_writer(
    tmp_path, stub_index, mcp_server
):
    import hashlib

    from grounding.eval.answers.prompts import (
        ABSTENTION_NO_SOURCE_SYSTEM,
        CORRECTNESS_JUDGE_SYSTEM,
        answer_system_prompt,
    )

    # The default wording is unchanged for fixtures that set no persona.
    assert answer_system_prompt(True).startswith(
        "You are a technical reference assistant for a practicing mechanical engineer."
    )
    for judge_prompt in (CORRECTNESS_JUDGE_SYSTEM, ABSTENTION_NO_SOURCE_SYSTEM):
        assert "question writer" in judge_prompt.lower()
        assert "engineer who set the question" not in judge_prompt

    fixture = tmp_path / "persona.yaml"
    fixture.write_text('persona: "a NASA structures engineer"\n' + MINI_ANSWERS_YAML.read_text())
    client = FakeClient(ScriptedModel())
    args = _parse(_base_argv(stub_index, tmp_path / "out")
                  + ["--items", "ans-001", "--conditions", "ungrounded", "--skip-scoring"])
    args.fixtures = fixture
    assert answers_cli.eval_answers_command(
        args, client=client, run_eval_fn=stub_run_eval, env={}
    ) == answers_cli.EXIT_OK
    (call,) = client.messages.calls
    assert call["system"].startswith(
        "You are a technical reference assistant for a NASA structures engineer."
    )
    (run_dir,) = list((tmp_path / "out").iterdir())
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["prompts"]["persona"] == "a NASA structures engineer"
    # The recorded hash is of the rendered prompt, so a persona change is visible.
    default = hashlib.sha256(answer_system_prompt(False).encode()).hexdigest()
    assert manifest["prompts"]["answer_system_sha256"]["ungrounded"] != default


def test_the_ungrounded_prompt_asks_for_clause_section_or_paragraph_numbers():
    assert "clause, section or paragraph number" in UNGROUNDED_CITATION_INSTRUCTIONS
    assert "p. N" in UNGROUNDED_CITATION_INSTRUCTIONS
    # The grounded prompt tells the model what to do when the sources fall short.
    assert "say so plainly" in GROUNDED_CITATION_INSTRUCTIONS
    assert "label it clearly as not from the sources" in GROUNDED_CITATION_INSTRUCTIONS


def test_dry_run_needs_no_key_and_makes_no_calls(tmp_path, stub_index, capsys):
    client = FakeClient(ScriptedModel())
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--dry-run"])

    code = answers_cli.eval_answers_command(args, client=client, env={})

    assert code == answers_cli.EXIT_OK
    assert client.messages.calls == []
    assert not (tmp_path / "out").exists()
    out = capsys.readouterr().out
    assert "no API calls made, no key needed" in out
    assert "total" in out and "$" in out


def test_real_run_without_key_exits_before_any_output(tmp_path, stub_index):
    args = _parse(_base_argv(stub_index, tmp_path / "out"))
    code = answers_cli.eval_answers_command(args, env={})
    assert code == answers_cli.EXIT_NO_API_KEY
    assert not (tmp_path / "out").exists()


def test_max_cost_below_estimate_aborts_before_calling(tmp_path, stub_index, capsys):
    client = FakeClient(ScriptedModel())
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--max-cost", "0.01"])
    code = answers_cli.eval_answers_command(args, client=client, env={})
    assert code == answers_cli.EXIT_MAX_COST
    assert client.messages.calls == []
    assert "exceeds --max-cost" in capsys.readouterr().err


def test_budget_guard_stops_mid_run_and_keeps_spend(tmp_path, mini_fixtures):
    """Each fake answer costs $0.105; the worst case of the next call is ~$0.40."""
    responder = lambda params, n: make_response(  # noqa: E731
        [text_block("An answer (Gamma Notes, 1st ed., p. 10).")],
        input_tokens=1000,
        output_tokens=4000,
    )
    client = FakeClient(responder)
    budget = Budget(max_cost=0.45)

    summary = run_answers(
        client,
        items=mini_fixtures.items,
        conditions=(CONDITIONS["ungrounded"],),
        answer_model="claude-opus-5",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=tmp_path,
        run_dir=tmp_path / "run",
        run_id="r1",
        budget=budget,
    )

    assert summary.written == 1
    assert summary.aborted and "--max-cost" in summary.aborted
    assert summary.not_run == [
        ("ans-002", "ungrounded"), ("ans-003", "ungrounded"), ("ans-004", "ungrounded"),
    ]
    assert len(client.messages.calls) == 1
    assert budget.spent == pytest.approx(0.105)
    (error,) = read_jsonl(tmp_path / "run" / "errors.jsonl")
    assert error["failure_class"] == "budget_exceeded"
    assert error["item_id"] == "ans-002"


# ---------------------------------------------------------------------------
# Tool loop behavior
# ---------------------------------------------------------------------------

class _AlwaysSearches:
    def __call__(self, params, n):
        if params.get("tool_choice") == {"type": "none"}:
            return make_response([text_block("Forced final answer.")])
        return make_response([tool_use_block(f"toolu_{n}", "more")], stop_reason="tool_use")


class _RecordingTool:
    def __init__(self):
        self.queries = []

    def execute(self, name, tool_input):
        from grounding.eval.answers.tool import ToolExecution

        self.queries.append(tool_input["query"])
        return ToolExecution(tool_input["query"], 5, "## Corpus Search Results\n[x, p.1]\nbody", False, 0.0,
                             [{"prefix": "[x, p.1]", "slug": "x", "page_start": 1, "page_end": 1}])


def test_max_iterations_forces_a_final_text_answer(tmp_path, mini_fixtures):
    client = FakeClient(_AlwaysSearches())
    tool = _RecordingTool()
    run_answers(
        client,
        items=mini_fixtures.items[:1],
        conditions=(CONDITIONS["dense"],),
        answer_model="claude-opus-5",
        corpus_dir=MINI_CORPUS,
        embeddings_dir=tmp_path,
        run_dir=tmp_path / "run",
        run_id="r1",
        max_iterations=2,
        tool_factory=lambda cond: tool,
    )
    (row,) = read_jsonl(tmp_path / "run" / "transcripts.jsonl")
    assert row["status"] == "max_iterations"
    assert row["n_tool_rounds"] == 2 and row["n_model_calls"] == 3
    assert row["final_text"] == "Forced final answer."
    assert "tool_choice" not in client.messages.calls[0]
    assert client.messages.calls[2]["tool_choice"] == {"type": "none"}
    # Both tool results went back as tool_result blocks with matching ids.
    tool_msgs = [m for m in client.messages.calls[2]["messages"] if m["role"] == "user"][1:]
    assert [m["content"][0]["tool_use_id"] for m in tool_msgs] == ["toolu_1", "toolu_2"]


def test_parallel_tool_calls_return_in_one_user_message(tmp_path, mini_fixtures):
    def responder(params, n):
        if n == 1:
            return make_response(
                [tool_use_block("toolu_a", "first"), tool_use_block("toolu_b", "second")],
                stop_reason="tool_use",
            )
        return make_response([text_block("Done [x, p.1].")])

    client = FakeClient(responder)
    tool = _RecordingTool()
    run_answers(
        client, items=mini_fixtures.items[:1], conditions=(CONDITIONS["dense"],),
        answer_model="claude-opus-5", corpus_dir=MINI_CORPUS, embeddings_dir=tmp_path,
        run_dir=tmp_path / "run", run_id="r1", tool_factory=lambda cond: tool,
    )
    second = client.messages.calls[1]["messages"]
    assert [b["tool_use_id"] for b in second[-1]["content"]] == ["toolu_a", "toolu_b"]
    assert tool.queries == ["first", "second"]
    (row,) = read_jsonl(tmp_path / "run" / "transcripts.jsonl")
    assert [t["id"] for t in row["tool_calls"]] == ["toolu_a", "toolu_b"]


def test_refusal_and_truncation_are_recorded_as_outcomes(tmp_path, mini_fixtures):
    stops = iter(["refusal", "max_tokens"])
    client = FakeClient(lambda p, n: make_response([text_block("")], stop_reason=next(stops)))
    run_answers(
        client, items=mini_fixtures.items[:2], conditions=(CONDITIONS["ungrounded"],),
        answer_model="claude-opus-5", corpus_dir=MINI_CORPUS, embeddings_dir=tmp_path,
        run_dir=tmp_path / "run", run_id="r1",
    )
    rows = read_jsonl(tmp_path / "run" / "transcripts.jsonl")
    assert [r["status"] for r in rows] == ["refusal", "truncated"]


# ---------------------------------------------------------------------------
# Harness failures go to errors.jsonl, never to transcripts
# ---------------------------------------------------------------------------

def test_retryable_error_is_retried_and_attempts_recorded():
    outcomes = iter([FakeAPIError(529, "overloaded"), make_response([text_block("ok")])])
    client = FakeClient(lambda p, n: next(outcomes))
    sleeps = []
    response, record = call_model(
        client,
        {"model": "claude-opus-5", "max_tokens": 10, "messages": []},
        purpose="answer",
        sleep=sleeps.append,
        jitter=lambda: 0.0,
    )
    assert record.attempts == 2
    assert sleeps == [2.0]
    assert record.stop_reason == "end_turn"


def test_non_retryable_error_and_served_model_mismatch_are_errors(tmp_path, mini_fixtures):
    outcomes = iter(
        [
            FakeAPIError(400, "bad request"),
            make_response([text_block("from the wrong model")], model="claude-sonnet-5"),
            make_response([text_block("fine")]),
        ]
    )
    client = FakeClient(lambda p, n: next(outcomes))
    budget = Budget(None)
    summary = run_answers(
        client, items=mini_fixtures.items[:3], conditions=(CONDITIONS["ungrounded"],),
        answer_model="claude-opus-5", corpus_dir=MINI_CORPUS, embeddings_dir=tmp_path,
        run_dir=tmp_path / "run", run_id="r1", budget=budget,
    )
    assert summary.written == 1 and summary.errors == 2
    errors = read_jsonl(tmp_path / "run" / "errors.jsonl")
    assert [e["failure_class"] for e in errors] == ["api_error", "served_model_mismatch"]
    assert [e["item_id"] for e in errors] == ["ans-001", "ans-002"]
    # The mismatched call was billed; its usage and cost are kept.
    assert errors[1]["usage"]["input_tokens"] == 1000 and errors[1]["cost_usd"] > 0
    (row,) = read_jsonl(tmp_path / "run" / "transcripts.jsonl")
    assert row["item_id"] == "ans-003"


def test_retrieval_failure_aborts_the_answer_as_tool_error(tmp_path, mini_fixtures):
    broken = SimpleNamespace(
        search_corpus=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index corrupt")),
        format_results_for_context=lambda results, query: "",
        _index_cache={},
    )
    tool = SearchCorpusTool(
        CONDITIONS["dense"], corpus_dir=MINI_CORPUS, embeddings_dir=tmp_path, server=broken
    )
    with pytest.raises(HarnessToolError):
        tool.execute("search_corpus", {"query": "q"})

    client = FakeClient(ScriptedModel())
    summary = run_answers(
        client, items=mini_fixtures.items[:1], conditions=(CONDITIONS["dense"],),
        answer_model="claude-opus-5", corpus_dir=MINI_CORPUS, embeddings_dir=tmp_path,
        run_dir=tmp_path / "run", run_id="r1", tool_factory=lambda cond: tool,
    )
    assert summary.errors == 1
    (error,) = read_jsonl(tmp_path / "run" / "errors.jsonl")
    assert error["failure_class"] == "tool_error"
    assert error["partial_model_calls"] == 1
    assert not (tmp_path / "run" / "transcripts.jsonl").exists()


# ---------------------------------------------------------------------------
# The tool mirrors the MCP server
# ---------------------------------------------------------------------------

def test_tool_definition_mirrors_the_mcp_server(mcp_server):
    import asyncio

    tools = asyncio.run(mcp_server.module.list_tools())
    mcp_tool = next(t for t in tools if t.name == "search_corpus")
    assert TOOL_DEFINITION["name"] == mcp_tool.name
    assert TOOL_DEFINITION["description"] == mcp_tool.description
    props = TOOL_DEFINITION["input_schema"]["properties"]
    for key in ("query", "top_k"):
        assert props[key] == mcp_tool.inputSchema["properties"][key]


@pytest.mark.parametrize("condition", ["dense", "hybrid", "hybrid-rerank"])
def test_tool_output_is_exactly_what_the_mcp_server_returns(stub_index, mcp_server, condition):
    server = mcp_server.module
    spec = CONDITIONS[condition]
    tool = SearchCorpusTool(spec, corpus_dir=MINI_CORPUS, embeddings_dir=stub_index)
    execution = tool.execute("search_corpus", {"query": "bootstrap confidence interval", "top_k": 3})

    with _server_env(MINI_CORPUS.resolve(), stub_index.resolve().parent):
        expected = server.format_results_for_context(
            server.search_corpus(
                "bootstrap confidence interval", stub_index.name, 3,
                rerank_config=spec.rerank, hybrid_config=spec.hybrid,
            ),
            "bootstrap confidence interval",
        )
    assert execution.output_text == expected
    assert execution.results[0]["prefix"] in execution.output_text
    assert execution.results[0]["slug"] == "beta"  # slug of source beta.pdf
    assert "The bootstrap is a resampling method" in execution.output_text


def test_tool_rejects_empty_query_like_the_mcp_server(stub_index):
    tool = SearchCorpusTool(
        CONDITIONS["dense"], corpus_dir=MINI_CORPUS, embeddings_dir=stub_index,
        server=SimpleNamespace(_index_cache={}),
    )
    execution = tool.execute("search_corpus", {"query": "  "})
    assert execution.output_text == "Error: query is required"
    assert execution.is_error is True


def test_server_env_is_restored(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_DIR", "/before")
    monkeypatch.delenv("EMBEDDINGS_DIR", raising=False)
    import os

    with _server_env(tmp_path, tmp_path):
        assert os.environ["EMBEDDINGS_DIR"] == str(tmp_path)
    assert os.environ["CORPUS_DIR"] == "/before"
    assert "EMBEDDINGS_DIR" not in os.environ


# ---------------------------------------------------------------------------
# Model options, SDK objects, item selection, recall
# ---------------------------------------------------------------------------

def test_generation_options_are_pinned_per_model_family():
    assert generation_options("claude-opus-5") == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    }
    assert generation_options("claude-sonnet-5")["output_config"] == {"effort": "high"}
    assert generation_options("claude-haiku-4-5") == {}


def test_call_model_normalizes_real_sdk_message_objects():
    anthropic = pytest.importorskip("anthropic")
    message = anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [
                {"type": "text", "text": "Searching."},
                {"type": "tool_use", "id": "toolu_1", "name": "search_corpus",
                 "input": {"query": "q"}},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }
    )
    client = FakeClient(lambda p, n: message)
    _, record = call_model(client, {"model": "claude-opus-5", "messages": []}, purpose="answer")
    assert record.usage == {
        "input_tokens": 12, "output_tokens": 7,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }
    assert record.content == [
        {"type": "text", "text": "Searching."},
        {"type": "tool_use", "id": "toolu_1", "name": "search_corpus", "input": {"query": "q"}},
    ]
    json.dumps(record.as_dict())  # transcript-ready


def test_select_items_filters_then_limits(mini_fixtures):
    picked = select_items(mini_fixtures, item_ids=["ans-003", "ans-001"], limit=1)
    assert [it.id for it in picked] == ["ans-001"]
    with pytest.raises(ValueError, match="ans-999"):
        select_items(mini_fixtures, item_ids=["ans-999"])


def _fixture_with_retrieval_only_item(tmp_path: Path) -> Path:
    """The mini answer fixture plus one Epic 16 style item with no answer: block."""
    text = MINI_ANSWERS_YAML.read_text()
    text += (
        "\n  - id: ret-only\n"
        '    query: "What is renormalization?"\n'
        "    expected:\n"
        '      doc_ids: ["doc-alpha"]\n'
    )
    path = tmp_path / "mixed.yaml"
    path.write_text(text)
    return path


def test_items_without_an_answer_block_are_refused_at_selection(tmp_path):
    fixtures = load_fixtures(_fixture_with_retrieval_only_item(tmp_path), agents_dir=MINI_AGENTS_DIR)
    with pytest.raises(ValueError, match="have no answer: block.*ret-only"):
        select_items(fixtures)
    # Leaving the item out with --items is the way around it.
    assert [it.id for it in select_items(fixtures, item_ids=["ans-001"])] == ["ans-001"]


def test_cli_refuses_items_without_an_answer_block_before_the_estimate(tmp_path, stub_index, capsys):
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--dry-run"])
    args.fixtures = _fixture_with_retrieval_only_item(tmp_path)
    code = answers_cli.eval_answers_command(args, env={})
    assert code == answers_cli.EXIT_BAD_INPUT
    captured = capsys.readouterr()
    assert "ret-only" in captured.err and "no answer: block" in captured.err
    assert "total" not in captured.out  # no estimate was printed


def _unpaged_copy_of_mini_corpus(tmp_path: Path, doc_dir: str) -> Path:
    """The mini corpus with page_start/page_end stripped from one document's chunks."""
    import shutil

    root = tmp_path / "corpus"
    shutil.copytree(MINI_CORPUS, root)
    for chunk in (root / doc_dir / "chunks").glob("ch_*.md"):
        lines = [ln for ln in chunk.read_text().splitlines()
                 if not ln.startswith(("page_start:", "page_end:"))]
        chunk.write_text("\n".join(lines) + "\n")
    return root


def test_check_page_index_lists_items_whose_gold_document_has_no_pages(tmp_path, stub_index, mini_fixtures):
    corpus = CorpusIndex(_unpaged_copy_of_mini_corpus(tmp_path, "alpha-paper"), stub_index)
    with pytest.raises(ValueError) as exc_info:
        check_page_index(mini_fixtures.items, corpus)
    message = str(exc_info.value)
    assert "ans-003 (doc doc-alpha: no page_start in any chunk)" in message
    assert "ans-001" not in message and "ans-004" not in message  # paged, and unanswerable
    # A gold document missing from the index is refused the same way.
    items = [replace(mini_fixtures.items[0], expected=replace(
        mini_fixtures.items[0].expected, doc_ids=("doc-missing",)))]
    with pytest.raises(ValueError, match=r"ans-001 \(doc doc-missing: not in the agent's index\)"):
        check_page_index(items, CorpusIndex(MINI_CORPUS, stub_index))


def test_cli_refuses_unpaged_gold_documents_before_the_estimate(tmp_path, stub_index, capsys):
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--dry-run"])
    args.corpus = _unpaged_copy_of_mini_corpus(tmp_path, "alpha-paper")
    code = answers_cli.eval_answers_command(args, env={})
    assert code == answers_cli.EXIT_BAD_INPUT
    captured = capsys.readouterr()
    assert "ans-003 (doc doc-alpha: no page_start in any chunk)" in captured.err
    assert "total" not in captured.out
    # Leaving the item out lets the run go ahead.
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--dry-run", "--items", "ans-001,ans-004"])
    args.corpus = _unpaged_copy_of_mini_corpus(tmp_path / "again", "alpha-paper")
    assert answers_cli.eval_answers_command(args, env={}) == answers_cli.EXIT_OK


def test_gold_page_recall_is_stricter_than_doc_recall(mini_fixtures):
    """A top-5 chunk from the right document but the wrong page is a doc hit only."""

    def fake_run_eval(fixture_set, agent, **kwargs):
        items = []
        for it in fixture_set.items:
            if it.answer and not it.answer.answerable:
                continue
            doc = it.expected.doc_ids[0]
            items.append(
                EvalItemResult(
                    item_id=it.id, query=it.query, expected_doc_ids=it.expected.doc_ids,
                    retrieved=(RetrievedChunk(doc, "c1", 0.1, 1, page_start=900, page_end=901),),
                    first_hit_rank=1, strict_first_hit_rank=None, tags=it.tags,
                    expected_page=it.expected.page,
                )
            )
        return SimpleNamespace(items=tuple(items), skipped=("ans-004",))

    recall = compute_retrieval_recall(
        mini_fixtures, mini_fixtures.items, parse_conditions("ungrounded,dense"),
        corpus_dir=MINI_CORPUS, embeddings_dir=Path("."), run_eval_fn=fake_run_eval,
    )
    dense = recall["conditions"]["dense"]
    assert set(recall["conditions"]) == {"dense"}
    assert dense["recall_at_5_doc"] == 1.0
    assert dense["recall_at_5_gold_page"] == 0.0
    assert dense["per_item"]["ans-001"] == {"doc_hit_rank": 1, "page_hit_rank": None}


def test_missing_bm25_sidecar_is_refused_for_hybrid(tmp_path, stub_index):
    (stub_index / "_bm25.pkl").unlink()
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--conditions", "hybrid", "--dry-run"])
    assert answers_cli.eval_answers_command(args, env={}) == answers_cli.EXIT_INDEX_MISSING
    args = _parse(_base_argv(stub_index, tmp_path / "out") + ["--conditions", "dense", "--dry-run"])
    assert answers_cli.eval_answers_command(args, env={}) == answers_cli.EXIT_OK
