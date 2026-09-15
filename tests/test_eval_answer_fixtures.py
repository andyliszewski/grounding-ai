"""Answer-fixture schema tests (Epic 25, Story 25.1).

Covers every validation path of the optional ``answer:`` block and the
top-level ``page_offsets:`` map, and proves the Epic 16 retrieval runner
skips ``answerable: false`` items while still scoring the rest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from grounding.eval import (
    AnswerSpec,
    FixtureValidationError,
    NumericAnswer,
    load_fixtures,
    run_eval,
)

FIXTURES_ROOT = Path(__file__).resolve().parent / "eval_fixtures"
MINI_ANSWERS_YAML = FIXTURES_ROOT / "mini_answers.yaml"
MINI_AGENTS_DIR = FIXTURES_ROOT / "agents"

HEADER = """
agent: scientist
version: 1
"""


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "scientist.yaml").write_text(
        "name: scientist\ndescription: test\n", encoding="utf-8"
    )
    return agents


def _load(tmp_path: Path, agents_dir: Path, body: str):
    path = tmp_path / "answers.yaml"
    path.write_text(HEADER + body, encoding="utf-8")
    return load_fixtures(path, agents_dir=agents_dir)


def _error(tmp_path: Path, agents_dir: Path, body: str) -> FixtureValidationError:
    with pytest.raises(FixtureValidationError) as exc_info:
        _load(tmp_path, agents_dir, body)
    return exc_info.value


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_answer_block_and_page_offsets_parse_into_typed_fields(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
page_offsets:
  doc-a: 14
  doc-b: -2
items:
  - id: q1
    query: "Modulus of elasticity of carbon steel?"
    expected:
      doc_ids: ["doc-a"]
      page: 40
    answer:
      category: table
      gold: "About 200 GPa."
      numeric: {value: 200, unit: GPa, rel_tol: 0.05}
      must_include: ["200 GPa"]
""",
    )

    assert fixtures.page_offsets == {"doc-a": 14, "doc-b": -2}
    answer = fixtures.items[0].answer
    assert answer == AnswerSpec(
        category="table",
        gold="About 200 GPa.",
        numeric=NumericAnswer(value=200.0, unit="GPa", rel_tol=0.05, abs_tol=None),
        must_include=("200 GPa",),
        answerable=True,
    )
    # FixtureSet stays hashable even though page_offsets is a dict.
    hash(fixtures)


def test_items_without_answer_block_are_unchanged(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "plain retrieval item"
    expected:
      doc_ids: ["doc-a"]
""",
    )
    assert fixtures.items[0].answer is None
    assert fixtures.items[0].expected.page is None
    assert fixtures.page_offsets == {}


def test_unanswerable_items_need_no_doc_ids_and_no_page(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
items:
  - id: q-none-1
    query: "Not in the corpus"
    answer:
      category: unanswerable
      answerable: false
      unanswerable_kind: not_in_corpus
  - id: q-none-2
    query: "Also not in the corpus"
    expected:
      doc_ids: []
    answer:
      category: unanswerable
      answerable: false
      unanswerable_kind: no_source
      gold: "Should decline."
""",
    )
    first, second = fixtures.items
    assert first.expected.doc_ids == ()
    assert first.expected.page is None
    assert first.answer.answerable is False
    assert first.answer.gold == ""
    assert first.answer.unanswerable_kind == "not_in_corpus"
    assert second.expected.doc_ids == ()
    assert second.answer.gold == "Should decline."
    assert second.answer.unanswerable_kind == "no_source"


@pytest.mark.parametrize(
    "answer_block, reason",
    [
        ("{category: unanswerable, answerable: false}", "need unanswerable_kind"),
        ("{category: unanswerable, answerable: false, unanswerable_kind: trivia}",
         "unknown unanswerable_kind 'trivia'"),
        ("{category: table, gold: g, unanswerable_kind: no_source}",
         "only unanswerable items"),
    ],
    ids=["missing-kind", "unknown-kind", "kind-on-answerable-item"],
)
def test_unanswerable_kind_is_required_known_and_only_on_unanswerable_items(
    tmp_path, agents_dir, answer_block, reason
):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"], page: 3}}
    answer: {answer_block}
""",
    )
    assert err.field == "answer.unanswerable_kind"
    assert reason in err.reason


def test_page_offsets_accept_section_for_section_paged_works(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
page_offsets:
  handbook-27e: section
  shigley-10e: 22
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["shigley-10e"]}
""",
    )
    assert fixtures.page_offsets == {"handbook-27e": "section", "shigley-10e": 22}
    hash(fixtures)


def test_revisions_and_identifiers_parse_for_government_style_documents(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
revisions:
  doc-a: B
  doc-b: "Rev. C"
  doc-c: 2016
  doc-d: "Rev. 1"
identifiers:
  doc-a: NASA-STD-5001B
  doc-c: "21 CFR Part 820"
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"]}
""",
    )
    assert fixtures.revisions == {"doc-a": "B", "doc-b": "Rev. C", "doc-c": 2016, "doc-d": "Rev. 1"}
    assert fixtures.identifiers == {"doc-a": "NASA-STD-5001B", "doc-c": "21 CFR Part 820"}
    hash(fixtures)


@pytest.mark.parametrize(
    "block, field_name",
    [
        ("revisions: {doc-a: 10}", "revisions"),  # a numbered edition belongs in editions:
        ("revisions: {doc-a: draft}", "revisions"),
        ("revisions: [B]", "revisions"),
        ("identifiers: {doc-a: 5001}", "identifiers"),
        ("identifiers: {doc-a: ''}", "identifiers"),
    ],
    ids=["edition-number", "free-text", "list", "non-string", "empty"],
)
def test_revisions_and_identifiers_reject_values_they_cannot_check(
    tmp_path, agents_dir, block, field_name
):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
{block}
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"]}}
""",
    )
    assert err.field == field_name


def test_persona_and_source_license_parse(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
persona: "a NASA structures engineer"
source_license: public_domain
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"]}
""",
    )
    assert fixtures.persona == "a NASA structures engineer"
    assert fixtures.source_license == "public_domain"
    # Absent: the benchmark's default persona, and source text stays local.
    (tmp_path / "plain").mkdir()
    plain = _load(tmp_path / "plain", agents_dir, """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"]}
""")
    assert plain.persona is None and plain.source_license is None


@pytest.mark.parametrize(
    "block, field_name",
    [
        ("persona: 5", "persona"),
        ('persona: ""', "persona"),
        ("source_license: cc-by", "source_license"),
        ("source_license: true", "source_license"),
    ],
    ids=["persona-number", "persona-empty", "unknown-license", "bool-license"],
)
def test_persona_and_source_license_reject_bad_values(tmp_path, agents_dir, block, field_name):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
{block}
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"]}}
""",
    )
    assert err.field == field_name


def test_numeric_abs_tol_accepts_the_band_edges(tmp_path, agents_dir):
    fixtures = _load(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "Thread count?"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer:
      category: table
      gold: "20 threads per inch."
      numeric: {value: 20, abs_tol: 0}
""",
    )
    numeric = fixtures.items[0].answer.numeric
    assert numeric.unit is None
    assert numeric.accepts(20.0) is True
    assert numeric.accepts(20.5) is False


def test_mini_answers_fixture_loads():
    fixtures = load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR)
    categories = [it.answer.category for it in fixtures.items]
    assert categories == ["guidance", "formula", "judgment", "unanswerable"]
    assert fixtures.page_offsets == {"doc-gamma": 2, "doc-beta": 0}


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------

def test_numeric_without_value_is_rejected(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer:
      category: table
      gold: "g"
      numeric: {unit: MPa, rel_tol: 0.02}
""",
    )
    assert err.item_id == "q1"
    assert err.field == "answer.numeric.value"


@pytest.mark.parametrize(
    "numeric",
    [
        "{value: 5, unit: mm, rel_tol: 0.1, abs_tol: 0.5}",
        "{value: 5, unit: mm}",
    ],
    ids=["both-tolerances", "no-tolerance"],
)
def test_numeric_needs_exactly_one_tolerance(tmp_path, agents_dir, numeric):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"], page: 3}}
    answer:
      category: formula
      gold: "g"
      numeric: {numeric}
""",
    )
    assert err.field == "answer.numeric"
    assert "exactly one of rel_tol or abs_tol" in err.reason


def test_negative_tolerance_is_rejected(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer:
      category: formula
      gold: "g"
      numeric: {value: 5, rel_tol: -0.1}
""",
    )
    assert err.field == "answer.numeric.rel_tol"


def test_unknown_category_is_rejected(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer:
      category: trivia
      gold: "g"
""",
    )
    assert err.field == "answer.category"
    assert "unknown category 'trivia'" in err.reason


def test_answerable_answer_item_needs_a_gold_page(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"]}
    answer:
      category: standard
      gold: "g"
""",
    )
    assert err.field == "expected.page"


def test_answerable_answer_item_still_needs_doc_ids(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: [], page: 3}
    answer:
      category: standard
      gold: "g"
""",
    )
    assert err.field == "expected.doc_ids"
    assert err.reason == "must be non-empty"


@pytest.mark.parametrize(
    "answer_block",
    [
        "{category: unanswerable, gold: g}",
        "{category: table, answerable: false}",
    ],
    ids=["unanswerable-but-answerable", "answerable-false-with-other-category"],
)
def test_category_and_answerable_must_agree(tmp_path, agents_dir, answer_block):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"], page: 3}}
    answer: {answer_block}
""",
    )
    assert err.field == "answer.answerable"


def test_answerable_item_needs_gold_text(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer: {category: guidance}
""",
    )
    assert err.field == "answer.gold"


def test_unknown_answer_key_is_rejected(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    expected: {doc_ids: ["doc-a"], page: 3}
    answer:
      category: guidance
      gold: "g"
      must_inlcude: ["typo"]
""",
    )
    assert err.field == "answer"
    assert "must_inlcude" in err.reason


def test_unanswerable_item_cannot_carry_numeric_gold(tmp_path, agents_dir):
    err = _error(
        tmp_path,
        agents_dir,
        """
items:
  - id: q1
    query: "q"
    answer:
      category: unanswerable
      answerable: false
      unanswerable_kind: no_source
      numeric: {value: 1, abs_tol: 0}
""",
    )
    assert err.field == "answer.numeric"


@pytest.mark.parametrize(
    "offsets",
    ["{doc-a: 1.5}", "{doc-a: true}", "[1, 2]", "{doc-a: sections}"],
    ids=["float", "bool", "list", "misspelled-section"],
)
def test_page_offsets_must_map_doc_ids_to_integers(tmp_path, agents_dir, offsets):
    err = _error(
        tmp_path,
        agents_dir,
        f"""
page_offsets: {offsets}
items:
  - id: q1
    query: "q"
    expected: {{doc_ids: ["doc-a"]}}
""",
    )
    assert err.field == "page_offsets"


# ---------------------------------------------------------------------------
# Epic 16 runner compatibility
# ---------------------------------------------------------------------------

def test_retrieval_runner_skips_unanswerable_items(tmp_path):
    """run_eval scores the answerable items and lists the unanswerable one as skipped."""
    fixtures = load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR)
    chunk_map = {
        "format_version": "1.1",
        "chunks": [
            {"chunk_id": "g1", "doc_id": "doc-gamma", "embedding_index": 0},
            {"chunk_id": "b1", "doc_id": "doc-beta", "embedding_index": 1},
            {"chunk_id": "a1", "doc_id": "doc-alpha", "embedding_index": 2},
        ],
    }
    hits = {
        fixtures.items[0].query: [("g1", 0.1)],
        fixtures.items[1].query: [("b1", 0.1)],
        fixtures.items[2].query: [("a1", 0.1)],
    }
    searched: list[str] = []

    def search(_index, _chunk_map, query, _top_k):
        searched.append(query)
        return hits[query]

    result = run_eval(
        fixtures,
        "mini",
        corpus_dir=tmp_path,
        embeddings_dir=tmp_path,
        embed_fn=lambda text: text,
        search_fn=search,
        load_index_fn=lambda _dir: (object(), chunk_map),
        load_manifest_doc_ids_fn=lambda _dir: {"doc-alpha", "doc-beta", "doc-gamma"},
    )

    assert [it.item_id for it in result.items] == ["ans-001", "ans-002", "ans-003"]
    assert result.skipped == ("ans-004",)
    assert fixtures.items[3].query not in searched
    assert result.aggregate.recall_at_1 == 1.0
