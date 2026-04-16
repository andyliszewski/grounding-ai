"""Tests for grounding.eval.fixtures (Story 16.1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from grounding.eval import (
    Expected,
    FixtureItem,
    FixtureSet,
    FixtureValidationError,
    UnknownAgentError,
    load_fixtures,
)


VALID_YAML = """
agent: scientist
version: 1
items:
  - id: sci-001
    query: "What distinguishes science from pseudoscience?"
    expected:
      doc_ids: ["abc12345"]
      chunk_ids: ["abc12345/ch_0023"]
    tags: ["methodology", "philosophy"]
    notes: "Popper, LSD ch. 1."
  - id: sci-002
    query: "Bootstrap CI for small sample mean"
    expected:
      doc_ids: ["def67890", "ghi11223"]
"""


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "scientist.yaml").write_text(
        "name: scientist\ndescription: test\n", encoding="utf-8"
    )
    return agents


def _write(tmp_path: Path, body: str, name: str = "fixture.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_valid_fixture_returns_typed_set(tmp_path: Path, agents_dir: Path) -> None:
    path = _write(tmp_path, VALID_YAML)

    fixtures = load_fixtures(path, agents_dir=agents_dir)

    assert isinstance(fixtures, FixtureSet)
    assert fixtures.agent == "scientist"
    assert fixtures.version == 1
    assert fixtures.source_path == path
    assert len(fixtures.items) == 2

    first = fixtures.items[0]
    assert isinstance(first, FixtureItem)
    assert first.id == "sci-001"
    assert first.query.startswith("What distinguishes")
    assert isinstance(first.expected, Expected)
    assert first.expected.doc_ids == ("abc12345",)
    assert first.expected.chunk_ids == ("abc12345/ch_0023",)
    assert first.tags == ("methodology", "philosophy")
    assert first.notes == "Popper, LSD ch. 1."

    second = fixtures.items[1]
    assert second.expected.doc_ids == ("def67890", "ghi11223")
    assert second.expected.chunk_ids == ()
    assert second.tags == ()
    assert second.notes == ""


def test_fixture_dataclasses_are_frozen(tmp_path: Path, agents_dir: Path) -> None:
    fixtures = load_fixtures(_write(tmp_path, VALID_YAML), agents_dir=agents_dir)
    with pytest.raises(Exception):
        fixtures.items[0].id = "mutated"  # type: ignore[misc]


def test_load_missing_required_field_raises_with_context(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: sci-001
    expected:
      doc_ids: ["abc"]
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.path == path
    assert err.item_id == "sci-001"
    assert err.field == "query"
    assert "missing" in err.reason or "empty" in err.reason


def test_load_missing_expected_doc_ids_reports_field(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: sci-001
    query: "test"
    expected:
      doc_ids: []
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.path == path
    assert err.item_id == "sci-001"
    assert err.field == "expected.doc_ids"
    assert err.reason == "must be non-empty"


def test_load_unknown_agent_raises(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: data-scientest
version: 1
items:
  - id: s-1
    query: "q"
    expected:
      doc_ids: ["a"]
"""
    path = _write(tmp_path, body)

    with pytest.raises(UnknownAgentError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert isinstance(err, FixtureValidationError)
    assert err.path == path
    assert err.field == "agent"
    assert "data-scientest" in err.reason


def test_load_duplicate_item_id_raises(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: dup-1
    query: "a"
    expected:
      doc_ids: ["x"]
  - id: dup-1
    query: "b"
    expected:
      doc_ids: ["y"]
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.item_id == "dup-1"
    assert err.field == "id"
    assert err.reason == "duplicate item id"


def test_load_empty_items_raises(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 1
items: []
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.field == "items"
    assert err.reason == "must be non-empty"


def test_load_malformed_yaml_raises_with_path(
    tmp_path: Path, agents_dir: Path
) -> None:
    path = _write(tmp_path, "agent: scientist\nversion: 1\nitems:\n  - id: [unclosed\n")

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.path == path
    assert "malformed YAML" in err.reason


def test_load_empty_file_raises(tmp_path: Path, agents_dir: Path) -> None:
    path = _write(tmp_path, "")

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.path == path
    assert err.reason == "fixture file is empty"


def test_load_chunk_id_format_validation_good(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: "q"
    expected:
      doc_ids: ["7a9b2c1f"]
      chunk_ids: ["7a9b2c1f/ch_0001", "7a9b2c1f/ch_9999"]
"""
    fixtures = load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert fixtures.items[0].expected.chunk_ids == (
        "7a9b2c1f/ch_0001",
        "7a9b2c1f/ch_9999",
    )


def test_load_chunk_id_format_validation_bad(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: "q"
    expected:
      doc_ids: ["7a9b2c1f"]
      chunk_ids: ["7a9b2c1f-ch-0001"]
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    err = exc_info.value
    assert err.item_id == "s1"
    assert err.field == "expected.chunk_ids"
    assert "ch_NNNN" in err.reason


def test_load_unsupported_version_raises(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 2
items:
  - id: s1
    query: "q"
    expected:
      doc_ids: ["x"]
"""
    path = _write(tmp_path, body)

    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(path, agents_dir=agents_dir)

    assert exc_info.value.field == "version"


def test_load_missing_fixture_file(tmp_path: Path, agents_dir: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(missing, agents_dir=agents_dir)
    assert exc_info.value.path == missing


def test_loader_does_not_touch_corpus_or_faiss(
    tmp_path: Path, agents_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loader must work with no corpus dir anywhere reachable."""
    work = tmp_path / "isolated"
    work.mkdir()
    monkeypatch.chdir(work)

    fixture_path = work / "fixture.yaml"
    fixture_path.write_text(VALID_YAML, encoding="utf-8")

    fixtures = load_fixtures(fixture_path, agents_dir=agents_dir)

    assert isinstance(fixtures, FixtureSet)
    assert not (work / "corpus").exists()
    assert not (work / "embeddings").exists()


def test_snapshot_known_good_fixture(tmp_path: Path, agents_dir: Path) -> None:
    fixtures = load_fixtures(_write(tmp_path, VALID_YAML), agents_dir=agents_dir)

    snapshot = {
        "agent": fixtures.agent,
        "version": fixtures.version,
        "item_count": len(fixtures.items),
        "items": [
            {
                "id": item.id,
                "query": item.query,
                "doc_ids": list(item.expected.doc_ids),
                "chunk_ids": list(item.expected.chunk_ids),
                "tags": list(item.tags),
                "notes": item.notes,
            }
            for item in fixtures.items
        ],
    }

    assert snapshot == {
        "agent": "scientist",
        "version": 1,
        "item_count": 2,
        "items": [
            {
                "id": "sci-001",
                "query": "What distinguishes science from pseudoscience?",
                "doc_ids": ["abc12345"],
                "chunk_ids": ["abc12345/ch_0023"],
                "tags": ["methodology", "philosophy"],
                "notes": "Popper, LSD ch. 1.",
            },
            {
                "id": "sci-002",
                "query": "Bootstrap CI for small sample mean",
                "doc_ids": ["def67890", "ghi11223"],
                "chunk_ids": [],
                "tags": [],
                "notes": "",
            },
        ],
    }


def test_expected_page_accepts_int(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      page: 247
      section: "3.2 Bootstrap Methods"
"""
    fixtures = load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert fixtures.items[0].expected.page == 247
    assert fixtures.items[0].expected.section == "3.2 Bootstrap Methods"


def test_expected_page_accepts_list(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      page: [15, 18]
"""
    fixtures = load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert fixtures.items[0].expected.page == (15, 18)
    assert fixtures.items[0].expected.section is None


def test_expected_page_rejects_negative(tmp_path: Path, agents_dir: Path) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      page: -3
"""
    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert exc_info.value.field == "expected.page"


def test_expected_page_rejects_out_of_order_range(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      page: [18, 15]
"""
    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert exc_info.value.field == "expected.page"
    assert "start" in exc_info.value.reason


def test_expected_page_rejects_wrong_length_range(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      page: [1, 2, 3]
"""
    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert exc_info.value.field == "expected.page"


def test_expected_section_rejects_empty_string(
    tmp_path: Path, agents_dir: Path
) -> None:
    body = """
agent: scientist
version: 1
items:
  - id: s1
    query: q
    expected:
      doc_ids: ["a"]
      section: "   "
"""
    with pytest.raises(FixtureValidationError) as exc_info:
        load_fixtures(_write(tmp_path, body), agents_dir=agents_dir)
    assert exc_info.value.field == "expected.section"


def test_expected_page_and_section_default_to_none(
    tmp_path: Path, agents_dir: Path
) -> None:
    fixtures = load_fixtures(_write(tmp_path, VALID_YAML), agents_dir=agents_dir)
    for fixture_item in fixtures.items:
        assert fixture_item.expected.page is None
        assert fixture_item.expected.section is None


def test_bundled_example_fixture_loads(tmp_path: Path) -> None:
    """The committed example fixture must parse against agents/examples/."""
    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "docs" / "eval" / "fixtures" / "example.yaml"
    agents_examples = repo_root / "agents" / "examples"

    fixtures = load_fixtures(example, agents_dir=agents_examples)

    assert fixtures.agent == "scientist"
    assert len(fixtures.items) >= 2
    ids = {item.id for item in fixtures.items}
    assert len(ids) == len(fixtures.items)
