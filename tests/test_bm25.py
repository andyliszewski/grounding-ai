"""Unit tests for grounding.bm25 (Story 19.1)."""

import json
import pickle
from pathlib import Path

import pytest

from grounding.bm25 import (
    BM25FormatError,
    BM25_MAP_FILENAME,
    BM25_PICKLE_FILENAME,
    FORMAT_VERSION,
    TOKENIZER_IDENTITY,
    append_to_bm25_index,
    load_bm25_index,
    search_bm25,
    tokenize,
    tombstone_bm25_documents,
    write_bm25_index,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("MiXeD.CaSe?!") == ["mixed", "case"]


def test_tokenize_preserves_code_identifiers_as_single_token():
    # `\w+` treats the CamelCase identifier as one token.
    assert tokenize("FooBarV2Client") == ["foobarv2client"]


def test_tokenize_keeps_numeric_tokens():
    assert tokenize("release v2 on 2026-04-15") == ["release", "v2", "on", "2026", "04", "15"]


def test_tokenize_unicode_letters():
    # \w with UNICODE flag accepts letter characters.
    assert tokenize("café résumé") == ["café", "résumé"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_write_and_load_round_trip(tmp_path):
    bodies = ["quick brown fox", "lazy dog jumps", "fox and hound"]
    ids = ["c1", "c2", "c3"]
    doc_ids = ["d1", "d1", "d2"]
    write_bm25_index(bodies, ids, tmp_path, chunk_doc_ids=doc_ids)

    assert (tmp_path / BM25_PICKLE_FILENAME).exists()
    assert (tmp_path / BM25_MAP_FILENAME).exists()

    idx = load_bm25_index(tmp_path)
    assert idx is not None
    assert idx.chunk_map["format_version"] == FORMAT_VERSION
    assert idx.chunk_map["tokenizer"] == TOKENIZER_IDENTITY
    assert idx.chunk_map["total"] == 3
    assert [c["chunk_id"] for c in idx.chunk_map["chunks"]] == ids
    assert [c["doc_id"] for c in idx.chunk_map["chunks"]] == doc_ids


def test_tokenizer_identity_mismatch_raises(tmp_path):
    write_bm25_index(["some text"], ["c1"], tmp_path)
    map_path = tmp_path / BM25_MAP_FILENAME
    m = json.loads(map_path.read_text())
    m["tokenizer"] = "porter_v1"
    map_path.write_text(json.dumps(m))

    with pytest.raises(BM25FormatError) as excinfo:
        load_bm25_index(tmp_path)
    assert "rebuild" in str(excinfo.value).lower()


def test_format_version_mismatch_raises(tmp_path):
    write_bm25_index(["some text"], ["c1"], tmp_path)
    map_path = tmp_path / BM25_MAP_FILENAME
    m = json.loads(map_path.read_text())
    m["format_version"] = 99
    map_path.write_text(json.dumps(m))

    with pytest.raises(BM25FormatError) as excinfo:
        load_bm25_index(tmp_path)
    msg = str(excinfo.value)
    assert "99" in msg or "v1" in msg.lower()


def test_append_preserves_existing_chunks(tmp_path):
    write_bm25_index(
        ["one fish two fish", "red fish blue fish"],
        ["c1", "c2"],
        tmp_path,
        chunk_doc_ids=["d1", "d1"],
    )
    added = append_to_bm25_index(
        ["green eggs and ham"], ["c3"], tmp_path, new_doc_ids=["d2"]
    )
    assert added == 1

    idx = load_bm25_index(tmp_path)
    ids = [c["chunk_id"] for c in idx.chunk_map["chunks"]]
    assert ids == ["c1", "c2", "c3"]
    # bm25_index values match position
    assert [c["bm25_index"] for c in idx.chunk_map["chunks"]] == [0, 1, 2]


def test_append_when_index_missing_creates_new(tmp_path):
    added = append_to_bm25_index(
        ["fresh start body"], ["c1"], tmp_path, new_doc_ids=["d1"]
    )
    assert added == 1
    idx = load_bm25_index(tmp_path)
    assert idx.chunk_map["total"] == 1


def test_tombstone_filters_in_search(tmp_path):
    bodies = ["alpha bravo charlie", "delta echo foxtrot", "alpha echo india"]
    ids = ["c1", "c2", "c3"]
    doc_ids = ["d1", "d2", "d3"]
    write_bm25_index(bodies, ids, tmp_path, chunk_doc_ids=doc_ids)

    n = tombstone_bm25_documents(["d1"], tmp_path)
    assert n == 1

    idx = load_bm25_index(tmp_path)
    results = search_bm25(idx, tokenize("alpha"), top_k=5)
    result_ids = [r[0] for r in results]
    assert "c1" not in result_ids
    assert "c3" in result_ids


def test_search_returns_tuples_with_1_indexed_rank(tmp_path):
    bodies = ["bootstrap confidence interval", "neural network transformer", "bootstrap methods paper"]
    ids = ["c1", "c2", "c3"]
    write_bm25_index(bodies, ids, tmp_path, chunk_doc_ids=["d1", "d2", "d3"])

    idx = load_bm25_index(tmp_path)
    results = search_bm25(idx, tokenize("bootstrap"), top_k=3)
    assert len(results) >= 1
    # Each result: (chunk_id, rank, score)
    ranks = [r[1] for r in results]
    assert ranks == list(range(1, len(results) + 1))
    # All scores are floats
    for r in results:
        assert isinstance(r[0], str)
        assert isinstance(r[1], int)
        assert isinstance(r[2], float)


def test_empty_corpus_full_build_no_crash(tmp_path):
    write_bm25_index([], [], tmp_path)
    idx = load_bm25_index(tmp_path)
    assert idx is not None
    assert idx.bm25 is None
    assert idx.chunk_map["total"] == 0
    assert search_bm25(idx, tokenize("anything"), top_k=5) == []


def test_single_chunk_corpus_search(tmp_path):
    write_bm25_index(["lonely document content"], ["c1"], tmp_path, chunk_doc_ids=["d1"])
    idx = load_bm25_index(tmp_path)
    results = search_bm25(idx, tokenize("lonely"), top_k=5)
    assert len(results) == 1
    assert results[0][0] == "c1"
    assert results[0][1] == 1


def test_load_missing_files_returns_none(tmp_path):
    assert load_bm25_index(tmp_path) is None
    # Only one file present still yields None.
    (tmp_path / BM25_MAP_FILENAME).write_text("{}")
    assert load_bm25_index(tmp_path) is None


def test_tombstone_updates_count_and_timestamp(tmp_path):
    bodies = ["a b c", "d e f"]
    write_bm25_index(bodies, ["c1", "c2"], tmp_path, chunk_doc_ids=["d1", "d2"])
    tombstone_bm25_documents(["d1"], tmp_path)
    m = json.loads((tmp_path / BM25_MAP_FILENAME).read_text())
    assert m["tombstone_count"] == 1
    assert any(c["deleted_utc"] is not None for c in m["chunks"] if c["chunk_id"] == "c1")


def test_tombstone_idempotent(tmp_path):
    write_bm25_index(["body one", "body two"], ["c1", "c2"], tmp_path, chunk_doc_ids=["d1", "d2"])
    assert tombstone_bm25_documents(["d1"], tmp_path) == 1
    # Second call should not double-increment.
    assert tombstone_bm25_documents(["d1"], tmp_path) == 0
    m = json.loads((tmp_path / BM25_MAP_FILENAME).read_text())
    assert m["tombstone_count"] == 1
