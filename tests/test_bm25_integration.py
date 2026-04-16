"""Integration test: mini-corpus BM25 artifacts load and match the
committed fixture (Story 19.1, Task 9)."""

from pathlib import Path

from grounding.bm25 import (
    BM25_MAP_FILENAME,
    BM25_PICKLE_FILENAME,
    FORMAT_VERSION,
    TOKENIZER_IDENTITY,
    load_bm25_index,
    search_bm25,
    tokenize,
)

FIXTURE_DIR = Path(__file__).parent / "eval_fixtures" / "mini_index"


def test_mini_index_bm25_artifacts_exist():
    assert (FIXTURE_DIR / BM25_PICKLE_FILENAME).exists()
    assert (FIXTURE_DIR / BM25_MAP_FILENAME).exists()


def test_mini_index_bm25_loads_and_has_five_chunks():
    idx = load_bm25_index(FIXTURE_DIR)
    assert idx is not None
    assert idx.bm25 is not None
    assert idx.chunk_map["format_version"] == FORMAT_VERSION
    assert idx.chunk_map["tokenizer"] == TOKENIZER_IDENTITY
    assert idx.chunk_map["total"] == 5
    assert idx.chunk_map["tombstone_count"] == 0
    chunk_ids = [c["chunk_id"] for c in idx.chunk_map["chunks"]]
    assert chunk_ids == [
        "doc-alpha_ch_0001",
        "doc-alpha_ch_0002",
        "doc-beta_ch_0001",
        "doc-beta_ch_0002",
        "doc-gamma_ch_0001",
    ]


def test_mini_index_bm25_search_recovers_expected_doc():
    idx = load_bm25_index(FIXTURE_DIR)
    # Query aligned with beta-study bootstrap content per mini_fixtures.yaml.
    results = search_bm25(idx, tokenize("bootstrap confidence interval"), top_k=3)
    assert results
    top_chunk_id = results[0][0]
    assert top_chunk_id.startswith("doc-beta")
