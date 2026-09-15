"""Tests for undersized-chunk merging (``ChunkConfig.min_chunk_size``).

The splitter emits one chunk per separator-delimited block that will not fit in
the current window, so a short block between two long ones is emitted alone.
That strands table captions and section titles in chunks of their own, where
they embed as a bare title carrying none of the data they name -- while the
data embeds carrying nothing that says what it is.

Covers:
- the real-world case: a table caption merges forward into its table.
- no emitted chunk is left under the threshold.
- ``min_chunk_size=0`` preserves the previous (unmerged) behaviour exactly.
- merged text is the faithful source slice, not a join (overlap not duplicated).
- offsets stay monotonic and aligned to the source.
- a trailing orphan folds backward.
- config validation and determinism.
"""
from __future__ import annotations

import pytest

from grounding.chunker import (
    ChunkConfig,
    _raw_split,
    split_markdown,
    split_markdown_with_map,
)


def _body(marker: str, length: int) -> str:
    """A block of `length` characters with no internal blank line."""
    unit = f"{marker} "
    return (unit * (length // len(unit) + 1))[:length].strip()


# A caption sandwiched between two blocks that each fill the window. This is the
# shape that produced the observed failure: the caption cannot join either
# neighbour, so it is emitted alone.
CAPTION = "Table 6.4.13 Properties of Cemented Carbides"
DOC = "\n\n".join([_body("alpha", 1180), CAPTION, _body("beta", 1180)])


def test_caption_merges_forward_into_the_block_it_names():
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    chunks = split_markdown(DOC, cfg)

    holding = [c for c in chunks if CAPTION in c]
    assert holding, "caption disappeared from the output"
    for chunk in holding:
        assert chunk.strip() != CAPTION, "caption still stranded in a chunk of its own"
    # It merged forward, into the block it introduces.
    assert any("beta" in c for c in holding)


def test_caption_is_stranded_without_merging():
    """Guards the premise: with merging off the caption really is orphaned."""
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=0)
    chunks = split_markdown(DOC, cfg)
    assert any(c.strip() == CAPTION for c in chunks)


def test_no_chunk_left_under_threshold():
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    for text in (DOC, "\n\n".join([_body("x", 1180), "tiny", "also short", _body("y", 1180)])):
        chunks = split_markdown(text, cfg)
        assert len(chunks) > 1, "test text should split into several chunks"
        for chunk in chunks:
            assert len(chunk.strip()) >= cfg.min_chunk_size


def test_min_chunk_size_zero_matches_raw_splitter():
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=0)
    assert split_markdown(DOC, cfg) == _raw_split(DOC, cfg)


def test_merged_text_is_the_source_slice_not_a_join():
    """Chunks overlap, so joining them would duplicate the overlap region."""
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    records = split_markdown_with_map(DOC, (), cfg)
    for rec in records:
        assert rec.text == DOC[rec.char_start : rec.char_end]


def test_offsets_stay_monotonic_and_ordered():
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    records = split_markdown_with_map(DOC, (), cfg)
    starts = [r.char_start for r in records]
    assert starts == sorted(starts)
    for rec in records:
        assert 0 <= rec.char_start <= rec.char_end <= len(DOC)


def test_trailing_orphan_folds_backward():
    """A short final block has nothing after it to absorb, so it folds back."""
    text = "\n\n".join([_body("alpha", 1180), _body("beta", 1180), "coda"])
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    chunks = split_markdown(text, cfg)
    assert not any(c.strip() == "coda" for c in chunks)
    assert any("coda" in c for c in chunks), "trailing text was dropped"
    for chunk in chunks:
        assert len(chunk.strip()) >= cfg.min_chunk_size


def test_document_shorter_than_threshold_survives():
    """A whole document under the threshold must not be dropped or merged away."""
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    chunks = split_markdown("short note", cfg)
    assert chunks == ["short note"]


def test_default_enables_merging():
    assert ChunkConfig().resolved_min_chunk_size() == 200


@pytest.mark.parametrize("bad", [-1, 1200, 5000])
def test_validation_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=bad).validate()


def test_deterministic_across_runs():
    cfg = ChunkConfig(chunk_size=1200, chunk_overlap=150, min_chunk_size=200)
    assert split_markdown(DOC, cfg) == split_markdown(DOC, cfg)
