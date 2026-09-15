"""Chunking utilities for grounding.

Implements Epic 3 Story 3.1 by wrapping LangChain's RecursiveCharacterTextSplitter.

Provenance API (Story 17.2)
---------------------------
``split_markdown_with_map`` is an additive companion to ``split_markdown`` that
returns :class:`ChunkWithProvenance` records carrying ``char_start`` / ``char_end``
offsets into the source Markdown. ``derive_chunk_metadata`` consumes those
offsets together with a :class:`FormattedElement` map (from Story 17.1) to
compute ``page_start``, ``page_end``, and ``section_heading`` for each chunk.

Boundary convention: when a chunk straddles a heading boundary, the **earlier**
section wins — ``section_heading`` reflects the section the chunk's start is
inside, not the section its end crosses into. See AC #6 of Story 17.2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from grounding.formatter import FormattedElement

logger = logging.getLogger("grounding.chunker")

#: Default merge threshold, in characters. Applied only when it stays comfortably
#: below ``chunk_size`` -- see :meth:`ChunkConfig.resolved_min_chunk_size`.
DEFAULT_MIN_CHUNK_SIZE = 200


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for Markdown chunking."""

    chunk_size: int = 1_200
    chunk_overlap: int = 150
    separators: tuple[str, ...] = ("\n\n", "\n", " ")
    min_chunk_size: Optional[int] = None

    def resolved_min_chunk_size(self) -> int:
        """The merge threshold actually applied.

        ``None`` (the default) means auto: ``DEFAULT_MIN_CHUNK_SIZE``, held below
        ``chunk_size`` so that a caller who shrinks ``chunk_size`` alone -- as
        tests and small-document callers do -- never lands on a threshold their
        chunks cannot clear. An explicit value is used as given.
        """
        if self.min_chunk_size is None:
            return min(DEFAULT_MIN_CHUNK_SIZE, self.chunk_size // 4)
        return self.min_chunk_size

    def validate(self) -> None:
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size is not None:
            if self.min_chunk_size < 0:
                raise ValueError("min_chunk_size cannot be negative")
            if self.min_chunk_size >= self.chunk_size:
                raise ValueError("min_chunk_size must be smaller than chunk_size")
        if not self.separators:
            raise ValueError("separators must contain at least one separator string")
        for separator in self.separators:
            if not isinstance(separator, str):
                raise ValueError("all separators must be strings")


def split_markdown(text: str, config: ChunkConfig | None = None) -> List[str]:
    """
    Split Markdown into deterministic chunks using LangChain.

    Args:
        text: Normalized Markdown document to chunk.
        config: Optional ChunkConfig overriding defaults.

    Returns:
        List of chunk strings ready for metadata decoration.

    Raises:
        TypeError: When text is not a string.
        ValueError: When configuration values are invalid.
    """
    return [rec.text for rec in split_markdown_with_map(text, (), config)]


def _raw_split(text: str, cfg: ChunkConfig) -> List[str]:
    """Run the LangChain splitter. No merging, no logging."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Provenance API (Story 17.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkWithProvenance:
    """A chunk carrying its character offsets into the source Markdown."""

    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ChunkProvenance:
    """Derived page/section metadata for a chunk."""

    page_start: Optional[int]
    page_end: Optional[int]
    section_heading: Optional[str]


def split_markdown_with_map(
    text: str,
    elements: Sequence[FormattedElement],
    config: ChunkConfig | None = None,
) -> List[ChunkWithProvenance]:
    """Split Markdown and return chunks with character offsets.

    Runs the same LangChain splitter as :func:`split_markdown`, then computes
    each chunk's ``char_start`` / ``char_end`` by locating successive chunks in
    ``text`` with a running cursor. The cursor scan is deterministic and
    correctly handles chunk overlap: chunk N+1 begins roughly at
    ``char_end(N) - chunk_overlap`` regardless of duplicate text elsewhere in
    the document, because the search always starts from the current cursor.

    The ``elements`` parameter is accepted for API symmetry with
    :func:`derive_chunk_metadata` but is not used for splitting — it is the
    caller's responsibility to pass the same element map to
    ``derive_chunk_metadata`` for each chunk.

    Args:
        text: The full Markdown text (may include front matter).
        elements: Ordered element map produced by ``format_markdown_with_map``.
            Accepted for symmetry; not used by the splitter itself.
        config: Optional :class:`ChunkConfig`.

    Returns:
        A list of :class:`ChunkWithProvenance`, ordered by ``char_start`` and
        with monotonically non-decreasing offsets.

    Raises:
        TypeError: When ``text`` is not a string.
        ValueError: When chunk config is invalid or a chunk cannot be located
            in ``text`` (indicates the splitter mutated content — should not
            happen with ``RecursiveCharacterTextSplitter``).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string containing Markdown content")
    del elements  # not used here; kept for API symmetry

    cfg = config or ChunkConfig()
    cfg.validate()

    raw_chunks = _raw_split(text, cfg)

    records: List[ChunkWithProvenance] = []
    cursor = 0
    for idx, chunk in enumerate(raw_chunks):
        if chunk == "":
            # Empty chunk: record at current cursor, do not advance search.
            records.append(ChunkWithProvenance(text="", char_start=cursor, char_end=cursor))
            continue
        pos = text.find(chunk, cursor)
        if pos < 0:
            raise ValueError(
                f"chunk {idx} not found in source text at or after cursor {cursor}; "
                "splitter output is inconsistent with input"
            )
        char_start = pos
        char_end = pos + len(chunk)
        records.append(
            ChunkWithProvenance(text=chunk, char_start=char_start, char_end=char_end)
        )
        # Advance cursor past the non-overlap portion. With chunk_overlap=O
        # and chunk of length L, next chunk should begin at char_end - O,
        # but we clamp to char_start+1 to guarantee monotonic progress.
        next_cursor = char_end - cfg.chunk_overlap
        cursor = max(char_start + 1, next_cursor)

    records = _merge_undersized(records, text, cfg)

    # Invariant: monotonic non-decreasing char_start.
    prev = -1
    for rec in records:
        if rec.char_start < prev:
            raise ValueError("chunk char_starts are not monotonically non-decreasing")
        prev = rec.char_start

    logger.info(
        "Chunked Markdown length=%d chunk_size=%d chunk_overlap=%d "
        "min_chunk_size=%d raw_chunks=%d chunks=%d",
        len(text),
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.resolved_min_chunk_size(),
        len(raw_chunks),
        len(records),
    )

    return records


def _span_text(records: Sequence[ChunkWithProvenance], text: str, i: int, j: int) -> str:
    """The source slice spanning records ``i`` through ``j`` inclusive.

    Slicing the source rather than joining the chunk strings is what keeps the
    merged text faithful: successive chunks overlap by ``chunk_overlap``, so a
    naive join would duplicate the overlapping region.
    """
    return text[records[i].char_start : records[j].char_end]


def _merge_undersized(
    records: List[ChunkWithProvenance],
    text: str,
    cfg: ChunkConfig,
) -> List[ChunkWithProvenance]:
    """Fold chunks shorter than ``min_chunk_size`` into their neighbour.

    The splitter emits a chunk per separator-delimited block that will not fit
    in the current window, so a short block sandwiched between two long ones is
    emitted alone. In practice that strands exactly the text you most want kept
    with its body -- a table caption, a section title, a figure label -- in a
    chunk of its own, where it embeds as a bare title with none of the data it
    names, while the data embeds with nothing saying what it is.

    Undersized chunks are merged **forward** (into what follows), because these
    orphans are overwhelmingly labels introducing the block after them. A run of
    short chunks accumulates until the span clears the threshold. A trailing
    group with nothing left to absorb folds backward into its predecessor.

    Deterministic and offset-preserving: the merged record spans from the first
    member's ``char_start`` to the last member's ``char_end``.

    Size contract: merging necessarily produces chunks larger than ``chunk_size``
    -- an orphan can only be absorbed by a neighbour that is already full, which
    is precisely why the splitter stranded it. Growth is bounded: a group stops
    absorbing the moment it clears the threshold, so no emitted chunk exceeds
    ``chunk_size + min_chunk_size``.
    """
    threshold = cfg.resolved_min_chunk_size()
    if threshold <= 0 or not records:
        return records

    groups: List[Tuple[int, int]] = []
    i = 0
    n = len(records)
    while i < n:
        j = i
        while j < n - 1 and len(_span_text(records, text, i, j).strip()) < threshold:
            j += 1
        groups.append((i, j))
        i = j + 1

    # A trailing group can still be undersized: nothing follows it to absorb.
    if len(groups) > 1:
        start, end = groups[-1]
        if len(_span_text(records, text, start, end).strip()) < threshold:
            prev_start, _ = groups[-2]
            groups[-2] = (prev_start, end)
            groups.pop()

    merged: List[ChunkWithProvenance] = []
    for start, end in groups:
        if start == end:
            merged.append(records[start])
            continue
        merged.append(
            ChunkWithProvenance(
                text=_span_text(records, text, start, end),
                char_start=records[start].char_start,
                char_end=records[end].char_end,
            )
        )
    return merged


def derive_chunk_metadata(
    chunk: ChunkWithProvenance,
    elements: Sequence[FormattedElement],
) -> ChunkProvenance:
    """Derive page range and section heading for a chunk.

    - ``page_start`` / ``page_end``: min/max of ``page_number`` over elements
      whose span overlaps ``[chunk.char_start, chunk.char_end)``, ignoring
      ``None``. If no overlapping element has a page number, both are ``None``.
    - ``section_heading``: last entry of ``heading_stack`` of the element at or
      before ``chunk.char_start`` whose stack is non-empty. When a chunk
      straddles a heading boundary, the **earlier** section wins (the section
      the chunk's start is inside). If the chunk begins before any heading,
      returns ``None``.

    Pure function. No I/O. No INFO-level logging.
    """
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_heading: Optional[str] = None

    pages: List[int] = []
    for el in elements:
        if el.char_end <= chunk.char_start or el.char_start >= chunk.char_end:
            continue  # no overlap
        if el.page_number is not None:
            pages.append(el.page_number)
    if pages:
        page_start = min(pages)
        page_end = max(pages)

    # section_heading: element at or before chunk.char_start with non-empty stack.
    for el in elements:
        if el.char_start > chunk.char_start:
            break
        if el.heading_stack:
            section_heading = el.heading_stack[-1]

    return ChunkProvenance(
        page_start=page_start,
        page_end=page_end,
        section_heading=section_heading,
    )
