"""Chunk metadata helpers for grounding.

Implements Epic 3 Story 3.3 by generating YAML front matter for chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

import yaml

logger = logging.getLogger("grounding.chunk_metadata")


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata describing a single Markdown chunk."""

    doc_id: str
    source: str
    chunk_id: str
    index: int
    page_start: Optional[int]
    page_end: Optional[int]
    content_hash: str
    created_utc: str
    section_heading: Optional[str] = None
    has_embedding: bool = False
    formula_count: Optional[int] = None
    inline_formula_count: Optional[int] = None
    display_formula_count: Optional[int] = None
    formula_ids: Optional[list[str]] = None

    def items(self) -> Sequence[tuple[str, Any]]:
        """
        Ordered key/value pairs for YAML rendering.

        Returns:
            Sequence of metadata pairs in deterministic order.
        """
        base_items: list[tuple[str, Any]] = [
            ("doc_id", self.doc_id),
            ("source", self.source),
            ("chunk_id", self.chunk_id),
            ("page_start", self.page_start),
            ("page_end", self.page_end),
            ("hash", self.content_hash),
            ("created_utc", self.created_utc),
        ]
        if self.section_heading is not None:
            base_items.append(("section_heading", self.section_heading))
        if self.has_embedding:
            base_items.append(("has_embedding", self.has_embedding))
        if self.formula_count is not None:
            base_items.append(("formula_count", self.formula_count))
        if self.inline_formula_count is not None:
            base_items.append(("inline_formula_count", self.inline_formula_count))
        if self.display_formula_count is not None:
            base_items.append(("display_formula_count", self.display_formula_count))
        if self.formula_ids is not None:
            base_items.append(("formula_ids", self.formula_ids))
        return base_items


def build_chunk_metadata(
    *,
    doc_id: str,
    source: str,
    chunk_index: int,
    chunk_hash: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    section_heading: Optional[str] = None,
    created_utc: Optional[datetime] = None,
    has_embedding: bool = False,
    formula_count: Optional[int] = None,
    inline_formula_count: Optional[int] = None,
    display_formula_count: Optional[int] = None,
    formula_ids: Optional[list[str]] = None,
) -> ChunkMetadata:
    """
    Construct chunk metadata from pipeline context.

    Args:
        doc_id: Document identifier (8 character SHA-1 prefix).
        source: Original source filename.
        chunk_index: 1-based chunk index.
        chunk_hash: Hash of chunk content (excluding YAML).
        page_start: Starting page number (if known).
        page_end: Ending page number (if known).
        section_heading: Optional nearest section heading.
        created_utc: Optional timestamp; defaults to current UTC time.
        has_embedding: Whether this chunk has an associated embedding.
        formula_count: Total number of formulas in this chunk.
        inline_formula_count: Number of inline formulas in this chunk.
        display_formula_count: Number of display formulas in this chunk.
        formula_ids: List of formula IDs present in this chunk.

    Returns:
        ChunkMetadata instance ready for rendering.
    """
    if chunk_index < 1:
        raise ValueError("chunk_index must be >= 1")

    timestamp = created_utc or datetime.now(timezone.utc)
    created_iso = timestamp.replace(microsecond=0).isoformat()

    if page_start is None or page_end is None:
        logger.debug(
            "Chunk %s missing page metadata (start=%s end=%s)",
            f"{doc_id}-{chunk_index:04d}",
            page_start,
            page_end,
        )

    return ChunkMetadata(
        doc_id=doc_id,
        source=source,
        chunk_id=f"{doc_id}-{chunk_index:04d}",
        index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        content_hash=chunk_hash,
        created_utc=created_iso,
        section_heading=section_heading,
        has_embedding=has_embedding,
        formula_count=formula_count,
        inline_formula_count=inline_formula_count,
        display_formula_count=display_formula_count,
        formula_ids=formula_ids,
    )


def render_chunk(metadata: ChunkMetadata, content: str) -> str:
    """
    Render chunk metadata and content as YAML front matter plus Markdown body.

    Args:
        metadata: Chunk metadata describing this chunk.
        content: Markdown content of the chunk (without front matter).

    Returns:
        String containing YAML front matter followed by the chunk content.
    """
    lines = ["---"]
    for key, value in metadata.items():
        serialized = _serialize_yaml_value(value)
        lines.append(f"{key}: {serialized}")
    lines.append("---")

    front_matter = "\n".join(lines)
    if not front_matter.endswith("\n"):
        front_matter += "\n"

    # Ensure content ends with newline for deterministic formatting.
    if content and not content.endswith("\n"):
        content = f"{content}\n"

    return f"{front_matter}\n{content}"


def _serialize_yaml_value(value: Any) -> str:
    """Serialize individual scalar values for YAML front matter."""
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    dumped = yaml.safe_dump(
        value,
        default_flow_style=True,
        allow_unicode=False,
    ).strip()
    if dumped.endswith("..."):
        dumped = dumped[:-3].strip()
    return dumped
