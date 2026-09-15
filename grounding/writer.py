"""Output writer helpers for grounding.

Implements Epic 4 Story 4.1 by persisting doc.md and chunk files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from grounding.pipeline import FileContext
from grounding.utils import atomic_write, ensure_dir

logger = logging.getLogger("grounding.writer")


def write_document(
    context: FileContext,
    markdown: str,
    chunks: Sequence[str],
    *,
    dry_run: bool = False,
) -> None:
    """
    Write document Markdown and chunk files to disk.

    Args:
        context: FileContext containing slug and output path information.
        markdown: Full document Markdown (already normalized).
        chunks: Ordered sequence of chunk strings (with YAML front matter).
        dry_run: When True, log operations without touching disk.

    Raises:
        ValueError: If `context.output_path` is missing when not in dry-run mode.
    """
    doc_path = context.output_path
    if doc_path is None:
        message = "FileContext.output_path is required to write document outputs"
        if dry_run:
            logger.info("Dry-run: %s; skipping writes for slug=%s", message, context.slug)
            return
        raise ValueError(message)

    chunk_dir = doc_path.parent / "chunks"

    if dry_run:
        _log_dry_run(context.slug, doc_path, chunk_dir, len(chunks))
        return

    ensure_dir(doc_path.parent)
    ensure_dir(chunk_dir)

    # Story 23.1: clear chunk files from any prior version of THIS document
    # before writing the new set. A revision that produces fewer chunks than the
    # previous version would otherwise leave orphaned ch_*.md behind, which the
    # embedder's directory glob (cli.py) would pick up and embed as stale content
    # citing pages/sections that no longer exist. Upstream slug-collision
    # detection (Story 23.2, controller.py) guarantees this chunk dir belongs to
    # the same document being written, so this only ever removes our own stale
    # chunks. Scoped to ch_*.md — doc.md, meta.yaml, music/ and formulas/ outputs
    # are untouched.
    stale_chunks = sorted(chunk_dir.glob("ch_*.md"))
    for stale in stale_chunks:
        stale.unlink()
    if stale_chunks:
        logger.debug(
            "Removed %d stale chunk file(s) before re-write slug=%s",
            len(stale_chunks),
            context.slug,
        )

    atomic_write(doc_path, markdown)

    for index, chunk in enumerate(chunks, start=1):
        chunk_name = f"ch_{index:04d}.md"
        chunk_path = chunk_dir / chunk_name
        atomic_write(chunk_path, chunk)

    logger.info(
        "Wrote document outputs slug=%s doc=%s chunks=%d",
        context.slug,
        doc_path,
        len(chunks),
    )


def _log_dry_run(slug: str, doc_path: Path, chunk_dir: Path, chunk_count: int) -> None:
    logger.info(
        "Dry-run: would write doc=%s and %d chunk(s) under %s for slug=%s",
        doc_path,
        chunk_count,
        chunk_dir,
        slug,
    )
