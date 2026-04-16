"""Citation prefix formatting for retrieval output (Story 17.3).

Renders a compact bracketed prefix the LLM can cite directly, e.g.

    [alpha-paper, p.247, §3.2 Bootstrap Methods]
    [alpha-paper, p.247–249, §3.2 Bootstrap Methods]
    [alpha-paper, p.247]
    [beta-study, §4. Methods]
    [gamma-notes]

Pure function, no I/O. Page ranges use U+2013 EN DASH. The `source`
argument may be either a raw filename (slugified via
`grounding.utils.slugify`) or an already-slugified string (returned
as-is when it has no file extension and already matches kebab-case).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .utils import slugify

EN_DASH = "\u2013"

_SLUG_SHAPE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _derive_slug(source: str) -> str:
    """Return slug for a `source` string that may be a filename or slug.

    Rules:
    1. If `source` has a file extension (e.g. ``alpha-paper.pdf``),
       slugify the stem.
    2. If `source` is already kebab-case with no extension, pass through.
    3. Otherwise, slugify defensively.
    """
    if not source:
        return ""
    has_extension = bool(Path(source).suffix)
    if not has_extension and _SLUG_SHAPE.match(source):
        return source
    return slugify(source)


def format_citation_prefix(
    source: str,
    page_start: Optional[int],
    page_end: Optional[int],
    section_heading: Optional[str],
) -> str:
    """Format a bracketed citation prefix for a retrieved chunk.

    Variants (in priority order):
      * Full:        ``[<slug>, p.<start>, §<section>]``          both sides present, single page
      * Range:       ``[<slug>, p.<start>–<end>, §<section>]``    page range with section
      * Page-only:   ``[<slug>, p.<start>[–<end>]]``              section is null
      * Section-only:``[<slug>, §<section>]``                     pages are null
      * Slug-only:   ``[<slug>]``                                 all null

    Missing-field inputs degrade silently to whichever smaller variant
    applies. Never raises on ``None`` inputs.
    """
    slug = _derive_slug(source)

    page_fragment = _page_fragment(page_start, page_end)
    section_fragment = _section_fragment(section_heading)

    parts = [slug]
    if page_fragment:
        parts.append(page_fragment)
    if section_fragment:
        parts.append(section_fragment)

    return "[" + ", ".join(p for p in parts if p) + "]"


def _page_fragment(page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"p.{page_start}"
    return f"p.{page_start}{EN_DASH}{page_end}"


def _section_fragment(section_heading: Optional[str]) -> str:
    if section_heading is None:
        return ""
    heading = section_heading.strip()
    if not heading:
        return ""
    return f"§{heading}"
