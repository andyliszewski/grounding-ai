"""Markdown formatter adapter for grounding.

Converts parsed elements to normalized Markdown by joining text content.

Element-map API (Story 17.1)
----------------------------
`format_markdown_with_map` is an additive companion to `format_markdown`. It
emits an ordered, non-overlapping element map (`FormattedElement`) alongside
the Markdown string so downstream chunking (Story 17.2) can derive
`page_start`, `page_end`, and `section_heading` per chunk.

Heading detection rules:
  * Unstructured elements: `element.category == "Title"` (or `"Header"`) is
    treated as a heading. Heading level is taken from
    `element.metadata.category_depth` (0-indexed) when present, else 1.
  * `TextElement` inputs (pdftotext fallback, EPUB ebooklib fallback, raw
    Markdown ingest): no `page_number` is available; a single leading
    `^#{1,6}\\s+...$` line is treated as a heading block (one element per
    heading or paragraph). pdftotext content has no `#` lines, so its stack
    stays empty.

Marker-produced Markdown: out of scope for 17.1 — page metadata lives in
Marker's JSON sidecar, not the Markdown it returns. Marker output reaching
the formatter as a single-string `TextElement` is treated as raw Markdown
(no pages, headings from `#`/`##`). Story 17.2 will revisit.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from grounding.parser import TextElement

logger = logging.getLogger("grounding.formatter")

_HEADING_FIRST_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*(?:\n|$)")
_UNSTRUCTURED_HEADING_CATEGORIES = frozenset({"Title", "Header"})


@dataclass(frozen=True)
class FormattedElement:
    """Record describing one source block's span in the formatted Markdown."""

    char_start: int
    char_end: int
    page_number: Optional[int]
    heading_stack: Tuple[str, ...]


@dataclass(frozen=True)
class FormatResult:
    """Return value of :func:`format_markdown_with_map`."""

    markdown: str
    elements: Tuple[FormattedElement, ...]


def _is_text_elements(elements: Sequence[Any]) -> bool:
    """Check if elements are TextElement (from fast pdftotext extraction)."""
    return elements and isinstance(elements[0], TextElement)


class FormatError(Exception):
    """Raised when formatting fails."""

    def __init__(self, source_name: str | None, message: str):
        self.source_name = source_name or "unknown"
        super().__init__(message)


def format_markdown(
    elements: Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    allow_plaintext_fallback: bool = False,  # Kept for API compatibility, ignored
    source_name: str | None = None,
) -> str:
    """
    Convert parsed elements to Markdown by joining text content.

    Args:
        elements: Sequence of parsed elements (TextElement or unstructured).
        metadata: Optional metadata dict to emit as front matter.
        allow_plaintext_fallback: Ignored (kept for API compatibility).
        source_name: Optional identifier used for logging (e.g., slug).

    Returns:
        Normalized Markdown string.

    Raises:
        TypeError: If elements is not a sequence.
    """
    if isinstance(elements, (str, bytes)):
        raise TypeError("elements must be a sequence of parsed items")
    if not isinstance(elements, Sequence):
        raise TypeError("elements must be a sequence of parsed items")

    metadata_copy: MutableMapping[str, Any] | None = dict(metadata) if metadata else None
    start = time.perf_counter()

    # Join all element text directly
    if _is_text_elements(elements):
        logger.debug("Formatting TextElements for %s", source_name or "unknown")
    else:
        logger.debug("Formatting unstructured elements for %s", source_name or "unknown")

    markdown = _join_plaintext(elements)

    elapsed_ms = (time.perf_counter() - start) * 1000
    normalized = _normalize_markdown(markdown)
    if metadata_copy:
        normalized = _render_front_matter(metadata_copy) + normalized

    logger.info(
        "Formatted %s element_count=%d elapsed_ms=%.2f",
        source_name or "unknown",
        len(elements),
        elapsed_ms,
    )
    return normalized


def _normalize_markdown(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    # Remove trailing blank lines for determinism
    while lines and lines[-1] == "":
        lines.pop()
    normalized = "\n".join(lines)
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _render_front_matter(metadata: Mapping[str, Any]) -> str:
    lines = ["---"]
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n"


def _join_plaintext(elements: Sequence[Any]) -> str:
    chunks = []
    for element in elements:
        text = getattr(element, "text", None)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Element map API (Story 17.1)
# ---------------------------------------------------------------------------


def _normalize_block_text(text: str) -> str:
    """Per-element normalization mirroring `_normalize_markdown`.

    Normalizes line endings, right-strips each line, and strips trailing
    newlines so the block can be joined with a fixed ``\\n\\n`` separator
    without producing extra blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n")


def _compute_heading_stacks(
    per_element_heading: Sequence[Optional[Tuple[int, str]]],
) -> List[Tuple[str, ...]]:
    """Compute the heading stack at each element position.

    Args:
        per_element_heading: One entry per kept element. ``None`` if the
            element is a body block; ``(level, text)`` if it is a heading.

    Returns:
        A list the same length as input, with the heading stack tuple for
        each element. Heading elements include themselves in their stack.
    """
    stacks: List[Tuple[str, ...]] = []
    current: List[Tuple[int, str]] = []
    for item in per_element_heading:
        if item is not None:
            level, text = item
            while current and current[-1][0] >= level:
                current.pop()
            current.append((level, text))
        stacks.append(tuple(t for _, t in current))
    return stacks


def _unstructured_page_number(element: Any) -> Optional[int]:
    md = getattr(element, "metadata", None)
    if md is None:
        return None
    page = getattr(md, "page_number", None)
    if page is None and isinstance(md, Mapping):
        page = md.get("page_number")
    return page if isinstance(page, int) else None


def _unstructured_heading_level(element: Any) -> int:
    md = getattr(element, "metadata", None)
    depth = None
    if md is not None:
        depth = getattr(md, "category_depth", None)
        if depth is None and isinstance(md, Mapping):
            depth = md.get("category_depth")
    if isinstance(depth, int) and depth >= 0:
        return depth + 1
    return 1


def _classify_element(
    element: Any, normalized_text: str
) -> Tuple[Optional[int], Optional[Tuple[int, str]]]:
    """Return ``(page_number, heading_info)`` for one element.

    ``heading_info`` is ``(level, text)`` if the element is a heading, else
    ``None``. TextElement inputs never yield a page number.
    """
    if isinstance(element, TextElement):
        match = _HEADING_FIRST_LINE_RE.match(normalized_text)
        if match and match.group(0).strip() == normalized_text.strip():
            level = len(match.group(1))
            return None, (level, match.group(2).strip())
        return None, None

    category = getattr(element, "category", None) or ""
    page_number = _unstructured_page_number(element)
    if category in _UNSTRUCTURED_HEADING_CATEGORIES:
        level = _unstructured_heading_level(element)
        heading_text = normalized_text.split("\n", 1)[0].strip()
        return page_number, (level, heading_text)
    return page_number, None


def format_markdown_with_map(
    elements: Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    source_name: str | None = None,
) -> FormatResult:
    """Format elements to Markdown and return a parallel element map.

    Additive companion to :func:`format_markdown`. Returns the formatted
    Markdown along with an ordered tuple of :class:`FormattedElement`
    records, one per non-empty input block, carrying ``page_number`` and
    ``heading_stack`` metadata. Spans are ordered by ``char_start`` and
    non-overlapping; the union covers every non-whitespace position in the
    body of the returned Markdown (front matter, if any, precedes the
    first element).

    See the module docstring for heading-detection and page-number rules.
    """
    if isinstance(elements, (str, bytes)):
        raise TypeError("elements must be a sequence of parsed items")
    if not isinstance(elements, Sequence):
        raise TypeError("elements must be a sequence of parsed items")

    start = time.perf_counter()

    kept: List[Tuple[str, Optional[int], Optional[Tuple[int, str]]]] = []
    textelement_seen = False
    for element in elements:
        raw = getattr(element, "text", None)
        if not raw:
            continue
        normalized = _normalize_block_text(raw)
        if not normalized or not normalized.strip():
            continue
        if isinstance(element, TextElement):
            textelement_seen = True
        page_number, heading_info = _classify_element(element, normalized)
        kept.append((normalized, page_number, heading_info))

    stacks = _compute_heading_stacks([item[2] for item in kept])

    front_matter = ""
    if metadata:
        front_matter = _render_front_matter(dict(metadata))
    offset = len(front_matter)

    body_parts: List[str] = []
    formatted: List[FormattedElement] = []
    pos = 0
    for idx, (normalized, page_number, _heading_info) in enumerate(kept):
        if idx > 0:
            body_parts.append("\n\n")
            pos += 2
        span_start = offset + pos
        body_parts.append(normalized)
        pos += len(normalized)
        span_end = offset + pos
        formatted.append(
            FormattedElement(
                char_start=span_start,
                char_end=span_end,
                page_number=page_number,
                heading_stack=stacks[idx],
            )
        )
    body = "".join(body_parts)
    if body:
        body += "\n"
    markdown = front_matter + body

    if textelement_seen:
        logger.debug(
            "TextElement input for %s: page metadata unavailable on fallback path",
            source_name or "unknown",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Formatted-with-map %s element_count=%d mapped=%d elapsed_ms=%.2f",
        source_name or "unknown",
        len(elements),
        len(formatted),
        elapsed_ms,
    )
    return FormatResult(markdown=markdown, elements=tuple(formatted))


def _coverage_check(
    markdown: str, elements: Sequence[FormattedElement]
) -> bool:
    """Invariant check used by tests.

    Verifies that elements are ordered by ``char_start``, spans are
    non-overlapping, and every non-whitespace character between the first
    element's start and the last element's end is covered by some span.
    Inter-block whitespace may be unmapped.
    """
    if not elements:
        return True
    starts = [e.char_start for e in elements]
    if starts != sorted(starts):
        return False
    for prev, curr in zip(elements, elements[1:]):
        if prev.char_end > curr.char_start:
            return False
    body_start = elements[0].char_start
    body_end = elements[-1].char_end
    if body_end > len(markdown) or body_start < 0:
        return False
    covered = bytearray(body_end - body_start)
    for e in elements:
        for i in range(e.char_start - body_start, e.char_end - body_start):
            covered[i] = 1
    for i in range(body_end - body_start):
        if not covered[i] and not markdown[body_start + i].isspace():
            return False
    return True
