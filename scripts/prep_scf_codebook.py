#!/usr/bin/env python3
"""Prepare an SCF codebook .txt for corpus ingestion.

The Federal Reserve Survey of Consumer Finances codebook (e.g.
``codebk2022.txt``) is a fixed-width plain-text file with no Markdown
structure. Dropped into ``staging/`` as-is it (a) is rejected by the
watcher — ``.txt`` is an unsupported extension — and (b) even renamed to
``.md`` chunks badly: the splitter cuts mid-variable-entry and every chunk
lands with ``section_heading: null``.

This script makes the codebook's existing structure explicit so the
``grounding`` chunker can exploit it:

* Each *run* of consecutive variable-definition lines (``X<digits>`` at
  column 0) is prefixed with a ``## <codes>`` heading. SCF stacks related
  codes — ``X411(#1)`` / ``X419(#2)`` / ``X425(#3)`` share one question
  block — so a run becomes one section.
* Each section name delimited by dashed rules (``PRINCIPAL RESIDENCE``,
  ``OTHER CONSUMER LOANS``, …) is promoted to a ``# <name>`` heading.

The output is a ``.md`` file ready to drop into ``staging/<collection>/``.
Body text is preserved verbatim — only heading lines are inserted.

Usage::

    python scripts/prep_scf_codebook.py ~/Desktop/codebk2022.txt
    python scripts/prep_scf_codebook.py in.txt -o ~/staging/research-methods/codebk2022.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A variable-definition line: an SCF X-code at column 0 (no leading space).
_VAR_LINE = re.compile(r"^X\d")
# A horizontal rule delimiting a section name.
_DASHES = re.compile(r"^-{10,}\s*$")
# Codebook pads the code field, then >=2 spaces, then the question text.
_CODE_FIELD = re.compile(r"\s{2,}")

_HEADING_MAX = 120


def _heading_for_run(run_lines: list[str]) -> str:
    """Build the ``## `` heading text for a run of variable-header lines.

    Takes the code field (everything before the first 2+-space gap) of each
    line in the run and joins them — so a stacked ``X411(#1)`` / ``X419(#2)``
    / ``X425(#3)`` block becomes ``X411(#1) X419(#2) X425(#3)``.
    """
    codes = [_CODE_FIELD.split(line, 1)[0].strip() for line in run_lines]
    heading = " ".join(c for c in codes if c)
    if len(heading) > _HEADING_MAX:
        heading = heading[: _HEADING_MAX - 1].rstrip() + "…"
    return heading


def prep_codebook(text: str) -> tuple[str, int, int]:
    """Insert Markdown headings into raw SCF-codebook text.

    Returns ``(markdown_text, n_variable_sections, n_named_sections)``.
    """
    lines = text.split("\n")
    n = len(lines)
    out: list[str] = []
    var_sections = 0
    named_sections = 0
    i = 0
    while i < n:
        line = lines[i]
        prev_dashes = i > 0 and _DASHES.match(lines[i - 1]) is not None
        next_dashes = i + 1 < n and _DASHES.match(lines[i + 1]) is not None

        # Section name: a non-indented, non-code line fenced by dashed rules.
        if (
            line.strip()
            and not line[0].isspace()
            and not _VAR_LINE.match(line)
            and not _DASHES.match(line)
            and prev_dashes
            and next_dashes
        ):
            out.append("")
            out.append(f"# {line.strip()}")
            out.append(line)
            named_sections += 1
            i += 1
            continue

        # Start of a run of consecutive variable-definition lines.
        if _VAR_LINE.match(line) and (i == 0 or not _VAR_LINE.match(lines[i - 1])):
            j = i
            while j < n and _VAR_LINE.match(lines[j]):
                j += 1
            out.append("")
            out.append(f"## {_heading_for_run(lines[i:j])}")
            var_sections += 1
            # fall through — the run's lines are emitted normally below

        out.append(line)
        i += 1

    return "\n".join(out), var_sections, named_sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="SCF codebook .txt file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output .md path (default: input with .md suffix)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    output = args.output or args.input.with_suffix(".md")
    text = args.input.read_text(encoding="utf-8", errors="replace")
    markdown, var_sections, named_sections = prep_codebook(text)
    output.write_text(markdown, encoding="utf-8")

    print(f"wrote {output}")
    print(f"  {named_sections} named section(s)  -> '# ' headings")
    print(f"  {var_sections} variable entr(ies)  -> '## ' headings")
    print(f"  {len(text):,} -> {len(markdown):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
