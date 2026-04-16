"""End-to-end integration test for Story 17.2 citation metadata.

Runs the real pipeline (format_markdown_with_map → split_markdown_with_map →
derive_chunk_metadata → render_chunk → writer) against a small set of
stubbed unstructured-style elements, then reads back the on-disk chunk YAML
and asserts that page_start / page_end / section_heading were populated as
expected.

Using monkeypatched parser output (rather than a binary PDF fixture) keeps
the test fast, deterministic, and independent of unstructured's per-PDF
classification heuristics while still exercising the full pipeline →
controller → writer path end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from grounding import pipeline as pipeline_mod
from grounding.controller import run_controller
from grounding.pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# Stub elements matching Unstructured's duck-typed shape.
# ---------------------------------------------------------------------------


@dataclass
class _StubMetadata:
    page_number: Optional[int] = None
    category_depth: Optional[int] = None


@dataclass
class _StubElement:
    text: str
    category: str = "NarrativeText"
    metadata: Any = field(default_factory=_StubMetadata)


def _title(text: str, *, page: int, level: int = 1) -> _StubElement:
    return _StubElement(
        text=text,
        category="Title",
        metadata=_StubMetadata(page_number=page, category_depth=level - 1),
    )


def _para(text: str, *, page: int) -> _StubElement:
    return _StubElement(
        text=text,
        category="NarrativeText",
        metadata=_StubMetadata(page_number=page),
    )


def _fake_pdf_elements():
    # 3-page document with a Section heading on page 2.
    body_p1 = " ".join(["Sentence on page one."] * 6)
    body_p2a = " ".join(["Methods body on page two."] * 6)
    body_p2b = " ".join(["More methods text on page two."] * 4)
    body_p3 = " ".join(["Concluding remarks on page three."] * 6)
    return [
        _title("Introduction", page=1, level=1),
        _para(body_p1, page=1),
        _title("Methods", page=2, level=1),
        _para(body_p2a, page=2),
        _para(body_p2b, page=2),
        _para(body_p3, page=3),
    ]


def test_pipeline_writes_populated_front_matter(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    dummy_pdf = input_dir / "tiny.pdf"
    dummy_pdf.write_bytes(b"%PDF-placeholder\n")  # scan_pdfs needs the file to exist

    def _fake_parse_pdf(path, ocr_mode="auto"):
        return _fake_pdf_elements()

    monkeypatch.setattr(pipeline_mod, "parse_pdf", _fake_parse_pdf)

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        parser="unstructured",
        ocr_mode="off",
        metadata={"chunk_size": 200, "chunk_overlap": 30},
    )
    run_controller(config, files=[dummy_pdf])

    # Inspect on-disk chunks.
    chunks_dir = output_dir / "tiny-pdf" / "chunks"
    if not chunks_dir.exists():
        # Slug follows slugify(path.name) which strips the extension; try without.
        candidates = [p for p in output_dir.iterdir() if p.is_dir()]
        assert candidates, f"no slug dir under {output_dir}"
        chunks_dir = candidates[0] / "chunks"
    chunk_files = sorted(chunks_dir.glob("ch_*.md"))
    assert chunk_files, "no chunks written"

    def _front_matter(path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"missing front matter in {path}"
        _, _, rest = text[4:].partition("---\n")
        fm_text = text[4:-len(rest) - 4]
        return yaml.safe_load(fm_text)

    parsed = [_front_matter(p) for p in chunk_files]

    # Every chunk carries page_start / page_end keys (may be None).
    for fm in parsed:
        assert "page_start" in fm
        assert "page_end" in fm

    # First chunk starts at offset 0 (inside the doc-level front matter,
    # before any element) — per AC #6 its section is None.
    assert parsed[0].get("section_heading") is None
    # But its overlapping elements are on page 1.
    assert parsed[0]["page_start"] == 1

    # At least one chunk cites Introduction on page 1.
    intro_chunks = [p for p in parsed if p.get("section_heading") == "Introduction"]
    assert intro_chunks, "expected at least one chunk under 'Introduction'"
    assert any(p["page_start"] == 1 for p in intro_chunks)

    # At least one chunk cites Methods on page 2.
    methods_chunks = [p for p in parsed if p.get("section_heading") == "Methods"]
    assert methods_chunks, "expected at least one chunk under 'Methods'"
    assert any(p["page_start"] == 2 for p in methods_chunks)

    # At least one chunk reaches page 3.
    assert any(p.get("page_end") == 3 for p in parsed)

    # page_start <= page_end on every populated chunk.
    for p in parsed:
        ps, pe = p.get("page_start"), p.get("page_end")
        if ps is not None and pe is not None:
            assert ps <= pe


def test_pipeline_text_element_fallback_all_null_pages(tmp_path, monkeypatch):
    # Simulate the pdftotext/TextElement fallback: page_number=None,
    # no `#` headings → chunks land with page_start=None, page_end=None,
    # section_heading=None.
    from grounding.parser import TextElement

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    dummy_pdf = input_dir / "scanned.pdf"
    dummy_pdf.write_bytes(b"%PDF-placeholder\n")

    def _fake_parse_pdf(path, ocr_mode="auto"):
        return [
            TextElement(text=("Line without markdown heading syntax. " * 15)),
            TextElement(text=("Another paragraph, pdftotext style. " * 15)),
        ]

    monkeypatch.setattr(pipeline_mod, "parse_pdf", _fake_parse_pdf)

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        parser="unstructured",
        ocr_mode="off",
        metadata={"chunk_size": 200, "chunk_overlap": 30},
    )
    run_controller(config, files=[dummy_pdf])

    candidates = [p for p in output_dir.iterdir() if p.is_dir()]
    chunks_dir = candidates[0] / "chunks"
    chunk_files = sorted(chunks_dir.glob("ch_*.md"))
    assert chunk_files

    for path in chunk_files:
        text = path.read_text(encoding="utf-8")
        _, _, rest = text[4:].partition("---\n")
        fm = yaml.safe_load(text[4:-len(rest) - 4])
        assert fm["page_start"] is None
        assert fm["page_end"] is None
        # section_heading is only emitted when non-None — verify omission.
        assert fm.get("section_heading") is None
