"""Tests for grounding.controller."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from grounding.controller import run_controller
from grounding.formatter import FormatResult
from grounding.pipeline import PipelineConfig


def _as_result(fake):
    """Adapt string-returning fake formatter to the element-map API."""
    def wrapper(elements, *, metadata=None, source_name=None, **extra):
        extra.pop("allow_plaintext_fallback", None)
        out = fake(
            elements,
            metadata=metadata,
            allow_plaintext_fallback=True,
            source_name=source_name,
            **extra,
        )
        if isinstance(out, FormatResult):
            return out
        return FormatResult(markdown=out, elements=())
    return wrapper


@pytest.fixture
def sample_config(tmp_path: Path) -> PipelineConfig:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    (input_dir / "alpha.pdf").write_text("PDF", encoding="utf-8")
    (input_dir / "beta.pdf").write_text("PDF", encoding="utf-8")
    metadata = {"chunk_size": 50, "chunk_overlap": 10}
    return PipelineConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        metadata=metadata,
    )


def test_run_controller_generates_outputs(
    monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig
) -> None:
    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[SimpleNamespace]:
        return [SimpleNamespace(text=f"section-{path.name}")]

    def fake_format_markdown(
        elements: List[SimpleNamespace],
        *,
        metadata=None,
        allow_plaintext_fallback=False,
        source_name: str,
    ) -> str:
        return "\n\n".join(element.text for element in elements) + "\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    result = run_controller(sample_config)

    assert result.stats.total_files == 2
    assert result.stats.succeeded == 2
    assert result.stats.total_chunks == 2

    for slug in ("alpha", "beta"):
        slug_dir = sample_config.output_dir / slug
        doc_path = slug_dir / "doc.md"
        chunk_files = sorted((slug_dir / "chunks").glob("ch_*.md"))
        meta_path = slug_dir / "meta.yaml"

        assert doc_path.exists()
        assert chunk_files
        assert meta_path.exists()
        context = next(ctx for ctx in result.files if ctx.slug == slug)
        assert context.chunk_count == len(chunk_files)

    manifest_path = sample_config.output_dir / "_index.json"
    assert manifest_path.exists()
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "alpha" in manifest and "beta" in manifest
    assert "chunk_count" in manifest


def test_run_controller_dry_run(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig) -> None:
    sample_config.dry_run = True

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[SimpleNamespace]:
        return [SimpleNamespace(text="content")]

    def fake_format_markdown(
        elements: List[SimpleNamespace],
        *,
        metadata=None,
        allow_plaintext_fallback=False,
        source_name: str,
    ) -> str:
        return "content\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    result = run_controller(sample_config)

    assert result.stats.total_files == 2
    assert result.stats.succeeded == 2
    assert result.stats.failed == 0  # dry-run skips writing but pipeline succeeded
    assert not (sample_config.output_dir / "alpha").exists()


def test_run_controller_handles_chunker_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir()
    (input_dir / "fail.pdf").write_text("PDF", encoding="utf-8")
    (input_dir / "ok.pdf").write_text("PDF", encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
    )

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[SimpleNamespace]:
        return [SimpleNamespace(text=f"chunk-{path.name}")]

    def fake_format_markdown(
        elements: List[SimpleNamespace],
        *,
        metadata=None,
        allow_plaintext_fallback=False,
        source_name: str,
    ) -> str:
        return "\n".join(element.text for element in elements) + "\n"

    def fake_split_markdown_with_map(text: str, elements, config):
        if "fail.pdf" in text:
            raise RuntimeError("chunk boom")
        from grounding.chunker import ChunkWithProvenance
        return [ChunkWithProvenance(text=text, char_start=0, char_end=len(text))]

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))
    monkeypatch.setattr("grounding.controller.split_markdown_with_map", fake_split_markdown_with_map)

    result = run_controller(config)

    assert result.stats.total_files == 2
    assert result.stats.succeeded == 1
    assert result.stats.failed == 1
    assert any(reason.startswith("chunker:") for reason in (f["reason"] for f in result.stats.failed_files))
    assert not (config.output_dir / "fail").exists()
    assert (config.output_dir / "ok" / "doc.md").exists()
