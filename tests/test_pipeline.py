"""Tests for grounding.pipeline."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, List

import pytest

from grounding.pipeline import PipelineConfig, PipelineResult, run_pipeline
from grounding.hashing import hash_document
from grounding.formatter import FormatError, FormatResult
from grounding.parser import ParseError


class DummyElement:
    """Minimal element with text attribute for formatter fallback tests."""

    def __init__(self, text: str):
        self.text = text


def _as_result(fake):
    """Wrap a string-returning fake formatter as a FormatResult producer.

    Pipeline now calls ``format_markdown_with_map`` which returns a
    :class:`FormatResult`. These legacy fakes still return a markdown string
    and expect an ``allow_plaintext_fallback`` kwarg (which the pipeline no
    longer passes). The wrapper bridges both.
    """
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


def create_pdfs(directory: Path, names: List[str]) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name in names:
        path = directory / name
        path.write_text("PDF content", encoding="utf-8")
        files.append(path)
    return files


def test_run_pipeline_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["a.pdf", "b.pdf"])

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        assert ocr_mode == "auto"
        return [DummyElement(f"content-{path.name}")]

    metadata_calls: List[dict[str, Any]] = []

    markdown_outputs: List[str] = []

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        assert metadata is not None
        metadata_calls.append(dict(metadata))
        assert metadata["source"]
        assert metadata.get("sha1")
        markdown = f"# {elements[0].text}\n"
        markdown_outputs.append(markdown)
        return markdown

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
        ocr_mode="auto",
    )

    result = run_pipeline(config, files=pdfs)

    assert isinstance(result, PipelineResult)
    assert result.stats.total_files == 2
    assert result.stats.processed == 2
    assert result.stats.succeeded == 2
    assert result.stats.failed == 0
    assert result.stats.parsed_count == 2
    assert result.stats.formatted_count == 2
    assert result.stats.skipped == 0
    assert result.stats.total_parse_ms >= 0
    assert result.stats.total_format_ms >= 0
    outputs = sorted(config.output_dir.glob("*/doc.md"))
    assert len(outputs) == 2
    assert outputs[0].read_text(encoding="utf-8").startswith("# content-")
    expected_hash = hashlib.sha1("PDF content".encode("utf-8")).hexdigest()
    assert all(context.sha1 == expected_hash for context in result.files)
    assert all(call["sha1"] == expected_hash for call in metadata_calls)
    # Doc hashes/IDs populated from formatted Markdown.
    assert all(context.doc_sha1 is not None for context in result.files)
    assert all(context.doc_id is not None and len(context.doc_id) == 8 for context in result.files)
    assert all(context.doc_hashes is not None for context in result.files)
    # Doc hash map should contain both algorithms.
    for context, markdown in zip(result.files, markdown_outputs):
        assert context.doc_sha1 is not None
        assert context.doc_id == context.doc_sha1[:8]
        expected_hashes = hash_document(markdown)
        assert context.doc_hashes == expected_hashes

    manifest_path = config.output_dir / "_index.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["slug"] for entry in manifest["docs"]] == ["a", "b"]


def test_run_pipeline_handles_parse_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["ok.pdf", "bad.pdf"])

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        if path.name == "bad.pdf":
            raise ParseError(path, "parse boom")
        return [DummyElement("ok")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        return "OK\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )

    result = run_pipeline(config, files=pdfs)

    assert result.stats.total_files == 2
    assert result.stats.succeeded == 1
    assert result.stats.failed == 1
    assert result.stats.parsed_count == 1
    assert result.stats.formatted_count == 1
    assert result.stats.skipped == 0
    assert any(ctx.source_path.name == "bad.pdf" and ctx.status == "failed" for ctx in result.files)
    assert any(ctx.source_path.name == "ok.pdf" and ctx.status == "success" for ctx in result.files)
    outputs = sorted(config.output_dir.glob("*/doc.md"))
    assert len(outputs) == 1
    assert outputs[0].read_text(encoding="utf-8") == "OK\n"
    manifest_path = config.output_dir / "_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["docs"]) == 1
    assert manifest["docs"][0]["slug"] == "ok"
    failure_reasons = [f["reason"] for f in result.stats.failed_files]
    assert any(reason.startswith("parser:") for reason in failure_reasons)


def test_run_pipeline_handles_format_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["ok.pdf", "bad.pdf"])

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        return [DummyElement(f"element-{path.name}")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        if source_name == "bad":
            raise FormatError(source_name, "format boom")
        return "OK\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )

    result = run_pipeline(config, files=pdfs)

    assert result.stats.total_files == 2
    assert result.stats.succeeded == 1
    assert result.stats.failed == 1
    assert result.stats.skipped == 0
    bad_context = next(ctx for ctx in result.files if ctx.source_path.name == "bad.pdf")
    good_context = next(ctx for ctx in result.files if ctx.source_path.name == "ok.pdf")
    assert bad_context.status == "failed"
    assert "format boom" in (bad_context.error or "")
    assert good_context.status == "success"
    outputs = sorted(config.output_dir.glob("*/doc.md"))
    assert len(outputs) == 1
    assert outputs[0].read_text(encoding="utf-8") == "OK\n"
    failure_reasons = [f["reason"] for f in result.stats.failed_files]
    assert any(reason.startswith("formatter:") for reason in failure_reasons)


def test_run_pipeline_fallback_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["fallback.pdf"])

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        return [DummyElement("fallback text")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        if allow_plaintext_fallback:
            return "---\nfallback: true\n---\n\n" + elements[0].text + "\n"
        raise FormatError(source_name, "no fallback")

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
        allow_plaintext_fallback=True,
    )

    result = run_pipeline(config, files=pdfs)

    assert result.stats.succeeded == 1
    assert result.stats.failed == 0
    assert result.stats.parsed_count == 1
    assert result.stats.formatted_count == 1
    assert result.stats.skipped == 0
    context = result.files[0]
    assert context.fallback_used is True
    assert context.status == "success"
    output_path = config.output_dir / context.slug / "doc.md"
    assert output_path.exists()
    assert "fallback: true" in output_path.read_text(encoding="utf-8")
    manifest_path = config.output_dir / "_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["docs"][0]["doc_id"] == context.doc_id


def test_run_pipeline_respects_ocr_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["ocr.pdf"])

    observed_modes: List[str] = []

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        observed_modes.append(ocr_mode)
        return [DummyElement("ocr")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        return "ocr\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
        ocr_mode="on",
    )

    result = run_pipeline(config, files=pdfs)

    assert observed_modes == ["on"]
    assert result.stats.skipped == 0
    manifest_path = config.output_dir / "_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["docs"]) == 1


def test_run_pipeline_rejects_invalid_ocr_mode(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "doc.pdf").write_text("PDF", encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        ocr_mode="invalid",
    )

    with pytest.raises(ValueError):
        run_pipeline(config)


def test_run_pipeline_detects_doc_id_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="grounding.pipeline")
    pdfs = create_pdfs(tmp_path / "pdfs", ["first.pdf", "second.pdf"])

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        return [DummyElement(f"element-{path.name}")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        return f"## {source_name}\n"

    # Force compute_sha1/short_doc_id to trigger a collision with different SHA-1 values.
    def fake_compute_sha1(markdown: str) -> str:
        if "first" in markdown:
            return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def fake_short_doc_id(_sha1: str) -> str:
        return "deadbeef"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))
    monkeypatch.setattr("grounding.pipeline.compute_sha1", fake_compute_sha1)
    monkeypatch.setattr("grounding.pipeline.short_doc_id", fake_short_doc_id)

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )

    result = run_pipeline(config, files=pdfs)

    assert len(result.stats.doc_id_collisions) == 1
    collision = result.stats.doc_id_collisions[0]
    assert collision["doc_id"] == "deadbeef"
    assert {collision["existing_slug"], collision["new_slug"]} == {"first", "second"}
    assert "Doc ID collision detected" in "".join(caplog.messages)
    manifest_path = config.output_dir / "_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["docs"]) == 1
    assert manifest["docs"][0]["slug"] == "second"

def test_run_pipeline_clean_removes_existing_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["doc.pdf"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "obsolete.txt").write_text("old", encoding="utf-8")

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        return [DummyElement("content")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        return "doc\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=out_dir,
        clean=True,
    )

    run_pipeline(config, files=pdfs)

    assert not (out_dir / "obsolete.txt").exists()
    assert (out_dir / "doc" / "doc.md").exists()
    manifest = json.loads((out_dir / "_index.json").read_text(encoding="utf-8"))
    assert len(manifest["docs"]) == 1


def test_run_pipeline_dry_run_skips_file_ops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pdfs = create_pdfs(tmp_path / "pdfs", ["doc.pdf"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("keep", encoding="utf-8")

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[DummyElement]:
        return [DummyElement("content")]

    def fake_format_markdown(
        elements: List[DummyElement],
        *,
        metadata: dict[str, Any] | None = None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        return "doc\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown_with_map", _as_result(fake_format_markdown))

    config = PipelineConfig(
        input_dir=tmp_path / "pdfs",
        output_dir=out_dir,
        clean=True,
        dry_run=True,
    )

    caplog.set_level("INFO")
    run_pipeline(config, files=pdfs)

    assert (out_dir / "existing.txt").exists()
    assert not (out_dir / "doc" / "doc.md").exists()
    assert not (out_dir / "_index.json").exists()
    assert any("Dry-run" in message for message in caplog.messages)
