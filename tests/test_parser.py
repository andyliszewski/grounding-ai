"""Tests for grounding.parser."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# Ensure unstructured dependency is stubbed if not installed
if "unstructured.partition.pdf" not in sys.modules:
    unstructured_module = ModuleType("unstructured")
    partition_module = ModuleType("unstructured.partition")
    pdf_module = ModuleType("unstructured.partition.pdf")

    def _default_partition_pdf(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - guard only
        raise RuntimeError("partition_pdf stub not configured")

    pdf_module.partition_pdf = _default_partition_pdf
    partition_module.pdf = pdf_module  # type: ignore[attr-defined]
    unstructured_module.partition = SimpleNamespace(pdf=pdf_module)

    sys.modules["unstructured"] = unstructured_module
    sys.modules["unstructured.partition"] = partition_module
    sys.modules["unstructured.partition.pdf"] = pdf_module

parser_module = importlib.import_module("grounding.parser")
importlib.reload(parser_module)
ParseError = parser_module.ParseError
parse_pdf = parser_module.parse_pdf


@pytest.fixture(autouse=True)
def reset_partition_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset partition_pdf stub before each test."""
    parser_module._partition_pdf = None  # type: ignore[attr-defined]

    def _stub(*_args: Any, **_kwargs: Any) -> list[str]:
        return []

    import unstructured.partition.pdf as pdf_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(pdf_mod, "partition_pdf", _stub)


@pytest.mark.integration
def test_parse_pdf_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.INFO, logger="grounding.parser")

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_text("content", encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def fake_partition_pdf(**kwargs: Any) -> list[str]:
        calls.append(kwargs)
        return ["title", "paragraph"]

    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake_partition_pdf)

    elements = parse_pdf(pdf_file, ocr_mode="auto")

    assert elements == ["title", "paragraph"]
    assert calls[0]["filename"] == str(pdf_file)
    assert calls[0]["ocr_strategy"] == "auto"
    assert calls[0]["infer_table_structure"] is True
    assert any("Parsed sample.pdf" in message for message in caplog.messages)


@pytest.mark.parametrize(
    "mode,expected_strategy",
    [
        ("auto", "auto"),
        ("on", "always"),
        pytest.param(
            "off", "never",
            marks=pytest.mark.integration,
        ),
        ("AUTO", "auto"),
    ],
)
def test_parse_pdf_ocr_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, expected_strategy: str) -> None:
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_text("x", encoding="utf-8")

    def fake_partition_pdf(**kwargs: Any) -> list[str]:
        assert kwargs["ocr_strategy"] == expected_strategy
        return []

    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake_partition_pdf)

    parse_pdf(pdf_file, ocr_mode=mode)


def test_parse_pdf_invalid_mode(tmp_path: Path) -> None:
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        parse_pdf(pdf_file, ocr_mode="unsupported")

    assert "Invalid ocr_mode" in str(exc.value)


def test_parse_pdf_missing_file(tmp_path: Path) -> None:
    pdf_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        parse_pdf(pdf_file)


def test_parse_pdf_directory(tmp_path: Path) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        parse_pdf(directory)


def test_parse_pdf_non_path_argument(tmp_path: Path) -> None:
    file_path = tmp_path / "file.pdf"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(TypeError):
        parse_pdf(str(file_path))  # type: ignore[arg-type]


def test_parse_pdf_wraps_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_file = tmp_path / "broken.pdf"
    pdf_file.write_text("broken", encoding="utf-8")

    def fake_partition_pdf(**_kwargs: Any) -> list[str]:
        raise RuntimeError("Boom")

    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake_partition_pdf)

    with pytest.raises(ParseError) as exc:
        parse_pdf(pdf_file)

    assert exc.value.file_path == pdf_file
    assert "broken.pdf" in str(exc.value)
    assert exc.value.__cause__ is not None
