"""Tests for grounding.formatter."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import List

import pytest

formatter_module = importlib.import_module("grounding.formatter")
importlib.reload(formatter_module)
FormatError = formatter_module.FormatError
format_markdown = formatter_module.format_markdown


@dataclass
class DummyElement:
    text: str


def test_format_markdown_success(caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("grounding").propagate = True
    caplog.set_level(logging.INFO, logger="grounding.formatter")

    elements = [DummyElement("Line 1"), DummyElement("Line 2")]

    output = format_markdown(elements, source_name="sample")

    assert output == "Line 1\n\nLine 2\n"
    assert any("element_count=2" in message for message in caplog.messages)


def test_format_markdown_with_metadata() -> None:
    metadata = {"title": "Doc", "status": "draft"}
    elements = [DummyElement("Content here")]

    output = format_markdown(elements, metadata=metadata, source_name="sample")

    assert output == "---\nstatus: draft\ntitle: Doc\n---\n\nContent here\n"


def test_format_markdown_metadata_unchanged() -> None:
    """Ensure original metadata dict is not mutated."""
    metadata = {"title": "Doc", "status": "draft"}
    elements = [DummyElement("Content")]

    format_markdown(elements, metadata=metadata, source_name="sample")

    assert metadata == {"title": "Doc", "status": "draft"}


def test_format_markdown_deterministic() -> None:
    elements = [DummyElement("A"), DummyElement("B")]

    first = format_markdown(elements, source_name="sample")
    second = format_markdown(elements, source_name="sample")

    assert first == second == "A\n\nB\n"


def test_format_markdown_rejects_non_sequence() -> None:
    with pytest.raises(TypeError):
        format_markdown("not-sequence")  # type: ignore[arg-type]


def test_format_markdown_rejects_string() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        format_markdown("string input")  # type: ignore[arg-type]


def test_format_markdown_rejects_bytes() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        format_markdown(b"bytes input")  # type: ignore[arg-type]


def test_format_markdown_empty_elements() -> None:
    elements: List[DummyElement] = []

    output = format_markdown(elements, source_name="empty")

    assert output == ""


def test_format_markdown_normalizes_line_endings() -> None:
    elements = [DummyElement("Line with\r\nWindows\rendings")]

    output = format_markdown(elements, source_name="sample")

    assert "\r" not in output
    assert "Line with\nWindows\nendings\n" == output


def test_format_markdown_strips_trailing_whitespace() -> None:
    elements = [DummyElement("Line with trailing   ")]

    output = format_markdown(elements, source_name="sample")

    assert output == "Line with trailing\n"


def test_format_markdown_boolean_metadata() -> None:
    metadata = {"draft": True, "published": False}
    elements = [DummyElement("Content")]

    output = format_markdown(elements, metadata=metadata, source_name="sample")

    assert "draft: true" in output
    assert "published: false" in output


def test_format_markdown_allow_plaintext_fallback_ignored() -> None:
    """allow_plaintext_fallback parameter is kept for API compat but ignored."""
    elements = [DummyElement("Content")]

    # Should work the same regardless of flag value
    output1 = format_markdown(elements, allow_plaintext_fallback=True, source_name="sample")
    output2 = format_markdown(elements, allow_plaintext_fallback=False, source_name="sample")

    assert output1 == output2 == "Content\n"
