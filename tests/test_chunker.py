"""Tests for grounding.chunker."""
from __future__ import annotations

import pytest

from grounding.chunker import ChunkConfig, split_markdown


SAMPLE_SECTION = """# Heading {index}

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum vitae orci
vel ligula rhoncus pulvinar. Sed in suscipit metus. Nunc ac eros nec sem
ullamcorper pretium. Vivamus eget orci ac velit porttitor aliquet sit amet non
diam.

| Column | Value |
| ------ | ----- |
| alpha  | beta  |

```python
def greet():
    print("hello world")
```
"""


def build_document(section_count: int = 4) -> str:
    """Create a deterministic Markdown document for chunking tests."""
    sections = [SAMPLE_SECTION.format(index=i) for i in range(section_count)]
    return "\n\n".join(sections)


def test_split_markdown_default_config_produces_chunks() -> None:
    document = build_document(section_count=4)
    chunks = split_markdown(document)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) for chunk in chunks)

    # Ensure table and code block remain intact within a single chunk.
    table_header = "| Column | Value |"
    table_value = "| alpha  | beta  |"
    for chunk in chunks:
        if table_header in chunk:
            assert table_value in chunk

    code_marker = "```python"
    for chunk in chunks:
        if code_marker in chunk:
            backtick_markers = chunk.count("```")
            assert backtick_markers % 2 == 0

    # Overlap should replicate some lines into the next chunk.
    if len(chunks) > 1:
        first_lines = set(chunks[0].splitlines())
        second_lines = set(chunks[1].splitlines())
        assert first_lines.intersection(second_lines), "Expected overlapping lines between chunks"


def test_split_markdown_respects_custom_config() -> None:
    document = build_document(section_count=3)

    default_chunks = split_markdown(document)
    custom_config = ChunkConfig(chunk_size=300, chunk_overlap=50)
    custom_chunks = split_markdown(document, custom_config)

    assert len(custom_chunks) > len(default_chunks)
    assert all(len(chunk) <= custom_config.chunk_size + 20 for chunk in custom_chunks)


@pytest.mark.parametrize(
    "config",
    [
        ChunkConfig(chunk_size=0, chunk_overlap=0),
        ChunkConfig(chunk_size=100, chunk_overlap=-1),
        ChunkConfig(chunk_size=100, chunk_overlap=100),
        ChunkConfig(chunk_size=100, chunk_overlap=50, separators=()),
        ChunkConfig(chunk_size=100, chunk_overlap=50, separators=("ok", 1)),  # type: ignore[arg-type]
    ],
)
def test_split_markdown_invalid_config_raises_value_error(config: ChunkConfig) -> None:
    with pytest.raises(ValueError):
        split_markdown("Some Markdown", config)


def test_split_markdown_requires_string_input() -> None:
    with pytest.raises(TypeError):
        split_markdown(123)  # type: ignore[arg-type]


def test_split_markdown_is_deterministic() -> None:
    document = build_document(section_count=2)
    config = ChunkConfig(chunk_size=250, chunk_overlap=40)

    first = split_markdown(document, config)
    second = split_markdown(document, config)

    assert first == second
