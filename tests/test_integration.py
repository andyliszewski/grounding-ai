"""Integration tests for grounding controller/CLI."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import hashlib
import pytest
from typer.testing import CliRunner

from grounding.cli import app
from grounding.controller import run_controller
from grounding.formatter import FormatError
from grounding.parser import ParseError
from grounding.pipeline import PipelineConfig
from tests.integration_utils import copy_fixtures, hash_directory

runner = CliRunner()


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_failures: set[str] | None = None,
    format_failures: set[str] | None = None,
) -> None:
    parse_failures = parse_failures or set()
    format_failures = format_failures or set()

    def fake_parse_pdf(path: Path, ocr_mode: str = "auto") -> List[SimpleNamespace]:
        stem = path.stem
        if stem in parse_failures:
            raise ParseError(path, f"unable to parse {stem}")
        lines = [" ".join([f"{stem}-{i}"] * 12) for i in range(6)]
        return [SimpleNamespace(text=line, metadata={"page_number": i + 1}) for i, line in enumerate(lines)]

    def fake_format_markdown(
        elements: List[SimpleNamespace],
        *,
        metadata=None,
        allow_plaintext_fallback: bool,
        source_name: str,
    ) -> str:
        if source_name in format_failures:
            raise FormatError(source_name, "format boom")
        return "\n".join(element.text for element in elements) + "\n"

    monkeypatch.setattr("grounding.pipeline.parse_pdf", fake_parse_pdf)
    monkeypatch.setattr("grounding.pipeline.format_markdown", fake_format_markdown)


class TestControllerIntegration:
    def test_controller_end_to_end_outputs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf", "table.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        assert result.stats.succeeded == 2
        assert result.stats.total_chunks > 0

        manifest = json.loads((config.output_dir / "_index.json").read_text(encoding="utf-8"))
        assert len(manifest["docs"]) == 2
        for entry in manifest["docs"]:
            slug = entry["slug"]
            slug_dir = config.output_dir / slug
            assert (slug_dir / "doc.md").exists()
            chunk_files = sorted((slug_dir / "chunks").glob("ch_*.md"))
            assert chunk_files
            assert entry["chunk_count"] == len(chunk_files)

    def test_controller_rerun_is_deterministic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf", "table.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            metadata={"chunk_size": 60, "chunk_overlap": 10},
        )

        run_controller(config)
        first_hash = hash_directory(config.output_dir)

        run_controller(config)
        second_hash = hash_directory(config.output_dir)

        assert first_hash == second_hash


class TestCLIIntegration:
    def test_controller_reports_failure_and_continues(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that parser failures are handled gracefully and other files continue processing."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf", "malformed.pdf"], input_dir)
        _install_stubs(monkeypatch, parse_failures={"malformed"})

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        assert result.stats.failed == 1
        assert result.stats.succeeded == 1
        manifest = json.loads((tmp_path / "out" / "_index.json").read_text(encoding="utf-8"))
        slugs = {entry["slug"] for entry in manifest["docs"]}
        assert slugs == {"plain"}

    def test_controller_with_clean_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that clean flag removes output directory before processing."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["table.pdf"], input_dir)
        _install_stubs(monkeypatch)

        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        (output_dir / "old_file.txt").write_text("old content")

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata={"chunk_size": 80, "chunk_overlap": 20},
            clean=True,
        )

        result = run_controller(config)

        assert result.stats.succeeded == 1
        assert not (output_dir / "old_file.txt").exists()
        manifest = json.loads((output_dir / "_index.json").read_text(encoding="utf-8"))
        assert len(manifest["docs"]) == 1

    def test_controller_dry_run_no_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that dry-run mode doesn't create output files."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            metadata={"chunk_size": 50, "chunk_overlap": 10},
            dry_run=True,
        )

        result = run_controller(config)

        assert result.stats.succeeded == 1
        assert not (tmp_path / "out" / "_index.json").exists()


class TestEmbeddingIntegration:
    """Integration tests for embedding generation (Story 6.2)."""

    def test_embeddings_generated_when_flag_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that embeddings are generated for all chunks when flag is enabled."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            emit_embeddings=True,
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        assert result.stats.succeeded == 1
        assert result.stats.total_chunks > 0

        # Verify embeddings were generated
        assert len(config.embeddings) > 0
        assert len(config.embeddings) == result.stats.total_chunks

        # Verify embedding dimensions (384 for all-MiniLM-L6-v2)
        for chunk_id, embedding in config.embeddings.items():
            assert embedding.shape == (384,)
            # chunk_id format: <doc_id>-<index> (e.g., "d45e4535-0001")
            assert "-" in chunk_id
            parts = chunk_id.split("-")
            assert len(parts) == 2
            assert len(parts[0]) == 8  # doc_id is 8 chars
            assert parts[1].isdigit()  # index is numeric

        # Verify chunk metadata includes has_embedding flag
        slug_dir = config.output_dir / "plain"
        chunk_files = sorted((slug_dir / "chunks").glob("ch_*.md"))
        assert len(chunk_files) > 0

        for chunk_file in chunk_files:
            content = chunk_file.read_text(encoding="utf-8")
            assert "has_embedding: true" in content

    def test_no_embeddings_when_flag_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that embeddings are not generated when flag is disabled."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            emit_embeddings=False,
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        assert result.stats.succeeded == 1
        assert result.stats.total_chunks > 0

        # Verify no embeddings were generated
        assert len(config.embeddings) == 0

        # Verify chunk metadata does not include has_embedding flag
        slug_dir = config.output_dir / "plain"
        chunk_files = sorted((slug_dir / "chunks").glob("ch_*.md"))
        assert len(chunk_files) > 0

        for chunk_file in chunk_files:
            content = chunk_file.read_text(encoding="utf-8")
            assert "has_embedding" not in content

    def test_embedding_count_matches_chunk_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the number of embeddings matches the number of chunks."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf", "table.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            emit_embeddings=True,
            metadata={"chunk_size": 40, "chunk_overlap": 5},
        )

        result = run_controller(config)

        assert result.stats.succeeded == 2
        total_chunks = result.stats.total_chunks
        assert total_chunks > 0

        # Verify embedding count matches chunk count
        assert len(config.embeddings) == total_chunks

        # Count actual chunk files
        chunk_file_count = 0
        for slug in ["plain", "table"]:
            slug_dir = config.output_dir / slug
            chunk_files = list((slug_dir / "chunks").glob("ch_*.md"))
            chunk_file_count += len(chunk_files)

        assert len(config.embeddings) == chunk_file_count

    def test_embeddings_accessible_in_write_phase(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that embeddings are accessible during the write phase."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf"], input_dir)
        _install_stubs(monkeypatch)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            emit_embeddings=True,
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        assert result.stats.succeeded == 1

        # Embeddings should be stored in config and accessible
        assert len(config.embeddings) > 0

        # Verify we can retrieve embeddings by chunk_id
        for chunk_id in config.embeddings.keys():
            embedding = config.embeddings[chunk_id]
            assert embedding is not None
            assert embedding.shape == (384,)

    def test_embedding_error_handling(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that pipeline continues when embedding generation fails for a chunk."""
        input_dir = tmp_path / "pdfs"
        copy_fixtures(["plain.pdf"], input_dir)
        _install_stubs(monkeypatch)

        # Mock generate_embedding to fail on specific chunks
        call_count = {"count": 0}

        def failing_generate_embedding(text: str):
            call_count["count"] += 1
            if call_count["count"] == 2:  # Fail on second chunk
                raise RuntimeError("Embedding generation failed")
            import numpy as np
            return np.random.rand(384)

        monkeypatch.setattr("grounding.controller.generate_embedding", failing_generate_embedding)

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
            emit_embeddings=True,
            metadata={"chunk_size": 50, "chunk_overlap": 10},
        )

        result = run_controller(config)

        # Pipeline should complete successfully despite embedding failure
        assert result.stats.succeeded == 1
        # Should have embeddings for all chunks except the one that failed
        assert len(config.embeddings) == result.stats.total_chunks - 1
