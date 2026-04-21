"""Unit tests for grounding.cli module."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def run_cli(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run the CLI with given arguments using subprocess."""
    cmd = [sys.executable, "-m", "grounding.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a dummy PDF file."""
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "test.pdf").write_bytes(b"%PDF-1.4 dummy")
    return d


class TestCLIValidation:
    """Test CLI argument validation."""

    def test_missing_required_args(self):
        """Test error when required args are missing."""
        result = run_cli()
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "arguments" in result.stderr.lower()

    def test_nonexistent_input_directory(self, tmp_path: Path):
        """Test error when input directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        result = run_cli(str(nonexistent), str(tmp_path / "out"))
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    def test_empty_input_directory(self, tmp_path: Path):
        """Test error when input directory has no PDFs."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = run_cli(str(empty_dir), str(tmp_path / "out"))
        assert result.returncode == 1
        assert "No PDF or EPUB files found" in result.stderr

    def test_chunk_overlap_greater_than_size(self, pdf_dir: Path, tmp_path: Path):
        """Test error when chunk overlap >= chunk size."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--chunk-size", "100",
            "--chunk-overlap", "200"
        )
        assert result.returncode == 1
        assert "overlap" in result.stderr.lower()
        assert "must be less than" in result.stderr.lower()

    def test_dry_run_mode(self, pdf_dir: Path, tmp_path: Path):
        """Test dry-run mode prints config and exits."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--dry-run"
        )
        assert result.returncode == 0
        assert "Dry-run mode" in result.stdout
        assert "Configuration:" in result.stdout

    def test_help_flag(self):
        """Test --help flag shows usage."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "Convert PDFs" in result.stdout
        assert "input_dir" in result.stdout
        assert "output_dir" in result.stdout

    def test_parser_choice_validation(self, pdf_dir: Path, tmp_path: Path):
        """Test parser choice is validated."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--parser", "unstructured",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "parser: unstructured" in result.stdout.lower()

    def test_ocr_mode_validation(self, pdf_dir: Path, tmp_path: Path):
        """Test OCR mode is validated."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--ocr", "auto",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "ocr: auto" in result.stdout.lower()

    def test_custom_chunk_parameters(self, pdf_dir: Path, tmp_path: Path):
        """Test custom chunk size and overlap."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--chunk-size", "800",
            "--chunk-overlap", "100",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "chunk_size: 800" in result.stdout.lower()
        assert "chunk_overlap: 100" in result.stdout.lower()

    def test_clean_flag(self, pdf_dir: Path, tmp_path: Path):
        """Test --clean flag is accepted."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--clean",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "clean: true" in result.stdout.lower()


class TestCLICollections:
    """Test CLI --collections flag (Story 10.4)."""

    def test_cli_with_collections(self, pdf_dir: Path, tmp_path: Path):
        """Test collections are parsed correctly."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "science,biology",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "collections:" in result.stdout.lower()
        assert "science" in result.stdout
        assert "biology" in result.stdout

    def test_cli_collections_multiple(self, pdf_dir: Path, tmp_path: Path):
        """Test multiple collections work."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "science,biology,reference,textbook",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "science" in result.stdout
        assert "biology" in result.stdout
        assert "reference" in result.stdout
        assert "textbook" in result.stdout

    def test_cli_invalid_collection_uppercase(self, pdf_dir: Path, tmp_path: Path):
        """Test error for uppercase collection name."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "Science",
            "--dry-run"
        )
        assert result.returncode == 1
        assert "Invalid collection name" in result.stderr
        assert "Science" in result.stderr
        assert "lowercase kebab-case" in result.stderr

    def test_cli_invalid_collection_underscore(self, pdf_dir: Path, tmp_path: Path):
        """Test error for underscore in collection name."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "music_theory",
            "--dry-run"
        )
        assert result.returncode == 1
        assert "Invalid collection name" in result.stderr
        assert "music_theory" in result.stderr
        assert "lowercase kebab-case" in result.stderr

    def test_cli_invalid_collection_space(self, pdf_dir: Path, tmp_path: Path):
        """Test error for space in collection name."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "music theory",
            "--dry-run"
        )
        assert result.returncode == 1
        assert "Invalid collection name" in result.stderr

    def test_cli_without_collections(self, pdf_dir: Path, tmp_path: Path):
        """Test CLI works without --collections flag (backward compatibility)."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--dry-run"
        )
        assert result.returncode == 0
        assert "collections: None" in result.stdout

    def test_cli_collections_kebab_case(self, pdf_dir: Path, tmp_path: Path):
        """Test valid kebab-case collection names."""
        result = run_cli(
            str(pdf_dir),
            str(tmp_path / "out"),
            "--collections", "music-theory,data-science,ai-ml",
            "--dry-run"
        )
        assert result.returncode == 0
        assert "music-theory" in result.stdout
        assert "data-science" in result.stdout
        assert "ai-ml" in result.stdout

    def test_cli_collections_help_text(self):
        """Test help text documents --collections flag."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "--collections" in result.stdout
        assert "collection" in result.stdout.lower()


class TestCLIEmbeddingsSubcommand:
    """Test CLI embeddings subcommand (Story 10.5)."""

    @pytest.fixture
    def corpus_with_docs(self, tmp_path: Path) -> Path:
        """Create a sample corpus with documents and manifest."""
        import json
        from datetime import datetime, timezone

        corpus = tmp_path / "corpus"
        corpus.mkdir()

        # Create manifest
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest = {
            "created_utc": now,
            "updated_utc": now,
            "docs": [
                {
                    "doc_id": "abc12345",
                    "slug": "test-doc",
                    "orig_name": "test.pdf",
                    "chunk_count": 2,
                    "collections": ["science"],
                },
                {
                    "doc_id": "def67890",
                    "slug": "other-doc",
                    "orig_name": "other.pdf",
                    "chunk_count": 1,
                    "collections": ["literature"],
                },
            ],
        }
        (corpus / "_index.json").write_text(json.dumps(manifest))

        # Create chunks for test-doc
        doc_dir = corpus / "test-doc" / "chunks"
        doc_dir.mkdir(parents=True)
        (doc_dir / "ch_0001.md").write_text("---\ndoc_id: abc12345\n---\nThis is chunk one of test document.")
        (doc_dir / "ch_0002.md").write_text("---\ndoc_id: abc12345\n---\nThis is chunk two of test document.")

        # Create chunks for other-doc
        doc_dir2 = corpus / "other-doc" / "chunks"
        doc_dir2.mkdir(parents=True)
        (doc_dir2 / "ch_0001.md").write_text("---\ndoc_id: def67890\n---\nThis is chunk one of other document.")

        return corpus

    @pytest.fixture
    def agents_dir(self, tmp_path: Path) -> Path:
        """Create agents directory with sample agent configs."""
        agents = tmp_path / "agents"
        agents.mkdir()

        (agents / "scientist.yaml").write_text("""
name: scientist
description: Science-focused agent
corpus_filter:
  collections:
    - science
""")
        (agents / "librarian.yaml").write_text("""
name: librarian
description: Literature-focused agent
corpus_filter:
  collections:
    - literature
""")
        return agents

    def test_embeddings_help(self):
        """Test embeddings --help shows correct info."""
        result = run_cli("embeddings", "--help")
        assert result.returncode == 0
        assert "--corpus" in result.stdout
        assert "--agent" in result.stdout
        assert "--agents-dir" in result.stdout
        assert "--check" in result.stdout
        assert "--out" in result.stdout

    def test_embeddings_missing_corpus(self, tmp_path: Path):
        """Test error when corpus doesn't exist."""
        result = run_cli(
            "embeddings",
            "--corpus", str(tmp_path / "nonexistent")
        )
        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_embeddings_missing_manifest(self, tmp_path: Path):
        """Test error when manifest doesn't exist."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus)
        )
        assert result.returncode == 1
        assert "Manifest not found" in result.stderr

    def test_embeddings_with_agent_flag(self, corpus_with_docs: Path, agents_dir: Path):
        """Test agent flag parsed correctly."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0
        assert "Agent: scientist" in result.stdout
        assert "Filtered: 1 documents (from 2 total)" in result.stdout

    def test_embeddings_agent_filtering(self, corpus_with_docs: Path, agents_dir: Path):
        """Test only filtered docs are embedded."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0
        # Scientist should only see science collection (1 doc with 2 chunks)
        assert "Chunks indexed: 2" in result.stdout

    def test_embeddings_agent_output_dir(self, corpus_with_docs: Path, agents_dir: Path):
        """Test output in agent subdirectory."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0

        # Check output directory structure
        output_dir = corpus_with_docs / "embeddings" / "scientist"
        assert output_dir.exists()
        assert (output_dir / "_embeddings.faiss").exists()
        assert (output_dir / "_chunk_map.json").exists()

    def test_embeddings_timestamp_written(self, corpus_with_docs: Path, agents_dir: Path):
        """Test .timestamp file is created."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0

        timestamp_file = corpus_with_docs / "embeddings" / "scientist" / ".timestamp"
        assert timestamp_file.exists()

        # Verify timestamp is valid ISO format
        from datetime import datetime
        timestamp = timestamp_file.read_text().strip()
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_embeddings_check_stale(self, corpus_with_docs: Path, agents_dir: Path):
        """Test --check returns exit code 1 when embeddings are stale."""
        import json
        from datetime import datetime, timezone

        # First generate embeddings
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0

        # Add a new document matching the scientist's filter (science collection)
        manifest_path = corpus_with_docs / "_index.json"
        manifest = json.loads(manifest_path.read_text())
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest["docs"].append({
            "doc_id": "new12345",
            "slug": "new-science-doc",
            "orig_name": "new-science.pdf",
            "chunk_count": 1,
            "collections": ["science"],  # Matches scientist filter
        })
        manifest["updated_utc"] = now
        manifest_path.write_text(json.dumps(manifest))

        # Create new document folder with chunk
        new_doc_dir = corpus_with_docs / "new-science-doc" / "chunks"
        new_doc_dir.mkdir(parents=True)
        (new_doc_dir / "ch_0001.md").write_text("---\ndoc_id: new12345\n---\nNew science content.")

        # Check should return stale (new doc detected)
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir),
            "--check"
        )
        assert result.returncode == 1
        assert "STALE" in result.stdout
        assert "New documents: 1" in result.stdout

    def test_embeddings_check_fresh(self, corpus_with_docs: Path, agents_dir: Path):
        """Test --check returns exit code 0 when embeddings are fresh."""
        # First generate embeddings
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 0

        # Check immediately should return fresh
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir),
            "--check"
        )
        assert result.returncode == 0
        assert "UP TO DATE" in result.stdout

    def test_embeddings_without_agent(self, corpus_with_docs: Path):
        """Test full corpus embedding (backward compatible)."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs)
        )
        assert result.returncode == 0
        assert "Documents: 2" in result.stdout
        # Should have 3 chunks total (2 + 1)
        assert "Chunks indexed: 3" in result.stdout

        # Output should be in 'full' directory
        output_dir = corpus_with_docs / "embeddings" / "full"
        assert output_dir.exists()
        assert (output_dir / "_embeddings.faiss").exists()

    def test_embeddings_missing_agent(self, corpus_with_docs: Path, agents_dir: Path):
        """Test error for unknown agent."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "nonexistent",
            "--agents-dir", str(agents_dir)
        )
        assert result.returncode == 1
        assert "Error loading agent config" in result.stderr

    def test_embeddings_explicit_output_dir(self, corpus_with_docs: Path, tmp_path: Path):
        """Test explicit --out directory."""
        custom_out = tmp_path / "custom-embeddings"

        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--out", str(custom_out)
        )
        assert result.returncode == 0

        # Check output is in custom directory
        assert custom_out.exists()
        assert (custom_out / "_embeddings.faiss").exists()

    def test_embeddings_check_missing(self, corpus_with_docs: Path, agents_dir: Path):
        """Test --check when embeddings don't exist."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_docs),
            "--agent", "scientist",
            "--agents-dir", str(agents_dir),
            "--check"
        )
        assert result.returncode == 1
        assert "MISSING" in result.stdout


class TestCLIEmbeddingsIncremental:
    """Test CLI embeddings --incremental flag (Story 14.2)."""

    @pytest.fixture
    def corpus_with_meta(self, tmp_path: Path) -> Path:
        """Create a corpus with documents, manifest, and meta.yaml files."""
        import json
        import yaml
        from datetime import datetime, timezone

        corpus = tmp_path / "corpus"
        corpus.mkdir()

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest = {
            "created_utc": now,
            "updated_utc": now,
            "docs": [
                {
                    "doc_id": "abc12345",
                    "slug": "test-doc",
                    "orig_name": "test.pdf",
                    "chunk_count": 2,
                    "collections": [],
                },
                {
                    "doc_id": "def67890",
                    "slug": "other-doc",
                    "orig_name": "other.pdf",
                    "chunk_count": 1,
                    "collections": [],
                },
            ],
        }
        (corpus / "_index.json").write_text(json.dumps(manifest))

        # Create test-doc with meta.yaml
        doc_dir = corpus / "test-doc"
        (doc_dir / "chunks").mkdir(parents=True)
        (doc_dir / "chunks" / "ch_0001.md").write_text("---\ndoc_id: abc12345\n---\nChunk one.")
        (doc_dir / "chunks" / "ch_0002.md").write_text("---\ndoc_id: abc12345\n---\nChunk two.")
        meta = {
            "doc_id": "abc12345",
            "slug": "test-doc",
            "hashes": {"file_sha1": "sha1_abc12345"},
        }
        (doc_dir / "meta.yaml").write_text(yaml.dump(meta))

        # Create other-doc with meta.yaml
        doc_dir2 = corpus / "other-doc"
        (doc_dir2 / "chunks").mkdir(parents=True)
        (doc_dir2 / "chunks" / "ch_0001.md").write_text("---\ndoc_id: def67890\n---\nOther chunk.")
        meta2 = {
            "doc_id": "def67890",
            "slug": "other-doc",
            "hashes": {"file_sha1": "sha1_def67890"},
        }
        (doc_dir2 / "meta.yaml").write_text(yaml.dump(meta2))

        return corpus

    def test_incremental_help_text(self):
        """Test --incremental flag appears in help."""
        result = run_cli("embeddings", "--help")
        assert result.returncode == 0
        assert "--incremental" in result.stdout
        assert "incremental" in result.stdout.lower()

    def test_incremental_fallback_no_index(self, corpus_with_meta: Path):
        """Test --incremental falls back to full when no index exists."""
        result = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--incremental"
        )
        assert result.returncode == 0
        # Should fall back to full generation
        assert "Mode: Full generation" in result.stdout

    def test_incremental_no_changes(self, corpus_with_meta: Path):
        """Test --incremental with no changes shows up-to-date message."""
        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Second run: incremental with no changes
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--incremental"
        )
        assert result2.returncode == 0
        assert "No changes detected" in result2.stdout or "Embeddings are up to date" in result2.stdout

    def test_incremental_new_document(self, corpus_with_meta: Path):
        """Test --incremental detects and embeds new document."""
        import json
        import yaml
        from datetime import datetime, timezone

        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Add a new document
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_path = corpus_with_meta / "_index.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["docs"].append({
            "doc_id": "new12345",
            "slug": "new-doc",
            "orig_name": "new.pdf",
            "chunk_count": 1,
            "collections": [],
        })
        manifest["updated_utc"] = now
        manifest_path.write_text(json.dumps(manifest))

        # Create new document folder
        new_doc_dir = corpus_with_meta / "new-doc"
        (new_doc_dir / "chunks").mkdir(parents=True)
        (new_doc_dir / "chunks" / "ch_0001.md").write_text("---\ndoc_id: new12345\n---\nNew content.")
        meta = {
            "doc_id": "new12345",
            "slug": "new-doc",
            "hashes": {"file_sha1": "sha1_new12345"},
        }
        (new_doc_dir / "meta.yaml").write_text(yaml.dump(meta))

        # Second run: incremental should detect new doc
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--incremental"
        )
        assert result2.returncode == 0
        assert "Mode: Incremental update" in result2.stdout
        assert "New documents: 1" in result2.stdout

    def test_incremental_deleted_document(self, corpus_with_meta: Path):
        """Test --incremental tombstones deleted document."""
        import json
        import shutil
        from datetime import datetime, timezone

        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Remove a document from manifest
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_path = corpus_with_meta / "_index.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["docs"] = [d for d in manifest["docs"] if d["doc_id"] != "def67890"]
        manifest["updated_utc"] = now
        manifest_path.write_text(json.dumps(manifest))

        # Remove document folder
        shutil.rmtree(corpus_with_meta / "other-doc")

        # Second run: incremental should tombstone deleted doc
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--incremental"
        )
        assert result2.returncode == 0
        assert "Mode: Incremental update" in result2.stdout
        assert "Deleted documents: 1" in result2.stdout
        assert "Tombstoning" in result2.stdout

    def test_incremental_summary_output(self, corpus_with_meta: Path):
        """Test incremental mode shows summary statistics."""
        import json
        import yaml
        from datetime import datetime, timezone

        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Add a new document
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_path = corpus_with_meta / "_index.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["docs"].append({
            "doc_id": "sum12345",
            "slug": "summary-test",
            "orig_name": "summary.pdf",
            "chunk_count": 1,
            "collections": [],
        })
        manifest["updated_utc"] = now
        manifest_path.write_text(json.dumps(manifest))

        new_doc_dir = corpus_with_meta / "summary-test"
        (new_doc_dir / "chunks").mkdir(parents=True)
        (new_doc_dir / "chunks" / "ch_0001.md").write_text("---\ndoc_id: sum12345\n---\nSummary test.")
        meta = {"doc_id": "sum12345", "slug": "summary-test", "hashes": {"file_sha1": "sha1_sum"}}
        (new_doc_dir / "meta.yaml").write_text(yaml.dump(meta))

        # Incremental run
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--incremental"
        )
        assert result2.returncode == 0
        assert "Incremental Update Summary" in result2.stdout
        assert "New:" in result2.stdout
        assert "Skipped:" in result2.stdout

    def test_check_shows_recommendation(self, corpus_with_meta: Path):
        """Test --check shows incremental recommendation when stale."""
        import json
        import yaml
        from datetime import datetime, timezone

        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Add a new document
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest_path = corpus_with_meta / "_index.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["docs"].append({
            "doc_id": "chk12345",
            "slug": "check-test",
            "orig_name": "check.pdf",
            "chunk_count": 1,
            "collections": [],
        })
        manifest["updated_utc"] = now
        manifest_path.write_text(json.dumps(manifest))

        new_doc_dir = corpus_with_meta / "check-test"
        (new_doc_dir / "chunks").mkdir(parents=True)
        (new_doc_dir / "chunks" / "ch_0001.md").write_text("---\ndoc_id: chk12345\n---\nCheck test.")
        meta = {"doc_id": "chk12345", "slug": "check-test", "hashes": {"file_sha1": "sha1_chk"}}
        (new_doc_dir / "meta.yaml").write_text(yaml.dump(meta))

        # Check should show recommendation
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--check"
        )
        assert result2.returncode == 1
        assert "STALE" in result2.stdout
        assert "Changes detected:" in result2.stdout
        assert "New documents:" in result2.stdout
        assert "--incremental" in result2.stdout

    def test_check_shows_index_info(self, corpus_with_meta: Path):
        """Test --check shows detailed index information."""
        # First run: create initial embeddings
        result1 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta)
        )
        assert result1.returncode == 0

        # Check should show index details
        result2 = run_cli(
            "embeddings",
            "--corpus", str(corpus_with_meta),
            "--check"
        )
        assert result2.returncode == 0
        assert "Vectors:" in result2.stdout
        assert "Tombstones:" in result2.stdout
        assert "Created:" in result2.stdout
