"""Unit tests for grounding.utils."""
import pytest
from pathlib import Path
import tempfile
import shutil

from grounding.utils import slugify, atomic_write, ensure_dir, validate_collection_name


class TestSlugify:
    """Test slugify() function."""

    def test_basic_filename(self):
        """Test basic filename conversion."""
        assert slugify("Report_2024.pdf") == "report-2024"

    def test_spaces(self):
        """Test filename with spaces."""
        assert slugify("Q3 Financial Report.pdf") == "q3-financial-report"

    def test_multiple_underscores(self):
        """Test multiple underscores."""
        assert slugify("Annual___Review___2024.pdf") == "annual-review-2024"

    def test_special_characters(self):
        """Test special character removal."""
        assert slugify("File@#$%Name!.pdf") == "filename"

    def test_mixed_case(self):
        """Test mixed case conversion."""
        assert slugify("FinalReport_V2.pdf") == "finalreport-v2"

    def test_leading_trailing_hyphens(self):
        """Test hyphen stripping."""
        assert slugify("__Report__.pdf") == "report"

    def test_collapse_hyphens(self):
        """Test multiple hyphens collapse."""
        assert slugify("A---B---C.pdf") == "a-b-c"

    def test_empty_string(self):
        """Test empty string."""
        assert slugify("") == ""

    def test_extension_only(self):
        """Test filename with only extension."""
        # Edge case: ".pdf" is treated as stem by pathlib, becomes "pdf"
        assert slugify(".pdf") == "pdf"

    def test_annual_review_final(self):
        """Test example from story: Annual_Review_2024_FINAL.pdf."""
        assert slugify("Annual_Review_2024_FINAL.pdf") == "annual-review-2024-final"

    def test_spaces_everywhere(self):
        """Test example with excessive spaces."""
        assert slugify("   Spaces__Everywhere  .pdf") == "spaces-everywhere"


class TestAtomicWrite:
    """Test atomic_write() function."""

    def test_basic_write(self, tmp_path):
        """Test basic file write."""
        target = tmp_path / "test.txt"
        content = "Hello, world!"

        result = atomic_write(target, content)

        assert result == target
        assert target.exists()
        assert target.read_text() == content

    def test_unicode_content(self, tmp_path):
        """Test writing Unicode content."""
        target = tmp_path / "unicode.txt"
        content = "Hello, 世界! 🌍"

        atomic_write(target, content)

        assert target.read_text(encoding="utf-8") == content

    def test_creates_parent_dirs(self, tmp_path):
        """Test parent directory creation."""
        target = tmp_path / "nested" / "dir" / "file.txt"
        content = "nested content"

        atomic_write(target, content)

        assert target.exists()
        assert target.read_text() == content

    def test_overwrites_existing(self, tmp_path):
        """Test overwriting existing file."""
        target = tmp_path / "existing.txt"
        target.write_text("old content")

        atomic_write(target, "new content")

        assert target.read_text() == "new content"

    def test_multiline_content(self, tmp_path):
        """Test multiline content."""
        target = tmp_path / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3"

        atomic_write(target, content)

        assert target.read_text() == content

    def test_string_path(self, tmp_path):
        """Test with string path (not Path object)."""
        target = str(tmp_path / "string_path.txt")
        content = "string path test"

        result = atomic_write(target, content)

        assert isinstance(result, Path)
        assert result.read_text() == content

    def test_yaml_frontmatter(self, tmp_path):
        """Test writing YAML frontmatter (common use case)."""
        target = tmp_path / "chunk.md"
        content = """---
doc_id: abc123
chunk_id: 1
---

# Heading

Content here.
"""

        atomic_write(target, content)

        assert target.read_text() == content

    def test_no_temp_files_left(self, tmp_path):
        """Test that no temporary files remain after successful write."""
        target = tmp_path / "clean.txt"
        content = "test content"

        atomic_write(target, content)

        # Check for any .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

        # Check for hidden temp files
        hidden_files = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
        assert len(hidden_files) == 0


class TestEnsureDir:
    """Test ensure_dir() function."""

    def test_create_directory(self, tmp_path):
        """Test directory creation."""
        target = tmp_path / "newdir"

        result = ensure_dir(target)

        assert result == target
        assert target.exists()
        assert target.is_dir()

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        target = tmp_path / "existing"
        target.mkdir()

        result = ensure_dir(target)

        assert result == target
        assert target.exists()

    def test_nested_directory(self, tmp_path):
        """Test nested directory creation."""
        target = tmp_path / "a" / "b" / "c"

        ensure_dir(target)

        assert target.exists()
        assert target.is_dir()

    def test_returns_path(self, tmp_path):
        """Test return value is Path."""
        target = tmp_path / "testdir"

        result = ensure_dir(target)

        assert isinstance(result, Path)

    def test_string_path(self, tmp_path):
        """Test with string path (not Path object)."""
        target = str(tmp_path / "string_dir")

        result = ensure_dir(target)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

    def test_typical_output_structure(self, tmp_path):
        """Test creating typical output directory structure."""
        base = tmp_path / "corpus"
        slug_dir = base / "report-2024"
        chunks_dir = slug_dir / "chunks"

        # Create nested structure
        ensure_dir(chunks_dir)

        assert base.exists()
        assert slug_dir.exists()
        assert chunks_dir.exists()
        assert chunks_dir.is_dir()


class TestValidateCollectionName:
    """Test validate_collection_name() function."""

    def test_valid_simple_name(self):
        """Test simple lowercase names pass."""
        assert validate_collection_name("science") is True
        assert validate_collection_name("biology") is True
        assert validate_collection_name("reference") is True

    def test_valid_kebab_case(self):
        """Test kebab-case names pass."""
        assert validate_collection_name("music-theory") is True
        assert validate_collection_name("ap-biology") is True
        assert validate_collection_name("reference-2024") is True

    def test_valid_with_numbers(self):
        """Test names with numbers pass."""
        assert validate_collection_name("science123") is True
        assert validate_collection_name("2024-reference") is True
        assert validate_collection_name("level1-docs") is True

    def test_invalid_uppercase(self):
        """Test uppercase letters fail."""
        assert validate_collection_name("Science") is False
        assert validate_collection_name("BIOLOGY") is False
        assert validate_collection_name("Music-Theory") is False

    def test_invalid_underscore(self):
        """Test underscores fail."""
        assert validate_collection_name("music_theory") is False
        assert validate_collection_name("ap_biology") is False

    def test_invalid_space(self):
        """Test spaces fail."""
        assert validate_collection_name("music theory") is False
        assert validate_collection_name("ap biology") is False

    def test_invalid_leading_hyphen(self):
        """Test leading hyphen fails."""
        assert validate_collection_name("-music") is False
        assert validate_collection_name("-science-docs") is False

    def test_invalid_trailing_hyphen(self):
        """Test trailing hyphen fails."""
        assert validate_collection_name("music-") is False
        assert validate_collection_name("science-docs-") is False

    def test_invalid_empty(self):
        """Test empty string fails."""
        assert validate_collection_name("") is False

    def test_invalid_none(self):
        """Test None fails."""
        assert validate_collection_name(None) is False

    def test_invalid_special_chars(self):
        """Test special characters fail."""
        assert validate_collection_name("science!") is False
        assert validate_collection_name("biology@2024") is False
        assert validate_collection_name("music.theory") is False

    def test_invalid_consecutive_hyphens(self):
        """Test consecutive hyphens fail."""
        assert validate_collection_name("music--theory") is False
        assert validate_collection_name("a---b") is False
