"""Unit tests for grounding.stats module."""
import pytest
import time
from grounding.stats import ProcessingStats


class TestProcessingStats:
    """Test ProcessingStats class."""

    def test_initialization(self):
        """Test stats initialization with default values."""
        stats = ProcessingStats()
        assert stats.total_files == 0
        assert stats.processed == 0
        assert stats.succeeded == 0
        assert stats.failed == 0
        assert stats.total_chunks == 0
        assert stats.failed_files == []
        assert stats.start_time > 0
        assert stats.end_time == 0.0

    def test_initialization_with_total_files(self):
        """Test stats initialization with total_files."""
        stats = ProcessingStats(total_files=10)
        assert stats.total_files == 10
        assert stats.processed == 0

    def test_record_success(self):
        """Test recording successful file processing."""
        stats = ProcessingStats(total_files=10)
        stats.record_success("doc1.pdf", 25)

        assert stats.processed == 1
        assert stats.succeeded == 1
        assert stats.failed == 0
        assert stats.total_chunks == 25
        assert len(stats.failed_files) == 0

    def test_record_multiple_successes(self):
        """Test recording multiple successful files."""
        stats = ProcessingStats(total_files=10)
        stats.record_success("doc1.pdf", 25)
        stats.record_success("doc2.pdf", 30)
        stats.record_success("doc3.pdf", 15)

        assert stats.processed == 3
        assert stats.succeeded == 3
        assert stats.failed == 0
        assert stats.total_chunks == 70

    def test_record_failure(self):
        """Test recording failed file processing."""
        stats = ProcessingStats(total_files=10)
        stats.record_failure("bad.pdf", "corrupted file")

        assert stats.processed == 1
        assert stats.succeeded == 0
        assert stats.failed == 1
        assert len(stats.failed_files) == 1
        assert stats.failed_files[0]["file"] == "bad.pdf"
        assert stats.failed_files[0]["reason"] == "corrupted file"

    def test_record_mixed_results(self):
        """Test recording mix of successes and failures."""
        stats = ProcessingStats(total_files=5)
        stats.record_success("doc1.pdf", 25)
        stats.record_success("doc2.pdf", 30)
        stats.record_failure("bad.pdf", "corrupted")
        stats.record_success("doc3.pdf", 15)
        stats.record_failure("empty.pdf", "no content")

        assert stats.processed == 5
        assert stats.succeeded == 3
        assert stats.failed == 2
        assert stats.total_chunks == 70
        assert len(stats.failed_files) == 2

    def test_duration_before_finish(self):
        """Test duration calculation before calling finish()."""
        stats = ProcessingStats()
        time.sleep(0.1)
        duration = stats.duration

        assert duration >= 0.1
        assert stats.end_time == 0.0  # Not yet finished

    def test_duration_after_finish(self):
        """Test duration calculation after calling finish()."""
        stats = ProcessingStats()
        time.sleep(0.1)
        stats.finish()
        duration = stats.duration

        assert duration >= 0.1
        assert stats.end_time > 0

    def test_get_summary_success_only(self):
        """Test summary with only successes."""
        stats = ProcessingStats(total_files=3)
        stats.record_success("doc1.pdf", 25)
        stats.record_success("doc2.pdf", 30)
        stats.record_success("doc3.pdf", 15)
        stats.finish()

        summary = stats.get_summary()

        assert "Summary:" in summary
        assert "Files processed: 3" in summary
        assert "Succeeded: 3" in summary
        assert "Failed: 0" in summary
        assert "Total chunks: 70" in summary
        assert "Duration:" in summary

    def test_get_summary_with_failures(self):
        """Test summary includes failed files with reasons."""
        stats = ProcessingStats(total_files=5)
        stats.record_success("doc1.pdf", 25)
        stats.record_failure("bad.pdf", "corrupted file")
        stats.record_failure("empty.pdf", "no content")
        stats.finish()

        summary = stats.get_summary()

        assert "Files processed: 3" in summary
        assert "Succeeded: 1" in summary
        assert "Failed: 2" in summary
        assert "bad.pdf: corrupted file" in summary
        assert "empty.pdf: no content" in summary

    def test_get_summary_format(self):
        """Test summary is properly formatted."""
        stats = ProcessingStats(total_files=1)
        stats.record_success("doc.pdf", 10)
        stats.finish()

        summary = stats.get_summary()
        lines = summary.split("\n")

        assert lines[0] == "Summary:"
        assert lines[1].startswith("  Files processed:")
        assert lines[2].startswith("  Succeeded:")
        assert lines[3].startswith("  Failed:")
