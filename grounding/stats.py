"""Statistics tracker for grounding processing."""
from dataclasses import dataclass, field
from time import time
from typing import List, Dict


@dataclass
class ProcessingStats:
    """Track processing statistics."""

    # Counts
    total_files: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    parsed_count: int = 0
    formatted_count: int = 0
    skipped: int = 0
    total_chunks: int = 0
    total_parse_ms: float = 0.0
    total_format_ms: float = 0.0

    # Failed files with reasons
    failed_files: List[Dict[str, str]] = field(default_factory=list)
    skipped_files: List[Dict[str, str]] = field(default_factory=list)
    doc_id_collisions: List[Dict[str, str]] = field(default_factory=list)

    # Timing
    start_time: float = field(default_factory=time)
    end_time: float = 0.0

    def record_success(self, file_name: str, chunk_count: int) -> None:
        """
        Record successful file processing.

        Args:
            file_name: Name of processed file
            chunk_count: Number of chunks generated
        """
        self.processed += 1
        self.succeeded += 1
        self.total_chunks += chunk_count

    def add_chunks(self, chunk_count: int) -> None:
        """Accumulate chunk count without altering success counters."""
        self.total_chunks += chunk_count

    def record_failure(self, file_name: str, reason: str) -> None:
        """
        Record failed file processing.

        Args:
            file_name: Name of failed file
            reason: Failure reason
        """
        self.processed += 1
        self.failed += 1
        self.failed_files.append({"file": file_name, "reason": reason})

    def finish(self) -> None:
        """Mark processing as finished."""
        self.end_time = time()

    def record_skip(self, file_name: str, reason: str | None = None) -> None:
        """
        Record skipped file without treating as success or failure.

        Args:
            file_name: Name of skipped file.
            reason: Optional note describing why it was skipped.
        """
        self.skipped += 1
        self.skipped_files.append({"file": file_name, "reason": reason or "skipped"})

    def record_parse_time(self, milliseconds: float) -> None:
        """
        Accumulate parser runtime.

        Args:
            milliseconds: Duration in milliseconds
        """
        self.total_parse_ms += milliseconds
        self.parsed_count += 1

    def record_format_time(self, milliseconds: float) -> None:
        """
        Accumulate formatter runtime.

        Args:
            milliseconds: Duration in milliseconds
        """
        self.total_format_ms += milliseconds
        self.formatted_count += 1

    def record_doc_id_collision(
        self,
        *,
        doc_id: str,
        existing_slug: str,
        existing_sha1: str,
        new_slug: str,
        new_sha1: str,
    ) -> None:
        """
        Record a document ID collision detected during processing.

        Args:
            doc_id: Colliding short document identifier.
            existing_slug: Slug that already claimed the doc_id.
            existing_sha1: Full SHA-1 hash for the existing document.
            new_slug: Slug for the new document causing the collision.
            new_sha1: Full SHA-1 hash for the new document.
        """
        self.doc_id_collisions.append(
            {
                "doc_id": doc_id,
                "existing_slug": existing_slug,
                "existing_sha1": existing_sha1,
                "new_slug": new_slug,
                "new_sha1": new_sha1,
            }
        )

    def record_postprocess_failure(self, file_name: str, reason: str) -> None:
        """Convert a previously counted success into a failure."""
        if self.succeeded > 0:
            self.succeeded -= 1
        self.failed += 1
        self.failed_files.append({"file": file_name, "reason": reason})

    @property
    def duration(self) -> float:
        """
        Get processing duration in seconds.

        Returns:
            Duration in seconds
        """
        end = self.end_time if self.end_time > 0 else time()
        return end - self.start_time

    def get_summary(self) -> str:
        """
        Generate summary string.

        Returns:
            Formatted summary text

        Examples:
            >>> stats = ProcessingStats(total_files=10)
            >>> stats.record_success("doc.pdf", 25)
            >>> stats.finish()
            >>> print(stats.get_summary())
            Summary:
              Files processed: 1
              Succeeded: 1
              Failed: 0
              Total chunks: 25
              Duration: 0.0s
        """
        lines = [
            "Summary:",
            f"  Files processed: {self.processed}",
            f"  Succeeded: {self.succeeded}",
            f"  Failed: {self.failed}",
            f"  Parsed: {self.parsed_count}",
            f"  Formatted: {self.formatted_count}",
            f"  Skipped: {self.skipped}",
        ]

        # List failed files
        if self.failed_files:
            for fail in self.failed_files:
                lines.append(f"    - {fail['file']}: {fail['reason']}")

        if self.skipped_files:
            lines.append("  Skipped files:")
            for skip in self.skipped_files:
                lines.append(f"    - {skip['file']}: {skip['reason']}")

        lines.append(f"  Total chunks: {self.total_chunks}")
        lines.append(f"  Duration: {self.duration:.1f}s")
        lines.append(f"  Parse time: {self.total_parse_ms:.2f}ms")
        lines.append(f"  Format time: {self.total_format_ms:.2f}ms")
        if self.doc_id_collisions:
            lines.append(f"  Doc ID collisions: {len(self.doc_id_collisions)}")
            for collision in self.doc_id_collisions:
                lines.append(
                    f"    - {collision['doc_id']}: {collision['existing_slug']} ({collision['existing_sha1']}) "
                    f"vs {collision['new_slug']} ({collision['new_sha1']})"
                )

        return "\n".join(lines)
