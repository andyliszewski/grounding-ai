"""Processing pipeline for grounding.

Wires together scanning, parsing, formatting, and writing as described in Epic 2 Story 2.4.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from grounding.formatter import FormatError, FormattedElement, format_markdown_with_map
from grounding.hashing import compute_sha1, hash_document, short_doc_id
from grounding.manifest import ManifestEntry, ManifestManager
from grounding.parser import OCR_MODE_MAP, ParseError, parse_epub, parse_pdf
from grounding.scanner import scan_pdfs
from grounding.stats import ProcessingStats
from grounding.utils import atomic_write, ensure_dir, slugify

logger = logging.getLogger("grounding.pipeline")

ProgressCallback = Callable[["FileContext"], None]


@dataclass
class PipelineConfig:
    """Configuration for running the grounding processing pipeline."""

    input_dir: Path
    output_dir: Path
    parser: str = "unstructured"
    ocr_mode: str = "auto"
    allow_plaintext_fallback: bool = True  # Always use plaintext formatting
    dry_run: bool = False
    clean: bool = False
    emit_embeddings: bool = False
    emit_music_embeddings: bool = False
    extract_formulas: bool = False
    music_format: str = "musicxml"
    formula_format: str = "latex"
    metadata: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    collections: List[str] | None = None


@dataclass
class FileContext:
    """Context recorded for each processed file."""

    source_path: Path
    slug: str
    output_path: Optional[Path]
    sha1: Optional[str] = None
    doc_sha1: Optional[str] = None
    doc_id: Optional[str] = None
    doc_hashes: Optional[Dict[str, str]] = None
    status: str = "pending"
    error: Optional[str] = None
    parse_ms: float = 0.0
    format_ms: float = 0.0
    fallback_used: bool = False
    chunk_count: int = 0
    markdown: Optional[str] = None
    element_map: tuple = ()
    formulas: Optional[List] = None
    formula_stats: Optional[Dict] = None


@dataclass
class PipelineResult:
    """Result bundle for a pipeline run."""

    stats: ProcessingStats
    files: List[FileContext]


def _compute_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-1 hash for the provided file path."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _did_use_fallback(markdown: str) -> bool:
    """Detect whether fallback front matter is present in the Markdown output."""
    if not markdown.startswith("---"):
        return False
    for line in markdown.splitlines()[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if stripped.lower().startswith("fallback:"):
            _, _, value = stripped.partition(":")
            return value.strip().lower() == "true"
    return False


def run_pipeline(
    config: PipelineConfig,
    *,
    files: Optional[Sequence[Path]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    logger_override: Optional[logging.Logger] = None,
    generate_outputs: bool = True,
) -> PipelineResult:
    """
    Execute the processing pipeline for the provided configuration.

    Args:
        config: Pipeline configuration.
        files: Optional pre-scanned list of PDF paths. If omitted the scanner is used.
        progress_callback: Optional callback invoked after each file is processed.
        logger_override: Optional logger instance to use instead of module logger.

    Returns:
        PipelineResult containing processing stats and per-file contexts.
    """
    active_logger = logger_override or logger

    normalized_ocr_mode = config.ocr_mode.lower()
    if normalized_ocr_mode not in OCR_MODE_MAP:
        raise ValueError(
            f"Invalid ocr_mode '{config.ocr_mode}'. Expected one of {sorted(OCR_MODE_MAP)}."
        )

    pdf_files: Sequence[Path] = files if files is not None else scan_pdfs(config.input_dir)
    pdf_files = list(pdf_files)

    stats = ProcessingStats(total_files=len(pdf_files))
    contexts: List[FileContext] = []
    seen_doc_ids: dict[str, tuple[str, str]] = {}

    def emit_progress(context: FileContext) -> None:
        active_logger.debug(
            "pipeline.file slug=%s sha1=%s doc_id=%s status=%s ocr_mode=%s parse_ms=%.2f format_ms=%.2f fallback=%s error=%s",
            context.slug,
            context.sha1 or "",
            context.doc_id or "",
            context.status,
            normalized_ocr_mode,
            context.parse_ms,
            context.format_ms,
            context.fallback_used,
            context.error or "",
        )
        if progress_callback:
            progress_callback(context)

    if not pdf_files:
        active_logger.warning("No PDF files discovered in %s", config.input_dir)
        stats.finish()
        return PipelineResult(stats=stats, files=contexts)

    if config.clean:
        if config.dry_run:
            if config.output_dir.exists():
                active_logger.debug(
                    "Dry-run: would remove output directory %s", config.output_dir
                )
        else:
            if config.output_dir.exists():
                active_logger.debug("Cleaning output directory %s", config.output_dir)
                shutil.rmtree(config.output_dir)

    if not config.dry_run:
        try:
            ensure_dir(config.output_dir)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to create output directory {config.output_dir}: {exc}"
            ) from exc
    else:
        active_logger.debug(
            "Dry-run: would ensure output directory %s exists", config.output_dir
        )

    manifest_path = config.output_dir / "_index.json"
    manifest_data = (
        ManifestManager.load(manifest_path)
        if generate_outputs
        else None
    )

    active_logger.debug(
        "Pipeline starting input=%s output=%s total=%d ocr_mode=%s dry_run=%s clean=%s allow_fallback=%s",
        config.input_dir,
        config.output_dir,
        len(pdf_files),
        normalized_ocr_mode,
        config.dry_run,
        config.clean,
        config.allow_plaintext_fallback,
    )

    for path in pdf_files:
        slug = slugify(path.name) or path.stem
        output_path = (
            config.output_dir / slug / "doc.md"
            if not config.dry_run
            else None
        )
        try:
            file_hash = _compute_sha1(path)
        except OSError as exc:
            active_logger.warning("Unable to compute SHA-1 for %s: %s", path.name, exc)
            file_hash = None

        context = FileContext(
            source_path=path,
            slug=slug,
            output_path=output_path,
            sha1=file_hash,
        )
        contexts.append(context)

        active_logger.debug("Processing file=%s slug=%s sha1=%s", path, slug, context.sha1)

        # Notify progress bar that we're starting this file
        context.status = "parsing"
        emit_progress(context)

        try:
            parse_start = time.perf_counter()
            # Route to appropriate parser based on file extension
            if path.suffix.lower() == ".epub":
                elements = parse_epub(path)
            else:
                elements = parse_pdf(path, ocr_mode=normalized_ocr_mode)
            context.parse_ms = (time.perf_counter() - parse_start) * 1000
            stats.record_parse_time(context.parse_ms)
            active_logger.debug(
                "Parsed file=%s elements=%d parse_ms=%.2f",
                path.name,
                len(elements),
                context.parse_ms,
            )
        except (ParseError, FileNotFoundError, IsADirectoryError, ValueError, TypeError) as exc:
            context.status = "failed"
            context.error = f"parser: {exc}"
            stats.record_failure(path.name, context.error)
            active_logger.warning(
                "Parser failure for file=%s ocr_mode=%s error=%s",
                path.name,
                normalized_ocr_mode,
                exc,
            )
            emit_progress(context)
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            context.status = "failed"
            context.error = f"parser: {exc}"
            stats.record_failure(path.name, context.error)
            active_logger.error(
                "Unexpected parser error for file=%s",
                path.name,
                exc_info=True,
            )
            emit_progress(context)
            continue

        try:
            format_start = time.perf_counter()
            metadata: dict[str, Any] = {"source": path.name}
            if context.sha1:
                metadata["sha1"] = context.sha1
            if config.metadata:
                metadata.update(config.metadata)
            format_result = format_markdown_with_map(
                elements,
                metadata=metadata,
                source_name=slug,
            )
            markdown = format_result.markdown
            context.markdown = markdown
            context.element_map = format_result.elements
            context.format_ms = (time.perf_counter() - format_start) * 1000
            stats.record_format_time(context.format_ms)
            context.fallback_used = _did_use_fallback(markdown)
            active_logger.debug(
                "Formatted file=%s format_ms=%.2f fallback=%s",
                path.name,
                context.format_ms,
                context.fallback_used,
            )

            doc_sha1 = compute_sha1(markdown)
            doc_id = short_doc_id(doc_sha1)
            doc_hashes = hash_document(markdown)
            context.doc_sha1 = doc_sha1
            context.doc_id = doc_id
            context.doc_hashes = doc_hashes

            existing = seen_doc_ids.get(doc_id)
            if existing and existing[0] != doc_sha1:
                previous_sha1, previous_slug = existing
                stats.record_doc_id_collision(
                    doc_id=doc_id,
                    existing_slug=previous_slug,
                    existing_sha1=previous_sha1,
                    new_slug=slug,
                    new_sha1=doc_sha1,
                )
                active_logger.warning(
                    "Doc ID collision detected doc_id=%s slug_a=%s slug_b=%s sha1_a=%s sha1_b=%s",
                    doc_id,
                    previous_slug,
                    slug,
                    previous_sha1,
                    doc_sha1,
                )
            else:
                seen_doc_ids[doc_id] = (doc_sha1, slug)

            active_logger.debug(
                "Document hashed slug=%s blake3=%s sha256=%s",
                slug,
                doc_hashes["blake3"],
                doc_hashes["sha256"],
            )

        except FormatError as exc:
            context.status = "failed"
            context.error = f"formatter: {exc}"
            stats.record_failure(path.name, context.error)
            active_logger.warning(
                "Formatter failure for file=%s: %s",
                path.name,
                exc,
            )
            emit_progress(context)
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            context.status = "failed"
            context.error = f"formatter: {exc}"
            stats.record_failure(path.name, context.error)
            active_logger.error(
                "Unexpected formatter error for file=%s",
                path.name,
                exc_info=True,
            )
            emit_progress(context)
            continue

        if generate_outputs:
            if not config.dry_run and context.output_path is not None:
                ensure_dir(context.output_path.parent)
                atomic_write(context.output_path, markdown)
            elif config.dry_run:
                active_logger.debug(
                    "Dry-run: would write document for slug=%s to %s",
                    context.slug,
                    context.output_path,
                )

        context.status = "success"
        stats.record_success(path.name, chunk_count=0)

        emit_progress(context)

        doc_id_for_manifest = context.doc_id or context.slug
        if generate_outputs and manifest_data is not None:
            entry = ManifestEntry(
                doc_id=doc_id_for_manifest,
                slug=context.slug,
                orig_name=context.source_path.name,
                strategy=config.parser,
                chunk_count=0,
                doc_path=str(Path(context.slug) / "doc.md"),
                meta_path=str(Path(context.slug) / "meta.yaml"),
            )
            manifest_data = ManifestManager.register_document(manifest_data, entry)

    stats.finish()

    if generate_outputs and manifest_data is not None:
        if config.dry_run:
            active_logger.debug(
                "Dry-run: skipping manifest write to %s", manifest_path
            )
        else:
            ManifestManager.write(manifest_data, manifest_path)

    active_logger.debug(
        "Pipeline completed total=%d processed=%d succeeded=%d failed=%d skipped=%d duration_ms=%.2f parse_ms=%.2f format_ms=%.2f",
        stats.total_files,
        stats.processed,
        stats.succeeded,
        stats.failed,
        stats.skipped,
        stats.duration * 1000,
        stats.total_parse_ms,
        stats.total_format_ms,
    )

    return PipelineResult(stats=stats, files=contexts)
