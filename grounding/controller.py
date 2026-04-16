"""Controller for full grounding pipeline (Epic 5 Story 5.1)."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from grounding.chunk_metadata import build_chunk_metadata, render_chunk
from grounding.chunker import (
    ChunkConfig,
    ChunkWithProvenance,
    derive_chunk_metadata,
    split_markdown_with_map,
)
from grounding.embedder import generate_embedding
from grounding.hashing import hash_chunk, compute_sha1, short_doc_id
from grounding.manifest import ManifestEntry, ManifestManager
from grounding.meta import build_meta_yaml
from grounding.pipeline import FileContext, ProgressCallback, PipelineConfig, PipelineResult, run_pipeline
from grounding.scanner import scan_pdfs
from grounding.stats import ProcessingStats
from grounding.utils import atomic_write, slugify, ensure_dir
from grounding.vector_store import write_vector_index
from grounding.writer import write_document

logger = logging.getLogger("grounding.controller")


def _merge_text_and_formulas(text_markdown: str, formulas: list) -> tuple[str, dict]:
    """
    Merge text and formulas while preserving reading order.

    Args:
        text_markdown: Markdown text from parser
        formulas: List of FormulaElement objects

    Returns:
        Tuple of (merged_markdown, formula_mapping) where formula_mapping
        maps chunk indices to list of formula IDs.
    """
    if not formulas:
        return text_markdown, {}

    # Sort formulas by page and position (top to bottom, left to right)
    sorted_formulas = sorted(
        formulas,
        key=lambda f: (f.page_num, f.bbox[1], f.bbox[0])  # page, y1 (top), x1 (left)
    )

    # For MVP: Append formulas at end of document by page
    # In production, would parse text structure and insert at correct positions
    lines = text_markdown.split('\n')
    formula_mapping = {}

    # Group formulas by page
    from itertools import groupby
    formulas_by_page = {
        page: list(group)
        for page, group in groupby(sorted_formulas, key=lambda f: f.page_num)
    }

    # Append formulas as sections per page
    for page_num, page_formulas in sorted(formulas_by_page.items()):
        lines.append(f"\n## Formulas from Page {page_num + 1}\n")

        for idx, formula in enumerate(page_formulas, start=1):
            formula_id = f"formula_{page_num:03d}_{idx:03d}"

            # Use $ for inline, $$ for display
            if formula.formula_type == "inline":
                lines.append(f"Formula {idx}: ${formula.latex_str}$\n")
            else:  # display
                lines.append(f"Formula {idx}:\n$$\n{formula.latex_str}\n$$\n")

    merged = '\n'.join(lines)
    return merged, formula_mapping


def _embed_formulas_in_chunks(
    chunks: list[str],
    formulas: list,
    doc_id: str
) -> tuple[list[str], dict]:
    """
    Embed formula metadata in chunks and track formula counts.

    Args:
        chunks: List of text chunks
        formulas: List of FormulaElement objects
        doc_id: Document ID for formula IDs

    Returns:
        Tuple of (chunks, formula_stats) where formula_stats maps chunk index
        to dict with formula counts and IDs.
    """
    formula_stats = {}

    # For each chunk, analyze formula content
    for chunk_idx, chunk in enumerate(chunks, start=1):
        # Count display formulas ($$) first
        display_count = chunk.count('$$') // 2  # Each formula has opening and closing $$

        # Count all $ characters, then subtract the ones used in $$
        total_dollar_signs = chunk.count('$')
        dollar_signs_in_display = display_count * 4  # Each $$ uses 2 $ chars, times 2 for open/close
        remaining_dollar_signs = total_dollar_signs - dollar_signs_in_display

        # Inline formulas use the remaining single $ delimiters
        inline_count = remaining_dollar_signs // 2

        total_count = inline_count + display_count

        if total_count > 0:
            # Generate formula IDs
            formula_ids = [f"{doc_id}_formula_{chunk_idx:04d}_{i:03d}" for i in range(1, total_count + 1)]

            formula_stats[chunk_idx] = {
                'formula_count': total_count,
                'inline_formula_count': inline_count,
                'display_formula_count': display_count,
                'formula_ids': formula_ids,
            }

    return chunks, formula_stats


def _process_music_pdf(
    pdf_path: Path,
    config: PipelineConfig,
    stats: ProcessingStats,
    active_logger: logging.Logger,
) -> Optional[FileContext]:
    """
    Process a single music notation PDF using OMR pipeline.

    Returns FileContext on success, None on failure.
    """
    try:
        from grounding import omr_parser, music_formatter

        # Create context
        slug = slugify(pdf_path.stem)

        # Compute file SHA-1 from PDF bytes
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        file_sha1 = compute_sha1(pdf_bytes)
        doc_id = short_doc_id(file_sha1)

        # Set output path to doc.md file (writer expects full path)
        output_path = config.output_dir / slug / "doc.md" if not config.dry_run else None

        context = FileContext(
            source_path=pdf_path,
            slug=slug,
            doc_id=doc_id,
            output_path=output_path,
            status="parsing",
        )

        active_logger.info("Processing music PDF: %s", pdf_path.name)

        # Parse music notation
        music_elements = omr_parser.parse_music_pdf(pdf_path)

        # Generate output formats based on config
        music_outputs = {}
        formats = config.music_format
        if formats == "all":
            formats = ["musicxml", "abc", "midi"]
        else:
            formats = [formats]

        if "musicxml" in formats:
            music_outputs["musicxml"] = music_formatter.format_to_musicxml(music_elements)
        if "abc" in formats:
            music_outputs["abc"] = music_formatter.format_to_abc(music_elements)
        if "midi" in formats:
            music_outputs["midi"] = music_formatter.format_to_midi(music_elements)

        # Generate markdown metadata summary for chunking
        markdown_content = music_formatter.format_to_markdown(music_elements)

        context.markdown = markdown_content
        context.music_outputs = music_outputs
        context.music_elements = music_elements
        context.status = "success"

        # Note: chunk_count will be updated later by chunking stage
        stats.record_success(pdf_path.name, chunk_count=0)
        return context

    except Exception as exc:
        error_msg = f"OMR processing failed: {exc}"
        active_logger.error("Music PDF processing failed: %s", pdf_path.name, exc_info=True)
        stats.record_failure(pdf_path.name, error_msg)
        return None


def run_controller(
    config: PipelineConfig,
    *,
    files: Optional[Sequence[Path]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    logger_override: Optional[logging.Logger] = None,
) -> PipelineResult:
    """Execute the full grounding pipeline including chunking and metadata."""
    active_logger = logger_override or logger

    pdf_files = list(files if files is not None else scan_pdfs(config.input_dir))

    # Route to OMR/Hybrid pipeline if needed
    if config.parser in ["omr", "hybrid"]:
        from grounding.stats import ProcessingStats

        stats = ProcessingStats()
        processed_contexts = []

        for pdf_path in pdf_files:
            if config.parser == "omr":
                # Process as pure music notation
                context = _process_music_pdf(pdf_path, config, stats, active_logger)
                if context:
                    processed_contexts.append(context)
                    if progress_callback:
                        progress_callback(context)

            elif config.parser == "hybrid":
                # Use hybrid processor for auto-detection
                try:
                    from grounding import hybrid_processor

                    slug = slugify(pdf_path.stem)
                    doc_id = short_doc_id(compute_sha1(pdf_path))
                    output_path = config.output_dir / slug

                    active_logger.info("Processing hybrid PDF: %s", pdf_path.name)

                    # Process with hybrid processor
                    output_formats = [config.music_format] if config.music_format != "all" else ["musicxml", "abc", "midi"]
                    hybrid_result = hybrid_processor.process_hybrid_pdf(
                        pdf_path,
                        parser_strategy=config.parser if config.parser not in ["omr", "hybrid"] else "unstructured",
                        output_formats=output_formats,
                    )

                    # Create context from hybrid result
                    context = FileContext(
                        source_path=pdf_path,
                        slug=slug,
                        doc_id=doc_id,
                        output_path=output_path,
                        status="success",
                    )
                    context.markdown = hybrid_result.get("combined_markdown", "")
                    context.music_outputs = hybrid_result.get("music_outputs", {})
                    context.hybrid_result = hybrid_result

                    processed_contexts.append(context)
                    stats.record_success(pdf_path.name)

                    if progress_callback:
                        progress_callback(context)

                except Exception as exc:
                    error_msg = f"Hybrid processing failed: {exc}"
                    active_logger.error("Hybrid PDF processing failed: %s", pdf_path.name, exc_info=True)
                    stats.record_failure(pdf_path.name, error_msg)

        # Create result object for OMR/hybrid processing
        result = PipelineResult(files=processed_contexts, stats=stats)
    else:
        # Use standard text pipeline
        result = run_pipeline(
            config,
            files=pdf_files,
            progress_callback=progress_callback,
            logger_override=active_logger,
            generate_outputs=False,
        )

    if config.dry_run:
        active_logger.debug("Dry-run: skipping output generation")
        return result

    manifest_path = config.output_dir / "_index.json"
    manifest_data = ManifestManager.load(manifest_path)
    stats = result.stats

    chunk_size = config.metadata.get("chunk_size", 1_200)
    chunk_overlap = config.metadata.get("chunk_overlap", 150)
    chunk_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    params_for_meta = dict(config.metadata)
    params_for_meta.setdefault("chunk_size", chunk_size)
    params_for_meta.setdefault("chunk_overlap", chunk_overlap)
    params_for_meta.setdefault("parser", config.parser)
    params_for_meta.setdefault("ocr_mode", config.ocr_mode)

    def handle_failure(stage: str, context: FileContext, exc: Exception) -> None:
        message = f"{stage}: {exc}"
        context.status = "failed"
        context.error = message
        stats.record_postprocess_failure(context.source_path.name, message)
        active_logger.error(
            "Controller failure stage=%s slug=%s", stage, context.slug, exc_info=True
        )
        if progress_callback:
            progress_callback(context)

    for context in result.files:
        if context.status != "success":
            continue
        if context.output_path is None:
            active_logger.warning("Missing output path for slug=%s; skipping", context.slug)
            continue
        if not context.markdown:
            active_logger.warning("No Markdown captured for slug=%s; skipping", context.slug)
            continue

        # Extract formulas if enabled
        if config.extract_formulas:
            try:
                from grounding.formula_extractor import extract_formulas
                from grounding import formula_formatter

                active_logger.debug("Extracting formulas for slug=%s", context.slug)
                formulas = extract_formulas(context.source_path)
                context.formulas = formulas

                if formulas:
                    active_logger.info(
                        "Extracted %d formulas from %s",
                        len(formulas),
                        context.source_path.name
                    )

                    # Generate formula outputs based on config
                    formula_outputs = {}
                    formats = config.formula_format

                    if formats in ["latex", "both"]:
                        try:
                            formula_outputs["latex"] = formula_formatter.format_to_latex(formulas)
                            active_logger.debug("Formatted %d formulas to LaTeX", len(formula_outputs["latex"]))
                        except Exception as exc:
                            active_logger.warning("LaTeX formatting failed: %s", exc, exc_info=True)

                    if formats in ["mathml", "both"]:
                        try:
                            formula_outputs["mathml"] = formula_formatter.format_to_mathml(formulas)
                            active_logger.debug("Formatted %d formulas to MathML", len(formula_outputs["mathml"]))
                        except Exception as exc:
                            active_logger.warning("MathML formatting failed: %s", exc, exc_info=True)

                    context.formula_outputs = formula_outputs

                    # Merge formulas into markdown
                    context.markdown, _ = _merge_text_and_formulas(context.markdown, formulas)
            except Exception as exc:
                active_logger.warning(
                    "Formula extraction failed for slug=%s: %s",
                    context.slug,
                    exc,
                    exc_info=True
                )
                # Continue processing without formulas

        try:
            element_map = getattr(context, "element_map", ()) or ()
            chunk_records = split_markdown_with_map(
                context.markdown, element_map, chunk_config
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            handle_failure("chunker", context, exc)
            continue

        if not chunk_records:
            chunk_records = [
                ChunkWithProvenance(
                    text=context.markdown,
                    char_start=0,
                    char_end=len(context.markdown),
                )
            ]

        chunks = [rec.text for rec in chunk_records]

        # Analyze formula content in chunks if formulas were extracted
        if config.extract_formulas and hasattr(context, 'formulas') and context.formulas:
            _, context.formula_stats = _embed_formulas_in_chunks(
                chunks,
                context.formulas,
                context.doc_id or context.slug
            )
            active_logger.debug(
                "Calculated formula stats for %d chunks in slug=%s",
                len(context.formula_stats) if context.formula_stats else 0,
                context.slug
            )

        # Generate embeddings if flag is enabled
        if config.emit_embeddings:
            embedding_start = time.perf_counter()
            embeddings_generated = 0
            embeddings_failed = 0

            for index, chunk_body in enumerate(chunks, start=1):
                chunk_id = f"{context.doc_id or context.slug}-{index:04d}"
                try:
                    if not chunk_body.strip():
                        active_logger.warning(
                            "Skipping embedding for empty chunk slug=%s chunk_id=%s",
                            context.slug,
                            chunk_id,
                        )
                        continue

                    embedding = generate_embedding(chunk_body)
                    config.embeddings[chunk_id] = embedding
                    embeddings_generated += 1

                    active_logger.debug(
                        "Generated embedding slug=%s chunk_id=%s dim=%d",
                        context.slug,
                        chunk_id,
                        embedding.shape[0],
                    )
                except Exception as exc:
                    embeddings_failed += 1
                    active_logger.error(
                        "Failed to generate embedding slug=%s chunk_id=%s error=%s",
                        context.slug,
                        chunk_id,
                        exc,
                        exc_info=True,
                    )
                    # Continue with next chunk - don't fail entire pipeline

            embedding_time_ms = (time.perf_counter() - embedding_start) * 1000
            active_logger.debug(
                "Embedding complete slug=%s generated=%d failed=%d time_ms=%.2f avg_ms=%.2f",
                context.slug,
                embeddings_generated,
                embeddings_failed,
                embedding_time_ms,
                embedding_time_ms / max(embeddings_generated, 1),
            )

            # Warn if embedding generation is slow
            if embeddings_generated > 0 and (embedding_time_ms / embeddings_generated) > 1000:
                active_logger.warning(
                    "Slow embedding generation detected slug=%s avg_time_ms=%.2f (>1000ms)",
                    context.slug,
                    embedding_time_ms / embeddings_generated,
                )

        # Generate music embeddings if flag is enabled and this is a music document
        if config.emit_music_embeddings and hasattr(context, 'music_elements') and context.music_elements:
            try:
                from grounding import music_descriptions, music_formatter

                active_logger.info("Generating music embeddings for slug=%s", context.slug)
                music_embedding_start = time.perf_counter()

                # Convert music elements to music21 stream for analysis
                stream = music_formatter._convert_elements_to_stream(context.music_elements)

                # Generate natural language description
                description = music_descriptions.generate_music_description(stream)

                # Generate embedding from description
                music_embedding = generate_embedding(description)

                # Store music embedding with special chunk_id
                # Use chunk index 0 for the music chunk (or append after text chunks)
                music_chunk_id = f"{context.doc_id or context.slug}_music_0001"
                config.embeddings[music_chunk_id] = music_embedding

                music_embedding_time_ms = (time.perf_counter() - music_embedding_start) * 1000

                active_logger.info(
                    "Music embedding generated slug=%s chunk_id=%s time_ms=%.2f",
                    context.slug,
                    music_chunk_id,
                    music_embedding_time_ms,
                )

                # Store music metadata for vector store
                if not hasattr(config, 'music_embedding_metadata'):
                    config.music_embedding_metadata = {}

                # Extract musical features for metadata
                tonic, mode = music_descriptions._analyze_key_signature(stream)
                time_sig = music_descriptions._get_time_signature(stream)
                key_obj = stream.analyze('key')
                harmony = music_descriptions._analyze_harmony(stream, key_obj)
                rhythm = music_descriptions._analyze_rhythm(stream)

                config.music_embedding_metadata[music_chunk_id] = {
                    "is_music": True,
                    "music_metadata": {
                        "key": f"{tonic} {mode}",
                        "time_signature": time_sig,
                        "harmony": harmony,
                        "rhythm": rhythm
                    },
                    "description": description,
                    "doc_id": context.doc_id or context.slug,
                    "file_path": str(Path(context.slug) / "music_chunk.md")
                }

                # Warn if music embedding generation is slow (target <100ms)
                if music_embedding_time_ms > 100:
                    active_logger.warning(
                        "Slow music embedding generation slug=%s time_ms=%.2f (>100ms target)",
                        context.slug,
                        music_embedding_time_ms,
                    )

            except Exception as exc:
                active_logger.error(
                    "Failed to generate music embedding slug=%s error=%s",
                    context.slug,
                    exc,
                    exc_info=True,
                )
                # Continue - don't fail pipeline on music embedding failure

        try:
            rendered_chunks = []
            for index, record in enumerate(chunk_records, start=1):
                chunk_body = record.text
                chunk_hash = hash_chunk(chunk_body, skip_front_matter=False)
                chunk_id = f"{context.doc_id or context.slug}-{index:04d}"
                has_embedding = chunk_id in config.embeddings

                # Derive page/section citation metadata from the element map.
                if element_map:
                    provenance = derive_chunk_metadata(record, element_map)
                    page_start = provenance.page_start
                    page_end = provenance.page_end
                    section_heading = provenance.section_heading
                else:
                    page_start = None
                    page_end = None
                    section_heading = None

                # Get formula stats for this chunk if available
                formula_kwargs = {}
                if hasattr(context, 'formula_stats') and context.formula_stats and index in context.formula_stats:
                    chunk_formula_stats = context.formula_stats[index]
                    formula_kwargs = {
                        'formula_count': chunk_formula_stats.get('formula_count'),
                        'inline_formula_count': chunk_formula_stats.get('inline_formula_count'),
                        'display_formula_count': chunk_formula_stats.get('display_formula_count'),
                        'formula_ids': chunk_formula_stats.get('formula_ids'),
                    }

                metadata = build_chunk_metadata(
                    doc_id=context.doc_id or context.slug,
                    source=context.source_path.name,
                    chunk_index=index,
                    chunk_hash=chunk_hash,
                    page_start=page_start,
                    page_end=page_end,
                    section_heading=section_heading,
                    has_embedding=has_embedding,
                    **formula_kwargs,
                )
                rendered_chunks.append(render_chunk(metadata, chunk_body))
        except Exception as exc:  # pragma: no cover - defensive guard
            handle_failure("chunk_metadata", context, exc)
            continue

        try:
            write_document(
                context,
                context.markdown,
                rendered_chunks,
                dry_run=config.dry_run,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            handle_failure("writer", context, exc)
            continue

        # Write music output files if present
        if hasattr(context, 'music_outputs') and context.music_outputs:
            try:
                # Music files go in same directory as doc.md
                music_dir = context.output_path.parent
                ensure_dir(music_dir)
                for format_name, content in context.music_outputs.items():
                    if format_name == "musicxml":
                        music_path = music_dir / "music.musicxml"
                        atomic_write(music_path, content)
                    elif format_name == "abc":
                        music_path = music_dir / "music.abc"
                        atomic_write(music_path, content)
                    elif format_name == "midi":
                        music_path = music_dir / "music.mid"
                        music_path.write_bytes(content)  # MIDI is binary
                    active_logger.debug("Wrote music file: %s", music_path)
            except Exception as exc:
                active_logger.error("Failed to write music files for slug=%s: %s", context.slug, exc, exc_info=True)
                # Don't fail the whole document if music files fail

        # Write formula output files if present
        if hasattr(context, 'formula_outputs') and context.formula_outputs:
            try:
                # Formula files go in formulas/ subdirectory
                formulas_dir = context.output_path.parent / "formulas"
                ensure_dir(formulas_dir)

                for format_name, formulas_dict in context.formula_outputs.items():
                    for formula_id, formula_content in formulas_dict.items():
                        if format_name == "latex":
                            formula_path = formulas_dir / f"{formula_id}.tex"
                            atomic_write(formula_path, formula_content)
                        elif format_name == "mathml":
                            formula_path = formulas_dir / f"{formula_id}.mathml"
                            atomic_write(formula_path, formula_content)
                        active_logger.debug("Wrote formula file: %s", formula_path)

                active_logger.info(
                    "Wrote %d formula files for slug=%s",
                    sum(len(formulas_dict) for formulas_dict in context.formula_outputs.values()),
                    context.slug
                )
            except Exception as exc:
                active_logger.error("Failed to write formula files for slug=%s: %s", context.slug, exc, exc_info=True)
                # Don't fail the whole document if formula files fail

        try:
            meta_yaml = build_meta_yaml(
                context,
                params=params_for_meta,
                tooling=None,
                collections=config.collections,
            )
            meta_path = context.output_path.parent / "meta.yaml"
            atomic_write(meta_path, meta_yaml)
        except Exception as exc:  # pragma: no cover - defensive guard
            handle_failure("meta", context, exc)
            continue

        try:
            # Build manifest entry with music metadata if present
            entry_kwargs = {
                "doc_id": context.doc_id or context.slug,
                "slug": context.slug,
                "orig_name": context.source_path.name,
                "chunk_count": len(rendered_chunks),
                "strategy": config.parser,
                "doc_path": str(Path(context.slug) / "doc.md"),
                "meta_path": str(Path(context.slug) / "meta.yaml"),
                "collections": config.collections,
            }

            # Add music-specific fields if this is a music document
            if hasattr(context, 'music_outputs') and context.music_outputs:
                entry_kwargs["content_type"] = "music" if config.parser == "omr" else "hybrid"
                entry_kwargs["music_format"] = config.music_format

                # Collect music file paths
                music_files = []
                for format_name in context.music_outputs.keys():
                    if format_name == "musicxml":
                        music_files.append(str(Path(context.slug) / "music.musicxml"))
                    elif format_name == "abc":
                        music_files.append(str(Path(context.slug) / "music.abc"))
                    elif format_name == "midi":
                        music_files.append(str(Path(context.slug) / "music.mid"))
                entry_kwargs["music_files"] = music_files

                # Extract music metadata from elements if available
                if hasattr(context, 'music_elements') and context.music_elements:
                    from grounding import music_formatter
                    metadata = music_formatter.extract_music_metadata(context.music_elements)
                    entry_kwargs["music_metadata"] = metadata

            # Add formula-specific fields if formulas were extracted
            if hasattr(context, 'formula_outputs') and context.formula_outputs:
                # Calculate formula statistics
                inline_count = sum(
                    1 for f in context.formulas if f.formula_type == "inline"
                ) if context.formulas else 0
                display_count = sum(
                    1 for f in context.formulas if f.formula_type == "display"
                ) if context.formulas else 0
                total_formulas = len(context.formulas) if context.formulas else 0

                # Determine complexity based on formula count
                if total_formulas < 5:
                    complexity = "simple"
                elif total_formulas < 15:
                    complexity = "moderate"
                else:
                    complexity = "complex"

                entry_kwargs["formula_metadata"] = {
                    "formula_count": total_formulas,
                    "inline_count": inline_count,
                    "display_count": display_count,
                    "complexity": complexity,
                }

                # Collect formula file paths
                formula_files = []
                for format_name, formulas_dict in context.formula_outputs.items():
                    for formula_id in formulas_dict.keys():
                        if format_name == "latex":
                            formula_files.append(str(Path(context.slug) / "formulas" / f"{formula_id}.tex"))
                        elif format_name == "mathml":
                            formula_files.append(str(Path(context.slug) / "formulas" / f"{formula_id}.mathml"))
                entry_kwargs["formula_files"] = formula_files

                # Set content_type if not already set by music
                if "content_type" not in entry_kwargs:
                    entry_kwargs["content_type"] = "scientific"

            entry = ManifestEntry(**entry_kwargs)
            manifest_data = ManifestManager.register_document(manifest_data, entry)
        except Exception as exc:  # pragma: no cover - defensive guard
            handle_failure("manifest", context, exc)
            continue

        context.chunk_count = len(rendered_chunks)
        stats.add_chunks(len(rendered_chunks))
        if progress_callback:
            progress_callback(context)

    # Write vector store if embeddings were generated
    if config.emit_embeddings and config.embeddings:
        try:
            # Collect music metadata if music embeddings were generated
            chunk_metadata = None
            if config.emit_music_embeddings and hasattr(config, 'music_embedding_metadata'):
                chunk_metadata = config.music_embedding_metadata
                active_logger.debug(
                    "Writing vector store with %d embeddings (%d music) to %s",
                    len(config.embeddings),
                    len(chunk_metadata),
                    config.output_dir,
                )
            else:
                active_logger.debug(
                    "Writing vector store with %d embeddings to %s",
                    len(config.embeddings),
                    config.output_dir,
                )

            write_vector_index(config.embeddings, config.output_dir, chunk_metadata=chunk_metadata)
        except Exception as exc:
            active_logger.error(
                "Failed to write vector store: %s",
                exc,
                exc_info=True,
            )
            # Don't fail entire pipeline - vector store is supplementary

    ManifestManager.write(manifest_data, manifest_path)
    return result
