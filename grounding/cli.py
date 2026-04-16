"""CLI entry point for grounding."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from grounding.logging_setup import setup_logging
from grounding.pipeline import PipelineConfig
from grounding.controller import run_controller
from grounding.scanner import scan_pdfs
from grounding.utils import validate_collection_name, atomic_write


def agents_list_command(args: argparse.Namespace) -> int:
    """Handle the 'agents list' subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from grounding.agent_filter import load_agent_config, AgentFilterError

    agents_dir = args.agents_dir.resolve()

    if not agents_dir.exists():
        print(f"Error: Agents directory does not exist: {agents_dir}", file=sys.stderr)
        return 1

    # Find all YAML files in the agents directory
    agent_files = sorted(agents_dir.glob("*.yaml")) + sorted(agents_dir.glob("*.yml"))

    if not agent_files:
        print(f"No agent definitions found in: {agents_dir}")
        return 0

    print(f"\n=== Available Agents ({len(agent_files)}) ===\n")
    print(f"{'Name':<20} {'Collections':<30} {'Description'}")
    print("-" * 80)

    for agent_file in agent_files:
        agent_name = agent_file.stem
        try:
            config = load_agent_config(agent_name, agents_dir)
            collections = ", ".join(config.collections) if config.collections else "-"
            description = config.description[:40] + "..." if len(config.description) > 40 else config.description
            print(f"{config.name:<20} {collections:<30} {description}")
        except AgentFilterError as exc:
            print(f"{agent_name:<20} {'<error>':<30} {str(exc)[:40]}")

    print(f"\nAgents directory: {agents_dir}")
    return 0


def agents_show_command(args: argparse.Namespace) -> int:
    """Handle the 'agents show' subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from grounding.agent_filter import load_agent_config, AgentFilterError
    from grounding.manifest import ManifestManager

    agents_dir = args.agents_dir.resolve()
    agent_name = args.agent_name

    if not agents_dir.exists():
        print(f"Error: Agents directory does not exist: {agents_dir}", file=sys.stderr)
        return 1

    try:
        config = load_agent_config(agent_name, agents_dir)
    except AgentFilterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== Agent: {config.name} ===\n")

    if config.description:
        print(f"Description: {config.description}\n")

    print("Filter Configuration:")
    if config.collections:
        print(f"  Collections: {', '.join(config.collections)}")
    else:
        print("  Collections: (none)")

    if config.slugs:
        print(f"  Explicit slugs: {len(config.slugs)}")
        for slug in config.slugs[:10]:  # Show first 10
            print(f"    - {slug}")
        if len(config.slugs) > 10:
            print(f"    ... and {len(config.slugs) - 10} more")
    else:
        print("  Explicit slugs: (none)")

    if config.exclude_slugs:
        print(f"  Excluded slugs: {', '.join(config.exclude_slugs)}")

    # If corpus is specified, show matching documents
    if args.corpus:
        corpus_path = args.corpus.resolve()
        manifest_path = corpus_path / "_index.json"

        if manifest_path.exists():
            from grounding.agent_filter import filter_manifest

            manifest = ManifestManager.load(manifest_path)
            filtered = filter_manifest(manifest, config)

            print(f"\nMatching Documents ({len(filtered.docs)}/{len(manifest.docs)}):")
            for doc in filtered.docs[:20]:  # Show first 20
                collections_str = f" [{', '.join(doc.collections)}]" if doc.collections else ""
                print(f"  - {doc.slug}{collections_str}")
            if len(filtered.docs) > 20:
                print(f"  ... and {len(filtered.docs) - 20} more")
        else:
            print(f"\n(Corpus not found at {corpus_path} - cannot show matching documents)")

    print(f"\nAgent file: {agents_dir / f'{agent_name}.yaml'}")
    return 0


def _create_agents_parser(subparsers) -> argparse.ArgumentParser:
    """Create the agents subcommand parser with sub-subcommands."""
    agents_parser = subparsers.add_parser(
        "agents",
        help="Manage agent definitions",
        description="List and inspect agent definitions for corpus filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all agents
  grounding agents list --agents-dir ./agents

  # Show agent details
  grounding agents show scientist --agents-dir ./agents

  # Show agent with matching documents from corpus
  grounding agents show scientist --agents-dir ./agents --corpus ./corpus
        """,
    )

    agents_subparsers = agents_parser.add_subparsers(dest="agents_command", help="Agent commands")

    # agents list
    list_parser = agents_subparsers.add_parser(
        "list",
        help="List all available agents",
        description="List all agent definitions in the agents directory.",
    )
    list_parser.add_argument(
        "--agents-dir",
        type=Path,
        required=True,
        help="Directory containing agent YAML definitions",
    )
    list_parser.set_defaults(func=agents_list_command)

    # agents show
    show_parser = agents_subparsers.add_parser(
        "show",
        help="Show details of a specific agent",
        description="Display detailed information about an agent's filter configuration.",
    )
    show_parser.add_argument(
        "agent_name",
        type=str,
        help="Name of the agent to show (without .yaml extension)",
    )
    show_parser.add_argument(
        "--agents-dir",
        type=Path,
        required=True,
        help="Directory containing agent YAML definitions",
    )
    show_parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus directory to show matching documents",
    )
    show_parser.set_defaults(func=agents_show_command)

    return agents_parser


def embeddings_command(args: argparse.Namespace) -> int:
    """Handle the 'embeddings' subcommand for generating vector embeddings.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error/stale)
    """
    from grounding.manifest import ManifestManager
    from grounding.embedder import generate_embedding
    from grounding.vector_store import (
        write_vector_index,
        check_index_staleness,
        append_to_vector_index,
        tombstone_documents,
        FAISS_INDEX_FILENAME,
    )
    from grounding.bm25 import (
        write_bm25_index,
        append_to_bm25_index,
        tombstone_bm25_documents,
    )

    logger = setup_logging(verbose=args.verbose, quiet_progress=not args.verbose)

    corpus_path = args.corpus.resolve()
    manifest_path = corpus_path / "_index.json"

    # Validate corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus directory does not exist: {corpus_path}", file=sys.stderr)
        return 1

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Load manifest
    manifest = ManifestManager.load(manifest_path)
    total_docs = len(manifest.docs)

    # Load agent config if specified
    agent_config = None
    if args.agent:
        from grounding.agent_filter import load_agent_config, filter_manifest, AgentFilterError

        # Determine agents directory
        if args.agents_dir:
            agents_dir = args.agents_dir.resolve()
        else:
            # Default: look for agents/ next to corpus or in parent
            agents_dir = corpus_path / "agents"
            if not agents_dir.exists():
                agents_dir = corpus_path.parent / "agents"

        if not agents_dir.exists():
            print(
                f"Error: Agents directory not found: {agents_dir}\n"
                f"Specify with --agents-dir or create agents/ directory",
                file=sys.stderr
            )
            return 1

        try:
            agent_config = load_agent_config(args.agent, agents_dir)
            manifest = filter_manifest(manifest, agent_config)
        except AgentFilterError as exc:
            print(f"Error loading agent config: {exc}", file=sys.stderr)
            return 1

    filtered_docs = len(manifest.docs)

    # Determine output directory
    if args.out:
        output_dir = args.out.resolve()
    elif args.agent:
        output_dir = corpus_path / "embeddings" / args.agent
    else:
        output_dir = corpus_path / "embeddings" / "full"

    # Handle --check flag (staleness check)
    if args.check:
        return _check_embeddings_staleness(
            corpus_path, output_dir, args.agent, filtered_docs, total_docs, manifest
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if incremental mode and index exists
    index_exists = (output_dir / FAISS_INDEX_FILENAME).exists()
    incremental_mode = args.incremental and index_exists

    # Build doc_id set from filtered manifest for quick lookup
    manifest_doc_ids = {doc.doc_id for doc in manifest.docs}
    manifest_doc_map = {doc.doc_id: doc for doc in manifest.docs}

    # Initialize incremental tracking
    new_doc_ids = set()
    deleted_doc_ids = set()
    updated_doc_ids = set()
    skipped_doc_ids = set()

    if incremental_mode:
        # Run staleness check to determine what needs processing
        try:
            staleness_report = check_index_staleness(corpus_path, output_dir)
            new_doc_ids = set(staleness_report.new_docs) & manifest_doc_ids
            deleted_doc_ids = set(staleness_report.deleted_docs)
            updated_doc_ids = set(staleness_report.updated_docs) & manifest_doc_ids

            # Warn if rebuild recommended
            if staleness_report.should_rebuild:
                print(
                    f"Warning: Tombstone ratio is high ({staleness_report.tombstone_ratio:.1%}). "
                    f"Consider a full rebuild for better performance.",
                    file=sys.stderr
                )

            # Documents to skip (already in index and not updated)
            skipped_doc_ids = manifest_doc_ids - new_doc_ids - updated_doc_ids

        except FileNotFoundError:
            # Index doesn't exist - fall back to full generation
            print("Index not found. Falling back to full generation.", file=sys.stderr)
            incremental_mode = False
            new_doc_ids = manifest_doc_ids

    # Print progress header
    print("\n=== Embedding Generation ===\n")
    if args.agent:
        print(f"Agent: {args.agent}")
        if args.agents_dir:
            print(f"Agents directory: {args.agents_dir}")
        print(f"Corpus: {corpus_path}")
        print(f"\nLoading agent configuration...")
        if agent_config:
            slugs_count = len(agent_config.slugs) if agent_config.slugs else 0
            cols_count = len(agent_config.collections) if agent_config.collections else 0
            print(f"  Agent '{args.agent}' filter: {slugs_count} slugs, {cols_count} collections")
        print(f"\nFiltering corpus...")
        print(f"  Filtered: {filtered_docs} documents (from {total_docs} total)")
    else:
        print(f"Corpus: {corpus_path}")
        print(f"Documents: {total_docs}")

    if filtered_docs == 0:
        print("\nNo documents match the filter. Nothing to embed.")
        return 0

    # Show mode-specific info
    if incremental_mode:
        print(f"\nMode: Incremental update")
        print(f"  New documents: {len(new_doc_ids)}")
        print(f"  Updated documents: {len(updated_doc_ids)}")
        print(f"  Deleted documents: {len(deleted_doc_ids)}")
        print(f"  Skipping: {len(skipped_doc_ids)} existing documents")

        docs_to_process = new_doc_ids | updated_doc_ids
        if not docs_to_process and not deleted_doc_ids:
            print("\nNo changes detected. Embeddings are up to date.")
            return 0
    else:
        print(f"\nMode: Full generation")
        docs_to_process = manifest_doc_ids

    # Tombstone deleted documents first (incremental mode only)
    if incremental_mode and deleted_doc_ids:
        print(f"\nTombstoning {len(deleted_doc_ids)} deleted documents...")
        tombstoned_count = tombstone_documents(list(deleted_doc_ids), output_dir)
        print(f"  Tombstoned {tombstoned_count} chunks")
        tombstone_bm25_documents(list(deleted_doc_ids), output_dir)

    # Filter documents to only those we need to process
    docs_to_embed = [doc for doc in manifest.docs if doc.doc_id in docs_to_process]

    if not docs_to_embed:
        if incremental_mode and deleted_doc_ids:
            # Only deletions, no new embeddings
            print("\nDone! (tombstones applied, no new embeddings)")
            return 0
        print("\nNo documents to embed.")
        return 0

    print(f"\nGenerating embeddings for {len(docs_to_embed)} documents...")

    # Collect all chunks from documents to process
    embeddings = {}
    chunk_metadata = {}
    chunk_bodies: dict[str, str] = {}
    total_chunks = 0

    with tqdm(total=len(docs_to_embed), desc="Processing docs", unit="doc", file=sys.stderr) as pbar:
        for doc in docs_to_embed:
            doc_dir = corpus_path / doc.slug / "chunks"
            if not doc_dir.exists():
                logger.warning(f"Chunks directory not found for {doc.slug}, skipping")
                pbar.update(1)
                continue

            # Get file_sha1 from meta.yaml for update tracking
            file_sha1 = None
            meta_path = corpus_path / doc.slug / "meta.yaml"
            if meta_path.exists():
                try:
                    import yaml
                    with open(meta_path, "r") as mf:
                        meta = yaml.safe_load(mf)
                        if meta and "hashes" in meta:
                            file_sha1 = meta["hashes"].get("file_sha1")
                except Exception:
                    pass

            # Read all chunk files
            chunk_files = sorted(doc_dir.glob("ch_*.md"))
            for chunk_file in chunk_files:
                chunk_text = chunk_file.read_text(encoding="utf-8")

                # Skip YAML front matter if present
                if chunk_text.startswith("---"):
                    end_marker = chunk_text.find("---", 3)
                    if end_marker != -1:
                        chunk_text = chunk_text[end_marker + 3:].strip()

                if not chunk_text.strip():
                    continue

                # Generate embedding
                chunk_id = f"{doc.doc_id}_{chunk_file.stem}"
                try:
                    embedding = generate_embedding(chunk_text)
                    embeddings[chunk_id] = embedding
                    chunk_metadata[chunk_id] = {
                        "doc_id": doc.doc_id,
                        "file_path": str(chunk_file.relative_to(corpus_path)),
                        "is_music": doc.content_type == "music" if doc.content_type else False,
                        "file_sha1": file_sha1,
                    }
                    chunk_bodies[chunk_id] = chunk_text
                    total_chunks += 1
                except Exception as exc:
                    logger.warning(f"Failed to embed {chunk_id}: {exc}")

            pbar.update(1)

    if not embeddings:
        print("\nNo chunks found to embed.")
        return 0

    # Write vector index (full or append)
    print(f"\nWriting vector index...")
    # Parallel arrays for BM25 (preserve insertion order, same as FAISS)
    ordered_ids = list(embeddings.keys())
    ordered_bodies = [chunk_bodies[cid] for cid in ordered_ids]
    ordered_doc_ids = [chunk_metadata[cid].get("doc_id") for cid in ordered_ids]

    if incremental_mode:
        # Tombstone updated docs before appending new embeddings
        if updated_doc_ids:
            print(f"  Tombstoning {len(updated_doc_ids)} updated documents...")
            tombstone_documents(list(updated_doc_ids), output_dir)
            tombstone_bm25_documents(list(updated_doc_ids), output_dir)

        # Append new embeddings
        vectors_added = append_to_vector_index(embeddings, output_dir, chunk_metadata)
        print(f"  Appended {vectors_added} vectors")
        bm25_added = append_to_bm25_index(
            ordered_bodies, ordered_ids, output_dir, new_doc_ids=ordered_doc_ids
        )
        print(f"  Appended {bm25_added} BM25 entries")
    else:
        write_vector_index(embeddings, output_dir, chunk_metadata)
        write_bm25_index(
            ordered_bodies, ordered_ids, output_dir, chunk_doc_ids=ordered_doc_ids
        )
        print(f"  Wrote BM25 index: {len(ordered_ids)} chunks")

    # Write timestamp file
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    timestamp_file = output_dir / ".timestamp"
    atomic_write(timestamp_file, timestamp)

    print(f"  Output: {output_dir / FAISS_INDEX_FILENAME}")
    print(f"  Chunks indexed: {total_chunks:,}")
    print(f"  Timestamp: {timestamp}")

    # Print summary for incremental mode
    if incremental_mode:
        print(f"\n=== Incremental Update Summary ===")
        print(f"  New: {len(new_doc_ids)} documents")
        print(f"  Updated: {len(updated_doc_ids)} documents")
        print(f"  Deleted: {len(deleted_doc_ids)} documents")
        print(f"  Skipped: {len(skipped_doc_ids)} existing documents")

    print(f"\nDone!")

    return 0


def _check_embeddings_staleness(
    corpus_path: Path,
    embeddings_path: Path,
    agent_name: Optional[str],
    filtered_docs: int,
    total_docs: int,
    manifest,
) -> int:
    """Check if embeddings are stale compared to corpus.

    Enhanced output shows new/deleted/updated document counts and
    recommendations for incremental vs full rebuild.

    Returns:
        0 if embeddings are up to date, 1 if stale or missing
    """
    from grounding.vector_store import (
        check_index_staleness,
        load_vector_index,
        FAISS_INDEX_FILENAME,
        CHUNK_MAP_FILENAME,
    )

    agent_label = agent_name if agent_name else "full"
    print(f"\n=== Embedding Staleness Check: {agent_label} ===\n")

    index_path = embeddings_path / FAISS_INDEX_FILENAME
    timestamp_file = embeddings_path / ".timestamp"

    # Check if index exists
    if not index_path.exists():
        print(f"Index: {embeddings_path}")
        print(f"  Status: MISSING")
        print(f"\nRun 'grounding embeddings' to generate embeddings")
        return 1

    # Load index info
    try:
        index, chunk_map = load_vector_index(embeddings_path)
        index_size = index.ntotal
        tombstone_count = chunk_map.get("tombstone_count", 0)
        tombstone_pct = (tombstone_count / index_size * 100) if index_size > 0 else 0
        created_utc = chunk_map.get("created_utc", "unknown")
        updated_utc = chunk_map.get("updated_utc", created_utc)
    except Exception as exc:
        print(f"Index: {embeddings_path}")
        print(f"  Status: ERROR ({exc})")
        return 1

    # Display index info
    print(f"Index: {embeddings_path}")
    print(f"  Vectors: {index_size:,}")
    print(f"  Tombstones: {tombstone_count} ({tombstone_pct:.2f}%)")
    print(f"  Created: {created_utc}")
    print(f"  Updated: {updated_utc}")

    # Check staleness
    try:
        staleness_report = check_index_staleness(corpus_path, embeddings_path)
    except Exception as exc:
        print(f"\nStaleness check failed: {exc}")
        return 1

    # Filter staleness report by the filtered manifest docs if agent is used
    # (new/deleted/updated should only count docs that match the agent filter)
    filtered_doc_ids = {doc.doc_id for doc in manifest.docs}
    new_docs_filtered = [d for d in staleness_report.new_docs if d in filtered_doc_ids]
    updated_docs_filtered = [d for d in staleness_report.updated_docs if d in filtered_doc_ids]
    # For deleted docs, they should be in index but not in filtered manifest
    deleted_docs_filtered = [d for d in staleness_report.deleted_docs if d not in filtered_doc_ids or d in staleness_report.deleted_docs]
    # Re-filter: deleted docs are those in index but not in corpus manifest anymore
    # For agent filter: only report if we expected them in the agent's scope
    deleted_docs_filtered = staleness_report.deleted_docs  # Keep as-is (index has them, manifest doesn't)

    # Display changes
    print(f"\nChanges detected:")
    print(f"  New documents: {len(new_docs_filtered)}")
    print(f"  Deleted documents: {len(deleted_docs_filtered)}")
    print(f"  Updated documents: {len(updated_docs_filtered)}")

    # Determine staleness based on filtered counts
    is_stale_filtered = bool(new_docs_filtered or deleted_docs_filtered or updated_docs_filtered)

    # Show recommendation
    if not is_stale_filtered:
        print(f"\nStatus: UP TO DATE")
        return 0

    print(f"\nStatus: STALE")

    # Estimate time savings
    new_count = len(new_docs_filtered) + len(updated_docs_filtered)
    skip_count = filtered_docs - new_count

    if staleness_report.should_rebuild:
        print(f"\nRecommendation: Full rebuild recommended")
        print(f"  Reason: Tombstone ratio ({staleness_report.tombstone_ratio:.1%}) exceeds threshold")
        print(f"  Run: grounding embeddings --agent {agent_name} --corpus <path>")
    elif new_count > 0:
        # Rough estimate: 1 second per chunk, ~10 chunks per doc
        est_incr_time = new_count * 10  # seconds
        est_full_time = filtered_docs * 10  # seconds

        def format_time(seconds):
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds // 60}m"
            else:
                return f"{seconds / 3600:.1f}h"

        print(f"\nRecommendation: Use --incremental")
        print(f"  Estimated time: ~{format_time(est_incr_time)} incremental vs ~{format_time(est_full_time)} full")
        print(f"  Run: grounding embeddings --agent {agent_name} --corpus <path> --incremental")
    else:
        print(f"\nRecommendation: Run incremental to apply deletions")
        print(f"  Run: grounding embeddings --agent {agent_name} --corpus <path> --incremental")

    return 1


def _create_embeddings_parser(subparsers) -> argparse.ArgumentParser:
    """Create the embeddings subcommand parser."""
    embeddings_parser = subparsers.add_parser(
        "embeddings",
        help="Generate vector embeddings from an existing corpus",
        description="Generate FAISS vector embeddings for semantic search on a processed corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for specific agent
  grounding embeddings --agent scientist --corpus ./corpus_root/corpus

  # With explicit output directory
  grounding embeddings --agent scientist --corpus ./corpus --out ./my-embeddings/scientist

  # Check if embeddings are stale
  grounding embeddings --agent scientist --corpus ./corpus --check

  # Incremental update (skip existing documents)
  grounding embeddings --agent scientist --corpus ./corpus --incremental

  # Full corpus (backward compatible)
  grounding embeddings --corpus ./corpus --out ./embeddings

  # With custom agents directory
  grounding embeddings --agent scientist --corpus ./corpus --agents-dir ./custom-agents
        """,
    )

    embeddings_parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the processed corpus directory (containing _index.json)",
    )
    embeddings_parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent name for filtered embeddings (e.g., 'scientist')",
    )
    embeddings_parser.add_argument(
        "--agents-dir",
        type=Path,
        default=None,
        help="Directory containing agent YAML configs (default: <corpus>/../agents/ or <corpus>/agents/)",
    )
    embeddings_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for embeddings (default: <corpus>/embeddings/<agent>/ or <corpus>/embeddings/full/)",
    )
    embeddings_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if embeddings are stale (exit code 1 if stale, 0 if fresh)",
    )
    embeddings_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental update: only embed new/updated documents, tombstone deleted ones",
    )
    embeddings_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    embeddings_parser.set_defaults(func=embeddings_command)
    return embeddings_parser


def main():
    """CLI entry point for grounding."""
    # Check if first argument is a subcommand
    subcommands = {"embeddings", "agents", "eval"}
    if len(sys.argv) > 1 and sys.argv[1] in subcommands:
        from grounding.eval.cli import _create_eval_parser

        # Use subcommand parser
        parser = argparse.ArgumentParser(
            description="grounding - Convert PDFs to LLM-ready Markdown chunks.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        _create_embeddings_parser(subparsers)
        _create_agents_parser(subparsers)
        _create_eval_parser(subparsers)

        args = parser.parse_args()
        if hasattr(args, "func"):
            sys.exit(args.func(args))
        else:
            parser.print_help()
            sys.exit(1)

    # Original CLI behavior for PDF conversion
    parser = argparse.ArgumentParser(
        description="Convert PDFs to LLM-ready Markdown chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  grounding ./pdfs ./corpus
  grounding ./docs ./output --chunk-size 800 --parser unstructured
  grounding ./research ./corpus --ocr on --dry-run
  grounding ./sheet-music ./corpus --parser omr --music-format musicxml
  grounding ./music-books ./corpus --parser hybrid --music-format all
  grounding ./papers ./corpus --extract-formulas --formula-format latex
  grounding ./textbooks ./corpus --extract-formulas --formula-format both

Subcommands:
  grounding embeddings --corpus ./corpus --agent scientist
  grounding agents list --agents-dir ./agents
  grounding agents show scientist --agents-dir ./agents
        """,
    )

    # Positional arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing PDF or EPUB files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for processed corpus",
    )

    # Optional arguments
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Target characters per chunk (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap characters between chunks (default: 150)",
    )
    parser.add_argument(
        "--parser",
        choices=["unstructured", "omr", "hybrid"],
        default="unstructured",
        help="PDF parser: unstructured, omr (music notation), hybrid (auto-detect text/music) (default: unstructured)",
    )
    parser.add_argument(
        "--ocr",
        choices=["auto", "on", "off"],
        default="auto",
        help="OCR mode: auto (detect), on (always), off (never) (default: auto)",
    )
    parser.add_argument(
        "--music-format",
        choices=["musicxml", "abc", "midi", "all"],
        default="musicxml",
        help="Music output format: musicxml, abc, midi, all (default: musicxml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print operations without writing files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before processing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--emit-embeddings",
        action="store_true",
        help="Generate vector embeddings for semantic search",
    )
    parser.add_argument(
        "--emit-music-embeddings",
        action="store_true",
        help="Generate embeddings for music content (requires --parser omr or hybrid)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["faiss"],
        default="faiss",
        help="Vector database to use (currently only 'faiss' supported) (default: faiss)",
    )
    parser.add_argument(
        "--extract-formulas",
        action="store_true",
        help="Extract mathematical formulas from PDFs (requires pix2tex)",
    )
    parser.add_argument(
        "--formula-format",
        choices=["latex", "mathml", "both"],
        default="latex",
        help="Formula output format: latex, mathml, both (default: latex)",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default=None,
        help="Comma-separated collection tags for documents (e.g., 'science,biology'). Tags must be lowercase kebab-case. Used for agent filtering.",
    )

    args = parser.parse_args()

    # Parse and validate collections
    collection_list = None
    if args.collections:
        collection_list = [c.strip() for c in args.collections.split(",") if c.strip()]
        for name in collection_list:
            if not validate_collection_name(name):
                print(
                    f"Error: Invalid collection name '{name}'. "
                    "Collections must be lowercase kebab-case (e.g., 'science', 'music-theory').",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Validate chunk parameters
    if args.chunk_overlap >= args.chunk_size:
        print(
            f"Error: Chunk overlap ({args.chunk_overlap}) must be less than chunk size ({args.chunk_size})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate input directory exists
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Validate vector database selection
    if args.emit_embeddings and args.vector_db != "faiss":
        print(
            f"Error: --vector-db '{args.vector_db}' not yet implemented. Use 'faiss'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate music embeddings
    if args.emit_music_embeddings and args.parser not in ["omr", "hybrid"]:
        print(
            f"Error: --emit-music-embeddings requires --parser omr or hybrid. "
            f"Current parser: {args.parser}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate OMR prerequisites
    if args.parser in ["omr", "hybrid"]:
        try:
            from grounding.omr_parser import check_audiveris_available
            if not check_audiveris_available():
                print(
                    f"Error: --parser '{args.parser}' requires Audiveris and JRE >=11.\n"
                    "See docs/epics/epic-7-installation-guide.md for installation instructions.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except ImportError as exc:
            print(
                f"Error: OMR support not available: {exc}\n"
                "Install required dependencies: pip install music21 pillow",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate formula extraction prerequisites
    if args.extract_formulas:
        try:
            import pix2tex  # noqa: F401
        except ImportError:
            print(
                "Error: --extract-formulas requires pix2tex.\n"
                "Install with: pip install pix2tex torch pypdfium2 scipy\n"
                "See docs/epics/epic-8-installation-guide.md for detailed instructions.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Discover PDFs
    pdf_files = scan_pdfs(args.input_dir)
    if not pdf_files:
        print(f"Error: No PDF or EPUB files found in input directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Prepare pipeline configuration
    config = PipelineConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        parser=args.parser,
        ocr_mode=args.ocr,
        allow_plaintext_fallback=True,  # Always use plaintext formatting
        dry_run=args.dry_run,
        clean=args.clean,
        emit_embeddings=args.emit_embeddings,
        emit_music_embeddings=args.emit_music_embeddings,
        extract_formulas=args.extract_formulas,
        music_format=args.music_format,
        formula_format=args.formula_format,
        collections=collection_list,
        metadata={
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "parser": args.parser,
            "music_format": args.music_format,
            "formula_format": args.formula_format,
        },
    )

    # Set up logging - suppress INFO logs to keep progress bar clean
    logger = setup_logging(verbose=args.verbose, quiet_progress=not args.verbose)

    # Dry-run mode: print config and exit
    if args.dry_run:
        print("Dry-run mode enabled. Configuration:")
        print(f"  input_dir: {config.input_dir}")
        print(f"  output_dir: {config.output_dir}")
        print(f"  chunk_size: {args.chunk_size}")
        print(f"  chunk_overlap: {args.chunk_overlap}")
        print(f"  parser: {config.parser}")
        print(f"  ocr: {config.ocr_mode}")
        print(f"  music_format: {config.music_format}")
        print(f"  extract_formulas: {config.extract_formulas}")
        print(f"  formula_format: {config.formula_format}")
        print(f"  collections: {config.collections}")
        print(f"  dry_run: {config.dry_run}")
        print(f"  clean: {config.clean}")
        print(f"  emit_embeddings: {config.emit_embeddings}")
        print(f"  emit_music_embeddings: {config.emit_music_embeddings}")
        print(f"  vector_db: {args.vector_db}")
        print(f"\nFound {len(pdf_files)} document(s) in {args.input_dir}")
        sys.exit(0)

    logger.debug("Processing %d document(s) from %s", len(pdf_files), args.input_dir)
    logger.debug("Configuration: %s", config)

    # Calculate total for progress tracking
    total_files = len(pdf_files)
    total_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)

    # Use MB-based progress for better feedback on large files
    with tqdm(
        total=total_mb,
        desc="Processing PDFs",
        unit="MB",
        unit_scale=False,
        unit_divisor=1.0,
        leave=True,
        file=sys.stderr,
        dynamic_ncols=True,
    ) as pbar:
        def on_progress(context) -> None:
            # Show current file being processed
            if context.status == "parsing":
                file_size_mb = context.source_path.stat().st_size / (1024 * 1024)
                pbar.set_description(f"Processing: {context.source_path.name[:40]} ({file_size_mb:.1f}MB)")
            # Increment progress by file size when file completes (success or failure)
            elif context.status in ("success", "failed"):
                file_size_mb = context.source_path.stat().st_size / (1024 * 1024)
                pbar.update(file_size_mb)
                pbar.set_description("Processing PDFs")

        result = run_controller(
            config,
            files=pdf_files,
            progress_callback=on_progress,
            logger_override=logger,
        )
        stats = result.stats

    # Print summary
    print("\n" + stats.get_summary())

    # Exit with error code if any files failed
    if stats.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
