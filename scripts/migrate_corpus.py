#!/usr/bin/env python3
"""
Corpus Migration Script

One-time migration script that merges existing per-agent corpora into a
single centralized corpus structure.

Usage:
    python scripts/migrate_corpus.py --source-dir /path/to/agents --target-dir ./corpus_root --dry-run
    python scripts/migrate_corpus.py --source-dir /path/to/agents --target-dir ./corpus_root
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Global verbose flag
VERBOSE = False


def log(message: str, verbose_only: bool = False) -> None:
    """Print message to stdout, optionally only in verbose mode."""
    if verbose_only and not VERBOSE:
        return
    print(message)


@dataclass
class DocumentInfo:
    """Information about a document to migrate."""

    agent_name: str
    slug: str
    source_path: Path
    doc_id: Optional[str] = None
    target_slug: Optional[str] = None  # May differ if collision resolved


@dataclass
class MigrationPlan:
    """Plan for the migration."""

    documents: list[DocumentInfo] = field(default_factory=list)
    collisions: dict[str, list[DocumentInfo]] = field(default_factory=dict)
    agent_doc_counts: dict[str, int] = field(default_factory=dict)
    skipped_agents: list[str] = field(default_factory=list)


@dataclass
class MigrationResult:
    """Result of the migration."""

    docs_copied: int = 0
    collisions_resolved: int = 0
    agent_defs_created: int = 0
    manifest_entries: int = 0
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate per-agent corpora to unified corpus structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run first (recommended)
    python scripts/migrate_corpus.py \\
        --source-dir "/Volumes/My Passport/Agents" \\
        --target-dir "./corpus_root" \\
        --dry-run

    # Actual migration
    python scripts/migrate_corpus.py \\
        --source-dir "/Volumes/My Passport/Agents" \\
        --target-dir "./corpus_root"
""",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Source directory containing agent folders (e.g., /Volumes/My Passport/Agents)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Target directory for unified corpus (e.g., ./corpus_root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def discover_agents(source_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """
    Discover agent folders with corpus subdirectories.

    Returns:
        Tuple of (agent_name -> corpus_path dict, list of skipped agent names)
    """
    agents: dict[str, Path] = {}
    skipped: list[str] = []

    if not source_dir.exists():
        log(f"ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)

    for item in sorted(source_dir.iterdir()):
        if not item.is_dir():
            continue

        corpus_path = item / "corpus"
        if corpus_path.exists() and corpus_path.is_dir():
            # Check if corpus has any document folders
            doc_folders = [
                d for d in corpus_path.iterdir() if d.is_dir() and not d.name.startswith("_")
            ]
            if doc_folders:
                agents[item.name] = corpus_path
            else:
                skipped.append(item.name)

    return agents, skipped


def get_doc_id_from_meta(meta_path: Path) -> Optional[str]:
    """Extract doc_id from meta.yaml file."""
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
            return meta.get("doc_id")
    except Exception:
        return None


def build_document_inventory(agents: dict[str, Path]) -> list[DocumentInfo]:
    """Build inventory of all documents across agents."""
    documents: list[DocumentInfo] = []

    for agent_name, corpus_path in agents.items():
        for doc_folder in sorted(corpus_path.iterdir()):
            if not doc_folder.is_dir() or doc_folder.name.startswith("_"):
                continue

            meta_path = doc_folder / "meta.yaml"
            doc_id = get_doc_id_from_meta(meta_path)

            documents.append(
                DocumentInfo(
                    agent_name=agent_name,
                    slug=doc_folder.name,
                    source_path=doc_folder,
                    doc_id=doc_id,
                    target_slug=doc_folder.name,  # Initially same as source
                )
            )

    return documents


def detect_collisions(documents: list[DocumentInfo]) -> dict[str, list[DocumentInfo]]:
    """
    Detect slug collisions across agents.

    Returns:
        Dict mapping slug -> list of documents with that slug (only where len > 1)
    """
    slug_to_docs: dict[str, list[DocumentInfo]] = defaultdict(list)

    for doc in documents:
        slug_to_docs[doc.slug].append(doc)

    # Return only actual collisions
    return {slug: docs for slug, docs in slug_to_docs.items() if len(docs) > 1}


def resolve_collisions(collisions: dict[str, list[DocumentInfo]]) -> None:
    """
    Resolve slug collisions by appending doc_id suffix to duplicates.

    Modifies documents in place, updating target_slug.
    """
    for slug, docs in collisions.items():
        # First document keeps original slug
        # Subsequent documents get doc_id suffix
        for i, doc in enumerate(docs[1:], start=1):
            if doc.doc_id:
                suffix = doc.doc_id[:8]
            else:
                # Fallback: use agent name if no doc_id
                suffix = doc.agent_name.lower()[:8]
            doc.target_slug = f"{slug}-{suffix}"


def build_migration_plan(source_dir: Path) -> MigrationPlan:
    """Build complete migration plan."""
    plan = MigrationPlan()

    # Discover agents
    agents, skipped = discover_agents(source_dir)
    plan.skipped_agents = skipped

    # Count docs per agent
    for agent_name, corpus_path in agents.items():
        doc_count = len(
            [d for d in corpus_path.iterdir() if d.is_dir() and not d.name.startswith("_")]
        )
        plan.agent_doc_counts[agent_name] = doc_count

    # Build document inventory
    plan.documents = build_document_inventory(agents)

    # Detect and resolve collisions
    plan.collisions = detect_collisions(plan.documents)
    resolve_collisions(plan.collisions)

    return plan


def copy_document(
    doc: DocumentInfo, target_corpus: Path, dry_run: bool
) -> bool:
    """
    Copy a document folder to the target corpus.

    Returns:
        True if successful, False otherwise
    """
    target_path = target_corpus / doc.target_slug

    if target_path.exists():
        log(f"  [SKIP] Already exists: {doc.target_slug}")
        return True

    if dry_run:
        if doc.slug != doc.target_slug:
            log(
                f"  [DRY RUN] Would copy: {doc.agent_name}/corpus/{doc.slug} -> corpus/{doc.target_slug}"
            )
        else:
            log(
                f"  [DRY RUN] Would copy: {doc.agent_name}/corpus/{doc.slug} -> corpus/{doc.slug}"
            )
        return True

    try:
        shutil.copytree(doc.source_path, target_path)
        if doc.slug != doc.target_slug:
            log(f"  Copied: {doc.agent_name}/corpus/{doc.slug} -> corpus/{doc.target_slug}")
        else:
            log(f"  Copied: {doc.agent_name}/corpus/{doc.slug}")
        return True
    except Exception as e:
        log(f"  [ERROR] Failed to copy {doc.slug}: {e}")
        return False


def update_meta_yaml(
    doc: DocumentInfo, target_corpus: Path, dry_run: bool
) -> bool:
    """
    Update meta.yaml with source_agent field.

    Returns:
        True if successful, False otherwise
    """
    meta_path = target_corpus / doc.target_slug / "meta.yaml"

    if dry_run:
        log(f"  [DRY RUN] Would update meta.yaml: source_agent={doc.agent_name.lower()}", verbose_only=True)
        return True

    try:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
        else:
            # Create minimal meta if missing
            meta = {"slug": doc.target_slug}

        # Add source_agent (lowercase, normalized)
        meta["source_agent"] = doc.agent_name.lower().replace(" ", "_").replace("-", "_")

        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return True
    except Exception as e:
        log(f"  [ERROR] Failed to update meta.yaml for {doc.target_slug}: {e}")
        return False


def load_manifest(manifest_path: Path) -> dict:
    """Load a manifest file, returning empty structure if not found."""
    if not manifest_path.exists():
        return {"docs": []}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"docs": []}


def merge_manifests(
    plan: MigrationPlan, target_corpus: Path, dry_run: bool
) -> int:
    """
    Merge all source manifests into single target manifest.

    Returns:
        Number of entries in merged manifest
    """
    merged_docs: list[dict] = []
    seen_doc_ids: set[str] = set()

    # Build lookup for target slugs using doc_id as key (unique identifier)
    doc_id_to_target_slug = {doc.doc_id: doc.target_slug for doc in plan.documents if doc.doc_id}
    # Fallback: also map by (agent_name, slug) for docs without doc_id
    agent_slug_to_target = {(doc.agent_name, doc.slug): doc.target_slug for doc in plan.documents}

    # Group documents by agent to find their manifests
    agent_docs: dict[str, list[DocumentInfo]] = defaultdict(list)
    for doc in plan.documents:
        agent_docs[doc.agent_name].append(doc)

    # Process each agent's manifest
    for agent_name, docs in agent_docs.items():
        if not docs:
            continue

        # Find source manifest
        source_manifest_path = docs[0].source_path.parent / "_index.json"
        source_manifest = load_manifest(source_manifest_path)

        for entry in source_manifest.get("docs", []):
            doc_id = entry.get("doc_id")
            if doc_id in seen_doc_ids:
                continue  # Skip duplicates

            # Update paths if slug was renamed
            orig_slug = entry.get("slug")
            # Look up target slug by doc_id first, then by (agent_name, slug)
            if doc_id and doc_id in doc_id_to_target_slug:
                new_slug = doc_id_to_target_slug[doc_id]
            else:
                new_slug = agent_slug_to_target.get((agent_name, orig_slug), orig_slug)

            if new_slug != orig_slug:
                entry["slug"] = new_slug
                if "doc_path" in entry:
                    entry["doc_path"] = entry["doc_path"].replace(orig_slug, new_slug)
                if "meta_path" in entry:
                    entry["meta_path"] = entry["meta_path"].replace(orig_slug, new_slug)

            merged_docs.append(entry)
            if doc_id:
                seen_doc_ids.add(doc_id)

    # Write merged manifest
    merged_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "docs": merged_docs,
    }

    target_manifest_path = target_corpus / "_index.json"

    if dry_run:
        log(f"\n[DRY RUN] Would write merged manifest with {len(merged_docs)} entries")
        return len(merged_docs)

    try:
        with open(target_manifest_path, "w", encoding="utf-8") as f:
            json.dump(merged_manifest, f, indent=2, ensure_ascii=False)
        log(f"\nWrote merged manifest: {len(merged_docs)} entries")
        return len(merged_docs)
    except Exception as e:
        log(f"[ERROR] Failed to write merged manifest: {e}")
        return 0


def generate_agent_definitions(
    plan: MigrationPlan, target_dir: Path, dry_run: bool
) -> int:
    """
    Generate agent definition YAML files.

    Returns:
        Number of agent definitions created
    """
    agents_dir = target_dir / "agents"

    if not dry_run:
        agents_dir.mkdir(parents=True, exist_ok=True)

    # Group documents by agent
    agent_slugs: dict[str, list[str]] = defaultdict(list)
    for doc in plan.documents:
        agent_slugs[doc.agent_name].append(doc.target_slug)

    created = 0
    for agent_name, slugs in agent_slugs.items():
        agent_file = agents_dir / f"{agent_name.lower()}.yaml"

        agent_def = {
            "name": agent_name.lower(),
            "description": f"Migrated from {agent_name} agent corpus",
            "corpus_filter": {"slugs": sorted(slugs)},
        }

        if dry_run:
            log(f"[DRY RUN] Would create: agents/{agent_name.lower()}.yaml ({len(slugs)} slugs)")
        else:
            try:
                with open(agent_file, "w", encoding="utf-8") as f:
                    yaml.dump(agent_def, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                log(f"Created: agents/{agent_name.lower()}.yaml ({len(slugs)} slugs)")
                created += 1
            except Exception as e:
                log(f"[ERROR] Failed to create agent definition for {agent_name}: {e}")

    return created if not dry_run else len(agent_slugs)


def print_plan(plan: MigrationPlan, source_dir: Path) -> None:
    """Print the migration plan."""
    print("\n" + "=" * 50)
    print("=== Corpus Migration ===")
    print("=" * 50)

    print(f"\nScanning source directory: {source_dir}")

    print("\nDiscovered agents:")
    for agent_name, count in sorted(plan.agent_doc_counts.items()):
        print(f"  - {agent_name}: {count} documents")
    for agent_name in plan.skipped_agents:
        print(f"  - {agent_name}: 0 documents (skipping)")

    if plan.collisions:
        print("\nCollision detection:")
        for slug, docs in plan.collisions.items():
            agents = ", ".join(d.agent_name for d in docs)
            renamed = [d for d in docs[1:]]
            rename_info = ", ".join(f"{d.agent_name}->{d.target_slug}" for d in renamed)
            print(f"  - {slug}: {agents} (will rename: {rename_info})")
    else:
        print("\nNo slug collisions detected.")

    total_docs = len(plan.documents)
    collision_count = sum(len(docs) - 1 for docs in plan.collisions.values())
    agent_count = len(plan.agent_doc_counts)

    print("\nMigration plan:")
    print(f"  - {total_docs} documents to copy")
    print(f"  - {collision_count} collisions to resolve")
    print(f"  - {agent_count} agent definitions to create")


def print_summary(result: MigrationResult, target_dir: Path, source_dir: Path, dry_run: bool) -> None:
    """Print migration summary."""
    print("\n" + "=" * 50)
    if dry_run:
        print("=== Dry Run Summary ===")
    else:
        print("=== Migration Summary ===")
    print("=" * 50)

    print(f"  - Documents copied: {result.docs_copied}")
    print(f"  - Collisions resolved: {result.collisions_resolved}")
    print(f"  - Agent definitions created: {result.agent_defs_created}")
    print(f"  - Merged manifest entries: {result.manifest_entries}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    if not dry_run:
        print("\n" + "-" * 50)
        print("Rollback instructions:")
        print(f"  To undo this migration, run:")
        print(f"    rm -rf {target_dir}/corpus {target_dir}/agents")
        print(f"  Original data remains at: {source_dir}")
        print("-" * 50)
        print("\nMigration complete!")
    else:
        print("\n" + "-" * 50)
        print("This was a dry run. No files were modified.")
        print("To perform the actual migration, run without --dry-run")
        print("-" * 50)


def validate_manifest(target_corpus: Path) -> bool:
    """Validate the merged manifest can be loaded."""
    manifest_path = target_corpus / "_index.json"
    if not manifest_path.exists():
        return True  # OK for dry run

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Basic validation
        if "docs" not in manifest:
            log("WARNING: Manifest missing 'docs' key")
            return False
        return True
    except Exception as e:
        log(f"ERROR: Manifest validation failed: {e}")
        return False


def run_migration(args: argparse.Namespace) -> int:
    """
    Run the migration.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    if args.verbose:
        global VERBOSE
        VERBOSE = True

    # Print backup reminder
    if not args.dry_run:
        print("\n" + "!" * 50)
        print("IMPORTANT: Ensure you have a backup before proceeding!")
        print(f"  Source: {args.source_dir}")
        print(f"  Target: {args.target_dir}")
        print("!" * 50)

    # Build migration plan
    plan = build_migration_plan(args.source_dir)

    if not plan.documents:
        log("ERROR: No documents found to migrate.")
        return 1

    # Print plan
    print_plan(plan, args.source_dir)

    # Create target directories
    target_corpus = args.target_dir / "corpus"
    target_originals = args.target_dir / "originals"

    if not args.dry_run:
        target_corpus.mkdir(parents=True, exist_ok=True)
        target_originals.mkdir(parents=True, exist_ok=True)

    result = MigrationResult()

    # Copy documents
    print("\nCopying documents:")
    for doc in plan.documents:
        if copy_document(doc, target_corpus, args.dry_run):
            result.docs_copied += 1
            # Track collision resolutions
            if doc.slug != doc.target_slug:
                result.collisions_resolved += 1
        else:
            result.errors.append(f"Failed to copy: {doc.slug}")

    # Update meta.yaml files
    if not args.dry_run:
        print("\nUpdating meta.yaml files:")
        for doc in plan.documents:
            if not update_meta_yaml(doc, target_corpus, args.dry_run):
                result.errors.append(f"Failed to update meta: {doc.target_slug}")

    # Merge manifests
    result.manifest_entries = merge_manifests(plan, target_corpus, args.dry_run)

    # Generate agent definitions
    print("\nGenerating agent definitions:")
    result.agent_defs_created = generate_agent_definitions(plan, args.target_dir, args.dry_run)

    # Validate
    if not args.dry_run:
        if not validate_manifest(target_corpus):
            result.errors.append("Manifest validation failed")

    # Print summary
    print_summary(result, args.target_dir, args.source_dir, args.dry_run)

    return 0 if not result.errors else 1


def main() -> None:
    """Main entry point."""
    args = parse_args()
    sys.exit(run_migration(args))


if __name__ == "__main__":
    main()
