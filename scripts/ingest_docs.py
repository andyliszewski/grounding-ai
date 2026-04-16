#!/usr/bin/env python3
"""
Ingest markdown and Word documents into the corpus.

Supports: .md, .docx, .doc
Uses grounding's chunker and writer for consistency.
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent dir to path for grounding imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from grounding.chunker import split_markdown, ChunkConfig
from grounding.utils import slugify, atomic_write, ensure_dir
from grounding.manifest import ManifestManager, ManifestEntry

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def read_markdown(file_path: Path) -> str:
    """Read markdown file directly."""
    return file_path.read_text(encoding='utf-8')


def read_docx(file_path: Path) -> str:
    """Read .docx file and convert to markdown-ish text."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    doc = Document(file_path)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Basic style detection for headings
        if para.style.name.startswith('Heading'):
            level = para.style.name.replace('Heading ', '')
            try:
                level_num = int(level)
                text = '#' * level_num + ' ' + text
            except ValueError:
                pass

        paragraphs.append(text)

    return '\n\n'.join(paragraphs)


def read_doc(file_path: Path) -> str:
    """Read .doc file using antiword or catdoc if available."""
    import subprocess
    import shutil

    # Try antiword first
    if shutil.which('antiword'):
        try:
            result = subprocess.run(
                ['antiword', str(file_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

    # Try catdoc
    if shutil.which('catdoc'):
        try:
            result = subprocess.run(
                ['catdoc', str(file_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass

    raise RuntimeError(
        f"Cannot read .doc file. Install antiword or catdoc: "
        f"sudo apt install antiword"
    )


def compute_hashes(content: str) -> dict:
    """Compute content hashes."""
    content_bytes = content.encode('utf-8')
    return {
        'sha256': hashlib.sha256(content_bytes).hexdigest(),
        'blake3': hashlib.sha256(content_bytes).hexdigest(),  # Use sha256 as fallback
    }


def ingest_document(
    file_path: Path,
    output_dir: Path,
    collections: list[str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> bool:
    """
    Ingest a markdown or Word document into the corpus.

    Returns True on success, False on failure.
    """
    ext = file_path.suffix.lower()

    # Read content based on file type
    try:
        if ext == '.md':
            content = read_markdown(file_path)
        elif ext == '.docx':
            content = read_docx(file_path)
        elif ext == '.doc':
            content = read_doc(file_path)
        else:
            logger.error(f"Unsupported format: {ext}")
            return False
    except Exception as e:
        logger.error(f"Failed to read {file_path.name}: {e}")
        return False

    if not content.strip():
        logger.warning(f"Empty content in {file_path.name}")
        return False

    # Generate identifiers
    slug = slugify(file_path.stem)
    file_sha1 = hashlib.sha1(file_path.read_bytes()).hexdigest()
    doc_id = file_sha1[:8]
    hashes = compute_hashes(content)

    # Create output directory
    doc_dir = output_dir / slug
    chunks_dir = doc_dir / 'chunks'
    ensure_dir(chunks_dir)

    # Write full document
    doc_path = doc_dir / 'doc.md'
    atomic_write(doc_path, content)
    logger.info(f"Wrote {doc_path}")

    # Chunk the content
    chunk_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = split_markdown(content, config=chunk_config)
    logger.info(f"Created {len(chunks)} chunks for {file_path.name}")

    # Write chunks with metadata
    now_utc = datetime.now(timezone.utc).isoformat()

    for i, chunk_text_content in enumerate(chunks):
        chunk_id = f"{doc_id}-{i:04d}"
        chunk_hash = hashlib.sha256(chunk_text_content.encode()).hexdigest()

        front_matter = f"""---
doc_id: {doc_id}
source: {file_path.name}
chunk_id: {chunk_id}
page_start: null
page_end: null
hash: {chunk_hash}
created_utc: '{now_utc}'
---

"""
        chunk_path = chunks_dir / f"ch_{i:04d}.md"
        atomic_write(chunk_path, front_matter + chunk_text_content + '\n')

    # Write meta.yaml
    meta_content = f"""doc_id: {doc_id}
slug: {slug}
orig_name: {file_path.name}
collections: {collections or []}
strategy: direct
tooling:
  parser: ingest_docs
params:
  chunk_size: {chunk_size}
  chunk_overlap: {chunk_overlap}
hashes:
  file_sha1: {file_sha1}
  blake3: {hashes['blake3']}
  sha256: {hashes['sha256']}
"""
    meta_path = doc_dir / 'meta.yaml'
    atomic_write(meta_path, meta_content)

    # Update manifest
    manifest_path = output_dir / '_index.json'
    manifest = ManifestManager.load(manifest_path)
    entry = ManifestEntry(
        doc_id=doc_id,
        slug=slug,
        orig_name=file_path.name,
        chunk_count=len(chunks),
        strategy='direct',
        doc_path=f"{slug}/doc.md",
        meta_path=f"{slug}/meta.yaml",
        collections=collections or [],
    )
    manifest = ManifestManager.register_document(manifest, entry)
    ManifestManager.write(manifest, manifest_path)

    logger.info(f"Successfully ingested {file_path.name} -> {slug}/ ({len(chunks)} chunks)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Ingest markdown and Word documents')
    parser.add_argument('input', type=Path, help='Input file or directory')
    parser.add_argument('output', type=Path, help='Output corpus directory')
    parser.add_argument('--collections', type=str, help='Comma-separated collection tags')
    parser.add_argument('--chunk-size', type=int, default=1200)
    parser.add_argument('--chunk-overlap', type=int, default=150)

    args = parser.parse_args()

    collections = args.collections.split(',') if args.collections else None

    if args.input.is_file():
        files = [args.input]
    else:
        files = list(args.input.glob('*.md')) + \
                list(args.input.glob('*.docx')) + \
                list(args.input.glob('*.doc'))

    success = 0
    failed = 0

    for f in files:
        if ingest_document(f, args.output, collections, args.chunk_size, args.chunk_overlap):
            success += 1
        else:
            failed += 1

    print(f"\nSummary: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
