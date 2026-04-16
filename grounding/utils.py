"""Utility functions for grounding."""
import re
import tempfile
from pathlib import Path
from typing import Union


def slugify(filename: str) -> str:
    """
    Convert filename to lowercase kebab-case slug.

    Removes file extension, replaces spaces/underscores with hyphens,
    removes special characters, and collapses multiple hyphens.

    Args:
        filename: Original filename (e.g., "Report_2024.pdf")

    Returns:
        Kebab-case slug (e.g., "report-2024")

    Examples:
        >>> slugify("Report_2024.pdf")
        'report-2024'
        >>> slugify("Q3 Financial Report.pdf")
        'q3-financial-report'
        >>> slugify("Annual_Review_2024_FINAL.pdf")
        'annual-review-2024-final'
    """
    # Remove file extension
    name = Path(filename).stem

    # Convert to lowercase
    slug = name.lower()

    # Replace spaces and underscores with hyphens
    slug = slug.replace(" ", "-").replace("_", "-")

    # Remove all non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)

    # Collapse multiple hyphens into one
    slug = re.sub(r"-+", "-", slug)

    # Strip leading/trailing hyphens
    slug = slug.strip("-")

    return slug


def atomic_write(path: Union[str, Path], content: str, encoding: str = "utf-8") -> Path:
    """
    Write content to file atomically using temp-then-rename.

    This prevents file corruption if the process is interrupted during write.
    The temp file is written first, then atomically renamed to the target path.

    Args:
        path: Target file path
        content: Content to write
        encoding: Text encoding (default: utf-8)

    Returns:
        Path to written file

    Raises:
        IOError: If write fails

    Examples:
        >>> atomic_write("output/doc.md", "# Title\\n\\nContent")
        PosixPath('output/doc.md')
    """
    path = Path(path)

    # Create parent directories if needed
    ensure_dir(path.parent)

    # Write to temporary file first
    # Use same directory as target to ensure atomic rename works
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    try:
        # Write content to temp file
        with open(temp_fd, "w", encoding=encoding) as f:
            f.write(content)

        # Atomically rename temp file to target path
        # This is atomic on POSIX systems and Windows
        Path(temp_path).replace(path)

        return path

    except Exception:
        # Clean up temp file on failure
        Path(temp_path).unlink(missing_ok=True)
        raise


def validate_collection_name(name: str) -> bool:
    """Validate collection name follows kebab-case convention.

    Args:
        name: Collection name to validate

    Returns:
        True if valid kebab-case, False otherwise

    Examples:
        >>> validate_collection_name("science")
        True
        >>> validate_collection_name("music-theory")
        True
        >>> validate_collection_name("Science")
        False
    """
    if not name or not isinstance(name, str):
        return False
    pattern = r'^[a-z0-9]+(-[a-z0-9]+)*$'
    return bool(re.match(pattern, name))


def ensure_dir(path: Union[str, Path], mode: int = 0o755) -> Path:
    """
    Create directory if it doesn't exist.

    Creates parent directories recursively. Does nothing if directory exists.

    Args:
        path: Directory path
        mode: Directory permissions (default: 0o755)

    Returns:
        Path to directory

    Examples:
        >>> ensure_dir("output/chunks")
        PosixPath('output/chunks')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    return path
