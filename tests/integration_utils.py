"""Helpers for integration tests."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Dict

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_pdfs"


def copy_fixtures(names: Iterable[str], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in names:
        src = FIXTURE_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Fixture {name} not found in {FIXTURE_DIR}")
        dest = destination / name
        shutil.copyfile(src, dest)
        copied.append(dest)
    return copied


def hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_directory(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for file in sorted(root.rglob("*")):
        if file.is_file():
            hashes[file.relative_to(root).as_posix()] = hash_file(file)
    return hashes
