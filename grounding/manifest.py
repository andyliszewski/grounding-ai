"""Corpus manifest manager for grounding.

Implements Epic 4 Story 4.3 by loading/updating `_index.json`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from grounding.utils import atomic_write

logger = logging.getLogger("grounding.manifest")


@dataclass
class ManifestEntry:
    doc_id: str
    slug: str
    orig_name: str
    pages: int | None = None
    strategy: str | None = None
    chunk_count: int | None = None
    meta_path: str | None = None
    doc_path: str | None = None
    content_type: str | None = None  # "text", "music", "scientific", or "hybrid"
    music_format: str | None = None  # "musicxml", "abc", "midi", "all"
    music_metadata: Dict | None = None  # {"key": "C major", "time_signature": "4/4", "phrase_count": 12, "measure_count": 48}
    music_files: List[str] | None = None  # ["slug/music.musicxml", "slug/music.abc", "slug/music.mid"]
    formula_metadata: Dict | None = None  # {"formula_count": 18, "inline_count": 12, "display_count": 6, "complexity": "moderate"}
    formula_files: List[str] | None = None  # ["slug/formulas/formula_0001_0001.tex", "slug/formulas/formula_0001_0001.mathml"]
    collections: List[str] | None = None  # ["science", "biology", "reference"]
    source_agent: str | None = None  # "scientist"

    def to_dict(self) -> dict:
        data = {
            "doc_id": self.doc_id,
            "slug": self.slug,
            "orig_name": self.orig_name,
            "pages": self.pages,
            "strategy": self.strategy,
            "chunk_count": self.chunk_count,
            "meta_path": self.meta_path,
            "doc_path": self.doc_path,
        }
        # Only include music fields if present
        if self.content_type:
            data["content_type"] = self.content_type
        if self.music_format:
            data["music_format"] = self.music_format
        if self.music_metadata:
            data["music_metadata"] = self.music_metadata
        if self.music_files:
            data["music_files"] = self.music_files
        # Only include formula fields if present
        if self.formula_metadata:
            data["formula_metadata"] = self.formula_metadata
        if self.formula_files:
            data["formula_files"] = self.formula_files
        # Only include collection fields if present
        if self.collections:
            data["collections"] = self.collections
        if self.source_agent:
            data["source_agent"] = self.source_agent
        return data


@dataclass
class ManifestData:
    created_utc: str | None = None
    updated_utc: str | None = None
    docs: List[ManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "docs": [entry.to_dict() for entry in self.docs],
        }


class ManifestError(RuntimeError):
    """Raised when manifest loading or validation fails."""


class ManifestManager:
    """Manage loading, updating, and writing `_index.json`."""

    @staticmethod
    def load(path: Path) -> ManifestData:
        if not path.exists():
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            logger.info("Manifest not found at %s; initializing new manifest", path)
            return ManifestData(created_utc=now, updated_utc=now, docs=[])

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Failed to parse manifest {path}: {exc}") from exc

        ManifestManager._validate_raw_manifest(raw)
        docs = [ManifestEntry(**entry) for entry in raw["docs"]]
        return ManifestData(
            created_utc=raw.get("created_utc"),
            updated_utc=raw.get("updated_utc"),
            docs=docs,
        )

    @staticmethod
    def register_document(manifest: ManifestData, entry: ManifestEntry) -> ManifestData:
        docs_by_id: Dict[str, ManifestEntry] = {doc.doc_id: doc for doc in manifest.docs}
        docs_by_id[entry.doc_id] = entry
        sorted_docs = sorted(docs_by_id.values(), key=lambda doc: (doc.slug, doc.doc_id))
        manifest.docs = sorted_docs
        manifest.updated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return manifest

    @staticmethod
    def write(manifest: ManifestData, path: Path) -> Path:
        json_text = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        written = atomic_write(path, json_text)
        logger.info("Manifest updated at %s entries=%d", written, len(manifest.docs))
        return written

    @staticmethod
    def _validate_raw_manifest(raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ManifestError("Manifest root must be an object")
        required_root_fields = {"created_utc", "updated_utc", "docs"}
        missing = required_root_fields - set(raw)
        if missing:
            raise ManifestError(f"Manifest missing required fields: {sorted(missing)}")
        docs = raw.get("docs")
        if not isinstance(docs, list):
            raise ManifestError("Manifest 'docs' must be a list")
        for entry in docs:
            if not isinstance(entry, dict):
                raise ManifestError("Manifest entries must be objects")
            if "doc_id" not in entry or "slug" not in entry or "orig_name" not in entry:
                raise ManifestError("Manifest entry missing required doc fields")
