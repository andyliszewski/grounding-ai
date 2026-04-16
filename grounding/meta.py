"""Document metadata serialization for grounding.

Implements Epic 4 Story 4.2 by generating meta.yaml content.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, List, Mapping, MutableMapping

import yaml

from grounding.pipeline import FileContext
from grounding.utils import validate_collection_name

logger = logging.getLogger("grounding.meta")

PACKAGE_KEYS = {
    "parser": "unstructured",
    "chunker": "langchain-text-splitters",
    "hashing": "blake3",
    "scanner": "unstructured",
}


def build_meta_yaml(
    context: FileContext,
    *,
    params: Mapping[str, Any],
    tooling: Mapping[str, str] | None = None,
    generated_utc: datetime | None = None,
    collections: List[str] | None = None,
    source_agent: str | None = None,
) -> str:
    """
    Create meta.yaml string for a processed document.

    Args:
        context: FileContext produced by the pipeline with slug/hash info.
        params: Execution parameters (ocr_mode, chunk_size, etc.).
        tooling: Optional overrides for tooling versions.
        generated_utc: Optional timestamp (defaults to current UTC).
        collections: Optional list of collection labels (e.g., ["science", "biology"]).
        source_agent: Optional source agent identifier (e.g., "scientist").

    Returns:
        YAML string representing document metadata.
    """
    timestamp = generated_utc or datetime.now(timezone.utc)
    context_slug = context.slug
    orig_name = context.source_path.name

    tooling_versions = _collect_tooling_versions(tooling)
    hashes = _collect_hashes(context)

    meta_dict = {
        "doc_id": context.doc_id,
        "slug": context_slug,
        "orig_name": orig_name,
        "created_utc": timestamp.replace(microsecond=0).isoformat(),
    }

    # Add collections if provided (validate each name)
    if collections:
        valid_collections = [c for c in collections if validate_collection_name(c)]
        if valid_collections:
            meta_dict["collections"] = valid_collections
        invalid = [c for c in collections if not validate_collection_name(c)]
        if invalid:
            logger.warning("Skipping invalid collection names: %s", invalid)

    # Add source_agent if provided
    if source_agent:
        meta_dict["source_agent"] = source_agent

    meta_dict["tooling"] = tooling_versions
    meta_dict["params"] = dict(params)
    meta_dict["hashes"] = hashes

    rendered = yaml.safe_dump(
        meta_dict,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=False,
    )
    return rendered


def _collect_tooling_versions(overrides: Mapping[str, str] | None) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for key, package in PACKAGE_KEYS.items():
        if overrides and key in overrides:
            versions[key] = overrides[key]
            continue
        versions[key] = _get_package_version(package)
    return versions


def _collect_hashes(context: FileContext) -> Dict[str, Any]:
    hashes: Dict[str, Any] = {
        "file_sha1": context.sha1,
        "doc_sha1": context.doc_sha1,
    }
    if context.doc_hashes:
        hashes.update(context.doc_hashes)
    else:
        if context.doc_id:
            logger.warning("Missing doc_hashes for slug=%s doc_id=%s", context.slug, context.doc_id)
    return hashes


def _get_package_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        logger.debug("Package %s not installed; reporting version as 'unknown'", dist_name)
        return "unknown"
