"""Agent filter module for corpus filtering.

Implements Story 10.2: Agent Filter Module.
Provides AgentConfig loading and manifest filtering based on agent configurations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from grounding.manifest import ManifestData, ManifestEntry

logger = logging.getLogger("grounding.agent_filter")


class AgentFilterError(RuntimeError):
    """Raised when agent configuration loading or filtering fails."""


@dataclass
class AgentConfig:
    """Agent configuration for corpus filtering.

    Attributes:
        name: Agent identifier (e.g., "scientist")
        description: Human-readable description of the agent's purpose
        slugs: Explicit list of document slugs to include (optional)
        collections: List of collection tags to match (optional)
        exclude_slugs: List of document slugs to exclude (optional)
    """

    name: str
    description: str = ""
    slugs: List[str] | None = None
    collections: List[str] | None = None
    exclude_slugs: List[str] | None = None


def load_agent_config(agent_name: str, agents_dir: Path) -> AgentConfig:
    """Load agent configuration from YAML file.

    Args:
        agent_name: Agent identifier (e.g., "scientist")
        agents_dir: Directory containing agent YAML files

    Returns:
        AgentConfig with filter criteria

    Raises:
        AgentFilterError: If agent file not found or invalid
    """
    agent_path = agents_dir / f"{agent_name}.yaml"
    if not agent_path.exists():
        raise AgentFilterError(f"Agent file not found: {agent_path}")

    try:
        raw = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AgentFilterError(f"Invalid YAML in agent file {agent_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise AgentFilterError(f"Agent file must contain a YAML object: {agent_path}")

    if "name" not in raw:
        raise AgentFilterError(f"Agent file missing required 'name' field: {agent_path}")

    # Extract corpus_filter settings
    corpus_filter = raw.get("corpus_filter", {})
    if not isinstance(corpus_filter, dict):
        corpus_filter = {}

    return AgentConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        slugs=corpus_filter.get("slugs"),
        collections=corpus_filter.get("collections"),
        exclude_slugs=corpus_filter.get("exclude_slugs"),
    )


def filter_manifest(manifest: ManifestData, config: AgentConfig) -> ManifestData:
    """Filter manifest entries based on agent configuration.

    Matching logic:
    1. If slugs provided: include docs with matching slug
    2. If collections provided: include docs with any matching collection
    3. If both: union of slug matches and collection matches
    4. If neither: include all docs (no filter)
    5. Apply exclude_slugs to remove specific docs

    Args:
        manifest: Full corpus manifest
        config: Agent filter configuration

    Returns:
        New ManifestData containing only matching documents
    """
    # Start with all docs if no inclusion filters
    if not config.slugs and not config.collections:
        matched_docs = list(manifest.docs)
    else:
        matched_docs = []
        slug_set = set(config.slugs) if config.slugs else set()
        collection_set = set(config.collections) if config.collections else set()

        for doc in manifest.docs:
            # Check slug match
            if doc.slug in slug_set:
                matched_docs.append(doc)
                continue

            # Check collection match
            if collection_set and doc.collections:
                if any(c in collection_set for c in doc.collections):
                    matched_docs.append(doc)

    # Apply exclusions
    if config.exclude_slugs:
        exclude_set = set(config.exclude_slugs)
        matched_docs = [doc for doc in matched_docs if doc.slug not in exclude_set]

    logger.debug(
        "Filtered manifest for agent=%s: %d/%d docs",
        config.name,
        len(matched_docs),
        len(manifest.docs),
    )

    # Return new ManifestData with filtered docs, preserving timestamps
    return ManifestData(
        created_utc=manifest.created_utc,
        updated_utc=manifest.updated_utc,
        docs=matched_docs,
    )
