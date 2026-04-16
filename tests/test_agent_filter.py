"""Tests for agent_filter module (Story 10.2)."""
from __future__ import annotations

import pytest
from pathlib import Path

from grounding.agent_filter import (
    AgentConfig,
    AgentFilterError,
    filter_manifest,
    load_agent_config,
)
from grounding.manifest import ManifestData, ManifestEntry


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_manifest() -> ManifestData:
    """Sample manifest for filtering tests."""
    return ManifestData(
        created_utc="2025-12-23T00:00:00+00:00",
        updated_utc="2025-12-23T12:00:00+00:00",
        docs=[
            ManifestEntry(
                doc_id="a1",
                slug="book-science-1",
                orig_name="Science Book 1.pdf",
                collections=["science"],
            ),
            ManifestEntry(
                doc_id="a2",
                slug="book-science-2",
                orig_name="Science Book 2.pdf",
                collections=["science", "biology"],
            ),
            ManifestEntry(
                doc_id="b1",
                slug="book-music-1",
                orig_name="Music Book 1.pdf",
                collections=["music"],
            ),
            ManifestEntry(
                doc_id="c1",
                slug="book-no-collection",
                orig_name="No Collection Book.pdf",
                collections=None,
            ),
        ],
    )


@pytest.fixture
def scientist_config() -> AgentConfig:
    """Science agent config with collection filter."""
    return AgentConfig(
        name="scientist",
        description="Science and research specialist",
        collections=["science"],
    )


@pytest.fixture
def tmp_agents_dir(tmp_path: Path) -> Path:
    """Create temp agents directory with sample configs."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Valid scientist agent
    scientist_yaml = agents_dir / "scientist.yaml"
    scientist_yaml.write_text(
        """name: scientist
description: Science and research specialist

corpus_filter:
  collections:
    - science
    - research-methods
  exclude_slugs:
    - solutions-manual
"""
    )

    # Valid musician agent with slugs
    musician_yaml = agents_dir / "musician.yaml"
    musician_yaml.write_text(
        """name: musician
description: Music specialist

corpus_filter:
  slugs:
    - book-music-1
    - book-music-2
"""
    )

    # Agent with both slugs and collections
    combined_yaml = agents_dir / "combined.yaml"
    combined_yaml.write_text(
        """name: combined
description: Combined filter agent

corpus_filter:
  slugs:
    - special-book
  collections:
    - science
"""
    )

    # Minimal valid agent (no corpus_filter)
    minimal_yaml = agents_dir / "minimal.yaml"
    minimal_yaml.write_text(
        """name: minimal
description: Agent with no filters
"""
    )

    # Invalid YAML
    invalid_yaml = agents_dir / "invalid.yaml"
    invalid_yaml.write_text("name: invalid\ndescription: [unclosed bracket")

    # Missing name field
    missing_name_yaml = agents_dir / "missing_name.yaml"
    missing_name_yaml.write_text("description: No name field")

    return agents_dir


# ─────────────────────────────────────────────────────────────────────────────
# AgentConfig Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_config_creation():
    """Test AgentConfig dataclass creation."""
    config = AgentConfig(
        name="test",
        description="Test agent",
        slugs=["book-a", "book-b"],
        collections=["science"],
        exclude_slugs=["excluded"],
    )
    assert config.name == "test"
    assert config.description == "Test agent"
    assert config.slugs == ["book-a", "book-b"]
    assert config.collections == ["science"]
    assert config.exclude_slugs == ["excluded"]


def test_agent_config_defaults():
    """Test AgentConfig with minimal fields."""
    config = AgentConfig(name="minimal")
    assert config.name == "minimal"
    assert config.description == ""
    assert config.slugs is None
    assert config.collections is None
    assert config.exclude_slugs is None


# ─────────────────────────────────────────────────────────────────────────────
# load_agent_config Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_load_agent_config_valid(tmp_agents_dir: Path):
    """Test loading a valid agent config."""
    config = load_agent_config("scientist", tmp_agents_dir)
    assert config.name == "scientist"
    assert config.description == "Science and research specialist"
    assert config.collections == ["science", "research-methods"]
    assert config.exclude_slugs == ["solutions-manual"]
    assert config.slugs is None


def test_load_agent_config_with_slugs(tmp_agents_dir: Path):
    """Test loading agent config with slug filter."""
    config = load_agent_config("musician", tmp_agents_dir)
    assert config.name == "musician"
    assert config.slugs == ["book-music-1", "book-music-2"]
    assert config.collections is None


def test_load_agent_config_minimal(tmp_agents_dir: Path):
    """Test loading minimal agent config (no corpus_filter)."""
    config = load_agent_config("minimal", tmp_agents_dir)
    assert config.name == "minimal"
    assert config.slugs is None
    assert config.collections is None
    assert config.exclude_slugs is None


def test_load_agent_config_missing_file(tmp_agents_dir: Path):
    """Test error when agent file doesn't exist."""
    with pytest.raises(AgentFilterError) as exc_info:
        load_agent_config("nonexistent", tmp_agents_dir)
    assert "Agent file not found" in str(exc_info.value)


def test_load_agent_config_invalid_yaml(tmp_agents_dir: Path):
    """Test error when YAML is malformed."""
    with pytest.raises(AgentFilterError) as exc_info:
        load_agent_config("invalid", tmp_agents_dir)
    assert "Invalid YAML" in str(exc_info.value)


def test_load_agent_config_missing_name(tmp_agents_dir: Path):
    """Test error when name field is missing."""
    with pytest.raises(AgentFilterError) as exc_info:
        load_agent_config("missing_name", tmp_agents_dir)
    assert "missing required 'name' field" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# filter_manifest Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_filter_by_slugs(sample_manifest: ManifestData):
    """Test filtering by explicit slug list."""
    config = AgentConfig(
        name="slug_filter",
        slugs=["book-science-1", "book-music-1"],
    )
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 2
    slugs = {doc.slug for doc in result.docs}
    assert slugs == {"book-science-1", "book-music-1"}


def test_filter_by_collections(sample_manifest: ManifestData, scientist_config: AgentConfig):
    """Test filtering by collection tags."""
    result = filter_manifest(sample_manifest, scientist_config)

    assert len(result.docs) == 2
    slugs = {doc.slug for doc in result.docs}
    assert slugs == {"book-science-1", "book-science-2"}


def test_filter_by_both_slugs_and_collections(sample_manifest: ManifestData):
    """Test filtering with both slugs and collections (union)."""
    config = AgentConfig(
        name="combined",
        slugs=["book-no-collection"],
        collections=["music"],
    )
    result = filter_manifest(sample_manifest, config)

    # Should include: book-no-collection (slug match) + book-music-1 (collection match)
    assert len(result.docs) == 2
    slugs = {doc.slug for doc in result.docs}
    assert slugs == {"book-no-collection", "book-music-1"}


def test_filter_with_exclude(sample_manifest: ManifestData):
    """Test exclude_slugs removes specific docs after inclusion."""
    config = AgentConfig(
        name="exclude_test",
        collections=["science"],
        exclude_slugs=["book-science-2"],
    )
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 1
    assert result.docs[0].slug == "book-science-1"


def test_filter_empty_config(sample_manifest: ManifestData):
    """Test empty filter returns all docs."""
    config = AgentConfig(name="empty")
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 4


def test_filter_no_matches(sample_manifest: ManifestData):
    """Test filter with no matching docs returns empty list."""
    config = AgentConfig(
        name="no_match",
        slugs=["nonexistent-slug"],
    )
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 0


def test_filter_preserves_timestamps(sample_manifest: ManifestData, scientist_config: AgentConfig):
    """Test that filtered manifest preserves original timestamps."""
    result = filter_manifest(sample_manifest, scientist_config)

    assert result.created_utc == "2025-12-23T00:00:00+00:00"
    assert result.updated_utc == "2025-12-23T12:00:00+00:00"


def test_filter_doc_without_collections_slug_match(sample_manifest: ManifestData):
    """Test that doc without collections can be matched by slug."""
    config = AgentConfig(
        name="slug_only",
        slugs=["book-no-collection"],
        collections=["science"],
    )
    result = filter_manifest(sample_manifest, config)

    slugs = {doc.slug for doc in result.docs}
    assert "book-no-collection" in slugs


def test_filter_doc_without_collections_no_collection_match(sample_manifest: ManifestData):
    """Test that doc without collections doesn't match collection filter."""
    config = AgentConfig(
        name="collection_only",
        collections=["science"],
    )
    result = filter_manifest(sample_manifest, config)

    slugs = {doc.slug for doc in result.docs}
    assert "book-no-collection" not in slugs


def test_filter_multiple_collection_any_match(sample_manifest: ManifestData):
    """Test that doc with multiple collections matches if any collection matches."""
    config = AgentConfig(
        name="bio_filter",
        collections=["biology"],
    )
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 1
    assert result.docs[0].slug == "book-science-2"


def test_filter_exclude_all(sample_manifest: ManifestData):
    """Test excluding all included docs results in empty list."""
    config = AgentConfig(
        name="exclude_all",
        collections=["science"],
        exclude_slugs=["book-science-1", "book-science-2"],
    )
    result = filter_manifest(sample_manifest, config)

    assert len(result.docs) == 0
