"""Tests for the corpus migration script."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

# Import the migration module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from migrate_corpus import (
    DocumentInfo,
    MigrationPlan,
    discover_agents,
    build_document_inventory,
    detect_collisions,
    resolve_collisions,
    build_migration_plan,
    copy_document,
    update_meta_yaml,
    merge_manifests,
    generate_agent_definitions,
)


@pytest.fixture
def mock_agents_dir(tmp_path):
    """Create a mock agents directory structure."""
    agents_dir = tmp_path / "agents"

    # AgentA with one document
    agent_a = agents_dir / "AgentA" / "corpus"
    agent_a.mkdir(parents=True)
    doc_a = agent_a / "doc-a"
    doc_a.mkdir()
    (doc_a / "chunks").mkdir()
    (doc_a / "doc.md").write_text("# Doc A\nContent from AgentA")
    (doc_a / "meta.yaml").write_text(
        yaml.dump({"doc_id": "a1b2c3d4", "slug": "doc-a", "orig_name": "DocA.pdf"})
    )
    (agent_a / "_index.json").write_text(
        json.dumps({
            "created_utc": "2025-01-01T00:00:00Z",
            "docs": [{"doc_id": "a1b2c3d4", "slug": "doc-a", "doc_path": "doc-a/doc.md"}]
        })
    )

    # AgentB with two documents (one collision)
    agent_b = agents_dir / "AgentB" / "corpus"
    agent_b.mkdir(parents=True)

    doc_b = agent_b / "doc-b"
    doc_b.mkdir()
    (doc_b / "doc.md").write_text("# Doc B\nContent from AgentB")
    (doc_b / "meta.yaml").write_text(
        yaml.dump({"doc_id": "e5f6g7h8", "slug": "doc-b"})
    )

    # Collision: same slug as AgentA
    doc_a_collision = agent_b / "doc-a"
    doc_a_collision.mkdir()
    (doc_a_collision / "doc.md").write_text("# Doc A Variant\nDifferent content")
    (doc_a_collision / "meta.yaml").write_text(
        yaml.dump({"doc_id": "i9j0k1l2", "slug": "doc-a"})
    )

    (agent_b / "_index.json").write_text(
        json.dumps({
            "created_utc": "2025-01-01T00:00:00Z",
            "docs": [
                {"doc_id": "e5f6g7h8", "slug": "doc-b", "doc_path": "doc-b/doc.md"},
                {"doc_id": "i9j0k1l2", "slug": "doc-a", "doc_path": "doc-a/doc.md"}
            ]
        })
    )

    # Empty agent (should be skipped)
    (agents_dir / "EmptyAgent" / "corpus").mkdir(parents=True)

    return agents_dir


@pytest.fixture
def target_dir(tmp_path):
    """Create a target directory for migration."""
    target = tmp_path / "target"
    target.mkdir()
    return target


class TestDiscoverAgents:
    """Tests for agent discovery."""

    def test_discovers_agents_with_documents(self, mock_agents_dir):
        agents, skipped = discover_agents(mock_agents_dir)
        assert "AgentA" in agents
        assert "AgentB" in agents
        assert len(agents) == 2

    def test_skips_empty_agents(self, mock_agents_dir):
        agents, skipped = discover_agents(mock_agents_dir)
        assert "EmptyAgent" in skipped

    def test_returns_corpus_paths(self, mock_agents_dir):
        agents, _ = discover_agents(mock_agents_dir)
        assert agents["AgentA"].name == "corpus"
        assert agents["AgentA"].parent.name == "AgentA"


class TestBuildDocumentInventory:
    """Tests for document inventory building."""

    def test_builds_inventory(self, mock_agents_dir):
        agents, _ = discover_agents(mock_agents_dir)
        docs = build_document_inventory(agents)
        assert len(docs) == 3  # 1 from AgentA + 2 from AgentB

    def test_extracts_doc_id(self, mock_agents_dir):
        agents, _ = discover_agents(mock_agents_dir)
        docs = build_document_inventory(agents)
        doc_ids = {d.doc_id for d in docs}
        assert "a1b2c3d4" in doc_ids
        assert "e5f6g7h8" in doc_ids


class TestCollisionDetection:
    """Tests for slug collision detection."""

    def test_detects_collisions(self, mock_agents_dir):
        agents, _ = discover_agents(mock_agents_dir)
        docs = build_document_inventory(agents)
        collisions = detect_collisions(docs)

        assert "doc-a" in collisions
        assert len(collisions["doc-a"]) == 2

    def test_no_collision_for_unique_slugs(self, mock_agents_dir):
        agents, _ = discover_agents(mock_agents_dir)
        docs = build_document_inventory(agents)
        collisions = detect_collisions(docs)

        assert "doc-b" not in collisions


class TestCollisionResolution:
    """Tests for slug collision resolution."""

    def test_resolves_by_appending_doc_id(self):
        docs = [
            DocumentInfo("AgentA", "doc-a", Path("/a"), "aaa11111", "doc-a"),
            DocumentInfo("AgentB", "doc-a", Path("/b"), "bbb22222", "doc-a"),
        ]
        collisions = {"doc-a": docs}
        resolve_collisions(collisions)

        assert docs[0].target_slug == "doc-a"  # First keeps original
        assert docs[1].target_slug == "doc-a-bbb22222"  # Second gets suffix

    def test_uses_agent_name_fallback_without_doc_id(self):
        docs = [
            DocumentInfo("AgentA", "doc-x", Path("/a"), None, "doc-x"),
            DocumentInfo("AgentB", "doc-x", Path("/b"), None, "doc-x"),
        ]
        collisions = {"doc-x": docs}
        resolve_collisions(collisions)

        assert docs[0].target_slug == "doc-x"
        assert docs[1].target_slug == "doc-x-agentb"


class TestDocumentCopying:
    """Tests for document copying."""

    def test_copies_document(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        doc = DocumentInfo(
            agent_name="AgentA",
            slug="doc-a",
            source_path=mock_agents_dir / "AgentA" / "corpus" / "doc-a",
            doc_id="a1b2c3d4",
            target_slug="doc-a"
        )

        result = copy_document(doc, target_corpus, dry_run=False)

        assert result is True
        assert (target_corpus / "doc-a" / "doc.md").exists()

    def test_dry_run_does_not_copy(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        doc = DocumentInfo(
            agent_name="AgentA",
            slug="doc-a",
            source_path=mock_agents_dir / "AgentA" / "corpus" / "doc-a",
            doc_id="a1b2c3d4",
            target_slug="doc-a"
        )

        result = copy_document(doc, target_corpus, dry_run=True)

        assert result is True
        assert not (target_corpus / "doc-a").exists()

    def test_skips_existing(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()
        (target_corpus / "doc-a").mkdir()

        doc = DocumentInfo(
            agent_name="AgentA",
            slug="doc-a",
            source_path=mock_agents_dir / "AgentA" / "corpus" / "doc-a",
            doc_id="a1b2c3d4",
            target_slug="doc-a"
        )

        result = copy_document(doc, target_corpus, dry_run=False)
        assert result is True


class TestMetaYamlUpdate:
    """Tests for meta.yaml updates."""

    def test_adds_source_agent(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        # Copy first
        doc = DocumentInfo(
            agent_name="AgentA",
            slug="doc-a",
            source_path=mock_agents_dir / "AgentA" / "corpus" / "doc-a",
            doc_id="a1b2c3d4",
            target_slug="doc-a"
        )
        copy_document(doc, target_corpus, dry_run=False)

        # Update meta
        result = update_meta_yaml(doc, target_corpus, dry_run=False)

        assert result is True
        meta_path = target_corpus / "doc-a" / "meta.yaml"
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["source_agent"] == "agenta"


class TestAgentDefinitionGeneration:
    """Tests for agent definition file generation."""

    def test_generates_agent_yaml(self, mock_agents_dir, target_dir):
        plan = build_migration_plan(mock_agents_dir)

        count = generate_agent_definitions(plan, target_dir, dry_run=False)

        assert count == 2
        assert (target_dir / "agents" / "agenta.yaml").exists()
        assert (target_dir / "agents" / "agentb.yaml").exists()

    def test_includes_slugs_list(self, mock_agents_dir, target_dir):
        plan = build_migration_plan(mock_agents_dir)
        generate_agent_definitions(plan, target_dir, dry_run=False)

        with open(target_dir / "agents" / "agentb.yaml") as f:
            agent_def = yaml.safe_load(f)

        assert "corpus_filter" in agent_def
        assert "slugs" in agent_def["corpus_filter"]
        # AgentB has doc-b and doc-a (renamed to doc-a-i9j0k1l2)
        assert len(agent_def["corpus_filter"]["slugs"]) == 2


class TestManifestMerge:
    """Tests for manifest merging."""

    def test_merges_manifests(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        plan = build_migration_plan(mock_agents_dir)

        # Copy documents first
        for doc in plan.documents:
            copy_document(doc, target_corpus, dry_run=False)

        count = merge_manifests(plan, target_corpus, dry_run=False)

        assert count == 3

        with open(target_corpus / "_index.json") as f:
            manifest = json.load(f)

        assert len(manifest["docs"]) == 3

    def test_updates_paths_for_renamed_docs(self, mock_agents_dir, target_dir):
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        plan = build_migration_plan(mock_agents_dir)

        for doc in plan.documents:
            copy_document(doc, target_corpus, dry_run=False)

        merge_manifests(plan, target_corpus, dry_run=False)

        with open(target_corpus / "_index.json") as f:
            manifest = json.load(f)

        # Find the renamed doc
        renamed_docs = [d for d in manifest["docs"] if "-i9j0k1l2" in d.get("slug", "")]
        assert len(renamed_docs) == 1
        assert "doc-a-i9j0k1l2" in renamed_docs[0]["doc_path"]


class TestEndToEndMigration:
    """End-to-end migration tests."""

    def test_full_migration(self, mock_agents_dir, target_dir):
        plan = build_migration_plan(mock_agents_dir)

        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()
        (target_dir / "originals").mkdir()

        # Copy all documents
        for doc in plan.documents:
            copy_document(doc, target_corpus, dry_run=False)
            update_meta_yaml(doc, target_corpus, dry_run=False)

        # Merge manifests
        merge_manifests(plan, target_corpus, dry_run=False)

        # Generate agent definitions
        generate_agent_definitions(plan, target_dir, dry_run=False)

        # Verify structure
        assert (target_corpus / "_index.json").exists()
        assert (target_corpus / "doc-a" / "doc.md").exists()
        assert (target_corpus / "doc-b" / "doc.md").exists()
        assert (target_corpus / "doc-a-i9j0k1l2" / "doc.md").exists()
        assert (target_dir / "agents" / "agenta.yaml").exists()
        assert (target_dir / "agents" / "agentb.yaml").exists()

    def test_original_preserved(self, mock_agents_dir, target_dir):
        """Verify original directories are not modified."""
        original_content = (mock_agents_dir / "AgentA" / "corpus" / "doc-a" / "doc.md").read_text()

        plan = build_migration_plan(mock_agents_dir)
        target_corpus = target_dir / "corpus"
        target_corpus.mkdir()

        for doc in plan.documents:
            copy_document(doc, target_corpus, dry_run=False)

        # Original should be unchanged
        assert (mock_agents_dir / "AgentA" / "corpus" / "doc-a" / "doc.md").read_text() == original_content
