"""Tests for the agent command generator script."""

import tempfile
from pathlib import Path

import pytest
import yaml

# Import functions from the generator script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_agent_commands import (
    load_agent_config,
    generate_command_content,
    generate_display_name,
    format_expertise_list,
    format_collections,
    generate_commands,
    DEFAULT_PERSONA,
)


class TestParsePersonaFromYaml:
    """Test persona block extraction from YAML."""

    def test_parse_full_persona(self, tmp_path):
        """Test parsing agent YAML with complete persona block."""
        agent_yaml = """
name: test-agent
description: Test agent description

persona:
  icon: "🧪"
  style: |
    You are a test agent.
    You speak in tests.
  expertise:
    - Testing
    - Quality assurance
  greeting: Hello, I am a test agent.

corpus_filter:
  collections:
    - test-collection
"""
        agent_file = tmp_path / "test-agent.yaml"
        agent_file.write_text(agent_yaml)

        config = load_agent_config(agent_file)

        assert config["name"] == "test-agent"
        assert config["description"] == "Test agent description"
        assert config["persona"]["icon"] == "🧪"
        assert "You are a test agent." in config["persona"]["style"]
        assert len(config["persona"]["expertise"]) == 2
        assert config["persona"]["greeting"] == "Hello, I am a test agent."

    def test_parse_minimal_persona(self, tmp_path):
        """Test parsing agent YAML with minimal persona block."""
        agent_yaml = """
name: minimal-agent
description: Minimal agent

persona:
  icon: "📦"

corpus_filter:
  collections:
    - general
"""
        agent_file = tmp_path / "minimal-agent.yaml"
        agent_file.write_text(agent_yaml)

        config = load_agent_config(agent_file)

        assert config["name"] == "minimal-agent"
        assert config["persona"]["icon"] == "📦"
        assert "style" not in config["persona"]
        assert "expertise" not in config["persona"]


class TestGenerateCommandFile:
    """Test command file generation."""

    def test_generate_complete_command(self):
        """Test generating command file from complete config."""
        config = {
            "name": "test-agent",
            "description": "Test agent description",
            "persona": {
                "icon": "🧪",
                "style": "You speak like a tester.",
                "expertise": ["Testing", "QA"],
                "greeting": "Hello, tester here.",
            },
            "corpus_filter": {
                "collections": ["testing", "quality"],
            },
        }

        content = generate_command_content(config)

        # Check front matter
        assert "---" in content
        assert 'description: "Activate test-agent agent' in content

        # Check persona section
        assert "🧪 **Test Agent**" in content
        assert "You speak like a tester." in content

        # Check expertise
        assert "- Testing" in content
        assert "- QA" in content

        # Check collections
        assert "testing, quality" in content

        # Check embeddings path
        assert "embeddings/test-agent/_embeddings.faiss" in content

        # Check greeting
        assert "Hello, tester here." in content

    def test_generate_command_preserves_multiline_style(self):
        """Test that multiline style content is preserved."""
        config = {
            "name": "multi-line",
            "description": "Multi-line agent",
            "persona": {
                "icon": "📝",
                "style": "Line one.\nLine two.\nLine three.",
                "expertise": ["Writing"],
                "greeting": "Hello.",
            },
            "corpus_filter": {"collections": ["writing"]},
        }

        content = generate_command_content(config)

        assert "Line one." in content
        assert "Line two." in content
        assert "Line three." in content


class TestMissingPersonaUsesDefaults:
    """Test graceful fallback when persona block is missing."""

    def test_missing_persona_uses_defaults(self):
        """Test that missing persona uses default values."""
        config = {
            "name": "no-persona-agent",
            "description": "Agent without persona",
            "corpus_filter": {"collections": ["general"]},
        }

        content = generate_command_content(config)

        # Should use default icon
        assert DEFAULT_PERSONA["icon"] in content

        # Should use default style
        assert DEFAULT_PERSONA["style"] in content

        # Should use default greeting
        assert DEFAULT_PERSONA["greeting"] in content

    def test_partial_persona_fills_missing(self):
        """Test that partial persona fills missing fields with defaults."""
        config = {
            "name": "partial-persona",
            "description": "Partial persona agent",
            "persona": {
                "icon": "🔧",
                # Missing style, expertise, greeting
            },
            "corpus_filter": {"collections": ["tools"]},
        }

        content = generate_command_content(config)

        # Should use provided icon
        assert "🔧" in content

        # Should use default for missing fields
        assert DEFAULT_PERSONA["style"] in content


class TestGeneratorIdempotent:
    """Test that generator produces identical output for same input."""

    def test_same_input_same_output(self, tmp_path):
        """Test idempotency - same YAML produces same command file."""
        agent_yaml = """
name: idempotent-agent
description: Test idempotency

persona:
  icon: "🔄"
  style: Consistent style.
  expertise:
    - Consistency
  greeting: Always the same.

corpus_filter:
  collections:
    - consistency
"""
        agent_file = tmp_path / "idempotent-agent.yaml"
        agent_file.write_text(agent_yaml)

        config = load_agent_config(agent_file)

        # Generate content multiple times
        content1 = generate_command_content(config)
        content2 = generate_command_content(config)
        content3 = generate_command_content(config)

        assert content1 == content2
        assert content2 == content3

    def test_regenerate_same_files(self, tmp_path):
        """Test that regenerating command files produces identical results."""
        agents_dir = tmp_path / "agents"
        commands_dir = tmp_path / "commands"
        agents_dir.mkdir()

        agent_yaml = """
name: regen-test
description: Regeneration test

persona:
  icon: "♻️"
  style: Test.
  expertise:
    - Testing
  greeting: Hi.

corpus_filter:
  collections:
    - test
"""
        (agents_dir / "regen-test.yaml").write_text(agent_yaml)

        # Generate first time
        generate_commands(agents_dir, commands_dir)
        first_content = (commands_dir / "regen-test.md").read_text()

        # Regenerate
        generate_commands(agents_dir, commands_dir)
        second_content = (commands_dir / "regen-test.md").read_text()

        assert first_content == second_content


class TestAllAgentsHaveCommands:
    """Test that all agents get command files generated."""

    def test_all_yaml_files_generate_commands(self, tmp_path):
        """Test that each YAML file produces a corresponding command file."""
        agents_dir = tmp_path / "agents"
        commands_dir = tmp_path / "commands"
        agents_dir.mkdir()

        # Create multiple agent files
        agent_names = ["agent-one", "agent-two", "agent-three"]
        for name in agent_names:
            yaml_content = f"""
name: {name}
description: {name} description

corpus_filter:
  collections:
    - general
"""
            (agents_dir / f"{name}.yaml").write_text(yaml_content)

        # Generate commands
        generated, skipped = generate_commands(agents_dir, commands_dir)

        assert generated == 3
        assert skipped == 0

        # Verify each command file exists
        for name in agent_names:
            command_file = commands_dir / f"{name}.md"
            assert command_file.exists(), f"Missing command file for {name}"


class TestCommandFrontMatterValid:
    """Test that generated YAML front matter is valid."""

    def test_front_matter_parses_as_yaml(self):
        """Test that front matter can be parsed as valid YAML."""
        config = {
            "name": "front-matter-test",
            "description": "Testing front matter",
            "persona": {
                "icon": "📋",
                "style": "Formal.",
                "expertise": ["Documentation"],
                "greeting": "Hello.",
            },
            "corpus_filter": {"collections": ["docs"]},
        }

        content = generate_command_content(config)

        # Extract front matter between --- markers
        parts = content.split("---")
        assert len(parts) >= 3, "Should have front matter delimiters"

        front_matter = parts[1].strip()

        # Parse as YAML
        parsed = yaml.safe_load(front_matter)

        assert "description" in parsed
        assert "front-matter-test" in parsed["description"]

    def test_front_matter_special_characters(self):
        """Test front matter handles special characters in description."""
        config = {
            "name": "special-chars",
            "description": "Agent with: colons, 'quotes', and \"double quotes\"",
            "persona": {
                "icon": "⚠️",
                "style": "Careful.",
                "expertise": ["Special handling"],
                "greeting": "Hi.",
            },
            "corpus_filter": {"collections": ["special"]},
        }

        content = generate_command_content(config)

        # Should not raise when parsing
        parts = content.split("---")
        front_matter = parts[1].strip()
        parsed = yaml.safe_load(front_matter)

        assert "description" in parsed


class TestDisplayNameGeneration:
    """Test display name generation from agent names."""

    def test_simple_name(self):
        """Test simple single-word name."""
        assert generate_display_name("scientist") == "Scientist"

    def test_hyphenated_name(self):
        """Test hyphenated multi-word name."""
        assert generate_display_name("data-scientist") == "Data Scientist"
        assert generate_display_name("corp-dev-friendly") == "Corp Dev Friendly"

    def test_acronym_handling(self):
        """Test that common acronyms stay uppercase."""
        assert generate_display_name("ceo") == "CEO"
        assert generate_display_name("ip") == "IP"
        assert generate_display_name("qa") == "QA"

    def test_mixed_acronyms_and_words(self):
        """Test names with both acronyms and regular words."""
        # If we had an agent named "vp-engineering"
        assert generate_display_name("vp-sales") == "VP Sales"


class TestFormatHelpers:
    """Test formatting helper functions."""

    def test_format_expertise_list(self):
        """Test expertise list formatting."""
        expertise = ["Item one", "Item two", "Item three"]
        result = format_expertise_list(expertise)

        assert "- Item one" in result
        assert "- Item two" in result
        assert "- Item three" in result
        assert result.count("-") == 3

    def test_format_collections(self):
        """Test collections formatting."""
        collections = ["finance", "strategy", "leadership"]
        result = format_collections(collections)

        assert result == "finance, strategy, leadership"

    def test_format_empty_collections(self):
        """Test formatting empty collections list."""
        result = format_collections([])
        assert result == ""


class TestGenerateCommandsFunction:
    """Test the main generate_commands function."""

    def test_dry_run_no_files_created(self, tmp_path):
        """Test that dry run doesn't create files."""
        agents_dir = tmp_path / "agents"
        commands_dir = tmp_path / "commands"
        agents_dir.mkdir()

        agent_yaml = """
name: dry-run-test
description: Dry run test

corpus_filter:
  collections:
    - test
"""
        (agents_dir / "dry-run-test.yaml").write_text(agent_yaml)

        generate_commands(agents_dir, commands_dir, dry_run=True)

        # Commands directory should not exist
        assert not commands_dir.exists()

    def test_skip_invalid_yaml(self, tmp_path):
        """Test that invalid YAML files are skipped."""
        agents_dir = tmp_path / "agents"
        commands_dir = tmp_path / "commands"
        agents_dir.mkdir()

        # Valid agent
        valid_yaml = """
name: valid-agent
description: Valid

corpus_filter:
  collections:
    - test
"""
        (agents_dir / "valid-agent.yaml").write_text(valid_yaml)

        # Invalid YAML
        invalid_yaml = """
name: invalid
description: [unclosed bracket
"""
        (agents_dir / "invalid-agent.yaml").write_text(invalid_yaml)

        generated, skipped = generate_commands(agents_dir, commands_dir)

        assert generated == 1
        assert skipped == 1

    def test_skip_agent_without_name(self, tmp_path):
        """Test that agents without name field are skipped."""
        agents_dir = tmp_path / "agents"
        commands_dir = tmp_path / "commands"
        agents_dir.mkdir()

        # Agent without name
        no_name_yaml = """
description: No name field

corpus_filter:
  collections:
    - test
"""
        (agents_dir / "no-name.yaml").write_text(no_name_yaml)

        generated, skipped = generate_commands(agents_dir, commands_dir)

        assert generated == 0
        assert skipped == 1


class TestRealAgents:
    """Integration tests using real agent files."""

    def test_parse_real_ceo_agent(self):
        """Test parsing the real CEO agent YAML."""
        agent_file = Path("agents/ceo.yaml")
        if not agent_file.exists():
            pytest.skip("Real agent files not available")

        config = load_agent_config(agent_file)

        assert config["name"] == "ceo"
        assert "persona" in config
        assert config["persona"]["icon"] == "🎯"
        assert len(config["persona"]["expertise"]) > 0
        assert "corpus_filter" in config
        assert len(config["corpus_filter"]["collections"]) > 0

    def test_all_real_agents_have_persona(self):
        """Test that all real agent files have persona blocks."""
        agents_dir = Path("agents")
        if not agents_dir.exists():
            pytest.skip("Real agent files not available")

        for agent_file in agents_dir.glob("*.yaml"):
            config = load_agent_config(agent_file)
            assert "persona" in config, f"{agent_file.name} missing persona block"
            assert "icon" in config["persona"], f"{agent_file.name} missing icon"
            assert "style" in config["persona"], f"{agent_file.name} missing style"
            assert "greeting" in config["persona"], f"{agent_file.name} missing greeting"
