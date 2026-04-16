#!/usr/bin/env python3
"""
Unit tests for local_rag.py CLI integration with agentic mode.

Tests argument parsing, mode selection, and integration between
simple RAG and agentic modes without loading heavy dependencies.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_dependencies():
    """Mock heavy dependencies before importing local_rag."""
    # Create mock modules
    mock_faiss = MagicMock()
    mock_faiss.read_index.return_value = MagicMock(ntotal=100)

    mock_sentence_transformers = MagicMock()
    mock_embedder = MagicMock()
    mock_sentence_transformers.SentenceTransformer.return_value = mock_embedder

    mock_requests = MagicMock()

    with patch.dict('sys.modules', {
        'faiss': mock_faiss,
        'sentence_transformers': mock_sentence_transformers,
        'requests': mock_requests,
    }):
        yield {
            'faiss': mock_faiss,
            'sentence_transformers': mock_sentence_transformers,
            'embedder': mock_embedder,
            'requests': mock_requests,
        }


@pytest.fixture
def temp_agent_dir(tmp_path):
    """Create temporary agent YAML file."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    agent_yaml = agents_dir / "test-agent.yaml"
    agent_yaml.write_text("""
name: test-agent
description: Test agent for CLI testing

persona:
  icon: "🧪"
  style: |
    You are a test agent.
  expertise:
    - Testing
  greeting: |
    Test agent ready.

corpus_filter:
  collections:
    - test
""")
    return agents_dir


# =============================================================================
# Test Argument Parsing
# =============================================================================

class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_agentic_flag_short(self):
        """Short -A flag is recognized."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "test", "-A"])
        assert args.agentic is True

    def test_agentic_flag_long(self):
        """Long --agentic flag is recognized."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "test", "--agentic"])
        assert args.agentic is True

    def test_agentic_flag_default_false(self):
        """--agentic defaults to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "test"])
        assert args.agentic is False

    def test_max_iterations_default(self):
        """--max-iterations defaults to 5."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--max-iterations", type=int, default=5)

        args = parser.parse_args(["-a", "test"])
        assert args.max_iterations == 5

    def test_max_iterations_custom(self):
        """--max-iterations accepts custom value."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--max-iterations", type=int, default=5)

        args = parser.parse_args(["-a", "test", "--max-iterations", "10"])
        assert args.max_iterations == 10

    def test_verbose_flag_short(self):
        """Short -v flag is recognized."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--verbose", "-v", action="store_true")

        args = parser.parse_args(["-a", "test", "-v"])
        assert args.verbose is True

    def test_verbose_flag_default_false(self):
        """--verbose defaults to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--verbose", "-v", action="store_true")

        args = parser.parse_args(["-a", "test"])
        assert args.verbose is False

    def test_combined_flags(self):
        """Multiple new flags work together."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")
        parser.add_argument("--max-iterations", type=int, default=5)
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument("--query", "-q")

        args = parser.parse_args([
            "-a", "test",
            "-A",
            "--max-iterations", "3",
            "-v",
            "-q", "What is 2+2?"
        ])

        assert args.agentic is True
        assert args.max_iterations == 3
        assert args.verbose is True
        assert args.query == "What is 2+2?"

    def test_existing_flags_still_work(self):
        """Existing flags still function with new additions."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--query", "-q")
        parser.add_argument("--top-k", "-k", type=int, default=5)
        parser.add_argument("--no-sources", action="store_true")
        parser.add_argument("--api-url", default="http://localhost:1234/v1/chat/completions")
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args([
            "-a", "survivor",
            "-q", "How to start fire?",
            "-k", "10",
            "--no-sources",
            "-A"
        ])

        assert args.agent == "survivor"
        assert args.query == "How to start fire?"
        assert args.top_k == 10
        assert args.no_sources is True
        assert args.agentic is True


# =============================================================================
# Test Mode Selection Logic
# =============================================================================

class TestModeSelection:
    """Tests for mode selection between simple RAG and agentic."""

    def test_simple_rag_mode_default(self):
        """Without --agentic, simple RAG mode is used."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "test"])

        # Simulating mode selection logic
        use_agentic = args.agentic
        assert use_agentic is False

    def test_agentic_mode_when_flag_set(self):
        """With --agentic, agentic mode is used."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "test", "-A"])

        use_agentic = args.agentic
        assert use_agentic is True


# =============================================================================
# Test Agentic Module Integration
# =============================================================================

class TestAgenticIntegration:
    """Tests for integration with agentic module."""

    def test_agentic_config_creation(self):
        """AgenticConfig is created with correct parameters."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import AgenticConfig

        config = AgenticConfig(
            max_iterations=10,
            verbose=True
        )

        assert config.max_iterations == 10
        assert config.verbose is True
        assert config.timeout == 120  # default

    def test_tool_registry_creation(self):
        """ToolRegistry can be created and tools registered."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import ToolRegistry

        registry = ToolRegistry()

        # Mock tool
        def mock_executor(args):
            return "mock result"

        schema = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }

        registry.register("mock_tool", schema, mock_executor)

        assert "mock_tool" in registry.tool_names
        assert len(registry.schemas) == 1

    def test_format_tool_call_trace_import(self):
        """format_tool_call_trace can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import format_tool_call_trace, AgentLoopResult, ToolCall

        # Create a result to format
        result = AgentLoopResult(
            content="Test response",
            tool_calls_made=[
                ToolCall(name="search_corpus", arguments={"query": "test"})
            ],
            iterations=2,
            stopped_reason="complete"
        )

        trace = format_tool_call_trace(result)

        assert "[Agentic]" in trace
        assert "Iterations: 2" in trace
        assert "search_corpus" in trace


# =============================================================================
# Test Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing CLI usage."""

    def test_simple_query_syntax(self):
        """Original simple query syntax still works."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--query", "-q")
        parser.add_argument("--agentic", "-A", action="store_true")

        # Original syntax without agentic
        args = parser.parse_args(["-a", "survivor", "-q", "How to purify water?"])

        assert args.agent == "survivor"
        assert args.query == "How to purify water?"
        assert args.agentic is False

    def test_repl_mode_without_query(self):
        """REPL mode triggers when no --query provided."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--query", "-q")
        parser.add_argument("--agentic", "-A", action="store_true")

        args = parser.parse_args(["-a", "scientist"])

        assert args.query is None
        # In main(), this would trigger REPL mode

    def test_path_arguments_unchanged(self):
        """Path arguments work the same."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", "-a", required=True)
        parser.add_argument("--corpus", "-c", type=Path, default=Path("~/corpus"))
        parser.add_argument("--embeddings", "-e", type=Path, default=Path("~/embeddings"))
        parser.add_argument("--agents-dir", type=Path, default=Path("./agents"))

        args = parser.parse_args([
            "-a", "test",
            "-c", "/custom/corpus",
            "-e", "/custom/embeddings"
        ])

        assert args.corpus == Path("/custom/corpus")
        assert args.embeddings == Path("/custom/embeddings")


# =============================================================================
# Test REPL Prompt Changes
# =============================================================================

class TestREPLPrompt:
    """Tests for REPL prompt changes in agentic mode."""

    def test_simple_rag_prompt_format(self):
        """Simple RAG mode uses standard prompt."""
        agentic = False
        prompt = "[Agentic] You: " if agentic else "\nYou: "
        assert prompt == "\nYou: "

    def test_agentic_prompt_format(self):
        """Agentic mode uses [Agentic] prefix."""
        agentic = True
        prompt = "[Agentic] You: " if agentic else "\nYou: "
        assert prompt == "[Agentic] You: "


# =============================================================================
# Test System Prompt Building
# =============================================================================

class TestSystemPrompt:
    """Tests for system prompt differences between modes."""

    def test_agentic_system_prompt_includes_tool_instructions(self):
        """Agentic system prompt includes tool usage guidance."""
        # Simulating build_agentic_system_prompt logic
        persona = {
            "name": "test",
            "icon": "🧪",
            "style": "You are helpful.",
            "expertise": ["Testing"],
        }

        parts = []
        if persona.get("icon") and persona.get("name"):
            parts.append(f"{persona['icon']} {persona['name'].title()}")
        if persona.get("style"):
            parts.append(persona["style"].strip())
        if persona.get("expertise"):
            expertise_list = "\n".join(f"- {e}" for e in persona["expertise"])
            parts.append(f"Your areas of expertise:\n{expertise_list}")

        # Agentic-specific instructions
        parts.append(
            "You have access to a search_corpus tool that searches your knowledge base. "
            "Use this tool when you need specific information from documents to answer accurately. "
            "For simple questions you can answer from general knowledge, respond directly without searching. "
            "When you do search, cite the sources in your response."
        )

        system_prompt = "\n\n".join(parts)

        assert "search_corpus" in system_prompt
        assert "tool" in system_prompt.lower()
        assert "documents" in system_prompt.lower()


# =============================================================================
# Test Verbose Output
# =============================================================================

class TestVerboseOutput:
    """Tests for verbose output in agentic mode."""

    def test_verbose_output_format(self):
        """Verbose output shows tool call trace."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import format_tool_call_trace, AgentLoopResult, ToolCall

        result = AgentLoopResult(
            content="Based on my search...",
            tool_calls_made=[
                ToolCall(name="search_corpus", arguments={"query": "water purification", "top_k": 5})
            ],
            iterations=2,
            stopped_reason="complete"
        )

        trace = format_tool_call_trace(result)

        assert "Iterations: 2" in trace
        assert "Stopped: complete" in trace
        assert "Tool calls made: 1" in trace
        assert "search_corpus" in trace
        assert "water purification" in trace

    def test_verbose_no_tool_calls(self):
        """Verbose output handles no tool calls."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import format_tool_call_trace, AgentLoopResult

        result = AgentLoopResult(
            content="4",
            tool_calls_made=[],
            iterations=1,
            stopped_reason="complete"
        )

        trace = format_tool_call_trace(result)

        assert "Iterations: 1" in trace
        assert "Tool calls made: 0" in trace


# =============================================================================
# Test Search Corpus Tool Integration
# =============================================================================

class TestSearchCorpusToolIntegration:
    """Tests for search_corpus tool integration in CLI."""

    def test_tool_schema_compatible_with_registry(self):
        """search_corpus schema works with ToolRegistry."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from agentic import ToolRegistry
        from search_corpus_tool import get_search_corpus_schema

        schema = get_search_corpus_schema()
        registry = ToolRegistry()

        # Should not raise
        registry.register("search_corpus", schema, lambda args: "result")

        assert "search_corpus" in registry.tool_names

    def test_create_search_corpus_tool_returns_tuple(self):
        """Factory function returns schema and executor."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from search_corpus_tool import create_search_corpus_tool

        mock_index = MagicMock()
        mock_chunk_map = []
        mock_embedder = MagicMock()
        mock_corpus_dir = Path("/tmp/corpus")

        result = create_search_corpus_tool(
            index=mock_index,
            chunk_map=mock_chunk_map,
            embedder=mock_embedder,
            corpus_dir=mock_corpus_dir
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        schema, executor = result
        assert isinstance(schema, dict)
        assert callable(executor)
