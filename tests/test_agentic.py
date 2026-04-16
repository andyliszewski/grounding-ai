"""Tests for agentic tool calling module (Story 12.1)."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

# Import from scripts directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from agentic import (
    AgenticConfig,
    AgentLoopResult,
    ToolCall,
    ToolRegistry,
    format_tool_call_trace,
    parse_tool_calls,
    query_ollama_with_tools,
    run_agentic_loop,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock Responses
# ─────────────────────────────────────────────────────────────────────────────

MOCK_TOOL_CALL_RESPONSE = {
    "message": {
        "role": "assistant",
        "content": "I'll search for that information.",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "search_corpus",
                "arguments": {"query": "test query", "top_k": 5}
            }
        }]
    }
}

MOCK_FINAL_RESPONSE = {
    "message": {
        "role": "assistant",
        "content": "Based on the search results, here is the answer..."
    }
}

MOCK_MULTIPLE_TOOL_CALLS_RESPONSE = {
    "message": {
        "role": "assistant",
        "content": "I need to search multiple topics.",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "arguments": {"query": "topic one", "top_k": 3}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "arguments": {"query": "topic two", "top_k": 3}
                }
            }
        ]
    }
}

MOCK_TOOL_CALL_ARGS_AS_STRING = {
    "message": {
        "role": "assistant",
        "content": "Searching...",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "search_corpus",
                "arguments": '{"query": "string args", "top_k": 5}'
            }
        }]
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tool_schema() -> dict:
    """Sample tool schema for search_corpus."""
    return {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the corpus for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }


@pytest.fixture
def mock_tool_executor():
    """Mock tool executor that returns predictable results."""
    def executor(name: str, arguments: dict) -> str:
        if name == "search_corpus":
            return f"Found 3 results for: {arguments.get('query', 'unknown')}"
        return f"Unknown tool: {name}"
    return executor


@pytest.fixture
def failing_tool_executor():
    """Tool executor that raises an exception."""
    def executor(name: str, arguments: dict) -> str:
        raise ValueError("Tool execution failed!")
    return executor


# ─────────────────────────────────────────────────────────────────────────────
# ToolRegistry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self, sample_tool_schema):
        """Tool can be registered with schema and executor."""
        registry = ToolRegistry()
        executor = lambda args: "result"

        registry.register("search_corpus", sample_tool_schema, executor)

        assert "search_corpus" in registry.tool_names
        assert len(registry.schemas) == 1
        assert registry.schemas[0] == sample_tool_schema

    def test_execute_registered_tool(self, sample_tool_schema):
        """Registered tool executes correctly."""
        registry = ToolRegistry()
        executor = lambda args: f"searched: {args['query']}"
        registry.register("search_corpus", sample_tool_schema, executor)

        result = registry.execute("search_corpus", {"query": "test"})

        assert result == "searched: test"

    def test_execute_unknown_tool(self):
        """Unknown tool returns error message (AC5)."""
        registry = ToolRegistry()

        result = registry.execute("unknown_tool", {})

        assert "Unknown tool: unknown_tool" in result

    def test_schemas_returns_copy(self, sample_tool_schema):
        """schemas property returns a copy, not the original list."""
        registry = ToolRegistry()
        registry.register("test", sample_tool_schema, lambda x: "")

        schemas1 = registry.schemas
        schemas2 = registry.schemas

        assert schemas1 == schemas2
        assert schemas1 is not schemas2  # Different list instances


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_parse_single_tool_call(self):
        """Single tool call is correctly parsed (AC2)."""
        result = parse_tool_calls(MOCK_TOOL_CALL_RESPONSE)

        assert len(result) == 1
        assert result[0].name == "search_corpus"
        assert result[0].arguments == {"query": "test query", "top_k": 5}

    def test_parse_multiple_tool_calls(self):
        """Multiple tool calls in one response are parsed (AC8)."""
        result = parse_tool_calls(MOCK_MULTIPLE_TOOL_CALLS_RESPONSE)

        assert len(result) == 2
        assert result[0].name == "search_corpus"
        assert result[0].arguments["query"] == "topic one"
        assert result[1].arguments["query"] == "topic two"

    def test_parse_no_tool_calls(self):
        """Response without tool calls returns empty list."""
        result = parse_tool_calls(MOCK_FINAL_RESPONSE)

        assert result == []

    def test_parse_tool_call_args_as_string(self):
        """Arguments as JSON string are parsed correctly."""
        result = parse_tool_calls(MOCK_TOOL_CALL_ARGS_AS_STRING)

        assert len(result) == 1
        assert result[0].arguments == {"query": "string args", "top_k": 5}

    def test_parse_empty_response(self):
        """Empty response returns empty list."""
        result = parse_tool_calls({})

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Agent Loop Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgenticLoop:
    """Tests for run_agentic_loop function."""

    @patch("agentic.query_ollama_with_tools")
    def test_no_tool_calls_returns_immediately(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """Direct response without tool calls returns on first iteration (AC3)."""
        mock_query.return_value = MOCK_FINAL_RESPONSE

        result = run_agentic_loop(
            user_query="What is 2+2?",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        assert result.iterations == 1
        assert result.stopped_reason == "complete"
        assert len(result.tool_calls_made) == 0
        assert "Based on the search results" in result.content

    @patch("agentic.query_ollama_with_tools")
    def test_single_tool_call_executed(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """Single tool call is executed and result appended (AC2, AC6)."""
        # First call returns tool call, second returns final response
        mock_query.side_effect = [MOCK_TOOL_CALL_RESPONSE, MOCK_FINAL_RESPONSE]

        result = run_agentic_loop(
            user_query="Search for something",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        assert result.iterations == 2
        assert result.stopped_reason == "complete"
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].name == "search_corpus"

    @patch("agentic.query_ollama_with_tools")
    def test_multiple_tool_calls_per_turn(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """Multiple tool calls in one response are all executed (AC8)."""
        mock_query.side_effect = [
            MOCK_MULTIPLE_TOOL_CALLS_RESPONSE,
            MOCK_FINAL_RESPONSE
        ]

        result = run_agentic_loop(
            user_query="Compare two topics",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        assert result.iterations == 2
        assert len(result.tool_calls_made) == 2
        assert result.tool_calls_made[0].arguments["query"] == "topic one"
        assert result.tool_calls_made[1].arguments["query"] == "topic two"

    @patch("agentic.query_ollama_with_tools")
    def test_max_iterations_enforced(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """Loop stops at max_iterations (AC4)."""
        # Always return tool call - never completes
        mock_query.return_value = MOCK_TOOL_CALL_RESPONSE

        config = AgenticConfig(max_iterations=3)

        result = run_agentic_loop(
            user_query="Endless search",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions",
            config=config
        )

        assert result.iterations == 3
        assert result.stopped_reason == "max_iterations"
        assert "[Note: Maximum iterations reached]" in result.content

    @patch("agentic.query_ollama_with_tools")
    def test_tool_execution_error_handled(
        self, mock_query, sample_tool_schema, failing_tool_executor
    ):
        """Errors in tool execution returned as string, not raised (AC5)."""
        mock_query.side_effect = [MOCK_TOOL_CALL_RESPONSE, MOCK_FINAL_RESPONSE]

        # Should NOT raise - error is captured and returned
        result = run_agentic_loop(
            user_query="Search with failing tool",
            tools=[sample_tool_schema],
            tool_executor=failing_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        assert result.stopped_reason == "complete"
        # The tool error was sent to the LLM, which then responded
        assert mock_query.call_count == 2

    @patch("agentic.query_ollama_with_tools")
    def test_message_history_accumulates(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """Messages correctly accumulate through the loop (AC6)."""
        mock_query.side_effect = [MOCK_TOOL_CALL_RESPONSE, MOCK_FINAL_RESPONSE]

        result = run_agentic_loop(
            user_query="Test message history",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions",
            system_prompt="You are helpful."
        )

        # Verify the second call had accumulated messages
        second_call_messages = mock_query.call_args_list[1][1]["messages"]

        # Should have: system, user, assistant (with tool call), tool result
        assert len(second_call_messages) >= 4
        assert second_call_messages[0]["role"] == "system"
        assert second_call_messages[1]["role"] == "user"
        assert second_call_messages[2]["role"] == "assistant"
        assert second_call_messages[3]["role"] == "tool"

    @patch("agentic.query_ollama_with_tools")
    def test_system_prompt_included(
        self, mock_query, sample_tool_schema, mock_tool_executor
    ):
        """System prompt is included in messages."""
        mock_query.return_value = MOCK_FINAL_RESPONSE

        run_agentic_loop(
            user_query="Test",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions",
            system_prompt="You are a test assistant."
        )

        call_messages = mock_query.call_args[1]["messages"]
        assert call_messages[0] == {"role": "system", "content": "You are a test assistant."}

    @patch("agentic.query_ollama_with_tools")
    def test_default_config_used(self, mock_query, sample_tool_schema, mock_tool_executor):
        """Default AgenticConfig is used when not provided."""
        mock_query.return_value = MOCK_FINAL_RESPONSE

        result = run_agentic_loop(
            user_query="Test",
            tools=[sample_tool_schema],
            tool_executor=mock_tool_executor,
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
            # No config provided
        )

        assert result.stopped_reason == "complete"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataclasses:
    """Tests for dataclass definitions."""

    def test_agentic_config_defaults(self):
        """AgenticConfig has correct defaults."""
        config = AgenticConfig()

        assert config.max_iterations == 5
        assert config.verbose is False
        assert config.timeout == 120

    def test_tool_call_creation(self):
        """ToolCall can be created with name and arguments."""
        tc = ToolCall(name="search", arguments={"query": "test"})

        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_agent_loop_result_defaults(self):
        """AgentLoopResult has correct defaults."""
        result = AgentLoopResult(content="test")

        assert result.content == "test"
        assert result.tool_calls_made == []
        assert result.iterations == 0
        assert result.stopped_reason == "complete"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Function Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatToolCallTrace:
    """Tests for format_tool_call_trace function."""

    def test_format_with_tool_calls(self):
        """Trace includes tool call details."""
        result = AgentLoopResult(
            content="Final answer",
            tool_calls_made=[
                ToolCall(name="search_corpus", arguments={"query": "test"}),
                ToolCall(name="search_corpus", arguments={"query": "another"})
            ],
            iterations=3,
            stopped_reason="complete"
        )

        trace = format_tool_call_trace(result)

        assert "[Agentic] Iterations: 3" in trace
        assert "[Agentic] Stopped: complete" in trace
        assert "[Agentic] Tool calls made: 2" in trace
        assert "search_corpus" in trace
        assert '"query": "test"' in trace

    def test_format_without_tool_calls(self):
        """Trace handles zero tool calls."""
        result = AgentLoopResult(
            content="Direct answer",
            tool_calls_made=[],
            iterations=1,
            stopped_reason="complete"
        )

        trace = format_tool_call_trace(result)

        assert "[Agentic] Tool calls made: 0" in trace


# ─────────────────────────────────────────────────────────────────────────────
# API Integration Tests (with mocking)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryOllamaWithTools:
    """Tests for query_ollama_with_tools function."""

    @patch("agentic.requests.post")
    def test_sends_correct_payload(self, mock_post, sample_tool_schema):
        """Request includes model, messages, tools, and stream=False (AC1)."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_FINAL_RESPONSE
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        query_ollama_with_tools(
            messages=messages,
            tools=[sample_tool_schema],
            model="qwen2.5:14b",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]

        assert payload["model"] == "qwen2.5:14b"
        assert payload["messages"] == messages
        assert payload["tools"] == [sample_tool_schema]
        assert payload["stream"] is False

    @patch("agentic.requests.post")
    def test_handles_empty_tools_list(self, mock_post):
        """Empty tools list is handled (tools param omitted)."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_FINAL_RESPONSE
        mock_post.return_value = mock_response

        query_ollama_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions"
        )

        payload = mock_post.call_args[1]["json"]
        assert "tools" not in payload  # Empty list filtered out

    @patch("agentic.requests.post")
    def test_respects_timeout(self, mock_post):
        """Timeout is passed to requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_FINAL_RESPONSE
        mock_post.return_value = mock_response

        query_ollama_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            model="test-model",
            api_url="http://localhost:11434/v1/chat/completions",
            timeout=60
        )

        assert mock_post.call_args[1]["timeout"] == 60
