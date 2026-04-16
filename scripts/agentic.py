#!/usr/bin/env python3
"""
Agentic tool calling infrastructure for local LLMs.

This module provides the core loop mechanism for multi-turn tool calling
with Ollama's OpenAI-compatible API. It enables LLMs to autonomously
decide when to call tools and process their results.

Usage:
    from agentic import run_agentic_loop, AgenticConfig, ToolRegistry

    registry = ToolRegistry()
    registry.register("search_corpus", schema, executor_func)

    result = run_agentic_loop(
        user_query="What is thermodynamics?",
        tools=registry.schemas,
        tool_executor=registry.execute,
        model="qwen2.5:14b",
        api_url="http://localhost:11434/v1/chat/completions"
    )
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import json
import logging
import sys
import threading
import itertools

import requests

logger = logging.getLogger("agentic")


# =============================================================================
# UI Helpers
# =============================================================================

class Spinner:
    """Simple terminal spinner for indicating work in progress."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

    def _spin(self):
        while self.running:
            char = next(self.spinner_chars)
            sys.stdout.write(f'\r{char} {self.message}...')
            sys.stdout.flush()
            threading.Event().wait(0.1)
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def update(self, message: str):
        """Update the spinner message."""
        self.message = message


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgenticConfig:
    """Configuration for agentic loop behavior."""
    max_iterations: int = 5
    verbose: bool = False
    timeout: int = 120  # seconds per LLM call
    show_spinner: bool = True  # Show spinner while waiting for LLM
    num_predict: int = 2048  # Max tokens in response (Ollama default is often 128)


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentLoopResult:
    """Result from running the agentic loop."""
    content: str
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "complete"  # "complete", "max_iterations", "error"
    messages: list[dict] = field(default_factory=list)  # Conversation history for continuity


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Registry for tool executors.

    Manages tool schemas and their corresponding executor functions.
    Tools are registered with a name, JSON schema, and executor callable.
    """

    def __init__(self):
        self._tools: dict[str, Callable[[dict], str]] = {}
        self._schemas: list[dict] = []

    def register(
        self,
        name: str,
        schema: dict,
        executor: Callable[[dict], str]
    ) -> None:
        """
        Register a tool with its schema and executor.

        Args:
            name: Tool name (must match schema function name)
            schema: OpenAI-format tool schema
            executor: Function that takes arguments dict and returns string result
        """
        self._tools[name] = executor
        self._schemas.append(schema)

    def execute(self, name: str, arguments: dict) -> str:
        """
        Execute a tool by name.

        Args:
            name: Tool name to execute
            arguments: Arguments dict to pass to the tool

        Returns:
            Tool result as string, or error message if tool not found
        """
        if name not in self._tools:
            return f"Unknown tool: {name}"
        return self._tools[name](arguments)

    @property
    def schemas(self) -> list[dict]:
        """Return list of all registered tool schemas."""
        return self._schemas.copy()

    @property
    def tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())


# =============================================================================
# Ollama API Integration
# =============================================================================

def query_ollama_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    api_url: str,
    timeout: int = 120,
    verbose: bool = False,
    num_predict: int = 2048
) -> dict:
    """
    Send a request to Ollama with tool definitions.

    Uses the OpenAI-compatible chat completions endpoint with tools parameter.

    Args:
        messages: Conversation history (system, user, assistant, tool messages)
        tools: List of tool schemas in OpenAI format
        model: Ollama model name (e.g., "qwen2.5:14b")
        api_url: Ollama API endpoint (e.g., "http://localhost:11434/v1/chat/completions")
        timeout: Request timeout in seconds
        verbose: If True, log debug information
        num_predict: Maximum tokens in response (default 2048, Ollama default is often 128)

    Returns:
        Response dict with 'message' key containing:
        - role: "assistant"
        - content: Optional text response
        - tool_calls: Optional list of tool call requests

    Raises:
        requests.RequestException: If the API request fails
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools if tools else None,
        "stream": False,
        "options": {
            "num_predict": num_predict
        }
    }

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    if verbose:
        logger.info(f"Sending {len(tools)} tools to {model}")
        tool_names = [t.get('function', {}).get('name', '?') for t in tools]
        logger.info(f"Tool names: {tool_names}")

    response = requests.post(
        api_url,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    result = response.json()

    if verbose:
        # Log response structure
        msg = result.get("message") or (result.get("choices", [{}])[0].get("message") if result.get("choices") else {})
        if msg:
            has_tools = bool(msg.get("tool_calls"))
            content_preview = (msg.get("content", "") or "")[:100]
            logger.info(f"Response has tool_calls: {has_tools}, content: {content_preview}...")

    return result


def parse_tool_calls(response: dict, verbose: bool = False) -> list[ToolCall]:
    """
    Parse tool calls from an Ollama response.

    Args:
        response: Raw response dict from Ollama API
        verbose: If True, log debug information about parsing

    Returns:
        List of ToolCall objects, empty if no tool calls
    """
    # Try primary format (Ollama native)
    message = response.get("message", {})
    raw_tool_calls = message.get("tool_calls", [])

    # Try alternate format (OpenAI-compatible choices array)
    if not raw_tool_calls:
        choices = response.get("choices", [])
        if choices:
            alt_msg = choices[0].get("message", {})
            raw_tool_calls = alt_msg.get("tool_calls", [])

    if verbose:
        logger.info(f"Raw response keys: {response.keys()}")
        if message:
            logger.info(f"Message keys: {message.keys()}")
        logger.info(f"Found {len(raw_tool_calls)} raw tool calls")

    parsed = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        arguments = func.get("arguments", {})

        # Arguments may be a JSON string or already parsed dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        parsed.append(ToolCall(name=name, arguments=arguments))

    return parsed


# =============================================================================
# Agent Loop
# =============================================================================

def run_agentic_loop(
    user_query: str,
    tools: list[dict],
    tool_executor: Callable[[str, dict], str],
    model: str,
    api_url: str,
    config: AgenticConfig | None = None,
    system_prompt: str | None = None,
    conversation_history: list[dict] | None = None
) -> AgentLoopResult:
    """
    Run the agentic loop until completion or max iterations.

    This is the main entry point for agentic tool calling. It sends the
    user query to the LLM, processes any tool calls, and continues until
    the LLM returns a final response (no tool calls) or max iterations
    is reached.

    Args:
        user_query: The user's question
        tools: List of tool schemas in OpenAI format
        tool_executor: Function that takes (tool_name, arguments) and returns result string
        model: Ollama model name
        api_url: Ollama API endpoint
        config: Optional configuration (defaults to AgenticConfig())
        system_prompt: Optional system message to prepend
        conversation_history: Optional list of previous messages for multi-turn conversations

    Returns:
        AgentLoopResult with:
        - content: Final text response
        - tool_calls_made: List of all tool calls executed
        - iterations: Number of LLM calls made
        - stopped_reason: "complete", "max_iterations", or "error"
        - messages: Updated conversation history (for continuity)
    """
    config = config or AgenticConfig()

    # Start with existing conversation history or create new
    if conversation_history:
        messages = list(conversation_history)  # Copy to avoid mutation
    else:
        messages = []
        # Add system prompt only for new conversations
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

    # Add user query
    messages.append({"role": "user", "content": user_query})

    tool_calls_made: list[ToolCall] = []
    iteration = 0

    # Log available tools in verbose mode
    if config.verbose:
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        logger.info(f"Available tools: {tool_names}")

    # Create spinner if enabled
    spinner = None
    if config.show_spinner:
        spinner = Spinner("Thinking")

    while iteration < config.max_iterations:
        iteration += 1

        if config.verbose:
            logger.info(f"Agentic loop iteration {iteration}/{config.max_iterations}")

        # Start spinner
        if spinner:
            spinner.update(f"Thinking (iteration {iteration})")
            spinner.start()

        # Call LLM with tools
        try:
            response = query_ollama_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                api_url=api_url,
                timeout=config.timeout,
                verbose=config.verbose,
                num_predict=config.num_predict
            )
        except requests.RequestException as e:
            if spinner:
                spinner.stop()
            logger.error(f"API request failed: {e}")
            return AgentLoopResult(
                content=f"Error calling LLM API: {e}",
                tool_calls_made=tool_calls_made,
                iterations=iteration,
                stopped_reason="error",
                messages=messages
            )
        finally:
            if spinner:
                spinner.stop()

        # Extract assistant message
        assistant_msg = response.get("message", {})
        if not assistant_msg:
            # Try alternate response format (choices array)
            choices = response.get("choices", [])
            if choices:
                assistant_msg = choices[0].get("message", {})

        # Append assistant message to history
        messages.append(assistant_msg)

        # Parse tool calls
        tool_calls = parse_tool_calls(response, verbose=config.verbose)

        if not tool_calls:
            # No tool calls - LLM is done
            final_content = assistant_msg.get("content", "")
            if config.verbose:
                # Log if the model seems to be describing tool use instead of calling
                lower_content = final_content.lower()
                if any(word in lower_content for word in ['let me', "i'll", 'i will', 'searching', 'looking up']):
                    logger.warning("Model may be describing actions instead of calling tools")
                    logger.warning(f"Content preview: {final_content[:200]}")
            return AgentLoopResult(
                content=final_content,
                tool_calls_made=tool_calls_made,
                iterations=iteration,
                stopped_reason="complete",
                messages=messages
            )

        # Process each tool call
        for tc in tool_calls:
            tool_calls_made.append(tc)

            if config.verbose:
                logger.info(f"Tool call: {tc.name}({tc.arguments})")

            # Execute tool with error handling (AC5: errors returned as content)
            try:
                result = tool_executor(tc.name, tc.arguments)
            except Exception as e:
                result = f"Error executing tool '{tc.name}': {type(e).__name__}: {e}"
                logger.warning(f"Tool execution error: {e}")

            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": f"{tc.name}_{iteration}",
                "name": tc.name,
                "content": result
            })

        # Log warning when approaching limit
        if iteration == config.max_iterations - 1 and config.verbose:
            logger.warning("Approaching max iterations limit")

    # Max iterations reached
    logger.warning(f"Max iterations ({config.max_iterations}) reached")

    # Extract final content from last assistant message
    final_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_content = msg["content"]
            break

    return AgentLoopResult(
        content=final_content + "\n\n[Note: Maximum iterations reached]",
        tool_calls_made=tool_calls_made,
        iterations=iteration,
        stopped_reason="max_iterations",
        messages=messages
    )


# =============================================================================
# Utility Functions
# =============================================================================

def format_tool_call_trace(result: AgentLoopResult) -> str:
    """
    Format a verbose trace of tool calls for display.

    Args:
        result: AgentLoopResult from run_agentic_loop

    Returns:
        Formatted string showing iterations and tool calls
    """
    lines = [
        f"[Agentic] Iterations: {result.iterations}",
        f"[Agentic] Stopped: {result.stopped_reason}",
        f"[Agentic] Tool calls made: {len(result.tool_calls_made)}"
    ]

    for tc in result.tool_calls_made:
        args_str = json.dumps(tc.arguments, ensure_ascii=False)
        lines.append(f"  - {tc.name}({args_str})")

    return "\n".join(lines)
