#!/usr/bin/env python3
"""
Filesystem tools for agentic LLM tool calling.

This module implements filesystem access tools that allow local LLMs
to read files, write files, execute shell commands, and search files.

Usage:
    from filesystem_tools import create_all_filesystem_tools
    from agentic import ToolRegistry

    registry = ToolRegistry()
    for name, schema, executor in create_all_filesystem_tools():
        registry.register(name, schema, executor)
"""

import glob as glob_module
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

logger = logging.getLogger("filesystem_tools")


# =============================================================================
# Read Tool
# =============================================================================

def get_read_schema() -> dict:
    """Return the JSON schema for the read_file tool."""
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file at the given path. "
                "Returns the file contents as text. Use this to examine "
                "documents, code, configuration files, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path to the file to read. "
                            "Paths starting with ~ will be expanded."
                        )
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to read. Default is 500. "
                            "Use for large files to avoid overwhelming context."
                        ),
                        "default": 500
                    }
                },
                "required": ["path"]
            }
        }
    }


@dataclass
class ReadTool:
    """Tool executor for reading files."""

    def execute(self, arguments: dict) -> str:
        path = arguments.get("path", "")
        max_lines = arguments.get("max_lines", 500)

        if not path:
            return "Error: 'path' parameter is required"

        # Expand path
        expanded = Path(path).expanduser().resolve()

        if not expanded.exists():
            return f"Error: File not found: {expanded}"

        if not expanded.is_file():
            return f"Error: Path is not a file: {expanded}"

        try:
            content = expanded.read_text()
            lines = content.splitlines()

            if len(lines) > max_lines:
                truncated = "\n".join(lines[:max_lines])
                return f"{truncated}\n\n[Truncated: showing {max_lines} of {len(lines)} lines]"

            return content

        except PermissionError:
            return f"Error: Permission denied reading: {expanded}"
        except UnicodeDecodeError:
            return f"Error: File is not valid UTF-8 text: {expanded}"
        except Exception as e:
            return f"Error reading file: {e}"


def create_read_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create read_file tool with schema and executor."""
    tool = ReadTool()
    return get_read_schema(), tool.execute


# =============================================================================
# Write Tool
# =============================================================================

def get_write_schema() -> dict:
    """Return the JSON schema for the write_file tool."""
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file at the given path. "
                "Creates parent directories if needed. "
                "Overwrites existing files. Use with caution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path to the file to write. "
                            "Paths starting with ~ will be expanded."
                        )
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    }


@dataclass
class WriteTool:
    """Tool executor for writing files."""

    def execute(self, arguments: dict) -> str:
        path = arguments.get("path", "")
        content = arguments.get("content", "")

        if not path:
            return "Error: 'path' parameter is required"

        # Expand path
        expanded = Path(path).expanduser().resolve()

        try:
            # Create parent directories if needed
            expanded.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            expanded.write_text(content)
            return f"Successfully wrote {len(content)} bytes to {expanded}"

        except PermissionError:
            return f"Error: Permission denied writing to: {expanded}"
        except Exception as e:
            return f"Error writing file: {e}"


def create_write_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create write_file tool with schema and executor."""
    tool = WriteTool()
    return get_write_schema(), tool.execute


# =============================================================================
# Bash Tool
# =============================================================================

def get_bash_schema() -> dict:
    """Return the JSON schema for the bash tool."""
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command and return its output. "
                "Use for running system commands, scripts, git operations, etc. "
                "Commands run in a bash shell with a timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum seconds to wait for command completion. "
                            "Default is 30 seconds."
                        ),
                        "default": 30
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Working directory for the command. "
                            "Default is current directory."
                        )
                    }
                },
                "required": ["command"]
            }
        }
    }


@dataclass
class BashTool:
    """Tool executor for shell commands."""

    def execute(self, arguments: dict) -> str:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", 30)
        cwd = arguments.get("cwd")

        if not command:
            return "Error: 'command' parameter is required"

        # Expand cwd if provided
        if cwd:
            cwd = str(Path(cwd).expanduser().resolve())

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            output_parts = []

            if result.stdout:
                output_parts.append(result.stdout)

            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            if result.returncode != 0:
                output_parts.append(f"[exit code: {result.returncode}]")

            output = "\n".join(output_parts) if output_parts else "[no output]"

            # Truncate if too long
            if len(output) > 10000:
                output = output[:10000] + "\n\n[Output truncated at 10000 chars]"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"


def create_bash_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create bash tool with schema and executor."""
    tool = BashTool()
    return get_bash_schema(), tool.execute


# =============================================================================
# Glob Tool
# =============================================================================

def get_glob_schema() -> dict:
    """Return the JSON schema for the glob tool."""
    return {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files matching a glob pattern. "
                "Supports patterns like '*.py', '**/*.md', 'src/**/*.js'. "
                "Returns a list of matching file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to match files. Use ** for recursive matching. "
                            "Examples: '*.txt', 'docs/**/*.md', 'src/**/test_*.py'"
                        )
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Base directory to search from. Default is current directory. "
                            "Paths starting with ~ will be expanded."
                        )
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default is 100.",
                        "default": 100
                    }
                },
                "required": ["pattern"]
            }
        }
    }


@dataclass
class GlobTool:
    """Tool executor for file glob matching."""

    def execute(self, arguments: dict) -> str:
        pattern = arguments.get("pattern", "")
        base_path = arguments.get("path", ".")
        max_results = arguments.get("max_results", 100)

        if not pattern:
            return "Error: 'pattern' parameter is required"

        # Expand base path
        expanded = Path(base_path).expanduser().resolve()

        if not expanded.exists():
            return f"Error: Directory not found: {expanded}"

        try:
            # Combine base path with pattern
            full_pattern = str(expanded / pattern)

            # Use recursive glob
            matches = list(glob_module.glob(full_pattern, recursive=True))

            # Sort and limit results
            matches = sorted(matches)[:max_results]

            if not matches:
                return f"No files found matching: {pattern}"

            result = f"Found {len(matches)} file(s):\n"
            result += "\n".join(matches)

            if len(matches) == max_results:
                result += f"\n\n[Results limited to {max_results}]"

            return result

        except Exception as e:
            return f"Error in glob search: {e}"


def create_glob_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create glob tool with schema and executor."""
    tool = GlobTool()
    return get_glob_schema(), tool.execute


# =============================================================================
# Grep Tool
# =============================================================================

def get_grep_schema() -> dict:
    """Return the JSON schema for the grep tool."""
    return {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a pattern in files. "
                "Supports regular expressions. "
                "Returns matching lines with file paths and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Regular expression pattern to search for. "
                            "Use simple strings for literal matching."
                        )
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "File or directory to search in. "
                            "If a directory, searches recursively. "
                            "Default is current directory."
                        )
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to filter files. Example: '*.py' to only search Python files."
                        )
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search. Default is false.",
                        "default": False
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Default is 50.",
                        "default": 50
                    }
                },
                "required": ["pattern"]
            }
        }
    }


@dataclass
class GrepTool:
    """Tool executor for searching file contents."""

    def execute(self, arguments: dict) -> str:
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", ".")
        file_pattern = arguments.get("file_pattern", "*")
        ignore_case = arguments.get("ignore_case", False)
        max_matches = arguments.get("max_matches", 50)

        if not pattern:
            return "Error: 'pattern' parameter is required"

        # Expand search path
        expanded = Path(search_path).expanduser().resolve()

        if not expanded.exists():
            return f"Error: Path not found: {expanded}"

        try:
            # Compile regex
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []

        try:
            if expanded.is_file():
                # Search single file
                matches = self._search_file(expanded, regex, max_matches)
            else:
                # Search directory recursively
                for filepath in expanded.rglob(file_pattern):
                    if filepath.is_file():
                        file_matches = self._search_file(filepath, regex, max_matches - len(matches))
                        matches.extend(file_matches)

                        if len(matches) >= max_matches:
                            break

        except Exception as e:
            return f"Error during search: {e}"

        if not matches:
            return f"No matches found for: {pattern}"

        result = f"Found {len(matches)} match(es):\n\n"
        result += "\n".join(matches)

        if len(matches) >= max_matches:
            result += f"\n\n[Results limited to {max_matches}]"

        return result

    def _search_file(self, filepath: Path, regex: re.Pattern, max_matches: int) -> list[str]:
        """Search a single file for matches."""
        matches = []

        try:
            content = filepath.read_text()
        except (UnicodeDecodeError, PermissionError):
            return []

        for line_num, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{filepath}:{line_num}: {line.strip()}")

                if len(matches) >= max_matches:
                    break

        return matches


def create_grep_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create grep tool with schema and executor."""
    tool = GrepTool()
    return get_grep_schema(), tool.execute


# =============================================================================
# Edit Tool
# =============================================================================

def get_edit_schema() -> dict:
    """Return the JSON schema for the edit_file tool."""
    return {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a targeted edit to a file by replacing a specific string. "
                "Use this instead of write_file when you only need to change part of a file. "
                "The old_string must match exactly (including whitespace)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace. Must be unique in the file."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace it with."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences. Default is false (replace first only).",
                        "default": False
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    }


@dataclass
class EditTool:
    """Tool executor for targeted file edits."""

    def execute(self, arguments: dict) -> str:
        path = arguments.get("path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        replace_all = arguments.get("replace_all", False)

        if not path:
            return "Error: 'path' parameter is required"
        if not old_string:
            return "Error: 'old_string' parameter is required"

        expanded = Path(path).expanduser().resolve()

        if not expanded.exists():
            return f"Error: File not found: {expanded}"

        try:
            content = expanded.read_text()

            # Check if old_string exists
            count = content.count(old_string)
            if count == 0:
                return f"Error: old_string not found in file"

            if count > 1 and not replace_all:
                return f"Error: old_string found {count} times. Use replace_all=true or provide a more unique string."

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                expanded.write_text(new_content)
                return f"Replaced {count} occurrence(s) in {expanded}"
            else:
                new_content = content.replace(old_string, new_string, 1)
                expanded.write_text(new_content)
                return f"Replaced 1 occurrence in {expanded}"

        except PermissionError:
            return f"Error: Permission denied: {expanded}"
        except Exception as e:
            return f"Error editing file: {e}"


def create_edit_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create edit_file tool with schema and executor."""
    tool = EditTool()
    return get_edit_schema(), tool.execute


# =============================================================================
# NotebookEdit Tool
# =============================================================================

def get_notebook_edit_schema() -> dict:
    """Return the JSON schema for the notebook_edit tool."""
    return {
        "type": "function",
        "function": {
            "name": "notebook_edit",
            "description": (
                "Edit a Jupyter notebook (.ipynb file). "
                "Can replace cell contents, insert new cells, or delete cells. "
                "Cells are identified by index (0-based)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the .ipynb notebook file."
                    },
                    "cell_index": {
                        "type": "integer",
                        "description": "Index of the cell to edit (0-based). For insert, the new cell goes after this index."
                    },
                    "new_source": {
                        "type": "string",
                        "description": "New source code/content for the cell."
                    },
                    "cell_type": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "description": "Type of cell. Required for insert mode.",
                        "default": "code"
                    },
                    "edit_mode": {
                        "type": "string",
                        "enum": ["replace", "insert", "delete"],
                        "description": "Edit mode: replace cell contents, insert new cell, or delete cell.",
                        "default": "replace"
                    }
                },
                "required": ["path", "cell_index", "new_source"]
            }
        }
    }


@dataclass
class NotebookEditTool:
    """Tool executor for Jupyter notebook editing."""

    def execute(self, arguments: dict) -> str:
        path = arguments.get("path", "")
        cell_index = arguments.get("cell_index")
        new_source = arguments.get("new_source", "")
        cell_type = arguments.get("cell_type", "code")
        edit_mode = arguments.get("edit_mode", "replace")

        if not path:
            return "Error: 'path' parameter is required"
        if cell_index is None:
            return "Error: 'cell_index' parameter is required"

        expanded = Path(path).expanduser().resolve()

        if not expanded.exists():
            return f"Error: Notebook not found: {expanded}"

        if not str(expanded).endswith(".ipynb"):
            return "Error: File must be a .ipynb notebook"

        try:
            # Load notebook
            with open(expanded) as f:
                notebook = json.load(f)

            cells = notebook.get("cells", [])

            if edit_mode == "delete":
                if cell_index < 0 or cell_index >= len(cells):
                    return f"Error: cell_index {cell_index} out of range (0-{len(cells)-1})"
                del cells[cell_index]
                notebook["cells"] = cells
                with open(expanded, "w") as f:
                    json.dump(notebook, f, indent=1)
                return f"Deleted cell {cell_index} from {expanded}"

            elif edit_mode == "insert":
                new_cell = {
                    "cell_type": cell_type,
                    "source": new_source.splitlines(keepends=True),
                    "metadata": {},
                }
                if cell_type == "code":
                    new_cell["outputs"] = []
                    new_cell["execution_count"] = None

                insert_at = cell_index + 1 if cell_index >= 0 else 0
                cells.insert(insert_at, new_cell)
                notebook["cells"] = cells
                with open(expanded, "w") as f:
                    json.dump(notebook, f, indent=1)
                return f"Inserted new {cell_type} cell at index {insert_at} in {expanded}"

            else:  # replace
                if cell_index < 0 or cell_index >= len(cells):
                    return f"Error: cell_index {cell_index} out of range (0-{len(cells)-1})"
                cells[cell_index]["source"] = new_source.splitlines(keepends=True)
                if cell_type:
                    cells[cell_index]["cell_type"] = cell_type
                with open(expanded, "w") as f:
                    json.dump(notebook, f, indent=1)
                return f"Replaced cell {cell_index} in {expanded}"

        except json.JSONDecodeError:
            return f"Error: Invalid notebook JSON: {expanded}"
        except Exception as e:
            return f"Error editing notebook: {e}"


def create_notebook_edit_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create notebook_edit tool with schema and executor."""
    tool = NotebookEditTool()
    return get_notebook_edit_schema(), tool.execute


# =============================================================================
# WebFetch Tool
# =============================================================================

def get_web_fetch_schema() -> dict:
    """Return the JSON schema for the web_fetch tool."""
    return {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch content from a URL and return it as text. "
                "Useful for reading web pages, APIs, documentation, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds. Default is 30.",
                        "default": 30
                    }
                },
                "required": ["url"]
            }
        }
    }


@dataclass
class WebFetchTool:
    """Tool executor for fetching web content."""

    def execute(self, arguments: dict) -> str:
        url = arguments.get("url", "")
        timeout = arguments.get("timeout", 30)

        if not url:
            return "Error: 'url' parameter is required"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; LocalAgent/1.0)"
            }
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()

            content = response.text

            # Truncate if too long
            if len(content) > 50000:
                content = content[:50000] + "\n\n[Content truncated at 50000 chars]"

            return content

        except requests.exceptions.Timeout:
            return f"Error: Request timed out after {timeout} seconds"
        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {e}"


def create_web_fetch_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create web_fetch tool with schema and executor."""
    tool = WebFetchTool()
    return get_web_fetch_schema(), tool.execute


# =============================================================================
# WebSearch Tool
# =============================================================================

def get_web_search_schema() -> dict:
    """Return the JSON schema for the web_search tool."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. "
                "Returns a list of search results with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default is 5.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }


@dataclass
class WebSearchTool:
    """Tool executor for web search via DuckDuckGo."""

    def execute(self, arguments: dict) -> str:
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        if not query:
            return "Error: 'query' parameter is required"

        try:
            # Use DuckDuckGo HTML search (no API key needed)
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; LocalAgent/1.0)"
            }
            response = requests.post(
                url,
                data={"q": query},
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            # Parse results (simple regex extraction)
            results = []
            html = response.text

            # Extract result blocks
            import re
            result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'

            links = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)

            for i, (href, title) in enumerate(links[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                results.append(f"{i+1}. {title.strip()}\n   URL: {href}\n   {snippet.strip()}")

            if not results:
                return f"No search results found for: {query}"

            return f"Search results for '{query}':\n\n" + "\n\n".join(results)

        except Exception as e:
            return f"Error searching: {e}"


def create_web_search_tool() -> tuple[dict, Callable[[dict], str]]:
    """Create web_search tool with schema and executor."""
    tool = WebSearchTool()
    return get_web_search_schema(), tool.execute


# =============================================================================
# Factory Function
# =============================================================================

def create_all_filesystem_tools() -> list[tuple[str, dict, Callable[[dict], str]]]:
    """
    Create all agent tools.

    Returns:
        List of (name, schema, executor) tuples ready for registration.

    Example:
        registry = ToolRegistry()
        for name, schema, executor in create_all_filesystem_tools():
            registry.register(name, schema, executor)
    """
    tools = []

    # Read
    schema, executor = create_read_tool()
    tools.append(("read_file", schema, executor))

    # Write
    schema, executor = create_write_tool()
    tools.append(("write_file", schema, executor))

    # Edit
    schema, executor = create_edit_tool()
    tools.append(("edit_file", schema, executor))

    # Bash
    schema, executor = create_bash_tool()
    tools.append(("bash", schema, executor))

    # Glob
    schema, executor = create_glob_tool()
    tools.append(("glob", schema, executor))

    # Grep
    schema, executor = create_grep_tool()
    tools.append(("grep", schema, executor))

    # NotebookEdit
    schema, executor = create_notebook_edit_tool()
    tools.append(("notebook_edit", schema, executor))

    # WebFetch
    schema, executor = create_web_fetch_tool()
    tools.append(("web_fetch", schema, executor))

    # WebSearch
    schema, executor = create_web_search_tool()
    tools.append(("web_search", schema, executor))

    return tools
