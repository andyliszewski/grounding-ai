#!/usr/bin/env python3
"""Generate .claude/commands/ files from agent YAML definitions.

This script reads corpus agent configurations from the agents/ directory
and generates corresponding slash command files in .claude/commands/.

Usage:
    python scripts/generate_agent_commands.py
    python scripts/generate_agent_commands.py --agents-dir ./agents --commands-dir ./.claude/commands
    python scripts/generate_agent_commands.py --dry-run
"""

from pathlib import Path
import argparse
import sys

import yaml


# Default paths relative to project root
DEFAULT_AGENTS_DIR = Path("agents")
DEFAULT_COMMANDS_DIR = Path(".claude/commands")

# Command file template
COMMAND_TEMPLATE = '''---
description: "Activate {name} agent - {description}"
---

You are now activating as the **{display_name}** corpus agent.

## Persona

{icon} **{display_name}**

{style}

## Expertise Areas

{expertise_list}

## Corpus Scope

Your knowledge is grounded in documents from these collections:
{collections}

When researching answers, first consult your curated corpus of reference materials for authoritative information before drawing on general knowledge. If a topic falls outside your collections, acknowledge the limitation.

## Embeddings Path

`embeddings/{name}/_embeddings.faiss`

## Research Methodology

**IMPORTANT**: Follow this protocol for all domain questions:

1. **CORPUS FIRST**: Always search your corpus before answering:
   ```
   mcp__corpus_search__search_corpus(query="<your search terms>", agent="{name}", top_k=5)
   ```

2. **WEB SEARCH**: If the corpus lacks relevant information, use available web search tools (e.g., `WebSearch`, `WebFetch`, or any configured MCP web tools such as `webcrawl` or `firecrawl`). Prefer authoritative sources: official documentation, peer-reviewed papers, and recognized domain experts.

3. **CITATION REQUIRED**: Cite sources for important facts and principles:
   - Corpus sources: {citation_corpus_example}
   - Online sources: {citation_web_example}
   - Derived analysis: Mark as [Derived] rather than presenting as established fact

4. **VERIFY TRAINING KNOWLEDGE**: If citing a book, author, or concept from general training, verify it exists before citing. Do not fabricate references.

5. **CORPUS RECOMMENDATIONS**: If an authoritative source would strengthen your corpus but is not yet ingested, recommend it for addition (e.g., "{recommendation_example}").

---

{greeting}
'''

# Default persona values for agents without persona block
DEFAULT_PERSONA = {
    "icon": "📚",
    "style": "You are a knowledgeable assistant focused on your corpus domain.",
    "expertise": ["Domain expertise based on corpus content"],
    "greeting": "I'm ready to help with questions related to my knowledge domain.",
    "citation_corpus_example": "[Source: Author, Title, corpus]",
    "citation_web_example": "[Source: Author, Title, via web search]",
    "recommendation_example": "Consider adding <Title> by <Author> to the <collection> collection",
}


def load_agent_config(agent_file: Path) -> dict:
    """Load and parse an agent YAML configuration file."""
    with open(agent_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_expertise_list(expertise: list[str]) -> str:
    """Format expertise list as markdown bullet points."""
    return "\n".join(f"- {item}" for item in expertise)


def format_collections(collections: list[str]) -> str:
    """Format collections list as comma-separated string."""
    return ", ".join(collections)


def generate_display_name(name: str) -> str:
    """Convert agent name to display format (e.g., 'corp-dev-friendly' -> 'Corp Dev Friendly').

    Handles common acronyms that should remain uppercase.
    """
    # Acronyms that should be all caps
    acronyms = {"ceo", "ip", "qa", "hr", "cfo", "cto", "vp", "ma"}

    words = []
    for word in name.split("-"):
        if word.lower() in acronyms:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def generate_command_content(config: dict) -> str:
    """Generate the command file content from agent configuration."""
    name = config["name"]
    # Escape double quotes in description for YAML front matter
    description = config.get("description", "Corpus agent").replace('"', '\\"')

    # Get persona block with defaults
    persona = config.get("persona", {})
    icon = persona.get("icon", DEFAULT_PERSONA["icon"])
    style = persona.get("style", DEFAULT_PERSONA["style"]).strip()
    expertise = persona.get("expertise", DEFAULT_PERSONA["expertise"])
    greeting = persona.get("greeting", DEFAULT_PERSONA["greeting"]).strip()
    citation_corpus_example = persona.get(
        "citation_corpus_example", DEFAULT_PERSONA["citation_corpus_example"]
    ).strip()
    citation_web_example = persona.get(
        "citation_web_example", DEFAULT_PERSONA["citation_web_example"]
    ).strip()
    recommendation_example = persona.get(
        "recommendation_example", DEFAULT_PERSONA["recommendation_example"]
    ).strip()

    # Get corpus filter collections
    corpus_filter = config.get("corpus_filter", {})
    collections = corpus_filter.get("collections", [])

    return COMMAND_TEMPLATE.format(
        name=name,
        description=description,
        display_name=generate_display_name(name),
        icon=icon,
        style=style,
        expertise_list=format_expertise_list(expertise),
        collections=format_collections(collections) if collections else "No specific collections defined",
        greeting=greeting,
        citation_corpus_example=citation_corpus_example,
        citation_web_example=citation_web_example,
        recommendation_example=recommendation_example,
    )


def generate_commands(
    agents_dir: Path,
    commands_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Generate command files for all agents.

    Returns:
        Tuple of (generated_count, skipped_count)
    """
    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}", file=sys.stderr)
        return 0, 0

    # Ensure commands directory exists
    if not dry_run:
        commands_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for agent_file in sorted(agents_dir.glob("*.yaml")):
        try:
            config = load_agent_config(agent_file)
            name = config.get("name")

            if not name:
                print(f"Warning: Skipping {agent_file.name} - missing 'name' field", file=sys.stderr)
                skipped += 1
                continue

            command_file = commands_dir / f"{name}.md"
            content = generate_command_content(config)

            if dry_run:
                print(f"Would generate: {command_file}")
                if verbose:
                    print(f"  From: {agent_file}")
                    print(f"  Collections: {config.get('corpus_filter', {}).get('collections', [])}")
            else:
                command_file.write_text(content, encoding="utf-8")
                print(f"Generated: {command_file}")
                if verbose:
                    print(f"  From: {agent_file}")

            generated += 1

        except yaml.YAMLError as e:
            print(f"Error parsing {agent_file.name}: {e}", file=sys.stderr)
            skipped += 1
        except Exception as e:
            print(f"Error processing {agent_file.name}: {e}", file=sys.stderr)
            skipped += 1

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate .claude/commands/ files from agent YAML definitions"
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=DEFAULT_AGENTS_DIR,
        help=f"Directory containing agent YAML files (default: {DEFAULT_AGENTS_DIR})",
    )
    parser.add_argument(
        "--commands-dir",
        type=Path,
        default=DEFAULT_COMMANDS_DIR,
        help=f"Output directory for command files (default: {DEFAULT_COMMANDS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    print(f"Generating agent commands...")
    print(f"  Agents dir: {args.agents_dir}")
    print(f"  Commands dir: {args.commands_dir}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    generated, skipped = generate_commands(
        agents_dir=args.agents_dir,
        commands_dir=args.commands_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print()
    print(f"Summary: {generated} generated, {skipped} skipped")

    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
