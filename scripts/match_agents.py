#!/usr/bin/env python3
"""Print agent names that should have their embeddings updated for an ingestion batch.

Used by ``scripts/staging-watcher.sh`` (Epic 24, Story 24.3 / W6) to replace the
watcher's hand-rolled bash YAML parsing — which only understood block-style,
unquoted lists and silently missed flow-style (``collections: [a, b]``) and
quoted (``- "a"``) entries — with the repo's own real YAML parser
(``grounding.agent_filter.load_agent_config``). This way the watcher and
``grounding agents show`` agree by construction and no valid YAML list form
silently drops an agent from embedding updates.

Output: each matching agent's **file stem** (e.g. ``scientist`` for
``scientist.yaml``) on its own line — the value the watcher passes to
``grounding embeddings --agent <name>``, which resolves ``<name>.yaml`` by
filename. No match -> no output. Exit code is always 0 (a missing/unreadable
single agent file is logged to stderr and skipped, preserving the watcher's
prior best-effort behavior).

``--collection`` takes a watcher staging directory name, which may be a
comma-joined multi-collection name (``parenting,child-psychology``). It is split
on commas and an agent matches on **any** element; see ``split_collections`` for
why the previous whole-string comparison silently skipped every such document.

``--slug`` (repeatable) takes the slugs of the documents just ingested, so an
agent that reaches a document through a ``corpus_filter.slugs`` pin rather than
through its collection is matched too. Collections alone cannot see those: the
pinned document's collection often belongs to a different agent, or to no agent
at all, so nothing ever fired and the pin never took effect.

Usage:
    match_agents.py --agents-dir DIR --collection NAME
    match_agents.py --agents-dir DIR --collection NAME1,NAME2
    match_agents.py --agents-dir DIR --collection NAME --slug SLUG [--slug SLUG ...]
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Make the repo importable when run as a standalone script, so we can reuse the
# canonical YAML parser without requiring `grounding` to be pip-installed (only
# PyYAML is needed; everything else under grounding.agent_filter is stdlib).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grounding.agent_filter import AgentFilterError, load_agent_config  # noqa: E402


def split_collections(collection: str) -> list[str]:
    """Split a ``--collection`` value into its component collection names.

    The watcher passes the staging subdirectory name **verbatim**, and a
    multi-collection drop is a comma-joined directory name such as
    ``parenting,child-psychology`` — the same form ``grounding``'s own CLI
    splits on commas (``cli.py``: ``args.collections.split(",")``), which is why
    those documents land in the corpus correctly tagged.

    This function exists because matching the *joined* string against an agent's
    collections list can never hit: no agent declares a collection literally
    named ``parenting,child-psychology``. Every document dropped into a
    comma-named staging directory therefore resolved to zero matching agents and
    silently triggered no embedding update, leaving the docs ingested but
    unsearchable. Single-name directories were unaffected, which is what kept
    the bug hidden.
    """
    return [part.strip() for part in collection.split(",") if part.strip()]


def find_matching_agents(
    agents_dir: Path, collection: str, slugs: Sequence[str] = ()
) -> list[str]:
    """Return file stems of agents that should have their embeddings updated.

    An agent matches on **either** axis:

    * **Collection** — its ``corpus_filter.collections`` intersects `collection`,
      a watcher staging directory name that may be comma-joined (see
      ``split_collections``). A document tagged ``parenting,child-psychology``
      belongs to every agent that wants either one.
    * **Slug pin** — its ``corpus_filter.slugs`` contains one of `slugs`, the
      slugs of the documents just ingested. An agent reaches a document this way
      when it wants that specific book but does not declare its collection, and
      matching on collections alone can never see it: the document's collection
      may belong to a different agent entirely, or to none.

    ``corpus_filter.exclude_slugs`` suppresses the slug axis only. A document the
    agent explicitly excludes must not pull that agent into an update on its own
    account, but exclusion of one document in a batch says nothing about the
    others, so it never cancels a collection match.

    The two axes are deliberately asymmetric in strictness because the costs are
    asymmetric: an over-trigger is one redundant ``--incremental`` run, which is
    idempotent and cheap, while an under-trigger leaves a document silently
    unsearchable for that agent, which is the failure this function exists to
    prevent.

    Iterates agent YAML files in sorted (glob) order — matching the watcher's
    historical ``*.yaml`` ordering — and uses ``load_agent_config`` (a real
    ``yaml.safe_load``) so every valid list form is honored.
    """
    if not agents_dir.is_dir():
        return []

    wanted = set(split_collections(collection))
    pinned = {s.strip() for s in slugs if s and s.strip()}
    if not wanted and not pinned:
        return []

    matches: list[str] = []
    for agent_file in sorted(agents_dir.glob("*.yaml")):
        agent_name = agent_file.stem
        try:
            config = load_agent_config(agent_name, agents_dir)
        except AgentFilterError as exc:
            print(f"match_agents: skipping {agent_file.name}: {exc}", file=sys.stderr)
            continue
        if wanted & set(config.collections or []):
            matches.append(agent_name)
            continue
        agent_slugs = set(config.slugs or []) - set(config.exclude_slugs or [])
        if pinned & agent_slugs:
            matches.append(agent_name)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match agents by collection and/or ingested document slug."
    )
    parser.add_argument("--agents-dir", required=True, type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--slug",
        action="append",
        default=[],
        metavar="SLUG",
        help="Slug of a document in this batch; repeatable. Matches agents that "
        "pin the slug in corpus_filter.slugs.",
    )
    args = parser.parse_args()

    for name in find_matching_agents(args.agents_dir, args.collection, args.slug):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
