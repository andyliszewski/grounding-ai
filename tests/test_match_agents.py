"""Tests for scripts/match_agents.py (Epic 24, Story 24.3 / W6).

The watcher's old hand-rolled bash YAML matcher only understood block-style,
unquoted lists. These tests pin the replacement (a real yaml.safe_load via
grounding.agent_filter) against every valid YAML list form so no agent can be
silently skipped for embedding updates because of YAML style.
"""

import importlib.util
from pathlib import Path

import pytest

# Load scripts/match_agents.py as a module (it isn't an importable package).
_MATCH_AGENTS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "match_agents.py"
_spec = importlib.util.spec_from_file_location("match_agents", _MATCH_AGENTS_PATH)
match_agents = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match_agents)
find_matching_agents = match_agents.find_matching_agents


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    """Agent YAMLs covering block, flow, quoted, and no-filter forms."""
    d = tmp_path / "agents"
    d.mkdir()

    (d / "scientist.yaml").write_text(
        "name: scientist\n"
        "corpus_filter:\n"
        "  collections:\n"
        "    - science\n"
        "    - biology\n"
    )
    # Flow style: the old matcher returned ZERO matches for this.
    (d / "flow-agent.yaml").write_text(
        "name: flow-agent\n"
        "corpus_filter:\n"
        "  collections: [physics, science]\n"
    )
    # Quoted entries (double + single): the old matcher kept the quotes and
    # failed the equality check.
    (d / "quoted-agent.yaml").write_text(
        "name: quoted-agent\n"
        "corpus_filter:\n"
        "  collections:\n"
        '    - "geology"\n'
        "    - 'meteorology'\n"
    )
    # No corpus_filter at all -> never matches anything.
    (d / "no-filter.yaml").write_text("name: no-filter\ndescription: nothing\n")
    return d


def test_block_style_match(agents_dir):
    assert find_matching_agents(agents_dir, "biology") == ["scientist"]


def test_flow_style_match(agents_dir):
    # W6: flow-style list must match.
    assert find_matching_agents(agents_dir, "physics") == ["flow-agent"]


def test_double_quoted_match(agents_dir):
    # W6: double-quoted entry must match.
    assert find_matching_agents(agents_dir, "geology") == ["quoted-agent"]


def test_single_quoted_match(agents_dir):
    # W6: single-quoted entry must match.
    assert find_matching_agents(agents_dir, "meteorology") == ["quoted-agent"]


def test_shared_collection_matches_all_forms(agents_dir):
    # 'science' appears in a block list AND a flow list -> both match, sorted by
    # file stem (glob order).
    assert find_matching_agents(agents_dir, "science") == ["flow-agent", "scientist"]


def test_no_match_returns_empty(agents_dir):
    assert find_matching_agents(agents_dir, "nonexistent") == []


def test_agent_without_filter_never_matches(agents_dir):
    # no-filter.yaml has no corpus_filter; it must not appear for any collection.
    for collection in ("science", "physics", "geology", "anything"):
        assert "no-filter" not in find_matching_agents(agents_dir, collection)


def test_missing_agents_dir_returns_empty(tmp_path):
    assert find_matching_agents(tmp_path / "does-not-exist", "science") == []


def test_returns_file_stem_not_name_field(tmp_path):
    """The watcher passes the result to `grounding embeddings --agent <name>`,
    which resolves <name>.yaml by filename, so the matcher must return the file
    stem even if the YAML `name:` field differs."""
    d = tmp_path / "agents"
    d.mkdir()
    (d / "file-stem.yaml").write_text(
        "name: different-name-field\n"
        "corpus_filter:\n"
        "  collections:\n"
        "    - science\n"
    )
    assert find_matching_agents(d, "science") == ["file-stem"]


def test_comma_joined_collection_matches_on_first_element(agents_dir):
    """A multi-collection staging dir name must match an agent declaring ANY element.

    Regression: the whole string "biology,physics" was compared against each
    agent's collections list, so no agent ever matched and every document dropped
    into a comma-named staging directory was ingested with no embedding update.
    """
    assert find_matching_agents(agents_dir, "biology,physics") == [
        "flow-agent",
        "scientist",
    ]


def test_comma_joined_collection_matches_on_later_element(agents_dir):
    assert find_matching_agents(agents_dir, "nonexistent,geology") == ["quoted-agent"]


def test_comma_joined_whitespace_is_tolerated(agents_dir):
    assert find_matching_agents(agents_dir, "biology, physics") == [
        "flow-agent",
        "scientist",
    ]
    assert find_matching_agents(agents_dir, " geology ") == ["quoted-agent"]


def test_agent_matching_multiple_elements_appears_once(agents_dir):
    """scientist declares both science and biology; one embedding update, not two."""
    assert find_matching_agents(agents_dir, "science,biology") == [
        "flow-agent",
        "scientist",
    ]


def test_comma_joined_with_no_matching_element_returns_empty(agents_dir):
    assert find_matching_agents(agents_dir, "nonexistent,also-missing") == []


def test_empty_and_degenerate_collection_values_return_empty(agents_dir):
    for value in ("", ",", " , ", "   "):
        assert find_matching_agents(agents_dir, value) == []


def test_split_collections_helper():
    split = match_agents.split_collections
    assert split("a") == ["a"]
    assert split("a,b,c") == ["a", "b", "c"]
    assert split("a, b ,c") == ["a", "b", "c"]
    assert split("a,,b") == ["a", "b"]
    assert split("") == []
    assert split(",") == []


@pytest.fixture
def pinning_agents_dir(agents_dir: Path) -> Path:
    """Adds agents that reach documents through corpus_filter.slugs pins."""
    # Pins a doc whose collection (elite-power) no agent declares. Matching on
    # collections alone can never fire for this agent.
    (agents_dir / "pinner.yaml").write_text(
        "name: pinner\n"
        "corpus_filter:\n"
        "  collections:\n"
        "    - strategy\n"
        "  slugs:\n"
        "    - surveillance-capitalism-zuboff\n"
        "    - some-other-pinned-book\n"
    )
    # Pins a doc and then excludes it: the exclusion must win on the slug axis.
    (agents_dir / "excluder.yaml").write_text(
        "name: excluder\n"
        "corpus_filter:\n"
        "  collections:\n"
        "    - science\n"
        "  slugs:\n"
        "    - surveillance-capitalism-zuboff\n"
        "  exclude_slugs:\n"
        "    - surveillance-capitalism-zuboff\n"
    )
    return agents_dir


def test_slug_pin_matches_when_collection_does_not(pinning_agents_dir):
    """The regression: a slug-pinned doc must trigger its agent's update.

    'elite-power' is declared by no agent, so the collection axis returns
    nothing and the pinned document was silently never embedded for `pinner`.
    """
    assert find_matching_agents(pinning_agents_dir, "elite-power") == []
    assert find_matching_agents(
        pinning_agents_dir, "elite-power", ["surveillance-capitalism-zuboff"]
    ) == ["pinner"]


def test_slug_and_collection_axes_union(pinning_agents_dir):
    """A batch matches agents from either axis, each listed once."""
    result = find_matching_agents(
        pinning_agents_dir, "biology", ["surveillance-capitalism-zuboff"]
    )
    assert result == ["pinner", "scientist"]


def test_agent_matching_both_axes_appears_once(pinning_agents_dir):
    """pinner declares 'strategy' and pins the slug; one update, not two."""
    assert find_matching_agents(
        pinning_agents_dir, "strategy", ["surveillance-capitalism-zuboff"]
    ) == ["pinner"]


def test_exclude_slugs_suppresses_the_slug_axis(pinning_agents_dir):
    """`excluder` pins the slug but also excludes it, so the pin must not fire."""
    result = find_matching_agents(
        pinning_agents_dir, "elite-power", ["surveillance-capitalism-zuboff"]
    )
    assert "excluder" not in result


def test_exclude_slugs_does_not_cancel_a_collection_match(pinning_agents_dir):
    """Excluding one document says nothing about the rest of the batch.

    `excluder` declares 'science', so a science batch must still update it even
    though one document in that batch is on its exclude list.
    """
    result = find_matching_agents(
        pinning_agents_dir, "science", ["surveillance-capitalism-zuboff"]
    )
    assert "excluder" in result


def test_unknown_slug_matches_nothing(pinning_agents_dir):
    assert find_matching_agents(pinning_agents_dir, "nonexistent", ["not-pinned-anywhere"]) == []


def test_slugs_default_to_empty_preserving_old_call_sites(agents_dir):
    """Two-argument calls must behave exactly as before the slug axis existed."""
    assert find_matching_agents(agents_dir, "science") == ["flow-agent", "scientist"]
    assert find_matching_agents(agents_dir, "science", []) == ["flow-agent", "scientist"]


def test_empty_collection_with_slugs_still_matches(pinning_agents_dir):
    """A blank collection must not short-circuit the slug axis."""
    assert find_matching_agents(
        pinning_agents_dir, "", ["surveillance-capitalism-zuboff"]
    ) == ["pinner"]


def test_blank_slugs_are_ignored(pinning_agents_dir):
    assert find_matching_agents(pinning_agents_dir, "", ["", "  "]) == []


def test_malformed_agent_is_skipped_not_fatal(agents_dir):
    """A single unparseable/invalid agent file is skipped, not fatal — the
    watcher's best-effort behavior is preserved."""
    # Missing required 'name' field -> load_agent_config raises -> skipped.
    (agents_dir / "broken.yaml").write_text("corpus_filter:\n  collections:\n    - science\n")
    # The good block-style agent still matches; broken one is silently skipped.
    result = find_matching_agents(agents_dir, "science")
    assert "scientist" in result
    assert "broken" not in result
