# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sample `systemd` unit file for the staging watcher at `scripts/grounding-watcher.service.example`.
- Research Methodology block auto-generated for every agent's slash command by `scripts/generate_agent_commands.py`. Includes corpus-first protocol, web-search fallback, citation format, and corpus-recommendation prompt.
- Optional `persona.citation_corpus_example`, `persona.citation_web_example`, and `persona.recommendation_example` fields on agent YAMLs for per-agent flavor in the methodology block.
- README troubleshooting section and end-to-end sample session output.
- Dedicated "Staging Watcher" section in the README covering single-machine setup, systemd deployment, and a reference to the multi-machine guide.

### Changed
- README Quick Start now leads with `pip install grounding-ai` (PyPI) and treats editable install as the developer path.
- README adds PyPI, Python version, and license badges.

### Fixed
- `CONTRIBUTING.md` referenced `master`; corrected to `main`.
- `scripts/publish.sh` leak scanner flagged the script itself as a false positive (its `LEAK_PATTERNS` literally contains the strings it searches for); scan now excludes `publish.sh`.
- `scripts/publish.sh` excluded from the public artifact so its `LEAK_PATTERNS`/`SCRUBS` (which enumerate private codenames) don't ship publicly.

### Docs
- `docs/ROADMAP.md` rewritten as a forward-looking, non-binding plan organized into retrieval-quality tiers (evaluation harness, page/section citations, cross-encoder reranking, hybrid retrieval, and follow-on items).

## [0.3.0] — 2025

Current PyPI release. Prior release notes live in `docs/MIGRATION-v0.1-to-v0.2.md` and `docs/ROADMAP.md`.
