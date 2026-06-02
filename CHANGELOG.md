# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Floor Starlette at `>=1.0.1` for CVE-2026-48710 ("BadHost"), in both the corpus-search MCP `Dockerfile` and the new `[mcp]` extra (see Added). Starlette `< 1.0.1` fails to validate the Host header and poisons `request.url.path`, enabling path-based auth bypass in ASGI middleware. It arrives transitively via `mcp` (floored only at `>=0.27`). The server runs over stdio (no HTTP listener), so the vector is not exploitable as shipped; the explicit floor keeps a known-vulnerable Starlette out of both the published image and dev installs. Local venv resolves to Starlette `1.2.1`.

### Added
- `[mcp]` optional-dependencies extra (`pip install -e .[mcp]`) declaring the corpus-search MCP server's extra runtime deps (`mcp`, `starlette>=1.0.1`). Previously these were installed only by the `Dockerfile`, so a local `pip install -e .` left the server unrunnable. The heavy retrieval deps it also needs are already in `[project.dependencies]`.
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
