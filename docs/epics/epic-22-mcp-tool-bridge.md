# Epic 22: MCP Tool Bridge

**Epic ID:** E22
**Owner:** Andy
**Status:** Draft
**Priority:** P1
**Completed Stories:** 0/3
**Dependencies:** Epic 12 (Agentic Tool Calling)
**Coordinates with:** Epic 21 (Agentic-Mode Safeguards) — forward-compatible category propagation
**Target Completion:** TBD

---

## Branching Plan

This epic ships entirely on the public mirror. Per-story summary:

| Story | Branch target | Private-only content? | Cross-repo coordination |
|-------|---------------|------------------------|--------------------------|
| 22.1  | public `main` (feature branch → squash PR) | No | None |
| 22.2  | public `main` (feature branch → squash PR) | No | None — webcrawl-mcp is its own public repo (`andyliszewski/webcrawl-mcp`); only its install path is referenced |
| 22.3  | public `main` (feature branch → squash PR) | No | None |

**Defaults:** the bridge ships **disabled** by default. No MCP servers are
spawned unless the user opts in via `mcp_servers:` config block. A fresh
clone behaves bit-for-bit identically to pre-22 — no new processes, no
new tools registered, no new failure modes on the happy path of "I just
want simple RAG." Same opt-in discipline as Epic 18 (rerank) and 19 (hybrid).

**Cadence:** one squashed commit per story on public `main` as each PR
merges. 22.1 lands the abstraction, 22.2 exercises it against a real
out-of-repo MCP server (webcrawl-mcp), 22.3 closes with docs and an
end-to-end note in CLAUDE.md.

**Cross-repo impact:**
- `my-agents` repo: none. Agent YAML schema is unchanged.
- `Corpora/` Syncthing share: none. Ingestion path is untouched.
- `webcrawl-mcp` repo: none. The bridge consumes it as a pip-installable
  console script; no upstream changes required.

---

## Overview

`scripts/local_rag.py --agentic` (`scripts/local_rag.py:458-473`) registers
exactly the tools that ship in this repo: `search_corpus` plus the
filesystem suite (`bash`, `read_file`, `write_file`, `edit_file`, `glob`,
`grep`, `notebook_edit`). Anything outside that closed set — web search,
Gmail, Calendar, the user's other MCP servers — is invisible to the local
LLM, even when those servers are already running and used productively
in other tools (Claude Desktop, Claude Code).

The user has built one such server: `webcrawl-mcp`
(`~/Documents/webcrawl/`, public OSS at
`andyliszewski/webcrawl-mcp`). It's a standard MCP stdio server built on
FastMCP, exposing `webcrawl_scrape`, `webcrawl_search`, `webcrawl_map`,
and `webcrawl_crawl`. Wiring it into `--agentic` would close the
"local LLM can't reach the web" gap that surfaced in operational use
on 2026-04-25 (a current-events query against the `hari-seldon` agent
returned a training-data answer with no grounding).

Two ways to do it:

1. **Direct Python import.** Wrap webcrawl-mcp's underlying functions in
   a `create_webcrawl_tools()` factory analogous to
   `create_search_corpus_tool()` (`scripts/local_rag.py:460`). Small
   diff (~30 lines), works only for webcrawl-mcp, only when the package
   is importable inside grounding-ai's venv.

2. **Generic MCP bridge.** A single client-side adapter that spawns any
   configured MCP server as a stdio subprocess, performs `initialize` +
   `tools/list`, and registers each tool as a
   `(name, schema, executor)` triple via `ToolRegistry.register`
   (`scripts/agentic.py:122`). Larger diff (~150 lines), but reusable for
   webcrawl-mcp and every future MCP server (Gmail, Calendar, Drive,
   user-built servers, third-party servers).

This epic ships **path 2**. Webcrawl-mcp is the first concrete consumer
(Story 22.2), but the abstraction is the deliverable. The user already
runs multiple MCP servers in adjacent tools; a one-time investment in a
generic bridge avoids per-server glue code from here on.

**Problem Statement:**
- The local agentic loop is closed-world: the only registrable tools are
  those whose Python code lives in this repo. There is no mechanism to
  register tools that live in other processes, regardless of protocol.
- MCP is the de facto standard for tool packaging (Anthropic SDK,
  FastMCP, official servers from Anthropic and the broader ecosystem).
  Bypassing it forces every tool to be re-implemented as in-repo
  Python — duplicated effort, drift risk, no reuse across tools.
- The user has at least one ready-to-go MCP server (webcrawl-mcp) and
  uses several others (Gmail, Calendar, Drive, plus the corpus-search
  server in `mcp_servers/corpus_search/`). None are accessible to the
  local LLM today.
- Without web access, `--agentic` cannot answer current-events questions
  with grounding. The model either declines, hallucinates, or
  pattern-matches its training data — undermining the project's "ground
  every claim" thesis for any query whose answer post-dates training.

**Solution:**
- Add `scripts/mcp_tool_adapter.py` exporting `MCPClient` (manages one
  subprocess MCP server) and `register_mcp_tools()` (the integration
  helper that wires discovered tools into a `ToolRegistry`).
- Configure servers via a top-level `mcp_servers:` block in the existing
  `config.yaml` (or via a dedicated `--mcp-servers-config <path>` CLI
  flag). Per-server config: command, args, env, optional name override,
  optional category override (forward-compat with Epic 21).
- Wire `register_mcp_tools(registry, config)` into `local_rag.py`'s
  agentic-setup block (`scripts/local_rag.py:458-473`) right after the
  in-repo tools register.
- Lifecycle: subprocesses spawned at REPL launch, terminated at REPL
  exit (and on signal). Per-call timeout. Server crash mid-loop surfaces
  as a tool-result error message, not an unhandled exception.
- Tool-name collision policy: in-repo tools win; later-registering MCP
  tools that collide get namespaced (`<server_name>__<tool_name>`) and
  log a WARNING.

---

## Goals

1. Any pip-installable or absolute-path-invokable stdio MCP server can be
   registered into `local_rag.py --agentic` via config — no per-server
   Python glue in this repo.
2. The bridge is bit-for-bit transparent when no MCP servers are
   configured. A fresh clone runs identically to pre-22.
3. webcrawl-mcp is registered end-to-end and its four tools
   (`webcrawl_scrape`, `webcrawl_search`, `webcrawl_map`,
   `webcrawl_crawl`) are callable by the local LLM in the agentic loop.
4. The bridge handles realistic failure modes: server fails to start,
   server crashes mid-session, individual tool call times out, server
   returns malformed JSON-RPC. None should crash the REPL.
5. Tool-name collisions (in-repo vs. MCP, or MCP vs. MCP) resolve
   deterministically with a logged warning — no silent shadowing.
6. The adapter is reusable beyond `local_rag.py`: any future surface
   that builds a `ToolRegistry` (a CI eval runner, a different REPL,
   a future TUI) can call `register_mcp_tools(registry, config)` with
   no extra plumbing.
7. Forward-compatibility with Epic 21's `PermissionBroker`: each
   registered MCP tool carries a `category` (default `network` for the
   safest assumption; overridable per-server in config). When 21.1
   adds the broker, MCP tools route through it like any other.
8. The epic ships with a documented reference config for webcrawl-mcp
   in `config.example.yaml`, copy-pasteable into the user's `config.yaml`.

---

## Non-Goals

- HTTP-transport MCP servers. MCP supports stdio and HTTP/SSE; this
  epic ships stdio only. HTTP support is a follow-up if a real consumer
  emerges (none exist in the user's environment today).
- Auto-discovery of MCP servers from system config (e.g., reading
  Claude Desktop's `claude_desktop_config.json`). Out of scope —
  explicit config in this repo's `config.yaml` is clearer for
  reproducibility, eval, and CI. Doc-mention possible, not auto-import.
- Bundling any MCP server into this repo. Webcrawl-mcp lives in its own
  public repo and is installed separately.
- A new permissions / sandbox layer for MCP tools. Epic 21 owns that.
  This epic only ensures forward-compatible category propagation.
- Replacing the existing in-repo MCP server at
  `mcp_servers/corpus_search/server.py`. That server publishes
  `search_corpus` to *external* clients (Claude Desktop, Claude Code).
  This epic consumes external MCP servers *into* `local_rag.py`. The two
  are orthogonal directions of MCP traffic.
- Streaming tool results. MCP supports it via partial responses; we
  collect the full response before returning to the LLM, same as every
  other tool in `ToolRegistry`.
- Per-tool execution budgets, retries, or circuit breakers. Useful but
  out of scope; document as future work.

---

## Architecture

```
local_rag.py --agentic startup
    │
    ▼
┌────────────────────────────────────────┐
│ ToolRegistry()                         │   scripts/agentic.py:110
│  register("search_corpus", ...)        │   in-repo (existing)
│  register("read_file", ...)            │   in-repo (existing)
│  register("bash", ...)                 │   in-repo (existing)
│  register(...filesystem suite...)      │
│                                        │
│  register_mcp_tools(registry, config)  │   NEW: scripts/mcp_tool_adapter.py
│   ├─ for each server in config:        │
│   │    spawn subprocess (stdio)        │
│   │    initialize + tools/list         │
│   │    for each tool:                  │
│   │      register("<name>", schema,    │
│   │               executor)            │
│   └─ atexit.register(shutdown_all)     │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ run_agentic_loop()                     │   scripts/agentic.py:288
│   when LLM calls "webcrawl_search":    │
│     registry.execute(name, args)       │
│       └─ MCPClient.call_tool(name,     │
│             args)                      │
│            ├─ JSON-RPC tools/call      │
│            ├─ wait on stdout (timeout) │
│            └─ return stringified       │
│              content                   │
└────────────────────────────────────────┘
```

### Module surface (preview)

```python
# scripts/mcp_tool_adapter.py

@dataclass
class MCPServerSpec:
    name: str                          # logical name; default = pyproject script name
    command: list[str]                 # argv to spawn (use absolute paths for venv-bound servers)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[Path] = None
    timeout_s: float = 30.0            # per-tool-call timeout
    startup_timeout_s: float = 10.0    # for initialize handshake
    category: str = "network"          # forward-compat with Epic 21; per-server default
    tool_categories: dict[str, str] = field(default_factory=dict)  # per-tool override

class MCPClient:
    """Owns one subprocess MCP server. Stdio JSON-RPC."""
    def __init__(self, spec: MCPServerSpec): ...
    def start(self) -> list[dict]: ...   # returns tools/list response
    def call_tool(self, name: str, arguments: dict) -> str: ...
    def stop(self, timeout_s: float = 5.0) -> None: ...

def register_mcp_tools(
    registry: ToolRegistry,
    specs: list[MCPServerSpec],
    *,
    on_collision: Literal["namespace", "skip", "error"] = "namespace",
) -> list[MCPClient]:
    """
    Spawn each server, discover its tools, register each into the registry.
    Returns the live MCPClient list so the caller can shut them down at exit.
    """

def load_mcp_specs_from_config(
    config_path: Path | None = None,
    cli_override: list[str] | None = None,
) -> list[MCPServerSpec]:
    """Resolve specs from config.yaml + optional --mcp-server flag list."""
```

### Config schema (preview)

A new top-level block in `config.yaml`, mirroring the `retrieval:` block
established in Epic 18/19:

```yaml
mcp_servers:
  - name: webcrawl
    command:
      - ~/Documents/webcrawl/venv/bin/webcrawl-mcp
    # absolute path is required for venv-bound servers; the bridge does
    # not activate venvs
    env: {}
    timeout_s: 60                # web requests are slower than corpus
    category: network            # forward-compat with Epic 21

  - name: gmail
    command:
      - ~/.local/share/mcp-servers/gmail/venv/bin/gmail-mcp
    env:
      GMAIL_TOKEN_PATH: ~/.config/mcp/gmail-token.json
    category: network
    tool_categories:
      list_messages: read         # read-only ops can downgrade per-tool
      delete_message: write
```

When `mcp_servers:` is absent or empty, the bridge does nothing — no
subprocesses spawned, no behavior change.

### CLI override

```
local_rag.py --agentic --mcp-server <abs-path-to-server>  # quick one-off
local_rag.py --agentic --mcp-servers-config <path>        # alt config file
```

`--mcp-server` may be repeated. Servers given on the command line append
to (not replace) the config block. Useful for testing a freshly installed
server without editing config.

### Tool-name collision policy

Resolution order at registration time:

1. **In-repo tools always win.** If an MCP server exposes a tool whose
   name is already in the registry (e.g., MCP server exposing
   `search_corpus`), the MCP version is registered as
   `<server_name>__<tool_name>` (e.g., `webcrawl__search_corpus`) and a
   WARNING is logged.
2. **MCP vs. MCP collisions** (two MCP servers expose the same tool
   name): first server wins under its bare name; the second is
   namespaced as above.
3. The `on_collision` parameter to `register_mcp_tools` lets a caller
   choose `"skip"` (drop the colliding tool) or `"error"` (raise on
   collision) instead of `"namespace"` (the default). For the
   `local_rag.py` integration we use `"namespace"`.

The flat names exposed to the LLM are recorded in the JSONL audit when
Epic 21 lands; for now they're visible in the agentic-mode `--verbose`
trace.

### Lifecycle

- **Startup:** `register_mcp_tools` spawns each subprocess, sends MCP
  `initialize`, awaits the response within `startup_timeout_s`, sends
  `tools/list`, and registers each tool. A server that fails any of
  these steps is dropped with an ERROR log naming the spec and the
  failure reason; the REPL continues with whatever tools succeeded.
- **Per call:** `call_tool` writes a `tools/call` JSON-RPC message,
  reads one response (filtering server-initiated notifications), and
  returns the `content` field stringified. `timeout_s` per call; on
  timeout, return a `"<MCP timeout: ...>"` string to the LLM (which
  the loop treats like any other tool error message) and leave the
  subprocess running.
- **Shutdown:** `atexit.register` is called from `register_mcp_tools`
  with a closure over the live `MCPClient` list. SIGTERM the
  subprocesses, then SIGKILL after a grace period. Also wired to a
  signal handler so Ctrl-C in the REPL doesn't leak processes.
- **Mid-session crash:** if a subprocess exits while the bridge is
  waiting on a response, `call_tool` returns
  `"<MCP server '<name>' exited unexpectedly>"` to the LLM and marks
  the client as dead. Subsequent calls to that client return the same
  error without retrying. A future story (or epic) may add restart
  logic; this epic prefers fail-fast clarity over auto-restart.

### Forward-compat with Epic 21

Epic 21 adds a `category` arg to `ToolRegistry.register` (Story 21.1
AC 2). Until that lands, this epic's `register_mcp_tools` stores the
category on a side-channel attribute (`registry._tool_categories[name] =
category`) so it's queryable but not yet enforced. When 21.1 merges, the
adapter switches to passing `category=` directly. This is a one-line
change at integration time, called out explicitly in 22.3.

---

## Stories

### Story 22.1: MCP stdio bridge + registry integration

Foundational module. Implements `MCPServerSpec`, `MCPClient`, and
`register_mcp_tools` in `scripts/mcp_tool_adapter.py`. Wires a config
loader into `local_rag.py`'s agentic-setup path. Tests against the
in-repo `mcp_servers/corpus_search/server.py` as a fixture (no
network, no external deps). Webcrawl-mcp registration is **not** in
this story — that's 22.2. The deliverable here is the abstraction,
proven against an MCP server that already lives in this repo.

**Branch:** public `main` via feature branch.

See `docs/stories/22.1-mcp-tool-adapter.md` for full AC, tasks, and
dev notes.

**Status:** Draft

### Story 22.2: Register webcrawl-mcp as the first concrete consumer

End-to-end exercise of the 22.1 bridge against an out-of-repo MCP
server. Adds the `webcrawl` entry to `config.example.yaml` with the
absolute venv path pattern. Smoke-tests in the agentic loop: the
local LLM, given a current-events question, should call
`webcrawl_search`, then optionally `webcrawl_scrape` on a result URL,
and synthesize a grounded answer with the URL cited. Updates README's
agentic-mode section to note web access via MCP is now possible (and
opt-in).

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. `config.example.yaml` gains a fully commented `mcp_servers:` block
   showing webcrawl-mcp registration with the absolute-venv-path
   pattern. The block is commented out so a fresh clone behaves
   unchanged; opting in is a deliberate copy.
2. With the config block uncommented and webcrawl-mcp installed at
   `~/Documents/webcrawl/venv/bin/webcrawl-mcp` (or a
   user-equivalent path), `local_rag.py --agentic --verbose` shows
   the four `webcrawl_*` tools registered and visible to the LLM in
   the tool list.
3. Manual end-to-end smoke: a current-events query (e.g., "what is the
   latest news about X?") in the agentic loop triggers a
   `webcrawl_search` call, the result is returned to the LLM, and the
   final answer cites at least one URL surfaced by the search. The
   verbose trace is captured in the PR description as evidence.
4. README's agentic-mode subsection notes that web access is available
   via the MCP bridge using `webcrawl-mcp` as the reference
   implementation. Two-line setup pointer (install webcrawl-mcp,
   uncomment the config block) — full prose lives in 22.3.
5. No code changes in `scripts/mcp_tool_adapter.py` from 22.1 — if a
   real bug surfaces in the bridge during the smoke test, fix it as a
   22.1 amendment (or a 22.1 follow-up commit), not in 22.2.
6. Latency overhead measured: median wall-time for a `webcrawl_search`
   round-trip (LLM → bridge → subprocess → DuckDuckGo → bridge → LLM)
   captured and noted in the PR description. Single representative
   query is fine; this is a calibration number, not an eval.

**Status:** Draft

### Story 22.3: Lifecycle robustness, docs, and epic close

Hardens the failure-mode handling that 22.1 specifies but 22.2 may
have stress-tested under load. Lands the user-facing docs that close
the epic.

**Branch:** public `main` via feature branch.

**Acceptance Criteria:**

1. CLAUDE.md gains an "MCP Tool Bridge" section covering: how the
   bridge works, the `mcp_servers:` config block, the `--mcp-server`
   CLI override, the namespace-on-collision policy, the per-tool
   timeout knob, and the lifecycle guarantees (no leaked subprocesses
   on REPL exit, signal, or crash).
2. CLAUDE.md cross-links to Epic 21 (when shipped) for the
   permission-category implication. If 21 has not yet merged, note
   the forward-compat plan: MCP tools default to `category: network`
   under the broker; per-tool overrides land via `tool_categories:` in
   server config.
3. README's agentic-mode section gains a "Recommended: pluggable web
   tools" subsection (consistent with Epic 21.6 AC 5 wording) naming
   webcrawl-mcp as the reference web-access integration, with a
   minimal setup snippet.
4. `local_rag.py --help` text for `--agentic` notes that MCP servers
   listed in `config.yaml` will be spawned at startup; `--mcp-server`
   and `--mcp-servers-config` are documented in their own help lines.
5. Lifecycle audit: a written test plan committed at
   `docs/eval/mcp-bridge-lifecycle-checks.md` covering the four
   failure modes from Goals #4 (server-start failure, mid-session
   crash, tool timeout, malformed JSON-RPC). Each is a manual
   reproducer with expected behavior. Maintainer runs the checks,
   ticks them off in the PR, and commits the ticked-off file.
6. If the lifecycle audit surfaces real bugs in 22.1's implementation,
   they are fixed in this PR. Bug fixes that change `mcp_tool_adapter.py`
   are scoped to lifecycle/error-handling only — no new features.
7. ROADMAP updated: this epic closes; if a follow-up emerges (HTTP
   transport, auto-restart, per-tool budgets), file it as a separate
   epic candidate, do not extend this one.

**Status:** Draft

---

## Dependencies

### Epic Dependencies
- **Epic 12** — agentic loop and `ToolRegistry` are the integration
  point.
- **Epic 21** — coordinates on the future `category` arg. This epic
  ships first and prepares the ground; 21 lands the broker and the
  category enforcement.

### External Dependencies
- **No new pip dependencies.** The bridge speaks JSON-RPC over stdio
  using only stdlib (`subprocess`, `json`, `threading`, `queue`). MCP's
  message format is well-documented; we don't need the `mcp` SDK on
  the client side for stdio — we just write/read newline-delimited JSON.
- **Out-of-repo runtime dependency for 22.2:** webcrawl-mcp installed
  at a known absolute path (its own venv). User has it; CI does not
  need it (the `corpus_search` in-repo server is the test fixture for
  22.1).

### Code Dependencies
- `scripts/agentic.py:110` — `ToolRegistry` class (consumer).
- `scripts/agentic.py:122` — `ToolRegistry.register` signature
  (write target for 22.1's registration calls; coordinated update
  with 21.1 for `category` arg).
- `scripts/local_rag.py:458-473` — agentic-setup block (integration
  site for `register_mcp_tools` call).
- `mcp_servers/corpus_search/server.py` — in-repo MCP server, used as
  test fixture for 22.1 unit tests.
- `config.example.yaml` — new top-level `mcp_servers:` block (22.2).

---

## Implementation Order

```
Story 22.1 (MCP stdio bridge + registry integration)
    └── Foundation. Tested against in-repo corpus_search MCP server.

Story 22.2 (Register webcrawl-mcp)
    └── First out-of-repo consumer. Validates the abstraction.

Story 22.3 (Lifecycle robustness, docs, close)
    └── Documents the pattern; hardens against failures observed in 22.2.
```

Strictly sequential. 22.2 depends on 22.1's adapter API. 22.3 documents
both and may amend 22.1's lifecycle code based on 22.2's findings.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| MCP protocol drift (FastMCP version vs. our hand-rolled client) | Medium | Stdio JSON-RPC is the stable surface in MCP; FastMCP 2.x and the official Anthropic SDK both implement it identically for stdio. Pin to MCP spec rev in module docstring; revisit if FastMCP makes a breaking change. |
| Subprocess process-leak on Ctrl-C or hard exit | Medium | Atexit + signal handlers in 22.1 AC. 22.3 lifecycle audit tests the contract end-to-end. |
| Per-call timeout truncates legitimate long-running tool calls (e.g., a deep crawl) | Medium | Per-server `timeout_s` knob; user can raise it. Default is 30s, doubled to 60s for webcrawl in the example config (web is slow). |
| Tool-name collision shadows a tool the user expected | Low | `namespace` policy is loud (WARNING log + new flat name); `--verbose` shows the registered names. |
| Config error (typo, bad path, missing executable) bricks the REPL | Low | Server-spawn failure is per-server, not fatal. Failed server logged, REPL continues with the rest. Same posture as Epic 21's malformed-config rejection but without the hard exit (this is an opt-in feature, not a security control). |
| MCP server returns content shapes the LLM can't use (binary, large blobs) | Medium | Stringify all tool responses. Truncate at a documented limit (default 50KB) with a `<truncated, full size N bytes>` marker. Tunable per server. |
| MCP servers initialize slowly, blocking REPL launch | Low | `startup_timeout_s` per server (default 10s); failures don't block other servers. Total startup latency for N servers ≤ N × startup_timeout_s in the worst case. Spawn in parallel as a future optimization if it bites. |
| Forward-compat assumption about Epic 21's category arg is wrong | Low | The integration uses a tiny shim (side-channel dict). When 21.1 lands and the real signature changes, it's a one-line update in `register_mcp_tools` — called out in 22.3 explicitly. |

---

## Testing Strategy

### Unit Tests
- `MCPServerSpec` validation: required fields, default category, env
  passthrough.
- `MCPClient.start` happy path against the in-repo corpus_search server
  fixture: subprocess spawns, initialize succeeds, `tools/list` returns
  the `search_corpus` tool schema.
- `MCPClient.start` failure modes: bad command (executable not found),
  initialize timeout, malformed initialize response.
- `MCPClient.call_tool` happy path: round-trips a `search_corpus` call
  and returns the stringified content.
- `MCPClient.call_tool` failure modes: timeout, server crash mid-call,
  malformed response, JSON-RPC error response.
- `MCPClient.stop`: SIGTERM grace period observed, SIGKILL after.
  No leaked subprocesses (assertable via `psutil` or `os.waitpid`).
- `register_mcp_tools`: empty spec list is a no-op; one server registers
  N tools; collision with in-repo tool name produces the namespaced
  variant + WARNING; collision between two MCP servers handled per
  policy.
- `load_mcp_specs_from_config`: empty/missing block returns `[]`; valid
  block parses to specs; CLI overrides append.

### Integration Tests
- One end-to-end agentic-loop test in `tests/test_mcp_tool_adapter.py`
  that constructs a real `ToolRegistry`, calls `register_mcp_tools`
  against the in-repo corpus_search server, and asserts that
  `registry.execute("search_corpus", {...})` returns a result. No
  Ollama dependency — just exercises the registration + dispatch path.
- 22.2's smoke test against webcrawl-mcp is a manual reproducer
  documented in the PR; not in CI (CI does not have webcrawl-mcp
  installed and we are not adding it as a CI dependency).

### Manual Validation
- 22.2 manual smoke: current-events query in the live agentic loop;
  capture the verbose trace showing `webcrawl_search` invoked and a
  URL-cited answer returned.
- 22.3 lifecycle audit: four manual reproducers (server-start failure,
  mid-session crash, timeout, malformed JSON-RPC) with expected
  observed behavior, ticked off in
  `docs/eval/mcp-bridge-lifecycle-checks.md`.

---

## Acceptance Criteria (Epic Level)

1. `scripts/mcp_tool_adapter.py` exists and is importable.
2. `local_rag.py --agentic` with no `mcp_servers:` block configured
   behaves bit-for-bit identically to pre-22.
3. `local_rag.py --agentic` with one or more MCP servers configured
   spawns each, registers their tools, and shuts them down cleanly on
   REPL exit (no leaked subprocesses).
4. webcrawl-mcp's four tools (`webcrawl_scrape`, `webcrawl_search`,
   `webcrawl_map`, `webcrawl_crawl`) are usable end-to-end from the
   local agentic loop, demonstrated by a manual smoke trace in the
   22.2 PR.
5. Tool-name collisions resolve via the documented namespace policy
   with a logged WARNING — no silent shadowing.
6. Per-tool-call timeouts surface as a string error to the LLM, not
   an unhandled exception.
7. Subprocess crashes mid-session do not crash the REPL.
8. CLAUDE.md and README document the bridge, the config block, the CLI
   overrides, and the recommended webcrawl-mcp setup.
9. `config.example.yaml` includes a commented-out `mcp_servers:` block
   demonstrating the webcrawl-mcp registration shape.
10. Forward-compat with Epic 21 documented; integration with the
    PermissionBroker is one line of code when 21.1 lands.

---

## Definition of Done

- All three stories closed with AC met.
- `scripts/mcp_tool_adapter.py` lands with unit tests and one
  integration test against the in-repo corpus_search server fixture.
- `config.example.yaml` ships the commented-out `mcp_servers:` block.
- CLAUDE.md gains an "MCP Tool Bridge" section.
- README agentic-mode area links to webcrawl-mcp as the reference
  web-access integration.
- Lifecycle audit committed at
  `docs/eval/mcp-bridge-lifecycle-checks.md` with all four reproducers
  ticked off.
- CI gate green on the merge commits.

---

## Future Enhancements (Out of Scope)

- HTTP/SSE transport for MCP servers (currently stdio only).
- Auto-discovery from `claude_desktop_config.json` or other system MCP
  configs.
- Auto-restart of crashed MCP server subprocesses (currently fail-fast).
- Per-tool execution budgets (`max_calls`, `max_total_seconds`).
- Streaming MCP tool responses to the LLM (currently buffered).
- Hot-reload of the `mcp_servers:` config without REPL restart.
- Bundling specific MCP servers into this repo's pyproject.
- A `grounding mcp` CLI subcommand for one-off testing of registered
  servers (e.g., `grounding mcp call webcrawl webcrawl_search '{"query":"..."}'`).
  Reasonable follow-up if the bridge sees heavy use.
- Migrating the in-repo `mcp_servers/corpus_search/` server to be
  consumed via the bridge inside `local_rag.py` (so even our own MCP
  server runs through the same path). Currently `local_rag.py` consumes
  `search_corpus` directly via Python import; bridge-based registration
  would unify the dispatch path but adds a subprocess for no functional
  gain. Defer until there's a reason.

---

## References

- `scripts/agentic.py:110` — `ToolRegistry`, the integration target.
- `scripts/agentic.py:122` — `ToolRegistry.register`, the registration
  signature.
- `scripts/local_rag.py:458-473` — agentic-setup block, integration
  site for `register_mcp_tools`.
- `scripts/search_corpus_tool.py`, `scripts/filesystem_tools.py` — the
  `(schema, executor)` factory pattern this epic preserves.
- `mcp_servers/corpus_search/server.py` — in-repo MCP server used as
  test fixture for 22.1.
- `docs/epics/epic-12-agentic-tool-calling-v02.md` — agentic loop
  foundation.
- `docs/epics/epic-21-agentic-safeguards.md` — the permissions broker
  that will eventually gate MCP tool calls.
- webcrawl-mcp: `https://github.com/andyliszewski/webcrawl-mcp`
  (public OSS, FastMCP-based, the first concrete bridge consumer).
- MCP specification:
  `https://spec.modelcontextprotocol.io/specification/` — stdio
  transport and JSON-RPC message shapes.
- FastMCP: `https://github.com/jlowin/fastmcp` — the framework
  webcrawl-mcp is built on; our client speaks to its stdio output.
